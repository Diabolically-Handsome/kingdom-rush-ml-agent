#!/usr/bin/env python3
"""Loopback scoring server for the 8B strategy brain with a trained LoRA adapter (run inside WSL).

Same protocol as tools/model-worker.py (complete legal-label distribution: summed log-likelihood of every
label through EOS, softmax over the menu only), but on the RTX 5090 and with an explicit adapter directory
whose files must match the training receipt. Inference only; no weights are updated.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from alpharush_rl.model_broker import GPU_MODELS, tokenize_request, validate_request  # noqa: E402

GPU_5090 = "GPU-4d9f95ae-dfba-0ba5-9aad-09372fa19208"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--port", type=int, default=12083)
    parser.add_argument("--max-prompt-tokens", type=int, default=8192)
    args = parser.parse_args()
    adapter = Path(args.adapter)
    receipt = json.loads((adapter / "receipt.json").read_text(encoding="utf-8"))
    for name, digest in receipt["adapter_files"].items():
        if name != "receipt.json" and hashlib.sha256((adapter / name).read_bytes()).hexdigest() != digest:
            raise SystemExit(f"adapter file {name} differs from its training receipt")
    os.environ["CUDA_VISIBLE_DEVICES"] = GPU_5090
    os.environ["HF_HUB_OFFLINE"] = "1"
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForImageTextToText, AutoTokenizer, BitsAndBytesConfig
    from peft import PeftModel
    spec = GPU_MODELS["8b"]
    config = AutoConfig.from_pretrained(spec["path"], local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(spec["path"], local_files_only=True)
    loader = AutoModelForImageTextToText if hasattr(config, "text_config") else AutoModelForCausalLM
    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
                               bnb_4bit_compute_dtype=torch.bfloat16)
    base = loader.from_pretrained(spec["path"], local_files_only=True, quantization_config=quant,
                                  dtype=torch.bfloat16, device_map={"": 0}, attn_implementation="sdpa")
    model = PeftModel.from_pretrained(base, str(adapter)).eval()
    model_id = (f"{spec['repo']}@{spec['revision']}:nf4-bf16:adapter-"
                + hashlib.sha256(json.dumps(receipt["adapter_files"], sort_keys=True).encode()).hexdigest()[:12])
    lock = threading.Lock()

    def distribution(request):
        validate_request(request)
        with lock, torch.inference_mode():
            started = time.perf_counter()
            tokens = tokenize_request(tokenizer, request)
            if tokens["prompt_tokens"] > args.max_prompt_tokens:
                raise ValueError("prompt exceeds token budget; refusing silent truncation")
            prefix = torch.tensor([tokens["input_ids"]], dtype=torch.long, device="cuda:0")
            output = model(input_ids=prefix, use_cache=True, logits_to_keep=1)
            cache = output.past_key_values
            prefix_length = cache.get_seq_length()
            first = output.logits[0, -1].float().log_softmax(-1)
            scores = []
            for suffix in tokens["suffixes"]:
                score = float(first[suffix[0]].item())
                cont = torch.tensor([suffix[:-1]], dtype=torch.long, device="cuda:0")
                out = model(input_ids=cont, past_key_values=cache, use_cache=True, logits_to_keep=0)
                logp = out.logits[0].float().log_softmax(-1)
                targets = torch.tensor(suffix[1:], dtype=torch.long, device="cuda:0")
                score += float(logp.gather(1, targets[:, None]).sum().item())
                cache.crop(-(cache.get_seq_length() - prefix_length))
                scores.append(score)
            top = max(scores)
            weights = [math.exp(max(s - top, -690.0)) for s in scores]
            total = sum(weights)
            p = [w / total for w in weights]
            return {"id": request["id"], "labels": request["labels"], "p": p, "logp": [math.log(x) for x in p],
                    "choice": request["labels"][max(range(len(scores)), key=scores.__getitem__)],
                    "prompt_sha256": tokens["prompt_sha256"], "model": model_id, "seconds": time.perf_counter() - started,
                    "sequence_log_likelihood": scores, "complete_legal_distribution": True, "adapter": str(adapter),
                    "learning_updates": 0}

    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_GET(self):
            payload = json.dumps({"ready": True, "model": model_id}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def do_POST(self):
            status = 200
            try:
                size = int(self.headers.get("Content-Length", "0"))
                if self.path != "/distribution" or not 0 < size <= 4 * 1024 ** 2:
                    raise ValueError("invalid endpoint or request size")
                result = distribution(json.loads(self.rfile.read(size)))
            except Exception as error:
                status, result = 400, {"error": str(error)}
            payload = json.dumps(result, ensure_ascii=False, allow_nan=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(json.dumps({"ready": True, "port": args.port, "model": model_id}), flush=True)
    try:
        server.serve_forever()
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
