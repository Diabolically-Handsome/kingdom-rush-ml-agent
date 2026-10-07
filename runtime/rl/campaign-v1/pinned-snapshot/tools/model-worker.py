#!/usr/bin/env python3
"""Isolated WSL inference worker; explicit GPU UUID, local weights, no adapter.

Default invocation does one request and exits. HTTP mode is explicit --serve.
All legal output sequences, including multi-token labels, receive exact summed
conditional log likelihood through EOS. Softmax masks to the supplied menu.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from alpharush_rl.model_broker import GPU_MODELS, tokenize_request, validate_distribution, validate_request, verify_model_manifest


def preflight(spec: dict) -> dict:
    raw = subprocess.check_output([
        "nvidia-smi", "--query-gpu=uuid,name,memory.used,memory.total,utilization.gpu",
        "--format=csv,noheader,nounits"], text=True)
    rows = [[part.strip() for part in line.split(",")] for line in raw.splitlines()]
    match = next((row for row in rows if row[0] == spec["gpu_uuid"]), None)
    if match is None or match[1] != spec["gpu_name"]:
        raise RuntimeError("required physical GPU not found")
    apps = subprocess.check_output([
        "nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name",
        "--format=csv,noheader"], text=True)
    busy = [line for line in apps.splitlines() if line.split(",")[0].strip() == spec["gpu_uuid"]]
    if busy:
        raise RuntimeError("GPU already has a compute process; leaving it untouched: " + "; ".join(busy))
    free = float(match[3]) - float(match[2])
    required = 14000 if spec["repo"].find("24B") >= 0 else 7000
    if free < required:
        raise RuntimeError(f"GPU has insufficient available memory ({free:g} MiB)")
    return {"uuid": match[0], "name": match[1], "memory_used_mib": float(match[2]),
            "memory_total_mib": float(match[3]), "utilization_percent": float(match[4]),
            "existing_compute_processes": busy}


class Scorer:
    def __init__(self, model_key: str, max_prompt_tokens: int):
        spec = GPU_MODELS[model_key]
        self.manifest = verify_model_manifest(model_key)
        self.preflight = preflight(spec)
        os.environ["CUDA_VISIBLE_DEVICES"] = spec["gpu_uuid"]
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        import torch
        from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForImageTextToText, AutoTokenizer, BitsAndBytesConfig
        self.torch, self.spec, self.max_prompt_tokens = torch, spec, max_prompt_tokens
        if torch.cuda.device_count() != 1 or torch.cuda.get_device_name(0) != spec["gpu_name"]:
            raise RuntimeError("CUDA visibility must resolve to exactly the assigned GPU")
        path = spec["path"]
        config = AutoConfig.from_pretrained(path, local_files_only=True, trust_remote_code=False)
        self.tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True, trust_remote_code=False)
        loader = AutoModelForImageTextToText if hasattr(config, "text_config") else AutoModelForCausalLM
        quantization = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                         bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16)
        started = time.perf_counter()
        self.model = loader.from_pretrained(
            path, local_files_only=True, trust_remote_code=False, quantization_config=quantization,
            dtype=torch.bfloat16, device_map={"": 0}, attn_implementation="sdpa",
        ).eval()
        self.load_seconds = time.perf_counter() - started
        self.model_id = f"{spec['repo']}@{spec['revision']}:nf4-bf16:instruction-base:no-adapter"
        identity_names = {"config.json", "model.safetensors.index.json", "generation_config.json",
                          "tekken.json", "tokenizer.json", "tokenizer_config.json",
                          "special_tokens_map.json", "chat_template.jinja", "SYSTEM_PROMPT.txt"}
        self.identity_hashes = {name: record["sha256"]
                                for name, record in self.manifest["content"]["files"].items()
                                if name in identity_names}
        self.requests_seen = 0
        self.lock = threading.Lock()

    def distribution(self, request: dict) -> dict:
        validate_request(request)
        with self.lock, self.torch.inference_mode():
            torch = self.torch
            torch.cuda.synchronize()
            started = time.perf_counter()
            tokens = tokenize_request(self.tokenizer, request)
            if tokens["prompt_tokens"] > self.max_prompt_tokens:
                raise ValueError("native prompt exceeds token budget; refusing silent truncation")
            prefix = torch.tensor([tokens["input_ids"]], dtype=torch.long, device="cuda:0")
            torch.cuda.reset_peak_memory_stats()
            output = self.model(input_ids=prefix, use_cache=True, logits_to_keep=1)
            cache = output.past_key_values
            prefix_length = cache.get_seq_length()
            first_logp = output.logits[0, -1].float().log_softmax(-1)
            del output
            scores = []
            for suffix in tokens["suffixes"]:
                score = float(first_logp[suffix[0]].item())
                continuation = torch.tensor([suffix[:-1]], dtype=torch.long, device="cuda:0")
                output = self.model(input_ids=continuation, past_key_values=cache,
                                    use_cache=True, logits_to_keep=0)
                targets = torch.tensor(suffix[1:], dtype=torch.long, device="cuda:0")
                logp = output.logits[0].float().log_softmax(-1)
                score += float(logp.gather(1, targets[:, None]).sum().item())
                scores.append(score)
                # Negative counts remove suffix tokens. Positive counts still
                # mean absolute target lengths in the 5.17 compatibility path.
                cache.crop(-(cache.get_seq_length() - prefix_length))
                if cache.get_seq_length() != prefix_length:
                    raise RuntimeError("shared prefix KV cache did not restore exactly")
                del output, continuation, targets, logp
            torch.cuda.synchronize()
            seconds = time.perf_counter() - started
            maximum = max(scores)
            weights = [math.exp(max(score - maximum, -690.0)) for score in scores]
            total = sum(weights)
            probabilities = [weight / total for weight in weights]
            choice = request["labels"][max(range(len(scores)), key=scores.__getitem__)]
            response = {
                "id": request["id"], "labels": request["labels"], "p": probabilities,
                "logp": [math.log(p) for p in probabilities], "choice": choice,
                "prompt_sha256": tokens["prompt_sha256"], "model": self.model_id,
                "seconds": seconds, "load_seconds": self.load_seconds,
                "cold_first_request": self.requests_seen == 0,
                "measurement_phase": request.get("measurement_phase", "sanity"),
                "model_revision": self.spec["revision"], "identity_hashes": self.identity_hashes,
                "model_manifest_sha256": self.manifest["manifest_sha256"],
                "tokenized_prompt_ids_sha256": tokens["tokenized_prompt_ids_sha256"],
                "prompt_tokens": tokens["prompt_tokens"], "label_token_ids": tokens["label_token_ids"],
                "sequence_log_likelihood": scores, "scoring": "sum-label-and-eos-log-likelihood",
                "complete_legal_distribution": True, "gpu": self.preflight,
                "peak_memory_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
                "adapter": None, "learning_updates": 0,
            }
            self.requests_seen += 1
            return validate_distribution(response, request)


def write_result(path: str | None, result: dict):
    payload = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    if path:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(payload, encoding="utf-8")
    else:
        print(payload, flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=tuple(GPU_MODELS), required=True)
    parser.add_argument("--request", help="one native request JSON")
    parser.add_argument("--requests", help="JSON list of at most six native requests; load model once")
    parser.add_argument("--output")
    parser.add_argument("--tokenize-only", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--serve", action="store_true")
    parser.add_argument("--max-prompt-tokens", type=int, default=16384)
    args = parser.parse_args()
    if args.request and args.requests:
        parser.error("choose --request or --requests")
    if args.preflight_only:
        write_result(args.output, preflight(GPU_MODELS[args.model]))
        return
    if args.tokenize_only:
        if not (args.request or args.requests):
            parser.error("--tokenize-only requires --request or --requests")
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(GPU_MODELS[args.model]["path"], local_files_only=True)
        data = json.loads(Path(args.requests or args.request).read_text(encoding="utf-8-sig"))
        requests = data if args.requests else [data]
        if not isinstance(requests, list) or not 1 <= len(requests) <= 6:
            parser.error("batch must contain one to six requests")
        responses = []
        for request in requests:
            result = tokenize_request(tokenizer, request)
            if result["prompt_tokens"] > args.max_prompt_tokens:
                raise ValueError("native prompt exceeds token budget; refusing silent truncation")
            result.pop("input_ids")
            result.pop("suffixes")
            responses.append({"id": request["id"], "model": args.model, **result,
                              "max_prompt_tokens": args.max_prompt_tokens,
                              "truncated": False, "learning_updates": 0})
        write_result(args.output, {"responses": responses, "learning_updates": 0}
                     if args.requests else responses[0])
        return
    if not args.serve and not (args.request or args.requests):
        parser.error("provide --request for bounded one-shot scoring, or explicit --serve")
    requests = None
    if not args.serve:
        data = json.loads(Path(args.requests or args.request).read_text(encoding="utf-8-sig"))
        requests = data if args.requests else [data]
        if not isinstance(requests, list) or not 1 <= len(requests) <= 6:
            parser.error("batch must contain one to six requests")
        for request in requests:
            validate_request(request)
    scorer = Scorer(args.model, args.max_prompt_tokens)
    if not args.serve:
        responses = [scorer.distribution(request) for request in requests]
        write_result(args.output, {"model": scorer.model_id, "responses": responses,
                                   "load_seconds": scorer.load_seconds, "learning_updates": 0}
                     if args.requests else responses[0])
        return
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            payload = json.dumps({"ready": True, "model": scorer.model_id, "gpu": scorer.preflight}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def do_POST(self):
            status = 200
            try:
                size = int(self.headers.get("Content-Length", "0"))
                if self.path != "/distribution" or not 0 < size <= 4 * 1024**2:
                    raise ValueError("invalid endpoint or request size")
                request = json.loads(self.rfile.read(size))
                result = scorer.distribution(request)
            except Exception as error:
                status, result = 400, {"error": str(error)}
            payload = json.dumps(result, ensure_ascii=False, allow_nan=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
    server = ThreadingHTTPServer(("127.0.0.1", scorer.spec["port"]), Handler)
    print(json.dumps({"ready": True, "port": scorer.spec["port"], "model": scorer.model_id}), flush=True)
    try:
        server.serve_forever()
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
