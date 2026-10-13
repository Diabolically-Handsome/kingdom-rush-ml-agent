#!/usr/bin/env python3
"""elite-v3b standalone offline scorer (run inside WSL; RTX 5090 only; start it only while no native game job runs).

The model is loaded exactly as tools/model-worker.py loads it (local weights, nf4 + double quant, bf16 compute, sdpa,
one explicit GPU) and every label is scored exactly as the worker scores it (summed log-likelihood of the label and EOS
through a shared prompt KV cache, negative cache.crop, softmax over the legal labels). The shared code is imported, not
changed (alpharush_rl.model_broker.tokenize_request / GPU_MODELS / verify_model_manifest).

Input: a JSONL request file made by v3b_prompts.py, one consult per line:
  {"key", "arm", "system", "labels", "rotations": [{"order": [...], "user": "..."}], "ids": [...],
   "reason_user": "..." (reasoning arms only)}
Modes:
  label   p = mean over rotations of the label distribution mapped to plays through each rotation's order;
          choice = argmax (ties to the lower index), exactly strategy_brain.LanguageBrain._ask.
  reason  phase 1: greedy generation of the reasoning (fixed batch plan: consults sorted by key, fixed batch size,
          left padding, do_sample False, max_new_tokens); phase 2: every rotation re-scored with the reasoning (FINAL
          line removed) appended to the assistant turn followed by "\\nFINAL ANSWER: ", built from separately encoded
          pieces. Stated choice = the reasoning's last "FINAL: [Pk]" (else the conditioned argmax, counted).
Writes <out>.jsonl (one row per consult) and <out>.receipt.json."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
import time
from pathlib import Path

sys.dont_write_bytecode = True
ROOT = Path("/mnt/c/Users/<user>/Documents/AlphaRush")
sys.path.insert(0, str(ROOT))
from alpharush_rl.model_broker import GPU_MODELS, tokenize_request, verify_model_manifest  # noqa: E402

GPU_5090 = {"gpu_uuid": "GPU-4d9f95ae-dfba-0ba5-9aad-09372fa19208", "gpu_name": "NVIDIA GeForce RTX 5090"}
MAX_PROMPT_TOKENS = 8192
FINAL_JOINER = "\nFINAL ANSWER: "
_FINAL = re.compile(r"FINAL:\s*\[?(P\d)\]?", re.IGNORECASE)


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(model_key):
    spec = {**GPU_MODELS[model_key], **GPU_5090}
    manifest = verify_model_manifest(model_key)
    os.environ["CUDA_VISIBLE_DEVICES"] = spec["gpu_uuid"]
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    import torch
    from transformers import (AutoConfig, AutoModelForCausalLM, AutoModelForImageTextToText, AutoTokenizer,
                              BitsAndBytesConfig)
    if torch.cuda.device_count() != 1 or torch.cuda.get_device_name(0) != spec["gpu_name"]:
        raise RuntimeError("CUDA visibility must resolve to exactly the RTX 5090")
    path = spec["path"]
    config = AutoConfig.from_pretrained(path, local_files_only=True, trust_remote_code=False)
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True, trust_remote_code=False)
    loader = AutoModelForImageTextToText if hasattr(config, "text_config") else AutoModelForCausalLM
    quantization = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
                                      bnb_4bit_compute_dtype=torch.bfloat16)
    started = time.perf_counter()
    model = loader.from_pretrained(path, local_files_only=True, trust_remote_code=False,
                                   quantization_config=quantization, dtype=torch.bfloat16, device_map={"": 0},
                                   attn_implementation="sdpa").eval()
    identity = {"model": f"{spec['repo']}@{spec['revision']}:nf4-bf16:instruction-base:no-adapter",
                "manifest_sha256": manifest["manifest_sha256"], "tokenizer": type(tokenizer).__name__,
                "load_seconds": round(time.perf_counter() - started, 1)}
    return torch, tokenizer, model, identity


def eos_id(tokenizer):
    eos = tokenizer.eos_token_id
    if isinstance(eos, list):
        eos = eos[0]
    if not isinstance(eos, int):
        raise ValueError("instruction tokenizer requires EOS")
    return eos


def scores_for(torch, model, tokens, extra=()):
    """The worker's scoring loop (tools/model-worker.py L97-119) with optional extra prefix ids after the prompt."""
    ids = list(tokens["input_ids"]) + list(extra)
    if len(ids) > MAX_PROMPT_TOKENS:
        raise ValueError("prompt exceeds the token budget; refusing silent truncation")
    prefix = torch.tensor([ids], dtype=torch.long, device="cuda:0")
    output = model(input_ids=prefix, use_cache=True, logits_to_keep=1)
    cache = output.past_key_values
    prefix_length = cache.get_seq_length()
    first_logp = output.logits[0, -1].float().log_softmax(-1)
    del output
    scores = []
    for suffix in tokens["suffixes"]:
        score = float(first_logp[suffix[0]].item())
        continuation = torch.tensor([suffix[:-1]], dtype=torch.long, device="cuda:0")
        out = model(input_ids=continuation, past_key_values=cache, use_cache=True, logits_to_keep=0)
        targets = torch.tensor(suffix[1:], dtype=torch.long, device="cuda:0")
        logp = out.logits[0].float().log_softmax(-1)
        score += float(logp.gather(1, targets[:, None]).sum().item())
        scores.append(score)
        cache.crop(-(cache.get_seq_length() - prefix_length))
        if cache.get_seq_length() != prefix_length:
            raise RuntimeError("shared prefix KV cache did not restore exactly")
        del out, continuation, targets, logp
    return scores


def softmax(scores):
    top = max(scores)
    weights = [math.exp(max(s - top, -690.0)) for s in scores]
    total = sum(weights)
    return [w / total for w in weights]


def label_distribution(torch, model, tokenizer, row, extra=()):
    n = len(row["labels"])
    total = [0.0] * n
    rotations = []
    for k, rot in enumerate(row["rotations"]):
        request = {"id": f"{row['key']}:r{k}", "system": row["system"], "user": rot["user"], "labels": row["labels"]}
        tokens = tokenize_request(tokenizer, request)
        p_rot = softmax(scores_for(torch, model, tokens, extra))
        for position, i in enumerate(rot["order"]):
            total[i] += p_rot[position]
        rotations.append({"order": rot["order"], "p": p_rot, "prompt_sha256": tokens["prompt_sha256"],
                          "prompt_tokens": tokens["prompt_tokens"] + len(extra)})
    p = [x / len(row["rotations"]) for x in total]
    choice = max(range(n), key=lambda i: (p[i], -i))
    return p, choice, rotations


def generate(torch, model, tokenizer, rows, batch_size, max_new_tokens):
    """Greedy reasoning for every row in a fixed batch plan; returns {key: (text, token count)}."""
    eos = eos_id(tokenizer)
    pad = tokenizer.pad_token_id if isinstance(tokenizer.pad_token_id, int) else eos
    out = {}
    for start in range(0, len(rows), batch_size):
        batch = rows[start:start + batch_size]
        prompts = []
        for row in batch:
            tokens = tokenize_request(tokenizer, {"id": row["key"], "system": row["system"],
                                                  "user": row["reason_user"], "labels": row["labels"]})
            if tokens["prompt_tokens"] > MAX_PROMPT_TOKENS:
                raise ValueError("reasoning prompt exceeds the token budget")
            prompts.append(tokens["input_ids"])
        width = max(len(p) for p in prompts)
        ids = torch.tensor([[pad] * (width - len(p)) + p for p in prompts], dtype=torch.long, device="cuda:0")
        mask = torch.tensor([[0] * (width - len(p)) + [1] * len(p) for p in prompts], dtype=torch.long, device="cuda:0")
        with torch.inference_mode():
            gen = model.generate(input_ids=ids, attention_mask=mask, max_new_tokens=max_new_tokens, do_sample=False,
                                 temperature=None, top_p=None, top_k=None, eos_token_id=eos, pad_token_id=pad)
        for row, seq in zip(batch, gen[:, width:].tolist()):
            if eos in seq:
                seq = seq[:seq.index(eos)]
            while seq and seq[-1] == pad:
                seq = seq[:-1]
            out[row["key"]] = (tokenizer.decode(seq, skip_special_tokens=True), len(seq))
    return out


def strip_final(text):
    return re.sub(r"\n?[^\n]*FINAL:[^\n]*", "", text or "").rstrip()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", choices=("8b", "24b"), required=True)
    ap.add_argument("--requests", required=True)
    ap.add_argument("--out", required=True, help="output path without extension")
    ap.add_argument("--mode", choices=("label", "reason"), required=True)
    ap.add_argument("--limit", type=int, default=0, help="only the first N consults (sorted by key)")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--max-new-tokens", type=int, default=250)
    ap.add_argument("--batch1-check", type=int, default=20, help="reason mode: regenerate N consults at batch 1")
    args = ap.parse_args(argv)
    rows = [json.loads(line) for line in Path(args.requests).read_text(encoding="utf-8").splitlines() if line.strip()]
    rows.sort(key=lambda r: r["key"])
    if args.limit:
        rows = rows[:args.limit]
    torch, tokenizer, model, identity = load(args.model)
    started = time.perf_counter()
    results, generation = [], {}
    with torch.inference_mode():
        if args.mode == "reason":
            t0 = time.perf_counter()
            generation = generate(torch, model, tokenizer, rows, args.batch, args.max_new_tokens)
            gen_seconds = time.perf_counter() - t0
            check = generate(torch, model, tokenizer, rows[:args.batch1_check], 1, args.max_new_tokens)
            divergent = sum(1 for k, v in check.items() if v[0] != generation[k][0])
        for row in rows:
            extra = ()
            record = {"key": row["key"], "arm": row["arm"], "model": args.model, "ids": row["ids"]}
            if args.mode == "reason":
                text, n_tokens = generation[row["key"]]
                reasoning = strip_final(text)
                extra = (tokenizer.encode(reasoning, add_special_tokens=False)
                         + tokenizer.encode(FINAL_JOINER, add_special_tokens=False))
                if eos_id(tokenizer) in extra:
                    raise RuntimeError("EOS inside the reasoning prefix")
                found = _FINAL.findall(text)
                stated = row["ids"].index(found[-1].upper()) if found and found[-1].upper() in row["ids"] else None
                record.update({"reasoning": text, "reasoning_tokens": n_tokens, "stated": stated})
            p, choice, rotations = label_distribution(torch, model, tokenizer, row, extra)
            record.update({"p": p, "choice": choice, "rotations": rotations})
            if args.mode == "reason":
                record["final_choice"] = record["stated"] if record["stated"] is not None else choice
                record["stated_parsed"] = record["stated"] is not None
            results.append(record)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out.with_suffix(".jsonl"), "w", encoding="utf-8") as handle:
        for record in results:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    receipt = {"schema": "alpharush-v3b-scorer-v1", **identity, "gpu": GPU_5090["gpu_name"], "mode": args.mode,
               "requests": args.requests, "requests_sha256": sha256_file(args.requests), "consults": len(results),
               "script_sha256": sha256_file(__file__), "batch": args.batch, "max_new_tokens": args.max_new_tokens,
               "seconds": round(time.perf_counter() - started, 1),
               "output_sha256": sha256_file(out.with_suffix(".jsonl"))}
    if args.mode == "reason":
        receipt.update({"generation_seconds": round(gen_seconds, 1), "batch_plan": "sorted by key, fixed batch size",
                        "batch1_check": args.batch1_check, "batch1_divergent": divergent,
                        "stated_parsed": sum(r["stated_parsed"] for r in results),
                        "mean_reasoning_tokens": round(sum(r["reasoning_tokens"] for r in results)
                                                       / max(1, len(results)), 1)})
    out.with_suffix(".receipt.json").write_text(json.dumps(receipt, indent=1), encoding="utf-8")
    print(json.dumps({k: receipt[k] for k in receipt if k not in ("requests",)}))


if __name__ == "__main__":
    main()
