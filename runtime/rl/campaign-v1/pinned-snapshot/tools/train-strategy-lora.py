#!/usr/bin/env python3
"""LoRA fine-tuning of the 8B strategy brain on "next build-order step" choices (run inside WSL).

The base model is loaded exactly as the scoring worker loads it (local weights, nf4 4-bit, bf16 compute)
on one explicitly named GPU. A LoRA adapter on the text attention and MLP projections is trained with
the causal-LM loss on the target label (plus EOS) only. Before and after training the holdout examples are
scored with the worker's own protocol: every legal label's summed log-likelihood through EOS; the top label
counts as the choice. Writes the adapter and a receipt with data digests, losses and accuracies.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from alpharush_rl.model_broker import GPU_MODELS, tokenize_request  # noqa: E402

GPU_5090 = "GPU-4d9f95ae-dfba-0ba5-9aad-09372fa19208"


def load_rows(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def label_scores(model, tokenizer, row, torch):
    """Summed log-likelihood of every legal label (through EOS), sharing the prompt's KV cache."""
    request = {"id": row["id"], "system": row["system"], "user": row["user"], "labels": row["labels"]}
    tokens = tokenize_request(tokenizer, request)
    prefix = torch.tensor([tokens["input_ids"]], dtype=torch.long, device="cuda:0")
    output = model(input_ids=prefix, use_cache=True, logits_to_keep=1)
    cache = output.past_key_values
    prefix_length = cache.get_seq_length()
    first = output.logits[0, -1].float().log_softmax(-1)
    scores = []
    for suffix in tokens["suffixes"]:
        score = float(first[suffix[0]].item())
        if len(suffix) > 1:
            cont = torch.tensor([suffix[:-1]], dtype=torch.long, device="cuda:0")
            out = model(input_ids=cont, past_key_values=cache, use_cache=True, logits_to_keep=0)
            logp = out.logits[0].float().log_softmax(-1)
            targets = torch.tensor(suffix[1:], dtype=torch.long, device="cuda:0")
            score += float(logp.gather(1, targets[:, None]).sum().item())
            cache.crop(prefix_length)
        scores.append(score)
    return scores


def evaluate(model, tokenizer, rows, torch, limit):
    hits, ranks = 0, []
    sample = rows[:limit]
    with torch.inference_mode():
        for row in sample:
            scores = label_scores(model, tokenizer, row, torch)
            order = sorted(range(len(scores)), key=lambda i: -scores[i])
            target = row["labels"].index(row["target"])
            hits += order[0] == target
            ranks.append(order.index(target) + 1)
    return {"examples": len(sample), "accuracy": hits / max(1, len(sample)),
            "mean_rank": sum(ranks) / max(1, len(ranks))}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", required=True)
    parser.add_argument("--holdout", required=True)
    parser.add_argument("--out", required=True, help="adapter output directory")
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--accumulate", type=int, default=8)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--eval-limit", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    os.environ["CUDA_VISIBLE_DEVICES"] = GPU_5090
    os.environ["HF_HUB_OFFLINE"] = "1"
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForImageTextToText, AutoTokenizer, BitsAndBytesConfig
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    if torch.cuda.device_count() != 1 or "5090" not in torch.cuda.get_device_name(0):
        raise SystemExit("expected exactly the RTX 5090")
    spec = GPU_MODELS["8b"]
    started = time.time()
    config = AutoConfig.from_pretrained(spec["path"], local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(spec["path"], local_files_only=True)
    loader = AutoModelForImageTextToText if hasattr(config, "text_config") else AutoModelForCausalLM
    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
                               bnb_4bit_compute_dtype=torch.bfloat16)
    model = loader.from_pretrained(spec["path"], local_files_only=True, quantization_config=quant,
                                   dtype=torch.bfloat16, device_map={"": 0}, attn_implementation="sdpa")
    train_rows, holdout_rows = load_rows(args.train), load_rows(args.holdout)
    random.Random(args.seed).shuffle(holdout_rows)
    model.eval()
    before = evaluate(model, tokenizer, holdout_rows, torch, args.eval_limit)
    print("holdout before", before, flush=True)
    projections = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
    targets = sorted(name for name, _ in model.named_modules()
                     if name.rsplit(".", 1)[-1] in projections and "language_model" in name.split("."))
    if not targets:
        targets = sorted(name for name, _ in model.named_modules() if name.rsplit(".", 1)[-1] in projections
                         and not any("vision" in part for part in name.split(".")))
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    model = get_peft_model(model, LoraConfig(r=args.rank, lora_alpha=2 * args.rank, lora_dropout=0.05, bias="none",
                                             target_modules=targets, task_type="CAUSAL_LM"))
    model.train()
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=0.0)
    order = list(range(len(train_rows)))
    total = int(len(order) * args.epochs)
    rng = random.Random(args.seed)
    losses, step = [], 0
    optimizer.zero_grad()
    for index in range(total):
        if index % len(order) == 0:
            rng.shuffle(order)
        row = train_rows[order[index % len(order)]]
        request = {"id": row["id"], "system": row["system"], "user": row["user"], "labels": row["labels"]}
        tokens = tokenize_request(tokenizer, request)
        suffix = tokens["label_token_ids"][row["target"]]
        ids = torch.tensor([tokens["input_ids"] + suffix], dtype=torch.long, device="cuda:0")
        labels = ids.clone()
        labels[0, :len(tokens["input_ids"])] = -100
        out = model(input_ids=ids, labels=labels)
        (out.loss / args.accumulate).backward()
        losses.append(float(out.loss.item()))
        if (index + 1) % args.accumulate == 0 or index + 1 == total:
            optimizer.step()
            optimizer.zero_grad()
            step += 1
            if step % 20 == 0:
                recent = losses[-20 * args.accumulate:]
                print(f"step {step}: loss {sum(recent) / len(recent):.4f}", flush=True)
    model.eval()
    after = evaluate(model, tokenizer, holdout_rows, torch, args.eval_limit)
    print("holdout after", after, flush=True)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(out_dir))
    digests = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(out_dir.iterdir()) if p.is_file()}
    receipt = {"schema": "alpharush-strategy-lora-v1", "base": f"{spec['repo']}@{spec['revision']}:nf4-bf16",
               "gpu": torch.cuda.get_device_name(0), "train": args.train, "holdout": args.holdout,
               "train_sha256": hashlib.sha256(Path(args.train).read_bytes()).hexdigest(),
               "holdout_sha256": hashlib.sha256(Path(args.holdout).read_bytes()).hexdigest(),
               "examples_seen": total, "optimizer_steps": step, "lr": args.lr, "rank": args.rank,
               "accumulate": args.accumulate, "target_modules": len(targets), "loss_first": losses[:20],
               "loss_last": losses[-20:], "holdout_before": before, "holdout_after": after,
               "adapter_files": digests, "wall_seconds": time.time() - started}
    (out_dir / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: receipt[k] for k in ("holdout_before", "holdout_after", "optimizer_steps", "wall_seconds")}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
