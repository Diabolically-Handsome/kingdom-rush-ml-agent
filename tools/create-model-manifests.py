#!/usr/bin/env python3
"""One-time read-only audit of local instruction weights; cache in AlphaRush.

Run after GPU timing, so disk and hashing work do not affect latency results.
Repeated calls reuse the receipt only while all audited file stats match.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from alpharush_rl.model_broker import GPU_MODELS


def canonical_hash(value: dict) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":")).encode("utf-8")).hexdigest()


def file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(16 * 1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def create(model_key: str, output: Path, force: bool = False) -> dict:
    spec = GPU_MODELS[model_key]
    base = Path(spec["path"])
    index = json.loads((base / "model.safetensors.index.json").read_text())
    names = set(index["weight_map"].values())
    names.update({"config.json", "model.safetensors.index.json", "generation_config.json"})
    for name in ("tekken.json", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
                 "chat_template.jinja", "processor_config.json", "params.json", "SYSTEM_PROMPT.txt"):
        if (base / name).is_file():
            names.add(name)
    for name in tuple(names):
        metadata = Path(".cache/huggingface/download") / (name + ".metadata")
        if (base / metadata).is_file():
            revision = (base / metadata).read_text().splitlines()[0]
            if revision != spec["revision"]:
                raise RuntimeError(f"revision mismatch for {model_key}:{name}")
            names.add(str(metadata))
    names = sorted(names)
    if output.is_file() and not force:
        previous = json.loads(output.read_text())
        content = previous["content"]
        if previous["manifest_sha256"] == canonical_hash(content) and sorted(content["files"]) == names:
            unchanged = True
            for name in names:
                stat = (base / name).stat()
                entry = content["files"][name]
                unchanged &= entry["bytes"] == stat.st_size and entry["mtime_ns"] == stat.st_mtime_ns
            if unchanged and content["revision"] == spec["revision"]:
                print(json.dumps({"model": model_key, "reused": True,
                                  "manifest_sha256": previous["manifest_sha256"]}), flush=True)
                return previous
    started = time.perf_counter()
    records = {}
    for name in names:
        path = base / name
        before = path.stat()
        digest = file_digest(path)
        after = path.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise RuntimeError("base file changed during hash audit")
        records[name] = {"bytes": after.st_size, "mtime_ns": after.st_mtime_ns, "sha256": digest}
        print(json.dumps({"model": model_key, "file": name, "bytes": after.st_size,
                          "sha256": digest}), flush=True)
    content = {"schema": "alpharush-instruction-base-sha256-v1", "repo": spec["repo"],
               "revision": spec["revision"], "weights_directory": str(base), "files": records,
               "adapter": None}
    receipt = {"content": content, "manifest_sha256": canonical_hash(content),
               "hashing_seconds": time.perf_counter() - started,
               "total_bytes": sum(item["bytes"] for item in records.values()),
               "source_files_modified": False, "learning_updates": 0}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(receipt, indent=2, ensure_ascii=False) + "\n")
    return receipt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=("8b", "24b", "both"), default="both")
    parser.add_argument("--output-dir", default="/home/<user>/alpharush/manifests")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    for key in (GPU_MODELS if args.model == "both" else [args.model]):
        result = create(key, Path(args.output_dir) / ("model-" + key + ".json"), args.force)
        print(json.dumps({"model": key, "manifest_sha256": result["manifest_sha256"],
                          "hashing_seconds": result["hashing_seconds"], "total_bytes": result["total_bytes"]}), flush=True)


if __name__ == "__main__":
    main()
