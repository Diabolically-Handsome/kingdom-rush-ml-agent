"""elite-v3: seal the phase (2026-10-12, owner: the project is archived). Disables every job kind of the elite-v3 phase so
no native or GPU job can start from this configuration; nothing else changes.

usage: config_v3_seal.py (out=<file> | apply=1)   (apply=1 is refused while any runtime/rl/*/job.lock exists)"""
import json
import sys
from pathlib import Path

ROOT = Path(r"C:\Users\<user>\Documents\AlphaRush")
args = dict(a.split("=", 1) for a in sys.argv[1:])
apply = args.get("apply") == "1"
if apply == ("out" in args):
    raise SystemExit("give exactly one of out=<file> or apply=1")
if apply and any((ROOT / "runtime/rl").glob("*/job.lock")):
    raise SystemExit("refused: a job holds a lock")
p = ROOT / "configs/phases/elite-v3.json"
cfg = json.loads(p.read_text(encoding="utf-8"))
for job in cfg["jobs"].values():
    job["enabled"] = False
text = json.dumps(cfg, ensure_ascii=False, indent=1) + "\n"
if apply:
    p.write_text(text, encoding="utf-8", newline="\n")
else:
    Path(args["out"]).write_text(text, encoding="utf-8", newline="\n")
print("sealed:", {k: v["enabled"] for k, v in cfg["jobs"].items()})
