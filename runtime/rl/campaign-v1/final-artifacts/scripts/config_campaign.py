"""Configure native-campaign (rehearsal) or native-final from the latest search results.

usage: config_campaign.py rehearsal|final <jobs.json> <operator weights path> [seeds]
"""
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(r"C:\Users\<user>\Documents\AlphaRush")
APPROVAL = ("[用户原话已省略 / user's message omitted]"
            "（2026-10-05 用户授权自主推进至 2026-10-06 19:00）")
mode, source, weights = sys.argv[1], Path(sys.argv[2]), sys.argv[3]
seeds = [int(s) for s in (sys.argv[4] if len(sys.argv) > 4 else
                          ("5001,5002,5003,5004,5005" if mode == "rehearsal" else "6001,6002,6003,6004,6005")).split(",")]
plans = json.loads(source.read_text(encoding="utf-8"))["plans"]
p = ROOT / "configs/phases/campaign-v1.json"
cfg = json.loads(p.read_text(encoding="utf-8"))
job = {"enabled": True, "approval": APPROVAL,
       "purpose": ("full main campaign from a new save, levels 1-12 in order on Normal: the 8B strategy brain picks each "
                   "level's plan (hero, spells, wave calls) and the star-upgrade package, the operator network executes "
                   "every decision; up to 8 attempts per level with untried plans"
                   + ("; ONE-SHOT final evaluation on the final_campaign_run seeds" if mode == "final" else
                      "; dress rehearsal on evaluation seeds")),
       "max_wall_seconds": 7200, "total_wall_seconds": 14400, "max_jobs": 1 if mode == "final" else 6,
       "max_games": len(seeds) * 12 * 8 + 5, "gpu": "external-inference", "optimizer_steps": 0, "serial": False,
       "rng_mode": "isolate_sound+stable_pairs", "action_scope": "v2", "replay_fraction": 0.0,
       "time_reserve_seconds": 120,
       "episode_protocol": {"wait_ticks": 120, "max_interval_ticks": 600, "max_ticks": 240000, "max_decisions": 3000},
       "operator_weights": {"path": weights, "sha256": hashlib.sha256((ROOT / weights).read_bytes()).hexdigest()},
       "campaign": {"levels": list(range(1, 13)), "seeds": seeds, "attempts_per_level": 8, "plans": plans,
                    "policy": "operator", "brain": "8b"}}
name = "native-final" if mode == "final" else "native-campaign"
cfg["jobs"][name] = job
p.write_text(json.dumps(cfg, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
print("configured", name, "seeds", seeds, "plans per level", {k: len(v) for k, v in plans.items()})
