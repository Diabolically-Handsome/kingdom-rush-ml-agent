"""elite-v2 run 4: elite stages 22 and 21 again, warm-started from run 3's best 20 plans per level (re-played on fresh
seeds) with the rest of each population of 60 random; human-timed trap clicks on (no clicks on elite stages).
usage: config_v2_run4.py (out=<file> | apply=1) [hours=H]; apply=1 is refused while any job.lock exists."""
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
hours = float(args.get("hours", 3))
RUN3 = "native-search-e7bfb9f97528485b93a4d38b23b35c44"
p = ROOT / "configs/phases/elite-v2.json"
cfg = json.loads(p.read_text(encoding="utf-8"))
job = cfg["jobs"]["native-search"]
job["enabled"] = True
job["search"] = {
    "levels": [22, 21],
    "search_seed": 1135, "search_seeds": [1135, 1136, 1137, 1138],
    "population": 60, "evaluations_per_level": 1800, "min_evaluations": 300,
    "stop_lives": 19, "stop_wins": 6, "stall_evaluations": 700,
    "profile_stars_per_level": [3, 1.2, 13], "workers": 48,
    "validate_seeds": list(range(1001, 1011)) + list(range(1115, 1121)) + [1139, 1140, 1101, 1102], "validate_top": 12,
    "level_weights": {"22": 1, "21": 1},
    "warm_start": {"runs": [RUN3], "top": 20, "reevaluate": True},
    "rng_seed": "elite-v2-search-4",
}
job["max_wall_seconds"] = int(hours * 3600)
job["time_reserve_seconds"] = 2400
job["max_games"] = 1800 * 4 * 2 + 2 * 12 * 20 + 300
job["host_features"] = ["timed_clicks"]
job["episode_protocol"] = {**{k: v for k, v in job["episode_protocol"].items() if k != "click_ticks"}, "click_ticks": 9}
job["purpose"] = ("elite-v2 run 4: elite stages 22 (best run-3 plans reach wave 12-13 of 15, no win yet) and 21 (first "
                  "wins in run 3, best 5/20 on validation), warm-started from run 3's best 20 plans per level re-played "
                  "on 4 fresh train seeds (1135-1138), the rest of each population of 60 random; the best 12 validated on "
                  "20 train seeds (1001-1010, 1115-1120, 1139-1140, 1101-1102); option-A profile [3, 1.2, 13]; "
                  "human-timed trap clicks on (no clicks on elite stages)")
text = json.dumps(cfg, ensure_ascii=False, indent=1) + "\n"
if apply:
    p.write_text(text, encoding="utf-8", newline="\n")
else:
    Path(args["out"]).write_text(text, encoding="utf-8", newline="\n")
print("ok run 4 max_games", job["max_games"], "hours", hours)
