"""elite-v2: main-campaign protection check on train seeds 1111-1120 (not used to train any main operator): the plan
executor and a main operator network play the first plan of levels 1-10 and 12 and all level-11 portfolio plans.
usage: config_v2_eval_main.py <main operator weights path>   (apply only between jobs)"""
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(r"C:\Users\<user>\Documents\AlphaRush")
SCRATCH = Path(__file__).parent
if any((ROOT / "runtime/rl").glob("*/job.lock")):
    raise SystemExit("refused: a job holds a lock")
weights = sys.argv[1]
v1 = json.loads((ROOT / "configs/phases/campaign-v1.json").read_text(encoding="utf-8"))
main = v1["jobs"]["native-final"]["campaign"]["plans"]
l11 = json.loads((SCRATCH / "v2_l11_portfolio.json").read_text(encoding="utf-8"))
seeds = list(range(1111, 1121))
tasks = []
for level in list(range(1, 11)) + [12]:
    e = main[str(level)][0]
    genome = e["genome"] if isinstance(e, dict) and "genome" in e else e
    tasks += [{"level": level, "seed": s, "genome": genome} for s in seeds]
for p in l11:
    tasks += [{"level": 11, "seed": s, "genome": p["genome"]} for s in seeds]
p = ROOT / "configs/phases/elite-v2.json"
cfg = json.loads(p.read_text(encoding="utf-8"))
job = cfg["jobs"]["native-eval"]
job.update({"enabled": True, "max_wall_seconds": 5400, "total_wall_seconds": 43200,
            "max_games": 2 * len(tasks) + 10,
            "operator_weights": {"path": weights, "sha256": hashlib.sha256((ROOT / weights).read_bytes()).hexdigest()},
            "eval": {"tasks": tasks, "workers": 32, "profile_stars_per_level": [2.75, 1.0],
                     "players": ["plan", "operator"]}})
job["purpose"] = ("elite-v2 main-campaign protection check: the plan executor and the main operator network "
                  f"{Path(weights).stem} play the first portfolio plan of levels 1-10 and 12 and the 7 new level-11 plans "
                  "on train seeds 1111-1120, which no main operator was trained on; a new main network replaces "
                  "campaign-v1's only if it does not lose ground on any level")
p.write_text(json.dumps(cfg, ensure_ascii=False, indent=1) + "\n", encoding="utf-8", newline="\n")
print("ok", len(tasks), "tasks x 2 players")
