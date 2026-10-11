"""elite-v2: level-11 demonstrations / DAgger for the main-campaign operator (layout v3). usage: <beta>"""
import hashlib, json, sys
from pathlib import Path
ROOT = Path(r"C:\Users\<user>\Documents\AlphaRush")
SCRATCH = Path(__file__).parent
if any((ROOT / "runtime/rl").glob("*/job.lock")):
    raise SystemExit("refused: a job holds a lock")
beta = float(sys.argv[1])
weights = sys.argv[2] if len(sys.argv) > 2 else "runtime/rl/campaign-v1/models/operator-d86c06c72ae1.json"
plans = json.loads((SCRATCH / "v2_l11_portfolio.json").read_text(encoding="utf-8"))
tasks = [{"level": 11, "seed": s, "genome": p["genome"]} for p in plans
         for s in list(range(1001, 1011)) + list(range(1101, 1111))]
p = ROOT / "configs/phases/elite-v2.json"
cfg = json.loads(p.read_text(encoding="utf-8"))
job = cfg["jobs"]["native-collect"]
job.update({"enabled": True, "max_wall_seconds": 3600, "total_wall_seconds": 43200, "max_games": len(tasks) + 20,
            "operator_weights": {"path": weights, "sha256": hashlib.sha256((ROOT / weights).read_bytes()).hexdigest()},
            "collect": {"tasks": tasks, "workers": 32, "profile_stars_per_level": [2.75, 1.0], "dagger": {"beta": beta}}})
job["purpose"] = (f"elite-v2: main level 11 portfolio plans (run 1, 7 distinct plans) on 20 train seeds, recorded for the "
                  f"main-campaign operator network (layout v3); DAgger beta {beta} (1.0: the plan executor plays and "
                  "labels every decision; 0: the network plays, the executor labels)")
p.write_text(json.dumps(cfg, ensure_ascii=False, indent=1) + "\n", encoding="utf-8", newline="\n")
print("ok", len(tasks), "tasks, beta", beta)
