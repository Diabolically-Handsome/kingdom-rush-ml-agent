"""elite-v2 main-campaign evals on train seeds 1111-1120 (never used to train any main operator): the first plan of
levels 1-10 and 12 and all level-11 portfolio plans, optionally with the human-timing host features and the level-11
plans' early=2 twins.
usage: config_v2_eval.py weights=<main operator weights> [players=plan,operator] [features=timed_clicks,early_waves]
                         [click_ticks=9] [early2=1] [l12=all] [only=11,12] (out=<file> | apply=1)
l12=all plays all 8 level-12 portfolio plans (default: the first); only= keeps just these levels' tasks.
apply=1 writes the live elite-v2 config and is refused while any runtime/rl/*/job.lock exists."""
import copy
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(r"C:\Users\<user>\Documents\AlphaRush")
SCRATCH = Path(__file__).parent
args = dict(a.split("=", 1) for a in sys.argv[1:])
apply = args.get("apply") == "1"
if apply == ("out" in args):
    raise SystemExit("give exactly one of out=<file> or apply=1")
if apply and any((ROOT / "runtime/rl").glob("*/job.lock")):
    raise SystemExit("refused: a job holds a lock")
weights = args["weights"]
players = args.get("players", "plan,operator").split(",")
features = [f for f in args.get("features", "").split(",") if f]
click_ticks = int(args["click_ticks"]) if "click_ticks" in args else None
early2 = args.get("early2") == "1"
if early2 and "early_waves" not in features:
    raise SystemExit("early2 needs the early_waves feature")

v1 = json.loads((ROOT / "configs/phases/campaign-v1.json").read_text(encoding="utf-8"))
main = v1["jobs"]["native-final"]["campaign"]["plans"]
l11 = json.loads((SCRATCH / "v2_l11_portfolio.json").read_text(encoding="utf-8"))
seeds = list(range(1111, 1121))
tasks = []
for level in list(range(1, 11)) + [12]:
    for e in main[str(level)][:(None if level == 12 and args.get("l12") == "all" else 1)]:
        genome = e["genome"] if isinstance(e, dict) and "genome" in e else e
        tasks += [{"level": level, "seed": s, "genome": genome} for s in seeds]
for p in l11:
    tasks += [{"level": 11, "seed": s, "genome": p["genome"]} for s in seeds]
    if early2 and p["genome"].get("early") != 2:
        twin = copy.deepcopy(p["genome"])
        twin["early"] = 2
        tasks += [{"level": 11, "seed": s, "genome": twin} for s in seeds]

if "only" in args:
    keep = {int(x) for x in args["only"].split(",")}
    tasks = [t for t in tasks if t["level"] in keep]
p = ROOT / "configs/phases/elite-v2.json"
cfg = json.loads(p.read_text(encoding="utf-8"))
job = cfg["jobs"]["native-eval"]
job.update({"enabled": True, "max_wall_seconds": 7200, "total_wall_seconds": 43200,
            "max_games": len(players) * len(tasks) + 10,
            "operator_weights": {"path": weights, "sha256": hashlib.sha256((ROOT / weights).read_bytes()).hexdigest()},
            "eval": {"tasks": tasks, "workers": 32, "profile_stars_per_level": [2.75, 1.0], "players": players}})
job.pop("host_features", None)
if features:
    job["host_features"] = features
protocol = dict(job["episode_protocol"])
protocol.pop("click_ticks", None)
if click_ticks:
    protocol["click_ticks"] = click_ticks
job["episode_protocol"] = protocol
what = (f"with the human-timing host features {features}" + (f" and {click_ticks}-tick trap clicks" if click_ticks else "")
        + (" plus the early=2 twin of each level-11 plan" if early2 else "")) if features else "without host features"
what += ("; all 8 level-12 portfolio plans" if args.get("l12") == "all" else "") + (
    f"; only levels {args['only']}" if "only" in args else "")
job["purpose"] = (f"elite-v2 main-campaign check {what}: players {players} (operator network {Path(weights).stem}) "
                  "play the first portfolio plan of levels 1-10 and 12 and the 7 new level-11 plans on train seeds "
                  "1111-1120, which no main operator was trained on; a new main network or host feature is adopted "
                  "only if it does not lose ground on any level")
text = json.dumps(cfg, ensure_ascii=False, indent=1) + "\n"
if apply:
    p.write_text(text, encoding="utf-8", newline="\n")
else:
    Path(args["out"]).write_text(text, encoding="utf-8", newline="\n")
print("ok", len(tasks), "tasks x", len(players), "players;", what)
