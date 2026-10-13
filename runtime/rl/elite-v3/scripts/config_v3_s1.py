"""elite-v3 S1 headroom probe: the commander's arms on the weak elite stages, train seeds only (development evidence).

usage: config_v3_s1.py (out=<file> | apply=1) [seeds=1201-1210] [plans=3] [players=cmd_passive,cmd_rule,cmd_random,cmd_placebo]
Levels 15, 17, 19, 20, 21, 22, 26; each level's first N portfolio plans (elite-v2 final portfolio); action scope v4 with
the final's human-timed clicks; the plan executor inside every arm (commander arms wrap it). apply=1 writes the live
elite-v3 config and is refused while any runtime/rl/*/job.lock exists."""
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
lo, hi = (int(x) for x in args.get("seeds", "1201-1210").split("-"))
seeds = list(range(lo, hi + 1))
n = int(args.get("plans", 3))
players = args.get("players", "cmd_passive,cmd_rule,cmd_random,cmd_placebo").split(",")
LEVELS = [15, 17, 19, 20, 21, 22, 26]
portfolios = json.loads((SCRATCH / "portfolios_v2_r3.json").read_text(encoding="utf-8"))["elite"]
tasks = []
for level in LEVELS:
    for entry in portfolios[str(level)][:n]:
        tasks += [{"level": level, "seed": s, "genome": entry["genome"]} for s in seeds]
p = ROOT / "configs/phases/elite-v3.json"
cfg = json.loads(p.read_text(encoding="utf-8"))
job = cfg["jobs"]["native-eval"]
job.update({"enabled": True, "action_scope": "v4", "max_wall_seconds": 10800, "total_wall_seconds": 43200,
            "max_jobs": 16, "max_games": len(tasks) * len(players) + 10, "host_features": ["timed_clicks"],
            "episode_protocol": {**{k: v for k, v in job["episode_protocol"].items() if k != "click_ticks"},
                                 "click_ticks": 9},
            "eval": {"tasks": tasks, "workers": 32, "profile_stars_per_level": [3, 1.2, 13], "players": players}})
job.pop("operator_weights", None)
job["purpose"] = (f"elite-v3 S1 headroom probe (development evidence, train seeds {lo}-{hi} only): commander arms {players} "
                  f"wrapping the plan executor on elite stages {LEVELS}, the first {n} elite-v2 portfolio plans of each; "
                  "action scope v4 (free Rain of Fire / reinforcement targets, hero movement where available); passive = "
                  "today's behaviour, placebo = the rule commander delayed by one tick (pure-chaos floor)")
text = json.dumps(cfg, ensure_ascii=False, indent=1) + "\n"
if apply:
    p.write_text(text, encoding="utf-8", newline="\n")
else:
    Path(args["out"]).write_text(text, encoding="utf-8", newline="\n")
print("ok", len(tasks), "tasks x", len(players), "players =", len(tasks) * len(players), "games")
