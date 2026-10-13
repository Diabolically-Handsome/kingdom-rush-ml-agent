"""elite-v3b: deterministic replay of the 56 S1b steady base games with the screen captured at every consult
(player cmd_steady_capture = the steady brain + capture). Same tasks as S1b (levels 15,17,19,20,21,22,26; first 2
portfolio plans; train seeds 1201-1204), same scope/protocol/host features. Also appends the owner's elite-v3b approval
to the phase authorization list (once).

usage: config_v3b_replay.py (out=<file> | apply=1)
apply=1 writes the live elite-v3 config and is refused while any runtime/rl/*/job.lock exists."""
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
APPROVAL_V3B = "[用户原话已省略 / user's message omitted]"
LEVELS, SEEDS, PLANS = [15, 17, 19, 20, 21, 22, 26], list(range(1201, 1205)), 2
portfolios = json.loads((SCRATCH / "portfolios_v2_r3.json").read_text(encoding="utf-8"))["elite"]
tasks = []
for level in LEVELS:
    for entry in portfolios[str(level)][:PLANS]:
        tasks += [{"level": level, "seed": s, "genome": entry["genome"]} for s in SEEDS]
p = ROOT / "configs/phases/elite-v3.json"
cfg = json.loads(p.read_text(encoding="utf-8"))
if APPROVAL_V3B not in cfg["authorization"]:
    cfg["authorization"].append(APPROVAL_V3B)
job = cfg["jobs"]["native-eval"]
job.update({"enabled": True, "action_scope": "v4", "max_wall_seconds": 3600, "max_games": len(tasks) + 4,
            "host_features": ["timed_clicks"],
            "eval": {"tasks": tasks, "workers": 32, "profile_stars_per_level": [3, 1.2, 13],
                     "players": ["cmd_steady_capture"]}})
assert job["episode_protocol"]["click_ticks"] == 9 and job["action_scope"] == "v4"
job["purpose"] = ("elite-v3b replay with capture (development evidence, train seeds 1201-1204 only): the 56 S1b steady "
                  "base games replayed by the steady brain with the screen recorded at every consult (towers, enemies, "
                  "heroes, flag preview, spell buttons) and a memory ledger (lives lost per path, hero deaths), so the "
                  "fair re-test's rich briefings describe exactly the moments whose fork outcomes are already measured; "
                  "must reproduce S1b exactly (trace hashes, outcomes, 992/992 old prompt hashes)")
text = json.dumps(cfg, ensure_ascii=False, indent=1) + "\n"
if apply:
    p.write_text(text, encoding="utf-8", newline="\n")
else:
    Path(args["out"]).write_text(text, encoding="utf-8", newline="\n")
print("ok", len(tasks), "tasks")
