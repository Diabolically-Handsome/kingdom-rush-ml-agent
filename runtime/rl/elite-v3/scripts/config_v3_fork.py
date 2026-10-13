"""elite-v3 fork pilot: on a set of tasks, every consult point of the steady (= executor alone) game is branched to every
other play once, all other consults steady, so each play's effect at each moment is measured on the same seed.

usage: config_v3_fork.py <S1 run id> (out=<file> | apply=1) [levels=15,17,19,20,21,22,26] [seeds=1201-1203] [plans=1]
Reads the S1 run's cmd_steady rows (commander consult records: plays offered at each consult). Train seeds only;
development evidence. apply=1 writes the live elite-v3 config and is refused while any runtime/rl/*/job.lock exists."""
import json
import sys
from pathlib import Path

ROOT = Path(r"C:\Users\<user>\Documents\AlphaRush")
SCRATCH = Path(__file__).parent
run_id = sys.argv[1]
args = dict(a.split("=", 1) for a in sys.argv[2:])
apply = args.get("apply") == "1"
if apply == ("out" in args):
    raise SystemExit("give exactly one of out=<file> or apply=1")
if apply and any((ROOT / "runtime/rl").glob("*/job.lock")):
    raise SystemExit("refused: a job holds a lock")
levels = [int(x) for x in args.get("levels", "15,17,19,20,21,22,26").split(",")]
lo, hi = (int(x) for x in args.get("seeds", "1201-1203").split("-"))
seeds = set(range(lo, hi + 1))
n = int(args.get("plans", 1))
portfolios = json.loads((SCRATCH / "portfolios_v2_r3.json").read_text(encoding="utf-8"))["elite"]
wanted = {(level, json.dumps(e["genome"], sort_keys=True)) for level in levels for e in portfolios[str(level)][:n]}
rows = [json.loads(line)["payload"] for line in (ROOT / "runtime/rl/elite-v3/runs" / run_id / "episodes.jsonl")
        .read_text(encoding="utf-8").splitlines() if line.strip()]
p = ROOT / "configs/phases/elite-v3.json"
cfg = json.loads(p.read_text(encoding="utf-8"))
s1_tasks = cfg["jobs"]["native-eval"]["eval"]["tasks"]  # the S1 job's tasks: journal rows carry their index
assert any(r.get("player") == "cmd_steady" for r in rows), "not an S1 run with a steady arm"
tasks, base_games = [], 0
for row in rows:
    if row.get("player") != "cmd_steady" or row["seed"] not in seeds:
        continue
    genome = s1_tasks[row["index"]]["genome"]
    assert s1_tasks[row["index"]]["level"] == row["level"] and s1_tasks[row["index"]]["seed"] == row["seed"]
    key = (row["level"], json.dumps(genome, sort_keys=True))
    if key not in wanted:
        continue
    base_games += 1
    for consult in row["result"]["commander"]["consults"]:
        for play in range(1, len(consult["plays"])):
            tasks.append({"level": row["level"], "seed": row["seed"], "genome": genome,
                          "fork": {"base": "steady", "consult": consult["consult"], "play": play}})
job = cfg["jobs"]["native-eval"]
job["eval"] = {"tasks": tasks, "workers": 32, "profile_stars_per_level": [3, 1.2, 13], "players": ["cmd_fork"]}
job["max_games"] = len(tasks) + 10
job["max_wall_seconds"] = 14400
job["purpose"] = (f"elite-v3 fork pilot (development evidence, train seeds {lo}-{hi} only): for {base_games} steady games "
                  f"of S1 run {run_id} (elite stages {levels}, first {n} portfolio plan(s)), every consult is branched to "
                  "every other offered play once (all other consults steady), measuring each play's effect at each "
                  "moment on the same seed; base brain steady = the executor alone")
text = json.dumps(cfg, ensure_ascii=False, indent=1) + "\n"
if apply:
    p.write_text(text, encoding="utf-8", newline="\n")
else:
    Path(args["out"]).write_text(text, encoding="utf-8", newline="\n")
print("ok", base_games, "base games ->", len(tasks), "fork games")
