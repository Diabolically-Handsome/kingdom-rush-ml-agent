"""elite-v3b: the capture replay must reproduce the S1b steady base games exactly before any rich prompt is built.

usage: v3b_verify_replay.py <S1b run id> <replay run id>
Per (level, seed, genome_id): same trace_sha256, final_state_sha256, outcome, final tick; per consult: same consult
number, tick, wave, reason, plays and old-format prompt_sha256; and every replay consult carries capture + ledger +
translator. Exits non-zero on any difference."""
import json
import sys
from pathlib import Path

RUNS = Path(r"C:\Users\<user>\Documents\AlphaRush\runtime\rl\elite-v3\runs")


def rows(run_id, player):
    out = {}
    for line in (RUNS / run_id / "episodes.jsonl").read_text(encoding="utf-8").splitlines():
        if '"eval_episode"' not in line[:40]:
            continue
        r = json.loads(line)["payload"]
        if r.get("player") == player:
            out[(r["level"], r["seed"], r["genome_id"])] = r
    return out


base, replay = rows(sys.argv[1], "cmd_steady"), rows(sys.argv[2], "cmd_steady_capture")
problems, consults, games = [], 0, 0
if set(base) != set(replay):
    problems.append(f"task sets differ: {len(base)} base vs {len(replay)} replay")
for key in sorted(set(base) & set(replay)):
    b, r = base[key]["result"], replay[key]["result"]
    games += 1
    for field in ("trace_sha256", "final_state_sha256", "final_tick", "outcome", "status"):
        if b.get(field) != r.get(field):
            problems.append(f"{key}: {field} differs")
    bc, rc = b["commander"]["consults"], r["commander"]["consults"]
    if len(bc) != len(rc):
        problems.append(f"{key}: {len(bc)} vs {len(rc)} consults")
    for x, y in zip(bc, rc):
        consults += 1
        for field in ("consult", "tick", "wave", "reason", "group", "plays", "prompt_sha256", "briefing", "choice"):
            if x.get(field) != y.get(field):
                problems.append(f"{key} consult {x.get('consult')}: {field} differs")
        if not all(k in y for k in ("capture", "ledger", "translator")):
            problems.append(f"{key} consult {y.get('consult')}: capture missing")
print(json.dumps({"games": games, "consults": consults, "problems": len(problems)}))
for p in problems[:40]:
    print(" ", p)
sys.exit(1 if problems else 0)
