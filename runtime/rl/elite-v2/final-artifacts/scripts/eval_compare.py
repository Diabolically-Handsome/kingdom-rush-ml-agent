"""Read-only comparison of elite-v2 native-eval runs, per level and plan: wins/games and mean winning lives.
usage: eval_compare.py <label>=<run id>[:player] ...   (player defaults to every player in the run)
Each column is one (run, player); rows are (level, genome id); the last rows sum each column per level."""
import collections
import json
import sys
from pathlib import Path

RUNS = Path(r"C:\Users\<user>\Documents\AlphaRush\runtime\rl\elite-v2\runs")
cols = []
for arg in sys.argv[1:]:
    label, ref = arg.split("=", 1)
    run, _, player = ref.partition(":")
    rows = [json.loads(line) for line in (RUNS / run / "episodes.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()]
    rows = [r["payload"] for r in rows if r["kind"] == "eval_episode"]
    players = [player] if player else sorted({r["player"] for r in rows})
    for pl in players:
        table = collections.defaultdict(list)
        for r in rows:
            if r["player"] == pl:
                table[(r["level"], r["genome_id"])].append((r["won"], r["lives"], r["seed"]))
        cols.append((f"{label}:{pl}", table))
keys = sorted({k for _, t in cols for k in t})
head = f"{'level':>5} {'plan':16}" + "".join(f"{c:>18}" for c, _ in cols)
print(head)
for key in keys:
    cells = []
    for _, t in cols:
        games = t.get(key, [])
        wins = [l for w, l, _ in games if w]
        cells.append(f"{len(wins)}/{len(games)} ({sum(wins) / len(wins):.1f})" if wins else f"0/{len(games)}" if games else "-")
    print(f"{key[0]:>5} {key[1]:16}" + "".join(f"{c:>18}" for c in cells))
print()
for level in sorted({k[0] for k in keys}):
    cells = []
    for _, t in cols:
        games = [g for k, v in t.items() if k[0] == level for g in v]
        wins = sum(w for w, _, _ in games)
        cells.append(f"{wins}/{len(games)}" if games else "-")
    print(f"{level:>5} {'ALL':16}" + "".join(f"{c:>18}" for c in cells))
