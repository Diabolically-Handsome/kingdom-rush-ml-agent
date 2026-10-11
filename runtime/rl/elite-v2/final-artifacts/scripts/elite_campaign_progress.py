"""Per-seed progress of the latest (or given) elite-v1 native-campaign / native-final run (read-only)."""
import collections
import json
import sys
from pathlib import Path

runs = Path(r"C:\Users\<user>\Documents\AlphaRush\runtime\rl\elite-v1\runs")
run = Path(sys.argv[1]) if len(sys.argv) > 1 else max(
    [p for p in runs.iterdir() if p.name.startswith(("native-campaign-", "native-final-"))], key=lambda p: p.stat().st_mtime)
rows = [json.loads(l) for l in (run / "episodes.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()] \
    if (run / "episodes.jsonl").exists() else []
seeds = collections.defaultdict(list)
decisions = collections.Counter()
errors = collections.Counter()
for r in rows:
    p = r["payload"]
    kind = r["kind"]
    if kind in ("campaign_attempt", "star_replay", "challenge_attempt"):
        fs = p["result"].get("final_summary") or {}
        tag = {"campaign_attempt": "", "star_replay": "r", "challenge_attempt": f"m{p.get('mode')}"}[kind]
        attempt = p.get("attempt", p.get("replay"))
        cell = f"L{p['level']}{tag}{'' if attempt in (0, None) else '#' + str(attempt + 1)}:"
        if p["won"]:
            cell += f"{p['stars']}*" if kind != "challenge_attempt" else "W"
        else:
            cell += f"x{fs.get('wave')}"
        seeds[p["seed"]].append((kind, p["level"], p["won"], p.get("stars", 0), cell))
    elif kind.endswith("_error"):
        errors[kind] += 1
        seeds[p["seed"]].append((kind, p["level"], None, 0, f"L{p['level']}:ERR"))
    elif kind == "strategy_decision":
        decisions[p["kind"]] += 1
print(run.name, "rows", len(rows), "8B decisions", dict(decisions), "errors", dict(errors))
for seed in sorted(seeds):
    won = {}
    for kind, level, w, stars, _ in seeds[seed]:
        if kind != "challenge_attempt" and w:
            won[level] = max(won.get(level, 0), stars)
    challenges = sum(1 for kind, *_rest in seeds[seed] if kind == "challenge_attempt" and _rest[1])
    elites = sorted(l for l in won if l > 12)
    print(f"seed {seed}: games {len(seeds[seed])}, campaign stars {sum(won.values())}, challenge wins {challenges}, "
          f"elite won {elites}")
    print("   " + " ".join(c for *_x, c in seeds[seed][-30:]))
