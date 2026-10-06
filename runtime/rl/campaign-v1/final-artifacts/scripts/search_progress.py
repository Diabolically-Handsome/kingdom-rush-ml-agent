"""Read-only progress of the latest native-search run: per level evaluations, wins, best lives."""
import json, sys, collections, time
from pathlib import Path
runs = Path(r"C:\Users\<user>\Documents\AlphaRush\runtime\rl\campaign-v1\runs")
run = Path(sys.argv[1]) if len(sys.argv) > 1 else max(runs.glob("native-search-*"), key=lambda p: p.stat().st_mtime)
rows = [json.loads(l) for l in (run / "episodes.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
by = collections.defaultdict(list); errs = collections.Counter(); reps = []; val = collections.defaultdict(list)
for r in rows:
    p = r["payload"]
    if r["kind"] == "search_eval": by[p["level"]].append(p["fitness"])
    elif r["kind"] == "search_validate": val[p["level"]].append(p["fitness"] >= 10000)
    elif r["kind"] == "episode_error": errs[p["error"][:80]] += 1
    elif r["kind"] == "replay": reps.append(p["replay_verified"])
age = time.time() - (run / "episodes.jsonl").stat().st_mtime
print(run.name, "rows", len(rows), "errors", dict(errs), "replays ok", reps.count(True), "/", len(reps), "last write %.0fs ago" % age)
for level in sorted(by):
    f = by[level]; wins = [x for x in f if x >= 10000]
    print(f"L{level:>2} evals {len(f):>4} wins {len(wins):>4} best_lives {int((max(f) - 10000) // 100) if wins else '-':>3} best_lost_wave {int(max(x for x in f if x < 10000) // 100) + 1 if any(x < 10000 for x in f) else '-'}"
          + (f"  validate {sum(val[level])}/{len(val[level])}" if val[level] else ""))
for name in ("survey-summary.json", "supervisor-receipt.json"):
    if (run / name).exists():
        d = json.loads((run / name).read_text(encoding="utf-8")); print(name, d.get("status"), d.get("stopped_reason"), d.get("games_played"))
