"""Read-only progress of an elite-v1 native-search run: per level evaluations, wins (per seed and per plan), validation."""
import collections
import json
import sys
import time
from pathlib import Path

runs = Path(r"C:\Users\<user>\Documents\AlphaRush\runtime\rl\elite-v1\runs")
run = Path(sys.argv[1]) if len(sys.argv) > 1 else max(runs.glob("native-search-*"), key=lambda p: p.stat().st_mtime)
rows = [json.loads(line) for line in (run / "episodes.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
evals = collections.defaultdict(list)
plans = collections.defaultdict(lambda: collections.defaultdict(list))
val = collections.defaultdict(lambda: collections.defaultdict(list))
errs, reps, waves = collections.Counter(), [], collections.defaultdict(collections.Counter)
stalls = collections.Counter()
for r in rows:
    p = r["payload"]
    if r["kind"] == "search_eval":
        evals[p["level"]].append(p["fitness"])
        plans[p["level"]][p["genome_id"]].append(p["fitness"])
        res = p["result"]
        if p["fitness"] < 10000:
            waves[p["level"]][(res.get("final_summary") or {}).get("wave")] += 1
        if res.get("stall_reason"):
            stalls[(p["level"], res["stall_reason"])] += 1
    elif r["kind"] == "search_validate":
        val[p["level"]][p["genome_id"]].append(p["fitness"] >= 10000)
    elif r["kind"] == "episode_error":
        errs[p["error"][:100]] += 1
    elif r["kind"] == "replay":
        reps.append(p["replay_verified"])
age = time.time() - (run / "episodes.jsonl").stat().st_mtime
print(run.name, "rows", len(rows), "errors", dict(errs), "replays ok", reps.count(True), "/", len(reps),
      "last write %.0fs ago" % age)
for level in sorted(evals):
    f = evals[level]
    wins = [x for x in f if x >= 10000]
    full = [g for g, fs in plans[level].items() if len(fs) >= 2 and all(x >= 10000 for x in fs)]
    best_mean = max((sum(fs) / len(fs) for fs in plans[level].values() if len(fs) >= 2), default=None)
    lost_waves = dict(sorted(waves[level].items(), key=lambda kv: (kv[0] is None, kv[0] or 0)))
    line = (f"L{level:>2} evals {len(f):>4} seed-wins {len(wins):>4} plans-winning-all {len(full):>3} "
            f"best_mean {best_mean if best_mean is None else round(best_mean)} lost_at_wave {lost_waves}")
    if val[level]:
        line += "  validate " + " ".join(f"{sum(v)}/{len(v)}" for v in sorted(val[level].values(), key=lambda v: -sum(v)))
    print(line)
if stalls:
    print("stalls", dict(stalls))
for name in ("survey-summary.json", "supervisor-receipt.json"):
    if (run / name).exists():
        d = json.loads((run / name).read_text(encoding="utf-8"))
        print(name, d.get("status"), d.get("stopped_reason"), d.get("games_played"))
