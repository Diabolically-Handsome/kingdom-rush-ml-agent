"""Per-seed progress of the latest (or given) native-campaign / native-final run: level -> attempts, stars."""
import collections
import json
import sys
from pathlib import Path

runs = Path(r"C:\Users\<user>\Documents\AlphaRush\runtime\rl\campaign-v1\runs")
run = Path(sys.argv[1]) if len(sys.argv) > 1 else max(
    [p for p in runs.iterdir() if p.name.startswith(("native-campaign-", "native-final-"))], key=lambda p: p.stat().st_mtime)
rows = [json.loads(l) for l in (run / "episodes.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()] \
    if (run / "episodes.jsonl").exists() else []
seeds = collections.defaultdict(list)
decisions = collections.Counter()
for r in rows:
    p = r["payload"]
    if r["kind"] == "campaign_attempt":
        o = p["result"].get("outcome") or {}
        fs = p["result"].get("final_summary") or {}
        seeds[p["seed"]].append((p["level"], p["attempt"], p["won"], p["stars"], o.get("lives"), fs.get("wave")))
    elif r["kind"] == "campaign_error":
        seeds[p["seed"]].append((p["level"], p["attempt"], None, 0, None, "ERR"))
    elif r["kind"] == "strategy_decision":
        decisions[p["kind"]] += 1
print(run.name, "rows", len(rows), "8B decisions", dict(decisions))
for seed in sorted(seeds):
    cells = []
    for level, attempt, won, stars, lives, wave in seeds[seed]:
        cells.append(f"L{level}{'' if attempt == 0 else '#' + str(attempt + 1)}:" +
                     (f"{stars}*" if won else ("ERR" if won is None else f"lost@w{wave}")))
    won_levels = {lv for lv, _, w, *_ in seeds[seed] if w}
    print(seed, f"cleared {len(won_levels)}/12", " ".join(cells))
for name in ("survey-summary.json", "supervisor-receipt.json"):
    if (run / name).exists():
        d = json.loads((run / name).read_text(encoding="utf-8"))
        print(name, d.get("status"), d.get("stopped_reason"), d.get("games_played"))
