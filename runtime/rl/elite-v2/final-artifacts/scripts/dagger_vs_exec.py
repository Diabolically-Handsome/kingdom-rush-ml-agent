"""Read-only: per level, wins of a native-collect run (network plays, DAgger) next to the plan executor's recorded
search/validation result for the same plan and seed (elite-v2 searches; elite-v1 option-A searches as fallback)."""
import collections, json, sys
from pathlib import Path
R = Path(r"C:\Users\<user>\Documents\AlphaRush\runtime\rl")
run = R / "elite-v2/runs" / sys.argv[1]
search_runs = sorted((R / "elite-v2/runs").glob("native-search-*")) + sorted((R / "elite-v1/runs").glob("native-search-*"))
want = {}
rows = []
for line in (run / "episodes.jsonl").open(encoding="utf-8"):
    r = json.loads(line)
    if r["kind"] != "collect_episode":
        continue
    p = r["payload"]
    rows.append(p)
    want[(p["level"], p["genome_id"], p["seed"])] = None
for sr in search_runs:
    ev = (sr / "events.jsonl")
    for line in (sr / "episodes.jsonl").open(encoding="utf-8"):
        if '"search_eval"' not in line[:40] and '"search_validate"' not in line[:40]:
            continue
        r = json.loads(line); p = r["payload"]
        k = (p["level"], p["genome_id"], p["seed"])
        if k in want and want[k] is None:
            want[k] = p["fitness"] >= 10000
net = collections.Counter(); ex = collections.Counter(); n = collections.Counter(); m = collections.Counter()
for p in rows:
    k = (p["level"], p["genome_id"], p["seed"])
    won = bool((p["result"].get("outcome") or {}).get("level_won"))
    n[p["level"]] += 1; net[p["level"]] += won
    if want[k] is not None:
        m[p["level"]] += 1; ex[p["level"]] += want[k]
for lv in sorted(n):
    print(lv, f"network {net[lv]}/{n[lv]}", f"executor (recorded) {ex[lv]}/{m[lv]}")
