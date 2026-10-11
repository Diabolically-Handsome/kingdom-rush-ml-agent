"""Read-only: L22 plans of the overnight search, best first: hero, first steps, block, early; wave reached."""
import collections, json, sys
from pathlib import Path
run = Path(sys.argv[1])
plans = {}
for line in (run / "episodes.jsonl").open(encoding="utf-8"):
    if '"level":22' not in line and '"level": 22' not in line:
        continue
    r = json.loads(line)
    p = r["payload"]
    if r["kind"] != "search_eval" or p["level"] != 22:
        continue
    e = plans.setdefault(p["genome_id"], {"g": p["genome"], "f": [], "w": []})
    e["f"].append(p["fitness"]); e["w"].append((p["result"].get("final_summary") or {}).get("wave"))
rows = sorted(plans.values(), key=lambda e: -sum(e["f"]) / len(e["f"]))
first = collections.Counter()
for e in rows:
    s = e["g"]["steps"]
    first[tuple(tuple(x) for x in s[:1])] += 1
print("plans", len(rows), "first-step counts", first.most_common(8))
heroes = collections.Counter(e["g"]["hero"] for e in rows)
print("heroes", heroes.most_common())
for e in rows[:15]:
    g = e["g"]
    print(round(sum(e["f"]) / len(e["f"])), e["w"], g["hero"], "block", g.get("block"), "early", g["early"], "steps", g["steps"][:6])
