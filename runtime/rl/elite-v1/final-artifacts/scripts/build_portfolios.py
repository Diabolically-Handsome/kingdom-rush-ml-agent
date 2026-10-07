"""Retry portfolios from validated elite-v1 search runs (read-only), for plan A and plan B.

usage: build_portfolios.py <out.json> <plans per level> <run-8 id> [later option-A run ids, comma-separated]
Sources: elite B = run 5 (realistic campaign-only stars [2.6, 1.0]); elite A = run 8 (option-A profile [3, 1.0, 12]);
challenges: Heroic = run 6, Iron = run 7. Only games of the named run count for a level (profiles differ by run).
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, r"C:\Users\<user>\Documents\AlphaRush")
from alpharush_rl.search import check_genome  # noqa: E402

RUNS = Path(r"C:\Users\<user>\Documents\AlphaRush\runtime\rl\elite-v1\runs")
R5, R6, R7 = ("native-search-08f5b21f202b4b6596dbb2b4d5a8c619", "native-search-77f73818b99d43b98815025897d07bb3",
              "native-search-a9ceecb077f5459291a3768e0e2c6f84")
if len(sys.argv) < 4:
    raise SystemExit(__doc__)  # without run 8 the elite_A portfolios would silently stay empty
R8 = sys.argv[3]
# Later option-A runs (e.g. run 9 at [3, 1.2, 13]) replace run 8 for the levels they validated.
LATER_A = sys.argv[4].split(",") if len(sys.argv) > 4 else []
out_path = Path(sys.argv[1])
size = int(sys.argv[2])


def table(run_id):
    """{level: {genome_id: {"genome", "seeds": {seed: (won, lives)}}}} from search_eval + search_validate rows."""
    out = {}
    for line in (RUNS / run_id / "episodes.jsonl").read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if row["kind"] not in ("search_eval", "search_validate"):
            continue
        p = row["payload"]
        entry = out.setdefault(p["level"], {}).setdefault(p["genome_id"], {"genome": p["genome"], "seeds": {}})
        lives = (p["result"].get("outcome") or {}).get("lives") or 0
        entry["seeds"][p["seed"]] = (p["fitness"] >= 10000, lives)
    return out


def portfolio(entries, size):
    """Best smoothed win rate first (ties: mean lives when winning), then greedily the plan covering the most
    train seeds not yet covered; plans with an identical per-seed record are skipped."""
    rows = []
    for gid, e in entries.items():
        seeds = e["seeds"]
        if len(seeds) < 8:  # validated plans only (2 search seeds + 8 validation seeds)
            continue
        wins = sum(w for w, _ in seeds.values())
        if not wins:
            continue
        lives = [l for w, l in seeds.values() if w]
        rows.append({"genome_id": gid, "genome": check_genome(e["genome"]), "wins": wins, "games": len(seeds),
                     "won_seeds": {s for s, (w, _) in seeds.items() if w},
                     "signature": tuple(sorted((s, w) for s, (w, _) in seeds.items())),
                     "mean_lives": sum(lives) / len(lives)})
    rows.sort(key=lambda r: (-(r["wins"] + 1) / (r["games"] + 4), -r["mean_lives"], r["genome_id"]))
    chosen, covered, seen = [], set(), set()
    pool = list(rows)
    while pool and len(chosen) < size:
        best = max(pool, key=lambda r: (len(r["won_seeds"] - covered) if chosen else 0, -rows.index(r)))
        chosen.append(best)
        covered |= best["won_seeds"]
        seen.add(best["signature"])
        pool = [r for r in pool if r["signature"] not in seen]
    return chosen, covered


def evidence(r):
    return f"won {r['wins']} of {r['games']} practice games, {r['mean_lives']:.0f} lives left on average when winning"


result = {"elite_B": {}, "elite_A": {}, "challenges": {}, "report": {}}
for name, run_id, levels in (("elite_B", R5, range(13, 27)), ("elite_A", R8, range(13, 27))):
    if not run_id:
        continue
    t = table(run_id)
    if name == "elite_A" and LATER_A:
        merged = {}  # the later runs share one profile: a plan's seeds from all of them count together
        for later in LATER_A:
            for level, entries in table(later).items():
                for gid, e in entries.items():
                    m = merged.setdefault(level, {}).setdefault(gid, {"genome": e["genome"], "seeds": {}})
                    m["seeds"].update(e["seeds"])
        for level, entries in merged.items():
            if any(len(e["seeds"]) >= 8 for e in entries.values()):
                t[level] = entries
    for level in levels:
        chosen, covered = portfolio(t.get(level, {}), size)
        if chosen:
            result[name][str(level)] = [{"genome": r["genome"], "evidence": evidence(r)} for r in chosen]
            result["report"][f"{name}:{level}"] = {"plans": len(chosen), "best": f"{chosen[0]['wins']}/{chosen[0]['games']}",
                                                   "coverage": len(covered)}
for mode, run_id in ((2, R6), (3, R7)):
    t = table(run_id)
    for level in range(1, 13):
        chosen, covered = portfolio(t.get(level, {}), size)
        if chosen:
            key = f"{level}:{mode}"
            result["challenges"][key] = [{"genome": r["genome"], "evidence": evidence(r)} for r in chosen]
            result["report"][f"challenge {key}"] = {"plans": len(chosen), "best": f"{chosen[0]['wins']}/{chosen[0]['games']}",
                                                     "coverage": len(covered)}
out_path.write_text(json.dumps(result, indent=1), encoding="utf-8", newline="\n")
for key, value in result["report"].items():
    print(key, value)
