"""elite-v2 retry portfolios (read-only): every level's validated plans from the option-A elite searches of elite-v1
(runs 8-13) and all elite-v2 searches, merged per plan, behavioural twins dropped (same trace on every shared game),
then greedily by train-seed coverage. Level 11 from elite-v2 search 1 (main campaign profile). Challenges as in
elite-v1 (scratchpad portfolios.json). Train seeds only (1001-1010, 1101-1140).

usage: build_portfolios_v2.py <out.json> [plans per level]
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, r"C:\Users\<user>\Documents\AlphaRush")
from alpharush_rl.search import check_genome, genome_id  # noqa: E402

ROOT = Path(r"C:\Users\<user>\Documents\AlphaRush")
V1 = ROOT / "runtime/rl/elite-v1/runs"
V2 = ROOT / "runtime/rl/elite-v2/runs"
# elite-v1 searches at the option-A profiles ([3, 1.0, 12] and [3, 1.2, 13]); earlier runs used other star budgets.
V1_A = ["native-search-1d4786a9538b4ba7bce718a29f7fca4a", "native-search-c23208d7fb2d4362956640f540c6af48",
        "native-search-55aeddcb01824591bbf2c0fb950c669f", "native-search-f7e0e33fbba849399e9bda62a0b629de",
        "native-search-fddef745dc724eb194892eb667e9393b", "native-search-61355d05d9e04a9eb32fe22e2742e3b8"]
TRAIN = set(range(1001, 1011)) | set(range(1101, 1141))
out_path = Path(sys.argv[1])
size = int(sys.argv[2]) if len(sys.argv) > 2 else 10


def records(run_dirs):
    """{level: {genome_id: {"genome", "games": {(run, seed): (won, lives, trace)}}}}; mode-1 search games only."""
    table = {}
    for run in run_dirs:
        events = (run / "events.jsonl").read_text(encoding="utf-8").splitlines()
        start = next((json.loads(e)["payload"] for e in events if '"search_start"' in e), {})
        if start.get("spec", {}).get("mode", 1) != 1:
            continue
        for line in (run / "episodes.jsonl").read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            if row["kind"] not in ("search_eval", "search_validate"):
                continue
            p = row["payload"]
            if p["seed"] not in TRAIN:
                continue
            entry = table.setdefault(p["level"], {}).setdefault(p["genome_id"], {"genome": p["genome"], "games": {}})
            lives = (p["result"].get("outcome") or {}).get("lives") or 0
            entry["games"][(run.name, p["seed"])] = (p["fitness"] >= 10000, lives, p["result"].get("trace_sha256"))
    return table


def twin(a, b):
    shared = set(a) & set(b)
    return bool(shared) and all(a[k][2] is not None and a[k][2] == b[k][2] for k in shared)


def portfolio(entries, size, min_games=8):
    rows = []
    for gid, e in entries.items():
        games = e["games"]
        seeds_won = {s for (_, s), (w, _, _) in games.items() if w}
        n = len({s for _, s in games})
        if n < min_games or not seeds_won:
            continue
        lives = [l for w, l, _ in games.values() if w]
        rows.append({"gid": gid, "genome": check_genome(e["genome"]), "games": games, "won_seeds": seeds_won,
                     "seeds": n, "wins": len(seeds_won), "mean_lives": sum(lives) / len(lives)})
    rows.sort(key=lambda r: (-(r["wins"] + 1) / (r["seeds"] + 4), -r["mean_lives"], r["gid"]))
    chosen, covered = [], set()
    pool = list(rows)
    while pool and len(chosen) < size:
        best = max(pool, key=lambda r: (len(r["won_seeds"] - covered) if chosen else 0, -rows.index(r)))
        pool.remove(best)
        if any(twin(best["games"], c["games"]) for c in chosen):
            continue
        chosen.append(best)
        covered |= best["won_seeds"]
    return chosen, covered


def evidence(r):
    return (f"won {r['wins']} of {r['seeds']} practice seeds, {r['mean_lives']:.0f} lives left on average when "
            f"winning; hero {(r['genome']['hero'] or 'none').replace('hero_', '')}")


v2_runs = sorted(p for p in V2.glob("native-search-*") if (p / "episodes.jsonl").exists())
table = records([V1 / r for r in V1_A] + v2_runs)
result = {"elite": {}, "report": {}}
for level in [11] + list(range(13, 27)):
    entries = table.get(level, {})
    if level == 11:  # only elite-v2's main-campaign-profile games count for level 11
        entries = {g: {**e, "games": {k: v for k, v in e["games"].items() if k[0] in {r.name for r in v2_runs}}}
                   for g, e in entries.items()}
    chosen, covered = portfolio(entries, size)
    if chosen:
        result["elite"][str(level)] = [{"genome": r["genome"], "evidence": evidence(r)} for r in chosen]
        seeds_seen = {s for e in entries.values() for _, s in e["games"]}
        result["report"][str(level)] = {"plans": len(chosen), "best": f"{chosen[0]['wins']}/{chosen[0]['seeds']}",
                                        "coverage": f"{len(covered)}/{len(seeds_seen)}",
                                        "heroes": sorted({(r['genome']['hero'] or 'none') for r in chosen})}
old = json.loads((Path(__file__).parent / "portfolios.json").read_text(encoding="utf-8"))
result["challenges"] = old["challenges"]
out_path.write_text(json.dumps(result, indent=1), encoding="utf-8", newline="\n")
for level, rep in result["report"].items():
    print(level, rep)
