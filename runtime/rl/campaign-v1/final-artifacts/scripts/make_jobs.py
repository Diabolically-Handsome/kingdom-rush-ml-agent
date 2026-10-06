"""Build collect/eval/campaign job bodies from native-search results (prints JSON to a file)."""
import json
import sys
from pathlib import Path

sys.path.insert(0, r"C:\Users\<user>\Documents\AlphaRush")
from alpharush_rl.search import check_genome, genome_id  # noqa: E402

PLANS_PER_LEVEL = 5
PLANS_HARD = {11: 8, 12: 8}  # bigger retry portfolios where plans win only some seeds
RUNS = Path(r"C:\Users\<user>\Documents\AlphaRush\runtime\rl\campaign-v1\runs")


def load_validation(run_ids, fresh_levels=(), fresh_runs=()):
    """Per level: {genome_id: {"genome", "seeds": {seed: (won, lives, fitness)}}} over all given runs
    (for fresh_levels, only rows of fresh_runs count: results from before a rule change are stale)."""
    table = {}
    for run_id in run_ids:
        for line in (RUNS / run_id / "episodes.jsonl").read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            if row["kind"] not in ("search_eval", "search_validate"):
                continue
            p = row["payload"]
            if p["level"] in fresh_levels and run_id not in fresh_runs:
                continue
            entry = table.setdefault(p["level"], {}).setdefault(p["genome_id"], {"genome": p["genome"], "seeds": {}})
            outcome = p["result"].get("outcome") or {}
            entry["seeds"][p["seed"]] = (p["fitness"] >= 10000, outcome.get("lives") or 0, p["fitness"])
    return table


def ranked(table, level, min_seeds=1):
    rows = []
    for gid, entry in table.get(level, {}).items():
        seeds = entry["seeds"]
        if len(seeds) < min_seeds:
            continue
        wins = sum(won for won, _, _ in seeds.values())
        lives = sorted(l for won, l, _ in seeds.values() if won)
        rows.append({"genome_id": gid, "genome": check_genome(entry["genome"]), "games": len(seeds), "wins": wins,
                     "won_seeds": frozenset(seed for seed, (won, _, _) in seeds.items() if won),
                     "signature": tuple(sorted((seed, won, l) for seed, (won, l, _) in seeds.items())),
                     "min_lives": lives[0] if lives and wins == len(seeds) else 0,
                     "mean_lives": sum(lives) / len(lives) if lives else 0})
    # Smoothed win rate (prior worth 1 win in 4 games): a plan proven on more seeds outranks a short record.
    rows.sort(key=lambda r: (-(r["wins"] + 1) / (r["games"] + 4), -r["mean_lives"], r["genome_id"]))
    return rows


def portfolio(rows, size):
    """Retry portfolio: the best-ranked plan first, then greedily the plan winning most seeds not yet covered
    (ties by rank); plans with an identical per-seed record to a chosen one are skipped."""
    chosen, covered, seen = [], set(), set()
    pool = [r for r in rows if r["signature"] not in seen]
    while pool and len(chosen) < size:
        best = max(pool, key=lambda r: (len(r["won_seeds"] - covered) if chosen else 0, -rows.index(r)))
        chosen.append(best)
        covered |= best["won_seeds"]
        seen.add(best["signature"])
        pool = [r for r in pool if r["signature"] not in seen]
    return chosen


def evidence(r):
    return (f"won {r['wins']} of {r['games']} practice games"
            + (f", {r['mean_lives']:.0f} lives left on average when winning" if r["wins"] else ""))


def main():
    run_ids = sys.argv[1].split(",")
    out = Path(sys.argv[2])
    only = {int(x) for x in sys.argv[3].split(",")} if len(sys.argv) > 3 else set(range(1, 13))
    fresh_levels = {int(x) for x in sys.argv[4].split(",")} if len(sys.argv) > 4 else set()
    fresh_runs = set(sys.argv[5].split(",")) if len(sys.argv) > 5 else set()
    table = load_validation(run_ids, fresh_levels, fresh_runs)
    report, collect, evals, plans = {}, [], [], {}
    for level in range(1, 13):
        rows = ranked(table, level, min_seeds=2) or ranked(table, level)
        winners = [r for r in rows if r["wins"]]
        report[level] = [{k: r[k] for k in ("genome_id", "games", "wins", "mean_lives")} for r in rows[:5]]
        validated = [r for r in rows if r["games"] >= 10]
        size = PLANS_HARD.get(level, PLANS_PER_LEVEL)
        picked = portfolio(validated, size)
        picked += [r for r in rows if r not in picked][:size - len(picked)]
        for r in ([r for r in picked if r["wins"]][:3] if level in only else []):
            for seed in range(1001, 1011):
                collect.append({"level": level, "seed": seed, "genome": r["genome"]})
        if rows and level in only:
            for seed in range(5001, 5011):
                evals.append({"level": level, "seed": seed, "genome": picked[0]["genome"]})
        if rows:
            plans[str(level)] = [{"genome": r["genome"], "evidence": evidence(r)} for r in picked]
            report[level] = [{k: r[k] for k in ("genome_id", "games", "wins", "mean_lives")} for r in picked]
    out.write_text(json.dumps({"report": report, "collect": collect, "eval": evals, "plans": plans}, indent=1),
                   encoding="utf-8")
    for level, rows in report.items():
        print(level, rows[:3])
    print("collect tasks", len(collect), "eval tasks", len(evals), "levels with plans", sorted(plans, key=int))


if __name__ == "__main__":
    main()
