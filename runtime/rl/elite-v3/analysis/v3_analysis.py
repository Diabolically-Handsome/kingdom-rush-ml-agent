"""elite-v3 analysis (read-only): S1 arms and the fork pilot.

usage: v3_analysis.py s1 <S1 run id>
       v3_analysis.py fork <S1 run id> <fork run id> [8b choices json]
Progress score of a game: R = won + 0.5 * wave reached / wave total + 0.25 * lives left / 20.
Fork: for every consult of a steady (executor alone) base game, the base outcome is play P0; each fork row gives the
outcome of play j at that consult (all other consults steady). Headroom = best play - P0; a brain's regret = best play -
the play it picks. Rule and random choices come from the consult's shadow record (what those brains would have chosen
at exactly that briefing); the 8B's from an offline scoring file {"<task index>:<consult>": play index}."""
import collections
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(r"C:\Users\<user>\Documents\AlphaRush")
RUNS = ROOT / "runtime/rl/elite-v3/runs"


def rows(run_id):
    return [json.loads(line)["payload"] for line in (RUNS / run_id / "episodes.jsonl").read_text(encoding="utf-8")
            .splitlines() if line.strip() and '"eval_episode"' in line[:40]]


def score(row):
    res = row["result"]
    out, summ = res.get("outcome") or {}, res.get("final_summary") or {}
    if res.get("status") != "terminal":
        return None
    total = summ.get("wave_total") or 15
    return (1.0 if out.get("level_won") else 0.0) + 0.5 * min(1.0, (summ.get("wave") or 0) / total) \
        + 0.25 * (out.get("lives") or 0) / 20


def s1(run_id):
    data = rows(run_id)
    by = collections.defaultdict(list)
    for r in data:
        by[(r["player"], r["level"])].append(r)
    players = sorted({r["player"] for r in data})
    levels = sorted({r["level"] for r in data})
    print("arm x level: wins/games, mean R, voids")
    for player in players:
        cells = []
        for level in levels:
            rs = by[(player, level)]
            scores = [s for s in map(score, rs) if s is not None]
            wins = sum(1 for r in rs if (r["result"].get("outcome") or {}).get("level_won"))
            voids = sum(1 for r in rs if r["result"].get("status") != "terminal")
            cells.append(f"L{level} {wins}/{len(rs)} R{statistics.mean(scores):.2f}" + (f" v{voids}" if voids else ""))
        allr = [s for r in data if r["player"] == player for s in [score(r)] if s is not None]
        allw = sum(1 for r in data if r["player"] == player and (r["result"].get("outcome") or {}).get("level_won"))
        print(f"{player:12s} total {allw}/{sum(1 for r in data if r['player'] == player)} R{statistics.mean(allr):.3f} | "
              + " | ".join(cells))
    # paired by task: each arm minus steady
    base = {(r["level"], r["seed"], r["genome_id"]): score(r) for r in data if r["player"] == "cmd_steady"}
    for player in players:
        if player == "cmd_steady":
            continue
        diffs = [score(r) - base[(r["level"], r["seed"], r["genome_id"])] for r in data if r["player"] == player
                 and score(r) is not None and base.get((r["level"], r["seed"], r["genome_id"])) is not None]
        if diffs:
            sd = statistics.pstdev(diffs) / max(1, len(diffs)) ** 0.5
            print(f"paired {player} - steady: mean {statistics.mean(diffs):+.3f} +- {1.96 * sd:.3f} (n={len(diffs)}; "
                  f"better {sum(d > 1e-9 for d in diffs)}, worse {sum(d < -1e-9 for d in diffs)})")


def fork(s1_id, fork_id, llm_path=None):
    base_rows = [r for r in rows(s1_id) if r["player"] == "cmd_steady"]
    fork_rows = rows(fork_id)
    llm = json.loads(Path(llm_path).read_text(encoding="utf-8")) if llm_path else {}
    base = {(r["level"], r["seed"], r["genome_id"]): r for r in base_rows}
    outcome = collections.defaultdict(dict)  # (level, seed, gid, consult) -> {play: R}
    voids = 0
    for r in fork_rows:
        f = r.get("fork") or {}
        s = score(r)
        if s is None:
            voids += 1
            continue
        outcome[(r["level"], r["seed"], r["genome_id"], f["consult"])][f["play"]] = s
    table = []
    for key, plays in sorted(outcome.items()):
        b = base.get(key[:3])
        if b is None:
            continue
        consult = next((c for c in b["result"]["commander"]["consults"] if c["consult"] == key[3]), None)
        if consult is None:
            continue
        plays = {0: score(b), **plays}
        n = len(consult["plays"])
        if len(plays) < n:
            continue  # a branch voided
        best = max(plays.values())
        rule = consult["shadow"].get("rule")
        rand = statistics.mean(plays.values())
        row = {"key": key, "n": n, "p0": plays[0], "best": best, "headroom": best - plays[0],
               "rule_regret": best - plays.get(rule, plays[0]), "random_regret": best - rand,
               "steady_regret": best - plays[0], "plays": plays, "rule": rule}
        llm_key = f"{key[0]}:{key[1]}:{key[2]}:{key[3]}"
        if llm_key in llm:
            row["llm_regret"] = best - plays[llm[llm_key]]
        table.append(row)
    print(f"consults with every branch: {len(table)} (void branches {voids})")
    if not table:
        return
    for name in ("headroom", "steady_regret", "rule_regret", "random_regret", "llm_regret"):
        vals = [t[name] for t in table if name in t]
        if vals:
            print(f"{name:14s} mean {statistics.mean(vals):.4f}  nonzero {sum(v > 1e-9 for v in vals)}/{len(vals)}  "
                  f"max {max(vals):.3f}")
    decisive = [t for t in table if t["headroom"] > 0.05]
    print(f"decisive consults (headroom > 0.05): {len(decisive)}")
    by_level = collections.defaultdict(list)
    for t in table:
        by_level[t["key"][0]].append(t)
    for level, ts in sorted(by_level.items()):
        print(f"  L{level}: consults {len(ts)}, mean headroom {statistics.mean(t['headroom'] for t in ts):.4f}, "
              f"rule regret {statistics.mean(t['rule_regret'] for t in ts):.4f}, "
              f"random regret {statistics.mean(t['random_regret'] for t in ts):.4f}")
    out = Path(__file__).parent / f"fork_table_{fork_id[-12:]}.json"
    out.write_text(json.dumps([{**t, "key": list(t["key"]), "plays": {str(k): v for k, v in t["plays"].items()}}
                               for t in table], indent=1), encoding="utf-8")
    print("table ->", out)


if __name__ == "__main__":
    if sys.argv[1] == "s1":
        s1(sys.argv[2])
    else:
        fork(sys.argv[2], sys.argv[3], sys.argv[4] if len(sys.argv) > 4 else None)
