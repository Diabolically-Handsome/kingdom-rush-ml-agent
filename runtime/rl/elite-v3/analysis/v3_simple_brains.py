"""elite-v3 (read-only, development evidence): cross-fitted one-step test of simple commander brains on the fork table.

usage: v3_simple_brains.py <S1 run id> <fork table json> [--boot 4000]
Two folds by seed (fit on 1201-1202 -> score on 1203-1204, and the reverse). Fitted on the fit fold only:
  play ranking = play ids ordered by mean (y_j - mean of the consult's non-P0 plays).
Brains (all decide from what was observable at the consult; none sees outcomes of the scored fold):
  steady            P0 always
  rule              the rule brain's shadow choice
  rule_timing_best  P0 when the rule brain picks P0; otherwise the best-ranked offered play id from the fit fold
  always_best       the best-ranked offered play id from the fit fold at every consult (never P0)
Statistic: gain vs steady = mean over all consults of the scored fold (y_b - y_0), unconditional on the base outcome,
with a 95% cluster bootstrap over base games; also d_np = y_b - mean non-P0 for deviating consults. A brain is a
candidate for a whole-game test only if its gain vs steady is positive in BOTH folds."""
import argparse
import collections
import json
import random
import statistics
from pathlib import Path

ROOT = Path(r"C:\Users\<user>\Documents\AlphaRush")
RUNS = ROOT / "runtime/rl/elite-v3/runs"
ap = argparse.ArgumentParser()
ap.add_argument("s1")
ap.add_argument("table")
ap.add_argument("--boot", type=int, default=4000)
args = ap.parse_args()
consults = {}
for line in (RUNS / args.s1 / "episodes.jsonl").read_text(encoding="utf-8").splitlines():
    if '"cmd_steady"' not in line:
        continue
    r = json.loads(line)["payload"]
    if r.get("player") == "cmd_steady":
        for c in r["result"]["commander"]["consults"]:
            consults[f"{r['level']}:{r['seed']}:{r['genome_id']}:{c['consult']}"] = c
T = []
for e in json.loads(Path(args.table).read_text(encoding="utf-8")):
    key = ":".join(map(str, e["key"]))
    c = consults[key]
    n = len(c["plays"])
    if n < 2:
        continue
    y = [e["plays"][str(j)] for j in range(n)]
    T.append({"key": key, "game": tuple(e["key"][:3]), "seed": e["key"][1], "y": y,
              "ids": [p["id"] for p in c["plays"]], "rule": c["shadow"]["rule"]})


def ranking(rows):
    eff = collections.defaultdict(list)
    for t in rows:
        m = statistics.mean(t["y"][1:])
        for j in range(1, len(t["y"])):
            eff[t["ids"][j]].append(t["y"][j] - m)
    return sorted(eff, key=lambda pid: -statistics.mean(eff[pid])), {k: round(statistics.mean(v), 4) for k, v in eff.items()}


def boot(by_game, reps, rng):
    games = sorted(by_game)
    stat = lambda gs: sum(sum(by_game[g]) for g in gs) / max(1, sum(len(by_game[g]) for g in gs))
    vals = sorted(stat([rng.choice(games) for _ in games]) for _ in range(reps))
    return stat(games), vals[int(0.025 * reps)], vals[int(0.975 * reps) - 1]


rng = random.Random(11)
out = {}
for fit, score in (({1201, 1202}, {1203, 1204}), ({1203, 1204}, {1201, 1202})):
    order, effects = ranking([t for t in T if t["seed"] in fit])
    best = lambda t: next(t["ids"].index(p) for p in order if p in t["ids"][1:])
    brains = {"steady": lambda t: 0, "rule": lambda t: t["rule"],
              "rule_timing_best": lambda t: 0 if t["rule"] == 0 else best(t), "always_best": best}
    name = f"fit {sorted(fit)} -> score {sorted(score)}"
    print(f"== {name}: fitted play ranking {order} effects {effects}")
    out[name] = {}
    rows = [t for t in T if t["seed"] in score]
    for b, fn in brains.items():
        gain, dnp = collections.defaultdict(list), collections.defaultdict(list)
        dev = 0
        for t in rows:
            i = fn(t)
            gain[t["game"]].append(t["y"][i] - t["y"][0])
            if i:
                dev += 1
                dnp[t["game"]].append(t["y"][i] - statistics.mean(t["y"][1:]))
        g = boot(gain, args.boot, rng)
        d = boot(dnp, args.boot, rng) if dnp else (0, 0, 0)
        out[name][b] = {"gain_vs_steady": [round(x, 4) for x in g], "d_np": [round(x, 4) for x in d],
                        "deviations": dev, "consults": len(rows), "games": len(gain)}
        print(f"  {b:17s} gain vs steady {g[0]:+.3f} [{g[1]:+.3f},{g[2]:+.3f}]  d_np {d[0]:+.3f} [{d[1]:+.3f},{d[2]:+.3f}]"
              f"  deviations {dev}/{len(rows)}  games {len(gain)}")
Path(args.table).with_name("simple_brains.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
