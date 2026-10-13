"""elite-v3 (read-only): does a model's probability order the NON-P0 plays (the deviations) by their outcome, beyond the
fixed play-id ranking? Removes the P0 / base-outcome artefact entirely: only the non-P0 branches of each consult are used.

usage: v3_nonp0_ranking.py <S1 run id> <fork table json> --seeds 1203,1204 --fit 1201,1202 name=<records.json> ...
Per consult with >= 2 non-P0 plays: pick = argmax of the model's p among non-P0 plays; d_np = y_pick - mean(non-P0 y);
rho = Spearman(p, y) over the non-P0 plays. Baseline 'fixed ranking': play ids ordered by mean (y_j - mean non-P0) on the
--fit seeds. 95% cluster bootstrap over base games; paired difference model - fixed ranking."""
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
ap.add_argument("--seeds", required=True)
ap.add_argument("--fit", required=True)
ap.add_argument("--boot", type=int, default=4000)
ap.add_argument("models", nargs="+")
a = ap.parse_args()
seeds = {int(x) for x in a.seeds.split(",")}
fit = {int(x) for x in a.fit.split(",")}
assert not seeds & fit
ids = {}
for line in (RUNS / a.s1 / "episodes.jsonl").read_text(encoding="utf-8").splitlines():
    if '"cmd_steady"' not in line:
        continue
    r = json.loads(line)["payload"]
    if r.get("player") == "cmd_steady":
        for c in r["result"]["commander"]["consults"]:
            ids[f"{r['level']}:{r['seed']}:{r['genome_id']}:{c['consult']}"] = [p["id"] for p in c["plays"]]
T = []
for e in json.loads(Path(a.table).read_text(encoding="utf-8")):
    key = ":".join(map(str, e["key"]))
    n = len(ids[key])
    if n < 3:
        continue  # fewer than two deviations: nothing to order
    T.append({"key": key, "game": tuple(e["key"][:3]), "seed": e["key"][1], "ids": ids[key],
              "y": [e["plays"][str(j)] for j in range(n)]})
eff = collections.defaultdict(list)
for t in T:
    if t["seed"] in fit:
        m = statistics.mean(t["y"][1:])
        for j in range(1, len(t["y"])):
            eff[t["ids"][j]].append(t["y"][j] - m)
rank = {pid: statistics.mean(v) for pid, v in eff.items()}
rows = [t for t in T if t["seed"] in seeds]


def spearman(xs, ys):
    def rk(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(v):
            j = i
            while j + 1 < len(v) and v[order[j + 1]] == v[order[i]]:
                j += 1
            for k in range(i, j + 1):
                r[order[k]] = (i + j) / 2
            i = j + 1
        return r
    rx, ry = rk(xs), rk(ys)
    mx, my = statistics.mean(rx), statistics.mean(ry)
    num = sum((p - mx) * (q - my) for p, q in zip(rx, ry))
    den = (sum((p - mx) ** 2 for p in rx) * sum((q - my) ** 2 for q in ry)) ** 0.5
    return num / den if den else None


rng = random.Random(3)


def boot(by_game):
    games = sorted(by_game)
    stat = lambda gs: sum(sum(by_game[g]) for g in gs) / max(1, sum(len(by_game[g]) for g in gs))
    vals = sorted(stat([rng.choice(games) for _ in games]) for _ in range(a.boot))
    return stat(games), vals[int(0.025 * a.boot)], vals[int(0.975 * a.boot) - 1]


def evaluate(name, score_of):
    d, rho, diff = collections.defaultdict(list), collections.defaultdict(list), collections.defaultdict(list)
    for t in rows:
        s = score_of(t)
        if s is None:
            raise SystemExit(f"{name}: no scores for {t['key']}")
        nonp0 = list(range(1, len(t["y"])))
        pick = max(nonp0, key=lambda j: (s[j], -j))
        fixed = max(nonp0, key=lambda j: (rank.get(t["ids"][j], 0.0), -j))
        m = statistics.mean(t["y"][1:])
        d[t["game"]].append(t["y"][pick] - m)
        diff[t["game"]].append(t["y"][pick] - t["y"][fixed])
        r = spearman([s[j] for j in nonp0], [t["y"][j] for j in nonp0])
        if r is not None:
            rho[t["game"]].append(r)
    res = {"d_np": boot(d), "spearman_nonp0": boot(rho), "minus_fixed_ranking": boot(diff)}
    print(f"{name:14s} d_np {res['d_np'][0]:+.3f} [{res['d_np'][1]:+.3f},{res['d_np'][2]:+.3f}]  "
          f"rho(nonP0) {res['spearman_nonp0'][0]:+.3f} [{res['spearman_nonp0'][1]:+.3f},{res['spearman_nonp0'][2]:+.3f}]  "
          f"minus fixed ranking {res['minus_fixed_ranking'][0]:+.3f} [{res['minus_fixed_ranking'][1]:+.3f},"
          f"{res['minus_fixed_ranking'][2]:+.3f}]")
    return res


print(f"consults with >= 2 deviations on seeds {sorted(seeds)}: {len(rows)} in {len({t['game'] for t in rows})} games; "
      f"fixed ranking from {sorted(fit)}: {sorted(rank, key=lambda k: -rank[k])}")
out = {"fixed_ranking": evaluate("fixed_ranking", lambda t: [0.0] + [rank.get(p, 0.0) for p in t["ids"][1:]])}
for spec in a.models:
    name, path = spec.split("=", 1)
    p = {r["key"]: r["p"] for r in json.loads(Path(path).read_text(encoding="utf-8")) if r.get("p")}
    out[name] = evaluate(name, lambda t, p=p: p.get(t["key"]))
print(json.dumps({k: {m: [round(x, 4) for x in v] for m, v in r.items()} for k, r in out.items()}))
