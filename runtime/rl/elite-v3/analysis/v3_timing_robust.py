"""elite-v3 robustness check (NOT pre-registered; added after the timing result): is the within-game timing signal just a
shared time trend? The benefit of deviating, g = mean(non-P0 branches) - P0, changes systematically along a game (an early
change re-rolls more of the future), so any signal that merely rises or falls with the wave could covary with g.

usage: v3_timing_robust.py <S1 run id> <fork table json> <p json> [<p json> ...] [--perm 5000]
For each signal: within every base game, regress the signal and g on [1, t, t^2] (t = wave / wave total of the consult)
and keep residuals; statistic = mean over consults of resid_signal * resid_g; null = residual signal shuffled within the
game. Also: the raw within-game covariance split by base game won / lost, and the signal's own within-game correlation
with t (how much of it is a clock)."""
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
ap.add_argument("signals", nargs="+")
ap.add_argument("--perm", type=int, default=5000)
a = ap.parse_args()
consults, totals = {}, {}
for line in (RUNS / a.s1 / "episodes.jsonl").read_text(encoding="utf-8").splitlines():
    if '"cmd_steady"' not in line:
        continue
    r = json.loads(line)["payload"]
    if r.get("player") == "cmd_steady":
        for c in r["result"]["commander"]["consults"]:
            consults[f"{r['level']}:{r['seed']}:{r['genome_id']}:{c['consult']}"] = c
        totals[(r["level"], r["seed"], r["genome_id"])] = (r["result"].get("final_summary") or {}).get("wave_total") or 15
T = []
for e in json.loads(Path(a.table).read_text(encoding="utf-8")):
    key = ":".join(map(str, e["key"]))
    c = consults[key]
    n = len(c["plays"])
    if n < 2:
        continue
    y = [e["plays"][str(j)] for j in range(n)]
    game = tuple(e["key"][:3])
    T.append({"key": key, "game": game, "t": c["wave"] / totals[game], "g": statistics.mean(y[1:]) - y[0],
              "won": y[0] >= 1.0, "rule": 1.0 if c["shadow"]["rule"] else 0.0})
by_game = collections.defaultdict(list)
for t in T:
    by_game[t["game"]].append(t)


def residuals(xs, ts):
    """Least-squares residuals of xs on [1, t, t^2] (falls back to fewer terms for tiny games)."""
    import itertools
    k = min(3, len(set(ts)))
    if k <= 1:
        m = statistics.mean(xs)
        return [x - m for x in xs]
    cols = [[1.0] * len(ts), list(ts), [t * t for t in ts]][:k]
    # normal equations (k <= 3), solved by Gaussian elimination
    A = [[sum(ci * cj for ci, cj in zip(cols[i], cols[j])) for j in range(k)] for i in range(k)]
    b = [sum(ci * x for ci, x in zip(cols[i], xs)) for i in range(k)]
    for i in range(k):
        piv = max(range(i, k), key=lambda r: abs(A[r][i]))
        A[i], A[piv], b[i], b[piv] = A[piv], A[i], b[piv], b[i]
        if abs(A[i][i]) < 1e-12:
            m = statistics.mean(xs)
            return [x - m for x in xs]
        for r in range(k):
            if r != i:
                f = A[r][i] / A[i][i]
                A[r] = [u - f * v for u, v in zip(A[r], A[i])]
                b[r] -= f * b[i]
    beta = [b[i] / A[i][i] for i in range(k)]
    return [x - sum(beta[j] * cols[j][idx] for j in range(k)) for idx, x in enumerate(xs)]


signals = {"rule": {t["key"]: t["rule"] for t in T}}
for path in a.signals:
    signals[Path(path).stem] = json.loads(Path(path).read_text(encoding="utf-8"))["p_intervene"]
rng = random.Random(9)
out = {}
for name, sig in signals.items():
    res_s, res_g, raw = {}, {}, {"won": [], "lost": []}
    clock = []
    for g, rows in by_game.items():
        ts = [r["t"] for r in rows]
        xs = [sig[r["key"]] for r in rows]
        gs = [r["g"] for r in rows]
        res_s[g], res_g[g] = residuals(xs, ts), residuals(gs, ts)
        mx, mg = statistics.mean(xs), statistics.mean(gs)
        raw["won" if rows[0]["won"] else "lost"].extend((x - mx) * (y - mg) for x, y in zip(xs, gs))
        if len(rows) > 2 and statistics.pstdev(xs) > 0 and statistics.pstdev(ts) > 0:
            mt = statistics.mean(ts)
            clock.append(sum((x - mx) * (t - mt) for x, t in zip(xs, ts))
                         / (len(xs) * statistics.pstdev(xs) * statistics.pstdev(ts)))
    count = sum(len(v) for v in res_s.values())
    stat = lambda S: sum(sum(s * q for s, q in zip(S[g], res_g[g])) for g in S) / count
    obs = stat(res_s)
    null = []
    for _ in range(a.perm):
        shuffled = {}
        for g, v in res_s.items():
            v = list(v)
            rng.shuffle(v)
            shuffled[g] = v
        null.append(stat(shuffled))
    p = (1 + sum(v >= obs for v in null)) / (1 + a.perm)
    sd = statistics.pstdev(null)
    out[name] = {"detrended_cov": round(obs, 6), "z": round(obs / sd, 2) if sd else None, "perm_p": round(p, 4),
                 "raw_cov_won_games": round(statistics.mean(raw["won"]), 6),
                 "raw_cov_lost_games": round(statistics.mean(raw["lost"]), 6),
                 "mean_within_game_corr_with_t": round(statistics.mean(clock), 3) if clock else None}
    print(f"{name:12s} detrended cov {obs:+.6f} (z {out[name]['z']:+.2f}) perm p {p:.4f} | raw cov won {out[name]['raw_cov_won_games']:+.5f}"
          f" lost {out[name]['raw_cov_lost_games']:+.5f} | signal~t corr {out[name]['mean_within_game_corr_with_t']}")
Path(a.table).with_name("llm_timing_robust.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
