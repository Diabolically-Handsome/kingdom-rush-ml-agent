"""elite-v3 (read-only; from the review copy, --name labels the language-model brain): does a commander brain show skill on the fork pilot, beyond chaos?

usage: v3_skill_tests.py [S1 run id] [fork run id] [8b choices json] [--boot 4000] [--perm 1000] [--json out.json]

Reads only the two runs' episodes.jsonl and the offline 8B choice file (+ .records.json beside it). Writes nothing
unless --json is given. For every consult of a steady base game with ALL branches measured (n >= 2 plays), y_j is the
progress score R of branch j (y_0 = the base game). Per brain b:
  d_all = y_b - mean_j y_j          (vs uniform over all offered plays, P0 included)
  d_np  = y_b - mean_{j>=1} y_j     (vs uniform over the NON-P0 plays; removes the base-game regression-to-the-mean)
  argmax share (ties count as hits; chance = mean(#argmax / n)) and mean normalised rank (1 = best, chance 0.5).
CIs: cluster bootstrap over base games (resample games, consult-weighted mean).
Permutation tests (outcomes fixed, choices randomised):
  uniform   : each consult's choice uniform over the n offered plays (the requested null: "no better than random")
  nonp0     : choices that were non-P0 re-drawn uniformly among the non-P0 plays (null: "picks among deviations at random")
  marginal  : the brain's chosen play IDs permuted across consults sharing (level, offered id set) (null: "its choice
              carries no consult-specific information beyond its own marginal policy", e.g. always-P1)
Noise floor: placebo-vs-rule (one native tick delay) paired R differences in S1; within-consult / between-consult /
between-game variance of branch outcomes; cross-seed replicability of play-id effects at matched (level, plan, wave
group) consults (split-seed correlation + label-permutation p)."""
import argparse
import collections
import json
import math
import random
import statistics
from pathlib import Path

ROOT = Path(r"C:\Users\<user>\Documents\AlphaRush")
RUNS = ROOT / "runtime/rl/elite-v3/runs"
SCRATCH = Path(__file__).resolve().parent
EPS = 1e-9

ap = argparse.ArgumentParser()
ap.add_argument("s1", nargs="?", default="native-eval-d043199d066d4c6db33991b4685ff6f3")
ap.add_argument("fork", nargs="?", default="native-eval-da28d8e25ce443c1b6c4c7f8724de6be")
ap.add_argument("llm", nargs="?", default=str(SCRATCH / "llm_choices_s1b.json"))
ap.add_argument("--boot", type=int, default=4000)
ap.add_argument("--perm", type=int, default=1000)
ap.add_argument("--seed", type=int, default=7)
ap.add_argument("--json")
ap.add_argument("--name", default="8b", help="label of the language-model brain in the output")
ap.add_argument("--seeds", help="only consults on these seeds, e.g. 1203,1204 (a LoRA's held-out fold)")
args = ap.parse_args()
NAME = args.name


def rows(run_id):
    out = []
    with open(RUNS / run_id / "episodes.jsonl", encoding="utf-8") as handle:
        for line in handle:
            if line.strip() and '"eval_episode"' in line[:40]:
                try:
                    out.append(json.loads(line)["payload"])
                except json.JSONDecodeError:
                    pass  # the growing file's last line may be partial
    return out


def score(row):  # identical to v3_analysis.score
    res = row["result"]
    out, summ = res.get("outcome") or {}, res.get("final_summary") or {}
    if res.get("status") != "terminal":
        return None
    total = summ.get("wave_total") or 15
    return (1.0 if out.get("level_won") else 0.0) + 0.5 * min(1.0, (summ.get("wave") or 0) / total) \
        + 0.25 * (out.get("lives") or 0) / 20


s1_rows = rows(args.s1)
fork_rows = rows(args.fork)
base = {(r["level"], r["seed"], r["genome_id"]): r for r in s1_rows if r["player"] == "cmd_steady"}
llm = json.loads(Path(args.llm).read_text(encoding="utf-8"))
llm_p = {}
rec_path = Path(args.llm).with_suffix(".records.json")
if rec_path.exists():
    llm_p = {r["key"]: r["p"] for r in json.loads(rec_path.read_text(encoding="utf-8"))}

branch = collections.defaultdict(dict)
for r in fork_rows:
    s = score(r)
    if s is not None:
        branch[(r["level"], r["seed"], r["genome_id"], r["fork"]["consult"])][r["fork"]["play"]] = s

C = []  # complete consults with n >= 2
for key, got in branch.items():
    b = base.get(key[:3])
    c = next((c for c in b["result"]["commander"]["consults"] if c["consult"] == key[3]), None) if b else None
    if c is None:
        continue
    n = len(c["plays"])
    y = {0: score(b), **got}
    if n < 2 or any(j not in y for j in range(n)):
        continue
    k = ":".join(map(str, key))
    C.append({"key": k, "game": key[:3], "level": key[0], "seed": key[1], "gid": key[2], "consult": key[3],
              "wave": c["wave"], "group": c["group"], "reason": c["reason"], "n": n, "y": [y[j] for j in range(n)],
              "ids": [p["id"] for p in c["plays"]], "base_won": bool(b["won"]),
              "rule": c["shadow"].get("rule"), "random": c["shadow"].get("random"), "llm": llm.get(k),
              "llm_p": llm_p.get(k)})
if args.seeds:
    keep = {int(x) for x in args.seeds.split(",")}
    C = [t for t in C if t["seed"] in keep]
missing = sum(1 for t in C if t["llm"] is None)
if missing:
    raise SystemExit(f"{missing} consults have no {NAME} choice: score them or restrict --seeds")
C.sort(key=lambda t: (t["game"], t["consult"]))
games = sorted({t["game"] for t in C})
rng = random.Random(args.seed)
OUT = {"consults": len(C), "games": len(games), "games_won": sum(base[g]["won"] for g in games),
       "consults_in_won_games": sum(t["base_won"] for t in C)}
print(f"complete consults (n>=2): {len(C)} in {len(games)} base games "
      f"({OUT['games_won']} won; {OUT['consults_in_won_games']} consults from won games)")


def idx_of(t, pid):
    return t["ids"].index(pid) if pid in t["ids"] else None


BRAINS = {
    "steady": lambda t: 0,
    NAME: lambda t: t["llm"],
    "rule": lambda t: t["rule"],
    "random(shadow)": lambda t: t["random"],
    "always_P1": lambda t: idx_of(t, "P1"),
    "always_P2": lambda t: idx_of(t, "P2"),
    "always_P4": lambda t: idx_of(t, "P4"),
    "oracle": lambda t: max(range(t["n"]), key=lambda j: (t["y"][j], -j)),
}


def per_consult(t, b):
    y, n = t["y"], t["n"]
    m_all, m_np = statistics.mean(y), statistics.mean(y[1:])
    top = max(y)
    ranks = sorted(y)
    below = sum(v < y[b] - EPS for v in y)
    ties = sum(abs(v - y[b]) <= EPS for v in y)
    rank = (below + (ties - 1) / 2) / (n - 1)  # 0 worst .. 1 best, ties averaged
    return {"y": y[b], "d_all": y[b] - m_all, "d_np": y[b] - m_np, "hit": abs(y[b] - top) <= EPS,
            "chance_hit": sum(abs(v - top) <= EPS for v in y) / n, "rank": rank, "nonp0": b != 0}


def cluster_boot(values_by_game, reps):
    """values_by_game: {game: [v...]} -> (mean, lo, hi) consult-weighted, resampling games."""
    keys = list(values_by_game)
    flat = [v for g in keys for v in values_by_game[g]]
    if not flat:
        return None
    stats = []
    for _ in range(reps):
        pick = [keys[rng.randrange(len(keys))] for _ in keys]
        vals = [v for g in pick for v in values_by_game[g]]
        stats.append(statistics.mean(vals))
    stats.sort()
    return statistics.mean(flat), stats[int(0.025 * reps)], stats[int(0.975 * reps) - 1]


def fmt(ci):
    return "n/a" if ci is None else f"{ci[0]:+.3f} [{ci[1]:+.3f},{ci[2]:+.3f}]"


def evaluate(name, fn, subset):
    by_game = collections.defaultdict(lambda: collections.defaultdict(list))
    used = 0
    for t in subset:
        b = fn(t)
        if b is None:
            continue
        used += 1
        pc = per_consult(t, b)
        for k, v in pc.items():
            by_game[k][t["game"]].append(float(v))
    if not used:
        return None
    res = {"consults": used}
    for k in ("y", "d_all", "d_np", "hit", "chance_hit", "rank", "nonp0"):
        vals = [v for g in by_game[k].values() for v in g]
        res[k] = round(statistics.mean(vals), 4)
    for k in ("d_all", "d_np", "rank"):
        ci = cluster_boot(by_game[k], args.boot)
        res[k + "_ci"] = [round(x, 4) for x in ci]
    return res


print("\n== brain vs branches (all complete consults); CI = 95% cluster bootstrap over base games ==")
print(f"{'brain':15s} {'n':>4s} {'meanR':>6s} {'nonP0':>6s} {'d_all (chosen-mean all)':>30s} "
      f"{'d_np (chosen-mean nonP0)':>30s} {'argmax':>7s} {'chance':>7s} {'rank':>6s}")
OUT["brains"] = {}
for strat_name, subset in (("all", C), ("base_lost", [t for t in C if not t["base_won"]]),
                           ("base_won", [t for t in C if t["base_won"]])):
    print(f"-- {strat_name} ({len(subset)} consults, {len({t['game'] for t in subset})} games)")
    OUT["brains"][strat_name] = {}
    for name, fn in BRAINS.items():
        r = evaluate(name, fn, subset)
        if r is None:
            continue
        OUT["brains"][strat_name][name] = r
        print(f"{name:15s} {r['consults']:4d} {r['y']:6.3f} {r['nonp0']:6.2f} "
              f"{fmt(r['d_all_ci']):>30s} {fmt(r['d_np_ci']):>30s} {r['hit']:7.3f} {r['chance_hit']:7.3f} "
              f"{r['rank']:6.3f}")

# regression to the mean: non-P0 branches vs the base outcome, by base status
print("\n== regression to the mean: mean(non-P0) - P0, and share of non-P0 branches beating / losing to P0 ==")
OUT["rtm"] = {}
for status in (False, True):
    sub = [t for t in C if t["base_won"] == status]
    if not sub:
        continue
    diffs = [y - t["y"][0] for t in sub for y in t["y"][1:]]
    better = sum(d > EPS for d in diffs) / len(diffs)
    worse = sum(d < -EPS for d in diffs) / len(diffs)
    same = 1 - better - worse
    OUT["rtm"]["won" if status else "lost"] = {"branches": len(diffs), "mean": round(statistics.mean(diffs), 4),
                                               "better": round(better, 3), "worse": round(worse, 3),
                                               "identical": round(same, 3)}
    print(f"base {'WON ' if status else 'LOST'}: branches {len(diffs)}  mean(nonP0 - P0) {statistics.mean(diffs):+.3f}  "
          f"better {better:.2f} worse {worse:.2f} identical {same:.2f}")
won_flip = [(t, j) for t in C for j in range(1, t["n"])]
flips = sum((t["y"][j] >= 1.0) != (t["y"][0] >= 1.0) for t, j in won_flip)
print(f"win/loss flips vs base among all non-P0 branches: {flips}/{len(won_flip)} = {flips / len(won_flip):.3f}")
OUT["win_flip_rate"] = round(flips / len(won_flip), 4)


# permutation tests
def perm_test(name, fn, mode, reps):
    used = [(t, fn(t)) for t in C if fn(t) is not None]
    obs = statistics.mean(t["y"][b] for t, b in used)
    strata = collections.defaultdict(list)
    if mode == "marginal":
        for i, (t, b) in enumerate(used):
            strata[(t["level"], tuple(t["ids"]))].append(i)
    ge = 0
    null = []
    for _ in range(reps):
        if mode == "uniform":
            s = statistics.mean(t["y"][rng.randrange(t["n"])] for t, b in used)
        elif mode == "nonp0":
            s = statistics.mean(t["y"][b] if b == 0 else t["y"][rng.randrange(1, t["n"])] for t, b in used)
        else:  # marginal: permute chosen play ids within (level, menu) strata
            vals = []
            for idxs in strata.values():
                chosen = [used[i][0]["ids"][used[i][1]] for i in idxs]
                rng.shuffle(chosen)
                for i, pid in zip(idxs, chosen):
                    t = used[i][0]
                    vals.append(t["y"][t["ids"].index(pid)])
            s = statistics.mean(vals)
        null.append(s)
        ge += s >= obs - 1e-12
    return {"obs_meanR": round(obs, 4), "null_mean": round(statistics.mean(null), 4),
            "null_sd": round(statistics.pstdev(null), 4), "p_one_sided": round((1 + ge) / (1 + reps), 4),
            "consults": len(used)}


print(f"\n== permutation tests ({args.perm}x; statistic = mean chosen R; one-sided p = P(null >= observed)) ==")
OUT["perm"] = {}
for name in (NAME, "rule", "random(shadow)", "always_P1"):
    OUT["perm"][name] = {}
    for mode in ("uniform", "nonp0", "marginal"):
        r = perm_test(name, BRAINS[name], mode, args.perm)
        OUT["perm"][name][mode] = r
        print(f"{name:15s} {mode:9s} obs {r['obs_meanR']:.4f}  null {r['null_mean']:.4f} (sd {r['null_sd']:.4f})  "
              f"p {r['p_one_sided']:.4f}  n={r['consults']}")


# 8B probability vs outcome inside a consult (uses the full distribution, not just the argmax)
def spearman(a, b):
    def rk(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(v):
            j = i
            while j + 1 < len(v) and abs(v[order[j + 1]] - v[order[i]]) <= EPS:
                j += 1
            for k in range(i, j + 1):
                r[order[k]] = (i + j) / 2
            i = j + 1
        return r
    ra, rb = rk(a), rk(b)
    ma, mb = statistics.mean(ra), statistics.mean(rb)
    va = sum((x - ma) ** 2 for x in ra)
    vb = sum((x - mb) ** 2 for x in rb)
    if va <= EPS or vb <= EPS:
        return None
    return sum((x - ma) * (z - mb) for x, z in zip(ra, rb)) / math.sqrt(va * vb)


by_game = collections.defaultdict(list)
for t in C:
    if t["llm_p"] and len(t["llm_p"]) == t["n"]:
        rho = spearman(t["llm_p"], t["y"])
        if rho is not None:
            by_game[t["game"]].append(rho)
ci = cluster_boot(by_game, args.boot)
null_ge, obs = 0, ci[0] if ci else None
if ci:
    items = [(t["llm_p"], t["y"]) for t in C if t["llm_p"] and len(t["llm_p"]) == t["n"] and spearman(t["llm_p"], t["y"])
             is not None]
    for _ in range(args.perm):
        vals = []
        for p, y in items:
            q = p[:]
            rng.shuffle(q)
            r = spearman(q, y)
            vals.append(0.0 if r is None else r)
        null_ge += statistics.mean(vals) >= obs - 1e-12
    OUT["llm_p_spearman"] = {"mean": round(ci[0], 4), "ci": [round(ci[1], 4), round(ci[2], 4)],
                             "consults": len(items), "perm_p": round((1 + null_ge) / (1 + args.perm), 4)}
    print(f"\n{NAME} p vs outcome, within-consult Spearman: {fmt(ci)} over {len(items)} consults with outcome ties broken; "
          f"perm p {(1 + null_ge) / (1 + args.perm):.4f}")
pmax = [max(t["llm_p"]) for t in C if t["llm_p"]]
if pmax:
    print(f"{NAME} top-play probability: median {statistics.median(pmax):.3f}, mean {statistics.mean(pmax):.3f}")
    OUT["llm_pmax_median"] = round(statistics.median(pmax), 4)

# noise floor 1: placebo vs rule in S1 (identical plays, commands one native tick late)
print("\n== noise floor ==")
s1_by = {(r["player"], r["level"], r["seed"], r["genome_id"]): r for r in s1_rows}
pairs = []
for (player, level, seed, gid), r in s1_by.items():
    if player != "cmd_rule":
        continue
    q = s1_by.get(("cmd_placebo", level, seed, gid))
    st = s1_by.get(("cmd_steady", level, seed, gid))
    if q is None or score(r) is None or score(q) is None:
        continue
    pairs.append((score(r), score(q), bool(r["won"]), bool(q["won"]), score(st) if st else None,
                  bool(st["won"]) if st else None))
if pairs:
    d = [a - b for a, b, *_ in pairs]
    disc = sum(x != z for _, _, x, z, *_ in pairs)
    d_rs = [a - s for a, _, _, _, s, _ in pairs if s is not None]
    disc_rs = sum(x != sw for _, _, x, _, _, sw in pairs if sw is not None)
    OUT["placebo_vs_rule"] = {"tasks": len(pairs), "mean": round(statistics.mean(d), 4),
                              "sd": round(statistics.pstdev(d), 4), "win_discordant": disc,
                              "rule_vs_steady_sd": round(statistics.pstdev(d_rs), 4),
                              "rule_vs_steady_win_discordant": disc_rs}
    print(f"S1 rule - placebo (same plays, 1 tick later): tasks {len(pairs)}, mean {statistics.mean(d):+.3f}, "
          f"SD {statistics.pstdev(d):.3f}, win-discordant {disc}/{len(pairs)}")
    print(f"S1 rule - steady (different plays):             SD {statistics.pstdev(d_rs):.3f}, "
          f"win-discordant {disc_rs}/{len(d_rs)}")

# noise floor 2: variance components of non-P0 branch outcomes
within, w_df = 0.0, 0
consult_means = collections.defaultdict(list)
for t in C:
    ys = t["y"][1:]
    if len(ys) >= 2:
        m = statistics.mean(ys)
        within += sum((v - m) ** 2 for v in ys)
        w_df += len(ys) - 1
    consult_means[t["game"]].append(statistics.mean(ys))
var_within = within / max(1, w_df)
bc, bc_df = 0.0, 0
for g, ms in consult_means.items():
    if len(ms) >= 2:
        m = statistics.mean(ms)
        bc += sum((v - m) ** 2 for v in ms)
        bc_df += len(ms) - 1
var_between_consult = bc / max(1, bc_df)
game_means = [statistics.mean(ms) for ms in consult_means.values()]
var_between_game = statistics.variance(game_means) if len(game_means) > 1 else float("nan")
OUT["var"] = {"within_consult_nonP0": round(var_within, 4), "between_consult_means_within_game":
              round(var_between_consult, 4), "between_game_means": round(var_between_game, 4)}
print(f"variance of non-P0 branch R: within consult {var_within:.4f} (SD {math.sqrt(var_within):.3f}, df {w_df}); "
      f"consult means within game {var_between_consult:.4f}; game means {var_between_game:.4f}")

# noise floor 3: do play-id effects replicate across seeds at matched situations?
# situation = (level, plan, wave group) of a 'wave' consult; effect = y_j - mean_j y (centred within the consult)
sit = collections.defaultdict(dict)  # situation -> seed -> {pid: effect}
for t in C:
    if t["reason"] != "wave" or t["group"] is None:
        continue
    m = statistics.mean(t["y"])
    sit[(t["level"], t["gid"], t["group"])][t["seed"]] = {pid: t["y"][j] - m for j, pid in enumerate(t["ids"])}


def split_corr(sitmap):
    xs, zs = [], []
    for s in sitmap.values():
        seeds = sorted(s)
        for a in range(len(seeds)):
            for b in range(a + 1, len(seeds)):
                common = sorted(set(s[seeds[a]]) & set(s[seeds[b]]))
                if len(common) < 2:
                    continue
                for pid in common:
                    xs.append(s[seeds[a]][pid])
                    zs.append(s[seeds[b]][pid])
    if len(xs) < 3 or statistics.pstdev(xs) < EPS or statistics.pstdev(zs) < EPS:
        return None, len(xs)
    mx, mz = statistics.mean(xs), statistics.mean(zs)
    r = sum((a - mx) * (b - mz) for a, b in zip(xs, zs)) / math.sqrt(sum((a - mx) ** 2 for a in xs)
                                                                      * sum((b - mz) ** 2 for b in zs))
    return r, len(xs)


r_obs, npairs = split_corr(sit)
ge = 0
if r_obs is not None:
    for _ in range(args.perm):
        shuffled = {}
        for k, s in sit.items():
            shuffled[k] = {}
            for seed, eff in s.items():
                pids = list(eff)
                vals = list(eff.values())
                rng.shuffle(vals)
                shuffled[k][seed] = dict(zip(pids, vals))
        r, _ = split_corr(shuffled)
        ge += r is not None and r >= r_obs - 1e-12
multi = sum(1 for s in sit.values() if len(s) >= 2)
OUT["replication"] = {"situations": len(sit), "situations_on_2plus_seeds": multi, "pairs": npairs,
                      "split_seed_corr": None if r_obs is None else round(r_obs, 4),
                      "perm_p": None if r_obs is None else round((1 + ge) / (1 + args.perm), 4)}
print(f"cross-seed replication of play-id effects: situations {len(sit)} ({multi} seen on >=2 seeds), "
      f"seed-pair x play cells {npairs}, split-seed corr "
      f"{'n/a' if r_obs is None else f'{r_obs:+.3f}'}, label-permutation p "
      f"{'n/a' if r_obs is None else f'{(1 + ge) / (1 + args.perm):.4f}'}")

# play-id main effects (does any play id help on average, controlling for the consult?)
print("\n== play-id main effects: y_j - mean(non-P0 of same consult) and y_j - y_0, cluster CI over games ==")
OUT["play_effects"] = {}
for pid in ("P1", "P2", "P3", "P4"):
    e_np, e_0 = collections.defaultdict(list), collections.defaultdict(list)
    for t in C:
        if pid in t["ids"]:
            j = t["ids"].index(pid)
            e_np[t["game"]].append(t["y"][j] - statistics.mean(t["y"][1:]))
            e_0[t["game"]].append(t["y"][j] - t["y"][0])
    if e_np:
        a, b = cluster_boot(e_np, args.boot), cluster_boot(e_0, args.boot)
        OUT["play_effects"][pid] = {"consults": sum(map(len, e_np.values())), "vs_nonP0": [round(x, 4) for x in a],
                                    "vs_P0": [round(x, 4) for x in b]}
        print(f"{pid}: consults {sum(map(len, e_np.values())):4d}  vs mean nonP0 {fmt(a)}   vs P0 {fmt(b)}")

# effective sample size: how many distinct choices did the 8B and rule actually make?
for name in (NAME, "rule"):
    ids = collections.Counter(t["ids"][BRAINS[name](t)] for t in C if BRAINS[name](t) is not None)
    OUT[f"{name}_choice_ids"] = dict(ids)
    print(f"{name} chosen play ids on complete consults: {dict(ids)}")
ag = sum(1 for t in C if t["llm"] is not None and t["ids"][t["llm"]] == "P1" and "P1" in t["ids"])
print(f"{NAME} == always_P1 on {ag}/{len(C)} complete consults")
OUT["llm_equals_always_P1"] = ag

if args.json:
    Path(args.json).write_text(json.dumps(OUT, indent=1), encoding="utf-8")
    print("->", args.json)

# ---- extra diagnostics for the existing scripts' method ------------------------------------------------------
print("\n== extra: does anything beyond the play-id main effect replicate across seeds? ==")
main = collections.defaultdict(list)
for s in sit.values():
    for eff in s.values():
        for pid, e in eff.items():
            main[pid].append(e)
main = {pid: statistics.mean(v) for pid, v in main.items()}
resid = {k: {seed: {pid: e - main[pid] for pid, e in eff.items()} for seed, eff in s.items()} for k, s in sit.items()}
r_res, n_res = split_corr(resid)
ge = 0
if r_res is not None:
    for _ in range(args.perm):
        sh = {}
        for k, s in resid.items():
            sh[k] = {}
            for seed, eff in s.items():
                vals = list(eff.values())
                rng.shuffle(vals)
                sh[k][seed] = dict(zip(eff, vals))
        r, _ = split_corr(sh)
        ge += r is not None and r >= r_res - 1e-12
OUT["replication_residual"] = {"split_seed_corr": None if r_res is None else round(r_res, 4),
                               "perm_p": None if r_res is None else round((1 + ge) / (1 + args.perm), 4)}
print(f"situation-specific (play-id main effect removed) split-seed corr "
      f"{'n/a' if r_res is None else f'{r_res:+.3f}'}, perm p {(1 + ge) / (1 + args.perm):.4f}")

print("\n== extra: play-id effect (vs mean non-P0) by level and by seed ==")
OUT["play_by"] = {}
for pid in ("P1", "P4"):
    for dim in ("level", "seed"):
        cells = collections.defaultdict(list)
        for t in C:
            if pid in t["ids"]:
                cells[t[dim]].append(t["y"][t["ids"].index(pid)] - statistics.mean(t["y"][1:]))
        txt = "  ".join(f"{k}:{statistics.mean(v):+.3f}(n{len(v)})" for k, v in sorted(cells.items()))
        OUT["play_by"][f"{pid}_{dim}"] = {str(k): round(statistics.mean(v), 4) for k, v in cells.items()}
        print(f"{pid} by {dim}: {txt}")

if C and C[0]["llm_p"] is not None:
    share = collections.defaultdict(list)
    for t in C:
        if t["llm_p"]:
            for j, pid in enumerate(t["ids"]):
                share[pid].append(t["llm_p"][j])
    OUT["llm_mean_p_by_id"] = {pid: round(statistics.mean(v), 4) for pid, v in sorted(share.items())}
    print(f"{NAME} mean probability by play id:", OUT["llm_mean_p_by_id"])

print("\n== extra: build_commander_sft labels (MARGIN 0.01) by split, and the base games behind them ==")
TRAIN, HOLD = {1201, 1202}, {1203, 1204}
for name, seeds in (("train", TRAIN), ("holdout", HOLD)):
    sub = [t for t in C if t["seed"] in seeds]
    tg = []
    for t in sub:
        best = max(range(t["n"]), key=lambda i: (t["y"][i], -i))
        tg.append(t["ids"][best] if t["y"][best] - t["y"][0] >= 0.01 else "P0")
    gs = sorted({t["game"] for t in sub})
    won = sum(base[g]["won"] for g in gs)
    OUT[f"sft_{name}"] = {"consults": len(sub), "games": len(gs), "won_games": won,
                          "targets": dict(collections.Counter(tg))}
    print(f"{name}: consults {len(sub)}, base games {len(gs)} (won {won}), targets {dict(collections.Counter(tg))}")
    # how predictable is the P0 label from base status alone?
    if sub:
        agree = sum((lab == "P0") == t["base_won"] for lab, t in zip(tg, sub))
        print(f"   label==P0 iff base game won: {agree}/{len(sub)}")
        OUT[f"sft_{name}"]["p0_iff_won"] = agree

print("\n== extra: v3_regret_eval-style gain vs steady on holdout seeds, naive vs cluster CI ==")
hold = [t for t in C if t["seed"] in HOLD]
OUT["holdout_gain"] = {}
for name in (NAME, "rule", "random(shadow)", "always_P1", "always_P4"):
    by = collections.defaultdict(list)
    for t in hold:
        b = BRAINS[name](t)
        if b is not None:
            by[t["game"]].append(t["y"][b] - t["y"][0])
    flat = [v for g in by.values() for v in g]
    if len(flat) < 2:
        continue
    naive = 1.96 * statistics.pstdev(flat) / math.sqrt(len(flat))
    ci = cluster_boot(by, args.boot)
    OUT["holdout_gain"][name] = {"mean": round(ci[0], 4), "naive_half": round(naive, 4),
                                 "cluster_ci": [round(ci[1], 4), round(ci[2], 4)], "games": len(by)}
    print(f"{name:15s} gain vs steady {ci[0]:+.3f}  naive +-{naive:.3f}   cluster [{ci[1]:+.3f},{ci[2]:+.3f}] "
          f"({len(by)} games, all base-lost: {all(not base[g]['won'] for g in by)})")

print("\n== extra: power for d_np (chosen - mean nonP0), cluster = base game ==")
for name in (NAME, "always_P4", "rule"):
    gm = []
    for g in games:
        v = [per_consult(t, BRAINS[name](t))["d_np"] for t in C if t["game"] == g and BRAINS[name](t) is not None]
        if v:
            gm.append(statistics.mean(v))
    sd = statistics.stdev(gm)
    need = {d: math.ceil(((1.96 + 0.84) * sd / d) ** 2) for d in (0.05, 0.1)}
    OUT.setdefault("power", {})[name] = {"sd_game_mean_dnp": round(sd, 4), "games_for_80pct": need}
    print(f"{name:10s} SD of per-game mean d_np {sd:.3f} -> base games for 80% power: delta 0.05: {need[0.05]}, "
          f"delta 0.10: {need[0.1]}")
if pairs:
    sd = statistics.pstdev([a - s for a, _, _, _, s, _ in pairs if s is not None])
    need = {d: math.ceil(((1.96 + 0.84) * sd / d) ** 2) for d in (0.05, 0.1, 0.2)}
    OUT.setdefault("power", {})["whole_game_paired"] = {"sd": round(sd, 4), "tasks_for_80pct": need}
    print(f"whole-game paired arm-vs-steady (SD {sd:.3f}): tasks for 80% power: {need}")

same_trace = [r for r in fork_rows if r["result"]["trace_sha256"] == base[(r["level"], r["seed"], r["genome_id"])]["result"]["trace_sha256"]]
OUT["noop_forks_same_trace"] = len(same_trace)
OUT["noop_forks_same_R"] = sum(score(r) == score(base[(r["level"], r["seed"], r["genome_id"])]) for r in same_trace)
print(f"\nforks with a trace identical to the base game: {len(same_trace)} (same R: {OUT['noop_forks_same_R']})")
if args.json:
    Path(args.json).write_text(json.dumps(OUT, indent=1), encoding="utf-8")
    print("->", args.json)

print("\n== extra: is the oracle just 'some re-roll wins'? (lost base games; independent-coin model) ==")
lost = [t for t in C if not t["base_won"]]
br = [(t, j) for t in lost for j in range(1, t["n"])]
q = sum(t["y"][j] >= 1.0 for t, j in br) / max(1, len(br))
obs_flip = sum(any(v >= 1.0 for v in t["y"][1:]) for t in lost) / max(1, len(lost))
pred_flip = statistics.mean(1 - (1 - q) ** (t["n"] - 1) for t in lost) if lost else float("nan")
by_id = collections.defaultdict(list)
for t, j in br:
    by_id[t["ids"][j]].append(t["y"][j] >= 1.0)
OUT["chaos_model"] = {"branch_win_rate_lost_base": round(q, 4), "consults_with_a_winning_branch": round(obs_flip, 4),
                      "independent_coin_prediction": round(pred_flip, 4),
                      "win_rate_by_id": {k: [round(sum(v) / len(v), 4), len(v)] for k, v in sorted(by_id.items())}}
print(f"non-P0 branch win rate in lost base games q={q:.3f}; consults with >=1 winning branch: observed {obs_flip:.3f}, "
      f"independent-coin prediction {pred_flip:.3f}; win rate by play id "
      + ", ".join(f"{k} {sum(v) / len(v):.3f} (n{len(v)})" for k, v in sorted(by_id.items())))
if args.json:
    Path(args.json).write_text(json.dumps(OUT, indent=1), encoding="utf-8")

print("\n== extra: deviation timing (does the brain deviate where deviating helps, within the same base game?) ==")
# y_0 is constant within a game, so the value of deviating at consult c is m_np(c) - y_0; within a game only m_np(c)
# varies. Statistic: sum over games of sum_c (dev_c - mean_g dev) * (m_np(c) - mean_g m_np); null: permute dev within game.
OUT["timing"] = {}
for name in ("rule", "random(shadow)", NAME):
    per_game = collections.defaultdict(list)
    for t in C:
        b = BRAINS[name](t)
        if b is not None:
            per_game[t["game"]].append((1.0 if b != 0 else 0.0, statistics.mean(t["y"][1:])))

    def stat(pg):
        s = 0.0
        for v in pg.values():
            md, mm = statistics.mean(d for d, _ in v), statistics.mean(m for _, m in v)
            s += sum((d - md) * (m - mm) for d, m in v)
        return s / sum(len(v) for v in pg.values())

    obs = stat(per_game)
    ge = 0
    for _ in range(args.perm):
        sh = {}
        for g, v in per_game.items():
            devs = [d for d, _ in v]
            rng.shuffle(devs)
            sh[g] = [(d, m) for d, (_, m) in zip(devs, v)]
        ge += stat(sh) >= obs - 1e-12
    OUT["timing"][name] = {"cov": round(obs, 5), "perm_p": round((1 + ge) / (1 + args.perm), 4)}
    print(f"{name:15s} within-game cov(deviate, mean nonP0) {obs:+.5f}  perm p {(1 + ge) / (1 + args.perm):.4f}")
if args.json:
    Path(args.json).write_text(json.dumps(OUT, indent=1), encoding="utf-8")
