"""elite-v3b pre-registered analysis (frozen before any new-format LLM result exists; see the pre-registration in
docs/elite/elite-v3b-preregistration-2026-10-12.md). Read-only.

usage: v3b_analysis.py <S1b run id> <replay run id> <fork table json> <results dir> [--boot 4000] [--perm 2000]
                       [--json out.json]

Unit: the 985 consults with >= 2 plays of the 56 steady base games (train seeds 1201-1204); y_j = progress score of the
branch taking play j at that consult (y_0 = the base game). Intervals: 95% percentile cluster bootstrap over base games.

M1 (primary, "deviation choice"): for every consult, the brain's preferred deviation = its most probable non-P0 play
     (language models: argmax of p over plays 1..n-1, ties to the lower index; rule baselines: their rule);
     score = y_pick - mean(y_1..y_{n-1}). Independent of the base game's outcome; 0 = a random deviation.
M2 (secondary, "deviate or stay"): the brain's actual choice minus P0, y_choice - y_0 (P0 -> 0), with the deviation
     rate and a game-progress (clock) control of the deviation timing (per-game quadratic detrend in wave fraction).
M3 (secondary): within-consult Spearman of p vs outcomes, over all plays and over non-P0 plays, and a permutation test
     of M1 that keeps the brain's own preferred-play frequencies within (level, offered non-P0 id set).
Tiers: T1 rich - thin_new (same model) M1 > 0, paired CI excludes 0, Holm over the two models; T2 primary arm
(24b rich_reason) M1 > 0 (CI excludes 0) and >= 1/2 of the cross-fitted fixed ranking's M1 in both folds; T3
(exploratory) primary arm M1 - fixed ranking M1 > 0 in both folds and pooled CI excludes 0."""
import argparse
import collections
import json
import math
import random
import statistics
import sys
from pathlib import Path

sys.dont_write_bytecode = True
ROOT = Path(r"C:\Users\<user>\Documents\AlphaRush")
sys.path.insert(0, str(ROOT))
from alpharush_rl import rich_briefing as rb  # noqa: E402

RUNS = ROOT / "runtime/rl/elite-v3/runs"
ANALYSIS_V3 = ROOT / "runtime/rl/elite-v3/analysis"
GEO = ROOT / "runtime/rl/elite-v3b/geometry"
FOLDS = {"A": ({1203, 1204}, {1201, 1202}), "B": ({1201, 1202}, {1203, 1204})}  # name: (fit seeds, scored seeds)
MODELS = ("8b", "24b")
ARMS = ("thin_new", "rich", "rich_reason")
PRIMARY = ("24b", "rich_reason")


def load_rows(run_id, player):
    out = {}
    for line in (RUNS / run_id / "episodes.jsonl").read_text(encoding="utf-8").splitlines():
        if '"eval_episode"' not in line[:40]:
            continue
        r = json.loads(line)["payload"]
        if r.get("player") == player:
            out[(r["level"], r["seed"], r["genome_id"])] = r
    return out


def consult_table(s1_id, replay_id, table_path):
    base, replay = load_rows(s1_id, "cmd_steady"), load_rows(replay_id, "cmd_steady_capture")
    table = {":".join(map(str, e["key"])): e for e in json.loads(Path(table_path).read_text(encoding="utf-8"))}
    cards = rb.load_cards(GEO / "enemy_cards.json")
    maps, T = {}, []
    for game, r in sorted(base.items()):
        total = (r["result"].get("final_summary") or {}).get("wave_total") or 15
        caps = {c["consult"]: c for c in replay[game]["result"]["commander"]["consults"]}
        prev_lives = None
        for c in r["result"]["commander"]["consults"]:
            key = f"{game[0]}:{game[1]}:{game[2]}:{c['consult']}"
            cap = caps[c["consult"]]
            lives = c["briefing"].get("lives")
            lost_since = (prev_lives - lives) if isinstance(prev_lives, int) and isinstance(lives, int) else 0
            prev_lives = lives
            if len(c["plays"]) < 2:
                continue
            entry = table[key]
            n = len(c["plays"])
            if game[0] not in maps:
                maps[game[0]] = rb.LevelMap.load(GEO, game[0])
            T.append({"key": key, "game": game, "level": game[0], "seed": game[1], "n": n,
                      "ids": [p["id"] for p in c["plays"]], "plays": c["plays"], "data": c["briefing"],
                      "y": [entry["plays"][str(j)] for j in range(n)], "t": (c.get("wave") or 0) / total,
                      "rule": c["shadow"].get("rule"), "random": c["shadow"].get("random"),
                      "lost_since": lost_since, "capture": cap.get("capture"), "map": maps[game[0]]})
    return T, cards


# -- brains ----------------------------------------------------------------------------------------------------------
def nonp0_argmax(p):
    return max(range(1, len(p)), key=lambda i: (p[i], -i))


def ranking(rows, by_level=False):
    eff = collections.defaultdict(list)
    for t in rows:
        m = statistics.mean(t["y"][1:])
        for j in range(1, t["n"]):
            eff[(t["level"] if by_level else None, t["ids"][j])].append(t["y"][j] - m)
    return {k: statistics.mean(v) for k, v in eff.items()}


def ranked_pick(t, rank, by_level=False):
    def score(j):
        pid = t["ids"][j]
        v = rank.get((t["level"] if by_level else None, pid))
        return v if v is not None else rank.get((None, pid), 0.0)
    return max(range(1, t["n"]), key=lambda j: (score(j), -j))


def threat_flags(t, cards):
    """(flying or cannot be blocked) for the threat of the first counter play (the top threat)."""
    top = next((p["threat"] for p in t["plays"] if p.get("threat")), None)
    if not top:
        return False
    entry = cards.get(top["template"]) or {}
    card = " ".join(entry.get("card") or []).lower()
    return bool(entry.get("flying_template") or "fly" in card or "unblockable" in (top.get("traits") or []))


def rule_rich_pick(t, cards):
    """Same-information rule (fixed a priori): 1 guard the exit that next-wave enemies head for; 2 against a flying or
    unblockable top threat focus the spells; 3 if the top threat's path has no defence over its last half, counter
    (send the hero); 4 otherwise focus the spells (else the first counter)."""
    ids = t["ids"]
    if "P3" in ids and "next wave there" in t["plays"][ids.index("P3")]["text"]:
        return ids.index("P3")
    if threat_flags(t, cards) and "P4" in ids:
        return ids.index("P4")
    top = next((p["threat"] for p in t["plays"] if p.get("threat")), None)
    if top and t["capture"] and t["map"] and 1 <= top["path"] <= len(t["map"].paths):
        quarters = rb.coverage(t["map"], t["capture"].get("towers") or [], top["path"])
        if not quarters[2] and not quarters[3]:
            counter = next((j for j, p in enumerate(t["plays"]) if p["name"] == "counter"
                            and p["threat"]["path"] == top["path"]), None)
            if counter is not None:
                return counter
    if "P4" in ids:
        return ids.index("P4")
    return next(j for j in range(1, t["n"]))


def cluster_boot(per_game, reps, rng):
    games = sorted(per_game)
    if not games:
        return (math.nan, math.nan, math.nan, math.nan)
    def stat(gs):
        num = sum(sum(per_game[g]) for g in gs)
        den = sum(len(per_game[g]) for g in gs)
        return num / den if den else math.nan
    obs = stat(games)
    vals = sorted(stat([rng.choice(games) for _ in games]) for _ in range(reps))
    lo, hi = vals[int(0.025 * reps)], vals[int(0.975 * reps) - 1]
    p_two = 2 * min(sum(v <= 0 for v in vals), sum(v >= 0 for v in vals)) / reps
    return obs, lo, hi, min(1.0, p_two)


def by_game(rows, values):
    out = collections.defaultdict(list)
    for t, v in zip(rows, values):
        if v is not None:
            out[t["game"]].append(v)
    return out


def m1_values(rows, picks):
    """M1 per consult; None where the brain has no preferred deviation (it is then left out)."""
    return [None if j is None else t["y"][j] - statistics.mean(t["y"][1:]) for t, j in zip(rows, picks)]


def spearman(xs, ys):
    def ranks(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r, i = [0.0] * len(v), 0
        while i < len(v):
            j = i
            while j + 1 < len(v) and v[order[j + 1]] == v[order[i]]:
                j += 1
            for k in range(i, j + 1):
                r[order[k]] = (i + j) / 2
            i = j + 1
        return r
    if len(xs) < 2:
        return None
    rx, ry = ranks(xs), ranks(ys)
    mx, my = statistics.mean(rx), statistics.mean(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = math.sqrt(sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry))
    return num / den if den else None


def detrended_timing(rows, signal):
    """Within-game covariance of the brain's deviation signal with g = mean(non-P0) - y_0 after removing a per-game
    quadratic trend in wave fraction from both (the clock control); returns (cov, z, one-sided perm p)."""
    rng = random.Random(9)
    games = collections.defaultdict(list)
    for t, s in zip(rows, signal):
        games[t["game"]].append((t["t"], s, statistics.mean(t["y"][1:]) - t["y"][0]))

    def resid(xs, ts):
        k = min(3, len(set(ts)))
        if k <= 1:
            m = statistics.mean(xs)
            return [x - m for x in xs]
        cols = [[1.0] * len(ts), list(ts), [u * u for u in ts]][:k]
        A = [[sum(a * b for a, b in zip(cols[i], cols[j])) for j in range(k)] for i in range(k)]
        bvec = [sum(a * x for a, x in zip(cols[i], xs)) for i in range(k)]
        for i in range(k):
            piv = max(range(i, k), key=lambda r: abs(A[r][i]))
            A[i], A[piv], bvec[i], bvec[piv] = A[piv], A[i], bvec[piv], bvec[i]
            if abs(A[i][i]) < 1e-12:
                m = statistics.mean(xs)
                return [x - m for x in xs]
            for r in range(k):
                if r != i:
                    f = A[r][i] / A[i][i]
                    A[r] = [u - f * v for u, v in zip(A[r], A[i])]
                    bvec[r] -= f * bvec[i]
        beta = [bvec[i] / A[i][i] for i in range(k)]
        return [x - sum(beta[j] * cols[j][n] for j in range(k)) for n, x in enumerate(xs)]

    rs, rg = {}, {}
    for g, items in games.items():
        ts = [i[0] for i in items]
        rs[g], rg[g] = resid([i[1] for i in items], ts), resid([i[2] for i in items], ts)
    count = sum(len(v) for v in rs.values())
    stat = lambda S: sum(sum(a * b for a, b in zip(S[g], rg[g])) for g in S) / count
    obs = stat(rs)
    null = []
    for _ in range(2000):
        shuffled = {}
        for g, v in rs.items():
            v = list(v)
            rng.shuffle(v)
            shuffled[g] = v
        null.append(stat(shuffled))
    sd = statistics.pstdev(null)
    return obs, (obs / sd if sd else None), (1 + sum(v >= obs for v in null)) / (1 + len(null))


def marginal_perm(rows, picks, reps, rng):
    """One-sided p that M1 >= observed when the brain's preferred play ids are shuffled among consults sharing
    (level, offered non-P0 id set) (keeps its own frequencies; destroys consult-specific information)."""
    strata = collections.defaultdict(list)
    for idx, t in enumerate(rows):
        if picks[idx] is not None:
            strata[(t["level"], tuple(sorted(t["ids"][1:])))].append(idx)
    used = [i for i in range(len(rows)) if picks[i] is not None]
    mean_m1 = lambda ps: statistics.mean(rows[i]["y"][ps[i]] - statistics.mean(rows[i]["y"][1:]) for i in used)
    obs = mean_m1(picks)
    ge = 0
    for _ in range(reps):
        new = list(picks)
        for members in strata.values():
            pids = [rows[i]["ids"][picks[i]] for i in members]
            rng.shuffle(pids)
            for i, pid in zip(members, pids):
                new[i] = rows[i]["ids"].index(pid)
        ge += mean_m1(new) >= obs
    return (1 + ge) / (1 + reps)


def holm(pvalues):
    order = sorted(range(len(pvalues)), key=lambda i: pvalues[i])
    adjusted, running = [0.0] * len(pvalues), 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(pvalues) - rank) * pvalues[i]))
        adjusted[i] = running
    return adjusted


# -- main -------------------------------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("s1")
    ap.add_argument("replay")
    ap.add_argument("table")
    ap.add_argument("results")
    ap.add_argument("--boot", type=int, default=4000)
    ap.add_argument("--perm", type=int, default=2000)
    ap.add_argument("--json")
    args = ap.parse_args()
    rng = random.Random(20261012)
    T, cards = consult_table(args.s1, args.replay, args.table)
    keys = [t["key"] for t in T]
    out = {"consults": len(T), "games": len({t["game"] for t in T}), "brains": {}, "contrasts": {}, "tiers": {}}

    # Language-model arms: p and actual choice per consult.
    lm = {}
    for model, path in (("8b", ANALYSIS_V3 / "llm_choices_s1b.records.json"),
                        ("24b", ANALYSIS_V3 / "llm24_choices_s1b.records.json")):
        recs = {r["key"]: r for r in json.loads(path.read_text(encoding="utf-8"))}
        lm[(model, "thin_old")] = {k: {"p": recs[k]["p"], "choice": recs[k]["choice"]} for k in keys}
    for model in MODELS:
        for arm in ARMS:
            path = Path(args.results) / f"{model}_{arm}.jsonl"
            if not path.exists():
                continue
            recs = {}
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    r = json.loads(line)
                    recs[r["key"]] = {"p": r["p"], "choice": r.get("final_choice", r["choice"]), "stated":
                                      r.get("stated_parsed")}
            missing = [k for k in keys if k not in recs]
            if missing:
                raise SystemExit(f"{model}_{arm}: {len(missing)} consults missing")
            lm[(model, arm)] = recs

    def report(name, picks, choices=None, p_lists=None):
        info = {"consults_with_preferred_deviation": sum(j is not None for j in picks)}
        m1 = m1_values(T, picks)
        info["M1"] = cluster_boot(by_game(T, m1), args.boot, rng)
        for fold, (_, scored) in FOLDS.items():
            rows = [i for i, t in enumerate(T) if t["seed"] in scored]
            info[f"M1_fold{fold}"] = cluster_boot(by_game([T[i] for i in rows], [m1[i] for i in rows]), args.boot, rng)
        if choices is not None:
            m2 = [t["y"][c] - t["y"][0] for t, c in zip(T, choices)]
            info["M2"] = cluster_boot(by_game(T, m2), args.boot, rng)
            info["deviation_rate"] = round(sum(c != 0 for c in choices) / len(choices), 4)
            info["choice_ids"] = dict(collections.Counter(t["ids"][c] for t, c in zip(T, choices)))
            info["timing_detrended"] = detrended_timing(T, [1.0 if c else 0.0 for c in choices])
        info["preferred_ids"] = dict(collections.Counter(t["ids"][j] for t, j in zip(T, picks) if j is not None))
        info["perm_p_marginal"] = marginal_perm(T, picks, args.perm, rng) if any(j is not None for j in picks) else None
        if p_lists is not None:
            rho_all = [spearman(p, t["y"]) for t, p in zip(T, p_lists)]
            rho_np = [spearman(p[1:], t["y"][1:]) if t["n"] >= 3 else None for t, p in zip(T, p_lists)]
            info["spearman_all"] = cluster_boot(by_game(T, rho_all), args.boot, rng)
            info["spearman_nonp0"] = cluster_boot(by_game(T, rho_np), args.boot, rng)
            info["timing_detrended_p"] = detrended_timing(T, [1 - p[0] for p in p_lists])
        out["brains"][name] = info
        return m1

    m1s = {}
    for (model, arm), recs in sorted(lm.items()):
        p_lists = [recs[k]["p"] for k in keys]
        m1s[(model, arm)] = report(f"{model}_{arm}", [nonp0_argmax(p) for p in p_lists],
                                   [recs[k]["choice"] for k in keys], p_lists)
    # Baselines (cross-fitted where they learn from outcomes).
    fixed, perlevel = [None] * len(T), [None] * len(T)
    for fold, (fit, scored) in FOLDS.items():
        fit_rows = [t for t in T if t["seed"] in fit]
        g_rank, l_rank = ranking(fit_rows), ranking(fit_rows, by_level=True)
        for i, t in enumerate(T):
            if t["seed"] in scored:
                fixed[i] = ranked_pick(t, g_rank)
                perlevel[i] = ranked_pick(t, {**g_rank, **l_rank}, by_level=True)
    m1s["fixed"] = report("fixed_ranking_crossfit", fixed)
    m1s["perlevel"] = report("per_level_ranking_crossfit", perlevel)
    m1s["rule_rich"] = report("rule_rich", [rule_rich_pick(t, cards) for t in T])
    for pid in ("P1", "P4"):
        report(f"always_{pid}", [t["ids"].index(pid) if pid in t["ids"] else 1 for t in T])
    report("rule_shadow (M1 on its deviations)", [t["rule"] if t["rule"] else None for t in T], [t["rule"] for t in T])
    report("steady (M2 only)", [None] * len(T), [0] * len(T))
    loss_choice = [fixed[i] if T[i]["lost_since"] >= 2 else 0 for i in range(len(T))]
    report("loss_history_control", fixed, loss_choice)
    oracle = [max(range(1, t["n"]), key=lambda j: (t["y"][j], -j)) for t in T]
    report("oracle_nonp0", oracle)

    def contrast(name, a, b):
        diff = [x - y for x, y in zip(a, b)]
        res = {"pooled": cluster_boot(by_game(T, diff), args.boot, rng)}
        for fold, (_, scored) in FOLDS.items():
            rows = [i for i, t in enumerate(T) if t["seed"] in scored]
            res[f"fold{fold}"] = cluster_boot(by_game([T[i] for i in rows], [diff[i] for i in rows]), args.boot, rng)
        out["contrasts"][name] = res
        return res

    # Primary (a): rich - thin_new, same model (Holm over the two models).
    t1 = {}
    for model in MODELS:
        if (model, "rich") in m1s and (model, "thin_new") in m1s:
            t1[model] = contrast(f"{model}: rich - thin_new", m1s[(model, "rich")], m1s[(model, "thin_new")])
        if (model, "thin_new") in m1s:
            contrast(f"{model}: thin_new - thin_old (format check)", m1s[(model, "thin_new")], m1s[(model, "thin_old")])
        if (model, "rich_reason") in m1s and (model, "rich") in m1s:
            contrast(f"{model}: rich_reason - rich (reasoning)", m1s[(model, "rich_reason")], m1s[(model, "rich")])
    if t1:
        adj = holm([r["pooled"][3] for r in t1.values()])
        out["tiers"]["T1"] = {model: {"diff": r["pooled"][:3], "p": r["pooled"][3], "p_holm": a,
                                      "pass": r["pooled"][1] > 0 and a < 0.05} for (model, r), a in zip(t1.items(), adj)}
        out["tiers"]["T1"]["pass_any"] = any(v["pass"] for v in out["tiers"]["T1"].values())
    # Secondary (b): primary arm vs cross-fitted fixed ranking; (c): rich arms vs rule_rich (Holm over secondaries).
    secondary = {}
    if PRIMARY in m1s:
        secondary["primary - fixed"] = contrast("24b rich_reason - fixed ranking", m1s[PRIMARY], m1s["fixed"])
    for model in MODELS:
        if (model, "rich") in m1s:
            secondary[f"{model} rich - rule_rich"] = contrast(f"{model} rich - rule_rich", m1s[(model, "rich")],
                                                              m1s["rule_rich"])
    if secondary:
        adj = holm([r["pooled"][3] for r in secondary.values()])
        out["contrasts_holm"] = {k: a for k, a in zip(secondary, adj)}
    if PRIMARY in m1s:
        prim = out["brains"]["24b_rich_reason"]
        fix = out["brains"]["fixed_ranking_crossfit"]
        half = all(prim[f"M1_fold{f}"][0] >= 0.5 * fix[f"M1_fold{f}"][0] for f in FOLDS)
        out["tiers"]["T2"] = {"M1": prim["M1"][:3], "fixed_folds": {f: fix[f"M1_fold{f}"][0] for f in FOLDS},
                              "primary_folds": {f: prim[f"M1_fold{f}"][0] for f in FOLDS},
                              "pass": prim["M1"][1] > 0 and half}
        c = out["contrasts"]["24b rich_reason - fixed ranking"]
        out["tiers"]["T3_exploratory"] = {"pass": c["pooled"][1] > 0 and all(c[f"fold{f}"][0] > 0 for f in FOLDS),
                                          "pooled": c["pooled"][:3]}
    # Print.
    fmt = lambda r: f"{r[0]:+.3f} [{r[1]:+.3f},{r[2]:+.3f}]" if r and not math.isnan(r[0]) else "n/a"
    print(f"consults {out['consults']} in {out['games']} base games")
    for name, info in out["brains"].items():
        line = (f"{name:34s} M1 {fmt(info['M1'])}  A {fmt(info['M1_foldA'])}  B {fmt(info['M1_foldB'])}  "
                f"perm p {info['perm_p_marginal'] if info['perm_p_marginal'] is not None else float('nan'):.4f}")
        if "M2" in info:
            line += f"  | M2 {fmt(info['M2'])} dev {info['deviation_rate']:.2f}"
        if "spearman_nonp0" in info:
            line += f"  | rho_np {fmt(info['spearman_nonp0'])}"
        print(line)
    for name, res in out["contrasts"].items():
        print(f"contrast {name:45s} pooled {fmt(res['pooled'])} p {res['pooled'][3]:.4f}  A {fmt(res['foldA'])}  B {fmt(res['foldB'])}")
    print("tiers:", json.dumps(out["tiers"], default=str))
    if "contrasts_holm" in out:
        print("Holm (secondary):", json.dumps(out["contrasts_holm"]))
    if args.json:
        Path(args.json).write_text(json.dumps(out, indent=1, default=str), encoding="utf-8")


if __name__ == "__main__":
    main()
