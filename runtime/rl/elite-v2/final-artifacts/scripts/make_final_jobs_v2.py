"""Set one elite-v2 job (eval | dagger | campaign | final) from build_portfolios_v2.py output (option A).

usage: make_final_jobs_v2.py <kind> <portfolios_v2.json> A [key=value ...]
elite-v2: level 11 takes its new portfolio (elite-v2 search 1) topped up with campaign-v1's level-11 plans; fillers
come from the searches of both phases; main=<main operator path> picks the main-campaign network.
options: weights=<elite operator path>  main=<main operator path>  seeds=a,b,..  attempts=N  star_attempts=N  challenge_attempts=N
         workers=N  start=main  plans=N (elite candidates per level)  dagger=beta  players=plan,operator  out=<dry-run file>
         levels=13,15 challenges=0 (eval/dagger/collect: only these elite levels, no challenge tasks)  apply=1
         brain=8b|rule  detour=1 (campaign/final: the stuck-player route, owner-approved 2026-10-10)
         features=timed_clicks,early_waves  click_ticks=N (every job kind: the human-timing host features)
Option A: the main campaign, star replays and the Heroic/Iron challenges of the 3-star main levels, then the elite
stages (A portfolios, searched at [3, 1.0, 12] and [3, 1.2, 13]). Option B: campaign mode only (main campaign, star replays, the elite
stages with the B portfolios searched at [2.6, 1.0]). Elite levels without a validated plan in the family get the
best plans of every elite search (by wins, then mean fitness) so the campaign can still try them. Behavioural
twins (the same game on every shared train game) are dropped from every portfolio and challenge list.
"""
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, r"C:\Users\<user>\Documents\AlphaRush")
from alpharush_rl.search import check_genome, genome_id  # noqa: E402

ROOT = Path(r"C:\Users\<user>\Documents\AlphaRush")
RUN_DIRS = [ROOT / "runtime/rl/elite-v1/runs", ROOT / "runtime/rl/elite-v2/runs"]
CONFIG = ROOT / "configs/phases/elite-v2.json"
APPROVAL = json.loads(CONFIG.read_text(encoding="utf-8"))["authorization"][-1]
PROTOCOL = {"wait_ticks": 120, "max_interval_ticks": 600, "max_ticks": 240000, "max_decisions": 3000}
MAIN_WEIGHTS = "runtime/rl/campaign-v1/models/operator-d86c06c72ae1.json"
MAIN_START = {"1": 3, "2": 3, "3": 3, "4": 2, "5": 3, "6": 2, "7": 3, "8": 3, "9": 3, "10": 3, "11": 2, "12": 2}
PROFILES = {"A": [3, 1.2, 13], "B": [3, 1.0]}  # A: ~62 stars at L24 (brain dry runs: 59-63); B: star replays, no challenges
ELITE = range(13, 27)
CHALLENGE_RUNS = {"native-search-77f73818b99d43b98815025897d07bb3", "native-search-a9ceecb077f5459291a3768e0e2c6f84"}


def weights_of(path):
    return {"path": path, "sha256": hashlib.sha256((ROOT / path).read_bytes()).hexdigest()}


RUN_MODE = {"native-search-77f73818b99d43b98815025897d07bb3": 2, "native-search-a9ceecb077f5459291a3768e0e2c6f84": 3}


def search_table():
    """{(level, mode): {genome_id: {"genome", "seeds": {(run, seed): (won, fitness, trace_sha256)}}}} over every
    elite-v1 search (train seeds only: search jobs are gated to the train pool)."""
    table = {}
    for run in sorted(p for d in RUN_DIRS for p in d.glob("native-search-*") if (p / "episodes.jsonl").exists()):
        mode = RUN_MODE.get(run.name, 1)
        for line in (run / "episodes.jsonl").read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            if row["kind"] not in ("search_eval", "search_validate"):
                continue
            p = row["payload"]
            entry = table.setdefault((p["level"], mode), {}).setdefault(
                p["genome_id"], {"genome": p["genome"], "seeds": {}})
            entry["seeds"][(run.name, p["seed"])] = (p["fitness"] >= 10000, p["fitness"],
                                                    p["result"].get("trace_sha256"))
    return table


def twin(a, b):
    """Behavioural twins: the very same game (trace) on every train game both plans played (at least one)."""
    shared = set(a) & set(b)
    return bool(shared) and all(a[k][2] is not None and a[k][2] == b[k][2] for k in shared)


def dedupe(entries, records):
    """Drop entries that are twins of an earlier entry (a deterministic replay of an already lost game)."""
    kept, seen = [], []
    for entry in entries:
        seeds = records.get(genome_id(check_genome(entry["genome"])), {}).get("seeds", {})
        if any(twin(seeds, other) for other in seen):
            continue
        kept.append(entry)
        seen.append(seeds)
    return kept, seen


def filler(records, exclude, seen, count):
    """The next distinct plans of every run: by smoothed win rate, then mean fitness less a short-record penalty
    (a zero-win plan measured on 10 games outranks a slightly higher 2-game mean)."""
    rows = []
    for gid, entry in records.items():
        games = len(entry["seeds"])
        if gid in exclude or games < 2:
            continue
        wins = sum(w for w, _, _ in entry["seeds"].values())
        fitness = sum(f for _, f, _ in entry["seeds"].values()) / games
        rows.append(((wins + 1) / (games + 4), fitness - 100 / games, fitness, gid, entry, wins, games))
    rows.sort(key=lambda r: (-r[0], -r[1], r[3]))
    out = []
    for _, _, fitness, gid, entry, wins, games in rows:
        if len(out) >= count:
            break
        if any(twin(entry["seeds"], other) for other in seen):
            continue
        seen.append(entry["seeds"])
        text = (f"won {wins} of {games} practice games (other star budgets)" if wins else
                f"no win in {games} practice games; the furthest-reaching plan found (mean score {fitness:.0f})")
        out.append({"genome": check_genome(entry["genome"]), "evidence": text})
    return out


def elite_plans(data, family, size, table):
    plans = {}
    for level in ELITE:
        records = table.get((level, 1), {})
        chosen, seen = dedupe(list(data[f"elite_{family}"].get(str(level), []))[:size], records)
        ids = {genome_id(check_genome(e["genome"])) for e in chosen}
        if len(chosen) < size:
            chosen += filler(records, ids, seen, size - len(chosen))
        plans[str(level)] = chosen
    return plans


def challenge_plans(data, table):
    out = {}
    for key, entries in data["challenges"].items():
        level, mode = (int(x) for x in key.split(":"))
        out[key] = dedupe(entries, table.get((level, mode), {}))[0]
    return out


def main_plans(data, table, size=10):
    """campaign-v1's frozen main-campaign portfolio; level 11 from elite-v2 search 1, topped up with the old plans,
    whose evidence is their elite-v2 re-measurement (their campaign-v1 record came from other seeds)."""
    cfg = json.loads((ROOT / "configs/phases/campaign-v1.json").read_text(encoding="utf-8"))
    plans = dict(cfg["jobs"]["native-final"]["campaign"]["plans"])
    new11 = list(data.get("elite", {}).get("11", []))
    ids = {genome_id(check_genome(e["genome"])) for e in new11}
    def remeasured(e):
        g = check_genome(e["genome"] if isinstance(e, dict) and "genome" in e else e)
        rec = table.get((11, 1), {}).get(genome_id(g), {}).get("seeds", {})
        return -sum(w for w, _, _ in rec.values()) / max(1, len(rec))
    for e in sorted(plans["11"], key=remeasured):  # the best re-measured older plans first
        g = check_genome(e["genome"] if isinstance(e, dict) and "genome" in e else e)
        if len(new11) < size and genome_id(g) not in ids:
            ids.add(genome_id(g))
            rec = table.get((11, 1), {}).get(genome_id(g), {}).get("seeds", {})
            wins, n = sum(w for w, _, _ in rec.values()), len(rec)
            new11.append({"genome": g, "evidence": (f"won {wins} of {n} practice seeds (older plan, re-measured)" if n
                                                    else "older plan, not re-measured")})
    plans["11"] = new11
    return plans


def main():
    kind, source, family = sys.argv[1], Path(sys.argv[2]), sys.argv[3]
    opts = dict(arg.split("=", 1) for arg in sys.argv[4:])
    data = json.loads(source.read_text(encoding="utf-8"))
    data.setdefault("elite_A", {k: v for k, v in data.get("elite", {}).items() if int(k) >= 13})
    profile = PROFILES[family]
    main_weights = opts.get("main", MAIN_WEIGHTS)
    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    base = {"enabled": True, "approval": APPROVAL, "gpu": False, "optimizer_steps": 0, "serial": False,
            "rng_mode": "isolate_sound+stable_pairs", "action_scope": "v3", "replay_fraction": 0.0,
            "time_reserve_seconds": 120, "episode_protocol": dict(PROTOCOL), "max_jobs": 8}
    if kind in ("eval", "dagger"):
        seeds = ([int(s) for s in opts["seeds"].split(",")] if "seeds" in opts else
                 list(range(7101, 7111)) if kind == "eval" else list(range(1001, 1011)))
        per_level = int(opts.get("plans", 1 if kind == "eval" else 3))
        tasks = []
        only = {int(x) for x in opts["levels"].split(",")} if "levels" in opts else None  # e.g. levels=13,15
        for level, entries in sorted(data[f"elite_{family}"].items(), key=lambda kv: int(kv[0])):
            if only is not None and int(level) not in only:
                continue
            for entry in entries[:per_level]:
                tasks += [{"level": int(level), "seed": s, "genome": entry["genome"]} for s in seeds]
        if family == "A" and opts.get("challenges", "1") == "1":
            for key, entries in sorted(data["challenges"].items()):
                level, mode = (int(x) for x in key.split(":"))
                for entry in entries[:max(1, per_level - 1)]:
                    tasks += [{"level": level, "seed": s, "genome": entry["genome"], "mode": mode} for s in seeds]
        if kind == "eval":
            players = opts.get("players", "plan,operator").split(",")
            on_train = all(s < 7000 for s in seeds)
            where = (f"held-out train seeds {seeds[0]}-{seeds[-1]} (no elite operator was trained on them; used to "
                     "choose between operator networks)" if on_train else
                     "evaluation seeds (never used for training or selection)")
            job = {**base, "purpose": (f"option {family}: per-task results of players {players} (elite operator network "
                                       f"{Path(opts['weights']).stem if 'operator' in players else '-'}, layout v4) on "
                                       f"{where}, profile {profile}"),
                   "max_wall_seconds": 7200, "total_wall_seconds": 21600,
                   "max_games": len(tasks) * len(players) + 10,
                   "eval": {"tasks": tasks, "workers": int(opts.get("workers", 32)),
                            "profile_stars_per_level": profile, "players": players}}
            if "operator" in players:
                job["operator_weights"] = weights_of(opts["weights"])
            name = "native-eval"
        else:
            job = {**base, "purpose": (f"option {family}: DAgger: the elite operator network plays the portfolio "
                                       f"plans on train seeds while the plan executor labels every state it reaches, "
                                       f"profile {profile}"),
                   "max_wall_seconds": 7200, "total_wall_seconds": 21600, "max_games": len(tasks) + 20,
                   "operator_weights": weights_of(opts["weights"]),
                   "collect": {"tasks": tasks, "workers": int(opts.get("workers", 32)),
                               "profile_stars_per_level": profile,
                               "dagger": {"beta": float(opts.get("dagger", 0.0))}}}
            name = "native-collect"
    elif kind in ("campaign", "final"):
        final = kind == "final"
        seeds = [int(s) for s in opts.get("seeds", "8101,8102,8103,8104,8105" if final else
                                          "7101,7102,7103,7104,7105").split(",")]
        attempts = int(opts.get("attempts", 10))
        star_attempts = int(opts.get("star_attempts", 2))
        challenge_attempts = int(opts.get("challenge_attempts", 3))
        start = None if final or opts.get("start") != "main" else dict(MAIN_START)
        levels = list(range(13, 27)) if start else list(range(1, 27))
        table = search_table()
        plans = {**main_plans(data, table), **elite_plans(data, family, int(opts.get("plans", 10)), table)}
        plans = {str(level): plans[str(level)] for level in levels}
        campaign = {"levels": levels, "seeds": seeds, "attempts_per_level": attempts, "plans": plans,
                    "policy": "operator", "brain": opts.get("brain", "8b")}
        if start:
            campaign["start"] = start
        if star_attempts:
            campaign["star_attempts"] = star_attempts
        if family == "A":
            campaign["challenges"] = challenge_plans(data, table)
            campaign["challenge_attempts"] = challenge_attempts
        detour = opts.get("detour") == "1"
        if detour:
            campaign["detour"] = True
        games = len(levels) * attempts + (12 * star_attempts if star_attempts else 0) + (
            len(campaign["challenges"]) * challenge_attempts if family == "A" else 0)
        if detour:  # each main level may be retried once after a detour (more replays and challenge attempts)
            games += 12 * attempts + 12 * star_attempts + (
                len(campaign["challenges"]) * challenge_attempts if family == "A" else 0)
        what = ("elite stages 13-26 from a main-cleared save (campaign-v1 final seed 6001's stars)" if start else
                "all 26 levels from a new save")
        extra = ("; after level 12, main levels won with fewer than 3 stars are replayed for stars" if star_attempts
                 else "")
        if family == "A":
            extra += ("; then the Heroic/Iron challenges of the 3-star main levels (each win adds a star to the "
                      "upgrade budget) before the elite stages")
        if detour:
            extra += ("; a main level lost with every plan is retried after a stuck player's detour (replays of the "
                      "earlier levels for stars and their open challenges) if the detour raised the star total")
        who = ("the 8B strategy brain picks" if campaign["brain"] == "8b" else
               "the fixed rule brain (ablation of the 8B strategy brain) picks")
        job = {**base, "purpose": (f"option {family}: {what} on Normal{extra}: {who} each "
                                   "attempt's plan and star allocation (all 13 heroes from level 12 on); the main-"
                                   "campaign operator network (levels 1-12, action scope v2) and the elite operator "
                                   "network (elite stages and challenges, v3) "
                                   f"execute every decision; up to {attempts} attempts per level with untried plans"
                                   + ("; ONE-SHOT final evaluation on the final_campaign_run seeds" if final
                                      else "; rehearsal on evaluation seeds")),
               "max_wall_seconds": int(opts.get("wall", 14400)),
               "total_wall_seconds": 43200 if not final else int(opts.get("wall", 14400)),
               "max_jobs": 1 if final else 8, "max_games": len(seeds) * games + 5,
               "gpu": "external-inference" if campaign["brain"] == "8b" else False,
               "operator_weights": weights_of(main_weights), "elite_operator_weights": weights_of(opts["weights"]),
               "campaign": campaign}
        name = "native-final" if final else "native-campaign"
    else:
        raise SystemExit("unknown kind")
    features = [f for f in opts.get("features", "").split(",") if f]
    if features:
        job["host_features"] = features
        job["purpose"] += f"; human-timing host features {features}"
    if "click_ticks" in opts:
        job["episode_protocol"]["click_ticks"] = int(opts["click_ticks"])
        job["purpose"] += f"; {int(opts['click_ticks'])}-tick trap clicks"
    cfg["jobs"][name] = job
    # The live config is pinned while a job runs: write it only with apply=1 and no job.lock; else to out=<file>.
    if opts.get("apply") == "1":
        if any((ROOT / "runtime/rl").glob("*/job.lock")):
            raise SystemExit("refused: a job holds a runtime/rl/*/job.lock (the config is pinned)")
        target = CONFIG
    elif "out" in opts:
        target = Path(opts["out"])
    else:
        raise SystemExit("give out=<file> (dry run) or apply=1 (live config, only between jobs)")
    target.write_text(json.dumps(cfg, ensure_ascii=False, indent=1) + "\n", encoding="utf-8", newline="\n")
    body = job.get("eval") or job.get("collect") or job.get("campaign")
    print("set", name, "max_games", job["max_games"], "tasks", len(body.get("tasks", [])) if "tasks" in body else "",
          "levels" if "plans" in body else "", {k: len(v) for k, v in body.get("plans", {}).items()})


if __name__ == "__main__":
    main()
