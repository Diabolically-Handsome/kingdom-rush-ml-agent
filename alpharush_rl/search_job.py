"""Parallel build-order search over campaign levels (the ``native-search`` job body).

Every native game is claimed from the job context in the main thread before it
starts and every result is journaled in the main thread; workers (threads, or
processes so that Python work scales past one core) only play games, each on
its own isolated port. Only train-pool seeds are used:
search episodes on ``search_seed``, then the best plans of each level are
re-played on every ``validate_seeds`` seed. Each evaluation is deterministic
(plan + seed + profile), so any journaled row can be cold-replayed later.
"""

import concurrent.futures as futures
import hashlib
import importlib
import json
import multiprocessing
import re
import time
from dataclasses import asdict
from pathlib import Path

from . import phase
from .campaign import heroes_available, level_profile
from .episode import EpisodeProtocol, replay_unsupported, replay_verify, run_episode
from .ops import GateRefused
from .search import (BuildOrderPolicy, LevelSearch, check_genome, fitness, genome_id, level_holders,
                     level_specials, search_signals, ToughestEnemyWatch)

SEARCH_KEYS = {"levels", "search_seed", "population", "evaluations_per_level", "min_evaluations", "stop_lives",
               "profile_stars_per_level", "workers", "validate_seeds", "validate_top", "rng_seed"}
# Optional: warm_start {"runs": [earlier native-search run ids], "top": N, "reevaluate": bool,
# "packages": [star allocations]} takes each level's N best journaled plans (adopted as measured, or
# re-played first; each is also queued with every listed allocation); search_seeds [train seeds]
# scores a plan by its mean fitness over these seeds instead of search_seed alone; levels_only limits the
# search phase to some levels while validation still covers every level.
OPTIONAL_KEYS = {"warm_start", "search_seeds"}
RUN_ID = re.compile(r"native-search-[0-9a-f]{32}")
MAX_WORKERS = 48
WIN = 10000


def _int(value, low, high=None):
    return (not isinstance(value, bool) and isinstance(value, int) and value >= low
            and (high is None or value <= high))


def check_search(spec, pools):
    """(spec, issues): a search job's parameters, with every seed in the train pool."""
    issues = []
    if not isinstance(spec, dict) or not SEARCH_KEYS <= set(spec) <= SEARCH_KEYS | OPTIONAL_KEYS:
        return None, [f"search must be an object with the keys {sorted(SEARCH_KEYS)} (optionally warm_start)"]
    warm = spec.get("warm_start")
    from .campaign import PACKAGES
    if warm is not None and (not isinstance(warm, dict)
                             or not {"runs", "top"} <= set(warm) <= {"runs", "top", "reevaluate", "packages"}
                             or warm.get("reevaluate", False) not in (True, False)
                             or not isinstance(warm.get("packages", []), list)
                             or not set(warm.get("packages", [])) <= set(PACKAGES)
                             or not isinstance(warm["runs"], list) or not warm["runs"]
                             or not all(isinstance(r, str) and RUN_ID.fullmatch(r) for r in warm["runs"])
                             or not _int(warm["top"], 1, 100)):
        issues.append("warm_start must be {runs: [native-search run ids], top: 1..100} "
                      "(optionally reevaluate: bool, packages: [star allocations])")
    levels = spec["levels"]
    if not isinstance(levels, list) or not levels or not all(_int(level, 1, 99) for level in levels) \
            or len(set(levels)) != len(levels):
        issues.append("levels must be distinct positive integers")
    for key, low, high in (("population", 4, 200), ("evaluations_per_level", 1, 100000),
                           ("min_evaluations", 0, 100000), ("stop_lives", 1, 20),
                           ("workers", 1, MAX_WORKERS), ("validate_top", 0, 20)):
        if not _int(spec[key], low, high):
            issues.append(f"{key} must be an integer in {low}..{high}")
    rate = spec["profile_stars_per_level"]
    if isinstance(rate, bool) or not isinstance(rate, (int, float)) or not 1 <= rate <= 3:
        issues.append("profile_stars_per_level must be a number from 1 to 3 (an average star rate)")
    search_seeds = spec.get("search_seeds", [spec["search_seed"]])
    if not isinstance(search_seeds, list) or not search_seeds or spec["search_seed"] not in search_seeds \
            or len(set(map(str, search_seeds))) != len(search_seeds):
        issues.append("search_seeds must be distinct seeds that include search_seed")
        search_seeds = []
    seeds = [spec["search_seed"], *search_seeds] + (
        spec["validate_seeds"] if isinstance(spec["validate_seeds"], list) else [None])
    if not isinstance(spec["validate_seeds"], list) or len(set(map(str, spec["validate_seeds"]))) != len(
            spec["validate_seeds"]):
        issues.append("validate_seeds must be a list of distinct seeds")
    for seed in seeds:
        role = phase.seed_role(pools, seed)
        if role != "train":
            issues.append(f"seed {seed!r} is {role}; a search may only play train seeds")
    if not isinstance(spec["rng_seed"], str) or not spec["rng_seed"]:
        issues.append("rng_seed must be a nonempty string")
    return (None if issues else dict(spec)), issues


def planned_search_games(spec, replay_fraction):
    """Upper bound of native games: probes, evaluations, validations and sampled replays."""
    levels = len(spec["levels"])
    search_seeds = spec.get("search_seeds", [spec["search_seed"]])
    validations = levels * spec["validate_top"] * len([s for s in spec["validate_seeds"] if s not in search_seeds])
    games = levels * (1 + spec["evaluations_per_level"] * len(search_seeds)) + validations
    return games + int(games * replay_fraction + 0.999999)


def _sampled(key, fraction):
    if fraction <= 0:
        return False
    digest = int(hashlib.sha256(key.encode()).hexdigest()[:8], 16)
    return digest / 0xFFFFFFFF < fraction


def _slim(result):
    """An episode result without its per-decision log (the plan replays it exactly)."""
    return {key: value for key, value in result.items() if key != "decisions"}


def _rules(result):
    counts = {}
    for decision in result.get("decisions", []):
        rule = (decision.get("meta") or {}).get("rule")
        counts[rule] = counts.get(rule, 0) + 1
    return dict(sorted(counts.items(), key=lambda kv: str(kv[0])))


def _resolve(path):
    module, _, name = path.partition(":")
    return getattr(importlib.import_module(module), name)


def play_task(task, port, factory, protocol, ctx, difficulty=2):
    """One game (probe, search/validate episode or cold replay); runs in a worker thread or process.

    ``factory`` is a callable ``(task, port) -> env`` or, picklable for processes, a
    ``("module:function", kwargs)`` pair called as ``function(task, port, **kwargs)``.
    """
    if isinstance(factory, tuple):
        env = _resolve(factory[0])(task, port, **factory[1])
    else:
        env = factory(task, port)
    try:
        if task["purpose"] == "probe":
            state = env.reset()
            reader = getattr(env, "meta", None)
            return {"holders": level_holders(state), "specials": level_specials(state),
                    "meta": reader() if callable(reader) else {}}
        if task["purpose"] == "replay":
            return replay_verify(lambda: env, task["result"])
        watch = ToughestEnemyWatch(BuildOrderPolicy(task["genome"]))
        result = run_episode(env, watch, protocol, seed=task["seed"], level=task["level"],
                             difficulty=difficulty, episode_id=task["episode_id"], check=ctx.check, observe_meta=True)
        final_state = getattr(env, "state", None) or {}
        watch.watch(final_state)
        signals = search_signals(result, final_state, toughest=watch.toughest or {"hp": 0.0, "hp_max": 0.0})
        return {"result": {**_slim(result), "search_signals": signals}, "rules": _rules(result)}
    finally:
        env.close()


def native_env(task, port, *, rng_mode, action_scope, tag):
    """The native env of one search task (module-level so worker processes can build it)."""
    from .env import NativeEnv
    identity = f"search_{tag}_{task['identity_index']:06d}" + ("_replay" if task["replay"] else "")
    return NativeEnv(seed=task["seed"], level=task["level"], port=port, difficulty=2, identity=identity,
                     rng_mode=rng_mode, action_scope=action_scope, profile=task["profile"])


def warm_entries(spec, runs_dir, protocol):
    """Each level's best journaled plans of earlier runs played under the same seed, profile and protocol.

    Evaluations are deterministic under those conditions, so their fitness is adopted, not re-measured.
    """
    warm = spec.get("warm_start")
    if not warm:
        return {}
    search_seeds = spec.get("search_seeds", [spec["search_seed"]])
    found = {}
    for run_id in warm["runs"]:
        run = Path(runs_dir) / run_id
        events = [json.loads(line) for line in (run / "events.jsonl").read_text(encoding="utf-8").splitlines() if line]
        start = next((e["payload"] for e in events if e.get("kind") == "search_start"), None)
        if start is None:
            raise GateRefused(f"warm_start run {run_id} has no search_start record")
        earlier = start["spec"]
        # Adopted scores must come from identical conditions; re-played plans only need to be plans.
        if not warm.get("reevaluate"):
            if (earlier["search_seed"], earlier["profile_stars_per_level"]) != (spec["search_seed"],
                                                                                spec["profile_stars_per_level"]):
                raise GateRefused(f"warm_start run {run_id} used another seed or profile")
            same = {k: v for k, v in start["protocol"].items() if k != "name"} == {
                k: v for k, v in asdict(protocol).items() if k != "name"}
            if not same:
                raise GateRefused(f"warm_start run {run_id} used another episode protocol")
        for line in (run / "episodes.jsonl").read_text(encoding="utf-8").splitlines():
            if not line:
                continue
            row = json.loads(line)
            if row.get("kind") != "search_eval":
                continue
            payload = row["payload"]
            if payload["level"] not in spec["levels"] or payload["seed"] not in search_seeds:
                continue
            key = genome_id(payload["genome"])
            entry = found.setdefault(payload["level"], {}).setdefault(
                key, {"genome": check_genome(payload["genome"]), "seeds": {}, "source_run": run_id,
                      "source_sha256": row["sha256"]})
            # The latest measurement of a plan on a seed wins (runs are listed oldest first).
            entry["seeds"][payload["seed"]] = payload["fitness"]
    ranked = {}
    for level, entries in found.items():
        for entry in entries.values():
            # Mean over the search seeds, a missing seed counting 0: plans proven on every seed rank first.
            entry["fitness"] = sum(entry["seeds"].get(seed, 0.0) for seed in search_seeds) / len(search_seeds)
        ranked[level] = sorted(entries.values(), key=lambda e: (-e["fitness"], genome_id(e["genome"])))[:warm["top"]]
    return ranked


def search_loop(spec, env_factory, ctx, out, *, pools, protocol, replay_fraction=0.0, time_reserve_seconds=0,
                clock=time.monotonic, ports=None, difficulty=2, processes=False, runs_dir=None):
    """Run the whole search; returns the summary. ``env_factory`` is described in ``play_task``;
    with ``processes`` it must be the picklable pair form.

    ``task`` holds ``level``, ``seed``, ``profile``, ``identity_index`` and ``replay``.
    """
    spec, issues = check_search(spec, pools)
    if issues:
        raise GateRefused("; ".join(issues))
    if not isinstance(protocol, EpisodeProtocol):
        raise TypeError("protocol must be an EpisodeProtocol")
    workers = spec["workers"]
    ports = list(ports) if ports is not None else list(range(9879, 9879 + workers))
    if len(ports) < workers:
        raise ValueError("fewer ports than workers")
    free_ports = list(ports[:workers])
    warm = warm_entries(spec, runs_dir, protocol) if spec.get("warm_start") else {}
    stars = spec["profile_stars_per_level"]
    search_seeds = spec.get("search_seeds", [spec["search_seed"]])
    multi = {}
    ctx.record("search_start", spec=spec, protocol=asdict(protocol), replay_fraction=replay_fraction,
               planned_games=planned_search_games(spec, replay_fraction), max_games=ctx.max_games)
    counter = [0]
    errors, stopped_reason = [], None
    searches, done_levels = {}, set()
    validation = {}
    replays = []

    def profile_for(level, hero, package="balanced"):
        return level_profile(level, stars, hero=hero, package=package)

    def make_task(purpose, level, seed, genome=None, result=None):
        counter[0] += 1
        hero = genome["hero"] if genome else None
        package = genome.get("pkg", "balanced") if genome else "balanced"
        return {"purpose": purpose, "level": level, "seed": seed, "genome": genome, "result": result,
                "profile": profile_for(level, hero, package), "identity_index": counter[0],
                "replay": purpose == "replay", "episode_id": f"{ctx.run_id}-{counter[0]:06d}"}

    def time_left():
        return not time_reserve_seconds or ctx.deadline - clock() >= time_reserve_seconds

    if processes:
        if not isinstance(env_factory, tuple):
            raise TypeError("worker processes need a picklable (module:function, kwargs) env factory")
        executor = futures.ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn"))
    else:
        executor = futures.ThreadPoolExecutor(max_workers=workers)
    inflight = {}

    def submit(task):
        ctx.claim_game()
        ctx.record("game_claimed", index=task["identity_index"], purpose=task["purpose"], level=task["level"],
                   games_played=ctx.games_played)
        task["port"] = free_ports.pop(0)
        inflight[executor.submit(play_task, task, task["port"], env_factory, protocol, ctx, difficulty)] = task

    def handle(task, future):
        nonlocal stopped_reason
        free_ports.append(task["port"])
        base = {"run_id": ctx.run_id, "index": task["identity_index"], "level": task["level"], "seed": task["seed"],
                "purpose": task["purpose"]}
        try:
            value = future.result()
        except GateRefused:
            raise
        except Exception as exc:
            errors.append(task["identity_index"])
            out.append("episode_error", {**base, "genome_id": genome_id(task["genome"]) if task["genome"] else None,
                                         "error": f"{type(exc).__name__}: {exc}"})
            if task["purpose"] in ("search", "validate") and task["level"] in searches:
                searches[task["level"]].report(task["genome"], -2.0)
            if len(errors) >= 10 and len(errors) * 5 > ctx.games_played:
                stopped_reason = "engine_errors"
            return
        if task["purpose"] == "probe":
            out.append("search_probe", {**base, "holders": value["holders"], "specials": value.get("specials", []),
                                        "meta": value["meta"]})
            search = LevelSearch(task["level"], value["holders"], heroes_available(task["level"]),
                                 spec["rng_seed"], population=spec["population"], specials=value.get("specials", []))
            for entry in warm.get(task["level"], []):
                if spec["warm_start"].get("reevaluate"):
                    search.queue(entry["genome"])
                else:
                    search.adopt(entry["genome"], entry["fitness"])
                out.append("search_import", {"run_id": ctx.run_id, "level": task["level"],
                                             "genome_id": genome_id(entry["genome"]), "genome": entry["genome"],
                                             "fitness": entry["fitness"], "source_run": entry["source_run"],
                                             "source_sha256": entry["source_sha256"]})
                for package in spec["warm_start"].get("packages", []):
                    variant = check_genome({**entry["genome"], "pkg": package})
                    if genome_id(variant) != genome_id(entry["genome"]):
                        search.queue(variant)
                        out.append("search_variant", {"run_id": ctx.run_id, "level": task["level"],
                                                      "genome_id": genome_id(variant), "genome": variant,
                                                      "parent_id": genome_id(entry["genome"])})
            searches[task["level"]] = search
            return
        if task["purpose"] == "replay":
            row = out.append("replay", {**base, "genome_id": genome_id(task["genome"]),
                                        "episode_sha256": task["episode_sha256"], **value})
            replays.append((task["identity_index"], value["replay_verified"]))
            return
        rules, value = value["rules"], value["result"]
        score = fitness(value)
        row = out.append("search_eval" if task["purpose"] == "search" else "search_validate",
                         {**base, "genome_id": genome_id(task["genome"]), "genome": task["genome"],
                          "fitness": score, "rules": rules, "result": value})
        if task["purpose"] == "search":
            key = (task["level"], genome_id(task["genome"]))
            group = multi.setdefault(key, [])
            group.append(score)
            if len(group) == len(search_seeds):
                searches[task["level"]].report(task["genome"], sum(group) / len(group))
                del multi[key]
        else:
            validation.setdefault(task["level"], {}).setdefault(genome_id(task["genome"]), []).append(
                {"seed": task["seed"], "fitness": score, "won": score >= WIN,
                 "lives": (value.get("outcome") or {}).get("lives")})
        if (_sampled(f"{task['level']}:{task['seed']}:{genome_id(task['genome'])}", replay_fraction)
                and not replay_unsupported(value) and ctx.games_remaining >= 1 and time_left()):
            replay = make_task("replay", task["level"], task["seed"], genome=task["genome"], result=value)
            replay["episode_sha256"] = row["sha256"]
            pending_replays.append(replay)

    pending_replays = []

    def level_finished(level):
        search = searches[level]
        best = search.best()
        if search.evaluations + len(search.pending) >= spec["evaluations_per_level"]:
            return True
        return (best is not None and best[0] >= WIN + 100 * spec["stop_lives"]
                and search.evaluations >= spec["min_evaluations"])

    def drain(block=True):
        if not inflight:
            return
        done, _ = futures.wait(list(inflight), return_when=futures.FIRST_COMPLETED if block else futures.ALL_COMPLETED,
                               timeout=None)
        for future in done:
            task = inflight.pop(future)
            handle(task, future)

    try:
        # 1. Probe each level's holders.
        for level in spec["levels"]:
            if ctx.games_remaining < 1 or not time_left():
                stopped_reason = stopped_reason or "budget"
                break
            submit(make_task("probe", level, spec["search_seed"]))
            while len(inflight) >= workers:
                drain()
        while inflight:
            drain()
        # 2. Steady-state search, always feeding the level with the fewest evaluations.
        while stopped_reason is None:
            while pending_replays and len(inflight) < workers and ctx.games_remaining >= 1:
                submit(pending_replays.pop(0))
            open_levels = [level for level in spec["levels"]
                           if level in searches and level not in done_levels and not level_finished(level)]
            for level in [l for l in searches if l not in done_levels and level_finished(l)]:
                done_levels.add(level)
                best = searches[level].best()
                ctx.record("level_search_done", level=level, evaluations=searches[level].evaluations,
                           best_fitness=best[0] if best else None, best_genome_id=best[1] if best else None)
            submitted = False
            while open_levels and len(inflight) + len(search_seeds) <= workers:
                if ctx.games_remaining < 1:
                    stopped_reason = "max_games"
                    break
                if not time_left():
                    stopped_reason = "time_reserve"
                    break
                level = min(open_levels, key=lambda l: (searches[l].evaluations + len(searches[l].pending), l))
                genome = None if level_finished(level) else searches[level].propose()
                if genome is None:
                    open_levels.remove(level)
                    continue
                for seed in search_seeds:
                    submit(make_task("search", level, seed, genome=genome))
                submitted = True
            if not inflight:
                break
            drain()
            if not open_levels and not submitted and not inflight and not pending_replays:
                break
        while inflight:
            drain()
        # 3. Validate the best plans of every level on the other train seeds.
        if stopped_reason is None and spec["validate_top"]:
            chosen = {}
            for level, search in sorted(searches.items()):
                # The best plans by (mean) search fitness, winners first: with several search seeds a plan
                # that won most of them is worth validating too.
                chosen[level] = search.population[:spec["validate_top"]]
                for entry in chosen[level]:
                    validation.setdefault(level, {}).setdefault(entry[1], []).append(
                        {"seed": "search", "fitness": entry[0], "won": entry[0] >= WIN, "lives": None})
            tasks = [make_task("validate", level, seed, genome=entry[2])
                     for level, entries in sorted(chosen.items()) for entry in entries
                     for seed in spec["validate_seeds"] if seed not in search_seeds]
            for task in tasks:
                if ctx.games_remaining < 1 or not time_left():
                    stopped_reason = "validation_budget"
                    break
                while pending_replays and len(inflight) < workers and ctx.games_remaining >= 1:
                    submit(pending_replays.pop(0))
                while len(inflight) >= workers:
                    drain()
                if ctx.games_remaining < 1:
                    stopped_reason = "validation_budget"
                    break
                submit(task)
            while inflight or pending_replays:
                while pending_replays and len(inflight) < workers and ctx.games_remaining >= 1:
                    submit(pending_replays.pop(0))
                if not inflight:
                    break
                drain()
    finally:
        executor.shutdown(wait=True, cancel_futures=True)

    levels = {}
    for level, search in sorted(searches.items()):
        best = search.best()
        checks = validation.get(level, {})
        ranked = sorted(checks.items(), key=lambda kv: (-sum(r["won"] for r in kv[1]),
                                                        -min((r["fitness"] for r in kv[1]), default=0), kv[0]))
        levels[str(level)] = {
            "evaluations": search.evaluations,
            "best_fitness": best[0] if best else None, "best_genome_id": best[1] if best else None,
            "best_genome": best[2] if best else None,
            "validated": [{"genome_id": key, "games": len(rows), "wins": sum(r["won"] for r in rows),
                           "min_fitness": min(r["fitness"] for r in rows),
                           "genome": next((e[2] for e in search.population if e[1] == key), None)}
                          for key, rows in ranked]}
    journal = out.verify()
    summary = {"schema": "alpharush-campaign-search-v1", "run_id": ctx.run_id, "spec": spec,
               "games_played": ctx.games_played, "max_games": ctx.max_games, "levels": levels,
               "episode_errors": errors, "replays_sampled": len(replays),
               "replays_verified": sum(ok is True for _, ok in replays),
               "replay_failures": [index for index, ok in replays if ok is False],
               "stopped_reason": stopped_reason, "protocol": asdict(protocol), "episodes_journal": journal,
               "optimizer_steps": 0}
    ctx.record("search_end", games_played=ctx.games_played, stopped_reason=stopped_reason,
               episodes_tip_sha256=journal["tip_sha256"])
    return summary
