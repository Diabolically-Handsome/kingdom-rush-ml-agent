"""Per-level evaluation of a plan player (the ``native-eval`` job body).

Each task plays one level on one seed with one plan, either through the scripted
plan executor ("plan") or through the operator network following the plan's
instructions ("operator"). Seeds may be train or evaluation seeds: nothing here
updates any weights. The summary reports wins and lives per level and player.
"""

import concurrent.futures as futures
import queue

from . import phase
from .campaign import level_profile
from .episode import EpisodeProtocol, run_episode
from .ops import GateRefused
from .search import BuildOrderPolicy, check_genome, genome_id

EVAL_KEYS = {"tasks", "workers", "profile_stars_per_level", "players"}
PLAYERS = ("plan", "operator", "llm_steps")


def check_eval(spec, pools):
    if not isinstance(spec, dict) or set(spec) != EVAL_KEYS:
        return None, [f"eval must be an object with exactly the keys {sorted(EVAL_KEYS)}"]
    issues = []
    if isinstance(spec["workers"], bool) or not isinstance(spec["workers"], int) or not 1 <= spec["workers"] <= 32:
        issues.append("workers must be 1..32")
    rate = spec["profile_stars_per_level"]
    if isinstance(rate, bool) or not isinstance(rate, (int, float)) or not 1 <= rate <= 3:
        issues.append("profile_stars_per_level must be a number from 1 to 3 (an average star rate)")
    if not isinstance(spec["players"], list) or not spec["players"] or set(spec["players"]) - set(PLAYERS):
        issues.append(f"players must be a nonempty subset of {PLAYERS}")
    tasks = spec["tasks"]
    if not isinstance(tasks, list) or not tasks:
        return None, issues + ["tasks must be a nonempty list"]
    roles, seen = set(), set()
    for index, task in enumerate(tasks):
        if not isinstance(task, dict) or set(task) != {"level", "seed", "genome"}:
            issues.append(f"tasks[{index}] must have exactly level, seed and genome")
            continue
        role = phase.seed_role(pools, task["seed"])
        roles.add(role)
        if role not in ("train", "evaluation"):
            issues.append(f"tasks[{index}]: seed {task['seed']!r} is {role}")
        try:
            key = (task["level"], task["seed"], genome_id(check_genome(task["genome"])))
        except ValueError as exc:
            issues.append(f"tasks[{index}]: {exc}")
            continue
        if key in seen:
            issues.append(f"tasks[{index}] repeats an earlier task")
        seen.add(key)
    return (None if issues else dict(spec)), issues


def eval_loop(spec, env_factory, ctx, out, *, pools, protocol, ports, make_operator=None, make_steps=None,
              difficulty=2):
    """``env_factory(task, profile, index, player, port)``; ``make_operator(genome)`` for operator players."""
    spec, issues = check_eval(spec, pools)
    if issues:
        raise GateRefused("; ".join(issues))
    if not isinstance(protocol, EpisodeProtocol):
        raise TypeError("protocol must be an EpisodeProtocol")
    if "operator" in spec["players"] and make_operator is None:
        raise GateRefused("an operator evaluation needs operator weights")
    if "llm_steps" in spec["players"] and make_steps is None:
        raise GateRefused("an llm_steps evaluation needs the operator weights and the strategy worker")
    free = queue.Queue()
    for port in ports[:spec["workers"]]:
        free.put(port)
    jobs = [(i, player, task) for i, task in enumerate(spec["tasks"]) for player in spec["players"]]
    ctx.record("eval_start", tasks=len(spec["tasks"]), players=spec["players"], games=len(jobs))

    def play(index, player, task):
        port = free.get()
        try:
            genome = check_genome(task["genome"])
            profile = level_profile(task["level"], spec["profile_stars_per_level"], hero=genome["hero"],
                                    package=genome.get("pkg", "balanced"))
            if player == "plan":
                policy = BuildOrderPolicy(genome)
            elif player == "operator":
                policy = make_operator(genome)
            else:
                policy = make_steps(genome)
            env = env_factory(task, profile, index, player, port)
            try:
                return run_episode(env, policy, protocol, seed=task["seed"], level=task["level"], difficulty=difficulty,
                                   episode_id=f"{ctx.run_id}-{index:05d}-{player}", check=ctx.check)
            finally:
                env.close()
        finally:
            free.put(port)

    results, errors, stopped_reason = {}, [], None
    executor = futures.ThreadPoolExecutor(max_workers=spec["workers"])
    inflight = {}
    try:
        while jobs or inflight:
            while jobs and len(inflight) < spec["workers"]:
                if ctx.games_remaining < 1:
                    stopped_reason, jobs = "max_games", []
                    break
                index, player, task = jobs.pop(0)
                ctx.claim_game()
                ctx.record("game_claimed", index=index, player=player, level=task["level"], games_played=ctx.games_played)
                inflight[executor.submit(play, index, player, task)] = (index, player, task)
            if not inflight:
                break
            finished, _ = futures.wait(list(inflight), return_when=futures.FIRST_COMPLETED)
            for future in finished:
                index, player, task = inflight.pop(future)
                base = {"run_id": ctx.run_id, "index": index, "player": player, "level": task["level"],
                        "seed": task["seed"], "genome_id": genome_id(task["genome"])}
                try:
                    result = future.result()
                except GateRefused:
                    raise
                except Exception as exc:
                    errors.append([index, player])
                    out.append("episode_error", {**base, "error": f"{type(exc).__name__}: {exc}"})
                    continue
                outcome = result.get("outcome") or {}
                won = result["status"] == "terminal" and bool(outcome.get("level_won"))
                out.append("eval_episode", {**base, "won": won, "lives": outcome.get("lives"),
                                            "result": {k: v for k, v in result.items() if k != "decisions"}})
                entry = results.setdefault(player, {}).setdefault(str(task["level"]), {"games": 0, "wins": 0, "lives": []})
                entry["games"] += 1
                entry["wins"] += won
                if won:
                    entry["lives"].append(outcome.get("lives"))
    finally:
        executor.shutdown(wait=True, cancel_futures=True)
    journal = out.verify()
    summary = {"schema": "alpharush-eval-v1", "run_id": ctx.run_id, "players": spec["players"], "results": results,
               "episode_errors": errors, "games_played": ctx.games_played, "stopped_reason": stopped_reason,
               "episodes_journal": journal, "optimizer_steps": 0}
    ctx.record("eval_end", games_played=ctx.games_played, errors=len(errors))
    return summary
