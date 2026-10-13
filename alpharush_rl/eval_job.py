"""Per-level evaluation of a plan player (the ``native-eval`` job body).

Each task plays one level on one seed with one plan, either through the scripted
plan executor ("plan") or through the operator network following the plan's
instructions ("operator"). The commander players ("cmd_*", action scope v4) wrap
the plan executor with an in-battle commander brain (alpharush_rl.commander);
"cmd_fork" plays a task's fork: its base brain, with one consult forced to one
play. Seeds may be train or evaluation seeds: nothing here updates any weights.
Commander players are development evidence for now: elite stages on train seeds
only. The summary reports wins and lives per level and player.
"""

import concurrent.futures as futures
import queue

from . import commander, phase
from .campaign import LAST_LEVEL, MAIN_LEVELS, task_profile
from .episode import EpisodeProtocol, run_episode
from .journal import canonical_json
from .ops import GateRefused
from .search import BuildOrderPolicy, check_genome, genome_id

EVAL_KEYS = {"tasks", "workers", "profile_stars_per_level", "players"}
TASK_KEYS = {"level", "seed", "genome", "mode", "fork"}
COMMANDER_PLAYERS = tuple(commander.PLAYER_BRAINS) + ("cmd_fork",)
PLAYERS = ("plan", "operator", "llm_steps") + COMMANDER_PLAYERS
FORK_KEYS = {"base", "consult", "play"}
# A fork's base brain plays on after the forced consult: a rule brain that acts. The 8B joins only once its
# determinism check (the same prompt scored twice gives the same p) has passed.
FORK_BASES = ("steady", "rule", "random", "placebo")


def needs_language_model(spec):
    """True when a commander player of this eval spec consults the 8B (cmd_8b, or a fork on it)."""
    return "cmd_8b" in spec["players"] or ("cmd_fork" in spec["players"] and any(
        task.get("fork", {}).get("base") == "8b" for task in spec["tasks"]))


def _fork_problems(fork):
    if not isinstance(fork, dict) or set(fork) != FORK_KEYS:
        return [f"fork must be an object with exactly the keys {sorted(FORK_KEYS)}"]
    issues = []
    if fork["base"] not in FORK_BASES:
        issues.append(f"fork base must be one of {FORK_BASES}")
    if not isinstance(fork["consult"], int) or isinstance(fork["consult"], bool) or fork["consult"] < 0:
        issues.append("fork consult must be a consult index >= 0")
    if not isinstance(fork["play"], int) or isinstance(fork["play"], bool) or not 0 <= fork["play"] < commander.MAX_PLAYS:
        issues.append(f"fork play must be a play index 0..{commander.MAX_PLAYS - 1}")
    return issues


def check_eval(spec, pools):
    if not isinstance(spec, dict) or set(spec) != EVAL_KEYS:
        return None, [f"eval must be an object with exactly the keys {sorted(EVAL_KEYS)}"]
    issues = []
    if isinstance(spec["workers"], bool) or not isinstance(spec["workers"], int) or not 1 <= spec["workers"] <= 32:
        issues.append("workers must be 1..32")
    from .campaign import star_rate_ok
    if not star_rate_ok(spec["profile_stars_per_level"]):
        issues.append("profile_stars_per_level must be a number from 1 to 3 (an average star rate) "
                      "or [main rate, elite rate]")
    if not isinstance(spec["players"], list) or not spec["players"] or set(spec["players"]) - set(PLAYERS):
        issues.append(f"players must be a nonempty subset of {PLAYERS}")
    tasks = spec["tasks"]
    if not isinstance(tasks, list) or not tasks:
        return None, issues + ["tasks must be a nonempty list"]
    forking = isinstance(spec["players"], list) and "cmd_fork" in spec["players"]
    if forking and spec["players"] != ["cmd_fork"]:
        issues.append("cmd_fork is played alone (each fork task is one forced consult)")
    # Commander players (cmd_*, cmd_fork included) are development evidence (plan section 5.2): elite stages on
    # train seeds only, never an evaluation seed.
    commanding = isinstance(spec["players"], list) and any(
        isinstance(player, str) and player in COMMANDER_PLAYERS for player in spec["players"])
    roles, seen = set(), set()
    for index, task in enumerate(tasks):
        if not isinstance(task, dict) or not {"level", "seed", "genome"} <= set(task) <= TASK_KEYS:
            issues.append(f"tasks[{index}] must have exactly level, seed and genome (and optionally mode and fork)")
            continue
        if forking != ("fork" in task):
            issues.append(f"tasks[{index}]: a fork task (with fork) is played by cmd_fork, and only fork tasks are")
            continue
        if "fork" in task:
            problems = _fork_problems(task["fork"])
            if problems:
                issues.extend(f"tasks[{index}]: {problem}" for problem in problems)
                continue
        if task.get("mode", 1) not in (1, 2, 3) or isinstance(task.get("mode"), bool) or (
                task.get("mode", 1) != 1 and not (isinstance(task["level"], int) and 1 <= task["level"] <= 12)):
            issues.append(f"tasks[{index}]: mode must be 1, or 2/3 (Heroic/Iron) on levels 1-12")
            continue
        role = phase.seed_role(pools, task["seed"])
        roles.add(role)
        if role not in ("train", "evaluation"):
            issues.append(f"tasks[{index}]: seed {task['seed']!r} is {role}")
        elif commanding and role != "train":
            issues.append(f"tasks[{index}]: commander players (cmd_*) play train seeds only; seed {task['seed']!r} "
                          f"is {role}")
        level = task["level"]
        if commanding and (isinstance(level, bool) or not isinstance(level, int)
                           or not MAIN_LEVELS < level <= LAST_LEVEL):
            issues.append(f"tasks[{index}]: commander players (cmd_*) play only the elite stages "
                          f"(levels {MAIN_LEVELS + 1}-{LAST_LEVEL}), not level {level!r}")
        try:
            key = (task["level"], task.get("mode", 1), task["seed"], genome_id(check_genome(task["genome"])))
            if "fork" in task:
                key += (canonical_json(task["fork"]),)
        except ValueError as exc:
            issues.append(f"tasks[{index}]: {exc}")
            continue
        if key in seen:
            issues.append(f"tasks[{index}] repeats an earlier task")
        seen.add(key)
    return (None if issues else dict(spec)), issues


def eval_loop(spec, env_factory, ctx, out, *, pools, protocol, ports, make_operator=None, make_steps=None,
              difficulty=2, make_commander=None):
    """``env_factory(task, profile, index, player, port)``; ``make_operator(genome)`` for operator players;
    ``make_commander(player, genome, task)`` for commander players (default: commander.make_player without a
    language model, which serves every brain but the 8B)."""
    spec, issues = check_eval(spec, pools)
    if issues:
        raise GateRefused("; ".join(issues))
    if not isinstance(protocol, EpisodeProtocol):
        raise TypeError("protocol must be an EpisodeProtocol")
    if "operator" in spec["players"] and make_operator is None:
        raise GateRefused("an operator evaluation needs operator weights")
    if "llm_steps" in spec["players"] and make_steps is None:
        raise GateRefused("an llm_steps evaluation needs the operator weights and the strategy worker")
    if needs_language_model(spec) and make_commander is None:
        raise GateRefused("an 8B commander evaluation needs the strategy worker")
    make_commander = make_commander or commander.make_player
    free = queue.Queue()
    for port in ports[:spec["workers"]]:
        free.put(port)
    jobs = [(i, player, task) for i, task in enumerate(spec["tasks"]) for player in spec["players"]]
    ctx.record("eval_start", tasks=len(spec["tasks"]), players=spec["players"], games=len(jobs))

    def play(index, player, task):
        port = free.get()
        try:
            genome = check_genome(task["genome"])
            profile = task_profile(task["level"], spec["profile_stars_per_level"], hero=genome["hero"],
                                   package=genome.get("pkg", "balanced"), mode=task.get("mode", 1))
            if player == "plan":
                policy = BuildOrderPolicy(genome)
            elif player == "operator":
                policy = make_operator(genome)
            elif player in COMMANDER_PLAYERS:
                policy = make_commander(player, genome, task)
            else:
                policy = make_steps(genome)
            env = env_factory(task, profile, index, player, port)
            try:
                result = run_episode(env, policy, protocol, seed=task["seed"], level=task["level"],
                                     difficulty=difficulty, episode_id=f"{ctx.run_id}-{index:05d}-{player}",
                                     check=ctx.check)
            finally:
                env.close()
            if player in COMMANDER_PLAYERS:
                # The journal row drops the decisions: keep every consult (briefing, plays, choice, shadows), the
                # commander's own commands and the post-hoc pace check (a violation voids the run in analysis).
                result["commander"] = {**policy.summary(), "command_log": commander.command_log(result["decisions"]),
                                       "pace_violations": commander.pace_violations(result["decisions"])}
            return result
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
                        "seed": task["seed"], "genome_id": genome_id(task["genome"]),
                        **({"mode": task["mode"]} if task.get("mode", 1) != 1 else {}),
                        **({"fork": task["fork"]} if "fork" in task else {})}
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
                key = str(task["level"]) + (f":{task['mode']}" if task.get("mode", 1) != 1 else "")
                entry = results.setdefault(player, {}).setdefault(key, {"games": 0, "wins": 0, "lives": []})
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
