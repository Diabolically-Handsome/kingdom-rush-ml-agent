"""Play the main campaign in order from a new save, the way a person would.

Level N starts from exactly the progress the earlier victories earned: stars by
the game's own rule from the lives left, star upgrades bought by the fixed rule
after every victory, and a hero only once the hero room offers it. A lost level
is retried (the save is unchanged) with the next candidate plan for that level;
when a level's attempts run out, the campaign ends there. Every attempt is
journaled with its profile, plan and result, so each one can be cold-replayed.
"""

import queue
import threading
import time
from concurrent import futures

from .campaign import (CHALLENGE_MODES, MAIN_LEVELS, challenge_stars, check_profile, check_upgrades, heroes_available,
                       prerequisite, stars_for_lives, upgrades_cost)
from .episode import EpisodeProtocol, run_episode
from .ops import GateRefused
from .search import check_genome, genome_id
from .strategy_brain import RuleBrain, describe_plan

CAMPAIGN_KEYS = {"levels", "seeds", "attempts_per_level", "plans", "policy", "brain"}
# Optional: "start" = {level: stars} victories a rehearsal starts from (never a final run); "challenges" =
# {"level:mode": [plans]} Heroic (2) / Iron (3) challenges of campaign-won main levels, played once the main
# campaign is won and before the elite stages, each with up to "challenge_attempts" attempts; a won challenge
# adds one star to the upgrade budget, a lost one costs nothing else. "star_attempts" = up to that many replays
# of each main level won with fewer than 3 stars, after level 12 and before the challenges (a re-win with more
# lives raises the stored stars, and a level's challenges open only at 3 stars).
OPTIONAL_CAMPAIGN_KEYS = {"start", "challenges", "challenge_attempts", "star_attempts"}
POLICIES = ("plan", "operator", "steps")
BRAINS = ("rule", "8b")


def check_campaign(spec, pools, phase_module):
    """(spec, issues). ``plans`` maps each level to its candidate plans in trial order."""
    if not isinstance(spec, dict) or not CAMPAIGN_KEYS <= set(spec) <= CAMPAIGN_KEYS | OPTIONAL_CAMPAIGN_KEYS:
        return None, [f"campaign must be an object with exactly the keys {sorted(CAMPAIGN_KEYS)} "
                      f"(optionally {sorted(OPTIONAL_CAMPAIGN_KEYS)})"]
    issues = []
    start = spec.get("start", {})
    first = 1
    try:
        start = check_profile({"levels": start})["levels"] if isinstance(start, dict) else None
    except (TypeError, ValueError):
        start = None
    if start is None or (start and sorted(start) != list(range(1, len(start) + 1))):
        issues.append("start must map levels 1..K to their won stars")
    elif start:
        first = len(start) + 1
    levels = spec["levels"]
    if not isinstance(levels, list) or levels != list(range(first, first + len(levels))) or not levels \
            or levels[-1] > 26:
        issues.append("levels must be 1..N in order (a campaign starts from a new save, or K+1..N after start)")
    attempts = spec["attempts_per_level"]
    if isinstance(attempts, bool) or not isinstance(attempts, int) or not 1 <= attempts <= 10:
        issues.append("attempts_per_level must be 1..10")
    challenges = spec.get("challenges", {})
    if ("challenges" in spec) != ("challenge_attempts" in spec):
        issues.append("challenges and challenge_attempts come together")
    tries = spec.get("challenge_attempts", 1)
    if isinstance(tries, bool) or not isinstance(tries, int) or not 1 <= tries <= 10:
        issues.append("challenge_attempts must be 1..10")
    if not isinstance(challenges, dict):
        issues.append("challenges must map 'level:mode' to lists of plans")
        challenges = {}
    replays = spec.get("star_attempts", 0)
    if isinstance(replays, bool) or not isinstance(replays, int) or not 0 <= replays <= 5:
        issues.append("star_attempts must be 0..5")
    reaches_12 = MAIN_LEVELS in (spec.get("levels") or []) or str(MAIN_LEVELS) in (spec.get("start") or {})
    if (challenges or replays) and not reaches_12:
        issues.append("challenges and star replays need a campaign that reaches level 12")
    for key, entries in challenges.items():
        level, _, mode = str(key).partition(":")
        if not (level.isdigit() and mode.isdigit() and 1 <= int(level) <= MAIN_LEVELS
                and int(mode) in CHALLENGE_MODES and key == f"{int(level)}:{int(mode)}"):
            issues.append(f"challenge key {key!r} must be 'level:mode' with level 1-12 and mode 2 or 3")
            continue
        if not isinstance(entries, list) or not entries:
            issues.append(f"challenge {key} has no candidate plan")
            continue
        for index, genome in enumerate(entries):
            try:
                check_genome(plan_genome(genome))
            except ValueError as exc:
                issues.append(f"challenge {key} plan {index}: {exc}")
    if spec["policy"] not in POLICIES:
        issues.append(f"policy must be one of {POLICIES}")
    if spec["brain"] not in BRAINS:
        issues.append(f"brain must be one of {BRAINS}")
    seeds = spec["seeds"]
    if not isinstance(seeds, list) or not seeds or len(set(map(str, seeds))) != len(seeds):
        issues.append("seeds must be a nonempty list of distinct seeds")
        seeds = []
    roles = {phase_module.seed_role(pools, seed) for seed in seeds}
    for seed in seeds:
        role = phase_module.seed_role(pools, seed)
        if role not in ("train", "evaluation", "final_campaign_run"):
            issues.append(f"seed {seed!r} is {role}")
    if len(roles) > 1:
        issues.append("one campaign job plays seeds of a single pool")
    plans = spec["plans"]
    if not isinstance(plans, dict):
        issues.append("plans must map level numbers to lists of plans")
    else:
        for level in levels if isinstance(levels, list) else []:
            candidates = plans.get(str(level))
            if not isinstance(candidates, list) or not candidates:
                issues.append(f"level {level} has no candidate plan")
                continue
            for index, genome in enumerate(candidates):
                try:
                    genome = check_genome(plan_genome(genome))
                except ValueError as exc:
                    issues.append(f"level {level} plan {index}: {exc}")
                    continue
                if genome["hero"] is not None and genome["hero"] not in heroes_available(level):
                    issues.append(f"level {level} plan {index}: hero {genome['hero']} is not unlocked yet")
    return (None if issues else dict(spec)), issues


def plan_genome(entry):
    """A candidate is a plan, or {"genome": plan, "evidence": text shown to the strategy brain}."""
    return entry["genome"] if isinstance(entry, dict) and "genome" in entry and "evidence" in entry else entry


def plan_evidence(entry):
    return entry["evidence"] if isinstance(entry, dict) and "genome" in entry and "evidence" in entry else None


class _Locked:
    """Serializes every method call of a shared journal/context across campaign threads."""

    def __init__(self, target, lock):
        self._target, self._lock = target, lock

    def __getattr__(self, name):
        value = getattr(self._target, name)
        if not callable(value):
            return value

        def call(*args, **kwargs):
            with self._lock:
                return value(*args, **kwargs)
        return call


def run_campaigns(spec, env_factory, ctx, out, *, protocol, make_policy, ports, difficulty=2, brain=None,
                  make_elite_policy=None):
    """One independent campaign per seed, in parallel, each on its own port; returns the summary.
    ``env_factory(level, seed, profile, attempt, port)`` returns an env."""
    if not isinstance(protocol, EpisodeProtocol):
        raise TypeError("protocol must be an EpisodeProtocol")
    lock = threading.RLock()
    shared_ctx, shared_out = _Locked(ctx, lock), _Locked(out, lock)
    free = queue.Queue()
    for port in ports[:len(spec["seeds"])]:
        free.put(port)

    def one(seed):
        port = free.get()
        try:
            return run_campaign({**spec, "seed": seed}, lambda *a, **k: env_factory(*a, port, **k), shared_ctx, shared_out,
                                protocol=protocol, make_policy=make_policy, difficulty=difficulty, brain=brain,
                                make_elite_policy=make_elite_policy)
        finally:
            free.put(port)
    with futures.ThreadPoolExecutor(max_workers=len(spec["seeds"])) as executor:
        results = list(executor.map(one, spec["seeds"]))
    journal = out.verify()
    stopped = [r["stopped_reason"] for r in results if r["stopped_reason"] not in (None, "level_failed")]
    return {"schema": "alpharush-campaign-run-v1", "run_id": ctx.run_id, "policy": spec["policy"],
            "seeds": spec["seeds"], "campaigns": results, "completed": sum(r["completed"] for r in results),
            "games_played": ctx.games_played, "stopped_reason": stopped[0] if stopped else None,
            "protocol": {"name": protocol.name}, "episodes_journal": journal, "optimizer_steps": 0}


def _close(env, out, base):
    """Close a game; a failing close is journaled (the job's process guard reaps the game) and never ends a run."""
    if env is None:
        return
    try:
        env.close()
    except Exception as exc:  # noqa: BLE001
        out.append("env_close_error", {**base, "error": f"{type(exc).__name__}: {exc}"[:300]})


def _progress(won):
    if not won:
        return "new save, no level won yet"
    return ", ".join(f"level {level} won with {stars} star{'s' if stars > 1 else ''}" for level, stars in won.items())


def run_campaign(spec, env_factory, ctx, out, *, protocol, make_policy, difficulty=2, brain=None,
                 make_elite_policy=None):
    """Play one seed's campaign serially; ``env_factory(level, seed, profile, attempt)`` returns an env
    and ``make_policy(genome)`` the decision maker for one attempt (``make_elite_policy``, if given, for the
    elite stages after level 12). The strategy ``brain`` (default: the fixed rules) picks each attempt's
    plan and buys star upgrades after every victory. A lost main-campaign level ends the campaign; a lost
    elite stage only blocks the stages its victory would unlock. Returns that campaign's summary."""
    brain = brain or RuleBrain()
    won, owned = {int(k): v for k, v in (spec.get("start") or {}).items()}, check_upgrades({})
    attempts_log = []
    stopped_reason, failed_level = None, None
    failed_levels, blocked_levels = [], []
    won_challenges, challenge_log, replay_log = {}, [], []
    challenges_played = [not (spec.get("challenges") or spec.get("star_attempts"))]
    started = time.monotonic()
    ctx.record("campaign_start", seed=spec["seed"], levels=spec["levels"], policy=spec["policy"],
               attempts_per_level=spec["attempts_per_level"], brain=brain.name)

    def journal_strategy(records, level, attempt, mode=None):
        for record in records:
            out.append("strategy_decision", {"run_id": ctx.run_id, "seed": spec["seed"], "level": level,
                                             "attempt": attempt, "brain": brain.name,
                                             **({"mode": mode} if mode else {}), **record})

    def total_stars():
        return sum(won.values()) + challenge_stars(won_challenges)

    def allocate(genome, level, mode=None):
        """Like a player on the map screen: look at the plan, then (re)allocate all stars (the upgrades screen's
        reset refunds every star, so each attempt may use the allocation its plan needs)."""
        kinds = {}
        for step in genome["steps"]:
            if step[0] == "b":
                kinds[step[2]] = kinds.get(step[2], 0) + 1
        buy_context = {"progress": _progress(won), "next_level": level, "won": dict(won),
                       "plan": describe_plan(genome, None), "plan_kinds": kinds,
                       "plan_package": genome.get("pkg", "balanced")}
        if won_challenges or mode:
            buy_context.update(challenge_stars=challenge_stars(won_challenges), **({"mode": mode} if mode else {}))
        bought, records = brain.buy(total_stars(), owned, buy_context)
        bought = check_upgrades(bought)
        if upgrades_cost(bought) > total_stars():
            raise RuntimeError("strategy brain bought more upgrades than the stars allow")
        return bought, records

    def replay_for_stars():
        """Replays of main levels won with fewer than 3 stars (their portfolio plans in order, now with the
        upgrades of the whole main campaign); a re-win with more lives raises the stored stars."""
        nonlocal owned
        for level in range(1, MAIN_LEVELS + 1):
            if won.get(level, 3) >= 3 or str(level) not in spec["plans"]:
                continue
            candidates = [(check_genome(plan_genome(e)), plan_evidence(e)) for e in spec["plans"][str(level)]]
            for attempt in range(spec.get("star_attempts", 0)):
                index = attempt % len(candidates)
                genome = candidates[index][0]
                owned, records = allocate(genome, level)
                journal_strategy(records, level, f"star{attempt}")
                profile = check_profile({"upgrades": owned, "hero": genome["hero"], "levels": won,
                                         "challenges": won_challenges})
                if ctx.games_remaining < 1:
                    return "max_games"
                ctx.claim_game()
                ctx.record("game_claimed", level=level, replay=attempt, games_played=ctx.games_played)
                env = None
                base = {"run_id": ctx.run_id, "level": level, "replay": attempt, "seed": spec["seed"],
                        "genome_id": genome_id(genome), "plan_index": index, "genome": genome, "profile": profile,
                        "stars_before": won[level]}
                try:
                    env = env_factory(level, spec["seed"], profile, f"r{attempt}")
                    result = run_episode(env, make_policy(genome), protocol, seed=spec["seed"], level=level,
                                         difficulty=difficulty, check=ctx.check, observe_meta=True,
                                         episode_id=f"{ctx.run_id}-L{level:02d}-r{attempt}")
                except GateRefused:
                    raise
                except Exception as exc:
                    out.append("star_replay_error", {**base, "error": f"{type(exc).__name__}: {exc}"})
                    continue
                finally:
                    _close(env, out, base)
                outcome = result.get("outcome") or {}
                victory = result["status"] == "terminal" and bool(outcome.get("level_won"))
                stars = stars_for_lives(int(outcome["lives"])) if victory else 0
                out.append("star_replay", {**base, "won": victory, "stars": stars,
                                           "result": {k: v for k, v in result.items() if k != "decisions"}})
                replay_log.append({"level": level, "replay": attempt, "won": victory, "stars": stars})
                won[level] = max(won[level], stars)
                if won[level] == 3:
                    break
        return None

    def play_challenges():
        """The 3-star main levels' Heroic/Iron challenges, in level order; returns a stop reason or None."""
        nonlocal owned
        challenges_played[0] = True
        stop = replay_for_stars() if spec.get("star_attempts") else None
        if stop:
            return stop
        make = make_elite_policy if make_elite_policy is not None else make_policy
        for key in sorted(spec.get("challenges", {}), key=lambda k: tuple(int(x) for x in k.split(":"))):
            level, mode = (int(x) for x in key.split(":"))
            if won.get(level) != 3:
                continue  # the map keeps a level's challenges locked below 3 campaign stars
            candidates = [(check_genome(plan_genome(e)), plan_evidence(e)) for e in spec["challenges"][key]]
            failed = []
            for attempt in range(spec["challenge_attempts"]):
                open_plans = [i for i in range(len(candidates)) if i not in failed] or list(range(len(candidates)))
                context = {"progress": _progress(won), "upgrades": dict(owned), "failed": [], "mode": mode}
                choice, records = brain.choose_plan(level, [candidates[i] for i in open_plans], attempt, context)
                index = open_plans[choice % len(open_plans)]
                for record in records:
                    record["plan_index"] = index
                journal_strategy(records, level, attempt, mode)
                genome = candidates[index][0]
                owned, records = allocate(genome, level, mode)
                journal_strategy(records, level, attempt, mode)
                profile = check_profile({"upgrades": owned, "hero": genome["hero"], "levels": won,
                                         "challenges": won_challenges})
                if ctx.games_remaining < 1:
                    return "max_games"
                ctx.claim_game()
                ctx.record("game_claimed", level=level, mode=mode, attempt=attempt, games_played=ctx.games_played)
                env = None
                base = {"run_id": ctx.run_id, "level": level, "mode": mode, "attempt": attempt, "seed": spec["seed"],
                        "genome_id": genome_id(genome), "plan_index": index, "genome": genome, "profile": profile}
                try:
                    env = env_factory(level, spec["seed"], profile, attempt, mode=mode)
                    result = run_episode(env, make(genome), protocol, seed=spec["seed"], level=level,
                                         difficulty=difficulty, check=ctx.check, observe_meta=True,
                                         episode_id=f"{ctx.run_id}-L{level:02d}-m{mode}-a{attempt}")
                except GateRefused:
                    raise
                except Exception as exc:
                    out.append("challenge_error", {**base, "error": f"{type(exc).__name__}: {exc}"})
                    challenge_log.append({"level": level, "mode": mode, "attempt": attempt, "error": True})
                    continue
                finally:
                    _close(env, out, base)
                outcome = result.get("outcome") or {}
                victory = result["status"] == "terminal" and bool(outcome.get("level_won"))
                out.append("challenge_attempt", {**base, "won": victory,
                                                 "result": {k: v for k, v in result.items() if k != "decisions"}})
                challenge_log.append({"level": level, "mode": mode, "attempt": attempt, "won": victory})
                if victory:
                    won_challenges.setdefault(level, []).append(mode)
                    won_challenges[level].sort()
                    break
                failed.append(index)
        return None

    try:
        for level in spec["levels"]:
            if level > MAIN_LEVELS and not challenges_played[0] and MAIN_LEVELS in won:
                stopped_reason = play_challenges()
                if stopped_reason:
                    break
            needed = prerequisite(level)
            if needed is not None and needed not in won:
                # Not unlocked: the elite stage before it in its range was lost.
                blocked_levels.append(level)
                attempts_log.append({"level": level, "blocked_by": needed})
                ctx.record("level_blocked", seed=spec["seed"], level=level, prerequisite=needed)
                continue
            entries = spec["plans"][str(level)]
            candidates = [(check_genome(plan_genome(e)), plan_evidence(e)) for e in entries]
            victory, failed = False, []
            for attempt in range(spec["attempts_per_level"]):
                # The engine is deterministic: a plan already lost here with this save and seed loses again,
                # so a retry offers only the untried plans (all of them again once every plan was tried).
                open_plans = [i for i in range(len(candidates)) if i not in failed] or list(range(len(candidates)))
                context = {"progress": _progress(won), "upgrades": dict(owned), "failed": []}
                choice, records = brain.choose_plan(level, [candidates[i] for i in open_plans], attempt, context)
                index = open_plans[choice % len(open_plans)]
                for record in records:
                    record["plan_index"] = index
                journal_strategy(records, level, attempt)
                genome = candidates[index][0]
                if won:
                    owned, records = allocate(genome, level)
                    journal_strategy(records, level, attempt)
                    ctx.record("upgrades_bought", seed=spec["seed"], before_level=level, upgrades=owned)
                profile = check_profile({"upgrades": owned, "hero": genome["hero"], "levels": won,
                                         "challenges": won_challenges})
                if ctx.games_remaining < 1:
                    stopped_reason = "max_games"
                    break
                ctx.claim_game()
                ctx.record("game_claimed", level=level, attempt=attempt, games_played=ctx.games_played)
                env = None
                base = {"run_id": ctx.run_id, "level": level, "attempt": attempt, "seed": spec["seed"],
                        "genome_id": genome_id(genome), "plan_index": index, "genome": genome, "profile": profile}
                make = make_elite_policy if make_elite_policy is not None and level > MAIN_LEVELS else make_policy
                try:
                    env = env_factory(level, spec["seed"], profile, attempt)
                    result = run_episode(env, make(genome), protocol, seed=spec["seed"], level=level,
                                         difficulty=difficulty, episode_id=f"{ctx.run_id}-L{level:02d}-a{attempt}",
                                         check=ctx.check, observe_meta=True)
                except GateRefused:
                    raise
                except Exception as exc:
                    out.append("campaign_error", {**base, "error": f"{type(exc).__name__}: {exc}"})
                    attempts_log.append({"level": level, "attempt": attempt, "error": True})
                    continue
                finally:
                    _close(env, out, base)
                outcome = result.get("outcome") or {}
                victory = result["status"] == "terminal" and bool(outcome.get("level_won"))
                stars = stars_for_lives(int(outcome["lives"])) if victory else 0
                out.append("campaign_attempt", {**base, "won": victory, "stars": stars,
                                                "result": {k: v for k, v in result.items() if k != "decisions"}})
                attempts_log.append({"level": level, "attempt": attempt, "won": victory, "stars": stars,
                                     "lives": outcome.get("lives"), "wave": (result.get("final_summary") or {}).get("wave")})
                if victory:
                    won[level] = stars
                    break
                failed.append(index)
            if stopped_reason:
                break
            if not victory:
                failed_levels.append(level)
                if failed_level is None:
                    failed_level = level
                if level <= MAIN_LEVELS:
                    stopped_reason = "level_failed"
                    break
        if not stopped_reason and not challenges_played[0] and MAIN_LEVELS in won:
            stopped_reason = play_challenges()
    except GateRefused:
        raise
    except Exception as exc:  # noqa: BLE001
        # An engineering failure outside a game (e.g. the strategy brain still unreachable after the
        # broker's retries) ends this seed's campaign only: its results so far stay in the summary.
        stopped_reason = f"exception: {type(exc).__name__}: {exc}"[:300]
        out.append("campaign_exception", {"run_id": ctx.run_id, "seed": spec["seed"],
                                          "error": stopped_reason})
    played = set(spec["levels"])
    summary = {"seed": spec["seed"], "completed": not failed_levels and not blocked_levels and stopped_reason is None
               and played <= set(won), "won": {str(k): v for k, v in won.items()},
               "total_stars": sum(won.values()), "failed_level": failed_level, "stopped_reason": stopped_reason,
               "attempts": attempts_log, "final_upgrades": owned, "brain": brain.name,
               "wall_seconds": time.monotonic() - started}
    if spec["levels"][-1] > MAIN_LEVELS:
        summary.update(failed_levels=failed_levels, blocked_levels=blocked_levels,
                       levels_won=sorted(level for level in won if level in played))
    if spec.get("star_attempts"):
        summary.update(star_replays=replay_log)
    if spec.get("challenges"):
        summary.update(challenges_won={str(k): v for k, v in won_challenges.items()},
                       challenge_stars=challenge_stars(won_challenges), challenge_attempts=challenge_log)
        summary["total_stars"] = total_stars()
    ctx.record("campaign_end", seed=spec["seed"], completed=summary["completed"], won=summary["won"],
               failed_level=failed_level)
    return summary
