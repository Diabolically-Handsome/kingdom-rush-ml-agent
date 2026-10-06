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

from .campaign import check_profile, check_upgrades, heroes_available, stars_for_lives, upgrades_cost
from .episode import EpisodeProtocol, run_episode
from .ops import GateRefused
from .search import check_genome, genome_id
from .strategy_brain import RuleBrain, describe_plan

CAMPAIGN_KEYS = {"levels", "seeds", "attempts_per_level", "plans", "policy", "brain"}
POLICIES = ("plan", "operator", "steps")
BRAINS = ("rule", "8b")


def check_campaign(spec, pools, phase_module):
    """(spec, issues). ``plans`` maps each level to its candidate plans in trial order."""
    if not isinstance(spec, dict) or set(spec) != CAMPAIGN_KEYS:
        return None, [f"campaign must be an object with exactly the keys {sorted(CAMPAIGN_KEYS)}"]
    issues = []
    levels = spec["levels"]
    if not isinstance(levels, list) or levels != list(range(1, len(levels) + 1)):
        issues.append("levels must be 1..N in order (a campaign starts from a new save)")
    attempts = spec["attempts_per_level"]
    if isinstance(attempts, bool) or not isinstance(attempts, int) or not 1 <= attempts <= 10:
        issues.append("attempts_per_level must be 1..10")
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


def run_campaigns(spec, env_factory, ctx, out, *, protocol, make_policy, ports, difficulty=2, brain=None):
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
            return run_campaign({**spec, "seed": seed}, lambda *a: env_factory(*a, port), shared_ctx, shared_out,
                                protocol=protocol, make_policy=make_policy, difficulty=difficulty, brain=brain)
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


def _progress(won):
    if not won:
        return "new save, no level won yet"
    return ", ".join(f"level {level} won with {stars} star{'s' if stars > 1 else ''}" for level, stars in won.items())


def run_campaign(spec, env_factory, ctx, out, *, protocol, make_policy, difficulty=2, brain=None):
    """Play one seed's campaign serially; ``env_factory(level, seed, profile, attempt)`` returns an env
    and ``make_policy(genome)`` the decision maker for one attempt. The strategy ``brain`` (default:
    the fixed rules) picks each attempt's plan and buys star upgrades after every victory.
    Returns that campaign's summary."""
    brain = brain or RuleBrain()
    won, owned = {}, check_upgrades({})
    attempts_log = []
    stopped_reason, failed_level = None, None
    started = time.monotonic()
    ctx.record("campaign_start", seed=spec["seed"], levels=spec["levels"], policy=spec["policy"],
               attempts_per_level=spec["attempts_per_level"], brain=brain.name)

    def journal_strategy(records, level, attempt):
        for record in records:
            out.append("strategy_decision", {"run_id": ctx.run_id, "seed": spec["seed"], "level": level,
                                             "attempt": attempt, "brain": brain.name, **record})

    for level in spec["levels"]:
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
                # Like a player on the map screen: look at the plan, then (re)allocate all stars (the upgrades
                # screen's reset refunds every star, so each attempt may use the allocation its plan needs).
                kinds = {}
                for step in genome["steps"]:
                    if step[0] == "b":
                        kinds[step[2]] = kinds.get(step[2], 0) + 1
                buy_context = {"progress": _progress(won), "next_level": level, "won": dict(won),
                               "plan": describe_plan(genome, None), "plan_kinds": kinds,
                               "plan_package": genome.get("pkg", "balanced")}
                owned, records = brain.buy(sum(won.values()), owned, buy_context)
                owned = check_upgrades(owned)
                if upgrades_cost(owned) > sum(won.values()):
                    raise RuntimeError("strategy brain bought more upgrades than the stars allow")
                journal_strategy(records, level, attempt)
                ctx.record("upgrades_bought", seed=spec["seed"], before_level=level, upgrades=owned)
            profile = check_profile({"upgrades": owned, "hero": genome["hero"], "levels": won})
            if ctx.games_remaining < 1:
                stopped_reason = "max_games"
                break
            ctx.claim_game()
            ctx.record("game_claimed", level=level, attempt=attempt, games_played=ctx.games_played)
            env = None
            base = {"run_id": ctx.run_id, "level": level, "attempt": attempt, "seed": spec["seed"],
                    "genome_id": genome_id(genome), "plan_index": index, "genome": genome, "profile": profile}
            try:
                env = env_factory(level, spec["seed"], profile, attempt)
                result = run_episode(env, make_policy(genome), protocol, seed=spec["seed"], level=level,
                                     difficulty=difficulty, episode_id=f"{ctx.run_id}-L{level:02d}-a{attempt}",
                                     check=ctx.check, observe_meta=True)
            except GateRefused:
                raise
            except Exception as exc:
                out.append("campaign_error", {**base, "error": f"{type(exc).__name__}: {exc}"})
                attempts_log.append({"level": level, "attempt": attempt, "error": True})
                continue
            finally:
                if env is not None:
                    env.close()
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
            failed_level = level
            stopped_reason = "level_failed"
            break
    summary = {"seed": spec["seed"], "completed": failed_level is None and stopped_reason is None
               and len(won) == len(spec["levels"]), "won": {str(k): v for k, v in won.items()},
               "total_stars": sum(won.values()), "failed_level": failed_level, "stopped_reason": stopped_reason,
               "attempts": attempts_log, "final_upgrades": owned, "brain": brain.name,
               "wall_seconds": time.monotonic() - started}
    ctx.record("campaign_end", seed=spec["seed"], completed=summary["completed"], won=summary["won"],
               failed_level=failed_level)
    return summary
