#!/usr/bin/env python3
"""Campaign-v1 native survey skeleton: scripted baselines over a confirmed run list.

Phase 0 allows only ``--check-only`` (the default) and ``--freeze``. ``--run``
passes the same checks, which refuse while ``jobs.native-survey`` is disabled
or declares no ``run_list``; never re-freeze to bypass a refusal.

``--run`` mirrors ``supervise()`` in tools/collect-level1-rewards.py: an owned
Windows job, a permit file, STOP polling and a wall deadline with a 10 second
cleanup reserve, here inside ``phase.job_context`` (lock, ledger, receipt).
The ``--worker`` child plays each run item serially with ``run_episode`` and
appends every result to the run directory's hash-chained ``episodes.jsonl``.
No reward is computed and no gradient is taken here.
"""
from __future__ import annotations

import argparse
import copy
import re
from dataclasses import asdict
import json
import math
import os
from pathlib import Path
import random
import socket
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from alpharush_rl import phase
from alpharush_rl.episode import EpisodeProtocol, replay_unsupported, replay_verify, run_episode
from alpharush_rl.journal import Journal, JournalError, canonical_json, sha256_data
from alpharush_rl.ops import GateRefused, _inside, sha256_file
from alpharush_rl.scripted_policies import make_policy
from alpharush_rl import search_job, campaign_run, collect_job, eval_job

CONFIG = ROOT / "configs/phases/campaign-v1.json"
JOB_KIND = "native-survey"
# Job kinds this supervised tool may run; each has its own caps and ledger count in the phase config.
JOB_KINDS = ("native-survey", "native-diagnose", "native-verify", "native-teacher", "native-search",
             "native-campaign", "native-collect", "native-eval", "native-final")
# Job kinds whose body is the parallel build-order search instead of a run list.
SEARCH_JOB_KINDS = ("native-search",)
# Job kinds with their own body key instead of a run list.
BODY_KEYS = {"native-search": "search", "native-campaign": "campaign", "native-collect": "collect",
             "native-eval": "eval", "native-final": "campaign"}
# The one-shot final campaign evaluation: its own job kind, final_campaign_run seeds only, one job.
FINAL_JOB_KIND = "native-final"
BRAIN_ENDPOINT = "http://127.0.0.1:12081"
STEPS_ENDPOINT = "http://127.0.0.1:12083"
SCHEMA = "alpharush-campaign-survey-v1"
SURVEY_ROLES = ("train", "evaluation")
ITEM_KEYS = ("difficulty", "level", "policy", "seed")
HISTORIC_KEYS = ("branch", "historic_replay")
# Earlier verified native branches whose command plans are replayed cold first;
# any trace or outcome difference stops the whole survey.
HISTORIC_SOURCES = ("runtime/rl/native-validation/branches.json",
                    "runtime/rl/level1-24b-phase1/data/branches.json")
PLAN_REPLAY_KEYS = ("episode", "plan_replay", "repeats", "rng_mode")
# Must equal alpharush_rl.engine.RNG_MODES (checked by a test; engine is not imported here).
RNG_TOKENS = ("audit", "isolate_sound", "stable_pairs")
RNG_MODES = tuple("+".join(t for i, t in enumerate(RNG_TOKENS) if mask >> i & 1) for mask in range(8))
# Must equal alpharush_rl.engine.ACTION_SCOPES (checked by a test). "v1" is the
# build/send_wave scope every earlier plan was recorded under.
ACTION_SCOPES = ("v1", "v2")
DEFAULT_ACTION_SCOPE = "v1"
SURVEY_RUN_ID = re.compile(r"native-survey-[0-9a-f]{32}")
DIFF_LIMIT = 40
DEFAULT_TIME_RESERVE_SECONDS = 0
# Consecutive engineering errors (e.g. a level that will not load) before the survey stops.
MAX_CONSECUTIVE_ERRORS = 5
PORT = 9879
CLEANUP_RESERVE_SECONDS = 10
PERMIT = "owned-job-permit.json"
EPISODES = "episodes.jsonl"
EVENTS = "events.jsonl"
SUMMARY = "survey-summary.json"
SUPERVISOR_RECEIPT = "supervisor-receipt.json"


class SurveyRefused(GateRefused):
    """A refused check; ``result`` is the complete check-only document."""

    def __init__(self, result):
        super().__init__("; ".join(result["issues"]) or "survey refused")
        self.result = result


def _print(value):
    print(json.dumps(value, ensure_ascii=False, indent=2))


def _write_new_json(path, value):
    """Exclusive create: an existing file is evidence and raises FileExistsError."""
    with Path(path).open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n")
    return path


def _integer(value, minimum):
    return not isinstance(value, bool) and isinstance(value, int) and value >= minimum


def replay_fraction_value(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) \
            or not 0 <= value <= 1:
        raise GateRefused(f"replay_fraction must be a number in [0, 1], not {value!r}")
    return float(value)


def time_reserve_value(value):
    """Seconds kept free before the deadline: no new game starts inside this reserve."""
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        raise GateRefused(f"time_reserve_seconds must be a finite number >= 0, not {value!r}")
    return float(value)


def episode_protocol(job):
    """EpisodeProtocol from the optional ``episode_protocol`` mapping; defaults otherwise."""
    fields = job.get("episode_protocol", {})
    if not isinstance(fields, dict):
        raise GateRefused("episode_protocol must be an object")
    try:
        return EpisodeProtocol(**fields)
    except (TypeError, ValueError) as exc:
        raise GateRefused(f"episode_protocol refused: {exc}") from exc


def _core(item):
    if item.get("kind") == "historic_replay":
        return {"historic_replay": item["source"], "branch": item["branch"]}
    if item.get("kind") == "plan_replay":
        return {"plan_replay": item["source_run"], "episode": item["episode"], "rng_mode": item["rng_mode"],
                "repeats": item["repeats"]}
    return {key: item[key] for key in ITEM_KEYS}


def load_survey_episode(root, run_id, index):
    """(episode result, episodes.jsonl sha256, action scope) from an earlier survey run.

    Read-only and journal verified. Rows written before action scopes existed
    were played under "v1".
    """
    if not isinstance(run_id, str) or not SURVEY_RUN_ID.fullmatch(run_id):
        raise GateRefused(f"plan_replay source {run_id!r} is not a native-survey run id")
    path = _inside(Path(root), f"runtime/rl/campaign-v1/runs/{run_id}/{EPISODES}")
    journal = Journal(path)
    journal.verify()
    matches = [row["payload"] for row in journal.entries()
               if row.get("kind") == "episode" and row["payload"].get("index") == index]
    if len(matches) != 1:
        raise GateRefused(f"survey run {run_id} has {len(matches)} episodes with index {index!r}")
    result = matches[0]["result"]
    if not isinstance(result.get("plan"), list) or not isinstance(result.get("trace_sha256"), str):
        raise GateRefused(f"survey episode {run_id}#{index} has no replayable plan")
    return result, sha256_file(path), matches[0].get("action_scope", DEFAULT_ACTION_SCOPE)


def load_historic_branch(root, source, label):
    """(branch, source_sha256) from an allowlisted earlier evidence file; read-only."""
    if source not in HISTORIC_SOURCES:
        raise GateRefused(f"historic source {source!r} is not one of {list(HISTORIC_SOURCES)}")
    path = _inside(Path(root), source)
    document = json.loads(path.read_text(encoding="utf-8"))
    matches = [b for b in document.get("branches", []) if isinstance(b, dict) and b.get("label") == label]
    if len(matches) != 1:
        raise GateRefused(f"historic source {source!r} has {len(matches)} branches labelled {label!r}")
    branch = matches[0]
    trace, plan, outcome = branch.get("trace"), branch.get("plan"), branch.get("outcome")
    if not (isinstance(trace, list) and trace and trace[0].get("kind") == "reset"
            and isinstance(plan, list) and isinstance(outcome, dict)):
        raise GateRefused(f"historic branch {source}#{label} lacks a reset trace, plan or outcome")
    return branch, sha256_file(path)


def historic_expected_outcome(branch):
    """The native terminal() record; later datasets wrap it with scoring context."""
    outcome = branch["outcome"]
    return outcome.get("native_raw", outcome)


def replay_sampled(item, fraction):
    """Deterministic draw from the item itself (never the run id), so plans are reproducible."""
    if item.get("kind") in ("historic_replay", "plan_replay"):
        return False  # the item is itself a cold replay
    if fraction <= 0:
        return False
    draw = random.Random(f"campaign-survey-replay:{item['index']}:{canonical_json(_core(item))}").random()
    return draw < fraction


def planned_games(items, fraction):
    """Native games the items need: one episode each plus every sampled cold replay."""
    return sum(item["repeats"] if item.get("kind") == "plan_replay" else 1 + replay_sampled(item, fraction)
               for item in items)


def _check_plan_replay(raw, index, pools, root):
    """(item, problems) for a {plan_replay, episode, rng_mode, repeats} diagnostic item."""
    problems = []
    if raw["rng_mode"] not in RNG_MODES:
        problems.append(f"rng_mode {raw['rng_mode']!r} is not one of {list(RNG_MODES)}")
    if not _integer(raw["repeats"], 2) or raw["repeats"] > 4:
        problems.append(f"repeats {raw['repeats']!r} must be an integer from 2 to 4")
    if not _integer(raw["episode"], 0):
        problems.append(f"episode {raw['episode']!r} must be a nonnegative integer")
    if problems:
        return None, problems
    try:
        result, digest, scope = load_survey_episode(root, raw["plan_replay"], raw["episode"])
    except (GateRefused, JournalError, OSError, ValueError, TypeError, KeyError) as exc:
        return None, [str(exc)]
    seed, level, difficulty = result.get("seed"), result.get("level"), result.get("difficulty")
    role = phase.seed_role(pools, seed) if _integer(seed, 0) else "never_use"
    if role not in SURVEY_ROLES:
        problems.append(f"episode seed {seed!r} is {role}; only train/evaluation seeds may be replayed")
    if difficulty != pools.get("difficulty"):
        problems.append(f"episode difficulty {difficulty!r} differs from the pool difficulty")
    if scope not in ACTION_SCOPES:
        problems.append(f"episode action_scope {scope!r} is not one of {list(ACTION_SCOPES)}")
    # The plan is replayed under the action scope it was recorded with, never the job's.
    item = {"index": index, "kind": "plan_replay", "source_run": raw["plan_replay"], "episode": raw["episode"],
            "rng_mode": raw["rng_mode"], "repeats": raw["repeats"], "source_sha256": digest,
            "plan_sha256": sha256_data(result["plan"]), "seed": seed, "level": level, "difficulty": difficulty,
            "pool": role, "action_scope": scope}
    return item, problems


def _check_historic(raw, index, pools, root):
    """(item, problems) for a {historic_replay, branch} item."""
    try:
        branch, digest = load_historic_branch(root, raw["historic_replay"], raw["branch"])
    except (GateRefused, OSError, ValueError, TypeError, AttributeError) as exc:
        return None, [str(exc)]
    reset = branch["trace"][0]
    seed, level = reset.get("seed"), reset.get("level")
    role = phase.seed_role(pools, seed) if _integer(seed, 0) else "never_use"
    problems = []
    if role not in SURVEY_ROLES:
        problems.append(f"historic seed {seed!r} is {role}; only train/evaluation seeds may be replayed")
    if not _integer(level, 1):
        problems.append(f"historic level {level!r} must be a positive integer")
    item = {"index": index, "kind": "historic_replay", "source": raw["historic_replay"], "branch": raw["branch"],
            "source_sha256": digest, "seed": seed, "level": level, "difficulty": pools.get("difficulty"),
            "pool": role}
    return item, problems


def check_run_list(run_list, pools, root=None, rng_mode="", action_scope=DEFAULT_ACTION_SCOPE):
    """(items, issues). Every item is checked before any game; any issue refuses all.

    ``rng_mode`` is the job's native RNG mode for episodes and historic replays;
    plan_replay items carry their own. ``action_scope`` is the job's native
    action scope for episodes (and their sampled cold replays). Historic and
    plan replays keep the scope their plan was recorded under: "v1" for the
    historic sources, the source row's for a plan_replay.
    """
    issues = phase.pool_problems(pools)
    if rng_mode not in RNG_MODES:
        issues.append(f"rng_mode {rng_mode!r} is not one of {list(RNG_MODES)}")
    if action_scope not in ACTION_SCOPES:
        issues.append(f"action_scope {action_scope!r} is not one of {list(ACTION_SCOPES)}")
    if issues:
        return [], issues
    if not isinstance(run_list, list) or not run_list:
        return [], ["run_list must be a nonempty list of {level, seed, policy, difficulty} objects"]
    expected_difficulty = pools.get("difficulty")
    items, seen = [], set()
    for index, raw in enumerate(run_list):
        where = f"run_list[{index}]"
        if isinstance(raw, dict) and set(raw) in (set(HISTORIC_KEYS), set(PLAN_REPLAY_KEYS)):
            if root is None:
                issues.append(f"{where}: historic and plan replays need the workspace root")
                continue
            check = _check_historic if set(raw) == set(HISTORIC_KEYS) else _check_plan_replay
            item, problems = check(raw, index, pools, root)
            if problems:
                issues.extend(f"{where}: {problem}" for problem in problems)
                continue
            key = canonical_json(_core(item))
            if key in seen:
                issues.append(f"{where} repeats an earlier item; a deterministic replay adds nothing")
                continue
            seen.add(key)
            if item["kind"] == "historic_replay":
                item["rng_mode"] = rng_mode
                # Historic traces hold v1 states; any other scope changes every state hash.
                item["action_scope"] = DEFAULT_ACTION_SCOPE
            items.append(item)
            continue
        if not isinstance(raw, dict) or set(raw) != set(ITEM_KEYS):
            issues.append(f"{where} must be an object with exactly the keys {list(ITEM_KEYS)}")
            continue
        problems = []
        if not _integer(raw["level"], 1):
            problems.append(f"level {raw['level']!r} must be a positive integer")
        role = phase.seed_role(pools, raw["seed"])
        if role not in SURVEY_ROLES:
            problems.append(f"seed {raw['seed']!r} is {role}; only train/evaluation seeds may be surveyed")
        difficulty = raw["difficulty"]
        if not _integer(difficulty, 0) or (expected_difficulty is not None and difficulty != expected_difficulty):
            problems.append(f"difficulty {difficulty!r} differs from the pool difficulty {expected_difficulty!r}")
        try:
            # The canonical spec is the recorded policy name, so aliases are refused.
            if make_policy(raw["policy"]).name != raw["policy"]:
                problems.append(f"policy {raw['policy']!r} is not a canonical scripted policy spec")
        except ValueError as exc:
            problems.append(f"policy {raw['policy']!r}: {exc}")
        if problems:
            issues.extend(f"{where}: {problem}" for problem in problems)
            continue
        key = canonical_json(_core(raw))
        if key in seen:
            issues.append(f"{where} repeats an earlier item; a deterministic replay adds nothing")
            continue
        seen.add(key)
        items.append({"index": index, **_core(raw), "pool": role, "rng_mode": rng_mode,
                      "action_scope": action_scope})
    return ([] if issues else items), issues


def plan_survey(config=CONFIG, job_kind=JOB_KIND):
    """Read-only check of the declared survey plan; content problems are issues, not raises."""
    plan = dict(items=None, planned_games=None, replay_fraction=None, max_games=None,
                episode_protocol=None, pools=None, issues=[])
    issues = plan["issues"]
    try:
        _, cfg, root = phase.load_phase(config)
    except (OSError, ValueError, GateRefused) as exc:
        issues.append(f"Phase configuration unusable: {exc}")
        return plan
    jobs = cfg.get("jobs")
    if job_kind not in JOB_KINDS:
        issues.append(f"Job kind {job_kind!r} is not one of {list(JOB_KINDS)}")
        return plan
    job = jobs.get(job_kind) if isinstance(jobs, dict) else None
    if not isinstance(job, dict):
        issues.append(f"Unknown job kind {job_kind}")
        return plan
    plan["max_games"] = job.get("max_games")
    searching = job_kind in SEARCH_JOB_KINDS
    body = BODY_KEYS.get(job_kind)
    if body and body not in job:
        issues.append(f"{job_kind} declares no {body} parameters")
    if not body and "run_list" not in job:
        issues.append(f"{job_kind} declares no run_list; the user must confirm one before enabling")
    fraction = None
    if "replay_fraction" not in job:
        issues.append(f"{job_kind} declares no replay_fraction (cold replays are native games too)")
    else:
        try:
            fraction = plan["replay_fraction"] = replay_fraction_value(job["replay_fraction"])
        except GateRefused as exc:
            issues.append(str(exc))
    try:
        plan["episode_protocol"] = asdict(episode_protocol(job))
    except GateRefused as exc:
        issues.append(str(exc))
    try:
        plan["time_reserve_seconds"] = time_reserve_value(job.get("time_reserve_seconds", DEFAULT_TIME_RESERVE_SECONDS))
    except GateRefused as exc:
        issues.append(str(exc))
    try:
        pools = json.loads(_inside(root, cfg["pools_path"]).read_text(encoding="utf-8-sig"))
    except (OSError, ValueError, GateRefused) as exc:
        issues.append(f"Pool manifest unreadable: {exc}")
        return plan
    if phase.pool_problems(pools):  # the preflight lists each pool problem
        issues.append("run_list cannot be checked against an invalid pool manifest")
        return plan
    plan["rng_mode"] = job.get("rng_mode", "")
    plan["action_scope"] = job.get("action_scope", DEFAULT_ACTION_SCOPE)
    if searching and "search" in job:
        spec, problems = search_job.check_search(job["search"], pools)
        issues.extend(f"search: {problem}" for problem in problems)
        if plan["rng_mode"] not in RNG_MODES:
            issues.append(f"rng_mode {plan['rng_mode']!r} is not one of {list(RNG_MODES)}")
        if plan["action_scope"] != "v2":
            issues.append("a build-order search needs action_scope v2")
        if spec and fraction is not None:
            plan["planned_games"] = search_job.planned_search_games(spec, fraction)
            plan["workers"] = spec["workers"]
            if _integer(plan["max_games"], 0) and plan["planned_games"] > plan["max_games"]:
                issues.append(f"search needs up to {plan['planned_games']} native games but max_games is "
                              f"{plan['max_games']}")
    if job_kind == "native-campaign" and "campaign" in job:
        spec, problems = campaign_run.check_campaign(job["campaign"], pools, phase)
        issues.extend(f"campaign: {problem}" for problem in problems)
        issues.extend(_operator_problems(job, root))
        if plan["action_scope"] != "v2" or "isolate_sound" not in plan["rng_mode"].split("+"):
            issues.append("a campaign needs action_scope v2 and a deterministic rng_mode")
        if spec:
            plan["planned_games"] = len(spec["seeds"]) * len(spec["levels"]) * spec["attempts_per_level"]
            plan["workers"] = len(spec["seeds"])
            if _integer(plan["max_games"], 0) and plan["planned_games"] > plan["max_games"]:
                issues.append(f"campaign needs up to {plan['planned_games']} games but max_games is {plan['max_games']}")
    if job_kind == "native-eval" and "eval" in job:
        spec, problems = eval_job.check_eval(job["eval"], pools)
        issues.extend(f"eval: {problem}" for problem in problems)
        if spec and ({"operator", "llm_steps"} & set(spec["players"])):
            issues.extend(_operator_problems({**job, "campaign": {"policy": "operator"}}, root))
        if spec and "llm_steps" in spec["players"] and job.get("gpu") != "external-inference":
            issues.append("an llm_steps evaluation must declare gpu=\"external-inference\"")
        if plan["action_scope"] != "v2" or "isolate_sound" not in plan["rng_mode"].split("+"):
            issues.append("evaluation needs action_scope v2 and a deterministic rng_mode")
        if spec:
            plan["planned_games"] = len(spec["tasks"]) * len(spec["players"])
            plan["workers"] = spec["workers"]
            if _integer(plan["max_games"], 0) and plan["planned_games"] > plan["max_games"]:
                issues.append(f"evaluation needs {plan['planned_games']} games but max_games is {plan['max_games']}")
    if job_kind in ("native-campaign", FINAL_JOB_KIND) and isinstance(job.get("campaign"), dict):
        seeds = job["campaign"].get("seeds") if isinstance(job["campaign"].get("seeds"), list) else []
        final = [s for s in seeds if phase.seed_role(pools, s) == "final_campaign_run"]
        if job_kind == FINAL_JOB_KIND and (len(final) != len(seeds) or job.get("max_jobs") != 1):
            issues.append("native-final plays only final_campaign_run seeds, in exactly one job (max_jobs 1)")
        if job_kind == "native-campaign" and final:
            issues.append("final_campaign_run seeds are reserved for the one native-final job")
    if job_kind == FINAL_JOB_KIND and "campaign" in job:
        spec, problems = campaign_run.check_campaign(job["campaign"], pools, phase)
        issues.extend(f"campaign: {problem}" for problem in problems)
        issues.extend(_operator_problems(job, root))
        if spec:
            plan["planned_games"] = len(spec["seeds"]) * len(spec["levels"]) * spec["attempts_per_level"]
            plan["workers"] = len(spec["seeds"])
    if job_kind in ("native-campaign", FINAL_JOB_KIND) and "campaign" in job and (
            job["campaign"].get("brain") == "8b" or job["campaign"].get("policy") == "steps"):
        if job.get("gpu") != "external-inference":
            issues.append("an 8b-brain campaign must declare gpu=\"external-inference\"")
    if job_kind == "native-collect" and "collect" in job:
        spec, problems = collect_job.check_collect(job["collect"], pools)
        issues.extend(f"collect: {problem}" for problem in problems)
        if spec and spec.get("dagger") is not None:
            issues.extend(_operator_problems({**job, "campaign": {"policy": "operator"}}, root))
        if plan["action_scope"] != "v2" or "isolate_sound" not in plan["rng_mode"].split("+"):
            issues.append("collection needs action_scope v2 and a deterministic rng_mode")
        if spec:
            plan["planned_games"] = len(spec["tasks"])
            plan["workers"] = spec["workers"]
            if _integer(plan["max_games"], 0) and plan["planned_games"] > plan["max_games"]:
                issues.append(f"collection needs {plan['planned_games']} games but max_games is {plan['max_games']}")
    if not body and "run_list" in job:
        items, problems = check_run_list(job["run_list"], pools, root, plan["rng_mode"], plan["action_scope"])
        issues.extend(problems)
        if items:
            plan["items"] = len(items)
            plan["pools"] = {role: sum(item["pool"] == role for item in items) for role in SURVEY_ROLES}
        if items and fraction is not None:
            plan["planned_games"] = planned_games(items, fraction)
            if _integer(plan["max_games"], 0) and plan["planned_games"] > plan["max_games"]:
                issues.append(f"run_list needs {plan['planned_games']} native games (episodes plus sampled "
                              f"replays) but max_games is {plan['max_games']}")
    return plan


def engine_ready():
    """Stat-only check of the prepared engine manifest; never rebuilds. Issue text or None."""
    from alpharush_rl import engine
    try:
        engine.prepare()
    except (RuntimeError, OSError) as exc:
        return f"Engine not prepared: {exc}"
    return None


def check_only(config=CONFIG, job_kind=JOB_KIND):
    """Phase preflight plus the survey plan; strictly read-only, never starts anything."""
    result = phase.preflight_phase(config, job_kind)
    if result["phase_id"] is not None:
        plan = plan_survey(config, job_kind)
        result["survey"] = {key: value for key, value in plan.items() if key != "issues"}
        result["issues"] = result["issues"] + [f"Survey plan: {issue}" for issue in plan["issues"]]
    # An unprepared engine would fail the first reset and spend the job in the ledger.
    engine_issue = engine_ready()
    if engine_issue:
        result["issues"] = result["issues"] + [engine_issue]
    result["ok"] = not result["issues"]
    return result


def verify_frozen(config, expected, job_kind=JOB_KIND):
    """Recheck the exact config, pins and pools the supervisor accepted; return (job, pools)."""
    cp, cfg, root = phase.load_phase(config)
    if sha256_file(cp) != expected["config_sha256"]:
        raise GateRefused("Phase configuration changed after the supervisor check")
    pins = _inside(root, cfg["pins_manifest"])
    if sha256_file(pins) != expected["pins_sha256"]:
        raise GateRefused("Pin manifest changed after the supervisor check")
    for relative, digest in sorted(json.loads(pins.read_text(encoding="utf-8"))["files"].items()):
        if sha256_file(_inside(root, relative)) != digest:
            raise GateRefused(f"Code/config SHA mismatch: {relative}")
    pools_path = _inside(root, cfg["pools_path"])
    if sha256_file(pools_path) != expected["pools_sha256"]:
        raise GateRefused("Pool manifest changed after the supervisor check")
    return cfg["jobs"][job_kind], json.loads(pools_path.read_text(encoding="utf-8-sig"))


def native_env_factory(run_id):
    """One fresh NativeEnv (fresh save identity) per game; imported lazily, never in tests."""
    tag = run_id.rsplit("-", 1)[-1][:12]

    def make(item, replay=False, attempt=None):
        from alpharush_rl.env import NativeEnv
        identity = (f"survey_{tag}_{item['index']:04d}" + ("_replay" if replay else "")
                    + ("" if attempt is None else f"_r{attempt}"))
        return NativeEnv(seed=item["seed"], level=item["level"], port=PORT, difficulty=item["difficulty"],
                         identity=identity, rng_mode=item.get("rng_mode", ""),
                         action_scope=item.get("action_scope", DEFAULT_ACTION_SCOPE))
    return make


def _claim(ctx, item, purpose):
    """Count a game against max_games before it starts, and journal the claim."""
    ctx.claim_game()
    ctx.record("game_claimed", index=item["index"], purpose=purpose, games_played=ctx.games_played)


def _historic_attempt(item, branch, env_factory):
    """One cold replay. Returns (stage, error, trace, outcome); stage is where it stopped."""
    env, stage, error = None, "start", None
    try:
        env = env_factory(dict(item), replay=True)
        env.reset()
        stage = "replay"
        env.replay(copy.deepcopy(branch["plan"]))
        stage = "done"
    except GateRefused:
        raise
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    finally:
        trace = copy.deepcopy(env.trace) if env is not None else []
        outcome = env.terminal() if env is not None and error is None else None
        if env is not None:
            env.close()
    return stage, error, trace, outcome


def _historic_replay(item, env_factory, ctx, out, root):
    """Cold replay of an earlier verified branch: "verified", "mismatch" or "engine_error".

    A process that fails to start or reset is an engineering error, retried once
    in a fresh process; it is never reported as a determinism verdict. Anything
    that goes wrong after the level loaded (including a refused replay command)
    is compared and reported as a mismatch.
    """
    branch, digest = load_historic_branch(root, item["source"], item["branch"])
    base = {"run_id": ctx.run_id, "index": item["index"], "item": _core(item), "source_sha256": digest}
    if digest != item["source_sha256"]:
        out.append("aborted", {**base, "stage": "historic", "error": "historic source changed after the check"})
        raise GateRefused(f"historic source changed: {item['source']}")
    expected_trace, expected_outcome = branch["trace"], historic_expected_outcome(branch)
    for attempt in (1, 2):
        stage, error, trace, outcome = _historic_attempt(item, branch, env_factory)
        if stage != "start":
            break
        out.append("historic_engine_error", {**base, "attempt": attempt, "error": error})
        if attempt == 2 or ctx.games_remaining < 1:
            return "engine_error"
        _claim(ctx, item, "historic_replay_retry")
    mismatch = next((i for i, (a, b) in enumerate(zip(trace, expected_trace)) if a != b),
                    None if len(trace) == len(expected_trace) else min(len(trace), len(expected_trace)))
    row = {**base, "attempt": attempt, "trace_match": error is None and mismatch is None,
           "outcome_match": outcome == expected_outcome, "first_trace_mismatch": mismatch,
           "trace_entries": len(trace), "expected_trace_entries": len(expected_trace),
           "replay_trace_sha256": sha256_data(trace), "expected_trace_sha256": sha256_data(expected_trace),
           "outcome": outcome, "expected_outcome": expected_outcome, "error": error, "stopped_at": stage}
    if mismatch is not None and mismatch < min(len(trace), len(expected_trace)):
        row["mismatch_entry"], row["expected_mismatch_entry"] = trace[mismatch], expected_trace[mismatch]
    row["verified"] = row["trace_match"] and row["outcome_match"]
    out.append("historic_replay", row)
    return "verified" if row["verified"] else "mismatch"


def _short(value):
    text = json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
    return text if len(text) <= 160 else text[:157] + "..."


def state_diff(a, b, path="", out=None, limit=DIFF_LIMIT):
    """[(path, a, b)] for differing leaves (dict keys and list items), at most ``limit`` entries."""
    out = [] if out is None else out
    if len(out) >= limit or a == b:
        return out
    if isinstance(a, dict) and isinstance(b, dict):
        for key in sorted(set(a) | set(b), key=str):
            if len(out) >= limit:
                break
            if key not in a or key not in b:
                out.append((f"{path}/{key}", _short(a.get(key)), _short(b.get(key))))
            elif a[key] != b[key]:
                state_diff(a[key], b[key], f"{path}/{key}", out, limit)
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            out.append((f"{path}#len", len(a), len(b)))
        for i, (x, y) in enumerate(zip(a, b)):
            if len(out) >= limit:
                break
            if x != y:
                state_diff(x, y, f"{path}[{i}]", out, limit)
    else:
        out.append((path, _short(a), _short(b)))
    return out


def _count_diff(a, b):
    a, b = a or {}, b or {}
    return {key: [a.get(key, 0), b.get(key, 0)] for key in sorted(set(a) | set(b)) if a.get(key, 0) != b.get(key, 0)}


def _instrumented_replay(item, plan, env_factory, attempt, audit):
    """Cold replay recording (command, tick, state SHA, cumulative RNG counts) after reset and each command."""
    env, stage, error, steps, states = None, "start", None, [], []

    def record(command):
        state = env.state
        steps.append({"command": command, "tick": state.get("tick"), "state_sha256": sha256_data(state),
                      "rng": env.rng_audit()["counts"] if audit else None})
        states.append(state)
    try:
        env = env_factory(dict(item), replay=True, attempt=attempt)
        env.reset()
        stage = "replay"
        record(-1)
        for index, command in enumerate(copy.deepcopy(plan)):
            if "action" in command:
                env.act(command["action"])
            else:
                env.advance(command["ticks"])
            record(index)
        stage = "done"
    except GateRefused:
        raise
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    finally:
        trace_sha256 = sha256_data(env.trace) if env is not None else None
        outcome = env.terminal() if env is not None and error is None else None
        if env is not None:
            env.close()
    return {"attempt": attempt, "stage": stage, "error": error, "steps": steps, "states": states,
            "trace_sha256": trace_sha256, "outcome": outcome}


def _plan_replay_diagnosis(item, env_factory, ctx, out, root):
    """Replay one survey plan ``repeats`` times; locate the first divergence and its RNG callers."""
    result, digest, _ = load_survey_episode(root, item["source_run"], item["episode"])
    # The scope the plan is replayed under: its source episode's, never the job's.
    base = {"run_id": ctx.run_id, "index": item["index"], "item": _core(item), "source_sha256": digest,
            "action_scope": item["action_scope"]}
    if digest != item["source_sha256"] or sha256_data(result["plan"]) != item["plan_sha256"]:
        out.append("aborted", {**base, "stage": "plan_replay", "error": "source episode changed after the check"})
        raise GateRefused(f"plan_replay source changed: {item['source_run']}#{item['episode']}")
    audit = "audit" in item["rng_mode"]
    runs = []
    for attempt in range(1, item["repeats"] + 1):
        ctx.check()
        _claim(ctx, item, f"plan_replay_{attempt}")
        runs.append(_instrumented_replay(item, result["plan"], env_factory, attempt, audit))
    first = runs[0]
    divergences = []
    for other in runs[1:]:
        shas = [(s["state_sha256"], s["command"]) for s in first["steps"]], \
               [(s["state_sha256"], s["command"]) for s in other["steps"]]
        step = next((i for i, (x, y) in enumerate(zip(*shas)) if x != y),
                    None if len(shas[0]) == len(shas[1]) else min(len(shas[0]), len(shas[1])))
        entry = {"attempt": other["attempt"], "first_divergent_step": step}
        if step is not None and step < min(len(first["steps"]), len(other["steps"])):
            a, b = first["steps"][step], other["steps"][step]
            entry.update(command=a["command"], plan_command=(result["plan"][a["command"]] if a["command"] >= 0 else "reset"),
                         tick=[a["tick"], b["tick"]],
                         state_diff=state_diff(first["states"][step], other["states"][step]),
                         rng_diff_at_step=_count_diff(a["rng"], b["rng"]),
                         rng_diff_before=(_count_diff(first["steps"][step - 1]["rng"], other["steps"][step - 1]["rng"])
                                          if step > 0 else {}))
        divergences.append(entry)
    summary_runs = [{"attempt": r["attempt"], "stage": r["stage"], "error": r["error"], "steps": len(r["steps"]),
                     "trace_sha256": r["trace_sha256"], "outcome": r["outcome"],
                     "matches_original": r["trace_sha256"] == result["trace_sha256"],
                     "final_rng": r["steps"][-1]["rng"] if r["steps"] else None,
                     "state_sha256s": [s["state_sha256"] for s in r["steps"]]} for r in runs]
    all_equal = all(d["first_divergent_step"] is None for d in divergences) and \
        len({(r["error"], r["trace_sha256"], json.dumps(r["outcome"], sort_keys=True)) for r in runs}) == 1
    row = {**base, "runs": summary_runs, "divergences": divergences, "all_repeats_equal": all_equal,
           "original_trace_sha256": result["trace_sha256"], "original_outcome": result["outcome"]}
    out.append("plan_replay_diagnosis", row)
    return all_equal


def worker_loop(run_list, env_factory, ctx, out, *, pools, replay_fraction=0.0, protocol=None,
                root=None, time_reserve_seconds=DEFAULT_TIME_RESERVE_SECONDS, clock=time.monotonic, rng_mode="",
                action_scope=DEFAULT_ACTION_SCOPE):
    """Play every run item serially; ``env_factory(item, replay=False)`` returns an env.

    Each item handed to the factory carries its ``rng_mode`` and ``action_scope``
    (see ``check_run_list``). All items are validated before the first game. Each game (episode, sampled
    cold replay or historic replay) is claimed from ``ctx`` before it starts. The
    loop stops when the next item no longer fits in ``max_games``, when less than
    ``time_reserve_seconds`` remain before ``ctx.deadline``, or at the first
    historic replay whose trace or outcome differs. ``ctx.check`` (STOP/deadline)
    aborts the loop; journaled results stay.
    """
    protocol = EpisodeProtocol() if protocol is None else protocol
    if not isinstance(protocol, EpisodeProtocol):
        raise TypeError("protocol must be an EpisodeProtocol")
    fraction = replay_fraction_value(replay_fraction)
    reserve = time_reserve_value(time_reserve_seconds)
    items, issues = check_run_list(run_list, pools, root, rng_mode, action_scope)
    if issues:
        raise GateRefused("; ".join(issues))
    ctx.record("survey_start", items=len(items), planned_games=planned_games(items, fraction),
               max_games=ctx.max_games, replay_fraction=fraction, protocol=asdict(protocol),
               time_reserve_seconds=reserve)
    statuses, outcomes, by_pool, replays = {}, {"won": 0, "lost": 0}, {}, []
    historic = {"replayed": 0, "verified": 0, "mismatches": [], "engine_errors": []}
    diagnoses = {"items": 0, "repeats_equal": [], "repeats_differ": []}
    errors, consecutive_errors = [], 0
    stopped_reason, not_run = None, []
    for position, item in enumerate(items):
        replay = replay_sampled(item, fraction)
        if ctx.games_remaining < 1 + replay:
            stopped_reason, not_run = "max_games", [rest["index"] for rest in items[position:]]
            ctx.record("max_games_reached", games_played=ctx.games_played, max_games=ctx.max_games,
                       not_run=not_run)
            break
        if reserve and ctx.deadline - clock() < reserve:
            stopped_reason, not_run = "time_reserve", [rest["index"] for rest in items[position:]]
            ctx.record("time_reserve_reached", games_played=ctx.games_played, reserve_seconds=reserve,
                       not_run=not_run)
            break
        if item.get("kind") == "plan_replay":
            if ctx.games_remaining < item["repeats"]:
                stopped_reason, not_run = "max_games", [rest["index"] for rest in items[position:]]
                ctx.record("max_games_reached", games_played=ctx.games_played, max_games=ctx.max_games,
                           not_run=not_run)
                break
            diagnoses["items"] += 1
            equal = _plan_replay_diagnosis(item, env_factory, ctx, out, root)
            diagnoses["repeats_equal" if equal else "repeats_differ"].append(item["index"])
            continue
        if item.get("kind") == "historic_replay":
            ctx.check()
            _claim(ctx, item, "historic_replay")
            historic["replayed"] += 1
            verdict = _historic_replay(item, env_factory, ctx, out, root)
            if verdict == "verified":
                historic["verified"] += 1
                continue
            not_run = [rest["index"] for rest in items[position + 1:]]
            if verdict == "engine_error":
                historic["engine_errors"].append(item["index"])
                stopped_reason = "historic_engine_error"
            else:
                historic["mismatches"].append(item["index"])
                stopped_reason = "determinism_mismatch"
            ctx.record(stopped_reason, index=item["index"], not_run=not_run)
            break
        base = {"run_id": ctx.run_id, "index": item["index"], "item": _core(item)}
        _claim(ctx, item, "episode")
        env = None
        try:
            env = env_factory(dict(item), replay=False)
            result = run_episode(env, make_policy(item["policy"]), protocol, seed=item["seed"],
                                 level=item["level"], difficulty=item["difficulty"],
                                 episode_id=f"{ctx.run_id}-{item['index']:04d}", check=ctx.check,
                                 observe_meta=True)
        except GateRefused as exc:  # STOP or deadline: stop everything
            out.append("aborted", {**base, "stage": "episode", "error": f"{type(exc).__name__}: {exc}"})
            raise
        except Exception as exc:
            # An engineering failure voids this game only; a fresh process starts the next item.
            error = f"{type(exc).__name__}: {exc}"
            out.append("episode_error", {**base, "pool": item["pool"], "error": error})
            errors.append(item["index"])
            consecutive_errors += 1
            if consecutive_errors >= MAX_CONSECUTIVE_ERRORS:
                stopped_reason, not_run = "consecutive_errors", [rest["index"] for rest in items[position + 1:]]
                ctx.record("consecutive_errors", index=item["index"], errors=errors[-MAX_CONSECUTIVE_ERRORS:],
                           not_run=not_run)
                break
            continue
        except BaseException as exc:
            out.append("aborted", {**base, "stage": "episode", "error": f"{type(exc).__name__}: {exc}"})
            raise
        finally:
            if env is not None:
                try:
                    env.close()
                except Exception as exc:
                    ctx.record("env_close_error", index=item["index"], error=f"{type(exc).__name__}: {exc}")
        consecutive_errors = 0
        row = out.append("episode", {**base, "pool": item["pool"], "rng_mode": item.get("rng_mode", ""),
                                     "action_scope": item.get("action_scope", DEFAULT_ACTION_SCOPE),
                                     "replay_sampled": replay, "result": result})
        statuses[result["status"]] = statuses.get(result["status"], 0) + 1
        by_pool[item["pool"]] = by_pool.get(item["pool"], 0) + 1
        if result["outcome"]:
            outcomes["won" if result["outcome"]["level_won"] else "lost"] += 1
        if replay and replay_unsupported(result):
            # Trace holds a step with no plan entry; a cold replay cannot match it.
            replay = False
            out.append("replay_skipped", {**base, "episode_sha256": row["sha256"],
                                          "reason": "void_after_native_side_effect"})
        if replay and reserve and ctx.deadline - clock() < reserve:
            replay = False
            out.append("replay_skipped", {**base, "episode_sha256": row["sha256"], "reason": "time_reserve"})
        if replay:
            _claim(ctx, item, "replay")
            try:
                verdict = replay_verify(lambda: env_factory(dict(item), replay=True), result)
            except GateRefused as exc:
                out.append("aborted", {**base, "stage": "replay", "error": f"{type(exc).__name__}: {exc}"})
                raise
            except Exception as exc:
                # The replay process failed to start or reset: not a determinism verdict.
                verdict = {"replay_verified": None, "replay_trace_sha256": None, "outcome_match": None,
                           "replay_error": f"{type(exc).__name__}: {exc}", "replay_skipped": "replay_engine_error"}
            except BaseException as exc:
                out.append("aborted", {**base, "stage": "replay", "error": f"{type(exc).__name__}: {exc}"})
                raise
            out.append("replay", {**base, "episode_sha256": row["sha256"], **verdict})
            replays.append((item["index"], verdict["replay_verified"]))
    journal = out.verify()
    summary = {"schema": SCHEMA, "run_id": ctx.run_id, "items": len(items),
               "episodes": sum(statuses.values()), "games_played": ctx.games_played, "max_games": ctx.max_games,
               "status_counts": dict(sorted(statuses.items())), "outcomes": outcomes,
               "pools": dict(sorted(by_pool.items())), "replay_fraction": fraction,
               "replays_sampled": len(replays), "replays_verified": sum(ok is True for _, ok in replays),
               "replay_failures": [index for index, ok in replays if ok is False],
               "replay_engine_errors": [index for index, ok in replays if ok is None],
               "episode_errors": errors,
               "historic_replays": historic, "plan_replay_diagnoses": diagnoses, "time_reserve_seconds": reserve,
               "stopped_reason": stopped_reason, "not_run": not_run, "protocol": asdict(protocol), "rng_mode": rng_mode,
               "action_scope": action_scope, "episodes_journal": journal, "optimizer_steps": 0}
    ctx.record("survey_end", episodes=summary["episodes"], games_played=ctx.games_played,
               stopped_reason=stopped_reason, episodes_tip_sha256=journal["tip_sha256"])
    return summary


def _probe_port(port=PORT):
    """Fail before any game if the isolated native port is already occupied."""
    probe = socket.socket()
    try:
        probe.bind(("127.0.0.1", port))
    finally:
        probe.close()


def _job_ports(workers):
    """Isolated native ports a job uses: one per parallel worker (one for serial jobs)."""
    return list(range(PORT, PORT + (workers if isinstance(workers, int) and workers > 0 else 1)))


def _operator_problems(job, root):
    """A campaign played by the operator network names its weights file and their SHA256."""
    if job.get("campaign", {}).get("policy") not in ("operator", "steps"):
        return []
    weights = job.get("operator_weights")
    if not isinstance(weights, dict) or set(weights) != {"path", "sha256"}:
        return ["an operator campaign declares operator_weights {path, sha256}"]
    try:
        path = _inside(root, weights["path"])
    except GateRefused as exc:
        return [f"operator_weights: {exc}"]
    if not path.exists() or sha256_file(path) != weights["sha256"]:
        return ["operator_weights file is missing or its SHA256 differs"]
    return []


def campaign_env_factory(run_id, rng_mode, action_scope):
    tag = run_id.rsplit("-", 1)[-1][:12]

    def make(level, seed, profile, attempt, port):
        from alpharush_rl.env import NativeEnv
        identity = f"campaign_{tag}_{seed}_L{level:02d}_a{attempt}"
        return NativeEnv(seed=seed, level=level, port=port, difficulty=2, identity=identity, rng_mode=rng_mode,
                         action_scope=action_scope, profile=profile)
    return make


def collect_env_factory(run_id, rng_mode, action_scope):
    tag = run_id.rsplit("-", 1)[-1][:12]

    def make(task, profile, index, port):
        from alpharush_rl.env import NativeEnv
        return NativeEnv(seed=task["seed"], level=task["level"], port=port, difficulty=2,
                         identity=f"collect_{tag}_{index:05d}", rng_mode=rng_mode, action_scope=action_scope,
                         profile=profile)
    return make


def search_env_factory(run_id, rng_mode, action_scope):
    """A fresh NativeEnv per search task, on the task's port, from the task's campaign profile."""
    tag = run_id.rsplit("-", 1)[-1][:12]

    def make(task, port):
        from alpharush_rl.env import NativeEnv
        identity = f"search_{tag}_{task['identity_index']:06d}" + ("_replay" if task["replay"] else "")
        return NativeEnv(seed=task["seed"], level=task["level"], port=port, difficulty=2, identity=identity,
                         rng_mode=rng_mode, action_scope=action_scope, profile=task["profile"])
    return make


def _stop_requested(ctx):
    return any((directory / name).exists() for directory in (ctx.state_dir, *ctx.stop_dirs)
               for name in phase.STOP_NAMES)


def _games_claimed(ctx):
    """Claims journaled by the worker; unreadable evidence is charged the full cap."""
    path = ctx.output_dir / EVENTS
    try:
        entries = Journal(path).entries() if path.exists() else []
    except (JournalError, OSError) as exc:
        return ctx.max_games, f"cap_charged: {type(exc).__name__}: {exc}"
    return sum(row.get("kind") == "game_claimed" for row in entries), "worker_events_journal"


def _supervise_run(ctx, cp, root, check, job):
    hard_deadline = ctx.deadline
    execution_deadline = hard_deadline - CLEANUP_RESERVE_SECONDS
    started = time.monotonic()
    permit = ctx.output_dir / PERMIT
    process = None
    status, error, confirmed, exit_code = "failed", None, False, None
    try:
        with (ctx.output_dir / "worker-stdout.json").open("wb") as out, \
                (ctx.output_dir / "worker-stderr.log").open("wb") as err:
            process = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "--worker",
                                        "--config", str(cp), "--run-dir", str(ctx.output_dir),
                                        "--job", ctx.job_kind],
                                       stdout=out, stderr=err, cwd=root, creationflags=subprocess.CREATE_NO_WINDOW)
            job.assign(process)
            # The child cannot start a game until it belongs to this owned job;
            # the permit appears atomically, so the child never reads half of it.
            draft = _write_new_json(permit.with_name(f".{PERMIT}.tmp"), {
                "parent_pid": os.getpid(), "launcher_pid": process.pid, "owned_job_name": job.name,
                "deadline": execution_deadline, "run_id": ctx.run_id, "job_kind": ctx.job_kind,
                "state_dir": str(ctx.state_dir), "stop_dirs": [str(d) for d in ctx.stop_dirs],
                "max_games": ctx.max_games, "config_sha256": check["config_sha256"],
                "pins_sha256": check["pins_sha256"], "pools_sha256": check["pools_sha256"]})
            os.replace(draft, permit)
            while process.poll() is None:
                stopped = _stop_requested(ctx)
                if stopped or time.monotonic() >= execution_deadline:
                    status = "stopped" if stopped else "wall_deadline"
                    job.terminate()
                    break
                time.sleep(0.2)
            process.wait(timeout=max(0.1, hard_deadline - time.monotonic() - 2))
            confirmed, exit_code = True, process.returncode
            if status == "failed" and exit_code == 0 and (ctx.output_dir / SUMMARY).exists():
                stopped = json.loads((ctx.output_dir / SUMMARY).read_text(encoding="utf-8")).get("stopped_reason")
                status = "survey_completed" if stopped is None else "survey_stopped_early"
    except BaseException as exc:
        error = repr(exc)
        job.terminate()
        if process and process.poll() is None:
            process.kill()
            process.wait(timeout=max(0.1, hard_deadline - time.monotonic()))
        confirmed = process is None or process.poll() is not None
        exit_code = None if process is None else process.returncode
        raise
    finally:
        job.close()  # all our game descendants exit, even after child errors
        if confirmed and permit.exists():
            permit.unlink()
        if time.monotonic() >= hard_deadline:
            status = "budget_exceeded"
        ctx.games_played, games_source = _games_claimed(ctx)  # job_context's receipt reads this
        receipt = {"status": status, "error": error, "run_id": ctx.run_id, "worker_exit_confirmed": confirmed,
                   "exit_code": exit_code, "wall_seconds": time.monotonic() - started,
                   "games_played": ctx.games_played, "games_source": games_source, "max_games": ctx.max_games,
                   "optimizer_steps": 0, "gpu": False}
        if (ctx.output_dir / SUMMARY).exists():
            summary = json.loads((ctx.output_dir / SUMMARY).read_text(encoding="utf-8"))
            receipt["summary_sha256"] = sha256_file(ctx.output_dir / SUMMARY)
            receipt["stopped_reason"] = summary.get("stopped_reason")
            receipt["not_run_count"] = len(summary.get("not_run") or [])
        if (ctx.output_dir / EPISODES).exists():
            receipt["episodes_journal"] = Journal(ctx.output_dir / EPISODES).verify()
        _write_new_json(ctx.output_dir / SUPERVISOR_RECEIPT, receipt)
    if status not in ("survey_completed", "survey_stopped_early"):
        refusal = GateRefused if status in ("stopped", "wall_deadline") else RuntimeError
        raise refusal(f"Native survey was not accepted: {status}, exit={exit_code}")
    return receipt


def supervise(config=CONFIG, job_kind=JOB_KIND):
    """Hard deadline/STOP even during native RPC; closing the owned job kills our descendants."""
    check = check_only(config, job_kind)
    if not check["ok"]:
        raise SurveyRefused(check)
    cp, _, root = phase.load_phase(config)
    # Environment failures refuse here, before the sole job is spent in the ledger.
    for port in _job_ports(check.get("survey", {}).get("workers")):
        _probe_port(port)
    from alpharush_rl.windows_job import OwnedProcessJob
    job = OwnedProcessJob()
    try:
        with phase.job_context(cp, job_kind) as ctx:
            if sha256_file(cp) != check["config_sha256"]:
                raise GateRefused("Phase configuration changed after the survey check")
            return _supervise_run(ctx, cp, root, check, job)
    finally:
        job.close()


def worker_main(config, run_dir, *, permit_wait=10.0, job_kind=JOB_KIND):
    run_dir = Path(run_dir).resolve()
    permit_path = run_dir / PERMIT
    deadline = time.monotonic() + permit_wait
    while not permit_path.exists():
        if time.monotonic() >= deadline:
            raise RuntimeError("No owned Windows job permit; no game was started")
        time.sleep(0.05)
    permit = json.loads(permit_path.read_text(encoding="utf-8"))
    direct = permit["launcher_pid"] == os.getpid() and permit["parent_pid"] == os.getppid()
    redirector = permit["launcher_pid"] == os.getppid()
    if not (direct or redirector) or time.monotonic() >= permit["deadline"]:
        raise RuntimeError("Invalid live survey permit")
    state_dir = Path(permit["state_dir"])
    if (job_kind not in JOB_KINDS or permit["job_kind"] != job_kind or permit["run_id"] != run_dir.name
            or run_dir.parent != state_dir / "runs"):
        raise RuntimeError("Survey permit belongs to another run")
    from alpharush_rl.windows_job import confirm_current_job
    confirm_current_job(permit["owned_job_name"])
    ctx = phase.PhaseRunContext(permit["run_id"], run_dir, state_dir, float(permit["deadline"]),
                                int(permit["max_games"]), job_kind=job_kind,
                                stop_dirs=tuple(Path(p) for p in permit["stop_dirs"]))
    ctx.check()
    job, pools = verify_frozen(config, permit, job_kind)
    if job_kind in SEARCH_JOB_KINDS:
        return _search_main(config, permit, ctx, job, pools, run_dir, job_kind)
    if job_kind == "native-campaign":
        return _campaign_main(config, permit, ctx, job, pools, run_dir, job_kind)
    if job_kind == "native-collect":
        return _collect_main(config, permit, ctx, job, pools, run_dir, job_kind)
    if job_kind == "native-eval":
        return _eval_main(config, permit, ctx, job, pools, run_dir, job_kind)
    if job_kind == FINAL_JOB_KIND:
        return _campaign_main(config, permit, ctx, job, pools, run_dir, job_kind)
    if "run_list" not in job:
        raise GateRefused(f"{job_kind} declares no run_list; no game was started")
    _probe_port()
    engine_before = engine_identity()
    ctx.record("engine_identity", **engine_before)
    summary = worker_loop(job["run_list"], native_env_factory(ctx.run_id), ctx, Journal(run_dir / EPISODES),
                          pools=pools, replay_fraction=job.get("replay_fraction"), protocol=episode_protocol(job),
                          root=ROOT, time_reserve_seconds=job.get("time_reserve_seconds", DEFAULT_TIME_RESERVE_SECONDS),
                          rng_mode=job.get("rng_mode", ""), action_scope=job.get("action_scope", DEFAULT_ACTION_SCOPE))
    summary["engine"] = engine_before
    # Results are written first, so a failed final check never discards them.
    _write_new_json(run_dir / SUMMARY, summary)
    verify_frozen(config, permit, job_kind)  # sources unchanged during the native run
    if engine_identity() != engine_before:
        raise GateRefused("Prepared engine changed during the native run")
    if summary["stopped_reason"] in ("determinism_mismatch", "historic_engine_error"):
        # The summary is kept as evidence, but the job must not be accepted.
        raise RuntimeError(f"Historic re-verification did not pass ({summary['stopped_reason']}); survey stopped")
    return summary


def _search_main(config, permit, ctx, job, pools, run_dir, job_kind):
    if "search" not in job:
        raise GateRefused(f"{job_kind} declares no search parameters; no game was started")
    spec, problems = search_job.check_search(job["search"], pools)
    if problems:
        raise GateRefused("; ".join(problems))
    ports = _job_ports(spec["workers"])
    for port in ports:
        _probe_port(port)
    engine_before = engine_identity()
    ctx.record("engine_identity", **engine_before)
    rng_mode, action_scope = job.get("rng_mode", ""), job.get("action_scope", DEFAULT_ACTION_SCOPE)
    # Worker processes (not threads): per-game Python work otherwise saturates one core.
    factory = ("alpharush_rl.search_job:native_env",
               {"rng_mode": rng_mode, "action_scope": action_scope, "tag": ctx.run_id.rsplit("-", 1)[-1][:12]})
    summary = search_job.search_loop(
        spec, factory, ctx, Journal(run_dir / EPISODES),
        pools=pools, protocol=episode_protocol(job), replay_fraction=replay_fraction_value(job.get("replay_fraction")),
        time_reserve_seconds=time_reserve_value(job.get("time_reserve_seconds", DEFAULT_TIME_RESERVE_SECONDS)),
        ports=ports, processes=True, runs_dir=run_dir.parent)
    summary.update(engine=engine_before, rng_mode=rng_mode, action_scope=action_scope)
    _write_new_json(run_dir / SUMMARY, summary)
    verify_frozen(config, permit, job_kind)
    if engine_identity() != engine_before:
        raise GateRefused("Prepared engine changed during the native run")
    return summary


def _finish(config, permit, run_dir, job_kind, summary, engine_before, job):
    summary.update(engine=engine_before, rng_mode=job.get("rng_mode", ""),
                   action_scope=job.get("action_scope", DEFAULT_ACTION_SCOPE))
    _write_new_json(run_dir / SUMMARY, summary)
    verify_frozen(config, permit, job_kind)
    if engine_identity() != engine_before:
        raise GateRefused("Prepared engine changed during the native run")
    return summary


def _campaign_main(config, permit, ctx, job, pools, run_dir, job_kind):
    spec, problems = campaign_run.check_campaign(job.get("campaign"), pools, phase)
    problems += _operator_problems(job, ROOT)
    if problems:
        raise GateRefused("; ".join(problems))
    ports = _job_ports(len(spec["seeds"]))
    for port in ports:
        _probe_port(port)
    engine_before = engine_identity()
    ctx.record("engine_identity", **engine_before)
    if spec["policy"] == "operator":
        from alpharush_rl.operator_net import OperatorPolicy, OptionScorer
        weights_path = _inside(ROOT, job["operator_weights"]["path"])
        net = OptionScorer.from_json(json.loads(weights_path.read_text(encoding="utf-8")))
        name = "operator:" + job["operator_weights"]["sha256"][:12]
        make = lambda genome: OperatorPolicy(net, genome, name=name)
    elif spec["policy"] == "steps":
        import urllib.request
        from alpharush_rl.llm_steps import LanguageStepSource, StepOperatorPolicy
        from alpharush_rl.model_broker import LanguageModelBroker
        from alpharush_rl.operator_net import OptionScorer
        with urllib.request.urlopen(STEPS_ENDPOINT + "/", timeout=10) as stream:
            health = json.load(stream)
        if health.get("ready") is not True:
            raise GateRefused("the strategy worker is not ready")
        ctx.record("strategy_worker", endpoint=STEPS_ENDPOINT, model=health.get("model"))
        weights_path = _inside(ROOT, job["operator_weights"]["path"])
        net = OptionScorer.from_json(json.loads(weights_path.read_text(encoding="utf-8")))
        name = "8b-steps+operator:" + job["operator_weights"]["sha256"][:12]

        def make(genome):
            source = LanguageStepSource(LanguageModelBroker(STEPS_ENDPOINT, timeout=300), cast=genome["cast"],
                                        early=genome["early"], branches=genome["branches"])
            return StepOperatorPolicy(net, source, name=name)
    else:
        from alpharush_rl.search import BuildOrderPolicy
        make = BuildOrderPolicy
    rng_mode, action_scope = job.get("rng_mode", ""), job.get("action_scope", DEFAULT_ACTION_SCOPE)
    brain = None
    if spec["brain"] == "8b":
        from alpharush_rl.model_broker import LanguageModelBroker
        from alpharush_rl.strategy_brain import LanguageBrain
        import urllib.request
        with urllib.request.urlopen(BRAIN_ENDPOINT + "/", timeout=10) as stream:
            health = json.load(stream)
        if health.get("ready") is not True:
            raise GateRefused("the 8B scoring server is not ready")
        ctx.record("strategy_brain", endpoint=BRAIN_ENDPOINT, model=health.get("model"))
        brain = LanguageBrain(LanguageModelBroker(BRAIN_ENDPOINT, timeout=180), name="8b")
    summary = campaign_run.run_campaigns(spec, campaign_env_factory(ctx.run_id, rng_mode, action_scope), ctx,
                                         Journal(run_dir / EPISODES), protocol=episode_protocol(job),
                                         make_policy=make, ports=ports, brain=brain)
    return _finish(config, permit, run_dir, job_kind, summary, engine_before, job)


def _eval_main(config, permit, ctx, job, pools, run_dir, job_kind):
    spec, problems = eval_job.check_eval(job.get("eval"), pools)
    if problems:
        raise GateRefused("; ".join(problems))
    make_operator = make_steps = None
    if "llm_steps" in spec["players"]:
        problems = _operator_problems({**job, "campaign": {"policy": "operator"}}, ROOT)
        if problems:
            raise GateRefused("; ".join(problems))
        import urllib.request
        from alpharush_rl.llm_steps import LanguageStepSource, StepOperatorPolicy
        from alpharush_rl.model_broker import LanguageModelBroker
        from alpharush_rl.operator_net import OptionScorer
        with urllib.request.urlopen(STEPS_ENDPOINT + "/", timeout=10) as stream:
            health = json.load(stream)
        if health.get("ready") is not True:
            raise GateRefused("the strategy worker is not ready")
        ctx.record("strategy_worker", endpoint=STEPS_ENDPOINT, model=health.get("model"))
        steps_net = OptionScorer.from_json(json.loads(_inside(ROOT, job["operator_weights"]["path"]).read_text(encoding="utf-8")))
        steps_name = "8b-steps+operator:" + job["operator_weights"]["sha256"][:12]

        def make_steps(genome):
            source = LanguageStepSource(LanguageModelBroker(STEPS_ENDPOINT, timeout=180), cast=genome["cast"],
                                        early=genome["early"], branches=genome["branches"])
            return StepOperatorPolicy(steps_net, source, name=steps_name)
    if "operator" in spec["players"]:
        problems = _operator_problems({**job, "campaign": {"policy": "operator"}}, ROOT)
        if problems:
            raise GateRefused("; ".join(problems))
        from alpharush_rl.operator_net import OperatorPolicy, OptionScorer
        data = json.loads(_inside(ROOT, job["operator_weights"]["path"]).read_text(encoding="utf-8"))
        net = OptionScorer.from_json(data)
        solo = data.get("solo") is True  # a solo network ignores the plan's instructions
        name = ("solo:" if solo else "operator:") + job["operator_weights"]["sha256"][:12]
        make_operator = lambda genome: OperatorPolicy(net, None if solo else genome, name=name)
    ports = _job_ports(spec["workers"])
    for port in ports:
        _probe_port(port)
    engine_before = engine_identity()
    ctx.record("engine_identity", **engine_before)
    rng_mode, action_scope = job.get("rng_mode", ""), job.get("action_scope", DEFAULT_ACTION_SCOPE)
    tag = ctx.run_id.rsplit("-", 1)[-1][:12]

    def make(task, profile, index, player, port):
        from alpharush_rl.env import NativeEnv
        return NativeEnv(seed=task["seed"], level=task["level"], port=port, difficulty=2,
                         identity=f"eval_{tag}_{index:05d}_{player}", rng_mode=rng_mode, action_scope=action_scope,
                         profile=profile)
    summary = eval_job.eval_loop(spec, make, ctx, Journal(run_dir / EPISODES), pools=pools,
                                 protocol=episode_protocol(job), ports=ports, make_operator=make_operator,
                                 make_steps=make_steps)
    return _finish(config, permit, run_dir, job_kind, summary, engine_before, job)


def _collect_main(config, permit, ctx, job, pools, run_dir, job_kind):
    spec, problems = collect_job.check_collect(job.get("collect"), pools)
    if problems:
        raise GateRefused("; ".join(problems))
    ports = _job_ports(spec["workers"])
    for port in ports:
        _probe_port(port)
    engine_before = engine_identity()
    ctx.record("engine_identity", **engine_before)
    rng_mode, action_scope = job.get("rng_mode", ""), job.get("action_scope", DEFAULT_ACTION_SCOPE)
    net, net_name = None, "dagger"
    if spec.get("dagger") is not None:
        problems = _operator_problems({**job, "campaign": {"policy": "operator"}}, ROOT)
        if problems:
            raise GateRefused("; ".join(problems))
        from alpharush_rl.operator_net import OptionScorer
        net = OptionScorer.from_json(json.loads(_inside(ROOT, job["operator_weights"]["path"]).read_text(encoding="utf-8")))
        net_name = "dagger:" + job["operator_weights"]["sha256"][:12]
    summary = collect_job.collect_loop(spec, collect_env_factory(ctx.run_id, rng_mode, action_scope), ctx,
                                       Journal(run_dir / EPISODES), run_dir / "data", pools=pools,
                                       protocol=episode_protocol(job), ports=ports, net=net, net_name=net_name)
    return _finish(config, permit, run_dir, job_kind, summary, engine_before, job)


def engine_identity():
    """Which prepared engine actually ran: manifest digest plus its exe and host-copy records."""
    from alpharush_rl import engine
    engine.prepare()  # stat-only; refuses a changed or unprepared engine
    manifest_path = ROOT / "runtime/rl-engine/manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    return {"manifest_sha256": sha256_file(manifest_path), "recipe_sha256": manifest.get("recipe_sha256"),
            "exe_sha256": manifest["exe"]["sha256"],
            "host_copy_sha256": manifest["runtime"]["runtime/rl-engine/alpha_rl_host.lua"]["sha256"]}


def main(argv=None):
    parser = argparse.ArgumentParser(description="Campaign-v1 native survey (phase 0: check-only and freeze)")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check-only", action="store_true", help="print the read-only preflight (default)")
    mode.add_argument("--freeze", action="store_true", help="explicitly pin this phase's code and configs")
    mode.add_argument("--run", action="store_true", help="supervised native survey; refused unless enabled")
    mode.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--config", default=str(CONFIG), help="phase configuration (default: campaign-v1)")
    parser.add_argument("--run-dir", help=argparse.SUPPRESS)
    parser.add_argument("--job", default=JOB_KIND, choices=JOB_KINDS, help="job kind in the phase config")
    args = parser.parse_args(argv)
    if args.worker != bool(args.run_dir):
        parser.error("--run-dir belongs to the internal --worker mode only")
    if args.worker:
        _print(worker_main(args.config, args.run_dir, job_kind=args.job))
        return 0
    if args.freeze:
        try:
            result = phase.freeze_phase(args.config)
        except (GateRefused, JournalError, OSError, ValueError) as exc:
            _print({"ok": False, "refused": "freeze", "error": f"{type(exc).__name__}: {exc}"})
            return 2
        _print(result)
        return 0
    if args.run:
        try:
            result = supervise(args.config, args.job)
        except SurveyRefused as exc:
            _print(exc.result)
            return 2
        except GateRefused as exc:
            _print({"ok": False, "refused": "run", "error": f"{type(exc).__name__}: {exc}"})
            return 2
        _print(result)
        return 0
    result = check_only(args.config, args.job)
    _print(result)
    return 0 if result["ok"] else 2


if __name__ == "__main__":
    sys.exit(main())
