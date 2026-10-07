"""Demonstration collection for the operator network (the ``native-collect`` job body).

Each task plays one plan on one train seed with the plan executor wrapped in a
``Recorder``; the decisions (feature arrays and the executor's choice, with the
plan step as the operator's instruction) are saved as one npz per episode next
to the journal, which records the episode result and the npz digest. Only
train seeds are accepted: evaluation seeds never feed a gradient.
"""

import concurrent.futures as futures
import hashlib
import json
import queue
import time
from pathlib import Path

from . import phase
from .campaign import level_profile
from .episode import EpisodeProtocol, run_episode
from .ops import GateRefused
from .operator_net import Recorder, save_rows
from .search import BuildOrderPolicy, check_genome, genome_id

COLLECT_KEYS = {"tasks", "workers", "profile_stars_per_level"}
# Optional: {"beta": 0..1} plays the operator network (job operator_weights) with expert labels (DAgger).
OPTIONAL_COLLECT = {"dagger"}


def check_collect(spec, pools):
    if not isinstance(spec, dict) or not COLLECT_KEYS <= set(spec) <= COLLECT_KEYS | OPTIONAL_COLLECT:
        return None, [f"collect must be an object with the keys {sorted(COLLECT_KEYS)} (optionally dagger)"]
    issues = []
    dagger = spec.get("dagger")
    if dagger is not None and (not isinstance(dagger, dict) or set(dagger) != {"beta"}
                               or isinstance(dagger["beta"], bool) or not isinstance(dagger["beta"], (int, float))
                               or not 0 <= dagger["beta"] <= 1):
        issues.append("dagger must be {beta: 0..1}")
    if isinstance(spec["workers"], bool) or not isinstance(spec["workers"], int) or not 1 <= spec["workers"] <= 32:
        issues.append("workers must be 1..32")
    rate = spec["profile_stars_per_level"]
    if isinstance(rate, bool) or not isinstance(rate, (int, float)) or not 1 <= rate <= 3:
        issues.append("profile_stars_per_level must be a number from 1 to 3 (an average star rate)")
    tasks = spec["tasks"]
    if not isinstance(tasks, list) or not tasks:
        return None, issues + ["tasks must be a nonempty list"]
    seen = set()
    for index, task in enumerate(tasks):
        if not isinstance(task, dict) or set(task) != {"level", "seed", "genome"}:
            issues.append(f"tasks[{index}] must have exactly level, seed and genome")
            continue
        if phase.seed_role(pools, task["seed"]) != "train":
            issues.append(f"tasks[{index}]: seed {task['seed']!r} is not a train seed")
        try:
            key = (task["level"], task["seed"], genome_id(check_genome(task["genome"])))
        except ValueError as exc:
            issues.append(f"tasks[{index}]: {exc}")
            continue
        if key in seen:
            issues.append(f"tasks[{index}] repeats an earlier task")
        seen.add(key)
    return (None if issues else dict(spec)), issues


def collect_loop(spec, env_factory, ctx, out, data_dir, *, pools, protocol, ports, difficulty=2, net=None,
                 net_name="dagger"):
    spec, issues = check_collect(spec, pools)
    if issues:
        raise GateRefused("; ".join(issues))
    if not isinstance(protocol, EpisodeProtocol):
        raise TypeError("protocol must be an EpisodeProtocol")
    if spec.get("dagger") is not None and net is None:
        raise GateRefused("a DAgger collection needs the operator network")
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    free = queue.Queue()
    for port in ports[:spec["workers"]]:
        free.put(port)
    ctx.record("collect_start", tasks=len(spec["tasks"]), workers=spec["workers"])

    def play(index, task):
        port = free.get()
        try:
            genome = check_genome(task["genome"])
            profile = level_profile(task["level"], spec["profile_stars_per_level"], hero=genome["hero"],
                                    package=genome.get("pkg", "balanced"))
            env = env_factory(task, profile, index, port)
            try:
                if spec.get("dagger") is not None:
                    from .operator_net import DaggerRecorder
                    recorder = DaggerRecorder(net, genome, beta=spec["dagger"]["beta"], seed=index, name=net_name)
                else:
                    recorder = Recorder(BuildOrderPolicy(genome))
                result = run_episode(env, recorder, protocol, seed=task["seed"], level=task["level"],
                                     difficulty=difficulty, episode_id=f"{ctx.run_id}-{index:05d}", check=ctx.check)
            finally:
                env.close()
            path = data_dir / f"L{task['level']:02d}_s{task['seed']}_{genome_id(genome)}.npz"
            outcome = result.get("outcome") or {}
            path.with_suffix(".macro.json").write_text(json.dumps(recorder.macro, separators=(",", ":")),
                                                       encoding="utf-8")
            save_rows(path, recorder.rows, {"level": task["level"], "seed": task["seed"],
                                            "genome_id": genome_id(genome), "won": bool(outcome.get("level_won")),
                                            "lives": outcome.get("lives"), "status": result["status"],
                                            "labels": "expert", "player": "network" if spec.get("dagger") is not None
                                            else "expert",
                                            "agreement": (recorder.agreements / max(1, len(recorder.rows)))
                                            if spec.get("dagger") is not None else 1.0})
            return result, path
        finally:
            free.put(port)

    executor = futures.ThreadPoolExecutor(max_workers=spec["workers"])
    inflight, errors, done, stopped_reason = {}, [], 0, None
    tasks = list(enumerate(spec["tasks"]))
    try:
        while tasks or inflight:
            while tasks and len(inflight) < spec["workers"]:
                if ctx.games_remaining < 1:
                    stopped_reason = "max_games"
                    tasks = []
                    break
                index, task = tasks.pop(0)
                ctx.claim_game()
                ctx.record("game_claimed", index=index, level=task["level"], games_played=ctx.games_played)
                inflight[executor.submit(play, index, task)] = (index, task)
            if not inflight:
                break
            finished, _ = futures.wait(list(inflight), return_when=futures.FIRST_COMPLETED)
            for future in finished:
                index, task = inflight.pop(future)
                base = {"run_id": ctx.run_id, "index": index, "level": task["level"], "seed": task["seed"],
                        "genome_id": genome_id(task["genome"])}
                try:
                    result, path = future.result()
                except GateRefused:
                    raise
                except Exception as exc:
                    errors.append(index)
                    out.append("episode_error", {**base, "error": f"{type(exc).__name__}: {exc}"})
                    continue
                done += 1
                out.append("collect_episode", {**base, "data": path.name,
                                               "data_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                                               "result": {k: v for k, v in result.items() if k != "decisions"}})
    finally:
        executor.shutdown(wait=True, cancel_futures=True)
    journal = out.verify()
    summary = {"schema": "alpharush-collect-v1", "run_id": ctx.run_id, "episodes": done, "episode_errors": errors,
               "games_played": ctx.games_played, "stopped_reason": stopped_reason, "data_dir": data_dir.name,
               "episodes_journal": journal, "optimizer_steps": 0}
    ctx.record("collect_end", episodes=done, errors=len(errors))
    return summary
