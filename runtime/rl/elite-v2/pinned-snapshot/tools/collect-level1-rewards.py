#!/usr/bin/env python3
"""A new, bounded all-legal native fork; never overwrite deployment evidence."""
from __future__ import annotations
import argparse
import copy
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from alpharush_rl.env import NativeEnv
from alpharush_rl.journal import Journal, sha256_data
from alpharush_rl.menus import build_menu, prompt_sha256
from alpharush_rl.ops import audit_records, sha256_file
from alpharush_rl.validation import establish_fork, finish, write_json

PHASE = ROOT / "runtime/rl/level1-24b-phase1"
DATA = PHASE / "data"
PLAN = PHASE / "collection-plan-r1.json"
SEED = 1002
LIMIT = 300
SUPERVISOR_LEDGER = PHASE / "collection-supervisor-r1-ledger.jsonl"
SUPERVISOR_RECEIPT = PHASE / "collection-supervisor-r1-receipt.json"
HISTORY = {
    "collection-plan.json": "a0a98e2bdc49865ac3ec77f59402f265ae1a561ea6bf138316d72b3476f493c2",
    "collection-supervisor-ledger.jsonl": "ba8dc58557856e9a7b3daec41fc761628b2011ef7e230d5b602ac80c0e63b496",
    "collection-supervisor-receipt.json": "c9265c37a93e9e08108b2c0d5f18c2528041dfe4590b51d74f3c92228aa52913",
    "collection-worker-stderr.log": "3c36a2d676912e3caa5e18d637decfb00a31bf920141bfa5734f7e2d5f06e411",
    "collection-worker-stdout.json": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
}


def historical_seconds():
    for name, expected in HISTORY.items():
        if sha256_file(PHASE / name) != expected:
            raise RuntimeError(f"Known pre-native attempt evidence changed: {name}")
    receipt = json.loads((PHASE / "collection-supervisor-receipt.json").read_text())
    if receipt["exit_code"] != 1 or receipt["worker_exit_confirmed"] is not True or receipt["optimizer_steps"] != 0:
        raise RuntimeError("Only the exact confirmed pre-native gate failure allows engineering r1")
    return receipt["wall_seconds"]


def policy_state(raw):
    """Do not disclose all future wave_db group counts to the option policy."""
    result = copy.deepcopy(raw)
    result.pop("level_path_wave_counts", None)
    return result


def inputs():
    return ["tools/collect-level1-rewards.py", "alpharush_rl/env.py", "alpharush_rl/engine.py",
            "alpharush_rl/validation.py", "alpharush_rl/menus.py", "alpharush_rl/journal.py", "alpharush_rl/windows_job.py",
            "alpharush_rl/reward_level1.py", "configs/scoring-level1-terminal-v2.json",
            "configs/pools.json", "configs/prices.json", "runtime/rl-engine/manifest.json",
            "runtime/rl/native-evidence.json", "runtime/rl/native-validation/dataset.json",
            "runtime/rl/native-validation/state-and-actions.json", "alpharush_rl/assets/host.lua",
            "alpharush_rl/assets/wrapper.lua", "Lumi_Nox/games/kingdom_rush/bridge.lua"]


def freeze():
    if PLAN.exists():
        raise RuntimeError("Collection already registered; do not overwrite its protocol")
    files = {p: sha256_file(ROOT / p) for p in inputs()}
    plan = {"schema_version": 1,
            "owner_words": "[用户原话已省略 / user's message omitted]",
            "seed": SEED, "level": 1, "difficulty": 2, "wall_seconds": LIMIT,
            "max_jobs": 1, "all_legal_candidates": True, "cold_replay_each": True,
            "continuation": "release clear waves; fixed 600 ticks; no further purchases",
            "heldout_consumed": False, "files": files,
            "engineering_revision": "r1: authenticate named Windows job across verified venv redirector ancestry",
            "preserved_pre_native_failure": HISTORY, "previous_wall_seconds": historical_seconds(),
            "execution_wall_seconds": LIMIT-historical_seconds()-10,
            "native_collection_jobs_cap": 1}
    write_json(PLAN, plan)
    return plan


def check(*, active_worker=False):
    plan = json.loads(PLAN.read_text(encoding="utf-8"))
    for p, sha in plan["files"].items():
        if sha256_file(ROOT / p) != sha:
            raise RuntimeError(f"Collection source SHA mismatch: {p}")
    for directory in (ROOT / "runtime/rl", PHASE, PHASE / "state"):
        for name in ("STOP", "ENGINEERING-STOP", "job.lock"):
            if (directory / name).exists():
                raise RuntimeError(f"Collection stopped/locked: {directory / name}")
    if (PHASE / "collection-ledger.jsonl").exists():
        raise RuntimeError("The sole collection job has already been spent")
    historical_seconds()
    if SUPERVISOR_LEDGER.exists() and not active_worker:
        raise RuntimeError("The sole supervised collection job has already been spent")
    pools = json.loads((ROOT / "configs/pools.json").read_text())
    if audit_records([{"seed": SEED, "level": 1}], pools):
        raise RuntimeError("Collection seed outside train pool")
    return {"ok": True, "check_only": True, "plan_sha256": sha256_file(PLAN), "plan": plan}


def collect():
    proof = check(active_worker=True)
    from alpharush_rl.reward_level1 import score_level1_outcome
    # Fail before a job is opened if the isolated native port is already occupied.
    probe = socket.socket()
    try:
        probe.bind(("127.0.0.1", 9879))
    finally:
        probe.close()
    PHASE.mkdir(parents=True, exist_ok=True)
    fd = os.open(PHASE / "collection.lock", os.O_WRONLY | os.O_CREAT | os.O_EXCL)
    os.write(fd, str(os.getpid()).encode()); os.close(fd)
    ledger = Journal(PHASE / "collection-ledger.jsonl")
    started = time.monotonic()
    ledger.append("open", {"pid": os.getpid(), "plan_sha256": proof["plan_sha256"], "optimizer_steps": 0})
    status, error = "failed", None
    try:
        DATA.mkdir(parents=True, exist_ok=False)
        def boundary():
            if time.monotonic() - started > LIMIT-historical_seconds():
                raise RuntimeError("Collection wall budget exhausted; no invented reward")
            for d in (ROOT / "runtime/rl", PHASE, PHASE / "state"):
                if any((d / n).exists() for n in ("STOP", "ENGINEERING-STOP")):
                    raise RuntimeError("Collection stopped")
        with NativeEnv(seed=SEED) as env:
            initial, middle, fork = establish_fork(env)
            menu = build_menu(fork)
        if initial["lives"] != 20 or initial["level_idx"] != 1 or fork["tick"] != 361:
            raise RuntimeError("Native scope differs from declared level-1 Normal fork")
        boundary()
        reward = json.loads((ROOT / "configs/scoring-level1-terminal-v2.json").read_text())
        branches = []
        for option in menu:
            boundary()
            with NativeEnv(seed=SEED) as env:
                _, _, current = establish_fork(env)
                if current != fork or build_menu(current) != menu:
                    raise RuntimeError("Native counterfactual prefix/menu differs")
                receipt = env.act(option["action"])
                raw = finish(env, boundary)
                enriched = {**raw, "level": 1, "seed": SEED, "difficulty": 2,
                            "initial_lives": initial["lives"], "native_raw": copy.deepcopy(raw)}
                branch = {"label": option["label"], "action": option["action"],
                          "receipt": receipt, "native_raw_outcome": raw, "outcome": enriched,
                          "return": score_level1_outcome(enriched, reward),
                          "final_state": copy.deepcopy(env.state), "plan": copy.deepcopy(env.plan),
                          "trace": copy.deepcopy(env.trace)}
            boundary()
            with NativeEnv(seed=SEED) as replay:
                for command in branch["plan"]:
                    boundary()
                    replay.act(command["action"]) if "action" in command else replay.advance(command["ticks"])
                if replay.trace != branch["trace"] or replay.terminal() != raw:
                    raise RuntimeError(f"Full cold replay differs: {option['label']}")
            boundary()
            path = DATA / f"branch-{option['label']}.jsonl"
            journal = Journal(path)
            journal.append("fork", {"state": fork, "menu": menu, "prompt_sha256": prompt_sha256(fork, menu)})
            for event in branch["trace"]:
                journal.append("native_trace", event)
            journal.append("terminal", {"outcome": enriched, "native_raw_outcome": raw,
                                        "plan": branch["plan"], "replay_verified": True})
            branch.update(replay_verified=True, journal=journal.verify(),
                          journal_path=path.relative_to(ROOT).as_posix())
            write_json(DATA / f"branch-{option['label']}.json", branch)
            branches.append(branch)
            write_json(PHASE / "collection-progress.json", {
                "completed": len(branches), "total": len(menu), "wall_seconds": time.monotonic()-started,
                "last": {"label": option["label"], "lives": raw["lives"], "win": raw["level_won"], "reward": branch["return"]}})
        if {b["label"] for b in branches} != {m["label"] for m in menu}:
            raise RuntimeError("Incomplete legal menu coverage; refuse training")
        base = json.loads((ROOT / "runtime/rl/native-validation/dataset.json").read_text())
        prices = json.loads((ROOT / "configs/prices.json").read_text())
        common = {"pool": "train", "data_kind": "native", "level": 1, "seed": SEED}
        group = {**common, "fork_id": f"kr1-level1-full-menu-{SEED}", "state": policy_state(fork), "menu": menu,
                 "native_fork_state_sha256": sha256_data(fork),
                 "scorer_sha256": sha256_data(reward), "price_sha256": sha256_data(prices),
                 "candidates": [{"label": b["label"], "return": b["return"], "native_outcome": b["outcome"],
                                 "receipt_verified": True, "replay_verified": True,
                                 "journal_tip_sha256": b["journal"]["tip_sha256"],
                                 "journal_path": b["journal_path"]} for b in branches]}
        dataset = {"schema_version": 1, "source": "real_game", "scope": {**base["scope"], "reward_version": reward["name"],
                   "policy_removed_fields": ["level_path_wave_counts"]},
                   "pool_registry": base["pool_registry"], "scoring_contract": reward, "scorer_sha256": sha256_data(reward),
                   "price_contract": prices, "price_sha256": sha256_data(prices), "groups": [group],
                   "anchors": {role: [{**common, "id": f"kr1-level1-{role}-{SEED}", "state": policy_state(s), "menu": build_menu(s)}]
                               for role, s in (("A1", initial), ("A2", middle))},
                   "validation_forks": [{**{k: v for k, v in r.items() if k != "reference"}, "state": policy_state(r["state"])}
                                        for r in base["validation_forks"]],
                   "heldout_consumed": False, "all_legal_candidates_verified": True}
        boundary()
        for p, sha in proof["plan"]["files"].items():
            if sha256_file(ROOT / p) != sha:
                raise RuntimeError(f"Collection source changed during native run: {p}")
        write_json(DATA / "dataset.json", dataset)
        write_json(DATA / "branches.json", {"scope": dataset["scope"], "fork_state": fork, "menu": menu, "branches": branches})
        old = json.loads((ROOT / "runtime/rl/native-evidence.json").read_text())
        gates = copy.deepcopy(old["gates"])
        for name in ("deterministic_replay", "branch_replay"):
            gates[name] = {"status": "verified", "scope": dataset["scope"],
                           "artifact_path": (DATA / "branches.json").relative_to(ROOT).as_posix(),
                           "artifact_sha256": sha256_file(DATA / "branches.json")}
        evidence = {"schema_version": 1, "scope": dataset["scope"], "gates": gates,
                    "dataset_path": (DATA / "dataset.json").relative_to(ROOT).as_posix(),
                    "dataset_sha256": sha256_file(DATA / "dataset.json"), "heldout_consumed": False,
                    "engine_manifest_sha256": sha256_file(ROOT / "runtime/rl-engine/manifest.json"),
                    "collection_plan_sha256": sha256_file(PLAN), "wall_seconds": time.monotonic()-started}
        write_json(DATA / "evidence.json", evidence)
        boundary()
        status = "verified_all_legal_native_rewards"
        return evidence
    except BaseException as exc:
        error = repr(exc)
        raise
    finally:
        ledger.append("close", {"status": status, "error": error, "wall_seconds": time.monotonic()-started,
                                "optimizer_steps": 0, "heldout_consumed": False})
        (PHASE / "collection.lock").unlink()


def supervise():
    """Hard deadline/STOP even during native RPC; close kills our descendants."""
    check()
    from alpharush_rl.windows_job import OwnedProcessJob
    started = time.monotonic()
    previous = historical_seconds()
    hard_deadline = started + LIMIT-previous
    ledger = Journal(SUPERVISOR_LEDGER)
    ledger.append("open", {"pid": os.getpid(), "plan_sha256": sha256_file(PLAN), "wall_cap_seconds": LIMIT,
                           "previous_wall_seconds": previous,
                           "execution_seconds": LIMIT-previous-10, "cleanup_reserve_seconds": 10})
    gate = PHASE / "collection-owned-job-ready-r1.json"
    process, job = None, None
    status, error, confirmed, exit_code = "failed", None, False, None
    try:
        job = OwnedProcessJob()
        with (PHASE / "collection-worker-r1-stdout.json").open("wb") as out, (PHASE / "collection-worker-r1-stderr.log").open("wb") as err:
            process = subprocess.Popen([sys.executable, str(Path(__file__)), "--worker"], stdout=out, stderr=err,
                                       cwd=ROOT, creationflags=subprocess.CREATE_NO_WINDOW)
            job.assign(process)
            # The child cannot start the game until it belongs to this owned job.
            write_json(gate, {"parent_pid": os.getpid(), "launcher_pid": process.pid,
                             "owned_job_name": job.name, "deadline": hard_deadline})
            while process.poll() is None:
                stopped = any((d / n).exists() for d in (ROOT / "runtime/rl", PHASE, PHASE / "state")
                              for n in ("STOP", "ENGINEERING-STOP"))
                if stopped or time.monotonic() >= hard_deadline - 10:
                    status = "stopped" if stopped else "wall_deadline"
                    job.terminate()
                    break
                time.sleep(0.2)
            process.wait(timeout=max(0.1, hard_deadline-time.monotonic()-2))
            confirmed, exit_code = True, process.returncode
            if status == "failed" and exit_code == 0 and (DATA / "evidence.json").exists():
                status = "verified_all_legal_native_rewards"
    except BaseException as exc:
        error = repr(exc)
        if job:
            job.terminate()
        if process and process.poll() is None:
            process.kill()
            process.wait(timeout=max(0.1, hard_deadline-time.monotonic()))
        confirmed = process is None or process.poll() is not None
        exit_code = None if process is None else process.returncode
        raise
    finally:
        if job:
            job.close()  # all our game descendants exit, even after child errors
        if confirmed and gate.exists():
            gate.unlink()
        if time.monotonic()-started+previous >= LIMIT:
            status = "budget_exceeded"
        receipt = {"status": status, "error": error, "worker_exit_confirmed": confirmed,
                   "exit_code": exit_code, "wall_seconds": time.monotonic()-started,
                   "previous_wall_seconds": previous, "cumulative_wall_seconds": time.monotonic()-started+previous,
                   "optimizer_steps": 0, "heldout_consumed": False}
        if status == "verified_all_legal_native_rewards":
            receipt["evidence_sha256"] = sha256_file(DATA / "evidence.json")
        ledger.append("close", receipt)
        write_json(SUPERVISOR_RECEIPT, receipt)
    if status != "verified_all_legal_native_rewards":
        raise RuntimeError(f"Native collection was not accepted: {status}, exit={exit_code}")
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze", action="store_true")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.freeze and args.run:
        parser.error("freeze and run must be separate operations")
    if args.worker:
        deadline = time.monotonic() + 10
        while not (PHASE / "collection-owned-job-ready-r1.json").exists():
            if time.monotonic() >= deadline:
                raise RuntimeError("No owned Windows job permit; no game was started")
            time.sleep(0.05)
        permit = json.loads((PHASE / "collection-owned-job-ready-r1.json").read_text())
        direct = permit["launcher_pid"] == os.getpid() and permit["parent_pid"] == os.getppid()
        redirector = permit["launcher_pid"] == os.getppid()
        if not (direct or redirector) or time.monotonic() >= permit["deadline"]:
            raise RuntimeError("Invalid live collection permit")
        from alpharush_rl.windows_job import confirm_current_job
        confirm_current_job(permit["owned_job_name"])
        result = collect()
    else:
        result = freeze() if args.freeze else supervise() if args.run else check()
    print(json.dumps(result, ensure_ascii=False, indent=2))
