#!/usr/bin/env python3
"""Dedicated first-level 24B reward phase; default is CPU read-only inspection."""
import argparse
from datetime import datetime, timezone
import importlib.util
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from types import SimpleNamespace
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from alpharush_rl.bounded_llm_training import (Permit, full_objective_gate, independent_reference,
                                             supervisor_acceptance, terminate_owned, write_json)
from alpharush_rl.journal import Journal, canonical_json, sha256_data
from alpharush_rl.menus import build_menu, decision_prompt, validate_menu
from alpharush_rl.model_broker import GPU_MODELS
from alpharush_rl.ops import GateRefused, preflight, sha256_file

PLAN_PATH = ROOT / "configs/level1-24b-training.json"
OPS_PATH = ROOT / "configs/level1-24b-ops.json"
PHASE = ROOT / "runtime/rl/level1-24b-phase1"
STATE = PHASE / "state"
GLOBAL = ROOT / "runtime/rl"
SYSTEM = "You control Kingdom Rush. Choose exactly one option label from the legal menu. Optimize native level victory and remaining lives. Reply with the option label only."


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def existing_journal(path, expected_tip=None):
    path = Path(path).resolve()
    if not path.is_relative_to(ROOT.resolve()) or not path.is_file():
        raise GateRefused("Native journal is missing or leaves the workspace")
    journal = Journal(path)
    return journal.verify(expected_tip), journal.entries()


def bind_native_branches(data, directory):
    """Read the actual chained evidence, not only candidate verified flags."""
    doc = load(directory / "branches.json")
    group = data["groups"][0]
    fork = doc["fork_state"]
    projected = {key: value for key, value in fork.items() if key != "level_path_wave_counts"}
    if (sha256_data(fork) != group["native_fork_state_sha256"] or projected != group["state"]
        or doc["menu"] != group["menu"] or fork.get("level_idx") != 1 or fork.get("tick") != 361):
        raise GateRefused("Raw native fork, its exact policy projection, or complete menu differs")
    branches = doc["branches"]
    labels = validate_menu(group["menu"])
    if doc["scope"] != data["scope"] or len(branches) != 26 or len(labels) != 26 or set(b["label"] for b in branches) != set(labels):
        raise GateRefused("The phase requires all 26 distinct native legal branches")
    by_label = {branch["label"]: branch for branch in branches}
    options = {option["label"]: option for option in group["menu"]}
    for candidate in group["candidates"]:
        branch = by_label[candidate["label"]]
        expected_path = (directory / ("branch-" + candidate["label"] + ".jsonl")).relative_to(ROOT).as_posix()
        if candidate["journal_path"] != expected_path:
            raise GateRefused("Candidate journal must belong to this exact phase/label")
        actual, rows = existing_journal(ROOT / candidate["journal_path"], candidate["journal_tip_sha256"])
        if (not rows or rows[0]["kind"] != "fork" or rows[-1]["kind"] != "terminal"
            or any(row["kind"] != "native_trace" for row in rows[1:-1])
            or rows[0]["payload"]["state"] != fork or rows[0]["payload"]["menu"] != group["menu"]
            or [row["payload"] for row in rows[1:-1]] != branch["trace"]
            or actual != branch["journal"] or branch["journal_path"] != candidate["journal_path"]):
            raise GateRefused("Actual native journal chain/fork/trace differs from branch binding")
        terminal = rows[-1]["payload"]
        raw, outcome, final = branch["native_raw_outcome"], branch["outcome"], branch["final_state"]
        if (terminal != {"outcome": outcome, "native_raw_outcome": raw, "plan": branch["plan"], "replay_verified": True}
            or candidate["native_outcome"] != outcome or candidate["return"] != branch["return"]
            or outcome.get("native_raw") != raw or outcome.get("source") != "native"
            or (outcome.get("level"), outcome.get("difficulty"), outcome.get("seed"), outcome.get("initial_lives")) != (1, 2, 1002, 20)
            or any(outcome.get(key) != value for key, value in raw.items())
            or raw != {"source": "native", "terminal": True, "level_won": bool(final["level_won"]),
                       "level_lost": bool(final["level_lost"]), "lives": final["lives"],
                       "wave": final["wave"], "tick": final["tick"], "state_sha256": sha256_data(final)}):
            raise GateRefused("Candidate terminal reward is not bound to the raw native final state")
        receipt = branch["receipt"]
        if (branch["action"] != options[candidate["label"]]["action"] or receipt.get("action") != branch["action"]
            or receipt.get("accepted") is not True or receipt.get("executed") is not True
            or receipt.get("tick_before") != 361 or receipt.get("before_sha256") != sha256_data(fork)
            or {"kind": "action", "receipt": receipt} not in branch["trace"]
            or (branch["trace"][0].get("kind"), branch["trace"][0].get("seed"), branch["trace"][0].get("level")) != ("reset", 1002, 1)
            or branch.get("replay_verified") is not True):
            raise GateRefused("Branch action receipt is not bound to the native fork/action/trace")
    return {"branches": 26, "journal_chains_verified": True, "raw_policy_projection_verified": True}


def plan():
    cfg = load(PLAN_PATH)
    fixed = {"authorized": True, "model": "24b", "max_jobs": 1, "max_optimizer_steps": 1,
             "max_wall_seconds": 1200, "execution_seconds": 1190, "cleanup_reserve_seconds": 10,
             "backend_training_seconds": 900, "max_prompt_tokens": 6144, "cpu_threads": 4,
             "seed": 0, "native_seed": 1002, "learning_rate": 3e-7, "alignment_max_delta_p": 1e-4,
             "scoring_execution": "canonical_teacher_forced_no_cache_v2",
             "heldout_access": False, "auto_retry": False, "auto_learning_rate_search": False}
    if any(cfg.get(key) != value for key, value in fixed.items()) or cfg["gpu_uuid"] != GPU_MODELS["24b"]["gpu_uuid"]:
        raise GateRefused("Fixed first-phase model/authorization/step/budget/protocol changed")
    if cfg["owner_words"] != "[用户原话已省略 / user's message omitted]":
        raise GateRefused("Current recorded owner authorization differs")
    if cfg["adapter"] != {"rank": 16, "alpha": 32, "dropout": 0.0,
                           "modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
                           "text_only": True, "zero_initial_residual": True}:
        raise GateRefused("Fresh text-only zero-residual adapter preregistration changed")
    return cfg


def native_inputs(cfg):
    from alpharush_rl.pools import audit_dataset
    from alpharush_rl.trainer import group_advantages, score_native_outcome, validate_price_contract
    data = load(ROOT / cfg["dataset"])
    reward = load(ROOT / cfg["reward_contract"])
    if data.get("source") != "real_game" or data.get("scoring_contract") != reward or reward.get("name") != cfg["reward_name"]:
        raise GateRefused("New phase requires the versioned first-level real native reward contract")
    if data["scorer_sha256"] != sha256_data(reward) or data["price_sha256"] != sha256_data(data["price_contract"]):
        raise GateRefused("Native reward/price contract pins differ")
    hygiene = audit_dataset(data)
    if len(data["groups"]) != 1 or any(len(data["anchors"][role]) != 1 for role in ("A1", "A2")) or len(data["validation_forks"]) != 1:
        raise GateRefused("First phase requires one fork, A1/A2 and reward-free validation state")
    group = data["groups"][0]
    if group["level"] != 1 or group["seed"] != 1002 or group.get("pool") != "train":
        raise GateRefused("Training fork is outside the authorized first-level seed1002 pool")
    rows = [group, data["anchors"]["A1"][0], data["anchors"]["A2"][0], data["validation_forks"][0]]
    if any((row.get("level"), row.get("seed"), row.get("pool")) != (1, 1002, "train") for row in rows[:3]):
        raise GateRefused("Native first-level fork and both anchors must use the same authorized train seed")
    requests = []
    for row in rows:
        if "level_path_wave_counts" in row["state"]:
            raise GateRefused("Policy state contains forbidden future-wave count information")
        labels = validate_menu(row["menu"])
        if canonical_json(row["menu"]) != canonical_json(build_menu(row["state"])):
            raise GateRefused("Native complete legal menu differs from structured state")
        validate_price_contract(row["menu"], data["price_contract"])
        row.pop("reference", None)
        request = {"id": row.get("fork_id", row.get("id")), "system": SYSTEM,
                   "user": decision_prompt(row["state"], row["menu"]), "labels": labels}
        row["llm_request"] = request
        requests.append(request)
    labels = validate_menu(group["menu"])
    candidates = group["candidates"]
    if len(candidates) != len(labels) or set(c["label"] for c in candidates) != set(labels):
        raise GateRefused("Every legal label needs exactly one native candidate and cold replay")
    for candidate in candidates:
        if candidate.get("receipt_verified") is not True or candidate.get("replay_verified") is not True or "native_outcomes" in candidate:
            raise GateRefused("Every legal candidate requires one verified native outcome and cold replay")
        outcome = candidate["native_outcome"]
        if score_native_outcome(outcome, reward) != candidate["return"]:
            raise GateRefused("Recorded reward differs from the frozen native reward function")
    advantages = group_advantages(labels, candidates)
    if not any(advantages):
        raise GateRefused("All native candidates have the same reward; no learning signal")
    hygiene["native_branch_binding"] = bind_native_branches(data, (ROOT / cfg["dataset"]).parent)
    return data, requests, hygiene


def inspect(active=False):
    cfg = plan()
    issues = []
    check = preflight(OPS_PATH, "cpu-native-rollout", data_path=cfg["dataset"])
    issues.extend(check["issues"])
    try:
        collection_receipt = load(ROOT / cfg["collection_receipt"])
        if (collection_receipt.get("status") != "verified_all_legal_native_rewards"
            or collection_receipt.get("worker_exit_confirmed") is not True
            or collection_receipt.get("exit_code") != 0):
            issues.append("Native collection supervisor has not confirmed successful worker exit")
        else:
            cumulative = collection_receipt.get("cumulative_wall_seconds")
            if (isinstance(cumulative, bool) or not isinstance(cumulative, (float, int))
                or not math.isfinite(cumulative) or not 0 <= cumulative <= 300):
                issues.append("Collector cumulative wall budget including the original attempt exceeds 300s")
            evidence = load(ROOT / cfg["evidence"])
            _, collection_rows = existing_journal(PHASE / "collection-supervisor-r1-ledger.jsonl")
            collection_plan = load(ROOT / cfg["collection_plan"])
            original_receipt = load(PHASE / "collection-supervisor-receipt.json")
            if (collection_receipt.get("evidence_sha256") != sha256_file(ROOT / cfg["evidence"])
                or evidence.get("collection_plan_sha256") != sha256_file(ROOT / cfg["collection_plan"])
                or len(collection_rows) != 2 or collection_rows[0]["kind"] != "open"
                or collection_rows[-1]["kind"] != "close"
                or collection_rows[0]["payload"].get("plan_sha256") != sha256_file(ROOT / cfg["collection_plan"])
                or collection_rows[-1]["payload"] != collection_receipt):
                issues.append("Successful collector r1 receipt/ledger/evidence/plan binding differs")
            previous = collection_receipt.get("previous_wall_seconds")
            if (previous != original_receipt.get("wall_seconds") or previous != collection_plan.get("previous_wall_seconds")
                or cumulative < collection_receipt["wall_seconds"] + previous
                or cumulative - collection_receipt["wall_seconds"] - previous > 0.1
                or collection_receipt.get("optimizer_steps") != 0 or collection_receipt.get("heldout_consumed") is not False
                or collection_receipt.get("error") is not None
                or (collection_plan.get("level"), collection_plan.get("difficulty"), collection_plan.get("seed"),
                    collection_plan.get("all_legal_candidates"), collection_plan.get("cold_replay_each")) != (1, 2, 1002, True, True)):
                issues.append("Collector r1 scope/history/zero-update/cumulative-duration binding differs")
    except (OSError, ValueError, KeyError) as exc:
        issues.append("Native collection supervisor receipt pending: " + str(exc))
    inputs = None
    try:
        inputs = native_inputs(cfg)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        issues.append("Native reward dataset pending/ineligible: " + str(exc))
    if (STATE / "training-ledger.jsonl").exists() and not active:
        issues.append("The sole 24B phase job has already opened; no automatic retry")
    for state in (GLOBAL, STATE):
        if any((state / name).exists() for name in ("STOP", "ENGINEERING-STOP")):
            issues.append("A phase/global STOP is present")
    manifest = load(ROOT / "runtime/rl-models/model-24b-manifest.json")
    if (manifest["manifest_sha256"] != sha256_data(manifest["content"])
        or manifest["content"]["revision"] != GPU_MODELS["24b"]["revision"]
        or manifest["content"]["repo"] != GPU_MODELS["24b"]["repo"]):
        issues.append("24B complete base manifest identity differs")
    if any(name in sys.modules for name in ("torch", "peft", "transformers")):
        raise GateRefused("CPU inspection unexpectedly imported model runtime")
    return {"ok": not issues, "check_only": True, "issues": issues, "gpu_queried": False,
            "gpu_started": False, "optimizer_steps": 0, "plan": cfg, "native_preflight": check,
            "native_dataset_eligible": inputs is not None,
            "canonical_p0_pending_independent_gpu_generation": True,
            "phase": "level1-24b-phase1", "long_training_enabled": False}, inputs


def runtime_worker_module(deadline=None):
    spec = importlib.util.spec_from_file_location("phase24b_frozen_worker", ROOT / "tools/model-worker.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if deadline is not None:
        def bounded_query(*args, **kwargs):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise GateRefused("GPU query exceeded the absolute job execution deadline")
            kwargs["timeout"] = min(5.0, remaining)
            return subprocess.check_output(*args, **kwargs)
        module.subprocess = SimpleNamespace(check_output=bounded_query)
    return module


def source_bookend(cfg, pins_sha256):
    result = preflight(OPS_PATH, "cpu-native-rollout", data_path=cfg["dataset"])
    if not result["ok"] or result.get("pins_sha256") != pins_sha256:
        raise GateRefused("Final source/data/evidence pins changed: " + "; ".join(result["issues"]))
    return result


def worker(path):
    if os.name != "posix" or "ALPHARUSH_24B_INHERITED_LOCK_FD" not in os.environ:
        raise GateRefused("24B worker requires a real inherited supervisor permit")
    permit = load(path)
    ctx = Permit(permit["deadline"], permit["deadline"], permit["parent_pid"], permit["token"],
                 int(os.environ["ALPHARUSH_24B_INHERITED_LOCK_FD"]), GLOBAL, STATE)
    ctx.check()
    check, inputs = inspect(active=True)
    if (not check["ok"] or sha256_data(check["plan"]) != permit["plan_sha256"]
        or check["native_preflight"].get("pins_sha256") != permit["pins_file_sha256"]):
        raise GateRefused("24B worker source/authorization preflight changed")
    cfg = check["plan"]
    data, requests, hygiene = inputs
    if sha256_file(ROOT / cfg["dataset"]) != permit["dataset_file_sha256"]:
        raise GateRefused("Reward dataset changed after launch")
    from alpharush_rl.llm_training import LLMSmokeConfig, one_step_smoke, prepare_lora, text_attention_targets
    output = Path(permit["output_dir"])
    def progress(stage):
        ctx.check()
        write_json(output / "progress.json", {"stage": stage, "wall_seconds": time.monotonic()-permit["started_monotonic"],
            "optimizer_steps_cap": 1}, atomic=True)
    module = runtime_worker_module(permit["deadline"])
    progress("base_loading")
    scorer = module.Scorer("24b", cfg["max_prompt_tokens"])
    progress("base_loaded")
    torch = scorer.torch
    torch.set_num_threads(4)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.cuda.reset_peak_memory_stats()
    model = scorer.model
    base = [(value, str(value.dtype)) for value in model.parameters()]
    for parameter, _ in base:
        parameter.requires_grad_(False)
    targets = text_attention_targets(model)
    if any(not any(part in ("language_model", "lang_model") for part in name.split(".")) for name in targets):
        raise GateRefused("24B adapter targets must belong to the text decoder")
    model = prepare_lora(model, rank=16, alpha=32, target_modules=targets,
                         check_only=False, launch_context=ctx.stage(599))
    if any(str(value.dtype) != dtype or value.requires_grad for value, dtype in base):
        raise GateRefused("Base dtype or trainability changed")
    for model_cfg in (model.config, getattr(model.config, "text_config", None)):
        if model_cfg is not None:
            model_cfg.use_cache = False
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    refdir = output / "canonical-reference"
    refdir.mkdir()
    initial = {name: value.detach().cpu().clone() for name, value in model.named_parameters() if value.requires_grad}
    if not initial or any(torch.count_nonzero(value).item() for name, value in initial.items() if "lora_B" in name):
        raise GateRefused("24B fresh zero-residual initialization failed")
    torch.save(initial, refdir / "zero_adapter.pt")
    identity = {"model": scorer.model_id, "base_revision": scorer.spec["revision"],
                "quantization": "4bit_nf4_bf16", "tokenizer_files_sha256": sha256_data(scorer.identity_hashes),
                "identity_hashes": scorer.identity_hashes, "adapter_sha256": None,
                "equivalent_initial_adapter": "zero_init_lora", "scoring": "sum-label-and-eos-log-likelihood",
                "scoring_execution": cfg["scoring_execution"], "model_manifest_sha256": scorer.manifest["manifest_sha256"],
                "gpu_uuid": scorer.spec["gpu_uuid"]}
    progress("canonical_reference_started")
    responses = independent_reference(model, scorer.tokenizer, requests, identity, ctx, cfg["max_prompt_tokens"])
    rows = [data["groups"][0], data["anchors"]["A1"][0], data["anchors"]["A2"][0], data["validation_forks"][0]]
    for row, response in zip(rows, responses):
        row["llm_reference"] = response
    for name, value in (("responses.json", {"responses": responses}), ("dataset.json", data), ("identity.json", identity)):
        write_json(refdir / name, value)
    write_json(refdir / "FROZEN-REFERENCE.json", {"optimizer_steps_at_freeze": 0,
        "source_dataset_file_sha256": permit["dataset_file_sha256"], "identity_sha256": sha256_data(identity),
        "files": {name: sha256_file(refdir / name) for name in ("zero_adapter.pt", "responses.json", "dataset.json", "identity.json")}})
    progress("canonical_reference_completed")
    core_cfg = LLMSmokeConfig(learning_rate=cfg["learning_rate"], beta_fork=cfg["beta_fork"], beta_anchor=cfg["beta_anchor"],
        max_grad_norm=cfg["max_grad_norm"], alignment_max_delta_p=cfg["alignment_max_delta_p"],
        alignment_min_argmax_agreement=1.0, minimum_calibration_rows=4, max_groups=1, max_anchor_rows=1,
        max_validation_rows=1, max_prompt_tokens=cfg["max_prompt_tokens"], wall_cap_seconds=900,
        hard_fork_mean_kl=cfg["hard_validation_mean_kl"], hard_train_p90_kl=cfg["hard_train_p90_kl"],
        hard_anchor_mean_kl=cfg["hard_anchor_mean_kl"], entropy_reference_fraction=cfg["entropy_reference_fraction"])
    progress("one_step_backend_started")
    inner = one_step_smoke(model, scorer.tokenizer, data, identity, core_cfg, bounded_smoke=True,
                          launch_context=ctx.stage(899), outputdir=output / "backend")
    progress("one_step_backend_completed")
    gate = full_objective_gate(inner, cfg["beta_fork"], cfg["beta_anchor"])
    progress("outer_objective_gate_completed")
    accepted = (inner["status"] == "accepted_lora_one_step_smoke" and inner["optimizer_steps"] == 1
                and inner.get("parameter_update_verified") is True and gate["passed"])
    candidate = output / "backend/candidate_adapter"
    candidate_location = None
    if candidate.exists():
        destination = output / ("accepted_candidate_adapter" if accepted else "rejected_candidate_adapter")
        if not candidate.resolve().is_relative_to(output.resolve()) or not destination.resolve().is_relative_to(output.resolve()):
            raise GateRefused("Candidate isolation path leaves this phase run")
        candidate.rename(destination)
        candidate_location = str(destination.relative_to(ROOT))
    if not accepted:
        with torch.no_grad():
            for name, value in model.named_parameters():
                if name in initial:
                    value.copy_(initial[name])
                    value.grad = None
    restored = all(torch.equal(value.detach().cpu(), initial[name]) for name, value in model.named_parameters() if name in initial)
    module.verify_model_manifest("24b")
    final_pins = source_bookend(cfg, permit["pins_file_sha256"])
    ctx.check()
    report = {"kind": "bounded_level1_24b_real_reward_training", "accepted": accepted,
        "status": "accepted_reward_step" if accepted else "rejected_reward_step",
        "reason": inner["reason"] if not gate["passed"] and "before" not in gate else ("full_objective_nonincrease_and_backend_gates_passed" if accepted else "backend_or_full_objective_rejected"),
        "optimizer_steps": inner["optimizer_steps"], "parameter_update_verified": inner.get("parameter_update_verified"),
        "full_objective_gate": gate, "backend_status": inner["status"], "backend_reason": inner["reason"],
        "backend_report_file_sha256": sha256_file(output / "backend/LLM-SMOKE-REPORT.json"),
        "rejected_model_restored": not accepted and restored,
        "candidate_location": candidate_location, "candidate_accepted": accepted,
        "candidate_metadata_authority": "this outer reward report; backend smoke acceptance alone is insufficient",
        "requires_supervisor_acceptance": True,
        "auto_promoted": False, "heldout_accessed": False,
        "source_dataset_file_sha256": permit["dataset_file_sha256"], "scorer_sha256": data["scorer_sha256"],
        "source_bookend": final_pins, "native_branch_binding": hygiene["native_branch_binding"],
        "canonical_reference_manifest_sha256": sha256_file(refdir / "FROZEN-REFERENCE.json"),
        "gpu": scorer.preflight, "seed": 0, "base_dtype_preserved": True,
        "peak_memory_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
        "peak_memory_reserved_mib": torch.cuda.max_memory_reserved() / 2**20,
        "cold_load_seconds": scorer.load_seconds, "backend_metrics_before": inner.get("before"),
        "backend_metrics_after": inner.get("after"), "gameplay_improvement": "not established by one optimizer step"}
    write_json(output / "REWARD-TRAINING-REPORT.json", report)
    if candidate_location:
        write_json(ROOT / candidate_location / "REWARD-ACCEPTANCE.json", report)
    return 0 if accepted else 3


def run(check, inputs):
    if not check["ok"]:
        raise GateRefused("; ".join(check["issues"]))
    if os.name != "posix" or Path(sys.prefix).resolve() != Path("/home/<user>/alpharush/.venv").resolve():
        raise GateRefused("Training must use the independent AlphaRush WSL environment")
    import fcntl
    cfg = check["plan"]
    STATE.mkdir(parents=True, exist_ok=True)
    if (STATE / "training-ledger.jsonl").exists():
        raise GateRefused("The single new 24B training job is already spent")
    journal = None
    token, started = uuid.uuid4().hex, time.monotonic()
    hard_deadline = started + 1200
    lock_path = GLOBAL / "model-gpu.lock"
    marker_fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    linux_lock, process = None, None
    opened, report = False, None
    status, error, exit_code, steps = "failed", None, -1, "unknown"
    run_id = "level1-24b-" + token
    output = PHASE / "runs" / run_id
    try:
        def interrupt(_signum, _frame):
            raise KeyboardInterrupt("24B supervisor interrupted")
        signal.signal(signal.SIGTERM, interrupt)
        os.write(marker_fd, canonical_json({"pid": os.getpid(), "token": token, "run_id": run_id}).encode())
        os.fsync(marker_fd)
        linux_lock = (GLOBAL / "model-gpu-wsl.lock").open("a")
        fcntl.flock(linux_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        current, _ = inspect(active=True)
        if not current["ok"]:
            raise GateRefused("Source preflight changed after acquiring locks")
        module = runtime_worker_module(started + 1190)
        gpu = module.preflight(GPU_MODELS["24b"])
        if gpu["memory_used_mib"] > cfg["busy_memory_used_mib"] or gpu["utilization_percent"] > cfg["busy_utilization_percent"]:
            raise GateRefused("5090 busy; external work left untouched")
        output.mkdir(parents=True)
        permit = {"parent_pid": os.getpid(), "token": token, "deadline": started + 1190, "started_monotonic": started,
                  "hard_deadline": hard_deadline, "output_dir": str(output), "plan_sha256": sha256_data(cfg),
                  "dataset_file_sha256": sha256_file(ROOT / cfg["dataset"]),
                  "pins_file_sha256": sha256_file(PHASE / "pins.json")}
        write_json(output / "permit.json", permit)
        write_json(output / "FROZEN-LAUNCH.json", {"preflight": current, "gpu": gpu, "permit": permit})
        journal = Journal(STATE / "training-ledger.jsonl")
        journal.append("open", {"run_id": run_id, "owner_words": cfg["owner_words"], "plan": cfg,
            "gpu": gpu, "dataset_file_sha256": permit["dataset_file_sha256"],
            "pins_file_sha256": sha256_file(PHASE / "pins.json"), "step_cap": 1})
        opened = True
        env = os.environ.copy()
        env.update({key: "4" for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS")})
        env.update(PYTHONHASHSEED="0", TOKENIZERS_PARALLELISM="false", ALPHARUSH_24B_INHERITED_LOCK_FD=str(linux_lock.fileno()))
        with (output / "stdout.log").open("w") as stdout, (output / "stderr.log").open("w") as stderr:
            process = subprocess.Popen([sys.executable, "-B", str(Path(__file__).resolve()), "--worker", str(output / "permit.json")],
                env=env, stdout=stdout, stderr=stderr, start_new_session=True, pass_fds=(linux_lock.fileno(),))
            while process.poll() is None:
                if time.monotonic() >= permit["deadline"] or any((state / name).exists() for state in (GLOBAL, STATE) for name in ("STOP", "ENGINEERING-STOP")):
                    raise GateRefused("24B STOP or absolute1190 execution deadline reached")
                time.sleep(0.1)
        exit_code = process.returncode
        if (output / "REWARD-TRAINING-REPORT.json").exists():
            report = load(output / "REWARD-TRAINING-REPORT.json")
            steps = report["optimizer_steps"]
            status = report["status"]
            if isinstance(steps, bool) or steps not in (0, 1):
                raise GateRefused("Worker optimizer steps are outside the exact phase cap")
            if report.get("accepted") and exit_code != 0:
                status = "worker_failed_after_report"
                error = "Worker wrote a provisional accepted report but exited nonzero"
        else:
            raise GateRefused("Worker exited without a complete outer reward receipt; steps unknown")
    except BaseException as exc:
        error = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        exited, cleanup_error = terminate_owned(process, hard_deadline)
        if not exited:
            status, error, steps = "worker_exit_unconfirmed", cleanup_error, "unknown"
        source_result = None
        if opened:
            try:
                source_result = source_bookend(cfg, permit["pins_file_sha256"])
            except (OSError, ValueError, TypeError, KeyError, GateRefused) as exc:
                status, error = "source_pins_changed", str(exc)
        try:
            if linux_lock is not None:
                linux_lock.close()
            os.close(marker_fd)
            if exited and lock_path.exists() and load(lock_path).get("token") == token:
                lock_path.unlink()
        except (OSError, ValueError) as exc:
            status, error, cleanup_error = "cleanup_failed", str(exc), str(exc)
        if opened:
            elapsed = time.monotonic() - started
            if elapsed >= 1200:
                status = "budget_exceeded"
            summary = {"run_id": run_id, "status": status, "optimizer_steps": steps,
                "accepted": (status == "accepted_reward_step" and supervisor_acceptance(report, exit_code=exit_code,
                    exited=exited, cleanup_error=cleanup_error, error=error, wall_seconds=elapsed)),
                "parameter_update_verified": report.get("parameter_update_verified") if report else None,
                "reason": report.get("reason") if report else error, "error": error, "exit_code": exit_code,
                "wall_seconds": elapsed, "worker_exit_confirmed": exited, "cleanup_error": cleanup_error,
                "report_path": (output / "REWARD-TRAINING-REPORT.json").relative_to(ROOT).as_posix(),
                "candidate_path": report.get("candidate_location") if report else None,
                "full_objective_improved": report["full_objective_gate"].get("improved", False) if report else False,
                "full_objective_nonincreasing": report["full_objective_gate"].get("passed", False) if report else False,
                "source_bookend": source_result,
                "scope": "one real first-level 24B reward optimizer step", "auto_promoted": False}
            if status == "accepted_reward_step" and not summary["accepted"]:
                status = "supervisor_acceptance_rejected"
                summary.update(status=status, reason="Actual clean exit/step/objective/source/budget acceptance failed")
            journal.append("close", summary)
            write_json(output / "SUPERVISOR-RECEIPT.json", summary)
            write_json(PHASE / "latest-training.json", summary, atomic=True)
            if summary["candidate_path"]:
                write_json(ROOT / summary["candidate_path"] / "SUPERVISOR-ACCEPTANCE.json", summary)
            observed = time.monotonic() - started
            if observed >= 1200:
                status = "budget_exceeded"
                summary.update(status=status, accepted=False, reason="Evidence/cleanup exceeded total wall budget")
                journal.append("acceptance_revoked", {"run_id": run_id, "status": status, "accepted": False,
                    "wall_seconds_observed": observed, "reason": summary["reason"]})
                write_json(PHASE / "latest-training.json", summary, atomic=True)
                if summary["candidate_path"]:
                    write_json(ROOT / summary["candidate_path"] / "SUPERVISOR-ACCEPTANCE.json", summary, atomic=True)
            journal.append("audit_close_timing", {"run_id": run_id, "wall_seconds_observed": observed,
                "observation_boundary": "after-worker-lock-receipt-latest-before-final-audit-append", "cap_seconds": 1200})
            print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if status == "accepted_reward_step" else 3


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--check-only", action="store_true")
    mode.add_argument("--run", action="store_true")
    mode.add_argument("--worker", help=argparse.SUPPRESS)
    args = ap.parse_args()
    if args.worker:
        return worker(args.worker)
    check, inputs = inspect()
    if args.run:
        return run(check, inputs)
    print(json.dumps(check, ensure_ascii=False, indent=2))
    return 0 if check["ok"] else 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, KeyError, TypeError, GateRefused) as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(2)
