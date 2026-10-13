#!/usr/bin/env python3
"""One preregistered 8B LoRA step, with a separate Linux supervisor.

Default is read-only check-only: no Torch/PEFT import, GPU query, model load,
lock, ledger append or optimizer. --prepare-data explicitly saves CPU-derived
inputs. --run is one WSL job, never a long training loop or model promotion.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from alpharush_rl.journal import canonical_bytes, sha256_data
from alpharush_rl.menus import decision_prompt
from alpharush_rl.model_broker import GPU_MODELS, tokenize_request, validate_distribution
from alpharush_rl.ops import GateRefused, preflight, sha256_file

PLAN = "configs/llm-smoke.json"
SEED_PLAN = "configs/llm-smoke-seed.json"
REVISION_PLAN = "configs/llm-smoke-r2.json"
SOURCE = "runtime/rl/native-validation/dataset.json"
REQUESTS = "runtime/rl-models/native-reference-batch.json"
BASELINE = "runtime/rl-models/native-reference-8b-baseline-enriched.json"
BINDING = "runtime/rl-models/native-reference-identity-binding.json"
MANIFEST = "runtime/rl-models/model-8b-manifest.json"
PREPARED = "runtime/rl/llm-smoke-prepared-r2"
REQUIRED_PINS = ("tools/train-llm-smoke.py", "alpharush_rl/llm_training.py",
                 "alpharush_rl/model_broker.py", "tools/model-worker.py",
                 PLAN, SEED_PLAN, REVISION_PLAN, SOURCE, REQUESTS, BASELINE, BINDING, MANIFEST)
EXPECTED_IDS = ("kr1-native-fork-1001", "kr1-native-anchor-A1-1001",
                "kr1-native-anchor-A2-1001", "kr2-validation-state-2001")


def read(relative):
    return json.loads((ROOT / relative).read_text(encoding="utf-8-sig"))


def emit(value):
    print(json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2), flush=True)


def write_new_or_equal(path, value):
    payload = canonical_bytes(value) + b"\n"
    if path.exists():
        if path.read_bytes() != payload:
            raise GateRefused(f"Prepared input differs; preserve it and inspect: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def append(path, value):
    with path.open("ab", buffering=0) as stream:
        stream.write(canonical_bytes(value) + b"\n")
        os.fsync(stream.fileno())


def revision_history(active_run_id=None):
    """Exactly one known, closed zero-update alignment failure permits r2."""
    revision = read(REVISION_PLAN)
    if (revision.get("revision_id") != "KR1-8B-TRAIN-MODE-SMOKE-R2-CANONICAL"
        or revision.get("maximum_jobs_total") != 2 or revision.get("maximum_optimizer_steps_total") != 1
        or revision.get("cumulative_wall_cap_seconds") != 600 or revision.get("revision_wall_cap_seconds") != 480
        or revision.get("revision_execution_seconds") != 470 or revision.get("cleanup_reserve_seconds") != 10
        or revision.get("scoring_execution") != "canonical_teacher_forced_no_cache_v2"
        or revision.get("learning_rate") != 3e-7 or revision.get("alignment_max_delta_p") != 1e-4
        or revision.get("formal_training") is not False or revision.get("heldout_access") is not False):
        raise GateRefused("Fixed r2 protocol/job/step/budget boundaries changed")
    ledger = ROOT / "runtime/rl/llm-smoke-ledger.jsonl"
    lines = ledger.read_bytes().splitlines(keepends=True)
    if len(lines) not in (2, 3) or (len(lines) == 3 and active_run_id is None):
        raise GateRefused("r2 permits one closed r1 plus only this supervised active r2; no retries")
    if hashlib.sha256(b"".join(lines[:2])).hexdigest() != revision["previous_ledger_prefix_sha256"]:
        raise GateRefused("Original r1 ledger prefix changed")
    rows = [json.loads(line) for line in lines]
    first, close = rows[:2]
    parent = revision["previous_run_id"]
    if (first.get("event") != "open" or close.get("event") != "close"
        or first.get("run_id") != parent or close.get("run_id") != parent
        or close.get("optimizer_steps") != 0 or close.get("status") != "rejected_one_step_smoke"
        or close.get("exit_code") != 3 or close.get("error") is not None
        or close.get("wall_seconds") != revision["previous_wall_seconds"]):
        raise GateRefused("r1 must have exact closed, known-zero scientific rejection; unknown/updated jobs are ineligible")
    if len(rows) == 3 and (rows[2].get("event") != "open" or rows[2].get("run_id") != active_run_id
                          or rows[2].get("revision_id") != revision["revision_id"]
                          or rows[2].get("previous_run_id") != parent):
        raise GateRefused("Active r2 ledger/permit lineage mismatch")
    for relative, expected in revision["previous_artifact_file_sha256"].items():
        path = (ROOT / relative).resolve()
        if not path.is_relative_to((ROOT / "runtime/rl/llm-smoke-runs" / parent).resolve()) or sha256_file(path) != expected:
            raise GateRefused("Preserved r1 artifact SHA mismatch")
    parent_dir = ROOT / "runtime/rl/llm-smoke-runs" / parent
    backend = json.loads((parent_dir / "training/LLM-SMOKE-REPORT.json").read_text())
    worker_report = json.loads((parent_dir / "worker-report.json").read_text())
    frozen = json.loads((parent_dir / "training/FROZEN-MANIFEST.json").read_text())
    receipt = json.loads((parent_dir / "training/UPDATE-RECEIPT.json").read_text())
    journal_rows = [json.loads(line) for line in (parent_dir / "training/updates.jsonl").read_text().splitlines()]
    previous = "0" * 64
    for sequence, entry in enumerate(journal_rows):
        unsigned = {key: value for key, value in entry.items() if key != "sha256"}
        if entry.get("seq") != sequence or entry.get("previous_sha256") != previous or sha256_data(unsigned) != entry["sha256"]:
            raise GateRefused("Preserved r1 journal hash chain is invalid")
        previous = entry["sha256"]
    # The fixed r1 receipt file was written after finish added its self-pin to
    # the shared payload object. The journal line predates that extra field.
    # Preserve both originals and admit only this exact pinned representation.
    receipt_without_self_pin = json.loads(json.dumps(receipt))
    self_pin = receipt_without_self_pin["payload"].pop("receipt_sha256", None)
    if self_pin != receipt["sha256"]:
        raise GateRefused("Unexpected r1 receipt self-pin representation")
    if (backend.get("reason") != "same_model_broker_alignment_failed" or backend.get("optimizer_steps") != 0
        or worker_report.get("optimizer_steps") != 0 or backend.get("candidate_saved") is not False
        or any(backend.get(name) is not True for name in ("reference_checkpoint_preserved", "broker_reference_preserved", "rejected_model_restored"))
        or worker_report["parameter_delta_diagnostic"]["changed_elements"] != 0
        or backend["alignment_calibration"]["passed"] is not False
        or backend["alignment_calibration"]["required_max_delta_p"] != 1e-4
        or any(row.get("kind") == "one_optimizer_update" for row in journal_rows)
        or receipt_without_self_pin != journal_rows[-1] or receipt["sha256"] != backend["receipt_sha256"]
        or sha256_file(parent_dir / "training/reference_adapter.pt") != frozen["reference_adapter_file_sha256"]):
        raise GateRefused("r1 zero-update/checkpoint/alignment rejection proof is incomplete")
    if close["wall_seconds"] + revision["revision_wall_cap_seconds"] > 600:
        raise GateRefused("Insufficient cumulative 600-second budget for r2")
    return revision, close


def preregistered_config(plan):
    from alpharush_rl.llm_training import LLMSmokeConfig
    adapter = plan.get("adapter", {})
    if (plan.get("plan_id") != "KR1-8B-TRAIN-MODE-SMOKE-ONE-STEP-V1"
        or plan.get("kind") != "bounded_training_mode_validation"
        or plan.get("model") != "8b" or plan.get("gpu_uuid") != GPU_MODELS["8b"]["gpu_uuid"]
        or plan.get("max_steps") != 1 or plan.get("wall_seconds") != 600
        or plan.get("max_prompt_tokens") != 6144 or plan.get("learning_rate") != 3e-7
        or plan.get("step0_max_probability_delta") != 1e-4
        or plan.get("max_grad_norm") != 1.0 or plan.get("beta_fork") != 0.03 or plan.get("beta_anchor") != 0.03
        or plan.get("hard_validation_mean_kl") != 0.05 or plan.get("hard_train_p90_kl") != 0.1
        or plan.get("hard_anchor_mean_kl") != 0.05 or plan.get("entropy_reference_fraction") != 0.5
        or plan.get("quantization") != "4bit_nf4_bf16" or plan.get("formal_training") is not False
        or plan.get("heldout_access") is not False or plan.get("optimizer") != "AdamW"
        or plan.get("gradient_protocol") != "two_pass_exact_score_derivative_per_legal_label_and_EOS"
        or plan.get("gradient_checkpointing") is not True or plan.get("use_cache_during_training") is not False
        or adapter != {"new_kr1_adapter": True, "rank": 16, "alpha": 32, "dropout": 0.0,
                       "modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
                       "text_decoder_only": True, "initialize_as_zero_residual": True}):
        raise GateRefused("One-step preregistration changed; no automatic tolerance/LR revision")
    config = LLMSmokeConfig(learning_rate=plan["learning_rate"], beta_fork=plan["beta_fork"],
        beta_anchor=plan["beta_anchor"], max_grad_norm=plan["max_grad_norm"],
        alignment_max_delta_p=plan["step0_max_probability_delta"], alignment_min_argmax_agreement=1.0,
        minimum_calibration_rows=4, hard_fork_mean_kl=plan["hard_validation_mean_kl"],
        hard_train_p90_kl=plan["hard_train_p90_kl"], hard_anchor_mean_kl=plan["hard_anchor_mean_kl"],
        entropy_reference_fraction=plan["entropy_reference_fraction"], max_prompt_tokens=6144,
        max_groups=1, max_anchor_rows=1, max_validation_rows=1, wall_cap_seconds=480,
        checkpoint_options=True)
    config.validate(updating=True)
    return config


def derive_inputs():
    plan, data, requests, baseline, binding, manifest = (read(name) for name in
        (PLAN, SOURCE, REQUESTS, BASELINE, BINDING, MANIFEST))
    config = preregistered_config(plan)
    seed_plan = read(SEED_PLAN)
    if (seed_plan.get("seed") != 0 or seed_plan.get("python_hash_seed") != 0
        or seed_plan.get("torch_manual_seed") != 0 or seed_plan.get("cuda_manual_seed_all") != 0
        or seed_plan.get("formal_training") is not False):
        raise GateRefused("Fixed one-step seed control differs from preregistration")
    if [item["id"] for item in requests] != list(EXPECTED_IDS):
        raise GateRefused("Expected exactly four original native references")
    if (sha256_file(ROOT / BASELINE) != binding["enriched_file_sha256"]
        or sha256_file(ROOT / REQUESTS) != binding["requests_file_sha256"]
        or binding.get("probabilities_changed") is not False
        or binding.get("performed_after_gpu_inference") is not True):
        raise GateRefused("Reference identity enrichment/source binding differs")
    content = manifest["content"]
    if sha256_data(content) != manifest["manifest_sha256"] or binding["model_manifest_sha256"] != manifest["manifest_sha256"]:
        raise GateRefused("Complete base-model manifest identity mismatch")
    if content["repo"] != GPU_MODELS["8b"]["repo"] or content["revision"] != GPU_MODELS["8b"]["revision"]:
        raise GateRefused("8B base/revision mismatch")
    if len(baseline["responses"]) != 4 or baseline["learning_updates"] != 0:
        raise GateRefused("Complete no-update four-reference baseline required")
    if len(data["groups"]) != 1 or any(len(data["anchors"][role]) != 1 for role in ("A1", "A2")) or len(data["validation_forks"]) != 1:
        raise GateRefused("One fork, A1, A2 and reward-free validation state required")
    rows = [data["groups"][0], data["anchors"]["A1"][0], data["anchors"]["A2"][0], data["validation_forks"][0]]
    identities = []
    for row, request, response in zip(rows, requests, baseline["responses"]):
        if row.get("fork_id", row.get("id")) != request["id"] or request["user"] != decision_prompt(row["state"], row["menu"]):
            raise GateRefused("Original native structured state and request bytes differ")
        validate_distribution(response, request)
        if response["prompt_tokens"] > 6144 or response.get("adapter") is not None or response.get("learning_updates") != 0:
            raise GateRefused("Baseline token cap/adapter/update identity mismatch")
        if response["model_manifest_sha256"] != manifest["manifest_sha256"] or response["tokenized_prompt_ids_sha256"] != binding["request_prompt_token_sha256"][request["id"]]:
            raise GateRefused("Baseline prompt-token/model manifest pins mismatch")
        for name, digest in response["identity_hashes"].items():
            if content["files"][name]["sha256"] != digest:
                raise GateRefused("Baseline identity file differs from complete model audit")
        if not {"tekken.json", "tokenizer_config.json", "chat_template.jinja"} <= set(response["identity_hashes"]):
            raise GateRefused("Real tokenizer and chat template pins required")
        identities.append(sha256_data(response["identity_hashes"]))
        row.pop("reference", None)  # Remove Tiny p0 rather than mixing backends.
        row["llm_request"], row["llm_reference"] = request, response
    if len(set(identities)) != 1:
        raise GateRefused("Four baseline references use different tokenizer identities")
    spec = GPU_MODELS["8b"]
    identity = {"model": f"{spec['repo']}@{spec['revision']}:nf4-bf16:instruction-base:no-adapter",
                "base_revision": spec["revision"], "quantization": "4bit_nf4_bf16",
                "tokenizer_files_sha256": identities[0], "adapter_sha256": None,
                "equivalent_initial_adapter": "zero_init_lora",
                "scoring": "sum-label-and-eos-log-likelihood",
                "model_manifest_sha256": manifest["manifest_sha256"]}
    source_hashes = {name: sha256_file(ROOT / name) for name in (PLAN, SEED_PLAN, REVISION_PLAN, SOURCE, REQUESTS, BASELINE, BINDING, MANIFEST)}
    lineage = {"schema_version": 1, "purpose": "one-step-train-mode-validation",
               "original_dataset_file_sha256": source_hashes[SOURCE], "source_file_sha256": source_hashes,
               "original_files_modified": False, "tiny_references_removed": 4,
               "llm_reference_rows": 4, "source_reference_identity_enriched_after_inference": True,
               "seed_control": seed_plan,
               "derived_dataset_canonical_sha256": sha256_data(data), "identity_sha256": sha256_data(identity),
               "heldout_accessed": False, "formal_training": False}
    return plan, config, data, identity, lineage


def cpu_inspect(active_run_id=None):
    from alpharush_rl.llm_training import check_only
    plan, config, data, identity, lineage = derive_inputs()
    deployment = read("configs/rl-deployment.json")
    check = preflight(ROOT / "configs/rl-deployment.json", "cpu-native-rollout", data_path=SOURCE)
    issues = list(check["issues"])
    revision, previous_close = revision_history(active_run_id)
    if plan.get("owner_words") != deployment.get("deployment_authorization", {}).get("cpu_smoke_selection_owner_words") or not plan.get("owner_words"):
        issues.append("Recorded owner authorization differs from the bounded validation plan")
    if revision.get("owner_words") != plan["owner_words"]:
        issues.append("Engineering revision authorization differs from the bounded owner words")
    if deployment.get("formal_training_enabled") is not False:
        issues.append("Formal training must remain disabled")
    pins = read("runtime/rl/pins.json")["files"]
    for name in REQUIRED_PINS:
        if name not in pins or pins[name] != sha256_file(ROOT / name):
            issues.append(f"Final one-step pin missing or changed: {name}")
    mathematical = check_only(data, identity, config)
    if not mathematical["native_eligibility"]:
        issues.append("Differentiable backend native eligibility failed: " + mathematical.get("reason", "unknown"))
    report = {"check_only": True, "ok": not issues, "issues": issues,
              "gpu_queried": False, "gpu_started": False, "optimizer_steps": 0,
              "formal_training_started": False, "scope": plan["scope"], "owner_words": plan["owner_words"],
              "preregistration": plan, "engineering_revision": revision,
              "previous_known_optimizer_steps": 0, "cumulative_wall_seconds_before_r2": previous_close["wall_seconds"],
              "canonical_p0_pending_new_gpu_reference": True, "cpu_native_preflight": check,
              "llm_native_check": mathematical, "lineage": lineage, "identity": identity}
    if any(name in sys.modules for name in ("torch", "peft", "transformers")):
        raise GateRefused("Check-only unexpectedly imported a model runtime")
    return report, (plan, config, data, identity, lineage)


@dataclass
class LivePermit:
    deadline: float
    state_dir: Path
    parent_pid: int
    lock_fd: int
    token: str
    kind: str = "llm-one-step-smoke"
    gpu_authorization_verified: bool = True
    gpu_lock_held: bool = True
    max_optimizer_steps: int = 1

    def check(self):
        if time.monotonic() >= self.deadline or os.getppid() != self.parent_pid:
            raise GateRefused("One-step deadline/parent permit expired")
        if any((self.state_dir / name).exists() for name in ("STOP", "ENGINEERING-STOP")):
            raise GateRefused("STOP present during one-step validation")
        descriptor = os.fstat(self.lock_fd)
        source = (self.state_dir / "model-gpu-wsl.lock").stat()
        if (descriptor.st_dev, descriptor.st_ino) != (source.st_dev, source.st_ino):
            raise GateRefused("Inherited descriptor does not identify the shared Linux GPU lock")
        lock = json.loads((self.state_dir / "model-gpu.lock").read_text())
        if lock.get("token") != self.token or lock.get("pid") != self.parent_pid:
            raise GateRefused("Supervisor lock identity changed")


def independent_canonical_references(model, tokenizer, requests, identity, context):
    """Independent frozen-base p0 implementation; no training scorer is called.

    For a prefix of L tokens and a suffix of m label+EOS tokens, feed L+m-1
    tokens. Causal logits at L-1 .. L+m-2 predict all m suffix targets.
    Every option gets a fresh full sequence; there is no shared KV cache.
    """
    import torch
    if not callable(getattr(model, "disable_adapter", None)):
        raise GateRefused("Independent reference requires an explicit disabled-adapter context")
    responses = []
    device = model.get_input_embeddings().weight.device
    model.eval()
    with model.disable_adapter(), torch.no_grad():
        for request in requests:
            context.check()
            started = time.monotonic()
            encoded = tokenize_request(tokenizer, request)
            prefix, suffixes = encoded["input_ids"], encoded["suffixes"]
            if not prefix or len(prefix) > 6144:
                raise GateRefused("Independent reference token cap exceeded; no truncation")
            scores = []
            for suffix in suffixes:
                context.check()
                sequence = torch.tensor([prefix + suffix[:-1]], dtype=torch.long, device=device)
                output = model(input_ids=sequence, use_cache=False, logits_to_keep=len(suffix))
                logits = output.logits[0]
                if logits.shape[0] == sequence.shape[1]:
                    logits = logits[len(prefix) - 1:len(prefix) - 1 + len(suffix)]
                elif logits.shape[0] != len(suffix):
                    raise GateRefused("Independent reference cannot establish full causal suffix shift")
                targets = torch.tensor(suffix, dtype=torch.long, device=device)
                log_probs = logits.float().log_softmax(-1)
                scores.append(float(log_probs.gather(1, targets[:, None]).sum().cpu()))
                del output, sequence, logits, targets, log_probs
            logp = torch.tensor(scores, dtype=torch.float64, device=device).log_softmax(0)
            probabilities = logp.exp().tolist()
            index = max(range(len(scores)), key=scores.__getitem__)
            response = {"id": request["id"], "labels": request["labels"],
                "p": probabilities, "logp": logp.tolist(), "choice": request["labels"][index],
                "prompt_sha256": encoded["prompt_sha256"], "prompt_tokens": len(prefix),
                "label_token_ids": encoded["label_token_ids"],
                "tokenized_prompt_ids_sha256": encoded["tokenized_prompt_ids_sha256"],
                "prompt_token_ids_sha256": sha256_data(prefix),
                "prompt_token_ids": prefix,
                "model": identity["model"], "model_revision": identity["base_revision"],
                "identity_hashes": identity["identity_hashes"], "adapter": None,
                "model_manifest_sha256": identity["model_manifest_sha256"],
                "gpu_uuid": GPU_MODELS["8b"]["gpu_uuid"],
                "scoring": "sum-label-and-eos-log-likelihood",
                "scoring_execution": "canonical_teacher_forced_no_cache_v2",
                "sequence_log_likelihood": scores, "complete_legal_distribution": True,
                "learning_updates": 0, "seconds": time.monotonic() - started,
                "reference_source": "independent_frozen_base_disable_adapter_full_teacher_forcing",
                "reference_scorer_file_sha256": sha256_file(Path(__file__)),
                "use_cache": False, "causal_suffix_shift": "prefix_length_minus_one"}
            response["normalization"] = "float64-log-softmax-on-model-device"
            responses.append(validate_distribution(response, request))
    return responses


def terminate_owned_worker(process, hard_deadline):
    """Bounded own-session cleanup; races and uncertain exit remain explicit."""
    if process is None or process.poll() is not None:
        return True, None
    try:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            process.wait(timeout=min(5, max(0.01, hard_deadline - time.monotonic() - 2)))
            return True, None
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=max(0.01, hard_deadline - time.monotonic() - 2))
                return True, None
            except subprocess.TimeoutExpired:
                return False, "Own worker exit unconfirmed before absolute cleanup deadline"
    except OSError as exc:
        return process.poll() is not None, f"Own-session cleanup: {type(exc).__name__}: {exc}"


def worker(permit_path):
    if os.name != "posix" or not os.environ.get("ALPHARUSH_ONE_STEP_INHERITED_LOCK_FD"):
        raise GateRefused("Worker requires a live Linux supervisor/inherited GPU lock")
    permit = json.loads(Path(permit_path).read_text())
    ctx = LivePermit(deadline=permit["deadline"], state_dir=ROOT / "runtime/rl",
                     parent_pid=permit["parent_pid"], lock_fd=int(os.environ["ALPHARUSH_ONE_STEP_INHERITED_LOCK_FD"]), token=permit["token"])
    ctx.check()
    check, (_, config, data, identity, lineage) = cpu_inspect(active_run_id=permit["run_id"])
    if not check["ok"] or sha256_data(lineage) != permit["lineage_sha256"]:
        raise GateRefused("Worker CPU preflight/source lineage changed")
    # Runtime imports occur only beyond authorization, pins and live permit.
    spec = importlib.util.spec_from_file_location("alpharush_frozen_inference", ROOT / "tools/model-worker.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    from alpharush_rl.llm_training import one_step_smoke, prepare_lora, text_attention_targets
    scorer = module.Scorer("8b", 6144)  # UUID, busy check, NF4/BF16 and full manifest checks.
    torch = scorer.torch
    torch.set_num_threads(4)
    torch.cuda.reset_peak_memory_stats()
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    ctx.check()
    model = scorer.model
    base_parameters = [(name, parameter, str(parameter.dtype)) for name, parameter in model.named_parameters()]
    for _, parameter, _ in base_parameters:
        parameter.requires_grad_(False)
    targets = text_attention_targets(model)
    if any(not any(part in ("language_model", "lang_model") for part in name.split("."))
           or any("vision" in part or "visual" in part for part in name.split(".")) for name in targets):
        raise GateRefused("Actual LoRA target names must belong solely to the text decoder")
    model = prepare_lora(model, rank=16, alpha=32, target_modules=targets,
                         check_only=False, launch_context=ctx)
    if any(str(parameter.dtype) != dtype or parameter.requires_grad for _, parameter, dtype in base_parameters):
        raise GateRefused("Base parameter dtype/trainability changed during adapter preparation")
    for cfg in (model.config, getattr(model.config, "text_config", None)):
        if cfg is not None:
            cfg.use_cache = False
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    outputdir = Path(permit["output_dir"])
    zero_adapter = {name: value.detach().cpu().clone() for name, value in model.named_parameters() if value.requires_grad}
    if not zero_adapter or any(torch.count_nonzero(value).item() for name, value in zero_adapter.items() if "lora_B" in name):
        raise GateRefused("Fresh reference requires exactly zero residual LoRA")
    reference_dir = outputdir / "canonical-reference"
    reference_dir.mkdir()
    torch.save(zero_adapter, reference_dir / "zero_adapter.pt")
    del zero_adapter
    identity = dict(identity, scoring_execution="canonical_teacher_forced_no_cache_v2",
                    identity_hashes=scorer.identity_hashes)
    rows = [data["groups"][0], data["anchors"]["A1"][0], data["anchors"]["A2"][0], data["validation_forks"][0]]
    responses = independent_canonical_references(model, scorer.tokenizer,
                                                [row["llm_request"] for row in rows], identity, ctx)
    for row, response in zip(rows, responses):
        row["llm_reference"] = response
    for name, value in (("responses.json", {"responses": responses, "learning_updates": 0}),
                        ("dataset.json", data), ("identity.json", identity)):
        write_new_or_equal(reference_dir / name, value)
    canonical_manifest = {"scoring_execution": identity["scoring_execution"],
        "optimizer_steps_at_reference_freeze": 0, "adapter_disabled_for_reference": True,
        "source_cached_baseline_file_sha256": sha256_file(ROOT / BASELINE),
        "cached_baseline_preserved": True, "old_p0_reused": False,
        "dataset_canonical_sha256": sha256_data(data), "identity_canonical_sha256": sha256_data(identity),
        "file_sha256": {name: sha256_file(reference_dir / name) for name in
                        ("responses.json", "dataset.json", "identity.json", "zero_adapter.pt")},
        "seed_control": read(SEED_PLAN), "revision": read(REVISION_PLAN)}
    write_new_or_equal(reference_dir / "FROZEN-CANONICAL-REFERENCE.json", canonical_manifest)
    ctx.check()
    report = one_step_smoke(model, scorer.tokenizer, data, identity, config,
                           bounded_smoke=True, launch_context=ctx, outputdir=outputdir / "training")
    report["canonical_reference"] = {"directory": str(reference_dir),
        "frozen_manifest_file_sha256": sha256_file(reference_dir / "FROZEN-CANONICAL-REFERENCE.json"),
        "training_identity_sha256": sha256_data(identity), "training_dataset_sha256": sha256_data(data),
        "scoring_execution": identity["scoring_execution"], "old_cached_p0_reused": False}
    reference_parameters = torch.load(outputdir / "training/reference_adapter.pt", map_location="cpu", weights_only=True)
    square_delta, maximum_delta, changed_parameters = 0.0, 0.0, 0
    for name, parameter in model.named_parameters():
        if name in reference_parameters:
            difference = parameter.detach().cpu().double() - reference_parameters[name].double()
            square_delta += float(difference.square().sum())
            maximum_delta = max(maximum_delta, float(difference.abs().max()))
            changed_parameters += int(torch.count_nonzero(difference))
    report["parameter_delta_diagnostic"] = {
        "scope": "returned_adapter_vs_preserved_reference_after_any_rejection_rollback",
        "l2": square_delta ** 0.5, "max_abs": maximum_delta, "changed_elements": changed_parameters,
        "actual_step_hash_changed_before_rollback": report.get("parameter_update_verified"),
        "post_step_parameter_sha256": report.get("post_update_parameter_sha256"),
        "rejected_model_restored": report.get("rejected_model_restored")}
    del reference_parameters
    module.verify_model_manifest("8b")
    report["launcher_runtime"] = {"gpu": scorer.preflight, "cold_load_seconds": scorer.load_seconds,
        "seed_control": read(SEED_PLAN),
        "cpu_threads": 4, "base_dtype_preserved": True, "base_trainable_parameters": 0,
        "peak_memory_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
        "peak_memory_reserved_mib": torch.cuda.max_memory_reserved() / 2**20,
        "memory_measurement_scope": "after-base-load:LoRA-canonical-p0-calibration-gradient-post-update",
        "text_attention_targets": targets, "kbit_upcast_preparation_used": False,
        "source_model_files_modified": False, "model_manifest_sha256": scorer.manifest["manifest_sha256"]}
    write_new_or_equal(outputdir / "worker-report.json", report)
    emit(report)
    return 0 if report["status"] == "accepted_lora_one_step_smoke" else 3


def supervise(check, inputs):
    if os.name != "posix" or not Path("/proc/version").exists() or "microsoft" not in Path("/proc/version").read_text().lower():
        raise GateRefused("Explicit --run must use the isolated AlphaRush WSL Python")
    if not check["ok"]:
        raise GateRefused("; ".join(check["issues"]))
    if Path(sys.prefix).resolve() != Path("/home/<user>/alpharush/.venv").resolve():
        raise GateRefused("Use the independent /home/<user>/alpharush/.venv runtime")
    import fcntl
    plan, config, data, identity, lineage = inputs
    state = ROOT / "runtime/rl"
    ledger = state / "llm-smoke-ledger.jsonl"
    revision, previous_close = revision_history()
    token = uuid.uuid4().hex
    lock_path = state / "model-gpu.lock"
    windows_fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    linux_lock = None
    process = None
    opened = False
    started = time.monotonic()
    run_id = "llm-one-step-" + token
    outputdir = state / "llm-smoke-runs" / run_id
    status, error, exit_code, steps = "failed", None, -1, "unknown"
    report = None
    hard_deadline = started + revision["revision_wall_cap_seconds"]
    common = {"run_id": run_id, "kind": "llm-one-step-smoke", "owner_words": plan["owner_words"],
              "scope": plan["scope"], "gpu_uuid": plan["gpu_uuid"], "max_optimizer_steps": 1,
              "wall_cap_seconds": 480, "execution_wall_cap_seconds": 470, "cleanup_reserve_seconds": 10,
              "cumulative_wall_cap_seconds": 600, "previous_wall_seconds": previous_close["wall_seconds"],
              "revision_id": revision["revision_id"], "previous_run_id": revision["previous_run_id"],
              "cpu_threads": 4, "max_prompt_tokens": 6144,
              "formal_training_started": False, "heldout_accessed": False,
              "pins_file_sha256": sha256_file(state / "pins.json"),
              "evidence_file_sha256": sha256_file(state / "native-evidence.json"), **lineage}
    try:
        def supervisor_signal(_signum, _frame):
            raise KeyboardInterrupt("One-step supervisor was interrupted")
        signal.signal(signal.SIGTERM, supervisor_signal)
        os.write(windows_fd, canonical_bytes({"token": token, "pid": os.getpid(), "run_id": run_id}))
        os.fsync(windows_fd)
        linux_lock = (state / "model-gpu-wsl.lock").open("a")
        fcntl.flock(linux_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        current, current_inputs = cpu_inspect()
        if not current["ok"] or sha256_data(current_inputs[-1]) != sha256_data(lineage):
            raise GateRefused("CPU gates/source lineage changed after acquiring locks")
        gpu_module_spec = importlib.util.spec_from_file_location("alpharush_preflight_only", ROOT / "tools/model-worker.py")
        gpu_module = importlib.util.module_from_spec(gpu_module_spec)
        gpu_module_spec.loader.exec_module(gpu_module)
        gpu = gpu_module.preflight(GPU_MODELS["8b"])
        if gpu["memory_used_mib"] > 4096 or gpu["utilization_percent"] > 10:
            raise GateRefused("Assigned 5080 is busy; external work remains untouched")
        outputdir.mkdir(parents=True, exist_ok=False)
        for name, value in (("dataset.json", data), ("identity.json", identity),
                            ("lineage.json", lineage), ("preflight.json", current),
                            ("frozen-plan.json", {"one_step": plan, "seed_control": read(SEED_PLAN), "revision": revision})):
            write_new_or_equal(outputdir / "inputs" / name, value)
        permit = {"parent_pid": os.getpid(), "token": token, "run_id": run_id,
                  "deadline": started + revision["revision_execution_seconds"], "hard_deadline": hard_deadline,
                  "lineage_sha256": sha256_data(lineage), "output_dir": str(outputdir)}
        permit_path = outputdir / "launch-permit.json"
        write_new_or_equal(permit_path, permit)
        append(ledger, {"event": "open", "time": datetime.now(timezone.utc).isoformat(), "gpu": gpu, **common})
        opened = True
        environment = os.environ.copy()
        environment.update({name: "4" for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS")})
        environment["TOKENIZERS_PARALLELISM"] = "false"
        environment["PYTHONHASHSEED"] = "0"
        environment["ALPHARUSH_ONE_STEP_INHERITED_LOCK_FD"] = str(linux_lock.fileno())
        with (outputdir / "stdout.log").open("w") as stdout, (outputdir / "stderr.log").open("w") as stderr:
            process = subprocess.Popen([sys.executable, "-B", str(Path(__file__).resolve()), "--worker", str(permit_path)],
                env=environment, stdout=stdout, stderr=stderr, start_new_session=True, pass_fds=(linux_lock.fileno(),))
            while process.poll() is None:
                if time.monotonic() >= permit["deadline"] or any((state / name).exists() for name in ("STOP", "ENGINEERING-STOP")):
                    raise GateRefused("One-step STOP or absolute 470-second execution deadline reached")
                time.sleep(0.1)
        exit_code = process.returncode
        report_path = outputdir / "worker-report.json"
        if report_path.is_file():
            report = json.loads(report_path.read_text())
            steps = report["optimizer_steps"]
            if steps not in (0, 1):
                raise GateRefused("Unexpected optimizer step count")
            status = "accepted_one_step_smoke" if exit_code == 0 and report["status"] == "accepted_lora_one_step_smoke" else "rejected_one_step_smoke"
        else:
            raise GateRefused(f"Worker failed without final training receipt, exit {exit_code}")
    except BaseException as exc:
        error = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        confirmed_exit, cleanup_error = terminate_owned_worker(process, hard_deadline)
        if not confirmed_exit:
            status, error, steps = "worker_exit_unconfirmed", cleanup_error, "unknown"
        try:
            if linux_lock is not None:
                linux_lock.close()
            os.close(windows_fd)
            if confirmed_exit and lock_path.exists() and json.loads(lock_path.read_text()).get("token") == token:
                lock_path.unlink()
        except (OSError, ValueError) as exc:
            cleanup_error = f"Lock cleanup: {type(exc).__name__}: {exc}"
            status, error = "cleanup_failed", cleanup_error
        if opened:
            wall_seconds = time.monotonic() - started
            cumulative_wall_seconds = previous_close["wall_seconds"] + wall_seconds
            if wall_seconds >= 480 or cumulative_wall_seconds >= 600:
                status, error = "budget_exceeded", "Actual cumulative or local wall budget exceeded"
            receipt = {"event": "close", "time": datetime.now(timezone.utc).isoformat(),
                       "status": status, "error": error, "exit_code": exit_code,
                       "optimizer_steps": steps, "wall_seconds": wall_seconds,
                       "cumulative_wall_seconds": cumulative_wall_seconds,
                       "cumulative_optimizer_steps": steps if isinstance(steps, int) else "unknown",
                       "worker_exit_confirmed": confirmed_exit, "cleanup_error": cleanup_error,
                       "lock_marker_preserved": lock_path.exists(),
                       "output_dir": str(outputdir), **common}
            append(ledger, receipt)
            write_new_or_equal(outputdir / "supervisor-receipt.json", receipt)
            latest = {"schema_version": 1, "run_id": run_id, "status": status,
                "optimizer_steps": steps, "parameter_update_verified": report.get("parameter_update_verified") if report else None,
                "accepted": status == "accepted_one_step_smoke", "report_path": (outputdir / "worker-report.json").relative_to(ROOT).as_posix(),
                "reason": report.get("reason") if report else error,
                "revision_id": revision["revision_id"], "cumulative_wall_seconds": cumulative_wall_seconds}
            latest_path = state / "latest-llm-training.json"
            temporary = latest_path.with_name(latest_path.name + "." + token + ".tmp")
            write_new_or_equal(temporary, latest)
            os.replace(temporary, latest_path)
            final_elapsed = time.monotonic() - started
            if final_elapsed >= 480 or previous_close["wall_seconds"] + final_elapsed >= 600:
                status = "budget_exceeded"
                latest.update(status=status, accepted=False, reason="Final evidence/cleanup wall budget exceeded")
                temporary = latest_path.with_name(latest_path.name + ".audit." + token + ".tmp")
                write_new_or_equal(temporary, latest)
                os.replace(temporary, latest_path)
            append(ledger, {"event": "audit_close_timing", "run_id": run_id,
                "total_wall_seconds_observed": final_elapsed,
                "cumulative_wall_seconds_observed": previous_close["wall_seconds"] + final_elapsed,
                "observation_boundary": "after-worker-lock-cleanup-close-receipt-and-latest-before-final-audit-append",
                "status": status, "worker_exit_confirmed": confirmed_exit,
                "optimizer_steps": steps, "local_cap_seconds": 480, "cumulative_cap_seconds": 600})
            emit(receipt)
    return 0 if status == "accepted_one_step_smoke" else 3


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check-only", action="store_true")
    mode.add_argument("--prepare-data", action="store_true")
    mode.add_argument("--run", action="store_true")
    mode.add_argument("--worker", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        return worker(args.worker)
    check, inputs = cpu_inspect()
    if args.prepare_data:
        _, _, data, identity, lineage = inputs
        destination = ROOT / PREPARED
        for name, value in (("dataset.json", data), ("identity.json", identity), ("lineage.json", lineage)):
            write_new_or_equal(destination / name, value)
        emit({"prepared": True, "directory": str(destination), "gpu_started": False, "optimizer_steps": 0, "check": check})
        return 0
    if args.run:
        return supervise(check, inputs)
    emit(check)
    return 0 if check["ok"] else 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (GateRefused, OSError, ValueError, KeyError, TypeError) as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(2)
