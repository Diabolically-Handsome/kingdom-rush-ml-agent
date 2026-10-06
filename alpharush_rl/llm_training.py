"""Differentiable full-label LoRA RL backend; importing never loads Torch/GPU.

Models/tokenizers are supplied by an explicitly authorized external launcher.
Default entry points are check-only. The only update entry point is one bounded
optimizer step; long GPU training is deliberately not exposed here.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import time

from .journal import Journal, canonical_bytes, canonical_json, sha256_data
from .menus import build_menu, decision_prompt, validate_menu
from .model_broker import prompt_hash, tokenize_request, validate_distribution, validate_request
from .pools import audit_dataset
from .trainer import EligibilityError, group_advantages, score_native_outcome, validate_price_contract

SCORING_PROTOCOL = "sum-label-and-eos-log-likelihood"
CANONICAL_EXECUTION = "canonical_teacher_forced_no_cache_v2"


def _torch():
    import torch
    return torch


def differentiable_label_scores(model, tokenized: dict, *, max_prompt_tokens: int = 16384,
                                 checkpoint_options: bool = False, boundary_check=None):
    """Sum causal log probabilities for every label token AND its terminal EOS.

    Each legal suffix is independently teacher-forced. No first-token shortcut,
    length normalization, top-N truncation, or non-differentiable .item() occurs.
    logits_to_keep limits vocabulary-sized prompt activations on modern HF
    models; a full-logits model is also accepted (including the CPU test model).
    """
    torch = _torch()
    prefix, suffixes = tokenized["input_ids"], tokenized["suffixes"]
    if not isinstance(prefix, list) or not prefix or len(prefix) > max_prompt_tokens:
        raise ValueError("nonempty prompt within explicit token budget required")
    if not isinstance(suffixes, list) or not suffixes or any(not isinstance(suffix, list) or len(suffix) < 2 for suffix in suffixes):
        raise ValueError("every legal label requires its complete token sequence plus EOS")
    if len({tuple(suffix) for suffix in suffixes}) != len(suffixes):
        raise ValueError("different legal labels tokenized to the same complete suffix")
    try:
        device = model.get_input_embeddings().weight.device
    except (AttributeError, TypeError):
        device = next(model.parameters()).device
    scores = []
    for suffix in suffixes:
        if boundary_check is not None:
            boundary_check()
        input_ids = torch.tensor([prefix + suffix[:-1]], dtype=torch.long, device=device)
        targets = torch.tensor(suffix, dtype=torch.long, device=device)

        def score_suffix(ids, target_ids):
            count = target_ids.shape[0]
            output = model(input_ids=ids, use_cache=False, logits_to_keep=count)
            logits = output.logits[0]
            if logits.shape[0] == ids.shape[1]:
                logits = logits[len(prefix) - 1:len(prefix) - 1 + count]
            elif logits.shape[0] != count:
                raise ValueError("causal model did not return the requested complete suffix logits")
            if logits.dtype in (torch.float16, torch.bfloat16):
                logits = logits.float()
            return logits.log_softmax(-1).gather(1, target_ids[:, None]).sum()

        if checkpoint_options:
            from torch.utils.checkpoint import checkpoint
            scores.append(checkpoint(score_suffix, input_ids, targets, use_reentrant=False))
        else:
            scores.append(score_suffix(input_ids, targets))
    return torch.stack(scores)


def differentiable_distribution(model, tokenizer, request: dict, **score_options):
    tokens = tokenize_request(tokenizer, request)
    scores = differentiable_label_scores(model, tokens, **score_options)
    # Match the broker's double precision normalization of float token scores.
    logp = scores.double().log_softmax(dim=0)
    return {"p": logp.exp(), "logp": logp, "scores": scores, "tokenized": tokens}


def full_legal_pi_adv_loss(logp, advantages, reference_p, beta: float):
    """-sum_legal pi(a)*adv(a) + beta*KL(pi || frozen pi0)."""
    torch = _torch()
    if not math.isfinite(beta) or beta < 0:
        raise ValueError("finite nonnegative KL beta required")
    if logp.ndim != 1 or not torch.isfinite(logp).all():
        raise ValueError("finite full-legal log probabilities required")
    p = logp.exp()
    advantage = torch.as_tensor(advantages, dtype=p.dtype, device=p.device).detach()
    # Detach AND clone: no optimizer or caller mutation can update pi0 in place.
    p0 = torch.as_tensor(reference_p, dtype=p.dtype, device=p.device).detach().clone()
    if p0.shape != p.shape or advantage.shape != p.shape or not torch.isfinite(advantage).all() or not torch.isfinite(p0).all() or torch.any(p0 <= 0):
        raise ValueError("matching complete advantages and positive reference distribution required")
    if not torch.allclose(p.sum(), p.new_tensor(1.0), atol=1e-8, rtol=0) or not torch.allclose(p0.sum(), p0.new_tensor(1.0), atol=1e-8, rtol=0):
        raise ValueError("full legal distributions must be normalized")
    expected_advantage = (p * advantage).sum()
    kl = (p * (logp - p0.log())).sum()
    entropy = -(p * logp).sum()
    return -expected_advantage + beta * kl, {
        "expected_advantage": expected_advantage, "kl": kl, "entropy": entropy,
        "reference_entropy": -(p0 * p0.log()).sum(), "p0": p0}


def full_legal_anchor_loss(logp, reference_p, beta: float):
    torch = _torch()
    return full_legal_pi_adv_loss(logp, torch.zeros_like(logp), reference_p, beta)


@dataclass
class LLMSmokeConfig:
    learning_rate: float | None = None  # Must be selected for the actual model.
    beta_fork: float = 0.03
    beta_anchor: float = 0.03
    beta_factor: float = 1.5
    beta_min: float = 1e-5
    beta_max: float = 10.0
    target_fork_kl: float = 0.01
    target_anchor_kl: float = 0.005
    hard_fork_mean_kl: float = 0.05
    hard_train_p90_kl: float = 0.10
    hard_anchor_mean_kl: float = 0.05
    entropy_reference_fraction: float = 0.5
    max_grad_norm: float = 1.0
    alignment_max_delta_p: float = 1e-4
    alignment_min_argmax_agreement: float = 0.98
    minimum_calibration_rows: int = 4
    minimum_expected_advantage_delta: float = 1e-8
    max_prompt_tokens: int = 16384
    max_groups: int = 2
    max_anchor_rows: int = 4
    max_validation_rows: int = 4
    wall_cap_seconds: float = 600.0
    checkpoint_options: bool = True
    recompute_score_tolerance: float = 1e-5

    def validate(self, updating: bool = False):
        if updating and self.learning_rate is None:
            raise EligibilityError("actual LLM smoke learning rate must be explicitly preregistered")
        if self.learning_rate is not None and (not math.isfinite(self.learning_rate) or self.learning_rate <= 0):
            raise ValueError("finite positive learning rate required")
        for name in ("beta_fork", "beta_anchor", "beta_min", "beta_max", "target_fork_kl", "target_anchor_kl",
                     "hard_fork_mean_kl", "hard_train_p90_kl", "hard_anchor_mean_kl", "max_grad_norm",
                     "wall_cap_seconds", "alignment_max_delta_p"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"finite positive {name} required")
        if not math.isfinite(self.recompute_score_tolerance) or self.recompute_score_tolerance < 0:
            raise ValueError("finite nonnegative recomputation tolerance required")
        if not self.beta_min <= min(self.beta_fork, self.beta_anchor) <= max(self.beta_fork, self.beta_anchor) <= self.beta_max:
            raise ValueError("KL betas must lie within declared positive bounds")
        if self.beta_factor <= 1 or not 0.5 <= self.entropy_reference_fraction <= 1 or not 0 <= self.alignment_min_argmax_agreement <= 1:
            raise ValueError("invalid adaptive beta, entropy floor or alignment agreement")
        if not 0 < self.wall_cap_seconds <= 900:
            raise ValueError("bounded smoke wall cap must be at most 900 seconds")
        for name in ("max_prompt_tokens", "max_groups", "max_anchor_rows", "max_validation_rows", "minimum_calibration_rows"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"positive integer {name} required")
        if not math.isfinite(self.minimum_expected_advantage_delta) or self.minimum_expected_advantage_delta < 0:
            raise ValueError("invalid learncheck delta")


def _identity(identity: dict):
    if not isinstance(identity, dict):
        raise EligibilityError("frozen LLM identity required")
    for key in ("model", "base_revision", "quantization", "tokenizer_files_sha256"):
        if not isinstance(identity.get(key), str) or not identity[key]:
            raise EligibilityError(f"frozen LLM {key} required")
    if identity["model"].startswith("cpu-tiny"):
        raise EligibilityError("Tiny CPU reference cannot stand in for a language-model reference")
    if "adapter_sha256" not in identity or identity.get("scoring") != SCORING_PROTOCOL:
        raise EligibilityError("exact base/adapter and label-plus-EOS scoring identity required")
    for key in ("tokenizer_files_sha256", "adapter_sha256"):
        pin = identity[key]
        if pin is not None and (len(pin) != 64 or any(c not in "0123456789abcdef" for c in pin)):
            raise EligibilityError(f"invalid {key} SHA256")
    if identity.get("equivalent_initial_adapter") not in (None, "zero_init_lora"):
        raise EligibilityError("unsupported initial adapter equivalence claim")
    if identity.get("equivalent_initial_adapter") == "zero_init_lora" and identity["adapter_sha256"] is not None:
        raise EligibilityError("zero-init equivalence applies only to an unadapted broker reference")
    if identity.get("scoring_execution") not in (None, CANONICAL_EXECUTION):
        raise EligibilityError("unsupported scoring execution protocol")
    if "identity_hashes" in identity and sha256_data(identity["identity_hashes"]) != identity["tokenizer_files_sha256"]:
        raise EligibilityError("identity file hashes differ from frozen tokenizer/base pin")
    return sha256_data(identity)


def _llm_row(row: dict, identity: dict, role: str, advantages=None):
    state, menu = row.get("state"), row.get("menu")
    if not isinstance(state, dict):
        raise EligibilityError("structured native state required for LLM row")
    labels = validate_menu(menu)
    wait = next((option["action"].get("ticks") for option in menu if option["action"]["action"] == "wait"), 30)
    if canonical_json(menu) != canonical_json(build_menu(state, wait)):
        raise EligibilityError("LLM row does not contain the complete supported native menu")
    request = row.get("llm_request")
    if not isinstance(request, dict):
        raise EligibilityError("exact LLM broker request required for every train/anchor/validation row")
    validate_request(request)
    if request["labels"] != labels or request["user"] != decision_prompt(state, menu):
        raise EligibilityError("LLM request labels/native prompt bytes mismatch")
    broker = row.get("llm_reference")
    if not isinstance(broker, dict):
        raise EligibilityError("same-model broker LLM reference required; Tiny references are ineligible")
    validate_distribution(broker, request)
    if broker["model"] != identity["model"] or broker.get("model_revision") != identity["base_revision"]:
        raise EligibilityError("broker reference uses a different base/revision")
    if broker.get("adapter") != identity["adapter_sha256"] or broker.get("scoring") != SCORING_PROTOCOL:
        raise EligibilityError("broker reference adapter or sequence scoring mismatch")
    if broker.get("scoring_execution") != identity.get("scoring_execution"):
        raise EligibilityError("reference and trainer scoring execution protocols differ")
    if sha256_data(broker.get("identity_hashes")) != identity["tokenizer_files_sha256"]:
        raise EligibilityError("broker tokenizer/base identity files differ from frozen identity")
    if row.get("reference", {}).get("model", "").startswith("cpu-tiny"):
        raise EligibilityError("remove Tiny reference from LLM dataset; cross-backend substitution is forbidden")
    return {"request": request, "reference": broker, "role": role, "advantages": advantages}


def validate_llm_dataset(data: dict, identity: dict, config: LLMSmokeConfig):
    """Native outcome eligibility is shared with the CPU path, never its pi0."""
    _identity(identity)
    if data.get("source") != "real_game":
        raise EligibilityError("LLM RL requires real native continuation evidence")
    hygiene = audit_dataset(data)
    contract, prices = data.get("scoring_contract"), data.get("price_contract")
    if not isinstance(contract, dict) or data.get("scorer_sha256") != sha256_data(contract):
        raise EligibilityError("frozen native scoring contract required")
    if not isinstance(prices, dict) or data.get("price_sha256") != sha256_data(prices):
        raise EligibilityError("frozen native prices required")
    groups = data.get("groups", [])
    if not 1 <= len(groups) <= config.max_groups:
        raise EligibilityError("bounded LLM smoke group count outside preregistered limit")
    rows, signal = [], False
    for group in groups:
        if group.get("scorer_sha256") != data["scorer_sha256"] or group.get("price_sha256") != data["price_sha256"]:
            raise EligibilityError("group scoring/price pins mismatch")
        candidates = group.get("candidates", [])
        for candidate in candidates:
            if candidate.get("receipt_verified") is not True or candidate.get("replay_verified") is not True or "native_outcomes" in candidate:
                raise EligibilityError("each candidate needs one verified native receipt/replay/outcome")
            outcome = candidate.get("native_outcome")
            if not isinstance(outcome, dict):
                raise EligibilityError("exactly one native terminal outcome required")
            value = score_native_outcome(outcome, contract)
            reward = candidate.get("return")
            if isinstance(reward, bool) or not isinstance(reward, (int, float)) or not math.isclose(value, reward, abs_tol=1e-10, rel_tol=0):
                raise EligibilityError("native return differs from frozen scorer")
            if outcome.get("level", group["level"]) != group["level"] or outcome.get("seed", group["seed"]) != group["seed"]:
                raise EligibilityError("native continuation level/seed lineage differs")
        advantage = group_advantages(validate_menu(group["menu"]), candidates)
        signal = signal or bool(any(advantage))
        validate_price_contract(group["menu"], prices)
        rows.append(_llm_row(group, identity, "fork", advantage.tolist()))
    if not signal:
        raise EligibilityError("entire dataset has no native advantage signal")
    for role in ("A1", "A2", "validation"):
        source = data.get("anchors", {}).get(role, []) if role != "validation" else data.get("validation_forks", [])
        cap = config.max_anchor_rows if role != "validation" else config.max_validation_rows
        if not 1 <= len(source) <= cap:
            raise EligibilityError(f"bounded smoke requires declared {role} rows within limit")
        for row in source:
            validate_price_contract(row["menu"], prices)
            rows.append(_llm_row(row, identity, role))
    return rows, hygiene


def check_only(data: dict, identity: dict, config: LLMSmokeConfig | dict | None = None) -> dict:
    """No Torch import, weight load, HTTP request, GPU query or optimization."""
    config = config if isinstance(config, LLMSmokeConfig) else LLMSmokeConfig(**(config or {}))
    config.validate()
    report = {"check_only": True, "optimizer_steps": 0, "gpu_started": False,
              "llm_training_validated": False, "formal_training_enabled": False,
              "dependencies": {name: importlib.util.find_spec(name) is not None for name in ("torch", "peft", "transformers")},
              "config": asdict(config)}
    try:
        rows, hygiene = validate_llm_dataset(data, identity, config)
        report.update({"native_eligibility": True, "hygiene": hygiene, "rows": len(rows),
                       "identity_sha256": _identity(identity), "status": "requires_broker_calibration_and_outer_launcher_permit"})
    except (ValueError, TypeError, KeyError) as exc:
        report.update({"native_eligibility": False, "status": "blocked", "reason": str(exc)})
    return report


def _floats(metrics):
    return {key: float(value.detach().cpu()) for key, value in metrics.items() if key != "p0"}


def _append_report_receipt(journal: Journal, kind: str, report: dict) -> dict:
    # The report subsequently gains its receipt hash. Retaining the caller's
    # mutable object would change the returned receipt after its hash was fixed.
    return journal.append(kind, json.loads(canonical_json(report)))


def exact_reference_probabilities(current_p, reference_p) -> bool:
    """Exact equality of every complete legal probability, never a delta gate."""
    torch = _torch()
    current = torch.as_tensor(current_p, dtype=torch.float64).detach().cpu()
    reference = torch.as_tensor(reference_p, dtype=torch.float64).detach().cpu()
    if current.ndim != 1 or current.numel() == 0 or current.shape != reference.shape:
        return False
    if not torch.isfinite(current).all() or torch.any(current <= 0):
        return False
    if not torch.allclose(current.sum(), current.new_tensor(1.0), atol=1e-12, rtol=0):
        return False
    return bool(torch.equal(current, reference))


def accumulate_two_pass_gradient(model, tokenizer, request: dict, advantages,
                                  reference_p, beta: float, *, scale: float = 1.0,
                                  max_prompt_tokens: int = 16384,
                                  checkpoint_options: bool = False,
                                  recompute_score_tolerance: float = 1e-5,
                                  boundary_check=None, first_scores=None) -> dict:
    """Exact score-Jacobian chain rule, holding only one option graph at a time.

    Pass one obtains all scores without graphs and derives dL/dscore. Pass two
    teacher-forces each suffix again, accumulating coefficient*dscore/dtheta.
    It is algebraically the same complete objective as full autograd, rather
    than sampled policy gradient. Existing parameter gradients are accumulated.
    """
    torch = _torch()
    _require_deterministic_forward(model)
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("positive finite objective averaging scale required")
    tokens = tokenize_request(tokenizer, request)
    reused = first_scores is not None
    if first_scores is None:
        with torch.no_grad():
            first_scores = differentiable_label_scores(model, tokens, max_prompt_tokens=max_prompt_tokens,
                                                       boundary_check=boundary_check)
    else:
        first_scores = torch.as_tensor(first_scores, dtype=torch.float64)
        if first_scores.shape != (len(tokens["suffixes"]),) or not torch.isfinite(first_scores).all():
            raise ValueError("complete finite cached calibration scores required")
    leaves = first_scores.detach().double().clone().requires_grad_(True)
    loss, metrics = full_legal_pi_adv_loss(leaves.log_softmax(0), advantages, reference_p, beta)
    coefficients, = torch.autograd.grad(loss, leaves)
    max_delta = 0.0
    for index, suffix in enumerate(tokens["suffixes"]):
        if boundary_check is not None:
            boundary_check()
        selected = {**tokens, "suffixes": [suffix]}
        score = differentiable_label_scores(model, selected, max_prompt_tokens=max_prompt_tokens,
                                            checkpoint_options=checkpoint_options)[0]
        delta = float((score.detach().double() - leaves[index].detach().to(score.device)).abs().cpu())
        max_delta = max(max_delta, delta)
        if delta > recompute_score_tolerance:
            raise ValueError("two-pass forward scores drifted; stochastic/recomputed graph cannot be trusted")
        (score * (coefficients[index].detach().to(device=score.device, dtype=score.dtype) * scale)).backward()
    return {"loss": float(loss.detach().cpu()), **_floats(metrics),
            "score_coefficients": coefficients.detach().cpu().tolist(),
            "recompute_max_score_delta": max_delta, "legal_options": len(tokens["suffixes"]),
            "reused_step0_calibration_scores": reused,
            "gradient_method": "exact_two_pass_complete_legal_sequence_scores"}


def _require_permit(context, config: LLMSmokeConfig):
    """Launcher owns real authorization, exclusive lock, busy check and budget."""
    if context is None or not callable(getattr(context, "check", None)):
        raise EligibilityError("outer launcher RunContext.check and explicit GPU training permit required")
    for name, expected in (("kind", "llm-one-step-smoke"), ("gpu_authorization_verified", True),
                           ("gpu_lock_held", True), ("max_optimizer_steps", 1)):
        if getattr(context, name, None) != expected:
            raise EligibilityError(f"outer launcher permit must declare {name}={expected!r}")
    deadline = getattr(context, "deadline", None)
    if not isinstance(deadline, (int, float)) or not math.isfinite(deadline) or not 0 < deadline - time.monotonic() <= config.wall_cap_seconds:
        raise EligibilityError("outer launcher needs a live bounded monotonic deadline")
    context.check()


def prepare_lora(model=None, *, rank: int = 16, alpha: int = 32,
                 target_modules=None,
                 check_only: bool = True, launch_context=None):
    """Attach an actual trainable PEFT adapter to an externally loaded model.

    No weight download or device movement occurs. The caller owns an isolated
    model instance. Existing adapters are refused rather than silently merged.
    """
    if check_only:
        return {"check_only": True, "adapter_created": False, "rank": rank, "alpha": alpha,
                "target_modules": list(target_modules or ("q_proj", "k_proj", "v_proj", "o_proj")),
                "text_only_targets_require_loaded_model_inspection": target_modules is None,
                "dropout": 0.0, "formal_training_enabled": False}
    if model is None or hasattr(model, "peft_config"):
        raise EligibilityError("externally loaded isolated base model without an existing adapter required")
    _require_permit(launch_context, LLMSmokeConfig())
    target_modules = text_attention_targets(model) if target_modules is None else target_modules
    if isinstance(rank, bool) or not isinstance(rank, int) or rank < 1 or isinstance(alpha, bool) or not isinstance(alpha, int) or alpha < 1:
        raise ValueError("positive integer LoRA rank/alpha required")
    if not target_modules or any(not isinstance(name, str) or not name for name in target_modules):
        raise ValueError("explicit attention target modules required")
    from peft import LoraConfig, get_peft_model
    adapter = get_peft_model(model, LoraConfig(r=rank, lora_alpha=alpha,
                             target_modules=list(target_modules), lora_dropout=0.0,
                             bias="none", task_type="CAUSAL_LM", init_lora_weights=True))
    adapter.eval()
    return adapter


def text_attention_targets(model) -> list[str]:
    """Select actual text attention names, excluding image towers in Mistral3."""
    projections = ("q_proj", "k_proj", "v_proj", "o_proj")
    names = [name for name, _ in model.named_modules() if name.rsplit(".", 1)[-1] in projections]
    has_language_prefix = any("language_model" in name.split(".") for name in names)
    if has_language_prefix:
        names = [name for name in names if "language_model" in name.split(".")]
    elif any("vision" in part or "visual" in part for name in names for part in name.split(".")):
        names = [name for name in names if not any("vision" in part or "visual" in part for part in name.split("."))]
    if not names or any(not any(name.endswith("." + suffix) for name in names) for suffix in projections):
        raise EligibilityError("cannot establish all four text attention projection targets")
    return sorted(names)


def _require_deterministic_forward(model):
    torch = _torch()
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout) and module.training and module.p != 0:
            raise EligibilityError("active dropout is incompatible with exact two-pass gradients")
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm) and module.training:
            raise EligibilityError("stateful training-mode batch normalization is incompatible with two-pass scoring")
    # HF attention implements dropout functionally rather than nn.Dropout.
    for module in model.modules():
        cfg = getattr(module, "config", None)
        if module.training and cfg is not None:
            for name in ("attention_dropout", "attention_probs_dropout_prob", "hidden_dropout_prob"):
                if float(getattr(cfg, name, 0) or 0) != 0:
                    raise EligibilityError(f"nonzero functional {name} is incompatible with deterministic two-pass")


def _adapter_parameters(model):
    if not hasattr(model, "peft_config") or not isinstance(model.peft_config, dict):
        raise EligibilityError("actual PEFT model required; CPU toy is not an LLM adapter smoke")
    selected = {}
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            if "lora_" not in name:
                raise EligibilityError("base/non-LoRA trainable parameters found; refusing full model updates")
            selected[name] = parameter
    if not selected:
        raise EligibilityError("no trainable LoRA parameters")
    for peft_config in model.peft_config.values():
        if float(getattr(peft_config, "lora_dropout", 0)) != 0:
            raise EligibilityError("zero LoRA dropout required for exact deterministic two-pass scoring")
    return selected


def _adapter_sha(parameters):
    digest = hashlib.sha256()
    for name in sorted(parameters):
        tensor = parameters[name].detach().cpu().contiguous()
        digest.update(canonical_bytes({"name": name, "shape": list(tensor.shape), "dtype": str(tensor.dtype)}))
        digest.update(tensor.view(_torch().uint8).numpy().tobytes())
    return digest.hexdigest()


def _measure_alignment(model, tokenizer, rows, identity, config, boundary_check=None):
    torch = _torch()
    measurements = []
    model.eval()
    with torch.no_grad():
        for row in rows:
            measured = differentiable_distribution(model, tokenizer, row["request"], max_prompt_tokens=config.max_prompt_tokens,
                                                  boundary_check=boundary_check)
            tokens, reference = measured["tokenized"], row["reference"]
            if reference.get("prompt_tokens") != tokens["prompt_tokens"] or reference.get("label_token_ids") != tokens["label_token_ids"]:
                raise EligibilityError("broker/training prompt token IDs or label-plus-EOS sequences do not align")
            if identity.get("scoring_execution") == CANONICAL_EXECUTION and reference.get("prompt_token_ids_sha256") != sha256_data(tokens["input_ids"]):
                raise EligibilityError("canonical independent reference prompt token bytes do not align")
            p = measured["p"].detach().cpu().tolist()
            maximum_delta = max(abs(left - right) for left, right in zip(p, reference["p"]))
            agreement = max(range(len(p)), key=p.__getitem__) == max(range(len(p)), key=reference["p"].__getitem__)
            measurements.append({"role": row["role"], "prompt_sha256": tokens["prompt_sha256"],
                                 "request_sha256": sha256_data(row["request"]),
                                 "reference_sha256": sha256_data(reference), "maximum_delta_p": maximum_delta,
                                 "argmax_agree": agreement, "tokenized_sha256": sha256_data(tokens),
                                 "sequence_scores": measured["scores"].detach().cpu().tolist(),
                                 "p": p, "logp": measured["logp"].detach().cpu().tolist(),
                                 "complete_distribution_exact_equal": exact_reference_probabilities(p, reference["p"])})
    max_delta = max(item["maximum_delta_p"] for item in measurements)
    agreement = sum(item["argmax_agree"] for item in measurements) / len(measurements)
    passed = (len(measurements) >= config.minimum_calibration_rows
              and max_delta <= config.alignment_max_delta_p
              and agreement >= config.alignment_min_argmax_agreement)
    return {"schema_version": 1, "passed": passed, "kind": "same_base_adapter_broker_vs_differentiable_backend",
            "identity_sha256": _identity(identity), "rows": measurements,
            "measured_max_delta_p": max_delta, "measured_argmax_agreement": agreement,
            "required_max_delta_p": config.alignment_max_delta_p,
            "required_argmax_agreement": config.alignment_min_argmax_agreement,
            "optimizer_updates_during_calibration": 0}


def _llm_metrics(model, tokenizer, rows, config, precomputed=None, boundary_check=None):
    torch = _torch()
    values = {role: [] for role in ("fork", "A1", "A2", "validation")}
    with torch.no_grad():
        for row in rows:
            cached = None if precomputed is None else precomputed.get(sha256_data(row["request"]))
            distribution = ({"logp": torch.as_tensor(cached["logp"], dtype=torch.float64)} if cached is not None
                            else differentiable_distribution(model, tokenizer, row["request"], max_prompt_tokens=config.max_prompt_tokens,
                                                             boundary_check=boundary_check))
            adv = row["advantages"] if row["role"] == "fork" else [0.0] * len(row["request"]["labels"])
            _, metrics = full_legal_pi_adv_loss(distribution["logp"], adv, row["reference"]["p"], 0.0)
            values[row["role"]].append(_floats(metrics))
    values["anchors"] = values["A1"] + values["A2"]
    result = {}
    for role, items in values.items():
        kls = sorted(item["kl"] for item in items)
        # Match NumPy's linear p90 interpolation without depending on NumPy here.
        position = (len(kls) - 1) * 0.9
        lo, hi = math.floor(position), math.ceil(position)
        result[role] = {name: sum(item[name] for item in items) / len(items) for name in items[0]}
        result[role].update({"count": len(items), "kl_p90": kls[lo] + (kls[hi] - kls[lo]) * (position - lo)})
    return result


def _safeguards(metrics, config):
    failures = []
    for role, key, maximum in (("validation", "kl", config.hard_fork_mean_kl),
                               ("fork", "kl_p90", config.hard_train_p90_kl),
                               ("anchors", "kl", config.hard_anchor_mean_kl)):
        if metrics[role][key] > maximum:
            failures.append(f"{role}_{key}")
    for role in ("fork", "validation", "A1", "A2"):
        if metrics[role]["entropy"] + 1e-12 < config.entropy_reference_fraction * metrics[role]["reference_entropy"]:
            failures.append(f"{role}_entropy_floor")
    return failures


def one_step_smoke(model, tokenizer, data: dict, identity: dict,
                   config: LLMSmokeConfig | dict | None = None, *,
                   bounded_smoke: bool = False, launch_context=None,
                   outputdir: str | Path | None = None) -> dict:
    """Only explicit bounded invocation can perform one actual LoRA update.

    The external launcher must kill on its deadline, hold an exclusive GPU lock,
    verify owner authorization and budget, and supply live boundary checks. A
    rejected update is restored in the supplied model and never saved as a
    candidate. Validation rows are metrics only; heldout never enters this call.
    """
    config = config if isinstance(config, LLMSmokeConfig) else LLMSmokeConfig(**(config or {}))
    if not bounded_smoke:
        return check_only(data, identity, config)
    config.validate(updating=True)
    _require_permit(launch_context, config)
    if outputdir is None:
        raise EligibilityError("fresh bounded experiment output directory required")
    rows, hygiene = validate_llm_dataset(data, identity, config)
    parameters = _adapter_parameters(model)
    destination = Path(outputdir)
    destination.mkdir(parents=True, exist_ok=True)
    if any(destination.iterdir()):
        raise FileExistsError("actual LLM smoke requires a fresh output directory")
    torch = _torch()
    was_training = model.training
    model.eval()
    original = {name: value.detach().cpu().clone() for name, value in parameters.items()}
    original_sha = _adapter_sha(original)
    torch.save(original, destination / "reference_adapter.pt")
    reference_file_sha = hashlib.sha256((destination / "reference_adapter.pt").read_bytes()).hexdigest()
    broker_reference_pins = [sha256_data(row["reference"]) for row in rows]
    journal = Journal(destination / "updates.jsonl")
    manifest = {"schema_version": 1, "identity": identity, "identity_sha256": _identity(identity),
                "reference_adapter_parameter_sha256": original_sha, "dataset_sha256": sha256_data(data),
                "reference_adapter_file_sha256": reference_file_sha,
                "broker_reference_sha256": broker_reference_pins,
                "config": asdict(config), "optimizer_step_cap": 1, "formal_training_enabled": False,
                "backend_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    manifest["actual_adapter_topology"] = {name: {
        "rank": int(getattr(value, "r")), "alpha": int(getattr(value, "lora_alpha")),
        "dropout": float(getattr(value, "lora_dropout")),
        "target_modules": sorted(getattr(value, "target_modules"))}
        for name, value in model.peft_config.items()}
    (destination / "FROZEN-MANIFEST.json").write_bytes(canonical_bytes(manifest) + b"\n")
    journal.append("reference_frozen", manifest)
    report = {"status": "rejected", "kind": "actual_lora_one_step_smoke", "optimizer_steps": 0,
              "candidate_saved": False, "formal_training_enabled": False, "heldout_accessed": False,
              "gameplay_improvement": "unmeasured", "hygiene": hygiene,
              "reference_adapter_parameter_sha256": original_sha, "config": asdict(config)}

    def finish(reason, accepted=False):
        if not accepted:
            with torch.no_grad():
                for name, parameter in parameters.items():
                    parameter.copy_(original[name])
                    parameter.grad = None
        report.update({"status": "accepted_lora_one_step_smoke" if accepted else "rejected",
                       "reason": reason, "reference_checkpoint_preserved": (
                           _adapter_sha(original) == original_sha and hashlib.sha256(
                               (destination / "reference_adapter.pt").read_bytes()).hexdigest() == reference_file_sha),
                       "broker_reference_preserved": [sha256_data(row["reference"]) for row in rows] == broker_reference_pins,
                       "rejected_model_restored": (not accepted and _adapter_sha(parameters) == original_sha)})
        receipt = _append_report_receipt(journal, "llm_smoke_receipt", report)
        report["receipt_sha256"] = receipt["sha256"]
        (destination / "LLM-SMOKE-REPORT.json").write_bytes(canonical_bytes(report) + b"\n")
        (destination / "UPDATE-RECEIPT.json").write_bytes(canonical_bytes(receipt) + b"\n")
        model.train(was_training)
        return report

    try:
        launch_context.check()
        if identity.get("equivalent_initial_adapter") == "zero_init_lora":
            b_parameters = {name: parameter for name, parameter in parameters.items() if "lora_B" in name}
            if not b_parameters or any(torch.count_nonzero(parameter.detach()).item() != 0 for parameter in b_parameters.values()):
                return finish("unadapted_broker_equivalence_requires_exactly_zero_lora_B")
            report["initial_adapter_equivalence"] = {"kind": "zero_init_lora", "verified_lora_B_zero": True,
                                                     "broker_reference_adapter": None,
                                                     "scope": "same_base_zero_delta_before_first_update"}
        elif identity["adapter_sha256"] is None:
            return finish("unadapted_broker_requires_explicit_zero_init_equivalence_mapping")
        calibration = _measure_alignment(model, tokenizer, rows, identity, config, boundary_check=launch_context.check)
        report["alignment_calibration"] = calibration
        journal.append("broker_alignment", calibration)
        if not calibration["passed"]:
            return finish("same_model_broker_alignment_failed")
        cached_step0 = {measurement["request_sha256"]: measurement for measurement in calibration["rows"]}
        before = _llm_metrics(model, tokenizer, rows, config, precomputed=cached_step0)
        report["before"] = before
        report["step0_forward_reuse"] = {"before_metrics_from_calibration": True,
                                        "two_pass_first_scores_from_calibration": True}
        failures = _safeguards(before, config)
        if failures:
            report["gate_failures"] = failures
            return finish("step0_safeguard_failed")
        optimizer = torch.optim.AdamW(list(parameters.values()), lr=config.learning_rate, weight_decay=0.0)
        optimizer.zero_grad(set_to_none=True)
        # HF layer-level checkpointing is gated on training=True. Every active
        # stochastic operation must be zero; score recomputation is checked too.
        if config.checkpoint_options and hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            model.train()
        _require_deterministic_forward(model)
        updates = []
        forks = [row for row in rows if row["role"] == "fork"]
        anchors = [row for row in rows if row["role"] in ("A1", "A2")]
        for selected, beta in ((forks, config.beta_fork), (anchors, config.beta_anchor)):
            for row in selected:
                cached = cached_step0[sha256_data(row["request"])]
                if row["role"] in ("A1", "A2") and cached["complete_distribution_exact_equal"]:
                    updates.append({"role": row["role"], "gradient_method": "analytic_zero_kl_gradient_at_exact_full_reference",
                                    "complete_distribution_exact_equal": True, "legal_options": len(row["request"]["labels"]),
                                    "loss": 0.0, "kl": 0.0, "all_legal_score_derivatives": [0.0] * len(row["request"]["labels"]),
                                    "no_tolerance_used_for_zero_shortcut": True})
                    continue
                advantage = row["advantages"] if row["role"] == "fork" else [0.0] * len(row["request"]["labels"])
                updates.append({"role": row["role"], **accumulate_two_pass_gradient(
                    model, tokenizer, row["request"], advantage, row["reference"]["p"], beta,
                    scale=1.0 / len(selected), max_prompt_tokens=config.max_prompt_tokens,
                    checkpoint_options=config.checkpoint_options,
                    recompute_score_tolerance=config.recompute_score_tolerance,
                    boundary_check=launch_context.check, first_scores=cached["sequence_scores"])})
        launch_context.check()
        gradient_norm = torch.nn.utils.clip_grad_norm_(list(parameters.values()), config.max_grad_norm, error_if_nonfinite=True)
        optimizer.step()
        report["optimizer_steps"] = 1
        updated_sha = _adapter_sha(parameters)
        report.update({"parameter_update_verified": updated_sha != original_sha,
                       "post_update_parameter_sha256": updated_sha,
                       "optimizer_runtime_completed": True})
        model.eval()
        after = _llm_metrics(model, tokenizer, rows, config, boundary_check=launch_context.check)
        report["after"] = after
        report["gradient_norm"] = float(gradient_norm.detach().cpu())
        failures = _safeguards(after, config)
        journal.append("one_optimizer_update", {"step": 1, "row_gradients": updates, "metrics": after,
                       "gradient_norm": report["gradient_norm"], "gate_failures": failures,
                       "adapter_parameter_sha256": _adapter_sha(parameters)})
        if failures:
            report["gate_failures"] = failures
            return finish("post_update_safeguard_failed")
        delta = after["fork"]["expected_advantage"] - before["fork"]["expected_advantage"]
        report["learncheck"] = {"scope": "native_training_direction_only", "expected_advantage_delta": delta,
                                "passed": delta > config.minimum_expected_advantage_delta}
        if not report["learncheck"]["passed"]:
            return finish("one_step_learncheck_failed")
        if [sha256_data(row["reference"]) for row in rows] != broker_reference_pins:
            return finish("frozen_broker_reference_changed")
        # Adaptive betas are reported for the next separately authorized run.
        fork_beta, anchor_beta = config.beta_fork, config.beta_anchor
        if after["validation"]["kl"] > config.target_fork_kl:
            fork_beta = min(config.beta_max, fork_beta * config.beta_factor)
        if after["anchors"]["kl"] > config.target_anchor_kl:
            anchor_beta = min(config.beta_max, anchor_beta * config.beta_factor)
        elif after["anchors"]["kl"] < config.target_anchor_kl / 2:
            anchor_beta = max(config.beta_min, anchor_beta / config.beta_factor)
        report["next_run_betas"] = {"fork": fork_beta, "combined_A1_A2": anchor_beta}
        launch_context.check()
        model.save_pretrained(destination / "candidate_adapter", safe_serialization=True)
        report["candidate_saved"] = True
        report["candidate_parameter_sha256"] = _adapter_sha(parameters)
        return finish("actual_lora_optimizer_step_and_gates_passed", accepted=True)
    except Exception as exc:
        return finish(f"runtime_rejection: {type(exc).__name__}: {exc}")
