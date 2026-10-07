"""Auditable full-legal pi_adv training, with native continuation eligibility.

The compact CPU scorer is a framework validation model. This module does not
launch a game, start GPU work, promote a model, or inspect heldout outcomes.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from .journal import Journal, canonical_bytes, canonical_json, sha256_data
from .menus import build_menu, prompt_sha256, validate_menu
from .policies import TinyOptionPolicy, option_features
from .pools import HygieneError, PoolRegistry, audit_dataset


class EligibilityError(ValueError):
    pass


@dataclass
class TrainingConfig:
    seed: int = 0
    hidden_size: int = 16
    steps: int = 40
    learning_rate: float = 0.03
    max_grad_norm: float = 1.0
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
    alignment_tolerance: float = 1e-10
    learncheck_min_delta: float = 1e-6
    reference_path: str | None = None
    stop_file: str | None = None

    def validate(self):
        if isinstance(self.steps, bool) or not isinstance(self.steps, int) or not 1 <= self.steps <= 1000:
            raise ValueError("CPU validation run requires 1..1000 steps")
        if isinstance(self.hidden_size, bool) or not isinstance(self.hidden_size, int) or not 1 <= self.hidden_size <= 256:
            raise ValueError("hidden_size must be 1..256")
        for key in ("learning_rate", "max_grad_norm", "beta_factor", "beta_min", "beta_max",
                    "target_fork_kl", "target_anchor_kl", "hard_fork_mean_kl",
                    "hard_train_p90_kl", "hard_anchor_mean_kl", "alignment_tolerance"):
            value = getattr(self, key)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{key} must be finite and positive")
        for key in ("beta_fork", "beta_anchor"):
            value = getattr(self, key)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not self.beta_min <= value <= self.beta_max:
                raise ValueError(f"{key} must be positive and within adaptive beta limits")
        for key in ("learncheck_min_delta",):
            value = getattr(self, key)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{key} must be finite and nonnegative")
        if not 0.5 <= self.entropy_reference_fraction <= 1.0:
            raise ValueError("entropy fraction must preserve at least half of reference entropy")
        if self.beta_factor <= 1 or self.beta_max < self.beta_min:
            raise ValueError("invalid adaptive beta limits")


def _sha(value, name):
    if not isinstance(value, str) or len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
        raise EligibilityError(f"{name} requires a lowercase SHA256 pin")
    return value


def score_native_outcome(outcome: dict, contract: dict) -> float:
    """Frozen KR1 protocol: terminal win/loss plus a declared lives weight.

    A timeout, command rejection or missing terminal result is an engineering
    invalidation, never a fabricated zero reward.
    """
    if contract.get("name") == "kr1-level1-terminal-v2":
        from .reward_level1 import score_level1_outcome
        return score_level1_outcome(outcome, contract)
    if contract.get("name") != "kr1-terminal-v1":
        raise EligibilityError("unsupported scoring contract")
    if outcome.get("source") != "native" or outcome.get("terminal") is not True:
        raise EligibilityError("verified native terminal outcome required")
    won, lost = outcome.get("level_won"), outcome.get("level_lost")
    if not isinstance(won, bool) or not isinstance(lost, bool) or won == lost:
        raise EligibilityError("native outcome must record exactly one of win/loss")
    lives = outcome.get("lives")
    if isinstance(lives, bool) or not isinstance(lives, (int, float)) or not math.isfinite(lives) or lives < 0:
        raise EligibilityError("native terminal lives required")
    for key in ("win", "loss", "lives_weight"):
        value = contract.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise EligibilityError(f"scoring contract requires finite {key}")
    return float(contract["win"] if won else contract["loss"]) + float(contract["lives_weight"]) * float(lives)


def group_advantages(labels: list[str], candidates: list[dict]) -> np.ndarray:
    """All legal labels; uncontinued labels receive minimum candidate R.

    Baseline is the arithmetic mean of the K continued candidates, including
    each candidate exactly once. This is E_pi[adv], not sampled log-pi loss.
    """
    if len(candidates) < 2:
        raise EligibilityError("at least two native continuations per fork required")
    rewards = {}
    for candidate in candidates:
        label, reward = candidate.get("label"), candidate.get("return")
        if label not in labels or label in rewards:
            raise EligibilityError("candidate must be a distinct legal label")
        if isinstance(reward, bool) or not isinstance(reward, (int, float)) or not math.isfinite(reward):
            raise EligibilityError("candidate return must be finite")
        rewards[label] = float(reward)
    baseline = sum(rewards.values()) / len(rewards)
    minimum = min(rewards.values())
    return np.asarray([rewards.get(label, minimum) - baseline for label in labels], dtype=np.float64)


def validate_price_contract(menu: list[dict], contract: dict) -> None:
    """Compare every legal option against frozen final-template native prices.

    Build-animation templates can cost zero; the charged final tower template
    is selected through the separately frozen build_templates mapping.
    """
    templates, build_templates = contract.get("templates"), contract.get("build_templates")
    if not isinstance(templates, dict) or not isinstance(build_templates, dict):
        raise EligibilityError("price contract requires templates and build_templates mappings")
    for option in menu:
        action = option["action"]
        action_name = action["action"]
        if action_name == "build_tower":
            template = build_templates.get(action.get("tower_type"))
            expected = templates.get(template)
        elif action_name == "upgrade_tower":
            expected = templates.get(action.get("target"))
        else:
            expected = 0
        actual = option.get("cost")
        if any(isinstance(value, bool) or not isinstance(value, (int, float))
               or not math.isfinite(value) or value < 0 for value in (actual, expected)):
            raise EligibilityError("unknown or invalid native option price")
        if not math.isclose(actual, expected, abs_tol=1e-10, rel_tol=0):
            raise EligibilityError("legal option cost differs from frozen native price contract")


def distribution_metrics(p: np.ndarray, p0: np.ndarray) -> dict:
    p, p0 = np.asarray(p), np.asarray(p0)
    if p.shape != p0.shape or p.ndim != 1 or np.any(p <= 0) or np.any(p0 <= 0):
        raise ValueError("matching, strictly positive complete legal distributions required")
    if not np.isclose(p.sum(), 1.0, atol=1e-10, rtol=0) or not np.isclose(p0.sum(), 1.0, atol=1e-10, rtol=0):
        raise ValueError("distributions must be normalized")
    logp = np.log(p)
    return {"kl": float(np.dot(p, logp - np.log(p0))),
            "entropy": float(-np.dot(p, logp)),
            "reference_entropy": float(-np.dot(p0, np.log(p0))),
            "argmax_changed": bool(p.argmax() != p0.argmax()),
            "max_probability_delta": float(np.max(np.abs(p - p0)))}


def pi_adv_loss_and_grad(policy: TinyOptionPolicy, features: np.ndarray,
                         advantages: np.ndarray, reference_p: np.ndarray,
                         beta: float = 0.0):
    """Exact differentiable -E_pi[adv] + beta KL(pi || pi0)."""
    p, cache = policy.forward(features)
    advantages = np.asarray(advantages, dtype=np.float64)
    reference_p = np.asarray(reference_p, dtype=np.float64)
    if advantages.shape != p.shape or not np.all(np.isfinite(advantages)) or beta < 0:
        raise ValueError("invalid advantages or KL beta")
    metrics = distribution_metrics(p, reference_p)
    objective = float(np.dot(p, advantages))
    # Softmax Jacobian applies to dL/dp. The constant +1 term in dKL/dp
    # cancels through that Jacobian, but is kept to show the exact derivative.
    dprob = -advantages + beta * (np.log(p) - np.log(reference_p) + 1.0)
    dlogits = p * (dprob - np.dot(p, dprob))
    metrics["expected_advantage"] = objective
    loss = -objective + beta * metrics["kl"]
    return float(loss), policy.backward(cache, dlogits), metrics


def anchor_loss_and_grad(policy: TinyOptionPolicy, features: np.ndarray,
                         reference_p: np.ndarray, beta: float):
    return pi_adv_loss_and_grad(policy, features, np.zeros(len(features)), reference_p, beta)


@dataclass
class PreparedRow:
    features: np.ndarray
    reference_p: np.ndarray
    advantages: np.ndarray | None
    fork_id: str
    labels: list[str]
    prompt_hash: str
    reference_origin: str


def _prepare_row(row: dict, reference: TinyOptionPolicy, tolerance: float,
                 advantages: np.ndarray | None = None) -> PreparedRow:
    state, menu = row.get("state"), row.get("menu")
    if not isinstance(state, dict):
        raise EligibilityError("native fork state required")
    labels = validate_menu(menu)
    wait = next((option["action"].get("ticks") for option in menu if option["action"]["action"] == "wait"), 30)
    native_menu = build_menu(state, wait_ticks=wait)
    if canonical_json(menu) != canonical_json(native_menu):
        raise EligibilityError("menu differs from complete supported native legal action scope")
    measured = reference.distribution(state, menu)
    ref = row.get("reference", measured)
    if ref.get("labels") != labels or ref.get("prompt_sha256") != measured["prompt_sha256"]:
        raise EligibilityError("step0 reference labels/prompt bytes do not align")
    if ref.get("model") != measured["model"]:
        raise EligibilityError("step0 reference model identity does not align")
    p0 = np.asarray(ref.get("p", []), dtype=np.float64)
    logp = np.asarray(ref.get("logp", []), dtype=np.float64)
    if p0.shape != (len(labels),) or logp.shape != p0.shape or not np.all(np.isfinite(logp)):
        raise EligibilityError("step0 full legal p/logp required")
    if np.any(p0 <= 0) or not np.isclose(p0.sum(), 1, atol=tolerance, rtol=0) or not np.allclose(np.log(p0), logp, atol=tolerance, rtol=0):
        raise EligibilityError("invalid step0 complete legal distribution")
    if np.max(np.abs(p0 - measured["p"])) > tolerance:
        raise EligibilityError("step0 reference probability alignment failed")
    return PreparedRow(option_features(state, menu), p0, advantages,
                       str(row.get("fork_id", "anchor")), labels, measured["prompt_sha256"],
                       "supplied" if "reference" in row else "frozen_local_cpu")


def _prepare_data(data: dict, reference: TinyOptionPolicy, config: TrainingConfig):
    if data.get("source") != "real_game":
        raise EligibilityError("only real_game native outcomes are eligible; synthetic fixtures cannot become real training")
    hygiene = audit_dataset(data)
    if not data.get("groups"):
        raise EligibilityError("no verified native continuation groups; RL training stays blocked")
    contract = data.get("scoring_contract")
    if not isinstance(contract, dict):
        raise EligibilityError("frozen scoring contract required")
    scorer_sha = _sha(data.get("scorer_sha256"), "scorer_sha256")
    price_sha = _sha(data.get("price_sha256"), "price_sha256")
    if scorer_sha != sha256_data(contract):
        raise EligibilityError("scoring contract SHA256 mismatch")
    if not isinstance(data.get("price_contract"), dict):
        raise EligibilityError("frozen native price contract required")
    if price_sha != sha256_data(data["price_contract"]):
        raise EligibilityError("price contract SHA256 mismatch")
    groups = []
    for group in data["groups"]:
        if group.get("scorer_sha256") != scorer_sha or group.get("price_sha256") != price_sha:
            raise EligibilityError("group scoring/price pins differ from frozen contract")
        candidates = group.get("candidates", [])
        if not isinstance(candidates, list):
            raise EligibilityError("one continuation record per candidate required")
        for candidate in candidates:
            if candidate.get("receipt_verified") is not True or candidate.get("replay_verified") is not True:
                raise EligibilityError("native action receipt and fork replay must both pass")
            if "native_outcomes" in candidate:
                raise EligibilityError("each candidate must contain exactly one native outcome")
            native = candidate.get("native_outcome")
            if not isinstance(native, dict):
                raise EligibilityError("exactly one structured native outcome required")
            measured = score_native_outcome(native, contract)
            reward = candidate.get("return")
            if isinstance(reward, bool) or not isinstance(reward, (float, int)) or not math.isclose(measured, reward, abs_tol=1e-10, rel_tol=0):
                raise EligibilityError("candidate return differs from frozen native scorer")
            if native.get("level", group["level"]) != group["level"] or native.get("seed", group["seed"]) != group["seed"]:
                raise EligibilityError("continuation outcome has different level/seed lineage")
        labels = validate_menu(group.get("menu"))
        validate_price_contract(group["menu"], data["price_contract"])
        advantages = group_advantages(labels, candidates)
        groups.append(_prepare_row(group, reference, config.alignment_tolerance, advantages))
    flat_groups = sum(not np.any(row.advantages) for row in groups)
    hygiene.update({"groups_flat_reward": flat_groups, "groups_with_reward_signal": len(groups) - flat_groups})
    if flat_groups == len(groups):
        raise EligibilityError("entire dataset has no differentiated native reward signal")
    anchors = {name: [_prepare_row(row, reference, config.alignment_tolerance)
                      for row in data.get("anchors", {}).get(name, [])] for name in ("A1", "A2")}
    if not anchors["A1"] or not anchors["A2"]:
        raise EligibilityError("both A1 and A2 train-pool anchor sets are required")
    validation = [_prepare_row(row, reference, config.alignment_tolerance)
                  for row in data.get("validation_forks", [])]
    if not validation:
        raise EligibilityError("reward-free validation forks required for post-training safeguards")
    for row in data.get("anchors", {}).get("A1", []) + data.get("anchors", {}).get("A2", []) + data.get("validation_forks", []):
        validate_price_contract(row["menu"], data["price_contract"])
    return groups, anchors, validation, hygiene


def _average_metrics(policy: TinyOptionPolicy, rows: list[PreparedRow]):
    measurements = []
    for row in rows:
        p, _ = policy.forward(row.features)
        measured = distribution_metrics(p, row.reference_p)
        if row.advantages is not None:
            measured["expected_advantage"] = float(np.dot(p, row.advantages))
        measurements.append(measured)
    if not measurements:
        return {"count": 0}
    result = {key: float(np.mean([item[key] for item in measurements])) for key in measurements[0]}
    result.update({"count": len(rows), "kl_p90": float(np.quantile([item["kl"] for item in measurements], 0.9))})
    return result


def _metrics(policy, groups, anchors, validation):
    return {"train_forks": _average_metrics(policy, groups),
            "validation_forks": _average_metrics(policy, validation),
            "anchors": _average_metrics(policy, anchors["A1"] + anchors["A2"]),
            "A1": _average_metrics(policy, anchors["A1"]),
            "A2": _average_metrics(policy, anchors["A2"])}


def _gate_failures(metrics: dict, config: TrainingConfig) -> list[str]:
    failures = []
    if metrics["validation_forks"]["kl"] > config.hard_fork_mean_kl:
        failures.append("validation_fork_mean_kl")
    if metrics["train_forks"]["kl_p90"] > config.hard_train_p90_kl:
        failures.append("train_fork_p90_kl")
    if metrics["anchors"]["kl"] > config.hard_anchor_mean_kl:
        failures.append("anchor_mean_kl")
    for name in ("train_forks", "validation_forks", "A1", "A2"):
        item = metrics[name]
        if item["entropy"] + 1e-12 < config.entropy_reference_fraction * item["reference_entropy"]:
            failures.append(f"{name}_entropy_floor")
    return failures


def _combined_loss(policy, groups, anchors, beta_fork, beta_anchor):
    gradients = {name: np.zeros_like(value) for name, value in policy.params.items()}
    total_loss = 0.0
    for rows, beta, is_fork in ((groups, beta_fork, True), (anchors["A1"] + anchors["A2"], beta_anchor, False)):
        for row in rows:
            adv = row.advantages if is_fork else np.zeros(len(row.labels))
            loss, grad, _ = pi_adv_loss_and_grad(policy, row.features, adv, row.reference_p, beta)
            total_loss += loss / len(rows)
            for name in gradients:
                gradients[name] += grad[name] / len(rows)
    return total_loss, gradients


def _write_json(path: Path, value):
    path.write_bytes(canonical_bytes(value) + b"\n")


def train_groups(data: dict | str | Path, config: TrainingConfig | dict | None,
                 outputdir: str | Path) -> dict:
    """Run a bounded CPU experiment; rejection preserves the frozen reference.

    Output includes config/data/contract pins, per-step update evidence and a
    receipt. Acceptance certifies a mathematical training smoke with verified
    real game rows; it does not certify policy improvement or training readiness.
    """
    if isinstance(data, (str, Path)):
        data = json.loads(Path(data).read_text(encoding="utf-8"))
    config = config if isinstance(config, TrainingConfig) else TrainingConfig(**(config or {}))
    config.validate()
    destination = Path(outputdir)
    destination.mkdir(parents=True, exist_ok=True)
    if any((destination / name).exists() for name in ("TRAINING-REPORT.json", "candidate.npz", "updates.jsonl", "reference.npz")):
        raise FileExistsError("training output must be a fresh experiment directory")
    reference = TinyOptionPolicy.load(config.reference_path) if config.reference_path else TinyOptionPolicy(config.seed, config.hidden_size)
    reference_sha = reference.save(destination / "reference.npz")
    journal = Journal(destination / "updates.jsonl")
    dataset_sha = sha256_data(data)
    manifest = {"schema_version": 1, "dataset_sha256": dataset_sha, "config": asdict(config),
                "reference_sha256": reference_sha, "scorer_sha256": data.get("scorer_sha256"),
                "price_sha256": data.get("price_sha256"), "source": data.get("source"),
                "trainer_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    _write_json(destination / "FROZEN-MANIFEST.json", manifest)
    journal.append("reference_frozen", manifest)
    report = {"schema_version": 1, "status": "rejected", "validation_kind": "unexecuted",
              "dataset_sha256": dataset_sha, "reference_sha256": reference_sha,
              "candidate_saved": False, "steps_completed": 0, "training_readiness": "unproven",
              "heldout_accessed": False, "gameplay_improvement": "unmeasured", "config": asdict(config)}

    def finish(reason, status="rejected"):
        report["status"], report["reason"] = status, reason
        # Reference is loaded back and compared even when a candidate is rejected.
        report["reference_preserved"] = TinyOptionPolicy.load(destination / "reference.npz").parameter_sha256() == reference_sha
        receipt = journal.append("training_receipt", {
            "status": status, "reason": reason, "steps_completed": report["steps_completed"],
            "candidate_saved": report["candidate_saved"], "reference_sha256": reference_sha,
            "reference_preserved": report["reference_preserved"], "heldout_accessed": False})
        report["receipt_sha256"] = receipt["sha256"]
        report["journal_tip_sha256"] = journal.verify()["tip_sha256"]
        _write_json(destination / "TRAINING-REPORT.json", report)
        _write_json(destination / "UPDATE-RECEIPT.json", receipt)
        return report

    if config.stop_file and Path(config.stop_file).exists():
        return finish("stop_file_present")
    try:
        groups, anchors, validation, hygiene = _prepare_data(data, reference, config)
    except (ValueError, TypeError, KeyError) as exc:
        return finish(f"eligibility: {exc}")
    report["hygiene"] = hygiene
    report["validation_kind"] = "real_game_cpu_training"
    report["step0_alignment"] = {"passed": True, "tolerance": config.alignment_tolerance,
                                 "reference_origins": sorted({row.reference_origin for row in groups + anchors["A1"] + anchors["A2"] + validation}),
                                 "scope": "same_cpu_implementation", "external_broker_calibrated": False}
    candidate = reference.clone()
    initial = _metrics(candidate, groups, anchors, validation)
    report["before"] = initial
    journal.append("eligibility_and_step0", {"hygiene": hygiene, "alignment": report["step0_alignment"], "metrics": initial})
    beta_fork, beta_anchor = config.beta_fork, config.beta_anchor
    for step in range(1, config.steps + 1):
        if config.stop_file and Path(config.stop_file).exists():
            return finish("stop_file_present")
        loss, gradients = _combined_loss(candidate, groups, anchors, beta_fork, beta_anchor)
        try:
            gradient_norm = candidate.update(gradients, config.learning_rate, config.max_grad_norm)
            metrics = _metrics(candidate, groups, anchors, validation)
        except ValueError as exc:
            return finish(f"nonfinite_update: {exc}")
        report["steps_completed"] = step
        report["after"] = metrics
        failures = _gate_failures(metrics, config)
        journal.append("update", {"step": step, "loss": loss, "gradient_norm": gradient_norm,
                                  "learning_rate": config.learning_rate, "beta_fork": beta_fork,
                                  "beta_anchor": beta_anchor, "metrics": metrics,
                                  "parameter_sha256": candidate.parameter_sha256(), "gate_failures": failures})
        if failures:
            report["gate_failures"] = failures
            return finish("hard_or_entropy_gate")
        # Follow the source experiment: fork beta only rises; anchor beta
        # adapts in both directions using the combined A1+A2 anchor mean.
        if metrics["validation_forks"]["kl"] > config.target_fork_kl:
            beta_fork = min(config.beta_max, max(config.beta_min, beta_fork * config.beta_factor))
        if metrics["anchors"]["kl"] > config.target_anchor_kl:
            beta_anchor = min(config.beta_max, max(config.beta_min, beta_anchor * config.beta_factor))
        elif metrics["anchors"]["kl"] < config.target_anchor_kl / 2:
            beta_anchor = max(config.beta_min, beta_anchor / config.beta_factor)
    final = report["after"]
    delta = final["train_forks"]["expected_advantage"] - initial["train_forks"]["expected_advantage"]
    report["learncheck"] = {"passed": delta > config.learncheck_min_delta,
                            "train_expected_advantage_delta": delta, "minimum_delta": config.learncheck_min_delta,
                            "scope": "training_direction_only", "heldout_policy_improvement": "unmeasured"}
    if not report["learncheck"]["passed"]:
        return finish("learncheck_no_training_direction_signal")
    report["candidate_sha256"] = candidate.save(destination / "candidate.npz")
    report["candidate_saved"] = True
    report["final_betas"] = {"fork": beta_fork, "combined_A1_A2": beta_anchor}
    return finish("verified_native_rows_and_cpu_update_passed", "accepted_cpu_smoke")
