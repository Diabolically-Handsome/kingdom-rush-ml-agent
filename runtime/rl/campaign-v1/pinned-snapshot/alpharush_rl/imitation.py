"""CPU SFT and explicit DAgger relabel queues, isolated from native branch RL."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from .journal import Journal, canonical_bytes, sha256_data
from .menus import validate_menu
from .policies import TinyOptionPolicy
from .pools import HygieneError, PoolRegistry
from .trainer import EligibilityError, _prepare_row, distribution_metrics, validate_price_contract

EXPERT_SOURCES = frozenset(("rules", "human"))


def cross_entropy_loss_and_grad(policy: TinyOptionPolicy, features: np.ndarray,
                                label_index: int, reference_p: np.ndarray | None = None,
                                beta: float = 0.0):
    """Full-legal supervised cross entropy plus optional KL(pi || pi0)."""
    p, cache = policy.forward(features)
    if isinstance(label_index, bool) or not isinstance(label_index, (int, np.integer)) or not 0 <= label_index < len(p):
        raise ValueError("expert label index must be a legal option")
    if not math.isfinite(beta) or beta < 0:
        raise ValueError("invalid reference KL beta")
    gradient = p.copy()
    gradient[label_index] -= 1.0
    cross_entropy = float(-np.log(p[label_index]))
    metrics = {"cross_entropy": cross_entropy, "label_probability": float(p[label_index]),
               "label_correct": bool(p.argmax() == label_index), "kl": 0.0,
               "entropy": float(-np.dot(p, np.log(p)))}
    if reference_p is not None:
        reference_metrics = distribution_metrics(p, np.asarray(reference_p))
        metrics.update(reference_metrics)
        dprob = np.log(p) - np.log(reference_p) + 1.0
        gradient += beta * p * (dprob - np.dot(p, dprob))
    loss = cross_entropy + beta * metrics["kl"]
    return loss, policy.backward(cache, gradient), metrics


@dataclass
class ImitationConfig:
    seed: int = 0
    hidden_size: int = 16
    steps: int = 40
    learning_rate: float = 0.03
    max_grad_norm: float = 1.0
    beta: float = 0.03
    target_kl: float = 0.01
    hard_mean_kl: float = 0.05
    entropy_reference_fraction: float = 0.5
    alignment_tolerance: float = 1e-10
    minimum_ce_improvement: float = 1e-6
    reference_path: str | None = None
    stop_file: str | None = None

    def validate(self):
        if isinstance(self.steps, bool) or not isinstance(self.steps, int) or not 1 <= self.steps <= 1000:
            raise ValueError("CPU SFT validation requires 1..1000 steps")
        if isinstance(self.hidden_size, bool) or not isinstance(self.hidden_size, int) or not 1 <= self.hidden_size <= 256:
            raise ValueError("hidden_size requires 1..256")
        for name in ("learning_rate", "max_grad_norm", "target_kl", "hard_mean_kl", "alignment_tolerance"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"invalid {name}")
        if not math.isfinite(self.beta) or self.beta < 0 or not math.isfinite(self.minimum_ce_improvement) or self.minimum_ce_improvement < 0:
            raise ValueError("invalid beta or learncheck threshold")
        if not 0.5 <= self.entropy_reference_fraction <= 1:
            raise ValueError("entropy floor must preserve at least half of reference")


def _validate_examples(data: dict, reference: TinyOptionPolicy, config: ImitationConfig):
    if data.get("source") != "real_game":
        raise EligibilityError("SFT accepts real_game observations only")
    registry = PoolRegistry(data["pool_registry"])
    if any(data.get(name) for name in ("heldout", "heldout_forks", "heldout_groups", "validation_examples")):
        raise HygieneError("SFT gradients cannot receive validation or heldout examples")
    examples = data.get("examples", [])
    if not isinstance(examples, list) or not examples:
        raise EligibilityError("verified explicitly labeled SFT examples required")
    prices = data.get("price_contract")
    if not isinstance(prices, dict) or data.get("price_sha256") != sha256_data(prices):
        raise EligibilityError("frozen native price contract/SHA required")
    prepared = []
    for example in examples:
        registry.check_row(example, "train")
        if example.get("receipt_verified") is not True:
            raise EligibilityError("native observation action receipt required")
        if example.get("source", data["source"]) != "real_game":
            raise EligibilityError("synthetic row in native SFT dataset")
        if example.get("expert_source") not in EXPERT_SOURCES:
            raise EligibilityError("explicit rules/human label provenance required; model choice is not an expert label")
        labels = validate_menu(example.get("menu"))
        if example.get("label") not in labels:
            raise EligibilityError("explicit legal expert label required; unlabeled DAgger queue cannot train")
        validate_price_contract(example["menu"], prices)
        row = _prepare_row(example, reference, config.alignment_tolerance)
        prepared.append((row, labels.index(example["label"]), example["expert_source"]))
    return prepared, registry.audit()


def _sft_metrics(policy, prepared):
    items = [cross_entropy_loss_and_grad(policy, row.features, index, row.reference_p)[2]
             for row, index, _ in prepared]
    return {name: float(np.mean([item[name] for item in items])) for name in items[0]}


def train_imitation(data: dict | str | Path, config: ImitationConfig | dict | None,
                    outputdir: str | Path) -> dict:
    """Train the small CPU scorer without touching heldout or future LLM assets."""
    if isinstance(data, (str, Path)):
        data = json.loads(Path(data).read_text(encoding="utf-8"))
    config = config if isinstance(config, ImitationConfig) else ImitationConfig(**(config or {}))
    config.validate()
    destination = Path(outputdir)
    destination.mkdir(parents=True, exist_ok=True)
    if any((destination / name).exists() for name in ("reference.npz", "candidate.npz", "updates.jsonl", "SFT-REPORT.json")):
        raise FileExistsError("SFT experiment requires a fresh output directory")
    reference = TinyOptionPolicy.load(config.reference_path) if config.reference_path else TinyOptionPolicy(config.seed, config.hidden_size)
    reference_sha = reference.save(destination / "reference.npz")
    journal = Journal(destination / "updates.jsonl")
    frozen = {"schema_version": 1, "dataset_sha256": sha256_data(data), "reference_sha256": reference_sha,
              "config": asdict(config), "price_sha256": data.get("price_sha256"),
              "imitation_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "training_kind": "cpu_supervised_imitation", "llm_adapter_training": "not_executed_by_cpu_trainer"}
    (destination / "FROZEN-MANIFEST.json").write_bytes(canonical_bytes(frozen) + b"\n")
    journal.append("reference_frozen", frozen)
    report = {"schema_version": 1, "status": "rejected", "training_kind": "unexecuted",
              "reference_sha256": reference_sha, "steps_completed": 0, "candidate_saved": False,
              "heldout_accessed": False, "gameplay_improvement": "unmeasured",
              "llm_adapter_training": "not_executed_by_cpu_trainer", "config": asdict(config)}

    def finish(reason, status="rejected"):
        report.update({"status": status, "reason": reason,
                       "reference_preserved": TinyOptionPolicy.load(destination / "reference.npz").parameter_sha256() == reference_sha})
        receipt = journal.append("sft_receipt", {name: report[name] for name in
                                 ("status", "reason", "reference_sha256", "reference_preserved", "steps_completed", "candidate_saved", "heldout_accessed")})
        report["receipt_sha256"] = receipt["sha256"]
        (destination / "SFT-REPORT.json").write_bytes(canonical_bytes(report) + b"\n")
        (destination / "UPDATE-RECEIPT.json").write_bytes(canonical_bytes(receipt) + b"\n")
        return report

    if config.stop_file and Path(config.stop_file).exists():
        return finish("stop_file_present")
    try:
        prepared, hygiene = _validate_examples(data, reference, config)
    except (ValueError, TypeError, KeyError) as exc:
        return finish(f"eligibility: {exc}")
    report.update({"training_kind": "real_game_cpu_supervised_imitation", "hygiene": hygiene,
                   "label_provenance_counts": {source: sum(item[2] == source for item in prepared) for source in sorted(EXPERT_SOURCES)},
                   "step0_alignment": {"passed": True, "scope": "same_cpu_implementation"}})
    candidate, beta = reference.clone(), config.beta
    report["before"] = _sft_metrics(candidate, prepared)
    for step in range(1, config.steps + 1):
        if config.stop_file and Path(config.stop_file).exists():
            return finish("stop_file_present")
        gradients = {name: np.zeros_like(value) for name, value in candidate.params.items()}
        loss = 0.0
        for row, index, _ in prepared:
            value, derivative, _ = cross_entropy_loss_and_grad(candidate, row.features, index, row.reference_p, beta)
            loss += value / len(prepared)
            for name in gradients:
                gradients[name] += derivative[name] / len(prepared)
        try:
            gradient_norm = candidate.update(gradients, config.learning_rate, config.max_grad_norm)
            metrics = _sft_metrics(candidate, prepared)
        except ValueError as exc:
            return finish(f"nonfinite_update: {exc}")
        report["steps_completed"], report["after"] = step, metrics
        failures = []
        if metrics["kl"] > config.hard_mean_kl:
            failures.append("hard_mean_kl")
        if metrics["entropy"] < config.entropy_reference_fraction * metrics["reference_entropy"]:
            failures.append("entropy_floor")
        journal.append("sft_update", {"step": step, "loss": loss, "gradient_norm": gradient_norm,
                                     "beta": beta, "metrics": metrics, "gate_failures": failures,
                                     "parameter_sha256": candidate.parameter_sha256()})
        if failures:
            report["gate_failures"] = failures
            return finish("hard_or_entropy_gate")
        if metrics["kl"] > config.target_kl:
            beta = min(10.0, max(1e-5, beta * 1.5))
    improvement = report["before"]["cross_entropy"] - report["after"]["cross_entropy"]
    report["learncheck"] = {"passed": improvement > config.minimum_ce_improvement,
                            "train_cross_entropy_improvement": improvement,
                            "scope": "imitation_labels_only", "gameplay_improvement": "unmeasured"}
    if not report["learncheck"]["passed"]:
        return finish("learncheck_no_label_signal")
    report["candidate_sha256"] = candidate.save(destination / "candidate.npz")
    report["candidate_saved"] = True
    return finish("native_observation_cpu_sft_update_passed", "accepted_cpu_imitation_smoke")


def export_dagger_relabel(decisions: list[dict], registry: PoolRegistry | dict,
                          output_path: str | Path | None = None,
                          *, price_contract: dict | None = None) -> dict:
    """Export visited model states for an explicit oracle, keeping labels blank."""
    registry = registry if isinstance(registry, PoolRegistry) else PoolRegistry(registry)
    queue = []
    ids = set()
    for decision in decisions:
        registry.check_row(decision, "train")
        if decision.get("source") != "real_game" or decision.get("receipt_verified") is not True:
            raise EligibilityError("DAgger queue requires verified real game observations")
        if decision.get("provenance") != "model":
            raise EligibilityError("DAgger collection requires a model-visited state; rules/fallback remain separately attributed")
        labels = validate_menu(decision.get("menu"))
        if decision.get("choice") not in labels:
            raise EligibilityError("recorded model choice must be legal")
        decision_id = decision.get("decision_id")
        if not isinstance(decision_id, str) or not decision_id or decision_id in ids:
            raise EligibilityError("DAgger decision IDs must be distinct")
        ids.add(decision_id)
        queue.append({"decision_id": decision_id, "source": "real_game", "pool": "train",
                      "level": decision["level"], "seed": decision["seed"],
                      "state": decision["state"], "menu": decision["menu"],
                      "executed_model_label": decision["choice"], "receipt_verified": True,
                      "label": None, "expert_source": None, "status": "awaiting_explicit_relabel"})
    result = {"schema_version": 1, "kind": "dagger_relabel_queue", "source": "real_game",
              "pool_registry": registry.manifest, "examples": queue, "unlabeled": len(queue),
              "training_eligible": False, "heldout_accessed": False}
    if price_contract is not None:
        for row in queue:
            validate_price_contract(row["menu"], price_contract)
        result.update({"price_contract": price_contract, "price_sha256": sha256_data(price_contract)})
    if output_path is not None:
        target = Path(output_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("xb") as stream:
            stream.write(canonical_bytes(result) + b"\n")
    return result


def merge_dagger_labels(queue: dict, relabels: list[dict]) -> dict:
    """Materialize explicitly labeled rows; no original choice becomes a label."""
    if queue.get("kind") != "dagger_relabel_queue":
        raise EligibilityError("expected explicit DAgger relabel queue")
    known = {row["decision_id"]: row for row in queue.get("examples", [])}
    labeled, seen = [], set()
    for relabel in relabels:
        identity = relabel.get("decision_id")
        if identity not in known or identity in seen:
            raise EligibilityError("unknown or duplicate DAgger relabel ID")
        seen.add(identity)
        if relabel.get("expert_source") not in EXPERT_SOURCES:
            raise EligibilityError("relabel must explicitly credit rules or human oracle")
        row = known[identity]
        if relabel.get("label") not in validate_menu(row["menu"]):
            raise EligibilityError("relabel must be legal")
        labeled.append({**row, "label": relabel["label"], "expert_source": relabel["expert_source"],
                        "status": "explicitly_labeled", "label_origin": "dagger_relabel"})
    result = {"schema_version": 1, "source": "real_game", "pool_registry": queue["pool_registry"],
              "examples": labeled, "remaining_unlabeled": len(known) - len(labeled),
              "training_eligible": False, "heldout_accessed": False}
    if "price_contract" in queue and queue.get("price_sha256") == sha256_data(queue["price_contract"]):
        result.update({"price_contract": queue["price_contract"], "price_sha256": queue["price_sha256"],
                       "training_eligible": bool(labeled)})
    else:
        result["missing_training_contract"] = "native_prices"
    return result


train_sft = train_imitation
