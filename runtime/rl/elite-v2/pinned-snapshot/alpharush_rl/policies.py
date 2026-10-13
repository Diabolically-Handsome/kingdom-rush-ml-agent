"""CPU option scorer and a transport-free contract for a future model broker."""
from __future__ import annotations

import copy
import hashlib
import math
from pathlib import Path

import numpy as np

from .journal import canonical_bytes, canonical_json, sha256_data
from .menus import decision_prompt, prompt_sha256, validate_menu

FEATURE_VERSION = "native-option-features-v1"
FEATURE_DIM = 24


def _number(value, default=0.0):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        return float(default)
    return float(value)


def option_features(state: dict, menu: list[dict]) -> np.ndarray:
    """Deterministic, documented native-state features; no hidden engine edits."""
    validate_menu(menu)
    holders = {h.get("id"): h for h in state.get("holders", [])}
    towers = {t.get("id"): t for t in state.get("towers", [])}
    enemy_hp = sum(_number(e.get("hp")) for e in state.get("enemies", []))
    gold = _number(state.get("gold"))
    rows = []
    for option in menu:
        action = option["action"]
        name = action["action"]
        entity = holders.get(action.get("holder_id"), towers.get(action.get("tower_id"), {}))
        family = action.get("tower_type", entity.get("type", ""))
        if not family:
            target = str(action.get("target", ""))
            family = next((kind for kind in ("archer", "barrack", "mage", "engineer") if kind in target), "")
        cost = _number(option.get("cost"))
        rows.append([
            1.0,
            min(gold / 1000.0, 4.0),
            _number(state.get("lives")) / 20.0,
            _number(state.get("wave")) / 20.0,
            _number(state.get("wave_total")) / 20.0,
            min(len(state.get("enemies", [])) / 30.0, 4.0),
            min(enemy_hp / 3000.0, 4.0),
            len(towers) / 20.0,
            len(holders) / 20.0,
            float(state.get("wave_ready") is True),
            float(name == "wait"), float(name == "build_tower"),
            float(name == "upgrade_tower"), float(name == "send_wave"),
            float(family == "archer"), float(family == "barrack"),
            float(family == "mage"), float(family == "engineer"),
            cost / 400.0, (gold - cost) / 1000.0,
            _number(entity.get("x")) / 1024.0, _number(entity.get("y")) / 768.0,
            min(_number(entity.get("path_score")) / 100.0, 4.0),
            _number(entity.get("level")) / 4.0,
        ])
    return np.asarray(rows, dtype=np.float64)


def masked_softmax(logits: np.ndarray, legal_mask: np.ndarray | None = None) -> np.ndarray:
    logits = np.asarray(logits, dtype=np.float64)
    if logits.ndim != 1 or not np.all(np.isfinite(logits)):
        raise ValueError("finite one-dimensional logits required")
    mask = np.ones(logits.shape, dtype=bool) if legal_mask is None else np.asarray(legal_mask, dtype=bool)
    if mask.shape != logits.shape or not mask.any():
        raise ValueError("legal mask must match logits and include an option")
    result = np.zeros(logits.shape, dtype=np.float64)
    shifted = logits[mask] - logits[mask].max()
    weights = np.exp(shifted)
    result[mask] = weights / weights.sum()
    return result


class TinyOptionPolicy:
    """Small shared tanh scorer. This validates the RL plumbing on CPU only.

    Each legal option receives a score from its state/action feature vector.
    It is deliberately independent of the later language-model architecture.
    """
    def __init__(self, seed: int = 0, hidden_size: int = 16):
        if hidden_size < 1:
            raise ValueError("hidden_size must be positive")
        rng = np.random.default_rng(seed)
        self.hidden_size = int(hidden_size)
        self.params = {
            "W1": rng.normal(0, 0.08, (FEATURE_DIM, self.hidden_size)).astype(np.float64),
            "b1": np.zeros(self.hidden_size, dtype=np.float64),
            "W2": rng.normal(0, 0.08, self.hidden_size).astype(np.float64),
        }

    def clone(self) -> "TinyOptionPolicy":
        result = TinyOptionPolicy(hidden_size=self.hidden_size)
        result.params = {name: value.copy() for name, value in self.params.items()}
        return result

    def parameter_sha256(self) -> str:
        return sha256_data({"feature_version": FEATURE_VERSION, "hidden_size": self.hidden_size,
                            "parameters": {key: value.tolist() for key, value in self.params.items()}})

    @property
    def model_id(self) -> str:
        return f"cpu-tiny-tanh:{self.parameter_sha256()}"

    def forward(self, features: np.ndarray, legal_mask: np.ndarray | None = None):
        features = np.asarray(features, dtype=np.float64)
        if features.ndim != 2 or features.shape[1] != FEATURE_DIM or not np.all(np.isfinite(features)):
            raise ValueError("invalid feature matrix")
        hidden = np.tanh(features @ self.params["W1"] + self.params["b1"])
        logits = hidden @ self.params["W2"]
        probabilities = masked_softmax(logits, legal_mask)
        return probabilities, {"features": features, "hidden": hidden, "p": probabilities}

    def backward(self, cache: dict, dlogits: np.ndarray) -> dict[str, np.ndarray]:
        dlogits = np.asarray(dlogits, dtype=np.float64)
        if dlogits.shape != cache["p"].shape or not np.all(np.isfinite(dlogits)):
            raise ValueError("invalid score gradient")
        hidden = cache["hidden"]
        dpre = (dlogits[:, None] * self.params["W2"][None, :]) * (1.0 - hidden ** 2)
        return {"W1": cache["features"].T @ dpre,
                "b1": dpre.sum(axis=0), "W2": hidden.T @ dlogits}

    def distribution(self, state: dict, menu: list[dict]) -> dict:
        labels = validate_menu(menu)
        p, _ = self.forward(option_features(state, menu))
        return {"labels": labels, "p": p.tolist(), "logp": np.log(p).tolist(),
                "prompt_sha256": prompt_sha256(state, menu), "model": self.model_id,
                "feature_version": FEATURE_VERSION, "complete_legal_distribution": True}

    def choose(self, state: dict, menu: list[dict], rng: np.random.Generator | None = None) -> dict:
        distribution = self.distribution(state, menu)
        p = np.asarray(distribution["p"])
        index = int(p.argmax()) if rng is None else int(rng.choice(len(menu), p=p))
        return {"choice": distribution["labels"][index], "action": copy.deepcopy(menu[index]["action"]),
                "confidence": float(p[index]), "provenance": "model", "distribution": distribution,
                "fallback": False}

    def update(self, gradients: dict, learning_rate: float, max_grad_norm: float = 1.0) -> float:
        if learning_rate <= 0 or not math.isfinite(learning_rate):
            raise ValueError("learning_rate must be finite and positive")
        norm = math.sqrt(sum(float(np.square(gradients[key]).sum()) for key in self.params))
        if not math.isfinite(norm):
            raise ValueError("nonfinite parameter gradient")
        scale = min(1.0, max_grad_norm / max(norm, 1e-30))
        for key in self.params:
            if gradients[key].shape != self.params[key].shape:
                raise ValueError("parameter gradient shape mismatch")
            self.params[key] -= learning_rate * scale * gradients[key]
        return norm

    def save(self, path: str | Path) -> str:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        # File handle avoids silently appending .npz to the requested path.
        with path.open("wb") as stream:
            np.savez(stream, **self.params, hidden_size=np.asarray(self.hidden_size),
                     feature_version=np.asarray(FEATURE_VERSION))
        return self.parameter_sha256()

    @classmethod
    def load(cls, path: str | Path) -> "TinyOptionPolicy":
        with np.load(Path(path), allow_pickle=False) as data:
            if str(data["feature_version"]) != FEATURE_VERSION:
                raise ValueError("policy feature-version mismatch")
            result = cls(hidden_size=int(data["hidden_size"]))
            for key in result.params:
                array = np.asarray(data[key], dtype=np.float64)
                if array.shape != result.params[key].shape or not np.all(np.isfinite(array)):
                    raise ValueError("invalid policy parameters")
                result.params[key] = array.copy()
        return result


class RulesPolicy:
    """Conservative baseline, explicitly attributed to scripted decisions."""
    def choose(self, state: dict, menu: list[dict], rng=None) -> dict:
        labels = validate_menu(menu)
        investments = [option for option in menu if option["action"]["action"] in ("build_tower", "upgrade_tower")]
        if investments:
            choice = min(investments, key=lambda item: (float(item.get("cost", 0)), item["label"]))
        else:
            choice = next((option for option in menu if option["action"]["action"] == "send_wave"), menu[0])
        return {"choice": choice["label"], "action": copy.deepcopy(choice["action"]),
                "confidence": None, "provenance": "rules", "distribution": None, "fallback": False}


def resolve_choice(choice: str, state: dict, menu: list[dict], fallback=None) -> dict:
    """Bad broker responses remain visible and do not become model credit."""
    labels = validate_menu(menu)
    if choice in labels:
        option = menu[labels.index(choice)]
        return {"choice": choice, "action": copy.deepcopy(option["action"]),
                "provenance": "model", "fallback": False}
    decision = (fallback or RulesPolicy()).choose(state, menu)
    decision.update({"provenance": "rules_fallback", "fallback": True,
                     "invalid_model_choice": choice, "distribution": None})
    return decision


class BrokerAdapter:
    """Contract only; this class intentionally performs no HTTP requests."""
    def __init__(self, endpoint: str | None = None):
        self.endpoint = endpoint

    def request(self, request_id: str, state: dict, menu: list[dict]) -> dict:
        return {"id": request_id, "system": "Choose exactly one legal option label.",
                "user": decision_prompt(state, menu), "labels": validate_menu(menu)}

    def validate_response(self, response: dict, request: dict, *, require_distribution: bool = True) -> dict:
        if response.get("id") != request["id"] or response.get("choice") not in request["labels"]:
            raise ValueError("broker request ID or legal choice mismatch")
        if require_distribution:
            distribution = response.get("distribution", response)
            labels = distribution.get("labels")
            if labels != request["labels"]:
                raise ValueError("broker must return all legal labels in their original order")
            p = np.asarray(distribution.get("p", []), dtype=np.float64)
            logp = np.asarray(distribution.get("logp", []), dtype=np.float64)
            if p.shape != (len(labels),) or logp.shape != p.shape or not np.all(np.isfinite(logp)):
                raise ValueError("complete p/logp required; top-N probabilities are insufficient")
            if np.any(p <= 0) or not np.isclose(p.sum(), 1.0, atol=1e-8, rtol=0) or not np.allclose(np.log(p), logp, atol=1e-8, rtol=0):
                raise ValueError("invalid complete broker distribution")
            expected = hashlib.sha256(request["user"].encode("utf-8")).hexdigest()
            if distribution.get("prompt_sha256") != expected:
                raise ValueError("broker prompt byte hash mismatch")
            if not isinstance(distribution.get("model"), str):
                raise ValueError("broker model identity required")
        return response
