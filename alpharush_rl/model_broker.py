"""Complete legal-option language-model distribution protocol and HTTP client.

The GPU worker is separate from the game process. No other-project adapter is loaded.
``prompt_sha256`` hashes the exact UTF-8 user prompt for the existing RL contract.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import urllib.request


GPU_MODELS = {
    "8b": {
        "repo": "mistralai/Ministral-3-8B-Instruct-2512-BF16",
        "revision": "f6fae9795746f63c9be8344932f01275f3c63734",
        "path": "/home/<user>/option_brain_20260922/base-ministral8b/weights",
        "gpu_uuid": "GPU-750ffce3-8683-57c4-8737-35bd45ad462f",
        "gpu_name": "NVIDIA GeForce RTX 5080",
        "port": 12081,
    },
    "24b": {
        "repo": "mistralai/Mistral-Small-3.2-24B-Instruct-2506",
        "revision": "95a6d26c4bfb886c58daf9d3f7332c857cb27b43",
        "path": "/home/<user>/strategy_brain_sft_20260922/base-mistral24/weights",
        "gpu_uuid": "GPU-4d9f95ae-dfba-0ba5-9aad-09372fa19208",
        "gpu_name": "NVIDIA GeForce RTX 5090",
        "port": 12082,
    },
}


def validate_request(request: dict) -> dict:
    if not isinstance(request, dict):
        raise ValueError("request must be an object")
    if not isinstance(request.get("id"), str) or not request["id"]:
        raise ValueError("nonempty request id required")
    for name in ("system", "user"):
        if not isinstance(request.get(name), str):
            raise ValueError(f"{name} must be text")
    labels = request.get("labels")
    if not isinstance(labels, list) or not 1 <= len(labels) <= 256:
        raise ValueError("one to 256 legal labels required")
    if any(not isinstance(label, str) or not 1 <= len(label) <= 8
           or any(letter not in "ABCDEFGHIJKLMNOPQRSTUVWXYZ" for letter in label) for label in labels):
        raise ValueError("legal labels must be uppercase ASCII letters")
    if len(set(labels)) != len(labels):
        raise ValueError("duplicate legal label")
    return request


def prompt_hash(request: dict) -> str:
    return hashlib.sha256(request["user"].encode("utf-8")).hexdigest()


def verify_model_manifest(model_key: str, manifest_path: str | Path | None = None) -> dict:
    """Cheap launch check against a previously completed full SHA256 audit.

    Source size/mtime changes invalidate the receipt and require a new hash
    audit. The expensive shard scan is never performed for every request.
    """
    spec = GPU_MODELS[model_key]
    path = Path(manifest_path or f"/home/<user>/alpharush/manifests/model-{model_key}.json")
    receipt = json.loads(path.read_text(encoding="utf-8"))
    content = receipt["content"]
    digest = hashlib.sha256(json.dumps(content, sort_keys=True, ensure_ascii=False,
                                       separators=(",", ":")).encode("utf-8")).hexdigest()
    if receipt.get("manifest_sha256") != digest:
        raise ValueError("base manifest SHA256 mismatch")
    if content.get("repo") != spec["repo"] or content.get("revision") != spec["revision"]:
        raise ValueError("base model identity mismatch")
    base = Path(spec["path"]).resolve()
    if Path(content["weights_directory"]).resolve() != base:
        raise ValueError("base weights directory mismatch")
    files = content["files"]
    index = json.loads((base / "model.safetensors.index.json").read_text(encoding="utf-8"))
    required = set(index["weight_map"].values()) | {"config.json", "model.safetensors.index.json", "tekken.json"}
    if not required <= set(files):
        raise ValueError("base manifest does not cover complete weights/tokenizer")
    for name, record in files.items():
        source = (base / name).resolve()
        if not source.is_relative_to(base):
            raise ValueError("base manifest path leaves weights directory")
        stat = source.stat()
        if stat.st_size != record["bytes"] or stat.st_mtime_ns != record["mtime_ns"]:
            raise ValueError(f"base file changed since SHA256 audit: {name}")
    return receipt


def validate_distribution(response: dict, request: dict) -> dict:
    validate_request(request)
    if response.get("id") != request["id"] or response.get("labels") != request["labels"]:
        raise ValueError("response request identity or label order mismatch")
    if response.get("choice") not in request["labels"]:
        raise ValueError("illegal response choice")
    if response.get("prompt_sha256") != prompt_hash(request):
        raise ValueError("prompt byte hash mismatch")
    probabilities, log_probabilities = response.get("p"), response.get("logp")
    if not isinstance(probabilities, list) or not isinstance(log_probabilities, list):
        raise ValueError("complete p and logp arrays required")
    if len(probabilities) != len(request["labels"]) or len(log_probabilities) != len(probabilities):
        raise ValueError("complete distribution required; truncated top-N is invalid")
    for p, logp in zip(probabilities, log_probabilities):
        if isinstance(p, bool) or isinstance(logp, bool) or not math.isfinite(p) or not math.isfinite(logp):
            raise ValueError("finite probabilities required")
        if p <= 0 or p > 1 or not math.isclose(math.log(p), logp, abs_tol=1e-8):
            raise ValueError("inconsistent p/logp")
    if not math.isclose(sum(probabilities), 1.0, abs_tol=1e-8):
        raise ValueError("probabilities do not sum to one")
    if not isinstance(response.get("model"), str) or response.get("complete_legal_distribution") is not True:
        raise ValueError("model identity and complete-distribution flag required")
    return response


class LanguageModelBroker:
    """Loopback-only client; caller determines its native prompt and legal menu."""
    def __init__(self, endpoint: str, timeout: float = 180.0):
        from urllib.parse import urlsplit
        parsed = urlsplit(endpoint)
        if parsed.scheme != "http" or parsed.hostname not in ("127.0.0.1", "localhost", "::1"):
            raise ValueError("broker endpoint must use local loopback HTTP")
        self.endpoint, self.timeout = endpoint.rstrip("/"), timeout

    def distribution(self, request: dict) -> dict:
        validate_request(request)
        payload = json.dumps(request, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        req = urllib.request.Request(self.endpoint + "/distribution", data=payload,
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=self.timeout) as stream:
            response = json.load(stream)
        return validate_distribution(response, request)


class RetryingBroker:
    """Retries transport failures of a deterministic scoring server: the same request to the same model, so a
    retry can only reproduce the answer (never a rule fallback). Invalid answers still raise at once."""
    DELAYS = (5, 15, 30, 60, 120, 240)

    def __init__(self, broker, delays=DELAYS, on_retry=None, sleep=None):
        import time
        self.broker, self.delays, self.on_retry = broker, tuple(delays), on_retry
        self.sleep = time.sleep if sleep is None else sleep

    def distribution(self, request: dict) -> dict:
        import http.client
        import urllib.error
        transient = (urllib.error.URLError, http.client.HTTPException, ConnectionError, TimeoutError)
        for attempt, delay in enumerate((*self.delays, None)):
            try:
                return self.broker.distribution(request)
            except transient as exc:
                if delay is None:
                    raise
                if self.on_retry is not None:
                    self.on_retry(request.get("id"), attempt + 1, exc)
                self.sleep(delay)
        raise AssertionError("unreachable")


class LanguageModelPolicy:
    """Use the same native menu and prompt as the environment's existing policy.

    Broker errors propagate. They cannot silently turn a rule fallback into
    credit for the language model.
    """
    def __init__(self, endpoint: str, request_prefix: str = "kr-native", timeout: float = 180.0):
        self.broker = LanguageModelBroker(endpoint, timeout)
        self.request_prefix, self.counter = request_prefix, 0

    def choose(self, state: dict, menu: list[dict], rng=None) -> dict:
        import copy
        from .menus import decision_prompt, validate_menu
        self.counter += 1
        labels = validate_menu(menu)
        request = {"id": f"{self.request_prefix}-{self.counter}",
                   "system": "You control Kingdom Rush. Choose exactly one option label from the legal menu. "
                             "Optimize native level victory and remaining lives. Reply with the option label only.",
                   "user": decision_prompt(state, menu), "labels": labels}
        response = self.broker.distribution(request)
        index = labels.index(response["choice"])
        return {"choice": response["choice"], "action": copy.deepcopy(menu[index]["action"]),
                "confidence": response["p"][index], "provenance": "model", "fallback": False,
                "distribution": response}


def tokenize_request(tokenizer, request: dict) -> dict:
    """Support both MistralCommonBackend and TokenizersBackend chat outputs."""
    validate_request(request)
    result = tokenizer.apply_chat_template(
        [{"role": "system", "content": request["system"]},
         {"role": "user", "content": request["user"]}],
        tokenize=True, add_generation_prompt=True,
    )
    ids = result["input_ids"] if isinstance(result, dict) or hasattr(result, "keys") else result
    if hasattr(ids, "tolist"):
        ids = ids.tolist()
    if ids and isinstance(ids[0], list):
        if len(ids) != 1:
            raise ValueError("one prompt per request required")
        ids = ids[0]
    eos = tokenizer.eos_token_id
    if isinstance(eos, list):
        eos = eos[0]
    if not isinstance(eos, int):
        raise ValueError("instruction tokenizer requires EOS")
    suffixes = [tokenizer.encode(label, add_special_tokens=False) + [eos] for label in request["labels"]]
    if any(len(suffix) < 2 for suffix in suffixes):
        raise ValueError("empty label encoding")
    return {"input_ids": ids, "suffixes": suffixes, "prompt_tokens": len(ids),
            "label_token_ids": dict(zip(request["labels"], suffixes)),
            "prompt_sha256": prompt_hash(request), "tokenizer": type(tokenizer).__name__,
            "tokenized_prompt_ids_sha256": hashlib.sha256(
                json.dumps(ids, separators=(",", ":")).encode("utf-8")).hexdigest()}

