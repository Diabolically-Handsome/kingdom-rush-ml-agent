"""Shared primitives for new bounded training phases; lazy model imports."""
from dataclasses import dataclass, replace
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import time

from .journal import canonical_bytes, sha256_data
from .model_broker import tokenize_request, validate_distribution
from .ops import GateRefused, sha256_file


def write_json(path, value, *, atomic=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if atomic:
        temporary = path.with_name(path.name + "." + str(os.getpid()) + ".tmp")
        temporary.write_bytes(canonical_bytes(value) + b"\n")
        os.replace(temporary, path)
    else:
        with path.open("xb") as stream:
            stream.write(canonical_bytes(value) + b"\n")
            stream.flush()
            os.fsync(stream.fileno())


@dataclass
class Permit:
    deadline: float
    outer_deadline: float
    parent_pid: int
    token: str
    lock_fd: int
    global_state: Path
    phase_state: Path
    kind: str = "llm-one-step-smoke"
    gpu_authorization_verified: bool = True
    gpu_lock_held: bool = True
    max_optimizer_steps: int = 1

    def stage(self, seconds):
        return replace(self, deadline=min(self.outer_deadline, time.monotonic() + seconds))

    def check(self):
        if time.monotonic() >= min(self.deadline, self.outer_deadline) or os.getppid() != self.parent_pid:
            raise GateRefused("Bounded training deadline or supervisor permit expired")
        for state in (self.global_state, self.phase_state):
            if any((state / name).exists() for name in ("STOP", "ENGINEERING-STOP")):
                raise GateRefused("Training STOP is present")
        descriptor, file = os.fstat(self.lock_fd), (self.global_state / "model-gpu-wsl.lock").stat()
        if (descriptor.st_dev, descriptor.st_ino) != (file.st_dev, file.st_ino):
            raise GateRefused("Inherited shared GPU lock changed")
        marker = json.loads((self.global_state / "model-gpu.lock").read_text())
        if marker.get("token") != self.token or marker.get("pid") != self.parent_pid:
            raise GateRefused("Supervisor GPU lock marker changed")


def independent_reference(model, tokenizer, requests, identity, ctx, token_cap):
    """Full fresh teacher forcing, independently implemented from trainer."""
    import torch
    device = model.get_input_embeddings().weight.device
    model.eval()
    responses = []
    with model.disable_adapter(), torch.no_grad():
        for request in requests:
            encoded = tokenize_request(tokenizer, request)
            prefix = encoded["input_ids"]
            if not prefix or len(prefix) > token_cap:
                raise GateRefused("Native prompt token cap exceeded; no truncation")
            scores, started = [], time.monotonic()
            for suffix in encoded["suffixes"]:
                ctx.check()
                inputs = torch.tensor([prefix + suffix[:-1]], dtype=torch.long, device=device)
                output = model(input_ids=inputs, use_cache=False, logits_to_keep=len(suffix))
                logits = output.logits[0]
                if logits.shape[0] == inputs.shape[1]:
                    logits = logits[len(prefix) - 1:len(prefix) - 1 + len(suffix)]
                elif logits.shape[0] != len(suffix):
                    raise GateRefused("Independent p0 cannot establish full causal suffix shift")
                targets = torch.tensor(suffix, dtype=torch.long, device=device)
                scores.append(float(logits.float().log_softmax(-1).gather(1, targets[:, None]).sum().cpu()))
                del inputs, output, logits, targets
            logp = torch.tensor(scores, dtype=torch.float64, device=device).log_softmax(0)
            p = logp.exp().tolist()
            best = max(range(len(scores)), key=scores.__getitem__)
            response = {"id": request["id"], "labels": request["labels"], "p": p, "logp": logp.tolist(),
                "choice": request["labels"][best], "prompt_sha256": encoded["prompt_sha256"],
                "prompt_tokens": len(prefix), "prompt_token_ids": prefix,
                "prompt_token_ids_sha256": sha256_data(prefix), "label_token_ids": encoded["label_token_ids"],
                "model": identity["model"], "model_revision": identity["base_revision"], "adapter": None,
                "identity_hashes": identity["identity_hashes"], "model_manifest_sha256": identity["model_manifest_sha256"],
                "gpu_uuid": identity["gpu_uuid"], "scoring": "sum-label-and-eos-log-likelihood",
                "scoring_execution": "canonical_teacher_forced_no_cache_v2", "sequence_log_likelihood": scores,
                "complete_legal_distribution": True, "learning_updates": 0, "seconds": time.monotonic() - started,
                "reference_source": "independent_frozen_base_disable_adapter_full_teacher_forcing",
                "reference_scorer_file_sha256": sha256_file(Path(__file__)),
                "normalization": "float64-log-softmax-on-model-device", "use_cache": False}
            responses.append(validate_distribution(response, request))
    return responses


def full_objective_gate(report, beta_fork, beta_anchor):
    if any(isinstance(beta, bool) or not isinstance(beta, (int, float)) or not math.isfinite(beta) or beta < 0
           for beta in (beta_fork, beta_anchor)):
        return {"passed": False, "improved": False, "reason": "invalid full objective coefficients"}
    def objective(metrics):
        values = (metrics["fork"]["expected_advantage"], metrics["fork"]["kl"], metrics["anchors"]["kl"])
        if any(isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) for value in values):
            raise ValueError("nonfinite or nonnumeric full objective metrics")
        return -values[0] + beta_fork * values[1] + beta_anchor * values[2]
    try:
        before, after = objective(report["before"]), objective(report["after"])
    except (KeyError, TypeError, ValueError) as exc:
        return {"passed": False, "improved": False, "reason": "full metrics unavailable/invalid: " + str(exc)}
    if not math.isfinite(before) or not math.isfinite(after):
        return {"passed": False, "improved": False, "reason": "nonfinite full objective"}
    return {"before": before, "after": after, "delta": after - before,
            "passed": after <= before, "improved": after < before, "tolerance": 0.0,
            "formula": "-mean_fork_Eadv + beta_fork*mean_fork_KL + beta_anchor*mean_combined_A1_A2_KL"}


def supervisor_acceptance(report, *, exit_code, exited, cleanup_error, error, wall_seconds):
    """Scientific worker acceptance remains provisional until its clean exit."""
    return (isinstance(report, dict) and report.get("accepted") is True
            and report.get("status") == "accepted_reward_step"
            and report.get("optimizer_steps") == 1
            and report.get("parameter_update_verified") is True
            and report.get("full_objective_gate", {}).get("passed") is True
            and exit_code == 0 and exited is True and cleanup_error is None and error is None
            and 0 <= wall_seconds <= 1200)


def terminate_owned(process, hard_deadline):
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
                return False, "Own training worker exit unconfirmed"
    except OSError as exc:
        return process.poll() is not None, str(exc)
