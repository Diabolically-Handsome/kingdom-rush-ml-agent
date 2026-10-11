"""Bounded native validation. Counterfactuals replay from fresh native processes.

This protocol covers campaign level 1, Normal difficulty, fixed ticks, base
towers and wave release. It does not certify arbitrary engine serialization,
hidden RNG state, heroes, powers, upgrades, other levels or parallel workers.
"""
from __future__ import annotations

import copy
import json
import time
from pathlib import Path

from .engine import ROOT
from .env import NativeEnv, observation
from .journal import Journal, sha256_data
from .menus import build_menu, decision_prompt, prompt_sha256
from .ops import sha256_file
from .policies import TinyOptionPolicy
from .trainer import score_native_outcome

SCOPE = {
    "game": "kr1-desktop-6.4.46", "level": 1, "difficulty": 2,
    "actions": ["wait", "build_tower", "send_wave"],
    "state": ["native ticks", "gold", "lives", "wave", "towers", "holders", "enemies", "native terminal outcome"],
    "replay": "cold process + fixed seed + exact command plan; compared exported state digests",
    "pending": ["hidden RNG serialization", "hero actions", "powers", "upgrades", "selling", "all-level fidelity", "parallel worker isolation"],
}


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return path


def write_new_json(path, value):
    """Create a JSON file; an existing file is evidence and raises FileExistsError."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    with path.open("x", encoding="utf-8") as stream:
        stream.write(encoded)
    return path


def _published_paths():
    """Global evidence copies; checked before the native run and again at write time."""
    paths = (ROOT / "runtime/rl/native-evidence.json", ROOT / "runtime/model-native-request.json")
    for target in paths:
        if target.exists():
            raise FileExistsError(f"Refusing to overwrite published evidence: {target.relative_to(ROOT).as_posix()}")
    return paths


def _deadline(start):
    if time.monotonic() - start > 42:
        raise RuntimeError("Native validation wall budget exhausted; no fabricated return")
    if any((ROOT / "runtime/rl" / x).exists() for x in ("STOP", "ENGINEERING-STOP")):
        raise RuntimeError("Native validation stopped")


def finish(env, check=lambda: None):
    """Frozen continuation: release clear waves; never buy extra towers."""
    for _ in range(60):
        check()
        if env.terminal():
            return env.terminal()
        if env.state.get("wave_ready"):
            env.act({"action": "send_wave"})
        env.advance(600)
    raise RuntimeError("Native episode timed out; engineering invalidation")


def establish_fork(env):
    initial = copy.deepcopy(env.state)
    env.act({"action": "build_tower", "holder_id": 20, "tower_type": "archer"})
    middle = copy.deepcopy(env.state)
    env.act({"action": "build_tower", "holder_id": 22, "tower_type": "archer"})
    return initial, middle, copy.deepcopy(env.state)


def collect_verified(output_dir=None, publish=False):
    """Create native evidence and a small training dataset; no gradients here.

    Evidence always lands in output_dir. Only publish=True writes the global
    runtime copies, and never over existing ones.
    """
    start = time.monotonic()
    _deadline(start)
    if publish:
        _published_paths()
    output = Path(output_dir or ROOT / "runtime/rl/native-validation")
    output.mkdir(parents=True, exist_ok=False)
    probe = {"scope": SCOPE}
    with NativeEnv() as env:
        before = copy.deepcopy(env.state)
        time.sleep(0.3)
        paused = observation(env.worker.rpc("state"))
        if before != paused:
            raise RuntimeError("Paused exported native state drifted")
        env.advance(60)
        if env.state["tick"] != before["tick"] + 60:
            raise RuntimeError("Fixed native tick mismatch")
        receipts = []
        for holder, kind in ((15, "mage"), (16, "barrack"), (17, "archer")):
            receipts.append(env.act({"action": "build_tower", "holder_id": holder, "tower_type": kind}))
        rejected_before = copy.deepcopy(env.state)
        failures = []
        for action in ({"action": "build_tower", "holder_id": 18, "tower_type": "engineer"},
                       {"action": "eval", "code": "return 1"}, {"action": "use_power", "power_id": "rain"}):
            try:
                env.worker.rpc(**action)
            except RuntimeError as exc:
                failures.append(str(exc))
            else:
                raise RuntimeError("Unauthorized/illegal native action was accepted")
        rejected_after = observation(env.worker.rpc("state"))
        if rejected_before != rejected_after:
            raise RuntimeError("Rejected action mutated native state")
        probe.update(paused_state_sha256=sha256_data(before), pause_verified=True,
                     fixed_step_verified=True, base_tower_receipts=receipts,
                     rejection_verified=True, rejection_errors=failures, trace=env.trace)
    _deadline(start)
    branches, anchor_states = [], None
    for action in ({"action": "wait", "ticks": 30},
                   {"action": "build_tower", "holder_id": 19, "tower_type": "engineer"}):
        with NativeEnv() as env:
            initial, middle, fork = establish_fork(env)
            anchor_states = (initial, middle)
            menu = build_menu(fork)
            label = next(m["label"] for m in menu if m["action"] == action)
            receipt = env.act(action)
            outcome = finish(env, lambda: _deadline(start))
            branch = {"fork_state": fork, "menu": menu, "label": label, "action": action,
                      "receipt": receipt, "outcome": outcome, "final_state": copy.deepcopy(env.state),
                      "plan": copy.deepcopy(env.plan), "trace": copy.deepcopy(env.trace)}
        _deadline(start)
        with NativeEnv() as replay:
            replay.replay(branch["plan"])
            if replay.trace != branch["trace"] or replay.terminal() != outcome:
                raise RuntimeError("Cold native branch replay differs")
            branch["replay_verified"] = True
            branch["replay_trace_sha256"] = sha256_data(replay.trace)
        _deadline(start)
        journal = Journal(output / f"branch-{label}.jsonl")
        journal.append("fork", {"state": fork, "menu": menu, "prompt_sha256": prompt_sha256(fork, menu)})
        for entry in branch["trace"]:
            journal.append("native_trace", entry)
        journal.append("terminal", {"outcome": outcome, "plan": branch["plan"], "replay_verified": True})
        branch["journal"] = journal.verify()
        branch["journal"]["engine_replay_verified"] = True
        branch["journal_path"] = (output / f"branch-{label}.jsonl").relative_to(ROOT).as_posix()
        branches.append(branch)
    if branches[0]["fork_state"] != branches[1]["fork_state"]:
        raise RuntimeError("Branches did not start from the same exported native state")
    # Validation is development-only: capture one state, no return/continuation.
    with NativeEnv(seed=2001, level=2) as val:
        validation_state = copy.deepcopy(val.state)
    _deadline(start)
    scoring = json.loads((ROOT / "configs/scoring.json").read_text())
    prices = json.loads((ROOT / "configs/prices.json").read_text())
    policy = TinyOptionPolicy(seed=0)
    fork, menu = branches[0]["fork_state"], branches[0]["menu"]
    common = {"pool": "train", "data_kind": "native", "level": 1, "seed": 1001}
    anchors = {}
    for name, state in zip(("A1", "A2"), anchor_states):
        anchor_menu = build_menu(state)
        anchors[name] = [{**common, "id": f"kr1-native-anchor-{name}-1001", "state": state,
                          "menu": anchor_menu, "reference": policy.distribution(state, anchor_menu)}]
    dataset = {
        "schema_version": 1, "source": "real_game", "scope": SCOPE,
        "pool_registry": json.loads((ROOT / "configs/pools.json").read_text()),
        "scoring_contract": scoring, "scorer_sha256": sha256_data(scoring),
        "price_contract": prices, "price_sha256": sha256_data(prices),
        "groups": [{**common, "fork_id": "kr1-native-fork-1001", "state": fork, "menu": menu,
                    "reference": policy.distribution(fork, menu),
                    "scorer_sha256": sha256_data(scoring), "price_sha256": sha256_data(prices),
                    "candidates": [{"label": b["label"], "return": score_native_outcome(b["outcome"], scoring),
                                    "native_outcome": b["outcome"], "receipt_verified": True,
                                    "replay_verified": True, "journal_tip_sha256": b["journal"]["tip_sha256"],
                                    "journal_path": b["journal_path"]} for b in branches]}],
        "anchors": anchors,
        "validation_forks": [{"id": "kr2-validation-state-2001", "pool": "validation", "data_kind": "native",
                              "level": 2, "seed": 2001, "state": validation_state,
                              "menu": build_menu(validation_state),
                              "reference": policy.distribution(validation_state, build_menu(validation_state))}],
        "heldout_consumed": False,
    }
    write_json(output / "state-and-actions.json", probe)
    write_json(output / "branches.json", {"scope": SCOPE, "branches": branches})
    write_json(output / "dataset.json", dataset)
    gates = {}
    for name, artifact in (("exact_state", "state-and-actions.json"), ("action_receipts", "state-and-actions.json"),
                           ("deterministic_replay", "branches.json"), ("branch_replay", "branches.json")):
        p = output / artifact
        gates[name] = {"status": "verified", "scope": SCOPE, "artifact_path": p.relative_to(ROOT).as_posix(),
                       "artifact_sha256": sha256_file(p)}
    evidence = {"schema_version": 1, "scope": SCOPE, "gates": gates,
                "engine_manifest_sha256": sha256_file(ROOT / "runtime/rl-engine/manifest.json"),
                "dataset_path": (output / "dataset.json").relative_to(ROOT).as_posix(),
                "dataset_sha256": sha256_file(output / "dataset.json"),
                "heldout_consumed": False, "wall_seconds": time.monotonic() - start}
    request = {"id": "kr1-native-fork-1001", "system": "You control Kingdom Rush. Choose exactly one option label from the legal menu. Optimize native level victory and remaining lives. Reply with the option label only.",
               "user": decision_prompt(fork, menu), "labels": [m["label"] for m in menu]}
    write_new_json(output / "native-evidence.json", evidence)
    write_new_json(output / "model-native-request.json", request)
    if publish:
        evidence_target, request_target = _published_paths()
        write_new_json(evidence_target, evidence)
        write_new_json(request_target, request)
    return evidence


if __name__ == "__main__":
    print(json.dumps(collect_verified(), ensure_ascii=False, indent=2))
