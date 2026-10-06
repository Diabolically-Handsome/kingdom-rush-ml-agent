"""User-facing deployment commands, with explicit scoped evidence and pins."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import statistics
import time
import uuid

from .engine import ROOT
from .journal import Journal, sha256_data
from .menus import build_menu, decision_prompt, prompt_sha256
from .model_broker import GPU_MODELS, validate_distribution, validate_request
from .ops import freeze, launch, preflight, sha256_file
from .validation import NativeEnv, establish_fork, finish, collect_verified, write_json, write_new_json

CONFIG = ROOT / "configs/rl-deployment.json"


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def _train_callback(context):
    from .trainer import train_groups
    evidence = read(ROOT / "runtime/rl/native-evidence.json")
    config = read(ROOT / "configs/training-smoke.json")
    config["stop_file"] = str(context.state_dir / "STOP")
    report = train_groups(ROOT / evidence["dataset_path"], config, context.output_dir / "training")
    context.record("native-cpu-update-result", status=report["status"], steps=report["steps_completed"])
    write_json(context.state_dir / "latest-training.json", {
        "run_id": context.run_id, "report_path": str((context.output_dir / "training/TRAINING-REPORT.json").relative_to(ROOT)),
        "status": report["status"], "validation_kind": report["validation_kind"]})
    return report


def _sft_callback(context):
    from .imitation import train_imitation
    from .policies import TinyOptionPolicy
    evidence = read(ROOT / "runtime/rl/native-evidence.json")
    data = read(ROOT / evidence["dataset_path"])
    fork = data["groups"][0]
    # This label is a declared fixed rules demonstration. It is not labeled as
    # a human expert or attributed to either language model.
    label = next(m["label"] for m in fork["menu"]
                 if m["action"] == {"action": "build_tower", "holder_id": 19, "tower_type": "engineer"})
    example = {"id": "kr1-rules-demo-1001", "pool": "train", "data_kind": "native", "level": 1, "seed": 1001,
               "state": fork["state"], "menu": fork["menu"], "label": label, "expert_source": "rules",
               "receipt_verified": True, "reference": TinyOptionPolicy(seed=0).distribution(fork["state"], fork["menu"])}
    imitation = {"source": "real_game", "pool_registry": data["pool_registry"],
                 "price_contract": data["price_contract"], "price_sha256": data["price_sha256"], "examples": [example]}
    write_json(context.output_dir / "sft-dataset.json", imitation)
    report = train_imitation(imitation, {"steps": 10, "learning_rate": 0.003,
                             "stop_file": str(context.state_dir / "STOP")}, context.output_dir / "sft")
    context.record("native-cpu-sft-result", status=report["status"])
    write_json(context.state_dir / "latest-sft.json", {"run_id": context.run_id,
               "report_path": str((context.output_dir / "sft/SFT-REPORT.json").relative_to(ROOT)), "status": report["status"]})
    return report


def status():
    cfg = read(CONFIG)
    evidence_path = ROOT / cfg["native_evidence_path"]
    evidence = read(evidence_path) if evidence_path.exists() else {}
    data = evidence.get("dataset_path")
    result = {"project": "AlphaRush", "native_scope": evidence.get("scope", "pending"),
              "cpu_training_preflight": preflight(CONFIG, "cpu-rl-smoke", data_path=data),
              "formal_training_enabled": cfg["formal_training_enabled"],
              "models": cfg["model"], "heldout_accessed": False}
    for name in ("latest-training", "latest-sft", "latest-llm-training", "model-comparison"):
        p = ROOT / f"runtime/rl/{name}.json"
        if p.exists():
            result[name] = read(p)
    current = ROOT / "runtime/rl/level1-24b-phase1/latest-training.json"
    if current.exists():
        result["level1-24b-training"] = read(current)
    waiting = ROOT / "runtime/rl/level1-24b-phase1/gpu-wait.json"
    if waiting.exists() and not current.exists():
        result["level1-24b-waiting"] = read(waiting)
    return result


def prepare_comparison(output_dir=None, import_batch=None):
    """First-level deployment comparison only; no heldout states accessed."""
    ev = read(ROOT / "runtime/rl/native-evidence.json")
    check = preflight(CONFIG, "cpu-native-rollout", data_path=ev["dataset_path"])
    if not check["ok"]:
        raise RuntimeError("; ".join(check["issues"]))
    out = Path(output_dir or ROOT / f"runtime/rl/comparison-{uuid.uuid4().hex[:12]}").resolve()
    if not out.is_relative_to(ROOT) or out.exists():
        raise ValueError("comparison must use a fresh directory inside AlphaRush")
    group = read(ROOT / ev["dataset_path"])["groups"][0]
    prompt = decision_prompt(group["state"], group["menu"])
    system = "You control Kingdom Rush. Choose exactly one option label from the legal menu. Optimize native level victory and remaining lives. Reply with the option label only."
    batch = [{"id": f"kr1-1001-{'warmup' if i < 2 else 'timed'}-{i}", "system": system,
              "user": prompt, "labels": [m["label"] for m in group["menu"]]} for i in range(5)]
    if import_batch:
        batch = read(import_batch)
        if not isinstance(batch, list) or len(batch) != 5:
            raise ValueError("Imported deployment smoke must have five original requests")
        for request in batch:
            validate_request(request)
            if request["user"] != prompt or request["system"] != system or request["labels"] != [m["label"] for m in group["menu"]]:
                raise ValueError("Imported inference smoke used different native prompt bytes")
    plan = {"schema_version": 1, "kind": "first_level_train_pool_exploratory_deployment_comparison",
            "pool": "train", "level": 1, "seed": 1001, "heldout_accessed": False,
            "warmup": 2, "timed": 3, "complete_legal_labels": len(group["menu"]),
            "state": group["state"], "menu": group["menu"], "prompt_sha256": prompt_sha256(group["state"], group["menu"]),
            "messages_sha256": sha256_data({"system": system, "user": prompt}),
            "rules_continuation": "send clear waves, advance 600 ticks, buy no additional towers",
            "pins_sha256": check["pins_sha256"], "native_evidence_sha256": check["evidence_sha256"]}
    if import_batch:
        plan.update(imported_prior_inference_smoke=True, inference_preregistered_under_final_pin_set=False,
                    original_batch_path=str(Path(import_batch).resolve()), original_batch_sha256=sha256_file(import_batch))
    write_json(out / "requests.json", batch)
    write_json(out / "plan.json", plan)
    return {"output_dir": str(out), "requests_path": str(out / "requests.json"), "plan_path": str(out / "plan.json")}


def finish_comparison(directory, response_paths=None):
    """Validate all model results, then replay chosen native branches."""
    directory = Path(directory).resolve()
    if not directory.is_relative_to(ROOT):
        raise ValueError("comparison directory must stay inside AlphaRush")
    plan, requests = read(directory / "plan.json"), read(directory / "requests.json")
    if sha256_file(ROOT / "runtime/rl/pins.json") != plan["pins_sha256"]:
        raise RuntimeError("Comparison pin set changed after preregistration")
    # Refuse before any native game: never append to or overwrite earlier evidence.
    if (ROOT / "runtime/rl/model-comparison.json").exists():
        raise FileExistsError("runtime/rl/model-comparison.json already exists; comparison results are never overwritten")
    if (directory / "summary.json").exists() or any(directory.glob("*-native.jsonl")):
        raise FileExistsError("This comparison directory already holds native results; use a new directory")
    summary = {"schema_version": 1, "kind": plan["kind"], "pool": "train", "level": 1, "seed": 1001,
               "matched_prompt_sha256": plan["prompt_sha256"], "heldout_accessed": False, "arms": {},
               "limits": ["one first-level decision", "training-pool exploratory test", "hardware and model versions both differ",
                          "remaining decisions are frozen rules", "unconstrained illegal-output rate unmeasured",
                          "does not measure full-game autonomous play or statistical superiority"]}
    scoring = read(ROOT / "configs/scoring.json")
    from .trainer import score_native_outcome
    for key in ("8b", "24b"):
        path = Path(response_paths[key]) if response_paths else directory / f"{key}.json"
        raw = read(path)
        responses = raw.get("responses", [raw])
        if len(responses) != 5:
            raise ValueError("comparison requires 2 warmup and 3 timed responses")
        for request, response in zip(requests, responses):
            validate_distribution(response, request)
            spec = GPU_MODELS[key]
            if (not response["model"].startswith(spec["repo"] + "@" + spec["revision"])
                    or response.get("gpu", {}).get("uuid") != spec["gpu_uuid"]
                    or response.get("adapter") is not None or response.get("learning_updates") != 0):
                raise ValueError("Model/GPU/adapter identity differs from the selected comparison arm")
        if len({r["model"] for r in responses}) != 1:
            raise ValueError("model identity changed within one comparison arm")
        timed = responses[2:]
        if len({r["choice"] for r in timed}) != 1:
            raise RuntimeError("Deterministic model decisions disagreed on identical native state")
        choice = timed[0]["choice"]
        selected = next(m for m in plan["menu"] if m["label"] == choice)
        started = time.monotonic()
        with NativeEnv() as env:
            _, _, fork = establish_fork(env)
            if fork != plan["state"]:
                raise RuntimeError("Model chosen branch did not replay to the same native fork")
            receipt = env.act(selected["action"])
            outcome = finish(env, lambda: _native_compare_deadline(started))
            native_plan, trace = copy.deepcopy(env.plan), copy.deepcopy(env.trace)
            stars = env.state.get("native_outcome", {}).get("stars")
        native_wall = time.monotonic() - started
        replay_started = time.monotonic()
        with NativeEnv() as replay:
            replay.replay(native_plan)
            if replay.trace != trace or replay.terminal() != outcome:
                raise RuntimeError("Chosen model branch native replay differs")
        journal = Journal(directory / f"{key}-native.jsonl")
        journal.append("model_decision", {"model": timed[0]["model"], "choice": choice, "prompt_sha256": plan["prompt_sha256"]})
        for row in trace:
            journal.append("native_trace", row)
        journal.append("terminal", {"outcome": outcome, "replay_verified": True, "plan": native_plan})
        latency = [r["seconds"] for r in timed]
        arm = {"model": timed[0]["model"], "gpu": timed[0]["gpu"], "choice": choice,
               "action": selected["text"], "warmup_seconds": [r["seconds"] for r in responses[:2]],
               "timed_seconds": latency, "median_seconds": statistics.median(latency),
               "p90_seconds": sorted(latency)[1] * 0.2 + sorted(latency)[2] * 0.8,
               "load_seconds": raw.get("load_seconds", timed[0]["load_seconds"]),
               "prompt_tokens": timed[0]["prompt_tokens"],
               "peak_memory_allocated_mib": max(r["peak_memory_allocated_mib"] for r in responses),
               "native_outcome": outcome, "native_stars": stars, "return": score_native_outcome(outcome, scoring),
               "receipt": receipt, "branch_replay_verified": True, "journal_tip": journal.verify()["tip_sha256"],
               "model_decisions": 1, "rules_fallback_decisions": 0,
               "rules_decisions": sum("action" in command for command in native_plan) - 1,
               "scripted_advance_commands": sum("ticks" in command for command in native_plan),
               "provenance_count_unit": "menu actions; fixed native-tick advances reported separately",
               "native_wall_seconds": native_wall,
               "replay_wall_seconds": time.monotonic() - replay_started,
               "approximate_hot_end_to_end_seconds": native_wall + statistics.median(latency),
               "learning_updates": 0}
        summary["arms"][key] = arm
        write_json(directory / "partial-summary.json", summary)
    write_json(directory / "summary.json", summary)
    published = ROOT / "runtime/rl/model-comparison.json"
    if published.exists():
        raise FileExistsError(f"Refusing to overwrite runtime/rl/model-comparison.json; this result is in {directory / 'summary.json'}")
    write_new_json(published, summary)
    return summary


def _native_compare_deadline(started):
    if time.monotonic() - started > 42:
        raise RuntimeError("Native model continuation wall budget exhausted")
    if any((ROOT / "runtime/rl" / n).exists() for n in ("STOP", "ENGINEERING-STOP")):
        raise RuntimeError("Native model comparison stopped")


def main():
    parser = argparse.ArgumentParser(description="AlphaRush deployment and bounded validation")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("status")
    sub.add_parser("freeze")
    v = sub.add_parser("validate-env")
    v.add_argument("--output-dir")
    sub.add_parser("train-smoke")
    sub.add_parser("sft-smoke")
    p = sub.add_parser("prepare-comparison")
    p.add_argument("--output-dir")
    p.add_argument("--import-batch", help="preserve request identities from prior deployment smoke; explicitly retrospective")
    f = sub.add_parser("finish-comparison")
    f.add_argument("directory")
    args = parser.parse_args()
    if args.command == "status":
        result = status()
    elif args.command == "freeze":
        extra = [p.relative_to(ROOT).as_posix() for directory in (ROOT / "alpharush_rl", ROOT / "configs", ROOT / "tools")
                 for p in directory.rglob("*") if p.is_file() and p.suffix in (".py", ".lua", ".json", ".ps1", ".sh")]
        entrypoints = [p.relative_to(ROOT).as_posix() for p in ROOT.iterdir()
                       if p.is_file() and p.suffix in (".ps1", ".cmd")]
        identities = [p.relative_to(ROOT).as_posix() for p in (ROOT / "runtime/rl-models").glob("model-*-manifest.json")]
        result = freeze(CONFIG, [*extra, *entrypoints, *identities, "requirements-local.txt", "runtime/rl-engine/manifest.json"])
    elif args.command == "validate-env":
        output = args.output_dir or ROOT / f"runtime/rl/native-validation-{uuid.uuid4().hex[:12]}"
        result = collect_verified(output, publish=False)
    elif args.command in ("train-smoke", "sft-smoke"):
        evidence = read(ROOT / "runtime/rl/native-evidence.json")
        result = launch(CONFIG, _train_callback if args.command == "train-smoke" else _sft_callback,
                        kind="cpu-rl-smoke", data_path=evidence["dataset_path"])
    elif args.command == "prepare-comparison":
        result = prepare_comparison(args.output_dir, args.import_batch)
    else:
        result = finish_comparison(args.directory)
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    return 2 if isinstance(result, dict) and result.get("status") == "rejected" else 0


if __name__ == "__main__":
    raise SystemExit(main())
