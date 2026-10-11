"""Read-only deployment-comparison gates; never imports torch or starts GPU work."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from alpharush_rl.journal import sha256_data
from alpharush_rl.menus import decision_prompt, prompt_sha256
from alpharush_rl.model_broker import GPU_MODELS, validate_distribution, validate_request
from alpharush_rl.ops import GateRefused, preflight, sha256_file


def inspect(model, directory, allow_result=False):
    directory = Path(directory).resolve()
    if not directory.is_relative_to(ROOT):
        raise GateRefused("Comparison directory must be inside AlphaRush")
    cfg = json.loads((ROOT / "configs/rl-deployment.json").read_text(encoding="utf-8-sig"))
    evidence = json.loads((ROOT / cfg["native_evidence_path"]).read_text(encoding="utf-8-sig"))
    check = preflight(ROOT / "configs/rl-deployment.json", "cpu-native-rollout", data_path=evidence["dataset_path"])
    if not check["ok"]:
        raise GateRefused("; ".join(check["issues"]))
    owner_words = cfg.get("deployment_authorization", {}).get("model_comparison_owner_words")
    bounded = cfg.get("bounded_model_inference", {})
    if not owner_words or bounded.get("authorized") is not True:
        raise GateRefused("Current user comparison authorization is missing")
    if bounded.get("training_enabled") or bounded.get("serve_enabled"):
        raise GateRefused("Bounded comparison must not enable training or automatic serving")
    cap = bounded.get("max_wall_seconds")
    if isinstance(cap, bool) or not isinstance(cap, int) or not 0 < cap <= 600:
        raise GateRefused("Inference cap must be 1..600 seconds")
    plan_path, batch_path = directory / "plan.json", directory / "requests.json"
    plan = json.loads(plan_path.read_text(encoding="utf-8-sig"))
    requests = json.loads(batch_path.read_text(encoding="utf-8-sig"))
    if plan.get("pins_sha256") != check["pins_sha256"] or plan.get("native_evidence_sha256") != check["evidence_sha256"]:
        raise GateRefused("Comparison plan was prepared for another pin/evidence set")
    if plan.get("pool") != "train" or plan.get("level") != 1 or plan.get("seed") != 1001 or plan.get("heldout_accessed") is not False:
        raise GateRefused("This launcher permits only the initial first-level training-pool exploration")
    if not isinstance(requests, list) or not 1 <= len(requests) <= min(6, bounded.get("max_batch_requests", 0)):
        raise GateRefused("Inference batch must contain one to six requests")
    expected_prompt = decision_prompt(plan["state"], plan["menu"])
    expected_labels = [item["label"] for item in plan["menu"]]
    if prompt_sha256(plan["state"], plan["menu"]) != plan.get("prompt_sha256"):
        raise GateRefused("Frozen native prompt differs from state/menu")
    for request in requests:
        validate_request(request)
        if request["user"] != expected_prompt or request["labels"] != expected_labels:
            raise GateRefused("Request differs from the identical frozen native state/menu")
        if sha256_data({"system": request["system"], "user": request["user"]}) != plan.get("messages_sha256"):
            raise GateRefused("Request system/user byte identity differs from plan")
    spec = GPU_MODELS[model]
    selected = cfg["model"]["models"][model]
    if selected["base"] != spec["repo"] or selected["gpu_uuid"] != spec["gpu_uuid"] or selected.get("adapter") is not None:
        raise GateRefused("Selected deployment model/GPU identity differs from worker configuration")
    output = directory / f"{model}.json"
    if output.exists() and not allow_result:
        raise GateRefused("Model result already exists; use a new prepared comparison directory")
    return dict(model=model, directory=str(directory), batch_path=str(batch_path), output_path=str(output),
                gpu_uuid=spec["gpu_uuid"], base=spec["repo"], owner_words=owner_words,
                max_requests=len(requests), max_wall_seconds=cap,
                total_wall_seconds=bounded["total_wall_seconds"], max_jobs=bounded["max_jobs"],
                busy_memory_used_mib=bounded["busy_memory_used_mib"][model],
                busy_utilization_percent=bounded["busy_utilization_percent"],
                pins_sha256=check["pins_sha256"], evidence_sha256=check["evidence_sha256"],
                plan_sha256=sha256_file(plan_path), batch_sha256=sha256_file(batch_path),
                venv_python_wsl="/home/<user>/alpharush/.venv/bin/python")


def verify_result(model, directory):
    job = inspect(model, directory, allow_result=True)
    requests = json.loads(Path(job["batch_path"]).read_text(encoding="utf-8-sig"))
    result = json.loads(Path(job["output_path"]).read_text(encoding="utf-8-sig"))
    responses = result.get("responses", [])
    if len(responses) != len(requests) or result.get("learning_updates") != 0:
        raise GateRefused("Incomplete batch or unexpected learning updates")
    spec = GPU_MODELS[model]
    identity = f"{spec['repo']}@{spec['revision']}:nf4-bf16:instruction-base:no-adapter"
    for request, response in zip(requests, responses):
        validate_distribution(response, request)
        if response.get("model") != identity or response.get("gpu", {}).get("uuid") != spec["gpu_uuid"]:
            raise GateRefused("Result base/revision/GPU identity differs from frozen deployment")
        if response.get("adapter") is not None or response.get("learning_updates") != 0:
            raise GateRefused("Unexpected adapter or learning updates")
    return dict(verified=True, model=model, responses=len(responses), result_sha256=sha256_file(job["output_path"]), scope="first_level_train_pool_deployment_inference_only", learning_updates=0)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=("8b", "24b"), required=True)
    ap.add_argument("--directory", required=True)
    ap.add_argument("--verify-result", action="store_true")
    args = ap.parse_args()
    try:
        print(json.dumps(verify_result(args.model, args.directory) if args.verify_result else inspect(args.model, args.directory), ensure_ascii=False, allow_nan=False))
    except (OSError, ValueError, KeyError, TypeError, GateRefused) as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(2)
