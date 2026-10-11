"""Independent CPU verification of the completed all-legal native collection."""
import copy
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from alpharush_rl.journal import Journal, sha256_data
from alpharush_rl.ops import sha256_file
from alpharush_rl.pools import audit_dataset
from alpharush_rl.reward_level1 import score_level1_outcome


def audit():
    phase = ROOT / "runtime/rl/level1-24b-phase1"
    receipt = json.loads((phase / "collection-supervisor-r1-receipt.json").read_text())
    evidence_path = phase / "data/evidence.json"
    evidence = json.loads(evidence_path.read_text())
    assert receipt["status"] == "verified_all_legal_native_rewards" and receipt["exit_code"] == 0
    assert receipt["worker_exit_confirmed"] is True and receipt["cumulative_wall_seconds"] < 300
    assert receipt["evidence_sha256"] == sha256_file(evidence_path)
    assert evidence["dataset_sha256"] == sha256_file(ROOT / evidence["dataset_path"])
    data = json.loads((ROOT / evidence["dataset_path"]).read_text())
    hygiene = audit_dataset(data)
    full = json.loads((phase / "data/branches.json").read_text())
    group = data["groups"][0]
    projection = copy.deepcopy(full["fork_state"])
    projection.pop("level_path_wave_counts", None)
    assert projection == group["state"] and group["native_fork_state_sha256"] == sha256_data(full["fork_state"])
    labels = [m["label"] for m in group["menu"]]
    assert len(labels) == 26 and labels == [b["label"] for b in full["branches"]]
    assert labels == [c["label"] for c in group["candidates"]]
    for candidate, branch, menu in zip(group["candidates"], full["branches"], group["menu"]):
        outcome = candidate["native_outcome"]
        assert outcome == branch["outcome"] and outcome["native_raw"] == branch["native_raw_outcome"]
        assert outcome["seed"] == 1002 and outcome["level"] == 1 and outcome["difficulty"] == 2
        assert branch["action"] == menu["action"] == branch["receipt"]["action"]
        assert branch["receipt"]["executed"] is True and branch["replay_verified"] is True
        assert candidate["return"] == branch["return"] == score_level1_outcome(outcome, data["scoring_contract"])
        path = ROOT / candidate["journal_path"]
        assert Journal(path).verify()["tip_sha256"] == candidate["journal_tip_sha256"] == branch["journal"]["tip_sha256"]
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        assert rows[0]["payload"]["state"] == full["fork_state"]
        assert rows[-1]["payload"]["outcome"] == outcome
        assert rows[-1]["payload"]["native_raw_outcome"] == outcome["native_raw"]
        assert [row["payload"] for row in rows[1:-1]] == branch["trace"]
        assert branch["trace"][-1]["state_sha256"] == outcome["state_sha256"] == sha256_data(branch["final_state"])
    for gate in evidence["gates"].values():
        assert sha256_file(ROOT / gate["artifact_path"]) == gate["artifact_sha256"]
    for role in ("A1", "A2"):
        assert data["anchors"][role][0]["seed"] == 1002 and data["anchors"][role][0]["level"] == 1
    assert all("level_path_wave_counts" not in row["state"] for row in
               [group, *data["anchors"]["A1"], *data["anchors"]["A2"], *data["validation_forks"]])
    assert all(not any(k in row for k in ("return", "reward", "candidates", "native_outcome")) for row in data["validation_forks"])
    assert data["heldout_consumed"] is False and not (ROOT / "runtime/rl/heldout-audit.jsonl").exists()
    return {"passed": True, "all_legal_native_branches": len(labels), "cold_replays": len(labels),
            "wins": sum(c["native_outcome"]["level_won"] for c in group["candidates"]),
            "losses": sum(c["native_outcome"]["level_lost"] for c in group["candidates"]),
            "top_candidates": [{"label": c["label"], "reward": c["return"], "lives": c["native_outcome"]["lives"]}
                               for c in sorted(group["candidates"], key=lambda c: c["return"], reverse=True)[:5]],
            "collection_wall_seconds": receipt["cumulative_wall_seconds"], "dataset_sha256": evidence["dataset_sha256"],
            "hygiene": hygiene, "heldout_accessed": False}


if __name__ == "__main__":
    print(json.dumps(audit(), indent=2))
