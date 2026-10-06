"""Collector boundary tests using a deterministic Python game handle only.

No native worker, external process, socket or GPU is opened by these tests.
The real menu, continuation, reward and journal implementations are exercised.
"""
from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from alpharush_rl.journal import Journal, sha256_data
from alpharush_rl.menus import build_menu
from alpharush_rl.ops import sha256_file
from alpharush_rl.reward_level1 import score_level1_outcome
from alpharush_rl.validation import write_json


WORKSPACE = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("level1_collector_under_test", WORKSPACE / "tools/collect-level1-rewards.py")
COLLECTOR = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(COLLECTOR)


def initial_state():
    return {"level_idx": 1, "difficulty": 2, "lives": 20, "tick": 1,
            "wave": 0, "gold": 265, "wave_ready": True,
            "holders": [{"id": holder, "blocked": False} for holder in range(15, 23)],
            "towers": [], "enemies": [], "heroes": [],
            "level_path_wave_counts": [{"future_wave": 7, "hidden_group_count": 99}]}


class SimulatedNativeHandle:
    """A small deterministic native-interface substitute, never a real reward row."""

    def __init__(self, factory, seed):
        if seed != 1002:
            raise AssertionError("collector changed the registered seed")
        self.factory = factory
        self.index = len(factory.handles)
        factory.handles.append(self)
        self.replay = self.index > 0 and self.index % 2 == 0
        self.candidate_index = (self.index + 1) // 2
        self.state = initial_state()
        self.plan, self.trace = [], []
        self.decision = None
        self.raw = None
        self._catalog()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def _catalog(self):
        self.state["action_catalog"] = [
            {"action": "build_tower", "holder_id": holder["id"], "tower_type": kind,
             "cost": cost, "available": True}
            for holder in self.state["holders"]
            for kind, cost in (("archer", 70), ("barrack", 70), ("mage", 100), ("engineer", 125))]

    def _record(self, command):
        self.plan.append(copy.deepcopy(command))
        self.trace.append({"command": copy.deepcopy(command), "state_sha256": sha256_data(self.state)})

    def act(self, action):
        action = copy.deepcopy(action)
        if action not in [option["action"] for option in build_menu(self.state)]:
            raise AssertionError(f"simulated handle was given an illegal action: {action}")
        if len(self.plan) >= 2 and self.decision is None:
            self.decision = copy.deepcopy(action)
        if action["action"] == "build_tower":
            cost = {"archer": 70, "barrack": 70, "mage": 100, "engineer": 125}[action["tower_type"]]
            self.state["gold"] -= cost
            self.state["holders"] = [h for h in self.state["holders"] if h["id"] != action["holder_id"]]
            self.state["towers"].append({"id": action["holder_id"], "template": "tower_" + action["tower_type"] + "_1"})
            self.state["tick"] += 180
            self._catalog()
        elif action["action"] == "wait":
            self.state["tick"] += action["ticks"]
        elif action["action"] == "send_wave":
            self.state["wave"] += 1
            self.state["wave_ready"] = False
        self._record({"action": action})
        return {"accepted": True, "executed": True, "action": action,
                "verification": {"status": "verified"}}

    def advance(self, ticks):
        if not self.replay and self.factory.fail_candidate == self.candidate_index:
            raise RuntimeError("simulated RPC timeout; no native terminal outcome")
        self.state["tick"] += ticks
        won = self.decision.get("tower_type") in ("mage", "engineer")
        self.state.update(lives=9 if won else 0, wave=7, level_won=won, level_lost=not won)
        self.raw = {"source": "native", "terminal": True, "level_won": won,
                    "level_lost": not won, "lives": self.state["lives"],
                    "wave": 7, "tick": self.state["tick"], "state_sha256": sha256_data(self.state)}
        self._record({"ticks": ticks})
        if self.replay and self.factory.replay_mismatch:
            self.trace[-1]["state_sha256"] = "f" * 64
        if not self.replay and self.factory.stop_candidate == self.candidate_index:
            (self.factory.phase / "STOP").write_text("test interrupt", encoding="utf-8")

    def terminal(self):
        return copy.deepcopy(self.raw)


class HandleFactory:
    def __init__(self, phase):
        self.phase = phase
        self.handles = []
        self.replay_mismatch = False
        self.fail_candidate = None
        self.stop_candidate = None

    def __call__(self, *, seed):
        return SimulatedNativeHandle(self, seed)


class FirstLevelCollectionTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="alpharush-collector-test-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.phase = self.root / "runtime/rl/level1-24b-phase1"
        self.data = self.phase / "data"
        self.plan = self.phase / "collection-plan-r1.json"
        self.supervisor_receipt = self.phase / "collection-supervisor-r1-receipt.json"
        for field, value in (("ROOT", self.root), ("PHASE", self.phase), ("DATA", self.data), ("PLAN", self.plan),
                             ("SUPERVISOR_LEDGER", self.phase / "collection-supervisor-r1-ledger.jsonl"),
                             ("SUPERVISOR_RECEIPT", self.supervisor_receipt)):
            patcher = patch.object(COLLECTOR, field, value)
            patcher.start()
            self.addCleanup(patcher.stop)
        # Historical real-run pins belong to the launcher, not this temporary
        # synthetic fixture; preserve the collector's live history untouched.
        history_patch = patch.object(COLLECTOR, "historical_seconds", return_value=0.0)
        history_patch.start()
        self.addCleanup(history_patch.stop)
        for filename in ("configs/pools.json", "configs/prices.json", "configs/scoring-level1-terminal-v2.json"):
            target = self.root / filename
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes((WORKSPACE / filename).read_bytes())
        pools = json.loads((self.root / "configs/pools.json").read_text(encoding="utf-8"))
        validation_state = initial_state()
        validation_state["level_idx"] = 2
        old_dataset = {"scope": {"game": "kr1-desktop-6.4.46", "level": 1, "difficulty": 2},
                       "pool_registry": pools, "validation_forks": [{"id": "reserved-val", "pool": "validation",
                            "data_kind": "native", "level": 2, "seed": 2001, "state": validation_state,
                            "menu": build_menu(validation_state), "reference": {"old_reference": "remove"}}]}
        write_json(self.root / "runtime/rl/native-validation/dataset.json", old_dataset)
        write_json(self.root / "runtime/rl/native-evidence.json", {"gates": {name: {"status": "verified", "old_proof": name}
                   for name in ("exact_state", "action_receipts", "deterministic_replay", "branch_replay")}})
        write_json(self.root / "runtime/rl-engine/manifest.json", {"test_engine_only": True})
        self.old_files = {str(path.relative_to(self.root)): path.read_bytes()
                          for path in self.root.rglob("*.json")}
        write_json(self.plan, {"schema_version": 1, "files": {relative: sha256_file(self.root / relative)
                    for relative in self.old_files}, "seed": 1002, "level": 1, "difficulty": 2,
                    "wall_seconds": 300, "max_jobs": 1, "all_legal_candidates": True})
        self.factory = HandleFactory(self.phase)
        handle_patch = patch.object(COLLECTOR, "NativeEnv", self.factory)
        handle_patch.start()
        self.addCleanup(handle_patch.stop)
        socket_patch = patch.object(COLLECTOR.socket, "socket")
        self.probe = socket_patch.start()
        self.addCleanup(socket_patch.stop)

    def read(self, path):
        return json.loads(Path(path).read_text(encoding="utf-8"))

    def close_row(self):
        rows = Journal(self.phase / "collection-ledger.jsonl").entries()
        self.assertEqual([row["kind"] for row in rows], ["open", "close"])
        self.assertTrue(Journal(self.phase / "collection-ledger.jsonl").verify()["integrity_verified"])
        return rows[-1]["payload"]

    def test_all_26_legal_candidates_have_pinned_receipt_replay_and_raw_outcome(self):
        evidence = COLLECTOR.collect()
        dataset = self.read(self.data / "dataset.json")
        branches = self.read(self.data / "branches.json")
        candidates = dataset["groups"][0]["candidates"]
        self.assertEqual(len(candidates), 26)
        self.assertEqual({r["label"] for r in candidates}, {r["label"] for r in branches["menu"]})
        self.assertEqual(len(self.factory.handles), 1 + 2 * 26)
        self.assertTrue(dataset["all_legal_candidates_verified"])
        for candidate, branch in zip(candidates, branches["branches"]):
            self.assertTrue(candidate["receipt_verified"] and candidate["replay_verified"])
            self.assertTrue(branch["receipt"]["accepted"] and branch["receipt"]["executed"])
            self.assertEqual(candidate["native_outcome"]["native_raw"], branch["native_raw_outcome"])
            self.assertEqual(candidate["return"], score_level1_outcome(candidate["native_outcome"], dataset["scoring_contract"]))
            journal = Journal(self.root / candidate["journal_path"])
            journal.verify(candidate["journal_tip_sha256"])
            self.assertEqual(journal.entries()[-1]["payload"]["native_raw_outcome"], branch["native_raw_outcome"])
        self.assertEqual(evidence["dataset_sha256"], sha256_file(self.data / "dataset.json"))
        self.assertFalse(dataset["heldout_consumed"])
        self.assertEqual(self.close_row()["status"], "verified_all_legal_native_rewards")
        for relative, original in self.old_files.items():
            self.assertEqual((self.root / relative).read_bytes(), original, relative)

    def test_policy_rows_hide_future_waves_and_validation_never_gains_returns(self):
        COLLECTOR.collect()
        data = self.read(self.data / "dataset.json")
        raw = self.read(self.data / "branches.json")["fork_state"]
        self.assertIn("level_path_wave_counts", raw)
        group = data["groups"][0]
        self.assertEqual(group["native_fork_state_sha256"], sha256_data(raw))
        self.assertNotIn("level_path_wave_counts", group["state"])
        for role in ("A1", "A2"):
            self.assertEqual(data["anchors"][role][0]["pool"], "train")
            self.assertNotIn("level_path_wave_counts", data["anchors"][role][0]["state"])
        for row in data["validation_forks"]:
            self.assertEqual(row["pool"], "validation")
            self.assertEqual(row["seed"], 2001)
            self.assertNotIn("level_path_wave_counts", row["state"])
            self.assertFalse({"return", "reward", "native_outcome", "candidates", "reference"} & row.keys())

    def test_cold_replay_trace_mismatch_refuses_any_final_dataset(self):
        self.factory.replay_mismatch = True
        with self.assertRaisesRegex(RuntimeError, "Full cold replay differs"):
            COLLECTOR.collect()
        self.assertFalse((self.data / "dataset.json").exists())
        self.assertFalse((self.data / "evidence.json").exists())
        self.assertEqual(self.close_row()["status"], "failed")
        self.assertFalse((self.phase / "collection.lock").exists())

    def test_partial_branch_files_do_not_become_a_final_dataset_or_a_retry(self):
        self.factory.fail_candidate = 2
        with self.assertRaisesRegex(RuntimeError, "RPC timeout"):
            COLLECTOR.collect()
        self.assertTrue((self.data / "branch-A.json").exists())
        self.assertEqual(self.read(self.phase / "collection-progress.json")["completed"], 1)
        self.assertFalse((self.data / "dataset.json").exists())
        self.assertFalse((self.data / "evidence.json").exists())
        self.assertEqual(self.close_row()["status"], "failed")
        count = len(self.factory.handles)
        with self.assertRaisesRegex(RuntimeError, "already been spent"):
            COLLECTOR.collect()
        self.assertEqual(len(self.factory.handles), count)

    def test_stop_before_open_and_during_continuation_refuses_the_job(self):
        (self.phase / "STOP").write_text("stop", encoding="utf-8")
        with self.assertRaisesRegex(RuntimeError, "stopped/locked"):
            COLLECTOR.collect()
        self.assertEqual(self.factory.handles, [])
        self.assertFalse((self.phase / "collection-ledger.jsonl").exists())
        (self.phase / "STOP").unlink()
        self.factory.stop_candidate = 1
        with self.assertRaisesRegex(RuntimeError, "Collection stopped"):
            COLLECTOR.collect()
        self.assertFalse((self.data / "dataset.json").exists())
        self.assertEqual(self.close_row()["status"], "failed")

    def test_source_change_after_last_branch_blocks_publication(self):
        original_writer = COLLECTOR.write_json

        def write_and_change_source(path, value):
            result = original_writer(path, value)
            if Path(path).name == "collection-progress.json" and value["completed"] == value["total"]:
                write_json(self.root / "runtime/rl-engine/manifest.json", {"changed_during_job": True})
            return result

        with patch.object(COLLECTOR, "write_json", write_and_change_source):
            with self.assertRaisesRegex(RuntimeError, "source changed during native run"):
                COLLECTOR.collect()
        self.assertEqual(self.read(self.phase / "collection-progress.json")["completed"], 26)
        self.assertFalse((self.data / "dataset.json").exists())
        self.assertEqual(self.close_row()["status"], "failed")

    def test_partial_publication_failure_has_no_verified_close_or_evidence(self):
        original_writer = COLLECTOR.write_json

        def fail_branches_write(path, value):
            if Path(path).name == "branches.json":
                raise OSError("simulated disk full during final publication")
            return original_writer(path, value)

        with patch.object(COLLECTOR, "write_json", fail_branches_write):
            with self.assertRaisesRegex(OSError, "disk full"):
                COLLECTOR.collect()
        self.assertFalse((self.data / "evidence.json").exists())
        self.assertEqual(self.close_row()["status"], "failed")
        # A leftover dataset file is diagnostic only: no final supervisor proof.
        self.assertFalse(self.supervisor_receipt.exists())

    def test_finished_job_cannot_open_another_collection(self):
        COLLECTOR.collect()
        count = len(self.factory.handles)
        with self.assertRaisesRegex(RuntimeError, "already been spent"):
            COLLECTOR.collect()
        self.assertEqual(len(self.factory.handles), count)
        self.assertEqual(self.close_row()["optimizer_steps"], 0)


if __name__ == "__main__":
    unittest.main()
