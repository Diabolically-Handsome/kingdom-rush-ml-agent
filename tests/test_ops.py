import json
from pathlib import Path
import shutil
import tempfile
import time
import unittest

from alpharush_rl import ops


def _small_job(context):
    context.check()
    context.record("optimizer-sanity", data_kind="synthetic")
    return {"parameter_updated": True, "data_kind": "synthetic"}


def _blocking_job(context):
    time.sleep(10)
    return None


class OpsTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="alpharush-ops-")
        self.root = Path(self.tmp.name)
        (self.root / "configs").mkdir()
        (self.root / "alpharush_rl").mkdir()
        shutil.copyfile(Path(ops.__file__), self.root / "alpharush_rl/ops.py")
        workspace = Path(__file__).resolve().parents[1]
        self.cfg = self.root / "configs/rl-deployment.json"
        self.pools = self.root / "configs/pools.json"
        shutil.copyfile(workspace / "configs/rl-deployment.json", self.cfg)
        shutil.copyfile(workspace / "configs/pools.json", self.pools)
        shutil.copyfile(workspace / "configs/models-comparison.json", self.root / "configs/models-comparison.json")
        ops.freeze(self.cfg)

    def tearDown(self):
        self.tmp.cleanup()

    def test_preflight_read_only_and_missing_pins(self):
        manifest = self.root / "runtime/rl/pins.json"
        manifest.unlink()
        before = set(self.root.rglob("*"))
        r = ops.preflight(self.cfg)
        self.assertFalse(r["ok"])
        self.assertEqual(before, set(self.root.rglob("*")))

    def test_code_mutation_is_refused(self):
        code = self.root / "alpharush_rl/ops.py"
        code.write_text(code.read_text(encoding="utf-8") + "\n# changed\n", encoding="utf-8")
        self.assertFalse(ops.preflight(self.cfg)["ok"])

    def test_stop_lock_budget_and_unclosed_are_refused(self):
        state = self.root / "runtime/rl"
        stop = state / "STOP"
        stop.touch()
        self.assertFalse(ops.preflight(self.cfg)["ok"])
        stop.unlink()
        lock = state / "job.lock"
        lock.write_text("{}", encoding="utf-8")
        self.assertFalse(ops.preflight(self.cfg)["ok"])
        lock.unlink()
        ledger = state / "ledger.jsonl"
        ops._append(ledger, {"event": "open", "run_id": "interrupted", "kind": "cpu-smoke"})
        self.assertFalse(ops.preflight(self.cfg)["ok"])
        ops._append(ledger, {"event": "close", "run_id": "interrupted", "kind": "cpu-smoke", "wall_seconds": 100})
        self.assertFalse(ops.preflight(self.cfg)["ok"])

    def test_gradient_pool_leaks_rejected_for_groups_and_anchors(self):
        pools = json.loads(self.pools.read_text(encoding="utf-8"))
        good = {"id": "train", "seed": 1001, "level": 1}
        self.assertEqual([], ops.audit_training_data([good], [dict(good, id="anchor", anchor_set="A1")], pools))
        self.assertTrue(ops.audit_training_data([dict(good, seed=2001)], [], pools))
        self.assertTrue(ops.audit_training_data([], [dict(good, seed=3001, anchor_set="A2")], pools))
        self.assertTrue(ops.audit_training_data([dict(good, level=2)], [], pools))
        self.assertTrue(ops.audit_training_data([dict(good, seed=9999)], [], pools))
        self.assertTrue(ops.audit_training_data([dict(good, seed=1001.0)], [], pools))
        self.assertTrue(ops.audit_training_data([good], [good], pools))

    def test_real_training_refuses_pending_native_evidence(self):
        dataset = self.root / "runtime/rl/data.json"
        dataset.write_text(json.dumps([{"id": "f1", "seed": 1001, "level": 1, "data_kind": "native"}]), encoding="utf-8")
        r = ops.preflight(self.cfg, "cpu-rl-smoke", data_path=dataset)
        self.assertFalse(r["ok"])
        self.assertTrue(any("Native evidence" in x for x in r["issues"]))

    def test_native_evidence_hashes_checked(self):
        state = self.root / "runtime/rl"
        artifact = state / "evidence.json"
        artifact.write_text('{"test":true}', encoding="utf-8")
        gate = {"status": "verified", "artifact_path": "runtime/rl/evidence.json", "artifact_sha256": ops.sha256_file(artifact)}
        evidence = {"schema_version": 1, "gates": {k: dict(gate) for k in ("exact_state", "action_receipts", "deterministic_replay", "branch_replay")}}
        ep = state / "native-evidence.json"
        ep.write_text(json.dumps(evidence), encoding="utf-8")
        self.assertTrue(ops.preflight(self.cfg, "cpu-native-rollout")["ok"])
        artifact.write_text("changed", encoding="utf-8")
        self.assertFalse(ops.preflight(self.cfg, "cpu-native-rollout")["ok"])

    def test_anchor_mapping_flattening_excludes_validation(self):
        doc = {"groups": [{"id": "g", "seed": 1001, "level": 1}],
               "anchors": {"A1": [{"id": "a1", "seed": 1002, "level": 1}],
                           "A2": [{"id": "a2", "seed": 1003, "level": 1}]},
               "validation_forks": [{"id": "v", "seed": 2001, "level": 2}]}
        dataset = self.root / "runtime/rl/data.json"
        dataset.write_text(json.dumps(doc), encoding="utf-8")
        rows = ops._dataset_rows(dataset)
        self.assertEqual(["g", "a1", "a2"], [r["id"] for r in rows])
        pools = json.loads(self.pools.read_text(encoding="utf-8"))
        self.assertEqual([], ops.audit_records(rows, pools))

    def test_native_dataset_is_bound_to_evidence_and_external_pools(self):
        state = self.root / "runtime/rl"
        pools = json.loads(self.pools.read_text(encoding="utf-8"))
        dataset = state / "data.json"
        doc = {"groups": [{"id": "g", "seed": 1001, "level": 1, "data_kind": "native"}],
               "anchors": {"A1": [], "A2": []}, "pool_registry": pools,
               "validation_forks": [{"id": "v", "seed": 2001, "level": 2}]}
        dataset.write_text(json.dumps(doc), encoding="utf-8")
        artifact = state / "evidence.json"
        artifact.write_text('{"verifiedFixture":true}', encoding="utf-8")
        gate = {"status": "verified", "artifact_path": "runtime/rl/evidence.json", "artifact_sha256": ops.sha256_file(artifact)}
        evidence = {"schema_version": 1, "dataset_path": "runtime/rl/data.json", "dataset_sha256": ops.sha256_file(dataset),
                    "gates": {k: dict(gate) for k in ("exact_state", "action_receipts", "deterministic_replay", "branch_replay")}}
        ep = state / "native-evidence.json"
        ep.write_text(json.dumps(evidence), encoding="utf-8")
        self.assertTrue(ops.preflight(self.cfg, "cpu-rl-smoke", data_path=dataset)["ok"])
        dataset.write_text(json.dumps(doc) + "\n", encoding="utf-8")
        self.assertFalse(ops.preflight(self.cfg, "cpu-rl-smoke", data_path=dataset)["ok"])
        doc["pool_registry"]["pools"]["heldout"]["seeds"].append(3999)
        dataset.write_text(json.dumps(doc), encoding="utf-8")
        evidence["dataset_sha256"] = ops.sha256_file(dataset)
        ep.write_text(json.dumps(evidence), encoding="utf-8")
        r = ops.preflight(self.cfg, "cpu-rl-smoke", data_path=dataset)
        self.assertFalse(r["ok"])
        self.assertTrue(any("pool registry" in item for item in r["issues"]))

    def test_heldout_reservation_once_and_no_pool_leak(self):
        records = [{"seed": 3001, "level": 3}]
        ops.claim_heldout_evaluation(self.cfg, "frozen-eval-1", "a" * 64, records)
        with self.assertRaises(ops.GateRefused):
            ops.claim_heldout_evaluation(self.cfg, "frozen-eval-2", "b" * 64, records)
        with self.assertRaises(ops.GateRefused):
            ops.claim_heldout_evaluation(self.cfg, "invalid", "c" * 64, [{"seed": 1001, "level": 1}])

    def test_gpu_disabled(self):
        self.assertFalse(ops.preflight(self.cfg, "gpu-train")["ok"])

    def test_job_context_receipt_and_lock_cleanup(self):
        with ops.job_context(self.cfg) as context:
            context.check()
            output = context.output_dir
        self.assertFalse((self.root / "runtime/rl/job.lock").exists())
        receipt = json.loads((output / "receipt.json").read_text(encoding="utf-8"))
        self.assertEqual("ok", receipt["status"])
        self.assertFalse(receipt["verified"])

    def test_spawned_job_runs_and_hard_deadline_kills_blocking_callback(self):
        result = ops.launch(self.cfg, _small_job)
        self.assertTrue(result["parameter_updated"])
        cfg = json.loads(self.cfg.read_text(encoding="utf-8"))
        cfg["jobs"]["cpu-smoke"]["max_wall_seconds"] = 0.5
        self.cfg.write_text(json.dumps(cfg), encoding="utf-8")
        ops.freeze(self.cfg)
        start = time.monotonic()
        with self.assertRaises(ops.GateRefused):
            ops.launch(self.cfg, _blocking_job)
        self.assertLess(time.monotonic() - start, 4)
        self.assertFalse((self.root / "runtime/rl/job.lock").exists())
        rows = ops._ledger(self.root / "runtime/rl/ledger.jsonl")
        self.assertEqual("close", rows[-1]["event"])
        self.assertEqual("stopped", rows[-1]["status"])


if __name__ == "__main__":
    unittest.main()
