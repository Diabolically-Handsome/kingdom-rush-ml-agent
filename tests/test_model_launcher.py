"""Pure CPU fixtures for comparison identity gates; no game/GPU invocation."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from alpharush_rl.journal import sha256_data
from alpharush_rl.menus import build_menu, decision_prompt, prompt_sha256
from alpharush_rl.model_broker import GPU_MODELS
from alpharush_rl.ops import GateRefused


spec = importlib.util.spec_from_file_location("launcher_preflight", Path(__file__).resolve().parents[1] / "tools/model-preflight.py")
helper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helper)


class ModelLauncherTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="alpharush-model-launch-")
        self.root = Path(self.temp.name)
        self.directory = self.root / "comparison"
        self.directory.mkdir()
        (self.root / "configs").mkdir()
        (self.root / "runtime/rl").mkdir(parents=True)
        origin = Path(__file__).resolve().parents[1]
        self.cfg = json.loads((origin / "configs/rl-deployment.json").read_text(encoding="utf-8-sig"))
        (self.root / "configs/rl-deployment.json").write_text(json.dumps(self.cfg), encoding="utf-8")
        (self.root / "runtime/rl/native-evidence.json").write_text('{"dataset_path":"fixture.json"}', encoding="utf-8")
        self.state = {"gold": 100, "lives": 20, "wave": 0, "holders": [], "towers": [], "enemies": [], "action_catalog": []}
        self.menu = build_menu(self.state)
        user = decision_prompt(self.state, self.menu)
        self.request = {"id": "fixture", "system": "Choose one label", "user": user, "labels": [m["label"] for m in self.menu]}
        self.requests = [dict(self.request, id=f"fixture-{i}") for i in range(5)]
        self.plan = {"pins_sha256": "a" * 64, "native_evidence_sha256": "b" * 64, "pool": "train", "level": 1,
                     "seed": 1001, "heldout_accessed": False, "state": self.state, "menu": self.menu,
                     "prompt_sha256": prompt_sha256(self.state, self.menu),
                     "messages_sha256": sha256_data({"system": self.request["system"], "user": user})}
        self._write()
        self.root_patch = patch.object(helper, "ROOT", self.root)
        self.preflight_patch = patch.object(helper, "preflight", return_value={"ok": True, "pins_sha256": "a" * 64, "evidence_sha256": "b" * 64})
        self.root_patch.start()
        self.preflight_patch.start()

    def tearDown(self):
        self.preflight_patch.stop()
        self.root_patch.stop()
        self.temp.cleanup()

    def _write(self):
        (self.directory / "plan.json").write_text(json.dumps(self.plan), encoding="utf-8")
        (self.directory / "requests.json").write_text(json.dumps(self.requests), encoding="utf-8")

    def test_matched_native_scope_and_max_six_batch(self):
        self.assertEqual(5, helper.inspect("8b", self.directory)["max_requests"])
        self.requests += [dict(self.request, id="x"), dict(self.request, id="y")]
        self._write()
        with self.assertRaises(GateRefused):
            helper.inspect("8b", self.directory)

    def test_modified_prompt_or_scope_is_refused(self):
        self.requests[0]["system"] += " changed"
        self._write()
        with self.assertRaises(GateRefused):
            helper.inspect("24b", self.directory)
        self.requests[0]["system"] = self.request["system"]
        self.plan["pool"] = "heldout"
        self._write()
        with self.assertRaises(GateRefused):
            helper.inspect("24b", self.directory)

    def test_verified_result_requires_full_distribution_exact_gpu_and_no_updates(self):
        model = GPU_MODELS["8b"]
        identity = f"{model['repo']}@{model['revision']}:nf4-bf16:instruction-base:no-adapter"
        responses = [{"id": request["id"], "labels": request["labels"], "choice": "A", "p": [1.0], "logp": [0.0],
                      "prompt_sha256": self.plan["prompt_sha256"], "model": identity, "complete_legal_distribution": True,
                      "gpu": {"uuid": model["gpu_uuid"]}, "adapter": None, "learning_updates": 0} for request in self.requests]
        raw = {"responses": responses, "learning_updates": 0}
        output = self.directory / "8b.json"
        output.write_text(json.dumps(raw), encoding="utf-8")
        self.assertTrue(helper.verify_result("8b", self.directory)["verified"])
        with self.assertRaises(GateRefused):
            helper.inspect("8b", self.directory)
        raw["responses"][0]["gpu"]["uuid"] = "wrong"
        output.write_text(json.dumps(raw), encoding="utf-8")
        with self.assertRaises(GateRefused):
            helper.verify_result("8b", self.directory)


if __name__ == "__main__":
    unittest.main()
