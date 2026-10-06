"""CPU mathematical audit of shared bounded-training helpers, never a GPU run."""
from __future__ import annotations

from contextlib import contextmanager
import copy
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import unittest

from alpharush_rl.bounded_llm_training import full_objective_gate, independent_reference, supervisor_acceptance
from alpharush_rl.journal import sha256_data
from alpharush_rl.ops import GateRefused, sha256_file


HAS_TORCH = importlib.util.find_spec("torch") is not None
if HAS_TORCH:
    import torch

    class ReferenceToyModel(torch.nn.Module):
        def __init__(self, output_shape="suffix"):
            super().__init__()
            generator = torch.Generator(device="cpu").manual_seed(501)
            self.embedding = torch.nn.Embedding(9, 2, dtype=torch.float64)
            self.base = torch.nn.Parameter(torch.randn(9, 9, generator=generator, dtype=torch.float64) * 0.4,
                                           requires_grad=False)
            self.adapter = torch.nn.Parameter(torch.randn(9, 9, generator=generator, dtype=torch.float64) * 0.3)
            self.adapter_enabled = True
            self.output_shape = output_shape
            self.calls = []

        def get_input_embeddings(self):
            return self.embedding

        @contextmanager
        def disable_adapter(self):
            previous = self.adapter_enabled
            self.adapter_enabled = False
            try:
                yield
            finally:
                self.adapter_enabled = previous

        def forward(self, input_ids, use_cache, logits_to_keep):
            self.calls.append({"ids": input_ids[0].tolist(), "use_cache": use_cache,
                               "logits_to_keep": logits_to_keep, "adapter_enabled": self.adapter_enabled,
                               "grad_enabled": torch.is_grad_enabled(), "training": self.training})
            table = self.base + self.adapter if self.adapter_enabled else self.base
            logits = table[input_ids]
            if self.output_shape == "suffix":
                logits = logits[:, -logits_to_keep:, :]
            elif self.output_shape == "bad":
                logits = logits[:, -1:, :]
            return SimpleNamespace(logits=logits)


class ReferenceTokenizer:
    eos_token_id = 4

    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        if not tokenize or not add_generation_prompt:
            raise AssertionError("reference requires the complete generation prefix")
        return {"input_ids": [1, 2, 3]}

    def encode(self, label, add_special_tokens):
        if add_special_tokens:
            raise AssertionError("label special tokens would change the suffix")
        return {"A": [5], "AA": [5, 5], "B": [6]}[label]


class CountingContext:
    def __init__(self):
        self.calls = 0

    def check(self):
        self.calls += 1


def metrics(advantage=0.0, fork_kl=0.0, anchor_kl=0.0):
    return {"fork": {"expected_advantage": advantage, "kl": fork_kl}, "anchors": {"kl": anchor_kl}}


class FullObjectiveGateTests(unittest.TestCase):
    def test_import_does_not_load_training_libraries(self):
        code = ("import sys; import alpharush_rl.bounded_llm_training; "
                "print('torch' in sys.modules, 'peft' in sys.modules, 'transformers' in sys.modules)")
        completed = subprocess.run([sys.executable, "-c", code], check=True, capture_output=True, text=True)
        self.assertEqual(completed.stdout.strip(), "False False False")

    def test_default_24b_cli_inspection_cannot_import_models_or_query_any_process(self):
        # Run the real default CLI in a fresh interpreter. Pending pins/data may
        # yield ok=False; read-only inspection must still return honest fields.
        code = """
import builtins, contextlib, io, json, runpy, subprocess, sys
original_import = builtins.__import__
def guarded_import(name, *args, **kwargs):
    if name.split('.')[0] in ('torch', 'peft', 'transformers'):
        raise AssertionError('default CLI imported a model runtime: ' + name)
    return original_import(name, *args, **kwargs)
builtins.__import__ = guarded_import
def forbid_process(*args, **kwargs):
    raise AssertionError('default CLI attempted an external process or GPU query')
subprocess.Popen = forbid_process
path = sys.argv[1]
cli = runpy.run_path(path, run_name='audit_default_24b_cli')
sys.argv = [path]
buffer = io.StringIO()
with contextlib.redirect_stdout(buffer):
    exit_code = cli['main']()
print(json.dumps({'exit_code': exit_code, 'result': json.loads(buffer.getvalue()),
                  'loaded_models': [name for name in ('torch', 'peft', 'transformers') if name in sys.modules]}))
"""
        completed = subprocess.run([sys.executable, "-c", code,
                                    str(Path(__file__).resolve().parents[1] / "tools/train-level1-24b.py")],
                                   check=True, capture_output=True, text=True)
        result = json.loads(completed.stdout)
        self.assertIn(result["exit_code"], (0, 2))
        self.assertEqual(result["loaded_models"], [])
        report = result["result"]
        self.assertIsInstance(report["ok"], bool)
        self.assertTrue(report["check_only"])
        self.assertFalse(report["gpu_queried"])
        self.assertFalse(report["gpu_started"])
        self.assertEqual(report["optimizer_steps"], 0)
        self.assertFalse(report["long_training_enabled"])

    def test_advantage_improvement_cannot_hide_a_worse_kl_penalized_objective(self):
        report = {"before": metrics(0.1, 0.1, 0.1), "after": metrics(0.11, 0.9, 0.4)}
        gate = full_objective_gate(report, 0.2, 0.3)
        self.assertGreater(report["after"]["fork"]["expected_advantage"], report["before"]["fork"]["expected_advantage"])
        self.assertFalse(gate["passed"])
        self.assertGreater(gate["delta"], 0)
        self.assertAlmostEqual(gate["before"], -0.05)
        self.assertAlmostEqual(gate["after"], 0.19)

    def test_objective_decrease_and_equality_pass_but_any_increase_fails(self):
        for after, passed in ((metrics(1e-10), True), (metrics(), True), (metrics(-1e-15), False)):
            with self.subTest(after=after):
                gate = full_objective_gate({"before": metrics(), "after": after}, 0.2, 0.3)
                self.assertEqual(gate["passed"], passed)
                self.assertEqual(gate["improved"], gate["delta"] < 0)
                self.assertEqual(gate["tolerance"], 0.0)

    def test_incomplete_metrics_fail_closed_with_a_reason(self):
        reports = [{}, {"before": metrics()}, {"before": {}, "after": metrics()},
                   {"before": metrics(), "after": {"fork": {"expected_advantage": 1, "kl": 0}}},
                   {"before": metrics(), "after": {"fork": {"expected_advantage": 1}, "anchors": {"kl": 0}}}]
        for report in reports:
            with self.subTest(report=report):
                gate = full_objective_gate(report, 0.2, 0.3)
                self.assertFalse(gate["passed"])
                self.assertTrue(gate.get("reason"))

    def test_nonfinite_metrics_are_rejected(self):
        for field in ("expected_advantage", "fork_kl", "anchor_kl"):
            for value in (math.nan, math.inf, -math.inf):
                values = {"advantage": 0.0, "fork_kl": 0.0, "anchor_kl": 0.0}
                values["advantage" if field == "expected_advantage" else field] = value
                with self.subTest(field=field, value=value):
                    gate = full_objective_gate({"before": metrics(), "after": metrics(**values)}, 0.2, 0.3)
                    self.assertFalse(gate["passed"])
                    self.assertTrue(gate.get("reason"))

    def test_worker_acceptance_requires_clean_supervisor_exit_and_complete_one_step_evidence(self):
        report = {"accepted": True, "status": "accepted_reward_step", "optimizer_steps": 1,
                  "parameter_update_verified": True, "full_objective_gate": {"passed": True, "improved": False}}
        closing = {"exit_code": 0, "exited": True, "cleanup_error": None, "error": None, "wall_seconds": 1199.0}
        self.assertTrue(supervisor_acceptance(report, **closing))
        for field, value in (("exit_code", 1), ("exit_code", -9), ("exited", False),
                             ("cleanup_error", "GPU lock removal failed"), ("error", "deadline reached"),
                             ("wall_seconds", 1200.0001), ("wall_seconds", -1),
                             ("wall_seconds", math.nan), ("wall_seconds", math.inf)):
            with self.subTest(close_field=field, value=value):
                self.assertFalse(supervisor_acceptance(report, **{**closing, field: value}))
        for fields in ({"optimizer_steps": 0}, {"optimizer_steps": 2}, {"optimizer_steps": "unknown"},
                       {"parameter_update_verified": False}, {"full_objective_gate": {"passed": False}},
                       {"full_objective_gate": {}}, {"accepted": False}, {"status": "rejected_reward_step"}):
            with self.subTest(report_fields=fields):
                self.assertFalse(supervisor_acceptance({**copy.deepcopy(report), **fields}, **closing))
        missing = copy.deepcopy(report)
        del missing["parameter_update_verified"]
        self.assertFalse(supervisor_acceptance(missing, **closing))
        self.assertFalse(supervisor_acceptance(None, **closing))


@unittest.skipUnless(HAS_TORCH, "CPU Torch tests run in the isolated WSL runtime")
class IndependentReferenceTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.tokenizer = ReferenceTokenizer()
        self.request = {"id": "cpu-reference-a-aa-eos", "system": "Select one legal option",
                        "user": "完整原生状态 fixture", "labels": ["A", "AA", "B"]}
        self.identity = {"model": "test-24b-base", "base_revision": "revision-24b-test",
                         "identity_hashes": {"tokenizer": "a" * 64, "base": "b" * 64},
                         "model_manifest_sha256": "c" * 64, "gpu_uuid": "GPU-dummy-5090-fixture"}

    def test_full_and_suffix_outputs_match_the_independent_causal_eos_sum(self):
        response_by_shape = []
        for shape in ("full", "suffix"):
            with self.subTest(shape=shape):
                model = ReferenceToyModel(shape)
                adapter_before = model.adapter.detach().clone()
                ctx = CountingContext()
                response = independent_reference(model, self.tokenizer, [self.request], self.identity, ctx, 32)[0]
                table = model.base.float().log_softmax(-1)
                expected_scores = torch.stack([table[3, 5] + table[5, 4],
                                               table[3, 5] + table[5, 5] + table[5, 4],
                                               table[3, 6] + table[6, 4]])
                expected_logp = expected_scores.double().log_softmax(0)
                torch.testing.assert_close(torch.tensor(response["sequence_log_likelihood"]), expected_scores, rtol=0, atol=0)
                self.assertEqual(response["p"], expected_logp.exp().tolist())
                self.assertEqual(response["logp"], expected_logp.tolist())
                self.assertEqual(response["label_token_ids"], {"A": [5, 4], "AA": [5, 5, 4], "B": [6, 4]})
                self.assertEqual([call["ids"] for call in model.calls], [[1, 2, 3, 5], [1, 2, 3, 5, 5], [1, 2, 3, 6]])
                self.assertEqual([call["logits_to_keep"] for call in model.calls], [2, 3, 2])
                self.assertTrue(all(call["use_cache"] is False and call["adapter_enabled"] is False
                                    and call["grad_enabled"] is False and call["training"] is False for call in model.calls))
                self.assertEqual(ctx.calls, 3)
                self.assertTrue(model.adapter_enabled)
                torch.testing.assert_close(model.adapter.detach(), adapter_before, rtol=0, atol=0)
                self.assertTrue(all(parameter.grad is None for parameter in model.parameters()))
                response_by_shape.append(response)
        self.assertEqual(response_by_shape[0]["p"], response_by_shape[1]["p"])

    def test_reference_metadata_is_bound_to_the_exact_prompt_tokens_and_passed_identity(self):
        model = ReferenceToyModel()
        response = independent_reference(model, self.tokenizer, [self.request], self.identity, CountingContext(), 32)[0]
        self.assertEqual(response["prompt_sha256"], hashlib.sha256(self.request["user"].encode("utf-8")).hexdigest())
        self.assertEqual(response["prompt_token_ids"], [1, 2, 3])
        self.assertEqual(response["prompt_token_ids_sha256"], sha256_data([1, 2, 3]))
        self.assertEqual(response["prompt_tokens"], 3)
        self.assertEqual(response["model"], self.identity["model"])
        self.assertEqual(response["model_revision"], self.identity["base_revision"])
        for field in ("identity_hashes", "model_manifest_sha256", "gpu_uuid"):
            self.assertEqual(response[field], self.identity[field])
        self.assertIsNone(response["adapter"])
        self.assertEqual(response["learning_updates"], 0)
        self.assertTrue(response["complete_legal_distribution"])
        self.assertEqual(response["scoring_execution"], "canonical_teacher_forced_no_cache_v2")
        self.assertEqual(response["reference_source"], "independent_frozen_base_disable_adapter_full_teacher_forcing")
        self.assertEqual(response["reference_scorer_file_sha256"],
                         sha256_file(Path(__file__).resolve().parents[1] / "alpharush_rl/bounded_llm_training.py"))

    def test_unestablished_causal_shape_and_prompt_truncation_are_rejected(self):
        with self.assertRaisesRegex(GateRefused, "full causal suffix shift"):
            independent_reference(ReferenceToyModel("bad"), self.tokenizer, [self.request], self.identity, CountingContext(), 32)
        model = ReferenceToyModel()
        with self.assertRaisesRegex(GateRefused, "no truncation"):
            independent_reference(model, self.tokenizer, [self.request], self.identity, CountingContext(), 2)
        self.assertEqual(model.calls, [])
        self.assertTrue(model.adapter_enabled)


if __name__ == "__main__":
    unittest.main()
