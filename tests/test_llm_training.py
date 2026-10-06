"""CPU toy causal/LoRA math tests; none constitutes an 8B/24B optimizer smoke."""
import copy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest

from alpharush_rl.llm_training import (accumulate_two_pass_gradient, check_only,
    differentiable_distribution, differentiable_label_scores, full_legal_pi_adv_loss,
    one_step_smoke, prepare_lora, exact_reference_probabilities, _append_report_receipt)
from alpharush_rl.journal import Journal, canonical_bytes, sha256_data
from alpharush_rl.trainer import EligibilityError

HAS_TORCH = importlib.util.find_spec("torch") is not None
if HAS_TORCH:
    import torch

    class ToyCausalLoRA(torch.nn.Module):
        def __init__(self):
            super().__init__()
            rng = torch.Generator(device="cpu").manual_seed(27)
            self.base = torch.nn.Parameter(torch.randn(9, 9, dtype=torch.float64, generator=rng) * 0.2,
                                           requires_grad=False)
            self.lora_A = torch.nn.Parameter(torch.randn(9, 2, dtype=torch.float64, generator=rng) * 0.1)
            self.lora_B = torch.nn.Parameter(torch.randn(2, 9, dtype=torch.float64, generator=rng) * 0.1)
            self.config = SimpleNamespace(attention_dropout=0.0)

        def forward(self, input_ids, use_cache, logits_to_keep):
            assert use_cache is False
            table = self.base + self.lora_A @ self.lora_B
            return SimpleNamespace(logits=table[input_ids][:, -logits_to_keep:, :])


class ToyTokenizer:
    eos_token_id = 4

    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        return {"input_ids": [1, 2, 3]}

    def encode(self, label, add_special_tokens):
        return {"A": [5], "AA": [5, 5], "B": [6]}[label]


def request_fixture():
    return {"id": "cpu-toy-only", "system": "Choose a label", "user": "mathematical fixture",
            "labels": ["A", "AA", "B"]}


class LazyAndPermitTests(unittest.TestCase):
    def test_saved_receipt_remains_hash_valid_after_report_gains_its_receipt_id(self):
        with tempfile.TemporaryDirectory() as temp:
            journal = Journal(Path(temp) / "updates.jsonl")
            report = {"status": "rejected", "optimizer_steps": 0,
                      "reference": {"preserved": True}}
            receipt = _append_report_receipt(journal, "llm_smoke_receipt", report)
            report["receipt_sha256"] = receipt["sha256"]
            report["reference"]["preserved"] = False
            destination = Path(temp) / "UPDATE-RECEIPT.json"
            destination.write_bytes(canonical_bytes(receipt) + b"\n")
            saved = json.loads(destination.read_text())
            self.assertEqual(saved, journal.entries()[-1])
            unsigned = {key: value for key, value in saved.items() if key != "sha256"}
            self.assertEqual(sha256_data(unsigned), saved["sha256"])
            self.assertNotIn("receipt_sha256", saved["payload"])
            self.assertTrue(saved["payload"]["reference"]["preserved"])
            self.assertTrue(journal.verify(saved["sha256"])["integrity_verified"])

    def test_import_does_not_import_torch_or_peft(self):
        completed = subprocess.run([sys.executable, "-c",
            "import sys; import alpharush_rl.llm_training; print('torch' in sys.modules, 'peft' in sys.modules)"],
            check=True, capture_output=True, text=True)
        self.assertEqual(completed.stdout.strip(), "False False")

    def test_default_is_check_only_and_tiny_reference_cannot_substitute(self):
        result = one_step_smoke(None, None, {}, {"model": "cpu-tiny-tanh:fixture"})
        self.assertTrue(result["check_only"])
        self.assertEqual(result["optimizer_steps"], 0)
        self.assertFalse(result["gpu_started"])
        self.assertFalse(result["llm_training_validated"])
        self.assertEqual(result["status"], "blocked")
        self.assertFalse(prepare_lora()["adapter_created"])

    def test_explicit_update_still_requires_external_launcher_permit(self):
        with self.assertRaises(EligibilityError):
            one_step_smoke(None, None, {}, {}, {"learning_rate": 3e-7}, bounded_smoke=True)


@unittest.skipUnless(HAS_TORCH, "run CPU toy tests in isolated WSL Torch runtime")
class DifferentiableCausalTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.model = ToyCausalLoRA().eval()
        self.tokenizer = ToyTokenizer()
        self.request = request_fixture()

    def test_complete_A_AA_EOS_sequences_and_causal_shift(self):
        dist = differentiable_distribution(self.model, self.tokenizer, self.request)
        table = (self.model.base + self.model.lora_A @ self.model.lora_B).log_softmax(-1)
        expected = torch.stack([table[3, 5] + table[5, 4],
                                table[3, 5] + table[5, 5] + table[5, 4],
                                table[3, 6] + table[6, 4]])
        torch.testing.assert_close(dist["scores"], expected, atol=1e-12, rtol=0)
        self.assertEqual(dist["tokenized"]["suffixes"], [[5, 4], [5, 5, 4], [6, 4]])
        self.assertNotEqual(float(dist["scores"][0].detach()), float(dist["scores"][1].detach()))
        self.assertTrue(dist["logp"].requires_grad)
        self.assertAlmostEqual(float(dist["p"].detach().sum()), 1.0)

    def test_full_objective_finite_difference_all_adapter_parameters(self):
        reference_p = torch.tensor([0.3, 0.45, 0.25], dtype=torch.float64, requires_grad=True)
        original_p0 = reference_p.detach().clone()
        advantage = [-0.8, 1.2, -0.8]

        def objective():
            dist = differentiable_distribution(self.model, self.tokenizer, self.request)
            return full_legal_pi_adv_loss(dist["logp"], advantage, reference_p, 0.21)[0]

        objective().backward()
        self.assertIsNone(reference_p.grad)
        torch.testing.assert_close(reference_p.detach(), original_p0, atol=0, rtol=0)
        epsilon = 1e-6
        for name, parameter in self.model.named_parameters():
            if not parameter.requires_grad:
                self.assertIsNone(parameter.grad)
                continue
            analytic = parameter.grad.detach().clone()
            for row in range(parameter.shape[0]):
                for col in range(parameter.shape[1]):
                    original = float(parameter[row, col].detach())
                    with torch.no_grad():
                        parameter[row, col] = original + epsilon
                    high = float(objective().detach())
                    with torch.no_grad():
                        parameter[row, col] = original - epsilon
                    low = float(objective().detach())
                    with torch.no_grad():
                        parameter[row, col] = original
                    self.assertAlmostEqual(float(analytic[row, col]), (high - low) / (2 * epsilon),
                                           delta=2e-8, msg=f"causal adapter gradient {name}[{row},{col}]")

    def test_exact_two_pass_matches_full_autograd_and_frozen_p0(self):
        twin = copy.deepcopy(self.model)
        p0 = torch.tensor([0.3, 0.45, 0.25], dtype=torch.float64, requires_grad=True)
        advantage = [-0.8, 1.2, -0.8]
        dist = differentiable_distribution(self.model, self.tokenizer, self.request)
        loss, _ = full_legal_pi_adv_loss(dist["logp"], advantage, p0, 0.21)
        loss.backward()
        measured = accumulate_two_pass_gradient(twin, self.tokenizer, self.request, advantage, p0, 0.21,
                                                checkpoint_options=True)
        self.assertAlmostEqual(measured["loss"], float(loss.detach()), places=12)
        self.assertEqual(measured["gradient_method"], "exact_two_pass_complete_legal_sequence_scores")
        self.assertEqual(measured["recompute_max_score_delta"], 0.0)
        for (name, parameter), (other_name, other_parameter) in zip(self.model.named_parameters(), twin.named_parameters()):
            self.assertEqual(name, other_name)
            if parameter.requires_grad:
                torch.testing.assert_close(parameter.grad, other_parameter.grad, atol=1e-12, rtol=1e-12)
            else:
                self.assertIsNone(other_parameter.grad)
        self.assertIsNone(p0.grad)

    def test_reused_calibration_scores_preserve_exact_complete_gradient(self):
        twin = copy.deepcopy(self.model)
        p0 = [0.3, 0.45, 0.25]
        with torch.no_grad():
            cached = differentiable_distribution(twin, self.tokenizer, self.request)["scores"].tolist()
        dist = differentiable_distribution(self.model, self.tokenizer, self.request)
        loss, _ = full_legal_pi_adv_loss(dist["logp"], [-0.8, 1.2, -0.8], p0, 0.21)
        loss.backward()
        measured = accumulate_two_pass_gradient(twin, self.tokenizer, self.request, [-0.8, 1.2, -0.8], p0, 0.21,
                                                first_scores=cached, checkpoint_options=True)
        self.assertTrue(measured["reused_step0_calibration_scores"])
        for parameter, other in zip(self.model.parameters(), twin.parameters()):
            if parameter.requires_grad:
                torch.testing.assert_close(parameter.grad, other.grad, atol=1e-12, rtol=1e-12)

    def test_exact_anchor_equality_has_no_small_delta_shortcut(self):
        p = [0.3, 0.45, 0.25]
        self.assertTrue(exact_reference_probabilities(p, p))
        changed = [0.3 + 1e-15, 0.45 - 1e-15, 0.25]
        self.assertFalse(exact_reference_probabilities(changed, p))
        self.assertFalse(exact_reference_probabilities([0.3, 0.45], p))
        self.assertFalse(exact_reference_probabilities([0, 0.5, 0.5], [0, 0.5, 0.5]))
        self.assertFalse(exact_reference_probabilities([0.3, 0.3], [0.3, 0.3]))

    def test_reused_stale_score_refuses_before_optimizer_update(self):
        with torch.no_grad():
            cached = differentiable_distribution(self.model, self.tokenizer, self.request)["scores"].tolist()
        cached[0] += 0.01
        before = self.model.lora_B.detach().clone()
        with self.assertRaises(ValueError):
            accumulate_two_pass_gradient(self.model, self.tokenizer, self.request, [-0.8, 1.2, -0.8], [0.3, 0.45, 0.25], 0.21,
                                         first_scores=cached)
        torch.testing.assert_close(before, self.model.lora_B.detach(), atol=0, rtol=0)

    def test_flat_group_zero_pg_preserves_kl_gradient(self):
        dist = differentiable_distribution(self.model, self.tokenizer, self.request)
        loss, metrics = full_legal_pi_adv_loss(dist["logp"], [0, 0, 0], [0.3, 0.45, 0.25], 0.21)
        self.assertEqual(float(metrics["expected_advantage"].detach()), 0.0)
        self.assertAlmostEqual(float(loss.detach()), 0.21 * float(metrics["kl"].detach()))
        loss.backward()
        self.assertGreater(float(self.model.lora_B.grad.abs().sum()), 0)

    def test_active_dropout_is_rejected_in_two_pass(self):
        self.model.dropout = torch.nn.Dropout(0.1)
        self.model.train()
        with self.assertRaises(EligibilityError):
            accumulate_two_pass_gradient(self.model, self.tokenizer, self.request, [0, 1, 0], [0.3, 0.45, 0.25], 0.21)

    def test_toy_never_initializes_cuda(self):
        self.assertFalse(torch.cuda.is_initialized())


if __name__ == "__main__":
    unittest.main()
