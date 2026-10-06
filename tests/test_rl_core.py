"""Mathematical/engineering fixtures never qualify as native gameplay evidence."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from alpharush_rl.journal import Journal, JournalError, canonical_bytes, snapshot_hash
from alpharush_rl.imitation import cross_entropy_loss_and_grad, export_dagger_relabel, merge_dagger_labels, train_imitation
from alpharush_rl.menus import build_menu, option_label, validate_menu
from alpharush_rl.policies import BrokerAdapter, TinyOptionPolicy, masked_softmax, option_features, resolve_choice
from alpharush_rl.pools import HygieneError, PoolRegistry, audit_dataset, claim_heldout_once
from alpharush_rl.trainer import EligibilityError, group_advantages, pi_adv_loss_and_grad, score_native_outcome, train_groups, validate_price_contract


def state_fixture(holders=1):
    native = {"gold": 200, "lives": 20, "wave": 0, "wave_total": 6, "tick": 0,
              "holders": [{"id": index + 1, "x": 100 + index, "y": 200,
                           "path_score": 10, "blocked": False} for index in range(holders)],
              "towers": [], "enemies": [], "heroes": [], "wave_ready": True,
              "action_catalog": []}
    for holder in native["holders"]:
        native["action_catalog"].extend([
            {"action": "build_tower", "holder_id": holder["id"], "tower_type": "archer", "cost": 70, "available": True},
            {"action": "build_tower", "holder_id": holder["id"], "tower_type": "mage", "cost": 100, "available": True},
        ])
    return native


def registry_fixture():
    return {"pools": {"train": {"levels": [1], "seeds": [101, 102]},
                      "validation": {"levels": [2], "seeds": [201]},
                      "heldout": {"levels": [3], "seeds": [301]}},
            "nevertrain_levels": [3], "nevertrain_seeds": [301]}


class MenuAndPolicyTests(unittest.TestCase):
    def test_native_legality_missing_cost_blocked_and_unsupported(self):
        state = state_fixture(2)
        state["holders"][1]["blocked"] = True
        state["action_catalog"].append({"action": "use_power", "available": True, "cost": 0})
        state["action_catalog"].append({"action": "build_tower", "holder_id": 1, "tower_type": "barrack", "available": True})
        menu = build_menu(state)
        builds = [option for option in menu if option["action"]["action"] == "build_tower"]
        self.assertEqual(len(builds), 2)
        self.assertEqual({option["action"]["holder_id"] for option in builds}, {1})
        self.assertFalse(any(option["action"]["action"] == "use_power" for option in menu))
        state["gold"] = 0
        self.assertEqual(len(build_menu(state)), 2)  # wait + native ready wave
        state.pop("action_catalog")
        self.assertEqual(len(build_menu(state)), 2)

    def test_no_top26_truncation_and_stable_order(self):
        state = state_fixture(20)
        menu = build_menu(state)
        self.assertEqual(len(menu), 42)
        self.assertEqual(option_label(26), "AA")
        reordered = copy.deepcopy(state)
        reordered["action_catalog"].reverse()
        self.assertEqual(build_menu(reordered), menu)
        policy = TinyOptionPolicy()
        dist = policy.distribution(state, menu)
        self.assertEqual(len(dist["p"]), 42)
        self.assertAlmostEqual(sum(dist["p"]), 1.0)
        self.assertTrue(dist["complete_legal_distribution"])

    def test_legal_mask_is_exact_not_epsilon(self):
        p = masked_softmax(np.asarray([10000.0, 1.0, 2.0]), np.asarray([False, True, True]))
        self.assertEqual(p[0], 0.0)
        self.assertAlmostEqual(p.sum(), 1.0)
        with self.assertRaises(ValueError):
            masked_softmax(np.zeros(2), np.zeros(2, dtype=bool))

    def test_fallback_does_not_receive_model_credit(self):
        state = state_fixture()
        decision = resolve_choice("INVALID", state, build_menu(state))
        self.assertEqual(decision["provenance"], "rules_fallback")
        self.assertTrue(decision["fallback"])
        self.assertIsNone(decision["distribution"])

    def test_save_load_keeps_parameter_and_distribution_identity(self):
        state = state_fixture()
        policy = TinyOptionPolicy(seed=29)
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "policy.npz"
            policy.save(path)
            loaded = TinyOptionPolicy.load(path)
            self.assertEqual(policy.parameter_sha256(), loaded.parameter_sha256())
            self.assertEqual(policy.distribution(state, build_menu(state)), loaded.distribution(state, build_menu(state)))

    def test_broker_rejects_truncated_probability_table(self):
        state, adapter = state_fixture(), BrokerAdapter()
        menu = build_menu(state)
        request = adapter.request("test-1", state, menu)
        dist = TinyOptionPolicy().distribution(state, menu)
        response = {"id": "test-1", "choice": menu[0]["label"], "distribution": dist}
        self.assertEqual(adapter.validate_response(response, request), response)
        truncated = copy.deepcopy(response)
        truncated["distribution"]["labels"] = dist["labels"][:2]
        with self.assertRaises(ValueError):
            adapter.validate_response(truncated, request)


class ObjectiveTests(unittest.TestCase):
    def test_final_template_price_contract_catches_stale_cost(self):
        menu = build_menu(state_fixture())
        prices = {"templates": {"tower_archer_1": 70, "tower_mage_1": 100},
                  "build_templates": {"archer": "tower_archer_1", "mage": "tower_mage_1"}}
        validate_price_contract(menu, prices)
        broken = copy.deepcopy(menu)
        next(option for option in broken if option["action"]["action"] == "build_tower")["cost"] = 0
        with self.assertRaises(EligibilityError):
            validate_price_contract(broken, prices)

    def test_candidates_mean_baseline_and_uncontinued_minimum(self):
        adv = group_advantages(["A", "B", "C"], [{"label": "A", "return": 2.0}, {"label": "B", "return": -1.0}])
        np.testing.assert_array_equal(adv, [1.5, -1.5, -1.5])
        for candidates in ([{"label": "A", "return": 1.0}],
                           [{"label": "A", "return": 1.0}, {"label": "A", "return": -1.0}]):
            with self.assertRaises(EligibilityError):
                group_advantages(["A", "B", "C"], candidates)

    def test_flat_reward_group_keeps_zero_pg_and_reference_kl(self):
        state = state_fixture()
        features = option_features(state, build_menu(state))
        labels = [option["label"] for option in build_menu(state)]
        advantages = group_advantages(labels, [{"label": "A", "return": 0.0}, {"label": "B", "return": 0.0}])
        np.testing.assert_array_equal(advantages, np.zeros(len(labels)))
        policy, reference = TinyOptionPolicy(seed=1), TinyOptionPolicy(seed=7)
        p0, _ = reference.forward(features)
        loss, gradient, metrics = pi_adv_loss_and_grad(policy, features, advantages, p0, beta=0.37)
        self.assertEqual(metrics["expected_advantage"], 0.0)
        self.assertAlmostEqual(loss, 0.37 * metrics["kl"])
        self.assertGreater(loss, 0.0)
        self.assertGreater(sum(float(np.abs(value).sum()) for value in gradient.values()), 0.0)

    def test_pi_adv_and_kl_full_gradient_finite_difference(self):
        state = state_fixture()
        features = option_features(state, build_menu(state))
        policy = TinyOptionPolicy(seed=13, hidden_size=3)
        reference = TinyOptionPolicy(seed=21, hidden_size=3)
        p0, _ = reference.forward(features)
        advantages = np.linspace(-0.7, 1.2, len(features))
        loss, gradients, metrics = pi_adv_loss_and_grad(policy, features, advantages, p0, beta=0.37)
        p, _ = policy.forward(features)
        self.assertAlmostEqual(loss, -np.dot(p, advantages) + 0.37 * np.dot(p, np.log(p / p0)))
        self.assertAlmostEqual(metrics["expected_advantage"], np.dot(p, advantages))
        epsilon = 1e-6
        for key, values in policy.params.items():
            for index in np.ndindex(values.shape):
                original = values[index]
                values[index] = original + epsilon
                high = pi_adv_loss_and_grad(policy, features, advantages, p0, beta=0.37)[0]
                values[index] = original - epsilon
                low = pi_adv_loss_and_grad(policy, features, advantages, p0, beta=0.37)[0]
                values[index] = original
                self.assertAlmostEqual(gradients[key][index], (high - low) / (2 * epsilon), delta=2e-8,
                                       msg=f"gradient {key}{index}")

    def test_exact_cpu_update_moves_expected_advantage(self):
        state = state_fixture()
        features = option_features(state, build_menu(state))
        policy = TinyOptionPolicy(seed=1, hidden_size=4)
        p0, _ = policy.forward(features)
        advantages = np.asarray([-1, 1, -1, -1], dtype=np.float64)
        before = pi_adv_loss_and_grad(policy, features, advantages, p0)[2]["expected_advantage"]
        for _ in range(20):
            _, gradient, _ = pi_adv_loss_and_grad(policy, features, advantages, p0, beta=0.03)
            policy.update(gradient, 0.03)
        after = pi_adv_loss_and_grad(policy, features, advantages, p0)[2]["expected_advantage"]
        self.assertGreater(after, before + 1e-5)

    def test_native_timeout_is_invalid_not_zero_return(self):
        contract = {"name": "kr1-terminal-v1", "win": 1, "loss": -1, "lives_weight": 0.01}
        outcome = {"source": "native", "terminal": True, "level_won": True, "level_lost": False, "lives": 0}
        self.assertEqual(score_native_outcome(outcome, contract), 1.0)
        outcome["terminal"] = False
        with self.assertRaises(EligibilityError):
            score_native_outcome(outcome, contract)


class HygieneAndEvidenceTests(unittest.TestCase):
    def test_both_level_and_seed_leakage_and_nevertrain_rejected(self):
        self.assertTrue(PoolRegistry(registry_fixture()).audit()["passed"])
        for field, value in (("levels", 1), ("seeds", 101)):
            registry = registry_fixture()
            registry["pools"]["heldout"][field].append(value)
            with self.assertRaises(HygieneError):
                PoolRegistry(registry)
        registry = registry_fixture()
        registry["nevertrain_levels"].append(1)
        with self.assertRaises(HygieneError):
            PoolRegistry(registry)

    def test_training_anchors_cannot_come_from_validation(self):
        data = {"pool_registry": registry_fixture(), "groups": [],
                "anchors": {"A1": [{"pool": "validation", "level": 2, "seed": 201}]}}
        with self.assertRaises(HygieneError):
            audit_dataset(data)

    def test_validation_forks_never_contain_rewards(self):
        data = {"pool_registry": registry_fixture(), "groups": [], "anchors": {},
                "validation_forks": [{"pool": "validation", "level": 2, "seed": 201, "return": 0}]}
        with self.assertRaises(HygieneError):
            audit_dataset(data)

    def test_no_branch_data_keeps_training_blocked_and_reference_preserved(self):
        data = {"source": "real_game", "pool_registry": registry_fixture(), "groups": [], "anchors": {}}
        with tempfile.TemporaryDirectory() as temp:
            report = train_groups(data, {"steps": 2}, Path(temp) / "blocked")
            self.assertEqual(report["status"], "rejected")
            self.assertEqual(report["steps_completed"], 0)
            self.assertEqual(report["validation_kind"], "unexecuted")
            self.assertTrue(report["reference_preserved"])
            self.assertFalse(report["candidate_saved"])
            self.assertFalse((Path(temp) / "blocked" / "candidate.npz").exists())
            self.assertTrue((Path(temp) / "blocked" / "UPDATE-RECEIPT.json").is_file())

    def test_synthetic_fixtures_cannot_enter_native_training(self):
        with tempfile.TemporaryDirectory() as temp:
            report = train_groups({"source": "synthetic_unit_test", "groups": []}, {"steps": 2}, temp)
            self.assertEqual(report["status"], "rejected")
            self.assertIn("synthetic", report["reason"])
            self.assertEqual(report["validation_kind"], "unexecuted")

    def test_journal_tamper_and_tip_truncation_are_detected(self):
        with tempfile.TemporaryDirectory() as temp:
            target = Path(temp) / "journal.jsonl"
            journal = Journal(target)
            journal.append("decision", {"source": "rules", "state": {"tick": 0}})
            tip = journal.append("receipt", {"verified": False})["sha256"]
            self.assertFalse(journal.verify(tip)["engine_replay_verified"])
            original = target.read_bytes()
            lines = original.splitlines()
            target.write_bytes(lines[0] + b"\n")
            with self.assertRaises(JournalError):
                Journal(target).verify(tip)
            row = json.loads(lines[0])
            row["payload"]["state"]["tick"] = 1
            target.write_bytes(canonical_bytes(row) + b"\n")
            with self.assertRaises(JournalError):
                Journal(target)

    def test_journal_hashes_integer_keys_as_stored(self):
        # A campaign profile holds {level: stars}; from ten levels on, integer and string keys sort differently.
        with tempfile.TemporaryDirectory() as temp:
            journal = Journal(Path(temp) / "journal.jsonl")
            row = journal.append("campaign_attempt", {"profile": {"levels": {i: 3 for i in range(1, 12)}}})
            self.assertEqual({str(i): 3 for i in range(1, 12)}, row["payload"]["profile"]["levels"])
            journal.append("next", {"ok": True})
            self.assertEqual(2, Journal(Path(temp) / "journal.jsonl").verify()["entries"])

    def test_snapshot_orders_entities_preserves_clock_and_rng(self):
        first = {"tick": 1, "rng": 7, "towers": [{"id": 2}, {"id": 1}]}
        second = {"tick": 1, "rng": 7, "towers": [{"id": 1}, {"id": 2}]}
        self.assertEqual(snapshot_hash(first), snapshot_hash(second))
        second["tick"] = 2
        self.assertNotEqual(snapshot_hash(first), snapshot_hash(second))
        second["tick"], second["rng"] = 1, 8
        self.assertNotEqual(snapshot_hash(first), snapshot_hash(second))

    def test_heldout_judge_reservation_is_once_even_if_crashed(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "judgment.json"
            claim_heldout_once(path, "a" * 64, "b" * 64)
            with self.assertRaises(HygieneError):
                claim_heldout_once(path, "a" * 64, "c" * 64)


class ImitationTests(unittest.TestCase):
    def test_supervised_cpu_gradient_moves_expert_label_probability(self):
        state = state_fixture()
        features = option_features(state, build_menu(state))
        policy = TinyOptionPolicy(seed=4, hidden_size=4)
        p0, _ = policy.forward(features)
        _, gradient, before = cross_entropy_loss_and_grad(policy, features, 1, p0, beta=0.03)
        # Compare a representative full CE+KL gradient by finite difference.
        original, epsilon = policy.params["W2"][0], 1e-6
        policy.params["W2"][0] = original + epsilon
        high = cross_entropy_loss_and_grad(policy, features, 1, p0, beta=0.03)[0]
        policy.params["W2"][0] = original - epsilon
        low = cross_entropy_loss_and_grad(policy, features, 1, p0, beta=0.03)[0]
        policy.params["W2"][0] = original
        self.assertAlmostEqual(gradient["W2"][0], (high - low) / (2 * epsilon), delta=2e-8)
        for _ in range(20):
            _, gradient, _ = cross_entropy_loss_and_grad(policy, features, 1, p0, beta=0.03)
            policy.update(gradient, 0.03)
        after = cross_entropy_loss_and_grad(policy, features, 1, p0)[2]
        self.assertGreater(after["label_probability"], before["label_probability"])
        self.assertLess(after["cross_entropy"], before["cross_entropy"])

    def test_sft_rejects_validation_gradient_example(self):
        state = state_fixture()
        prices = {"templates": {"tower_archer_1": 70, "tower_mage_1": 100},
                  "build_templates": {"archer": "tower_archer_1", "mage": "tower_mage_1"}}
        from alpharush_rl.journal import sha256_data
        data = {"source": "real_game", "pool_registry": registry_fixture(),
                "price_contract": prices, "price_sha256": sha256_data(prices),
                "examples": [{"pool": "validation", "level": 2, "seed": 201, "state": state,
                              "menu": build_menu(state), "label": "B", "expert_source": "rules", "receipt_verified": True}]}
        with tempfile.TemporaryDirectory() as temp:
            report = train_imitation(data, {"steps": 2}, temp)
            self.assertEqual(report["status"], "rejected")
            self.assertEqual(report["steps_completed"], 0)
            self.assertTrue(report["reference_preserved"])
            self.assertFalse(report["candidate_saved"])

    def test_dagger_keeps_model_choice_unlabeled_until_explicit_oracle(self):
        state = state_fixture()
        decision = {"decision_id": "d1", "source": "real_game", "pool": "train", "level": 1,
                    "seed": 101, "state": state, "menu": build_menu(state), "choice": "A",
                    "provenance": "model", "receipt_verified": True}
        queue = export_dagger_relabel([decision], registry_fixture())
        self.assertIsNone(queue["examples"][0]["label"])
        self.assertFalse(queue["training_eligible"])
        relabeled = merge_dagger_labels(queue, [{"decision_id": "d1", "label": "B", "expert_source": "rules"}])
        self.assertEqual(relabeled["examples"][0]["executed_model_label"], "A")
        self.assertEqual(relabeled["examples"][0]["label"], "B")
        self.assertEqual(relabeled["examples"][0]["expert_source"], "rules")
        with self.assertRaises(EligibilityError):
            merge_dagger_labels(queue, [{"decision_id": "d1", "label": "B", "expert_source": "model"}])


if __name__ == "__main__":
    unittest.main()
