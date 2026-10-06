"""Operator network: features, gradients, imitation of a plan executor on the toy level; no game is started."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from alpharush_rl.episode import run_episode
from alpharush_rl.operator_net import (G_DIM, O_DIM, OperatorPolicy, OptionScorer, Recorder, accuracy,
                                       decision_arrays, load_rows, save_rows, train)
from alpharush_rl.search import BuildOrderPolicy

from test_search import PROTOCOL, ToyEnv, genome


PLAN = genome([["b", "02", "mage"], ["u", "02"], ["b", "01", "archer"], ["u", "02"], ["b", "03", "barrack"]])


class FeatureTests(unittest.TestCase):
    def test_dimensions_and_instruction_matches(self):
        env = ToyEnv(gold=600)
        state = env.reset()
        from alpharush_rl.menus import build_menu
        menu = build_menu(state)
        g, o = decision_arrays(state, menu, {"step": ["b", "02", "mage"], "cast": 50, "early": 1})
        self.assertEqual((G_DIM,), g.shape)
        self.assertEqual((len(menu), O_DIM), o.shape)
        exact = [i for i, item in enumerate(menu) if item["action"] == {"action": "build_tower", "holder_id": 2,
                                                                         "tower_type": "mage"}]
        self.assertEqual(1, len(exact))
        self.assertEqual(1.0, o[exact[0], -1])  # same op on the same holder
        self.assertEqual(1.0, o[exact[0], -2])  # same kind
        g0, _ = decision_arrays(state, menu, None)
        self.assertEqual(0.0, g0[-1])


class GradientTests(unittest.TestCase):
    def test_analytic_gradients_match_finite_differences(self):
        rng = np.random.default_rng(3)
        net = OptionScorer(hidden=(8,), seed=1)
        batch = [(rng.normal(size=G_DIM).astype(np.float32), rng.normal(size=(n, O_DIM)).astype(np.float32),
                  int(rng.integers(n)), 1.0) for n in (3, 5, 2)]
        for p in net.params:
            p[...] = p.astype(np.float64)
        net.params = [p.astype(np.float64) for p in net.params]
        batch = [(g.astype(np.float64), o.astype(np.float64), t, w) for g, o, t, w in batch]
        loss, grads = net.loss_and_grads(batch)
        for index in (0, 1, 2):
            p = net.params[index]
            flat = p.reshape(-1)
            for j in rng.choice(flat.size, size=min(5, flat.size), replace=False):
                old = flat[j]
                flat[j] = old + 1e-5
                up, _ = net.loss_and_grads(batch)
                flat[j] = old - 1e-5
                down, _ = net.loss_and_grads(batch)
                flat[j] = old
                self.assertAlmostEqual((up - down) / 2e-5, grads[index].reshape(-1)[j], places=4)


class ImitationTests(unittest.TestCase):
    def test_record_train_and_play_on_toy_level(self):
        rows = []
        for seed in (1001, 1002, 1003):
            recorder = Recorder(BuildOrderPolicy(PLAN))
            result = run_episode(ToyEnv(seed=seed, gold=600), recorder, PROTOCOL, seed=seed, level=1)
            self.assertEqual(len(result["decisions"]), len(recorder.rows))
            rows += recorder.rows
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "episode.npz"
            save_rows(path, rows, {"level": 1})
            loaded, meta = load_rows(path)
        self.assertEqual({"level": 1}, meta)
        self.assertEqual(len(rows), len(loaded))
        self.assertTrue(all(np.array_equal(a[1], b[1]) and a[2] == b[2] for a, b in zip(rows, loaded)))
        decisions = [(g, o, t, 1.0) for g, o, t in loaded]
        net = OptionScorer(hidden=(32,), seed=0)
        history = train(net, decisions, epochs=60, batch=32, lr=3e-3)
        self.assertLess(history[-1], history[0])
        self.assertGreater(accuracy(net, decisions), 0.9)
        expert = run_episode(ToyEnv(seed=1004, gold=600), BuildOrderPolicy(PLAN), PROTOCOL, seed=1004, level=1)
        learned = run_episode(ToyEnv(seed=1004, gold=600), OperatorPolicy(net, PLAN), PROTOCOL, seed=1004, level=1)
        self.assertEqual("terminal", learned["status"])
        built = [d["action"] for d in learned["decisions"] if d["action"]["action"] == "build_tower"]
        self.assertEqual({"action": "build_tower", "holder_id": 2, "tower_type": "mage"}, built[0])
        self.assertTrue(all(d["provenance"] == "model:operator" for d in learned["decisions"]))
        self.assertEqual(expert["outcome"]["level_won"], learned["outcome"]["level_won"])

    def test_rows_and_weights_from_before_click_entity_still_load(self):
        import tempfile
        from pathlib import Path
        from alpharush_rl.operator_net import LEGACY_ACTIONS, load_rows
        net = OptionScorer(hidden=(16, 8), seed=2)
        old = net.to_json()
        old["o_dim"] = O_DIM - 1
        old["params"][0] = np.delete(np.asarray(old["params"][0]), G_DIM + LEGACY_ACTIONS, axis=0).tolist()
        again = OptionScorer.from_json(old)
        g = np.ones(G_DIM, dtype=np.float32)
        o = np.ones((3, O_DIM), dtype=np.float32)
        o[:, LEGACY_ACTIONS] = 0.0  # no click options in old data
        self.assertTrue(np.allclose(net.scores(g, o), again.scores(g, o), atol=1e-5))  # the column is zero
        self.assertEqual(net.params[0].shape, again.params[0].shape)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "old.npz"
            o_old = np.delete(o, LEGACY_ACTIONS, axis=1)
            np.savez_compressed(path, g=g[None, :], o=o_old, offsets=np.asarray([0, 3]), target=np.asarray([1]),
                                meta=np.asarray('{"seed": 1001}'))
            rows, meta = load_rows(path)
            self.assertEqual((3, O_DIM), rows[0][1].shape)
            self.assertTrue(np.allclose(rows[0][1], o))

    def test_weights_round_trip(self):
        net = OptionScorer(hidden=(16, 8), seed=2)
        again = OptionScorer.from_json(net.to_json())
        g, o = np.ones(G_DIM, dtype=np.float32), np.ones((4, O_DIM), dtype=np.float32)
        self.assertTrue(np.allclose(net.scores(g, o), again.scores(g, o)))



class DaggerTests(unittest.TestCase):
    def test_network_plays_while_the_executor_labels_its_states(self):
        from alpharush_rl.operator_net import DaggerRecorder
        expert_rows = []
        for seed in (1001, 1002):
            recorder = Recorder(BuildOrderPolicy(PLAN))
            run_episode(ToyEnv(seed=seed, gold=600), recorder, PROTOCOL, seed=seed, level=1)
            expert_rows += recorder.rows
        net = OptionScorer(hidden=(16,), seed=0)
        train(net, [(g, o, t, 1.0) for g, o, t in expert_rows], epochs=5, batch=32, lr=3e-3)
        dagger = DaggerRecorder(net, PLAN, beta=0.0, seed=3)
        result = run_episode(ToyEnv(seed=1003, gold=600), dagger, PROTOCOL, seed=1003, level=1)
        self.assertEqual(len(result["decisions"]), len(dagger.rows))
        self.assertEqual(len(dagger.rows), dagger.net_choices)
        self.assertTrue(all(d["provenance"] == "model:dagger" and d["meta"]["source"] == "network"
                            for d in result["decisions"]))
        # Labels are the plan executor's choices in the visited states, recorded as the training target.
        for decision, (_, _, target) in zip(result["decisions"], dagger.rows):
            self.assertEqual(decision["meta"]["expert_label"], decision["labels"][target])
        self.assertLessEqual(dagger.agreements, len(dagger.rows))
        mixed = DaggerRecorder(net, PLAN, beta=1.0, seed=3)
        run_episode(ToyEnv(seed=1003, gold=600), mixed, PROTOCOL, seed=1003, level=1)
        self.assertEqual(0, mixed.net_choices)
        expert_play = run_episode(ToyEnv(seed=1003, gold=600), BuildOrderPolicy(PLAN), PROTOCOL, seed=1003, level=1)
        self.assertEqual(expert_play["outcome"], run_episode(ToyEnv(seed=1003, gold=600), DaggerRecorder(
            net, PLAN, beta=1.0, seed=3), PROTOCOL, seed=1003, level=1)["outcome"])


if __name__ == "__main__":
    unittest.main()
