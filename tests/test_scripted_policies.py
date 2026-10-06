"""Scripted baselines: legal labels only, deterministic, state+menu inputs only."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import random
import unittest

from alpharush_rl.journal import canonical_json
from alpharush_rl.menus import build_menu, validate_menu
from alpharush_rl.scripted_policies import (PressureGreedy, RandomLegal, SendWaveOnly, SingleTowerType,
                                            make_policy, tower_counts)

COSTS = {"archer": 70, "barrack": 70, "mage": 100, "engineer": 125}


def state_fixture(scores=(10, 30, 30, 5), gold=500, wave_ready=True, towers=(), kinds=tuple(COSTS)):
    """Holder IDs start at 11; scores[i] is the path_score of holder 11+i."""
    native = {"gold": gold, "lives": 20, "wave": 0, "wave_total": 6, "tick": 0,
              "holders": [{"id": 11 + index, "x": 100 + index, "y": 200, "path_score": score,
                           "blocked": False} for index, score in enumerate(scores)],
              "towers": [{"id": 100 + index, "template": template, "holder_id": str(index)}
                         for index, template in enumerate(towers)],
              "enemies": [], "heroes": [], "wave_ready": wave_ready, "action_catalog": [],
              "level_path_wave_counts": [{"future_wave": 3, "hidden_group_count": 9}]}
    for holder in native["holders"]:
        native["action_catalog"].extend(
            {"action": "build_tower", "holder_id": holder["id"], "tower_type": kind,
             "cost": COSTS[kind], "available": True} for kind in kinds)
    return native


def context(index=0, seed=1001):
    return {"episode_id": "test", "decision_index": index, "level": 1, "seed": seed,
            "difficulty": 2, "tick": 0, "protocol": "kr1-episode-v1"}


def action_of(menu, label):
    return next(item["action"] for item in menu if item["label"] == label)


class ProtocolShapeTests(unittest.TestCase):
    SPECS = ("send_wave_only", "random:7", "single:archer", "single:barrack",
             "single:mage", "single:engineer", "pressure_greedy")

    def test_return_shape_and_provenance(self):
        state = state_fixture()
        menu = build_menu(state, wait_ticks=60)
        for spec in self.SPECS:
            with self.subTest(spec):
                policy = make_policy(spec)
                self.assertEqual(policy.name, spec)
                choice = policy.choose(state, menu, context())
                self.assertEqual(set(choice), {"label", "provenance", "distribution", "meta"})
                self.assertIn(choice["label"], validate_menu(menu))
                self.assertEqual(choice["provenance"], f"scripted:{spec}")
                self.assertIsNone(choice["distribution"])
                canonical_json(choice["meta"])  # JSON-serializable, no NaN/inf

    def test_only_legal_labels_on_random_states(self):
        rng = random.Random(20261005)
        all_kinds = tuple(COSTS)
        for trial in range(150):
            scores = [rng.choice([0, 1, 3, 3, 7, None]) for _ in range(rng.randint(0, 6))]
            towers = [rng.choice(["tower_archer_1", "tower_mage_2", "tower_build_barrack", "tower_engineer_1"])
                      for _ in range(rng.randint(0, 5))]
            kinds = tuple(k for k in all_kinds if rng.random() < 0.6)
            state = state_fixture(scores, gold=rng.choice([0, 70, 100, 500]), wave_ready=rng.random() < 0.5,
                                  towers=towers, kinds=kinds)
            menu = build_menu(state, wait_ticks=60)
            labels = validate_menu(menu)
            for spec in self.SPECS:
                with self.subTest(trial=trial, spec=spec):
                    before = copy.deepcopy(state)
                    first = make_policy(spec).choose(state, menu, context(trial))
                    self.assertIn(first["label"], labels)
                    self.assertEqual(state, before)  # never mutates the observation
                    self.assertEqual(make_policy(spec).choose(copy.deepcopy(state), copy.deepcopy(menu),
                                                              context(trial)), first)

    def test_malformed_menu_raises(self):
        menu = build_menu(state_fixture())
        broken = menu + [dict(menu[1])]
        for spec in self.SPECS:
            with self.subTest(spec), self.assertRaises(ValueError):
                make_policy(spec).choose(state_fixture(), broken, context())

    def test_non_random_policies_ignore_context(self):
        state = state_fixture()
        menu = build_menu(state)
        for spec in ("send_wave_only", "single:mage", "pressure_greedy"):
            policy = make_policy(spec)
            self.assertEqual(policy.choose(state, menu, context(0, 1)), policy.choose(state, menu, context(9, 5)))
            self.assertEqual(policy.choose(state, menu, {}), policy.choose(state, menu, context()))

    def test_make_policy_rejects_unknown_specs(self):
        for spec in ("", "random", "random:", "random:abc", "random:007", "random:+3", "random: 3",
                     "single:", "single:hero", "pressure", "send_wave_only:1", None, 3):
            with self.subTest(spec=spec), self.assertRaises(ValueError):
                make_policy(spec)
        self.assertEqual(make_policy("random:-4").name, "random:-4")
        with self.assertRaises(ValueError):
            RandomLegal(True)


class SendWaveOnlyTests(unittest.TestCase):
    def test_sends_wave_else_waits(self):
        policy = SendWaveOnly()
        state = state_fixture()
        menu = build_menu(state, wait_ticks=60)
        self.assertEqual(action_of(menu, policy.choose(state, menu, context())["label"]), {"action": "send_wave"})
        state["wave_ready"] = False
        menu = build_menu(state, wait_ticks=60)
        choice = policy.choose(state, menu, context())
        self.assertEqual(choice["label"], "A")
        self.assertEqual(action_of(menu, "A"), {"action": "wait", "ticks": 60})


class RandomLegalTests(unittest.TestCase):
    def test_seeded_by_seed_and_decision_index(self):
        state = state_fixture()
        menu = build_menu(state)
        labels = validate_menu(menu)
        policy = RandomLegal(7)
        picks = [policy.choose(state, menu, context(index))["label"] for index in range(60)]
        self.assertEqual(picks, [random.Random(f"7:{index}").choice(labels) for index in range(60)])
        self.assertEqual(picks, [RandomLegal(7).choose(state, menu, context(index))["label"] for index in range(60)])
        self.assertGreater(len(set(picks)), 5)  # uniform over the whole menu, wait included
        other = [RandomLegal(8).choose(state, menu, context(index))["label"] for index in range(60)]
        self.assertNotEqual(picks, other)
        # The episode seed in context does not enter the policy RNG.
        self.assertEqual(policy.choose(state, menu, context(3, seed=1))["label"],
                         policy.choose(state, menu, context(3, seed=2))["label"])


class SingleTowerTypeTests(unittest.TestCase):
    def test_highest_path_score_then_smallest_holder(self):
        state = state_fixture(scores=(10, 30, 30, 5))  # holders 12 and 13 tie on 30
        menu = build_menu(state)
        choice = SingleTowerType("mage").choose(state, menu, context())
        self.assertEqual(action_of(menu, choice["label"]),
                         {"action": "build_tower", "holder_id": 12, "tower_type": "mage"})
        self.assertEqual(choice["meta"]["holder_id"], 12)

    def test_unknown_score_ranks_last(self):
        state = state_fixture(scores=(None, 2))
        menu = build_menu(state)
        label = SingleTowerType("archer").choose(state, menu, context())["label"]
        self.assertEqual(action_of(menu, label)["holder_id"], 12)
        state = state_fixture(scores=(None,))
        menu = build_menu(state)
        label = SingleTowerType("archer").choose(state, menu, context())["label"]
        self.assertEqual(action_of(menu, label)["holder_id"], 11)

    def test_falls_back_to_send_wave_then_wait(self):
        state = state_fixture(gold=80)  # archer/barrack affordable, mage is not
        menu = build_menu(state)
        self.assertEqual(action_of(menu, SingleTowerType("mage").choose(state, menu, context())["label"]),
                         {"action": "send_wave"})
        state["wave_ready"] = False
        menu = build_menu(state)
        self.assertEqual(SingleTowerType("mage").choose(state, menu, context())["label"], "A")
        self.assertEqual(action_of(menu, SingleTowerType("barrack").choose(state, menu, context())["label"])
                         ["tower_type"], "barrack")
        with self.assertRaises(ValueError):
            SingleTowerType("hero")


class WaitOnlyTests(unittest.TestCase):
    def test_wait_only_always_waits_and_round_trips(self):
        from alpharush_rl.scripted_policies import WaitOnly, make_policy
        state = state_fixture(towers=())
        menu = build_menu(state)
        self.assertGreater(len(menu), 1)
        choice = WaitOnly().choose(state, menu, context())
        self.assertEqual(choice["label"], "A")
        self.assertEqual(choice["provenance"], "scripted:wait_only")
        self.assertEqual(make_policy("wait_only").name, "wait_only")


class PressureGreedyTests(unittest.TestCase):
    def choose(self, towers, **kwargs):
        state = state_fixture(towers=towers, **kwargs)
        menu = build_menu(state)
        choice = PressureGreedy().choose(state, menu, context())
        return action_of(menu, choice["label"]), choice

    def test_tower_counts_use_template_prefix(self):
        # Holders are not towers; a tower still under construction counts toward its kind.
        state = state_fixture(towers=("tower_archer_1", "tower_archer_3", "tower_build_mage",
                                      "tower_holder_grass", "tower_mage_2", "tower_engineer_1"))
        self.assertEqual(tower_counts(state), {"archer": 2, "barrack": 0, "mage": 2, "engineer": 1})

    def test_ratio_order(self):
        action, choice = self.choose(())  # all zero: the larger share (archer) first
        self.assertEqual(action, {"action": "build_tower", "holder_id": 12, "tower_type": "archer"})
        self.assertEqual(choice["meta"]["tower_counts"], {"archer": 0, "barrack": 0, "mage": 0, "engineer": 0})
        three = ("tower_archer_1",) * 3
        self.assertEqual(self.choose(three)[0]["tower_type"], "barrack")  # 0/1 < 3/3, ratio order breaks ties
        self.assertEqual(self.choose(("tower_archer_1",) * 2)[0]["tower_type"], "barrack")
        self.assertEqual(self.choose(("tower_archer_1",))[0]["tower_type"], "barrack")
        self.assertEqual(self.choose(three + ("tower_barrack_1",))[0]["tower_type"], "mage")
        full = three + ("tower_barrack_1", "tower_mage_1", "tower_engineer_1")
        self.assertEqual(self.choose(full)[0]["tower_type"], "archer")
        self.assertEqual(self.choose(full + ("tower_archer_2",) + ("tower_barrack_2", "tower_mage_2"))[0]["tower_type"],
                         "engineer")

    def test_only_buildable_types_considered(self):
        three = ("tower_archer_1",) * 3 + ("tower_barrack_1", "tower_mage_1")
        self.assertEqual(self.choose(three)[0]["tower_type"], "engineer")
        self.assertEqual(self.choose(three, gold=110)[0]["tower_type"], "archer")  # engineer unaffordable
        self.assertEqual(self.choose(three, kinds=("mage",))[0]["tower_type"], "mage")

    def test_no_builds_send_wave_then_wait(self):
        self.assertEqual(self.choose((), gold=0)[0], {"action": "send_wave"})
        self.assertEqual(self.choose((), gold=0, wave_ready=False)[0]["action"], "wait")

    def test_custom_ratio_and_validation(self):
        state = state_fixture()
        menu = build_menu(state)
        policy = PressureGreedy({"mage": 2, "engineer": 1})
        choice = policy.choose(state, menu, context())
        self.assertEqual(action_of(menu, choice["label"])["tower_type"], "mage")
        # A different ratio is a different baseline, never credited as the default one.
        self.assertEqual(choice["provenance"], "scripted:pressure_greedy:mage=2,engineer=1")
        choice["meta"]["ratio"]["mage"] = 99  # meta is a copy
        self.assertEqual(policy.ratio, {"mage": 2, "engineer": 1})
        self.assertEqual(PressureGreedy(dict(PressureGreedy().ratio)).name, "pressure_greedy")
        self.assertNotEqual(PressureGreedy({"barrack": 1, "archer": 3, "mage": 1, "engineer": 1}).name,
                            "pressure_greedy")
        for bad in ({}, {"hero": 1}, {"archer": 0}, {"archer": -1}, {"archer": float("nan")}, {"archer": True}):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                PressureGreedy(bad)


DATASET = Path(__file__).resolve().parents[1] / "runtime/rl/level1-24b-phase1/data/dataset.json"


@unittest.skipUnless(DATASET.exists(), "level-1 phase-1 dataset not present")
class RealLevel1StateTests(unittest.TestCase):
    """Read-only: real native level-1 states, menus rebuilt with the episode wait length."""

    def test_policies_pick_legal_labels(self):
        dataset = json.loads(DATASET.read_text(encoding="utf-8"))
        states = [dataset["groups"][0]["state"]] + [row["state"] for rows in dataset["anchors"].values() for row in rows]
        for index, state in enumerate(states):
            menu = build_menu(state, wait_ticks=60)
            holders = {holder["id"]: holder for holder in state["holders"]}
            for spec in ProtocolShapeTests.SPECS:
                with self.subTest(index=index, spec=spec):
                    choice = make_policy(spec).choose(state, menu, context(index))
                    action = action_of(menu, choice["label"])
                    if spec.startswith("single:") and action["action"] == "build_tower":
                        best = max(holders[item["action"]["holder_id"]]["path_score"] for item in menu
                                   if item["action"].get("tower_type") == action["tower_type"])
                        self.assertEqual(holders[action["holder_id"]]["path_score"], best)


if __name__ == "__main__":
    unittest.main()
