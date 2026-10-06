"""TeacherV2 (``teacher_v2``): rule order, legal labels only, deterministic, state+menu inputs only.

Menus are built here in build_menu's shape (wait first, then canonical action
order) with the action-scope v2 entries the native catalog offers, so these
tests do not depend on how menus.py renders them.
"""
from __future__ import annotations

import copy
import random
import unittest

from alpharush_rl.journal import canonical_json
from alpharush_rl.menus import option_label, validate_menu
from alpharush_rl.scripted_policies import (BRANCH_PREFERENCE, CAST_PROGRESS, DEFAULT_RATIO, PressureGreedy,
                                            TeacherV2, base_kind, make_policy, tower_counts)

COSTS = {"archer": 70, "barrack": 70, "mage": 100, "engineer": 125}
OTHER_BRANCH = {"archer": "tower_musketeer", "barrack": "tower_barbarian", "mage": "tower_sorcerer",
                "engineer": "tower_tesla"}
WAIT_TICKS = 60


def build(holder, kind, cost=None):
    return {"action": "build_tower", "holder_id": holder, "tower_type": kind}, COSTS[kind] if cost is None else cost


def upgrade(tower, target, cost=150):
    return {"action": "upgrade_tower", "tower_id": tower, "target": target}, cost


def power_up(tower, power, cost=200):
    return {"action": "upgrade_power", "tower_id": tower, "power": power}, cost


def sell(tower):
    return {"action": "sell_tower", "tower_id": tower}, 0


def cast(power, anchor, x=400, y=300):
    return {"action": "use_power", "power": power, "x": x, "y": y, "anchor_id": anchor}, 0


SEND = ({"action": "send_wave"}, 0)


def _text(action, cost):
    name = action["action"]
    if name == "build_tower":
        return f"Build {action['tower_type']} at holder {action['holder_id']} ({cost:g} gold)"
    if name == "upgrade_tower":
        return f"Upgrade tower {action['tower_id']} to {action['target']} ({cost:g} gold)"
    if name == "upgrade_power":
        return f"Upgrade power {action['power']} on tower {action['tower_id']} ({cost:g} gold)"
    if name == "sell_tower":
        return f"Sell tower {action['tower_id']}"
    if name == "use_power":
        spell = "Cast rain of fire" if action["power"] == 1 else "Call reinforcements"
        return f"{spell} at ({action['x']},{action['y']}) near enemy {action['anchor_id']}"
    return "Send the next wave"


def menu_of(*options):
    """Legal menu: wait first, then the options in canonical action order, labelled A, B, ..."""
    rest = sorted(({"text": _text(action, cost), "action": action, "cost": float(cost),
                    "legality_source": "native_catalog"} for action, cost in options),
                  key=lambda item: canonical_json(item["action"]))
    wait = {"text": f"Wait {WAIT_TICKS} native ticks", "action": {"action": "wait", "ticks": WAIT_TICKS},
            "cost": 0.0, "legality_source": "environment_wait"}
    return [{"label": option_label(index), **item} for index, item in enumerate([wait] + rest)]


def tower(tower_id, template, path_score=5, is_special=False):
    return {"id": tower_id, "template": template, "path_score": path_score, "is_special": is_special,
            "holder_id": str(tower_id), "level": 1}


def state_of(holders=(), towers=(), enemies=(), gold=1000, wave=1):
    """``holders`` are (id, path_score) pairs, ``enemies`` (id, path_progress) pairs."""
    return {"gold": gold, "lives": 20, "wave": wave, "wave_total": 6, "tick": 100,
            "holders": [{"id": holder_id, "mesh_id": str(holder_id), "blocked": False, "path_score": score,
                         "x": 100, "y": 200} for holder_id, score in holders],
            "towers": [dict(t) for t in towers],
            "enemies": [{"id": enemy_id, "path_progress": progress, "x": 300, "y": 300, "hp": 50}
                        for enemy_id, progress in enemies],
            "heroes": [], "action_catalog": [], "wave_ready": not enemies}


def context(index=0):
    return {"episode_id": "test", "decision_index": index, "level": 6, "seed": 1001, "difficulty": 2,
            "tick": 100, "protocol": "kr1-episode-v1"}


def choose(state, menu):
    choice = TeacherV2().choose(state, menu, context())
    action = next(item["action"] for item in menu if item["label"] == choice["label"])
    return action, choice


class ShapeTests(unittest.TestCase):
    def test_name_provenance_and_shape(self):
        policy = make_policy("teacher_v2")
        self.assertIsInstance(policy, TeacherV2)
        self.assertEqual(policy.name, "teacher_v2")
        state = state_of(holders=[(11, 3)], towers=[tower(201, "tower_ranger")], enemies=[(501, 0.6)])
        menu = menu_of(build(11, "archer"), power_up(201, "poison"), cast(1, 501), cast(2, 501), sell(201))
        choice = policy.choose(state, menu, context())
        self.assertEqual(set(choice), {"label", "provenance", "distribution", "meta"})
        self.assertIn(choice["label"], validate_menu(menu))
        self.assertEqual(choice["provenance"], "scripted:teacher_v2")
        self.assertIsNone(choice["distribution"])
        self.assertEqual(choice["meta"]["rule"], "use_power")
        canonical_json(choice["meta"])  # JSON-serializable, no NaN/inf

    def test_context_is_ignored(self):
        state = state_of(holders=[(11, 3), (12, 4)], towers=[tower(201, "tower_archer_1")])
        menu = menu_of(build(11, "mage"), build(12, "mage"), upgrade(201, "tower_archer_2"), SEND)
        policy = TeacherV2()
        self.assertEqual(policy.choose(state, menu, {}), policy.choose(state, menu, context(9)))

    def test_malformed_menu_raises(self):
        state = state_of(holders=[(11, 3)])
        menu = menu_of(build(11, "archer"), SEND)
        for broken in ([], menu + [dict(menu[1])], [dict(menu[0], label="a")] + menu[1:],
                       menu + [{"label": "Z", "action": {"action": "hire_hero"}}]):
            with self.subTest(broken=broken), self.assertRaises(ValueError):
                TeacherV2().choose(state, broken, context())


class CastRuleTests(unittest.TestCase):
    MENU = (build(11, "archer"), upgrade(201, "tower_archer_2"), power_up(202, "poison"), SEND)

    def test_rain_of_fire_at_the_furthest_anchor_before_anything_else(self):
        state = state_of(holders=[(11, 3)], towers=[tower(201, "tower_archer_1"), tower(202, "tower_ranger")],
                         enemies=[(501, 0.7), (502, 0.9), (503, 0.2)])
        menu = menu_of(*self.MENU, cast(1, 502, 410, 320), cast(1, 501, 300, 300), cast(1, 503, 100, 90),
                       cast(2, 502, 410, 320), cast(2, 501))
        action, choice = choose(state, menu)
        self.assertEqual(action, {"action": "use_power", "power": 1, "x": 410, "y": 320, "anchor_id": 502})
        self.assertEqual(choice["meta"], {"rule": "use_power", "power": 1, "anchor_id": 502, "anchor_progress": 0.9,
                                          "x": 410, "y": 320})

    def test_reinforcements_when_rain_of_fire_is_not_offered(self):
        state = state_of(enemies=[(501, 0.55), (502, 0.8)])
        action, choice = choose(state, menu_of(*self.MENU, cast(2, 501), cast(2, 502, 500, 100)))
        self.assertEqual(action, {"action": "use_power", "power": 2, "x": 500, "y": 100, "anchor_id": 502})
        self.assertEqual(choice["meta"]["power"], 2)

    def test_threshold_is_inclusive_and_needs_an_enemy_far_enough(self):
        menu = menu_of(cast(1, 501), cast(2, 501), SEND)
        self.assertEqual(CAST_PROGRESS, 0.5)
        self.assertEqual(choose(state_of(enemies=[(501, 0.5)]), menu)[0]["action"], "use_power")
        for enemies in ([(501, 0.49)], [(501, None)], [(501, "0.9")], []):
            with self.subTest(enemies=enemies):
                action, choice = choose(state_of(enemies=enemies), menu)
                self.assertNotEqual(action["action"], "use_power")
        # Below the threshold the other rules decide: here a build.
        state = state_of(holders=[(11, 3)], enemies=[(501, 0.3)])
        self.assertEqual(choose(state, menu_of(cast(1, 501), build(11, "archer")))[0]["action"], "build_tower")

    def test_ties_go_to_the_smaller_anchor_and_unknown_progress_ranks_last(self):
        state = state_of(enemies=[(507, 0.6), (503, 0.6), (509, None)])
        menu = menu_of(cast(1, 509), cast(1, 507), cast(1, 503))
        self.assertEqual(choose(state, menu)[0]["anchor_id"], 503)
        # An anchor missing from the state (or without progress) is offered but ranks last.
        state = state_of(enemies=[(507, 0.6), (509, None)])
        self.assertEqual(choose(state, menu_of(cast(1, 509), cast(1, 999), cast(1, 507)))[0]["anchor_id"], 507)
        self.assertEqual(choose(state, menu_of(cast(1, 509), cast(1, 999)))[0]["anchor_id"], 509)


class UpgradePowerTests(unittest.TestCase):
    def test_cheapest_power_of_the_highest_scoring_tower(self):
        towers = [tower(201, "tower_ranger", 3), tower(202, "tower_ranger", 7), tower(203, "tower_paladin", 7)]
        menu = menu_of(power_up(201, "poison", 10), power_up(202, "poison", 250), power_up(202, "thorn", 150),
                       power_up(203, "healing", 50), upgrade(204, "tower_mage_2", 5), SEND)
        action, choice = choose(state_of(towers=towers), menu)
        self.assertEqual(action, {"action": "upgrade_power", "tower_id": 202, "power": "thorn"})
        self.assertEqual(choice["meta"], {"rule": "upgrade_power", "tower_id": 202, "power": "thorn", "cost": 150.0,
                                          "path_score": 7})

    def test_equal_prices_go_by_power_name(self):
        menu = menu_of(power_up(202, "thorn", 150), power_up(202, "poison", 150))
        self.assertEqual(choose(state_of(towers=[tower(202, "tower_ranger")]), menu)[0]["power"], "poison")

    def test_unknown_tower_score_ranks_last(self):
        towers = [tower(201, "tower_ranger", None), tower(202, "tower_bfg", 0)]
        menu = menu_of(power_up(201, "poison"), power_up(202, "missile"), power_up(203, "bolt"))
        self.assertEqual(choose(state_of(towers=towers), menu)[0]["tower_id"], 202)


class UpgradeTowerTests(unittest.TestCase):
    def test_highest_path_score_then_smallest_tower_id(self):
        towers = [tower(301, "tower_archer_1", 5), tower(303, "tower_barrack_1", 9), tower(302, "tower_mage_2", 9)]
        menu = menu_of(upgrade(301, "tower_archer_2"), upgrade(303, "tower_barrack_2"), upgrade(302, "tower_mage_3"),
                       sell(301), SEND)
        action, choice = choose(state_of(towers=towers), menu)
        self.assertEqual(action, {"action": "upgrade_tower", "tower_id": 302, "target": "tower_mage_3"})
        self.assertEqual(choice["meta"], {"rule": "upgrade_tower", "tower_id": 302, "target": "tower_mage_3",
                                          "preferred_branch": False, "path_score": 9})

    def test_level_three_takes_the_preferred_branch_else_the_other(self):
        self.assertEqual(BRANCH_PREFERENCE, {"archer": "tower_ranger", "barrack": "tower_paladin",
                                             "mage": "tower_arcane_wizard", "engineer": "tower_bfg"})
        for kind, preferred in BRANCH_PREFERENCE.items():
            other = OTHER_BRANCH[kind]
            state = state_of(towers=[tower(301, f"tower_{kind}_3")])
            with self.subTest(kind=kind):
                for options in ((other, preferred), (preferred, other)):
                    action, choice = choose(state, menu_of(*(upgrade(301, target, 230) for target in options)))
                    self.assertEqual(action["target"], preferred)
                    self.assertTrue(choice["meta"]["preferred_branch"])
                action, choice = choose(state, menu_of(upgrade(301, other, 230)))  # preferred branch locked
                self.assertEqual(action["target"], other)
                self.assertFalse(choice["meta"]["preferred_branch"])

    def test_power_upgrades_come_before_tower_upgrades(self):
        towers = [tower(301, "tower_archer_1", 20), tower(302, "tower_ranger", 1)]
        menu = menu_of(upgrade(301, "tower_archer_2", 10), power_up(302, "thorn", 400))
        self.assertEqual(choose(state_of(towers=towers), menu)[0]["action"], "upgrade_power")


class BuildOrUpgradeTests(unittest.TestCase):
    """With both a build and an upgrade offered: build while empty holders > built towers / 2."""

    @staticmethod
    def scene(empty, built, special=0, power=False):
        holders = [(11 + i, i) for i in range(empty)]
        towers = ([tower(201 + i, "tower_archer_1", i) for i in range(built)]
                  + [tower(401 + i, "tower_elf", 50, is_special=True) for i in range(special)])
        options = [build(h, kind) for h, _ in holders for kind in COSTS]
        options += [upgrade(201 + i, "tower_archer_2") for i in range(built)]
        if power:
            options.append(power_up(201, "poison"))
        return state_of(holders=holders, towers=towers), menu_of(*options, SEND)

    def test_threshold(self):
        for empty, built, rule in ((1, 1, "build"), (4, 2, "build"), (3, 4, "build"), (2, 4, "upgrade_tower"),
                                   (1, 3, "upgrade_tower"), (3, 6, "upgrade_tower"), (4, 6, "build")):
            with self.subTest(empty=empty, built=built):
                state, menu = self.scene(empty, built)
                choice = choose(state, menu)[1]
                self.assertEqual(choice["meta"]["rule"], rule)
                self.assertEqual((choice["meta"]["build_first"], choice["meta"]["empty_holders"],
                                  choice["meta"]["built_towers"]), (rule == "build", empty, built))

    def test_empty_holders_count_holders_not_build_options(self):
        state, menu = self.scene(2, 4)  # 8 build options on 2 holders: 2 > 2 is false
        self.assertEqual(choose(state, menu)[1]["meta"]["empty_holders"], 2)

    def test_special_towers_are_not_built_towers(self):
        state, menu = self.scene(2, 2, special=3)
        choice = choose(state, menu)[1]
        self.assertEqual((choice["meta"]["rule"], choice["meta"]["built_towers"]), ("build", 2))

    def test_power_upgrades_compete_with_builds_too(self):
        state, menu = self.scene(1, 4, power=True)
        self.assertEqual(choose(state, menu)[1]["meta"]["rule"], "upgrade_power")
        state, menu = self.scene(3, 4, power=True)
        self.assertEqual(choose(state, menu)[1]["meta"]["rule"], "build")

    def test_no_build_order_meta_without_a_choice_to_make(self):
        state, menu = self.scene(2, 0)
        self.assertNotIn("build_first", choose(state, menu)[1]["meta"])
        state = state_of(towers=[tower(201, "tower_archer_1")])
        self.assertNotIn("build_first", choose(state, menu_of(upgrade(201, "tower_archer_2")))[1]["meta"])


class BuildRuleTests(unittest.TestCase):
    def test_same_choice_as_pressure_greedy_on_plain_towers(self):
        rng = random.Random(20261006)
        templates = ("tower_archer_1", "tower_archer_3", "tower_barrack_2", "tower_mage_1", "tower_engineer_2",
                     "tower_build_mage", "tower_holder_grass")
        for trial in range(120):
            holders = [(11 + i, rng.choice([0, 1, 3, 3, 7, None])) for i in range(rng.randint(1, 5))]
            towers = [tower(201 + i, rng.choice(templates)) for i in range(rng.randint(0, 6))]
            kinds = [kind for kind in COSTS if rng.random() < 0.7] or ["barrack"]
            state = state_of(holders=holders, towers=towers, wave=rng.randint(0, 3), enemies=())
            menu = menu_of(*(build(h, kind) for h, _ in holders for kind in kinds), SEND)
            with self.subTest(trial=trial):
                teacher = TeacherV2().choose(state, menu, context(trial))
                greedy = PressureGreedy().choose(state, menu, context(trial))
                self.assertEqual(teacher["label"], greedy["label"])
                self.assertEqual(teacher["meta"]["rule"], "build")
                self.assertEqual(teacher["meta"]["tower_counts"], tower_counts(state))
                self.assertEqual(teacher["meta"]["ratio"], DEFAULT_RATIO)

    def test_four_level_towers_count_toward_their_base_kind(self):
        towers = [tower(201, "tower_ranger"), tower(202, "tower_musketeer"), tower(203, "tower_archer_1"),
                  tower(204, "tower_paladin"), tower(205, "tower_arcane_wizard"), tower(206, "tower_bfg")]
        state = state_of(holders=[(11, 1)], towers=towers)
        menu = menu_of(*(build(11, kind) for kind in COSTS))
        action, choice = choose(state, menu)
        self.assertEqual(choice["meta"]["tower_counts"], {"archer": 3, "barrack": 1, "mage": 1, "engineer": 1})
        self.assertEqual(action["tower_type"], "archer")  # 3/3 = 1/1: ties prefer the larger share
        greedy = PressureGreedy().choose(state, menu, context())  # sees one archer and nothing else
        self.assertNotEqual(greedy["label"], choice["label"])

    def test_base_kind(self):
        cases = {"tower_archer_1": "archer", "tower_barrack_3": "barrack", "tower_build_engineer": "engineer",
                 "tower_sorcerer": "mage", "tower_tesla": "engineer", "tower_barbarian": "barrack",
                 "tower_musketeer": "archer", "tower_elf": None, "tower_holder_grass": None, "tower_sunray": None,
                 None: None, 7: None}
        for template, kind in cases.items():
            with self.subTest(template=template):
                self.assertEqual(base_kind(template), kind)


class SendWaitAndSellTests(unittest.TestCase):
    def test_send_wave_only_without_enemies(self):
        self.assertEqual(choose(state_of(), menu_of(SEND))[0], {"action": "send_wave"})
        action, choice = choose(state_of(enemies=[(501, 0.1)], wave=2), menu_of(SEND))
        self.assertEqual((action["action"], choice["meta"]), ("wait", {"rule": "wait"}))
        # Before the first wave the native wave_ready alone decides.
        self.assertEqual(choose(state_of(enemies=[(501, 0.1)], wave=0), menu_of(SEND))[0], {"action": "send_wave"})

    def test_builds_come_before_the_wave(self):
        state = state_of(holders=[(11, 1)])
        self.assertEqual(choose(state, menu_of(build(11, "mage"), SEND))[0]["action"], "build_tower")

    def test_never_sells_and_waits_when_nothing_applies(self):
        state = state_of(towers=[tower(201, "tower_archer_1")], enemies=[(501, 0.2)])
        for menu in (menu_of(sell(201)), menu_of(), menu_of(sell(201), cast(1, 501), cast(2, 501))):
            with self.subTest(menu=menu):
                action, choice = choose(state, menu)
                self.assertEqual(action, {"action": "wait", "ticks": WAIT_TICKS})
                self.assertEqual(choice["meta"], {"rule": "wait"})


class DeterminismTests(unittest.TestCase):
    RULE_ACTIONS = {"use_power": "use_power", "upgrade_power": "upgrade_power", "upgrade_tower": "upgrade_tower",
                    "build": "build_tower", "send_wave": "send_wave", "wait": "wait"}

    def test_only_legal_labels_deterministic_and_non_mutating(self):
        rng = random.Random(20261006)
        templates = ("tower_archer_1", "tower_archer_3", "tower_barrack_3", "tower_mage_2", "tower_ranger",
                     "tower_paladin", "tower_bfg", "tower_elf")
        for trial in range(300):
            holders = [(11 + i, rng.choice([0, 2, 2, 5, None])) for i in range(rng.randint(0, 5))]
            towers = [tower(201 + i, rng.choice(templates), rng.choice([0, 4, 4, 9, None]),
                            is_special=rng.random() < 0.1) for i in range(rng.randint(0, 6))]
            enemies = [(501 + i, rng.choice([0.1, 0.4, 0.5, 0.75, 0.75, None])) for i in range(rng.randint(0, 6))]
            state = state_of(holders=holders, towers=towers, enemies=enemies, wave=rng.randint(0, 4))
            options = [build(h, kind) for h, _ in holders for kind in COSTS if rng.random() < 0.6]
            for t in state["towers"]:
                if rng.random() < 0.5:
                    options.append(upgrade(t["id"], rng.choice(["tower_archer_2", "tower_ranger", "tower_musketeer"]),
                                           rng.choice([110, 230])))
                if rng.random() < 0.4:
                    options.append(power_up(t["id"], rng.choice(["poison", "thorn"]), rng.choice([100, 250])))
                if rng.random() < 0.5:
                    options.append(sell(t["id"]))
            anchors = [enemy_id for enemy_id, _ in enemies][:3]
            for power in (1, 2):
                if rng.random() < 0.6:
                    options.extend(cast(power, anchor, 100 + anchor, 200) for anchor in anchors)
            if rng.random() < 0.5:
                options.append(SEND)
            unique = {canonical_json(action): (action, cost) for action, cost in options}
            menu = menu_of(*unique.values())
            labels = validate_menu(menu)
            with self.subTest(trial=trial):
                before_state, before_menu = copy.deepcopy(state), copy.deepcopy(menu)
                first = TeacherV2().choose(state, menu, context(trial))
                self.assertIn(first["label"], labels)
                self.assertEqual((state, menu), (before_state, before_menu))
                again = make_policy("teacher_v2").choose(copy.deepcopy(state), copy.deepcopy(menu), context(0))
                self.assertEqual(again, first)
                canonical_json(first["meta"])
                action = next(item["action"] for item in menu if item["label"] == first["label"])
                self.assertEqual(action["action"], self.RULE_ACTIONS[first["meta"]["rule"]])
                self.assertNotEqual(action["action"], "sell_tower")



class TeacherParamsTests(unittest.TestCase):
    def test_variant_names_round_trip_canonically(self):
        from alpharush_rl.scripted_policies import TEACHER_DEFAULTS, make_policy, parse_teacher_params, teacher_name
        self.assertEqual("teacher_v2", make_policy("teacher_v2").name)
        params = {**TEACHER_DEFAULTS, "b": "mbst", "c": 30, "r": "2211"}
        name = teacher_name(params)
        self.assertEqual("teacher_v2:b=mbst,c=30,r=2211", name)
        self.assertEqual(params, parse_teacher_params(name.split(":", 1)[1]))
        self.assertEqual(name, make_policy(name).name)
        for bad in ("teacher_v2:", "teacher_v2:c=50", "teacher_v2:r=2211,c=30", "teacher_v2:b=xxxx",
                    "teacher_v2:r=0000", "teacher_v2:c=101", "teacher_v2:c=030", "teacher_v2:z=1"):
            with self.assertRaises(ValueError):
                make_policy(bad)

    def test_exploration_is_seeded_by_decision_index(self):
        from alpharush_rl.scripted_policies import make_policy
        policy = make_policy("teacher_v2:e=100,s=3")
        state = {"towers": [], "enemies": [], "holders": [{"id": 1, "mesh_id": "01", "path_score": 1}], "gold": 100}
        menu = [{"label": "A", "action": {"action": "wait", "ticks": 30}},
                {"label": "B", "action": {"action": "build_tower", "holder_id": 1, "tower_type": "archer"}},
                {"label": "C", "action": {"action": "build_tower", "holder_id": 1, "tower_type": "mage"}}]
        picks = [policy.choose(state, menu, {"decision_index": i})["label"] for i in range(20)]
        self.assertEqual(picks, [policy.choose(state, menu, {"decision_index": i})["label"] for i in range(20)])
        self.assertEqual({"B", "C"}, set(picks))
        self.assertTrue(all(policy.choose(state, menu, {"decision_index": i})["meta"]["rule"] == "explore"
                            for i in range(5)))



class AimTests(unittest.TestCase):
    def test_a_ready_sunray_fires_at_the_strongest_offered_enemy_first(self):
        from alpharush_rl.menus import build_menu
        from alpharush_rl.scripted_policies import make_policy
        from alpharush_rl.search import BuildOrderPolicy, BRANCHES, KINDS
        state = {"gold": 0, "lives": 20, "wave": 3, "wave_total": 9, "tick": 10, "wave_ready": False,
                 "holders": [], "towers": [{"id": 57, "template": "tower_sunray", "is_special": True,
                                            "holder_id": "71", "powers": [{"name": "ray", "level": 1, "max_level": 4}]}],
                 "enemies": [{"id": 5, "hp": 90, "path_progress": 0.9}, {"id": 6, "hp": 900, "path_progress": 0.2}],
                 "action_catalog": [{"action": "point_tower", "tower_id": 57, "x": x, "y": 1, "anchor_id": a,
                                     "cost": 0, "available": True} for a, x in ((5, 10), (6, 20))]}
        menu = build_menu(state)
        self.assertEqual(3, len(menu))  # wait and two aims
        self.assertIn("Fire tower 57 (tower_sunray) at enemy 6 (20,1)", [m["text"] for m in menu])
        for policy in (make_policy("teacher_v2"),
                       BuildOrderPolicy({"hero": None, "steps": [], "cast": 50, "early": 1,
                                         "branches": {k: BRANCHES[k][0] for k in KINDS}})):
            choice = policy.choose(state, menu, {"decision_index": 0})
            picked = next(m for m in menu if m["label"] == choice["label"])
            self.assertEqual(6, picked["action"]["anchor_id"])
            self.assertEqual("point_tower", choice["meta"]["rule"])
        normal = dict(state, towers=[dict(state["towers"][0], is_special=False)])
        self.assertEqual(1, len(build_menu(normal)))  # only special towers are aimed


if __name__ == "__main__":
    unittest.main()
