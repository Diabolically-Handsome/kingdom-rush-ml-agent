"""Elite stages (levels 13-26): rally plan steps, boss blocking, dormant bosses, unlock chains, the elite operator
layout and strategy prompts. Toy levels only; no game is started."""
from __future__ import annotations

import hashlib
from pathlib import Path
import random
import tempfile
import time
import unittest

import numpy as np

from alpharush_rl import phase
from alpharush_rl.campaign import ELITE_PREREQUISITE, prerequisite, slot_lua, unlocked_levels
from alpharush_rl.campaign_run import check_campaign, run_campaigns
from alpharush_rl.engine import level_scope
from alpharush_rl.env import _set_rally_receipt
from alpharush_rl.episode import run_episode
from alpharush_rl.journal import Journal, JournalError
from alpharush_rl.menus import build_menu
from alpharush_rl import operator_net
from alpharush_rl.search import (BuildOrderPolicy, LevelSearch, block_choice, check_genome, crossover, genome_id,
                                 mutate, random_genome)
from alpharush_rl import strategy_brain

from test_search import POOLS, PROTOCOL, ToyEnv, genome

HOLDERS = [{"mesh": f"0{i}", "x": 10 * i, "y": 5, "path_score": 1.0} for i in range(1, 6)]


def option(label, action, cost=0.0):
    return {"label": label, "text": "", "action": action, "cost": cost, "legality_source": "native_catalog"}


def rally(tower_id, which, x, y):
    return {"action": "set_rally", "tower_id": tower_id, "option": which, "x": x, "y": y}


WAIT = option("A", {"action": "wait", "ticks": 30})


class GenomeTests(unittest.TestCase):
    def test_rally_steps_and_block_are_valid_plan_parts(self):
        g = genome([["b", "01", "barrack"], ["r", "01", "exit"]], block=1)
        self.assertEqual(1, g["block"])
        self.assertNotIn("block", genome([["b", "01", "barrack"]], block=0))
        self.assertEqual(genome_id(genome([["b", "01", "mage"]])), genome_id(genome([["b", "01", "mage"]], block=0)))
        for bad in ([["r", "01"]], [["r", "01", "boss"]], [["r", "01", "exit", 1]], [["r", 1, "exit"]]):
            with self.assertRaises(ValueError):
                genome(bad)
        with self.assertRaises(ValueError):
            genome([], block=True)

    def test_fallback_build_first_gene(self):
        g = genome([["b", "01", "mage"]], f=400)
        self.assertEqual(400, g["f"])
        self.assertNotIn("f", genome([], f=50))
        for bad in (0, 60, 401, True, "400"):
            with self.assertRaises(ValueError):
                genome([], f=bad)
        self.assertEqual(400, BuildOrderPolicy(g).fallback.params["f"])
        self.assertEqual(50, BuildOrderPolicy(genome([])).fallback.params["f"])
        self.assertIn("upgrades a few strong towers first", strategy_brain.describe_plan(g))

    def test_tower_cap_gene(self):
        g = genome([], cap=8)
        self.assertEqual(8, BuildOrderPolicy(g).fallback.params["m"])
        self.assertNotIn("cap", genome([], cap=0))
        for bad in (5, -1, True, "8"):
            with self.assertRaises(ValueError):
                genome([], cap=bad)
        self.assertIn("never more than 8 towers", strategy_brain.describe_plan(g))
        towers = [{"id": i, "holder_id": str(i), "template": "tower_archer_1"} for i in range(1, 9)]
        state = {"gold": 500, "wave": 1, "enemies": [], "towers": towers,
                 "holders": [{"id": h, "mesh_id": str(h), "path_score": 1} for h in (50, 51, 52)]}
        menu = [WAIT, *[option(label, {"action": "build_tower", "holder_id": h, "tower_type": "archer"}, 70.0)
                        for label, h in (("B", 50), ("D", 51), ("E", 52))],
                option("C", {"action": "upgrade_tower", "tower_id": 1, "target": "tower_archer_2"}, 110.0)]
        capped = BuildOrderPolicy(genome([], cap=8, f=25)).choose(state, menu, {"decision_index": 0})
        free = BuildOrderPolicy(genome([], f=25)).choose(state, menu, {"decision_index": 0})
        self.assertEqual(("C", "B"), (capped["label"], free["label"]))

    def test_soft_tower_cap_builds_again_with_banked_gold(self):
        from alpharush_rl.scripted_policies import parse_teacher_params, teacher_name
        g = genome([], cap=-8)
        self.assertEqual(-8, BuildOrderPolicy(g).fallback.params["m"])
        self.assertIn("at most 8 towers until 1000 gold is banked", strategy_brain.describe_plan(g))
        self.assertEqual(-6, parse_teacher_params("m=-6")["m"])
        self.assertEqual("teacher_v2:m=-6", teacher_name({**parse_teacher_params(""), "m": -6}))
        for bad in ("m=-0", "m=--6", "c=-5"):
            with self.assertRaises(ValueError):
                parse_teacher_params(bad)
        towers = [{"id": i, "holder_id": str(i), "template": "tower_archer_1"} for i in range(1, 9)]
        menu = [WAIT, *[option(label, {"action": "build_tower", "holder_id": h, "tower_type": "archer"}, 70.0)
                        for label, h in (("B", 50), ("D", 51), ("E", 52))],
                option("C", {"action": "upgrade_tower", "tower_id": 1, "target": "tower_archer_2"}, 110.0)]
        for gold, soft, hard in ((500, "C", "C"), (1000, "B", "C")):
            state = {"gold": gold, "wave": 1, "enemies": [], "towers": towers,
                     "holders": [{"id": h, "mesh_id": str(h), "path_score": 1} for h in (50, 51, 52)]}
            got = [BuildOrderPolicy(genome([], cap=c, f=25)).choose(state, menu, {"decision_index": 0})["label"]
                   for c in (-8, 8)]
            self.assertEqual([soft, hard], got, gold)
        rng = random.Random(5)
        caps = {random_genome(HOLDERS, [], rng, rally=True).get("cap", 0) for _ in range(300)}
        self.assertTrue({-6, -8, -10, -12} <= caps and {6, 8, 10, 12} <= caps)

    def test_operators_never_add_rally_without_scope_v3(self):
        rng = random.Random(7)
        for _ in range(300):
            g = random_genome(HOLDERS, ["hero_gerald"], rng)
            g = mutate(crossover(g, random_genome(HOLDERS, [], rng), rng), HOLDERS, ["hero_gerald"], rng)
            self.assertNotIn("block", g)
            self.assertNotIn("f", g)
            self.assertNotIn("cap", g)
            self.assertFalse([s for s in g["steps"] if s[0] == "r"])

    def test_operators_with_rally_add_rally_steps_after_barracks_builds(self):
        rng = random.Random(3)
        steps, blocks = 0, set()
        g = random_genome(HOLDERS, [], rng, rally=True)
        for _ in range(400):
            g = mutate(g, HOLDERS, [], rng, rally=True)
            blocks.add(g.get("block", 0))
            built = set()
            for s in g["steps"]:
                if s[0] == "b":
                    built.add(s[1])
                elif s[0] == "r":
                    steps += 1
                    self.assertIn(s[1], built)  # a rally step follows its holder's build
        self.assertGreater(steps, 0)
        self.assertEqual({0, 1}, blocks)
        rng = random.Random(5)
        self.assertGreater(len({random_genome(HOLDERS, [], rng, rally=True).get("f", 50) for _ in range(40)}), 3)

    def test_level_search_passes_rally_to_its_operators(self):
        search = LevelSearch(13, HOLDERS, [], "s", population=8, rally=True)
        proposals = [search.propose() for _ in range(8)]
        for g in proposals[1:]:
            search.report(g, 1.0)
        self.assertTrue(any("block" in g for g in proposals[1:]))


class ExecutorTests(unittest.TestCase):
    def state(self, **extra):
        return {"gold": 100, "wave": 1, "holders": [], "enemies": [],
                "towers": [{"id": 7, "holder_id": "01", "template": "tower_barrack_2", "rally_x": 10, "rally_y": 10}],
                **extra}

    def test_rally_step_takes_its_offered_option_or_skips(self):
        menu = [WAIT, option("B", rally(7, "center", 50, 60)), option("C", rally(7, "exit", 90, 60))]
        policy = BuildOrderPolicy(genome([["r", "01", "exit"], ["r", "01", "entry"]]))
        choice = policy.choose(self.state(), menu, {})
        self.assertEqual(("C", "plan"), (choice["label"], choice["meta"]["rule"]))
        choice = policy.choose(self.state(), menu, {})  # no "entry" point: skipped, plan exhausted
        self.assertEqual(2, policy.cursor)
        self.assertNotEqual("plan", choice["meta"]["rule"])

    def test_rally_step_waits_while_its_tower_has_no_menu(self):
        policy = BuildOrderPolicy(genome([["r", "01", "exit"]]))
        choice = policy.choose(self.state(), [WAIT], {})  # e.g. stunned by Blackburn: no menu at all
        self.assertEqual((0, "plan_wait"), (policy.cursor, choice["meta"]["rule"]))
        menu = [WAIT, option("B", {"action": "sell_tower", "tower_id": 7})]
        policy.choose(self.state(), menu, {})  # a menu without rally points: the step is skipped
        self.assertEqual(1, policy.cursor)

    def test_block_rallies_far_barracks_onto_the_boss(self):
        state = self.state(towers=[{"id": 7, "holder_id": "01", "template": "tower_barrack_2", "rally_x": 10, "rally_y": 10},
                                   {"id": 8, "holder_id": "02", "template": "tower_barrack_2", "rally_x": 100, "rally_y": 100}])
        menu = [WAIT, option("B", rally(7, "boss", 20, 20)), option("C", rally(8, "boss", 300, 300))]
        self.assertEqual("C", block_choice(state, menu)["label"])  # 7 is already within BLOCK_RADIUS
        policy = BuildOrderPolicy(genome([], block=1))
        self.assertEqual(("C", "block"), (policy.choose(state, menu, {})["label"],
                                          policy.choose(state, menu, {})["meta"]["rule"]))
        self.assertNotEqual("block", BuildOrderPolicy(genome([])).choose(state, menu, {})["meta"]["rule"])

    def test_dormant_boss_neither_triggers_spells_nor_blocks_wave_calls(self):
        dormant = {"id": 5, "template": "enemy_demon_cerberus", "hp": 6000, "boss": True, "dormant": True,
                   "path_progress": 0.9, "x": 1, "y": 1}
        state = self.state(enemies=[dormant], wave_ready=True)
        menu = [WAIT, option("B", {"action": "send_wave"}),
                option("C", {"action": "use_power", "power": 1, "x": 1, "y": 1, "anchor_id": 5})]
        policy = BuildOrderPolicy(genome([["b", "05", "mage"]], cast=10, early=1, boss=1))
        choice = policy.choose(state, menu, {})
        self.assertEqual(("B", "fallback_send_wave"), (choice["label"], choice["meta"]["rule"]))
        awake = dict(dormant, dormant=False)
        choice = BuildOrderPolicy(genome([["b", "05", "mage"]], cast=10, early=1, boss=1)).choose(
            self.state(enemies=[awake]), menu, {})
        self.assertEqual("C", choice["label"])  # an awake boss (by flag, not eb_ name) gets the spell


class ReceiptAndMenuTests(unittest.TestCase):
    def test_rally_receipt_checks_the_reported_rally_point(self):
        after = {"towers": [{"id": 7, "rally_x": 300, "rally_y": 210}]}
        self.assertTrue(_set_rally_receipt({}, after, rally(7, "exit", 300, 210), 0)["executed"])
        self.assertFalse(_set_rally_receipt({}, after, rally(7, "exit", 301, 210), 0)["executed"])
        self.assertFalse(_set_rally_receipt({}, {"towers": []}, rally(7, "exit", 300, 210), 0)["executed"])

    def test_menu_offers_rally_options_of_known_towers_only(self):
        state = {"gold": 10, "wave": 1, "holders": [], "enemies": [],
                 "towers": [{"id": 7, "holder_id": "01", "template": "tower_barrack_2"}],
                 "action_catalog": [{**rally(7, "exit", 300, 210), "cost": 0, "available": True},
                                    {**rally(7, "boss", 140, 210), "cost": 0, "available": True},
                                    {**rally(9, "exit", 300, 210), "cost": 0, "available": True},
                                    {**rally(7, "nowhere", 1, 2), "cost": 0, "available": True},
                                    {**rally(7, "entry", 1.5, 2), "cost": 0, "available": True}]}
        menu = build_menu(state)
        actions = [item["action"] for item in menu[1:]]
        self.assertEqual([rally(7, "boss", 140, 210), rally(7, "exit", 300, 210)], actions)
        self.assertIn("boss's path point", menu[1]["text"])


class StarProfileTests(unittest.TestCase):
    def test_elite_rate_profiles(self):
        from alpharush_rl.campaign import level_profile
        self.assertEqual(level_profile(13, 2.75), level_profile(13, [2.75, 1.5]))  # no earlier elite stage
        plain, realistic = level_profile(20, 2.75), level_profile(20, [2.75, 1.5])
        self.assertEqual(52, sum(plain["levels"].values()))
        self.assertEqual(33 + 10, sum(realistic["levels"].values()))  # 12 main at 2.75, 7 elite at 1.5
        self.assertEqual({13: 2, 14: 2, 15: 2, 16: 1, 17: 1, 18: 1, 19: 1},
                         {k: v for k, v in realistic["levels"].items() if k > 12})
        for bad in ([2.75], [2.75, 0.5], [True, 2], "2"):
            with self.assertRaises(ValueError):
                level_profile(20, bad)


class ChallengeTests(unittest.TestCase):
    def test_profiles_slots_and_budget(self):
        from alpharush_rl.campaign import check_profile, task_profile
        won = {level: 3 for level in range(1, 13)}
        upgrades = {"rain": 5, "archers": 5, "mages": 5, "barracks": 4}  # 37 stars: 36 campaign + challenge stars
        profile = check_profile({"levels": won, "challenges": {"1": [3, 2], "5": [2]}, "upgrades": upgrades})
        with self.assertRaises(ValueError):
            check_profile({"levels": won, "upgrades": upgrades})
        self.assertEqual({1: [2, 3], 5: [2]}, profile["challenges"])
        self.assertNotIn("challenges", check_profile({"levels": won}))  # main profiles keep their old shape
        slot = slot_lua(profile)
        self.assertIn("[1] = {\n\t\t\t[1] = 2;\n\t\t\t[2] = 2;\n\t\t\t[3] = 2;", slot)
        for bad in ({"challenges": {"13": [2]}}, {"challenges": {"1": [1]}}, {"challenges": {"1": [2, 2]}},
                    {"challenges": {"1": []}}):
            with self.assertRaises(ValueError):
                check_profile({"levels": won, **bad})
        with self.assertRaises(ValueError):  # 39 stars cannot buy 40 stars of upgrades
            check_profile({"levels": won, "challenges": {"1": [2, 3], "2": [2]},
                           "upgrades": {"rain": 5, "archers": 5, "mages": 5, "barracks": 5}})
        iron5 = task_profile(5, [2.6, 1.0], mode=3)
        self.assertEqual(31, sum(iron5["levels"].values()))
        self.assertEqual(4, sum(len(m) for m in iron5["challenges"].values()))  # half of the 9 challenges before it
        iron10 = task_profile(10, [2.6, 1.0], mode=3)
        self.assertEqual(3, iron10["levels"][10])  # replayed for 3 stars: the game opens challenges only then
        with self.assertRaises(ValueError):  # a 2-star level's challenges stay locked
            check_profile({"levels": {**won, 2: 2}, "challenges": {"2": [2]}})
        self.assertNotIn("challenges", level_profile_of_rate(5, [2.6, 1.0, 24]))  # main levels: no challenge stars
        self.assertEqual(task_profile(5, [2.6, 1.0]), level_profile_of(5))
        full = task_profile(20, [3, 1.0, 24])
        self.assertEqual(24, sum(len(m) for m in full["challenges"].values()))
        partial = task_profile(20, [2.6, 1.0, 24])  # only the 3-star levels 1-7 can hold challenge wins
        self.assertEqual(14, sum(len(m) for m in partial["challenges"].values()))
        with self.assertRaises(ValueError):
            task_profile(13, [2.6, 1.0], mode=2)

    def test_campaign_plays_challenges_between_main_and_elite(self):
        made = []

        def factory(level, seed, profile, attempt, port, mode=1):
            made.append((level, mode, attempt, profile))
            toy = type("Toy", (LoseOn,), {"lose": (14,) if mode == 1 else ((2,) if mode == 3 else ())})
            return toy(seed=seed, level=level, gold=600)
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "runs" / "native-campaign-0001"
            run_dir.mkdir(parents=True)
            ctx = phase.PhaseRunContext("native-campaign-0001", run_dir, Path(tmp), time.monotonic() + 300, 200,
                                        job_kind="native-campaign")
            spec = {"levels": [13, 14], "seeds": [1001], "attempts_per_level": 1, "policy": "plan", "brain": "rule",
                    "plans": {"13": [WIN_PLAN], "14": [WIN_PLAN]}, "start": MAIN,
                    "challenges": {"1:2": [WIN_PLAN], "2:3": [WIN_PLAN], "2:2": [WIN_PLAN]}, "challenge_attempts": 2}
            self.assertEqual([], check_campaign(spec, POOLS, phase)[1])
            summary = run_campaigns(spec, factory, ctx, Journal(run_dir / "episodes.jsonl"), protocol=PROTOCOL,
                                    make_policy=BuildOrderPolicy, ports=[9001])
            campaign = summary["campaigns"][0]
            rows = Journal(run_dir / "episodes.jsonl").entries()
        self.assertEqual([(1, 2), (2, 2), (2, 3), (2, 3), (13, 1), (14, 1)], [(l, m) for l, m, _, _ in made])
        self.assertEqual({"1": [2], "2": [2]}, campaign["challenges_won"])  # the iron one lost twice
        self.assertEqual(2, campaign["challenge_stars"])
        self.assertEqual(sum(int(v) for v in MAIN.values()) + 3 + 2, campaign["total_stars"])  # + stars of 13
        self.assertEqual({1: [2], 2: [2]}, made[4][3]["challenges"])  # level 13 is played with both challenge stars
        self.assertEqual(4, sum(row["kind"] == "challenge_attempt" for row in rows))
        bad = dict(spec, challenges={"13:2": [WIN_PLAN]})
        self.assertIsNone(check_campaign(bad, POOLS, phase)[0])
        self.assertIsNone(check_campaign({k: v for k, v in spec.items() if k != "challenge_attempts"}, POOLS, phase)[0])

    def test_star_replays_unlock_challenges(self):
        made = []

        def factory(level, seed, profile, attempt, port, mode=1):
            made.append((level, mode, attempt))
            return ToyEnv(seed=seed, level=level, gold=600)
        start = {**MAIN, "4": 2}
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "runs" / "native-campaign-0001"
            run_dir.mkdir(parents=True)
            ctx = phase.PhaseRunContext("native-campaign-0001", run_dir, Path(tmp), time.monotonic() + 300, 200,
                                        job_kind="native-campaign")
            spec = {"levels": [13], "seeds": [1001], "attempts_per_level": 1, "policy": "plan", "brain": "rule",
                    "plans": {"4": [WIN_PLAN], "13": [WIN_PLAN]}, "start": start, "star_attempts": 2,
                    "challenges": {"4:2": [WIN_PLAN]}, "challenge_attempts": 1}
            self.assertEqual([], check_campaign(spec, POOLS, phase)[1])
            campaign = run_campaigns(spec, factory, ctx, Journal(run_dir / "episodes.jsonl"), protocol=PROTOCOL,
                                     make_policy=BuildOrderPolicy, ports=[9001])["campaigns"][0]
        self.assertEqual([(4, 1, "r0"), (4, 2, 0), (13, 1, 0)], made)  # replay to 3 stars, then the challenge
        self.assertEqual(3, int(campaign["won"]["4"]))
        self.assertEqual({"4": [2]}, campaign["challenges_won"])
        self.assertIsNone(check_campaign(dict(spec, levels=[13], start={"1": 3}), POOLS, phase)[0])

    def test_search_mode_only_for_main_levels(self):
        from alpharush_rl import search_job
        spec = {"levels": [3, 5], "search_seed": 1001, "population": 8, "evaluations_per_level": 10,
                "min_evaluations": 0, "stop_lives": 1, "profile_stars_per_level": [2.6, 1.0], "workers": 2,
                "validate_seeds": [1003], "validate_top": 1, "rng_seed": "t", "mode": 3}
        self.assertEqual([], search_job.check_search(spec, POOLS)[1])
        self.assertTrue(search_job.check_search({**spec, "levels": [13]}, POOLS)[1])
        self.assertTrue(search_job.check_search({**spec, "mode": 4}, POOLS)[1])

    def test_challenge_prompts(self):
        broker = BrainTests.Broker()
        brain = strategy_brain.LanguageBrain(broker)
        brain.choose_plan(4, [(WIN_PLAN, None), (WIN_PLAN, None)], 0, {"progress": "p", "upgrades": {}, "mode": 3})
        self.assertEqual(strategy_brain.CHALLENGE_SYSTEM, broker.requests[-1]["system"])
        self.assertIn("Level 4 Iron challenge (one long wave, 1 life, no hero) is next", broker.requests[-1]["user"])


def level_profile_of_rate(level, rate):
    from alpharush_rl.campaign import level_profile
    return level_profile(level, rate)


def level_profile_of(level):
    from alpharush_rl.campaign import level_profile
    return level_profile(level, [2.6, 1.0])


class UnlockTests(unittest.TestCase):
    def test_prerequisites_follow_the_game_ranges(self):
        self.assertIsNone(prerequisite(1))
        self.assertEqual(11, prerequisite(12))
        for level in (13, 14, 15, 16, 18, 20, 23):
            self.assertEqual(12, prerequisite(level))
        self.assertEqual({17: 16, 19: 18, 21: 20, 22: 15, 24: 23, 25: 24, 26: 25},
                         {k: v for k, v in ELITE_PREREQUISITE.items() if v != 12})
        with self.assertRaises(ValueError):
            prerequisite(27)

    def test_unlocked_levels(self):
        self.assertEqual([1], unlocked_levels({}))
        self.assertEqual([1, 2, 3, 4], unlocked_levels({1: 3, 2: 3, 3: 2}))
        main = {level: 3 for level in range(1, 13)}
        self.assertEqual(list(range(1, 17)) + [18, 20, 23], unlocked_levels(main))
        self.assertEqual(list(range(1, 19)) + [20, 22, 23], unlocked_levels({**main, 15: 3, 16: 2}))

    def test_slot_unlocks(self):
        main_only = slot_lua({"levels": {1: 3, 2: 3}})
        self.assertIn("[3] = {\n", main_only)
        self.assertNotIn("[4] =", main_only)
        elite = slot_lua({"levels": {level: 3 for level in range(1, 13)}})
        for level in (13, 14, 15, 16, 18, 20, 23):
            self.assertIn(f"[{level}] = {{\n", elite)
        for level in (17, 19, 21, 22, 24):
            self.assertNotIn(f"[{level}] =", elite)


class LoseOn(ToyEnv):
    """A toy level no plan can win on the levels in ``lose``."""
    lose = ()

    def _spawn(self, s):
        super()._spawn(s)
        if self.level in self.lose:
            for enemy in s["enemies"]:
                enemy["hp"] = 10 ** 6


WIN_PLAN = genome([["b", "01", "mage"], ["b", "02", "archer"], ["u", "01"], ["b", "03", "archer"]])
MAIN = {str(level): 3 for level in range(1, 13)}


class EliteCampaignTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="alpharush-elite-")
        self.addCleanup(self.tmp.cleanup)

    def run_spec(self, spec, lose=(), elite_policy=None):
        made, policies = [], []

        class Toy(LoseOn):
            pass
        Toy.lose = tuple(lose)

        def factory(level, seed, profile, attempt, port):
            made.append((level, attempt, profile))
            return Toy(seed=seed, level=level, gold=600)
        run_dir = Path(self.tmp.name) / "runs" / f"native-campaign-{len(list(Path(self.tmp.name).glob('*')))}"
        run_dir.mkdir(parents=True)
        ctx = phase.PhaseRunContext("native-campaign-0001", run_dir, Path(self.tmp.name), time.monotonic() + 300, 200,
                                    job_kind="native-campaign")

        def main_policy(g):
            policies.append("main")
            return BuildOrderPolicy(g)

        def elite(g):
            policies.append("elite")
            return BuildOrderPolicy(g)
        summary = run_campaigns(spec, factory, ctx, Journal(run_dir / "episodes.jsonl"), protocol=PROTOCOL,
                                make_policy=main_policy, ports=[9001], make_elite_policy=elite if elite_policy else None)
        return summary["campaigns"][0], made, policies

    def spec(self, levels, **overrides):
        return {"levels": levels, "seeds": [1001], "attempts_per_level": 1, "policy": "plan", "brain": "rule",
                "plans": {str(level): [WIN_PLAN] for level in levels}, **overrides}

    def test_a_lost_elite_stage_blocks_only_its_own_range(self):
        campaign, made, policies = self.run_spec(self.spec(list(range(13, 20)), start=MAIN), lose=(16,),
                                                 elite_policy=True)
        self.assertFalse(campaign["completed"])
        self.assertEqual([16], campaign["failed_levels"])
        self.assertEqual([17], campaign["blocked_levels"])
        self.assertEqual([13, 14, 15, 18, 19], campaign["levels_won"])
        self.assertIsNone(campaign["stopped_reason"])
        self.assertEqual(["elite"] * 6, policies)
        self.assertEqual({level: 3 for level in range(1, 13)},
                         {k: v for k, v in made[0][2]["levels"].items() if int(k) <= 12})

    def test_a_full_elite_rehearsal_completes(self):
        campaign, _, _ = self.run_spec(self.spec([13, 14], start=MAIN))
        self.assertTrue(campaign["completed"])
        self.assertEqual([13, 14], campaign["levels_won"])

    def test_a_lost_main_level_still_ends_the_campaign(self):
        campaign, made, _ = self.run_spec(self.spec([1, 2, 3]), lose=(2,))
        self.assertEqual(("level_failed", 2), (campaign["stopped_reason"], campaign["failed_level"]))
        self.assertNotIn("blocked_levels", campaign)  # main-campaign summaries keep their old shape
        self.assertEqual([1, 2], [level for level, _, _ in made])

    def test_spec_checks_for_start_progress(self):
        self.assertEqual([], check_campaign(self.spec(list(range(13, 27)), start=MAIN), POOLS, phase)[1])
        for bad in (self.spec([13], start={"1": 3, "3": 3}), self.spec([14], start=MAIN),
                    self.spec([13], start={"1": 4}), self.spec([27], start={**MAIN, **{str(l): 3 for l in range(13, 27)}}),
                    self.spec([13]), self.spec([13], start=[1])):
            self.assertIsNone(check_campaign(bad, POOLS, phase)[0])


class SearchBudgetTests(unittest.TestCase):
    def test_elite_search_mixes_in_immigrants_and_fallback_mutations(self):
        search = LevelSearch(13, HOLDERS, ["hero_gerald", "hero_ignus"], "s", population=8, rally=True)
        for _ in range(8):
            search.report(search.propose(), 1.0)
        children = [search.propose() for _ in range(200)]
        self.assertGreater(len({g.get("ratio", "3111") for g in children}), 3)
        plain = LevelSearch(5, HOLDERS, ["hero_gerald"], "s", population=8)
        for _ in range(8):
            plain.report(plain.propose(), 1.0)
        self.assertTrue(all("f" not in plain.propose() for _ in range(50)))

    def test_elite_population_keeps_one_plan_per_result(self):
        search = LevelSearch(13, HOLDERS, [], "s", population=8, rally=True)
        g = [search.propose() for _ in range(4)]
        for genome, score in zip(g, (5.0, 5.0, 3.0, 5.0)):
            search.report(genome, score)
        self.assertEqual([5.0, 3.0], [entry[0] for entry in search.population])
        self.assertEqual(genome_id(g[0]), search.population[0][1])  # the first plan with that result stays
        plain = LevelSearch(5, HOLDERS, [], "s", population=8)
        g = [plain.propose() for _ in range(3)]
        for genome in g:
            plain.report(genome, 5.0)
        self.assertEqual(3, len(plain.population))  # main-campaign searches are unchanged

    def test_improvement_tracking(self):
        search = LevelSearch(13, HOLDERS, [], "s", population=8, rally=True)
        g = [search.propose() for _ in range(3)]
        search.report(g[0], 5.0)
        search.report(g[1], 3.0)
        self.assertEqual((5.0, 1), (search.best_seen, search.improved_at))
        search.report(g[2], 9.0)
        self.assertEqual((9.0, 3), (search.best_seen, search.improved_at))

    def test_validation_runs_after_a_time_reserve_stop(self):
        from alpharush_rl import search_job
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "runs" / "native-search-0001"
            run_dir.mkdir(parents=True)
            start = time.monotonic()
            ctx = phase.PhaseRunContext("native-search-0001", run_dir, Path(tmp), start + 1000, 200,
                                        job_kind="native-search")
            calls = [0]

            def clock():  # the search phase runs into its 600 s reserve after a few proposals
                calls[0] += 1
                return start + (500 if calls[0] > 12 else 0)
            spec = {"levels": [1], "search_seed": 1001, "population": 4, "evaluations_per_level": 50,
                    "min_evaluations": 0, "stop_lives": 20, "profile_stars_per_level": 2, "workers": 2,
                    "validate_seeds": [1001, 1002], "validate_top": 2, "rng_seed": "unit"}
            out = Journal(run_dir / "episodes.jsonl")
            summary = search_job.search_loop(spec, lambda task, port: ToyEnv(seed=task["seed"], level=task["level"]),
                                             ctx, out, pools=POOLS, protocol=PROTOCOL, ports=[9001, 9002],
                                             time_reserve_seconds=600, clock=clock)
            self.assertEqual("time_reserve", summary["stopped_reason"])
            self.assertGreater(sum(row["kind"] == "search_validate" for row in out.entries()), 0)

    def test_seed_plans_are_played_first(self):
        from alpharush_rl import search_job
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "runs" / "native-search-0002"
            run_dir.mkdir(parents=True)
            ctx = phase.PhaseRunContext("native-search-0002", run_dir, Path(tmp), time.monotonic() + 300, 100,
                                        job_kind="native-search")
            spec = {"levels": [1], "search_seed": 1001, "population": 4, "evaluations_per_level": 3,
                    "min_evaluations": 0, "stop_lives": 20, "profile_stars_per_level": 2, "workers": 1,
                    "validate_seeds": [1002], "validate_top": 0, "rng_seed": "unit", "seed_plans": {"1": [WIN_PLAN]}}
            self.assertEqual([], search_job.check_search(spec, POOLS)[1])
            out = Journal(run_dir / "episodes.jsonl")
            search_job.search_loop(spec, lambda task, port: ToyEnv(seed=task["seed"], level=task["level"]), ctx, out,
                                   pools=POOLS, protocol=PROTOCOL, ports=[9001])
            rows = out.entries()
        evals = [r["payload"]["genome_id"] for r in rows if r["kind"] == "search_eval"]
        self.assertEqual(genome_id(WIN_PLAN), evals[0])
        self.assertEqual(1, sum(r["kind"] == "search_seed_plan" for r in rows))
        self.assertTrue(search_job.check_search({**spec, "seed_plans": {"2": [WIN_PLAN]}}, POOLS)[1])

    def test_spec_accepts_weights_and_stop_rules(self):
        from alpharush_rl import search_job
        spec = {"levels": [13, 14], "search_seed": 1001, "population": 8, "evaluations_per_level": 10,
                "min_evaluations": 0, "stop_lives": 19, "profile_stars_per_level": 2.75, "workers": 2,
                "validate_seeds": [1003], "validate_top": 1, "rng_seed": "t",
                "level_weights": {"13": 4, "14": 0.5}, "stop_wins": 3, "stall_evaluations": 50,
                "warm_start": {"runs": ["native-search-" + "0" * 32], "top": 4, "reevaluate": [13]}}
        self.assertEqual([], search_job.check_search(spec, POOLS)[1])
        for bad in ({"level_weights": {"15": 1}}, {"level_weights": {"13": 0}}, {"stop_wins": 0},
                    {"warm_start": {"runs": ["native-search-" + "0" * 32], "top": 4, "reevaluate": ["13"]}}):
            self.assertTrue(search_job.check_search({**spec, **bad}, POOLS)[1], bad)


class SafetyTests(unittest.TestCase):
    def test_retrying_broker_retries_transport_errors_only(self):
        import urllib.error
        from alpharush_rl.model_broker import RetryingBroker

        class Flaky:
            def __init__(self, failures):
                self.failures, self.calls = list(failures), 0

            def distribution(self, request):
                self.calls += 1
                if self.failures:
                    raise self.failures.pop(0)
                return {"p": [1.0]}
        slept, retries = [], []
        flaky = Flaky([urllib.error.URLError("down"), ConnectionResetError("reset")])
        broker = RetryingBroker(flaky, delays=(1, 2, 3), on_retry=lambda *a: retries.append(a[1]), sleep=slept.append)
        self.assertEqual({"p": [1.0]}, broker.distribution({"id": "x"}))
        self.assertEqual(([1, 2], [1, 2], 3), (slept, retries, flaky.calls))
        with self.assertRaises(ValueError):  # an invalid answer is not a transport failure
            RetryingBroker(Flaky([ValueError("bad")]), sleep=slept.append).distribution({"id": "y"})
        with self.assertRaises(urllib.error.URLError):
            RetryingBroker(Flaky([urllib.error.URLError("x")] * 3), delays=(0, 0), sleep=slept.append) \
                .distribution({"id": "z"})

    def test_an_exception_outside_a_game_ends_only_that_seed(self):
        from alpharush_rl.strategy_brain import RuleBrain

        class Broken(RuleBrain):
            def choose_plan(self, level, candidates, attempt, context):
                if level == 3:
                    raise ConnectionError("brain down")
                return super().choose_plan(level, candidates, attempt, context)
        with tempfile.TemporaryDirectory(prefix="alpharush-safety-") as tmp:
            run_dir = Path(tmp) / "runs" / "native-campaign-0001"
            run_dir.mkdir(parents=True)
            ctx = phase.PhaseRunContext("native-campaign-0001", run_dir, Path(tmp), time.monotonic() + 300, 200,
                                        job_kind="native-campaign")
            spec = {"levels": [1, 2, 3], "seeds": [1001, 1002], "attempts_per_level": 1, "policy": "plan",
                    "brain": "rule", "plans": {str(level): [WIN_PLAN] for level in (1, 2, 3)}}
            out = Journal(run_dir / "episodes.jsonl")
            summary = run_campaigns(spec, lambda level, seed, profile, attempt, port: ToyEnv(seed=seed, level=level,
                                                                                                  gold=600),
                                    ctx, out, protocol=PROTOCOL, make_policy=BuildOrderPolicy, ports=[9001, 9002],
                                    brain=Broken())
            for campaign in summary["campaigns"]:
                self.assertEqual({"1", "2"}, set(campaign["won"]))
                self.assertTrue(campaign["stopped_reason"].startswith("exception: ConnectionError"))
                self.assertFalse(campaign["completed"])
            self.assertTrue(summary["stopped_reason"].startswith("exception"))
            self.assertEqual(2, sum(row["kind"] == "campaign_exception" for row in out.entries()))


class WarmStartTests(unittest.TestCase):
    def test_replayed_plans_come_from_any_measured_seed(self):
        from dataclasses import asdict
        from alpharush_rl.ops import GateRefused
        from alpharush_rl.search_job import warm_entries
        import json as _json
        strong, weak, other = (genome([["b", "01", kind]]) for kind in ("mage", "archer", "barrack"))
        with tempfile.TemporaryDirectory(prefix="alpharush-warm-") as tmp:
            run = Path(tmp) / "native-search-0001"
            run.mkdir()
            spec_old = {"search_seed": 1001, "search_seeds": [1001, 1002], "profile_stars_per_level": [3, 1.2, 13]}
            (run / "events.jsonl").write_text(_json.dumps({"kind": "search_start", "payload": {
                "spec": spec_old, "protocol": asdict(PROTOCOL)}}) + "\n", encoding="utf-8")
            rows = [("search_eval", strong, 1001, 10500.0), ("search_validate", strong, 1005, 10400.0),
                    ("search_validate", strong, 1006, 900.0), ("search_eval", weak, 1001, 800.0),
                    ("search_validate", weak, 1005, 700.0), ("search_eval", other, 1002, 300.0)]
            (run / "episodes.jsonl").write_text("".join(_json.dumps({"kind": kind, "sha256": f"{i:064x}", "payload": {
                "level": 13, "seed": seed, "genome": g, "fitness": f}}) + "\n"
                for i, (kind, g, seed, f) in enumerate(rows)), encoding="utf-8")
            spec = {"levels": [13], "search_seed": 1005, "search_seeds": [1005, 1006, 1007, 1008],
                    "profile_stars_per_level": [3, 1.2, 13],
                    "warm_start": {"runs": [run.name], "top": 2, "reevaluate": True}}
            ranked = warm_entries(spec, tmp, PROTOCOL)[13]
            self.assertEqual([genome_id(strong), genome_id(weak)], [genome_id(e["genome"]) for e in ranked])
            with self.assertRaises(GateRefused):  # adopting scores still needs the same seeds and profile
                warm_entries({**spec, "warm_start": {"runs": [run.name], "top": 2}}, tmp, PROTOCOL)
            same = {**spec, "search_seed": 1001, "search_seeds": [1001, 1002],
                    "warm_start": {"runs": [run.name], "top": 3}}
            adopted = {genome_id(e["genome"]): e["fitness"]
                       for e in warm_entries(same, tmp, PROTOCOL)[13]}  # search_eval rows on its seeds only
            self.assertEqual({genome_id(strong): 5250.0, genome_id(weak): 400.0, genome_id(other): 150.0}, adopted)


class JournalTests(unittest.TestCase):
    def test_appends_track_the_tip_and_still_refuse_tampering(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "j.jsonl"
            journal = Journal(path)
            for i in range(30):
                journal.append("x", {"i": i})
            self.assertEqual(30, Journal(path).verify()["entries"])
            other = Journal(path)  # a second writer: each re-verifies when the file changed under it
            other.append("y", {})
            journal.append("z", {})
            self.assertEqual(32, Journal(path).verify()["entries"])
            text = path.read_text(encoding="utf-8").replace('"i":3}', '"i":4}', 1)
            path.write_text(text + "", encoding="utf-8")
            with self.assertRaises(JournalError):
                journal.append("w", {})


class GateTests(unittest.TestCase):
    """tools/campaign-survey.py: elite stages need scope v3 and the right operator layouts."""

    @classmethod
    def setUpClass(cls):
        import importlib.util
        tool = Path(__file__).resolve().parents[1] / "tools/campaign-survey.py"
        spec = importlib.util.spec_from_file_location("campaign_survey_gate_test", tool)
        cls.survey = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.survey)

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="alpharush-gates-")
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        for layout in ("v3", "v4"):
            net = operator_net.OptionScorer(hidden=(4,), layout=layout).to_json()
            (self.root / f"{layout}.json").write_text(__import__("json").dumps(net), encoding="utf-8")

    def job(self, scope="v3", main="v3.json", elite="v4.json"):
        job = {"action_scope": scope, "operator_weights": {"path": main, "sha256": "x"}}
        if elite:
            job["elite_operator_weights"] = {"path": elite, "sha256": "x"}
        return job

    def problems(self, job, levels, operator):
        return self.survey._elite_problems(job, self.root, levels, operator)

    def test_campaign_layouts(self):
        self.assertEqual([], self.problems(self.job(), list(range(1, 27)), "campaign"))
        self.assertEqual(["elite stages (levels 13-26) need action_scope v3"],
                         self.problems(self.job(scope="v2"), list(range(1, 27)), "campaign"))
        self.assertEqual(["an operator campaign with elite stages declares elite_operator_weights"],
                         self.problems(self.job(elite=None), list(range(1, 27)), "campaign"))
        self.assertEqual(["elite_operator_weights must be a layout-v4 network"],
                         self.problems(self.job(elite="v3.json"), [13, 14], "campaign"))
        self.assertEqual(["operator_weights (levels 1-12) must be a layout-v3 network"],
                         self.problems(self.job(main="v4.json"), [1, 13], "campaign"))
        self.assertEqual([], self.problems(self.job(scope="v2", elite=None), list(range(1, 13)), "campaign"))

    def test_single_network_jobs(self):
        self.assertEqual([], self.problems(self.job(main="v4.json", elite=None), [13, 20], "single"))
        self.assertEqual(["operator_weights must be a layout-v4 network for these levels"],
                         self.problems(self.job(elite=None), [13], "single"))
        self.assertIn("one operator network cannot play both main-campaign and elite tasks",
                      self.problems(self.job(main="v4.json", elite=None), [12, 13], "single"))
        self.assertEqual(["elite_operator_weights is only used by operator campaigns"],
                         self.problems(self.job(main="v4.json"), [13], "single"))
        self.assertEqual([], self.problems({"action_scope": "v3"}, [13, 26], None))
        self.assertEqual(["elite stages (levels 13-26) need action_scope v3"],
                         self.problems({"action_scope": "v2"}, [13], None))


class ScopeTests(unittest.TestCase):
    def test_v3_jobs_play_the_main_campaign_in_v2(self):
        self.assertEqual(["v2"] * 12 + ["v3"] * 14, [level_scope("v3", level) for level in range(1, 27)])
        self.assertEqual(["v2"] * 26, [level_scope("v2", level) for level in range(1, 27)])
        self.assertEqual("v1", level_scope("v1", 20))


class OperatorLayoutTests(unittest.TestCase):
    # sha256 over the toy episode's feature arrays computed by the operator module before the elite layout
    # existed (operator-d86c06c72ae1 era); layout v3 must keep producing exactly these numbers.
    V3_TOY_SHA256 = "7f38b88ab687348b223129d4dde5cbac3a9d7d0e5cdb9efc8f054faf7bd0dc4a"
    PLAN = genome([["b", "01", "mage"], ["b", "02", "archer"], ["u", "01"], ["u", "01"], ["k", "02"],
                   ["b", "03", "barrack"], ["u", "02"], ["u", "02"], ["u", "02"]], cast=40, early=0)

    def test_layout_v3_is_the_campaign_v1_feature_layout(self):
        self.assertEqual((62, 27), operator_net.DIMS["v3"])
        seen = []

        class Capture:
            name = "capture"

            def __init__(self, plan):
                self.policy, self.tracker = BuildOrderPolicy(plan), operator_net.PlanTracker(plan)

            def choose(self, state, menu, context):
                instruction = self.tracker.instruction(state, menu)
                choice = self.policy.choose(state, menu, context)
                index = next(i for i, item in enumerate(menu) if item["label"] == choice["label"])
                self.tracker.observe(instruction, menu[index]["action"])
                seen.append((state, menu, instruction))
                seen.append((state, menu, None))
                return choice
        capture = Capture(self.PLAN)
        for seed in (1001, 1002):
            run_episode(ToyEnv(seed=seed, gold=900), capture, PROTOCOL, seed=seed, level=1)
        digest = hashlib.sha256()
        for state, menu, instruction in seen:
            g, o = operator_net.decision_arrays(state, menu, instruction, "v3")
            digest.update(g.tobytes())
            digest.update(o.tobytes())
        self.assertEqual((66, self.V3_TOY_SHA256), (len(seen), digest.hexdigest()))

    def test_layout_v4_sees_levels_rally_and_bosses(self):
        g_dim, o_dim = operator_net.DIMS["v4"]
        state = {"level_idx": 20, "wave": 3, "wave_total": 15, "gold": 300, "lives": 20, "holders": [],
                 "towers": [{"id": 7, "holder_id": "01", "template": "tower_barrack_2", "rally_x": 10, "rally_y": 10}],
                 "enemies": [{"id": 5, "boss": True, "hp": 30, "hp_max": 60, "path_progress": 0.5, "x": 100, "y": 100},
                             {"id": 6, "boss": True, "dormant": True, "untargetable": True, "hp": 9, "hp_max": 9}]}
        menu = [WAIT, option("B", rally(7, "boss", 110, 100))]
        instruction = {"step": ["r", "01", "exit"], "cast": 50, "early": 1, "block": 1}
        g, o = operator_net.decision_arrays(state, menu, instruction, "v4")
        self.assertEqual(((g_dim,), (2, o_dim)), (g.shape, o.shape))
        self.assertEqual(1.0, g[19])  # level 20 one-hot
        elite = g[26 + 8 + 5 + 16 + 1 + 5:][:7]
        self.assertEqual([1.0, 0.5, 0.5, 1.0, 0.0, 0.0, 0.0], [round(float(x), 4) for x in elite])
        self.assertEqual(1.0, g[-12])  # the plan's block order
        self.assertEqual([0.3333, 0.1111, 0.1111, 0.1111, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                         [round(float(x), 4) for x in g[-11:]])  # default ratio 3111, f 50, no cap, first branches
        genes = {"step": None, "cast": 50, "early": 1, "block": 0, "ratio": "2021", "f": 400, "cap": 6,
                 "branches": {"archer": "tower_musketeer", "barrack": "tower_paladin", "mage": "tower_sorcerer",
                              "engineer": "tower_bfg"}}
        g2, _ = operator_net.decision_arrays(state, menu, genes, "v4")
        self.assertEqual([0.2222, 0.0, 0.2222, 0.1111, 1.0, 0.5, 1.0, 0.0, 1.0, 0.0, 1.0],
                         [round(float(x), 4) for x in g2[-11:]])
        tracker = operator_net.PlanTracker(genome([], ratio="2021", f=400))
        self.assertEqual(("2021", 400), (tracker.instruction(state, menu)["ratio"], tracker.instruction(state, menu)["f"]))
        rally_row = o[1]
        self.assertEqual(1.0, rally_row[operator_net.ACTIONS.index("set_rally")])
        self.assertEqual([0.0, 0.0, 0.0, 1.0], list(rally_row[-11:-7]))  # option "boss"
        self.assertAlmostEqual(np.hypot(100, 90) / 100, float(rally_row[-7]), places=4)  # from its rally point
        self.assertAlmostEqual(0.1, float(rally_row[-6]), places=4)  # to the awake boss
        self.assertEqual([1.0, 1.0, 0.0, 1.0], list(rally_row[-4:]))  # same op and holder, other option

    def test_weights_round_trip_keep_their_layout(self):
        for layout in ("v3", "v4"):
            net = operator_net.OptionScorer(hidden=(8,), seed=1, layout=layout)
            again = operator_net.OptionScorer.from_json(net.to_json())
            self.assertEqual(layout, again.layout)
            g = np.ones(operator_net.DIMS[layout][0], dtype=np.float32)
            o = np.ones((3, operator_net.DIMS[layout][1]), dtype=np.float32)
            self.assertTrue(np.allclose(net.scores(g, o), again.scores(g, o)))
        with self.assertRaises(ValueError):
            operator_net.OptionScorer.from_json({**operator_net.OptionScorer(layout="v4").to_json(), "layout": "v3"})


class BrainTests(unittest.TestCase):
    class Broker:
        def __init__(self):
            self.requests = []

        def distribution(self, request):
            self.requests.append(request)
            return {"p": [1.0 / len(request["labels"])] * len(request["labels"]), "model": "stub"}

    def test_elite_prompts_name_the_stage_and_main_prompts_are_unchanged(self):
        broker = self.Broker()
        brain = strategy_brain.LanguageBrain(broker)
        plans = [(genome([["b", "01", "barrack"], ["r", "01", "exit"]], block=1), None), (WIN_PLAN, None)]
        brain.choose_plan(19, plans, 0, {"progress": "p", "upgrades": {}})
        request = broker.requests[-1]
        self.assertEqual(strategy_brain.ELITE_SYSTEM, request["system"])
        self.assertIn("Elite stage 19 (Ha'Kraj Plateau; boss: Ulgukhai", request["user"])
        self.assertIn("1 barracks rally orders; barracks rally onto the boss to block it", request["user"])
        brain.choose_plan(5, [(WIN_PLAN, None), (WIN_PLAN, None)], 0, {"progress": "p", "upgrades": {}})
        request = broker.requests[-1]
        self.assertEqual(strategy_brain.SYSTEM, request["system"])
        self.assertTrue(request["user"].startswith("Campaign so far: p. Star upgrades owned: {}. Level 5 is next "
                                                   "(attempt 1). Holders are named by map slot ids."))
        self.assertNotIn("rally", request["user"])


if __name__ == "__main__":
    unittest.main()
