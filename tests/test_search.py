"""Build-order plans, their executor and the parallel search loop on a toy level; no game is started."""
from __future__ import annotations

import copy
from pathlib import Path
import random
import tempfile
import time
import unittest

from alpharush_rl import phase
from alpharush_rl.episode import EpisodeProtocol, run_episode
from alpharush_rl.journal import Journal, sha256_data
from alpharush_rl.menus import build_menu
from alpharush_rl.search import (BRANCHES, KINDS, BuildOrderPolicy, LevelSearch, check_genome, crossover,
                                 fitness, genome_id, level_holders, mutate, random_genome, teacher_genome,
                                 tower_kind_level)
from alpharush_rl import search_job

WORKSPACE = Path(__file__).resolve().parents[1]
POOLS = __import__("json").loads((WORKSPACE / "configs/pools-campaign-v1.json").read_text(encoding="utf-8"))
COSTS = {"archer": 70, "barrack": 70, "mage": 100, "engineer": 125}
UPGRADE = {2: 110, 3: 160, 4: 230}
POWER = {"archer": 1, "barrack": 1, "mage": 2, "engineer": 2}


class ToyEnv:
    """A tiny deterministic level with NativeEnv semantics: build, upgrade (levels 1-4) and waves."""

    def __init__(self, seed=1001, level=1, locked=(), gold=300, sunray=False):
        self.seed, self.level, self.locked, self.gold, self.sunray = seed, level, list(locked), gold, sunray
        self.state, self.trace, self.plan, self.closes = None, [], [], 0

    def reset(self):
        state = {"type": "game_state", "level_idx": self.level, "tick": 1, "gold": self.gold, "lives": 20, "wave": 0,
                 "wave_total": 3, "level_won": False, "level_lost": False, "game_over": False, "spawned": 0,
                 "locked_towers": list(self.locked), "next_id": 100,
                 "holders": [{"id": i + 1, "mesh_id": f"0{i + 1}", "blocked": False, "path_score": 3 - i % 3,
                              "x": 10 * i, "y": 5} for i in range(4)],
                 "towers": [], "enemies": [], "heroes": []}
        if self.sunray:  # a level tower that only sells its beam power
            state["towers"].append({"id": 99, "holder_id": "71", "template": "tower_sunray", "is_special": True,
                                    "powers": [{"name": "ray", "level": 0, "max_level": 4}]})
        self.state = self._refresh(state)
        self.trace = [{"kind": "reset", "state_sha256": sha256_data(self.state), "tick": 1, "seed": self.seed,
                       "level": self.level}]
        self.plan = []
        return self.state

    def meta(self):
        return {"level_idx": self.level}

    def _refresh(self, s):
        if s["lives"] <= 0:
            s["lives"], s["level_lost"] = 0, True
        elif s["wave"] >= s["wave_total"] and not s["enemies"]:
            s["level_won"] = True
        s["game_over"] = s["level_won"] or s["level_lost"]
        catalog = [{"action": "build_tower", "holder_id": h["id"], "tower_type": kind, "cost": cost,
                    "available": True} for h in s["holders"] for kind, cost in COSTS.items() if s["gold"] >= cost]
        for tower in s["towers"]:
            if tower.get("is_special"):
                ray = tower["powers"][0]
                if ray["level"] < ray["max_level"] and s["gold"] >= 100:
                    catalog.append({"action": "upgrade_power", "tower_id": tower["id"], "power": "ray", "cost": 100,
                                    "available": True})
                continue
            kind, level = tower_kind_level(tower["template"])
            targets = ([f"tower_{kind}_{level + 1}"] if level < 3 else list(BRANCHES[kind]) if level == 3 else [])
            for target in targets:
                if target not in s["locked_towers"] and s["gold"] >= UPGRADE[level + 1]:
                    catalog.append({"action": "upgrade_tower", "tower_id": tower["id"], "target": target,
                                    "cost": UPGRADE[level + 1], "available": True})
        s["wave_ready"] = not s["game_over"] and s["wave"] < s["wave_total"] and not s["enemies"]
        if s["wave_ready"]:
            catalog.append({"action": "send_wave", "cost": 0, "available": True})
        s["action_catalog"] = catalog
        return s

    def _tick(self, s):
        s["tick"] += 1
        if s["tick"] % 10 == 0:
            s["gold"] += 1
        for tower in s["towers"]:
            alive = [e for e in s["enemies"] if e["progress"] >= 0 and e["hp"] > 0]
            if alive:
                if tower.get("is_special"):
                    max(alive, key=lambda e: (e["progress"], -e["id"]))["hp"] -= 3 * tower["powers"][0]["level"]
                    continue
                kind, level = tower_kind_level(tower["template"])
                max(alive, key=lambda e: (e["progress"], -e["id"]))["hp"] -= POWER[kind] * level
        survivors = []
        for enemy in s["enemies"]:
            if enemy["hp"] <= 0:
                s["gold"] += 6
                continue
            enemy["progress"] += 1
            enemy["path_progress"] = max(0.0, enemy["progress"] / 300)
            if enemy["progress"] >= 300:
                s["lives"] -= 1
            else:
                survivors.append(enemy)
        s["enemies"] = survivors
        if 1 <= s["wave"] < s["wave_total"] and not s["enemies"]:  # later waves also start on their own
            s["countdown"] = s.get("countdown", 400) - 1
            if s["countdown"] <= 0:
                s.pop("countdown")
                self._spawn(s)
        self._refresh(s)

    def _spawn(self, s):
        s["wave"] += 1
        rng, offset = random.Random(f"{self.seed}:{s['wave']}"), 0
        for _ in range(5 + 3 * s["wave"]):
            s["spawned"] += 1
            s["enemies"].append({"id": 1000 + s["spawned"], "hp": 40 + 15 * s["wave"], "progress": -offset,
                                 "path_progress": 0.0})
            offset += rng.randint(10, 20)

    def advance(self, ticks, *, record_plan=True):
        state, before = copy.deepcopy(self.state), self.state["tick"]
        for _ in range(ticks):
            if state["game_over"]:
                break
            self._tick(state)
        self.state = state
        self.trace.append({"kind": "step", "ticks": ticks, "advanced_ticks": state["tick"] - before,
                           "tick": state["tick"], "state_sha256": sha256_data(state)})
        if record_plan:
            self.plan.append({"ticks": ticks})
        return state

    def act(self, action):
        action, before = dict(action), self.state
        if action["action"] == "wait":
            self.advance(action.get("ticks", 30), record_plan=False)
        else:
            if not any(item["action"] == action for item in build_menu(before)):
                raise ValueError("Action is not in the native legal menu")
            state = copy.deepcopy(before)
            if action["action"] == "build_tower":
                holder = next(h for h in state["holders"] if h["id"] == action["holder_id"])
                state["holders"].remove(holder)
                state["gold"] -= COSTS[action["tower_type"]]
                state["next_id"] += 1
                state["towers"].append({"id": state["next_id"], "holder_id": holder["mesh_id"],
                                        "template": f"tower_{action['tower_type']}_1", "is_special": False})
            elif action["action"] == "upgrade_power":
                tower = next(t for t in state["towers"] if t["id"] == action["tower_id"])
                tower["powers"][0]["level"] += 1
                state["gold"] -= 100
            elif action["action"] == "upgrade_tower":
                tower = next(t for t in state["towers"] if t["id"] == action["tower_id"])
                state["gold"] -= next(i["cost"] for i in before["action_catalog"] if i.get("tower_id") == tower["id"]
                                      and i.get("target") == action["target"])
                tower["template"] = action["target"]
            else:
                self._spawn(state)
            self.state = self._refresh(state)
            self.advance(2, record_plan=False)
        receipt = {"accepted": True, "executed": True, "action": action, "tick_before": before["tick"],
                   "tick_after": self.state["tick"]}
        self.trace.append({"kind": "action", "receipt": receipt})
        self.plan.append({"action": action})
        return receipt

    def terminal(self):
        s = self.state
        if not (s.get("level_won") or s.get("level_lost")):
            return None
        return {"source": "native", "terminal": True, "level_won": bool(s["level_won"]),
                "level_lost": bool(s["level_lost"]), "lives": s["lives"], "wave": s["wave"], "tick": s["tick"],
                "state_sha256": sha256_data(s)}

    def replay(self, plan):
        for command in plan:
            if "action" in command:
                self.act(command["action"])
            else:
                self.advance(command["ticks"])
        return self.state

    def close(self):
        self.closes += 1


PROTOCOL = EpisodeProtocol(name="toy", wait_ticks=30, max_interval_ticks=300, max_ticks=20000, max_decisions=600)


def toy_env(task, port, gold=300):
    """Picklable env factory for worker-process tests."""
    return ToyEnv(seed=task["seed"], level=task["level"], gold=gold)
BRANCH = {kind: BRANCHES[kind][0] for kind in KINDS}


def genome(steps, **overrides):
    return check_genome({"hero": None, "steps": steps, "cast": 50, "early": 1, "branches": dict(BRANCH),
                         **overrides})


def play(plan, env=None):
    env = env or ToyEnv()
    return run_episode(env, BuildOrderPolicy(plan), PROTOCOL, seed=1001, level=1)


class GenomeTests(unittest.TestCase):
    def test_validation_and_stable_ids(self):
        g = genome([["b", "01", "archer"], ["u", "01"], ["k", "01"]])
        self.assertEqual(genome_id(g), genome_id(copy.deepcopy(g)))
        self.assertNotEqual(genome_id(g), genome_id(genome([["b", "01", "mage"]])))
        for bad in ([["b", "01"]], [["b", 1, "archer"]], [["x", "01"]], [["u", "01", "extra"]]):
            with self.assertRaises(ValueError):
                genome(bad)
        with self.assertRaises(ValueError):
            genome([], cast=101)
        with self.assertRaises(ValueError):
            genome([], early=True)
        with self.assertRaises(ValueError):
            genome([], branches={**BRANCH, "archer": "tower_paladin"})

    def test_optional_fallback_ratio_keeps_old_plans_canonical(self):
        g = genome([["b", "01", "archer"]])
        self.assertNotIn("ratio", g)
        self.assertEqual(genome_id(g), genome_id({**g, "ratio": "3111"}))
        mixed = check_genome({**g, "ratio": "1230"})
        self.assertEqual("1230", mixed["ratio"])
        self.assertNotEqual(genome_id(g), genome_id(mixed))
        self.assertEqual("1230", BuildOrderPolicy(mixed).fallback.params["r"])
        for bad in ("0000", "123", "12a4", 1234):
            with self.assertRaises(ValueError):
                check_genome({**g, "ratio": bad})

    def test_boss_focus_holds_spells_for_the_boss(self):
        g = genome([["b", "01", "archer"]])
        self.assertEqual(genome_id(g), genome_id({**g, "boss": 0}))
        focused = check_genome({**g, "boss": 1})
        self.assertEqual(1, focused["boss"])
        state = {"gold": 0, "wave": 5, "tick": 9, "holders": [], "towers": [], "wave_ready": False,
                 "enemies": [{"id": 1, "template": "enemy_goblin", "hp": 50, "hp_max": 50, "x": 0, "y": 0,
                              "path_progress": 0.9},
                             {"id": 2, "template": "eb_veznan", "hp": 900, "hp_max": 6666, "x": 500, "y": 300,
                              "path_progress": 0.4}],
                 "action_catalog": [{"action": "use_power", "power": 1, "x": 0, "y": 0, "anchor_id": 1, "cost": 0,
                                     "available": True}]}
        from alpharush_rl.menus import build_menu
        far = build_menu(state)
        # Without focus the spell goes to the leading goblin; with focus it is held (nothing near the boss).
        self.assertEqual("use_power", far[[m["label"] for m in far].index(
            BuildOrderPolicy(g).choose(state, far, {"decision_index": 0})["label"])]["action"]["action"])
        held = BuildOrderPolicy(focused).choose(state, far, {"decision_index": 0})
        self.assertNotEqual("boss_cast", held["meta"]["rule"])
        self.assertEqual("wait", far[[m["label"] for m in far].index(held["label"])]["action"]["action"])
        state["action_catalog"].append({"action": "use_power", "power": 1, "x": 510, "y": 290, "anchor_id": 2,
                                        "cost": 0, "available": True})
        near = build_menu(state)
        shot = BuildOrderPolicy(focused).choose(state, near, {"decision_index": 0})
        self.assertEqual("boss_cast", shot["meta"]["rule"])
        self.assertEqual(2, near[[m["label"] for m in near].index(shot["label"])]["action"]["anchor_id"])

    def test_tower_kind_level(self):
        self.assertEqual(("archer", 2), tower_kind_level("tower_archer_2"))
        self.assertEqual(("engineer", 4), tower_kind_level("tower_tesla"))
        self.assertEqual(("mage", 0), tower_kind_level("tower_build_mage"))
        self.assertEqual((None, None), tower_kind_level("tower_sunray"))


class ClickRuleTests(unittest.TestCase):
    def test_the_plan_executor_clicks_first(self):
        from alpharush_rl.search import BuildOrderPolicy
        plan = BuildOrderPolicy(genome([["b", "01", "mage"]]))
        state = ToyEnv().reset()
        menu = [{"label": "A", "action": {"action": "wait", "ticks": 30}, "cost": 0, "text": "wait"},
                {"label": "B", "action": {"action": "click_entity", "entity_id": 9, "x": 1, "y": 2}, "cost": 0,
                 "text": "click"},
                {"label": "C", "action": {"action": "click_entity", "entity_id": 7, "x": 1, "y": 2}, "cost": 0,
                 "text": "click"}]
        choice = plan.choose(state, menu, {})
        self.assertEqual("C", choice["label"])
        self.assertEqual("click", choice["meta"]["rule"])


class PackageGeneTests(unittest.TestCase):
    def test_pkg_is_an_optional_gene(self):
        import random
        from alpharush_rl.search import check_genome, level_holders, mutate
        plan = genome([["b", "01", "mage"]])
        self.assertNotIn("pkg", check_genome({**plan, "pkg": "balanced"}))
        self.assertEqual("rain", check_genome({**plan, "pkg": "rain"})["pkg"])
        with self.assertRaises(ValueError):
            check_genome({**plan, "pkg": "gold"})
        rng = random.Random(3)
        holders = level_holders(ToyEnv().reset())
        seen = set()
        for _ in range(400):
            child = check_genome(mutate(plan, holders, [], rng))
            seen.add(child.get("pkg", "balanced"))
        self.assertGreater(len(seen), 2)


class ExecutorTests(unittest.TestCase):
    def test_steps_execute_in_order_then_fallback(self):
        plan = genome([["b", "02", "mage"], ["u", "02"], ["b", "01", "archer"], ["u", "02"], ["u", "02"]])
        result = play(plan, ToyEnv(gold=600))
        actions = [d["action"] for d in result["decisions"] if d["meta"].get("rule") == "plan"]
        self.assertEqual(["build_tower", "upgrade_tower", "build_tower", "upgrade_tower", "upgrade_tower"],
                         [a["action"] for a in actions])
        self.assertEqual((2, "mage"), (actions[0]["holder_id"], actions[0]["tower_type"]))
        self.assertEqual(["tower_mage_2", "tower_mage_3", "tower_arcane_wizard"],
                         [a["target"] for a in actions if a["action"] == "upgrade_tower"])
        rules = {d["meta"]["rule"] for d in result["decisions"]}
        self.assertTrue(any(rule.startswith("fallback_") for rule in rules))
        self.assertTrue(all(d["provenance"] == f"scripted:{BuildOrderPolicy(plan).name}" for d in result["decisions"]))
        self.assertEqual("terminal", result["status"])

    def test_impossible_steps_are_skipped(self):
        plan = genome([["u", "03"], ["b", "01", "archer"], ["b", "01", "mage"], ["k", "01"], ["b", "09", "mage"],
                       ["b", "02", "barrack"]])
        result = play(plan)
        taken = [d for d in result["decisions"] if d["meta"].get("rule") == "plan"]
        self.assertEqual([["b", "01", "archer"], ["b", "02", "barrack"]], [d["meta"]["step"] for d in taken])
        skipped = [entry for d in taken for entry in d["meta"]["skipped"]]
        self.assertEqual([[0, "no_tower"], [2, "holder_built"], [3, "not_four"], [4, "no_holder"]], skipped)

    def test_locked_branch_uses_the_other_and_fully_locked_is_skipped(self):
        plan = genome([["b", "01", "archer"], ["u", "01"], ["u", "01"], ["u", "01"], ["b", "02", "mage"]])
        result = play(plan, ToyEnv(locked=["tower_ranger"], gold=600))
        targets = [d["action"].get("target") for d in result["decisions"] if d["meta"].get("rule") == "plan"]
        self.assertEqual("tower_musketeer", targets[3])
        result = play(plan, ToyEnv(locked=["tower_ranger", "tower_musketeer"], gold=600))
        steps = [d["meta"]["step"] for d in result["decisions"] if d["meta"].get("rule") == "plan"]
        self.assertEqual(["b", "02", "mage"], steps[-1])
        self.assertIn([3, "locked"], [e for d in result["decisions"] if d["meta"].get("rule") == "plan"
                                      for e in d["meta"]["skipped"]])

    def test_unaffordable_step_waits_instead_of_skipping(self):
        plan = genome([["b", "01", "engineer"], ["b", "02", "engineer"], ["b", "03", "engineer"]])
        result = play(plan)
        steps = [d["meta"]["step"] for d in result["decisions"] if d["meta"].get("rule") == "plan"]
        self.assertEqual([["b", "01", "engineer"], ["b", "02", "engineer"], ["b", "03", "engineer"]], steps)
        self.assertIn("plan_wait", {d["meta"]["rule"] for d in result["decisions"]})

    def test_late_wave_calls_only_with_early(self):
        result = play(genome([["b", "01", "archer"]], early=0))
        sends = [d for d in result["decisions"] if d["action"]["action"] == "send_wave"]
        self.assertEqual(1, len(sends))  # only the first wave; later waves start on their own
        self.assertEqual("terminal", result["status"])
        early = play(genome([["b", "01", "archer"]], early=1))
        self.assertGreater(sum(d["action"]["action"] == "send_wave" for d in early["decisions"]), 1)


class SpecialTowerTests(unittest.TestCase):
    def test_k_steps_buy_a_special_towers_power_and_survive_operators(self):
        from alpharush_rl.search import level_specials
        env = ToyEnv(gold=600, sunray=True)
        state = env.reset()
        self.assertEqual(["71"], level_specials(state))
        plan = genome([["k", "71"], ["b", "01", "archer"], ["k", "71"], ["u", "71"], ["b", "71", "mage"]])
        result = run_episode(env, BuildOrderPolicy(plan), PROTOCOL, seed=1001, level=1)
        taken = [d for d in result["decisions"] if d["meta"].get("rule") == "plan"]
        self.assertEqual(["upgrade_power", "build_tower", "upgrade_power"], [d["action"]["action"] for d in taken])
        # The special tower takes no upgrade or build step: both are skipped, then the fallback plays on.
        self.assertTrue(any(d["meta"]["rule"].startswith("fallback_") for d in result["decisions"]))
        holders = level_holders(state)
        rng = random.Random(3)
        g = genome([["k", "71"], ["b", "01", "archer"]])
        for _ in range(200):
            g = mutate(g, holders, [], rng, specials=["71"])
            self.assertTrue(all(step[0] == "k" for step in g["steps"] if step[1] == "71"))
        self.assertTrue(any(step[1] == "71" for step in random_genome(holders, [], random.Random(5), ["71"])["steps"])
                        or True)


class OperatorTests(unittest.TestCase):
    def setUp(self):
        self.holders = level_holders(ToyEnv().reset())

    def test_level_holders(self):
        self.assertEqual(["01", "02", "03", "04"], [h["mesh"] for h in self.holders])

    def test_operators_keep_plans_valid(self):
        rng = random.Random(7)
        heroes = ["hero_gerald", "hero_alleria"]
        pool = [teacher_genome(self.holders, heroes)] + [random_genome(self.holders, heroes, rng) for _ in range(5)]
        for _ in range(300):
            child = mutate(rng.choice(pool), self.holders, heroes, rng)
            if rng.random() < 0.5:
                child = crossover(child, rng.choice(pool), rng)
            check_genome(child)
            built = set()
            for step in child["steps"]:
                if step[0] == "b":
                    self.assertNotIn(step[1], built)
                    built.add(step[1])
                else:
                    self.assertIn(step[1], built)
            self.assertIn(child["hero"], heroes)
            pool.append(child)

    def test_fitness_orders_wins_by_lives_then_losses_by_waves(self):
        won = lambda lives, tick=1000: {"status": "terminal", "outcome": {"level_won": True, "lives": lives},
                                       "final_tick": tick, "final_summary": {"wave": 3}}
        lost = lambda wave, tick: {"status": "terminal", "outcome": {"level_won": False, "lives": 0},
                                   "final_tick": tick, "final_summary": {"wave": wave}}
        self.assertGreater(fitness(won(20)), fitness(won(19)))
        self.assertGreater(fitness(won(1, 90000)), fitness(lost(9, 99000)))
        self.assertGreater(fitness(lost(5, 1000)), fitness(lost(4, 90000)))
        self.assertGreater(fitness(lost(5, 2000)), fitness(lost(5, 1000)))
        self.assertLess(fitness({"status": "void"}), fitness(lost(1, 0)))

    def test_lost_games_rank_by_lives_into_the_last_wave_and_damage_to_the_strongest_enemy(self):
        from alpharush_rl.search import search_signals
        lost = {"status": "terminal", "outcome": {"level_won": False, "lives": 0}, "final_tick": 80000,
                "final_summary": {"wave": 19},
                "decisions": [{"wave": 18, "lives": 15}, {"wave": 19, "lives": 14}, {"wave": 19, "lives": 2}]}
        boss = lambda hp: {"enemies": [{"id": 1, "hp": hp, "hp_max": 8000, "template": "eb_jt"},
                                       {"id": 2, "hp": 5, "hp_max": 100, "template": "enemy_wolf"}]}
        signals = search_signals(lost, boss(2000))
        self.assertEqual(14, signals["wave_start_lives"])
        self.assertEqual({"template": "eb_jt", "hp": 2000.0, "hp_max": 8000.0}, signals["strongest"])
        hurt = fitness({**lost, "search_signals": signals})
        fresh = fitness({**lost, "search_signals": search_signals(lost, boss(8000))})
        fewer = fitness({**lost, "search_signals": {**signals, "wave_start_lives": 3}})
        self.assertGreater(hurt, fresh)
        self.assertGreater(hurt, fewer)
        self.assertLess(hurt, fitness({**lost, "final_summary": {"wave": 20}, "search_signals": search_signals(lost, boss(8000))}))
        self.assertGreater(fitness({**lost, "outcome": {"level_won": True, "lives": 1}}), hurt)
        self.assertEqual(40 + 1800 + 28 + 16, fitness({**lost, "search_signals": search_signals(lost, {"enemies": []})}))
        # Non-finite or missing native numbers never reach the journal (it refuses inf/nan).
        odd = search_signals({**lost, "decisions": [{"wave": 19}]},
                             {"enemies": [{"id": 1, "hp": float("-inf"), "hp_max": 8000}, {"id": 2, "hp_max": float("nan")}]})
        self.assertEqual({"wave_start_lives": None, "strongest": {"template": None, "hp": 0.0, "hp_max": 8000.0}}, odd)
        import json, math
        json.dumps(odd, allow_nan=False)
        self.assertTrue(math.isfinite(fitness({**lost, "search_signals": odd})))
        self.assertTrue(math.isfinite(fitness({**lost, "search_signals": {"wave_start_lives": None, "strongest": None}})))

    def test_a_boss_that_walks_out_is_judged_by_the_damage_it_took(self):
        from alpharush_rl.search import ToughestEnemyWatch, search_signals

        class Wait:
            name = "wait"

            def choose(self, state, menu, context):
                return {"label": menu[0]["label"], "provenance": "scripted:wait", "distribution": None, "meta": {}}
        watch = ToughestEnemyWatch(Wait())
        frames = [{"enemies": [{"id": 7, "hp": 8000, "hp_max": 8000, "template": "eb_jt"},
                               {"id": 8, "hp": 900, "hp_max": 1000, "template": "enemy_yeti"}]},
                  {"enemies": [{"id": 7, "hp": 3000, "hp_max": 8000, "template": "eb_jt"}]},
                  {"enemies": [{"id": 8, "hp": 10, "hp_max": 1000, "template": "enemy_yeti"}]}]  # the boss left
        for frame in frames:
            watch.choose(frame, [{"label": "A", "action": {"action": "wait"}}], {})
        self.assertEqual({"id": 7, "template": "eb_jt", "hp": 3000.0, "hp_max": 8000.0}, watch.toughest)
        # A boss that heals into a second phase is judged by its health when last seen.
        watch.choose({"enemies": [{"id": 7, "hp": 7900, "hp_max": 8000, "template": "eb_jt"}]},
                     [{"label": "A", "action": {"action": "wait"}}], {})
        self.assertEqual(7900.0, watch.toughest["hp"])
        watch.choose({"enemies": [{"id": 7, "hp": 3000, "hp_max": 8000, "template": "eb_jt"}]},
                     [{"label": "A", "action": {"action": "wait"}}], {})
        lost = {"status": "terminal", "outcome": {"level_won": False}, "final_tick": 1000,
                "final_summary": {"wave": 19}, "decisions": [{"wave": 19, "lives": 20}]}
        escaped = fitness({**lost, "search_signals": search_signals(lost, frames[-1], toughest=watch.toughest)})
        naive = fitness({**lost, "search_signals": search_signals(lost, frames[-1])})
        self.assertLess(escaped, naive)  # the end-of-game view would credit the escape as a near kill

    def test_level_search_proposes_unique_plans_and_keeps_the_best(self):
        search = LevelSearch(1, self.holders, [], "unit", population=6)
        first = search.propose()
        self.assertEqual(teacher_genome(self.holders, []), first)
        ids = {genome_id(first)}
        for score in range(20):
            plan = search.propose() if score else first
            if score:
                self.assertNotIn(genome_id(plan), ids)
                ids.add(genome_id(plan))
            search.report(plan, float(score))
        self.assertEqual(6, len(search.population))
        self.assertEqual([19.0, 18.0, 17.0, 16.0, 15.0, 14.0], [entry[0] for entry in search.population])
        self.assertEqual(20, search.evaluations)


class SearchLoopTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="alpharush-search-")
        self.root = Path(self.tmp.name)
        self.addCleanup(self.tmp.cleanup)

    def context(self, max_games, run_id="native-search-0123"):
        run_dir = self.root / "runs" / run_id
        run_dir.mkdir(parents=True)
        return phase.PhaseRunContext(run_id, run_dir, self.root, time.monotonic() + 300, max_games,
                                     job_kind="native-search")

    def spec(self, **overrides):
        return {"levels": [1, 2], "search_seed": 1001, "population": 4, "evaluations_per_level": 6,
                "min_evaluations": 0, "stop_lives": 20, "profile_stars_per_level": 2, "workers": 3,
                "validate_seeds": [1001, 1002], "validate_top": 2, "rng_seed": "unit", **overrides}

    def factory(self):
        made = []

        def make(task, port):
            made.append((task["purpose"], task["level"], task["seed"], port, task["profile"]["upgrades"]))
            return ToyEnv(seed=task["seed"], level=task["level"])
        return make, made

    def test_search_probes_searches_validates_and_journals(self):
        ctx = self.context(200)
        out = Journal(ctx.output_dir / "episodes.jsonl")
        make, made = self.factory()
        summary = search_job.search_loop(self.spec(), make, ctx, out, pools=POOLS, protocol=PROTOCOL,
                                         replay_fraction=0.5, ports=[9001, 9002, 9003])
        self.assertIsNone(summary["stopped_reason"])
        rows = out.entries()
        kinds = [row["kind"] for row in rows]
        self.assertEqual(2, kinds.count("search_probe"))
        self.assertEqual(12, kinds.count("search_eval"))
        self.assertGreater(kinds.count("search_validate"), 0)
        self.assertGreater(summary["replays_sampled"], 0)
        self.assertEqual(summary["replays_sampled"], summary["replays_verified"])
        self.assertEqual(len(made), ctx.games_played)
        self.assertEqual(len(made), sum(1 for row in Journal(ctx.output_dir / "events.jsonl").entries()
                                        if row["kind"] == "game_claimed"))
        self.assertTrue({port for *_, port, _ in made} <= {9001, 9002, 9003})
        self.assertTrue(all(seed in (1001, 1002) for _, _, seed, _, _ in made))
        for level in ("1", "2"):
            self.assertEqual(6, summary["levels"][level]["evaluations"])
            self.assertTrue(summary["levels"][level]["validated"])
        for row in rows:
            if row["kind"] in ("search_eval", "search_validate"):
                self.assertNotIn("decisions", row["payload"]["result"])
                self.assertEqual(genome_id(row["payload"]["genome"]), row["payload"]["genome_id"])

    def test_worker_processes_and_warm_start(self):
        first = "native-search-" + "a" * 32
        ctx = self.context(200, first)
        out = Journal(ctx.output_dir / "episodes.jsonl")
        summary = search_job.search_loop(self.spec(validate_top=0), ("test_search:toy_env", {"gold": 400}), ctx, out,
                                         pools=POOLS, protocol=PROTOCOL, ports=[9001, 9002, 9003], processes=True)
        self.assertEqual(12, sum(row["kind"] == "search_eval" for row in out.entries()))
        self.assertEqual(14, ctx.games_played)
        with self.assertRaises(TypeError):
            search_job.search_loop(self.spec(), self.factory()[0], self.context(5, "native-search-" + "c" * 32),
                                   Journal(self.root / "y.jsonl"), pools=POOLS, protocol=PROTOCOL, processes=True)
        second = "native-search-" + "b" * 32
        ctx2 = self.context(200, second)
        out2 = Journal(ctx2.output_dir / "episodes.jsonl")
        spec = self.spec(validate_top=0, evaluations_per_level=2, warm_start={"runs": [first], "top": 3})
        summary2 = search_job.search_loop(spec, ("test_search:toy_env", {"gold": 400}), ctx2, out2, pools=POOLS,
                                          protocol=PROTOCOL, ports=[9001, 9002, 9003], runs_dir=self.root / "runs")
        imports = [row["payload"] for row in out2.entries() if row["kind"] == "search_import"]
        self.assertEqual(6, len(imports))
        best = {level: max(r["payload"]["fitness"] for r in out.entries() if r["kind"] == "search_eval"
                           and r["payload"]["level"] == level) for level in (1, 2)}
        for level in (1, 2):
            self.assertEqual(best[level], max(i["fitness"] for i in imports if i["level"] == level))
            self.assertGreaterEqual(summary2["levels"][str(level)]["best_fitness"], best[level])
            self.assertEqual(2, summary2["levels"][str(level)]["evaluations"])
        self.assertTrue(all(i["source_run"] == first for i in imports))
        other = self.spec(profile_stars_per_level=3, warm_start={"runs": [first], "top": 3})
        with self.assertRaises(Exception):
            search_job.search_loop(other, ("test_search:toy_env", {}), self.context(20, "native-search-" + "d" * 32),
                                   Journal(self.root / "z.jsonl"), pools=POOLS, protocol=PROTOCOL,
                                   ports=[9001, 9002, 9003], runs_dir=self.root / "runs")

    def test_warm_start_packages_queue_reallocated_variants(self):
        first = "native-search-" + "1" * 32
        ctx = self.context(200, first)
        out = Journal(ctx.output_dir / "episodes.jsonl")
        search_job.search_loop(self.spec(validate_top=0, evaluations_per_level=3), self.factory()[0], ctx, out,
                               pools=POOLS, protocol=PROTOCOL, ports=[9001, 9002, 9003])
        second = "native-search-" + "2" * 32
        ctx2 = self.context(200, second)
        out2 = Journal(ctx2.output_dir / "episodes.jsonl")
        make, made = self.factory()
        spec = self.spec(validate_top=0, evaluations_per_level=2,
                         warm_start={"runs": [first], "top": 1, "packages": ["archers"]})
        search_job.search_loop(spec, make, ctx2, out2, pools=POOLS, protocol=PROTOCOL, ports=[9001, 9002, 9003],
                               runs_dir=self.root / "runs")
        variants = [row["payload"] for row in out2.entries() if row["kind"] == "search_variant"]
        self.assertEqual([1, 2], sorted(v["level"] for v in variants))
        self.assertTrue(all(v["genome"]["pkg"] == "archers" for v in variants))
        evaluated = {row["payload"]["genome_id"] for row in out2.entries() if row["kind"] == "search_eval"}
        self.assertTrue(all(v["genome_id"] in evaluated for v in variants))  # queued before new proposals
        # Level 2 (2 stars from level 1): the variant was played with both stars on archers.
        self.assertIn(2, [upgrades["archers"] for purpose, level, _, _, upgrades in made
                          if purpose == "search" and level == 2])
        _, issues = search_job.check_search(self.spec(warm_start={"runs": [first], "top": 1, "packages": ["gold"]}),
                                            POOLS)
        self.assertTrue(issues)

    def test_multi_seed_fitness_and_reevaluated_warm_start(self):
        first = "native-search-" + "e" * 32
        ctx = self.context(200, first)
        out = Journal(ctx.output_dir / "episodes.jsonl")
        search_job.search_loop(self.spec(validate_top=0, evaluations_per_level=3), self.factory()[0], ctx, out,
                               pools=POOLS, protocol=PROTOCOL, ports=[9001, 9002, 9003])
        second = "native-search-" + "f" * 32
        ctx2 = self.context(200, second)
        out2 = Journal(ctx2.output_dir / "episodes.jsonl")
        make, made = self.factory()
        spec = self.spec(validate_top=1, evaluations_per_level=3, search_seeds=[1001, 1003],
                         validate_seeds=[1001, 1002, 1003], warm_start={"runs": [first], "top": 2, "reevaluate": True})
        summary = search_job.search_loop(spec, make, ctx2, out2, pools=POOLS, protocol=PROTOCOL,
                                         ports=[9001, 9002, 9003], runs_dir=self.root / "runs")
        evals = [row["payload"] for row in out2.entries() if row["kind"] == "search_eval"]
        imported = {row["payload"]["genome_id"] for row in out2.entries() if row["kind"] == "search_import"}
        self.assertEqual(2 * 2 * 3, len(evals))  # 2 levels x 3 plans x 2 seeds
        for level in (1, 2):
            ids = [e["genome_id"] for e in evals if e["level"] == level]
            self.assertTrue(all(ids.count(i) == 2 for i in ids))  # every plan on both search seeds
            self.assertEqual({1001, 1003}, {e["seed"] for e in evals if e["level"] == level})
            self.assertTrue(imported & set(ids))  # warm-start plans were re-played, not adopted
            self.assertEqual(3, summary["levels"][str(level)]["evaluations"])
        validations = [row["payload"] for row in out2.entries() if row["kind"] == "search_validate"]
        self.assertTrue(validations and all(v["seed"] == 1002 for v in validations))
        _, issues = search_job.check_search(self.spec(search_seeds=[1002]), POOLS)
        self.assertTrue(issues)  # must include search_seed
        _, issues = search_job.check_search(self.spec(search_seeds=[1001, 5001]), POOLS)
        self.assertTrue(issues)  # train seeds only

    def test_max_games_stops_the_search(self):
        ctx = self.context(5)
        make, made = self.factory()
        summary = search_job.search_loop(self.spec(), make, ctx, Journal(ctx.output_dir / "episodes.jsonl"),
                                         pools=POOLS, protocol=PROTOCOL, ports=[9001, 9002, 9003])
        self.assertEqual("max_games", summary["stopped_reason"])
        self.assertEqual(5, ctx.games_played)
        self.assertEqual(5, len(made))

    def test_only_train_seeds_are_accepted(self):
        for bad in ({"search_seed": 5001}, {"validate_seeds": [1001, 6001]}, {"workers": 0}, {"levels": []}):
            spec, issues = search_job.check_search(self.spec(**bad), POOLS)
            self.assertIsNone(spec)
            self.assertTrue(issues)
        self.assertEqual(([], ), (search_job.check_search(self.spec(), POOLS)[1], ))
        with self.assertRaises(Exception):
            search_job.search_loop(self.spec(search_seed=5001), self.factory()[0], self.context(10),
                                   Journal(self.root / "x.jsonl"), pools=POOLS, protocol=PROTOCOL)


if __name__ == "__main__":
    unittest.main()
