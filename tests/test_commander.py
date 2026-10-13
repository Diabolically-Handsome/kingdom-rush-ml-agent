"""Action scope v4 plumbing and the in-battle commander on toy levels and fake menus; no game, process or model
is started (the 8B is a fake scoring broker)."""
from __future__ import annotations

import copy
import json
import math
from pathlib import Path
import random
import tempfile
import time
import unittest
from unittest import mock

from alpharush_rl import commander, engine, env as env_module, eval_job, phase
from alpharush_rl.commander import (Briefing, CommanderPolicy, ForcedChoice, LanguageCommander, PassiveBrain,
                                    PlaceboCommander, PlayGenerator, RandomCommander, RuleCommander, Translator)
from alpharush_rl.episode import COMMANDER_ACTIONS, EpisodeProtocol, run_episode
from alpharush_rl.journal import Journal, sha256_data
from alpharush_rl.menus import SUPPORTED_ACTIONS, build_menu
from alpharush_rl.ops import GateRefused
from alpharush_rl.search import BuildOrderPolicy

from test_campaign_survey import SURVEY, TempWorkspace
from test_search import POOLS, ToyEnv, genome

PLAN = genome([["b", "01", "mage"], ["b", "02", "archer"], ["u", "01"], ["b", "03", "archer"]])
PROTOCOL = EpisodeProtocol(name="toy", wait_ticks=30, max_interval_ticks=300, max_ticks=20000, max_decisions=900)
COOLDOWN = {1: 600, 2: 300}
DAMAGE = {1: 60, 2: 20}
PATH_Y = {1: 100, 2: 300}
EXITS = {"1": (330, 100), "2": (330, 300)}
HOME = (300, 200)
HP_FACTOR = 4  # tougher than ToyEnv's enemies, so spells matter and lives are lost


def _x(enemy):
    return 10 + max(0, enemy["progress"])


class CommandToy(ToyEnv):
    """ToyEnv with what the commander reads: spiders (egg layers) on path 1 and zombies on path 2 with positions,
    a hero, both spells with cooldowns (use_power offers, as in v3), the next-wave preview, and with ``v4`` the
    host's aim_power / move_hero offers."""

    def __init__(self, *args, v4=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.v4 = v4

    def reset(self):
        super().reset()
        s = self.state
        s.update({"wave_total": 4, "cooldown": {"1": 0, "2": 0},
                  "heroes": [{"id": 7, "template": "hero_gerald", "hp": 400, "hp_max": 400, "dead": False,
                              "x": HOME[0], "y": HOME[1], "rally_x": HOME[0], "rally_y": HOME[1]}]})
        self.state = self._refresh(s)
        self.trace = [{"kind": "reset", "state_sha256": sha256_data(self.state), "tick": 1, "seed": self.seed,
                       "level": self.level}]
        return self.state

    def _spawn(self, s):
        before = {e["id"] for e in s["enemies"]}
        super()._spawn(s)
        for e in s["enemies"]:
            if e["id"] not in before:
                spider = e["id"] % 3 == 0
                e.update({"template": "enemy_spider_rotten" if spider else "enemy_zombie", "path_index": 1 if spider else 2,
                          "hp": e["hp"] * HP_FACTOR})

    def _tick(self, s):
        hero = s.get("heroes", [None])[0] if s.get("heroes") else None
        if hero is not None:
            dx, dy = hero["rally_x"] - hero["x"], hero["rally_y"] - hero["y"]
            step = min(3.0, math.hypot(dx, dy))
            if step > 0:
                hero["x"] += round(dx / math.hypot(dx, dy) * step)
                hero["y"] += round(dy / math.hypot(dx, dy) * step)
            for e in s["enemies"]:
                if e["progress"] >= 0 and math.hypot(_x(e) - hero["x"], PATH_Y[e["path_index"]] - hero["y"]) <= 40:
                    e["hp"] -= 2
                    break
        super()._tick(s)

    def _refresh(self, s):
        s = super()._refresh(s)
        if "cooldown" not in s:
            return s
        for e in s["enemies"]:
            e["x"], e["y"] = _x(e), PATH_Y[e["path_index"]]
        s["powers_ui"] = [{"id": p, "mode": "cooldown" if s["cooldown"][str(p)] > s["tick"] else "ready"}
                          for p in (1, 2)]
        s["wave_spawning"] = False
        s["next_wave"] = None if s["wave"] >= s["wave_total"] else {"group_idx": s["wave"] + 1, "paths": [
            {"path_index": 1, "spawns": [{"template": "enemy_spider_rotten", "count": 2 + s["wave"]}]},
            {"path_index": 2, "spawns": [{"template": "enemy_zombie", "count": 4 + 2 * s["wave"]}]}]}
        live = sorted((e for e in s["enemies"] if e["progress"] >= 0 and e["hp"] > 0),
                      key=lambda e: (-e["progress"], e["id"]))
        catalog = [item for item in s["action_catalog"] if item["action"] not in ("use_power", "aim_power", "move_hero")]
        for p in (1, 2):
            if s["cooldown"][str(p)] > s["tick"]:
                continue
            for e in live[:3]:
                catalog.append({"action": "use_power", "power": p, "x": e["x"], "y": e["y"], "anchor_id": e["id"],
                                "cost": 0, "available": True})
            if not self.v4 or not live:
                continue
            aims = [("cluster:1", live[0]["x"], live[0]["y"], None)]
            for e in live[:2]:
                aims.append((f"{'lead' if p == 1 else 'block'}:{e['id']}", min(310, e["x"] + 12), e["y"], e["id"]))
            if p == 1:
                aims += [(f"threat:{e['id']}", e["x"], e["y"], e["id"]) for e in live
                         if e["template"] == "enemy_spider_rotten"][:2]
            for option, x, y, target in aims:
                catalog.append({"action": "aim_power", "power": p, "x": x, "y": y, "option": option, "cost": 0,
                                "available": True, **({"target_id": target} if target is not None else {})})
        hero = s["heroes"][0]
        if self.v4 and not hero["dead"]:
            points = {"home": HOME, **{f"exit:{k}": point for k, point in EXITS.items()},
                      **{f"up:{p}": (110, y) for p, y in PATH_Y.items()}}
            for option, (x, y) in points.items():
                catalog.append({"action": "move_hero", "hero_id": hero["id"], "x": x, "y": y, "option": option,
                                "cost": 0, "available": True})
        s["action_catalog"] = catalog
        return s

    def act(self, action):
        name = action["action"]
        if name not in ("aim_power", "use_power", "move_hero"):
            return super().act(action)
        before = self.state
        if not any(item["action"] == action for item in build_menu(before)):
            raise ValueError("Action is not in the native legal menu")
        s = copy.deepcopy(before)
        if name == "move_hero":
            s["heroes"][0]["rally_x"], s["heroes"][0]["rally_y"] = action["x"], action["y"]
        else:
            p = action["power"]
            s["cooldown"][str(p)] = s["tick"] + COOLDOWN[p]
            for e in s["enemies"]:
                if e["progress"] >= 0 and math.hypot(e["x"] - action["x"], e["y"] - action["y"]) <= 60:
                    e["hp"] -= DAMAGE[p]
        self.state = self._refresh(s)
        self.advance(30 if name == "move_hero" else 2, record_plan=False)
        receipt = {"accepted": True, "executed": True, "action": dict(action), "tick_before": before["tick"],
                   "tick_after": self.state["tick"]}
        self.trace.append({"kind": "action", "receipt": receipt})
        self.plan.append({"action": dict(action)})
        return receipt


def play(policy, *, v4=True, seed=1001, protocol=PROTOCOL):
    return run_episode(CommandToy(seed=seed, gold=400, v4=v4), policy, protocol, seed=seed, level=1)


def commanded(result):
    return [d for d in result["decisions"] if d["provenance"].startswith("commander:")]


def executor_view(result):
    return [(d["tick"], d["action"]) for d in result["decisions"] if not d["provenance"].startswith("commander:")]


# -- fake v4 snapshots for translator / generator tests ---------------------------------------------------------
def enemy(eid, template="enemy_zombie", path=2, progress=0.6, x=None, y=None, **extra):
    return {"id": eid, "template": template, "path_index": path, "path_progress": progress, "hp": 300, "hp_max": 300,
            "x": 200 * progress if x is None else x, "y": PATH_Y.get(path, 200) if y is None else y, **extra}


def snapshot(*, tick=1000, enemies=(), hero=None, ready=(True, True), lives=20, wave=2, preview=None, aims=None,
             moves=True):
    hero = {"id": 7, "template": "hero_gerald", "hp": 400, "hp_max": 400, "dead": False, "x": 300, "y": 200,
            "rally_x": 300, "rally_y": 200, "home_x": 300, "home_y": 200, **(hero or {})}
    s = {"tick": tick, "lives": lives, "gold": 300, "wave": wave, "wave_total": 10, "level_idx": 22,
         "enemies": [dict(e) for e in enemies], "heroes": [hero], "towers": [], "holders": [],
         "powers_ui": [{"id": p, "mode": "ready" if ok else "cooldown"} for p, ok in zip((1, 2), ready)],
         "next_wave": preview, "wave_spawning": False}
    catalog = []
    for p, ok in zip((1, 2), ready):
        if not ok:
            continue
        for e in s["enemies"][:3]:
            catalog.append({"action": "use_power", "power": p, "x": round(e["x"]), "y": round(e["y"]),
                            "anchor_id": e["id"], "cost": 0, "available": True})
        for option, x, y, target in (aims or {}).get(p, []):
            catalog.append({"action": "aim_power", "power": p, "x": x, "y": y, "option": option, "cost": 0,
                            "available": True, **({"target_id": target} if target is not None else {})})
    if moves and not hero["dead"]:
        for option, (x, y) in {"home": (300, 200), "exit:1": EXITS["1"], "exit:2": EXITS["2"], "up:1": (110, 100),
                               "up:2": (110, 300)}.items():
            catalog.append({"action": "move_hero", "hero_id": 7, "x": x, "y": y, "option": option, "cost": 0,
                            "available": True})
    s["action_catalog"] = catalog
    return s


def named_aims(enemies, power):
    out = [("cluster:1", round(enemies[0]["x"]), round(enemies[0]["y"]), None)] if enemies else []
    kinds = ("lead", "threat") if power == 1 else ("block",)
    return out + [(f"{kind}:{e['id']}", round(e["x"]), round(e["y"]), e["id"]) for e in enemies for kind in kinds]


def preview(*groups):
    return {"group_idx": 3, "paths": [{"path_index": path, "spawns": [{"template": t, "count": c}]}
                                      for t, path, c in groups]}


class ScopePlumbingTests(unittest.TestCase):
    def test_v4_is_v3_plus_the_commander_actions_and_plays_main_levels_in_v2(self):
        self.assertEqual(("v1", "v2", "v3", "v4"), engine.ACTION_SCOPES)
        self.assertEqual(["v2"] * 12 + ["v4"] * 14, [engine.level_scope("v4", level) for level in range(1, 27)])
        self.assertEqual(["v2"] * 12 + ["v3"] * 14, [engine.level_scope("v3", level) for level in range(1, 27)])
        self.assertEqual(env_module.SCOPES["v3"] + ("aim_power", "move_hero"), env_module.SCOPES["v4"])
        self.assertTrue({"aim_power", "move_hero"} <= SUPPORTED_ACTIONS)
        self.assertEqual(frozenset(("aim_power", "move_hero")), COMMANDER_ACTIONS)
        self.assertEqual("v4", engine.Worker(action_scope="v4").action_scope)
        # An aimed spell costs use_power's 2 ticks; a hero order a rally's 30.
        self.assertNotIn("aim_power", env_module.ACTION_TICKS)
        self.assertEqual(30, env_module.ACTION_TICKS["move_hero"])

    def test_menu_offers_well_formed_commander_options_only(self):
        s = snapshot(enemies=[enemy(11), enemy(12, "enemy_spider_rotten", 1)])
        s["action_catalog"] += [
            {"action": "aim_power", "power": 1, "x": 5, "y": 6, "option": "lead:11", "target_id": 11, "cost": 0,
             "available": True},
            {"action": "aim_power", "power": 2, "x": 7, "y": 8, "option": "cluster:2", "cost": 0, "available": True},
            {"action": "aim_power", "power": 1, "x": 5, "y": 6, "option": "lead:12", "target_id": 11, "cost": 0,
             "available": True},  # the named enemy differs from the target
            {"action": "aim_power", "power": 1, "x": 5, "y": 6, "option": "threat:99", "target_id": 99, "cost": 0,
             "available": True},  # not a live enemy
            {"action": "aim_power", "power": 3, "x": 5, "y": 6, "option": "cluster:1", "cost": 0, "available": True},
            {"action": "aim_power", "power": 1, "x": 5.5, "y": 6, "option": "cluster:1", "cost": 0, "available": True},
            {"action": "aim_power", "power": 1, "x": 5, "y": 6, "option": "anywhere", "cost": 0, "available": True},
            {"action": "move_hero", "hero_id": 7, "x": 1, "y": 2, "option": "block:12", "cost": 0, "available": True},
            {"action": "move_hero", "hero_id": 7, "x": 1, "y": 2, "option": "block:99", "cost": 0, "available": True},
            {"action": "move_hero", "hero_id": 8, "x": 1, "y": 2, "option": "home", "cost": 0, "available": True},
            {"action": "move_hero", "hero_id": 7, "x": 1, "y": 2, "option": "upstream", "cost": 0, "available": True}]
        menu = build_menu(s)
        aims = [item["action"] for item in menu if item["action"]["action"] == "aim_power"]
        self.assertIn({"action": "aim_power", "power": 1, "x": 5, "y": 6, "option": "lead:11", "target_id": 11}, aims)
        self.assertIn({"action": "aim_power", "power": 2, "x": 7, "y": 8, "option": "cluster:2"}, aims)
        self.assertEqual(2, len(aims))
        moves = sorted(item["action"]["option"] for item in menu if item["action"]["action"] == "move_hero")
        self.assertEqual(["block:12", "exit:1", "exit:2", "home", "up:1", "up:2"], moves)
        text = {item["action"].get("option"): item["text"] for item in menu if "option" in item["action"]}
        self.assertEqual("Cast rain of fire at (5,6) where enemy 11 will be in 1.2 s", text["lead:11"])
        self.assertEqual("Move hero hero_gerald upstream on path 1 (110,100)", text["up:1"])
        dead = snapshot(hero={"dead": True})
        dead["action_catalog"].append({"action": "move_hero", "hero_id": 7, "x": 1, "y": 2, "option": "home",
                                       "cost": 0, "available": True})
        self.assertFalse([i for i in build_menu(dead) if i["action"]["action"] == "move_hero"])

    def test_each_spell_keeps_its_kinds_and_a_disabled_hero_takes_no_order(self):
        s = snapshot(enemies=[enemy(11), enemy(12, "enemy_spider_rotten", 1)], moves=False)
        s["action_catalog"] = [
            {"action": "aim_power", "power": p, "x": 5 + i, "y": 6, "option": option, "cost": 0, "available": True,
             **({} if option.startswith("cluster") else {"target_id": 11})}
            for p in (1, 2) for i, option in enumerate(("lead:11", "threat:11", "block:11", "cluster:1"))]
        offered = {(item["action"]["power"], item["action"]["option"]) for item in build_menu(s)
                   if item["action"]["action"] == "aim_power"}
        # Rain of Fire: lead, threat, cluster; reinforcements: block, cluster.
        self.assertEqual({(1, "lead:11"), (1, "threat:11"), (1, "cluster:1"), (2, "block:11"), (2, "cluster:1")},
                         offered)
        for hero, orders in (({"stunned": True}, 0), ({"can_click": False}, 0),
                             ({"stunned": True, "can_click": True}, 0), ({"stunned": False, "can_click": True}, 5),
                             ({}, 5)):  # {}: nothing exported, as in v3
            with self.subTest(hero=hero):
                self.assertEqual(orders, len([item for item in build_menu(snapshot(hero=hero))
                                              if item["action"]["action"] == "move_hero"]))


class FakeHost:
    """Stands in for engine.Worker inside the real NativeEnv: replies ok, then steps to a scripted state."""

    def __init__(self, **kwargs):
        self.kwargs, self.calls, self.after = kwargs, [], None

    def rpc(self, action, **kwargs):
        self.calls.append((action, kwargs))
        if action == "step":
            state = copy.deepcopy(self.after)
            state["tick"] = self.before_tick + kwargs["ticks"]
            return {"state": state, "tick_before": self.before_tick, "tick_after": state["tick"]}
        return {"type": "ok"}

    def close(self):
        pass


class ReceiptTests(unittest.TestCase):
    def make_env(self, scope):
        with mock.patch.object(env_module, "Worker", FakeHost):
            return env_module.NativeEnv(action_scope=scope)

    def test_move_hero_receipt_needs_the_rally_point_within_a_pixel(self):
        native = self.make_env("v4")
        before = snapshot()
        native.state, native.worker.before_tick = env_module.observation(before), before["tick"]
        action = {"action": "move_hero", "hero_id": 7, "x": 110, "y": 100, "option": "up:1"}
        after = copy.deepcopy(before)
        after["heroes"][0].update({"rally_x": 110.6, "rally_y": 99.2, "rally_new": True})
        native.worker.after = after
        receipt = native.act(action)
        self.assertTrue(receipt["executed"])
        self.assertEqual(before["tick"] + env_module.HERO_TICKS, receipt["tick_after"])
        self.assertEqual(("move_hero", {"hero_id": 7, "x": 110, "y": 100, "option": "up:1"}), native.worker.calls[0])
        self.assertTrue(receipt["rally_new_after"])
        native.state = env_module.observation(before)
        after["heroes"][0].update({"rally_x": 112.0, "rally_y": 100})
        with self.assertRaisesRegex(RuntimeError, "receipt failed"):
            native.act(action)

    def test_aim_power_receipt_is_use_powers_and_costs_two_ticks(self):
        native = self.make_env("v4")
        before = snapshot(enemies=[enemy(11)], aims={1: [("cluster:1", 40, 300, None)]})
        native.state, native.worker.before_tick = env_module.observation(before), before["tick"]
        after = copy.deepcopy(before)
        after["powers_ui"][0]["mode"] = "cooldown"
        native.worker.after = after
        receipt = native.act({"action": "aim_power", "power": 1, "x": 40, "y": 300, "option": "cluster:1"})
        self.assertEqual((True, "cooldown", before["tick"] + 2),
                         (receipt["executed"], receipt["power_mode_after"], receipt["tick_after"]))

    def test_v3_env_refuses_commander_actions(self):
        native = self.make_env("v3")
        before = snapshot(enemies=[enemy(11)])
        native.state = env_module.observation(before)
        with self.assertRaisesRegex(ValueError, "outside the v3 action scope"):
            native.act({"action": "move_hero", "hero_id": 7, "x": 110, "y": 100, "option": "up:1"})


class EpisodeHookTests(unittest.TestCase):
    def test_commander_options_never_make_decision_events(self):
        class Seeing(BuildOrderPolicy):
            """The plan executor, remembering which commander options its menus showed."""
            def __init__(self, genome):
                super().__init__(genome)
                self.seen = set()

            def choose(self, state, menu, context):
                self.seen |= {item["action"]["action"] for item in menu} & COMMANDER_ACTIONS
                return super().choose(state, menu, context)
        seeing = Seeing(PLAN)
        v3, v4 = play(BuildOrderPolicy(PLAN), v4=False), play(seeing, v4=True)
        self.assertEqual({"aim_power", "move_hero"}, seeing.seen)  # offered all along, never a decision event
        self.assertEqual("terminal", v3["status"])
        self.assertEqual([(d["tick"], d["action"]) for d in v3["decisions"]],
                         [(d["tick"], d["action"]) for d in v4["decisions"]])
        self.assertEqual(v3["plan"], v4["plan"])
        self.assertEqual(v3["outcome"], {**v4["outcome"], "state_sha256": v3["outcome"]["state_sha256"]})
        self.assertTrue(any(d["action"]["action"] == "use_power" for d in v4["decisions"]))

    class Asking:
        """The plan executor, plus decisions of its own (waits, as long as a forced wait) at the given ticks;
        ``hold`` ticks are held instead at ticks in ``holds``."""
        def __init__(self, ticks, holds=(), hold=1):
            self.inner, self.name, self.ticks, self.holds, self.hold = BuildOrderPolicy(PLAN), "asking", set(ticks), \
                set(holds), hold
            self.steps = []

        def wants_decision(self, state):
            self.steps.append(state["tick"])
            if state["tick"] in self.holds:
                return {"hold": self.hold}
            return state["tick"] in self.ticks

        def choose(self, state, menu, context):
            if state["tick"] in self.ticks:
                return {"label": "A", "provenance": "test:asked", "meta": {}}
            return self.inner.choose(state, menu, context)

    def test_wants_decision_adds_decisions_without_moving_the_executors(self):
        plain, silent = play(BuildOrderPolicy(PLAN), v4=False), self.Asking([])
        self.assertEqual(plain["plan"], play(silent, v4=False)["plan"])  # never asking changes nothing
        executor_ticks = {d["tick"] for d in plain["decisions"]}
        # Ask at steps where the executor had nothing due: the game runs exactly as alone (an own wait lasts as
        # long as the forced wait it replaces), and the executor's re-ask clock is untouched by the extra asks.
        quiet = [t for t in silent.steps if t not in executor_ticks][::3]
        self.assertGreaterEqual(len(quiet), 4)
        asking = play(self.Asking(quiet), v4=False)
        mine = [d for d in asking["decisions"] if d["provenance"] == "test:asked"]
        self.assertTrue(mine)
        executor = [(d["tick"], d["action"]) for d in asking["decisions"] if d["provenance"] != "test:asked"]
        self.assertEqual([(d["tick"], d["action"]) for d in plain["decisions"]], executor)
        self.assertEqual(plain["plan"], asking["plan"])
        # Asked while the executor has a decision due: the policy's own decision comes first and the executor's
        # follows at the next step, unchanged.
        first = play(self.Asking([1]), v4=False)
        self.assertEqual(("test:asked", 1), (first["decisions"][0]["provenance"], first["decisions"][0]["tick"]))
        self.assertEqual((1 + PROTOCOL.wait_ticks, plain["decisions"][0]["action"]),
                         (first["decisions"][1]["tick"], first["decisions"][1]["action"]))
        # A hold delays the step, a due decision included, by that many ticks.
        held = play(self.Asking([], holds=[1]), v4=False)
        self.assertEqual({"action": {"action": "wait", "ticks": 1}}, held["plan"][0])
        self.assertEqual((2, plain["decisions"][0]["action"]), (held["decisions"][0]["tick"],
                                                                held["decisions"][0]["action"]))
        self.assertEqual("terminal", held["status"])

    def test_a_failing_wants_decision_voids_the_episode(self):
        class Broken(BuildOrderPolicy):
            def wants_decision(self, state):
                raise KeyError("boom")
        result = play(Broken(PLAN), v4=False)
        self.assertEqual("void", result["status"])
        self.assertIn("policy_error: KeyError", result["void_reason"])

    def test_wants_decision_returns_a_bool_or_an_explicit_hold_only(self):
        class Answering(BuildOrderPolicy):
            def __init__(self, genome, answer):
                super().__init__(genome)
                self.answer = answer

            def wants_decision(self, state):
                return self.answer
        bad = (None, 1, 0, 1.0, "hold", [1], {"hold": 0}, {"hold": -1}, {"hold": 1.0}, {"hold": True},
               {"hold": PROTOCOL.wait_ticks}, {"hold": 1, "why": "late"}, {"wait": 1}, {})
        for answer in bad:
            with self.subTest(answer=answer):
                result = play(Answering(PLAN, answer), v4=False)
                self.assertEqual("void", result["status"])
                self.assertTrue(result["void_reason"].startswith("invalid_wants_decision: "), result["void_reason"])
                self.assertIn(f"0 < n < {PROTOCOL.wait_ticks}", result["void_reason"])
                self.assertEqual(([], []), (result["decisions"], result["plan"]))  # nothing was played
        plain = play(BuildOrderPolicy(PLAN), v4=False)
        self.assertEqual(plain["plan"], play(Answering(PLAN, False), v4=False)["plan"])
        longest = play(self.Asking([], holds=[1], hold=PROTOCOL.wait_ticks - 1), v4=False)
        self.assertEqual(({"action": {"action": "wait", "ticks": PROTOCOL.wait_ticks - 1}}, "terminal"),
                         (longest["plan"][0], longest["status"]))

    def test_at_most_one_hold_per_due_event(self):
        plain = play(BuildOrderPolicy(PLAN), v4=False)
        # Held at tick 1 with the first decision due; asked to hold it again at tick 2: not granted, the due
        # decision is taken.
        twice = play(self.Asking([], holds=[1, 2]), v4=False)
        self.assertEqual([{"action": {"action": "wait", "ticks": 1}}, {"action": plain["decisions"][0]["action"]}],
                         twice["plan"][:2])
        self.assertEqual((2, plain["decisions"][0]["provenance"]), (twice["decisions"][0]["tick"],
                                                                    twice["decisions"][0]["provenance"]))
        # The policy's own decision in between does not serve the due event, so it still counts as held.
        between = play(self.Asking([2], holds=[1, 2 + PROTOCOL.wait_ticks]), v4=False)
        self.assertEqual([(2, "test:asked"), (2 + PROTOCOL.wait_ticks, plain["decisions"][0]["provenance"])],
                         [(d["tick"], d["provenance"]) for d in between["decisions"][:2]])
        self.assertEqual(plain["decisions"][0]["action"], between["decisions"][1]["action"])
        # Once the due decision is taken, the next due event may be held once again.
        second = twice["decisions"][1]["tick"]
        again = play(self.Asking([], holds=[1, 2, second, second + 1]), v4=False)
        self.assertEqual(second + 1, again["decisions"][1]["tick"])
        self.assertEqual(2, again["plan"].count({"action": {"action": "wait", "ticks": 1}}))


class PassiveTests(unittest.TestCase):
    def test_passive_commander_is_the_executor_decision_for_decision(self):
        alone = play(BuildOrderPolicy(PLAN))
        passive = play(CommanderPolicy(BuildOrderPolicy(PLAN), PassiveBrain(), seed=1001))
        self.assertEqual("terminal", alone["status"])
        for key in ("plan", "trace_sha256", "outcome", "n_forced_waits"):
            self.assertEqual(alone[key], passive[key], key)
        self.assertEqual([(d["tick"], d["label"], d["action"], d["provenance"]) for d in alone["decisions"]],
                         [(d["tick"], d["label"], d["action"], d["provenance"]) for d in passive["decisions"]])
        self.assertTrue(any(d["action"]["action"] == "use_power" for d in passive["decisions"]))
        # It still records its consults (briefing, plays, shadow choices) on the executor's decisions.
        consults = [c for d in passive["decisions"] for c in d["meta"].get("commander_consults", [])]
        self.assertTrue(consults)
        self.assertEqual({"rule", "random"}, set(consults[0]["shadow"]))

    def test_passive_matches_the_executor_in_v3_too(self):
        self.assertEqual(play(BuildOrderPolicy(PLAN), v4=False)["plan"],
                         play(CommanderPolicy(BuildOrderPolicy(PLAN), PassiveBrain()), v4=False)["plan"])


class PlayTests(unittest.TestCase):
    def random_snapshot(self, rng):
        templates = ["enemy_zombie", "enemy_spider_rotten", "enemy_swamp_thing", "eb_kingpin", "enemy_gargoyle",
                     "enemy_demon_cerberus"]
        enemies = [enemy(10 + i, rng.choice(templates), rng.choice((1, 2)), round(rng.random(), 3))
                   for i in range(rng.randrange(0, 7))]
        ready = (rng.random() < 0.7, rng.random() < 0.7)
        live = sorted(enemies, key=lambda e: -e["path_progress"])
        aims = {p: named_aims(live[:3], p) for p in (1, 2)} if rng.random() < 0.8 else None
        groups = [(rng.choice(templates), rng.choice((1, 2)), rng.randrange(1, 9)) for _ in range(rng.randrange(0, 4))]
        hero = {"hp": rng.choice((80, 400)), "dead": rng.random() < 0.2, "rally_x": rng.choice((300, 110)),
                "rally_y": rng.choice((200, 100))}
        return snapshot(enemies=enemies, ready=ready, aims=aims, hero=hero, preview=preview(*groups) if groups else None,
                        tick=rng.randrange(1000, 5000))

    def test_at_most_six_plays_always_translated_into_offered_options(self):
        rng = random.Random(11)
        generator = PlayGenerator(cast=50)
        geometry = {"exits": dict(EXITS), "home_exit": "1", "path_exit": {"1": "1", "2": "2"}}
        commands = 0
        for _ in range(300):
            s = self.random_snapshot(rng)
            menu = build_menu(s)
            briefing = Briefing.from_state(s)
            plays = generator.plays(briefing, geometry)
            self.assertTrue(1 <= len(plays) <= commander.MAX_PLAYS)
            self.assertEqual("P0", plays[0]["id"])
            self.assertEqual(len({p["id"] for p in plays}), len(plays))
            json.dumps([briefing.data, plays], allow_nan=False)
            for p in plays:
                translator = Translator(cast=50)
                translator.adopt(p, s)
                translator.observe(s)
                out = translator.command(s, menu, commit=True)
                if out is not None:
                    commands += 1
                    self.assertTrue(any(item is out["item"] for item in menu))
                    self.assertIn(out["item"]["action"]["action"], ("aim_power", "use_power", "move_hero"))
        self.assertGreater(commands, 200)

    def test_threats_and_rule_choices(self):
        rule = RuleCommander()
        geometry = {"exits": dict(EXITS), "home_exit": "1", "path_exit": {"1": "1", "2": "2"}}

        def choose(*groups, field=()):
            s = snapshot(preview=preview(*groups) if groups else None, enemies=field)
            plays = PlayGenerator().plays(Briefing.from_state(s), geometry)
            index, record = rule.choose(Briefing.from_state(s), plays, {})
            return plays[index]["id"], record["rule"], plays
        play_id, why, plays = choose(("enemy_zombie", 2, 12), ("enemy_spider_rotten", 1, 3))
        self.assertEqual(["P0", "P1", "P2", "P3", "P4"], [p["id"] for p in plays])
        self.assertEqual("enemy_spider_rotten", plays[1]["threat"]["template"])  # 3 egg layers outrank 12 zombies
        self.assertEqual(("P1", "block_egg_layers"), (play_id, why))
        self.assertEqual(("P4", "focus_spells"), choose(("eb_kingpin", 1, 1), ("enemy_spider_rotten", 1, 2))[:2])
        self.assertEqual(("P1", "save_fire_for_threat"), choose(("enemy_swamp_thing", 2, 2))[:2])
        self.assertEqual(("P3", "guard_threatened_exit"), choose(("enemy_zombie", 2, 9))[:2])
        self.assertEqual(("P0", "steady"), choose(("enemy_zombie", 1, 9))[:2])
        # No flag showing (an alarm consult): the threats are what is on the field.
        self.assertEqual("P4", choose(field=[enemy(5, "eb_kingpin", 1, 0.3)])[0])

    def test_briefing_shows_only_what_the_screen_shows(self):
        s = snapshot(enemies=[enemy(11, progress=0.3), enemy(12, "eb_myconid", 1, 0.6, hp=1000, hp_max=4500)],
                     preview=preview(("enemy_spider_rotten", 1, 3)))
        s["enemies"][0].update({"armor": 0.5, "gold": 9, "lives_cost": 1})
        data = Briefing.from_state(s).data
        flat = json.dumps(data)
        for hidden in ("armor", "hp_max", "gold\": 9", "lives_cost", "speed"):
            self.assertNotIn(hidden, flat)
        self.assertEqual([{"template": "enemy_zombie", "path": 2, "band": 1, "count": 1}], data["field"])
        self.assertEqual(1, data["bosses"][0]["health"])  # a quarter bar, not 1000 HP
        self.assertEqual({"group": 3, "paths": [{"path": 1, "enemies": [["enemy_spider_rotten", 3]]}]},
                         data["next_wave"])
        text = Briefing.from_state(s).text()
        self.assertIn("Elite stage 22 (Fungal Forest)", text)
        self.assertIn("3 spider rotten (lays eggs)", text)
        self.assertIn("Rain of Fire ready", text)

    def test_a_dormant_boss_shows_no_health_bar(self):
        sleeping = enemy(40, "enemy_demon_cerberus", 1, 0.9, hp=1500, hp_max=6000, dormant=True)
        awake = enemy(41, "eb_myconid", 2, 0.6, hp=1000, hp_max=4500)
        briefing = Briefing.from_state(snapshot(enemies=[sleeping, awake]))
        bosses = {b["template"]: b for b in briefing.data["bosses"]}
        self.assertNotIn("health", bosses["enemy_demon_cerberus"])
        self.assertTrue(bosses["enemy_demon_cerberus"]["dormant"])
        self.assertEqual(1, bosses["eb_myconid"]["health"])
        text = briefing.text()
        self.assertIn("Boss demon cerberus: waiting.\n", text)
        self.assertIn("Boss myconid: on path 2 at 50-75%, health about 25%.", text)
        self.assertEqual(1, text.count("health about"))

    def test_threats_sort_without_comparing_none_to_a_name(self):
        # A boss the bridge exports without a template ties on score with a named group on the same path.
        nameless = enemy(40, None, 1, 0.5, boss=True)
        s = snapshot(enemies=[enemy(11, "enemy_zombie", 1, 0.4), nameless])
        threats = PlayGenerator.threats(Briefing.from_state(s))
        self.assertEqual(threats[0]["score"], threats[1]["score"])
        self.assertEqual([None, "enemy_zombie"], [t["template"] for t in threats])
        self.assertEqual(threats, PlayGenerator.threats(Briefing.from_state({**s, "enemies": s["enemies"][::-1]})))
        self.assertTrue(PlayGenerator().plays(Briefing.from_state(s))[1]["text"].startswith("Counter 1 ? on path 1:"))


class TranslatorTests(unittest.TestCase):
    def setUp(self):
        self.t = Translator(cast=50)

    def run_command(self, s, commit=True):
        self.t.observe(s)
        return self.t.command(s, build_menu(s), commit=commit)

    def test_hero_orders_are_paced_three_seconds_apart(self):
        self.t.adopt({"id": "P1", "hero": {"to": "path", "template": "enemy_spider_rotten", "path": 1, "exit": "1"},
                      "fire": commander.STEADY, "reinf": commander.STEADY}, snapshot())
        first = self.run_command(snapshot(tick=1000, ready=(False, False)))
        self.assertEqual(("hero", "up:1"), (first["kind"], first["item"]["action"]["option"]))
        self.t.adopt({"id": "P3", "hero": {"to": "exit", "exit": "2"}, "fire": commander.STEADY,
                      "reinf": commander.STEADY}, snapshot())
        hero = {"rally_x": 110, "rally_y": 100}
        self.assertIsNone(self.run_command(snapshot(tick=1179, ready=(False, False), hero=hero)))
        second = self.run_command(snapshot(tick=1180, ready=(False, False), hero=hero))
        self.assertEqual("exit:2", second["item"]["action"]["option"])
        # Already there: no order at all.
        self.assertIsNone(self.run_command(snapshot(tick=2000, ready=(False, False),
                                                    hero={"rally_x": 330, "rally_y": 300})))

    def test_retreat_reflex_and_no_new_orders_until_healed(self):
        self.t.adopt({"id": "P1", "hero": {"to": "path", "template": "enemy_zombie", "path": 2, "exit": None},
                      "fire": commander.STEADY, "reinf": commander.STEADY}, snapshot())
        away = {"rally_x": 110, "rally_y": 300, "hp": 100}
        out = self.run_command(snapshot(tick=1000, ready=(False, False), hero=away))
        self.assertEqual(("retreat", "home"), (out["kind"], out["item"]["action"]["option"]))
        healing = {"rally_x": 300, "rally_y": 200, "hp": 200}  # 50%: home, but not healed enough to go out again
        self.assertIsNone(self.run_command(snapshot(tick=1400, ready=(False, False), hero=healing)))
        out = self.run_command(snapshot(tick=1400, ready=(False, False), hero={"hp": 300}))
        self.assertEqual("up:2", out["item"]["action"]["option"])
        # A second retreat waits for the 10 s reflex gap even though hero orders are paced at 3 s.
        self.assertIsNone(self.run_command(snapshot(tick=1590, ready=(False, False), hero=away)))
        self.assertEqual("retreat", self.run_command(snapshot(tick=1600, ready=(False, False), hero=away))["kind"])

    def test_listed_boss_auras(self):
        kingpin = enemy(30, "eb_kingpin", 1, 0.5, x=120, y=110)
        self.t.adopt({"id": "P1", "hero": {"to": "path", "template": "eb_kingpin", "path": 1, "exit": None},
                      "fire": commander.STEADY, "reinf": commander.STEADY}, snapshot())
        # The upstream point (110, 100) lies inside Kingpin's 65 px aura: no order there.
        self.assertIsNone(self.run_command(snapshot(enemies=[kingpin], ready=(False, False))))
        inside = snapshot(tick=2000, enemies=[kingpin], ready=(False, False), hero={"rally_x": 150, "rally_y": 100})
        self.assertEqual("aura", self.run_command(inside)["kind"])
        asleep = enemy(31, "enemy_demon_cerberus", 1, 0.5, x=120, y=110, dormant=True)
        self.assertFalse(Translator._inside((150, 100), Translator._auras(snapshot(enemies=[asleep]))))

    def test_the_steady_play_leaves_the_spells_to_the_executor(self):
        zombies = [enemy(11, progress=0.7, x=150), enemy(12, progress=0.4, x=90)]
        s = snapshot(enemies=zombies, aims={1: named_aims(zombies, 1), 2: named_aims(zombies, 2)}, moves=False)
        self.assertIsNone(self.run_command(s))  # P0 owns no power: the executor casts as without a commander

    def test_one_cast_per_readiness_and_steady_targets(self):
        zombies = [enemy(11, progress=0.7, x=150), enemy(12, progress=0.4, x=90)]
        s = snapshot(enemies=zombies, aims={1: named_aims(zombies, 1), 2: named_aims(zombies, 2)}, moves=False)
        # A play that owns both powers (blocking a template that is not on the field) casts on the steady targets.
        self.t.adopt({"id": "P1", "hero": None, "fire": {"mode": "block", "template": "enemy_none", "path": 9},
                      "reinf": {"mode": "block", "template": "enemy_none", "path": 9}}, s)
        fire = self.run_command(s)
        self.assertEqual(("fire", "lead:11"), (fire["kind"], fire["item"]["action"]["option"]))
        reinf = self.run_command({**s, "tick": 1002})
        self.assertEqual(("reinf", "block:11"), (reinf["kind"], reinf["item"]["action"]["option"]))
        self.assertIsNone(self.run_command({**s, "tick": 1004}))  # both still shown ready: never twice
        self.run_command(snapshot(tick=1010, enemies=zombies, ready=(False, False), moves=False))
        self.assertEqual("fire", self.run_command({**s, "tick": 1700})["kind"])
        # Below the plan's cast threshold nothing is cast.
        early = [enemy(11, progress=0.2)]
        below = Translator(cast=50)
        below.play = {"id": "P1", "hero": None, "fire": {"mode": "block", "template": "enemy_none", "path": 9},
                      "reinf": commander.STEADY}
        self.assertIsNone(below.command(snapshot(enemies=early, aims={1: named_aims(early, 1)},
                                                               moves=False), build_menu(snapshot(
            enemies=early, aims={1: named_aims(early, 1)}, moves=False))))

    def test_saved_fire_waits_for_its_target_then_expires(self):
        zombies = [enemy(11, progress=0.8, x=160)]
        spiders = [enemy(21, "enemy_spider_rotten", 1, 0.1, x=20)]
        p1 = {"id": "P1", "hero": None, "fire": {"mode": "reserve", "template": "enemy_spider_rotten", "path": 1},
              "reinf": {"mode": "block", "template": "enemy_spider_rotten", "path": 1}}
        flag = preview(("enemy_spider_rotten", 1, 2))
        self.t.adopt(p1, snapshot(wave=2, preview=flag))
        s = snapshot(wave=2, preview=flag, enemies=zombies, aims={1: named_aims(zombies, 1), 2: named_aims(zombies, 2)},
                     moves=False)
        out = self.run_command(s)
        self.assertEqual(("reinf", "block:11"), (out["kind"], out["item"]["action"]["option"]))  # no spider: steady
        both = zombies + spiders
        s = snapshot(tick=1100, wave=3, enemies=both, ready=(True, False), aims={1: named_aims(both, 1)}, moves=False)
        out = self.run_command(s)
        self.assertEqual(("fire", "threat:21", 21), (out["kind"], out["item"]["action"]["option"], out["target"]))
        # Expiry: two lives lost within 10 s release the saved spell for the leading enemy.
        t = Translator(cast=50)
        t.adopt(p1, snapshot(wave=2, preview=flag))
        held = snapshot(tick=1000, wave=2, preview=flag, enemies=zombies, ready=(True, False),
                        aims={1: named_aims(zombies, 1)}, moves=False)
        t.observe(held)
        self.assertIsNone(t.command(held, build_menu(held)))
        lost = {**held, "tick": 1060, "lives": 18}
        t.observe(lost)
        self.assertEqual("lead:11", t.command(lost, build_menu(lost))["item"]["action"]["option"])
        # Also expired once its wave has gone by without it (not spawning, not on the next flag).
        t = Translator(cast=50)
        t.adopt(p1, snapshot(wave=2, preview=flag))
        later = snapshot(tick=3000, wave=3, preview=preview(("enemy_zombie", 2, 4)), enemies=zombies,
                         ready=(True, False), aims={1: named_aims(zombies, 1)}, moves=False)
        t.observe(later)
        self.assertEqual("lead:11", t.command(later, build_menu(later))["item"]["action"]["option"])


class CommanderRunTests(unittest.TestCase):
    def test_rule_commander_plays_paced_legal_commands(self):
        result = play(CommanderPolicy(BuildOrderPolicy(PLAN), RuleCommander(), seed=1001))
        self.assertEqual("terminal", result["status"])
        mine = commanded(result)
        kinds = {d["provenance"].rsplit(":", 1)[1] for d in mine}
        self.assertTrue({"hero", "fire", "reinf"} <= kinds, kinds)
        self.assertTrue(all(d["action"]["action"] in ("aim_power", "use_power", "move_hero", "wait") for d in mine))
        self.assertEqual([], commander.pace_violations(result["decisions"]))
        ticks = [d["tick"] for d in mine]
        self.assertTrue(all(b - a >= commander.TRIGGER_GAP for a, b in zip(ticks, ticks[1:])))
        consults = [c for d in result["decisions"] for c in d["meta"].get("commander_consults", [])]
        self.assertEqual(list(range(len(consults))), [c["consult"] for c in consults])
        self.assertEqual("wave", consults[0]["reason"])
        self.assertEqual(0, consults[0]["wave"])  # before wave 1
        self.assertTrue(all(len(c["plays"]) <= commander.MAX_PLAYS and c["play_id"] == c["plays"][c["choice"]]["id"]
                            for c in consults))
        self.assertTrue(all(len(c["prompt_sha256"]) == 64 for c in consults))
        # Executor decisions never see the commander's options nor, with an active brain, use_power.
        self.assertFalse([d for d in result["decisions"] if not d["provenance"].startswith("commander:")
                          and d["action"]["action"] in ("use_power", "aim_power", "move_hero")])
        replay = run_episode(CommandToy(seed=1001, gold=400), CommanderPolicy(BuildOrderPolicy(PLAN), RuleCommander(),
                                                                             seed=1001), PROTOCOL, seed=1001, level=1)
        self.assertEqual(result["trace_sha256"], replay["trace_sha256"])

    def test_consults_on_new_flags_and_at_most_one_alarm_per_wave(self):
        policy = CommanderPolicy(BuildOrderPolicy(PLAN), RuleCommander(), seed=1001)
        steps = [snapshot(tick=100, wave=2, preview=preview(("enemy_zombie", 2, 4))),  # flag of group 3
                 snapshot(tick=160, wave=2, preview=preview(("enemy_zombie", 2, 4))),  # same flag: no consult
                 snapshot(tick=200, wave=3, lives=17),                                 # 3 lives lost
                 snapshot(tick=300, wave=3, lives=14, hero={"dead": True}),            # same wave: no second alarm
                 snapshot(tick=400, wave=3, lives=14, hero={"dead": True},
                          enemies=[enemy(40, "eb_myconid", 1, 0.1)]),
                 snapshot(tick=500, wave=4, lives=14, hero={"dead": True})]            # 3 lost since the alarm
        for s in steps:
            policy.wants_decision(s)
        self.assertEqual([("wave", 100), ("alarm:lives_lost", 200), ("alarm:lives_lost", 500)],
                         [(c["reason"], c["tick"]) for c in policy.consults])
        record = policy.consults[1]
        for key in ("briefing", "plays", "choice", "play_id", "p", "latency_seconds", "prompt_sha256", "shadow",
                    "record", "group", "wave"):
            self.assertIn(key, record)
        # No flag at the alarm: its plays counter what is on the field (here nothing: steady, or guard the other exit
        # of this two-exit map, which the rule leaves alone without a threat heading there).
        self.assertEqual(["P0", "P3"], [p["id"] for p in record["plays"]])
        self.assertEqual("P0", record["play_id"])
        boss = CommanderPolicy(BuildOrderPolicy(PLAN), RuleCommander(), seed=1001)
        boss.wants_decision(steps[0])
        boss.wants_decision(steps[4])
        self.assertEqual(["wave", "alarm:boss_entry"], [c["reason"] for c in boss.consults])
        self.assertEqual("P4", boss.consults[1]["play_id"])  # the Myconid's aura: spells only, hero at home

    def test_operator_network_can_be_the_executor(self):
        from alpharush_rl.operator_net import OperatorPolicy, OptionScorer
        net = OptionScorer(hidden=(8,), seed=0)
        result = play(CommanderPolicy(OperatorPolicy(net, PLAN), RuleCommander(), seed=1001))
        self.assertIn(result["status"], ("terminal", "stalled"))
        self.assertTrue(commanded(result))
        self.assertFalse([d for d in result["decisions"] if d["provenance"].startswith("model:")
                          and d["action"]["action"] in ("use_power", "aim_power", "move_hero")])

    def test_the_plans_spell_genes_reach_the_commander_over_any_executor(self):
        from alpharush_rl.operator_net import OperatorPolicy, OptionScorer
        from alpharush_rl.search import genome_id
        plan = genome(PLAN["steps"], cast=30, boss=1)
        net = OptionScorer(hidden=(8,), seed=0)
        for inner, explicit in ((OperatorPolicy(net, plan), None), (BuildOrderPolicy(plan), None),
                                (OperatorPolicy(net), plan), (OperatorPolicy(net, plan), plan)):
            with self.subTest(inner=inner.name, explicit=explicit is not None):
                policy = CommanderPolicy(inner, RuleCommander(), seed=1001, genome=explicit)
                self.assertEqual((30, 30, True, genome_id(plan)),
                                 (policy.translator.cast, policy.generator.cast, policy.translator.boss_focus,
                                  policy.genome_id))
        # (genome_id also seeds the random arm's consults.) Without a plan the defaults stay; a plan that is not
        # the executor's is refused.
        bare = CommanderPolicy(OperatorPolicy(net), RuleCommander())
        self.assertEqual((50, False, "none"), (bare.translator.cast, bare.translator.boss_focus, bare.genome_id))
        with self.assertRaisesRegex(ValueError, "plan its executor follows"):
            CommanderPolicy(OperatorPolicy(net, PLAN), RuleCommander(), genome=plan)

    def test_pace_checker_flags_violations(self):
        clean = {"click_offered": False}
        decisions = [{"index": 0, "tick": 100, "provenance": "commander:rule:P1:hero",
                      "action": {"action": "move_hero", "option": "up:1"}, "meta": clean},
                     {"index": 1, "tick": 200, "provenance": "commander:rule:P1:retreat",
                      "action": {"action": "move_hero", "option": "home"}, "meta": clean},
                     {"index": 2, "tick": 300, "provenance": "scripted:x", "action": {"action": "move_hero"},
                      "meta": {"commander_consults": [{"consult": 0, "reason": "alarm:lives_lost", "wave": 2},
                                                      {"consult": 1, "reason": "alarm:hero_death", "wave": 2}]}}]
        problems = commander.pace_violations(decisions)
        self.assertEqual(4, len(problems), problems)
        # The commander's own decisions: at least TRIGGER_GAP ticks apart, never while a tap is offered (an
        # unrecorded menu counts as one).
        fire = {"action": "aim_power", "power": 1, "x": 5, "y": 6, "option": "cluster:1"}
        decisions = [{"index": 0, "tick": 1000, "provenance": "commander:rule:P0:fire", "action": fire, "meta": clean},
                     {"index": 1, "tick": 1029, "provenance": "commander:rule:P0:wait",
                      "action": {"action": "wait", "ticks": 30}, "meta": clean},
                     {"index": 2, "tick": 1100, "provenance": "commander:rule:P0:reinf", "action": fire,
                      "meta": {"click_offered": True}},
                     {"index": 3, "tick": 1200, "provenance": "commander:rule:P0:reinf", "action": fire, "meta": {}},
                     {"index": 4, "tick": 1201, "provenance": "scripted:x", "action": fire, "meta": {}}]
        self.assertEqual(["decision 2: a commander command while a click_entity was offered (or not recorded)",
                          "decision 3: a commander command while a click_entity was offered (or not recorded)",
                          f"commander decisions at ticks 1000 and 1029 are less than {commander.TRIGGER_GAP} ticks "
                          "apart"], commander.pace_violations(decisions))
        self.assertEqual([{"tick": 1000, "kind": "fire", "action": "aim_power", "option": "cluster:1", "play": "P0"},
                          {"tick": 1029, "kind": "wait", "action": "wait", "option": None, "play": "P0"}],
                         commander.command_log(decisions)[:2])
        self.assertEqual(4, len(commander.command_log(decisions)))

    def test_random_commander_is_seeded_per_consult(self):
        brain, plays = RandomCommander(), [{"id": f"P{i}"} for i in range(5)]
        key = {"seed": 1001, "genome_id": "abc", "wave": 3, "consult": 4}
        self.assertEqual(brain.choose(None, plays, key), brain.choose(None, plays, dict(key)))
        picks = {brain.choose(None, plays, {**key, "consult": k})[0] for k in range(40)}
        self.assertEqual(set(range(5)), picks)
        one = play(CommanderPolicy(BuildOrderPolicy(PLAN), RandomCommander(), seed=1001))
        two = play(CommanderPolicy(BuildOrderPolicy(PLAN), RandomCommander(), seed=1001))
        self.assertEqual(one["trace_sha256"], two["trace_sha256"])
        self.assertEqual("terminal", one["status"])

    def test_placebo_is_the_rule_arm_one_tick_late(self):
        rule = play(CommanderPolicy(BuildOrderPolicy(PLAN), RuleCommander(), seed=1001))
        placebo = play(CommanderPolicy(BuildOrderPolicy(PLAN), PlaceboCommander(), seed=1001))
        first_rule, first_placebo = commanded(rule)[0], commanded(placebo)[0]
        self.assertEqual(first_rule["tick"] + 1, first_placebo["tick"])
        self.assertEqual(first_rule["action"]["option"], first_placebo["action"]["option"])
        self.assertIn({"action": {"action": "wait", "ticks": 1}}, placebo["plan"])
        self.assertNotIn({"action": {"action": "wait", "ticks": 1}}, rule["plan"])
        self.assertTrue(all(d["provenance"].startswith("commander:placebo:") for d in commanded(placebo)))

    def test_forced_choice_forks_one_consult(self):
        base = play(CommanderPolicy(BuildOrderPolicy(PLAN), RuleCommander(), seed=1001))
        consults = [c for d in base["decisions"] for c in d["meta"].get("commander_consults", [])]
        target = consults[2]
        other = next(i for i in range(len(target["plays"])) if i != target["choice"])
        fork = play(CommanderPolicy(BuildOrderPolicy(PLAN), ForcedChoice(RuleCommander(), 2, other), seed=1001))
        forked = [c for d in fork["decisions"] for c in d["meta"].get("commander_consults", [])]
        self.assertEqual(other, forked[2]["choice"])
        self.assertTrue(forked[2]["record"]["forced"])
        self.assertEqual([c["choice"] for c in consults[:2]], [c["choice"] for c in forked[:2]])
        # Everything before the forced consult replays exactly.
        before = [(d["tick"], d["action"]) for d in base["decisions"] if d["tick"] < target["tick"]]
        self.assertEqual(before, [(d["tick"], d["action"]) for d in fork["decisions"] if d["tick"] < target["tick"]])
        # A play index the consult does not offer leaves the base brain's choice (recorded as unavailable).
        brain = ForcedChoice(RuleCommander(), 0, 5)
        index, record = brain.choose(None, [{"id": "P0", "threat": None}], {"consult": 0})
        self.assertEqual((0, False, 1), (index, record["forced"], record["fork_unavailable"]))


class FakeBroker:
    """Scores the play whose text contains ``prefer`` highest (0.7), the rest evenly; records requests."""

    def __init__(self, prefer="Counter"):
        self.prefer, self.requests = prefer, []

    def distribution(self, request):
        self.requests.append(request)
        lines = [line for line in request["user"].splitlines() if line[:2] in {f"{l}." for l in request["labels"]}]
        hits = [i for i, line in enumerate(lines) if self.prefer in line]
        pick = hits[0] if hits else len(lines) - 1
        p = [0.3 / max(1, len(lines) - 1)] * len(lines)
        p[pick] = 0.7
        total = sum(p)
        p = [x / total for x in p]
        return {"id": request["id"], "labels": request["labels"], "p": p, "logp": [math.log(x) for x in p],
                "choice": request["labels"][pick], "model": "fake"}


class LanguageCommanderTests(unittest.TestCase):
    def test_every_cyclic_order_is_scored_and_only_offered_plays_win(self):
        broker = FakeBroker(prefer="Focus")
        policy = CommanderPolicy(BuildOrderPolicy(PLAN), LanguageCommander(broker), seed=1001)
        result = play(policy)
        self.assertEqual("terminal", result["status"])
        consults = policy.summary()["consults"]
        self.assertTrue(consults)
        first = consults[0]
        n = len(first["plays"])
        self.assertEqual("P4", first["play_id"])
        self.assertEqual(n, first["record"]["rotations"])
        self.assertAlmostEqual(1.0, sum(first["p"]))
        self.assertEqual(first["prompt_sha256"], first["record"]["prompt_sha256"])
        asked = broker.requests[:n]
        self.assertEqual([[chr(65 + i) for i in range(n)]] * n, [r["labels"] for r in asked])
        self.assertEqual(commander.SYSTEM, asked[0]["system"])
        # In rotation k, option A is play k: every play is scored once in every position.
        for k, request in enumerate(asked):
            self.assertIn("A. " + first["plays"][k]["text"], request["user"])
        self.assertEqual(n * len([c for c in consults if len(c["plays"]) > 1]), len(broker.requests))

    def test_a_single_play_needs_no_request(self):
        broker = FakeBroker()
        index, record = LanguageCommander(broker).choose(Briefing({}), [{"id": "P0", "text": "x"}], {})
        self.assertEqual((0, True, []), (index, record["trivial"], broker.requests))


class EvalPlayerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="alpharush-commander-")
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def context(self, kind):
        run_dir = self.root / "runs" / f"{kind}-0001"
        run_dir.mkdir(parents=True)
        return phase.PhaseRunContext(f"{kind}-0001", run_dir, self.root, time.monotonic() + 300, 50,
                                     job_kind="native-eval")

    def spec(self, players, tasks=None):
        return {"tasks": tasks or [{"level": 22, "seed": 1001, "genome": PLAN}], "workers": 2,
                "profile_stars_per_level": 2, "players": players}

    def run_loop(self, spec, kind, **kwargs):
        ctx = self.context(kind)
        out = Journal(ctx.output_dir / "episodes.jsonl")
        factory = lambda task, profile, index, player, port: CommandToy(seed=task["seed"], gold=400)
        summary = eval_job.eval_loop(spec, factory, ctx, out, pools=POOLS, protocol=PROTOCOL, ports=[9001, 9002],
                                     **kwargs)
        return summary, [row["payload"] for row in out.entries() if row["kind"] == "eval_episode"]

    def test_commander_players(self):
        players = ["plan", "cmd_passive", "cmd_rule", "cmd_random", "cmd_placebo"]
        summary, rows = self.run_loop(self.spec(players), "native-eval")
        self.assertEqual(set(players), set(summary["results"]))
        self.assertEqual([], summary["episode_errors"])
        by_player = {row["player"]: row for row in rows}
        self.assertEqual(by_player["plan"]["result"]["plan"], by_player["cmd_passive"]["result"]["plan"])
        self.assertEqual("rule", by_player["cmd_rule"]["result"]["commander"]["brain"])
        self.assertTrue(by_player["cmd_rule"]["result"]["commander"]["consults"])
        self.assertNotIn("commander", by_player["plan"]["result"])
        self.assertTrue(by_player["cmd_placebo"]["result"]["commander"]["held_commands"] > 0)

    def test_eval_rows_carry_the_pace_evidence(self):
        players = ["cmd_passive", "cmd_rule", "cmd_placebo"]
        _, rows = self.run_loop(self.spec(players), "native-eval")
        by_player = {row["player"]: row["result"]["commander"] for row in rows}
        # The row's own copy of the commands, as the episode played them (replayed here: the toy is deterministic).
        replay = play(commander.make_player("cmd_rule", PLAN, {"seed": 1001}))
        self.assertEqual(commander.command_log(replay["decisions"]), by_player["cmd_rule"]["command_log"])
        self.assertTrue({"fire", "reinf", "hero"} <= {c["kind"] for c in by_player["cmd_rule"]["command_log"]})
        self.assertTrue(all(set(c) == {"tick", "kind", "action", "option", "play"}
                            for c in by_player["cmd_rule"]["command_log"]))
        self.assertTrue(by_player["cmd_placebo"]["command_log"])
        self.assertEqual([], by_player["cmd_passive"]["command_log"])
        for player in players:
            self.assertEqual([], by_player[player]["pace_violations"], player)
        # A violation is kept in the row, not hidden: the same episode with a doctored command tick.
        doctored = copy.deepcopy(replay["decisions"])
        mine = [d for d in doctored if d["provenance"].startswith("commander:")]
        mine[1]["tick"] = mine[0]["tick"] + 1
        self.assertIn(f"commander decisions at ticks {mine[0]['tick']} and {mine[1]['tick']} are less than "
                      f"{commander.TRIGGER_GAP} ticks apart", commander.pace_violations(doctored))

    def test_fork_tasks_and_the_8b_player(self):
        fork = {"base": "rule", "consult": 1, "play": 2}
        tasks = [{"level": 22, "seed": 1001, "genome": PLAN, "fork": fork},
                 {"level": 22, "seed": 1001, "genome": PLAN, "fork": {**fork, "play": 0}}]
        summary, rows = self.run_loop(self.spec(["cmd_fork"], tasks), "native-eval")
        self.assertEqual(2, summary["results"]["cmd_fork"]["22"]["games"])
        self.assertEqual([fork, {**fork, "play": 0}], [row["fork"] for row in sorted(rows, key=lambda r: r["index"])])
        consults = sorted(rows, key=lambda r: r["index"])[0]["result"]["commander"]["consults"]
        self.assertEqual((2, True), (consults[1]["choice"], consults[1]["record"]["forced"]))
        bad = [self.spec(["cmd_fork", "plan"], tasks), self.spec(["cmd_fork"]), self.spec(["cmd_rule"], tasks),
               self.spec(["cmd_fork"], [{**tasks[0], "fork": {**fork, "base": "passive"}}]),
               self.spec(["cmd_fork"], [{**tasks[0], "fork": {**fork, "play": 6}}]),
               self.spec(["cmd_fork"], [{**tasks[0], "fork": {**fork, "consult": -1}}]),
               self.spec(["cmd_fork"], [{**tasks[0], "fork": {"base": "rule"}}]),
               self.spec(["cmd_fork"], [tasks[0], tasks[0]]), self.spec(["cmd_magic"]),
               # No 8B fork before its determinism check has passed.
               self.spec(["cmd_fork"], [{**tasks[0], "fork": {**fork, "base": "8b"}}])]
        for spec in bad:
            self.assertIsNone(eval_job.check_eval(spec, POOLS)[0], spec)
        self.assertEqual(("steady", "rule", "random", "placebo"), eval_job.FORK_BASES)
        self.assertEqual([], eval_job.check_eval(self.spec(["cmd_fork"], tasks), POOLS)[1])
        # The 8B player needs the strategy worker; with a (fake) broker it plays like the others.
        with self.assertRaisesRegex(GateRefused, "strategy worker"):
            self.run_loop(self.spec(["cmd_8b"]), "native-eval2")
        self.assertTrue(eval_job.needs_language_model(self.spec(["cmd_fork"], [{**tasks[0], "fork": {**fork, "base": "8b"}}])))
        broker = FakeBroker()
        summary, rows = self.run_loop(self.spec(["cmd_8b"]), "native-eval3", make_commander=lambda player, genome, task:
                                      commander.make_player(player, genome, task, broker=broker))
        self.assertEqual(1, summary["results"]["cmd_8b"]["22"]["games"])
        self.assertTrue(broker.requests)
        self.assertEqual("8b", rows[0]["result"]["commander"]["brain"])

    def test_commander_players_play_elite_stages_on_train_seeds_only(self):
        evaluation_seed = POOLS["pools"]["evaluation"]["seeds"][0]
        fork = {"base": "rule", "consult": 0, "play": 1}
        for player in eval_job.COMMANDER_PLAYERS:
            extra = {"fork": fork} if player == "cmd_fork" else {}
            with self.subTest(player=player):
                ok = [{"level": level, "seed": 1001, "genome": PLAN, **extra} for level in (13, 22, 26)]
                self.assertEqual([], eval_job.check_eval(self.spec([player], ok), POOLS)[1])
                for task, problem in (
                        ({"level": 22, "seed": evaluation_seed}, f"seed {evaluation_seed} is evaluation"),
                        ({"level": 12, "seed": 1001}, "play only the elite stages (levels 13-26), not level 12"),
                        ({"level": 1, "seed": 1001}, "not level 1"), ({"level": 27, "seed": 1001}, "not level 27"),
                        ({"level": "22", "seed": 1001}, "not level '22'")):
                    spec, issues = eval_job.check_eval(self.spec([player], [{**task, "genome": PLAN, **extra}]),
                                                       POOLS)
                    self.assertIsNone(spec)
                    self.assertTrue(any(i.startswith("tasks[0]: commander players (cmd_*) play") and problem in i
                                        for i in issues), issues)
        # Mixed with the plan player, the plan player's tasks follow the same rule; alone, it plays anything.
        main = [{"level": 1, "seed": evaluation_seed, "genome": PLAN}]
        self.assertIsNone(eval_job.check_eval(self.spec(["plan", "cmd_rule"], main), POOLS)[0])
        self.assertEqual([], eval_job.check_eval(self.spec(["plan", "operator"], main), POOLS)[1])


class SurveyGateTests(TempWorkspace):
    def eval_issues(self, players, scope, **extra):
        job = self.cfg["jobs"]["native-eval"]
        job.update({"action_scope": scope, "rng_mode": "isolate_sound+stable_pairs", **extra,
                    "eval": {"tasks": [{"level": 22, "seed": 1001, "genome": PLAN}], "workers": 1,
                             "profile_stars_per_level": 2, "players": players}})
        self.config.write_text(json.dumps(self.cfg, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return SURVEY.plan_survey(self.config, "native-eval")["issues"]

    def test_commander_players_need_v4_and_networks_stay_out_of_it(self):
        self.assertIn("commander players (cmd_*) need action_scope v4", self.eval_issues(["cmd_rule"], "v3"))
        issues = self.eval_issues(["cmd_rule", "cmd_passive", "plan"], "v4")
        self.assertFalse([i for i in issues if "v4" in i or "cmd_" in i or "action_scope" in i], issues)
        self.assertIn('an 8B commander evaluation must declare gpu="external-inference"',
                      self.eval_issues(["cmd_8b"], "v4"))
        self.assertNotIn('an 8B commander evaluation must declare gpu="external-inference"',
                         self.eval_issues(["cmd_8b"], "v4", gpu="external-inference"))
        self.assertTrue(any(i.startswith("action scope v4 is played by the plan executor")
                            for i in self.eval_issues(["operator"], "v4")))
        self.assertFalse(any(i.startswith("action scope v4") for i in self.eval_issues(["operator"], "v3")))
        self.assertEqual([], SURVEY._v4_problems({"action_scope": "v4"}, False))

    def test_searches_and_collections_stay_off_v4(self):
        for kind in ("native-search", "native-collect"):
            refusal = f"action scope v4 has no operator feature layout yet (v5): {kind} needs v2 or v3"
            for scope in ("v4", "v3", "v2"):
                with self.subTest(kind=kind, scope=scope):
                    self.cfg["jobs"][kind]["action_scope"] = scope
                    self.config.write_text(json.dumps(self.cfg, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
                    issues = SURVEY.plan_survey(self.config, kind)["issues"]
                    self.assertIs(scope == "v4", refusal in issues, issues)
            self.assertEqual([refusal], SURVEY._v4_problems({"action_scope": "v4"}, True, kind))
        self.assertEqual([], SURVEY._v4_problems({"action_scope": "v4"}, False, "native-eval"))


if __name__ == "__main__":
    unittest.main()


class CommanderPartsTests(unittest.TestCase):
    def test_component_arms_control_only_their_parts(self):
        from alpharush_rl import commander as c
        spells, hero = c.make_brain("rule_spells"), c.make_brain("rule_hero")
        self.assertEqual(("spells",), spells.parts)
        self.assertEqual(("hero",), hero.parts)
        self.assertNotIn("parts", vars(c.RuleCommander))
        tr = c.Translator(parts=("hero",))
        self.assertFalse("spells" in tr.parts)
        self.assertEqual({"cmd_rule_spells", "cmd_rule_hero"} <= set(c.PLAYER_BRAINS), True)


class SteadyPlayTests(unittest.TestCase):
    def test_steady_brain_plays_exactly_as_the_executor(self):
        from alpharush_rl.commander import SteadyBrain
        alone = play(BuildOrderPolicy(PLAN))
        steady = play(CommanderPolicy(BuildOrderPolicy(PLAN), SteadyBrain(), seed=1001))
        for key in ("plan", "trace_sha256", "outcome", "n_forced_waits"):
            self.assertEqual(alone[key], steady[key], key)
        self.assertEqual([(d["tick"], d["action"]) for d in alone["decisions"]],
                         [(d["tick"], d["action"]) for d in steady["decisions"]])

    def test_a_play_owns_only_the_powers_it_changes(self):
        from alpharush_rl.commander import STEADY, Translator
        tr = Translator()
        tr.play = {"id": "P3", "hero": {"to": "exit", "exit": "2"}, "fire": STEADY, "reinf": STEADY}
        self.assertEqual(set(), tr.owned_powers())
        self.assertTrue(tr.hero_out())
        tr.play = {"id": "P4", "hero": {"to": "home"}, "fire": {"mode": "reserve", "template": "x", "path": 1},
                   "reinf": STEADY}
        self.assertEqual({1}, tr.owned_powers())
        self.assertFalse(tr.hero_out())
        self.assertEqual(set(), Translator(parts=("hero",)).owned_powers())


class SteadyHomeTests(unittest.TestCase):
    def test_the_steady_play_never_moves_a_hero_the_commander_has_not_moved(self):
        from alpharush_rl.commander import Translator
        tr = Translator()
        tr.play = {"id": "P0", "hero": {"to": "home"}, "fire": commander.STEADY, "reinf": commander.STEADY}
        away = snapshot(hero={"rally_x": 520, "rally_y": 260})  # not at the catalog's home point (300, 200)
        self.assertIsNone(tr.command(away, build_menu(away)))
        tr.moved_hero = True
        out = tr.command(away, build_menu(away))
        self.assertEqual(("hero", "home"), (out["kind"], out["item"]["action"]["option"]))
