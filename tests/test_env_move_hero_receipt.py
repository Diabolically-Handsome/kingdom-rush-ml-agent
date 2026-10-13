"""v4 move_hero completion receipts (alpharush_rl/env.py): the clicked point as the hero's rally point, or an order
the hero's own script demonstrably replaced with its own position inside the window (Ignus' Surge of Flame ends its
dash with nav_rally.pos = center = its position; host.lua hero_orders records it). Fake host, no game."""
from __future__ import annotations

import copy
import unittest
from unittest import mock

from alpharush_rl import env as env_module

ACTION = {"action": "move_hero", "hero_id": 35, "x": 637, "y": 208, "option": "up:2"}
TICK = 27143


def snapshot(tick=TICK, **hero):
    """A v4 state with hero 35 (Ignus) offered the move to (637,208)."""
    base = {"id": 35, "template": "hero_ignus", "level": 4, "hp": 477, "hp_max": 490, "dead": False,
            "x": 489.09090733085, "y": 517.25080923455, "rally_x": 488, "rally_y": 531, "rally_new": False,
            "can_click": True, "stunned": False, "home_x": 60, "home_y": 340}
    base.update(hero)
    return {"tick": tick, "lives": 20, "gold": 300, "wave": 7, "wave_total": 15, "level_idx": 15, "enemies": [],
            "heroes": [base], "towers": [], "holders": [], "powers_ui": [], "next_wave": None, "wave_spawning": False,
            "action_catalog": [{**{k: v for k, v in ACTION.items()}, "cost": 0, "available": True, "legal": True}]}


# Seed 1201's void (level 15): the click landed during the dash; at tick 27149 the dash ended and the surge set
# rally pos and center to the hero's position (464.52.., 521.87..); the hero then took that rally (rally_new false).
SURGED = {"rally_x": 464.52241569916, "rally_y": 521.87565722596, "rally_new": False, "x": 468.6, "y": 521.1,
          "order": {"x": 637, "y": 208, "tick": TICK,
                    "replaced": {"tick": 27149, "x": 464.52241569916, "y": 521.87565722596, "at_hero": True}}}


def receipt(after_hero, before_hero=None, before_tick=TICK, after_tick=TICK + 30, action=ACTION):
    before = env_module.observation(snapshot(before_tick, **(before_hero or {})))
    after = env_module.observation(snapshot(after_tick, **after_hero)) if after_hero is not None else \
        {**env_module.observation(snapshot(after_tick)), "heroes": []}
    return env_module._move_hero_receipt(before, after, action, 0.0)


class MoveHeroReceiptTests(unittest.TestCase):
    def test_the_clicked_point_within_a_pixel_is_executed(self):
        out = receipt({"rally_x": 637.4, "rally_y": 207.2, "rally_new": True,
                       "order": {"x": 637, "y": 208, "tick": TICK}})
        self.assertEqual((True, False, None), (out["executed"], out["superseded_by_hero_script"],
                                               out["order_replaced"]))
        self.assertTrue(receipt({"rally_x": 637, "rally_y": 208})["executed"])  # no order record needed

    def test_an_order_the_heros_script_replaced_with_its_own_position_is_executed_and_says_so(self):
        out = receipt(SURGED)
        self.assertEqual((True, True), (out["executed"], out["superseded_by_hero_script"]))
        self.assertEqual(SURGED["order"]["replaced"], out["order_replaced"])
        self.assertEqual([464.52241569916, 521.87565722596], out["rally_after"])
        # Seed 1202's: the window ended during the surge's end animation, before the hero took the new rally.
        still = copy.deepcopy(SURGED)
        still["rally_new"], still["order"]["replaced"]["tick"] = True, TICK + 30
        self.assertEqual((True, True), (receipt(still)["executed"], receipt(still)["superseded_by_hero_script"]))

    def test_anything_short_of_that_proof_fails(self):
        def changed(edit):
            hero = copy.deepcopy(SURGED)
            edit(hero)
            return hero
        cases = {
            "no order record (an older host)": changed(lambda h: h.pop("order")),
            "order without replacement": changed(lambda h: h["order"].pop("replaced")),
            "replaced by another point": changed(lambda h: h["order"]["replaced"].update(at_hero=False)),
            "at_hero not literally true": changed(lambda h: h["order"]["replaced"].update(at_hero=1)),
            "replaced before the click": changed(lambda h: h["order"]["replaced"].update(tick=TICK)),
            "replaced after the window": changed(lambda h: h["order"]["replaced"].update(tick=TICK + 31)),
            "replacement tick not a number": changed(lambda h: h["order"]["replaced"].update(tick=True)),
            "another click's record": changed(lambda h: h["order"].update(x=636)),
            "a stale record": changed(lambda h: h["order"].update(tick=TICK - 30)),
            "rally moved on since": changed(lambda h: h.update(rally_x=470.0)),
            "no rally point": changed(lambda h: h.update(rally_y=None)),
            "replacement point unknown": changed(lambda h: h["order"]["replaced"].update(x=None)),
            "order not a record": changed(lambda h: h.update(order=[637, 208])),
        }
        for name, hero in cases.items():
            with self.subTest(case=name):
                out = receipt(hero)
                self.assertEqual((False, False), (out["executed"], out["superseded_by_hero_script"]))
        self.assertFalse(receipt(None)["executed"])  # the hero is gone
        other = dict(ACTION, x=600)
        self.assertFalse(receipt(SURGED, action=other)["executed"])


class FakeHost:
    """Stands in for engine.Worker inside the real NativeEnv: replies ok, then steps to a scripted state."""

    def __init__(self, **kwargs):
        self.calls, self.after = [], None

    def rpc(self, action, **kwargs):
        self.calls.append((action, kwargs))
        if action == "step":
            state = copy.deepcopy(self.after)
            state["tick"] = TICK + kwargs["ticks"]
            return {"state": state, "tick_before": TICK, "tick_after": state["tick"]}
        return {"type": "ok"}

    def close(self):
        pass


class NativeActTests(unittest.TestCase):
    def native(self, after_hero):
        with mock.patch.object(env_module, "Worker", FakeHost):
            native = env_module.NativeEnv(action_scope="v4")
        native.state = env_module.observation(snapshot())
        native.worker.after = snapshot(**after_hero)
        return native

    def test_a_superseded_order_is_a_receipt_not_a_void(self):
        native = self.native(SURGED)
        out = native.act(ACTION)
        self.assertEqual((True, True, TICK + env_module.HERO_TICKS),
                         (out["executed"], out["superseded_by_hero_script"], out["tick_after"]))
        self.assertEqual([{"action": ACTION}], native.plan)
        self.assertEqual("action", native.trace[-1]["kind"])

    def test_an_unexplained_rally_still_voids(self):
        # Seed 1201's receipt as the host before hero_orders reported it: no record, so no proof.
        hero = {k: v for k, v in SURGED.items() if k != "order"}
        with self.assertRaisesRegex(RuntimeError, "receipt failed"):
            self.native(hero).act(ACTION)


if __name__ == "__main__":
    unittest.main()
