"""Tests for click_entity (host catalog/handler, env receipt, menu, executor rule, operator legacy rows)."""
import os
from pathlib import Path

ROOT = Path(os.environ.get("PATCH_ROOT", "C:/Users/<user>/Documents/AlphaRush"))


def patch(path, pairs):
    p = ROOT / path
    s = p.read_text(encoding="utf-8")
    for old, new in pairs:
        assert s.count(old) == 1, (path, old[:80], s.count(old))
        s = s.replace(old, new)
    p.write_text(s, encoding="utf-8", newline="\n")


patch("tests/test_env_actions_v2.py", [
    ('''        self.assertEqual(v2.scope, ["wait", "build_tower", "send_wave", "upgrade_tower", "upgrade_power",
                                    "sell_tower", "use_power", "point_tower"])''',
     '''        self.assertEqual(v2.scope, ["wait", "build_tower", "send_wave", "upgrade_tower", "upgrade_power",
                                    "sell_tower", "use_power", "point_tower", "click_entity"])'''),
    ('''if __name__ == "__main__":''',
     '''class ClickEntityTests(unittest.TestCase):
    def test_receipt_requires_the_click_to_have_been_offered(self):
        click = {"action": "click_entity", "entity_id": 500, "x": 10, "y": 21}
        offered = {"action_catalog": [{**click, "kind": "tower_trap", "cost": 0}]}
        gone = {"action_catalog": [{"action": "send_wave", "cost": 0}]}
        self.assertEqual({"executed": True, "offered_after": True},
                         env_module._click_entity_receipt(offered, offered, click, 0))
        self.assertEqual({"executed": True, "offered_after": False},
                         env_module._click_entity_receipt(offered, gone, click, 0))
        self.assertFalse(env_module._click_entity_receipt(gone, gone, click, 0)["executed"])

    def test_menu_offers_the_click_as_its_own_option(self):
        from alpharush_rl.menus import build_menu
        state = {"wave": 3, "gold": 0, "holders": [], "towers": [], "enemies": [], "wave_ready": False,
                 "action_catalog": [{"action": "click_entity", "entity_id": 500, "x": 10, "y": 21,
                                     "template": "mod_jt_tower", "kind": "tower_trap", "cost": 0, "available": True,
                                     "legal": True}]}
        menu = build_menu(state)
        clicks = [item for item in menu if item["action"]["action"] == "click_entity"]
        self.assertEqual([{"action": "click_entity", "entity_id": 500, "x": 10, "y": 21}],
                         [item["action"] for item in clicks])
        self.assertIn("mod_jt_tower", clicks[0]["text"])
        self.assertIn("free the trapped tower", clicks[0]["text"])


if __name__ == "__main__":'''),
])

patch("tests/test_host_actions_v2.py", [(
    '''class HostV2HandleTests(HostLuaCase):''',
    '''class HostV2ClickTests(HostLuaCase):
    """Clicks a game script waits for: J.T. knocked down (finishing tap) and ice on a tower (3 clicks)."""

    def setUp(self):
        super().setUp()
        self.lua.run("""
ENTITIES[500] = {template_name = "mod_jt_tower", pos = {x = 10.4, y = 20.6}, ui = {can_click = true},
                 modifier = {target_id = 20}, required_clicks = 3}
ENTITIES[501] = {template_name = "eb_jt", pos = {x = 332.46, y = 395}, ui = {can_click = true}, dying = true,
                 tap_decal = "decal_jt_tap", health = {hp = 4, dead = false}}
ENTITIES[502] = {template_name = "decal_sheep", pos = {x = 50, y = 50}, ui = {can_click = true}, required_clicks = 5}
ENTITIES[503] = {template_name = "enemy_yeti", pos = {x = 60, y = 60}, ui = {can_click = true},
                 health = {hp = 300, dead = false}}
""")

    def clicks(self):
        return [item for item in self.catalog() if item["action"] == "click_entity"]

    def test_only_waited_for_clicks_are_offered(self):
        offered = {(c["entity_id"], c["x"], c["y"], c["kind"], c["template"]) for c in self.clicks()}
        self.assertEqual({(500, 10, 21, "tower_trap", "mod_jt_tower"), (501, 332, 395, "downed_boss", "eb_jt")},
                         offered)
        self.lua.run("ENTITIES[501].health.dead = true; ENTITIES[500].required_clicks = 0")
        self.assertEqual([], self.clicks())

    def test_a_click_sets_the_gui_flag_once(self):
        self.catalog()
        reply = self.rpc('{"id":"c1","action":"click_entity","entity_id":500,"x":10,"y":21}')
        self.assertEqual(undump(reply["result"]["native_wire"]),
                         {"type": "ok", "action": "click_entity", "entity_id": 500, "x": 10, "y": 21})
        self.assertIs(self.entity(500)["ui"]["clicked"], True)
        self.assertEqual([501], [c["entity_id"] for c in self.clicks()])  # pending until the script takes it
        refused = self.rpc('{"id":"c2","action":"click_entity","entity_id":500,"x":10,"y":21}')
        self.assertIs(refused["ok"], False)
        refused = self.rpc('{"id":"c3","action":"click_entity","entity_id":502,"x":50,"y":50}')
        self.assertIs(refused["ok"], False)  # an easter-egg decoration is not a waited-for click


class HostV2HandleTests(HostLuaCase):''')])

patch("tests/test_search.py", [(
    '''class PackageGeneTests(unittest.TestCase):''',
    '''class ClickRuleTests(unittest.TestCase):
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


class PackageGeneTests(unittest.TestCase):''')])

patch("tests/test_operator_net.py", [(
    '''    def test_weights_round_trip(self):''',
    '''    def test_rows_and_weights_from_before_click_entity_still_load(self):
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

    def test_weights_round_trip(self):''')])
print("click tests patched")
