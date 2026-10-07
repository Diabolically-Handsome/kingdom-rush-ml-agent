"""host.lua action scope v3 (elite stages): rally points, enemy flags, per-lane progress, dormant-aware
wave gate. Runs in the game's own LuaJIT on the v2 test world plus a barrack, two path lanes and bosses."""
from __future__ import annotations

import unittest

try:
    from test_host_actions_v2 import NOT_ENABLED, OUTSIDE, HostLuaCase, as_list, native_state
except ImportError:  # run as tests.test_host_actions_v3 from the workspace root
    from tests.test_host_actions_v2 import NOT_ENABLED, OUTSIDE, HostLuaCase, as_list, native_state

# Barrack 22 (tower_barrack_3, menu with tw_rally) at (200, 200) with range_offset (0, 10): rally ellipse
# centred (200, 210), radius 100, aspect 0.7. Path 1 has two lanes of 11 nodes from x=100 to x=300
# (y 210 and 230); the node at (100, 210) is water, so it is no rally point. Enemies: 41 an awake boss
# inside the ellipse; 42 a sleeping (dormant) Cerberus; 43 a shielded boss (ignore_damage); 44 a flying,
# unblockable enemy walking lane 1 at node 6.
ELITE = r"""
F_BLOCK, F_BOSS, F_MINIBOSS, F_FLYING = 1, 32, 64, 128
F_ALL = 0xFFFFFFFF
local lane1, lane2 = {}, {}
for i = 0, 10 do
    lane1[#lane1 + 1] = {x = 100 + 20 * i, y = 210}
    lane2[#lane2 + 1] = {x = 100 + 20 * i, y = 230}
end
PATH_DB.paths = {{lane1, lane2}}
GROUND["100,210"] = {terrain = TERRAIN_WATER, nodes = 0}
local b = ENTITIES[22]
b.pos = {x = 200, y = 200}
b.tower.range_offset = {x = 0, y = 10}
b.barrack = {rally_range = 100, rally_pos = {x = 210, y = 215}, rally_terrains = TERRAIN_LAND}
ENTITIES[41] = {template_name = "eb_test", enemy = {}, health = {hp = 500}, vis = {flags = F_BOSS, bans = 0},
                pos = {x = 150, y = 215}}
ENTITIES[42] = {template_name = "enemy_demon_cerberus", enemy = {}, health = {hp = 6000, ignore_damage = false},
                vis = {flags = F_BOSS + F_MINIBOSS, bans = F_ALL}, sleeping = true}
ENTITIES[43] = {template_name = "eb_ulgukhai", enemy = {}, health = {hp = 9000, ignore_damage = true},
                vis = {flags = F_BOSS, bans = 0}}
ENTITIES[44] = {template_name = "enemy_gargoyle", enemy = {}, health = {hp = 50},
                vis = {flags = F_FLYING, bans = F_BLOCK}, nav_path = {pi = 1, spi = 1, ni = 6}}
NATIVE_STATE.enemies = {
    {id = 41, x = 150, y = 215, hp = 500, path_progress = 0.3},
    {id = 42, x = 800, y = 300, hp = 6000, path_progress = 0.95},
    {id = 43, x = 600, y = 400, hp = 9000, path_progress = 0.7},
    {id = 44, x = 200, y = 210, hp = 50, path_progress = 0.2},
}
NATIVE_STATE.enemy_count = 4
"""
RALLY = {"entry": (120, 210), "center": (200, 210), "exit": (300, 210), "boss": (140, 210)}


class HostV3Tests(HostLuaCase):
    scope = "v3"

    def setUp(self):
        super().setUp()
        self.lua.run(ELITE)

    def world(self, scope):
        lua = self.fresh(scope)
        lua.run(ELITE)
        return lua

    def rallies(self, lua=None):
        return {item["option"]: (item["x"], item["y"]) for item in self.catalog(lua)
                if item["action"] == "set_rally" and item["tower_id"] == 22}

    def enemies(self, lua=None):
        return {e["id"]: e for e in as_list(native_state(lua or self.lua)["enemies"])}

    def test_rally_options_follow_the_gui_rules(self):
        self.assertEqual(RALLY, self.rallies())
        # Every offered point passed the GUI checks at its rounded coordinates.
        self.assertEqual({22}, {item["tower_id"] for item in self.catalog() if item["action"] == "set_rally"})

    def test_no_boss_option_without_an_awake_boss_in_range(self):
        self.lua.run("ENTITIES[41].sleeping = true")
        self.assertEqual({k: v for k, v in RALLY.items() if k != "boss"}, self.rallies())
        self.lua.run("ENTITIES[41].sleeping = nil; NATIVE_STATE.enemies[1].x = 400")
        self.assertNotIn("boss", self.rallies())

    def test_rally_anywhere_skips_the_terrain_checks(self):
        self.lua.run("ENTITIES[22].barrack.rally_anywhere = true")
        self.assertEqual((100, 210), self.rallies()["entry"])

    def test_v2_offers_no_rally_and_no_flags(self):
        lua = self.world("v2")
        self.assertFalse([item for item in self.catalog(lua) if item["action"] == "set_rally"])
        state = native_state(lua)
        self.assertNotIn("active_enemy_count", state)
        self.assertTrue(all("dormant" not in e for e in as_list(state["enemies"])))
        reply = self.rpc_on(lua, '{"id":"r","action":"set_rally","tower_id":22,"option":"center","x":200,"y":210}')
        self.assertEqual(NOT_ENABLED, reply["error"])

    def rpc_on(self, lua, line):
        from test_host_actions_v2 import rpc
        return rpc(lua, line)

    def test_enemy_flags_and_lane_progress(self):
        enemies = self.enemies()
        self.assertEqual((True, False, False), (enemies[41]["boss"], enemies[41]["dormant"], enemies[41]["untargetable"]))
        self.assertEqual((True, True, True), (enemies[42]["boss"], enemies[42]["dormant"], enemies[42]["untargetable"]))
        self.assertEqual((True, False, True), (enemies[43]["boss"], enemies[43]["dormant"], enemies[43]["untargetable"]))
        self.assertEqual((True, True, False), (enemies[44]["flying"], enemies[44]["unblockable"], enemies[44]["boss"]))
        self.assertAlmostEqual(0.5, enemies[44]["path_progress"])  # node 6 of 11 on its own lane
        self.assertEqual(0.3, enemies[41]["path_progress"])  # no nav_path: the bridge's value stays
        self.assertEqual(3, native_state(self.lua)["active_enemy_count"])

    def test_progress_follows_path_connections(self):
        # Path 1 (11 nodes) continues into path 2 (21 nodes), like level 18's tunnels; path 3 stands alone.
        self.lua.run("""
            local p1, p2, p3 = {}, {}, {}
            for i = 1, 11 do p1[i] = {x = 1000 + i, y = 0} end
            for i = 1, 21 do p2[i] = {x = 2000 + i, y = 0} end
            for i = 1, 11 do p3[i] = {x = 3000 + i, y = 0} end
            PATH_DB.paths = {{p1}, {p2}, {p3}}
            PATH_DB.path_connections = {[1] = 2}
            ENTITIES[44].nav_path = {pi = 1, spi = 1, ni = 6}
            ENTITIES[43].nav_path = {pi = 2, spi = 1, ni = 11}
            ENTITIES[41].nav_path = {pi = 3, spi = 1, ni = 6}
        """)
        enemies = self.enemies()
        self.assertAlmostEqual(5 / 30, enemies[44]["path_progress"])  # 5 of 10 + 20 nodes along its route
        self.assertAlmostEqual(20 / 30, enemies[43]["path_progress"])  # entered path 2 after path 1's 10 nodes
        self.assertAlmostEqual(0.5, enemies[41]["path_progress"])  # no connection: the lane fraction

    def test_seated_boss_is_dormant(self):
        self.lua.run('ENTITIES[43].phase = "sitting"')
        self.assertTrue(self.enemies()[43]["dormant"])

    def test_spells_never_anchor_on_untargetable_enemies(self):
        anchors = {item["anchor_id"] for item in self.catalog() if item["action"] == "use_power"}
        self.assertEqual({44}, anchors)  # 41 is within ANCHOR_SPACING of 44, which is further along
        self.lua.run("table.remove(NATIVE_STATE.enemies, 4)")
        anchors = {item["anchor_id"] for item in self.catalog() if item["action"] == "use_power"}
        self.assertEqual({41}, anchors)  # the dormant 42 and the shielded 43 are further along, never anchors

    def test_dormant_boss_does_not_hold_back_the_wave_call(self):
        self.lua.run("NATIVE_STATE.wave = 3; NATIVE_STATE.enemies = {{id = 42, x = 800, y = 300, hp = 6000}}; "
                     "NATIVE_STATE.enemy_count = 1")
        state = native_state(self.lua)
        self.assertTrue(state["wave_ready"])
        self.assertIn({"action": "send_wave", "cost": 0}, self.catalog())
        lua = self.world("v2")
        lua.run("NATIVE_STATE.wave = 3; NATIVE_STATE.enemies = {{id = 42, x = 800, y = 300, hp = 6000}}; "
                "NATIVE_STATE.enemy_count = 1")
        self.assertFalse(native_state(lua)["wave_ready"])
        self.lua.run("NATIVE_STATE.enemies[2] = {id = 77, x = 1, y = 1, hp = 5}; NATIVE_STATE.enemy_count = 2")
        self.assertFalse(native_state(self.lua)["wave_ready"])  # an enemy without an entity still counts

    def test_set_rally_is_the_gui_click(self):
        reply = self.rpc('{"id":"r","action":"set_rally","tower_id":22,"option":"exit","x":300,"y":210}')
        self.assertTrue(reply["ok"], reply)
        barrack = self.entity(22)["barrack"]
        self.assertEqual(({"x": 300, "y": 210}, True), (barrack["rally_pos"], barrack["rally_new"]))

    def test_set_rally_must_match_the_catalog(self):
        for line in ('{"id":"r","action":"set_rally","tower_id":22,"option":"exit","x":301,"y":210}',
                     '{"id":"r","action":"set_rally","tower_id":22,"option":"center","x":300,"y":210}',
                     '{"id":"r","action":"set_rally","tower_id":21,"option":"exit","x":300,"y":210}',
                     '{"id":"r","action":"set_rally","tower_id":22,"x":300,"y":210}'):
            with self.subTest(line=line):
                self.assertEqual(OUTSIDE, self.rpc(line)["error"])
        self.assertEqual({"x": 210, "y": 215}, self.entity(22)["barrack"]["rally_pos"])

    def test_blocked_or_unclickable_barrack_offers_no_rally(self):
        self.lua.run("ENTITIES[22].tower.blocked = true")
        self.assertEqual({}, self.rallies())
        self.lua.run("ENTITIES[22].tower.blocked = nil; ENTITIES[22].ui = {can_click = false}")
        self.assertEqual({}, self.rallies())

    def test_hello_reports_v3(self):
        self.assertEqual("v3", self.rpc('{"id":"h","action":"hello"}')["result"]["action_scope"])
        self.assertEqual("v3", native_state(self.lua)["action_scope"])


if __name__ == "__main__":
    unittest.main()
