"""host.lua action scope v4 (commander): aimed spells (aim_power) and hero moves (move_hero) on top of v3.
Runs in the game's own LuaJIT on the v2 test world plus the v3 elite additions and a straight two-lane path."""
from __future__ import annotations

import re
import unittest

try:
    from test_host_actions_v2 import NOT_ENABLED, OUTSIDE, HostLuaCase, as_list, native_state, rpc, undump
    from test_host_actions_v3 import ELITE
except ImportError:  # run as tests.test_host_actions_v4 from the workspace root
    from tests.test_host_actions_v2 import NOT_ENABLED, OUTSIDE, HostLuaCase, as_list, native_state, rpc, undump
    from tests.test_host_actions_v3 import ELITE

# Path 1: lane 1 along y=300 and lane 2 along y=320, 51 nodes each from x=0 to x=1000 (one every 20 px), so an
# enemy at node ni has lane progress (ni - 1) / 50. Both spells ready. Enemies: 201 the leader (speed 50);
# 202 blocked; 203 an egg-laying spider (a named threat) on lane 2; 204 and 205 a 20 px pair (the only cluster);
# 206 a flying, unblockable gargoyle. The v3 elite enemies 41-44 stay entities but are not in the state.
AIM = r"""
local lane1, lane2 = {}, {}
for i = 0, 50 do
    lane1[#lane1 + 1] = {x = 20 * i, y = 300}
    lane2[#lane2 + 1] = {x = 20 * i, y = 320}
end
PATH_DB.paths = {{lane1, lane2}}
game.game_gui.power_2 = {mode = "ready"}
function enemy(id, template, x, y, spi, ni, hp, speed)
    ENTITIES[id] = {template_name = template, enemy = {blockers = {}}, health = {hp = hp}, vis = {flags = 0, bans = 0},
                    pos = {x = x, y = y}, motion = {max_speed = speed}, nav_path = {pi = 1, spi = spi, ni = ni, dir = 1}}
    return {id = id, template = template, x = x, y = y, hp = hp, speed = speed}
end
NATIVE_STATE.enemies = {
    enemy(201, "enemy_goblin", 600, 300, 1, 31, 100, 50),
    enemy(202, "enemy_orc", 500, 300, 1, 26, 300, 30),
    enemy(203, "enemy_spider_big", 300, 320, 2, 16, 900, 25),
    enemy(204, "enemy_goblin", 100, 300, 1, 6, 400, 30),
    enemy(205, "enemy_goblin", 120, 300, 1, 7, 400, 30),
    enemy(206, "enemy_gargoyle", 400, 300, 1, 21, 50, 40),
}
ENTITIES[202].enemy.blockers = {7}
ENTITIES[206].vis = {flags = F_FLYING, bans = F_BLOCK}
NATIVE_STATE.enemy_count = 6
"""
# (option, x, y, target_id) per power. Rain of fire: the spider where it stands; 201 1.2 s (60 px) ahead; the
# blocked 202 without lead; the pair's hp-weighted centroid. Reinforcements: 40 px ahead of the four leading
# spaced ground enemies (205 and 204 are 20 px apart), then the pair's centroid; never the gargoyle.
AIMS = {
    1: {("threat:203", 300, 320, 203), ("lead:201", 660, 300, 201), ("lead:202", 500, 300, 202),
        ("cluster:1", 110, 300, None)},
    2: {("block:201", 640, 300, 201), ("block:202", 540, 300, 202), ("block:203", 340, 320, 203),
        ("block:205", 160, 300, 205), ("cluster:1", 110, 300, None)},
}
OPTION = re.compile(r"^(threat|lead|block):(\d+)$|^cluster:[12]$")
# The v3 elite enemies 41-43 join the AIM field ahead of everyone, unblocked at 30 px/s: 41 the awake boss (lane 1,
# node 36); 42 the sleeping Cerberus, banned from every effect (lane 1, node 41); 43 the boss shielded by
# ignore_damage (lane 2, node 42), 28 px from 42.
ELITE_AHEAD = r"""
for _, at in ipairs({{41, 700, 300, 1, 36}, {42, 800, 300, 1, 41}, {43, 820, 320, 2, 42}}) do
    local id, x, y, spi, ni = unpack(at)
    local e = ENTITIES[id]
    e.pos, e.motion, e.enemy.blockers = {x = x, y = y}, {max_speed = 30}, {}
    e.nav_path = {pi = 1, spi = spi, ni = ni, dir = 1}
    NATIVE_STATE.enemies[#NATIVE_STATE.enemies + 1] = {id = id, template = e.template_name, x = x, y = y,
                                                       hp = e.health.hp}
end
NATIVE_STATE.enemy_count = #NATIVE_STATE.enemies
"""

# A second path (lanes y=600 and y=620) ends at exit 2; path 1 ends at exit 1, where hero 70 (Gerald) stands.
# The live enemies walk path 1 and the next wave comes on path 2. The grid stand-in has 16 px cells and the
# game's find_waypoints contract: a route (here: midpoint, then the point) or nil for an UNREACHABLE target
# cell, written into its one-entry waypoints_cache as the game's A* does; WAYPOINT_CALLS records each call.
HERO = r"""
local lane3, lane4 = {}, {}
for i = 0, 50 do
    lane3[#lane3 + 1] = {x = 20 * i, y = 600}
    lane4[#lane4 + 1] = {x = 20 * i, y = 620}
end
PATH_DB.paths[2] = {lane3, lane4}
for _, en in ipairs(NATIVE_STATE.enemies) do en.path_index = 1 end
NATIVE_STATE.next_wave = {wave = 1, paths = {{path_index = 2, spawns = {}}}}
game.store.level.locations = {exits = {{id = 1, pos = {x = 1000, y = 310}}, {id = 2, pos = {x = 1000, y = 610}}}}
ENTITIES[70] = {template_name = "hero_gerald", hero = {level = 3, xp = 900},
                health = {hp = 400, hp_max = 400, dead = false, ignore_damage = false, dead_lifetime = 15, death_ts = 0},
                ui = {can_click = true, can_select = true}, unit = {is_stunned = false}, pos = {x = 1000, y = 310},
                nav_rally = {pos = {x = 1000, y = 310}, center = {x = 1000, y = 310}, new = false,
                             requires_node_nearby = true},
                nav_grid = {valid_terrains = bit.bor(TERRAIN_LAND, TERRAIN_ICE), valid_terrains_dest = TERRAIN_LAND,
                            waypoints = {}}}
NATIVE_STATE.heroes = {{id = 70, template = "hero_gerald", level = 3, hp = 400, hp_max = 400, dead = false,
                        x = 1000, y = 310, rally_x = 1000, rally_y = 310}}
GRID_DB.grid, GRID_DB.cell_size, GRID_DB.ox, GRID_DB.oy = {}, 16, 0, 0
GRID_DB.waypoints_cache = {}
GRID_DB.get_coords = function(self, x, y)
    return math.ceil((x - self.ox) / self.cell_size), math.ceil((y - self.oy) / self.cell_size)
end
UNREACHABLE, WAYPOINT_CALLS = {}, {}
GRID_DB.find_waypoints = function(self, from, fallback, to, terrains, no_cache)
    assert(self == GRID_DB, "grid_db method called without self")
    WAYPOINT_CALLS[#WAYPOINT_CALLS + 1] = {from = {x = from.x, y = from.y}, fallback = {x = fallback.x, y = fallback.y},
                                           to = {x = to.x, y = to.y}, terrains = terrains}
    local cache = self.waypoints_cache
    local fx, fy = self:get_coords(from.x, from.y)
    local tx, ty = self:get_coords(to.x, to.y)
    cache.from_c, cache.to_c, cache.path_c, cache.path = nil, nil, nil, nil
    if UNREACHABLE[tx .. "," .. ty] then return nil end
    local path = {{x = (from.x + to.x) / 2, y = (from.y + to.y) / 2}, {x = to.x, y = to.y}}
    cache.from_c, cache.to_c, cache.path_c, cache.path = {x = fx, y = fy}, {x = tx, y = ty}, {}, path
    return path
end
function waypoint_calls() return #WAYPOINT_CALLS end
function waypoints_cache_dump() return dump(GRID_DB.waypoints_cache) end
"""
# home: exit 1 where the hero spawned; exit:k: node 46 of 51 (0.9) of the path ending there; up:pi: node 19
# (0.35) of the next wave's path 2, then of the live enemies' path 1.
MOVES = {("home", 1000, 310), ("exit:1", 900, 300), ("exit:2", 900, 600), ("up:2", 360, 600), ("up:1", 360, 300)}


def v4_key(item):
    """The host's total catalog order (catalog_v2's sort keys)."""
    power = item.get("power")
    named = power if isinstance(power, str) else item.get("option")
    owner = next((item[key] for key in ("holder_id", "tower_id", "entity_id", "hero_id") if key in item), 0)
    return (item["action"], owner, item.get("tower_type") or item.get("target") or named or "",
            power if isinstance(power, float) else 0, item.get("x", 0), item.get("y", 0), item.get("anchor_id", 0))


class HostV4Case(HostLuaCase):
    scope = "v4"

    def setUp(self):
        super().setUp()
        self.lua.run(ELITE)
        self.lua.run(AIM)

    def world(self, scope):
        lua = self.fresh(scope)
        lua.run(ELITE)
        lua.run(AIM)
        return lua

    def aims(self, lua=None):
        out = {1: set(), 2: set()}
        for item in self.catalog(lua):
            if item["action"] == "aim_power":
                out[int(item["power"])].add((item["option"], item["x"], item["y"], item.get("target_id")))
        return out

    def assertRefusedWithoutSideEffects(self, lines, error=OUTSIDE):
        game_before, calls_before = self.lua.call("game_dump"), self.lua.call("calls_dump")
        for line in lines:
            with self.subTest(command=line):
                reply = self.rpc(line)
                self.assertEqual((False, error), (reply["ok"], reply["error"]))
        self.assertEqual(game_before, self.lua.call("game_dump"))
        self.assertEqual(calls_before, self.lua.call("calls_dump"))


class HostV4AimTests(HostV4Case):
    def test_aimed_points_follow_the_named_rules(self):
        self.assertEqual(AIMS, self.aims())
        for item in self.catalog():
            if item["action"] == "aim_power":
                self.assertEqual(0, item["cost"])
                self.assertRegex(item["option"], OPTION)
                # target_id exactly when the option names an enemy, and then that enemy.
                named = item["option"].partition(":")[2]
                self.assertEqual(None if item["option"].startswith("cluster") else float(named),
                                 item.get("target_id"))

    def test_v4_keeps_every_v3_offer(self):
        v3 = self.world("v3")
        self.assertEqual(self.catalog(v3), [item for item in self.catalog() if item["action"] != "aim_power"])
        self.assertEqual("v4", native_state(self.lua)["action_scope"])
        self.assertEqual("v4", self.rpc('{"id":"h","action":"hello"}')["result"]["action_scope"])

    def test_every_point_was_asked_the_gui_placement_questions(self):
        self.lua.run("GROUND_CALLS = {}")
        catalog = self.catalog()
        asked = {}
        for call in as_list(undump(self.lua.call("ground_calls_dump"))):
            asked.setdefault((call["x"], call["y"]), set()).add((call["name"], call["flags"]))
        for item in catalog:
            if item["action"] == "aim_power":
                questions = ({("cell_is", 4 | 16), ("valid_node_nearby", 8)} if item["power"] == 1
                             else {("valid_node_nearby", 2), ("cell_is_only", 1 | 8)})
                self.assertLessEqual(questions, asked[(item["x"], item["y"])], item)

    def test_cliff_faerie_and_water_points_are_refused(self):
        # A cliff at 201's lead point: the next spaced leader, the gargoyle, is led instead (48 px at speed 40).
        self.lua.run('GROUND["660,300"] = {terrain = TERRAIN_CLIFF, nodes = bit.bor(NF_POWER_1, NF_RALLY)}')
        rain = self.aims()[1]
        self.assertNotIn(("lead:201", 660, 300, 201), rain)
        self.assertIn(("lead:206", 448, 300, 206), rain)
        # Faerie ground under the spider: no rain of fire there (reinforcements never aim at it anyway).
        self.lua.run('GROUND["300,320"] = {terrain = bit.bor(TERRAIN_LAND, TERRAIN_FAERIE), '
                     'nodes = bit.bor(NF_POWER_1, NF_RALLY)}')
        self.assertNotIn("threat:203", {o for o, *_ in self.aims()[1]})
        # Water ahead of 201: reinforcements refuse it (land or ice only); the other blocks stay.
        self.lua.run('GROUND["640,300"] = {terrain = TERRAIN_WATER, nodes = bit.bor(NF_POWER_1, NF_RALLY)}')
        self.assertEqual(AIMS[2] - {("block:201", 640, 300, 201)}, self.aims()[2])
        self.assertRefusedWithoutSideEffects([
            '{"id":"x","action":"aim_power","power":1,"x":660,"y":300,"option":"lead:201","target_id":201}',
            '{"id":"x","action":"aim_power","power":1,"x":300,"y":320,"option":"threat:203","target_id":203}',
            '{"id":"x","action":"aim_power","power":2,"x":640,"y":300,"option":"block:201","target_id":201}',
        ])

    def test_a_shared_point_keeps_its_first_kind_only(self):
        # The blocked spider now leads: its unled point is the threat's point, so no lead:203 is offered and
        # the next two spaced leaders are led instead.
        self.lua.run("ENTITIES[203].nav_path.ni = 46; ENTITIES[203].enemy.blockers = {9}")
        self.assertEqual(AIMS[1], self.aims()[1])

    def test_options_are_distinct_capped_and_totally_ordered(self):
        # A crowded field: 60 enemies on both lanes, a third of them named threats, some bosses.
        self.lua.run(r"""
NATIVE_STATE.enemies = {}
for i = 1, 60 do
    local template = i % 3 == 0 and "enemy_swamp_thing" or "enemy_goblin"
    local lane, ni = i % 2 + 1, 1 + (i * 7) % 50
    local en = enemy(300 + i, template, 20 * (ni - 1) + i % 5, lane == 1 and 300 or 320, lane, ni, 50 + i * 13, 20 + i)
    if i % 11 == 0 then ENTITIES[300 + i].vis.flags = F_BOSS end
    NATIVE_STATE.enemies[i] = en
end
""")
        catalog = self.catalog()
        aims = [item for item in catalog if item["action"] == "aim_power"]
        self.assertEqual(catalog, sorted(catalog, key=v4_key))
        self.assertEqual(len(catalog), len({v4_key(item) for item in catalog}))
        for power in (1, 2):
            mine = [item for item in aims if item["power"] == power]
            self.assertEqual(6, len(mine))
            self.assertEqual(len(mine), len({(item["x"], item["y"]) for item in mine}))
            self.assertEqual(len(mine), len({item["option"] for item in mine}))
            self.assertTrue(all(isinstance(item["x"], float) and item["x"] == int(item["x"]) for item in mine))
        kinds = {power: sorted(item["option"].split(":")[0] for item in aims if item["power"] == power)
                 for power in (1, 2)}
        self.assertEqual({1: ["cluster", "cluster", "lead", "lead", "threat", "threat"],
                          2: ["block", "block", "block", "block", "cluster", "cluster"]}, kinds)
        # Bosses are the first threats named (311, 322, 333, 344 and 355), the healthiest first.
        self.assertEqual({"threat:344", "threat:355"},
                         {item["option"] for item in aims if item["option"].startswith("threat")})
        # Traversal order never shows.
        self.lua.run("local e = NATIVE_STATE.enemies\n"
                     "for i = 1, math.floor(#e / 2) do e[i], e[#e + 1 - i] = e[#e + 1 - i], e[i] end")
        self.assertEqual(catalog, self.catalog())

    def test_untargetable_enemies_are_never_aimed_at(self):
        def named(aims):
            return {option for power in (1, 2) for option, *_ in aims[power]}

        def targets(aims):
            return {target for power in (1, 2) for *_, target in aims[power]}
        self.lua.run(ELITE_AHEAD)
        aims = self.aims()
        self.assertFalse({f"{kind}:{i}" for kind in ("threat", "lead", "block") for i in (42, 43)} & named(aims))
        self.assertFalse({42, 43} & targets(aims))
        self.assertIn(("threat:41", 700, 300, 41), aims[1])  # the awake boss is aimed at
        # Exactly the offers of the same field without them: they neither lead, block nor weigh in a cluster.
        alone = self.world("v4")
        alone.run(ELITE_AHEAD + "table.remove(NATIVE_STATE.enemies); table.remove(NATIVE_STATE.enemies)")
        self.assertEqual(self.aims(alone), aims)
        self.assertFalse({42, 43} & {item.get("anchor_id") for item in self.catalog()
                                     if item["action"] == "use_power"})
        # Awake and hittable at the same places, both are offered: the two healthiest bosses are Rain of Fire's
        # threats, their pair is the heaviest cluster and 43 leads the blockable ground enemies.
        self.lua.run("ENTITIES[42].sleeping = nil; ENTITIES[42].vis.bans = 0\n"
                     "ENTITIES[43].health.ignore_damage = false")
        woken = self.aims()
        self.assertLessEqual({("threat:43", 820, 320, 43), ("threat:42", 800, 300, 42), ("cluster:1", 812, 312, None)},
                             woken[1])
        self.assertIn("block:43", named(woken))
        # Awake but still banned from every effect: 42 stays out.
        self.lua.run("ENTITIES[42].vis.bans = F_ALL")
        self.assertFalse({"threat:42", "lead:42", "block:42"} & named(self.aims()))
        self.assertNotIn(42, targets(self.aims()))

    def test_only_ready_powers_are_aimed(self):
        self.lua.run('game.game_gui.power_1 = {mode = "cooldown"}; game.game_gui.power_2 = {mode = "locked"}')
        self.assertEqual({1: set(), 2: set()}, self.aims())
        self.lua.run('game.game_gui.power_1 = {mode = "unlocked"}; game.game_gui.power_2 = nil')
        self.assertEqual({1: AIMS[1], 2: set()}, self.aims())

    def test_an_aimed_spell_is_the_use_power_cast(self):
        self.catalog()
        reply = self.rpc('{"id":"a","action":"aim_power","power":1,"x":660,"y":300,"option":"lead:201","target_id":201}')
        self.assertEqual({"native_wire": "bridge-wire-1"}, reply["result"])
        reply = self.rpc('{"id":"b","action":"aim_power","power":2,"x":110,"y":300,"option":"cluster:1"}')
        self.assertEqual({"native_wire": "bridge-wire-2"}, reply["result"])
        reply = self.rpc('{"id":"c","action":"use_power","power":1,"x":600,"y":300,"anchor_id":201}')
        self.assertEqual({"native_wire": "bridge-wire-3"}, reply["result"])
        # The same native call as use_power: only power and point, no option or target.
        self.assertEqual([{"action": "use_power", "power": 1, "x": 660, "y": 300},
                          {"action": "use_power", "power": 2, "x": 110, "y": 300},
                          {"action": "use_power", "power": 1, "x": 600, "y": 300}], self.calls())

    def test_off_catalog_aims_are_refused_without_side_effects(self):
        self.catalog()
        self.assertRefusedWithoutSideEffects([
            '{"id":"x","action":"aim_power","power":1,"x":661,"y":300,"option":"lead:201","target_id":201}',
            '{"id":"x","action":"aim_power","power":1,"x":660,"y":300,"option":"lead:201"}',
            '{"id":"x","action":"aim_power","power":1,"x":660,"y":300,"option":"lead:202","target_id":201}',
            '{"id":"x","action":"aim_power","power":1,"x":660,"y":300,"option":"lead:201","target_id":202}',
            '{"id":"x","action":"aim_power","power":2,"x":660,"y":300,"option":"lead:201","target_id":201}',
            '{"id":"x","action":"aim_power","power":1,"x":110,"y":300,"option":"cluster:1","target_id":204}',
            '{"id":"x","action":"aim_power","power":1,"x":110,"y":300}',
            '{"id":"x","action":"aim_power","power":"1","x":110,"y":300,"option":"cluster:1"}',
            '{"id":"x","action":"aim_power","power":1,"x":400,"y":300,"option":"threat:206","target_id":206}',
            '{"id":"x","action":"aim_power","power":1,"x":110,"y":300,"option":"cluster:1","anchor_id":204}',
        ])

    def test_v2_and_v3_never_offer_or_accept_aimed_spells(self):
        for scope in ("v2", "v3"):
            with self.subTest(scope=scope):
                lua = self.world(scope)
                self.assertFalse([item for item in self.catalog(lua) if item["action"] == "aim_power"])
                reply = rpc(lua, '{"id":"a","action":"aim_power","power":1,"x":660,"y":300,'
                                 '"option":"lead:201","target_id":201}')
                self.assertEqual((False, NOT_ENABLED), (reply["ok"], reply["error"]))
                self.assertEqual([], as_list(undump(lua.call("calls_dump"))))

    def test_level_power_areas_are_asked_with_the_gui_mode_only_in_v4(self):
        # No NF_POWER_1 node near 202 (500,300), but the level allows fire there for the GUI's POWER_1 mode, as
        # levels 8, 10 and 12 do for their special areas. v3 never asks (its number test); v4 asks as the GUI does.
        area = ('GUI_MODE_POWER_1 = "POWER_1"; MODES = {}\n'
                'GROUND["500,300"] = {terrain = TERRAIN_LAND, nodes = NF_RALLY}\n'
                'game.store.level.fn_can_power = function(self, store, mode, pos)\n'
                '    assert(self == game.store.level and store == game.store)\n'
                '    MODES[#MODES + 1] = mode\n'
                '    return mode == GUI_MODE_POWER_1 and pos.x == 500 and pos.y == 300\n'
                'end')
        for scope, offered in (("v3", False), ("v4", True)):
            with self.subTest(scope=scope):
                lua = self.world(scope)
                lua.run(area)
                catalog = self.catalog(lua)
                spells = {(item["action"], item.get("option")) for item in catalog
                          if item.get("power") == 1 and (item["x"], item["y"]) == (500, 300)}
                expected = {("use_power", None), ("aim_power", "lead:202")} if offered else set()
                self.assertEqual(expected, spells)
                lua.run("function modes_dump() return dump(MODES) end")
                self.assertEqual({"POWER_1"} if offered else set(), set(as_list(undump(lua.call("modes_dump")))))
                reply = rpc(lua, '{"id":"u","action":"use_power","power":1,"x":500,"y":300,"anchor_id":202}')
                self.assertIs(offered, reply["ok"])


class HostV4HeroTests(HostV4Case):
    def setUp(self):
        super().setUp()
        self.lua.run(HERO)

    def world(self, scope):
        lua = super().world(scope)
        lua.run(HERO)
        return lua

    def moves(self, lua=None):
        items = [item for item in self.catalog(lua) if item["action"] == "move_hero"]
        self.assertTrue(all(item["hero_id"] == 70 and item["cost"] == 0 for item in items))
        return {(item["option"], item["x"], item["y"]) for item in items}

    def hero(self, lua=None):
        return as_list(native_state(lua or self.lua)["heroes"])[0]

    def move(self, option, x, y, hero_id=70):
        return self.rpc(f'{{"id":"m","action":"move_hero","hero_id":{hero_id},"x":{x},"y":{y},"option":"{option}"}}')

    def test_moves_follow_the_gui_rally_hero_rules(self):
        self.assertEqual(MOVES, self.moves())
        catalog = self.catalog()
        self.assertEqual(catalog, sorted(catalog, key=v4_key))
        # Every path point was asked the GUI's questions (NF_RALLY node, destination terrain, a route).
        self.lua.run("GROUND_CALLS = {}")
        self.catalog()
        asked = {}
        for call in as_list(undump(self.lua.call("ground_calls_dump"))):
            asked.setdefault((call["x"], call["y"]), set()).add((call["name"], call["flags"]))
        for _, x, y in MOVES:
            self.assertLessEqual({("valid_node_nearby", 2), ("cell_is_only", 1)}, asked[(x, y)])

    def test_v4_hero_exports(self):
        hero = self.hero()
        self.assertEqual((True, False, False, 1000, 310), (hero["can_click"], hero["rally_new"], hero["stunned"],
                                                          hero["home_x"], hero["home_y"]))
        self.assertNotIn("respawn_remaining", hero)
        # Dead since tick_ts 1.25 with a 15 s respawn at tick_ts 3.25: 13 s to go; clicks disabled meanwhile.
        self.lua.run("ENTITIES[70].health.dead = true; ENTITIES[70].health.death_ts = 1.25\n"
                     "ENTITIES[70].ui.can_click = false; ENTITIES[70].unit.is_stunned = true\n"
                     "ENTITIES[70].nav_rally.new = true")
        hero = self.hero()
        self.assertEqual((False, True, True, 13), (hero["can_click"], hero["rally_new"], hero["stunned"],
                                                   hero["respawn_remaining"]))
        self.lua.run("game.store.tick_ts = 99")
        self.assertEqual(0, self.hero()["respawn_remaining"])
        # A respawn point is home; so is the level's custom spawn point for a hero that uses it.
        self.lua.run("ENTITIES[70].hero.use_custom_spawn_point = true\n"
                     "game.store.level.custom_spawn_pos = {x = 170.4, y = 371.6}")
        self.assertEqual((170, 372), (self.hero()["home_x"], self.hero()["home_y"]))
        self.lua.run("ENTITIES[70].hero.respawn_point = {x = 640, y = 300}")
        self.assertEqual((640, 300), (self.hero()["home_x"], self.hero()["home_y"]))
        # v3 exports none of it.
        for key in ("can_click", "rally_new", "stunned", "home_x", "home_y", "respawn_remaining"):
            self.assertNotIn(key, self.hero(self.world("v3")))

    def test_heroes_the_gui_would_not_rally_offer_nothing(self):
        cases = {
            "dead": "ENTITIES[70].health.dead = true",
            "respawning (dead while ignoring damage)":
                "ENTITIES[70].health.dead = true; ENTITIES[70].health.ignore_damage = true",
            "stunned": "ENTITIES[70].unit.is_stunned = true",
            "not clickable": "ENTITIES[70].ui.can_click = false",
            "not selectable": "ENTITIES[70].ui.can_select = false",
            "no ui": "ENTITIES[70].ui = nil",
            "gui disabled": 'game.game_gui.mode = "DISABLED"',
            "notification": 'GUI_MODE_NOTIFICATION = "NOTIFICATION"; game.game_gui.mode = GUI_MODE_NOTIFICATION',
            "no gui": "game.game_gui = nil",
            "no nav_grid": "ENTITIES[70].nav_grid = nil",
        }
        for name, change in cases.items():
            with self.subTest(case=name):
                self.lua = self.world("v4")
                self.lua.run(change)
                self.assertEqual(set(), self.moves())
                self.assertRefusedWithoutSideEffects(['{"id":"m","action":"move_hero","hero_id":70,"x":900,"y":300,'
                                                      '"option":"exit:1"}'])
        self.lua = self.world("v4")
        self.lua.run('game.game_gui.mode = "TOWER_MENU"; ENTITIES[70].ui.can_click = nil')
        self.assertEqual(MOVES, self.moves())  # any other GUI mode; can_click unset is not false
        # The GUI's rally click never looks at ignore_damage: a living hero that ignores damage is rallied.
        self.lua = self.world("v4")
        self.lua.run("ENTITIES[70].health.ignore_damage = true")
        self.assertEqual(MOVES, self.moves())
        self.assertIs(True, self.move("exit:1", 900, 300)["ok"])

    def test_points_failing_the_gui_rules_move_to_the_nearest_passing_node(self):
        self.lua.run('GROUND["900,300"] = {terrain = TERRAIN_WATER, nodes = NF_RALLY}\n'  # not the hero's terrain
                     'GROUND["360,300"] = {terrain = TERRAIN_LAND, nodes = NF_POWER_1}\n'  # no rally node nearby
                     'GROUND["380,300"] = {terrain = TERRAIN_ICE, nodes = NF_RALLY}\n'  # walkable, not a destination
                     'UNREACHABLE["23,38"] = true')  # (360,600): no route
        self.assertEqual({("home", 1000, 310), ("exit:1", 920, 300), ("exit:2", 900, 600), ("up:2", 380, 600),
                          ("up:1", 340, 300)}, self.moves())
        # A hero that needs no rally node takes (360,300) again; ignore_waypoints skips the route.
        self.lua.run("ENTITIES[70].nav_rally.requires_node_nearby = false; ENTITIES[70].nav_grid.ignore_waypoints = true")
        self.assertEqual({("home", 1000, 310), ("exit:1", 920, 300), ("exit:2", 900, 600), ("up:2", 360, 600),
                          ("up:1", 360, 300)}, self.moves())

    def test_a_teleporting_hero_skips_the_route_only_beyond_its_min_distance(self):
        # No route to (360,600) or to home (1000,310), the hero's own cell: up:2 is a teleport beyond 100 px, home
        # is not (0 px), so home moves to the nearest passing lane node, exit 1's end node (1000,300).
        self.lua.run('UNREACHABLE["23,38"] = true; UNREACHABLE["63,20"] = true\n'
                     "ENTITIES[70].teleport = {min_distance = 100}")
        self.assertEqual(MOVES - {("home", 1000, 310)} | {("home", 1000, 300)}, self.moves())
        self.lua.run("ENTITIES[70].teleport.disabled = true")
        self.assertIn(("up:2", 380, 600), self.moves())
        self.lua.run("ENTITIES[70].teleport = {}")  # no min_distance: the GUI's own comparison fails on every point
        self.assertEqual(set(), self.moves())

    def test_options_are_capped_distinct_and_from_known_paths(self):
        self.lua.run(r"""
for pi = 3, 9 do
    local lane = {}
    for i = 0, 50 do lane[#lane + 1] = {x = 20 * i, y = 100 + 60 * pi} end
    PATH_DB.paths[pi] = {lane}
end
NATIVE_STATE.next_wave.paths = {{path_index = 9}, {path_index = 3}, {path_index = 42}, {path_index = 2}}
""")
        moves = self.moves()
        self.assertEqual(6, len(moves))
        self.assertEqual(6, len({(x, y) for _, x, y in moves}))
        # Next wave paths 2, 3 and 9 (42 is no path) come before the live enemies' path 1; the cap cuts after 2, 3.
        self.assertEqual({"home", "exit:1", "exit:2", "up:2", "up:3", "up:9"}, {option for option, *_ in moves})
        # The same point twice is offered once: a respawn point on exit:1's point keeps home only.
        self.lua.run("ENTITIES[70].hero.respawn_point = {x = 900, y = 300}")
        options = {option: (x, y) for option, x, y in self.moves()}
        self.assertEqual((900, 300), options["home"])
        self.assertNotIn("exit:1", options)

    def test_move_hero_writes_the_gui_click(self):
        self.lua.run("GRID_DB.waypoints_cache.path = {{x = 1, y = 1}}")
        self.catalog()
        reply = self.move("exit:1", 900, 300)
        self.assertEqual({"type": "ok", "action": "move_hero", "hero_id": 70, "option": "exit:1", "x": 900, "y": 300},
                         undump(reply["result"]["native_wire"]))
        hero = self.entity(70)
        self.assertEqual((True, {"x": 900, "y": 300}, {"x": 900, "y": 300}),
                         (hero["nav_rally"]["new"], hero["nav_rally"]["pos"], hero["nav_rally"]["center"]))
        self.assertEqual({1: {"x": 950, "y": 305}, 2: {"x": 900, "y": 300}}, hero["nav_grid"]["waypoints"])
        self.assertEqual([], self.calls())  # never the bridge
        # The game's waypoints_cache now holds the click's route, as after a player's click.
        cache = undump(self.lua.call("waypoints_cache_dump"))
        self.assertEqual(({"x": 63, "y": 20}, {"x": 57, "y": 19}), (cache["from_c"], cache["to_c"]))
        # rally_new until the hero's script takes it; the options now follow the hero's rally point as fallback.
        self.assertTrue(self.hero()["rally_new"])

    def test_teleport_and_ignore_waypoints_writes(self):
        self.lua.run('UNREACHABLE["23,38"] = true; ENTITIES[70].teleport = {min_distance = 100}')
        self.catalog()
        self.assertIs(True, self.move("up:2", 360, 600)["ok"])
        hero = self.entity(70)
        self.assertNotIn("waypoints", hero["nav_grid"])  # the GUI stores find_waypoints' nil
        self.assertEqual({"x": 360, "y": 600}, hero["nav_rally"]["pos"])
        lua = self.world("v4")
        lua.run("ENTITIES[70].nav_grid.ignore_waypoints = true; ENTITIES[70].nav_grid.waypoints = {{x = 7, y = 7}}")
        self.catalog(lua)
        calls = int(lua.call("waypoint_calls"))
        self.assertIs(True, rpc(lua, '{"id":"m","action":"move_hero","hero_id":70,"x":900,"y":600,'
                                     '"option":"exit:2"}')["ok"])
        hero = undump(lua.call("entity_dump", b"70"))
        self.assertEqual({1: {"x": 7, "y": 7}}, hero["nav_grid"]["waypoints"])
        self.assertEqual(calls, int(lua.call("waypoint_calls")))

    def test_asking_for_routes_leaves_the_game_cache_and_is_cached_by_cells(self):
        self.lua.run("GRID_DB.waypoints_cache = {from_c = {x = 1, y = 2}, to_c = {x = 3, y = 4}, path_c = {5}, "
                     "path = {{x = 6, y = 7}}}")
        before = self.lua.call("waypoints_cache_dump")
        self.catalog()
        self.assertEqual(before, self.lua.call("waypoints_cache_dump"))
        asked = int(self.lua.call("waypoint_calls"))
        self.assertEqual(len(MOVES), asked)  # each passing point asked once (the hero ignores no waypoints)
        self.catalog()
        self.lua.run("ENTITIES[70].pos = {x = 1001, y = 311}")  # the same cell
        self.catalog()
        self.assertEqual(asked, int(self.lua.call("waypoint_calls")))
        self.lua.run("ENTITIES[70].pos = {x = 980, y = 311}")  # another cell: asked again
        self.catalog()
        self.assertEqual(2 * asked, int(self.lua.call("waypoint_calls")))
        self.assertEqual(before, self.lua.call("waypoints_cache_dump"))

    def step(self, ticks, lua=None):
        reply = rpc(lua or self.lua, f'{{"id":"t","action":"step","ticks":{ticks}}}')
        self.assertIs(True, reply["ok"], reply)
        self.assertEqual(ticks, reply["result"]["tick_after"] - reply["result"]["tick_before"])
        return reply["result"]["state"]

    def script_hero(self, lua=None):
        """The native tick runs HERO_SCRIPT(tick) when set: a stand-in for the hero's own update script."""
        (lua or self.lua).run("HOST.install(function()\n"
                              "    local store = game and game.store\n"
                              "    if store and store.step then\n"
                              "        store.tick = store.tick + 1\n"
                              "        if HERO_SCRIPT then HERO_SCRIPT(store.tick) end\n"
                              "    end\n"
                              "end)")

    def test_the_last_order_is_exported_with_what_the_game_did_to_it(self):
        self.catalog()
        self.script_hero()
        self.assertNotIn("order", self.hero())  # never ordered
        # The hero walks (its pos changes, its rally vector stays) and, as the death script at a respawn, moves
        # its rally vector in place: neither replaces the order.
        self.assertIs(True, self.move("exit:1", 900, 300)["ok"])
        self.lua.run("HERO_SCRIPT = function(tick)\n"
                     "    local e = ENTITIES[70]\n"
                     "    e.pos = {x = e.pos.x - 10, y = e.pos.y}\n"
                     "    if tick == 103 then e.nav_rally.pos.x, e.nav_rally.pos.y = 1000, 310 end\n"
                     "end")
        hero = as_list(self.step(5)["heroes"])[0]
        self.assertEqual({"x": 900, "y": 300, "tick": 100}, hero["order"])
        self.assertEqual({"x": 1000, "y": 310}, self.entity(70)["nav_rally"]["pos"])

    def test_an_order_replaced_by_the_heros_own_position_is_recorded_once(self):
        # Ignus' Surge of Flame on arrival (game_scripts L2808-2809): center and pos become clones of its position.
        self.catalog()
        self.script_hero()
        self.assertIs(True, self.move("up:2", 360, 600)["ok"])
        self.lua.run("HERO_SCRIPT = function(tick)\n"
                     "    local e = ENTITIES[70]\n"
                     "    if tick <= 104 then e.pos = {x = e.pos.x - 20.25, y = e.pos.y + 0.5} end\n"
                     "    if tick == 104 then\n"
                     "        e.nav_rally.center = {x = e.pos.x, y = e.pos.y}\n"
                     "        e.nav_rally.pos = {x = e.pos.x, y = e.pos.y}\n"
                     "    end\n"
                     "    if tick == 106 then e.nav_rally.pos = {x = 1, y = 2} end  -- a later change: not recorded\n"
                     "end")
        state = self.step(3)
        self.assertEqual({"x": 360, "y": 600, "tick": 100}, as_list(state["heroes"])[0]["order"])
        hero = as_list(self.step(4)["heroes"])[0]
        self.assertEqual({"x": 360, "y": 600, "tick": 100,
                          "replaced": {"tick": 104, "x": 919, "y": 312, "at_hero": True}}, hero["order"])
        self.assertEqual({"x": 1, "y": 2}, self.entity(70)["nav_rally"]["pos"])
        # A new order starts a new record.
        self.lua.run("HERO_SCRIPT = nil")
        self.catalog()
        self.assertIs(True, self.move("exit:2", 900, 600)["ok"])
        self.assertEqual({"x": 900, "y": 600, "tick": 107}, self.hero()["order"])

    def test_a_rally_vector_that_is_not_the_heros_position_is_recorded_as_such(self):
        cases = {
            "another point": "e.nav_rally.pos = {x = 5, y = 6}; e.nav_rally.center = {x = 5, y = 6}",
            "pos only": "e.nav_rally.pos = {x = e.pos.x, y = e.pos.y}",
            "center elsewhere": "e.nav_rally.pos = {x = e.pos.x, y = e.pos.y}; e.nav_rally.center = {x = 0, y = 0}",
            "no pos": "e.nav_rally.pos = nil",
        }
        for name, change in cases.items():
            with self.subTest(case=name):
                self.lua = self.world("v4")
                self.catalog()
                self.script_hero()
                self.assertIs(True, self.move("exit:1", 900, 300)["ok"])
                self.lua.run("HERO_SCRIPT = function(tick)\n"
                             "    local e = ENTITIES[70]\n"
                             f"    if tick == 101 then {change} end\n"
                             "end")
                replaced = as_list(self.step(2)["heroes"])[0]["order"]["replaced"]
                self.assertEqual((101, False), (replaced["tick"], replaced["at_hero"]))

    def test_only_v4_watches_orders(self):
        for scope in ("v2", "v3"):
            with self.subTest(scope=scope):
                lua = self.world(scope)
                self.catalog(lua)
                self.script_hero(lua)
                lua.run("HERO_SCRIPT = function(tick) ENTITIES[70].nav_rally.pos = {x = 1, y = 1} end")
                self.assertNotIn("order", as_list(self.step(3, lua)["heroes"])[0])

    def test_off_catalog_moves_are_refused_without_side_effects(self):
        self.catalog()
        self.assertRefusedWithoutSideEffects([
            '{"id":"m","action":"move_hero","hero_id":70,"x":901,"y":300,"option":"exit:1"}',
            '{"id":"m","action":"move_hero","hero_id":70,"x":900,"y":300,"option":"exit:2"}',
            '{"id":"m","action":"move_hero","hero_id":70,"x":900,"y":300}',
            '{"id":"m","action":"move_hero","x":900,"y":300,"option":"exit:1"}',
            '{"id":"m","action":"move_hero","hero_id":71,"x":900,"y":300,"option":"exit:1"}',
            '{"id":"m","action":"move_hero","hero_id":70,"x":500,"y":300,"option":"home"}',
            '{"id":"m","action":"move_hero","hero_id":70,"x":900,"y":300,"option":"exit:1","target_id":70}',
            '{"id":"m","action":"move_hero","hero_id":"70","x":900,"y":300,"option":"exit:1"}',
        ])

    def test_no_scope_reaches_the_bridge_move_hero(self):
        for scope in ("<unset>", "v1", "v2", "v3"):
            with self.subTest(scope=scope):
                lua = self.world(scope)
                if scope in ("v2", "v3"):
                    self.assertEqual(set(), {item["action"] for item in self.catalog(lua)} & {"move_hero"})
                reply = rpc(lua, '{"id":"m","action":"move_hero","hero_id":70,"x":900,"y":300,"option":"exit:1"}')
                self.assertEqual((False, NOT_ENABLED), (reply["ok"], reply["error"]))
                self.assertEqual([], as_list(undump(lua.call("calls_dump"))))
                self.assertEqual({"x": 1000, "y": 310}, undump(lua.call("entity_dump", b"70"))["nav_rally"]["pos"])


if __name__ == "__main__":
    unittest.main()
