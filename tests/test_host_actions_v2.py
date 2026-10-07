"""host.lua action scope v2 (catalog, state, execution) in the game's own LuaJIT; no game, no sockets."""
from __future__ import annotations

import hashlib
import os
import unittest

try:
    from test_host_protocol import HOST, LUA_DLL, Lua, undump
except ImportError:  # run as tests.test_host_actions_v2 from the workspace root
    from tests.test_host_protocol import HOST, LUA_DLL, Lua, undump

# Stub socket, the native bridge, entity_db and LÖVE; load host.lua with a chosen
# ALPHARUSH_ACTION_SCOPE ("<unset>" = absent). Values are flattened to sorted
# "path=type:value" lines exactly as in test_host_protocol, so undump() applies.
SETUP = r"""
local host_path, scope = ...
print = function() end
local ENV = {ALPHARUSH_SEED = "1001", ALPHARUSH_PORT = "9879"}
if scope ~= "<unset>" then ENV.ALPHARUSH_ACTION_SCOPE = scope end
os.getenv = function(name) return ENV[name] end
local function hex(s) return (s:gsub(".", function(c) return string.format("%02x", c:byte()) end)) end
local function flatten(value, prefix, out)
    local t = type(value)
    if t == "table" then
        local any = false
        for k, v in pairs(value) do
            any = true
            flatten(v, prefix .. "/" .. (type(k) == "number" and ("#" .. tostring(k)) or hex(tostring(k))), out)
        end
        if not any then out[#out + 1] = prefix .. "=table:" end
    elseif t == "string" then out[#out + 1] = prefix .. "=string:" .. hex(value)
    elseif t == "number" then out[#out + 1] = prefix .. "=number:" .. string.format("%.17g", value)
    elseif t == "boolean" then out[#out + 1] = prefix .. "=boolean:" .. tostring(value)
    else out[#out + 1] = prefix .. "=" .. t .. ":" end  -- never an address
    return out
end
function dump(value)
    local out = flatten(value, "", {})
    table.sort(out)
    return table.concat(out, "\n")
end
local function copy(value)
    if type(value) ~= "table" then return value end
    local out = {}
    for k, v in pairs(value) do out[k] = copy(v) end
    return out
end
ENCODED, CALLS, INBOX, SENDS = {}, {}, {}, 0
local connected = false
local client = {
    settimeout = function() end, close = function() connected = false end,
    send = function(_, payload) SENDS = SENDS + 1; return #payload end,
    receive = function()
        local line = table.remove(INBOX, 1)
        if line then return line end
        return nil, "timeout", ""
    end,
}
local server = {settimeout = function() end,
                accept = function() if connected then return nil end; connected = true; return client end}
package.preload["socket"] = function() return {bind = function() return server end} end
package.preload["alpha_native_bridge"] = function()
    return {
        codec = {encode = function(value) ENCODED[#ENCODED + 1] = copy(value); return dump(value) end},
        collect_state = function() return copy(NATIVE_STATE) end,
        command_json = function(cmd) CALLS[#CALLS + 1] = copy(cmd); return "bridge-wire-" .. #CALLS end,
    }
end
package.preload["entity_db"] = function()
    return {get_template = function(self, name)
        if name == "tower_raises" then error("entity_db: no template " .. name) end
        return TEMPLATES[name]
    end}
end
love = {filesystem = {getSaveDirectory = function() return "C:/saves/alpharush_unit" end}}
HOST = dofile(host_path)
local function native_update()
    local store = game and game.store
    if store and store.step then store.tick = store.tick + 1 end
end
function install()
    local ok, err = pcall(HOST.install, native_update)
    return (ok and "OK|" or "ERR|") .. tostring(err) .. "|" .. tostring(HOST.installed)
end
function rpc(line)
    if not HOST.installed then HOST.install(native_update) end
    INBOX[#INBOX + 1] = line
    local before = SENDS
    HOST.update(1 / 60)
    if SENDS ~= before + 1 then return "NOREPLY" end
    local reply = ENCODED[#ENCODED]
    -- Error positions follow host.lua line numbers, which any edit shifts; compare the message.
    if type(reply.error) == "string" then reply.error = (reply.error:gsub("^.-%.lua:%d+: ", "")) end
    return dump(reply)
end
function calls_dump() return dump(CALLS) end
function game_dump() return dump(game) end
function entity_dump(id) return dump(game.store.entities[tonumber(id)]) end
"""

# A subset of the real data.tower_menus_data, entity_db templates and a native level:
# standard/blocked/special/pending holders, every standard tower family, special towers,
# locks, an exact-gold power price, and enemies for power anchors.
WORLD = r"""
local function up(arg) return {action = "tw_upgrade", action_arg = arg} end
local function power(name) return {action = "upgrade_power", action_arg = name} end
local SELL, RALLY = {action = "tw_sell"}, {action = "tw_rally"}
package.loaded["data.tower_menus_data"] = {
    holder = {{up("tower_build_archer"), up("tower_build_barrack"), up("tower_build_mage"), up("tower_build_engineer")}},
    archer = {{up("tower_archer_2"), SELL}, {up("tower_archer_3"), SELL},
              {up("tower_ranger"), up("tower_musketeer"), SELL}},
    barrack = {{up("tower_barrack_2"), RALLY, SELL}, {up("tower_barrack_3"), RALLY, SELL},
               {up("tower_paladin"), up("tower_barbarian"), RALLY, SELL}},
    mage = {{up("tower_mage_2"), SELL}, {up("tower_mage_3"), SELL},
            {up("tower_arcane_wizard"), up("tower_sorcerer"), SELL}},
    engineer = {{up("tower_engineer_2"), SELL}, {up("tower_engineer_3"), SELL},
                {up("tower_bfg"), up("tower_tesla"), SELL}},
    ranger = {{power("poison"), power("thorn"), SELL}},
    musketeer = {{power("sniper"), power("shrapnel"), SELL}},
    paladin = {{power("healing"), power("shield"), power("holystrike"), RALLY, SELL}},
    barbarian = {{power("dual_axes"), power("twister"), power("throwing_axes"), RALLY, SELL}},
    arcane_wizard = {{power("disintegrate"), power("teleport"), SELL}},
    sorcerer = {{power("polymorph"), power("elemental"), SELL}},
    bfg = {{power("missile"), power("cluster"), SELL}},
    tesla = {{power("bolt"), power("overcharge"), SELL}},
    holder_elf = {{up("tower_elf")}},
    elf = {{{action = "tw_buy_soldier"}}},
    holder_blocked_jungle = {{{action = "tw_unblock"}}},
    holder_sasquash = {{{action = "tw_none"}}},
    sunray = {{power("ray"), {action = "tw_point"}}},
}
local function priced(price) return {tower = {price = price}} end
TEMPLATES = {
    -- Build placeholders carry their own price, which the GUI ignores in favour of build_name's.
    tower_build_archer = {build_name = "tower_archer_1", tower = {price = 1}},
    tower_build_barrack = {build_name = "tower_barrack_1", tower = {price = 1}},
    tower_build_mage = {build_name = "tower_mage_1", tower = {price = 1}},
    tower_build_engineer = {build_name = "tower_engineer_1", tower = {price = 1}},
    tower_archer_1 = priced(70), tower_archer_2 = priced(110), tower_archer_3 = priced(160),
    tower_ranger = priced(230), tower_musketeer = priced(230),
    tower_barrack_1 = priced(70), tower_barrack_2 = priced(110), tower_barrack_3 = priced(160),
    tower_paladin = priced(230), tower_barbarian = priced(240),
    tower_mage_1 = priced(100), tower_mage_2 = priced(160), tower_mage_3 = priced(240),
    tower_arcane_wizard = priced(300), tower_sorcerer = priced(300),
    tower_engineer_1 = priced(125), tower_engineer_2 = priced(220), tower_engineer_3 = priced(320),
    tower_bfg = priced(400), tower_tesla = priced(375), tower_elf = priced(100),
}
local function holder(kind, extra)
    local e = {template_name = "tower_holder_grass", tower_holder = {}, tower = {type = kind}, ui = {nav_mesh_id = "1"}}
    for k, v in pairs(extra or {}) do e.tower[k] = v end
    return e
end
local function tower(kind, level, template, extra)
    local e = {template_name = template, tower = {type = kind, level = level, can_be_sold = true, can_be_mod = true}}
    for k, v in pairs(extra or {}) do e.tower[k] = v end
    return e
end
local blocked = holder("holder")
blocked.tower_holder.blocked = true
local jungle = holder("holder_blocked_jungle")
jungle.tower_holder.blocked = true
local ranger = tower("ranger", nil, "tower_ranger")
ranger.powers = {poison = {level = 1, max_level = 3, price_base = 150, price_inc = 50},
                 thorn = {level = 3, max_level = 3, price_base = 10, price_inc = 10}}
local paladin = tower("paladin", 1, "tower_paladin", {can_be_sold = false})
paladin.powers = {healing = {level = 0, max_level = 3, price_base = 235, price_inc = 10},
                  shield = {level = 2, max_level = 3, price_base = 100, price_inc = 70},
                  holystrike = {level = 0, max_level = 3}}
local bfg = tower("bfg", 1, "tower_bfg")
bfg.powers = {missile = {level = 0, max_level = 3, price_base = 250, price_inc = 0}}
local sunray = tower("sunray", 1, "tower_sunray", {can_be_sold = false, can_be_mod = false})
sunray.powers = {ray = {level = 0, max_level = 4, price_base = 10, price_inc = 0}}
ENTITIES = {
    [10] = holder("holder"), [11] = holder("holder"), [12] = blocked,
    [13] = holder("holder_elf"), [14] = holder("holder", {upgrade_to = "tower_build_mage"}),
    [15] = holder("holder_sasquash"), [16] = jungle,
    [20] = tower("archer", 1, "tower_archer_1"), [21] = tower("archer", 3, "tower_archer_3"),
    [22] = tower("barrack", 3, "tower_barrack_3"), [23] = ranger, [24] = paladin,
    [25] = tower("archer", 1, "tower_archer_1", {upgrade_to = "tower_archer_2"}),
    [26] = {template_name = "tower_build_mage", tower = {type = "build_animation"}},
    [27] = tower("elf", 1, "tower_elf"), [28] = sunray,
    [29] = tower("archer", 2, "tower_archer_2", {can_be_mod = false}),
    [30] = tower("mage", 3, "tower_mage_3"), [31] = bfg, [32] = tower("engineer", 1, "tower_engineer_1"),
    [33] = tower("musketeer", 1, "tower_musketeer"),
    [40] = {template_name = "enemy_goblin", enemy = {}, health = {hp = 10}},
}
game = {
    store = {tick = 100, tick_ts = 3.25, tick_length = 1 / 30, player_gold = 235, lives = 20,
             wave_group_number = 0, level = {locked_towers = {"tower_musketeer", "tower_engineer_1"}},
             entities = ENTITIES},
    game_gui = {power_1 = {mode = "ready"}, power_2 = {mode = "cooldown"}},
}
NATIVE_STATE = {
    type = "game_state", gold = 235, lives = 20, wave = 0, wave_total = 6, tick = 100,
    next_wave = {wave = 1}, wave_spawning = false, enemy_count = 6,
    locked_towers = {"tower_musketeer", "tower_engineer_1"},
    towers = {{id = 20, template = "tower_archer_1", holder_id = "1"}},
    holders = {{id = 10, mesh_id = "1"}},
    enemies = {
        {id = 101, x = 100.4, y = 200.6, path_progress = 0.9},
        {id = 103, x = 300.5, y = 199.5, path_progress = 0.8},
        {id = 102, x = 130, y = 210, path_progress = 0.8},  -- ties 103, lower id first, but near 101
        {id = 104, x = 500, y = 100, path_progress = 0.5},
        {id = 105, x = 700, y = 100, path_progress = 0.4},  -- spaced, but over the 3-anchor limit
        {id = 106, x = 900, y = 100},                       -- no path progress: never an anchor
    },
}
"""

# The game's placement modules and constants, as the GUI asks them before casting a spell
# (game_gui GUI_MODE_POWER_1/2). Constant values are stand-in distinct bits. GROUND["x,y"]
# configures one (rounded anchor) cell: {terrain = TERRAIN_* bits, nodes = NF_* bits of the
# path nodes nearby}, or "error" for a cell the game modules fail on. Unconfigured cells are
# land with both node kinds nearby, so every anchor is a legal point unless a test says otherwise.
# Only package.loaded and globals change, so the v1 transcript (game, bridge calls) is unaffected.
GROUND = r"""
assert(type(bit) == "table" and bit.bor and bit.band and bit.bnot, "LuaJIT bit library")
TERRAIN_LAND, TERRAIN_WATER, TERRAIN_CLIFF, TERRAIN_ICE, TERRAIN_FAERIE = 1, 2, 4, 8, 16
NF_RALLY, NF_POWER_1 = 2, 8
GROUND, GROUND_CALLS = {}, {}
local function cell(x, y)
    local c = GROUND[tostring(x) .. "," .. tostring(y)]
    if c == "error" then error("grid_db: no cell at " .. tostring(x) .. "," .. tostring(y)) end
    return c or {terrain = TERRAIN_LAND, nodes = bit.bor(NF_RALLY, NF_POWER_1)}
end
local function record(name, x, y, flags, range)
    GROUND_CALLS[#GROUND_CALLS + 1] = {name = name, x = x, y = y, flags = flags, range = range}
end
PATH_DB = {valid_node_nearby = function(self, x, y, range, flags)
    assert(self == PATH_DB, "path_db method called without self")
    record("valid_node_nearby", x, y, flags, range == nil and "nil" or range)
    return bit.band(cell(x, y).nodes, flags) ~= 0
end}
GRID_DB = {
    cell_is = function(self, x, y, flags)
        assert(self == GRID_DB, "grid_db method called without self")
        record("cell_is", x, y, flags)
        return bit.band(cell(x, y).terrain, flags) ~= 0
    end,
    cell_is_only = function(self, x, y, flags)
        assert(self == GRID_DB, "grid_db method called without self")
        record("cell_is_only", x, y, flags)
        local terrain = cell(x, y).terrain
        return terrain ~= 0 and bit.band(terrain, bit.bnot(flags)) == 0
    end,
}
package.loaded["path_db"], package.loaded["grid_db"] = PATH_DB, GRID_DB
function ground_calls_dump() return dump(GROUND_CALLS) end
"""

# v1 transcript: requests on the rich world, then a poor mid-wave world.
V1_RICH_REQUESTS = [
    '{"id":"1","action":"state"}',
    '{"id":"2","action":"build_tower","holder_id":10,"tower_type":"archer"}',
    '{"id":"3","action":"build_tower","holder_id":13,"tower_type":"mage"}',
    '{"id":"4","action":"build_tower","holder_id":12,"tower_type":"archer"}',
    '{"id":"5","action":"build_tower","holder_id":11,"tower_type":"barrack","target":"tower_build_mage"}',
    '{"id":"6","action":"build_tower","holder_id":11,"tower_type":"elf"}',
    '{"id":"7","action":"build_tower","holder_id":10,"tower_type":"archer","tower_id":5}',
    '{"id":"8","action":"build_tower","holder_id":14,"tower_type":"engineer"}',
    '{"id":"9","action":"send_wave"}',
    '{"id":"10","action":"upgrade_tower","tower_id":20,"target":"tower_archer_2"}',
    '{"id":"11","action":"upgrade_power","tower_id":23,"power":"poison"}',
    '{"id":"12","action":"sell_tower","tower_id":20}',
    '{"id":"13","action":"use_power","power":1,"x":100,"y":201,"anchor_id":101}',
    '{"id":"14","action":"move_hero","x":1,"y":2}',
    '{"id":"15","action":"step","ticks":1}',
    '{"id":"16","action":"pause"}',
]
V1_POOR_WORLD = "game.store.player_gold = 90; NATIVE_STATE.wave = 2; NATIVE_STATE.enemy_count = 3"
V1_POOR_REQUESTS = [
    '{"id":"17","action":"state"}',
    '{"id":"18","action":"build_tower","holder_id":10,"tower_type":"mage"}',
    '{"id":"19","action":"build_tower","holder_id":15,"tower_type":"barrack"}',
    '{"id":"20","action":"send_wave"}',
]
# Recorded from host.lua (sha256 2630f37d...) before action scope v2 existed: sha256 of
# v1_transcript(), with ALPHARUSH_ACTION_SCOPE unset and "v1" alike, and its hello reply.
V1_BASELINE_SHA256 = "03ee4ea5a7c53a0bde6ed2040be1527751b0b4cd17824a0379b070aeae3ebf18"
V1_BASELINE_HELLO = {"version": "alpharush-rl-v1", "save_directory": "C:/saves/alpharush_unit",
                     "seed": 1001, "port": 9879, "rng_mode": ""}
# The same baseline, readable: v1 offers every basic tower on every unblocked tower_holder
# (special and pending holders included) and ignores locks; send_wave when wave_ready.
V1_RICH_CATALOG = [("build_tower", holder, kind, cost) for holder in (10, 11, 13, 14, 15)
                   for kind, cost in (("archer", 70), ("barrack", 70), ("engineer", 125), ("mage", 100))]
V1_RICH_CATALOG.append(("send_wave", None, None, 0))
V1_POOR_CATALOG = [("build_tower", holder, kind, 70) for holder in (10, 11, 13, 14, 15)
                   for kind in ("archer", "barrack")]
OUTSIDE = "Action is outside the native legal catalog"
NOT_ENABLED = "Action not enabled in verified experiment scope"
V1_REJECTED = {"4": OUTSIDE, "6": OUTSIDE, "7": OUTSIDE, "10": NOT_ENABLED, "11": NOT_ENABLED,
               "12": NOT_ENABLED, "13": NOT_ENABLED, "14": NOT_ENABLED, "18": OUTSIDE, "20": OUTSIDE}

# Expected v2 catalog of WORLD (gold 235; tower_musketeer and tower_engineer_1 locked;
# power 1 ready, power 2 cooling down; every anchor on legal ground). Every entry also
# carries available/legal = true. Power prices are the GUI's: price_base for the first
# level, price_inc for each later one (ranger poison at level 1: 50; paladin shield at
# level 2: 70; bfg missile at level 0: 250 > 235).
V2_WORLD_CATALOG = [
    {"action": "build_tower", "holder_id": 10, "tower_type": "archer", "target": "tower_build_archer", "cost": 70},
    {"action": "build_tower", "holder_id": 10, "tower_type": "barrack", "target": "tower_build_barrack", "cost": 70},
    {"action": "build_tower", "holder_id": 10, "tower_type": "mage", "target": "tower_build_mage", "cost": 100},
    {"action": "build_tower", "holder_id": 11, "tower_type": "archer", "target": "tower_build_archer", "cost": 70},
    {"action": "build_tower", "holder_id": 11, "tower_type": "barrack", "target": "tower_build_barrack", "cost": 70},
    {"action": "build_tower", "holder_id": 11, "tower_type": "mage", "target": "tower_build_mage", "cost": 100},
    *({"action": "sell_tower", "tower_id": tower_id, "cost": 0} for tower_id in (20, 21, 22, 23, 30, 31, 32, 33)),
    {"action": "send_wave", "cost": 0},
    {"action": "upgrade_power", "tower_id": 23, "power": "poison", "cost": 50},
    {"action": "upgrade_power", "tower_id": 24, "power": "healing", "cost": 235},
    {"action": "upgrade_power", "tower_id": 24, "power": "holystrike", "cost": 0},
    {"action": "upgrade_power", "tower_id": 24, "power": "shield", "cost": 70},
    # The sunray level tower's menu sells only its beam power; nothing else of it is offered.
    {"action": "upgrade_power", "tower_id": 28, "power": "ray", "cost": 10},
    {"action": "upgrade_tower", "tower_id": 20, "target": "tower_archer_2", "cost": 110},
    {"action": "upgrade_tower", "tower_id": 21, "target": "tower_ranger", "cost": 230},
    {"action": "upgrade_tower", "tower_id": 22, "target": "tower_paladin", "cost": 230},
    {"action": "upgrade_tower", "tower_id": 32, "target": "tower_engineer_2", "cost": 220},
    {"action": "use_power", "power": 1, "x": 100, "y": 201, "anchor_id": 101, "cost": 0},
    {"action": "use_power", "power": 1, "x": 301, "y": 200, "anchor_id": 103, "cost": 0},
    {"action": "use_power", "power": 1, "x": 500, "y": 100, "anchor_id": 104, "cost": 0},
]


def load(scope="<unset>"):
    lua = Lua()
    lua.run(SETUP, str(HOST).encode("utf-8"), scope.encode("utf-8"))
    lua.run(WORLD)
    lua.run(GROUND)
    return lua


def v1_transcript(lua):
    out = []
    for line in V1_RICH_REQUESTS:
        out.append(line + "\n" + lua.call("rpc", line.encode()))
    lua.run(V1_POOR_WORLD)
    for line in V1_POOR_REQUESTS:
        out.append(line + "\n" + lua.call("rpc", line.encode()))
    out.append(lua.call("calls_dump"))
    out.append(lua.call("game_dump"))
    return out


def transcript_sha256(parts):
    return hashlib.sha256("\n\0\n".join(parts).encode("utf-8")).hexdigest()


def as_list(table):
    """Lua arrays undump to {1: ..., 2: ...}; an empty table undumps to {}."""
    return [table[index] for index in sorted(table)] if table else []


def rpc(lua, line):
    return undump(lua.call("rpc", line.encode("utf-8") if isinstance(line, str) else line))


def native_state(lua):
    reply = rpc(lua, '{"id":"s","action":"state"}')
    if reply.get("ok") is not True:
        raise AssertionError(reply)
    return reply["result"]


def catalog_of(lua):
    """The v2 catalog without its constant available/legal flags (which must be true)."""
    items = as_list(native_state(lua).get("action_catalog", {}))
    for item in items:
        if item.pop("available") is not True or item.pop("legal") is not True:
            raise AssertionError(f"catalog entry is not available and legal: {item}")
    return items


def sort_key(item):
    power = item.get("power")
    return (item["action"], item.get("holder_id", item.get("tower_id", 0)),
            item.get("tower_type") or item.get("target") or (power if isinstance(power, str) else ""),
            power if isinstance(power, float) else 0,
            item.get("x", 0), item.get("y", 0), item.get("anchor_id", 0))


@unittest.skipUnless(os.name == "nt" and LUA_DLL.exists(), "game LuaJIT runtime/rl-engine/lua51.dll unavailable")
class HostLuaCase(unittest.TestCase):
    scope = "v2"

    def setUp(self):
        self.lua = load(self.scope)
        self.addCleanup(self.lua.close)

    def fresh(self, scope=None):
        lua = load(scope or self.scope)
        self.addCleanup(lua.close)
        return lua

    def rpc(self, line):
        return rpc(self.lua, line)

    def catalog(self, lua=None):
        return catalog_of(lua or self.lua)

    def calls(self):
        return as_list(undump(self.lua.call("calls_dump")))

    def entity(self, entity_id):
        return undump(self.lua.call("entity_dump", str(entity_id).encode()))

    def gold(self):
        return undump(self.lua.call("game_dump"))["store"]["player_gold"]

    def spells(self, lua=None):
        return {(item["power"], item["anchor_id"]) for item in self.catalog(lua) if item["action"] == "use_power"}


class HostV1UnchangedTests(HostLuaCase):
    scope = "v1"

    def test_v1_transcript_is_identical_to_the_pre_v2_host(self):
        for scope in ("<unset>", "v1"):
            with self.subTest(scope=scope):
                parts = v1_transcript(self.fresh(scope))
                self.assertEqual(transcript_sha256(parts), V1_BASELINE_SHA256)
                replies = {}
                for part in parts[:-2]:
                    reply = undump(part.partition("\n")[2])
                    replies[reply["id"]] = reply
                self.assertEqual(len(replies), len(V1_RICH_REQUESTS) + len(V1_POOR_REQUESTS))
                for request_id, reply in replies.items():
                    if request_id in V1_REJECTED:
                        self.assertEqual((reply["ok"], reply["error"]), (False, V1_REJECTED[request_id]))
                    else:
                        self.assertIs(reply["ok"], True, request_id)

                def compact(state):
                    return [(item["action"], item.get("holder_id"), item.get("tower_type"), item["cost"])
                            for item in as_list(state["action_catalog"])]

                states = [replies["1"]["result"], replies["15"]["result"]["state"], replies["16"]["result"]]
                for state in states:
                    self.assertEqual(compact(state), V1_RICH_CATALOG)
                self.assertEqual(compact(replies["17"]["result"]), V1_POOR_CATALOG)
                for state in states + [replies["17"]["result"]]:
                    self.assertNotIn("powers_ui", state)
                    self.assertNotIn("action_scope", state)
                # v1 hands the whole command (id and unchecked build target included) to the bridge.
                calls = as_list(undump(parts[-2]))
                self.assertEqual([(c["id"], c["action"], c.get("holder_id")) for c in calls],
                                 [("2", "build_tower", 10), ("3", "build_tower", 13), ("5", "build_tower", 11),
                                  ("8", "build_tower", 14), ("9", "send_wave", None), ("19", "build_tower", 15)])
                self.assertEqual(calls[2]["target"], "tower_build_mage")

    def test_hello_only_gains_the_action_scope(self):
        for scope, expected in (("<unset>", "v1"), ("v1", "v1"), ("v2", "v2")):
            with self.subTest(scope=scope):
                reply = rpc(self.fresh(scope), '{"id":"h","action":"hello"}')
                self.assertEqual(reply["result"], {**V1_BASELINE_HELLO, "action_scope": expected, "headless": False})


class HostV2CatalogTests(HostLuaCase):
    def test_world_catalog_follows_the_game_menus(self):
        self.assertEqual(self.catalog(), V2_WORLD_CATALOG)
        state = native_state(self.lua)
        self.assertEqual(state["action_scope"], "v2")
        self.assertEqual(as_list(state["powers_ui"]), [{"id": 1, "mode": "ready"}, {"id": 2, "mode": "cooldown"}])
        self.assertIs(state["wave_ready"], True)

    def test_special_blocked_and_pending_entities_are_never_offered(self):
        def ids():
            return {item.get("holder_id", item.get("tower_id")) for item in self.catalog()}
        # 12/16 blocked, 13 elf / 15 sasquash holders, 14 and 25 pending upgrades,
        # 26 build animation, 27 elf tower, 29 not modifiable.
        self.assertFalse(ids() & {12, 13, 14, 15, 16, 25, 26, 27, 29})
        # 28, the sunray level tower, is offered its beam power and nothing else (no upgrade or sale).
        self.assertEqual([{"action": "upgrade_power", "tower_id": 28, "power": "ray", "cost": 10}],
                         [item for item in self.catalog() if item.get("tower_id") == 28])
        self.lua.run('ENTITIES[10].tower.upgrade_to = "tower_build_archer"; ENTITIES[20].tower.sell = true')
        self.assertFalse(ids() & {10, 20})
        # A tower level the game has no menu for offers nothing.
        self.lua.run('ENTITIES[21].tower.level = 4')
        self.assertNotIn(21, ids())
        self.assertIn(11, ids())

    def test_locks_from_the_state_copy_or_the_store_level(self):
        locks = '{"tower_build_mage", "tower_archer_1", "tower_archer_2", "tower_paladin"}'
        for source in ("NATIVE_STATE.locked_towers", "game.store.level.locked_towers"):
            with self.subTest(source=source):
                lua = self.fresh()
                lua.run(f"NATIVE_STATE.locked_towers = {{}}; game.store.level.locked_towers = {{}}; {source} = {locks}")
                catalog = self.catalog(lua)
                # Build placeholder or its level-1 tower locked: no build of that kind.
                self.assertEqual({(item["holder_id"], item["tower_type"], item["cost"])
                                  for item in catalog if item["action"] == "build_tower"},
                                 {(holder, kind, cost) for holder in (10, 11)
                                  for kind, cost in (("barrack", 70), ("engineer", 125))})
                self.assertEqual({(item["tower_id"], item["target"])
                                  for item in catalog if item["action"] == "upgrade_tower"},
                                 {(21, "tower_ranger"), (21, "tower_musketeer"), (32, "tower_engineer_2")})

    def test_prices_follow_build_name_and_gold(self):
        def priced(action):
            return {(item.get("holder_id", item.get("tower_id")), item.get("tower_type") or item.get("target")
                     or item.get("power")): item["cost"] for item in self.catalog() if item["action"] == action}
        # Without build_name a placeholder charges its own price; a dangling or failing
        # template drops the entry instead of guessing.
        self.lua.run('TEMPLATES.tower_build_barrack = {tower = {price = 60}}; TEMPLATES.tower_mage_1.tower.price = 236')
        self.lua.run('TEMPLATES.tower_build_archer = {build_name = "tower_missing", tower = {price = 1}}')
        self.lua.run('TEMPLATES.tower_archer_2 = nil')
        self.lua.run('table.insert(package.loaded["data.tower_menus_data"].engineer[1], 1, '
                     '{action = "tw_upgrade", action_arg = "tower_raises"})')
        self.assertEqual(priced("build_tower"), {(10, "barrack"): 60, (11, "barrack"): 60})
        self.assertEqual(priced("upgrade_tower"), {(21, "tower_ranger"): 230, (22, "tower_paladin"): 230,
                                                   (32, "tower_engineer_2"): 220})
        # Gold boundary: a price equal to the gold is affordable, one above is not.
        self.lua.run("game.store.player_gold = 230")
        self.assertEqual(priced("upgrade_power"), {(23, "poison"): 50, (24, "holystrike"): 0, (24, "shield"): 70,
                                                   (28, "ray"): 10})
        self.assertEqual(set(priced("upgrade_tower")), {(21, "tower_ranger"), (22, "tower_paladin"),
                                                        (32, "tower_engineer_2")})
        self.lua.run("game.store.player_gold = 59")
        self.assertEqual(priced("build_tower"), {})
        self.assertEqual(priced("upgrade_tower"), {})
        self.assertEqual(priced("upgrade_power"), {(23, "poison"): 50, (24, "holystrike"): 0, (28, "ray"): 10})
        self.lua.run("game.store.player_gold = 49")
        self.assertEqual(priced("upgrade_power"), {(24, "holystrike"): 0, (28, "ray"): 10})
        self.lua.run("game.store.player_gold = 9")
        self.assertEqual(priced("upgrade_power"), {(24, "holystrike"): 0})

    def test_power_levels_and_gui_price(self):
        def powers():
            return {(item["tower_id"], item["power"]): item["cost"]
                    for item in self.catalog() if item["action"] == "upgrade_power"}
        # The GUI charges price_base for the first level and price_inc for every later
        # level, never price_base + price_inc * level (the bridge's old formula).
        self.lua.run("ENTITIES[23].powers.poison.level = 2")  # still price_inc 50, not 150 + 50 * 2
        self.assertEqual(powers()[(23, "poison")], 50)
        self.lua.run("ENTITIES[23].powers.poison.price_inc = 236")
        self.assertNotIn((23, "poison"), powers())
        self.lua.run("game.store.player_gold = 236")
        self.assertEqual(powers()[(23, "poison")], 236)
        self.lua.run("ENTITIES[23].powers.poison.level = 3")  # at max_level
        self.assertNotIn((23, "poison"), powers())
        self.lua.run("ENTITIES[24].powers.healing.level = 1")  # past the first level: price_inc 10
        self.assertEqual(powers()[(24, "healing")], 10)
        self.lua.run("ENTITIES[31].powers.missile.level = 1")  # price_inc 0 after a 250 first level
        self.assertEqual(powers()[(31, "missile")], 0)
        # Malformed levels and prices drop the entry instead of guessing.
        self.lua.run('ENTITIES[24].powers.holystrike.level = "0"; ENTITIES[24].powers.healing = nil\n'
                     "ENTITIES[24].powers.shield.price_inc = -5; ENTITIES[31].powers.missile.price_inc = 0 / 0\n"
                     "ENTITIES[28].powers.ray.level = 4")  # the sunray beam at its max level
        self.assertEqual(powers(), {})

    def test_power_upgrades_walk_the_gui_prices_of_real_templates(self):
        # Real template prices: arcane wizard disintegrate 350/200, ranger thorn 300/150.
        self.lua.run(r"""
game.store.player_gold = 2000
ENTITIES[34] = {template_name = "tower_arcane_wizard",
                tower = {type = "arcane_wizard", level = 1, can_be_sold = true, can_be_mod = true, spent = 300},
                powers = {disintegrate = {level = 0, max_level = 3, price_base = 350, price_inc = 200}}}
ENTITIES[23].powers.thorn = {level = 0, max_level = 3, price_base = 300, price_inc = 150}
""")
        gold = 2000
        for tower_id, power, prices, spent in ((34, "disintegrate", (350, 200, 200), 300),
                                               (23, "thorn", (300, 150, 150), 0)):
            command = f'{{"id":"p","action":"upgrade_power","tower_id":{tower_id},"power":"{power}"}}'
            for level, price in enumerate(prices):
                with self.subTest(power=power, level=level):
                    offered = [item for item in self.catalog() if item["action"] == "upgrade_power"
                               and item["tower_id"] == tower_id and item["power"] == power]
                    self.assertEqual([item["cost"] for item in offered], [price])
                    reply = self.rpc(command)
                    self.assertEqual(undump(reply["result"]["native_wire"]),
                                     {"type": "ok", "action": "upgrade_power", "tower_id": tower_id, "power": power,
                                      "level": level + 1, "cost": price})
                    gold, spent = gold - price, spent + price
                    entity = self.entity(tower_id)
                    self.assertEqual((entity["powers"][power]["level"], entity["powers"][power]["changed"]),
                                     (level + 1, True))
                    self.assertEqual((entity["tower"]["spent"], self.gold()), (spent, gold))
            # At max_level: no longer offered, and the command is refused without side effects.
            self.assertNotIn((tower_id, power), {(item.get("tower_id"), item.get("power")) for item in self.catalog()})
            before = self.lua.call("game_dump")
            self.assertEqual(self.rpc(command)["error"], OUTSIDE)
            self.assertEqual(self.lua.call("game_dump"), before)
        # 350 + 200 + 200 and 300 + 150 + 150 (the old formula would charge 1650 and 1350).
        self.assertEqual(2000 - self.gold(), 750 + 600)
        self.assertEqual(self.calls(), [])  # GUI-native in the host, never the bridge's upgrade_power

    def test_sellability(self):
        def sells():
            return {item["tower_id"] for item in self.catalog() if item["action"] == "sell_tower"}
        self.assertNotIn(24, sells())
        self.lua.run("ENTITIES[24].tower.can_be_sold = nil; ENTITIES[20].tower.can_be_sold = false")
        self.assertEqual(sells(), {21, 22, 23, 24, 30, 31, 32, 33})
        self.lua.run("ENTITIES[20].tower.can_be_mod = nil")
        self.assertIn((20, "tower_archer_2"), {(item["tower_id"], item["target"])
                                               for item in self.catalog() if item["action"] == "upgrade_tower"})

    def test_spell_buttons(self):
        cases = [('{mode = "ready"}', '{mode = "cooldown"}', {1}, ["ready", "cooldown"]),
                 ('{mode = "unlocked"}', '{mode = "ready"}', {1, 2}, ["unlocked", "ready"]),
                 ('{mode = "locked"}', '{}', set(), ["locked", "missing"]),
                 ('{mode = "cooldown"}', 'nil', set(), ["cooldown", "missing"])]
        for first, second, expected, modes in cases:
            with self.subTest(first=first, second=second):
                self.lua.run(f"game.game_gui = {{power_1 = {first}, power_2 = {second}}}")
                spells = [item for item in self.catalog() if item["action"] == "use_power"]
                self.assertEqual({item["power"] for item in spells}, expected)
                self.assertEqual(len(spells), 3 * len(expected))
                self.assertEqual([item["mode"] for item in as_list(native_state(self.lua)["powers_ui"])], modes)
        self.lua.run("game.game_gui = nil")
        self.assertFalse([item for item in self.catalog() if item["action"] == "use_power"])
        self.assertEqual(as_list(native_state(self.lua)["powers_ui"]),
                         [{"id": 1, "mode": "missing"}, {"id": 2, "mode": "missing"}])

    def test_spell_anchor_spacing_limit_ties_and_rounding(self):
        def anchors(enemies):
            self.lua.run(f"NATIVE_STATE.enemies = {enemies}")
            return [(item["anchor_id"], item["x"], item["y"]) for item in self.catalog()
                    if item["action"] == "use_power"]
        # 2 is 59.9 from 1; 3 is exactly 60 from 1; 4 is exactly 60 from 1 but 53.7 from 3;
        # 5 is 60 from 1 and 84.9 from 3; 6 is spaced but over the limit; 7 has no progress.
        self.assertEqual(anchors("""{
            {id = 1, x = 0, y = 0, path_progress = 0.9}, {id = 2, x = 59.9, y = 0, path_progress = 0.8},
            {id = 3, x = 60, y = 0, path_progress = 0.7}, {id = 5, x = 0, y = 60, path_progress = 0.6},
            {id = 4, x = 36, y = -48, path_progress = 0.6}, {id = 6, x = -500, y = 0, path_progress = 0.1},
            {id = 7, x = 900, y = 900}}"""), [(1, 0, 0), (5, 0, 60), (3, 60, 0)])
        # Equal progress: lowest ids first; halves round up.
        self.assertEqual(anchors("""{
            {id = 40, x = 0, y = 0, path_progress = 0.5}, {id = 30, x = 99.5, y = 20.49, path_progress = 0.5},
            {id = 20, x = 300, y = -10.5, path_progress = 0.5}, {id = 10, x = 600.4, y = 0.5, path_progress = 0.5}}"""),
            [(30, 100, 20), (20, 300, -10), (10, 600, 1)])
        self.assertEqual(anchors("{{id = 1, x = 5, y = 5}}"), [])
        self.assertEqual(anchors("{}"), [])

    def test_unloaded_menus_offer_no_tower_actions_and_are_never_required(self):
        self.lua.run('package.loaded["data.tower_menus_data"] = nil; MENU_LOADS = 0\n'
                     'package.preload["data.tower_menus_data"] = function() MENU_LOADS = MENU_LOADS + 1; return {} end')
        self.assertEqual({item["action"] for item in self.catalog()}, {"send_wave", "use_power"})
        self.lua.run('assert(MENU_LOADS == 0 and package.loaded["data.tower_menus_data"] == nil)')

    def test_order_is_total_and_independent_of_traversal(self):
        first = self.catalog()
        self.assertEqual(first, sorted(first, key=sort_key))
        self.assertEqual(len({sort_key(item) for item in first}), len(first))
        self.lua.run(r"""
local ids = {}
for id in pairs(ENTITIES) do ids[#ids + 1] = id end
table.sort(ids, function(a, b) return a > b end)
local reordered = {}
for i = 1, 300 do reordered[1000 + i] = {template_name = "decal"} end
for _, id in ipairs(ids) do reordered[id] = ENTITIES[id] end
for i = 1, 300 do reordered[1000 + i] = nil end
game.store.entities = reordered
local enemies = NATIVE_STATE.enemies
for i = 1, math.floor(#enemies / 2) do enemies[i], enemies[#enemies + 1 - i] = enemies[#enemies + 1 - i], enemies[i] end
""")
        self.assertEqual(self.catalog(), first)


class HostV2SunrayTests(HostLuaCase):
    """The sunray level tower: its beam power is bought like a skill; once recharged it is aimed."""

    def ready(self, cooldown=3):
        self.lua.run(f"""
ENTITIES[28].powers.ray.level = 1
ENTITIES[28].user_selection = {{allowed = true}}
ENTITIES[28].attacks = {{list = {{{{ts = 0, cooldown = {cooldown}}}}}}}
NATIVE_STATE.enemies[1].hp = 50
NATIVE_STATE.enemies[2].hp = 400
NATIVE_STATE.enemies[3].hp = 400
NATIVE_STATE.enemies[4].hp = 10
""")

    def aims(self):
        return [item for item in self.catalog() if item["action"] == "point_tower"]

    def test_a_recharged_sunray_is_offered_the_strongest_enemies(self):
        self.assertEqual([], self.aims())  # no beam level yet
        self.ready()
        aims = self.aims()
        self.assertEqual({(102, 130, 210), (103, 301, 200), (101, 100, 201)},
                         {(a["anchor_id"], a["x"], a["y"]) for a in aims})
        self.assertTrue(all(a["tower_id"] == 28 and a["cost"] == 0 for a in aims))
        self.ready(cooldown=5)  # tick_ts 3.25 - ts 0 < 5: still recharging
        self.assertEqual([], self.aims())
        self.ready()
        self.lua.run("ENTITIES[28].user_selection.allowed = false")
        self.assertEqual([], self.aims())

    def test_aiming_sets_the_gui_point_once(self):
        self.ready()
        self.catalog()  # first update pauses the store
        reply = self.rpc('{"id":"a1","action":"point_tower","tower_id":28,"x":130,"y":210,"anchor_id":102}')
        self.assertEqual(undump(reply["result"]["native_wire"]),
                         {"type": "ok", "action": "point_tower", "tower_id": 28, "x": 130, "y": 210, "anchor_id": 102})
        self.assertEqual({"x": 130, "y": 210}, self.entity(28)["user_selection"]["new_pos"])
        self.assertEqual([], self.aims())  # aimed: not offered again until it has fired and recharged
        refused = self.rpc('{"id":"a2","action":"point_tower","tower_id":28,"x":130,"y":210,"anchor_id":102}')
        self.assertIs(refused["ok"], False)
        self.ready()
        refused = self.rpc('{"id":"a3","action":"point_tower","tower_id":28,"x":500,"y":100,"anchor_id":104}')
        self.assertIs(refused["ok"], False)  # 104 is not among the three strongest


class HostV2ClickTests(HostLuaCase):
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


class HostV2HandleTests(HostLuaCase):
    def test_legal_commands_take_the_native_paths(self):
        # Builds take the GUI path in the host (the holder upgrades into its build template; the game
        # charges the possibly discounted price), never the bridge with its base-price gold check.
        reply = self.rpc('{"id":"b1","action":"build_tower","holder_id":10,"tower_type":"archer"}')
        self.assertEqual(undump(reply["result"]["native_wire"]),
                         {"type": "ok", "action": "build_tower", "holder_id": 10, "tower_type": "archer",
                          "target": "tower_build_archer"})
        self.assertEqual(self.entity(10)["tower"]["upgrade_to"], "tower_build_archer")
        self.assertEqual(self.calls(), [])
        reply = self.rpc('{"id":"b2","action":"build_tower","holder_id":11,"tower_type":"mage","target":"tower_build_mage"}')
        self.assertIs(reply["ok"], True)
        self.assertEqual(self.entity(11)["tower"]["upgrade_to"], "tower_build_mage")

        reply = self.rpc('{"id":"u1","action":"upgrade_tower","tower_id":20,"target":"tower_archer_2"}')
        self.assertEqual(undump(reply["result"]["native_wire"]),
                         {"type": "ok", "action": "upgrade_tower", "tower_id": 20, "target": "tower_archer_2"})
        self.assertEqual(self.entity(20)["tower"]["upgrade_to"], "tower_archer_2")
        reply = self.rpc('{"id":"s1","action":"sell_tower","tower_id":21}')
        self.assertEqual(undump(reply["result"]["native_wire"]), {"type": "ok", "action": "sell_tower", "tower_id": 21})
        self.assertIs(self.entity(21)["tower"]["sell"], True)
        self.assertNotIn("upgrade_to", self.entity(21)["tower"])
        self.assertEqual(len(self.calls()), 0)  # builds, upgrades and sales never go through the bridge
        # Pending upgrade/sale: those towers leave the catalog.
        self.assertFalse({20, 21} & {item.get("tower_id") for item in self.catalog()})

        # Power upgrades are GUI-native in the host: level + 1, changed, the GUI price
        # (price_inc past the first level) charged and counted as spent; no bridge call.
        self.assertEqual(self.gold(), 235)  # pending upgrade/sale flags charge nothing yet
        reply = self.rpc('{"id":"p1","action":"upgrade_power","tower_id":23,"power":"poison"}')
        self.assertEqual(undump(reply["result"]["native_wire"]),
                         {"type": "ok", "action": "upgrade_power", "tower_id": 23, "power": "poison",
                          "level": 2, "cost": 50})
        ranger = self.entity(23)
        self.assertEqual(ranger["powers"]["poison"],
                         {"level": 2, "max_level": 3, "price_base": 150, "price_inc": 50, "changed": True})
        self.assertEqual(ranger["powers"]["thorn"], {"level": 3, "max_level": 3, "price_base": 10, "price_inc": 10})
        self.assertEqual((ranger["tower"]["spent"], self.gold()), (50, 185))
        self.assertEqual(len(self.calls()), 0)
        reply = self.rpc('{"id":"c1","action":"use_power","power":1,"x":100,"y":201,"anchor_id":101}')
        self.assertEqual(reply["result"], {"native_wire": "bridge-wire-1"})
        self.assertEqual(self.calls()[-1], {"action": "use_power", "power": 1, "x": 100, "y": 201})
        reply = self.rpc('{"id":"w1","action":"send_wave"}')
        self.assertEqual(reply["result"], {"native_wire": "bridge-wire-2"})
        self.assertEqual(self.calls()[-1], {"id": "w1", "action": "send_wave"})
        self.assertEqual(self.gold(), 185)

    def test_illegal_commands_are_refused_without_side_effects(self):
        self.assertEqual(self.catalog(), V2_WORLD_CATALOG)  # first update pauses the store
        game_before, calls_before = self.lua.call("game_dump"), self.lua.call("calls_dump")
        refused = [
            ('"build_tower","holder_id":13,"tower_type":"mage"', OUTSIDE),  # special holder (v1 allowed it)
            ('"build_tower","holder_id":14,"tower_type":"archer"', OUTSIDE),  # pending build
            ('"build_tower","holder_id":12,"tower_type":"archer"', OUTSIDE),  # blocked
            ('"build_tower","holder_id":10,"tower_type":"engineer"', OUTSIDE),  # tower_engineer_1 locked
            ('"build_tower","holder_id":10,"tower_type":"archer","target":"tower_build_mage"', OUTSIDE),
            ('"build_tower","holder_id":10,"tower_type":"archer","tower_id":20', OUTSIDE),
            ('"upgrade_tower","tower_id":21,"target":"tower_musketeer"', OUTSIDE),  # locked
            ('"upgrade_tower","tower_id":22,"target":"tower_barbarian"', OUTSIDE),  # 240 > 235
            ('"upgrade_tower","tower_id":20,"target":"tower_archer_3"', OUTSIDE),  # not in the level-1 menu
            ('"upgrade_tower","tower_id":20', OUTSIDE),
            ('"upgrade_tower","tower_id":27,"target":"tower_elf"', OUTSIDE),
            ('"upgrade_tower","tower_id":20,"target":"tower_archer_2","power":"poison"', OUTSIDE),
            ('"upgrade_power","tower_id":23,"power":"thorn"', OUTSIDE),  # max level
            ('"upgrade_power","tower_id":31,"power":"missile"', OUTSIDE),  # first level 250 > 235
            ('"upgrade_power","tower_id":28,"power":"beam"', OUTSIDE),  # not the sunray's power
            ('"sell_tower","tower_id":28', OUTSIDE),  # the sunray sells only its beam power
            ('"upgrade_tower","tower_id":28,"target":"tower_sunray"', OUTSIDE),
            ('"sell_tower","tower_id":24', OUTSIDE),  # can_be_sold = false
            ('"sell_tower","tower_id":29', OUTSIDE),  # can_be_mod = false
            ('"sell_tower","tower_id":10', OUTSIDE),  # a holder
            ('"use_power","power":2,"x":100,"y":201,"anchor_id":101', OUTSIDE),  # cooling down
            ('"use_power","power":1,"x":101,"y":201,"anchor_id":101', OUTSIDE),
            ('"use_power","power":1,"x":100,"y":201', OUTSIDE),
            ('"use_power","power":1,"x":130,"y":210,"anchor_id":102', OUTSIDE),  # not an anchor
            ('"use_power","power":"1","x":100,"y":201,"anchor_id":101', OUTSIDE),
            ('"use_power","power":1,"x":100.4,"y":200.6,"anchor_id":101', OUTSIDE),
            ('"send_wave","tower_id":20', OUTSIDE),
            ('"move_hero","x":1,"y":2', NOT_ENABLED),
            ('"set_rally_point","tower_id":22,"x":1,"y":2', NOT_ENABLED),
            ('"eval","code":"return 1"', NOT_ENABLED),
        ]
        for body, error in refused:
            with self.subTest(command=body):
                reply = self.rpc('{"id":"x","action":' + body + "}")
                self.assertEqual((reply["ok"], reply["error"]), (False, error))
        self.assertEqual(self.lua.call("game_dump"), game_before)
        self.assertEqual(self.lua.call("calls_dump"), calls_before)

    def test_send_wave_and_loading_follow_the_v1_rules(self):
        send = '{"id":"w","action":"send_wave"}'
        self.lua.run("NATIVE_STATE.wave = 2; NATIVE_STATE.enemy_count = 1")
        self.assertEqual(self.rpc(send)["error"], OUTSIDE)
        self.lua.run("NATIVE_STATE.enemy_count = 0; NATIVE_STATE.wave_spawning = true")
        self.assertEqual(self.rpc(send)["error"], OUTSIDE)
        self.lua.run("NATIVE_STATE.wave_spawning = false")
        self.assertIs(self.rpc(send)["ok"], True)
        self.lua.run("game.store.tick = nil")
        self.assertEqual(native_state(self.lua)["type"], "loading")
        reply = self.rpc('{"id":"b","action":"build_tower","holder_id":10,"tower_type":"archer"}')
        self.assertEqual(reply["error"], OUTSIDE)
        self.assertEqual(len(self.calls()), 1)


class HostV2GuiRuleTests(HostLuaCase):
    """What the game's GUI itself refuses: unselectable entities and illegal spell points."""

    def assertRefused(self, bodies):
        game_before, calls_before = self.lua.call("game_dump"), self.lua.call("calls_dump")
        for body in bodies:
            with self.subTest(command=body):
                reply = self.rpc('{"id":"x","action":' + body + "}")
                self.assertEqual((reply["ok"], reply["error"]), (False, OUTSIDE))
        self.assertEqual(self.lua.call("game_dump"), game_before)
        self.assertEqual(self.lua.call("calls_dump"), calls_before)

    def test_frozen_and_unclickable_entities_are_never_offered(self):
        def ids():
            return {item.get("holder_id", item.get("tower_id")) for item in self.catalog()}
        # Tower 20 frozen by a boss (tower.blocked); holder 10 and ranger 23 not selectable
        # (ui.can_click = false). An explicit can_click = true changes nothing.
        self.lua.run("ENTITIES[20].tower.blocked = true; ENTITIES[10].ui.can_click = false\n"
                     "ENTITIES[23].ui = {can_click = false}; ENTITIES[22].ui = {can_click = true}")
        self.assertFalse(ids() & {10, 20, 23})
        self.assertLessEqual({11, 21, 22, 24, 30, 31, 32, 33}, ids())
        self.assertRefused([
            '"build_tower","holder_id":10,"tower_type":"archer"',
            '"build_tower","holder_id":10,"tower_type":"barrack","target":"tower_build_barrack"',
            '"upgrade_tower","tower_id":20,"target":"tower_archer_2"',
            '"sell_tower","tower_id":20',
            '"upgrade_power","tower_id":23,"power":"poison"',
            '"sell_tower","tower_id":23',
        ])
        # Thawed and selectable again: the catalog is the full world catalog once more.
        self.lua.run("ENTITIES[20].tower.blocked = nil; ENTITIES[10].ui.can_click = nil\n"
                     "ENTITIES[23].ui.can_click = true")
        self.assertEqual(self.catalog(), V2_WORLD_CATALOG)

    def test_spell_points_follow_the_gui_placement_rules(self):
        self.lua.run('game.game_gui.power_2 = {mode = "ready"}')
        others = {(power, anchor) for power in (1, 2) for anchor in (103, 104)}
        self.assertEqual(self.spells(), others | {(1, 101), (2, 101)})
        # Anchor 101's cell (100,201): rain of fire needs no cliff/faerie and a NF_POWER_1
        # node nearby or water; reinforcements need a NF_RALLY node nearby on land/ice only.
        cases = [
            ("{terrain = TERRAIN_CLIFF, nodes = bit.bor(NF_POWER_1, NF_RALLY)}", set()),
            ("{terrain = bit.bor(TERRAIN_LAND, TERRAIN_FAERIE), nodes = bit.bor(NF_POWER_1, NF_RALLY)}", set()),
            ("{terrain = TERRAIN_LAND, nodes = 0}", set()),
            ("{terrain = TERRAIN_WATER, nodes = 0}", {1}),
            ("{terrain = TERRAIN_LAND, nodes = NF_POWER_1}", {1}),
            ("{terrain = TERRAIN_LAND, nodes = NF_RALLY}", {2}),
            ("{terrain = TERRAIN_ICE, nodes = NF_RALLY}", {2}),
            ("{terrain = bit.bor(TERRAIN_LAND, TERRAIN_WATER), nodes = bit.bor(NF_POWER_1, NF_RALLY)}", {1}),
            ("{terrain = bit.bor(TERRAIN_LAND, TERRAIN_ICE), nodes = NF_RALLY}", {2}),
            ('"error"', set()),  # a failing game module fails closed
        ]
        for cell, powers in cases:
            with self.subTest(cell=cell):
                self.lua.run(f'GROUND["100,201"] = {cell}')
                self.assertEqual(self.spells(), others | {(power, 101) for power in powers})
                self.assertRefused([f'"use_power","power":{power},"x":100,"y":201,"anchor_id":101'
                                    for power in {1, 2} - powers])
        # The host asks exactly the GUI's questions: power 1 with the rain-of-fire node range.
        self.lua.run('GROUND["100,201"] = nil; GROUND_CALLS = {}')
        self.catalog()
        asked = {(call["name"], call.get("range"), call["flags"])
                 for call in as_list(undump(self.lua.call("ground_calls_dump")))
                 if (call["x"], call["y"]) == (100, 201)}
        self.assertEqual(asked, {("cell_is", None, 4 | 16), ("valid_node_nearby", 1 / 0.7, 8),
                                 ("valid_node_nearby", "nil", 2), ("cell_is_only", None, 1 | 8)})

    def test_spells_need_the_game_placement_modules_already_loaded(self):
        cases = {
            "path_db unloaded": 'package.loaded["path_db"] = nil',
            "grid_db unloaded": 'package.loaded["grid_db"] = nil',
            "both unloaded": 'package.loaded["path_db"] = nil; package.loaded["grid_db"] = nil',
            "path_db without nodes": 'package.loaded["path_db"] = {}',
            "grid_db not a table": 'package.loaded["grid_db"] = true',
            "terrain constants missing": "TERRAIN_CLIFF = nil; TERRAIN_LAND = nil",
        }
        for name, change in cases.items():
            with self.subTest(case=name):
                lua = self.fresh()
                lua.run('game.game_gui.power_2 = {mode = "ready"}; MODULE_LOADS = 0\n'
                        'for _, name in ipairs({"path_db", "grid_db"}) do\n'
                        '    package.preload[name] = function() MODULE_LOADS = MODULE_LOADS + 1; return {} end\n'
                        'end\n' + change + "\n"
                        'WAS_NIL = {path_db = package.loaded["path_db"] == nil, '
                        'grid_db = package.loaded["grid_db"] == nil}')
                catalog = self.catalog(lua)
                self.assertFalse([item for item in catalog if item["action"] == "use_power"])
                self.assertLessEqual({"build_tower", "upgrade_tower", "upgrade_power", "sell_tower", "send_wave"},
                                     {item["action"] for item in catalog})
                self.assertEqual(as_list(native_state(lua)["powers_ui"]),
                                 [{"id": 1, "mode": "ready"}, {"id": 2, "mode": "ready"}])
                reply = rpc(lua, '{"id":"c","action":"use_power","power":1,"x":100,"y":201,"anchor_id":101}')
                self.assertEqual((reply["ok"], reply["error"]), (False, OUTSIDE))
                self.assertEqual(as_list(undump(lua.call("calls_dump"))), [])
                # Never require()d by the host: no loader ran and nothing new was loaded.
                lua.run("assert(MODULE_LOADS == 0, 'placement module required')\n"
                        "for name, was_nil in pairs(WAS_NIL) do\n"
                        "    assert(not was_nil or package.loaded[name] == nil, name .. ' was loaded')\n"
                        "end")


@unittest.skipUnless(os.name == "nt" and LUA_DLL.exists(), "game LuaJIT runtime/rl-engine/lua51.dll unavailable")
class HostActionScopeInstallTests(unittest.TestCase):
    def test_unknown_scopes_fail_closed_at_install(self):
        for scope in ("v9", "", "V2", "v1 ", "all"):
            with self.subTest(scope=scope):
                lua = load(scope)
                self.addCleanup(lua.close)
                status, message, installed = lua.call("install").split("|")
                self.assertEqual((status, installed), ("ERR", "false"))
                self.assertIn("Unsupported ALPHARUSH_ACTION_SCOPE", message)
                with self.assertRaises(RuntimeError):
                    lua.call("rpc", b'{"id":"1","action":"hello"}')

    def test_known_scopes_install(self):
        for scope in ("<unset>", "v1", "v2"):
            with self.subTest(scope=scope):
                lua = load(scope)
                self.addCleanup(lua.close)
                self.assertEqual(lua.call("install"), "OK|nil|true")


if __name__ == "__main__":
    unittest.main()
