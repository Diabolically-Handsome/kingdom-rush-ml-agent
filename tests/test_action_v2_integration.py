"""Action scope v2 end to end, across the host, Python env, menus, teacher and survey parts.

1. ``TeacherEpisodeTests``: TeacherV2 plays whole episodes with ``run_episode`` on the
   real NativeEnv over test_env_actions_v2's FakeWorker (host v2 semantics in Python);
   the plans replay cold with ``replay_verify``.
2. ``SurveyScopeTests``: the campaign survey worker loop with its real
   ``native_env_factory``; the job's action scope reaches every Worker, episode row
   and cold replay.
3. ``RealHostBridgeTests``: the real host.lua and the real native bridge
   (bridge.lua + engine.NATIVE_EXPORTS, exactly what engine._build packs) run in the
   game's LuaJIT behind the real engine.Worker (scope handshake, token, rpc) and
   NativeEnv. Only the game itself (store, entities, templates, path_db/grid_db, GUI
   power buttons, simulation) is a small Lua stand-in.

Nothing here starts the game, a process, a socket or the GPU.
"""
from __future__ import annotations

from collections import Counter
import copy
import json
import math
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from alpharush_rl import engine, env as env_module
from alpharush_rl.episode import EpisodeProtocol, replay_verify, run_episode
from alpharush_rl.journal import Journal, canonical_json, sha256_data
from alpharush_rl.menus import build_menu
from alpharush_rl.scripted_policies import BRANCH_PREFERENCE, make_policy

try:
    from test_campaign_survey import SURVEY, TempWorkspace, item
    from test_env_actions_v2 import StepClock, worker_class
    from test_host_protocol import HOST, LUA_DLL, Lua
except ImportError:  # run as tests.test_action_v2_integration from the workspace root
    from tests.test_campaign_survey import SURVEY, TempWorkspace, item
    from tests.test_env_actions_v2 import StepClock, worker_class
    from tests.test_host_protocol import HOST, LUA_DLL, Lua

ROOT = Path(__file__).resolve().parents[1]
BRIDGE = ROOT / "Lumi_Nox/games/kingdom_rush/bridge.lua"
V1_KINDS = {"wait", "build_tower", "send_wave"}
V2_KINDS = {"upgrade_tower", "upgrade_power", "sell_tower", "use_power"}
# TeacherV2's meta rule for each chosen native action.
RULES = {"use_power": "use_power", "upgrade_power": "upgrade_power", "upgrade_tower": "upgrade_tower",
         "build_tower": "build", "send_wave": "send_wave", "wait": "wait"}
# A FakeWorker level where the teacher reaches tier 4, buys a power and casts spells.
TEACHER_LEVEL = {"gold": 1000, "waves": 8}
# The keys a menu action keeps from a native catalog entry (menus.build_menu).
MENU_KEYS = {"build_tower": ("holder_id", "tower_type"), "upgrade_tower": ("tower_id", "target"),
             "upgrade_power": ("tower_id", "power"), "sell_tower": ("tower_id",),
             "use_power": ("power", "x", "y", "anchor_id"), "send_wave": ()}


def menu_action(entry):
    return {"action": entry["action"], **{key: entry[key] for key in MENU_KEYS[entry["action"]]}}


def action_receipts(trace):
    return [entry["receipt"] for entry in trace if entry["kind"] == "action"]


class RecordingEnv(env_module.NativeEnv):
    """NativeEnv that keeps every state it observed (episode results carry only hashes)."""

    def reset(self):
        self.seen = []
        state = super().reset()
        self.seen.append(state)
        return state

    def advance(self, ticks, *, record_plan=True):
        state = super().advance(ticks, record_plan=record_plan)
        self.seen.append(state)
        return state


class TeacherEpisodeTests(unittest.TestCase):
    def make(self, action_scope="v2", identity="it_teacher", **config):
        patcher = mock.patch.object(env_module, "Worker", worker_class(**{**TEACHER_LEVEL, **config}))
        patcher.start()
        self.addCleanup(patcher.stop)
        return env_module.NativeEnv(seed=1001, level=1, identity=identity, action_scope=action_scope)

    def play(self, env, policy="teacher_v2", protocol=None):
        try:
            return run_episode(env, make_policy(policy), protocol or EpisodeProtocol(), seed=1001, level=1,
                               episode_id="it-teacher", clock=StepClock())
        finally:
            env.close()

    def test_teacher_v2_plays_every_v2_kind_and_replays_cold(self):
        env = self.make()
        result = self.play(env)
        self.assertEqual((result["status"], result["void_reason"]), ("terminal", None))
        self.assertTrue(result["outcome"]["level_won"])
        kinds = Counter(d["action"]["action"] for d in result["decisions"])
        for kind in ("build_tower", "upgrade_tower", "upgrade_power", "use_power", "send_wave"):
            self.assertGreater(kinds[kind], 0, kinds)
        self.assertEqual(kinds["sell_tower"], 0)  # the teacher never sells
        # Tier 4 through the preferred branch, then that tower's powers.
        targets = [d["action"]["target"] for d in result["decisions"] if d["action"]["action"] == "upgrade_tower"]
        self.assertIn(BRANCH_PREFERENCE["archer"], targets)
        ranger = next(t["id"] for t in env.state["towers"] if t["template"] == "tower_ranger")
        self.assertTrue(all(d["action"]["tower_id"] == ranger for d in result["decisions"]
                            if d["action"]["action"] == "upgrade_power"))
        for decision in result["decisions"]:
            self.assertEqual(decision["provenance"], "scripted:teacher_v2")
            self.assertEqual(decision["meta"]["rule"], RULES[decision["action"]["action"]])
            if decision["action"]["action"] == "use_power":
                self.assertGreaterEqual(decision["meta"]["anchor_progress"], 0.5)
        # Every native action has an executed receipt, in plan order, and the hashes match.
        receipts = action_receipts(env.trace)
        self.assertTrue(all(r["accepted"] and r["executed"] for r in receipts))
        self.assertEqual([r["action"] for r in receipts], [p["action"] for p in result["plan"] if "action" in p])
        self.assertEqual(sha256_data(env.trace), result["trace_sha256"])
        spell = next(r for r in receipts if r["action"]["action"] == "use_power")
        self.assertEqual(spell["power_mode_after"], "cooldown")
        # The JSON-journaled result replays cold in a fresh v2 env.
        result = json.loads(json.dumps(result, allow_nan=False))
        verdict = replay_verify(lambda: self.make(identity="it_teacher_replay"), result)
        self.assertEqual((verdict["replay_verified"], verdict["outcome_match"], verdict["replay_error"]),
                         (True, True, None))
        self.assertEqual(verdict["replay_trace_sha256"], result["trace_sha256"])
        # A v1 env refuses the v2 plan, and a moved spell no longer matches the native menu.
        verdict = replay_verify(lambda: self.make("v1", identity="it_teacher_v1"), result)
        self.assertIs(verdict["replay_verified"], False)
        self.assertIn("ValueError", verdict["replay_error"])
        tampered = copy.deepcopy(result)
        cast = next(p["action"] for p in tampered["plan"] if p.get("action", {}).get("action") == "use_power")
        cast["x"] += 1
        verdict = replay_verify(lambda: self.make(identity="it_teacher_tampered"), tampered)
        self.assertIs(verdict["replay_verified"], False)
        self.assertIn("native legal menu", verdict["replay_error"])

    def test_teacher_v2_is_deterministic(self):
        first = self.play(self.make(identity="it_teacher_a"))
        second = self.play(self.make(identity="it_teacher_b"))
        self.assertEqual(first["plan"], second["plan"])
        self.assertEqual(first["trace_sha256"], second["trace_sha256"])
        self.assertEqual([d["label"] for d in first["decisions"]], [d["label"] for d in second["decisions"]])

    def test_teacher_v2_in_a_v1_env_stays_in_v1(self):
        env = self.make("v1", identity="it_teacher_v1_only")
        result = self.play(env)
        self.assertEqual(result["status"], "terminal")
        self.assertLessEqual({d["action"]["action"] for d in result["decisions"]}, V1_KINDS)
        self.assertNotIn("action_scope", env.worker.passed)  # the v1 Worker call is unchanged
        verdict = replay_verify(lambda: self.make("v1", identity="it_teacher_v1_replay"),
                                json.loads(json.dumps(result)))
        self.assertIs(verdict["replay_verified"], True)


class SurveyScopeTests(TempWorkspace):
    """worker_loop -> native_env_factory -> NativeEnv(action_scope) -> Worker(...)."""

    def run_job(self, run_list, **kwargs):
        workers = []

        class Recording(worker_class(**TEACHER_LEVEL)):
            def __init__(self, **passed):
                super().__init__(**passed)
                workers.append(self)

        ctx = self.context(max_games=2 * len(run_list))
        out = Journal(ctx.output_dir / SURVEY.EPISODES)
        with mock.patch.object(env_module, "Worker", Recording):
            summary = SURVEY.worker_loop(run_list, SURVEY.native_env_factory(ctx.run_id), ctx, out,
                                         pools=self.pools, replay_fraction=1.0, **kwargs)
        return summary, out.entries(), workers

    def test_v2_job_reaches_every_worker_episode_and_replay(self):
        summary, rows, workers = self.run_job([item(1001, "teacher_v2"), item(1002, "pressure_greedy")],
                                              action_scope="v2")
        self.assertEqual(summary["action_scope"], "v2")
        self.assertEqual((summary["episodes"], summary["replays_sampled"], summary["replays_verified"]), (2, 2, 2))
        self.assertEqual((summary["episode_errors"], summary["replay_failures"]), ([], []))
        self.assertEqual([row["kind"] for row in rows], ["episode", "replay", "episode", "replay"])
        # Episode, then its cold replay, each in a fresh Worker started for scope v2.
        self.assertEqual([(w.passed["identity"], w.passed.get("action_scope")) for w in workers],
                         [("survey_0123456789ab_0000", "v2"), ("survey_0123456789ab_0000_replay", "v2"),
                          ("survey_0123456789ab_0001", "v2"), ("survey_0123456789ab_0001_replay", "v2")])
        self.assertTrue(all(w.starts == 1 and w.closes >= 1 for w in workers))
        episodes = [row["payload"] for row in rows if row["kind"] == "episode"]
        self.assertEqual([e["action_scope"] for e in episodes], ["v2", "v2"])
        teacher = episodes[0]["result"]
        self.assertEqual((teacher["policy"], teacher["status"]), ("teacher_v2", "terminal"))
        self.assertLessEqual({"upgrade_tower", "upgrade_power", "use_power"},
                             {d["action"]["action"] for d in teacher["decisions"]})
        # PressureGreedy only builds and sends waves, in either scope.
        self.assertLessEqual({d["action"]["action"] for d in episodes[1]["result"]["decisions"]}, V1_KINDS)
        self.assertTrue(all(row["payload"]["replay_verified"] for row in rows if row["kind"] == "replay"))

    def test_default_job_keeps_the_v1_worker_call(self):
        summary, rows, workers = self.run_job([item(1001, "teacher_v2")])
        self.assertEqual(summary["action_scope"], "v1")
        self.assertEqual((summary["replays_sampled"], summary["replays_verified"]), (1, 1))
        self.assertTrue(workers and all("action_scope" not in w.passed and w.action_scope == "v1" for w in workers))
        episode = rows[0]["payload"]
        self.assertEqual(episode["action_scope"], "v1")
        self.assertLessEqual({d["action"]["action"] for d in episode["result"]["decisions"]}, V1_KINDS)


# ---------------------------------------------------------------------------------------------
# The real host.lua and native bridge in the game's LuaJIT.

# The process stand-in: stub socket/LÖVE, the game's entity_db/path_db modules, and
# alpha_native_bridge built from the real bridge source. Arguments: host path, the
# ALPHARUSH_* environment the Worker gave the process, the native bridge module source.
HARNESS = r"""
local host_path, env_text, native_source = ...
print = function() end
local ENV = {}
for key, value in env_text:gmatch("([^=\n]+)=([^\n]*)") do ENV[key] = value end
os.getenv = function(name) return ENV[name] end
local INBOX, OUTBOX, connected = {}, {}, false
local client = {
    settimeout = function() end, close = function() connected = false end,
    send = function(_, payload) OUTBOX[#OUTBOX + 1] = payload; return #payload end,
    receive = function()
        local line = table.remove(INBOX, 1)
        if line then return line end
        return nil, "timeout", ""
    end,
}
local server = {settimeout = function() end,
                accept = function() if connected then return nil end; connected = true; return client end}
package.preload["socket"] = function()
    return {bind = function() return server end, gettime = function() return 0 end}
end
package.preload["alpha_native_bridge"] = function()
    return assert(loadstring(native_source, "=alpha_native_bridge"))()
end
package.preload["entity_db"] = function()
    return {get_template = function(self, name) return TEMPLATES[name] end,
            create_entity = function(self, name)
                return {template_name = name, pos = {x = 0, y = 0}, user_selection = {}}
            end}
end
package.preload["path_db"] = function() return PATHS end
love = {filesystem = {getSaveDirectory = function() return "C:/saves/" .. tostring(ENV.ALPHARUSH_IDENTITY) end}}
HOST = dofile(host_path)
function rpc(line)
    if not HOST.installed then HOST.install(native_update) end
    INBOX[#INBOX + 1] = line
    HOST.update(1 / 60)
    local reply = table.concat(OUTBOX)
    OUTBOX = {}
    return reply
end
"""

# A small deterministic level: the game's own menu data (subset), templates with the
# bridge's prices, one straight path, standard/special/blocked holders, a level-3
# archer, a ranger with powers and a special elf tower. CONFIG is set from Python.
WORLD = r"""
local function up(arg) return {action = "tw_upgrade", action_arg = arg} end
local function power(name) return {action = "upgrade_power", action_arg = name} end
local SELL, RALLY = {action = "tw_sell"}, {action = "tw_rally"}
package.loaded["data.tower_menus_data"] = {
    holder = {{up("tower_build_archer"), up("tower_build_barrack"), up("tower_build_mage"), up("tower_build_engineer")}},
    archer = {{up("tower_archer_2"), SELL}, {up("tower_archer_3"), SELL}, {up("tower_ranger"), up("tower_musketeer"), SELL}},
    barrack = {{up("tower_barrack_2"), RALLY, SELL}, {up("tower_barrack_3"), RALLY, SELL},
               {up("tower_paladin"), up("tower_barbarian"), RALLY, SELL}},
    mage = {{up("tower_mage_2"), SELL}, {up("tower_mage_3"), SELL}, {up("tower_arcane_wizard"), up("tower_sorcerer"), SELL}},
    engineer = {{up("tower_engineer_2"), SELL}, {up("tower_engineer_3"), SELL}, {up("tower_bfg"), up("tower_tesla"), SELL}},
    ranger = {{power("poison"), power("thorn"), SELL}},
    musketeer = {{power("sniper"), power("shrapnel"), SELL}},
    paladin = {{power("healing"), power("shield"), RALLY, SELL}},
    barbarian = {{power("twister"), power("throwing_axes"), RALLY, SELL}},
    arcane_wizard = {{power("disintegrate"), power("teleport"), SELL}},
    sorcerer = {{power("polymorph"), power("elemental"), SELL}},
    bfg = {{power("missile"), power("cluster"), SELL}},
    tesla = {{power("bolt"), power("overcharge"), SELL}},
    holder_elf = {{up("tower_elf")}},
    elf = {{{action = "tw_buy_soldier"}}},
}

function sorted_keys(t)
    local keys = {}
    for k in pairs(t) do keys[#keys + 1] = k end
    table.sort(keys)
    return keys
end
local function tower(kind, level, price, damage, powers)
    return {tower = {type = kind, level = level, price = price}, damage = damage, powers = powers}
end
local function pw(base, inc) return {price_base = base, price_inc = inc, max_level = 3} end
TEMPLATES = {enemy_goblin = {health = {hp_max = 100}, enemy = {gold = 5, lives_cost = 1}, motion = {max_speed = 1}},
             tower_elf = tower("elf", 1, 100, 0)}
for kind, prices in pairs({archer = {70, 110, 160}, barrack = {70, 110, 160}, mage = {100, 160, 240},
                           engineer = {125, 220, 320}}) do
    TEMPLATES["tower_build_" .. kind] = {build_name = "tower_" .. kind .. "_1", tower = {type = "build_animation", price = 0}}
    for level, price in ipairs(prices) do TEMPLATES["tower_" .. kind .. "_" .. level] = tower(kind, level, price, level) end
end
for kind, spec in pairs({ranger = {230, {poison = pw(250, 0), thorn = pw(300, 50)}},
                         musketeer = {230, {sniper = pw(250, 150), shrapnel = pw(200, 100)}},
                         paladin = {230, {healing = pw(150, 50), shield = pw(100, 100)}},
                         barbarian = {230, {twister = pw(250, 0), throwing_axes = pw(200, 50)}},
                         arcane_wizard = {300, {disintegrate = pw(350, 0), teleport = pw(300, 0)}},
                         sorcerer = {300, {polymorph = pw(300, 0), elemental = pw(250, 0)}},
                         bfg = {400, {missile = pw(250, 0), cluster = pw(250, 0)}},
                         tesla = {375, {bolt = pw(250, 0), overcharge = pw(250, 0)}}}) do
    TEMPLATES["tower_" .. kind] = tower(kind, 1, spec[1], 4, spec[2])
end
local nodes = {}
for x = 0, 900, 10 do nodes[#nodes + 1] = {x = x, y = 400} end
PATHS = {paths = {{nodes}}}
-- The game's spell placement modules, already loaded as in the game (the host never
-- require()s them): both node kinds along the road within 30 px, land everywhere.
-- Constant values are stand-in distinct bits.
TERRAIN_LAND, TERRAIN_WATER, TERRAIN_CLIFF, TERRAIN_ICE, TERRAIN_FAERIE = 1, 2, 4, 8, 16
NF_RALLY, NF_POWER_1 = 2, 8
function PATHS:valid_node_nearby(x, y, range, flags)
    if bit.band(flags, bit.bor(NF_RALLY, NF_POWER_1)) == 0 then return false end
    for _, n in ipairs(nodes) do
        if math.abs(n.x - x) <= 30 and math.abs(n.y - y) <= 30 then return true end
    end
    return false
end
GRID = {cell_is = function(self, x, y, flags) return bit.band(TERRAIN_LAND, flags) ~= 0 end,
        cell_is_only = function(self, x, y, flags) return bit.band(TERRAIN_LAND, bit.bnot(flags)) == 0 end}
package.loaded["path_db"], package.loaded["grid_db"] = PATHS, GRID

local function group(index)
    return {group_idx = index, waves = {{path_index = 1, spawns = {{creep = "enemy_goblin", max = CONFIG.waves[index]}}}}}
end
game = {
    store = {tick = 1, ts = 0, tick_ts = 0, tick_length = 1 / 30, player_gold = CONFIG.gold, lives = 20,
             level_idx = 1, level_name = "level_unit", wave_group_number = 0, wave_group_total = #CONFIG.waves,
             waves_finished = false, level = {locked_towers = {"tower_musketeer"}, max_upgrade_level = 5},
             next_wave_group_ready = group(1), entities = {}},
    game_gui = {mode = 0, power_1 = {mode = "ready", cooldown = CONFIG.cooldowns[1]},
                power_2 = {mode = "ready", cooldown = CONFIG.cooldowns[2]}},
}
local NEXT_ID = 0
local function add(e)
    NEXT_ID = NEXT_ID + 1
    e.id = NEXT_ID
    game.store.entities[NEXT_ID] = e
    return e
end
game.simulation = {queue_insert_entity = function(self, e) add(e) end,
                   queue_remove_entity = function(self, e) game.store.entities[e.id] = nil end}
local function new_holder(mesh, x, y, kind)
    return {template_name = "tower_holder_grass", pos = {x = x, y = y}, ui = {nav_mesh_id = mesh}, tower_holder = {},
            tower = {type = kind or "holder", level = 1, holder_id = mesh, terrain_style = 1}}
end
local function new_tower(template, mesh, x, y, spent)
    local t = TEMPLATES[template]
    local e = {template_name = template, pos = {x = x, y = y}, attacks = {range = 300},
               tower = {type = t.tower.type, level = t.tower.level, spent = spent, holder_id = mesh,
                        can_be_sold = true, can_be_mod = true, refund_factor = 0.6, terrain_style = 1}}
    if t.powers then
        e.powers = {}
        for _, name in ipairs(sorted_keys(t.powers)) do
            local p = t.powers[name]
            e.powers[name] = {level = 0, max_level = p.max_level, price_base = p.price_base, price_inc = p.price_inc}
        end
    end
    return e
end
add(new_holder("1", 150, 300))
add(new_holder("2", 350, 300))
add(new_holder("3", 550, 300))
add(new_holder("4", 750, 620))
add(new_holder("5", 450, 520, "holder_elf"))
add(new_holder("6", 650, 300)).tower_holder.blocked = true
add(new_tower("tower_archer_3", "7", 250, 520, 340))
add(new_tower("tower_ranger", "8", 850, 300, 570))
add(new_tower("tower_elf", "9", 650, 520, 100)).tower.can_be_sold = false

local SPAWN = nil
-- One native frame while the host steps a paused store (store.step), like the game's update.
function native_update(dt)
    local store = game.store
    if not store.step then return end
    store.tick = store.tick + 1
    store.ts = store.ts + store.tick_length
    store.tick_ts = store.ts
    -- Tower system: GUI upgrade/sale flags resolve on the next frame into new entities.
    for _, id in ipairs(sorted_keys(store.entities)) do
        local e = store.entities[id]
        local t = e.tower
        if t and t.upgrade_to then
            local final = TEMPLATES[t.upgrade_to].build_name or t.upgrade_to
            local price = TEMPLATES[final].tower.price
            local holder = t.type == "holder"
            store.player_gold = store.player_gold - price
            store.entities[id] = nil
            add(new_tower(final, holder and e.ui.nav_mesh_id or t.holder_id, e.pos.x, e.pos.y,
                          (holder and 0 or t.spent) + price))
        elseif t and t.sell then
            local refund = store.wave_group_number == 0 and t.spent or math.floor(t.spent * t.refund_factor)
            store.player_gold = store.player_gold + refund
            store.entities[id] = nil
            add(new_holder(t.holder_id, e.pos.x, e.pos.y))
        elseif e.template_name == "power_fireball_control" then
            for _, en in pairs(store.entities) do
                if en.enemy then
                    local dx, dy = en.pos.x - e.pos.x, en.pos.y - e.pos.y
                    if dx * dx + dy * dy <= 3600 then en.health.hp = en.health.hp - CONFIG.blast end
                end
            end
            store.entities[id] = nil
        elseif e.template_name == "power_reinforcements_control" then
            store.entities[id] = nil
        end
    end
    for p = 1, 2 do
        local btn = game.game_gui["power_" .. p]
        if btn.mode == "cooldown" and store.ts - btn.start_ts >= btn.cooldown then btn.mode = "ready" end
    end
    if store.send_next_wave then
        store.send_next_wave = false
        store.wave_group_number = store.wave_group_number + 1
        store.next_wave_group_ready = nil
        SPAWN = {left = CONFIG.waves[store.wave_group_number], at = store.tick}
    end
    if SPAWN and store.tick >= SPAWN.at then
        local hp = CONFIG.enemy_hp + CONFIG.enemy_hp_step * store.wave_group_number
        add({template_name = "enemy_goblin", pos = {x = 0, y = 400}, enemy = {gold = 5, lives_cost = 1},
             health = {hp = hp, hp_max = hp, dead = false}, motion = {max_speed = 1},
             nav_path = {pi = 1, spi = 1, ni = 1, dir = 1}})
        SPAWN.left, SPAWN.at = SPAWN.left - 1, store.tick + 40
        if SPAWN.left == 0 then
            SPAWN = nil
            if CONFIG.waves[store.wave_group_number + 1] then
                store.next_wave_group_ready = group(store.wave_group_number + 1)
            else
                store.waves_finished = true
            end
        end
    end
    -- Towers hit the enemy furthest along; enemies die, walk or leak.
    local ids = sorted_keys(store.entities)
    for _, id in ipairs(ids) do
        local e = store.entities[id]
        local damage = e.tower and TEMPLATES[e.template_name] and TEMPLATES[e.template_name].damage or 0
        local target = nil
        for _, eid in ipairs(ids) do
            local en = store.entities[eid]
            if en.enemy and en.health.hp > 0 and (target == nil or en.pos.x > target.pos.x) then target = en end
        end
        if damage > 0 and target and store.tick % CONFIG.attack_every == 0 then
            for _, name in ipairs(sorted_keys(e.powers or {})) do damage = damage + e.powers[name].level end
            target.health.hp = target.health.hp - damage
        end
    end
    local alive = false
    for _, id in ipairs(ids) do
        local en = store.entities[id]
        if en.enemy then
            if en.health.hp <= 0 then
                store.entities[id] = nil
                store.player_gold = store.player_gold + 5
            else
                en.pos.x = en.pos.x + 1
                en.nav_path.ni = en.nav_path.ni + 1
                if en.pos.x >= 900 then
                    store.entities[id] = nil
                    store.lives = store.lives - 1
                else
                    alive = true
                end
            end
        end
    end
    if store.lives <= 0 then
        store.lives = 0
        store.game_outcome = {victory = false}
    elseif store.waves_finished and not alive and not SPAWN then
        store.game_outcome = {victory = true}
    end
end
"""
LUA_LEVEL = {"gold": 1500, "waves": (3, 4, 5, 6), "enemy_hp": 200, "enemy_hp_step": 150, "blast": 120,
             "cooldowns": (10, 4), "attack_every": 10}


def native_bridge_source():
    """alpha_native_bridge.lua exactly as engine._build packs it into the prepared exe."""
    return BRIDGE.read_text(encoding="utf-8").rsplit("return M", 1)[0] + engine.NATIVE_EXPORTS


def lua_config(config):
    def value(v):
        if isinstance(v, (tuple, list)):
            return "{" + ", ".join(value(x) for x in v) + "}"
        return repr(v)
    return "CONFIG = {" + ", ".join(f"{key} = {value(v)}" for key, v in sorted(config.items())) + "}"


class LuaGame:
    """The isolated game process: real host.lua + real native bridge in a LuaJIT state."""

    def __init__(self, environment, config, drop=()):
        self.environment = {k: v for k, v in environment.items() if k.startswith("ALPHARUSH_") and k not in drop}
        self.lua = Lua()
        text = "\n".join(f"{key}={value}" for key, value in sorted(self.environment.items()))
        self.lua.run(HARNESS, str(HOST).encode("utf-8"), text.encode("utf-8"), native_bridge_source().encode("utf-8"))
        self.lua.run(lua_config(config))
        self.lua.run(WORLD)
        self.alive = True
        self.requests = []

    def request(self, line):
        if not self.alive:
            raise ConnectionError("Isolated game exited")
        self.requests.append(json.loads(line))
        return self.lua.call("rpc", line).encode("utf-8")

    # -- subprocess.Popen stand-in
    def poll(self):
        return None if self.alive else 0

    def terminate(self):
        if self.alive:
            self.alive = False
            self.lua.close()

    kill = terminate

    def wait(self, timeout=None):
        return 0


class LuaSocket:
    """socket.create_connection stand-in: one request line in, the host's reply line(s) out."""

    def __init__(self, game):
        self.game, self.pending = game, b""

    def settimeout(self, timeout):
        pass

    def sendall(self, data):
        for line in data.split(b"\n"):
            if line:
                self.pending += self.game.request(line)

    def recv(self, size):
        data, self.pending = self.pending[:size], self.pending[size:]
        return data

    def close(self):
        pass


@unittest.skipUnless(os.name == "nt" and LUA_DLL.exists(), "game LuaJIT runtime/rl-engine/lua51.dll unavailable")
class RealHostBridgeTests(unittest.TestCase):
    """engine.Worker.start launches a LuaGame instead of the prepared exe."""

    def setUp(self):
        tmp = tempfile.TemporaryDirectory(prefix="alpharush-v2-integration-")
        self.addCleanup(tmp.cleanup)
        base = Path(tmp.name)
        self.games, self.config, self.drop = [], dict(LUA_LEVEL), ()

        def popen(args, **kwargs):
            game = LuaGame(kwargs["env"], self.config, self.drop)
            self.games.append(game)
            self.addCleanup(game.terminate)
            return game

        for patcher in (mock.patch.object(engine, "ROOT", base),
                        mock.patch.object(engine, "prepare", return_value=base / "rl-engine/Kingdom Rush.exe"),
                        mock.patch.object(engine.subprocess, "Popen", side_effect=popen),
                        mock.patch.object(engine.socket, "create_connection",
                                          side_effect=lambda *a, **k: LuaSocket(self.games[-1]))):
            patcher.start()
            self.addCleanup(patcher.stop)

    def env(self, action_scope="v2", identity="it_lua", cls=env_module.NativeEnv):
        native = cls(seed=1001, level=1, port=9879, identity=identity, action_scope=action_scope)
        self.addCleanup(native.close)
        return native

    def assertMenuMatchesCatalog(self, state):
        """Every host catalog entry is one menu option and the menu adds nothing beyond wait.

        After the native outcome the host still lists its catalog; the menu offers only wait.
        """
        catalog = {canonical_json(menu_action(entry)) for entry in state["action_catalog"]}
        menu = build_menu(state)
        self.assertEqual(menu[0]["action"]["action"], "wait")
        if state["level_won"] or state["level_lost"]:
            self.assertEqual(len(menu), 1)
            return menu
        self.assertEqual(catalog, {canonical_json(option["action"]) for option in menu[1:]})
        self.assertEqual(len(catalog), len(state["action_catalog"]))
        self.assertTrue(all(entry["available"] is True and entry["legal"] is True for entry in state["action_catalog"]))
        return menu

    def test_scope_handshake_through_the_real_host(self):
        native = self.env(identity="it_lua_v2")
        state = native.reset()
        game = self.games[-1]
        self.assertEqual(game.environment["ALPHARUSH_ACTION_SCOPE"], "v2")
        self.assertEqual(game.environment["ALPHARUSH_TOKEN"], native.worker.token)
        self.assertEqual(game.requests[0]["action"], "hello")
        self.assertEqual((state["action_scope"], state["powers_ui"]),
                         ("v2", [{"id": 1, "mode": "ready"}, {"id": 2, "mode": "ready"}]))
        self.assertTrue({e["action"] for e in state["action_catalog"]} >= {"build_tower", "upgrade_tower",
                                                                           "upgrade_power", "sell_tower", "send_wave"})
        # v1 (the default) asks for v1 explicitly and gets the unchanged v1 state.
        v1 = self.env("v1", identity="it_lua_v1").reset()
        self.assertEqual(self.games[-1].environment["ALPHARUSH_ACTION_SCOPE"], "v1")
        self.assertNotIn("powers_ui", v1)
        self.assertNotIn("action_scope", v1)
        self.assertEqual({e["action"] for e in v1["action_catalog"]}, {"build_tower", "send_wave"})
        # v1 offers every basic tower on every unblocked holder (the elf holder too) and ignores locks.
        self.assertEqual({e["holder_id"] for e in v1["action_catalog"] if e["action"] == "build_tower"}, {1, 2, 3, 4, 5})
        # A host that never saw the scope (a pre-v2 host) serves v1: a v2 Worker refuses it.
        self.drop = ("ALPHARUSH_ACTION_SCOPE",)
        with self.assertRaisesRegex(RuntimeError, "Native action scope 'v1' differs from 'v2'"):
            self.env(identity="it_lua_old").reset()

    def test_every_catalog_entry_is_a_menu_option_the_host_accepts(self):
        state = self.env(identity="it_lua_catalog").reset()
        menu = self.assertMenuMatchesCatalog(state)
        kinds = Counter(option["action"]["action"] for option in menu[1:])
        # Four kinds on the four standard holders, the archer_3 branches minus the locked
        # musketeer, both ranger powers, and selling the archer and the ranger.
        self.assertEqual(kinds, {"build_tower": 16, "upgrade_tower": 1, "upgrade_power": 2, "sell_tower": 2,
                                 "send_wave": 1})
        self.assertFalse({5, 6} & {o["action"].get("holder_id") for o in menu})  # elf and blocked holders
        self.assertNotIn(9, {o["action"].get("tower_id") for o in menu})       # the special elf tower
        for index, option in enumerate(menu[1:]):
            with self.subTest(action=option["action"]):
                native = self.env(identity=f"it_lua_accept_{index}")
                native.reset()
                receipt = native.act(option["action"])
                self.assertTrue(receipt["executed"])
                sent = {k: v for k, v in self.games[-1].requests[-2].items() if k not in ("id", "token")}
                self.assertEqual(sent, option["action"])  # Worker.rpc sends exactly the menu action
        # A command the catalog does not hold is refused by the host itself.
        native = self.env(identity="it_lua_refused")
        native.reset()
        for command in ({"action": "build_tower", "holder_id": 5, "tower_type": "archer"},
                        {"action": "upgrade_tower", "tower_id": 7, "target": "tower_musketeer"},
                        {"action": "sell_tower", "tower_id": 9}, {"action": "move_hero", "x": 1, "y": 2}):
            with self.assertRaisesRegex(RuntimeError, "native legal catalog|verified experiment scope"):
                native.worker.rpc(**command)

    def test_scripted_v2_actions_receipts_and_cold_replay(self):
        native = self.env(identity="it_lua_script", cls=RecordingEnv)
        state = native.reset()

        def act(name, **match):
            menu = self.assertMenuMatchesCatalog(native.state)
            action = next(o["action"] for o in menu if o["action"]["action"] == name
                          and all(o["action"].get(k) == v for k, v in match.items()))
            return native.act(action)

        gold = state["gold"]
        build = act("build_tower", holder_id=1, tower_type="archer")
        self.assertEqual((build["expected_cost"], build["gold_delta"], native.state["gold"]), (70, 70, gold - 70))
        archer = build["tower_id"]
        up = act("upgrade_tower", tower_id=7, target="tower_ranger")
        self.assertEqual((up["expected_cost"], up["gold_delta"]), (230, 230))
        ranger = up["tower_id_after"]
        self.assertEqual(next(t for t in native.state["towers"] if t["id"] == ranger)["holder_id"], "7")
        power = act("upgrade_power", tower_id=8, power="poison")
        self.assertEqual((power["power_level_before"], power["power_level_after"], power["gold_delta"]), (0, 1, 250))
        self.assertEqual((power["spent_before"], power["spent_after"]), (570, 820))
        sold = act("sell_tower", tower_id=archer)
        self.assertEqual(sold["gold_delta"], -70)  # full refund before the first wave
        self.assertEqual(next(h for h in native.state["holders"] if h["id"] == sold["holder_id_after"])["mesh_id"], "1")
        # GUI prices: thorn's first level costs price_base 300, the next price_inc 50 (not 300 + 50).
        thorns = [act("upgrade_power", tower_id=ranger, power="thorn") for _ in range(2)]
        self.assertEqual([(r["expected_cost"], r["gold_delta"], r["spent_after"] - r["spent_before"]) for r in thorns],
                         [(300, 300, 300), (50, 50, 50)])
        self.assertEqual(thorns[1]["spent_after"], 340 + 230 + 300 + 50)
        wave = act("send_wave")
        self.assertEqual(native.state["wave"], 1)
        for _ in range(20):
            if len(native.state["enemies"]) >= 2 and any(e["path_progress"] > 0.05 for e in native.state["enemies"]):
                break
            native.advance(30)
        enemies = sorted(native.state["enemies"], key=lambda e: (-e["path_progress"], e["id"]))
        # The anchors come from the bridge's live enemies: the leader first, coordinates rounded.
        anchors = [o["action"] for o in build_menu(native.state)
                   if o["action"]["action"] == "use_power" and o["action"]["power"] == 1]
        self.assertIn(enemies[0]["id"], [a["anchor_id"] for a in anchors])
        for anchor in anchors:
            enemy = next(e for e in enemies if e["id"] == anchor["anchor_id"])
            self.assertEqual((anchor["x"], anchor["y"]), (math.floor(enemy["x"] + 0.5), math.floor(enemy["y"] + 0.5)))
        fire = act("use_power", power=1, anchor_id=enemies[0]["id"])
        self.assertEqual((fire["power_mode_before"], fire["power_mode_after"]), ("ready", "cooldown"))
        reinforce = act("use_power", power=2)
        self.assertEqual(reinforce["power_mode_after"], "cooldown")
        self.assertFalse([o for o in build_menu(native.state) if o["action"]["action"] == "use_power"])
        self.assertTrue(all(r["executed"] for r in (build, up, power, sold, wave, fire, reinforce)))
        for state in native.seen:
            self.assertMenuMatchesCatalog(state)
        # Cold replay in a fresh process gives the same trace.
        plan, trace = json.loads(json.dumps(native.plan)), copy.deepcopy(native.trace)
        replay = self.env(identity="it_lua_script_replay")
        replay.reset()
        replay.replay(plan)
        self.assertEqual(sha256_data(replay.trace), sha256_data(trace))
        self.assertEqual(replay.state, native.state)

    def test_teacher_v2_episode_on_the_real_host_replays_cold(self):
        native = self.env(identity="it_lua_teacher", cls=RecordingEnv)
        result = run_episode(native, make_policy("teacher_v2"), EpisodeProtocol(max_ticks=20000), seed=1001, level=1,
                             episode_id="it-lua-teacher", clock=StepClock())
        self.assertEqual((result["status"], result["void_reason"]), ("terminal", None))
        self.assertTrue(result["outcome"]["level_won"])
        kinds = Counter(d["action"]["action"] for d in result["decisions"])
        for kind in ("build_tower", "upgrade_tower", "upgrade_power", "use_power", "send_wave"):
            self.assertGreater(kinds[kind], 0, kinds)
        self.assertEqual(kinds["sell_tower"], 0)
        for decision in result["decisions"]:
            self.assertEqual(decision["meta"]["rule"], RULES[decision["action"]["action"]])
        self.assertIn("tower_ranger", {d["action"].get("target") for d in result["decisions"]})
        self.assertTrue(all(r["executed"] for r in action_receipts(native.trace)))
        for state in native.seen:
            self.assertMenuMatchesCatalog(state)
        result = json.loads(json.dumps(result, allow_nan=False))
        verdict = replay_verify(lambda: self.env(identity="it_lua_teacher_replay"), result)
        self.assertEqual((verdict["replay_verified"], verdict["replay_error"]), (True, None))


if __name__ == "__main__":
    unittest.main()
