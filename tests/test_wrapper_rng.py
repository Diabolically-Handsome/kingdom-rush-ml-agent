"""wrapper.lua RNG modes, run inside the game's own LuaJIT with stubbed LÖVE (no game, no process)."""
from __future__ import annotations

import os
from pathlib import Path
import tempfile
import unittest

from test_host_protocol import LUA_DLL, Lua

ROOT = Path(__file__).resolve().parents[1]
WRAPPER = ROOT / "alpharush_rl/assets/wrapper.lua"

# Stub LÖVE just enough for wrapper.lua: the original main and host are no-ops,
# and newRandomGenerator is a tiny LCG so isolated sound draws are observable.
SETUP = r"""
local wrapper_path, host_path, mode = ...
local real_getenv = os.getenv
os.getenv = function(name)
    if name == "ALPHARUSH_RNG_MODE" then return mode end
    if name == "ALPHARUSH_HOST_FILE" then return host_path end
    if name == "ALPHARUSH_SEED" then return "1002" end
    return real_getenv(name)
end
print = function() end
local function generator(seed)
    local state = seed
    return {random = function(self, a, b)
        state = (state * 1103515245 + 12345) % 2147483648
        local x = state / 2147483648
        if a == nil then return x end
        if b == nil then return 1 + math.floor(x * a) end
        return a + math.floor(x * (b - a + 1))
    end}
end
love = {filesystem = {setIdentity = function() end, load = function() return function() end end},
        math = {setRandomSeed = function() end, newRandomGenerator = generator,
                random = function() return 0.5 end},
        audio = {setVolume = function() end}, update = function() end}
package.preload["features"] = function() return {platform_services = {}} end
ORIGINAL_RANDOM = math.random
dofile(wrapper_path)
-- Callers named like the game's chunks; only the source name matters to the dispatcher.
-- Not tail calls: a tail call drops the caller's frame, as in the game's own code paths
-- (sound_db uses random() inside comparisons and assignments).
local function chunk(name, code) return assert(loadstring(code, name))() end
GAME_DRAW = chunk("@all/systems.lua", "return function() local x = math.random(1, 1000000); return x end")
SOUND_DRAW = chunk("@all/sound_db.lua", "return function() local x = math.random(); return x end")
function draws(pattern)
    local out = {}
    for c in pattern:gmatch(".") do
        if c == "g" then out[#out + 1] = tostring(GAME_DRAW()) else SOUND_DRAW() end
    end
    return table.concat(out, ",")
end
function reference(n)
    math.randomseed(1002)
    local out = {}
    for i = 1, n do out[#out + 1] = tostring(ORIGINAL_RANDOM(1, 1000000)) end
    return table.concat(out, ",")
end
function state()
    local rng = ALPHARUSH_RNG
    if not rng then return "none|" .. tostring(math.random == ORIGINAL_RANDOM) end
    local keys = {}
    for k, v in pairs(rng.counts) do keys[#keys + 1] = k .. "=" .. v end
    table.sort(keys)
    return rng.mode .. "|" .. table.concat(keys, ";")
end
"""


@unittest.skipUnless(os.name == "nt" and LUA_DLL.exists(), "game LuaJIT runtime/rl-engine/lua51.dll unavailable")
class WrapperRngTests(unittest.TestCase):
    def load(self, mode):
        tmp = tempfile.TemporaryDirectory(prefix="alpharush-wrapper-")
        self.addCleanup(tmp.cleanup)
        host = Path(tmp.name) / "host.lua"
        host.write_text("return {installed=false, install=function() end, update=function() end}\n",
                        encoding="utf-8")
        lua = Lua()
        self.addCleanup(lua.close)
        lua.run(SETUP, str(WRAPPER).encode(), str(host).encode(), mode.encode())
        return lua

    def test_default_mode_keeps_the_original_random(self):
        lua = self.load("")
        self.assertEqual(lua.call("state"), "none|true")
        # Sound and game draws share one stream, exactly as in the unmodified game.
        lua.run("math.randomseed(1002)")
        mixed = lua.call("draws", b"gsgsgg").split(",")
        reference = lua.call("reference", b"6").split(",")
        self.assertEqual(mixed, [reference[0], reference[2], reference[4], reference[5]])

    def test_audit_counts_callers_without_changing_the_stream(self):
        lua = self.load("audit")
        lua.run("math.randomseed(1002)")
        mixed = lua.call("draws", b"gsgsgg").split(",")
        reference = lua.call("reference", b"6").split(",")
        self.assertEqual(mixed, [reference[0], reference[2], reference[4], reference[5]])
        mode, counts = lua.call("state").split("|")
        self.assertEqual(mode, "audit")
        counts = dict(item.split("=") for item in counts.split(";"))
        self.assertEqual(counts, {"math@@all/sound_db.lua:1": "2", "math@@all/systems.lua:1": "4"})

    def test_isolated_sound_never_consumes_the_gameplay_stream(self):
        lua = self.load("isolate_sound")
        lua.run("math.randomseed(1002)")
        game = lua.call("draws", b"gssssgsgsg").split(",")
        reference = lua.call("reference", b"4").split(",")
        self.assertEqual(game, reference)  # however many sound draws happen in between
        mode, counts = lua.call("state").split("|")
        self.assertEqual((mode, counts), ("isolate_sound", ""))

    def test_audit_and_isolation_together(self):
        lua = self.load("audit+isolate_sound")
        lua.run("math.randomseed(1002)")
        self.assertEqual(lua.call("draws", b"gsssg").split(","), lua.call("reference", b"2").split(","))
        counts = dict(item.split("=") for item in lua.call("state").split("|")[1].split(";"))
        self.assertEqual(counts, {"math@@all/sound_db.lua:1": "3", "math@@all/systems.lua:1": "2"})


STABLE = r"""
-- Coroutines created a, b, c; inserted into a table in a different order.
local a = coroutine.create(function() end)
local b = coroutine.create(function() end)
local c = coroutine.create(function() end)
NAMES = {}
NAMES[a], NAMES[b], NAMES[c] = "a", "b", "c"
function thread_order()
    local t = {}
    t[c] = 3; t[a] = 1; t[b] = 2
    local out = {}
    for key, value in pairs(t) do out[#out + 1] = NAMES[key] .. value end
    return table.concat(out, ",")
end
function removal_during_loop()
    local t = {}
    t[c] = 3; t[a] = 1; t[b] = 2
    local out = {}
    for key, value in pairs(t) do
        out[#out + 1] = NAMES[key]
        if NAMES[key] == "a" then t[b] = nil end
    end
    return table.concat(out, ",")
end
function id_order()
    local t = {}
    for _, id in ipairs({42, 7, 19}) do t[{id = id}] = id end
    local out = {}
    for key in pairs(t) do out[#out + 1] = tostring(key.id) end
    return table.concat(out, ",")
end
function mixed_fallback()
    local t = {}
    t[{id = 2}] = true; t[{name = "no id"}] = true
    local n = 0
    for _ in pairs(t) do n = n + 1 end
    return tostring(n)
end
function primitive_keys()
    local t = {x = 1, y = 2, [3] = 3}
    local f = pairs(t)
    local n = 0
    for _ in pairs(t) do n = n + 1 end
    return tostring(f == next) .. "," .. n
end
"""


@unittest.skipUnless(os.name == "nt" and LUA_DLL.exists(), "game LuaJIT runtime/rl-engine/lua51.dll unavailable")
class StablePairsTests(WrapperRngTests):
    def stable(self, mode="isolate_sound+stable_pairs"):
        lua = self.load(mode)
        lua.run(STABLE)
        return lua

    def test_coroutine_keys_iterate_in_creation_order(self):
        lua = self.stable()
        self.assertEqual(lua.call("thread_order"), "a1,b2,c3")
        self.assertEqual(lua.call("removal_during_loop"), "a,c")

    def test_object_keys_with_ids_sort_by_id_and_others_fall_back(self):
        lua = self.stable()
        self.assertEqual(lua.call("id_order"), "7,19,42")
        self.assertEqual(lua.call("mixed_fallback"), "2")
        self.assertEqual(lua.call("primitive_keys"), "true,3")

    def test_tokens_combine_and_default_leaves_pairs_untouched(self):
        lua = self.stable("audit+isolate_sound+stable_pairs")
        self.assertEqual(lua.call("thread_order"), "a1,b2,c3")
        lua.run("math.randomseed(1002)")
        self.assertEqual(lua.call("draws", b"gsg").split(","), lua.call("reference", b"2").split(","))
        plain = self.load("isolate_sound")
        plain.run("return 0")
        self.assertEqual(plain.call("state").split("|")[0], "isolate_sound")
        plain.run(STABLE)
        # Without stable_pairs the original pairs is kept (order is the VM's own).
        self.assertEqual(sorted(plain.call("thread_order").split(",")), ["a1", "b2", "c3"])


if __name__ == "__main__":
    unittest.main()
