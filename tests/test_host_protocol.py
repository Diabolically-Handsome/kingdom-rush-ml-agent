"""host.lua request decoding and meta, run inside the game's own LuaJIT (no game, no sockets)."""
from __future__ import annotations

import ctypes
import os
from pathlib import Path
import re
import unittest

ROOT = Path(__file__).resolve().parents[1]
HOST = ROOT / "alpharush_rl/assets/host.lua"
LUA_DIR = ROOT / "runtime/rl-engine"
LUA_DLL = LUA_DIR / "lua51.dll"
LUA_GLOBALSINDEX = -10002

# Test-side harness: stub modules, load host.lua, and flatten Lua values into
# sorted "path=type:value" lines (keys/strings hex-encoded) for Python to compare.
SETUP = r"""
local host_path, test_token = ...
print = function() end
local real_getenv = os.getenv
os.getenv = function(name)
    if name == "ALPHARUSH_TOKEN" then return test_token end
    return real_getenv(name)
end
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
    else out[#out + 1] = prefix .. "=" .. t .. ":" .. tostring(value) end
    return out
end
function dump(value)
    local out = flatten(value, "", {})
    table.sort(out)
    return table.concat(out, "\n")
end
SENT, INBOX, CONNECTED, CLOSED, PARTIAL, SEND_FAIL = {}, {}, false, false, nil, false
local client = {
    settimeout = function() end, close = function() CLOSED = true; CONNECTED = false end,
    send = function(_, payload)
        if SEND_FAIL then return nil, "closed" end
        return #payload
    end,
    receive = function()
        local line = table.remove(INBOX, 1)
        if line then return line end
        local partial = PARTIAL or ""
        PARTIAL = nil
        return nil, "timeout", partial
    end,
}
local server = {settimeout = function() end,
                accept = function() if CONNECTED then return nil end; CONNECTED = true; CLOSED = false; return client end}
package.preload["socket"] = function() return {bind = function() return server end} end
package.preload["alpha_native_bridge"] = function()
    return {codec = {encode = function(value) SENT[#SENT + 1] = value; return "{}" end}}
end
HOST = dofile(host_path)
function probe(line)
    local value, err = HOST._decode_request(line)
    if value == nil then return "ERR|" .. tostring(err) end
    return "OK|" .. dump(value)
end
function meta_dump() return dump(HOST._meta()) end
function game_dump() return dump(game) end
function pump()
    if not HOST.installed then HOST.install(function() end) end
    local before = #SENT
    HOST.update(1 / 60)
    if #SENT ~= before + 1 then return "NOREPLY" end
    return dump(SENT[#SENT])
end
function rpc(line)
    INBOX[#INBOX + 1] = line
    return pump()
end
function closed() return tostring(CLOSED) end
"""


def undump(text):
    root = {}
    for line in text.splitlines():
        path, rest = line.split("=", 1)
        kind, raw = rest.split(":", 1)
        keys = [int(p[1:]) if p.startswith("#") else bytes.fromhex(p).decode("utf-8")
                for p in path.split("/")[1:]]
        value = {"string": lambda: bytes.fromhex(raw).decode("utf-8"), "number": lambda: float(raw),
                 "boolean": lambda: raw == "true", "table": dict}[kind]()
        node = root
        for key in keys[:-1]:
            node = node.setdefault(key, {})
        if keys:
            node[keys[-1]] = value
    return root


class Lua:
    """Minimal ctypes wrapper over the game's LuaJIT 2.0.4 (read-only use)."""

    def __init__(self):
        self._dir = os.add_dll_directory(str(LUA_DIR))
        lib = self.lib = ctypes.CDLL(str(LUA_DLL))
        p, i = ctypes.c_void_p, ctypes.c_int
        lib.luaL_newstate.restype = p
        lib.luaL_openlibs.argtypes = [p]
        lib.luaL_loadstring.argtypes = [p, ctypes.c_char_p]
        lib.lua_pcall.argtypes = [p, i, i, i]
        lib.lua_tolstring.argtypes = [p, i, ctypes.POINTER(ctypes.c_size_t)]
        lib.lua_tolstring.restype = p
        lib.lua_pushlstring.argtypes = [p, ctypes.c_char_p, ctypes.c_size_t]
        lib.lua_getfield.argtypes = [p, i, ctypes.c_char_p]
        lib.lua_settop.argtypes = [p, i]
        lib.lua_close.argtypes = [p]
        self.L = lib.luaL_newstate()
        lib.luaL_openlibs(self.L)

    def _string(self, index):
        size = ctypes.c_size_t()
        pointer = self.lib.lua_tolstring(self.L, index, ctypes.byref(size))
        return None if not pointer else ctypes.string_at(pointer, size.value)

    def _check(self, status):
        if status != 0:
            message = self._string(-1)
            self.lib.lua_settop(self.L, 0)
            raise RuntimeError((message or b"lua error").decode("utf-8", "replace"))

    def run(self, code, *args):
        self._check(self.lib.luaL_loadstring(self.L, code.encode("utf-8")))
        for arg in args:
            self.lib.lua_pushlstring(self.L, arg, len(arg))
        self._check(self.lib.lua_pcall(self.L, len(args), 0, 0))

    def call(self, name, *args):
        self.lib.lua_getfield(self.L, LUA_GLOBALSINDEX, name.encode())
        for arg in args:
            self.lib.lua_pushlstring(self.L, arg, len(arg))
        self._check(self.lib.lua_pcall(self.L, len(args), 1, 0))
        result = self._string(-1)
        self.lib.lua_settop(self.L, 0)
        return result.decode("utf-8")

    def close(self):
        self.lib.lua_close(self.L)
        self._dir.close()


@unittest.skipUnless(os.name == "nt" and LUA_DLL.exists(), "game LuaJIT runtime/rl-engine/lua51.dll unavailable")
class HostProtocolTests(unittest.TestCase):
    token = b""

    def setUp(self):
        self.lua = Lua()
        self.addCleanup(self.lua.close)
        self.lua.run(SETUP, str(HOST).encode("utf-8"), self.token)

    def decode(self, line):
        if isinstance(line, str):
            line = line.encode("utf-8")
        result = self.lua.call("probe", line)
        status, _, body = result.partition("|")
        return (undump(body), None) if status == "OK" else (None, body)

    def assertRejected(self, line, fragment=""):
        value, error = self.decode(line)
        self.assertIsNone(value, f"accepted {line!r}")
        self.assertIn(fragment, error)
        return error

    def test_valid_flat_objects(self):
        value, error = self.decode('{"id":"7","action":"step","ticks":60,"neg":-3,"frac":0.25,'
                                   '"exp":1.5e3,"small":-2E-2,"zero":0,"yes":true,"no":false,"gone":null}')
        self.assertIsNone(error)
        self.assertEqual(value, {"id": "7", "action": "step", "ticks": 60, "neg": -3, "frac": 0.25,
                                 "exp": 1500, "small": -0.02, "zero": 0, "yes": True, "no": False})
        value, _ = self.decode(' \t{ "holder_id" : 20 , "tower_type" : "archer" } \t')
        self.assertEqual(value, {"holder_id": 20, "tower_type": "archer"})
        self.assertEqual(self.decode("{}")[0], {})

    def test_string_escapes_unicode_and_surrogates(self):
        value, error = self.decode(r'{"s":"a\"b\\c\/d\b\f\n\r\t","cjk":"中文","pair":"😀","nul":"x\u0000y"}')
        self.assertIsNone(error)
        self.assertEqual(value["s"], 'a"b\\c/d\b\f\n\r\t')
        self.assertEqual(value["cjk"], "中文")
        self.assertEqual(value["pair"], "\U0001F600")
        self.assertEqual(value["nul"], "x\x00y")
        self.assertRejected(r'{"s":"\ud83d"}', "surrogate")
        self.assertRejected(r'{"s":"\ude00"}', "surrogate")
        self.assertRejected(r'{"s":"\ud83dA"}', "surrogate")
        self.assertRejected(r'{"s":"\x41"}', "escape")
        self.assertRejected(r'{"s":"\u12G4"}', "unicode")

    def test_rejects_code_and_structure(self):
        self.assertRejected('{"action":"x"}os.exit(1)', "trailing")
        self.assertRejected("os.exit(1)", "JSON object")
        self.assertRejected('{"action":"state","x":(function() os.exit(1) end)()}', "invalid value")
        self.assertRejected('{["action"]="state"}', "string key")
        self.assertRejected('{"a":{"b":1}}', "nested")
        self.assertRejected('{"a":[1,2]}', "nested")
        self.assertRejected('{"a":1,"a":2}', "duplicate")
        self.assertRejected('{"a":null,"a":2}', "duplicate")
        self.assertRejected('{"a":"abc', "unterminated")
        self.assertRejected('{"a":"x\ny"}', "control")
        self.assertRejected('{"a":"x\x01y"}', "control")
        self.assertRejected('{"a":1', "comma or closing")
        self.assertRejected('{"a":1,}', "string key")
        self.assertRejected('{"a" 1}', "colon")
        self.assertRejected('{"a":1}{"b":2}', "trailing")
        self.assertRejected("{'a':1}", "string key")
        for top in ("[1]", '"text"', "1", "true", "null", "", "   "):
            self.assertRejected(top, "JSON object")

    def test_rejects_non_json_numbers_and_literals(self):
        for bad in ("01", "+1", "1.", ".5", "1e", "1e+", "-", "0x10", "1.5.2", "Infinity", "NaN", "tru", "nil"):
            self.assertRejected('{"a":%s}' % bad)
        self.assertRejected('{"a":1e999}', "out of range")

    def test_rejects_overlong_line(self):
        filler = "x" * 4096
        self.assertRejected('{"a":"%s"}' % filler, "4096")
        limit = '{"a":"%s"}' % ("y" * (4096 - 8))
        self.assertEqual(len(limit), 4096)
        self.assertEqual(self.decode(limit)[0], {"a": "y" * 4088})

    def test_host_never_compiles_requests(self):
        source = HOST.read_text(encoding="utf-8")
        self.assertNotIn("loadstring", source)
        self.assertNotIn("loadfile", source)
        self.assertNotIn("dofile", source)
        self.assertIsNone(re.search(r"(?<![A-Za-z0-9_])load\s*\(", source))
        self.assertNotIn("native.codec.decode", source)

    def test_meta_is_read_only_and_guarded(self):
        self.lua.run(r"""
game = {store = {level_idx = 3, level_mode = 1, level_difficulty = 2,
                 level = {locked_towers = {"tower_mage_3", "tower_engineer_3"}, locked_powers = {2},
                          locked_hero = true, max_upgrade_level = 2},
                 entities = {[10] = {hero = {level = 1}}, [11] = {enemy = {}}, [12] = {hero = {}}}}}
package.loaded["game_settings"] = {main_campaign_levels = 12}
""")
        before = self.lua.call("game_dump")
        meta = undump(self.lua.call("meta_dump"))
        self.assertEqual(self.lua.call("game_dump"), before)
        self.assertEqual(meta["level_idx"], 3)
        self.assertEqual(meta["level_mode"], 1)
        self.assertEqual(meta["level_difficulty"], 2)
        self.assertEqual(meta["locked_towers"], {1: "tower_mage_3", 2: "tower_engineer_3"})
        self.assertEqual(meta["locked_powers"], {1: 2})
        self.assertIs(meta["locked_hero"], True)
        self.assertEqual(meta["max_upgrade_level"], 2)
        self.assertEqual(meta["main_campaign_levels"], 12)
        self.assertEqual(meta["hero_count"], 2)
        self.assertEqual(meta["errors"], {})

    def test_meta_reports_unavailable_fields_instead_of_failing(self):
        self.lua.run(r"""
game = {store = {level_idx = 1, entities = {},
                 level = setmetatable({}, {__index = function(_, key) error("boom " .. key) end})}}
""")
        meta = undump(self.lua.call("meta_dump"))
        self.assertEqual(meta["level_idx"], 1)
        self.assertEqual(meta["hero_count"], 0)
        self.assertNotIn("locked_towers", meta)
        self.assertNotIn("main_campaign_levels", meta)
        errors = " | ".join(meta["errors"].values())
        for fragment in ("locked_towers: ", "boom locked_towers", "level_difficulty", "main_campaign_levels",
                         "kr1.game_settings"):
            self.assertIn(fragment, errors)
        self.lua.run("game = nil")
        meta = undump(self.lua.call("meta_dump"))
        self.assertIn("no native level is loaded", " | ".join(meta["errors"].values()))

    def test_meta_never_requires_unloaded_modules(self):
        # require() would run module code and leave a failure sentinel for the game's own require.
        self.lua.run(r"""
game = {store = {level_idx = 1, entities = {}}}
LOADER_RUNS = 0
package.preload["game_settings"] = function() LOADER_RUNS = LOADER_RUNS + 1; error("not ready") end
""")
        meta = undump(self.lua.call("meta_dump"))
        self.assertNotIn("main_campaign_levels", meta)
        self.assertIn("not loaded by the game", " | ".join(meta["errors"].values()))
        self.lua.run(r"""
assert(LOADER_RUNS == 0, "meta executed a module loader")
assert(package.loaded["game_settings"] == nil, "meta left a package.loaded entry")
""")

    def test_rpc_replies_to_malformed_requests_and_keeps_whitelist(self):
        reply = undump(self.lua.call("rpc", b'{"id":"1","action":'))
        self.assertNotIn("id", reply)
        self.assertIs(reply["ok"], False)
        self.assertTrue(reply["error"].startswith("Malformed request: "))
        # A malformed client is answered once and then disconnected.
        self.assertEqual(self.lua.call("closed"), "true")
        reply = undump(self.lua.call("rpc", b'{"id":"2","action":"eval","code":"return 1"}'))
        self.assertEqual(reply["id"], "2")
        self.assertIs(reply["ok"], False)
        self.assertIn("Action not enabled", reply["error"])
        reply = undump(self.lua.call("rpc", b'{"id":"3","action":"meta"}'))
        self.assertEqual(reply["id"], "3")
        self.assertIs(reply["ok"], True)
        self.assertIn("errors", reply["result"])

    def test_failed_reply_drops_client_without_escaping_the_game_loop(self):
        self.lua.run("SEND_FAIL = true")
        self.lua.call("rpc", b'{"id":"5","action":"meta"}')
        self.assertEqual(self.lua.call("closed"), "true")
        self.lua.call("rpc", b'garbage')
        self.assertEqual(self.lua.call("closed"), "true")
        self.lua.run("SEND_FAIL = false")
        reply = undump(self.lua.call("rpc", b'{"id":"6","action":"meta"}'))
        self.assertEqual(reply["id"], "6")

    def test_unsafe_ids_are_not_echoed(self):
        for line in (b'{"id":"\\b","action":"meta"}', b'{"id":"\\u0001","action":"meta"}',
                     b'{"id":"' + b"x" * 65 + b'","action":"meta"}', b'{"id":"a b","action":"meta"}'):
            reply = undump(self.lua.call("rpc", line))
            self.assertNotIn("id", reply, line)
            self.assertIs(reply["ok"], True)
        reply = undump(self.lua.call("rpc", b'{"id":"run-7:a.b_c","action":"meta"}'))
        self.assertEqual(reply["id"], "run-7:a.b_c")

    def test_unterminated_oversized_buffer_is_answered_and_dropped(self):
        self.lua.run("PARTIAL = ...", b'{"id":"4","action":"state","pad":"' + b"z" * 4000)
        self.assertEqual(self.lua.call("pump"), "NOREPLY")
        self.assertEqual(self.lua.call("closed"), "false")
        self.lua.run("PARTIAL = ...", b"z" * 200)
        reply = undump(self.lua.call("pump"))
        self.assertIs(reply["ok"], False)
        self.assertIn("exceeds 4096", reply["error"])
        self.assertEqual(self.lua.call("closed"), "true")


class HostTokenTests(HostProtocolTests):
    """Same protocol behaviour with a per-launch token, plus the token gate itself."""
    token = b"secret-token"

    def rpc(self, line):
        return undump(self.lua.call("rpc", line))

    def test_rpc_replies_to_malformed_requests_and_keeps_whitelist(self):
        self.skipTest("covered without token; token gate tested below")

    def test_failed_reply_drops_client_without_escaping_the_game_loop(self):
        self.skipTest("covered without token")

    def test_unsafe_ids_are_not_echoed(self):
        self.skipTest("covered without token")

    def test_requests_without_the_token_are_refused_and_dropped(self):
        for line in (b'{"id":"1","action":"meta"}', b'{"id":"1","action":"meta","token":"wrong"}',
                     b'{"id":"1","action":"meta","token":7}'):
            reply = self.rpc(line)
            self.assertIs(reply["ok"], False, line)
            self.assertEqual(reply["error"], "Unauthorized request")
            self.assertNotIn("id", reply)
            self.assertEqual(self.lua.call("closed"), "true")
        reply = self.rpc(b'{"id":"2","action":"meta","token":"secret-token"}')
        self.assertEqual(reply["id"], "2")
        self.assertIs(reply["ok"], True)
        reply = self.rpc(b'{"id":"3","action":"eval","code":"return 1","token":"secret-token"}')
        self.assertIn("Action not enabled", reply["error"])


if __name__ == "__main__":
    unittest.main()
