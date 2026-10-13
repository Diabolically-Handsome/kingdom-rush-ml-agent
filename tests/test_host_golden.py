"""Golden transcripts of host.lua action scopes v2, v3 and v3 + timed_clicks (the elite-v2 final's configuration),
recorded on 2026-10-11 from the host the elite-v2 final ran (host.lua sha256 9383decc9008d28f...) before action
scope v4 was added. Every later host change must keep them byte-identical: a v4 addition may not change what the
earlier scopes offer, accept, refuse or do. Runs in the game's own LuaJIT; no game, no sockets."""
from __future__ import annotations

import json
import unittest

try:
    from test_host_actions_v2 import GROUND, HOST, LUA_DLL, SETUP, WORLD, Lua, as_list, native_state
    from test_host_actions_v3 import ELITE
except ImportError:  # run as tests.test_host_golden from the workspace root
    from tests.test_host_actions_v2 import GROUND, HOST, LUA_DLL, SETUP, WORLD, Lua, as_list, native_state
    from tests.test_host_actions_v3 import ELITE

import hashlib
import os

ENV_LINE = 'if scope ~= "<unset>" then ENV.ALPHARUSH_ACTION_SCOPE = scope end'
assert SETUP.count(ENV_LINE) == 1
SETUP_FEATURES = SETUP.replace(ENV_LINE, ENV_LINE + "\nENV.ALPHARUSH_HOST_FEATURES = GOLDEN_HOST_FEATURES")

# Clickable entities: J.T.'s ice on a tower (no click window), J.T. knocked down, two Vez'nan holds with a click
# window (prompt sprite 2 shown on 600, hidden on 601), a sheep and a yeti.
CLICKS = r"""
ENTITIES[500] = {template_name = "mod_jt_tower", pos = {x = 10.4, y = 20.6}, ui = {can_click = true},
                 modifier = {target_id = 20}, required_clicks = 3}
ENTITIES[501] = {template_name = "eb_jt", pos = {x = 332.46, y = 395}, ui = {can_click = true}, dying = true,
                 tap_decal = "decal_jt_tap", health = {hp = 4, dead = false}}
ENTITIES[502] = {template_name = "decal_sheep", pos = {x = 50, y = 50}, ui = {can_click = true}, required_clicks = 5}
ENTITIES[503] = {template_name = "enemy_yeti", pos = {x = 60, y = 60}, ui = {can_click = true},
                 health = {hp = 300, dead = false}}
ENTITIES[600] = {template_name = "mod_veznan_tower", pos = {x = 70, y = 80}, ui = {can_click = true},
                 modifier = {target_id = 21}, required_clicks = 3, click_time = 4,
                 render = {sprites = {{hidden = false}, {hidden = false}}}}
ENTITIES[601] = {template_name = "mod_veznan_tower", pos = {x = 90, y = 80}, ui = {can_click = true},
                 modifier = {target_id = 22}, required_clicks = 3, click_time = 4,
                 render = {sprites = {{hidden = false}, {hidden = true}}}}
"""
PROBES = [
    '{"id":"p1","action":"move_hero","hero_id":7,"x":100,"y":200}',
    '{"id":"p2","action":"aim_power","power":1,"x":100,"y":201,"option":"cluster"}',
    '{"id":"p3","action":"no_such_action"}',
    '{"id":"p4","action":"use_power","power":1,"x":1,"y":1,"anchor_id":999}',
]
GOLDEN = {
    "v2": "d331e60dcaf6cc4572a0e105d71a20c5d020d1d3da604dcfa138eedef22de45e",
    "v3": "264acb5137a4f4c7d57e8750834554a787d117b7fa68d9928332973f16da9288",
    "v3+timed_clicks": "201bf66a20f99ceb70dbbe0732d6dbcc1c22af692bc28e3b78277acb9181b622",
}


def load(scope, features, elite):
    lua = Lua()
    if features is None:
        lua.run(SETUP, str(HOST).encode("utf-8"), scope.encode("utf-8"))
    else:
        lua.run(f"GOLDEN_HOST_FEATURES = {json.dumps(features)}")
        lua.run(SETUP_FEATURES, str(HOST).encode("utf-8"), scope.encode("utf-8"))
    lua.run(WORLD)
    lua.run(GROUND)
    if elite:
        lua.run(ELITE)
    lua.run(CLICKS)
    return lua


def transcript(lua):
    """state; every offered action executed in catalog order; refused probes; state; native calls; game."""
    out = []

    def call(line):
        out.append(line + "\n" + lua.call("rpc", line.encode("utf-8")))

    call('{"id":"s0","action":"state"}')
    catalog = as_list(native_state(lua).get("action_catalog", {}))
    for index, item in enumerate(catalog):
        request = {k: v for k, v in item.items() if k not in ("available", "legal", "cost")}
        request["id"] = f"a{index}"
        call(json.dumps(request, sort_keys=True, separators=(",", ":")))
    for line in PROBES:
        call(line)
    call('{"id":"s1","action":"state"}')
    out.append(lua.call("calls_dump"))
    out.append(lua.call("game_dump"))
    return out


def golden_sha256(name):
    scope, _, features = name.partition("+")
    lua = load(scope, features or None, elite=scope == "v3")
    try:
        return hashlib.sha256("\n\0\n".join(transcript(lua)).encode("utf-8")).hexdigest()
    finally:
        lua.close()


@unittest.skipUnless(os.name == "nt" and LUA_DLL.exists(), "game LuaJIT runtime/rl-engine/lua51.dll unavailable")
class HostGoldenTranscriptTests(unittest.TestCase):
    def test_scopes_v2_v3_and_timed_clicks_are_byte_identical_to_the_elite_v2_host(self):
        for name, expected in GOLDEN.items():
            with self.subTest(name=name):
                self.assertEqual(expected, golden_sha256(name))


if __name__ == "__main__":  # print the transcript hashes (used once to record GOLDEN)
    for key in GOLDEN:
        print(key, golden_sha256(key))
