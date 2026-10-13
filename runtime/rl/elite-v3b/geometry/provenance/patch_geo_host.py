"""elite-v3b geometry probe: add a DEV-ONLY read-only "paths" RPC to the scratch dev copy's host.lua (v4_merged).
Never applied to the real repository. Returns the static map a player sees: every path's lanes (node x/y), the
path connections, level exits, the custom hero spawn point, lane-1 ice flags and the level location keys."""
from pathlib import Path

HOST = Path(__file__).parent / "v4_merged" / "alpharush_rl" / "assets" / "host.lua"
s = HOST.read_text(encoding="utf-8")
assert "dev_paths" not in s, "already patched"
FUNC = r'''
-- DEV ONLY (elite-v3b geometry probe, scratch copy): the static level geometry a player sees on the map.
local function dev_paths()
    local P, GR = package.loaded["path_db"], package.loaded["grid_db"]
    local store = game and game.store
    if type(P) ~= "table" or type(P.paths) ~= "table" then return {error = "no path_db"} end
    local out = {paths = {}, connections = {}, exits = {}, ice = {}}
    for _, path in ipairs(P.paths) do
        local lanes = {}
        for _, lane in ipairs(type(path) == "table" and path or {}) do
            local nodes = {}
            for _, n in ipairs(type(lane) == "table" and lane or {}) do
                if type(n) == "table" and finite(n.x) and finite(n.y) then nodes[#nodes+1] = {n.x, n.y} end
            end
            lanes[#lanes+1] = nodes
        end
        out.paths[#out.paths+1] = lanes
        local ice = {}
        for _, n in ipairs(type(path) == "table" and type(path[1]) == "table" and path[1] or {}) do
            local on = false
            if type(GR) == "table" and type(TERRAIN_ICE) == "number" and type(n) == "table" then
                local ok, v = pcall(GR.cell_is, GR, n.x, n.y, TERRAIN_ICE)
                on = (ok and v) and true or false
            end
            ice[#ice+1] = on
        end
        out.ice[#out.ice+1] = ice
    end
    for from, to in pairs(type(P.path_connections) == "table" and P.path_connections or {}) do
        out.connections[#out.connections+1] = {from, to}
    end
    for _, ex in ipairs(level_exits(store)) do out.exits[#out.exits+1] = {k = ex.k, x = ex.x, y = ex.y} end
    local level = store and store.level
    local csp = type(level) == "table" and level.custom_spawn_pos
    if type(csp) == "table" and finite(csp.x) and finite(csp.y) then out.custom_spawn = {csp.x, csp.y} end
    local loc = type(level) == "table" and level.locations
    if type(loc) == "table" then
        local keys = {}
        for k, v in pairs(loc) do keys[#keys+1] = tostring(k) .. ":" .. type(v) end
        out.location_keys = keys
    end
    return out
end

local function handle(cmd)'''
old = "\nlocal function handle(cmd)"
assert s.count(old) == 1
s = s.replace(old, FUNC, 1)
old2 = '    elseif cmd.action == "rng_audit" then'
assert s.count(old2) == 1
s = s.replace(old2, '    elseif cmd.action == "paths" then return dev_paths()\n' + old2, 1)
HOST.write_text(s, encoding="utf-8", newline="\n")
print("patched", HOST)
