"""elite-v3b: add a DEV-ONLY read-only "tower_info" RPC to the scratch dev copy's host.lua (v4_merged): for each
tower template name, whether its attacks (or its barrack soldiers) can target flying enemies, its range and kind."""
from pathlib import Path

HOST = Path(__file__).parent / "v4_merged" / "alpharush_rl" / "assets" / "host.lua"
s = HOST.read_text(encoding="utf-8")
assert "dev_tower_info" not in s, "already patched"
FUNC = r'''
-- DEV ONLY (elite-v3b probe, scratch copy): can each tower template's attacks reach flying enemies?
local function dev_tower_info(names)
    local edb = require("entity_db")
    local out = {}
    for _, name in ipairs(type(names) == "table" and names or {}) do
        local ok, t = pcall(edb.get_template, edb, name)
        if ok and type(t) == "table" then
            local rec = {name = name, attacks = {}}
            local list = type(t.attacks) == "table" and type(t.attacks.list) == "table" and t.attacks.list or {}
            for i, a in ipairs(list) do
                local bans = type(a) == "table" and a.vis_bans or 0
                local flags = type(a) == "table" and a.vis_flags or 0
                rec.attacks[#rec.attacks+1] = {i = i, bans = finite(bans) and bans or 0, flags = finite(flags) and flags or 0,
                    bans_flying = finite(bans) and type(F_FLYING) == "number" and bit.band(bans, F_FLYING) ~= 0 or false,
                    range = type(a) == "table" and finite(a.range) and a.range or nil}
            end
            rec.has_barrack = type(t.barrack) == "table"
            rec.range = type(t.attacks) == "table" and finite(t.attacks.range) and t.attacks.range or nil
            out[#out+1] = rec
        else
            out[#out+1] = {name = name, missing = true}
        end
    end
    return out
end

local function handle(cmd)'''
old = "\nlocal function handle(cmd)"
assert s.count(old) == 1
s = s.replace(old, FUNC, 1)
old2 = '    elseif cmd.action == "paths" then return dev_paths()'
assert s.count(old2) == 1
s = s.replace(old2, old2 + '\n    elseif cmd.action == "tower_info" then return dev_tower_info(cmd.names)', 1)
HOST.write_text(s, encoding="utf-8", newline="\n")
print("patched")
