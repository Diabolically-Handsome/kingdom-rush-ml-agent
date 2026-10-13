"""elite-v3b geometry probe, third DEV-ONLY host patch (applied after patch_geo_host2.py to the scratch dev copy's
host.lua; never to the real repository). Written up after the run from the session's inline patch commands (same code):

1. tower_info takes its template names as one comma-separated string (the host's request parser refuses arrays);
2. tower_info also lists hero attacks (the melee / ranged / timed_attacks components, which heroes use instead of a
   tower's attacks list), so towers.json says which hero attacks can target flying enemies;
3. a new read-only "template_keys" RPC: each enemy template's in-game string key (info.i18n_key) and whether its
   template is flagged flying; tower_probe.py writes it to enemy_keys.json.
No game data is contained here."""
from pathlib import Path

HOST = Path(__file__).parent / "v4_merged" / "alpharush_rl" / "assets" / "host.lua"
s = HOST.read_text(encoding="utf-8")


def rep(old, new):
    global s
    assert s.count(old) == 1, old[:70]
    s = s.replace(old, new)


# 1. names as a comma-separated string
rep('''    for _, name in ipairs(type(names) == "table" and names or {}) do
        local ok, t = pcall(edb.get_template, edb, name)''',
    '''    for name in string.gmatch(type(names) == "string" and names or "", "[^,]+") do
        local ok, t = pcall(edb.get_template, edb, name)''')
# 2. hero attack components
rep('''            rec.has_barrack = type(t.barrack) == "table"''',
    '''            for _, comp in ipairs({"melee", "ranged", "timed_attacks"}) do
                local c = type(t[comp]) == "table" and t[comp] or nil
                local al = c and type(c.attacks) == "table" and c.attacks or {}
                for i, a in ipairs(al) do
                    local bans = type(a) == "table" and a.vis_bans or 0
                    rec.attacks[#rec.attacks+1] = {i = comp .. ":" .. i,
                        bans_flying = finite(bans) and type(F_FLYING) == "number" and bit.band(bans, F_FLYING) ~= 0 or false,
                        range = type(a) == "table" and (finite(a.max_range) and a.max_range or finite(a.range) and a.range) or nil}
                end
            end
            rec.has_barrack = type(t.barrack) == "table"''')
# 3. template_keys RPC
rep('''
local function handle(cmd)''', r'''
-- DEV ONLY (elite-v3b probe, scratch copy): each enemy template's info i18n key (its in-game name/card strings).
local function dev_template_keys(names)
    local edb = require("entity_db")
    local out = {}
    for name in string.gmatch(type(names) == "string" and names or "", "[^,]+") do
        local ok, t = pcall(edb.get_template, edb, name)
        local info = ok and type(t) == "table" and type(t.info) == "table" and t.info or nil
        local vis = ok and type(t) == "table" and type(t.vis) == "table" and t.vis or {}
        out[#out+1] = {name = name, key = info and info.i18n_key or nil,
                       flying = type(F_FLYING) == "number" and finite(vis.flags) and bit.band(vis.flags, F_FLYING) ~= 0 or false}
    end
    return out
end

local function handle(cmd)''')
rep('''    elseif cmd.action == "tower_info" then return dev_tower_info(cmd.names)''',
    '''    elseif cmd.action == "tower_info" then return dev_tower_info(cmd.names)
    elseif cmd.action == "template_keys" then return dev_template_keys(cmd.names)''')
HOST.write_text(s, encoding="utf-8", newline="\n")
print("patched", HOST)
