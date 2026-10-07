local socket = require("socket")
local native = require("alpha_native_bridge")
local M = {installed = false}
local server, original_update
local clients = {}
local controlled = false
local MAX_REQUEST_BYTES = 4096
-- Per-launch secret from the parent Worker; requests without it are refused.
local TOKEN = os.getenv("ALPHARUSH_TOKEN")
-- "v1" keeps the original build/send scope byte for byte; "v2" adds upgrades, tower
-- powers, selling and spells; "v3" (elite stages) is v2 plus barracks rally points, dormant /
-- untargetable / boss enemy flags and per-lane path progress. Any other value fails closed at install.
local ACTION_SCOPE = os.getenv("ALPHARUSH_ACTION_SCOPE") or "v1"

-- Strict flat JSON object decoder for RPC requests; it never compiles input.
-- Values: string, number, true, false, null (null = absent). Returns table or nil, err.
local function decode_request(line)
    if type(line) ~= "string" then return nil, "request is not a string" end
    if #line > MAX_REQUEST_BYTES then return nil, "request exceeds " .. MAX_REQUEST_BYTES .. " bytes" end
    local byte, sub, char, floor = string.byte, string.sub, string.char, math.floor
    local n, pos = #line, 1
    local NULL = {}
    local escapes = {[34]='"', [92]="\\", [47]="/", [98]="\b", [102]="\f", [110]="\n", [114]="\r", [116]="\t"}

    local function skip()
        while pos <= n do
            local c = byte(line, pos)
            if c ~= 32 and c ~= 9 and c ~= 10 and c ~= 13 then return end
            pos = pos + 1
        end
    end
    local function utf8(cp)
        if cp < 0x80 then return char(cp) end
        if cp < 0x800 then return char(0xC0 + floor(cp / 0x40), 0x80 + cp % 0x40) end
        if cp < 0x10000 then
            return char(0xE0 + floor(cp / 0x1000), 0x80 + floor(cp / 0x40) % 0x40, 0x80 + cp % 0x40)
        end
        return char(0xF0 + floor(cp / 0x40000), 0x80 + floor(cp / 0x1000) % 0x40,
                    0x80 + floor(cp / 0x40) % 0x40, 0x80 + cp % 0x40)
    end
    local function hex4()
        local digits = sub(line, pos, pos + 3)
        if not digits:match("^%x%x%x%x$") then return nil end
        pos = pos + 4
        return tonumber(digits, 16)
    end
    local function parse_string()
        pos = pos + 1
        local parts, start = {}, pos
        while true do
            local c = byte(line, pos)
            if not c then return nil, "unterminated string" end
            if c == 34 then
                parts[#parts + 1] = sub(line, start, pos - 1)
                pos = pos + 1
                return table.concat(parts)
            elseif c < 32 then
                return nil, "raw control character in string"
            elseif c == 92 then
                parts[#parts + 1] = sub(line, start, pos - 1)
                local e = byte(line, pos + 1)
                pos = pos + 2
                if e == 117 then
                    local cp = hex4()
                    if not cp then return nil, "invalid unicode escape" end
                    if cp >= 0xD800 and cp <= 0xDBFF then
                        if sub(line, pos, pos + 1) ~= "\\u" then return nil, "unpaired surrogate" end
                        pos = pos + 2
                        local low = hex4()
                        if not low or low < 0xDC00 or low > 0xDFFF then return nil, "unpaired surrogate" end
                        cp = 0x10000 + (cp - 0xD800) * 0x400 + (low - 0xDC00)
                    elseif cp >= 0xDC00 and cp <= 0xDFFF then
                        return nil, "unpaired surrogate"
                    end
                    parts[#parts + 1] = utf8(cp)
                elseif e and escapes[e] then
                    parts[#parts + 1] = escapes[e]
                else
                    return nil, "invalid escape"
                end
                start = pos
            else
                pos = pos + 1
            end
        end
    end
    local function parse_number()
        local start = pos
        if byte(line, pos) == 45 then pos = pos + 1 end
        local c = byte(line, pos)
        if c == 48 then
            pos = pos + 1
        elseif c and c >= 49 and c <= 57 then
            local _, last = line:find("^%d+", pos)
            pos = last + 1
        else
            return nil, "invalid number"
        end
        if byte(line, pos) == 46 then
            local _, last = line:find("^%d+", pos + 1)
            if not last then return nil, "invalid number fraction" end
            pos = last + 1
        end
        c = byte(line, pos)
        if c == 101 or c == 69 then
            local _, last = line:find("^[+-]?%d+", pos + 1)
            if not last then return nil, "invalid number exponent" end
            pos = last + 1
        end
        local value = tonumber(sub(line, start, pos - 1))
        if not value or value ~= value or value == math.huge or value == -math.huge then
            return nil, "number out of range"
        end
        return value
    end
    local function parse_value()
        local c = byte(line, pos)
        if c == 34 then return parse_string() end
        if c == 45 or (c and c >= 48 and c <= 57) then return parse_number() end
        if sub(line, pos, pos + 3) == "true" then pos = pos + 4; return true end
        if sub(line, pos, pos + 4) == "false" then pos = pos + 5; return false end
        if sub(line, pos, pos + 3) == "null" then pos = pos + 4; return NULL end
        if c == 123 or c == 91 then return nil, "nested objects and arrays are not accepted" end
        return nil, "invalid value"
    end
    local function fail(message) return nil, message .. " at byte " .. pos end

    skip()
    if byte(line, pos) ~= 123 then return fail("request must be a JSON object") end
    pos = pos + 1
    local result, seen = {}, {}
    skip()
    if byte(line, pos) == 125 then
        pos = pos + 1
    else
        while true do
            skip()
            if byte(line, pos) ~= 34 then return fail("expected string key") end
            local key, err = parse_string()
            if key == nil then return fail(err) end
            if seen[key] then return fail("duplicate key") end
            seen[key] = true
            skip()
            if byte(line, pos) ~= 58 then return fail("expected colon") end
            pos = pos + 1
            skip()
            local value
            value, err = parse_value()
            if value == nil then return fail(err) end
            if value ~= NULL then result[key] = value end
            skip()
            local c = byte(line, pos)
            if c == 125 then pos = pos + 1; break end
            if c ~= 44 then return fail("expected comma or closing brace") end
            pos = pos + 1
        end
    end
    skip()
    if pos <= n then return fail("trailing data after request object") end
    return result
end
M._decode_request = decode_request

local function ready()
    local s=game and game.store
    return s and s.entities and type(s.tick)=="number" and type(s.player_gold)=="number"
        and type(s.lives)=="number" and type(s.wave_group_number)=="number"
end

local function send(client, value)
    local payload = native.codec.encode(value) .. "\n"
    client:settimeout(5)
    local n, err = client:send(payload)
    client:settimeout(0)
    if n ~= #payload then error("RPC send failed: " .. tostring(err)) end
end

local function catalog(state)
    local edb = require("entity_db")
    local out = {}
    local store = game and game.store
    if not store then return out end
    for id,e in pairs(store.entities or {}) do
        if e.tower_holder and not e.tower_holder.blocked then
            for _,kind in ipairs({"archer", "barrack", "mage", "engineer"}) do
                local name = "tower_build_" .. kind
                local t = edb:get_template("tower_" .. kind .. "_1")
                local cost = t and t.tower and t.tower.price
                if cost and cost <= store.player_gold then
                    out[#out+1] = {action="build_tower", holder_id=id,
                        tower_type=kind, target=name, cost=cost, available=true, legal=true}
                end
            end
        end
    end
    if state.next_wave and not state.wave_spawning and (state.wave==0 or state.enemy_count==0) then
        out[#out+1]={action="send_wave",cost=0,available=true,legal=true}
    end
    table.sort(out, function(a,b)
        if a.action~=b.action then return a.action<b.action end
        if (a.holder_id or 0) ~= (b.holder_id or 0) then return (a.holder_id or 0)<(b.holder_id or 0) end
        return (a.tower_type or "") < (b.tower_type or "")
    end)
    return out
end

-- Action scope v2 mirrors the game's own tower menus (data.tower_menus_data) and prices.
local BASIC_TOWERS = {archer=true, barrack=true, mage=true, engineer=true}
-- Standard families and their level-4 branches; special towers/holders are never offered.
-- Level towers whose GUI menu sells only a power (level 9's sunray beam); nothing else is offered.
local POWER_SPECIALS = {sunray=true}
local STANDARD_TOWERS = {archer=true, barrack=true, mage=true, engineer=true,
    ranger=true, musketeer=true, paladin=true, barbarian=true,
    arcane_wizard=true, sorcerer=true, bfg=true, tesla=true}
local ANCHOR_SPACING, ANCHOR_LIMIT = 60, 3
local V2_ACTIONS = {build_tower=true, send_wave=true, upgrade_tower=true,
    upgrade_power=true, sell_tower=true, use_power=true, point_tower=true, click_entity=true}
local V3_ACTIONS = {build_tower=true, send_wave=true, upgrade_tower=true, upgrade_power=true, sell_tower=true,
    use_power=true, point_tower=true, click_entity=true, set_rally=true}
local V2_MATCH_KEYS = {"action", "holder_id", "tower_id", "tower_type", "target", "power", "x", "y", "anchor_id",
    "entity_id", "option"}

local function finite(value)
    return type(value) == "number" and value == value and value ~= math.huge and value ~= -math.huge
end

-- Only the table the game already loaded (see game_settings below for why not require()).
local function tower_menus(kind, level)
    local menus = package.loaded["data.tower_menus_data"]
    local by_level = type(menus) == "table" and menus[kind]
    local items = type(by_level) == "table" and by_level[level]
    return type(items) == "table" and items or {}
end

-- GUI price of a tw_upgrade target: a build placeholder charges its build_name's price.
local function upgrade_price(edb, target)
    local function template(name)
        local ok, t = pcall(edb.get_template, edb, name)
        return ok and type(t) == "table" and t or nil
    end
    local t = type(target) == "string" and template(target)
    if t and t.build_name ~= nil then t = template(t.build_name) end
    local price = t and type(t.tower) == "table" and t.tower.price
    return finite(price) and price >= 0 and price or nil
end

local function locked_towers(s, store)
    local locked = {}
    for _, list in pairs({s.locked_towers or false, store.level and store.level.locked_towers or false}) do
        if type(list) == "table" then
            for _, name in pairs(list) do locked[name] = true end
        end
    end
    return locked
end

-- Spell anchors: live enemies by path progress (desc, then id), each at least
-- ANCHOR_SPACING from those already chosen, at most ANCHOR_LIMIT, rounded to integers.
-- A charged sunray is aimed by the GUI's tw_point at a clicked enemy: offer the (at most
-- ANCHOR_LIMIT) live enemies with the most health, ties to the one further along its path.
local function strong_anchors(enemies)
    local live = {}
    for _, en in ipairs(type(enemies) == "table" and enemies or {}) do
        if type(en) == "table" and type(en.id) == "number" and finite(en.hp) and en.hp > 0
            and finite(en.x) and finite(en.y) then
            live[#live+1] = en
        end
    end
    table.sort(live, function(a, b)
        if a.hp ~= b.hp then return a.hp > b.hp end
        local pa, pb = a.path_progress or 0, b.path_progress or 0
        if pa ~= pb then return pa > pb end
        return a.id < b.id
    end)
    local chosen = {}
    for i = 1, math.min(ANCHOR_LIMIT, #live) do
        local en = live[i]
        chosen[#chosen+1] = {id=en.id, x=math.floor(en.x + 0.5), y=math.floor(en.y + 0.5)}
    end
    return chosen
end

-- The sunray script fires once its attack has recharged (tick_ts - ts >= cooldown) and the GUI
-- has set user_selection.new_pos; it then resets ts. Ready = allowed, unaimed and recharged.
local function point_ready(e, store)
    local us, ray = e.user_selection, type(e.powers) == "table" and e.powers.ray
    local a = type(e.attacks) == "table" and type(e.attacks.list) == "table" and e.attacks.list[1]
    return type(us) == "table" and us.allowed == true and us.new_pos == nil and type(ray) == "table"
        and finite(ray.level) and ray.level >= 1 and type(a) == "table" and finite(a.ts) and finite(a.cooldown)
        and finite(store.tick_ts) and store.tick_ts - a.ts >= a.cooldown
end

local function power_anchors(enemies)
    local live = {}
    for _, en in ipairs(type(enemies) == "table" and enemies or {}) do
        if type(en) == "table" and type(en.id) == "number" and finite(en.path_progress)
            and finite(en.x) and finite(en.y) then
            live[#live+1] = en
        end
    end
    table.sort(live, function(a, b)
        if a.path_progress ~= b.path_progress then return a.path_progress > b.path_progress end
        return a.id < b.id
    end)
    local chosen = {}
    for _, en in ipairs(live) do
        local spaced = true
        for _, c in ipairs(chosen) do
            local dx, dy = en.x - c.enemy.x, en.y - c.enemy.y
            if dx*dx + dy*dy < ANCHOR_SPACING*ANCHOR_SPACING then spaced = false; break end
        end
        if spaced then
            chosen[#chosen+1] = {enemy=en, id=en.id, x=math.floor(en.x + 0.5), y=math.floor(en.y + 0.5)}
            if #chosen >= ANCHOR_LIMIT then break end
        end
    end
    return chosen
end

-- The GUI's own placement rules (game_gui GUI_MODE_POWER_1/2): rain of fire is refused on
-- cliffs/faerie cells and needs a NF_POWER_1 path node nearby or water; reinforcements need a
-- rally node nearby on land/ice only. Fail closed when the game modules are not loaded.
local function power_point_valid(p, x, y)
    local P, GR = package.loaded["path_db"], package.loaded["grid_db"]
    if type(P) ~= "table" or type(GR) ~= "table" then return false end
    for _, flag in ipairs({TERRAIN_CLIFF, TERRAIN_FAERIE, TERRAIN_WATER, TERRAIN_LAND, TERRAIN_ICE,
                           NF_POWER_1, NF_RALLY}) do
        if type(flag) ~= "number" then return false end
    end
    local ok, valid = pcall(function()
        if p == 1 then
            if GR:cell_is(x, y, bit.bor(TERRAIN_CLIFF, TERRAIN_FAERIE)) then return false end
            if P:valid_node_nearby(x, y, 1 / 0.7, NF_POWER_1) or GR:cell_is(x, y, TERRAIN_WATER) then return true end
            -- Some levels (8, 10, 12) also allow fire inside level-defined areas.
            local level = game.store and game.store.level
            return type(level) == "table" and type(level.fn_can_power) == "function"
                and type(GUI_MODE_POWER_1) == "number"
                and level:fn_can_power(game.store, GUI_MODE_POWER_1, {x = x, y = y}) and true or false
        end
        return P:valid_node_nearby(x, y, nil, NF_RALLY) and GR:cell_is_only(x, y, bit.bor(TERRAIN_LAND, TERRAIN_ICE))
    end)
    return ok and valid and true or false
end

-- Entities the GUI cannot select (e.g. towers frozen by a boss) offer no menu.
local function clickable(e)
    return not (type(e.ui) == "table" and e.ui.can_click == false)
end

-- Clicks a game script is waiting for (the GUI's PickView sets e.ui.clicked on a clickable entity):
-- a boss knocked down awaiting the finishing tap (J.T.: dying, tap decal shown) and a click-to-break
-- modifier on a tower (J.T.'s ice; its script counts required_clicks itself). Returns the kind, or nil.
local function click_need(e)
    local ui = type(e) == "table" and e.ui
    if type(ui) ~= "table" or ui.can_click ~= true or ui.clicked then return nil end
    if type(e.pos) ~= "table" or not finite(e.pos.x) or not finite(e.pos.y) then return nil end
    if e.dying == true and type(e.tap_decal) == "string" and type(e.health) == "table" and not e.health.dead then
        return "downed_boss"
    end
    if type(e.modifier) == "table" and finite(e.required_clicks) and e.required_clicks > 0 then
        return "tower_trap"
    end
    return nil
end

-- v3: bit tests on vis flags/bans; a missing game constant never matches.
local function has_flag(mask, flag)
    return finite(mask) and type(flag) == "number" and bit.band(mask, flag) ~= 0
end

-- v3: progress 0..1 along the enemy's route: its own lane (path pi, sub-path spi, node ni), so levels with
-- several exits (17, 19, 24, 26) measure every enemy against the exit its path actually ends at, extended
-- through path_db.path_connections (level 18's tunnels move enemies from the end of paths 1 and 2 to node 1
-- of paths 5 and 6). A path's route = the longest chain of feeder paths before it (head) + its own nodes
-- + the connected paths after it (tail). Without connections this is exactly (ni - 1) / (#lane - 1).
local route_cache = setmetatable({}, {__mode = "k"})

local function route_of(P)
    local cached = route_cache[P.paths]
    if cached then return cached end
    local conn = type(P.path_connections) == "table" and P.path_connections or {}
    local nodes = {}
    for pi, path in pairs(P.paths) do
        local lane = type(path) == "table" and path[1]
        nodes[pi] = type(lane) == "table" and #lane or 0
    end
    local function tail(pi, depth)
        local next_pi = conn[pi]
        if next_pi == nil or depth > 8 or not nodes[next_pi] then return 0 end
        return math.max(0, nodes[next_pi] - 1) + tail(next_pi, depth + 1)
    end
    local function head(pi, depth)
        local best = 0
        if depth > 8 then return 0 end
        for from, to in pairs(conn) do
            if to == pi and nodes[from] then best = math.max(best, head(from, depth + 1) + math.max(0, nodes[from] - 1)) end
        end
        return best
    end
    local route = {}
    for pi in pairs(nodes) do route[pi] = {head = head(pi, 0), tail = tail(pi, 0)} end
    route_cache[P.paths] = route
    return route
end

local function route_progress(P, pi, n, ni)
    local r = route_of(P)[pi] or {head = 0, tail = 0}
    local total = r.head + (n - 1) + r.tail
    if total <= 0 then return 0 end
    return math.min(1, math.max(0, (r.head + (ni - 1)) / total))
end

local function lane_progress(e)
    local P, np = package.loaded["path_db"], e.nav_path
    if type(P) ~= "table" or type(P.paths) ~= "table" or type(np) ~= "table" or not finite(np.ni) then return nil end
    local path = P.paths[np.pi]
    local lane = type(path) == "table" and path[np.spi]
    if type(lane) ~= "table" or #lane < 2 then return nil end
    local ok, progress = pcall(route_progress, P, np.pi, #lane, np.ni)
    return ok and progress or nil
end

-- v3: flags on the exported enemies. Dormant = a boss waiting for its wave (level 20's sleeping
-- Cerberus, level 21's seated Moloch): it never moves before then, so it neither holds back an early
-- wave call nor anchors a spell. Untargetable = dormant, damage-immune (e.g. Ulgukhai's shield while
-- unblocked) or banned from every effect. Returns the number of enemies that are not dormant.
local function augment_enemies(s, store)
    local active = 0
    for _, en in ipairs(type(s.enemies) == "table" and s.enemies or {}) do
        local e = type(en) == "table" and store.entities[en.id]
        if type(e) ~= "table" then
            active = active + 1  -- unknown entity: counted, never flagged
        else
            local vis = type(e.vis) == "table" and e.vis or {}
            local bans = finite(vis.bans) and vis.bans or 0
            local dormant = e.sleeping == true or e.phase == "sitting"
            local banned_all = bans == -1 or (type(F_ALL) == "number" and bit.band(bans, F_ALL) == bit.tobit(F_ALL))
            en.dormant = dormant
            en.untargetable = dormant or (type(e.health) == "table" and e.health.ignore_damage == true) or banned_all
            en.flying = has_flag(vis.flags, F_FLYING)
            en.boss = has_flag(vis.flags, F_BOSS) or has_flag(vis.flags, F_MINIBOSS)
                or (type(e.template_name) == "string" and e.template_name:sub(1, 3) == "eb_")
            en.unblockable = has_flag(bans, F_BLOCK)
            local progress = lane_progress(e)
            if progress then en.path_progress = progress end
            if not dormant then active = active + 1 end
        end
    end
    return active
end

local function targetable(enemies)
    local out = {}
    for _, en in ipairs(type(enemies) == "table" and enemies or {}) do
        if type(en) == "table" and not en.untargetable then out[#out+1] = en end
    end
    return out
end

-- v3 rally points, as the GUI's GUI_MODE_RALLY_TOWER click allows them: inside the barrack's
-- rally_range ellipse (utils.is_inside_ellipse, aspect 0.7) around pos + tower.range_offset, and
-- (unless rally_anywhere) next to an NF_RALLY path node on rally_terrains only. Candidates are the
-- path nodes passing those checks (rounded to integers, then checked); the offered options are the
-- most upstream ("entry"), most downstream ("exit") and most central ("center") candidate, plus
-- "boss": the candidate nearest an awake boss standing inside the ellipse.
local RALLY_ASPECT = 0.7
local rally_cache = setmetatable({}, {__mode = "k"})

local function inside_ellipse(x, y, cx, cy, r)
    local a, b = r, r * RALLY_ASPECT
    return ((x - cx) / a) ^ 2 + ((y - cy) / b) ^ 2 <= 1
end

local function rally_center(e)
    local off = type(e.tower) == "table" and e.tower.range_offset
    local ox = type(off) == "table" and finite(off.x) and off.x or 0
    local oy = type(off) == "table" and finite(off.y) and off.y or 0
    return e.pos.x + ox, e.pos.y + oy
end

local function rally_candidates(e)
    local b = e.barrack
    local P, GR = package.loaded["path_db"], package.loaded["grid_db"]
    if type(P) ~= "table" or type(P.paths) ~= "table" or type(GR) ~= "table" or type(NF_RALLY) ~= "number" then
        return {}
    end
    local r = b.rally_range
    if not finite(r) or r <= 0 or type(e.pos) ~= "table" or not finite(e.pos.x) or not finite(e.pos.y) then return {} end
    local cx, cy = rally_center(e)
    local cached = rally_cache[e]
    if cached and cached.r == r and cached.cx == cx and cached.cy == cy and cached.anywhere == b.rally_anywhere then
        return cached.points
    end
    local points = {}
    local ok = pcall(function()
        for pi, path in ipairs(P.paths) do
            for spi, lane in ipairs(path) do
                local n = #lane
                for ni = 1, n do
                    local node = lane[ni]
                    if type(node) == "table" and finite(node.x) and finite(node.y) then
                        local x, y = math.floor(node.x + 0.5), math.floor(node.y + 0.5)
                        if inside_ellipse(x, y, cx, cy, r) and (b.rally_anywhere
                            or (P:valid_node_nearby(x, y, nil, NF_RALLY) and GR:cell_is_only(x, y, b.rally_terrains))) then
                            points[#points+1] = {x=x, y=y, progress=route_progress(P, pi, n, ni), pi=pi, spi=spi, ni=ni}
                        end
                    end
                end
            end
        end
    end)
    if not ok then points = {} end
    rally_cache[e] = {r=r, cx=cx, cy=cy, anywhere=b.rally_anywhere, points=points}
    return points
end

local function rally_options(e, s)
    local points = rally_candidates(e)
    if #points == 0 then return {} end
    local function order(a, b)
        if a.pi ~= b.pi then return a.pi < b.pi end
        if a.spi ~= b.spi then return a.spi < b.spi end
        return a.ni < b.ni
    end
    local function best(score)
        local chosen, value
        for _, p in ipairs(points) do
            local v = score(p)
            if chosen == nil or v < value or (v == value and order(p, chosen)) then chosen, value = p, v end
        end
        return chosen
    end
    local cx, cy = rally_center(e)
    local out = {
        {option="entry", point=best(function(p) return p.progress end)},
        {option="center", point=best(function(p) return (p.x - cx) ^ 2 + (p.y - cy) ^ 2 end)},
        {option="exit", point=best(function(p) return -p.progress end)},
    }
    local boss, boss_d
    for _, en in ipairs(type(s.enemies) == "table" and s.enemies or {}) do
        if type(en) == "table" and en.boss and not en.dormant and finite(en.x) and finite(en.y) and finite(en.hp)
            and en.hp > 0 and inside_ellipse(en.x, en.y, cx, cy, e.barrack.rally_range) then
            local d = (en.x - cx) ^ 2 + (en.y - cy) ^ 2
            if boss == nil or d < boss_d or (d == boss_d and en.id < boss.id) then boss, boss_d = en, d end
        end
    end
    if boss then
        out[#out+1] = {option="boss", point=best(function(p) return (p.x - boss.x) ^ 2 + (p.y - boss.y) ^ 2 end),
                       boss_id=boss.id}
    end
    return out
end

-- GUI price of the next power level: price_base for the first, price_inc for each later one.
local function power_price(pw)
    local price = pw.level == 0 and (pw.price_base or 0) or (pw.price_inc or 0)
    return finite(price) and price >= 0 and price or nil
end

local function power_button(p)
    local gui = game and game.game_gui
    local btn = type(gui) == "table" and gui["power_" .. p]
    return type(btn) == "table" and btn or nil
end

local function powers_ui()
    local out = {}
    for p = 1, 2 do
        local btn = power_button(p)
        local mode = btn and btn.mode
        out[p] = {id=p, mode=type(mode) == "string" and mode or "missing"}
    end
    return out
end

-- Live enemies that hold back an early wave call (v3: dormant bosses do not).
local function wave_count(s)
    if ACTION_SCOPE == "v3" then return s.active_enemy_count end
    return s.enemy_count
end

local function catalog_v2(s)
    local edb = require("entity_db")
    local out = {}
    local store = game and game.store
    if not store then return out end
    local gold = store.player_gold
    local locked = locked_towers(s, store)
    local function add(item)
        item.available, item.legal = true, true
        out[#out+1] = item
    end
    for id, e in pairs(store.entities or {}) do
        local tower = type(e) == "table" and e.tower
        local kind = type(tower) == "table" and tower.type
        if type(kind) ~= "string" or tower.upgrade_to or tower.sell then
            -- not a holder/tower, or a native build/upgrade/sale is already pending
        elseif not clickable(e) then
            -- the GUI cannot open this entity's menu
        elseif kind == "holder" then
            if e.tower_holder and not e.tower_holder.blocked then
                for _, item in ipairs(tower_menus("holder", 1)) do
                    local arg = item.action == "tw_upgrade" and item.action_arg
                    local base = type(arg) == "string" and arg:match("^tower_build_(.+)$")
                    if base and BASIC_TOWERS[base] and not locked[arg] and not locked["tower_" .. base .. "_1"] then
                        local price = upgrade_price(edb, arg)
                        if price and price <= gold then
                            add({action="build_tower", holder_id=id, tower_type=base, target=arg, cost=price})
                        end
                    end
                end
            end
        elseif STANDARD_TOWERS[kind] and tower.can_be_mod ~= false and not tower.blocked then
            for _, item in ipairs(tower_menus(kind, tower.level or 1)) do
                local arg = item.action_arg
                if item.action == "tw_upgrade" then
                    if type(arg) == "string" and not locked[arg] then
                        local price = upgrade_price(edb, arg)
                        if price and price <= gold then
                            add({action="upgrade_tower", tower_id=id, target=arg, cost=price})
                        end
                    end
                elseif item.action == "upgrade_power" then
                    local pw = type(e.powers) == "table" and type(arg) == "string" and e.powers[arg]
                    if type(pw) == "table" and finite(pw.level) and finite(pw.max_level) and pw.level < pw.max_level then
                        local price = power_price(pw)
                        if price and price <= gold then
                            add({action="upgrade_power", tower_id=id, power=arg, cost=price})
                        end
                    end
                elseif item.action == "tw_sell" then
                    if tower.can_be_sold ~= false then add({action="sell_tower", tower_id=id, cost=0}) end
                elseif item.action == "tw_rally" and ACTION_SCOPE == "v3" and type(e.barrack) == "table" then
                    for _, o in ipairs(rally_options(e, s)) do
                        add({action="set_rally", tower_id=id, option=o.option, x=o.point.x, y=o.point.y, cost=0})
                    end
                end
            end
        elseif POWER_SPECIALS[kind] and not tower.blocked then
            for _, item in ipairs(tower_menus(kind, tower.level or 1)) do
                local arg = item.action_arg
                if item.action == "upgrade_power" then
                    local pw = type(e.powers) == "table" and type(arg) == "string" and e.powers[arg]
                    if type(pw) == "table" and finite(pw.level) and finite(pw.max_level) and pw.level < pw.max_level then
                        local price = power_price(pw)
                        if price and price <= gold then
                            add({action="upgrade_power", tower_id=id, power=arg, cost=price})
                        end
                    end
                elseif item.action == "tw_point" and point_ready(e, store) then
                    for _, a in ipairs(strong_anchors(ACTION_SCOPE == "v3" and targetable(s.enemies) or s.enemies)) do
                        add({action="point_tower", tower_id=id, x=a.x, y=a.y, anchor_id=a.id, cost=0})
                    end
                end
            end
        end
    end
    for id, e in pairs(store.entities or {}) do
        local need = click_need(e)
        if need then
            add({action="click_entity", entity_id=id, x=math.floor(e.pos.x + 0.5), y=math.floor(e.pos.y + 0.5),
                 template=type(e.template_name) == "string" and e.template_name or "", kind=need, cost=0})
        end
    end
    local anchors = power_anchors(ACTION_SCOPE == "v3" and targetable(s.enemies) or s.enemies)
    for p = 1, 2 do
        local btn = power_button(p)
        if btn and btn.mode ~= nil and btn.mode ~= "locked" and btn.mode ~= "cooldown" then
            for _, a in ipairs(anchors) do
                if power_point_valid(p, a.x, a.y) then
                    add({action="use_power", power=p, x=a.x, y=a.y, anchor_id=a.id, cost=0})
                end
            end
        end
    end
    if s.next_wave and not s.wave_spawning and (s.wave==0 or wave_count(s)==0) then
        add({action="send_wave", cost=0})
    end
    -- Total order over the identifying keys, so entity traversal order never shows.
    local keys = {}
    for _, a in ipairs(out) do
        local named = type(a.power) == "string" and a.power or a.option
        keys[a] = {a.action or "", a.holder_id or a.tower_id or a.entity_id or 0, a.tower_type or a.target or named or "",
                   type(a.power) == "number" and a.power or 0, a.x or 0, a.y or 0, a.anchor_id or 0}
    end
    table.sort(out, function(a, b)
        local ka, kb = keys[a], keys[b]
        for i = 1, #ka do
            if ka[i] ~= kb[i] then return ka[i] < kb[i] end
        end
        return false
    end)
    return out
end

local function state()
    if not ready() then
        return {type="loading", save_directory=love.filesystem.getSaveDirectory(), controlled=false}
    end
    local s = native.collect_state()
    s.save_directory = love.filesystem.getSaveDirectory()
    s.controlled = controlled
    local store = game and game.store
    if store then
        s.tick_ts = store.tick_ts
        s.tick_length = store.tick_length
        s.native_outcome = store.game_outcome
        s.level_won = store.game_outcome and store.game_outcome.victory==true or false
        s.level_lost = store.game_outcome and store.game_outcome.victory==false or false
        if ACTION_SCOPE == "v3" then s.active_enemy_count = augment_enemies(s, store) end
        if ACTION_SCOPE == "v2" or ACTION_SCOPE == "v3" then
            s.action_catalog = catalog_v2(s)
            s.powers_ui = powers_ui()
            s.action_scope = ACTION_SCOPE
        else
            s.action_catalog = catalog(s)
        end
        s.wave_ready = s.next_wave ~= nil and not s.wave_spawning and (s.wave==0 or wave_count(s)==0)
    end
    return s
end

-- Copy plain data only, so replies never expose live game tables.
local function plain(value, depth)
    local t = type(value)
    if t == "number" then
        if value ~= value or value == math.huge or value == -math.huge then return nil end
        return value
    elseif t == "string" or t == "boolean" then
        return value
    elseif t == "table" and (depth or 0) < 4 then
        local out = {}
        for k, v in pairs(value) do
            if type(k) == "string" or type(k) == "number" then out[k] = plain(v, (depth or 0) + 1) end
        end
        return out
    end
    return nil
end

-- Only modules the game already loaded; require() would run module code and
-- could leave a failure sentinel in package.loaded for the game's own require.
local function game_settings()
    for _, name in ipairs({"game_settings", "kr1.game_settings"}) do
        if type(package.loaded[name]) == "table" then return package.loaded[name] end
    end
    return nil, "not loaded by the game (game_settings, kr1.game_settings)"
end

-- Read-only level/campaign facts; every read is guarded and nothing is written.
local function meta()
    local out, errors = {}, {}
    local function read(name, getter)
        local ok, value = pcall(getter)
        if ok then out[name] = plain(value) else errors[#errors + 1] = name .. ": " .. tostring(value) end
    end
    local store = game and game.store
    if store then
        read("level_idx", function() return store.level_idx end)
        read("level_mode", function() return store.level_mode end)
        read("level_difficulty", function() return store.level_difficulty end)
        read("locked_towers", function() return store.level and store.level.locked_towers end)
        read("locked_powers", function() return store.level and store.level.locked_powers end)
        read("locked_hero", function() return store.level and store.level.locked_hero end)
        read("max_upgrade_level", function() return store.level and store.level.max_upgrade_level end)
        read("hero_count", function()
            local count = 0
            for _, e in pairs(store.entities or {}) do
                if type(e) == "table" and e.hero then count = count + 1 end
            end
            return count
        end)
        -- What the level actually loaded from the save slot (campaign profile evidence).
        read("star_upgrades", function()
            local upgrades = package.loaded["upgrades"]
            return type(upgrades) == "table" and upgrades.levels or nil
        end)
        read("selected_hero", function() return store.selected_hero end)
        read("heroes", function()
            local names = {}
            for _, e in pairs(store.entities or {}) do
                if type(e) == "table" and e.hero and type(e.template_name) == "string" then
                    names[#names + 1] = e.template_name
                end
            end
            table.sort(names)
            return names
        end)
        if out.level_difficulty == nil then errors[#errors + 1] = "level_difficulty: not exposed by game.store" end
    else
        errors[#errors + 1] = "game.store: no native level is loaded"
    end
    local ok, settings, problem = pcall(game_settings)
    if not ok then
        errors[#errors + 1] = "main_campaign_levels: " .. tostring(settings)
    elseif not settings then
        errors[#errors + 1] = "main_campaign_levels: game_settings unavailable (" .. tostring(problem) .. ")"
    else
        read("main_campaign_levels", function() return settings.main_campaign_levels end)
        if out.main_campaign_levels == nil then
            errors[#errors + 1] = "main_campaign_levels: not present in game_settings"
        end
    end
    out.errors = errors
    return out
end
M._meta = meta

local function step(count)
    local store = game and game.store
    if not store then error("No native level is loaded") end
    count = tonumber(count) or 1
    if count < 0 or count > 3600 or count ~= math.floor(count) then error("Invalid tick count") end
    controlled = true
    store.paused = true
    local start_tick = store.tick
    local updates = 0
    while store.tick - start_tick < count do
        store.step = true
        original_update(store.tick_length or 1/60)
        updates = updates + 1
        if updates > count*3+4 then error("Native tick failed to advance") end
        if game.store ~= store then error("Native store changed during step") end
        if store.game_outcome and type(store.game_outcome.victory)=="boolean" then break end
    end
    if store.tick - start_tick > count then error("Native tick overshot requested count") end
    store.step = false
    store.paused = true
    return {tick_before=start_tick, tick_after=store.tick, requested_ticks=count,
            native_updates=updates, terminated=store.game_outcome~=nil, state=state()}
end

-- v2: the command must equal one catalog entry on every identifying key (absent = nil).
-- build_tower's target is implied by tower_type and may be omitted, as in v1 menus;
-- a stated target must still match.
local function handle_v2(cmd)
    if not (ACTION_SCOPE == "v3" and V3_ACTIONS or V2_ACTIONS)[cmd.action] then
        error("Action not enabled in verified experiment scope")
    end
    local legal = false
    for _, a in ipairs(state().action_catalog or {}) do
        local same = true
        for _, key in ipairs(V2_MATCH_KEYS) do
            local implied = key == "target" and a.action == "build_tower" and cmd.target == nil
            if not implied and cmd[key] ~= a[key] then same = false; break end
        end
        if same then legal = true; break end
    end
    if not legal then error("Action is outside the native legal catalog") end
    if cmd.action == "build_tower" then
        -- As the GUI build button (and the bridge, minus its base-price gold check that ignores star
        -- upgrades such as Hermetic Study): the holder upgrades into the build template and the game's
        -- tower system charges the actual (possibly discounted) price.
        local e = game.store.entities[cmd.holder_id]
        local target = "tower_build_" .. cmd.tower_type
        e.tower.upgrade_to = target
        return {native_wire=native.codec.encode({type="ok", action="build_tower", holder_id=cmd.holder_id,
                                                 tower_type=cmd.tower_type, target=target})}
    elseif cmd.action == "upgrade_tower" or cmd.action == "sell_tower" then
        -- The game's own GUI path: the tower systems charge/refund and swap entities.
        local e = game.store.entities[cmd.tower_id]
        if cmd.action == "upgrade_tower" then e.tower.upgrade_to = cmd.target else e.tower.sell = true end
        return {native_wire=native.codec.encode({type="ok", action=cmd.action,
                                                 tower_id=cmd.tower_id, target=cmd.target})}
    elseif cmd.action == "upgrade_power" then
        -- As the GUI button: next level, mark changed, charge the GUI price, count it as spent.
        local e = game.store.entities[cmd.tower_id]
        local pw = e.powers[cmd.power]
        local price = power_price(pw)
        pw.level = pw.level + 1
        pw.changed = true
        game.store.player_gold = game.store.player_gold - price
        e.tower.spent = (e.tower.spent or 0) + price
        return {native_wire=native.codec.encode({type="ok", action="upgrade_power", tower_id=cmd.tower_id,
                                                 power=cmd.power, level=pw.level, cost=price})}
    elseif cmd.action == "use_power" then
        return {native_wire=native.command_json({action="use_power", power=cmd.power, x=cmd.x, y=cmd.y})}
    elseif cmd.action == "click_entity" then
        -- As the GUI's PickView click on this entity: its script consumes ui.clicked.
        local e = game.store.entities[cmd.entity_id]
        e.ui.clicked = true
        return {native_wire=native.codec.encode({type="ok", action="click_entity", entity_id=cmd.entity_id,
                                                 x=cmd.x, y=cmd.y})}
    elseif cmd.action == "set_rally" then
        -- As the GUI's GUI_MODE_RALLY_TOWER click (game_gui): a new rally_pos vector and rally_new; the
        -- barrack's own script then moves its soldiers.
        local e = game.store.entities[cmd.tower_id]
        e.barrack.rally_pos = {x=cmd.x, y=cmd.y}
        e.barrack.rally_new = true
        return {native_wire=native.codec.encode({type="ok", action="set_rally", tower_id=cmd.tower_id,
                                                 option=cmd.option, x=cmd.x, y=cmd.y})}
    elseif cmd.action == "point_tower" then
        -- As the GUI's tw_point click: the tower's own script picks the enemy at this point and fires.
        local e = game.store.entities[cmd.tower_id]
        e.user_selection.new_pos = {x=cmd.x, y=cmd.y}
        return {native_wire=native.codec.encode({type="ok", action="point_tower", tower_id=cmd.tower_id,
                                                 x=cmd.x, y=cmd.y, anchor_id=cmd.anchor_id})}
    end
    return {native_wire=native.command_json(cmd)}
end

local function handle(cmd)
    if cmd.action == "hello" then
        return {version="alpharush-rl-v1", save_directory=love.filesystem.getSaveDirectory(),
                seed=tonumber(os.getenv("ALPHARUSH_SEED")), port=tonumber(os.getenv("ALPHARUSH_PORT")),
                rng_mode=ALPHARUSH_RNG and ALPHARUSH_RNG.mode or "", action_scope=ACTION_SCOPE,
                headless=os.getenv("ALPHARUSH_HEADLESS") == "1" and love.joystick == nil}
    elseif cmd.action == "state" then return state()
    elseif cmd.action == "step" then return step(cmd.ticks)
    elseif cmd.action == "pause" then
        controlled = true
        if game and game.store then game.store.paused=true end
        return state()
    elseif cmd.action == "diagnostics" then
        local keys={}
        if main and main.handler then
            for k,v in pairs(main.handler) do keys[tostring(k)]=type(v) end
        end
        local modules={}
        for k,v in pairs(package.loaded) do
            if tostring(k):match("screen") or k=="game" or k=="storage" then
                modules[tostring(k)]=type(v)
            end
        end
        return {handler=keys, modules=modules, state=state()}
    elseif cmd.action == "meta" then return meta()
    elseif cmd.action == "quit" then
        -- A normal LOVE shutdown releases the audio device; a killed process leaks it in the
        -- system audio service. The event is handled after this reply is sent.
        love.event.quit()
        return {quitting=true}
    elseif cmd.action == "rng_audit" then
        -- Cumulative random() call counts per calling chunk (diagnostic RNG modes only).
        local rng = ALPHARUSH_RNG
        return {mode = rng and rng.mode or "", counts = rng and plain(rng.counts, 3) or {}}
    elseif ACTION_SCOPE == "v2" or ACTION_SCOPE == "v3" then return handle_v2(cmd)
    else
        if cmd.action=="build_tower" or cmd.action=="send_wave" then
            local legal=false
            for _,a in ipairs(catalog(state())) do
                if a.action==cmd.action and a.holder_id==cmd.holder_id and a.tower_id==cmd.tower_id
                    and a.tower_type==cmd.tower_type and (cmd.action~="upgrade_tower" or a.target==cmd.target) then
                    legal=true;break
                end
            end
            if not legal then error("Action is outside the native legal catalog") end
        else
            error("Action not enabled in verified experiment scope")
        end
        -- Reuse the author's native-action implementations without starting its server.
        return {native_wire=native.command_json(cmd)}
    end
end

function M.install(update)
    if ACTION_SCOPE ~= "v1" and ACTION_SCOPE ~= "v2" and ACTION_SCOPE ~= "v3" then
        error("Unsupported ALPHARUSH_ACTION_SCOPE " .. string.format("%q", ACTION_SCOPE) .. " (expected v1, v2 or v3)")
    end
    original_update = update
    server = assert(socket.bind("127.0.0.1", tonumber(os.getenv("ALPHARUSH_PORT")) or 9879))
    server:settimeout(0)
    M.installed=true
    print("[AlphaRush] isolated RPC ready")
end

-- Echo only short plain ids; the reply encoder does not escape control bytes.
local function safe_id(id)
    if type(id) == "number" then return id end
    if type(id) == "string" and #id <= 64 and id:match("^[%w%-_.:]*$") then return id end
    return nil
end

local function drop(i)
    pcall(function() clients[i].socket:close() end)
    table.remove(clients, i)
end

-- A failed reply only drops that client; it never escapes into the game loop.
local function reply(i, value)
    if not pcall(send, clients[i].socket, value) then drop(i); return false end
    return true
end

-- One request of client i, waiting at most ``timeout`` seconds for it; true when a full line was answered.
local function serve(i, timeout)
    local item=clients[i]
    item.socket:settimeout(timeout)
    local line,err,partial = item.socket:receive("*l")
    item.socket:settimeout(0)
    if partial and #partial>0 then item.buffer=item.buffer..partial end
    if line then
        line=item.buffer..line; item.buffer=""
        local cmd,decode_error=decode_request(line)
        if not cmd then
            -- Always answer, so a client never waits out its socket timeout; then disconnect.
            if reply(i,{id=nil, ok=false, error="Malformed request: "..tostring(decode_error)}) then drop(i) end
        elseif TOKEN and TOKEN ~= "" and cmd.token ~= TOKEN then
            if reply(i,{id=nil, ok=false, error="Unauthorized request"}) then drop(i) end
        else
            cmd.token = nil
            local ok,result=pcall(handle,cmd)
            reply(i,{id=safe_id(cmd.id), ok=ok, result=ok and result or nil,
                     error=not ok and tostring(result) or nil})
        end
        return true
    elseif err=="closed" then drop(i)
    elseif #item.buffer > MAX_REQUEST_BYTES then
        if reply(i,{id=nil, ok=false, error="Malformed request: request exceeds "..MAX_REQUEST_BYTES.." bytes"}) then drop(i) end
    end
    return false
end

-- After an answered request the client usually sends its next one within a millisecond, so it is served
-- in the same frame (each further wait at most BURST_WAIT s, the frame at most BURST_SECONDS or BURST_MAX
-- requests). Windows caps the frame rate of hidden windows (about 57 frames/s measured), which otherwise
-- paced every request. Game time advances only inside "step", so serving faster changes no game.
local BURST_WAIT, BURST_SECONDS, BURST_MAX = 0.02, 0.1, 64

function M.update(dt)
    if not controlled then
        original_update(1/60)
        if ready() then controlled=true; game.store.paused=true; love.draw=function() end end
    end
    local c = server:accept()
    if c then c:settimeout(0); clients[#clients+1]={socket=c, buffer=""} end
    local served = false
    for i=#clients,1,-1 do
        if serve(i, 0) then served = true end
    end
    local clock = type(socket.gettime) == "function" and socket.gettime or nil
    local start, count = clock and clock(), 0
    while served and #clients > 0 and count < BURST_MAX and (not clock or clock() - start < BURST_SECONDS) do
        served, count = false, count + 1
        for i=#clients,1,-1 do
            if serve(i, BURST_WAIT) then served = true end
        end
    end
end

return M
