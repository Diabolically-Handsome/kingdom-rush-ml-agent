-- Preserve original game logic; isolate saves and seed both native RNG families.
local seed = tonumber(os.getenv("ALPHARUSH_SEED")) or 1001
local identity = os.getenv("ALPHARUSH_IDENTITY") or "alpharush_rl_probe"
local set_identity = love.filesystem.setIdentity
love.filesystem.setIdentity = function(_) return set_identity(identity) end
set_identity(identity)
local original_time = os.time
os.time = function(t) if t then return original_time(t) end return seed end
math.randomseed(seed)
if love.math and love.math.setRandomSeed then love.math.setRandomSeed(seed) end

-- Optional determinism modes (ALPHARUSH_RNG_MODE, "+"-joined; empty keeps the original game):
-- "audit" counts random() calls per calling chunk; "isolate_sound" gives sound_db its own
-- generator, because its playback-state checks follow real audio time and would otherwise
-- consume the gameplay stream nondeterministically; "stable_pairs" iterates tables keyed by
-- coroutines (the per-path wave spawners) or by objects with numeric ids in creation/id
-- order instead of memory-address order, which differs between processes. Installed before
-- the game loads, so modules that cache these functions cache the replacements.
local rng_mode = os.getenv("ALPHARUSH_RNG_MODE") or ""
local modes = {}
for token in rng_mode:gmatch("[^+]+") do modes[token] = true end
local audit_on, isolate_on = modes.audit == true, modes.isolate_sound == true
if modes.stable_pairs then
    local raw_next, raw_type, sort, create = next, type, table.sort, coroutine.create
    local order = setmetatable({}, {__mode = "k"})
    local created = 0
    coroutine.create = function(fn)
        local co = create(fn)
        created = created + 1
        order[co] = created
        return co
    end
    local function rank(key)
        if raw_type(key) == "thread" then return order[key] end
        if raw_type(key) == "table" then
            local id = rawget(key, "id")
            if raw_type(id) == "number" then return id end
        end
        return nil
    end
    local original_pairs = pairs
    pairs = function(t)
        local first = raw_next(t)
        local kind = raw_type(first)
        if first == nil or (kind ~= "thread" and kind ~= "table") then return original_pairs(t) end
        local keys = {}
        for key in raw_next, t do
            if rank(key) == nil then return original_pairs(t) end
            keys[#keys + 1] = key
        end
        sort(keys, function(a, b)
            local ra, rb = rank(a), rank(b)
            if raw_type(a) ~= raw_type(b) then return raw_type(a) < raw_type(b) end
            return ra < rb
        end)
        local i = 0
        return function()
            -- Keys removed during the loop are skipped, as with next().
            repeat
                i = i + 1
                local key = keys[i]
                if key == nil then return nil end
                local value = t[key]
                if value ~= nil then return key, value end
            until false
        end, t, nil
    end
end
if audit_on or isolate_on then
    local audit = {}
    ALPHARUSH_RNG = {mode = rng_mode, counts = audit}
    local getinfo = debug.getinfo
    local base_random = math.random
    local sound_random
    if isolate_on then
        local generator = love.math.newRandomGenerator(seed)
        sound_random = function(a, b)
            if a == nil then return generator:random() end
            if b == nil then return generator:random(a) end
            return generator:random(a, b)
        end
    end
    local function count(kind, info)
        local key = kind .. "@" .. tostring(info and info.source) .. ":" .. tostring(info and info.linedefined)
        audit[key] = (audit[key] or 0) + 1
    end
    math.random = function(...)
        local info = getinfo(2, "S")
        if audit_on then count("math", info) end
        if sound_random and info and info.source and info.source:find("sound_db", 1, true) then
            return sound_random(...)
        end
        return base_random(...)
    end
    if audit_on and love.math and love.math.random then
        local base_love_random = love.math.random
        love.math.random = function(...)
            count("love", getinfo(2, "S"))
            return base_love_random(...)
        end
    end
end

local original, err = love.filesystem.load("_alpha_original_main.lua")
if not original then error(err) end
original()
-- Experiment workers use local saves and do not publish progress/achievements.
local features = require("features")
for _,service in pairs(features.platform_services or {}) do
    if type(service) == "table" then service.enabled = false end
end
local host = assert(loadfile(os.getenv("ALPHARUSH_HOST_FILE")))()
local function install()
    if host.installed then return end
    if not love.update then return end
    host.install(love.update)
    love.update = host.update
    local function fail_closed(message)
        print("[AlphaRush worker failure] " .. tostring(message))
        io.stdout:flush()
        os.exit(1)
    end
    love.errhand = fail_closed
    love.errorhandler = fail_closed
    if love.audio then love.audio.setVolume(0) end
end
local original_load = love.load
love.load = function(...)
    if original_load then original_load(...) end
    install()
end
install()
