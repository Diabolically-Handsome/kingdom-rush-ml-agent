"""Recording copy for elite-v2 (scratchpad rec_v2 only, never the real repo): the host keeps drawing after the level is
under control, random draws made while drawing use their own generator (so the gameplay stream is unchanged),
tutorial pop-ups and their dark overlay are not drawn, and pipelined "quiet" steps return no state."""
from pathlib import Path

ROOT = Path(__file__).parent / "rec_v2"
REAL = Path(r"C:\Users\<user>\Documents\AlphaRush")
assert ROOT.resolve() != REAL.resolve()


def patch(rel, pairs):
    path = ROOT / rel
    text = path.read_bytes().decode("utf-8")
    for old, new in pairs:
        assert text.count(old) == 1, (rel, old[:80])
        text = text.replace(old, new)
    path.write_bytes(text.encode("utf-8"))


patch("alpharush_rl/assets/host.lua", [
    ('''local function step(count)
    local store = game and game.store
    if not store then error("No native level is loaded") end
''', '''local function step(count, quiet)
    local store = game and game.store
    if not store then error("No native level is loaded") end
    if quiet and store.game_outcome and type(store.game_outcome.victory) == "boolean" then
        -- RECORDING COPY ONLY: a pipelined step queued behind the level's end does nothing.
        return {tick_before=store.tick, tick_after=store.tick, requested_ticks=count, native_updates=0,
                terminated=true}
    end
'''),
    ('''            native_updates=updates, terminated=store.game_outcome~=nil, state=state()}
end''', '''            native_updates=updates, terminated=store.game_outcome~=nil, state=(not quiet) and state() or nil}
end'''),
    ('''    elseif cmd.action == "step" then return step(cmd.ticks)''',
     '''    elseif cmd.action == "step" then return step(cmd.ticks, cmd.quiet == true)'''),
    ('''        if ready() then controlled=true; game.store.paused=true; love.draw=function() end end''',
     '''        if ready() then
            controlled=true; game.store.paused=true
            if os.getenv("ALPHARUSH_RECORD") == "1" then
                -- RECORDING COPY ONLY: keep drawing; random draws made while rendering use their own
                -- generator (wrapper.lua), so the gameplay random stream is the same as in evaluation.
                local game_draw = love.draw
                love.draw = function()
                    -- Tutorial pop-ups wait for a human "OK" (the agent plays through the API, not the
                    -- GUI); they are simply not drawn, so they do not cover the map in the video.
                    local gui = game and game.game_gui
                    local nv = type(gui) == "table" and gui.notiview
                    if type(nv) == "table" and nv.hidden == false then nv.hidden = true end
                    -- ... and so is the dark modal overlay those pop-ups (and the pause view) put behind them.
                    local ov = type(gui) == "table" and gui.overlay
                    if type(ov) == "table" and ov.hidden == false then ov.hidden = true end
                    ALPHARUSH_DRAWING = true
                    local ok, e = pcall(game_draw)
                    ALPHARUSH_DRAWING = false
                    if not ok then print("[AlphaRush draw error] " .. tostring(e)) end
                end
            else
                love.draw=function() end
            end
        end'''),
])

patch("alpharush_rl/assets/host.lua", [
    ("""local BURST_WAIT, BURST_SECONDS, BURST_MAX = 0.02, 0.1, 64""",
     """local BURST_WAIT, BURST_SECONDS, BURST_MAX = 0.02, 0.1, 64
-- RECORDING COPY ONLY: one request per rendered frame, so every captured frame advances the same number of ticks.
if os.getenv("ALPHARUSH_RECORD") == "1" then BURST_MAX = 0 end"""),
])

patch("alpharush_rl/assets/wrapper.lua", [
    ('''    math.random = function(...)
        local info = getinfo(2, "S")''', '''    -- RECORDING COPY ONLY: random draws made while rendering (host sets ALPHARUSH_DRAWING) use their
    -- own generator, so drawing every frame cannot shift the gameplay stream.
    local recording = os.getenv("ALPHARUSH_RECORD") == "1"
    local draw_generator = recording and love.math.newRandomGenerator(seed + 1) or nil
    local function draw_random(a, b)
        if a == nil then return draw_generator:random() end
        if b == nil then return draw_generator:random(a) end
        return draw_generator:random(a, b)
    end
    math.random = function(...)
        if recording and ALPHARUSH_DRAWING then return draw_random(...) end
        local info = getinfo(2, "S")'''),
    ('''    if audit_on and love.math and love.math.random then
        local base_love_random = love.math.random
        love.math.random = function(...)
            count("love", getinfo(2, "S"))
            return base_love_random(...)
        end
    end''', '''    if (audit_on or recording) and love.math and love.math.random then
        local base_love_random = love.math.random
        love.math.random = function(...)
            if recording and ALPHARUSH_DRAWING then return draw_random(...) end
            if audit_on then count("love", getinfo(2, "S")) end
            return base_love_random(...)
        end
    end'''),
])
print("recording copy patched:", ROOT)
