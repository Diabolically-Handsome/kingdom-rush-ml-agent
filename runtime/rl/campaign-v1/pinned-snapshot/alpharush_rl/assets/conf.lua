-- AlphaRush isolated engine: the game's own conf, then (ALPHARUSH_HEADLESS=1) no joystick subsystem.
-- LOVE enumerates input devices before it opens the window; after hundreds of short-lived parallel
-- games that enumeration stalled every new process, and a scripted worker reads no gamepads.
-- The game touches love.joystick only when it is present (error screen vibration, console builds).
require("_alpha_original_conf")
local game_conf = love.conf
function love.conf(t)
    if game_conf then game_conf(t) end
    if os.getenv("ALPHARUSH_HEADLESS") == "1" then
        t.modules.joystick = false
    end
end
