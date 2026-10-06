"""Inspect LuaJIT bytecode prototypes (constants) of game files with the game's own LuaJIT. Read-only."""
import ctypes, os, sys, zipfile
sys.stdout.reconfigure(encoding="utf-8")
d = r"C:\Users\<user>\Documents\AlphaRush\runtime\rl-engine"
os.add_dll_directory(d)
L = ctypes.CDLL(os.path.join(d, "lua51.dll"))
P, I = ctypes.c_void_p, ctypes.c_int
L.luaL_newstate.restype = P; L.luaL_openlibs.argtypes = [P]
L.luaL_loadbuffer.argtypes = [P, ctypes.c_char_p, ctypes.c_size_t, ctypes.c_char_p]
L.luaL_loadstring.argtypes = [P, ctypes.c_char_p]
L.lua_pcall.argtypes = [P, I, I, I]
L.lua_tolstring.argtypes = [P, I, ctypes.POINTER(ctypes.c_size_t)]; L.lua_tolstring.restype = P
L.lua_setfield.argtypes = [P, I, ctypes.c_char_p]; L.lua_settop.argtypes = [P, I]
G = -10002
s = L.luaL_newstate(); L.luaL_openlibs(s)
def tostr(i=-1):
    n = ctypes.c_size_t(); p = L.lua_tolstring(s, i, ctypes.byref(n)); return None if not p else ctypes.string_at(p, n.value).decode("utf-8","replace")
def run(code):
    if L.luaL_loadstring(s, code.encode()) or L.lua_pcall(s, 0, 1, 0):
        raise RuntimeError(tostr())
    r = tostr(); L.lua_settop(s, 0); return r
z = zipfile.ZipFile(r"D:\SteamLibrary\steamapps\common\Kingdom Rush\Kingdom Rush.exe.bak")
FILTER = sys.argv[2] if len(sys.argv) > 2 else ""
for name in sys.argv[1].split(","):
    data = z.read(name)
    if L.luaL_loadbuffer(s, data, len(data), name.encode()): print("LOADERR", tostr()); continue
    L.lua_setfield(s, G, b"CHUNK")
    print("=====", name)
    print(run(r'''
local ju = require("jit.util")
local out = {}
local filter = "''' + FILTER + r'''"
local function walk(fn, path)
  local info = ju.funcinfo(fn)
  local consts = {}
  local i = -1
  while true do
    local k = ju.funck(fn, i)
    if k == nil then break end
    if type(k) == "string" then consts[#consts+1] = k end
    i = i - 1
  end
  local line = path .. " L" .. tostring(info.linedefined) .. "-" .. tostring(info.lastlinedefined) .. ": " .. table.concat(consts, " ")
  if filter == "" then out[#out+1] = line else
    for w in filter:gmatch("[^|]+") do if line:find(w, 1, true) then out[#out+1] = line; break end end
  end
  local j = -1
  while true do
    local k = ju.funck(fn, j)
    if k == nil then break end
    if type(k) == "proto" then walk(k, path .. "/" .. tostring(-j)) end
    j = j - 1
  end
end
walk(CHUNK, "main")
return table.concat(out, "\n")
'''))
