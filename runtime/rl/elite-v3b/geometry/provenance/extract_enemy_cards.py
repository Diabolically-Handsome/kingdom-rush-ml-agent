"""elite-v3b: build enemy_cards.json from YOUR OWN copy of the game and enemy_keys.json (tower_probe.py). Written up
after the run from the session's inline commands (same logic). It contains no game data: it reads the game's English
string table (_assets/kr1-desktop/strings/en.lua, LuaJIT bytecode inside the prepared engine archive), recovers its
key -> text pairs from the bytecode's string constants (each a ULEB128 length + 5 followed by the bytes; keys are
upper-case identifiers, each followed by its value), and for every template in enemy_keys.json keeps <KEY>_NAME and
the bullets of <KEY>_EXTRA (one per line, leading '- ' removed, lower-cased), plus <KEY>_SPECIAL.

usage: extract_enemy_cards.py <prepared engine exe (runtime/rl-engine/Kingdom Rush.exe)> <enemy_keys.json> <out json>
The full string table is not written anywhere (only the 53 enemies' names and bullets)."""
import json
import re
import sys
import zipfile

exe, keys_path, out_path = sys.argv[1:4]
data = zipfile.ZipFile(exe).read("_assets/kr1-desktop/strings/en.lua")


def uleb(i):
    n = shift = 0
    while True:
        c = data[i]
        i += 1
        n |= (c & 0x7f) << shift
        shift += 7
        if c < 0x80:
            return n, i


KEY = re.compile(rb"^[A-Z0-9_]+$")
pairs, i = {}, 0
while i < len(data) - 2:
    try:
        n, j = uleb(i)
    except IndexError:
        break
    if 6 <= n < 200 and j + n - 5 <= len(data) and KEY.match(data[j:j + n - 5]):
        key = data[j:j + n - 5].decode()
        try:
            m, v0 = uleb(j + n - 5)
        except IndexError:
            break
        if m >= 5 and v0 + m - 5 <= len(data):
            pairs.setdefault(key, data[v0:v0 + m - 5].decode("utf-8", "replace"))
            i = v0 + m - 5
            continue
    i += 1
cards = {}
for entry in json.load(open(keys_path, encoding="utf-8")):
    key = entry.get("key")
    extra = [line.strip().lstrip("-").strip() for line in (pairs.get(f"{key}_EXTRA") or "").split("\n")]
    cards[entry["name"]] = {"key": key, "name": pairs.get(f"{key}_NAME"),
                            "card": [e.lower() for e in extra if e],
                            "special": (pairs.get(f"{key}_SPECIAL") or "").lstrip("*").strip(),
                            "flying_template": bool(entry.get("flying"))}
json.dump(cards, open(out_path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(len(cards), "enemy cards")
