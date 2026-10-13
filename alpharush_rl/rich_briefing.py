"""elite-v3b: prompts for the fair re-test of the language-model commander (plan docs/elite/elite-v3b-fair-retest-plan-
2026-10-12.md). Pure functions over recorded data; nothing here touches a game.

Three prompt formats for the same consult:
- thin-oldformat: what the 8B/24B saw in elite-v3 (commander.SYSTEM, Briefing.text(), the original play texts).
- thin-newformat: the same situation information, with SYSTEM_V2 (a neutral note that P0 is the plan's tuned default,
  the basic game rules every player knows), the official in-game names, and honest play texts that state exactly what
  the translator does (commander.Translator), tagged with stable play ids.
- rich: thin-newformat plus what a human sees on the screen and remembers: tower coverage of every quarter of every
  relevant path, the map (exits, merges, ice, hero start), each enemy's in-game card (official bullets and health),
  hero level/class, and the lives lost per path so far. Never: the whole wave list, timers, random state, exact data
  the game does not show.

Static game knowledge comes from one-off read-only probes of a scratch copy (runtime/rl/elite-v3b/geometry/): path
geometry, exits and ice per level, which towers can hit flying enemies, and the in-game English names and card bullets
(strings/en.lua). Coverage uses the game's ellipse (horizontal radius = range, vertical = 0.7 x range)."""
import json
import math
import re
from pathlib import Path

from alpharush_rl.commander import ELITE_STAGES, LEVEL_CARDS, display, traits

TOWER_NAMES = {"tower_archer_1": "Archer Tower", "tower_archer_2": "Marksman Tower",
               "tower_archer_3": "Sharpshooter Tower", "tower_ranger": "Rangers Hideout",
               "tower_musketeer": "Musketeer Garrison", "tower_mage_1": "Mage Tower", "tower_mage_2": "Adept Tower",
               "tower_mage_3": "Wizard Tower", "tower_arcane_wizard": "Arcane Wizard",
               "tower_sorcerer": "Sorcerer Mage", "tower_engineer_1": "Dwarven Bombard",
               "tower_engineer_2": "Dwarven Artillery", "tower_engineer_3": "Dwarven Howitzer",
               "tower_bfg": "500mm Big Bertha", "tower_tesla": "Tesla x104", "tower_barrack_1": "Militia Barracks",
               "tower_barrack_2": "Footmen Barracks", "tower_barrack_3": "Knights Barracks",
               "tower_paladin": "Holy Order", "tower_barbarian": "Barbarian Hall"}
# Probe (towers.json): the artillery family's main attack bans flying targets; barracks soldiers can neither block
# nor fight flyers; every other family hits flying enemies.
NO_AIR = {"tower_engineer_1", "tower_engineer_2", "tower_engineer_3", "tower_bfg"}
BARRACKS = {"tower_barrack_1", "tower_barrack_2", "tower_barrack_3", "tower_paladin", "tower_barbarian"}
HEROES = {"hero_ignus": ("Ignus", "melee fighter, cannot fight flying enemies"),
          "hero_ingvar": ("Ingvar Bearclaw", "melee fighter, cannot fight flying enemies"),
          "hero_hacksaw": ("Hacksaw", "melee fighter with a ranged sawblade, can hit flying enemies"),
          "hero_magnus": ("Magnus Spellbane", "ranged mage, can hit flying enemies"),
          "hero_elora": ("Elora Wintersong", "ranged frost mage, can hit flying enemies"),
          "hero_bolin": ("Bolin Farslayer", "ranged rifleman, can hit flying enemies")}
MAX_LIVES = 20
BANDS = ("0-25%", "25-50%", "50-75%", "75-100%")
ASPECT = 0.7          # utils.is_inside_ellipse in the game (vertical radius = 0.7 x horizontal)
RALLY_NEAR_PX = 80    # a barracks rally point this close to a path holds that path
MERGE_PX = 30         # two paths closer than this from some node to their ends share the road

SYSTEM_V2 = (
    "You are the tactical commander of an agent playing Kingdom Rush (2011) on Normal difficulty. Towers and upgrades "
    "follow a fixed battle plan found by search. Once per wave (and after an alarm) you choose one play for the hero "
    "and the two spells, Rain of Fire and reinforcements; a fixed executor carries it out until your next choice. "
    "P0 (Steady) is the default the battle plan was searched and tuned with; the other plays deviate from it. "
    "Basic rules: soldiers and reinforcements cannot block or fight flying enemies, and enemies marked 'cannot be "
    "blocked' walk past them; artillery cannot hit flying enemies; armor reduces physical damage (arrows, artillery, "
    "soldiers) and magic resistance reduces mage damage; Rain of Fire falls on an area after a short delay; a hero sent "
    "away walks back to its start point by itself when its health drops below 35% or it stands in a boss's aura, and "
    "takes no other order until healed to 70%. Winning comes first, then keeping lives.")
LABEL_INSTRUCTION = "Answer with one label."
REASON_INSTRUCTION = ("Think about the situation in at most 150 words. Refer to plays only by their id in brackets "
                      "(for example [P2]), and finish with a last line of the form FINAL: [Pk].")
FINAL_JOINER = "\nFINAL ANSWER: "


# -- static knowledge -----------------------------------------------------------------------------------------
def load_cards(path):
    """template -> {"name", "card": [bullets], "special", "flying_template"} (enemy_cards.json)."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def enemy_name(template, cards):
    entry = (cards or {}).get(template) or {}
    return entry.get("name") or display(template)


def enemy_card(template, cards, hp=None):
    """The in-game card line: official bullets (or the special line when the card has none), health rounded."""
    entry = (cards or {}).get(template) or {}
    bits = list(entry.get("card") or [])
    if not bits and entry.get("special"):
        bits = [s.strip().lower() for s in entry["special"].split(",") if s.strip()]
    if entry.get("flying_template") and not any("fly" in b for b in bits):
        bits.insert(0, "flying")
    if hp:
        bits.append(f"health about {_round_hp(hp)}")
    return f"{enemy_name(template, cards)}: {', '.join(bits) if bits else 'no special traits'}"


def _round_hp(hp):
    hp = float(hp)
    if hp < 100:
        return int(round(hp, -1)) or int(hp)
    digits = int(math.floor(math.log10(hp))) - 1
    return int(round(hp, -digits))


def hero_label(template):
    name, kind = HEROES.get(template, (display(template).title(), "hero"))
    return name, kind


# -- map geometry -----------------------------------------------------------------------------------------------
class LevelMap:
    """Static geometry of one level from the probe: paths (lanes of [x, y] nodes), exits, lane-1 ice flags."""

    def __init__(self, geo):
        if geo.get("connections"):
            raise ValueError("path connections are not supported (none on the seven elite stages probed)")
        self.paths = [[[(float(x), float(y)) for x, y in lane] for lane in lanes] for lanes in geo["paths"]]
        self.exits = {int(e["k"]): (float(e["x"]), float(e["y"])) for e in geo.get("exits") or []}
        self.ice = [list(map(bool, flags)) for flags in geo.get("ice") or []]

    @classmethod
    def load(cls, directory, level):
        return cls(json.loads((Path(directory) / f"level_{level}.json").read_text(encoding="utf-8")))

    def lane1(self, pi):
        return self.paths[pi - 1][0]

    def band_nodes(self, pi, band):
        """Nodes of every lane of path pi whose progress (node index / last index) lies in quarter ``band``."""
        out = []
        for lane in self.paths[pi - 1]:
            last = max(1, len(lane) - 1)
            for ni, node in enumerate(lane):
                q = ni / last
                if band / 4 <= q < (band + 1) / 4 or (band == 3 and q == 1.0):
                    out.append(node)
        return out

    def exit_of(self, pi):
        if not self.exits:
            return None
        end = self.lane1(pi)[-1]
        return min(self.exits, key=lambda k: (math.dist(end, self.exits[k]), k))

    def nearest(self, x, y):
        """(path, progress, distance) of the lane-1 node nearest to (x, y)."""
        best = None
        for pi in range(1, len(self.paths) + 1):
            lane = self.lane1(pi)
            last = max(1, len(lane) - 1)
            for ni, node in enumerate(lane):
                d = math.dist(node, (x, y))
                if best is None or d < best[2]:
                    best = (pi, ni / last, d)
        return best

    def joins(self, pi):
        """[(other path, this path's progress where the shared road starts)] for every other path whose lane 1
        stays within MERGE_PX of this path's lane 1 from some node to this path's end."""
        lane = self.lane1(pi)
        out = []
        for pj in range(1, len(self.paths) + 1):
            if pj == pi:
                continue
            other = self.lane1(pj)
            close = [min(math.dist(node, o) for o in other) <= MERGE_PX for node in lane]
            start = len(lane)
            while start > 0 and close[start - 1]:
                start -= 1
            if start < len(lane) - 1:
                out.append((pj, start / max(1, len(lane) - 1)))
        return out

    def ice_span(self, pi):
        flags = self.ice[pi - 1] if pi - 1 < len(self.ice) else []
        idx = [i for i, f in enumerate(flags) if f]
        if not idx:
            return None
        last = max(1, len(flags) - 1)
        return idx[0] / last, idx[-1] / last


def _inside(node, x, y, r):
    dx, dy = (node[0] - x) / r, (node[1] - y) / (r * ASPECT)
    return dx * dx + dy * dy <= 1.0


def tower_text(tower):
    name = TOWER_NAMES.get(tower.get("template"), display(str(tower.get("template"))).title())
    return name + (" (no air)" if tower.get("template") in NO_AIR else "")


def coverage(level_map, towers, pi):
    """[names per quarter] of the towers whose range ellipse reaches a node of path pi in that quarter, plus
    barracks whose rally point holds the path in that quarter (with their soldiers alive)."""
    quarters = [[] for _ in BANDS]
    for band in range(4):
        nodes = level_map.band_nodes(pi, band)
        for t in towers:
            template = t.get("template")
            if template in BARRACKS:
                continue
            r = t.get("range")
            if not isinstance(r, (int, float)) or r <= 0 or not nodes:
                continue
            if any(_inside(node, float(t["x"]), float(t["y"]), float(r)) for node in nodes):
                quarters[band].append(tower_text(t))
    for t in towers:
        if t.get("template") not in BARRACKS or not isinstance(t.get("rally_x"), (int, float)):
            continue
        near = level_map.nearest(t["rally_x"], t["rally_y"])
        if near and near[0] == pi and near[2] <= RALLY_NEAR_PX:
            band = min(3, int(near[1] * 4))
            soldiers = f"{t.get('soldier_count', '?')}/{t.get('max_soldiers', '?')} soldiers"
            quarters[band].append(f"{tower_text(t)} ({soldiers} holding the road at {near[1]:.0%})")
    return [sorted(q) for q in quarters]


def _merge_names(names):
    counts = {}
    for n in names:
        counts[n] = counts.get(n, 0) + 1
    return ", ".join(f"{n} x{c}" if c > 1 else n for n, c in sorted(counts.items()))


# -- the situation texts -----------------------------------------------------------------------------------------
def _title(level):
    stage = ELITE_STAGES.get(level)
    return f"Elite stage {level} ({stage[0]})" if stage else f"Level {level}"


def _notes(level, cards):
    card = dict(LEVEL_CARDS.get(level, {}))
    stage = ELITE_STAGES.get(level)
    notes = ([f"boss: {stage[1]}"] if stage else []) + list(card.get("notes") or [])
    return notes


def _spells(data):
    powers = data.get("powers") or {}
    return f"Spells: Rain of Fire {powers.get('rain_of_fire')}; reinforcements {powers.get('reinforcements')}."


def thin_text_v2(data, cards):
    """Briefing.text()'s information only, with the official in-game names (the thin-newformat situation)."""
    level = data["level"]
    notes = _notes(level, cards) + ([f"{(LEVEL_CARDS.get(level) or {}).get('exits')} exits"]
                                    if (LEVEL_CARDS.get(level) or {}).get("exits") else [])
    sent = (f"wave {data['wave']} of {data['wave_total']} has been sent" if (data.get("wave") or 0) > 0
            else f"no wave has been sent yet ({data['wave_total']} in total)")
    lines = [_title(level) + (f" [{'; '.join(notes)}]" if notes else "")
             + f". Lives {data['lives']}, gold {float(data['gold']):g}; {sent}."]
    preview = data.get("next_wave")
    if preview and preview.get("paths"):
        lines.append("Next wave (flag preview): " + "; ".join(
            f"path {p['path']}: " + ", ".join(f"{c} {enemy_name(t, cards)}{_tag(t)}" for t, c in p["enemies"])
            for p in preview["paths"]) + ".")
    else:
        lines.append("No next-wave flag is showing.")
    field = "; ".join(f"{g['count']} {enemy_name(g['template'], cards)} on path {g['path']} at {BANDS[g['band']]}"
                      for g in data["field"])
    lines.append(f"On the field: {field or 'no enemies'}.")
    lines.extend(_boss_lines(data, cards))
    for hero in data["heroes"]:
        name, _ = hero_label(hero["template"])
        where = {None: "", True: ", at its start point", False: ", away from its start point"}[hero["at_home"]]
        lines.append(f"Hero {name}: " + (f"health {hero['health']}%{where}." if hero["alive"]
                                          else "fallen, respawning."))
    lines.append(_spells(data))
    return "\n".join(lines)


def _tag(template):
    names = {"egg": "lays eggs", "heal": "heals", "unblockable": "cannot be blocked", "aura": "deadly aura",
             "boss": "boss"}
    tags = [names[t] for t in traits(template)]
    return f" ({', '.join(tags)})" if tags else ""


def _boss_lines(data, cards):
    out = []
    for boss in data.get("bosses") or []:
        name = enemy_name(boss["template"], cards)
        if boss["dormant"]:
            out.append(f"Boss {name}: waiting.")
        else:
            out.append(f"Boss {name}: on path {boss['path']} at {BANDS[boss['band']]}, health about "
                       f"{25 * boss['health']}%.")
    return out


def rich_text(data, capture, ledger, level_map, cards):
    """The rich situation: the thin information plus map, defences, enemy cards, hero details and memory."""
    level = data["level"]
    lines = [_title(level) + (f" [{'; '.join(_notes(level, cards))}]" if _notes(level, cards) else "")
             + f". Lives {data['lives']} of {MAX_LIVES}, gold {float(data['gold']):g}; "
             + (f"wave {data['wave']} of {data['wave_total']} has been sent."
                if (data.get("wave") or 0) > 0 else f"no wave has been sent yet ({data['wave_total']} in total).")]
    preview = data.get("next_wave")
    active = sorted({p["path"] for p in (preview or {}).get("paths") or []} | {g["path"] for g in data["field"]}
                    | {b["path"] for b in data.get("bosses") or [] if not b["dormant"]}
                    | {int(k) for k in (ledger or {}).get("lives_lost", {}) if str(k).isdigit()})
    active = [p for p in active if 1 <= p <= len(level_map.paths)] or list(range(1, len(level_map.paths) + 1))
    # Map.
    lines.append("Map (paths are numbered as in the flag preview; % = how far along the path toward its exit):")
    for pi in active:
        bits = []
        exit_k = level_map.exit_of(pi)
        if exit_k is not None:
            bits.append(f"leaves by exit {exit_k}")
        for pj, at in level_map.joins(pi):
            bits.append(f"shares the road with path {pj} from {at:.0%}")
        ice = level_map.ice_span(pi)
        if ice:
            bits.append(f"ice from {ice[0]:.0%} to {ice[1]:.0%}")
        lines.append(f"- Path {pi}: " + ("; ".join(bits) if bits else "no special features") + ".")
    others = [pi for pi in range(1, len(level_map.paths) + 1) if pi not in active]
    if others:
        lines.append(f"- Paths {', '.join(map(str, others))}: no enemies now or in the next wave.")
    heroes = (capture or {}).get("heroes") or []
    for hero in heroes:
        if isinstance(hero.get("home_x"), (int, float)):
            near = level_map.nearest(hero["home_x"], hero["home_y"])
            if near:
                exit_k = level_map.exit_of(near[0])
                lines.append(f"- Hero start point: next to path {near[0]} at {near[1]:.0%}"
                             + (f" (that path leaves by exit {exit_k})" if exit_k is not None else "") + ".")
            break
    # Defences.
    towers = (capture or {}).get("towers") or []
    lines.append("Your defences reaching each quarter of those paths:")
    for pi in active:
        quarters = coverage(level_map, towers, pi)
        lines.append(f"- Path {pi}: " + "; ".join(
            f"{BANDS[b]} {_merge_names(q) if q else 'nothing'}" for b, q in enumerate(quarters)) + ".")
    # Enemies.
    templates, hp = [], {}
    for g in data["field"]:
        templates.append(g["template"])
    for p in (preview or {}).get("paths") or []:
        templates.extend(t for t, _ in p["enemies"])
    for e in (capture or {}).get("enemies") or []:
        if isinstance(e.get("hp_max"), (int, float)):
            hp.setdefault(e.get("template"), e["hp_max"])
    for p in ((capture or {}).get("next_wave") or {}).get("paths") or []:
        for s in (p.get("spawns") if isinstance(p, dict) else None) or []:
            if isinstance(s, dict) and isinstance(s.get("hp"), (int, float)):
                hp.setdefault(s.get("template"), s["hp"])
    field = "; ".join(f"{g['count']} {enemy_name(g['template'], cards)} on path {g['path']} at {BANDS[g['band']]}"
                      for g in data["field"])
    lines.append(f"On the field: {field or 'no enemies'}.")
    lines.extend(_boss_lines(data, cards))
    if preview and preview.get("paths"):
        lines.append("Next wave (flag preview): " + "; ".join(
            f"path {p['path']}: " + ", ".join(f"{c} {enemy_name(t, cards)}" for t, c in p["enemies"])
            for p in preview["paths"]) + ".")
    else:
        lines.append("No next-wave flag is showing.")
    seen = []
    for t in templates:
        if t not in seen:
            seen.append(t)
    if seen:
        lines.append("Enemy cards: " + "; ".join(enemy_card(t, cards, hp.get(t)) for t in seen) + ".")
    # Hero, spells, memory.
    for hero in data["heroes"]:
        name, kind = hero_label(hero["template"])
        live = next((h for h in heroes if h.get("template") == hero["template"]), {})
        level_txt = f", level {live['level']}" if isinstance(live.get("level"), int) else ""
        if not hero["alive"]:
            lines.append(f"Hero {name} ({kind}{level_txt}): fallen, respawning.")
            continue
        where = ""
        if hero["at_home"] is True:
            where = ", at its start point"
        elif isinstance(live.get("x"), (int, float)):
            near = level_map.nearest(live["x"], live["y"])
            where = f", away from its start point (near path {near[0]} at {near[1]:.0%})" if near else ""
        lines.append(f"Hero {name} ({kind}{level_txt}): health {hero['health']}%{where}.")
    lines.append(_spells(data))
    lost = (ledger or {}).get("lives_lost") or {}
    deaths = (ledger or {}).get("hero_deaths") or 0
    if lost or deaths:
        parts = [f"path {k}: {v}" if str(k).isdigit() else f"unclear where: {v}"
                 for k, v in sorted(lost.items(), key=lambda kv: (not str(kv[0]).isdigit(), str(kv[0]).zfill(3)))]
        lines.append("So far this game: lives lost " + (", ".join(parts) if parts else "none")
                     + (f"; the hero has fallen {deaths} time{'s' if deaths != 1 else ''}" if deaths else "") + ".")
    else:
        lines.append("So far this game: no lives lost; the hero has not fallen.")
    return "\n".join(lines)


# -- honest play texts -------------------------------------------------------------------------------------------
_CAST = re.compile(r"covered (\d+)% of its path")


def cast_percent(plays, default=50):
    for play in plays:
        match = _CAST.search(play.get("text") or "")
        if match:
            return int(match.group(1))
    return default


def play_text_v2(play, data, cast, cards, boss=False):
    """Exactly what the translator does for this play (commander.Translator), tagged with the play id. ``boss``: the
    plan's boss gene (while a boss is alive, steady casts go only at the boss)."""
    hero = bool(data.get("heroes"))
    pid = play["id"]
    on_field = {g["template"] for g in data["field"]} | {b["template"] for b in data.get("bosses") or []
                                                          if not b["dormant"]}
    if play["name"] == "steady":
        return (f"[{pid}] Steady (the plan's tuned default): " + ("the hero stays at its start point; " if hero else "")
                + f"once any enemy has covered {cast}% of its path, Rain of Fire and reinforcements are each cast at "
                "the most advanced enemy as soon as they are ready"
                + (" (while a boss is alive: only at the boss)." if boss else "."))
    threat = play.get("threat") or {}
    what = (f"{threat.get('count')} {enemy_name(threat.get('template'), cards)}{_tag(threat.get('template'))} "
            f"on path {threat.get('path')}")
    target = enemy_name(threat.get("template"), cards)
    there = " (some are on the field now)" if threat.get("template") in on_field else ""
    release = (f"held spells are released if 2 lives are lost within 10 seconds or no {target} can come any more")
    if play["name"] == "counter":
        hero_txt = (f"the hero walks to the point 35% along path {threat.get('path')} (if it cannot get there: to the "
                    f"leading {target}); " if hero else "")
        return (f"[{pid}] Counter {what}{there}: " + hero_txt
                + f"Rain of Fire is held until {target} are on the field, then cast at them; reinforcements are called "
                f"at the leading {target} whenever they are on the field (otherwise as in Steady); {release}.")
    if play["name"] == "guard_exit":
        exit_k = play["hero"]["exit"]
        heading = re.search(r"\(next wave there: path ([^)]*)\)", play.get("text") or "")
        # Only the positive fact the generator knew; "no threat there" could contradict the rich map.
        via = f" (next-wave enemies heading there: path {heading.group(1)})" if heading else ""
        return (f"[{pid}] Guard exit {exit_k}{via}: the hero walks to the point 90% along the path that leaves by exit "
                f"{exit_k}; the spells are used as in Steady.")
    if play["name"] == "focus":
        return (f"[{pid}] Focus both spells on {what}{there}: Rain of Fire and reinforcements are both held until "
                f"{target} are on the field, then both are cast at them; "
                + ("the hero stays at its start point; " if hero else "") + f"{release}.")
    raise ValueError(f"unknown play {play['name']!r}")


def options_v2(plays, data, cards, boss=False):
    cast = cast_percent(plays)
    return [play_text_v2(p, data, cast, cards, boss) for p in plays]


# -- requests -------------------------------------------------------------------------------------------------------
def user_text(situation):
    return situation + "\nChoose the play for the coming wave."


def label_requests(user, options, rotations=6):
    """The label-mode rotations exactly as strategy_brain.LanguageBrain._ask builds them: [(order, user text)]."""
    n = len(options)
    labels = [chr(65 + i) for i in range(n)]
    out = []
    for shift in range(min(n, rotations)):
        order = [(i + shift) % n for i in range(n)]
        out.append((order, user + "\n\nOptions:\n" + "\n".join(f"{label}. {options[i]}" for label, i in
                                                                zip(labels, order)) + "\n\n" + LABEL_INSTRUCTION))
    return labels, out


def reason_request(user, options):
    """The reasoning prompt: options in canonical order (P-ids in the texts), free text ending in FINAL: [Pk]."""
    labels = [chr(65 + i) for i in range(len(options))]
    return (user + "\n\nOptions:\n" + "\n".join(f"{label}. {text}" for label, text in zip(labels, options))
            + "\n\n" + REASON_INSTRUCTION)


_FINAL = re.compile(r"FINAL:\s*\[?(P\d)\]?", re.IGNORECASE)


def parse_final(text, ids):
    """The play index named by the last 'FINAL: [Pk]' line, or None."""
    found = _FINAL.findall(text or "")
    if not found:
        return None
    pid = found[-1].upper()
    return ids.index(pid) if pid in ids else None


def strip_final(text):
    """The reasoning without its FINAL line(s), for conditioned label scoring."""
    return re.sub(r"\n?[^\n]*FINAL:[^\n]*", "", text or "").rstrip()


FORBIDDEN = ("level_paths", "level_path_wave_counts", "wave_db", "timer", "seed", "random", "rng", "next_wave_timer",
             "spawn_counter")


def leaks(text):
    """Names of forbidden (non-human) data that appear in a prompt (should be empty)."""
    lower = text.lower()
    return [word for word in FORBIDDEN if word in lower]
