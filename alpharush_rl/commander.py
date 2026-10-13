"""In-battle tactical commander (action scope v4): once per wave a brain picks one of at most six plays, and a
deterministic translator turns the current play into the game's own aimed-spell and hero-move options.

Plan: docs/elite/elite-v3-commander-plan-2026-10-11.md, sections 2.2-2.3 and 4.

* ``Briefing``: only what a player sees on screen (lives, gold, wave n/total, the next-wave flag's preview per
  path and enemy type, visible enemies by type/path/progress band, hero health band and position, spell
  readiness) plus a written level card. Never exact health, armor, timers, later waves or random state.
* ``PlayGenerator``: at most six plays per consult, deterministic. P0 steady (the executor's own spell rule,
  hero at its start point); P1/P2 counter the next wave's two main threats (hero intercepts on their path, Rain
  of Fire saved for them, reinforcements block them); P3 guards the other exit of a multi-exit level; P4 focuses
  both spells on threat 1 with the hero at its start point.
* ``Translator``: shared by every brain, no learned parameters. It picks among the host's aim_power/move_hero
  catalog options by option name and target, and owns the reflexes (hero retreat below 35% health, staying out
  of listed boss auras) and the pace rules (hero orders at least 3 s apart, one cast per readiness, a saved
  spell released when its target appears or the reservation expires).
* Brains, ``choose(briefing, plays, key) -> (index, record)``: ``PassiveBrain`` (never acts: the wrapped
  executor plays exactly as before), ``RuleCommander`` (a deliberately strong rule baseline), ``RandomCommander``
  (uniform, seeded per consult), ``PlaceboCommander`` (the rule brain with every command one tick late),
  ``LanguageCommander`` (the 8B, scored like strategy_brain.LanguageBrain) and ``ForcedChoice`` (forks).
* ``CommanderPolicy`` wraps the plan executor (or the operator network): it consults the brain when a new
  next-wave flag shows and on an alarm, issues the translator's commands at decisions it asks for itself
  (``wants_decision``), and leaves every other decision to the wrapped policy, whose menu never shows the
  commander's options (nor use_power unless the brain is passive).
"""
from __future__ import annotations

import copy
import hashlib
import math
import random
import time

from .menus import build_menu, validate_menu
from .search import BuildOrderPolicy, check_genome, genome_id
from .strategy_brain import ELITE_STAGES, LanguageBrain

TICKS_PER_SECOND = 60
MAX_PLAYS = 6
HERO_ORDER_GAP = 3 * TICKS_PER_SECOND  # hero orders (voluntary or reflex) at least 3 s apart
RETREAT_GAP = 10 * TICKS_PER_SECOND    # at most one retreat reflex per 10 s
TRIGGER_GAP = 30       # decisions the commander asks for itself at least 30 ticks apart
RETREAT_HP = 0.35      # the hero walks home below this health fraction...
RETURN_HP = 0.7        # ...and takes no other order until healed to this one
ARRIVED_PX = 25        # a rally point this close to an option's point already serves it
CLUSTER_PX = 60        # a cluster point this close to an enemy hits it
BOSS_RADIUS = 80       # the plan's boss gene: spells only this close to an awake boss (search.BOSS_RADIUS)
RESERVE_LOSS_LIVES, RESERVE_LOSS_TICKS = 2, 10 * TICKS_PER_SECOND  # a saved spell is released after 2 lives
# lost within 10 s
ALARM_LIVES = 3        # lives lost since the last consult that call an extra consult (once per wave)
EXIT_PROGRESS = 0.85   # an enemy this far along its path shows where the path leaves the map
# Listed boss auras the hero stays out of (px): Kingpin, Blackburn's smash, Cerberus' death blast, Myconid spores.
AURAS = {"eb_kingpin": 65, "eb_blackburn": 106, "enemy_demon_cerberus": 120, "eb_myconid": 110}
# Static enemy traits a player learns from the in-game encyclopedia (docs/elite/elite-stages-study-2026-10-06.json):
# lays eggs (spiders, which a blocker may keep from laying), heals itself or others, cannot be blocked (flying,
# skating, Kingpin), a deadly aura, boss. Any "eb_" template is a boss.
TRAITS = {
    "enemy_spider_big": ("egg",), "enemy_spider_rotten": ("egg",),
    "enemy_shaman": ("heal",), "enemy_swamp_thing": ("heal",), "enemy_forest_troll": ("heal",),
    "enemy_troll_brute": ("heal",), "enemy_werewolf": ("heal",), "enemy_lycan_werewolf": ("heal",),
    "enemy_demon_mage": ("heal",),
    "enemy_troll_skater": ("unblockable",), "enemy_demon_gulaemon": ("unblockable",),
    "enemy_gargoyle": ("unblockable",), "enemy_demon_imp": ("unblockable",), "enemy_witch": ("unblockable",),
    "eb_kingpin": ("boss", "heal", "unblockable", "aura"), "eb_gulthak": ("boss", "heal"),
    "eb_blackburn": ("boss", "aura"), "eb_myconid": ("boss", "aura"), "enemy_demon_cerberus": ("boss", "aura"),
}
TRAIT_TEXT = {"egg": "lays eggs", "heal": "heals", "unblockable": "cannot be blocked", "aura": "deadly aura",
              "boss": "boss"}
TRAIT_WEIGHT = {"egg": 3.0, "heal": 2.0, "unblockable": 2.0, "aura": 1.0, "boss": 10.0}
# The written level card: what one look at the map tells (exits, special rules).
LEVEL_CARDS = {15: {"notes": ["fallen soldiers rise as enemies (graveyard)"]},
               17: {"exits": 2, "notes": ["the Kingpin cannot be blocked"]},
               19: {"exits": 3, "notes": ["skaters cannot be blocked on ice", "Ulgukhai is hurt only while blocked"]},
               20: {"notes": ["burning floors hurt soldiers", "Cerberus sleeps until the last wave"]},
               21: {"notes": ["burning floors hurt soldiers", "Moloch sits until the last wave"]},
               22: {"notes": ["fallen soldiers rise as enemies (graveyard)"]},
               24: {"exits": 2}, 26: {"exits": 3, "notes": ["Blackburn stuns nearby towers"]}}
SYSTEM = ("You are the tactical commander of an agent playing Kingdom Rush (2011) on Normal difficulty. Towers and "
          "upgrades follow a fixed battle plan; once per wave you choose one play for the hero and the two spells "
          "(Rain of Fire and reinforcements), and a fixed executor carries it out. Winning comes first, then "
          "keeping lives. Reply with exactly one option label from the list.")
POWER_PARTS = {1: "fire", 2: "reinf"}
STEADY = {"mode": "steady"}


def _n(value, default=0.0):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        return default
    return float(value)


def _int(value):
    return isinstance(value, int) and not isinstance(value, bool)


def _finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _distance(ax, ay, bx, by):
    values = [_n(v, math.nan) for v in (ax, ay, bx, by)]
    if any(math.isnan(v) for v in values):
        return math.inf
    return math.hypot(values[0] - values[2], values[1] - values[3])


def _enemies(state):
    return [e for e in state.get("enemies") or [] if isinstance(e, dict) and _int(e.get("id"))]


def _awake(state):
    return [e for e in _enemies(state) if not e.get("dormant")]


def traits(template):
    out = TRAITS.get(template, ())
    if isinstance(template, str) and template.startswith("eb_") and "boss" not in out:
        out = ("boss",) + out
    return out


def _is_boss(enemy):
    return bool(enemy.get("boss")) or "boss" in traits(enemy.get("template"))


def _heroes(state):
    return sorted((h for h in state.get("heroes") or [] if isinstance(h, dict) and _int(h.get("id"))),
                  key=lambda h: h["id"])


def _hero_fraction(hero):
    return _n(hero.get("hp")) / max(1.0, _n(hero.get("hp_max"), 1.0))


def power_ready(state, power):
    """Spell readiness as the GUI button shows it (the same rule as the host's use_power offers)."""
    for button in state.get("powers_ui") or []:
        if isinstance(button, dict) and button.get("id") == power:
            return button.get("mode") not in (None, "locked", "cooldown", "missing")
    return False


def _power_state(state, power):
    for button in state.get("powers_ui") or []:
        if isinstance(button, dict) and button.get("id") == power:
            mode = button.get("mode")
            if mode in (None, "locked", "missing"):
                return "locked"
            return "cooling down" if mode == "cooldown" else "ready"
    return "locked"


def display(template):
    name = str(template or "?")
    for prefix in ("enemy_", "eb_", "hero_"):
        if name.startswith(prefix):
            name = name[len(prefix):]
    return name.replace("_", " ")


def _band(progress):
    """Progress band 0-3 (quarters of the path), as a player judges it by eye."""
    return min(3, max(0, int(_n(progress) * 4)))


def _next_wave(state):
    """The next-wave flag's preview {"group", "paths": [{"path", "enemies": [[template, count], ...]}]}, or None."""
    preview = state.get("next_wave")
    if not isinstance(preview, dict):
        return None
    paths = preview.get("paths")
    paths = list(paths.values()) if isinstance(paths, dict) else paths if isinstance(paths, list) else []
    by_path = {}
    for entry in paths:
        if not isinstance(entry, dict) or not _int(entry.get("path_index")):
            continue
        spawns = entry.get("spawns")
        spawns = list(spawns.values()) if isinstance(spawns, dict) else spawns if isinstance(spawns, list) else []
        counts = by_path.setdefault(entry["path_index"], {})
        for spawn in spawns:
            if isinstance(spawn, dict) and isinstance(spawn.get("template"), str):
                count = spawn.get("count") if _int(spawn.get("count")) and spawn["count"] > 0 else 1
                counts[spawn["template"]] = counts.get(spawn["template"], 0) + count
    group = preview.get("group_idx")
    return {"group": group if _int(group) else None,
            "paths": [{"path": path, "enemies": [[t, c] for t, c in sorted(counts.items())]}
                      for path, counts in sorted(by_path.items()) if counts]}


class Briefing:
    """What a player sees on screen at a consult; ``data`` is plain JSON, ``text()`` the prompt's situation."""

    def __init__(self, data):
        self.data = data

    @classmethod
    def from_state(cls, state):
        level = state.get("level_idx")
        field, bosses = {}, []
        for e in _enemies(state):
            path = e.get("path_index") if _int(e.get("path_index")) else 0
            if _is_boss(e):
                boss = {"template": e.get("template"), "path": path, "band": _band(e.get("path_progress")),
                        "dormant": bool(e.get("dormant"))}
                if not boss["dormant"]:  # a dormant boss (sleeping Cerberus, seated Moloch) shows no health bar
                    fraction = _n(e.get("hp")) / max(1.0, _n(e.get("hp_max"), 1.0))
                    boss["health"] = min(4, max(0, math.ceil(fraction * 4)))
                bosses.append(boss)
                continue
            key = (str(e.get("template")), path, _band(e.get("path_progress")))
            field[key] = field.get(key, 0) + 1
        heroes = []
        for hero in _heroes(state):
            home = None
            if _finite(hero.get("home_x")) and _finite(hero.get("home_y")):  # v4 exports the start point
                home = _distance(hero.get("x"), hero.get("y"), hero["home_x"], hero["home_y"]) <= 2 * ARRIVED_PX
            heroes.append({"template": hero.get("template"), "alive": not hero.get("dead"),
                           "health": 10 * round(_hero_fraction(hero) * 10), "x": round(_n(hero.get("x"))),
                           "y": round(_n(hero.get("y"))), "at_home": home})
        card = dict(LEVEL_CARDS.get(level, {}))
        if level in ELITE_STAGES:
            card["stage"], card["boss"] = ELITE_STAGES[level]
        groups = sorted(field.items(), key=lambda kv: (kv[0][1], -kv[0][2], kv[0][0]))
        return cls({"level": level, "wave": state.get("wave"), "wave_total": state.get("wave_total"),
                    "lives": state.get("lives"), "gold": state.get("gold"), "next_wave": _next_wave(state),
                    "field": [{"template": t, "path": p, "band": b, "count": c} for (t, p, b), c in groups],
                    "bosses": sorted(bosses, key=lambda b: (str(b["template"]), b["path"])), "heroes": heroes,
                    "powers": {"rain_of_fire": _power_state(state, 1), "reinforcements": _power_state(state, 2)},
                    "card": card})

    def text(self):
        d = self.data
        card = d.get("card") or {}
        title = f"Elite stage {d['level']} ({card['stage']})" if card.get("stage") else f"Level {d['level']}"
        notes = ([f"boss: {card['boss']}"] if card.get("boss") else []) + list(card.get("notes") or []) \
            + ([f"{card['exits']} exits"] if card.get("exits") else [])
        sent = (f"wave {d['wave']} of {d['wave_total']} has been sent" if _n(d.get("wave")) > 0
                else f"no wave has been sent yet ({d['wave_total']} in total)")
        lines = [title + (f" [{'; '.join(notes)}]" if notes else "")
                 + f". Lives {d['lives']}, gold {_n(d['gold']):g}; {sent}."]
        preview = d.get("next_wave")
        if preview and preview["paths"]:
            lines.append("Next wave (flag preview): " + "; ".join(
                f"path {p['path']}: " + ", ".join(f"{c} {display(t)}" + _trait_note(t) for t, c in p["enemies"])
                for p in preview["paths"]) + ".")
        else:
            lines.append("No next-wave flag is showing.")
        bands = ("0-25%", "25-50%", "50-75%", "75-100%")
        field = "; ".join(f"{g['count']} {display(g['template'])} on path {g['path']} at {bands[g['band']]}"
                          for g in d["field"])
        lines.append(f"On the field: {field or 'no enemies'}.")
        for boss in d["bosses"]:
            if boss["dormant"]:
                lines.append(f"Boss {display(boss['template'])}: waiting.")
            else:
                lines.append(f"Boss {display(boss['template'])}: on path {boss['path']} at {bands[boss['band']]}, "
                             f"health about {25 * boss['health']}%.")
        for hero in d["heroes"]:
            where = {None: "", True: ", at its start point", False: ", away from its start point"}[hero["at_home"]]
            status = f"health {hero['health']}%{where}." if hero["alive"] else "fallen, respawning."
            lines.append(f"Hero {display(hero['template'])}: {status}")
        powers = d["powers"]
        lines.append(f"Spells: Rain of Fire {powers['rain_of_fire']}; reinforcements {powers['reinforcements']}.")
        return "\n".join(lines)


def _trait_note(template):
    names = [TRAIT_TEXT[t] for t in traits(template)]
    return f" ({', '.join(names)})" if names else ""


def threat_score(template, count):
    weight = 1.0 + sum(TRAIT_WEIGHT[t] for t in traits(template))
    return round(weight * (1.0 + math.log2(max(1, count))), 4)


class PlayGenerator:
    """At most MAX_PLAYS plays from a briefing and the map geometry learned so far (exit points, start point,
    which exit each path leaves by). A play is plain JSON: hero order, Rain of Fire rule, reinforcements rule."""

    def __init__(self, cast=50):
        self.cast = cast

    @staticmethod
    def threats(briefing):
        """(template, path) groups by threat score: the next-wave preview, or the field when no flag shows."""
        data = briefing.data if isinstance(briefing, Briefing) else briefing
        counts = {}
        if data.get("next_wave") and data["next_wave"]["paths"]:
            for entry in data["next_wave"]["paths"]:
                for template, count in entry["enemies"]:
                    counts[(template, entry["path"])] = counts.get((template, entry["path"]), 0) + count
        else:
            for group in data.get("field") or []:
                key = (group["template"], group["path"])
                counts[key] = counts.get(key, 0) + group["count"]
            for boss in data.get("bosses") or []:
                if not boss["dormant"]:
                    counts[(boss["template"], boss["path"])] = counts.get((boss["template"], boss["path"]), 0) + 1
        out = [{"template": t, "path": p, "count": c, "traits": list(traits(t)), "score": threat_score(t, c)}
               for (t, p), c in counts.items()]
        # str(): a boss without a template (None) ties with named groups without ever comparing None to a string.
        return sorted(out, key=lambda x: (-x["score"], str(x["template"]), x["path"]))

    def plays(self, briefing, geometry=None):
        geometry = geometry or {}
        data = briefing.data if isinstance(briefing, Briefing) else briefing
        hero = bool(data.get("heroes"))
        home = {"to": "home"} if hero else None
        threats = self.threats(data)
        plays = [{"id": "P0", "name": "steady", "threat": None, "hero": home, "fire": STEADY, "reinf": STEADY,
                  "text": (("Steady: the hero holds its start point; " if hero else "Steady: ")
                           + f"Rain of Fire hits the leading enemy once one has covered {self.cast}% of its path; "
                           + "reinforcements block the leading enemy.")}]
        path_exit = geometry.get("path_exit") or {}
        for index, threat in enumerate(threats[:2]):
            target = {"template": threat["template"], "path": threat["path"]}
            what = (f"{threat['count']} {display(threat['template'])}{_trait_note(threat['template'])} "
                    f"on path {threat['path']}")
            intercept = {"to": "path", **target, "exit": path_exit.get(str(threat["path"]))}
            plays.append({"id": f"P{index + 1}", "name": "counter", "threat": threat,
                          "hero": intercept if hero else None,
                          "fire": {"mode": "reserve", **target}, "reinf": {"mode": "block", **target},
                          "text": (f"Counter {what}: "
                                   + (f"the hero intercepts them upstream on path {threat['path']}; " if hero else "")
                                   + "Rain of Fire is saved for them; reinforcements block them.")})
        exits = geometry.get("exits") or {}
        home_exit = geometry.get("home_exit")
        others = [k for k in sorted(exits, key=int) if k != home_exit]
        if hero and len(exits) >= 2 and others:
            total = sum(t["score"] for t in threats) or 1.0
            share = {k: sum(t["score"] for t in threats if path_exit.get(str(t["path"])) == k) / total for k in others}
            exit_k = max(others, key=lambda k: (share[k], -int(k)))
            paths = sorted({t["path"] for t in threats if path_exit.get(str(t["path"])) == exit_k})
            via = f" (next wave there: path {', '.join(map(str, paths))})" if paths else ""
            plays.append({"id": "P3", "name": "guard_exit", "threat": None, "exit_threat": round(share[exit_k], 4),
                          "hero": {"to": "exit", "exit": exit_k}, "fire": STEADY, "reinf": STEADY,
                          "text": (f"Guard exit {exit_k}{via}: the hero moves to exit {exit_k}; spells as in the "
                                   "steady play.")})
        if threats:
            threat = threats[0]
            target = {"template": threat["template"], "path": threat["path"]}
            plays.append({"id": "P4", "name": "focus", "threat": threat, "hero": home,
                          "fire": {"mode": "reserve", **target}, "reinf": {"mode": "reserve", **target},
                          "text": (f"Focus both spells on {threat['count']} {display(threat['template'])}"
                                   f"{_trait_note(threat['template'])} on path {threat['path']} (held until they "
                                   "appear)" + ("; the hero holds its start point." if hero else "."))})
        return plays[:MAX_PLAYS]


def _point(item):
    return item["action"].get("x"), item["action"].get("y")


class Translator:
    """Turns the current play into one catalog option at a time; deterministic, shared by every brain."""

    def __init__(self, cast=50, boss_focus=False, parts=("spells", "hero")):
        self.cast, self.boss_focus = cast, boss_focus
        self.parts = frozenset(parts)  # what this commander controls: "spells" and/or "hero"
        self.play = {"id": "none", "hero": None, "fire": STEADY, "reinf": STEADY}
        self.reserve_wave = 0
        self.expired = set()          # saved-spell parts ("fire"/"reinf") released by expiry this play
        self.cast_issued = {1: False, 2: False}  # one cast per readiness
        self.hero_tick = self.retreat_tick = None
        self.moved_hero = False       # the commander has moved the hero (the steady play only then brings it home)
        self.retreating = set()       # heroes sent home that take no other order until healed
        self.losses = []              # ticks of each life lost
        self.lives = None

    def owned_powers(self):
        """Powers the current play casts itself: those whose spec is not steady (and only if this commander controls
        the spells); the executor keeps casting the others exactly as without a commander."""
        if "spells" not in self.parts:
            return set()
        return {power for power, part in POWER_PARTS.items() if (self.play.get(part) or STEADY)["mode"] != "steady"}

    def hero_out(self):
        """True while the play sends the hero away from its start point (hero reflexes apply only then)."""
        spec = self.play.get("hero")
        return "hero" in self.parts and bool(spec) and spec.get("to") != "home"

    def adopt(self, play, state):
        self.play = play
        self.expired = set()
        self.reserve_wave = _n(state.get("wave"), 0) + (1 if _next_wave(state) else 0)

    def observe(self, state):
        tick, lives = _n(state.get("tick")), state.get("lives")
        if _int(lives):
            if _int(self.lives) and lives < self.lives:
                self.losses.extend([tick] * (self.lives - lives))
            self.lives = lives
        self.losses = [t for t in self.losses if tick - t < RESERVE_LOSS_TICKS]
        for power in (1, 2):
            if not power_ready(state, power):
                self.cast_issued[power] = False
        for hero in _heroes(state):
            if hero.get("dead") or _hero_fraction(hero) >= RETURN_HP:
                self.retreating.discard(hero["id"])
        for part in ("fire", "reinf"):
            spec = self.play.get(part) or STEADY
            if spec["mode"] == "reserve" and part not in self.expired and not self._held(spec, state):
                self.expired.add(part)

    def _held(self, spec, state):
        """A saved spell stays saved while its target may still come: not after 2 lives lost within 10 s; while
        it is on the field, its wave has not started or is still spawning, or the next flag shows it."""
        if len(self.losses) >= RESERVE_LOSS_LIVES:
            return False
        if any(e.get("template") == spec["template"] for e in _awake(state)):
            return True
        if _n(state.get("wave"), 0) < self.reserve_wave or state.get("wave_spawning"):
            return True
        preview = _next_wave(state)
        return bool(preview) and any(t == spec["template"] for p in preview["paths"] for t, _ in p["enemies"])

    def command(self, state, menu, *, commit=False):
        """{"item", "kind", "reason", ...} for the option to take now, or None. ``commit`` books the pace."""
        tick = _n(state.get("tick"))
        hero, spells = "hero" in self.parts, "spells" in self.parts
        out = ((hero and self.hero_out() and self._reflex(state, menu, tick)) or (spells and self._spells(state, menu))
               or (hero and self._hero(state, menu, tick)) or None)
        if out and commit:
            action = out["item"]["action"]
            if action["action"] == "move_hero":
                self.hero_tick = tick
                self.moved_hero = True
                if out["kind"] in ("retreat", "aura"):
                    self.retreat_tick = tick
                    self.retreating.add(action["hero_id"])
            else:
                self.cast_issued[action["power"]] = True
        return out

    # -- hero -------------------------------------------------------------------------------------------
    def _moves(self, menu):
        """(hero, {option: item}) for the lowest-id hero the catalog lets the player move, else (None, {})."""
        heroes = {}
        for item in menu:
            action = item["action"]
            if action.get("action") == "move_hero":
                heroes.setdefault(action["hero_id"], {})[action["option"]] = item
        if not heroes:
            return None, {}
        hero_id = min(heroes)
        return hero_id, heroes[hero_id]

    @staticmethod
    def _auras(state):
        return [(e.get("x"), e.get("y"), AURAS[e["template"]]) for e in _awake(state) if e.get("template") in AURAS]

    @staticmethod
    def _inside(point, auras):
        return any(_distance(point[0], point[1], x, y) <= r for x, y, r in auras)

    def _reflex(self, state, menu, tick):
        hero_id, moves = self._moves(menu)
        hero = next((h for h in _heroes(state) if h["id"] == hero_id), None)
        home = moves.get("home")
        if hero is None or home is None or not self._paced(tick):
            return None
        if self.retreat_tick is not None and tick - self.retreat_tick < RETREAT_GAP:
            return None
        rally = (hero.get("rally_x"), hero.get("rally_y"))
        if _distance(*rally, *_point(home)) <= ARRIVED_PX:
            return None
        if _hero_fraction(hero) < RETREAT_HP:
            return {"item": home, "kind": "retreat", "reason": f"hero health below {RETREAT_HP:.0%}"}
        auras = self._auras(state)
        if self._inside(rally, auras) and not self._inside(_point(home), auras):
            return {"item": home, "kind": "aura", "reason": "hero inside a listed boss aura"}
        return None

    def _paced(self, tick):
        return self.hero_tick is None or tick - self.hero_tick >= HERO_ORDER_GAP

    def _hero(self, state, menu, tick):
        spec = self.play.get("hero")
        hero_id, moves = self._moves(menu)
        hero = next((h for h in _heroes(state) if h["id"] == hero_id), None)
        if not spec or hero is None or not self._paced(tick):
            return None
        if spec["to"] == "home" and not self.moved_hero:
            return None  # never moved by the commander: the hero is where the executor alone would have it
        names = [spec["to"]] if spec["to"] == "home" else [f"exit:{spec['exit']}"] if spec["to"] == "exit" else []
        if spec["to"] == "path":
            targets = sorted((e for e in _awake(state) if e.get("template") == spec["template"]),
                             key=lambda e: (e.get("path_index") != spec["path"], -_n(e.get("path_progress")), e["id"]))
            names = [f"up:{spec['path']}"] + [f"block:{e['id']}" for e in targets] \
                + ([f"exit:{spec['exit']}"] if spec.get("exit") is not None else [])
        item = next((moves[name] for name in names if name in moves), None)
        if item is None:
            return None
        if hero_id in self.retreating and spec["to"] != "home":
            return None
        if _distance(hero.get("rally_x"), hero.get("rally_y"), *_point(item)) <= ARRIVED_PX:
            return None
        if self._inside(_point(item), self._auras(state)):
            return None
        return {"item": item, "kind": "hero", "reason": f"play {self.play['id']}: hero {spec['to']}"}

    # -- spells -----------------------------------------------------------------------------------------
    def _spells(self, state, menu):
        owned = self.owned_powers()
        for power in (1, 2):
            if self.cast_issued[power] or power not in owned:
                continue
            items = [item for item in menu if item["action"].get("action") in ("aim_power", "use_power")
                     and item["action"].get("power") == power]
            if not items:
                continue
            part = POWER_PARTS[power]
            spec = self.play.get(part) or STEADY
            out = None
            if spec["mode"] in ("reserve", "block"):
                targets = sorted((e for e in _awake(state) if e.get("template") == spec["template"]),
                                 key=lambda e: (e.get("path_index") != spec["path"], -_n(e.get("path_progress")),
                                                e["id"]))
                for enemy in targets:
                    item = self._aim(power, enemy, items, CLUSTER_PX)
                    if item is not None:
                        out = {"item": item, "kind": part, "reason": f"{spec['mode']} {spec['template']}",
                               "target": enemy["id"]}
                        break
                if out is None and spec["mode"] == "reserve" and part not in self.expired:
                    continue  # saved for its target
            out = out or self._steady(power, part, state, items)
            if out:
                return out
        return None

    @staticmethod
    def _aim(power, enemy, items, near):
        """The option for this enemy: its named point (Rain of Fire: threat, then 1.2 s lead; reinforcements:
        block), else a cluster point within ``near`` px of it, else the plain cast anchored on it."""
        named = {item["action"].get("option"): item for item in items if item["action"]["action"] == "aim_power"}
        for kind in (("threat", "lead") if power == 1 else ("block",)):
            if f"{kind}:{enemy['id']}" in named:
                return named[f"{kind}:{enemy['id']}"]
        clusters = sorted((item for name, item in named.items() if str(name).startswith("cluster:")
                           and _distance(*_point(item), enemy.get("x"), enemy.get("y")) <= near),
                          key=lambda i: i["action"]["option"])
        if clusters:
            return clusters[0]
        return next((item for item in items if item["action"]["action"] == "use_power"
                     and item["action"].get("anchor_id") == enemy["id"]), None)

    def _steady(self, power, part, state, items):
        """The executor's rule with the new aim: once an enemy has covered cast% of its path, the leading enemy
        (its 1.2 s lead / block point, else the first cluster, else the plain cast at the most advanced anchor);
        with the plan's boss gene, only at an awake boss while one is alive."""
        enemies = _awake(state)
        bosses = [e for e in enemies if _is_boss(e) and _n(e.get("hp")) > 0]
        if self.boss_focus and bosses:
            boss = max(bosses, key=lambda e: (_n(e.get("hp")), -e["id"]))
            item = self._aim(power, boss, items, BOSS_RADIUS)
            return {"item": item, "kind": part, "reason": "boss focus", "target": boss["id"]} if item else None
        if not any(_n(e.get("path_progress")) >= self.cast / 100 for e in enemies):
            return None
        progress = {e["id"]: _n(e.get("path_progress")) for e in enemies}
        kind = "lead" if power == 1 else "block"
        named = [item for item in items if str(item["action"].get("option", "")).startswith(kind + ":")
                 and item["action"].get("target_id") in progress]
        if named:
            item = min(named, key=lambda i: (-progress[i["action"]["target_id"]], i["action"]["target_id"]))
            return {"item": item, "kind": part, "reason": f"steady {kind}", "target": item["action"]["target_id"]}
        clusters = sorted((i for i in items if str(i["action"].get("option", "")).startswith("cluster:")),
                          key=lambda i: i["action"]["option"])
        if clusters:
            return {"item": clusters[0], "kind": part, "reason": "steady cluster"}
        plain = [i for i in items if i["action"]["action"] == "use_power"]
        if plain:
            item = min(plain, key=lambda i: (-progress.get(i["action"].get("anchor_id"), -math.inf),
                                             str(i["action"].get("anchor_id"))))
            return {"item": item, "kind": part, "reason": "steady anchor", "target": item["action"].get("anchor_id")}
        return None


# -- brains ---------------------------------------------------------------------------------------------------
class PassiveBrain:
    """Never acts: the wrapped executor plays exactly as it does without a commander."""
    name = "passive"
    passive = True

    def choose(self, briefing, plays, key):
        return 0, {"brain": self.name}


class RuleCommander:
    """A deliberately strong rule baseline from the mechanics notes (thresholds fixed before any evaluation):
    a boss in the next wave first, then block egg-laying spiders with the hero and the reinforcements, save Rain
    of Fire for the main named threat (spells only, hero home, against an enemy that cannot be blocked or has a
    deadly aura), guard the exit most of the next wave heads for, else steady."""
    name = "rule"
    GUARD_SHARE = 0.5

    def choose(self, briefing, plays, key):
        ids = {play["id"]: i for i, play in enumerate(plays)}
        first = plays[ids["P1"]] if "P1" in ids else None
        if first and "boss" in first["threat"]["traits"]:
            return self._named(ids, first)
        for pid in ("P1", "P2"):
            play = plays[ids[pid]] if pid in ids else None
            if play and play["hero"] and "egg" in play["threat"]["traits"]:
                return ids[pid], {"brain": self.name, "rule": "block_egg_layers"}
        if first and first["threat"]["traits"]:
            return self._named(ids, first)
        if "P3" in ids and plays[ids["P3"]]["exit_threat"] >= self.GUARD_SHARE:
            return ids["P3"], {"brain": self.name, "rule": "guard_threatened_exit"}
        return ids.get("P0", 0), {"brain": self.name, "rule": "steady"}

    def _named(self, ids, first):
        """Rain of Fire saved for a named threat; the hero intercepts it unless it cannot be blocked or has a
        deadly aura (or there is no hero), then both spells focus on it with the hero at its start point."""
        if ({"unblockable", "aura"} & set(first["threat"]["traits"]) or not first["hero"]) and "P4" in ids:
            return ids["P4"], {"brain": self.name, "rule": "focus_spells"}
        return ids["P1"], {"brain": self.name, "rule": "save_fire_for_threat"}


class SteadyBrain:
    """Always the steady play P0 without being passive: plays exactly as the executor alone (the fork baseline)."""
    name = "steady"

    def choose(self, briefing, plays, key):
        return 0, {"brain": self.name}


class RuleSpellsCommander(RuleCommander):
    """The rule brain controlling only the spells; the hero stays where the executor leaves it (component ablation)."""
    name = "rule_spells"
    parts = ("spells",)


class RuleHeroCommander(RuleCommander):
    """The rule brain controlling only the hero; the executor casts the spells as without a commander (component
    ablation)."""
    name = "rule_hero"
    parts = ("hero",)


class PlaceboCommander(RuleCommander):
    """The rule brain with every command held back one native tick: its outcome discordance with the rule arm
    measures pure chaos."""
    name = "placebo"
    command_delay_ticks = 1


def consult_seed(key):
    text = f"{key.get('seed')}:{key.get('genome_id')}:{key.get('wave')}:{key.get('consult')}"
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest(), 16)


class RandomCommander:
    """Uniform over the plays, seeded by sha256(task seed, genome id, wave, consult index)."""
    name = "random"

    def choose(self, briefing, plays, key):
        return random.Random(consult_seed(key)).randrange(len(plays)), {"brain": self.name}


def consult_user(briefing):
    return briefing.text() + "\nChoose the play for the coming wave."


class LanguageCommander:
    """The 8B: every play label A-F scored by the local model worker's /distribution in every cyclic order,
    exactly like strategy_brain.LanguageBrain, so it can only pick an offered play."""

    def __init__(self, broker, name="8b"):
        self.scorer = LanguageBrain(broker, name=name)
        self.name = name

    def choose(self, briefing, plays, key):
        if len(plays) == 1:
            return 0, {"brain": self.name, "trivial": True}
        index, record = self.scorer._ask("commander", consult_user(briefing), [p["text"] for p in plays], SYSTEM)
        return index, {"brain": self.name, **{k: v for k, v in record.items() if k != "options"}}


class ForcedChoice:
    """A fork: the base brain, except that consult ``consult`` takes play ``play`` (if offered)."""

    def __init__(self, base, consult, play):
        self.base, self.consult, self.play = base, consult, play
        self.name = f"fork-{base.name}-c{consult}-p{play}"
        self.command_delay_ticks = getattr(base, "command_delay_ticks", 0)

    def choose(self, briefing, plays, key):
        if key.get("consult") == self.consult:
            if self.play < len(plays):
                return self.play, {"brain": self.name, "forced": True}
            index, record = self.base.choose(briefing, plays, key)
            return index, {**record, "brain": self.name, "forced": False, "fork_unavailable": len(plays)}
        return self.base.choose(briefing, plays, key)


RULE_BRAINS = {"passive": PassiveBrain, "steady": SteadyBrain, "rule": RuleCommander, "rule_spells": RuleSpellsCommander,
               "rule_hero": RuleHeroCommander, "random": RandomCommander,
               "placebo": PlaceboCommander}
BRAINS = tuple(RULE_BRAINS) + ("8b",)


def make_brain(name, broker=None):
    if name == "8b":
        if broker is None:
            raise ValueError("the 8b commander needs the strategy worker's broker")
        return LanguageCommander(broker)
    if name not in RULE_BRAINS:
        raise ValueError(f"unknown commander brain {name!r}")
    return RULE_BRAINS[name]()


def request_user(user, options):
    """The user text of the 8B's first (unrotated) request, as strategy_brain.LanguageBrain builds it."""
    return (user + "\n\nOptions:\n" + "\n".join(f"{chr(65 + i)}. {text}" for i, text in enumerate(options))
            + "\n\nAnswer with one label.")


# -- elite-v3b capture: what the screen shows at a consult, for offline prompts -----------------------------------
CAPTURE_FIELDS = {
    "towers": ("id", "template", "type", "level", "holder_id", "x", "y", "range", "soldier_count", "max_soldiers",
               "rally_range", "rally_x", "rally_y"),
    "enemies": ("id", "template", "hp", "hp_max", "x", "y", "speed", "lives_cost", "armor", "magic_armor",
                "path_index", "path_progress", "flying", "unblockable", "boss", "dormant", "untargetable"),
    "heroes": ("id", "template", "level", "xp", "hp", "hp_max", "dead", "x", "y", "rally_x", "rally_y", "home_x",
               "home_y", "respawn_remaining", "stunned"),
}
LOSS_PROGRESS = 0.9  # an enemy this far along when it disappears together with a life drop is the one that got out


def capture_state(state):
    """The screen at a consult (plain JSON): towers, enemies and heroes with the listed fields, the flag preview,
    the spell buttons, lives, gold, wave, tick and level. Read only; nothing here feeds back into the game."""
    out = {key: state.get(key) for key in ("tick", "wave", "wave_total", "lives", "gold", "level_idx")}
    for key, fields in CAPTURE_FIELDS.items():
        out[key] = [{f: item.get(f) for f in fields if f in item} for item in state.get(key) or []
                    if isinstance(item, dict)]
    out["next_wave"] = copy.deepcopy(state.get("next_wave")) if isinstance(state.get("next_wave"), dict) else None
    out["powers_ui"] = copy.deepcopy(state.get("powers_ui")) if isinstance(state.get("powers_ui"), list) else None
    return out


# -- the wrapper ----------------------------------------------------------------------------------------------
def executor_genome(inner):
    """The plan an executor follows: the plan executor's own, or the operator network's plan tracker's."""
    genome = getattr(inner, "genome", None)
    if genome is None:
        genome = getattr(getattr(inner, "tracker", None), "genome", None)
    return genome


class CommanderPolicy:
    """A commander over an executor policy (the plan executor, or the operator network). Its spell rule (cast
    threshold, boss focus) and consult seeds follow the plan: ``genome`` if given (it must be the executor's own
    plan when the executor has one), else the executor's plan."""

    def __init__(self, inner, brain, *, seed=0, clock=time.monotonic, genome=None, capture=False):
        self.inner, self.brain, self.seed, self.clock = inner, brain, seed, clock
        self.name = f"commander:{brain.name}:{inner.name}"
        own = executor_genome(inner)
        if genome is not None and own is not None and genome_id(genome) != genome_id(own):
            raise ValueError("the commander's plan must be the plan its executor follows")
        self.genome = check_genome(genome) if genome is not None else own
        self.genome_id = genome_id(self.genome) if self.genome is not None else "none"
        cast = self.genome["cast"] if self.genome is not None else 50
        self.parts = tuple(getattr(brain, "parts", ("spells", "hero")))
        self.translator = Translator(cast=cast, boss_focus=bool((self.genome or {}).get("boss")), parts=self.parts)
        self.generator = PlayGenerator(cast=cast)
        self.passive = bool(getattr(brain, "passive", False))
        self.delay = getattr(brain, "command_delay_ticks", 0)
        self.shadows = (RuleCommander(), RandomCommander())
        self.consults, self.pending = [], []
        self.commands, self.holds = {}, 0
        self.play_id = "none"
        self._groups = set()
        self._alarm = self._alarm_wave = self._lives_at_consult = None
        self._alive, self._bosses = {}, set()
        self._observed = self._requested = self._last_request = self._held_since = None
        self.exits, self.home, self.path_ends = {}, None, {}
        # elite-v3b: record the screen at each consult and a memory ledger (lives lost per path, hero deaths).
        self.capture = bool(capture)
        self.ledger = {"lives_lost": {}, "hero_deaths": 0}
        self._last_seen = None

    # -- observation -------------------------------------------------------------------------------------
    def _observe(self, state, menu):
        tick = state.get("tick")
        if tick == self._observed:
            return
        self._observed = tick
        self.translator.observe(state)
        for e in _awake(state):
            if _int(e.get("path_index")) and _n(e.get("path_progress")) >= EXIT_PROGRESS:
                self.path_ends[str(e["path_index"])] = (_n(e.get("x")), _n(e.get("y")))
            if _is_boss(e) and e["id"] not in self._bosses:
                self._bosses.add(e["id"])
                self._alarm = self._alarm or "boss_entry"
        for hero in _heroes(state):
            alive = not hero.get("dead")
            if self._alive.get(hero["id"]) and not alive:
                self._alarm = self._alarm or "hero_death"
            self._alive[hero["id"]] = alive
            if _finite(hero.get("home_x")) and _finite(hero.get("home_y")):
                self.home = (hero["home_x"], hero["home_y"])
        for item in menu:
            action = item["action"]
            if action.get("action") == "move_hero":
                kind, _, number = str(action.get("option")).partition(":")
                if kind == "exit":
                    self.exits[number] = (action["x"], action["y"])
                elif kind == "home" and self.home is None:
                    self.home = (action["x"], action["y"])
        lives = state.get("lives")
        if self._lives_at_consult is not None and _int(lives) and self._lives_at_consult - lives >= ALARM_LIVES:
            self._alarm = self._alarm or "lives_lost"
        if self.capture:
            self._remember(state)

    def _remember(self, state):
        """The memory ledger: each life drop is put on the paths of the enemies that were at least LOSS_PROGRESS
        along their path in the previous observation and are gone now (most advanced first, lives_cost each),
        the rest on "unknown"; hero deaths are counted. Uses only past and present observations."""
        lives = state.get("lives")
        now = {e["id"] for e in _enemies(state)}
        last = self._last_seen
        if last is not None and _int(lives) and _int(last["lives"]) and lives < last["lives"]:
            lost = last["lives"] - lives
            gone = sorted((e for e in last["enemies"] if e["id"] not in now and e["progress"] >= LOSS_PROGRESS),
                          key=lambda e: (-e["progress"], e["id"]))
            book = self.ledger["lives_lost"]
            for e in gone:
                if lost <= 0:
                    break
                take = min(lost, max(1, int(_n(e["cost"], 1))))
                book[e["path"]] = book.get(e["path"], 0) + take
                lost -= take
            if lost > 0:
                book["unknown"] = book.get("unknown", 0) + lost
        for hero in _heroes(state):
            if hero.get("dead") and last is not None and last["alive"].get(hero["id"]):
                self.ledger["hero_deaths"] += 1
        self._last_seen = {
            "lives": lives,
            "enemies": [{"id": e["id"], "progress": _n(e.get("path_progress")), "cost": e.get("lives_cost"),
                         "path": str(e["path_index"]) if _int(e.get("path_index")) else "unknown"}
                        for e in _enemies(state)],
            "alive": {h["id"]: not h.get("dead") for h in _heroes(state)}}

    def geometry(self):
        """Exit points, the exit nearest the hero's start point and the exit each seen path leaves by."""
        def nearest(point):
            return min(self.exits, key=lambda k: (_distance(*point, *self.exits[k]), int(k))) if self.exits else None
        return {"exits": dict(self.exits), "home_exit": nearest(self.home) if self.home else None,
                "path_exit": {path: nearest(end) for path, end in self.path_ends.items()} if self.exits else {}}

    # -- consults ----------------------------------------------------------------------------------------
    def _consult_reason(self, state):
        preview = state.get("next_wave")
        if isinstance(preview, dict):
            group = preview.get("group_idx") if _int(preview.get("group_idx")) else int(_n(state.get("wave"), 0)) + 1
            if group not in self._groups:
                return "wave", group
        if self._alarm and self.consults:
            if self._alarm_wave == state.get("wave"):
                self._alarm = None  # at most one alarm per wave
            else:
                return "alarm:" + self._alarm, None
        return None, None

    def _maybe_consult(self, state):
        reason, group = self._consult_reason(state)
        if reason is None:
            return
        if group is not None:
            self._groups.add(group)
        if reason.startswith("alarm"):
            self._alarm_wave = state.get("wave")
        self._alarm = None
        self._lives_at_consult = state.get("lives") if _int(state.get("lives")) else None
        briefing = Briefing.from_state(state)
        plays = self.generator.plays(briefing, self.geometry())
        key = {"seed": self.seed, "genome_id": self.genome_id, "wave": state.get("wave"), "consult": len(self.consults)}
        started = self.clock()
        index, record = self.brain.choose(briefing, plays, dict(key))
        latency = self.clock() - started
        if not (_int(index) and 0 <= index < len(plays)):
            raise ValueError(f"commander brain {self.brain.name} chose play {index!r} of {len(plays)}")
        shadow = {brain.name: brain.choose(briefing, plays, dict(key))[0] for brain in self.shadows}
        play = plays[index]
        self.play_id = play["id"]
        if not self.passive:
            self.translator.adopt(play, state)
        # The 8B's first request for this consult, so offline scoring can prove it saw the same prompt.
        prompt = request_user(consult_user(briefing), [p["text"] for p in plays])
        entry = {"consult": key["consult"], "reason": reason, "tick": state.get("tick"), "wave": state.get("wave"),
                 "group": group, "brain": self.brain.name, "choice": index, "play_id": play["id"],
                 "p": record.get("p"), "latency_seconds": latency,
                 "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
                 "shadow": shadow, "briefing": briefing.data, "plays": plays, "record": record}
        if self.capture:
            entry["capture"] = capture_state(state)
            entry["ledger"] = copy.deepcopy(self.ledger)
            entry["translator"] = {"cast": self.translator.cast, "boss_focus": self.translator.boss_focus}
        self.consults.append(entry)
        self.pending.append(entry)

    # -- the episode protocol ----------------------------------------------------------------------------
    def wants_decision(self, state):
        """True when the translator has a command now: it then comes before any decision the executor has due
        (which stays due). Commands are at least TRIGGER_GAP ticks apart and never ahead of a click the game
        waits for; the placebo holds each one back ``command_delay_ticks`` first (returned as the episode's explicit
        ``{"hold": n}``). Consults happen here too, so a flag is answered at its first step. Never for a passive
        brain. Returns only True, False or a hold."""
        if self.passive or state.get("level_won") or state.get("level_lost") or state.get("game_over"):
            return False
        tick = state.get("tick")
        menu = build_menu(state)
        self._observe(state, menu)
        self._maybe_consult(state)
        if self.translator.command(state, menu) is None:
            self._held_since = None
            return False
        if any(item["action"].get("action") == "click_entity" for item in menu):
            return False  # the executor taps first (a boss's finishing tap, a frozen tower)
        if self._last_request is not None and tick - self._last_request < TRIGGER_GAP:
            return False
        if self.delay:
            if self._held_since is None:
                self._held_since = tick
                self.holds += 1
            if tick - self._held_since < self.delay:
                return {"hold": self.delay - (tick - self._held_since)}
        self._held_since = None
        self._requested = self._last_request = tick
        return True

    def choose(self, state, menu, context):
        validate_menu(menu)
        tick = state.get("tick")
        self._observe(state, menu)
        self._maybe_consult(state)
        if not self.passive and self._requested == tick:
            # A decision the commander asked for: the executor keeps its own decision points.
            command = self.translator.command(state, menu, commit=True)
            kind = command["kind"] if command else "wait"
            item = command["item"] if command else next(i for i in menu if i["action"].get("action") == "wait")
            self.commands[kind] = self.commands.get(kind, 0) + 1
            # click_offered: this menu offered a tap the game waits for (pace_violations flags any command then).
            meta = {"rule": "commander_" + kind, "play": self.play_id, "option": item["action"].get("option"),
                    "reason": (command or {}).get("reason"), "target": (command or {}).get("target"),
                    "click_offered": any(i["action"].get("action") == "click_entity" for i in menu)}
            return {"label": item["label"], "provenance": f"commander:{self.brain.name}:{self.play_id}:{kind}",
                    "distribution": None, "meta": self._with_consults(meta)}
        owned = set() if self.passive else self.translator.owned_powers()
        shown = [item for item in menu if item["action"].get("action") not in ("aim_power", "move_hero")
                 and not (item["action"].get("action") == "use_power" and item["action"].get("power") in owned)]
        choice = self.inner.choose(state, shown, context)
        if isinstance(choice, dict) and self.pending:
            choice = {**choice, "meta": self._with_consults(dict(choice.get("meta") or {}))}
        return choice

    def _with_consults(self, meta):
        if self.pending:
            meta["commander_consults"] = copy.deepcopy(self.pending)
            self.pending = []
        return meta

    def summary(self):
        """Per-episode record for the evaluation journal (whose rows drop the decisions)."""
        return {"brain": self.brain.name, "consults": copy.deepcopy(self.consults), "commands": dict(self.commands),
                "held_commands": self.holds}


PLAYER_BRAINS = {"cmd_passive": "passive", "cmd_steady": "steady", "cmd_rule": "rule", "cmd_rule_spells": "rule_spells",
                 "cmd_rule_hero": "rule_hero", "cmd_random": "random", "cmd_placebo": "placebo",
                 "cmd_8b": "8b", "cmd_steady_capture": "steady"}
CAPTURE_PLAYERS = {"cmd_steady_capture"}  # elite-v3b: the steady brain that also records the screen at each consult


def make_player(player, genome, task, broker=None):
    """The commander arm ``player`` (eval_job) over the plan executor for one task; "cmd_fork" reads the task's
    ``fork`` {"base", "consult", "play"}."""
    if player == "cmd_fork":
        fork = task["fork"]
        brain = ForcedChoice(make_brain(fork["base"], broker), fork["consult"], fork["play"])
    else:
        brain = make_brain(PLAYER_BRAINS[player], broker)
    return CommanderPolicy(BuildOrderPolicy(genome), brain, seed=task["seed"], capture=player in CAPTURE_PLAYERS)


def _commanded(decision):
    """(play id, kind) of a decision the commander asked for itself, else None."""
    provenance = str(decision.get("provenance") or "")
    if not provenance.startswith("commander:") or provenance.count(":") < 3:
        return None
    _, play, kind = provenance.rsplit(":", 2)
    return play, kind


def command_log(decisions):
    """The commander's own decisions of one episode, compactly, for the evaluation journal (whose rows drop the
    decisions): tick, kind (fire/reinf/hero/retreat/aura/wait), action, option and the play in force."""
    out = []
    for decision in decisions:
        commanded = _commanded(decision)
        if commanded:
            action = decision.get("action") or {}
            out.append({"tick": decision.get("tick"), "kind": commanded[1], "action": action.get("action"),
                        "option": action.get("option"), "play": commanded[0]})
    return out


def pace_violations(decisions):
    """Post-hoc pace check of one episode's decisions (plan section 4): commander hero orders at least
    HERO_ORDER_GAP ticks apart, retreat reflexes at least RETREAT_GAP apart, the commander's own decisions at
    least TRIGGER_GAP apart and never while the menu offered a click_entity tap, at most one alarm consult per
    wave, one consult per next-wave group. Returns readable violations (empty = clean)."""
    out = []
    hero_ticks, retreat_ticks, command_ticks, alarms, groups = [], [], [], {}, set()
    for decision in decisions:
        commanded = _commanded(decision)
        kind = commanded[1] if commanded else None
        if commanded:
            command_ticks.append(decision["tick"])
            if (decision.get("meta") or {}).get("click_offered") is not False:
                out.append(f"decision {decision.get('index')}: a commander command while a click_entity was "
                           "offered (or not recorded)")
        if (decision.get("action") or {}).get("action") == "move_hero":
            if kind is None:
                out.append(f"decision {decision.get('index')}: a hero move not issued by the commander")
            hero_ticks.append(decision["tick"])
            if kind in ("retreat", "aura"):
                retreat_ticks.append(decision["tick"])
        for consult in (decision.get("meta") or {}).get("commander_consults") or []:
            if str(consult.get("reason")).startswith("alarm"):
                alarms[consult.get("wave")] = alarms.get(consult.get("wave"), 0) + 1
            elif consult.get("group") in groups:
                out.append(f"consult {consult.get('consult')}: next-wave group {consult.get('group')} asked twice")
            else:
                groups.add(consult.get("group"))
    for name, ticks, gap in (("hero orders", hero_ticks, HERO_ORDER_GAP), ("retreats", retreat_ticks, RETREAT_GAP),
                             ("commander decisions", command_ticks, TRIGGER_GAP)):
        for a, b in zip(ticks, ticks[1:]):
            if b - a < gap:
                out.append(f"{name} at ticks {a} and {b} are less than {gap} ticks apart")
    out.extend(f"wave {wave}: {count} alarm consults" for wave, count in sorted(alarms.items(), key=str) if count > 1)
    return out

