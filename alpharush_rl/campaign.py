"""Campaign progression the way a person plays it from a new save.

Stars come from the game's own victory rule, star upgrades are bought by one
fixed, human-readable rule, and a level is started from a save slot that holds
exactly that progress (a direct ``-level`` launch reads upgrades and the
selected hero from the active slot when the level loads).
"""

import os
import re
from pathlib import Path

# Star-upgrade trees as the slot names them, and each level's star price (kr1/upgrades.lua).
STAR_TREES = ("archers", "barracks", "mages", "engineers", "rain", "reinforcements")
STAR_PRICES = {"archers": (1, 1, 2, 2, 3), "barracks": (1, 1, 2, 2, 3), "mages": (1, 1, 2, 2, 3),
               "engineers": (1, 1, 2, 3, 3), "rain": (2, 2, 3, 3, 3), "reinforcements": (2, 3, 3, 3, 4)}
# Fixed purchase rule: always the lowest next level across trees, ties in this order.
BUY_PRIORITY = ("rain", "archers", "mages", "barracks", "engineers", "reinforcements")
# Every hero the slot template carries (a new save lists each with no xp and no skills).
HERO_NAMES = ("hero_10yr", "hero_alleria", "hero_bolin", "hero_denas", "hero_elora", "hero_gerald",
              "hero_hacksaw", "hero_ignus", "hero_ingvar", "hero_magnus", "hero_malik", "hero_oni", "hero_thor")
# Heroes a new desktop save can select without purchases, and the level from which the hero room
# offers each (map_data hero_data.available_level: a level N campaign has N levels unlocked).
FREE_HEROES = {"hero_gerald": 4, "hero_alleria": 6, "hero_malik": 8, "hero_bolin": 8, "hero_magnus": 9,
               "hero_ignus": 11}
VERSION_STRING = "kr1-desktop-6.4.46"
CAMPAIGN_MODE, NORMAL = 1, 2
MAX_STARS_PER_LEVEL = 3


def stars_for_lives(lives: int) -> int:
    """Campaign victory stars (systems.lua): 3 from 18 lives left, 2 from 6, else 1."""
    if isinstance(lives, bool) or not isinstance(lives, int) or lives < 1:
        raise ValueError("a won level has at least one life left")
    return 3 if lives >= 18 else 2 if lives >= 6 else 1


def upgrades_cost(upgrades: dict) -> int:
    return sum(sum(STAR_PRICES[tree][:upgrades.get(tree, 0)]) for tree in STAR_TREES)


def check_upgrades(upgrades: dict) -> dict:
    if not isinstance(upgrades, dict) or set(upgrades) - set(STAR_TREES):
        raise ValueError(f"star upgrades must be a dict over {STAR_TREES}")
    out = {}
    for tree in STAR_TREES:
        level = upgrades.get(tree, 0)
        if isinstance(level, bool) or not isinstance(level, int) or not 0 <= level <= len(STAR_PRICES[tree]):
            raise ValueError(f"star upgrade {tree} must be 0..{len(STAR_PRICES[tree])}")
        out[tree] = level
    return out


def buy_upgrades(stars: int, owned: dict | None = None) -> dict:
    """Spend ``stars`` (total earned) by the fixed rule on top of ``owned``; never sells."""
    if isinstance(stars, bool) or not isinstance(stars, int) or stars < 0:
        raise ValueError("stars must be a nonnegative integer")
    levels = check_upgrades(owned or {})
    left = stars - upgrades_cost(levels)
    if left < 0:
        raise ValueError("owned upgrades cost more stars than were earned")
    while True:
        offers = [(levels[tree], BUY_PRIORITY.index(tree), tree) for tree in BUY_PRIORITY
                  if levels[tree] < len(STAR_PRICES[tree]) and STAR_PRICES[tree][levels[tree]] <= left]
        if not offers:
            return levels
        _, _, tree = min(offers)
        left -= STAR_PRICES[tree][levels[tree]]
        levels[tree] += 1


def check_profile(profile: dict) -> dict:
    """Validated copy of a save profile: ``upgrades``, optional ``hero``, optional ``levels``
    ({level index: stars} for won campaign levels)."""
    if not isinstance(profile, dict) or set(profile) - {"upgrades", "hero", "hero_xp", "levels"}:
        raise ValueError("profile keys are upgrades, hero, hero_xp and levels")
    upgrades = check_upgrades(profile.get("upgrades", {}))
    hero = profile.get("hero")
    if hero is not None and hero not in HERO_NAMES:
        raise ValueError(f"unknown hero {hero!r}")
    hero_xp = profile.get("hero_xp", 0)
    if isinstance(hero_xp, bool) or not isinstance(hero_xp, int) or hero_xp < 0 or (hero_xp and hero is None):
        raise ValueError("hero_xp is a nonnegative integer for the selected hero")
    levels = {}
    for key, stars in (profile.get("levels") or {}).items():
        index = int(key)
        if str(index) != str(key) or index < 1 or isinstance(stars, bool) or stars not in (1, 2, 3):
            raise ValueError(f"invalid won level {key!r}: {stars!r}")
        levels[index] = stars
    if upgrades_cost(upgrades) > sum(levels.values()) and levels:
        raise ValueError("upgrades cost more stars than the won levels give")
    return {"upgrades": upgrades, "hero": hero, "hero_xp": hero_xp, "levels": dict(sorted(levels.items()))}


def _lua(value, indent=0) -> str:
    pad = "\t" * (indent + 1)
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, str):
        if not re.fullmatch(r"[A-Za-z0-9_.\-]*", value):
            raise ValueError(f"unsafe slot string {value!r}")
        return f'"{value}"'
    if isinstance(value, dict):
        lines = ["{"]
        for key in sorted(value, key=lambda k: (isinstance(k, str), k)):
            name = f"[{key}]" if isinstance(key, int) else f'["{key}"]'
            if isinstance(key, str) and not re.fullmatch(r"[A-Za-z0-9_]+", key):
                raise ValueError(f"unsafe slot key {key!r}")
            lines.append(f"{pad}{name} = {_lua(value[key], indent + 1)};")
        lines.append("\t" * indent + "}")
        return "\n".join(lines)
    raise ValueError(f"unsupported slot value {value!r}")


def slot_lua(profile: dict) -> str:
    """The slot file a new save would hold after this progress, in the game's own literal format."""
    profile = check_profile(profile)
    heroes = {"status": {name: {"skills": {}, "xp": 0} for name in HERO_NAMES}}
    if profile["hero"]:
        heroes["selected"] = profile["hero"]
        heroes["status"][profile["hero"]]["xp"] = profile["hero_xp"]
    levels = {index: {CAMPAIGN_MODE: NORMAL, "stars": stars} for index, stars in profile["levels"].items()}
    if profile["levels"]:
        # The map unlocks the next level after a campaign victory.
        following = max(profile["levels"]) + 1
        levels.setdefault(following, {})
    slot = {"achievements": {}, "bag": {}, "gems": 0, "heroes": heroes, "levels": levels, "seen": {},
            "upgrades": profile["upgrades"], "version_string": VERSION_STRING}
    return f"local obj1 = {_lua(slot)}\nreturn obj1\n"


def save_directory(identity: str) -> Path:
    appdata = os.environ.get("APPDATA")
    if not appdata:
        raise RuntimeError("APPDATA is not set; the game's save directory is unknown")
    return Path(appdata) / identity


def write_profile(identity: str, profile: dict) -> Path:
    """Write slot 1 into a new identity's save directory before the game starts.

    Refuses an identity whose save directory already holds a slot, so a profile
    never silently replaces a game's own progress.
    """
    directory = save_directory(identity)
    path = directory / "slot_1.lua"
    if path.exists():
        raise RuntimeError(f"save slot already exists for identity {identity!r}")
    directory.mkdir(parents=True, exist_ok=True)
    text = slot_lua(profile)
    path.write_text(text, encoding="utf-8", newline="\n")
    return path


def profile_matches(profile: dict, meta: dict) -> list[str]:
    """Problems between a profile and what the loaded level reports (host ``meta``)."""
    profile = check_profile(profile)
    problems = []
    loaded = meta.get("star_upgrades")
    if not isinstance(loaded, dict) or {t: loaded.get(t) for t in STAR_TREES} != profile["upgrades"]:
        problems.append(f"star upgrades loaded {loaded!r} != profile {profile['upgrades']!r}")
    if (meta.get("selected_hero") or None) != profile["hero"]:
        problems.append(f"selected hero loaded {meta.get('selected_hero')!r} != profile {profile['hero']!r}")
    return problems


# Allocation packages (the upgrades screen's reset refunds every star, so any package can be applied
# before any level): "balanced" is the fixed incremental rule; the others spend on their trees first.
PACKAGES = {"balanced": (), "rain": ("rain",), "spells": ("rain", "reinforcements"), "archers": ("archers",),
            "barracks": ("barracks",), "mages": ("mages",), "engineers": ("engineers",)}


def package_upgrades(package: str, won: dict) -> dict:
    """The star upgrades of ``package`` after the victories ``won`` ({level: stars})."""
    if package not in PACKAGES:
        raise ValueError(f"unknown upgrade package {package!r}")
    if package == "balanced":
        upgrades, stars = {}, 0
        for index in sorted(won, key=int):
            stars += won[index]
            upgrades = buy_upgrades(stars, upgrades)
        return check_upgrades(upgrades)
    stars = sum(won.values())
    levels = check_upgrades({})
    while True:
        left = stars - upgrades_cost(levels)
        offers = [(levels[t], t) for t in PACKAGES[package] if levels[t] < 5 and STAR_PRICES[t][levels[t]] <= left]
        if not offers:
            break
        levels[min(offers)[1]] += 1
    return buy_upgrades(stars, levels)


def heroes_available(level: int) -> list[str]:
    """Free heroes the hero room offers when ``level`` is the next campaign level."""
    return sorted((name for name, first in FREE_HEROES.items() if level >= first), key=lambda n: (FREE_HEROES[n], n))


def campaign_profile(won: dict, hero: str | None = None, hero_xp: int = 0) -> dict:
    """Profile after winning ``won`` ({level: stars}) in level order, buying upgrades by the
    fixed rule after every victory on top of what was already bought (nothing is ever refunded)."""
    upgrades, stars = {}, 0
    for index in sorted(won, key=int):
        stars += won[index]
        upgrades = buy_upgrades(stars, upgrades)
    return check_profile({"upgrades": upgrades, "hero": hero, "hero_xp": hero_xp, "levels": won})


def level_profile(level: int, stars_per_level: int = MAX_STARS_PER_LEVEL, hero: str | None = None,
                  package: str = "balanced") -> dict:
    """Progress a player has when starting ``level`` after winning every earlier level with
    ``stars_per_level`` stars; ``hero`` must be one the hero room offers at that level."""
    if isinstance(level, bool) or not isinstance(level, int) or level < 1:
        raise ValueError("level must be a positive integer")
    if isinstance(stars_per_level, bool) or not isinstance(stars_per_level, (int, float)) \
            or not 1 <= stars_per_level <= 3:
        raise ValueError("stars_per_level must be a number from 1 to 3")
    if hero is not None and hero not in heroes_available(level):
        raise ValueError(f"{hero!r} is not available at level {level}")
    # An average rate: the earliest levels get the higher star count, totalling round(rate * levels).
    earlier = level - 1
    total = int(round(stars_per_level * earlier))
    low = int(stars_per_level)
    won = {index: low for index in range(1, level)}
    for index in range(1, level):
        if sum(won.values()) >= total:
            break
        won[index] = min(3, low + 1)
    return check_profile({**campaign_profile(won, hero=hero), "upgrades": package_upgrades(package, won)})
