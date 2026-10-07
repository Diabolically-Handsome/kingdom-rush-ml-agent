"""Deterministic scripted baselines for the episode Policy protocol.

Each policy reads only ``state`` and the legal ``menu`` (RandomLegal also uses
``context["decision_index"]``) and returns a label from that menu. They are
baselines and survey drivers, never a fallback for a failed model decision.
"""
from __future__ import annotations

import math
import random

from .menus import validate_menu

TOWER_TYPES = ("archer", "barrack", "mage", "engineer")
DEFAULT_RATIO = {"archer": 3, "barrack": 1, "mage": 1, "engineer": 1}
# TeacherV2: the four-level branch taken when a level-3 tower may specialize;
# the other branch is taken only when this one is not on the menu.
BRANCH_PREFERENCE = {"archer": "tower_ranger", "barrack": "tower_paladin", "mage": "tower_arcane_wizard",
                     "engineer": "tower_bfg"}
# Four-level templates by base kind, so specialized towers still count toward the build ratio.
FOUR_LEVEL_KINDS = {"tower_ranger": "archer", "tower_musketeer": "archer", "tower_paladin": "barrack",
                    "tower_barbarian": "barrack", "tower_arcane_wizard": "mage", "tower_sorcerer": "mage",
                    "tower_bfg": "engineer", "tower_tesla": "engineer"}
# TeacherV2 casts a spell only once some enemy has covered this share of its path.
CAST_PROGRESS = 0.5


def _choice(policy, label: str, **meta) -> dict:
    return {"label": label, "provenance": f"scripted:{policy.name}", "distribution": None, "meta": meta}


def _wait_label(menu: list[dict]) -> str:
    for item in menu:
        if item["action"].get("action") == "wait":
            return item["label"]
    raise ValueError("legal menu has no wait option")


def _send_wave_label(menu: list[dict]) -> str | None:
    for item in menu:
        if item["action"].get("action") == "send_wave":
            return item["label"]
    return None


def _number(value) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        return -math.inf  # unknown score ranks last, but the option stays selectable
    return float(value)


def _id_key(value) -> tuple:
    # Native holder IDs are integers; anything else sorts after them, by text.
    if isinstance(value, int) and not isinstance(value, bool):
        return (0, value, "")
    return (1, 0, str(value))


def _builds(menu: list[dict], kind: str) -> list[dict]:
    return [item for item in menu
            if item["action"].get("action") == "build_tower" and item["action"].get("tower_type") == kind]


def _best_holder(state: dict, options: list[dict]) -> dict:
    """Highest holder path_score; ties go to the smallest holder_id."""
    holders = {holder.get("id"): holder for holder in state.get("holders", []) if isinstance(holder, dict)}

    def key(item):
        holder_id = item["action"].get("holder_id")
        return (-_number(holders.get(holder_id, {}).get("path_score")), _id_key(holder_id))

    return min(options, key=key)


def tower_counts(state: dict, kinds=TOWER_TYPES) -> dict[str, int]:
    counts = {kind: 0 for kind in kinds}
    for tower in state.get("towers", []):
        template = tower.get("template") if isinstance(tower, dict) else None
        if not isinstance(template, str):
            continue
        for kind in kinds:
            # A tower still under construction counts toward its kind as well.
            if template.startswith(f"tower_{kind}_") or template == f"tower_build_{kind}":
                counts[kind] += 1
    return counts


def _least_built_kind(menu: list[dict], counts: dict, ratio: dict) -> str | None:
    """Buildable kind least built relative to its target share; ties prefer the
    larger share, then the ratio's declared order. None when nothing is buildable."""
    order = list(ratio)
    buildable = [kind for kind in order if _builds(menu, kind)]
    if not buildable:
        return None
    return min(buildable, key=lambda k: (counts[k] / ratio[k], -ratio[k], order.index(k)))


class SendWaveOnly:
    name = "send_wave_only"

    def choose(self, state: dict, menu: list[dict], context: dict) -> dict:
        validate_menu(menu)
        label = _send_wave_label(menu)
        if label is not None:
            return _choice(self, label, rule="send_wave")
        return _choice(self, _wait_label(menu), rule="wait")


class WaitOnly:
    """Never builds or calls a wave; probes whether waves start on their own."""
    name = "wait_only"

    def choose(self, state: dict, menu: list[dict], context: dict) -> dict:
        validate_menu(menu)
        return _choice(self, _wait_label(menu), rule="wait")


class RandomLegal:
    def __init__(self, seed: int):
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise ValueError("random policy seed must be an integer")
        self.seed = seed
        self.name = f"random:{seed}"

    def choose(self, state: dict, menu: list[dict], context: dict) -> dict:
        labels = validate_menu(menu)
        # String seeds are hashed by random itself, independent of PYTHONHASHSEED.
        rng = random.Random(f"{self.seed}:{context['decision_index']}")
        return _choice(self, rng.choice(labels), rule="uniform", n_options=len(labels))


class SingleTowerType:
    def __init__(self, kind: str):
        if kind not in TOWER_TYPES:
            raise ValueError(f"unknown tower type: {kind!r}")
        self.kind = kind
        self.name = f"single:{kind}"

    def choose(self, state: dict, menu: list[dict], context: dict) -> dict:
        validate_menu(menu)
        options = _builds(menu, self.kind)
        if options:
            item = _best_holder(state, options)
            return _choice(self, item["label"], rule="build", tower_type=self.kind,
                           holder_id=item["action"].get("holder_id"))
        label = _send_wave_label(menu)
        if label is not None:
            return _choice(self, label, rule="send_wave")
        return _choice(self, _wait_label(menu), rule="wait")


class PressureGreedy:
    def __init__(self, ratio: dict | None = None):
        ratio = dict(DEFAULT_RATIO if ratio is None else ratio)
        if not ratio:
            raise ValueError("ratio must name at least one tower type")
        for kind, weight in ratio.items():
            if kind not in TOWER_TYPES:
                raise ValueError(f"unknown tower type: {kind!r}")
            if isinstance(weight, bool) or not isinstance(weight, (int, float)) or not math.isfinite(weight) or weight <= 0:
                raise ValueError("ratio weights must be positive finite numbers")
        self.ratio = ratio
        # Ratio order breaks ties, so only the default in its own order keeps the plain name.
        default = list(ratio.items()) == list(DEFAULT_RATIO.items())
        self.name = "pressure_greedy" if default else (
            "pressure_greedy:" + ",".join(f"{kind}={weight:g}" for kind, weight in ratio.items()))

    def choose(self, state: dict, menu: list[dict], context: dict) -> dict:
        validate_menu(menu)
        counts = tower_counts(state, tuple(self.ratio))
        kind = _least_built_kind(menu, counts, self.ratio)
        if kind is not None:
            item = _best_holder(state, _builds(menu, kind))
            return _choice(self, item["label"], rule="build", tower_type=kind,
                           holder_id=item["action"].get("holder_id"), tower_counts=counts, ratio=dict(self.ratio))
        label = _send_wave_label(menu)
        if label is not None:
            return _choice(self, label, rule="send_wave")
        return _choice(self, _wait_label(menu), rule="wait")


def base_kind(template) -> str | None:
    """Base kind of a tower template: levels 1-3, under construction, or a four-level branch."""
    if not isinstance(template, str):
        return None
    if template in FOUR_LEVEL_KINDS:
        return FOUR_LEVEL_KINDS[template]
    for kind in TOWER_TYPES:
        if template.startswith(f"tower_{kind}_") or template == f"tower_build_{kind}":
            return kind
    return None


def _price(item: dict) -> float:
    value = item.get("cost")
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        return math.inf  # unknown price ranks last
    return float(value)


def aim_choice(policy, state: dict, menu: list[dict]) -> dict | None:
    """Fire a ready level tower (sunray) at the offered enemy with the most health."""
    aims = [item for item in menu if item["action"].get("action") == "point_tower"]
    if not aims:
        return None
    hp = {_id_key(e.get("id")): _number(e.get("hp")) for e in state.get("enemies", []) if isinstance(e, dict)}
    item = min(aims, key=lambda i: (-hp.get(_id_key(i["action"].get("anchor_id")), 0.0),
                                    _id_key(i["action"].get("anchor_id"))))
    return _choice(policy, item["label"], rule="point_tower", anchor_id=item["action"].get("anchor_id"),
                   anchor_hp=hp.get(_id_key(item["action"].get("anchor_id"))))


def _is_power(value, number: int) -> bool:
    return not isinstance(value, bool) and isinstance(value, (int, float)) and value == number


# teacher_v2 variants: "teacher_v2" (defaults) or "teacher_v2:" + comma-joined key=value pairs in
# TEACHER_KEYS order, listing only non-default keys, so a variant's name round-trips canonically.
#   b  four branch letters for archer, barrack, mage, engineer:
#      r(anger)|m(usketeer), p(aladin)|b(arbarian), a(rcane wizard)|s(orcerer), b(fg)|t(esla)
#   c  cast threshold, percent of path progress (0..100)
#   f  build-first factor, percent: build first while empty holders > built towers * f / 100
#   r  tower ratio digits for archer, barrack, mage, engineer (0..9 each, not all 0)
#   e  exploration percent (0..100) and s its seed: at each decision, with probability e/100,
#      a uniformly drawn non-wait, non-sell option replaces the rule's choice.
#   m  tower cap (0 = none): no new build once this many standard towers stand, so gold goes to upgrades;
#      a negative m is soft: past -m towers builds resume while SOFT_CAP_GOLD or more is banked
TEACHER_BRANCHES = {
    "archer": {"r": "tower_ranger", "m": "tower_musketeer"},
    "barrack": {"p": "tower_paladin", "b": "tower_barbarian"},
    "mage": {"a": "tower_arcane_wizard", "s": "tower_sorcerer"},
    "engineer": {"b": "tower_bfg", "t": "tower_tesla"},
}
TEACHER_KEYS = ("b", "c", "f", "r", "e", "s", "m")
TEACHER_DEFAULTS = {"b": "rpab", "c": 50, "f": 50, "r": "3111", "e": 0, "s": 0, "m": 0}
# A soft tower cap (m < 0) builds again from this much banked gold: no single upgrade costs that much.
SOFT_CAP_GOLD = 1000


def parse_teacher_params(text: str) -> dict:
    """Validated parameters of a "teacher_v2:..." spec suffix (empty text = defaults)."""
    params = dict(TEACHER_DEFAULTS)
    if text == "":
        return params
    seen = []
    for part in text.split(","):
        key, separator, value = part.partition("=")
        if not separator or key not in TEACHER_KEYS or key in seen:
            raise ValueError(f"invalid teacher_v2 parameter: {part!r}")
        seen.append(key)
        if key == "b":
            if len(value) != 4 or any(letter not in TEACHER_BRANCHES[kind]
                                      for letter, kind in zip(value, TEACHER_BRANCHES)):
                raise ValueError(f"invalid teacher_v2 branches: {value!r}")
            params[key] = value
        elif key == "r":
            if len(value) != 4 or not value.isdigit() or value == "0000":
                raise ValueError(f"invalid teacher_v2 ratio: {value!r}")
            params[key] = value
        else:
            digits = value[1:] if key == "m" and value.startswith("-") else value
            if not digits.isdigit() or str(int(value)) != value:
                raise ValueError(f"invalid teacher_v2 number: {part!r}")
            number = int(value)
            if key in ("c", "f", "e") and number > 100:
                raise ValueError(f"teacher_v2 {key} must be 0..100: {part!r}")
            params[key] = number
    if seen != [key for key in TEACHER_KEYS if key in seen]:
        raise ValueError("teacher_v2 parameters must follow the canonical key order")
    if any(params[key] == TEACHER_DEFAULTS[key] for key in seen):
        raise ValueError("teacher_v2 parameters must list only non-default values")
    return params


def teacher_name(params: dict) -> str:
    parts = [f"{key}={params[key]}" for key in TEACHER_KEYS if params[key] != TEACHER_DEFAULTS[key]]
    return "teacher_v2" + (":" + ",".join(parts) if parts else "")


class TeacherV2:
    """Deterministic action-scope v2 teacher; the first matching rule wins.

    a. ``use_power`` once some enemy has path_progress >= CAST_PROGRESS: rain of
       fire (power 1) before reinforcements (power 2), at the offered anchor
       enemy furthest along its path;
    b. ``upgrade_power``: the cheapest power of the highest path_score tower;
    c. ``upgrade_tower``: the highest path_score tower; at level 3 into its
       BRANCH_PREFERENCE branch, the other branch only when that one is absent;
    d. ``build_tower``: PressureGreedy's default ratio and holder choice, with
       four-level towers counted toward their base kind;
    e. ``send_wave`` while no enemy is on the field (before the first wave the
       native wave_ready alone decides);
    f. wait.

    When a build and an upgrade (b or c) are both offered, building comes first
    while the empty holders outnumber half the built (non-special) towers.
    Selling is never chosen. Ties go to the smaller native ID.
    """
    def __init__(self, params: dict | None = None):
        self.params = dict(TEACHER_DEFAULTS if params is None else params)
        self.name = teacher_name(self.params)
        self.branches = {TEACHER_BRANCHES[kind][letter]
                         for kind, letter in zip(TEACHER_BRANCHES, self.params["b"])}
        self.ratio = {kind: int(digit) for kind, digit in zip(DEFAULT_RATIO, self.params["r"])}
        self.cast_progress = self.params["c"] / 100

    def choose(self, state: dict, menu: list[dict], context: dict) -> dict:
        validate_menu(menu)
        if self.params["e"]:
            rng = random.Random(f"teacher_v2:{self.params['s']}:{context['decision_index']}")
            if rng.random() < self.params["e"] / 100:
                others = [item for item in menu
                          if item["action"].get("action") not in ("wait", "sell_tower")]
                if others:
                    return _choice(self, rng.choice(others)["label"], rule="explore", n_options=len(others))
        options: dict[str, list[dict]] = {}
        for item in menu:
            options.setdefault(item["action"].get("action"), []).append(item)
        towers = {tower.get("id"): tower for tower in state.get("towers", []) if isinstance(tower, dict)}
        # A dormant boss (action scope v3) is not on the field yet: no spells for it, no wave hold-back.
        enemies = [enemy for enemy in state.get("enemies", []) if isinstance(enemy, dict) and not enemy.get("dormant")]
        aim = aim_choice(self, state, menu)
        if aim is not None:
            return aim
        if any(_number(enemy.get("path_progress")) >= self.cast_progress for enemy in enemies):
            choice = self._cast(enemies, options.get("use_power", []))
            if choice is not None:
                return choice
        builds = [item for item in options.get("build_tower", [])
                  if self.ratio.get(item["action"].get("tower_type"), 0) > 0]
        cap = self.params.get("m", 0)
        if cap and sum(not tower.get("is_special") and base_kind(tower.get("template")) is not None
                       for tower in towers.values()) >= abs(cap):
            if cap > 0 or _number(state.get("gold")) < SOFT_CAP_GOLD:
                builds = []  # the tower cap is reached: keep the gold for upgrades
        power_ups, tower_ups = options.get("upgrade_power", []), options.get("upgrade_tower", [])
        order = {}
        if builds and (power_ups or tower_ups):
            empty = len({_id_key(item["action"].get("holder_id")) for item in builds})
            built = sum(not tower.get("is_special") for tower in towers.values())
            order = {"build_first": empty > built * self.params["f"] / 100, "empty_holders": empty,
                     "built_towers": built}
            if order["build_first"]:
                return self._build(state, menu, **order)
        if power_ups:
            return self._upgrade_power(towers, power_ups, **order)
        if tower_ups:
            return self._upgrade_tower(towers, tower_ups, **order)
        if builds:
            return self._build(state, menu)
        sends = options.get("send_wave", [])
        if sends and (not enemies or state.get("wave") == 0):
            return _choice(self, sends[0]["label"], rule="send_wave")
        return _choice(self, _wait_label(menu), rule="wait")

    def _cast(self, enemies: list[dict], casts: list[dict]) -> dict | None:
        progress = {_id_key(enemy.get("id")): enemy.get("path_progress") for enemy in enemies}
        for power in (1, 2):
            offered = [item for item in casts if _is_power(item["action"].get("power"), power)]
            if not offered:
                continue
            item = min(offered, key=lambda i: (-_number(progress.get(_id_key(i["action"].get("anchor_id")))),
                                               _id_key(i["action"].get("anchor_id"))))
            action = item["action"]
            return _choice(self, item["label"], rule="use_power", power=power, anchor_id=action.get("anchor_id"),
                           anchor_progress=progress.get(_id_key(action.get("anchor_id"))),
                           x=action.get("x"), y=action.get("y"))
        return None

    def _upgrade_power(self, towers: dict, offered: list[dict], **order) -> dict:
        def key(item):
            tower_id = item["action"].get("tower_id")
            # Highest-scoring tower first, then its cheapest power.
            return (-_number(towers.get(tower_id, {}).get("path_score")), _id_key(tower_id), _price(item),
                    str(item["action"].get("power")))
        item = min(offered, key=key)
        tower_id = item["action"].get("tower_id")
        return _choice(self, item["label"], rule="upgrade_power", tower_id=tower_id,
                       power=item["action"].get("power"), cost=item.get("cost"),
                       path_score=towers.get(tower_id, {}).get("path_score"), **order)

    def _upgrade_tower(self, towers: dict, offered: list[dict], **order) -> dict:
        tower_id = min((item["action"].get("tower_id") for item in offered),
                       key=lambda t: (-_number(towers.get(t, {}).get("path_score")), _id_key(t)))
        own = [item for item in offered if item["action"].get("tower_id") == tower_id]
        # A tower's menu offers only its own kind's branches, so membership picks its preference.
        preferred = [item for item in own if item["action"].get("target") in self.branches]
        item = min(preferred or own, key=lambda i: str(i["action"].get("target")))
        return _choice(self, item["label"], rule="upgrade_tower", tower_id=tower_id,
                       target=item["action"].get("target"), preferred_branch=bool(preferred),
                       path_score=towers.get(tower_id, {}).get("path_score"), **order)

    def _build(self, state: dict, menu: list[dict], **order) -> dict:
        counts = {kind: 0 for kind in DEFAULT_RATIO}
        for tower in state.get("towers", []):
            kind = base_kind(tower.get("template") if isinstance(tower, dict) else None)
            if kind in counts:
                counts[kind] += 1
        ratio = {kind: weight for kind, weight in self.ratio.items() if weight > 0}
        kind = _least_built_kind(menu, counts, ratio)
        item = _best_holder(state, _builds(menu, kind))
        return _choice(self, item["label"], rule="build", tower_type=kind, holder_id=item["action"].get("holder_id"),
                       tower_counts=counts, ratio=ratio, **order)


def make_policy(spec: str):
    """``send_wave_only`` | ``wait_only`` | ``random:<seed>`` | ``single:<kind>`` | ``pressure_greedy``
    | ``teacher_v2``."""
    if not isinstance(spec, str):
        raise ValueError("policy spec must be a string")
    if spec == "send_wave_only":
        return SendWaveOnly()
    if spec == "wait_only":
        return WaitOnly()
    if spec == "pressure_greedy":
        return PressureGreedy()
    if spec == "teacher_v2" or spec.startswith("teacher_v2:"):
        params = parse_teacher_params(spec[len("teacher_v2:"):] if ":" in spec else "")
        policy = TeacherV2(params)
        if policy.name != spec:
            raise ValueError(f"non-canonical teacher_v2 spec: {spec!r} (canonical {policy.name!r})")
        return policy
    name, separator, argument = spec.partition(":")
    if separator and name == "random":
        try:
            seed = int(argument, 10)
        except ValueError:
            seed = None
        if seed is None or str(seed) != argument:  # canonical digits, so name round-trips
            raise ValueError(f"invalid random policy seed: {spec!r}")
        return RandomLegal(seed)
    if separator and name == "single":
        return SingleTowerType(argument)
    raise ValueError(f"unknown scripted policy: {spec!r}")
