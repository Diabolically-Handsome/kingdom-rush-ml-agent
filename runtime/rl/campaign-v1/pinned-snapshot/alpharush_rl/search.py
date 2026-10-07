"""Build-order plans for one level, their executor policy, and a steady-state genetic search.

A plan (genome) is strategy a person could write down before playing a level:
which hero to bring, which tower to build on which holder, in what order to
upgrade, how deep into the path enemies may come before spells are cast, and
whether waves are called early. The executor turns the plan into native menu
choices; once the plan is exhausted it falls back to the deterministic
``teacher_v2`` rules so leftover gold is still spent.

Holders are named by their mesh id (``holders[].mesh_id``, which a built
tower reports as ``towers[].holder_id``), so a plan is stable across seeds and
processes of the same level.
"""

import copy
import math
import hashlib
import json
import random

from .menus import validate_menu
from .scripted_policies import TeacherV2, _choice, _number, _wait_label, aim_choice

KINDS = ("archer", "barrack", "mage", "engineer")
BRANCHES = {"archer": ("tower_ranger", "tower_musketeer"), "barrack": ("tower_paladin", "tower_barbarian"),
            "mage": ("tower_arcane_wizard", "tower_sorcerer"), "engineer": ("tower_bfg", "tower_tesla")}
FOUR = {target: kind for kind, targets in BRANCHES.items() for target in targets}
# A pending step is dropped once this much gold is banked: no single step costs that much.
SKIP_GOLD = 800
GENOME_KEYS = ("hero", "steps", "cast", "early", "branches")
# Optional plan keys and their defaults (a canonical plan omits a key at its default).
OPTIONAL_GENOME = {"ratio": "3111", "boss": 0, "pkg": "balanced"}
# "boss" 1: while a boss is alive, spells are cast only at anchors within BOSS_RADIUS of it, else held.
BOSS_RADIUS = 80
MAX_STEPS = 80


def tower_kind_level(template):
    """(base kind, level 1-4) of a tower template, or (None, None) for anything else."""
    if not isinstance(template, str):
        return None, None
    if template in FOUR:
        return FOUR[template], 4
    for kind in KINDS:
        prefix = f"tower_{kind}_"
        if template.startswith(prefix) and template[len(prefix):] in ("1", "2", "3"):
            return kind, int(template[len(prefix):])
        if template == f"tower_build_{kind}":
            return kind, 0
    return None, None


def check_genome(genome: dict) -> dict:
    """Validated canonical copy of a plan."""
    if not isinstance(genome, dict) or not set(GENOME_KEYS) <= set(genome) <= set(GENOME_KEYS) | set(OPTIONAL_GENOME):
        raise ValueError(f"genome keys must be {GENOME_KEYS} (optionally {tuple(OPTIONAL_GENOME)})")
    ratio = genome.get("ratio", OPTIONAL_GENOME["ratio"])
    if not isinstance(ratio, str) or len(ratio) != 4 or not ratio.isdigit() or ratio == "0000":
        raise ValueError("ratio must be four digits for archer, barrack, mage, engineer (not all 0)")
    boss = genome.get("boss", OPTIONAL_GENOME["boss"])
    if boss not in (0, 1) or isinstance(boss, bool):
        raise ValueError("boss must be 0 or 1")
    from .campaign import PACKAGES
    pkg = genome.get("pkg", OPTIONAL_GENOME["pkg"])
    if pkg not in PACKAGES:
        raise ValueError(f"pkg must be one of {sorted(PACKAGES)}")
    hero = genome["hero"]
    if hero is not None and not isinstance(hero, str):
        raise ValueError("hero must be a name or None")
    cast = genome["cast"]
    if isinstance(cast, bool) or not isinstance(cast, int) or not 0 <= cast <= 100:
        raise ValueError("cast must be a percent 0..100")
    if genome["early"] not in (0, 1) or isinstance(genome["early"], bool):
        raise ValueError("early must be 0 or 1")
    branches = genome["branches"]
    if not isinstance(branches, dict) or set(branches) != set(KINDS) or any(
            branches[kind] not in BRANCHES[kind] for kind in KINDS):
        raise ValueError("branches must name one four-level branch per kind")
    steps = genome["steps"]
    if not isinstance(steps, list) or len(steps) > MAX_STEPS:
        raise ValueError(f"steps must be a list of at most {MAX_STEPS}")
    out = []
    for step in steps:
        if not isinstance(step, list) or not step or step[0] not in ("b", "u", "k"):
            raise ValueError(f"invalid step {step!r}")
        if not isinstance(step[1] if len(step) > 1 else None, str):
            raise ValueError(f"step holder must be a mesh id string: {step!r}")
        if step[0] == "b" and (len(step) != 3 or step[2] not in KINDS):
            raise ValueError(f"build step is ['b', mesh, kind]: {step!r}")
        if step[0] in ("u", "k") and len(step) != 2:
            raise ValueError(f"upgrade/skill step is [op, mesh]: {step!r}")
        out.append(list(step))
    extra = {} if ratio == OPTIONAL_GENOME["ratio"] else {"ratio": ratio}
    if boss != OPTIONAL_GENOME["boss"]:
        extra["boss"] = boss
    if pkg != OPTIONAL_GENOME["pkg"]:
        extra["pkg"] = pkg
    return {**extra, "hero": hero, "steps": out, "cast": cast, "early": genome["early"],
            "branches": {kind: branches[kind] for kind in KINDS}}


def genome_id(genome: dict) -> str:
    text = json.dumps(check_genome(genome), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode()).hexdigest()[:16]


class BuildOrderPolicy:
    """Executes a plan's steps in order; spells, wave calls and fallback follow fixed rules."""

    def __init__(self, genome: dict):
        self.genome = check_genome(genome)
        self.name = "build_order:" + genome_id(self.genome)
        self.cursor = 0
        self.pending = 0
        branch_letters = "".join(
            {"tower_ranger": "r", "tower_musketeer": "m", "tower_paladin": "p", "tower_barbarian": "b",
             "tower_arcane_wizard": "a", "tower_sorcerer": "s", "tower_bfg": "b", "tower_tesla": "t"}[
                self.genome["branches"][kind]] for kind in KINDS)
        from .scripted_policies import TEACHER_DEFAULTS
        self.fallback = TeacherV2({**TEACHER_DEFAULTS, "b": branch_letters, "c": self.genome["cast"],
                                   "r": self.genome.get("ratio", OPTIONAL_GENOME["ratio"])})

    def _step_option(self, step, state, menu, holders, towers):
        """("take", item) | ("wait", None) | ("skip", reason) for the current step."""
        op, mesh = step[0], step[1]
        tower = towers.get(mesh)
        locked = set(state.get("locked_towers") or [])
        if op == "b":
            if tower is not None:
                return "skip", "holder_built"
            if mesh not in holders:
                return "skip", "no_holder"
            for item in menu:
                action = item["action"]
                if (action.get("action") == "build_tower" and action.get("holder_id") == holders[mesh]
                        and action.get("tower_type") == step[2]):
                    return "take", item
            if f"tower_{step[2]}_1" in locked or f"tower_build_{step[2]}" in locked:
                return "skip", "locked"
            return "wait", None
        if tower is None:
            return "skip", "no_tower"
        own = [item for item in menu if item["action"].get("tower_id") == tower.get("id")]
        if tower.get("is_special"):
            # Only a power-selling special tower (the sunray) can take a "k" step.
            powers = [p for p in tower.get("powers") or [] if isinstance(p, dict)]
            if op != "k" or not powers:
                return "skip", "not_standard"
            if all(p.get("level") == p.get("max_level") for p in powers):
                return "skip", "skills_full"
            offered = [item for item in own if item["action"].get("action") == "upgrade_power"]
            if offered:
                return "take", min(offered, key=lambda i: (_number(i.get("cost")), str(i["action"].get("power"))))
            return "wait", None
        kind, level = tower_kind_level(tower.get("template"))
        if kind is None:
            return "skip", "not_standard"
        if level == 0:
            return "wait", None  # still under construction
        if op == "u":
            if level == 4:
                return "skip", "max_level"
            if level < 3:
                targets = [f"tower_{kind}_{level + 1}"]
            else:
                preferred = self.genome["branches"][kind]
                targets = [preferred] + [t for t in BRANCHES[kind] if t != preferred]
            targets = [t for t in targets if t not in locked]
            if not targets:
                return "skip", "locked"
            for target in targets:
                for item in own:
                    if item["action"].get("action") == "upgrade_tower" and item["action"].get("target") == target:
                        return "take", item
            return "wait", None
        # op == "k": the tower's cheapest offered skill level.
        if level != 4:
            return "skip", "not_four"
        powers = tower.get("powers") if isinstance(tower.get("powers"), list) else []
        if powers and all(isinstance(p, dict) and p.get("level") == p.get("max_level") for p in powers):
            return "skip", "skills_full"
        offered = [item for item in own if item["action"].get("action") == "upgrade_power"]
        if offered:
            return "take", min(offered, key=lambda i: (_number(i.get("cost")), str(i["action"].get("power"))))
        return "wait", None

    def choose(self, state: dict, menu: list[dict], context: dict) -> dict:
        validate_menu(menu)
        clicks = [item for item in menu if item["action"].get("action") == "click_entity"]
        if clicks:
            # A click a script waits for (finish a downed boss, break ice off a tower) is always taken first.
            item = min(clicks, key=lambda i: (_number(i["action"].get("entity_id")), i["label"]))
            return _choice(self, item["label"], rule="click", entity_id=item["action"].get("entity_id"),
                           plan_cursor=self.cursor)
        aim = aim_choice(self, state, menu)
        if aim is not None:
            return self._own(aim, "point_tower")
        enemies = [e for e in state.get("enemies", []) if isinstance(e, dict)]
        casts = [item for item in menu if item["action"].get("action") == "use_power"]
        bosses = [e for e in enemies if str(e.get("template") or "").startswith("eb_") and _number(e.get("hp")) > 0]
        if self.genome.get("boss") and bosses:
            # Boss focus: spells only near the boss (rain of fire first), otherwise held for it.
            boss = max(bosses, key=lambda e: (_number(e.get("hp")), -_number(e.get("id"))))
            near = [item for item in casts
                    if (_number(item["action"].get("x")) - _number(boss.get("x"))) ** 2
                    + (_number(item["action"].get("y")) - _number(boss.get("y"))) ** 2 <= BOSS_RADIUS ** 2]
            if near:
                item = min(near, key=lambda i: (_number(i["action"].get("power")), str(i["action"].get("anchor_id"))))
                return _choice(self, item["label"], rule="boss_cast", power=item["action"].get("power"),
                               boss=boss.get("template"), plan_cursor=self.cursor)
            casts = []
            # Held for the boss: neither the plan nor the fallback may spend the spells elsewhere.
            menu = [item for item in menu if item["action"].get("action") != "use_power"]
        if casts and any(_number(e.get("path_progress")) >= self.genome["cast"] / 100 for e in enemies):
            choice = self.fallback._cast(enemies, casts)
            if choice is not None:
                return self._own(choice, choice["meta"]["rule"])
        holders = {str(h.get("mesh_id")): h.get("id") for h in state.get("holders", [])
                   if isinstance(h, dict) and h.get("mesh_id") is not None}
        towers = {str(t.get("holder_id")): t for t in state.get("towers", [])
                  if isinstance(t, dict) and t.get("holder_id") is not None}
        steps = self.genome["steps"]
        skipped = []
        while self.cursor < len(steps):
            verdict, item = self._step_option(steps[self.cursor], state, menu, holders, towers)
            if verdict == "skip":
                skipped.append([self.cursor, item])
                self.cursor += 1
                self.pending = 0
                continue
            if verdict == "take":
                step = steps[self.cursor]
                self.cursor += 1
                self.pending = 0
                return _choice(self, item["label"], rule="plan", step=step, plan_cursor=self.cursor - 1,
                               skipped=skipped)
            # Waiting for gold; a step that stays unavailable with a full purse is impossible.
            if _number(state.get("gold")) >= SKIP_GOLD:
                skipped.append([self.cursor, "unavailable"])
                self.cursor += 1
                self.pending = 0
                continue
            self.pending += 1
            sends = [item for item in menu if item["action"].get("action") == "send_wave"]
            if sends and (state.get("wave") == 0 or (self.genome["early"] and not enemies)):
                return _choice(self, sends[0]["label"], rule="send_wave", plan_cursor=self.cursor, skipped=skipped)
            return _choice(self, _wait_label(menu), rule="plan_wait", plan_cursor=self.cursor, skipped=skipped)
        choice = self.fallback.choose(state, menu, context)
        rule = choice["meta"].get("rule")
        if rule == "send_wave" and not self.genome["early"] and state.get("wave") != 0:
            return _choice(self, _wait_label(menu), rule="fallback_wait", plan_cursor=self.cursor)
        return self._own(choice, f"fallback_{rule}")

    def _own(self, choice, rule):
        """A rule-module choice re-labelled as this plan's decision."""
        return _choice(self, choice["label"], **{**choice["meta"], "rule": rule, "plan_cursor": self.cursor})


# ---------------------------------------------------------------- search operators


def level_specials(state: dict) -> list[str]:
    """Mesh ids of a level's special towers that sell powers (plan "k" steps may buy them)."""
    return sorted(str(t["holder_id"]) for t in state.get("towers", [])
                  if isinstance(t, dict) and t.get("is_special") and t.get("holder_id") is not None
                  and any(isinstance(p, dict) for p in t.get("powers") or []))


def level_holders(state: dict) -> list[dict]:
    """Standard holders of a level's initial state: mesh id, position and path score."""
    out = []
    for holder in state.get("holders", []):
        if isinstance(holder, dict) and holder.get("mesh_id") is not None and not holder.get("blocked"):
            out.append({"mesh": str(holder["mesh_id"]), "x": holder.get("x"), "y": holder.get("y"),
                        "path_score": _number(holder.get("path_score"))})
    return sorted(out, key=lambda h: h["mesh"])


def _weighted_holders(holders, rng, count):
    pool = list(holders)
    chosen = []
    while pool and len(chosen) < count:
        weights = [1.0 + h["path_score"] for h in pool]
        pick = rng.choices(range(len(pool)), weights=weights)[0]
        chosen.append(pool.pop(pick)["mesh"])
    return chosen


def random_genome(holders: list[dict], heroes: list, rng: random.Random, specials=()) -> dict:
    """A plausible random plan: a handful of builds, upgrades of some of them, maybe skills."""
    count = rng.randint(max(1, min(3, len(holders))), max(1, min(len(holders), 10)))
    meshes = _weighted_holders(holders, rng, count)
    kinds = [rng.choices(KINDS, weights=(3, 2, 3, 2))[0] for _ in meshes]
    steps = [["b", mesh, kind] for mesh, kind in zip(meshes, kinds)]
    for _ in range(rng.randint(0, 3 * len(meshes))):
        mesh = rng.choice(meshes)
        first = next(i for i, s in enumerate(steps) if s[0] == "b" and s[1] == mesh)
        steps.insert(rng.randint(first + 1, len(steps)), ["u", mesh])
    for _ in range(rng.randint(0, 2)):
        steps.append(["k", rng.choice(meshes)])
    for mesh in specials:
        for _ in range(rng.randint(0, 4)):
            steps.insert(rng.randint(0, len(steps)), ["k", mesh])
    return check_genome({"hero": rng.choice(list(heroes)) if heroes else None, "steps": steps[:MAX_STEPS],
                         "cast": rng.choice((30, 40, 50, 60, 70)), "early": rng.choice((0, 1)),
                         "branches": {kind: rng.choice(BRANCHES[kind]) for kind in KINDS}})


def teacher_genome(holders: list[dict], heroes: list) -> dict:
    """The plain teacher's opening as a plan: best-scoring holders, 3:1:1:1, no explicit upgrades."""
    ranked = sorted(holders, key=lambda h: (-h["path_score"], h["mesh"]))
    cycle = ("archer", "barrack", "mage", "engineer", "archer", "archer")
    steps = [["b", h["mesh"], cycle[i % len(cycle)]] for i, h in enumerate(ranked[:6])]
    return check_genome({"hero": heroes[0] if heroes else None, "steps": steps, "cast": 50, "early": 1,
                         "branches": {kind: BRANCHES[kind][0] for kind in KINDS}})


def _repair(steps, specials=()):
    """Keep the first build per holder; drop upgrades/skills that precede their holder's build
    (special towers stand from the start and take only skill steps)."""
    built, out = set(), []
    for step in steps:
        if step[1] in specials:
            if step[0] != "k":
                continue
        elif step[0] == "b":
            if step[1] in built:
                continue
            built.add(step[1])
        elif step[1] not in built:
            continue
        out.append(step)
    return out[:MAX_STEPS]


def mutate(genome: dict, holders: list[dict], heroes: list, rng: random.Random, specials=()) -> dict:
    g = copy.deepcopy(check_genome(genome))
    steps = g["steps"]
    for _ in range(rng.choice((1, 1, 1, 2, 2, 3))):
        op = rng.choice(("add_build", "add_upgrade", "add_upgrade", "remove", "swap", "swap", "kind", "move",
                         "skill", "param", "relocate", "relocate", "burst"))
        built = [s[1] for s in steps if s[0] == "b"]
        free = [h for h in holders if h["mesh"] not in built]
        if op == "add_build" and free:
            mesh = _weighted_holders(free, rng, 1)[0]
            steps.insert(rng.randint(0, len(steps)), ["b", mesh, rng.choice(KINDS)])
        elif op == "add_upgrade" and built:
            mesh = rng.choice(built)
            first = next(i for i, s in enumerate(steps) if s[0] == "b" and s[1] == mesh)
            steps.insert(rng.randint(first + 1, len(steps)), ["u", mesh])
        elif op == "skill" and (built or specials):
            steps.insert(rng.randint(0, len(steps)), ["k", rng.choice(built + list(specials))])
        elif op == "remove" and steps:
            del steps[rng.randrange(len(steps))]
        elif op == "swap" and len(steps) > 1:
            i = rng.randrange(len(steps) - 1)
            steps[i], steps[i + 1] = steps[i + 1], steps[i]
        elif op == "move" and len(steps) > 1:
            step = steps.pop(rng.randrange(len(steps)))
            steps.insert(rng.randint(0, len(steps)), step)
        elif op == "relocate" and built and free:
            # The same tower (and its upgrades and skills) on another free holder.
            mesh = rng.choice(built)
            target = _weighted_holders(free, rng, 1)[0]
            steps[:] = [[s[0], target, *s[2:]] if s[1] == mesh else s for s in steps]
        elif op == "burst" and built:
            # Upgrade one tower twice in a row, right after its build or anywhere later.
            mesh = rng.choice(built)
            first = next(i for i, s in enumerate(steps) if s[0] == "b" and s[1] == mesh)
            at = rng.randint(first + 1, len(steps))
            steps[at:at] = [["u", mesh], ["u", mesh]]
        elif op == "kind" and built:
            index = rng.choice([i for i, s in enumerate(steps) if s[0] == "b"])
            steps[index] = ["b", steps[index][1], rng.choice([k for k in KINDS if k != steps[index][2]])]
        else:
            which = rng.choice(("cast", "early", "branch", "hero", "ratio", "ratio", "boss", "pkg", "pkg"))
            if which == "cast":
                g["cast"] = min(100, max(0, g["cast"] + rng.choice((-20, -10, 10, 20))))
            elif which == "ratio":
                digits = [int(d) for d in g.get("ratio", OPTIONAL_GENOME["ratio"])]
                index = rng.randrange(4)
                digits[index] = min(9, max(0, digits[index] + rng.choice((-1, 1, 2))))
                if any(digits):
                    g["ratio"] = "".join(map(str, digits))
            elif which == "early":
                g["early"] = 1 - g["early"]
            elif which == "boss":
                g["boss"] = 1 - g.get("boss", OPTIONAL_GENOME["boss"])
            elif which == "pkg":
                from .campaign import PACKAGES
                g["pkg"] = rng.choice(sorted(PACKAGES))
            elif which == "branch":
                kind = rng.choice(KINDS)
                g["branches"][kind] = rng.choice(BRANCHES[kind])
            elif heroes:
                g["hero"] = rng.choice(list(heroes))
        steps[:] = _repair(steps, specials)
    return check_genome(g)


def crossover(a: dict, b: dict, rng: random.Random, specials=()) -> dict:
    a, b = check_genome(a), check_genome(b)
    cut_a = rng.randint(0, len(a["steps"]))
    cut_b = rng.randint(0, len(b["steps"]))
    child = copy.deepcopy(a)
    child["steps"] = _repair(a["steps"][:cut_a] + b["steps"][cut_b:], specials)
    for key in ("hero", "cast", "early"):
        child[key] = rng.choice((a[key], b[key]))
    child["ratio"] = rng.choice((a.get("ratio", OPTIONAL_GENOME["ratio"]), b.get("ratio", OPTIONAL_GENOME["ratio"])))
    child["boss"] = rng.choice((a.get("boss", OPTIONAL_GENOME["boss"]), b.get("boss", OPTIONAL_GENOME["boss"])))
    child["pkg"] = rng.choice((a.get("pkg", OPTIONAL_GENOME["pkg"]), b.get("pkg", OPTIONAL_GENOME["pkg"])))
    child["branches"] = {kind: rng.choice((a["branches"][kind], b["branches"][kind])) for kind in KINDS}
    return check_genome(child)


def _finite(value, default=0.0) -> float:
    """A finite float (journals refuse inf/nan): unknown or non-finite values become ``default``."""
    number = _number(value)
    return number if math.isfinite(number) else default


class ToughestEnemyWatch:
    """Wraps a policy and remembers, over the whole episode, the enemy with the largest maximum health
    and the health it was last seen with (it may later die or walk out of the map; a boss may also heal
    into a second phase, so its lowest health ever is not the measure)."""

    def __init__(self, policy):
        self.policy, self.name = policy, policy.name
        self.toughest = None  # {"id", "template", "hp", "hp_max"} at its lowest seen health

    def watch(self, state):
        for enemy in state.get("enemies", []) if isinstance(state, dict) else []:
            if not isinstance(enemy, dict) or _finite(enemy.get("hp_max")) <= 0:
                continue
            hp_max, hp = _finite(enemy.get("hp_max")), max(0.0, _finite(enemy.get("hp")))
            known = self.toughest
            if known is None or hp_max > known["hp_max"]:
                self.toughest = {"id": enemy.get("id"), "template": enemy.get("template"), "hp": hp, "hp_max": hp_max}
            elif enemy.get("id") == known["id"]:
                known["hp"] = hp

    def choose(self, state, menu, context):
        self.watch(state)
        return self.policy.choose(state, menu, context)


def search_signals(result: dict, final_state: dict, toughest=None) -> dict:
    """Pressure measures of one played episode: lives when its last wave began, and the toughest enemy
    (usually a boss) at the lowest health it was ever seen at, from the watch when given, else the
    strongest enemy still on the field at the end."""
    decisions = result.get("decisions") or []
    last_wave = (result.get("final_summary") or {}).get("wave")
    lives = next((d.get("lives") for d in decisions if d.get("wave") == last_wave), None)
    strongest = None
    if toughest is not None:
        strongest = {"template": toughest.get("template"), "hp": max(0.0, _finite(toughest.get("hp"))),
                     "hp_max": _finite(toughest.get("hp_max"))}
    else:
        enemies = [e for e in final_state.get("enemies", []) if isinstance(e, dict) and _finite(e.get("hp_max")) > 0]
        best = max(enemies, key=lambda e: (_finite(e.get("hp_max")), -_finite(e.get("id"))), default=None)
        if best is not None:
            strongest = {"template": best.get("template"), "hp": max(0.0, _finite(best.get("hp"))),
                         "hp_max": _finite(best.get("hp_max"))}
    return {"wave_start_lives": None if lives is None else _finite(lives), "strongest": strongest}


def fitness(result: dict) -> float:
    """Wins first (more lives, then faster), else how far the defence held: waves, then (when the
    episode carries search signals) lives kept into the last wave and damage done to the strongest
    enemy still standing, then ticks."""
    outcome = result.get("outcome") or {}
    summary = result.get("final_summary") or {}
    if result.get("status") != "terminal":
        return -1.0
    if outcome.get("level_won"):
        return 10000 + 100 * _number(outcome.get("lives")) - _number(result.get("final_tick")) / 100000
    wave = _number(summary.get("wave"))
    signals = result.get("search_signals")
    if not isinstance(signals, dict):
        return 100 * max(0.0, wave - 1) + min(99.0, _number(result.get("final_tick")) / 1000)
    score = 100 * max(0.0, wave - 1) + 2 * min(20.0, max(0.0, _finite(signals.get("wave_start_lives"))))
    strongest = signals.get("strongest")
    if isinstance(strongest, dict) and _finite(strongest.get("hp_max")) > 0:
        score += 40 * (1 - min(1.0, max(0.0, _finite(strongest.get("hp")) / _finite(strongest.get("hp_max")))))
    else:
        score += 40  # nothing left standing
    return score + min(19.0, _number(result.get("final_tick")) / 5000)


class LevelSearch:
    """Steady-state GA over one level's plans (population kept sorted by fitness)."""

    def __init__(self, level: int, holders: list[dict], heroes: list, seed: str, population: int = 24,
                 seeds_for_wins=(), specials=()):
        self.level, self.holders, self.heroes = level, holders, list(heroes)
        self.specials = list(specials)
        self.rng = random.Random(f"search:{seed}:{level}")
        self.size = population
        self.population: list[tuple[float, str, dict]] = []  # (fitness, genome id, genome)
        self.seen: set[str] = set()
        self.evaluations = 0
        self.pending: set[str] = set()
        self.queued: list[dict] = []

    def propose(self) -> dict | None:
        """Next untried plan: queued plans first, then seeds, then children of tournament-selected parents."""
        while self.queued:
            genome = self.queued.pop(0)
            key = genome_id(genome)
            if key not in self.seen:
                self.seen.add(key)
                self.pending.add(key)
                return genome
        for _ in range(200):
            if len(self.population) + len(self.pending) < self.size // 2 or not self.population:
                if not self.seen:
                    genome = teacher_genome(self.holders, self.heroes)
                else:
                    genome = random_genome(self.holders, self.heroes, self.rng, self.specials)
            else:
                parent = self._tournament()
                if self.rng.random() < 0.3 and len(self.population) > 1:
                    genome = mutate(crossover(parent, self._tournament(), self.rng, self.specials), self.holders,
                                    self.heroes, self.rng, self.specials)
                else:
                    genome = mutate(parent, self.holders, self.heroes, self.rng, self.specials)
            key = genome_id(genome)
            if key not in self.seen:
                self.seen.add(key)
                self.pending.add(key)
                return genome
        return None

    def _tournament(self) -> dict:
        picks = [self.rng.randrange(len(self.population)) for _ in range(3)]
        return self.population[min(picks)][2]

    def queue(self, genome: dict):
        """Have this plan evaluated before any new proposal (e.g. a warm start to re-measure)."""
        self.queued.append(check_genome(genome))

    def adopt(self, genome: dict, score: float):
        """Insert an already evaluated plan (e.g. from an earlier run) without counting an evaluation."""
        key = genome_id(genome)
        if key in self.seen:
            return
        self.seen.add(key)
        self.population.append((score, key, check_genome(genome)))
        self.population.sort(key=lambda entry: (-entry[0], entry[1]))
        del self.population[self.size:]

    def report(self, genome: dict, score: float):
        key = genome_id(genome)
        self.pending.discard(key)
        self.evaluations += 1
        self.population.append((score, key, check_genome(genome)))
        self.population.sort(key=lambda entry: (-entry[0], entry[1]))
        del self.population[self.size:]

    def best(self):
        return self.population[0] if self.population else None
