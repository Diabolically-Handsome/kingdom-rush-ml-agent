"""Strategy-level decisions of a campaign: star upgrades between levels and the battle plan per level.

Two interchangeable brains make the same decisions from the same legal menus:
``RuleBrain`` (the fixed human-readable rules) and ``LanguageBrain`` (the 8B
model, scoring every legal option label by log-likelihood through the local
model worker, so it can only ever pick a legal option). A battle plan carries
its hero, so choosing the plan also chooses the hero. The operator that plays
the level only executes the chosen plan.
"""

import hashlib
import json

from .campaign import (BUY_PRIORITY, PACKAGES, STAR_PRICES, STAR_TREES, check_upgrades, package_upgrades,
                       upgrades_cost)
from .search import BRANCHES, KINDS, check_genome, genome_id

LABELS = [chr(ord("A") + i) for i in range(26)]
# The 8B's label scores carry a position bias (in probes it favoured option B whatever B said), so every
# menu is scored in up to ROTATIONS cyclic orders and each option's probability is averaged over them.
ROTATIONS = 6
UPGRADE_TEXT = {
    "archers": ("Salvage (better sell refund)", "Eagle Eye (+range)", "Piercing Shots (ignore armor)",
                "Far Shots (+range)", "Precision (critical shots)"),
    "barracks": ("Survival (+soldier health)", "Better Armor", "Improved Deployment (faster respawn, wider rally)",
                 "Survival 2 (+soldier health)", "Barbed Armor (thorns damage)"),
    "mages": ("Spell Reach (+range)", "Arcane Shatter (armor reduction)", "Hermetic Study (cheaper mage towers)",
              "Empowered Magic (+damage)", "Slow Curse"),
    "engineers": ("Concentrated Fire (+damage)", "Range Finder (+range)", "Field Logistics (cheaper upgrades)",
                  "Industrialization (cheaper towers)", "Efficiency (faster reload)"),
    "rain": ("Blazing Skies (more fireballs)", "Scorched Earth (burning ground)", "Bigger and Meaner (+damage)",
             "Blazing Earth (longer burn)", "Cataclysm (meteor storm)"),
    "reinforcements": ("Reinforcements level 1", "Reinforcements level 2", "Reinforcements level 3",
                       "Reinforcements level 4", "Reinforcements level 5"),
}
TREE_NAME = {"archers": "Archer towers", "barracks": "Barracks", "mages": "Mage towers",
             "engineers": "Artillery towers", "rain": "Rain of Fire spell", "reinforcements": "Reinforcements spell"}
HERO_TEXT = {"hero_gerald": "Gerald Lightseeker (sturdy paladin, blocks melee enemies)",
             "hero_alleria": "Alleria Swiftwind (ranged archer, calls a wildcat)",
             "hero_malik": "Malik Hammerfury (melee, area stun smash)",
             "hero_bolin": "Bolin Farslayer (rifleman, tar bombs and mines)",
             "hero_magnus": "Magnus Spellbane (mage, creates illusions, arcane rain)",
             "hero_ignus": "Ignus (fire elemental, area fire damage)"}
KIND_NAME = {"archer": "archer", "barrack": "barracks", "mage": "mage", "engineer": "artillery"}
SYSTEM = ("You are the strategy brain of an agent playing Kingdom Rush (2011) through the main campaign on Normal "
          "difficulty, starting from a new save and playing levels in order. You make only strategic choices; a "
          "separate operator executes them in battle. Winning every level comes first, then keeping lives (18+ "
          "lives left earns 3 stars, 6+ earns 2). Reply with exactly one option label from the list.")


class RuleBrain:
    """The fixed rules: buy evenly in a fixed tree order; plans in the given (validated) order."""
    name = "rule"

    def buy(self, stars, owned, context):
        """The allocation the chosen plan was tested with (the balanced rule unless the plan names one)."""
        return package_upgrades(context.get("plan_package", "balanced"), context["won"]), []

    def choose_plan(self, level, candidates, attempt, context):
        return 0, []  # the first offered plan (the runner offers untried plans in validated order)


def describe_plan(genome, evidence=None):
    genome = check_genome(genome)
    builds = [s for s in genome["steps"] if s[0] == "b"]
    upgrades = sum(s[0] == "u" for s in genome["steps"])
    skills = sum(s[0] == "k" for s in genome["steps"])
    counts = {kind: sum(s[2] == kind for s in builds) for kind in KINDS}
    order = ", ".join(f"{KIND_NAME[s[2]]}@{s[1]}" for s in builds[:6])
    branches = ", ".join(genome["branches"][kind].replace("tower_", "") for kind in KINDS)
    hero = HERO_TEXT.get(genome["hero"], "no hero") if genome["hero"] else "no hero"
    text = (f"hero: {hero}; opening builds: {order or 'none'}; planned towers: "
            + ", ".join(f"{counts[k]} {KIND_NAME[k]}" for k in KINDS if counts[k])
            + f"; {upgrades} planned upgrades, {skills} special-skill purchases; level-4 choices: {branches}; "
            + f"spells when the leading enemy has covered {genome['cast']}% of its path; "
            + ("calls the next wave early when the field is clear" if genome["early"] else "lets waves come on their own")
            + "; after the plan, keeps building archer:barracks:mage:artillery at "
            + ":".join(genome.get("ratio", "3111"))
            + ("; saves spells for the boss" if genome.get("boss") else "")
            + (f"; tested with star upgrades allocated {PACKAGE_TEXT[genome['pkg']]}" if genome.get("pkg") else ""))
    if evidence:
        text += f"; practice results: {evidence}"
    return text


def _greedy(stars, owned, order):
    """Buy repeatedly the cheapest affordable next level, ties in ``order`` (a focus puts its trees first)."""
    levels = dict(owned)
    while True:
        left = stars - upgrades_cost(levels)
        offers = [(levels[t], order.index(t), t) for t in order
                  if levels[t] < 5 and STAR_PRICES[t][levels[t]] <= left]
        if not offers:
            return levels
        _, _, tree = min(offers)
        levels[tree] += 1


def _focus(stars, owned, trees):
    """Spend on ``trees`` first (lowest level first), then evenly on the rest."""
    levels = dict(owned)
    while True:
        left = stars - upgrades_cost(levels)
        offers = [(levels[t], t) for t in trees if levels[t] < 5 and STAR_PRICES[t][levels[t]] <= left]
        if not offers:
            break
        levels[min(offers)[1]] += 1
    return _greedy(stars, levels, list(BUY_PRIORITY))


def _bought(before, after):
    parts = [f"{TREE_NAME[t]} to {after[t]}" for t in STAR_TREES if after[t] > before[t]]
    return ", ".join(parts) if parts else "nothing"


PACKAGE_TEXT = {"balanced": "the balanced allocation", "rain": "Rain of Fire first",
                "spells": "both spells first", "archers": "Archer towers first", "barracks": "Barracks first",
                "mages": "Mage towers first", "engineers": "Artillery towers first"}


def _allocation(upgrades):
    return ", ".join(f"{TREE_NAME[t]} {upgrades[t]}" for t in STAR_TREES if upgrades[t]) or "nothing"


def allocation_packages(won, owned, plan_package, plan_kinds):
    """[(text, upgrades)] for the next level: the plan's tested allocation first (recommended), the balanced
    one, a focus on the plan's main tower kind, both spells, and keeping the current allocation."""
    tree_of = {"archer": "archers", "barrack": "barracks", "mage": "mages", "engineer": "engineers"}
    main = max(plan_kinds, key=lambda k: (plan_kinds[k], k), default="mage")
    order = [plan_package, "balanced", tree_of.get(main, "mages"), "spells"]
    owned, seen, out = check_upgrades(owned), set(), []
    for name in order:
        if name in seen:
            continue
        seen.add(name)
        upgrades = package_upgrades(name, won)
        label = ("Reset and allocate the stars as this battle plan was tested with (recommended): "
                 if name == plan_package else f"Reset and allocate {PACKAGE_TEXT[name]}: ")
        out.append((label + _allocation(upgrades), upgrades))
    out.append(("Keep the current allocation: " + _allocation(owned), dict(owned)))
    return out


def purchase_packages(stars, owned, plan_kinds):
    """[(text, upgrades)]: the balanced rule the plans were tested with, two focused alternatives, keeping."""
    owned = check_upgrades(owned)
    balanced = _greedy(stars, owned, list(BUY_PRIORITY))
    tree_of = {"archer": "archers", "barrack": "barracks", "mage": "mages", "engineer": "engineers"}
    main = max(plan_kinds, key=lambda k: (plan_kinds[k], k), default="mage")
    focus_tree = tree_of.get(main, "mages")
    towers = _focus(stars, owned, [focus_tree])
    spells = _focus(stars, owned, ["rain", "reinforcements"])
    return [(f"Balanced purchase in the order the battle plans were tested with (recommended): buy "
             f"{_bought(owned, balanced)}", balanced),
            (f"Focus on {TREE_NAME[focus_tree]} (the towers this plan builds most): buy {_bought(owned, towers)}",
             towers),
            (f"Focus on the two spells: buy {_bought(owned, spells)}", spells),
            ("Keep all unspent stars for later", dict(owned))]


class LanguageBrain:
    """The 8B strategy brain; every decision is a legal menu scored by the local model worker."""

    def __init__(self, broker, name="8b"):
        self.broker = broker
        self.name = name
        self.counter = 0

    def _ask(self, kind, user, options):
        """The option with the highest probability averaged over cyclic orders of the menu (with at most
        ROTATIONS options every option is scored once in every position)."""
        n = len(options)
        labels = LABELS[:n]
        total, ids, hashes, model = [0.0] * n, [], [], None
        shifts = range(min(n, ROTATIONS))
        for shift in shifts:
            order = [(i + shift) % n for i in range(n)]
            self.counter += 1
            request = {"id": f"strategy-{kind}-{self.counter}", "system": SYSTEM,
                       "user": user + "\n\nOptions:\n" + "\n".join(f"{label}. {options[i]}"
                                                                 for label, i in zip(labels, order))
                       + "\n\nAnswer with one label.", "labels": labels}
            response = self.broker.distribution(request)
            for position, i in enumerate(order):
                total[i] += float(response["p"][position])
            ids.append(request["id"])
            hashes.append(hashlib.sha256(request["user"].encode("utf-8")).hexdigest())
            model = response.get("model")
        p = [x / len(shifts) for x in total]
        index = max(range(n), key=lambda i: (p[i], -i))
        record = {"kind": kind, "request_id": ids[0], "request_ids": ids, "prompt_sha256": hashes[0],
                  "prompt_sha256s": hashes, "options": options, "choice": index, "p": p, "model": model,
                  "rotations": len(shifts)}
        return index, record

    def buy(self, stars, owned, context):
        """One decision per level: how to allocate all earned stars (the upgrades screen's reset refunds
        every star for free, so any package may be applied before any level)."""
        owned = check_upgrades(owned or {})
        packages = allocation_packages(context["won"], owned, context.get("plan_package", "balanced"),
                                       context.get("plan_kinds") or {})
        if len({json.dumps(p[1], sort_keys=True) for p in packages}) <= 1:
            return packages[0][1], []
        status = ", ".join(f"{TREE_NAME[t]} {owned[t]}/5" for t in STAR_TREES)
        plan = f" The chosen battle plan for it: {context['plan']}." if context.get("plan") else ""
        user = (f"Campaign so far: {context['progress']}. Next level: {context['next_level']}.{plan} "
                f"Stars earned: {stars}. Current star upgrades: {status}. The upgrades screen can reset all "
                "upgrades for free and spend the stars again. Choose the allocation for the next level.")
        options = [text for text, _ in packages]
        index, record = self._ask("upgrade", user, options)
        return check_upgrades(packages[index][1]), [record]

    def buy_one_by_one(self, stars, owned, context):
        owned = check_upgrades(owned or {})
        records = []
        while True:
            left = stars - upgrades_cost(owned)
            offers = [tree for tree in BUY_PRIORITY if owned[tree] < 5 and STAR_PRICES[tree][owned[tree]] <= left]
            if not offers:
                return owned, records
            status = ", ".join(f"{TREE_NAME[t]} {owned[t]}/5" for t in STAR_TREES)
            plan = f" The chosen battle plan for it: {context['plan']}." if context.get("plan") else ""
            user = (f"Campaign so far: {context['progress']}. Next level: {context['next_level']}.{plan} "
                    f"Stars earned: {stars}, unspent: {left}. Star upgrades owned: {status}. "
                    "Upgrades are permanent and apply to every later level. Buy one upgrade now, or keep the "
                    "remaining stars.")
            options = [f"Buy {TREE_NAME[t]} level {owned[t] + 1}: {UPGRADE_TEXT[t][owned[t]]} "
                       f"(costs {STAR_PRICES[t][owned[t]]} stars)" for t in offers]
            options.append("Keep the remaining stars for later")
            index, record = self._ask("upgrade", user, options)
            records.append(record)
            if index == len(offers):
                return owned, records
            owned = {**owned, offers[index]: owned[offers[index]] + 1}

    def choose_plan(self, level, candidates, attempt, context):
        user = (f"Campaign so far: {context['progress']}. Star upgrades owned: {context['upgrades']}. "
                f"Level {level} is next (attempt {attempt + 1}). Holders are named by map slot ids. "
                "Choose the battle plan most likely to win this level with many lives left.")
        options = [describe_plan(genome, evidence) for genome, evidence in candidates]
        if context.get("failed"):
            user += " Plans already tried and lost on this level: " + ", ".join(
                LABELS[i] for i in context["failed"]) + "."
        index, record = self._ask("plan", user, options)
        return index, [record]
