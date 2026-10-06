"""Star allocation packages: the upgrades screen can reset (refund every star), so each level may start
from a fresh allocation. A plan may name the package it was tested with; the strategy brain chooses."""
import os
from pathlib import Path

ROOT = Path(os.environ.get("PATCH_ROOT", "C:/Users/<user>/Documents/AlphaRush"))


def patch(path, pairs):
    p = ROOT / path
    s = p.read_text(encoding="utf-8")
    for old, new in pairs:
        assert s.count(old) == 1, (path, old[:80], s.count(old))
        s = s.replace(old, new)
    p.write_text(s, encoding="utf-8", newline="\n")


# 1. campaign.py: packages computed from the total earned stars (the "balanced" one is the incremental rule).
patch("alpharush_rl/campaign.py", [(
    '''def heroes_available(level: int) -> list[str]:''',
    '''# Allocation packages (the upgrades screen's reset refunds every star, so any package can be applied
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


def heroes_available(level: int) -> list[str]:'''),
    ('''def level_profile(level: int, stars_per_level: int = MAX_STARS_PER_LEVEL, hero: str | None = None) -> dict:''',
     '''def level_profile(level: int, stars_per_level: int = MAX_STARS_PER_LEVEL, hero: str | None = None,
                  package: str = "balanced") -> dict:'''),
    ('''        won[index] = min(3, low + 1)
    return campaign_profile(won, hero=hero)''', '''        won[index] = min(3, low + 1)
    return check_profile({**campaign_profile(won, hero=hero), "upgrades": package_upgrades(package, won)})'''),
])

# 2. search.py: optional plan parameter "pkg".
patch("alpharush_rl/search.py", [
    ('''OPTIONAL_GENOME = {"ratio": "3111", "boss": 0}''', '''OPTIONAL_GENOME = {"ratio": "3111", "boss": 0, "pkg": "balanced"}'''),
    ('''    boss = genome.get("boss", OPTIONAL_GENOME["boss"])
    if boss not in (0, 1) or isinstance(boss, bool):
        raise ValueError("boss must be 0 or 1")''', '''    boss = genome.get("boss", OPTIONAL_GENOME["boss"])
    if boss not in (0, 1) or isinstance(boss, bool):
        raise ValueError("boss must be 0 or 1")
    from .campaign import PACKAGES
    pkg = genome.get("pkg", OPTIONAL_GENOME["pkg"])
    if pkg not in PACKAGES:
        raise ValueError(f"pkg must be one of {sorted(PACKAGES)}")'''),
    ('''    if boss != OPTIONAL_GENOME["boss"]:
        extra["boss"] = boss''', '''    if boss != OPTIONAL_GENOME["boss"]:
        extra["boss"] = boss
    if pkg != OPTIONAL_GENOME["pkg"]:
        extra["pkg"] = pkg'''),
    ('''            which = rng.choice(("cast", "early", "branch", "hero", "ratio", "ratio", "boss"))''',
     '''            which = rng.choice(("cast", "early", "branch", "hero", "ratio", "ratio", "boss", "pkg", "pkg"))'''),
    ('''            elif which == "boss":
                g["boss"] = 1 - g.get("boss", OPTIONAL_GENOME["boss"])''', '''            elif which == "boss":
                g["boss"] = 1 - g.get("boss", OPTIONAL_GENOME["boss"])
            elif which == "pkg":
                from .campaign import PACKAGES
                g["pkg"] = rng.choice(sorted(PACKAGES))'''),
    ('''    child["boss"] = rng.choice((a.get("boss", OPTIONAL_GENOME["boss"]), b.get("boss", OPTIONAL_GENOME["boss"])))''',
     '''    child["boss"] = rng.choice((a.get("boss", OPTIONAL_GENOME["boss"]), b.get("boss", OPTIONAL_GENOME["boss"])))
    child["pkg"] = rng.choice((a.get("pkg", OPTIONAL_GENOME["pkg"]), b.get("pkg", OPTIONAL_GENOME["pkg"])))'''),
])

# 3. Every evaluation (search, collect, eval) uses the plan's package.
patch("alpharush_rl/search_job.py", [(
    '''    def profile_for(level, hero):
        return level_profile(level, stars, hero=hero)

    def make_task(purpose, level, seed, genome=None, result=None):
        counter[0] += 1
        hero = genome["hero"] if genome else None
        return {"purpose": purpose, "level": level, "seed": seed, "genome": genome, "result": result,
                "profile": profile_for(level, hero), "identity_index": counter[0],''',
    '''    def profile_for(level, hero, package="balanced"):
        return level_profile(level, stars, hero=hero, package=package)

    def make_task(purpose, level, seed, genome=None, result=None):
        counter[0] += 1
        hero = genome["hero"] if genome else None
        package = genome.get("pkg", "balanced") if genome else "balanced"
        return {"purpose": purpose, "level": level, "seed": seed, "genome": genome, "result": result,
                "profile": profile_for(level, hero, package), "identity_index": counter[0],''')])
patch("alpharush_rl/collect_job.py", [(
    '''            profile = level_profile(task["level"], spec["profile_stars_per_level"], hero=genome["hero"])''',
    '''            profile = level_profile(task["level"], spec["profile_stars_per_level"], hero=genome["hero"],
                                    package=genome.get("pkg", "balanced"))''')])
patch("alpharush_rl/eval_job.py", [(
    '''            profile = level_profile(task["level"], spec["profile_stars_per_level"], hero=genome["hero"])''',
    '''            profile = level_profile(task["level"], spec["profile_stars_per_level"], hero=genome["hero"],
                                    package=genome.get("pkg", "balanced"))''')])

# 4. Strategy brain: before each level, choose an allocation of all earned stars (reset is free).
patch("alpharush_rl/strategy_brain.py", [
    ('''from .campaign import BUY_PRIORITY, STAR_PRICES, STAR_TREES, check_upgrades, upgrades_cost''',
     '''from .campaign import (BUY_PRIORITY, PACKAGES, STAR_PRICES, STAR_TREES, check_upgrades, package_upgrades,
                       upgrades_cost)'''),
    ('''    def buy(self, stars, owned, context):
        from .campaign import buy_upgrades
        return buy_upgrades(stars, owned), []''', '''    def buy(self, stars, owned, context):
        """The allocation the chosen plan was tested with (the balanced rule unless the plan names one)."""
        return package_upgrades(context.get("plan_package", "balanced"), context["won"]), []'''),
    ('''    def buy(self, stars, owned, context):
        """One decision per level: which purchase package to apply to the unspent stars."""
        owned = check_upgrades(owned or {})
        left = stars - upgrades_cost(owned)
        packages = purchase_packages(stars, owned, context.get("plan_kinds") or {})
        if left <= 0 or len({json.dumps(p[1], sort_keys=True) for p in packages}) <= 1:
            return owned, []
        status = ", ".join(f"{TREE_NAME[t]} {owned[t]}/5" for t in STAR_TREES)
        plan = f" The chosen battle plan for it: {context['plan']}." if context.get("plan") else ""
        user = (f"Campaign so far: {context['progress']}. Next level: {context['next_level']}.{plan} "
                f"Stars earned: {stars}, unspent: {left}. Star upgrades owned: {status}. Upgrades are permanent "
                "and apply to every later level. Choose how to spend the unspent stars.")
        options = [text for text, _ in packages]
        index, record = self._ask("upgrade", user, options)
        return check_upgrades(packages[index][1]), [record]''',
     '''    def buy(self, stars, owned, context):
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
        return check_upgrades(packages[index][1]), [record]'''),
    ('''def purchase_packages(stars, owned, plan_kinds):''', '''PACKAGE_TEXT = {"balanced": "the balanced allocation", "rain": "Rain of Fire first",
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


def purchase_packages(stars, owned, plan_kinds):'''),
    ('''            + ("; saves spells for the boss" if genome.get("boss") else ""))''',
     '''            + ("; saves spells for the boss" if genome.get("boss") else "")
            + (f"; tested with star upgrades allocated {PACKAGE_TEXT[genome['pkg']]}" if genome.get("pkg") else ""))'''),
])

# 5. Campaign runner: the allocation is chosen before every level from all stars won so far.
patch("alpharush_rl/campaign_run.py", [(
    '''                buy_context = {"progress": _progress(won), "next_level": level,
                               "plan": describe_plan(genome, None), "plan_kinds": kinds}''',
    '''                buy_context = {"progress": _progress(won), "next_level": level, "won": dict(won),
                               "plan": describe_plan(genome, None), "plan_kinds": kinds,
                               "plan_package": genome.get("pkg", "balanced")}''')])
print("packages patched")

# 6. Reset is free, so every attempt (not only the first) starts from the allocation chosen for its plan.
patch("alpharush_rl/campaign_run.py", [(
    '''            if attempt == 0 and won:
                # Like a player on the map screen: look at the next level's plan, then spend the stars.''',
    '''            if won:
                # Like a player on the map screen: look at the plan, then (re)allocate all stars (the upgrades
                # screen's reset refunds every star, so each attempt may use the allocation its plan needs).''')])
print("reallocation per attempt patched")
