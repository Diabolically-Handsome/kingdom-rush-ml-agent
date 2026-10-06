"""Tests for star allocation packages (reset and reallocate before each attempt)."""
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


patch("tests/test_strategy_brain.py", [(
    '''    def test_language_brain_chooses_one_purchase_package(self):
        from alpharush_rl.campaign import buy_upgrades
        from alpharush_rl.strategy_brain import purchase_packages
        context = {"progress": "x", "next_level": 4, "plan_kinds": {"mage": 3, "archer": 1}}
        owned, records = LanguageBrain(FakeBroker(prefer=("Balanced",))).buy(9, {}, context)
        self.assertEqual(buy_upgrades(9), owned)  # the rule the plans were validated with
        self.assertEqual(1, len(records))
        owned, _ = LanguageBrain(FakeBroker(prefer=("Focus on Mage",))).buy(9, {}, context)
        self.assertEqual(5, owned["mages"])  # 1+1+2+2+3 = 9 stars, all on the plan's main tower kind
        owned, _ = LanguageBrain(FakeBroker(prefer=("Keep",))).buy(9, {}, context)
        self.assertEqual(0, upgrades_cost(owned))
        for stars in range(0, 30):
            for _, levels in purchase_packages(stars, {}, {"archer": 2}):
                self.assertLessEqual(upgrades_cost(levels), stars)
        self.assertEqual(([], ), (LanguageBrain(FakeBroker()).buy(9, buy_upgrades(9), context)[1], ))''',
    '''    def test_language_brain_chooses_one_allocation_of_all_stars(self):
        from alpharush_rl.campaign import PACKAGES, package_upgrades
        from alpharush_rl.strategy_brain import allocation_packages
        won = {1: 3, 2: 3, 3: 3}
        context = {"progress": "x", "next_level": 4, "won": won, "plan_kinds": {"mage": 3, "archer": 1}}
        owned, records = LanguageBrain(FakeBroker(prefer=("recommended",))).buy(9, {}, context)
        self.assertEqual(package_upgrades("balanced", won), owned)  # the allocation the plan was tested with
        self.assertEqual(1, len(records))
        owned, _ = LanguageBrain(FakeBroker(prefer=("Mage towers first",))).buy(9, {}, context)
        self.assertEqual(5, owned["mages"])  # 1+1+2+2+3 = 9 stars, all on the plan's main tower kind
        kept = {"archers": 1, "barracks": 0, "mages": 0, "engineers": 0, "rain": 0, "reinforcements": 0}
        owned, _ = LanguageBrain(FakeBroker(prefer=("Keep",))).buy(9, kept, context)
        self.assertEqual(kept, owned)
        # A reset may lower a tree: a rain plan after a mage-heavy allocation.
        rain = LanguageBrain(FakeBroker(prefer=("recommended",))).buy(
            9, package_upgrades("mages", won), {**context, "plan_package": "rain"})[0]
        self.assertEqual(package_upgrades("rain", won), rain)
        self.assertLess(rain["mages"], 5)
        options = allocation_packages(won, kept, "rain", {"archer": 2})
        self.assertIn("recommended", options[0][0])
        self.assertEqual(package_upgrades("rain", won), options[0][1])
        for total in range(0, 13):
            won_n = {i: 3 for i in range(1, total // 3 + 1)}
            if total % 3:
                won_n[total // 3 + 1] = total % 3
            for name in PACKAGES:
                levels = package_upgrades(name, won_n)
                self.assertLessEqual(upgrades_cost(levels), sum(won_n.values()))
            for _, levels in allocation_packages(won_n, {}, "balanced", {"archer": 2}):
                self.assertLessEqual(upgrades_cost(levels), sum(won_n.values()))'''),
    ('''        self.assertTrue(all(d["kind"] != "upgrade" or len(d["options"]) == 4 for d in decisions))''',
     '''        self.assertTrue(all(d["kind"] != "upgrade" or len(d["options"]) == 4 for d in decisions))
        self.assertTrue(all(d["kind"] != "upgrade" or "recommended" in d["options"][0] for d in decisions))'''),
])

patch("tests/test_campaign_profile.py", [(
    '''    def test_hero_must_be_unlocked_for_the_level(self):''',
    '''    def test_allocation_packages_spend_all_stars_on_their_trees_first(self):
        from alpharush_rl.campaign import PACKAGES, package_upgrades
        balanced = level_profile(9, 2.75)
        self.assertEqual(balanced, level_profile(9, 2.75, package="balanced"))
        rain = level_profile(9, 2.75, package="rain")
        self.assertEqual({"archers": 2, "barracks": 2, "mages": 2, "engineers": 1, "rain": 5, "reinforcements": 1},
                         rain["upgrades"])
        self.assertEqual(balanced["levels"], rain["levels"])
        for name in PACKAGES:
            upgrades = level_profile(12, 3, package=name)["upgrades"]
            self.assertLessEqual(upgrades_cost(upgrades), 33)
        self.assertEqual(5, package_upgrades("mages", {1: 3, 2: 3, 3: 3})["mages"])
        with self.assertRaises(ValueError):
            package_upgrades("wizards", {1: 3})

    def test_hero_must_be_unlocked_for_the_level(self):''')])

patch("tests/test_search.py", [(
    '''class ExecutorTests(''',
    '''class PackageGeneTests(unittest.TestCase):
    def test_pkg_is_an_optional_gene(self):
        import random
        from alpharush_rl.search import check_genome, level_holders, mutate
        plan = genome([["b", "01", "mage"]])
        self.assertNotIn("pkg", check_genome({**plan, "pkg": "balanced"}))
        self.assertEqual("rain", check_genome({**plan, "pkg": "rain"})["pkg"])
        with self.assertRaises(ValueError):
            check_genome({**plan, "pkg": "gold"})
        rng = random.Random(3)
        holders = level_holders(ToyEnv().reset())
        seen = set()
        for _ in range(400):
            child = check_genome(mutate(plan, holders, [], rng))
            seen.add(child.get("pkg", "balanced"))
        self.assertGreater(len(seen), 2)


class ExecutorTests(''')])

patch("tests/test_campaign_run.py", [(
    '''    def test_retries_offer_only_untried_plans(self):''',
    '''    def test_each_attempt_reallocates_the_stars_for_its_plan(self):
        from alpharush_rl.campaign import package_upgrades
        plans = {"1": [WIN_PLAN], "2": [{**LOSE_PLAN, "pkg": "mages"}, WIN_PLAN]}
        spec = self.spec(levels=[1, 2], seeds=[1001], plans=plans)
        made = []

        def factory(level, seed, profile, attempt, port):
            made.append((level, attempt, profile))
            return (FirstTryLoses if level == 2 and attempt == 0 else ToyEnv)(seed=seed, level=level, gold=600)
        ctx = self.context(10)
        summary = run_campaigns(spec, factory, ctx, Journal(ctx.output_dir / "episodes.jsonl"), protocol=PROTOCOL,
                                make_policy=BuildOrderPolicy, ports=[9001])
        self.assertTrue(summary["campaigns"][0]["completed"])
        (_, _, first), (_, _, focus), (_, _, balanced) = made
        won = {int(k): v for k, v in focus["levels"].items()}
        self.assertEqual(package_upgrades("mages", won), focus["upgrades"])  # the losing plan's tested allocation
        self.assertEqual(package_upgrades("balanced", won), balanced["upgrades"])  # reset for the retry's plan
        self.assertNotEqual(focus["upgrades"], balanced["upgrades"])

    def test_retries_offer_only_untried_plans(self):'''),
    ('''        self.assertEqual(buy_upgrades(stars), second["upgrades"])  # one victory so far
            self.assertEqual(entries[0][2], entries[1][2])  # a retry starts from the same save''',
     '''        self.assertEqual(buy_upgrades(stars), second["upgrades"])  # one victory so far
            self.assertEqual(entries[2][2], entries[3][2])  # a retry starts from the same save''')])
print("package tests patched")

patch("tests/test_search.py", [(
    '''    def test_multi_seed_fitness_and_reevaluated_warm_start(self):''',
    '''    def test_warm_start_packages_queue_reallocated_variants(self):
        first = "native-search-" + "1" * 32
        ctx = self.context(200, first)
        out = Journal(ctx.output_dir / "episodes.jsonl")
        search_job.search_loop(self.spec(validate_top=0, evaluations_per_level=3), self.factory()[0], ctx, out,
                               pools=POOLS, protocol=PROTOCOL, ports=[9001, 9002, 9003])
        second = "native-search-" + "2" * 32
        ctx2 = self.context(200, second)
        out2 = Journal(ctx2.output_dir / "episodes.jsonl")
        make, made = self.factory()
        spec = self.spec(validate_top=0, evaluations_per_level=2,
                         warm_start={"runs": [first], "top": 1, "packages": ["archers"]})
        search_job.search_loop(spec, make, ctx2, out2, pools=POOLS, protocol=PROTOCOL, ports=[9001, 9002, 9003],
                               runs_dir=self.root / "runs")
        variants = [row["payload"] for row in out2.entries() if row["kind"] == "search_variant"]
        self.assertEqual([1, 2], sorted(v["level"] for v in variants))
        self.assertTrue(all(v["genome"]["pkg"] == "archers" for v in variants))
        evaluated = {row["payload"]["genome_id"] for row in out2.entries() if row["kind"] == "search_eval"}
        self.assertTrue(all(v["genome_id"] in evaluated for v in variants))  # queued before new proposals
        # Level 2 (2 stars from level 1): the variant was played with both stars on archers.
        self.assertIn(2, [upgrades["archers"] for purpose, level, _, _, upgrades in made
                          if purpose == "search" and level == 2])
        _, issues = search_job.check_search(self.spec(warm_start={"runs": [first], "top": 1, "packages": ["gold"]}),
                                            POOLS)
        self.assertTrue(issues)

    def test_multi_seed_fitness_and_reevaluated_warm_start(self):''')])
print("warm package test patched")
