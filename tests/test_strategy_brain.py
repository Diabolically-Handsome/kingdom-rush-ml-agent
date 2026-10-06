"""Strategy brains on the toy campaign with a fake scoring server; no model or game is started."""
from __future__ import annotations

import math
import time
import tempfile
import unittest
from pathlib import Path

from alpharush_rl import phase
from alpharush_rl.campaign import STAR_PRICES, upgrades_cost
from alpharush_rl.campaign_run import run_campaigns
from alpharush_rl.journal import Journal
from alpharush_rl.search import BuildOrderPolicy
from alpharush_rl.strategy_brain import LanguageBrain, RuleBrain, describe_plan

from test_campaign_run import FirstTryLoses, LOSE_PLAN, WIN_PLAN
from test_search import PROTOCOL, ToyEnv


class FakeBroker:
    """Picks the option whose text contains a wanted word, else the last option; records requests."""

    def __init__(self, prefer=("Mage", "won")):
        self.prefer, self.requests = prefer, []

    def distribution(self, request):
        self.requests.append(request)
        lines = [line for line in request["user"].splitlines() if line[:2] in {f"{l}." for l in request["labels"]}]
        pick = len(lines) - 1
        for word in self.prefer:
            hits = [i for i, line in enumerate(lines) if word in line]
            if hits:
                pick = hits[0]
                break
        p = [0.1 / max(1, len(lines) - 1)] * len(lines)
        p[pick] = 0.9 if len(lines) > 1 else 1.0
        total = sum(p)
        p = [x / total for x in p]
        return {"id": request["id"], "labels": request["labels"], "p": p, "logp": [math.log(x) for x in p],
                "choice": request["labels"][pick], "model": "fake"}


class BrainTests(unittest.TestCase):
    def test_language_brain_chooses_one_allocation_of_all_stars(self):
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
                self.assertLessEqual(upgrades_cost(levels), sum(won_n.values()))

    def test_choices_are_averaged_over_menu_orders(self):
        class Biased:
            """Always 0.6 on label B, the rest spread evenly; option 'good' gets +0.2 wherever it is."""
            def __init__(self):
                self.requests = []

            def distribution(self, request):
                self.requests.append(request)
                lines = [l for l in request["user"].splitlines() if l[:2] in {f"{x}." for x in request["labels"]}]
                n = len(lines)
                p = [0.4 / (n - 1)] * n
                p[1] = 0.6
                good = next(i for i, line in enumerate(lines) if "good" in line)
                p = [x + (0.2 if i == good else 0.0) for i, x in enumerate(p)]
                p = [x / sum(p) for x in p]
                return {"p": p, "choice": request["labels"][max(range(n), key=p.__getitem__)], "model": "biased"}
        broker = Biased()
        index, record = LanguageBrain(broker)._ask("plan", "pick", ["bad one", "bad two", "good", "bad three"])
        self.assertEqual(2, index)  # a single unrotated request would have answered B ("bad two")
        self.assertEqual(4, len(broker.requests))
        self.assertEqual(4, record["rotations"])
        self.assertAlmostEqual(1.0, sum(record["p"]))
        self.assertEqual(len(set(record["prompt_sha256s"])), 4)

    def test_plan_choice_and_descriptions(self):
        broker = FakeBroker(prefer=("won 3 of 3",))
        candidates = [(LOSE_PLAN, "won 0 of 3 practice games"), (WIN_PLAN, "won 3 of 3 practice games")]
        index, records = LanguageBrain(broker).choose_plan(4, candidates, 0, {"progress": "p", "upgrades": {}})
        self.assertEqual(1, index)
        self.assertIn("practice results: won 3 of 3", describe_plan(*candidates[1]))
        self.assertEqual((0, []), RuleBrain().choose_plan(4, candidates, 1, {}))  # the runner orders the offers

    def test_campaign_with_language_brain_journals_every_strategic_choice(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        root = Path(tmp.name)
        run_dir = root / "runs" / "native-campaign-0001"
        run_dir.mkdir(parents=True)
        ctx = phase.PhaseRunContext("native-campaign-0001", run_dir, root, time.monotonic() + 300, 50,
                                    job_kind="native-campaign")
        out = Journal(run_dir / "episodes.jsonl")
        plans = {str(level): [{"genome": LOSE_PLAN, "evidence": "won 0 of 3 practice games"},
                              {"genome": WIN_PLAN, "evidence": "won 3 of 3 practice games"}] for level in (1, 2, 3)}
        spec = {"levels": [1, 2, 3], "seeds": [1001], "attempts_per_level": 2, "plans": plans, "policy": "plan",
                "brain": "8b"}
        broker = FakeBroker(prefer=("won 3 of 3", "Mage"))
        profiles = []

        def factory(level, seed, profile, attempt, port):
            profiles.append((level, profile))
            return ToyEnv(seed=seed, level=level, gold=600)
        summary = run_campaigns(spec, factory, ctx, out, protocol=PROTOCOL, make_policy=BuildOrderPolicy,
                                ports=[9001], brain=LanguageBrain(broker))
        campaign = summary["campaigns"][0]
        self.assertTrue(campaign["completed"])
        self.assertEqual("8b", campaign["brain"])
        decisions = [row["payload"] for row in out.entries() if row["kind"] == "strategy_decision"]
        self.assertEqual({"plan", "upgrade"}, {d["kind"] for d in decisions})
        self.assertTrue(all(d["kind"] != "upgrade" or len(d["options"]) == 4 for d in decisions))
        self.assertTrue(all(d["kind"] != "upgrade" or "recommended" in d["options"][0] for d in decisions))
        self.assertEqual(3, sum(d["kind"] == "plan" for d in decisions))
        # Level 2 starts with level 1's stars spent on mages, bought before it (after its plan was chosen).
        level2 = next(profile for level, profile in profiles if level == 2)
        self.assertGreater(level2["upgrades"]["mages"], 0)
        self.assertLessEqual(upgrades_cost(level2["upgrades"]), sum(level2["levels"].values()))
        self.assertTrue(all(STAR_PRICES))


if __name__ == "__main__":
    unittest.main()
