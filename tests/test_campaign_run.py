"""Campaign runner and demonstration collection on the toy level; no game is started."""
from __future__ import annotations

from pathlib import Path
import tempfile
import time
import unittest

from alpharush_rl import collect_job, phase
from alpharush_rl.campaign import buy_upgrades
from alpharush_rl.campaign_run import check_campaign, run_campaigns
from alpharush_rl.journal import Journal
from alpharush_rl.operator_net import load_rows
from alpharush_rl.search import BuildOrderPolicy

from test_search import POOLS, PROTOCOL, ToyEnv, genome

WIN_PLAN = genome([["b", "01", "mage"], ["b", "02", "archer"], ["u", "01"], ["b", "03", "archer"]])
LOSE_PLAN = genome([["b", "04", "barrack"]])


class WeakToy(ToyEnv):
    """A toy level no plan can win (huge enemies): on level 3, or whenever ``always``."""
    always = False

    def _spawn(self, s):
        super()._spawn(s)
        if self.level == 3 or self.always:
            for enemy in s["enemies"]:
                enemy["hp"] = 10 ** 6


class FirstTryLoses(WeakToy):
    always = True


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="alpharush-campaign-")
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def context(self, max_games, kind="native-campaign"):
        run_dir = self.root / "runs" / f"{kind}-0001"
        run_dir.mkdir(parents=True)
        return phase.PhaseRunContext(f"{kind}-0001", run_dir, self.root, time.monotonic() + 300, max_games,
                                     job_kind=kind)


class CampaignTests(Base):
    def spec(self, **overrides):
        plans = {str(level): [LOSE_PLAN, WIN_PLAN] for level in (1, 2, 3)}
        return {"levels": [1, 2, 3], "seeds": [1001, 1002], "attempts_per_level": 2, "plans": plans,
                "policy": "plan", "brain": "rule", **overrides}

    def run_spec(self, spec, toy=ToyEnv, max_games=50, first_try_loses=False):
        made = []

        def factory(level, seed, profile, attempt, port):
            made.append((level, seed, attempt, port, profile))
            cls = FirstTryLoses if first_try_loses and attempt == 0 else toy
            return cls(seed=seed, level=level, gold=600)
        ctx = self.context(max_games)
        out = Journal(ctx.output_dir / "episodes.jsonl")
        summary = run_campaigns(spec, factory, ctx, out, protocol=PROTOCOL, make_policy=BuildOrderPolicy,
                                ports=[9001, 9002])
        return summary, made, out, ctx

    def test_campaign_buys_upgrades_after_each_victory_and_retries(self):
        summary, made, out, ctx = self.run_spec(self.spec(), first_try_loses=True)
        self.assertEqual(2, summary["completed"])
        for campaign in summary["campaigns"]:
            self.assertTrue(campaign["completed"])
            self.assertEqual(["1", "2", "3"], sorted(campaign["won"]))
            self.assertEqual(6, len(campaign["attempts"]))  # each level: a lost attempt, then the winning plan
        by_seed = {}
        for level, seed, attempt, port, profile in made:
            by_seed.setdefault(seed, []).append((level, attempt, profile))
        for seed, entries in by_seed.items():
            first, second = entries[0][2], entries[2][2]
            self.assertEqual({}, first["levels"])
            stars = sum(second["levels"].values())
            self.assertEqual(buy_upgrades(stars), second["upgrades"])  # one victory so far
            self.assertEqual(entries[2][2], entries[3][2])  # a retry starts from the same save
        self.assertEqual(ctx.games_played, len(made))
        kinds = [row["kind"] for row in out.entries()]
        self.assertEqual(12, kinds.count("campaign_attempt"))

    def test_each_attempt_reallocates_the_stars_for_its_plan(self):
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

    def test_retries_offer_only_untried_plans(self):
        seen = []

        class Recording:
            name = "recording"

            def choose_plan(self, level, candidates, attempt, context):
                seen.append((level, attempt, len(candidates)))
                return 0, []

            def buy(self, stars, owned, context):
                from alpharush_rl.campaign import buy_upgrades
                return buy_upgrades(stars, owned), []
        plans = {str(level): [LOSE_PLAN, LOSE_PLAN, WIN_PLAN] for level in (1, 2, 3)}
        spec = self.spec(seeds=[1001], attempts_per_level=3, plans=plans)

        def factory(level, seed, profile, attempt, port):
            return (FirstTryLoses if attempt < 2 else ToyEnv)(seed=seed, level=level, gold=600)
        ctx = self.context(30)
        summary = run_campaigns(spec, factory, ctx, Journal(ctx.output_dir / "episodes.jsonl"), protocol=PROTOCOL,
                                make_policy=BuildOrderPolicy, ports=[9001], brain=Recording())
        self.assertTrue(summary["campaigns"][0]["completed"])
        self.assertEqual([(1, 0, 3), (1, 1, 2), (1, 2, 1)], seen[:3])  # 3 plans, then 2 untried, then 1

    def test_an_unwinnable_level_ends_the_campaign(self):
        summary, made, _, _ = self.run_spec(self.spec(seeds=[1001]), toy=WeakToy)
        campaign = summary["campaigns"][0]
        self.assertFalse(campaign["completed"])
        self.assertEqual(3, campaign["failed_level"])
        self.assertEqual("level_failed", campaign["stopped_reason"])
        self.assertEqual(0, summary["completed"])
        self.assertIsNone(summary["stopped_reason"])

    def test_spec_checks(self):
        self.assertEqual([], check_campaign(self.spec(), POOLS, phase)[1])
        bad = [self.spec(levels=[2, 3]), self.spec(seeds=[1001, 5001]), self.spec(seeds=[42]),
               self.spec(attempts_per_level=0), self.spec(policy="magic"), self.spec(brain="oracle"),
               self.spec(plans={"1": [WIN_PLAN], "2": [WIN_PLAN]}),
               self.spec(plans={str(l): [{**WIN_PLAN, "hero": "hero_gerald"}] for l in (1, 2, 3)})]
        for spec in bad:
            self.assertIsNone(check_campaign(spec, POOLS, phase)[0])


class CollectTests(Base):
    def test_collect_writes_one_npz_per_episode_on_train_seeds_only(self):
        spec = {"tasks": [{"level": 1, "seed": seed, "genome": WIN_PLAN} for seed in (1001, 1002, 1003)],
                "workers": 2, "profile_stars_per_level": 2}
        ctx = self.context(10, "native-collect")
        out = Journal(ctx.output_dir / "episodes.jsonl")
        data = ctx.output_dir / "data"
        summary = collect_job.collect_loop(spec, lambda task, profile, index, port: ToyEnv(seed=task["seed"], gold=600),
                                           ctx, out, data, pools=POOLS, protocol=PROTOCOL, ports=[9001, 9002])
        self.assertEqual(3, summary["episodes"])
        rows = [row for row in out.entries() if row["kind"] == "collect_episode"]
        self.assertEqual(3, len(rows))
        for row in rows:
            decisions, meta = load_rows(data / row["payload"]["data"])
            self.assertEqual(row["payload"]["seed"], meta["seed"])
            self.assertTrue(decisions)
        bad = {**spec, "tasks": [{"level": 1, "seed": 5001, "genome": WIN_PLAN}]}
        self.assertIsNone(collect_job.check_collect(bad, POOLS)[0])
        dup = {**spec, "tasks": spec["tasks"] + spec["tasks"][:1]}
        self.assertIsNone(collect_job.check_collect(dup, POOLS)[0])
        self.assertEqual([], collect_job.check_collect({**spec, "profile_stars_per_level": 2.75}, POOLS)[1])
        for rate in (0.5, 3.5, True, "2"):
            self.assertIsNone(collect_job.check_collect({**spec, "profile_stars_per_level": rate}, POOLS)[0])



class EvalTests(Base):
    def test_plan_and_operator_players_on_train_and_evaluation_seeds(self):
        from alpharush_rl import eval_job
        from alpharush_rl.operator_net import OperatorPolicy, OptionScorer
        net = OptionScorer(hidden=(8,), seed=0)
        spec = {"tasks": [{"level": 1, "seed": seed, "genome": WIN_PLAN} for seed in (1001, 5001)], "workers": 2,
                "profile_stars_per_level": 2, "players": ["plan", "operator"]}
        ctx = self.context(10, "native-eval")
        out = Journal(ctx.output_dir / "episodes.jsonl")
        made = []

        def factory(task, profile, index, player, port):
            made.append((player, task["seed"], port))
            return ToyEnv(seed=task["seed"], gold=600)
        summary = eval_job.eval_loop(spec, factory, ctx, out, pools=POOLS, protocol=PROTOCOL, ports=[9001, 9002],
                                     make_operator=lambda genome: OperatorPolicy(net, genome))
        self.assertEqual(4, ctx.games_played)
        self.assertEqual({"plan", "operator"}, set(summary["results"]))
        self.assertEqual(2, summary["results"]["plan"]["1"]["games"])
        self.assertEqual(2, summary["results"]["plan"]["1"]["wins"])
        rows = [row["payload"] for row in out.entries() if row["kind"] == "eval_episode"]
        self.assertEqual({"plan", "operator"}, {row["player"] for row in rows})
        bad = {**spec, "tasks": [{"level": 1, "seed": 6001, "genome": WIN_PLAN}]}
        self.assertIsNone(eval_job.check_eval(bad, POOLS)[0])
        self.assertEqual([], eval_job.check_eval({**spec, "profile_stars_per_level": 2.75}, POOLS)[1])
        self.assertIsNone(eval_job.check_eval({**spec, "profile_stars_per_level": 4}, POOLS)[0])
        # The online-steps player needs its own factory; it plays like the others when given one.
        from alpharush_rl.llm_steps import LanguageStepSource, StepOperatorPolicy
        from test_llm_steps import ScriptedBroker
        steps_spec = {**spec, "players": ["llm_steps"], "tasks": spec["tasks"][:1]}
        with self.assertRaises(Exception):
            eval_job.eval_loop(steps_spec, factory, self.context(10, "native-eval3"), out, pools=POOLS,
                               protocol=PROTOCOL, ports=[9001, 9002])
        ctx3 = self.context(10, "native-eval4")
        summary3 = eval_job.eval_loop(
            steps_spec, factory, ctx3, Journal(ctx3.output_dir / "episodes.jsonl"), pools=POOLS, protocol=PROTOCOL,
            ports=[9001, 9002], make_steps=lambda g: StepOperatorPolicy(net, LanguageStepSource(
                ScriptedBroker(["build mage tower at slot 02"]), cast=g["cast"], early=g["early"],
                branches=g["branches"])))
        self.assertEqual(1, summary3["results"]["llm_steps"]["1"]["games"])
        with self.assertRaises(Exception):
            eval_job.eval_loop(spec, factory, self.context(10, "native-eval2"), out, pools=POOLS, protocol=PROTOCOL,
                               ports=[9001, 9002])


if __name__ == "__main__":
    unittest.main()
