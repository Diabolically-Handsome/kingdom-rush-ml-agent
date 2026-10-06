"""The strategy model issuing steps online with the operator executing them; toy level, fake scorer."""
from __future__ import annotations

import math
import unittest

from alpharush_rl.episode import run_episode
from alpharush_rl.llm_steps import LanguageStepSource, StepOperatorPolicy, step_candidates
from alpharush_rl.operator_net import OptionScorer, Recorder, macro_summary, train
from alpharush_rl.search import BuildOrderPolicy

from test_search import PROTOCOL, ToyEnv, genome


class ScriptedBroker:
    """Answers with the option whose text contains the first matching wanted phrase, else option A."""

    def __init__(self, wanted):
        self.wanted, self.requests = list(wanted), []

    def distribution(self, request):
        self.requests.append(request)
        lines = [line for line in request["user"].splitlines() if line.split(".", 1)[0] in request["labels"]]
        pick = 0
        for phrase in self.wanted:
            hits = [i for i, line in enumerate(lines) if phrase in line]
            if hits:
                pick = hits[0]
                self.wanted.remove(phrase)
                break
        p = [1e-3] * len(lines)
        p[pick] = 1.0
        total = sum(p)
        p = [x / total for x in p]
        return {"id": request["id"], "labels": request["labels"], "p": p, "logp": [math.log(x) for x in p],
                "choice": request["labels"][pick], "model": "scripted"}


class LanguageStepTests(unittest.TestCase):
    def test_candidates_cover_builds_upgrades_and_the_sunray(self):
        state = ToyEnv(gold=600, sunray=True).reset()
        options = step_candidates(macro_summary(state, [], 0, None, None))
        steps = [s for s, _ in options]
        self.assertEqual(4 * 4 + 1, len(steps))  # four empty slots x four kinds, plus the sunray beam
        self.assertIn(["b", "02", "mage"], steps)
        self.assertIn(["k", "71"], steps)

    def test_operator_executes_the_models_steps(self):
        plan = genome([["b", "02", "mage"], ["u", "02"], ["b", "01", "archer"]])
        rows = []
        for seed in (1001, 1002, 1003):
            recorder = Recorder(BuildOrderPolicy(plan))
            run_episode(ToyEnv(seed=seed, gold=600), recorder, PROTOCOL, seed=seed, level=1)
            rows += recorder.rows
        net = OptionScorer(hidden=(32,), seed=0)
        train(net, [(g, o, t, 1.0) for g, o, t in rows], epochs=40, batch=32, lr=3e-3)
        broker = ScriptedBroker(["build mage tower at slot 02", "upgrade the mage tower at slot 02",
                                 "build archer tower at slot 01"])
        source = LanguageStepSource(broker, cast=50, early=1, max_steps=3)
        result = run_episode(ToyEnv(seed=1004, gold=600), StepOperatorPolicy(net, source), PROTOCOL, seed=1004, level=1)
        builds = [d["action"] for d in result["decisions"] if d["action"]["action"] in ("build_tower", "upgrade_tower")]
        self.assertEqual({"action": "build_tower", "holder_id": 2, "tower_type": "mage"}, builds[0])
        self.assertEqual(["b", "02", "mage"], source.records[0]["step"])
        self.assertLessEqual(len(broker.requests), 3)  # bounded number of strategy calls
        self.assertTrue(all(d["provenance"] == "model:8b-steps+operator" for d in result["decisions"]))
        self.assertEqual("terminal", result["status"])


if __name__ == "__main__":
    unittest.main()
