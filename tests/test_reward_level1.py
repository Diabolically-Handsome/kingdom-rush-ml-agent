import copy
import json
from pathlib import Path
import unittest

from alpharush_rl.journal import sha256_data
from alpharush_rl.reward_level1 import default_contract, rescore_dataset, score_level1_outcome, validate_reward_contract
from alpharush_rl.trainer import EligibilityError


def outcome(won=True, lives=9, **extra):
    return {"source": "native", "terminal": True, "level": 1, "difficulty": 2,
            "seed": 1002, "level_won": won, "level_lost": not won, "lives": lives, **extra}


class FirstLevelRewardTests(unittest.TestCase):
    def setUp(self):
        self.contract = default_contract()

    def test_contract_file_matches_frozen_implementation(self):
        path = Path(__file__).resolve().parents[1] / "configs/scoring-level1-terminal-v2.json"
        contract = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(validate_reward_contract(contract), sha256_data(self.contract))

    def test_all_life_counts_have_correct_order_and_bounded_return(self):
        returns = [score_level1_outcome(outcome(lives=lives), self.contract) for lives in range(1, 21)]
        self.assertEqual(returns[0], 1.05)
        self.assertEqual(returns[-1], 2.0)
        self.assertEqual(returns[8], 1.45)
        self.assertTrue(all(left < right for left, right in zip(returns, returns[1:])))
        self.assertGreater(min(returns), score_level1_outcome(outcome(won=False, lives=0), self.contract))

    def test_contradictory_native_outcomes_are_rejected_without_clipping(self):
        with self.assertRaises(EligibilityError):
            score_level1_outcome(outcome(won=True, lives=0), self.contract)
        for lives in range(1, 21):
            with self.subTest(lives=lives), self.assertRaises(EligibilityError):
                score_level1_outcome(outcome(won=False, lives=lives), self.contract)
        enriched = outcome(lives=9, native_raw=outcome(won=True, lives=0))
        with self.assertRaises(EligibilityError):
            score_level1_outcome(enriched, self.contract)

    def test_money_waiting_speed_stars_and_tower_counts_do_not_change_reward(self):
        for won in (True, False):
            lives = 9 if won else 0
            early = outcome(won=won, lives=lives, gold=0, tick=1, seconds=0, stars=0, wave=0, tower_count=0, action_count=1)
            late_rich = outcome(won=won, lives=lives, gold=10**12, tick=10**12, seconds=10**9,
                                stars=3, wave=999, tower_count=999, action_count=10**9)
            self.assertEqual(score_level1_outcome(early, self.contract), score_level1_outcome(late_rich, self.contract))

    def test_truncated_engineering_and_unfinished_rows_never_get_zero_reward(self):
        invalids = [{"terminal": False}, {"source": "synthetic"}, {"level_won": False, "level_lost": False},
                    {"level_won": True, "level_lost": True}, {"error": "connection lost"}]
        invalids += [{flag: True} for flag in self.contract["engineering_invalid_flags"]]
        for fields in invalids:
            with self.subTest(fields=fields), self.assertRaises(EligibilityError):
                score_level1_outcome(outcome(**fields), self.contract)

    def test_scope_lives_and_boolean_zero_validation(self):
        invalids = [{"level": 2}, {"difficulty": 1}, {"mode": "iron"}, {"lives": True},
                    {"lives": -1}, {"lives": 21}, {"lives": 9.5}, {"lives": float("nan")},
                    {"lives": float("inf")}, {"level": True}]
        for fields in invalids:
            with self.subTest(fields=fields), self.assertRaises(EligibilityError):
                score_level1_outcome(outcome(**fields), self.contract)
        missing = outcome()
        del missing["difficulty"]
        with self.assertRaises(EligibilityError):
            score_level1_outcome(missing, self.contract)
        self.assertEqual(score_level1_outcome(outcome(won=False, lives=0), self.contract), -1.0)

    def test_preserved_native_raw_cannot_be_changed_to_raise_reward(self):
        enriched = outcome(native_raw={"source": "native", "terminal": True,
                                      "level_won": True, "level_lost": False, "lives": 9})
        self.assertEqual(score_level1_outcome(enriched, self.contract), 1.45)
        enriched["lives"] = 20
        with self.assertRaises(EligibilityError):
            score_level1_outcome(enriched, self.contract)

    def test_same_version_name_cannot_hide_changed_weights(self):
        changed = default_contract()
        changed["defeat_return"] = 0
        with self.assertRaises(EligibilityError):
            score_level1_outcome(outcome(won=False, lives=0), changed)

    def test_rescoring_creates_new_dataset_and_preserves_raw_evidence(self):
        registry = {"pools": {"train": {"levels": [1], "seeds": [1002]},
                              "validation": {"levels": [2], "seeds": [2001]},
                              "heldout": {"levels": [3], "seeds": [3001]}}}
        candidates = [{"label": label, "return": reward, "native_outcome": native,
                       "receipt_verified": True, "replay_verified": True}
                      for label, reward, native in (("A", -1.0, outcome(won=False, lives=0)),
                                                   ("B", 1.09, outcome(lives=9)))]
        data = {"source": "real_game", "pool_registry": registry, "scorer_sha256": "a" * 64,
                "groups": [{"fork_id": "level1-1002", "pool": "train", "level": 1, "seed": 1002,
                            "menu": [{"label": "A"}, {"label": "B"}], "candidates": candidates}], "anchors": {}}
        before = copy.deepcopy(data)
        revised = rescore_dataset(data, self.contract)
        self.assertEqual(data, before)
        self.assertEqual([row["return"] for row in revised["groups"][0]["candidates"]], [-1.0, 1.45])
        self.assertNotEqual(revised["scorer_sha256"], data["scorer_sha256"])
        for old, new in zip(data["groups"][0]["candidates"], revised["groups"][0]["candidates"]):
            self.assertEqual(old["native_outcome"], new["native_outcome"])
        self.assertEqual(revised["reward_revision"]["source_dataset_sha256"], sha256_data(data))
        data["groups"][0]["candidates"][0]["replay_verified"] = False
        with self.assertRaises(EligibilityError):
            rescore_dataset(data, self.contract)


if __name__ == "__main__":
    unittest.main()
