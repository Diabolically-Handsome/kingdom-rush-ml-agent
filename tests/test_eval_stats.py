"""Paired evaluation statistics against hand-computed and brute-force values."""
from __future__ import annotations

import itertools
import math
import random
import unittest

from alpharush_rl.eval_stats import (bootstrap_ci, minimum_detectable_effect, paired_differences, required_pairs,
                                     sign_flip_pvalue, summarize_paired, wilson_interval)
from alpharush_rl.journal import canonical_json


def brute_force_p(diffs):
    """Independent full 2**n enumeration of the two-sided sign-flip test."""
    observed = abs(math.fsum(diffs))
    tolerance = 1e-9 * math.fsum(abs(d) for d in diffs)
    hits = sum(1 for signs in itertools.product((1, -1), repeat=len(diffs))
               if abs(math.fsum(s * d for s, d in zip(signs, diffs))) >= observed - tolerance)
    return hits / 2 ** len(diffs)


class PairedDifferenceTests(unittest.TestCase):
    def test_sorted_keys_and_direction(self):
        a = {(2, 5001): 1.0, (1, 5002): 0.5, (1, 5001): 0}
        b = {(1, 5001): 1, (2, 5001): 0.25, (1, 5002): 0.5}
        self.assertEqual(paired_differences(a, b), [((1, 5001), -1.0), ((1, 5002), 0.0), ((2, 5001), 0.75)])
        self.assertEqual(paired_differences(b, a), [((1, 5001), 1.0), ((1, 5002), 0.0), ((2, 5001), -0.75)])

    def test_key_sets_must_match(self):
        with self.assertRaises(ValueError):
            paired_differences({"x": 1, "y": 2}, {"x": 1})
        with self.assertRaises(ValueError):
            paired_differences({"x": 1}, {"x": 1, "z": 0})

    def test_values_must_be_finite_numbers(self):
        for bad in (float("nan"), float("inf"), True, "1", None):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                paired_differences({"x": bad}, {"x": 0})

    def test_mixed_key_types_still_sort_deterministically(self):
        a, b = {1: 2.0, "1": 3.0}, {1: 1.0, "1": 1.0}
        self.assertEqual(paired_differences(a, b), paired_differences(dict(reversed(a.items())), b))


class SignFlipTests(unittest.TestCase):
    def test_known_exact_values(self):
        self.assertEqual(sign_flip_pvalue([1, 1, 1]), 0.25)
        self.assertEqual(sign_flip_pvalue([1, 2, 3]), 0.25)
        self.assertEqual(sign_flip_pvalue([3, 1]), 0.5)
        self.assertEqual(sign_flip_pvalue([1] * 5), 2 / 32)
        self.assertEqual(sign_flip_pvalue([1, -1]), 1.0)
        self.assertEqual(sign_flip_pvalue([2]), 1.0)
        self.assertEqual(sign_flip_pvalue([1] * 20), 2 / 2 ** 20)

    def test_all_zero_and_zero_padding(self):
        self.assertEqual(sign_flip_pvalue([0, 0, 0]), 1.0)
        self.assertEqual(sign_flip_pvalue([0.0] * 50), 1.0)
        self.assertEqual(sign_flip_pvalue([0, 0, 1, 1, 1]), 0.25)  # zeros never move the sum

    def test_matches_brute_force_including_float_ties(self):
        self.assertEqual(sign_flip_pvalue([0.1, 0.2, -0.3]), 1.0)
        rng = random.Random(3)
        for _ in range(40):
            diffs = [rng.choice([-2, -1, -0.5, 0.1, 0.2, 0.3, 1, 2.5]) for _ in range(rng.randint(1, 10))]
            with self.subTest(diffs=diffs):
                self.assertAlmostEqual(sign_flip_pvalue(diffs), brute_force_p(diffs), places=12)

    def test_symmetry(self):
        diffs = [0.5, -1.25, 2.0, 3.0, 0.75, -0.25]
        self.assertEqual(sign_flip_pvalue(diffs), sign_flip_pvalue([-d for d in diffs]))
        self.assertEqual(sign_flip_pvalue(diffs), sign_flip_pvalue(list(reversed(diffs))))
        a = {i: float(i * i % 7) for i in range(9)}
        b = {i: float(i % 4) for i in range(9)}
        self.assertEqual(sign_flip_pvalue([d for _, d in paired_differences(a, b)]),
                         sign_flip_pvalue([d for _, d in paired_differences(b, a)]))
        self.assertEqual(sign_flip_pvalue(diffs, exact_max_n=0, samples=500, seed=4),
                         sign_flip_pvalue([-d for d in diffs], exact_max_n=0, samples=500, seed=4))

    def test_monte_carlo_is_seeded_and_close(self):
        p = sign_flip_pvalue([1, 1, 1], exact_max_n=0, samples=20000, seed=11)
        self.assertLess(abs(p - 0.25), 0.02)
        self.assertEqual(p, sign_flip_pvalue([1, 1, 1], exact_max_n=0, samples=20000, seed=11))
        # 30 nonzero pairs exceed exact_max_n: Monte Carlo never reports p = 0.
        tiny = sign_flip_pvalue([1] * 30, samples=2000, seed=0)
        self.assertEqual(tiny, 1 / 2001)
        self.assertEqual(sign_flip_pvalue([1] * 30, exact_max_n=0, samples=2000, seed=0), tiny)

    def test_argument_validation(self):
        for kwargs in ({"exact_max_n": -1}, {"exact_max_n": 25}, {"exact_max_n": 2.0},
                       {"exact_max_n": 0, "samples": 0}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                sign_flip_pvalue([1, 2], **kwargs)
        for bad in ([], [float("nan")], [True]):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                sign_flip_pvalue(bad)


class BootstrapTests(unittest.TestCase):
    def test_constant_and_ordering(self):
        self.assertEqual(bootstrap_ci([0.5] * 7, samples=200), (0.5, 0.5))
        diffs = [0.1, 0.4, -0.2, 0.9, 0.3, 0.0, 0.6, 0.2]
        low, high = bootstrap_ci(diffs, samples=4000, seed=5)
        self.assertLess(low, sum(diffs) / len(diffs))
        self.assertGreater(high, sum(diffs) / len(diffs))
        self.assertGreaterEqual(low, min(diffs))
        self.assertLessEqual(high, max(diffs))
        self.assertEqual((low, high), bootstrap_ci(diffs, samples=4000, seed=5))
        narrow = bootstrap_ci(diffs, alpha=0.5, samples=4000, seed=5)
        self.assertTrue(low <= narrow[0] <= narrow[1] <= high)

    def test_negation_symmetry(self):
        diffs = [0.1, 0.4, -0.2, 0.9, 0.3]
        low, high = bootstrap_ci(diffs, samples=1000, seed=2)
        neg_low, neg_high = bootstrap_ci([-d for d in diffs], samples=1000, seed=2)
        self.assertAlmostEqual(neg_low, -high, places=12)
        self.assertAlmostEqual(neg_high, -low, places=12)

    def test_validation(self):
        for kwargs in ({"alpha": 0}, {"alpha": 1}, {"samples": 0}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                bootstrap_ci([1.0, 2.0], **kwargs)
        with self.assertRaises(ValueError):
            bootstrap_ci([])


class WilsonTests(unittest.TestCase):
    def test_known_value_and_bounds(self):
        low, high = wilson_interval(5, 10, z=1.96)
        self.assertAlmostEqual(low, 0.236593, places=5)
        self.assertAlmostEqual(high, 0.763407, places=5)
        self.assertEqual(wilson_interval(0, 10)[0], 0.0)
        self.assertEqual(wilson_interval(10, 10)[1], 1.0)
        self.assertEqual(wilson_interval(0, 0), (0.0, 1.0))
        self.assertAlmostEqual(wilson_interval(0, 10)[1], 0.277533, places=5)  # 3.84/13.84 at z=1.96

    def test_mirror_and_containment(self):
        for n in (1, 2, 7, 40):
            for k in range(n + 1):
                low, high = wilson_interval(k, n)
                mirror = wilson_interval(n - k, n)
                self.assertAlmostEqual(low, 1 - mirror[1], places=12)
                self.assertTrue(0.0 <= low <= k / n <= high <= 1.0)
        self.assertLess(wilson_interval(3, 10, alpha=0.2)[1], wilson_interval(3, 10)[1])

    def test_validation(self):
        for args in ((11, 10), (-1, 10), (1, -1), (True, 3), (1.0, 3)):
            with self.subTest(args=args), self.assertRaises(ValueError):
                wilson_interval(*args)


class PowerTests(unittest.TestCase):
    def test_known_values(self):
        self.assertAlmostEqual(minimum_detectable_effect(1.0, 100), (1.959964 + 0.841621) / 10, places=5)
        self.assertEqual(required_pairs(1.0, 0.5), 32)  # (2.8016 / 0.5)**2 = 31.4
        self.assertEqual(required_pairs(0.0, 0.1), 1)
        self.assertEqual(minimum_detectable_effect(0.0, 5), 0.0)

    def test_mde_and_required_pairs_are_inverse(self):
        for sigma in (0.05, 0.3, 1.0, 2.5, 17.0):
            for n in (1, 2, 3, 10, 20, 64, 333, 1000):
                for alpha, power in ((0.05, 0.8), (0.01, 0.9), (0.1, 0.5)):
                    with self.subTest(sigma=sigma, n=n, alpha=alpha, power=power):
                        delta = minimum_detectable_effect(sigma, n, alpha=alpha, power=power)
                        self.assertEqual(required_pairs(sigma, delta, alpha=alpha, power=power), n)
            for delta in (0.01, 0.07, 0.3, 1.1):
                k = required_pairs(sigma, delta)
                self.assertLessEqual(minimum_detectable_effect(sigma, k), delta * (1 + 1e-9))
                if k > 1:
                    self.assertGreater(minimum_detectable_effect(sigma, k - 1), delta)

    def test_validation(self):
        for call in (lambda: minimum_detectable_effect(-1, 3), lambda: minimum_detectable_effect(1, 0),
                     lambda: minimum_detectable_effect(1, 2.0), lambda: required_pairs(1, 0),
                     lambda: required_pairs(-1, 1), lambda: required_pairs(1, 1, power=1.0),
                     lambda: required_pairs(1, 1, alpha=0.0)):
            with self.assertRaises(ValueError):
                call()


class SummaryTests(unittest.TestCase):
    def test_summary_fields(self):
        a = {("L1", s): v for s, v in zip(range(5001, 5009), (1, 0, 1, 1, 0, 1, 1, 0))}
        b = {("L1", s): v for s, v in zip(range(5001, 5009), (0, 0, 1, 0, 0, 0, 1, 0))}
        summary = summarize_paired(a, b, bootstrap_samples=2000)
        diffs = [d for _, d in paired_differences(a, b)]
        self.assertEqual(summary["n"], 8)
        self.assertEqual(summary["mean"], 3 / 8)
        self.assertAlmostEqual(summary["sd"], 0.517549, places=5)
        self.assertEqual(summary["p"], sign_flip_pvalue(diffs))
        self.assertEqual(summary["p"], 0.25)  # three +1, five zeros
        self.assertEqual(summary["p_method"], "exact")
        self.assertEqual(summary["ci"], list(bootstrap_ci(diffs, samples=2000)))
        self.assertAlmostEqual(summary["mde"], minimum_detectable_effect(summary["sd"], 8))
        canonical_json(summary)
        flipped = summarize_paired(b, a, bootstrap_samples=2000)
        self.assertEqual(flipped["p"], summary["p"])
        self.assertEqual(flipped["mean"], -summary["mean"])

    def test_single_pair_and_mismatch(self):
        summary = summarize_paired({"k": 2.0}, {"k": 1.5}, bootstrap_samples=10)
        self.assertEqual((summary["n"], summary["mean"], summary["sd"], summary["mde"]), (1, 0.5, None, None))
        self.assertEqual(summary["ci"], [0.5, 0.5])
        with self.assertRaises(ValueError):
            summarize_paired({"k": 1}, {"j": 1})
        with self.assertRaises(ValueError):
            summarize_paired({}, {})


if __name__ == "__main__":
    unittest.main()
