"""Paired evaluation statistics (standard library only).

Pairs share a key such as ``(level, seed)``; every difference is ``a - b``.
Randomized procedures take an explicit seed so reports are reproducible.
"""
from __future__ import annotations

import math
import random
import statistics
from statistics import NormalDist

MAX_EXACT_N = 24  # exact enumeration holds 2**(n-1) sums in memory


def _finite(value, what: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{what} must be a finite number")
    return float(value)


def _diffs(diffs) -> list[float]:
    values = [_finite(value, "difference") for value in diffs]
    if not values:
        raise ValueError("at least one paired difference is required")
    return values


def _alpha(alpha) -> float:
    alpha = _finite(alpha, "alpha")
    if not 0 < alpha < 1:
        raise ValueError("alpha must lie in (0, 1)")
    return alpha


def paired_differences(a: dict, b: dict) -> list[tuple]:
    if set(a) != set(b):
        missing, extra = sorted(map(repr, set(a) - set(b))), sorted(map(repr, set(b) - set(a)))
        raise ValueError(f"paired keys differ: only in a={missing}, only in b={extra}")
    try:
        keys = sorted(a)
    except TypeError:
        keys = sorted(a, key=repr)
    return [(key, _finite(a[key], f"a[{key!r}]") - _finite(b[key], f"b[{key!r}]")) for key in keys]


def _sign_flip(diffs, exact_max_n: int, samples: int, seed: int) -> tuple[float, str]:
    values = _diffs(diffs)
    if isinstance(exact_max_n, bool) or not isinstance(exact_max_n, int) or not 0 <= exact_max_n <= MAX_EXACT_N:
        raise ValueError(f"exact_max_n must be an integer in [0, {MAX_EXACT_N}]")
    # Zero differences do not move the sum under any sign assignment.
    nonzero = [value for value in values if value != 0.0]
    if not nonzero:
        return 1.0, "exact"
    observed = abs(math.fsum(nonzero))
    tolerance = 1e-9 * math.fsum(abs(value) for value in nonzero)  # ties up to float rounding
    if len(nonzero) <= exact_max_n:
        # |sum| is symmetric under flipping every sign: fix the first sign.
        sums = [nonzero[0]]
        for value in nonzero[1:]:
            sums = [s + value for s in sums] + [s - value for s in sums]
        extreme = sum(1 for s in sums if abs(s) >= observed - tolerance)
        return extreme / len(sums), "exact"
    if isinstance(samples, bool) or not isinstance(samples, int) or samples < 1:
        raise ValueError("samples must be a positive integer")
    rng = random.Random(seed)
    total = math.fsum(nonzero)
    extreme = 0
    for _ in range(samples):
        bits = rng.getrandbits(len(nonzero))
        flipped = math.fsum(value for index, value in enumerate(nonzero) if bits >> index & 1)
        if abs(total - 2.0 * flipped) >= observed - tolerance:
            extreme += 1
    # Count the observed assignment itself so a Monte Carlo p is never zero.
    return (extreme + 1) / (samples + 1), "monte_carlo"


def sign_flip_pvalue(diffs, *, exact_max_n: int = 20, samples: int = 100000, seed: int = 0) -> float:
    """Two-sided paired sign-flip (randomization) test on the sum of differences.

    Zero differences are dropped first (they never change the sum), so exact
    enumeration applies whenever at most ``exact_max_n`` differences are nonzero.
    """
    return _sign_flip(diffs, exact_max_n, samples, seed)[0]


def _percentile(ordered: list[float], q: float) -> float:
    position = q * (len(ordered) - 1)
    lower = math.floor(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def bootstrap_ci(diffs, *, alpha: float = 0.05, samples: int = 10000, seed: int = 0) -> tuple[float, float]:
    """Percentile bootstrap interval for the mean difference."""
    values = _diffs(diffs)
    alpha = _alpha(alpha)
    if isinstance(samples, bool) or not isinstance(samples, int) or samples < 1:
        raise ValueError("samples must be a positive integer")
    rng = random.Random(seed)
    n = len(values)
    means = sorted(math.fsum(rng.choices(values, k=n)) / n for _ in range(samples))
    return _percentile(means, alpha / 2), _percentile(means, 1 - alpha / 2)


def wilson_interval(successes: int, n: int, *, z: float | None = None, alpha: float = 0.05) -> tuple[float, float]:
    for value, what in ((successes, "successes"), (n, "n")):
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"{what} must be an integer")
    if n < 0 or not 0 <= successes <= n:
        raise ValueError("require 0 <= successes <= n")
    if n == 0:
        return 0.0, 1.0
    z = NormalDist().inv_cdf(1 - _alpha(alpha) / 2) if z is None else _finite(z, "z")
    p = successes / n
    denominator = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denominator
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
    lower = 0.0 if successes == 0 else max(0.0, centre - half)
    upper = 1.0 if successes == n else min(1.0, centre + half)
    return lower, upper


def _z_sum(alpha: float, power: float) -> float:
    power = _finite(power, "power")
    if not 0 < power < 1:
        raise ValueError("power must lie in (0, 1)")
    normal = NormalDist()
    return normal.inv_cdf(1 - _alpha(alpha) / 2) + normal.inv_cdf(power)


def minimum_detectable_effect(sigma_d: float, n: int, *, alpha: float = 0.05, power: float = 0.8) -> float:
    """Two-sided normal-approximation MDE of the mean paired difference."""
    sigma_d = _finite(sigma_d, "sigma_d")
    if sigma_d < 0:
        raise ValueError("sigma_d must be nonnegative")
    if isinstance(n, bool) or not isinstance(n, int) or n < 1:
        raise ValueError("n must be a positive integer")
    return _z_sum(alpha, power) * sigma_d / math.sqrt(n)


def required_pairs(sigma_d: float, delta: float, *, alpha: float = 0.05, power: float = 0.8) -> int:
    """Smallest n whose MDE does not exceed ``delta`` (inverse of the MDE, ceil)."""
    sigma_d, delta = _finite(sigma_d, "sigma_d"), _finite(delta, "delta")
    if sigma_d < 0 or delta <= 0:
        raise ValueError("require sigma_d >= 0 and delta > 0")
    exact = (_z_sum(alpha, power) * sigma_d / delta) ** 2
    pairs = math.ceil(exact)
    if pairs > 1 and math.isclose(exact, pairs - 1, rel_tol=1e-9):
        pairs -= 1  # an integer up to float rounding is not one more pair
    return max(1, pairs)


def summarize_paired(a: dict, b: dict, *, alpha: float = 0.05, power: float = 0.8, seed: int = 0,
                     exact_max_n: int = 20, sign_flip_samples: int = 100000,
                     bootstrap_samples: int = 10000) -> dict:
    pairs = paired_differences(a, b)
    diffs = [diff for _, diff in pairs]
    n = len(diffs)
    if n == 0:
        raise ValueError("at least one paired difference is required")
    sd = statistics.stdev(diffs) if n > 1 else None
    p, method = _sign_flip(diffs, exact_max_n, sign_flip_samples, seed)
    lower, upper = bootstrap_ci(diffs, alpha=alpha, samples=bootstrap_samples, seed=seed)
    return {"n": n, "mean": math.fsum(diffs) / n, "sd": sd, "p": p, "p_method": method,
            "ci": [lower, upper], "alpha": alpha, "power": power, "seed": seed,
            "mde": None if sd is None else minimum_detectable_effect(sd, n, alpha=alpha, power=power)}
