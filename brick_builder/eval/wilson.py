"""Wilson score interval for a binomial success rate.

Used everywhere in the evaluation report to turn a raw "successes out of n" count into a 95 %
confidence interval, so that small held-out sets are not over-trusted (the acceptance criteria state
acceptance thresholds directly as Wilson lower bounds).
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class WilsonInterval:
    """A point estimate plus its Wilson score interval."""

    p_hat: float   # raw success rate, successes / n
    low: float     # interval lower bound
    high: float    # interval upper bound
    n: int


def wilson_interval(successes: int, n: int, z: float = 1.96) -> WilsonInterval:
    """Wilson score interval for a binomial proportion.

    `z=1.96` gives a 95 % interval. This is preferred over the naive normal-approximation
    interval because it stays inside [0, 1] and is much less optimistic at small `n` or at
    success rates near 0 or 1, which is exactly the regime evaluation sweeps run in (per-cell
    counts can be a few dozen episodes).

    Raises `ValueError` if `n <= 0` or `successes` is outside `[0, n]`.
    """
    if n <= 0:
        raise ValueError(f"n must be positive, got {n}")
    if not (0 <= successes <= n):
        raise ValueError(f"successes ({successes}) must be in [0, n={n}]")

    p_hat = successes / n
    z2 = z * z
    denom = 1.0 + z2 / n
    center = p_hat + z2 / (2 * n)
    margin = z * math.sqrt((p_hat * (1 - p_hat) + z2 / (4 * n)) / n)
    low = (center - margin) / denom
    high = (center + margin) / denom
    # Clip for floating-point edge cases at p_hat in {0, 1}.
    low = max(0.0, low)
    high = min(1.0, high)
    return WilsonInterval(p_hat=p_hat, low=low, high=high, n=n)


def intervals_disjoint(a: WilsonInterval, b: WilsonInterval) -> bool:
    """True when two Wilson intervals do not overlap.

    Used to check acceptance criteria phrased as "beats X with disjoint CIs" (the
    student must beat the scripted baseline by margin with disjoint confidence intervals).
    """
    return a.high < b.low or b.high < a.low
