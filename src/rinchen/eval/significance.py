"""Estimates with an interval and the paired comparison of two systems (D13, T9 and T10).

`estimate` bootstraps one system: the score of the whole corpus and the percentile interval of
the scores of the resamples. `compare` bootstraps two systems scored on the SAME units: ONE set
of draws is shared by both, so what differs between the systems is all that moves the
difference, and a constant offset gives an interval of zero width.

The p-value is two-sided with add-one smoothing,
`min(1, 2 * min((le + 1) / (B + 1), (ge + 1) / (B + 1)))`, with `le` the resamples whose
difference is at most 0 and `ge` those at least 0: identical systems give exactly 1.0 and the
smallest value is `2 / (B + 1)`. The verdict of a comparison is its p-value against alpha and
NEVER its interval, whose end can touch zero while the p-value is large (ties: many resamples
have a difference of exactly zero). There is no correction for several comparisons.

sacrebleu's own paired test is never used: it calls a system significantly different from itself
and reads its seed from the environment. A score is whatever the caller hands in, a function of
a list of Python integers (a column sum of the statistics); this module knows no metric.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from rinchen.eval.bootstrap import (
    BootstrapSettings,
    interval_bounds,
    require_unit_rows,
    resample_totals,
)

Scorer = Callable[[Sequence[int]], float]


@dataclass(frozen=True, slots=True)
class Estimate:
    """The score of the whole corpus and its percentile interval; `interval` is `None` with
    fewer than two units, where an interval would be a point."""

    value: float
    interval: tuple[float, float] | None
    units: int


@dataclass(frozen=True, slots=True)
class PairedComparison:
    """`delta` is the score of the first system minus the second's on the whole corpus;
    `interval` the percentile interval of the resample differences; `p_value` the two-sided
    p-value; `better_fraction` the share of resamples in which the first system is better
    (above for a higher-is-better metric, below otherwise). The last three are `None` with
    fewer than two units."""

    delta: float
    interval: tuple[float, float] | None
    p_value: float | None
    better_fraction: float | None
    units: int
    higher_is_better: bool = True


def _score_of_sum(rows: NDArray[np.int64], score: Scorer) -> float:
    return float(score(rows.sum(axis=0).tolist()))


def estimate(
    rows: NDArray[np.int64], score: Scorer, settings: BootstrapSettings
) -> Estimate:
    """The estimate of the corpus whose units are the rows of `rows`.

    Raises:
        ValueError: `rows` is not a `(units, columns)` int64 array with a unit (see
            `require_unit_rows`).
    """
    matrix = require_unit_rows(rows)
    value = _score_of_sum(matrix, score)
    units = matrix.shape[0]
    if units < 2:
        return Estimate(value, None, units)
    scores = [score(row) for row in resample_totals(matrix, settings).tolist()]
    return Estimate(value, interval_bounds(scores, settings.alpha), units)


def compare(
    rows_a: NDArray[np.int64],
    rows_b: NDArray[np.int64],
    score: Scorer,
    settings: BootstrapSettings,
    *,
    higher_is_better: bool = True,
) -> PairedComparison:
    """System `a` against system `b`, scored on the same units, with one set of draws.

    Raises:
        ValueError: an array that `require_unit_rows` refuses, or two arrays of different
            shapes.
    """
    a = require_unit_rows(rows_a)
    b = require_unit_rows(rows_b)
    if a.shape != b.shape:
        raise ValueError("the two systems must have rows of the same shape")
    units, width = a.shape
    delta = _score_of_sum(a, score) - _score_of_sum(b, score)
    if units < 2:
        return PairedComparison(delta, None, None, None, units, higher_is_better)
    # One draw over both systems' columns side by side: the same units for each resample.
    drawn = resample_totals(np.concatenate([a, b], axis=1), settings).tolist()
    differences = [
        float(score(totals[:width])) - float(score(totals[width:])) for totals in drawn
    ]
    resamples = settings.resamples
    at_most = sum(difference <= 0 for difference in differences)
    at_least = sum(difference >= 0 for difference in differences)
    better = sum(
        difference > 0 if higher_is_better else difference < 0
        for difference in differences
    )
    p_value = min(
        1.0, 2 * min((at_most + 1) / (resamples + 1), (at_least + 1) / (resamples + 1))
    )
    return PairedComparison(
        delta,
        interval_bounds(differences, settings.alpha),
        p_value,
        better / resamples,
        units,
        higher_is_better,
    )


def is_significant(comparison: PairedComparison, alpha: float) -> bool:
    """Whether the first system is significantly better: its p-value is below `alpha` AND its
    delta is better than zero for the metric's direction. The interval never decides.

    Raises:
        ValueError: `alpha` is not in (0, 0.5).
    """
    if not 0 < alpha < 0.5:
        raise ValueError("alpha must be a number above 0 and below 0.5")
    if comparison.p_value is None:
        return False
    if comparison.higher_is_better:
        improved = comparison.delta > 0
    else:
        improved = comparison.delta < 0
    return improved and comparison.p_value < alpha
