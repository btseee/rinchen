"""The estimates and the paired comparison of D13 (T9 and T10), against the golden literals of
`tests/support/eval_golden.py`: BLEU, chrF and chrF++ of the golden corpus with an interval over
the segments and over the five documents, the comparison of `beta` against `alpha`, and the
mean-type metric over twelve integer micro-scores.

Every literal was derived with the pure-Python reference of the design spike, not with the code
under test. BLEU's `exp` and `log` come from the C library, which may differ in the last digit on
another machine, so a BLEU interval end is compared at 1e-9 relative; every other figure with
`==`, and the integer sums are exact by construction. The verdict of a comparison is its p-value
against alpha and never the interval, which ties can make touch zero.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
import pytest

from rinchen.eval.bootstrap import (
    BootstrapSettings,
    group_rows,
    interval_bounds,
    resample_totals,
)
from rinchen.eval.significance import (
    Estimate,
    PairedComparison,
    compare,
    estimate,
    is_significant,
)
from rinchen.metrics.segment_stats import (
    score_from_statistics,
    segment_statistics,
    statistics_metric,
)
from tests.support.eval_golden import (
    ALPHA_ENG,
    ALPHA_KHK,
    BETA_ENG,
    BETA_KHK,
    GOLDEN_COUNT_METRICS,
    GOLDEN_DIRECTIONS,
    GOLDEN_DOCS,
    GOLDEN_DOCUMENT_INTERVALS,
    GOLDEN_ENG,
    GOLDEN_KHK,
    GOLDEN_PAIRED,
    GOLDEN_SEGMENT_ESTIMATES,
    GOLDEN_SYSTEMS,
    MEAN_ABOVE,
    MEAN_AT_OR_ABOVE,
    MEAN_AT_OR_BELOW,
    MEAN_BETTER_FRACTION,
    MEAN_INTERVAL_A,
    MEAN_INTERVAL_DIFFERENCE,
    MEAN_MICRO_A,
    MEAN_MICRO_B,
    MEAN_P_VALUE,
    MEAN_POINT_A,
    MEAN_POINT_B,
)
from tests.support.no_network import no_network  # noqa: F401 - the fixture of the mark

pytestmark = pytest.mark.usefixtures("no_network")

SETTINGS = BootstrapSettings()
TARGET = {"khk_Cyrl-eng_Latn": "eng_Latn", "eng_Latn-khk_Cyrl": "khk_Cyrl"}
REFERENCES = {"khk_Cyrl-eng_Latn": GOLDEN_ENG, "eng_Latn-khk_Cyrl": GOLDEN_KHK}
HYPOTHESES = {
    ("khk_Cyrl-eng_Latn", "alpha"): ALPHA_ENG,
    ("khk_Cyrl-eng_Latn", "beta"): BETA_ENG,
    ("eng_Latn-khk_Cyrl", "alpha"): ALPHA_KHK,
    ("eng_Latn-khk_Cyrl", "beta"): BETA_KHK,
}
Scorer = Callable[[Sequence[int]], float]


def golden_rows(
    direction: str, system: str, metric: str, unit: str
) -> tuple[np.ndarray, Scorer]:
    """The unit rows of a golden system and the scorer of its metric."""
    metric_ = statistics_metric(metric, TARGET[direction])
    statistics = segment_statistics(
        metric_, HYPOTHESES[direction, system], REFERENCES[direction]
    )
    groups = GOLDEN_DOCS if unit == "document" else None
    rows, _ = group_rows(statistics.rows, groups)
    return rows, lambda totals: score_from_statistics(metric_, totals)


def mean_rows(micro: Sequence[int]) -> np.ndarray:
    return np.array([[score, 1] for score in micro], dtype=np.int64)


def mean_score(totals: Sequence[int]) -> float:
    return totals[0] / (totals[1] * 1_000_000)


def total_score(totals: Sequence[int]) -> float:
    return float(totals[0])


def assert_interval(
    actual: tuple[float, float] | None, expected: tuple[float, float], metric: str
) -> None:
    assert actual is not None
    if metric == "bleu":
        assert actual == pytest.approx(expected, rel=1e-9)
    else:
        assert actual == expected


GOLDEN_KEYS = [
    (direction, system, metric)
    for direction in GOLDEN_DIRECTIONS
    for system in GOLDEN_SYSTEMS
    for metric in GOLDEN_COUNT_METRICS
]


# ---- estimate -----------------------------------------------------------------------------


@pytest.mark.parametrize("key", GOLDEN_KEYS, ids=" ".join)
def test_the_segment_estimate_is_the_golden_point_and_interval(
    key: tuple[str, str, str],
) -> None:
    direction, system, metric = key
    rows, score = golden_rows(direction, system, metric, "segment")
    golden = GOLDEN_SEGMENT_ESTIMATES[key]

    result = estimate(rows, score, SETTINGS)

    assert result.value == golden.point
    assert_interval(result.interval, (golden.low, golden.high), metric)
    assert result.units == 10


@pytest.mark.parametrize("key", GOLDEN_KEYS, ids=" ".join)
def test_the_document_estimate_is_the_golden_point_and_interval(
    key: tuple[str, str, str],
) -> None:
    direction, system, metric = key
    rows, score = golden_rows(direction, system, metric, "document")

    result = estimate(rows, score, SETTINGS)

    assert result.value == GOLDEN_SEGMENT_ESTIMATES[key].point
    assert_interval(result.interval, GOLDEN_DOCUMENT_INTERVALS[key], metric)
    assert result.units == 5


def test_the_mean_type_estimate_as_a_mean_with_its_interval() -> None:
    result = estimate(mean_rows(MEAN_MICRO_A), mean_score, SETTINGS)

    assert result == Estimate(MEAN_POINT_A, MEAN_INTERVAL_A, 12)


def test_the_estimate_values_are_python_floats_whatever_the_scorer_returns() -> None:
    def numpy_scorer(totals: Sequence[int]) -> float:
        return np.float64(totals[0]) / np.float64(totals[1] * 1_000_000)  # type: ignore[return-value]

    result = estimate(mean_rows(MEAN_MICRO_A), numpy_scorer, SETTINGS)

    assert type(result.value) is float
    assert result.interval is not None
    assert [type(end) for end in result.interval] == [float, float]
    assert result == Estimate(MEAN_POINT_A, MEAN_INTERVAL_A, 12)


def test_the_comparison_values_are_python_floats_whatever_the_scorer_returns() -> None:
    def numpy_scorer(totals: Sequence[int]) -> float:
        return np.float64(totals[0]) / np.float64(totals[1] * 1_000_000)  # type: ignore[return-value]

    result = compare(
        mean_rows(MEAN_MICRO_A), mean_rows(MEAN_MICRO_B), numpy_scorer, SETTINGS
    )

    assert result.interval is not None
    assert [
        type(number)
        for number in (
            result.delta,
            *result.interval,
            result.p_value,
            result.better_fraction,
        )
    ] == [float] * 5
    assert result.p_value == MEAN_P_VALUE


def test_the_scorer_gets_a_list_of_python_integers_the_first_time_over_the_whole_rows() -> (
    None
):
    seen: list[Any] = []

    def spy(totals: Sequence[int]) -> float:
        seen.append(totals)
        return mean_score(totals)

    estimate(mean_rows(MEAN_MICRO_A), spy, SETTINGS)

    assert len(seen) == 1 + 1000
    assert seen[0] == [sum(MEAN_MICRO_A), 12]
    assert all(type(totals) is list for totals in seen)
    assert all(type(cell) is int for totals in seen for cell in totals)


def test_the_interval_follows_the_alpha_and_the_resample_count_of_the_settings() -> (
    None
):
    settings = BootstrapSettings(resamples=200, seed=5, alpha=0.1)
    rows = mean_rows(MEAN_MICRO_A)
    scores = sorted(mean_score(row.tolist()) for row in resample_totals(rows, settings))

    result = estimate(rows, mean_score, settings)

    # k = floor(200 * 0.1 / 2) = 10: the positions 10 and 189
    assert result.interval == (scores[10], scores[189])
    assert result.interval == interval_bounds(scores, 0.1)


def test_the_estimate_follows_the_seed() -> None:
    rows = mean_rows(MEAN_MICRO_A)

    first = estimate(rows, mean_score, BootstrapSettings(seed=1))
    other = estimate(rows, mean_score, BootstrapSettings(seed=2))

    assert first.value == other.value
    assert first.interval != other.interval


def test_one_unit_has_a_value_and_no_interval() -> None:
    result = estimate(np.array([[912000, 1]], dtype=np.int64), mean_score, SETTINGS)

    assert result == Estimate(0.912, None, 1)


def test_no_unit_is_a_value_error() -> None:
    with pytest.raises(ValueError, match="unit"):
        estimate(np.zeros((0, 2), dtype=np.int64), mean_score, SETTINGS)


def test_rows_that_are_not_a_two_dimensional_int64_array_are_a_value_error() -> None:
    with pytest.raises(ValueError, match="int64"):
        estimate(np.zeros((3, 2)), mean_score, SETTINGS)


# ---- compare ------------------------------------------------------------------------------


@pytest.mark.parametrize(("direction", "unit"), sorted(GOLDEN_PAIRED))
def test_beta_against_alpha_on_chrf_pp_is_the_golden_comparison(
    direction: str, unit: str
) -> None:
    rows_beta, score = golden_rows(direction, "beta", "chrf_pp", unit)
    rows_alpha, _ = golden_rows(direction, "alpha", "chrf_pp", unit)
    golden = GOLDEN_PAIRED[direction, unit]

    result = compare(rows_beta, rows_alpha, score, SETTINGS)

    assert result == PairedComparison(
        golden.delta,
        (golden.low, golden.high),
        golden.p_value,
        golden.better_fraction,
        10 if unit == "segment" else 5,
    )


def test_a_comparison_of_two_mean_type_systems_through_its_p_value_and_its_better_fraction() -> (
    None
):
    result = compare(
        mean_rows(MEAN_MICRO_A), mean_rows(MEAN_MICRO_B), mean_score, SETTINGS
    )

    # 7 resamples at or below zero, 994 at or above, 993 above (see eval_golden)
    assert (MEAN_AT_OR_BELOW, MEAN_AT_OR_ABOVE, MEAN_ABOVE) == (7, 994, 993)
    assert result.p_value == MEAN_P_VALUE
    assert result.better_fraction == MEAN_BETTER_FRACTION
    assert result.delta == pytest.approx(MEAN_POINT_A - MEAN_POINT_B, abs=1e-12)
    assert result.interval == pytest.approx(MEAN_INTERVAL_DIFFERENCE, abs=1e-12)
    assert result.units == 12


def test_the_comparison_follows_the_alpha_the_resample_count_and_the_seed_of_the_settings() -> (
    None
):
    settings = BootstrapSettings(resamples=200, seed=5, alpha=0.1)
    # Two separate resamplings with the same settings draw the same units: the pairing the
    # comparison relies on, derived here from the resampling alone.
    totals_a = resample_totals(mean_rows(MEAN_MICRO_A), settings).tolist()
    totals_b = resample_totals(mean_rows(MEAN_MICRO_B), settings).tolist()
    differences = sorted(
        mean_score(a) - mean_score(b) for a, b in zip(totals_a, totals_b, strict=True)
    )
    at_most = sum(difference <= 0 for difference in differences)
    at_least = sum(difference >= 0 for difference in differences)

    result = compare(
        mean_rows(MEAN_MICRO_A), mean_rows(MEAN_MICRO_B), mean_score, settings
    )

    # k = floor(200 * 0.1 / 2) = 10: the positions 10 and 189
    assert result.interval == (differences[10], differences[189])
    assert result.p_value == min(
        1.0, 2 * min((at_most + 1) / 201, (at_least + 1) / 201)
    )
    assert result.better_fraction == sum(d > 0 for d in differences) / 200


def test_swapping_the_systems_negates_the_delta_and_the_interval_and_keeps_the_p_value() -> (
    None
):
    forward = compare(
        mean_rows(MEAN_MICRO_A), mean_rows(MEAN_MICRO_B), mean_score, SETTINGS
    )

    swapped = compare(
        mean_rows(MEAN_MICRO_B), mean_rows(MEAN_MICRO_A), mean_score, SETTINGS
    )

    assert forward.interval is not None
    assert swapped.interval is not None
    assert swapped.delta == -forward.delta
    assert swapped.interval == (-forward.interval[1], -forward.interval[0])
    assert swapped.p_value == forward.p_value == MEAN_P_VALUE
    # 6 resamples have B above A: 1,000 - 994
    assert swapped.better_fraction == 0.006


def test_swapping_golden_chrf_pp_systems_negates_and_keeps_the_p_value() -> None:
    rows_beta, score = golden_rows("khk_Cyrl-eng_Latn", "beta", "chrf_pp", "segment")
    rows_alpha, _ = golden_rows("khk_Cyrl-eng_Latn", "alpha", "chrf_pp", "segment")
    golden = GOLDEN_PAIRED["khk_Cyrl-eng_Latn", "segment"]

    swapped = compare(rows_alpha, rows_beta, score, SETTINGS)

    assert swapped.delta == -golden.delta
    assert swapped.interval == (-golden.high, -golden.low)
    assert swapped.p_value == golden.p_value
    assert swapped.better_fraction == 0.0


def test_identical_systems_have_no_difference_and_a_p_value_of_exactly_one() -> None:
    rows, score = golden_rows("khk_Cyrl-eng_Latn", "alpha", "chrf_pp", "segment")

    result = compare(rows, rows.copy(), score, SETTINGS)

    assert result == PairedComparison(0.0, (0.0, 0.0), 1.0, 0.0, 10)
    assert not is_significant(result, 0.05)


def test_identical_mean_type_systems_are_not_significant_either() -> None:
    result = compare(
        mean_rows(MEAN_MICRO_A), mean_rows(MEAN_MICRO_A), mean_score, SETTINGS
    )

    assert (result.delta, result.interval, result.p_value) == (0.0, (0.0, 0.0), 1.0)
    assert not is_significant(result, 0.49)


def test_a_constant_offset_on_every_unit_is_a_zero_width_interval_because_the_draws_are_shared() -> (
    None
):
    rows_a = mean_rows(MEAN_MICRO_A)
    rows_b = rows_a + np.array([1000, 0], dtype=np.int64)

    exact = compare(rows_a, rows_b, total_score, SETTINGS)
    as_means = compare(rows_a, rows_b, mean_score, SETTINGS)

    # a resample draws twelve units, so the sums differ by 12 * 1000 in every resample
    assert exact.delta == -12000.0
    assert exact.interval == (-12000.0, -12000.0)
    assert exact.p_value == 2 / 1001
    assert exact.better_fraction == 0.0
    # as means the floats differ in the last digits only; an unpaired draw would be 0.11 wide
    assert as_means.interval is not None
    assert as_means.interval[1] - as_means.interval[0] < 1e-12
    assert as_means.delta == pytest.approx(-0.001, abs=1e-12)


def test_the_two_systems_are_drawn_with_the_same_indices() -> None:
    # System b is system a read in another column: a shared draw makes their sums equal in every
    # resample, whatever the seed, so the scores never differ.
    rows = mean_rows(MEAN_MICRO_A)
    duplicated = rows.copy()

    for seed in (1, 2, 3):
        result = compare(rows, duplicated, mean_score, BootstrapSettings(seed=seed))

        assert result.interval == (0.0, 0.0)


def test_shapes_that_differ_are_a_value_error() -> None:
    with pytest.raises(ValueError, match="shape"):
        compare(
            mean_rows(MEAN_MICRO_A),
            mean_rows(MEAN_MICRO_B[:-1]),
            mean_score,
            SETTINGS,
        )
    with pytest.raises(ValueError, match="shape"):
        compare(
            mean_rows(MEAN_MICRO_A),
            np.zeros((12, 3), dtype=np.int64),
            mean_score,
            SETTINGS,
        )


def test_no_unit_in_a_comparison_is_a_value_error() -> None:
    empty = np.zeros((0, 2), dtype=np.int64)

    with pytest.raises(ValueError, match="unit"):
        compare(empty, empty, mean_score, SETTINGS)


def test_one_unit_gives_a_delta_and_nothing_else() -> None:
    a = np.array([[900000, 1]], dtype=np.int64)
    b = np.array([[800000, 1]], dtype=np.int64)

    result = compare(a, b, mean_score, SETTINGS)

    assert result == PairedComparison(pytest.approx(0.1), None, None, None, 1)
    assert not is_significant(result, 0.05)


def test_when_lower_is_better_the_better_fraction_counts_the_negative_differences() -> (
    None
):
    rows_a = mean_rows(MEAN_MICRO_A)
    rows_b = mean_rows(MEAN_MICRO_B)

    higher = compare(rows_a, rows_b, mean_score, SETTINGS)
    lower = compare(rows_a, rows_b, mean_score, SETTINGS, higher_is_better=False)

    assert higher.better_fraction == 0.993
    assert lower.better_fraction == 0.006
    assert lower.p_value == higher.p_value == MEAN_P_VALUE
    assert lower.delta == higher.delta
    assert lower.interval == higher.interval
    assert (higher.higher_is_better, lower.higher_is_better) == (True, False)


def test_the_default_direction_is_higher_is_better() -> None:
    result = compare(
        mean_rows(MEAN_MICRO_A), mean_rows(MEAN_MICRO_B), mean_score, SETTINGS
    )

    assert result.higher_is_better is True


def test_each_score_is_asked_for_once_per_resample_and_system() -> None:
    calls: list[int] = []

    def counting(totals: Sequence[int]) -> float:
        calls.append(1)
        return mean_score(totals)

    compare(mean_rows(MEAN_MICRO_A), mean_rows(MEAN_MICRO_B), counting, SETTINGS)

    assert len(calls) == 2 + 2 * 1000


# ---- is_significant -----------------------------------------------------------------------


def comparison(
    delta: float,
    p_value: float | None,
    interval: tuple[float, float] | None = (-1.0, 1.0),
    *,
    higher_is_better: bool = True,
) -> PairedComparison:
    return PairedComparison(
        delta,
        interval,
        p_value,
        None if p_value is None else 0.9,
        5,
        higher_is_better,
    )


def test_a_comparison_is_significant_when_the_p_value_is_below_alpha_and_delta_is_better() -> (
    None
):
    assert is_significant(comparison(0.5, 0.0499), 0.05)
    assert is_significant(comparison(0.5, MEAN_P_VALUE), 0.05)


def test_a_p_value_equal_to_alpha_or_above_is_not_significant() -> None:
    assert not is_significant(comparison(0.5, 0.05), 0.05)
    assert not is_significant(comparison(0.5, 0.2), 0.05)
    assert not is_significant(comparison(0.5, MEAN_P_VALUE), 0.01)


def test_the_verdict_is_the_p_value_and_never_the_interval() -> None:
    # ties make an interval end touch zero while the p-value is small, and an interval can sit
    # above zero while the p-value is large: only the p-value decides
    assert is_significant(comparison(0.2, 0.01, interval=(0.0, 1.0)), 0.05)
    assert is_significant(comparison(0.2, 0.01, interval=(-0.5, 1.0)), 0.05)
    assert not is_significant(comparison(0.2, 0.2, interval=(0.1, 2.0)), 0.05)


def test_a_delta_of_zero_or_in_the_wrong_direction_is_not_significant() -> None:
    assert not is_significant(comparison(0.0, 0.001), 0.05)
    assert not is_significant(comparison(-0.5, 0.001), 0.05)


def test_when_lower_is_better_a_negative_delta_is_the_better_one() -> None:
    assert is_significant(comparison(-0.5, 0.001, higher_is_better=False), 0.05)
    assert not is_significant(comparison(0.5, 0.001, higher_is_better=False), 0.05)
    assert not is_significant(comparison(0.0, 0.001, higher_is_better=False), 0.05)


def test_a_comparison_without_a_p_value_is_not_significant() -> None:
    assert not is_significant(comparison(0.5, None, interval=None), 0.05)


def test_the_golden_comparisons_are_significant_at_five_percent() -> None:
    for (direction, unit), golden in GOLDEN_PAIRED.items():
        rows_beta, score = golden_rows(direction, "beta", "chrf_pp", unit)
        rows_alpha, _ = golden_rows(direction, "alpha", "chrf_pp", unit)

        result = compare(rows_beta, rows_alpha, score, SETTINGS)

        assert golden.p_value < 0.05
        assert is_significant(result, 0.05)
        assert not is_significant(compare(rows_alpha, rows_beta, score, SETTINGS), 0.05)


@pytest.mark.parametrize("alpha", [0.0, 0.5, 1.0, -0.1, float("nan")])
def test_an_alpha_outside_zero_to_one_half_is_a_value_error(alpha: float) -> None:
    with pytest.raises(ValueError, match="alpha"):
        is_significant(comparison(0.5, 0.01), alpha)
