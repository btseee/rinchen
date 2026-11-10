"""Property tests of the bootstrap (D13, T9 and T10): whatever the seed and the units, every drawn
index is in range and the same call gives the same draws; the totals of identical rows and of a
column of ones are what counting says; a comparison of two systems is the mirror image of the
comparison the other way round, and a system never differs from itself. Under the `no_network`
guard like every D13 test module."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from rinchen.eval.bootstrap import (
    BootstrapSettings,
    draw_index,
    group_rows,
    interval_bounds,
    resample_indices,
    resample_totals,
)
from rinchen.eval.significance import compare, estimate, is_significant
from tests.support.no_network import no_network  # noqa: F401 - the fixture of the mark

pytestmark = pytest.mark.usefixtures("no_network")

SETTINGS = settings(
    max_examples=80, suppress_health_check=[HealthCheck.function_scoped_fixture]
)
SEEDS = st.integers(min_value=0, max_value=2**64 - 1)
UNITS = st.integers(min_value=1, max_value=60)
BOOTSTRAPS = st.builds(
    BootstrapSettings,
    resamples=st.integers(min_value=100, max_value=140),
    seed=SEEDS,
    alpha=st.sampled_from([0.01, 0.05, 0.1, 0.2]),
)
MICRO_SCORES = st.lists(
    st.integers(min_value=0, max_value=1_000_000), min_size=2, max_size=25
)


def mean_rows(micro: Sequence[int]) -> np.ndarray:
    return np.array([[score, 1] for score in micro], dtype=np.int64)


def mean_score(totals: Sequence[int]) -> float:
    return totals[0] / (totals[1] * 1_000_000)


@SETTINGS
@given(seed=SEEDS, units=UNITS, first=st.integers(min_value=0, max_value=500))
def test_every_index_is_in_range(seed: int, units: int, first: int) -> None:
    drawn = resample_indices(seed, first, first + 3, units)

    assert drawn.shape == (3, units)
    assert 0 <= int(drawn.min()) <= int(drawn.max()) < units


@SETTINGS
@given(seed=SEEDS, units=UNITS)
def test_the_same_call_twice_gives_equal_draws(seed: int, units: int) -> None:
    first = resample_indices(seed, 0, 4, units)
    again = resample_indices(seed, 0, 4, units)

    assert first.tolist() == again.tolist()
    assert first[0, 0] == draw_index(seed, 0, 0, units)


@SETTINGS
@given(
    row=st.lists(
        st.integers(min_value=-(10**6), max_value=10**6), min_size=1, max_size=5
    ),
    units=UNITS,
    bootstrap=BOOTSTRAPS,
)
def test_the_totals_of_identical_rows_are_the_row_times_the_units(
    row: list[int], units: int, bootstrap: BootstrapSettings
) -> None:
    rows = np.tile(np.array(row, dtype=np.int64), (units, 1))

    totals = resample_totals(rows, bootstrap)

    assert totals.shape == (bootstrap.resamples, len(row))
    assert (totals == np.array(row, dtype=np.int64) * units).all()


@SETTINGS
@given(units=UNITS, bootstrap=BOOTSTRAPS)
def test_the_totals_of_a_column_of_ones_are_the_number_of_units(
    units: int, bootstrap: BootstrapSettings
) -> None:
    totals = resample_totals(np.ones((units, 1), dtype=np.int64), bootstrap)

    assert set(totals[:, 0].tolist()) == {units}


@SETTINGS
@given(micro=MICRO_SCORES, bootstrap=BOOTSTRAPS)
def test_an_interval_is_ordered_and_a_mean_stays_in_the_range_of_the_scores(
    micro: list[int], bootstrap: BootstrapSettings
) -> None:
    result = estimate(mean_rows(micro), mean_score, bootstrap)

    assert result.interval is not None
    low, high = result.interval
    assert low <= high
    assert min(micro) / 1e6 - 1e-12 <= low
    assert high <= max(micro) / 1e6 + 1e-12
    assert result.units == len(micro)


@SETTINGS
@given(micro_a=MICRO_SCORES, micro_b=MICRO_SCORES, bootstrap=BOOTSTRAPS)
def test_the_comparison_the_other_way_round_is_its_mirror_image(
    micro_a: list[int], micro_b: list[int], bootstrap: BootstrapSettings
) -> None:
    size = min(len(micro_a), len(micro_b))
    rows_a, rows_b = mean_rows(micro_a[:size]), mean_rows(micro_b[:size])

    forward = compare(rows_a, rows_b, mean_score, bootstrap)
    backward = compare(rows_b, rows_a, mean_score, bootstrap)

    assert forward.interval is not None
    assert backward.interval is not None
    assert backward.delta == -forward.delta
    assert backward.interval == (-forward.interval[1], -forward.interval[0])
    assert backward.p_value == forward.p_value
    assert forward.better_fraction is not None
    assert backward.better_fraction is not None
    # the resamples with no difference at all are better for neither system
    assert forward.better_fraction + backward.better_fraction <= 1.0


@SETTINGS
@given(micro=MICRO_SCORES, bootstrap=BOOTSTRAPS)
def test_a_system_never_differs_from_itself(
    micro: list[int], bootstrap: BootstrapSettings
) -> None:
    rows = mean_rows(micro)

    result = compare(rows, rows.copy(), mean_score, bootstrap)

    assert (result.delta, result.interval, result.p_value) == (0.0, (0.0, 0.0), 1.0)
    assert result.better_fraction == 0.0
    assert not is_significant(result, bootstrap.alpha)


@SETTINGS
@given(micro_a=MICRO_SCORES, micro_b=MICRO_SCORES, bootstrap=BOOTSTRAPS)
def test_a_p_value_is_a_probability_with_the_add_one_floor(
    micro_a: list[int], micro_b: list[int], bootstrap: BootstrapSettings
) -> None:
    size = min(len(micro_a), len(micro_b))

    result = compare(
        mean_rows(micro_a[:size]), mean_rows(micro_b[:size]), mean_score, bootstrap
    )

    assert result.p_value is not None
    assert 2 / (bootstrap.resamples + 1) <= result.p_value <= 1.0


@SETTINGS
@given(
    rows=st.lists(
        st.lists(st.integers(-1000, 1000), min_size=2, max_size=2),
        min_size=1,
        max_size=20,
    ),
    data=st.data(),
)
def test_grouping_keeps_the_column_sums_and_numbers_groups_by_first_appearance(
    rows: list[list[int]], data: st.DataObject
) -> None:
    groups = data.draw(
        st.lists(
            st.sampled_from(["a", "b", "c", "d"]),
            min_size=len(rows),
            max_size=len(rows),
        )
    )

    grouped, labels = group_rows(rows, groups)

    assert grouped.sum(axis=0).tolist() == [sum(column) for column in zip(*rows)]
    assert labels == tuple(dict.fromkeys(groups))
    assert len(labels) == grouped.shape[0]


@SETTINGS
@given(values=st.lists(st.floats(-1e6, 1e6), min_size=1, max_size=300))
def test_the_bounds_are_two_of_the_values_in_order(values: list[float]) -> None:
    low, high = interval_bounds(values, 0.1)

    assert low <= high
    assert low in values
    assert high in values
