"""The bootstrap of D13 (T9): the counter-based index stream and its vectors, the draws of whole
resamples, the sums of the drawn rows, the units a document bootstrap makes, the positions of the
interval, the settings and the units a document bootstrap makes.

The literals are the ones of `tests/support/eval_golden.py` and of the plan; the pure-Python
reference they were derived with is frozen in `test_eval_bootstrap_reference.py`, and what the
bootstrap must never depend on is in `test_eval_bootstrap_determinism.py`.
"""

from __future__ import annotations

import dataclasses
import math
import time

import numpy as np
import pytest

from rinchen.errors import ConfigError
from rinchen.eval import bootstrap
from rinchen.eval.bootstrap import (
    BOOTSTRAP_SEED,
    DEFAULT_ALPHA,
    DEFAULT_RESAMPLES,
    MIN_RESAMPLES,
    RNG_NAME,
    BootstrapSettings,
    draw_index,
    group_rows,
    interval_bounds,
    interval_positions,
    resample_indices,
    resample_totals,
)
from tests.support.eval_golden import (
    GOLDEN_ALPHA,
    GOLDEN_DOCS,
    GOLDEN_POSITIONS,
    GOLDEN_RESAMPLES,
    GOLDEN_SEED,
    MEAN_MICRO_A,
    MEAN_SUMS_A,
)
from tests.support.no_network import no_network  # noqa: F401 - the fixture of the mark

pytestmark = pytest.mark.usefixtures("no_network")

PUBLISHED_HIGH_BITS = (0xE220A839, 0x6E789E6A, 0x06C45D18)


def mean_rows(micro: tuple[int, ...]) -> np.ndarray:
    return np.array([[score, 1] for score in micro], dtype=np.int64)


# ---- the index rule -----------------------------------------------------------------------


def test_the_constants_are_the_ones_of_the_spec() -> None:
    assert RNG_NAME == "splitmix64-counter/1"
    assert BOOTSTRAP_SEED == GOLDEN_SEED == 20261110
    assert DEFAULT_RESAMPLES == GOLDEN_RESAMPLES == 1000
    assert DEFAULT_ALPHA == GOLDEN_ALPHA == 0.05
    assert MIN_RESAMPLES == 100


def test_the_first_indices_of_seed_0_are_the_high_bits_of_the_published_splitmix64_outputs() -> (
    None
):
    # 0xE220A8397B1DCDAF, 0x6E789E6AA1B965F4 and 0x06C45D188009454F are the first three outputs of
    # the published splitmix64 for a state of 0: over 2^32 units the index IS the high 32 bits.
    drawn = [draw_index(0, 0, j, 2**32) for j in range(3)]

    assert drawn == [3793791033, 1853398634, 113532184]
    assert drawn == list(PUBLISHED_HIGH_BITS)


def test_the_first_indices_of_seed_0_over_a_thousand_units() -> None:
    assert [draw_index(0, 0, j, 1000) for j in range(5)] == [883, 431, 26, 970, 106]


def test_the_indices_of_the_default_seed_over_seven_units() -> None:
    def resample(b: int) -> list[int]:
        return [draw_index(GOLDEN_SEED, b, j, 7) for j in range(7)]

    assert resample(0) == [5, 2, 4, 4, 2, 3, 2]
    assert resample(1) == [0, 6, 2, 1, 2, 4, 1]
    assert resample(2) == [2, 5, 4, 5, 6, 4, 3]
    assert resample(999) == [5, 2, 6, 0, 2, 1, 4]


def test_the_indices_of_the_default_seed_over_1012_units() -> None:
    def head(b: int) -> list[int]:
        return [draw_index(GOLDEN_SEED, b, j, 1012) for j in range(5)]

    assert head(0) == [788, 407, 682, 697, 335]
    assert head(1) == [112, 510, 740, 790, 239]
    assert head(2) == [472, 156, 38, 493, 223]


def test_one_unit_is_always_the_unit_zero() -> None:
    assert [draw_index(GOLDEN_SEED, b, 0, 1) for b in range(50)] == [0] * 50


def test_two_units_over_the_first_four_resamples() -> None:
    drawn = [[draw_index(GOLDEN_SEED, b, j, 2) for j in range(2)] for b in range(4)]

    assert drawn == [[1, 0], [1, 1], [0, 1], [0, 0]]


@pytest.mark.parametrize(
    ("seed", "resample", "position", "units"),
    [
        (-1, 0, 0, 5),
        (2**64, 0, 0, 5),
        (0, -1, 0, 5),
        (0, 0, -1, 5),
        (0, 0, 5, 5),
        (0, 0, 0, 0),
        (0, 0, 0, 2**32 + 1),
    ],
)
def test_draw_index_refuses_what_is_not_a_cell_of_the_stream(
    seed: int, resample: int, position: int, units: int
) -> None:
    with pytest.raises(ValueError):
        draw_index(seed, resample, position, units)


def test_the_largest_seed_and_the_largest_unit_count_are_accepted() -> None:
    assert 0 <= draw_index(2**64 - 1, 0, 0, 2**32) < 2**32


# ---- resample_indices ---------------------------------------------------------------------


def test_resample_indices_equal_draw_index_cell_for_cell_on_40_by_1012() -> None:
    drawn = resample_indices(GOLDEN_SEED, 0, 40, 1012)

    assert drawn.shape == (40, 1012)
    assert drawn.dtype == np.int64
    expected = [
        [draw_index(GOLDEN_SEED, b, j, 1012) for j in range(1012)] for b in range(40)
    ]
    assert drawn.tolist() == expected


def test_resample_indices_equal_draw_index_when_the_first_resample_is_not_zero() -> (
    None
):
    drawn = resample_indices(GOLDEN_SEED, 17, 23, 31)

    assert drawn.shape == (6, 31)
    assert drawn.tolist() == [
        [draw_index(GOLDEN_SEED, b, j, 31) for j in range(31)] for b in range(17, 23)
    ]


def test_every_index_is_in_range() -> None:
    for units in (1, 2, 3, 7, 1012):
        drawn = resample_indices(GOLDEN_SEED, 0, 25, units)

        assert int(drawn.min()) >= 0
        assert int(drawn.max()) < units


def test_a_resample_does_not_depend_on_the_chunk_it_is_drawn_in() -> None:
    whole = resample_indices(GOLDEN_SEED, 0, 30, 53)

    middle = resample_indices(GOLDEN_SEED, 11, 19, 53)

    assert middle.tolist() == whole[11:19].tolist()


def test_no_resample_gives_an_empty_array_of_the_right_width() -> None:
    drawn = resample_indices(GOLDEN_SEED, 5, 5, 9)

    assert drawn.shape == (0, 9)


@pytest.mark.parametrize(
    ("seed", "first", "last", "units"),
    [
        (-1, 0, 1, 5),
        (2**64, 0, 1, 5),
        (0, -1, 1, 5),
        (0, 3, 2, 5),
        (0, 0, 1, 0),
        (0, 0, 1, 2**32 + 1),
    ],
)
def test_resample_indices_refuse_what_is_not_a_range_of_the_stream(
    seed: int, first: int, last: int, units: int
) -> None:
    with pytest.raises(ValueError):
        resample_indices(seed, first, last, units)


def test_the_draw_counts_of_1000_resamples_over_1012_units_have_no_bias_beyond_chance() -> (
    None
):
    units = 1012
    drawn = resample_indices(GOLDEN_SEED, 0, 1000, units)

    counts = np.bincount(drawn.ravel(), minlength=units)

    # Each unit is expected 1000 times; the spike measured a chi-square of 1058.938, and a chi
    # square over 1011 degrees of freedom is 1011 +- sqrt(2 * 1011).
    chi_square = float(((counts - 1000.0) ** 2 / 1000.0).sum())
    assert chi_square == pytest.approx(1058.938, abs=1e-3)
    assert abs(chi_square - 1011) < 4 * math.sqrt(2 * 1011)


# ---- resample_totals ----------------------------------------------------------------------


def test_the_totals_of_the_golden_mean_metric_at_the_positions_of_the_interval() -> (
    None
):
    totals = resample_totals(mean_rows(MEAN_MICRO_A), BootstrapSettings())

    sums = sorted(int(value) for value in totals[:, 0])
    assert totals.shape == (1000, 2)
    assert totals.dtype == np.int64
    assert (sums[GOLDEN_POSITIONS[0]], sums[GOLDEN_POSITIONS[1]]) == MEAN_SUMS_A
    # the second column counts the rows drawn: always as many as there are units
    assert set(totals[:, 1].tolist()) == {12}


def test_the_resamples_follow_the_seed() -> None:
    rows = mean_rows(MEAN_MICRO_A)

    first = resample_totals(rows, BootstrapSettings(seed=1))
    again = resample_totals(rows, BootstrapSettings(seed=1))
    other = resample_totals(rows, BootstrapSettings(seed=2))

    assert first.tolist() == again.tolist()
    assert first.tolist() != other.tolist()


def test_the_resample_count_is_the_number_of_totals() -> None:
    totals = resample_totals(mean_rows(MEAN_MICRO_A), BootstrapSettings(resamples=137))

    assert totals.shape == (137, 2)


def test_the_totals_do_not_depend_on_the_chunk_size(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rows = mean_rows(MEAN_MICRO_A)
    whole = resample_totals(rows, BootstrapSettings()).tolist()

    for cells in (12 * 7, 12, 1):
        monkeypatch.setattr(bootstrap, "CHUNK_CELLS", cells)

        assert resample_totals(rows, BootstrapSettings()).tolist() == whole


def test_the_draws_are_made_in_chunks_that_cover_every_resample_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[int, int, int]] = []
    real = bootstrap.resample_indices

    def recording(seed: int, first: int, last: int, units: int) -> np.ndarray:
        calls.append((first, last, units))
        return real(seed, first, last, units)

    monkeypatch.setattr(bootstrap, "CHUNK_CELLS", 12 * 300)
    monkeypatch.setattr(bootstrap, "resample_indices", recording)

    resample_totals(mean_rows(MEAN_MICRO_A), BootstrapSettings())

    assert calls == [(0, 300, 12), (300, 600, 12), (600, 900, 12), (900, 1000, 12)]


def test_no_unit_is_a_value_error() -> None:
    with pytest.raises(ValueError, match="unit"):
        resample_totals(np.zeros((0, 3), dtype=np.int64), BootstrapSettings())


@pytest.mark.parametrize(
    "rows",
    [
        np.zeros((3, 2), dtype=np.float64),
        np.zeros((3, 2), dtype=np.uint64),
        np.zeros((3, 2), dtype=np.int32),
        np.zeros(3, dtype=np.int64),
        np.zeros((2, 2, 2), dtype=np.int64),
        np.zeros((3, 0), dtype=np.int64),
        [[1, 2], [3, 4]],
    ],
    ids=["float", "uint64", "int32", "1-d", "3-d", "no column", "list"],
)
def test_rows_must_be_a_two_dimensional_int64_array(rows: object) -> None:
    with pytest.raises(ValueError, match="int64"):
        resample_totals(rows, BootstrapSettings())  # type: ignore[arg-type]


def test_rows_whose_sums_could_overflow_are_refused() -> None:
    rows = np.array([[2**62], [1], [1]], dtype=np.int64)

    with pytest.raises(ValueError, match="overflow"):
        resample_totals(rows, BootstrapSettings())


def test_one_row_at_the_limit_is_accepted() -> None:
    rows = np.array([[2**62]], dtype=np.int64)

    totals = resample_totals(rows, BootstrapSettings(resamples=100))

    assert set(totals[:, 0].tolist()) == {2**62}


def test_a_thousand_resamples_of_four_thousand_units_with_24_columns_run_in_seconds() -> (
    None
):
    # a runaway guard, not a benchmark: the spike measured 0.02 to 0.09 s on a quiet machine
    rows = (np.arange(4000 * 24, dtype=np.int64).reshape(4000, 24) * 7) % 101
    started = time.perf_counter()

    totals = resample_totals(rows, BootstrapSettings())

    assert time.perf_counter() - started < 3.0
    assert totals.shape == (1000, 24)


# ---- the positions of the interval --------------------------------------------------------


@pytest.mark.parametrize(
    ("resamples", "alpha", "positions"),
    [
        (1000, 0.05, (25, 974)),
        (100, 0.05, (2, 97)),
        (1000, 0.1, (50, 949)),
        (999, 0.05, (24, 974)),
        (1000, 0.07, (35, 964)),
        (1000, 0.01, (5, 994)),
        (200, 0.05, (5, 194)),
        # 200 * 0.29 / 2 is 28.999999999999996 in floats and 29 on paper: the 1e-9 is the rule
        (200, 0.29, (29, 170)),
        (360, 0.35, (63, 296)),
    ],
)
def test_the_interval_positions(
    resamples: int, alpha: float, positions: tuple[int, int]
) -> None:
    assert interval_positions(resamples, alpha) == positions


@pytest.mark.parametrize(
    ("resamples", "alpha"), [(0, 0.05), (-5, 0.05), (1000, 0.0), (1000, 0.5), (10, 0.9)]
)
def test_the_interval_positions_refuse_what_has_no_interval(
    resamples: int, alpha: float
) -> None:
    with pytest.raises(ValueError):
        interval_positions(resamples, alpha)


def test_the_bounds_of_a_thousand_ordered_values_are_their_positions() -> None:
    assert interval_bounds(range(1000), 0.05) == (25.0, 974.0)


def test_the_bounds_do_not_depend_on_the_order_of_the_values() -> None:
    shuffled = [(index * 389) % 1000 for index in range(1000)]
    assert sorted(shuffled) == list(range(1000))

    assert interval_bounds(shuffled, 0.05) == (25.0, 974.0)


def test_the_bounds_are_python_floats_even_for_numpy_values() -> None:
    low, high = interval_bounds(np.arange(1000, dtype=np.float32), 0.05)

    assert (type(low), type(high)) == (float, float)
    assert (low, high) == (25.0, 974.0)


def test_the_bounds_of_no_value_or_of_a_nan_are_a_value_error() -> None:
    with pytest.raises(ValueError):
        interval_bounds([], 0.05)
    with pytest.raises(ValueError, match="number"):
        interval_bounds([1.0, math.nan, 3.0], 0.05)


# ---- the settings -------------------------------------------------------------------------


def test_the_settings_default_to_the_spec() -> None:
    settings = BootstrapSettings()

    assert (settings.resamples, settings.seed, settings.alpha) == (1000, 20261110, 0.05)


def test_the_settings_are_frozen_and_hold_no_extra_attribute() -> None:
    settings = BootstrapSettings()

    with pytest.raises(dataclasses.FrozenInstanceError):
        settings.seed = 1  # type: ignore[misc]
    with pytest.raises((AttributeError, TypeError)):
        settings.extra = 1  # type: ignore[attr-defined]


def test_the_settings_accept_the_edges_of_their_ranges() -> None:
    settings = BootstrapSettings(resamples=100, seed=0, alpha=0.4999)
    assert settings.resamples == 100
    assert BootstrapSettings(seed=2**64 - 1).seed == 2**64 - 1
    assert BootstrapSettings(alpha=np.float64(0.1)).alpha == 0.1


@pytest.mark.parametrize(
    ("keyword", "given"),
    [
        ("resamples", 99),
        ("resamples", 0),
        ("resamples", True),
        ("resamples", 100.0),
        ("resamples", "1000"),
        ("alpha", 0.0),
        ("alpha", 0.5),
        ("alpha", -0.1),
        ("alpha", 1.0),
        ("alpha", math.nan),
        ("alpha", True),
        ("alpha", 0),
        ("alpha", "0.05"),
        ("seed", -1),
        ("seed", True),
        ("seed", 1.0),
        ("seed", 2**64),
        ("seed", "1"),
    ],
)
def test_the_settings_refuse_what_they_cannot_hold_and_name_the_field(
    keyword: str, given: object
) -> None:
    with pytest.raises(ConfigError, match=keyword):
        BootstrapSettings(**{keyword: given})  # type: ignore[arg-type]


# ---- group_rows ---------------------------------------------------------------------------


def test_ten_rows_become_five_documents_in_the_order_of_their_first_row() -> None:
    rows = [[i, 10 * i, 100 * i] for i in range(10)]

    grouped, labels = group_rows(rows, GOLDEN_DOCS)

    assert labels == ("d1", "d2", "d3", "d4", "d5")
    assert grouped.dtype == np.int64
    assert grouped.tolist() == [
        [1, 10, 100],
        [5, 50, 500],
        [9, 90, 900],
        [13, 130, 1300],
        [17, 170, 1700],
    ]


def test_groups_are_numbered_by_first_appearance_and_never_sorted() -> None:
    rows = [[1], [10], [100], [1000]]

    grouped, labels = group_rows(rows, ("b", "a", "b", "a"))

    assert labels == ("b", "a")
    assert grouped.tolist() == [[101], [1010]]


def test_a_group_that_comes_back_later_adds_to_its_first_unit() -> None:
    grouped, labels = group_rows([[1, 1], [2, 2], [4, 4], [8, 8]], ("x", "y", "z", "x"))

    assert labels == ("x", "y", "z")
    assert grouped.tolist() == [[9, 9], [2, 2], [4, 4]]


def test_without_groups_every_row_is_a_unit_labelled_by_its_position() -> None:
    grouped, labels = group_rows([[1, 2], [3, 4], [5, 6]], None)

    assert labels == ("0", "1", "2")
    assert grouped.tolist() == [[1, 2], [3, 4], [5, 6]]
    assert grouped.dtype == np.int64


def test_rows_may_be_tuples_of_tuples() -> None:
    grouped, _ = group_rows(((1, 2), (3, 4)), ("a", "a"))

    assert grouped.tolist() == [[4, 6]]


@pytest.mark.parametrize(
    ("rows", "groups"),
    [
        ([[1, 2], [3]], None),
        ([[1, 2], [3]], ("a", "b")),
        ([], None),
        ([], ()),
        ([[1], [2]], ("a",)),
        ([[1], [2]], ("a", "b", "c")),
        ([[]], None),
        ([[2**63]], None),
        ([[1.5, 2.0]], None),
        ([[True, False]], ("a",)),
        ([[2**62], [2**62]], ("a", "a")),
    ],
    ids=[
        "unequal width",
        "unequal width with groups",
        "no row",
        "no row, no group",
        "fewer groups than rows",
        "more groups than rows",
        "no column",
        "beyond int64",
        "a float cell",
        "a bool cell",
        "a sum beyond int64",
    ],
)
def test_group_rows_refuses_what_cannot_be_grouped(
    rows: list[list[int]], groups: tuple[str, ...] | None
) -> None:
    with pytest.raises(ValueError):
        group_rows(rows, groups)
