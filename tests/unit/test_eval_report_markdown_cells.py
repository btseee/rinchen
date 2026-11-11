"""The three cells that keep a figure honest (D13 fix wave 1: I3, I4, M6), as functions.

`status_cell` counts the outputs that were `ok` and names the others; `source_cells` says which
dataset and which revision a suite was built from, or that it has no source record; `verdict` turns
a p-value, a delta and the direction of the metric into `better`, `worse` or `no significant
difference`, by the p-value and never by the interval.
"""

from __future__ import annotations

import pytest

from rinchen.eval.report_markdown_cells import source_cells, status_cell, verdict

REVISION = "0123456789abcdef0123456789abcdef01234567"


@pytest.mark.parametrize(
    ("statuses", "segments", "cell"),
    [
        ({"ok": 10}, 10, "10 of 10"),
        ({"empty": 1, "ok": 6, "truncated": 3}, 10, "6 of 10 (1 empty, 3 truncated)"),
        ({"empty": 10}, 10, "0 of 10 (10 empty)"),
        (
            {"empty": 1, "no_output": 2, "ok": 1, "too_long": 3, "truncated": 4},
            11,
            "1 of 11 (1 empty, 2 no_output, 3 too_long, 4 truncated)",
        ),
        ({"truncated": 4, "ok": 7, "empty": 1}, 12, "7 of 12 (1 empty, 4 truncated)"),
    ],
)
def test_status_cell_counts_ok_of_the_segments_and_names_the_others_sorted(
    statuses: dict[str, int], segments: int, cell: str
) -> None:
    assert status_cell(statuses, segments) == cell


@pytest.mark.parametrize(
    ("dataset", "revision", "pinned", "cells"),
    [
        (None, None, False, ("n/a (no source record)", "n/a (no source record)")),
        ("org/data", None, False, ("org/data", "unpinned")),
        ("org/data", REVISION, True, ("org/data", REVISION)),
        ("org/data", REVISION, False, ("org/data", f"{REVISION} (unpinned)")),
    ],
)
def test_source_cells_say_the_dataset_and_the_revision_or_why_not(
    dataset: str | None, revision: str | None, pinned: bool, cells: tuple[str, str]
) -> None:
    assert source_cells(dataset, revision, pinned) == cells


@pytest.mark.parametrize(
    ("p_value", "delta", "higher", "expected"),
    [
        (0.002, 5.0, True, "better"),
        (0.002, -5.0, True, "worse"),
        (0.002, 5.0, False, "worse"),
        (0.002, -5.0, False, "better"),
        (0.049, 0.1, True, "better"),
        (0.05, 5.0, True, "no significant difference"),
        (0.3, 5.0, True, "no significant difference"),
        (0.3, -5.0, True, "no significant difference"),
        (1.0, 0.0, True, "no significant difference"),
        (0.0, 0.0, True, "no significant difference"),
        (None, 5.0, True, "n/a (fewer than two units)"),
        (None, -5.0, False, "n/a (fewer than two units)"),
    ],
)
def test_the_verdict_is_the_p_value_against_alpha_with_the_difference_in_the_right_direction(
    p_value: float | None, delta: float, higher: bool, expected: str
) -> None:
    assert verdict(p_value, delta, higher, 0.05) == expected
