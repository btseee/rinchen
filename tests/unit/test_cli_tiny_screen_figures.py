"""The figures of the lines `rinchen tiny run` prints (D15 fix wave, finding F11, the screen side of A-I1).

The screen line of a threshold wrote its value with `format(value, ".6g")` beside the bound: a figure of
49.99999996 appeared as `50` beside `>= 50.0` with the verdict `fail`, and an elapsed time of 1200.0000001 as
`1200` beside `<= 1200`, the defect the Markdown had at four digits. The screen now uses the one rule of the
Markdown (`figure_beside_bound`) with six digits: a figure that is not on the edge of its bound reads exactly as
before, and one that is gets as many digits as show its side. The rule's own tests (four digits) are in
`test_tiny_report_markdown.py` and `tests/property/test_tiny_report_figures_property.py`.
"""

from __future__ import annotations

import copy
import math
import re
from pathlib import Path

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from rinchen.cli.tiny_output import _threshold_line, result_lines
from rinchen.tiny.context import StageResult
from rinchen.tiny.report import WALL_TIME
from rinchen.tiny.report_markdown import figure_beside_bound
from rinchen.tiny.states import PART_IDS, PartRecord, StageRecord
from rinchen.tiny.thresholds import ThresholdResult
from tests.support.tiny_report_build import build
from tests.support.tiny_report_inputs import facts_of, golden_results, replaced

LINE = re.compile(r"threshold (\S+): (\S+) \(value (\S+), bound (>=|<=) (\S+)\)")
FIGURES = st.floats(
    min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False
)
LIMITS = st.floats(min_value=0.0, max_value=1e6, allow_nan=False, allow_infinity=False)
DIRECTIONS = st.sampled_from([">=", "<="])


def meets(number: float, direction: str, limit: float) -> bool:
    return number >= limit if direction == ">=" else number <= limit


def line_of(
    threshold_id: str, value: float | None, bound: str, elapsed: float = 1.0
) -> str:
    verdict = "pass" if value is not None else "fail"
    result = ThresholdResult(threshold_id, bound, value, verdict, "a detail")
    return _threshold_line(result, elapsed)


def parsed(line: str) -> tuple[float | None, str, float]:
    match = LINE.fullmatch(line)
    assert match, line
    shown = match.group(3)
    return (
        (None if shown == "-" else float(shown)),
        match.group(4),
        float(match.group(5)),
    )


# ---- the rule with six digits ---------------------------------------------------------------------------


def test_the_rule_still_starts_at_four_digits_for_the_markdown() -> None:
    assert figure_beside_bound(99.35851234, ">= 50.0") == "99.36"
    assert figure_beside_bound(1.6221680949143363e-06, "<= 0.001") == "1.622e-06"
    assert figure_beside_bound(1.0, ">= 1.0") == "1"


def test_with_six_digits_a_figure_far_from_its_bound_reads_as_six_digits_do() -> None:
    assert figure_beside_bound(99.35851234, ">= 50.0", digits=6) == "99.3585"
    assert figure_beside_bound(1.6221680949143363e-06, "<= 0.001", digits=6) == (
        "1.62217e-06"
    )
    assert figure_beside_bound(1.0, ">= 1.0", digits=6) == "1"
    assert figure_beside_bound(112.1612, "<= 1200", digits=6) == "112.161"


@pytest.mark.parametrize(
    ("value", "bound"),
    [
        (49.99999996, ">= 50.0"),
        (49.9999996, ">= 50.0"),
        (0.9999999999, ">= 1.0"),
        (1200.0000001, "<= 1200"),
        (0.0010000004, "<= 0.001"),
        (1.0000004, "<= 1.0"),
    ],
)
def test_a_figure_a_hair_from_its_bound_is_written_with_the_digits_that_show_its_side(
    value: float, bound: str
) -> None:
    text = figure_beside_bound(value, bound, digits=6)
    comparison, _, limit = bound.partition(" ")

    assert meets(float(text), comparison, float(limit)) == meets(
        value, comparison, float(limit)
    )
    assert float(text) != float(f"{value:.6g}")


@pytest.mark.parametrize(
    ("value", "bound", "text"),
    [
        (49.99999996000001, ">= 50.0", "49.99999996"),
        (1200.0000001000002, "<= 1200", "1200.0000001"),
        (0.9999999999000001, ">= 1.0", "0.9999999999"),
    ],
)
def test_a_figure_a_hair_from_its_bound_gets_the_fewest_digits_that_show_its_side_and_not_repr(
    value: float, bound: str, text: str
) -> None:
    assert figure_beside_bound(value, bound, digits=6) == text
    assert text != repr(value)


def test_the_digits_are_a_keyword_and_a_figure_beside_its_bound_has_between_one_and_seventeen() -> (
    None
):
    with pytest.raises(TypeError):
        figure_beside_bound(1.5, ">= 1.0", 6)
    for digits in (0, -1, 18):
        with pytest.raises(ValueError, match="digits"):
            figure_beside_bound(1.5, ">= 1.0", digits=digits)
    assert figure_beside_bound(0.1 + 0.2, ">= 0.3", digits=17) == repr(0.1 + 0.2)


# ---- the screen line -------------------------------------------------------------------------------------


def test_the_three_figures_of_the_report_review_read_as_the_rule_says() -> None:
    assert line_of("train_chrf", 49.99999996, ">= 50.0").endswith(
        "(value 49.99999996, bound >= 50.0)"
    )
    assert line_of(WALL_TIME, None, "<= 1200", 1200.0000001).endswith(
        "(value 1200.0000001, bound <= 1200)"
    )
    assert line_of("placeholder_acc", 237 / 238, ">= 1.0").endswith(
        "(value 0.995798, bound >= 1.0)"
    )


def test_a_null_is_a_hyphen_and_a_whole_number_is_itself() -> None:
    assert "(value -, bound >= 1.0)" in line_of("script_purity", None, ">= 1.0")
    assert "(value 0, bound <= 0)" in line_of("stages", 0, "<= 0")


def test_the_golden_report_prints_the_figures_it_printed_before(tmp_path: Path) -> None:
    lines = result_lines(build(), 123.456, tmp_path, tmp_path / "manifest.json")

    by_id = {
        line.split()[1].rstrip(":"): line
        for line in lines
        if line.startswith("threshold ")
    }
    assert len(by_id) == 8
    assert by_id["train_chrf"] == (
        "threshold train_chrf: pass (value 99.3585, bound >= 50.0)"
    )
    assert by_id["parity_delta"].endswith("(value 1.62217e-06, bound <= 0.001)")
    assert by_id["wall_time"].endswith("(value 123.456, bound <= 1200)")


def test_a_report_whose_train_chrf_is_a_hair_under_its_floor_prints_the_side_it_is_on(
    tmp_path: Path,
) -> None:
    facts = copy.deepcopy(dict(facts_of(golden_results(), "evaluate")))
    facts["memorised_chrf"]["rus_Cyrl-khk_Cyrl"]["int8"] = 49.99999996
    record = StageRecord(
        "evaluate",
        "ok",
        detail={},
        parts=tuple(PartRecord(p, "ok") for p in PART_IDS["evaluate"]),
    )
    report = build(replaced(golden_results(), "evaluate", StageResult(record, facts)))

    lines = result_lines(report, 1.0, tmp_path, tmp_path / "manifest.json")

    assert "threshold train_chrf: fail (value 49.99999996, bound >= 50.0)" in lines


def test_the_wall_time_line_shows_the_seconds_beside_its_bound_on_the_right_side(
    tmp_path: Path,
) -> None:
    lines = result_lines(build(), 1200.0000001, tmp_path, tmp_path / "manifest.json")

    assert "threshold wall_time: pass (value 1200.0000001, bound <= 1200)" in lines


# ---- the property -----------------------------------------------------------------------------------------


@st.composite
def figures_beside_bounds(draw: st.DrawFn) -> tuple[float, str, float]:
    """A figure, a direction and a limit: the figure far from the limit, within a thousandth of it, within
    a millionth of it or a few units in the last place from it."""
    limit = draw(LIMITS)
    direction = draw(DIRECTIONS)
    how = draw(st.sampled_from(["anywhere", "near", "very near", "by ulps"]))
    if how == "anywhere":
        return draw(FIGURES), direction, limit
    if how == "near":
        return limit * (1 + draw(st.floats(-1e-3, 1e-3))), direction, limit
    if how == "very near":
        return limit * (1 + draw(st.floats(-1e-6, 1e-6))), direction, limit
    figure = limit
    for _ in range(draw(st.integers(min_value=0, max_value=3))):
        figure = math.nextafter(figure, draw(st.sampled_from([-math.inf, math.inf])))
    return figure, direction, limit


@given(figures_beside_bounds())
@settings(max_examples=3000, deadline=None)
def test_a_printed_figure_read_back_is_on_the_side_of_the_printed_bound_the_figure_is_on(
    triple: tuple[float, str, float],
) -> None:
    figure, direction, limit = triple

    shown, comparison, bound = parsed(
        line_of("train_chrf", figure, f"{direction} {limit!r}")
    )

    assert shown is not None
    assert (comparison, bound) == (direction, limit)
    assert meets(shown, direction, limit) == meets(figure, direction, limit)


@given(figures_beside_bounds())
@settings(max_examples=3000, deadline=None)
def test_a_figure_not_on_the_edge_of_its_bound_is_printed_as_six_digits_always_were(
    triple: tuple[float, str, float],
) -> None:
    figure, direction, limit = triple
    old = format(figure, ".6g")
    if meets(float(old), direction, limit) != meets(figure, direction, limit):
        return

    line = line_of("train_chrf", figure, f"{direction} {limit!r}")

    assert f"(value {old}," in line


@given(figures_beside_bounds())
@settings(max_examples=2000, deadline=None)
def test_the_seconds_of_the_wall_time_are_printed_on_the_side_of_their_bound_too(
    triple: tuple[float, str, float],
) -> None:
    figure, direction, limit = triple
    seconds = abs(figure)

    shown, _, _ = parsed(line_of(WALL_TIME, None, f"{direction} {limit!r}", seconds))

    assert shown is not None
    assert meets(shown, direction, limit) == meets(seconds, direction, limit)
