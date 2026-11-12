"""Property test of the figures of the tiny pipeline's Markdown report (D15 fix wave, finding A-I1).

A figure written beside its bound (the thresholds table, the robustness table beside its floor) is the
text of the fewest significant digits, four or more, that keep it on its bound's side: read back as a
number, the text compares with the bound exactly as the figure does, so a value that fails is never
written as one that meets its bound and a value that passes is never written as one that misses it; and
the text stays short, which is to say it is the four-digit one unless that one would cross the bound.
Over random figures, bounds and directions, near the bound and a few units in the last place from it
included.
"""

from __future__ import annotations

import math
import re

from hypothesis import given, settings
from hypothesis import strategies as st

from rinchen.tiny.report_markdown import figure_beside_bound, render
from tests.support.tiny_report_random import random_report

# A figure a table may hold: a share, a score, a delta or a duration, positive or not.
FIGURES = st.floats(
    min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False
)
LIMITS = st.floats(min_value=0.0, max_value=1e6, allow_nan=False, allow_infinity=False)
DIRECTIONS = st.sampled_from([">=", "<="])
SMALL_STEPS = st.integers(min_value=-3, max_value=3)
TEXT = re.compile(r"-?[0-9]+(\.[0-9]+)?(e[+-][0-9]+)?")


def meets(number: float, direction: str, limit: float) -> bool:
    return number >= limit if direction == ">=" else number <= limit


@st.composite
def figures_beside_bounds(draw: st.DrawFn) -> tuple[float, str, float]:
    """A figure, a direction and a limit, the figure far from the limit, within a thousandth of it or
    a few units in the last place from it."""
    limit = draw(LIMITS)
    direction = draw(DIRECTIONS)
    how = draw(st.sampled_from(["anywhere", "near", "by ulps"]))
    if how == "anywhere":
        return draw(FIGURES), direction, limit
    if how == "near":
        return limit * (1 + draw(st.floats(-1e-3, 1e-3))), direction, limit
    figure = limit
    for _ in range(abs(draw(SMALL_STEPS))):
        figure = math.nextafter(figure, draw(st.sampled_from([-math.inf, math.inf])))
    return figure, direction, limit


@given(figures_beside_bounds())
@settings(max_examples=3000, deadline=None)
def test_a_figure_beside_its_bound_reads_back_on_the_same_side_of_it(
    triple: tuple[float, str, float],
) -> None:
    figure, direction, limit = triple
    text = figure_beside_bound(figure, f"{direction} {limit!r}")
    assert TEXT.fullmatch(text), text
    assert meets(float(text), direction, limit) == meets(figure, direction, limit)


@given(figures_beside_bounds())
@settings(max_examples=3000, deadline=None)
def test_a_figure_is_written_with_four_digits_unless_they_would_cross_its_bound(
    triple: tuple[float, str, float],
) -> None:
    figure, direction, limit = triple
    text = figure_beside_bound(figure, f"{direction} {limit!r}")
    short = f"{figure:.4g}"
    if meets(float(short), direction, limit) == meets(figure, direction, limit):
        assert text == short
    else:
        digits = re.sub(r"e[+-][0-9]+$", "", text).replace(".", "").lstrip("0-")
        assert len(digits) > 4
        assert float(text) != float(short)


@given(figures_beside_bounds())
@settings(max_examples=2000, deadline=None)
def test_a_figure_reads_back_within_its_digits_of_what_it_is(
    triple: tuple[float, str, float],
) -> None:
    figure, direction, limit = triple
    text = figure_beside_bound(figure, f"{direction} {limit!r}")
    assert math.isclose(float(text), figure, rel_tol=5e-4, abs_tol=0.0) or (
        float(text) == figure
    )


@given(st.integers(min_value=-(10**9), max_value=10**9), DIRECTIONS, LIMITS)
@settings(max_examples=300, deadline=None)
def test_a_whole_number_is_written_as_itself(
    number: int, direction: str, limit: float
) -> None:
    assert figure_beside_bound(number, f"{direction} {limit!r}") == str(number)


@given(st.integers(min_value=0, max_value=2**32 - 1))
@settings(max_examples=300, deadline=None)
def test_every_figure_of_a_rendered_report_reads_back_on_the_side_of_its_bound(
    seed: int,
) -> None:
    report = random_report(seed)
    markdown = render(report)
    by_id = {result.id: result for result in report.thresholds}
    for line in markdown.splitlines():
        cells = [cell.strip() for cell in line.strip("|").split(" | ")]
        if len(cells) == 5 and cells[0] in by_id and cells[2] not in ("-", ""):
            result = by_id[cells[0]]
            direction, _, limit = cells[1].partition(" ")
            passed = meets(float(cells[2]), direction, float(limit))
            assert passed == (result.verdict == "pass"), (cells, result.verdict)
