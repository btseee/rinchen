"""Property tests for `rinchen.mbr.select` with the real chrF utility over
short random texts (empty strings and duplicates included): one in-range index
per segment, the same answer on every call, the inputs left untouched, and a
winner that has the highest expected utility, the lowest index on a tie. The
expected utilities are computed here from `chrf` directly. The hypothesis
profile is loaded in `tests/conftest.py`."""

from __future__ import annotations

from statistics import fmean

from hypothesis import given
from hypothesis import strategies as st

from rinchen.mbr.mbr import select
from rinchen.mbr.utility import ChrfUtility
from rinchen.metrics.bleu_chrf import chrf

_TEXT = st.text(alphabet="abc ", max_size=8)
_SEGMENT = st.lists(_TEXT, min_size=1, max_size=4)


@st.composite
def _inputs(
    draw: st.DrawFn,
) -> tuple[list[list[str]], list[list[str]] | None]:
    cands = draw(st.lists(_SEGMENT, max_size=4))
    if draw(st.booleans()):
        return cands, None
    refs = [draw(st.lists(_TEXT, min_size=1, max_size=3)) for _ in cands]
    return cands, refs


def _expected_utilities(
    segment: list[str], references: list[str] | None
) -> list[float]:
    expected = []
    for index, candidate in enumerate(segment):
        if references is None:
            pool = [other for i, other in enumerate(segment) if i != index]
        else:
            pool = references
        expected.append(fmean(chrf(candidate, reference) for reference in pool))
    return expected


@given(_inputs())
def test_select_returns_one_index_in_range_per_segment(
    inputs: tuple[list[list[str]], list[list[str]] | None],
) -> None:
    cands, refs = inputs
    winners = select(cands, ChrfUtility(), refs)
    assert len(winners) == len(cands)
    assert all(
        0 <= winner < len(segment)
        for winner, segment in zip(winners, cands, strict=True)
    )


@given(_inputs())
def test_select_is_deterministic_and_leaves_its_inputs_alone(
    inputs: tuple[list[list[str]], list[list[str]] | None],
) -> None:
    cands, refs = inputs
    cands_before = [list(segment) for segment in cands]
    refs_before = None if refs is None else [list(segment) for segment in refs]
    assert select(cands, ChrfUtility(), refs) == select(cands, ChrfUtility(), refs)
    assert cands == cands_before
    assert refs == refs_before


@given(_inputs())
def test_select_returns_the_first_candidate_of_highest_expected_utility(
    inputs: tuple[list[list[str]], list[list[str]] | None],
) -> None:
    cands, refs = inputs
    winners = select(cands, ChrfUtility(), refs)
    for i, (segment, winner) in enumerate(zip(cands, winners, strict=True)):
        if len(segment) == 1:
            assert winner == 0
            continue
        expected = _expected_utilities(segment, None if refs is None else refs[i])
        assert expected[winner] == max(expected)
        assert all(value < expected[winner] for value in expected[:winner])
