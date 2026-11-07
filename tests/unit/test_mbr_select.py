"""Minimum Bayes risk selection: a planted candidate set whose winner is
computed by hand with chrF, the tie-break, the self-skip rule, explicit
pseudo-references, and the errors for malformed input."""

from __future__ import annotations

from collections.abc import Callable
from statistics import fmean

import pytest

from rinchen.mbr.mbr import select
from rinchen.mbr.utility import ChrfUtility, NegatedUtility
from rinchen.metrics.bleu_chrf import chrf

PLANTED = ["the cat sat", "the cat sat down", "a dog ran"]


class StubUtility:
    """Scores with a function (0.0 by default) and records every (hyp, ref) pair."""

    def __init__(self, fn: Callable[[str, str], float] | None = None) -> None:
        self._fn = fn if fn is not None else (lambda hyp, ref: 0.0)
        self.pairs: list[tuple[str, str]] = []

    def score(self, hyp: str, ref: str) -> float:
        self.pairs.append((hyp, ref))
        return self._fn(hyp, ref)


def _length_of_hypothesis(hyp: str, ref: str) -> float:
    return float(len(hyp))


def _self_match_scores_length(hyp: str, ref: str) -> float:
    return float(len(hyp)) if hyp == ref else 0.0


def _expected_chrf(candidate: str, references: list[str]) -> float:
    return fmean(chrf(candidate, reference) for reference in references)


# --- the planted set ---------------------------------------------------------


def test_select_picks_the_candidate_closest_to_the_others() -> None:
    others = {0: PLANTED[1:], 1: [PLANTED[0], PLANTED[2]], 2: PLANTED[:2]}
    expected = [_expected_chrf(PLANTED[i], others[i]) for i in range(3)]
    assert expected[1] > expected[0] > expected[2]  # about 50.5, 35.3, 6.2
    assert select([PLANTED], ChrfUtility()) == [1]


def test_select_with_a_negated_utility_picks_the_candidate_farthest_away() -> None:
    assert select([PLANTED], NegatedUtility(ChrfUtility())) == [2]


def test_select_returns_one_index_per_segment_in_order() -> None:
    cands = [["aa", "bbbb", "c"], ["xxx", "y"], ["p", "qq", "rrr"]]
    assert select(cands, StubUtility(_length_of_hypothesis)) == [1, 0, 2]


def test_select_of_no_segments_is_empty_and_scores_nothing() -> None:
    utility = StubUtility()
    assert select([], utility) == []
    assert utility.pairs == []


# --- one candidate -----------------------------------------------------------


def test_select_returns_zero_for_a_single_candidate_without_scoring() -> None:
    utility = StubUtility()
    assert select([["only"]], utility) == [0]
    assert select([["only"]], utility, [["r1", "r2"]]) == [0]
    assert select([["only"]], utility, [[]]) == [0]
    assert utility.pairs == []


def test_select_scores_the_multi_candidate_segments_around_a_single_one() -> None:
    cands = [["a", "bb"], ["solo"], ["c", "dd"]]
    assert select(cands, StubUtility(_length_of_hypothesis)) == [1, 0, 1]


# --- ties --------------------------------------------------------------------


def test_select_breaks_a_total_tie_toward_the_lowest_index() -> None:
    assert select([["x", "y", "z"]], StubUtility()) == [0]
    assert select([["x", "y", "z"]], StubUtility(), [["r"]]) == [0]


def test_select_breaks_a_tie_between_later_candidates_toward_the_lower_one() -> None:
    def prefers_b_and_c(hyp: str, ref: str) -> float:
        return 1.0 if hyp in ("b", "c") else 0.0

    assert select([["a", "b", "c"]], StubUtility(prefers_b_and_c)) == [1]
    assert select([["a", "b", "c"]], StubUtility(prefers_b_and_c), [["r"]]) == [1]


# --- candidates as their own references ---------------------------------------


def test_select_does_not_score_a_candidate_against_itself() -> None:
    utility = StubUtility()
    select([["x", "y"]], utility)
    assert sorted(utility.pairs) == [("x", "y"), ("y", "x")]


def test_select_compares_equal_strings_at_different_indexes() -> None:
    utility = StubUtility()
    select([["a", "b", "a"]], utility)
    assert sorted(utility.pairs) == [
        ("a", "a"),
        ("a", "a"),
        ("a", "b"),
        ("a", "b"),
        ("b", "a"),
        ("b", "a"),
    ]


def test_select_gives_a_candidate_no_credit_for_matching_itself() -> None:
    # Counting each candidate as its own reference would make "bbb" win.
    assert select([["a", "bbb", "c"]], StubUtility(_self_match_scores_length)) == [0]


def test_select_credits_a_candidate_for_a_duplicate_at_another_index() -> None:
    cands = [["a", "bbb", "bbb"]]
    assert select(cands, StubUtility(_self_match_scores_length)) == [1]


# --- explicit pseudo-references -----------------------------------------------


def test_select_scores_every_candidate_against_every_pseudo_reference() -> None:
    utility = StubUtility()
    select([["x", "y"]], utility, [["x", "r"]])
    assert sorted(utility.pairs) == [("x", "r"), ("x", "x"), ("y", "r"), ("y", "x")]


def test_select_pseudo_references_change_the_winner() -> None:
    assert select([PLANTED], ChrfUtility()) == [1]
    assert select([PLANTED], ChrfUtility(), [["the cat sat"]]) == [0]
    assert select([PLANTED], ChrfUtility(), [["the cat sat down"]]) == [1]
    assert select([PLANTED], ChrfUtility(), [["a dog ran"]]) == [2]


def test_select_averages_over_several_pseudo_references() -> None:
    references = ["the cat sat", "a dog ran"]
    expected = [_expected_chrf(candidate, references) for candidate in PLANTED]
    assert expected[0] > expected[2] > expected[1]  # about 52.3, 51.9, 50.5
    assert select([PLANTED], ChrfUtility(), [references]) == [0]


def test_select_uses_the_pseudo_references_of_the_same_segment() -> None:
    cands = [PLANTED, PLANTED, PLANTED]
    refs = [["the cat sat"], ["a dog ran"], ["the cat sat down"]]
    assert select(cands, ChrfUtility(), refs) == [0, 2, 1]


# --- errors ---------------------------------------------------------------------


@pytest.mark.parametrize("index", [0, 1, 2])
def test_select_rejects_a_segment_without_candidates(index: int) -> None:
    cands = [["a", "b"], ["c", "d"], ["e", "f"]]
    cands[index] = []
    with pytest.raises(ValueError, match=f"segment {index} "):
        select(cands, StubUtility())


@pytest.mark.parametrize(("refs_count", "cands_count"), [(7, 3), (2, 4)])
def test_select_rejects_pseudo_references_of_another_length(
    refs_count: int, cands_count: int
) -> None:
    cands = [["a", "b"]] * cands_count
    refs = [["r"]] * refs_count
    with pytest.raises(ValueError) as caught:
        select(cands, StubUtility(), refs)
    assert str(refs_count) in str(caught.value)
    assert str(cands_count) in str(caught.value)


@pytest.mark.parametrize("index", [0, 1])
def test_select_rejects_no_references_for_a_multi_candidate_segment(index: int) -> None:
    cands = [["a", "b"], ["c", "d"]]
    refs = [["r"], ["r"]]
    refs[index] = []
    with pytest.raises(ValueError, match=f"segment {index} "):
        select(cands, StubUtility(), refs)


def test_select_checks_every_segment_before_scoring_any() -> None:
    utility = StubUtility()
    with pytest.raises(ValueError):
        select([["a", "b"], []], utility)
    with pytest.raises(ValueError):
        select([["a", "b"], ["c", "d"]], utility, [["r"], []])
    with pytest.raises(ValueError):
        select([["a", "b"]], utility, [["r"], ["s"]])
    assert utility.pairs == []
