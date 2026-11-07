"""Minimum Bayes risk selection: a planted candidate set whose winner is
computed by hand with chrF, the tie-break, the self-skip rule, explicit
pseudo-references, the errors for malformed input and the errors for utilities
that are NaN, infinite or too large to average."""

from __future__ import annotations

import math
from collections.abc import Callable, Iterator
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


def _nan_at(*nan_pairs: tuple[str, str]) -> StubUtility:
    """A utility that scores 1.0 except for the given (hyp, ref) pairs, which are NaN."""
    return StubUtility(lambda hyp, ref: math.nan if (hyp, ref) in nan_pairs else 1.0)


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


# --- ties that only exact arithmetic sees -------------------------------------


@pytest.mark.parametrize(
    ("scores_p", "scores_q"),
    [([0.1, 0.2, 0.3], [0.3, 0.2, 0.1]), ([0.3, 0.2, 0.1], [0.1, 0.2, 0.3])],
)
def test_select_ties_on_the_same_scores_added_in_another_order(
    scores_p: list[float], scores_q: list[float]
) -> None:
    # Added left to right, 0.1 + 0.2 + 0.3 is 0.6000000000000001 and
    # 0.3 + 0.2 + 0.1 is 0.6: the two candidates still tie, so index 0 wins.
    references = ["r1", "r2", "r3"]
    table = {
        **{("p", ref): score for ref, score in zip(references, scores_p, strict=True)},
        **{("q", ref): score for ref, score in zip(references, scores_q, strict=True)},
    }
    utility = StubUtility(lambda hyp, ref: table[(hyp, ref)])
    assert select([["p", "q"]], utility, [references]) == [0]


# --- utilities that are NaN, infinite or too large -----------------------------

INF = math.inf


@pytest.mark.parametrize(
    ("nan_pair", "candidate"),
    [(("a", "b"), 0), (("b", "c"), 1), (("c", "a"), 2)],
)
def test_select_rejects_a_nan_utility_among_the_candidates(
    nan_pair: tuple[str, str], candidate: int
) -> None:
    cands = [["x", "y"], ["a", "b", "c"]]
    with pytest.raises(ValueError) as caught:
        select(cands, _nan_at(nan_pair))
    assert "segment 1" in str(caught.value)
    assert f"candidate {candidate}" in str(caught.value)


@pytest.mark.parametrize(
    ("nan_pair", "candidate"),
    [(("a", "s"), 0), (("b", "r"), 1), (("c", "r"), 2)],
)
def test_select_rejects_a_nan_utility_against_pseudo_references(
    nan_pair: tuple[str, str], candidate: int
) -> None:
    cands = [["x", "y"], ["a", "b", "c"]]
    refs = [["r"], ["r", "s"]]
    with pytest.raises(ValueError) as caught:
        select(cands, _nan_at(nan_pair), refs)
    assert "segment 1" in str(caught.value)
    assert f"candidate {candidate}" in str(caught.value)


def test_select_names_the_segment_and_the_candidate_of_a_nan_utility() -> None:
    cands = [["a", "b"]] * 3 + [["k", "l", "m"]]
    with pytest.raises(ValueError) as caught:
        select(cands, _nan_at(("m", "k")))
    assert "segment 3" in str(caught.value)
    assert "candidate 2" in str(caught.value)
    assert "'m'" in str(caught.value)


def test_select_shows_at_most_the_first_60_characters_of_the_candidate() -> None:
    long_text = "x" * 100
    with pytest.raises(ValueError) as caught:
        select([["a", long_text]], _nan_at((long_text, "a")))
    assert "x" * 60 in str(caught.value)
    assert "x" * 61 not in str(caught.value)
    with pytest.raises(ValueError) as caught:
        select([["the cat", "b"]], _nan_at(("the cat", "b")))
    assert "the cat" in str(caught.value)


def test_select_rejects_infinities_of_both_signs_with_the_original_as_cause() -> None:
    def mixed(hyp: str, ref: str) -> float:
        return INF if ref == "r1" else -INF

    with pytest.raises(ValueError) as caught:
        select([["a", "b"], ["c", "d"]], StubUtility(mixed), [["r1", "r2"]] * 2)
    assert "segment 0" in str(caught.value)
    assert "candidate 0" in str(caught.value)
    assert isinstance(caught.value.__cause__, ValueError)


def test_select_rejects_infinities_of_both_signs_among_the_candidates() -> None:
    def mixed(hyp: str, ref: str) -> float:
        return INF if ref == "b" else -INF

    with pytest.raises(ValueError) as caught:
        select([["a", "b", "c"]], StubUtility(mixed))
    assert "segment 0" in str(caught.value)
    assert "candidate 0" in str(caught.value)
    assert isinstance(caught.value.__cause__, ValueError)


@pytest.mark.parametrize("pseudo", [False, True])
def test_select_rejects_a_sum_beyond_the_float_range_with_the_cause(
    pseudo: bool,
) -> None:
    utility = StubUtility(lambda hyp, ref: 1.7e308)
    with pytest.raises(ValueError) as caught:
        if pseudo:
            select([["a", "b"]], utility, [["r1", "r2"]])
        else:
            select([["a", "b", "c"]], utility)
    assert "segment 0" in str(caught.value)
    assert "candidate 0" in str(caught.value)
    assert isinstance(caught.value.__cause__, OverflowError)


def test_select_still_averages_large_finite_utilities() -> None:
    scores = {"a": [1e300, 2e300, 3e300], "b": [3e300, 3e300, 3e300]}
    results: Iterator[float] = iter(scores["a"] + scores["b"])
    utility = StubUtility(lambda hyp, ref: next(results))
    assert select([["a", "b"]], utility, [["r1", "r2", "r3"]]) == [1]


def test_select_ranks_an_infinite_expected_utility_beyond_the_finite_ones() -> None:
    utility = StubUtility(lambda hyp, ref: INF if hyp == "b" else 5.0)
    assert select([["a", "b", "c"]], utility) == [1]
    assert select([["a", "b", "c"]], utility, [["r1", "r2"]]) == [1]


# --- the wrong kinds of input ---------------------------------------------------


def test_select_rejects_a_flat_list_of_strings_naming_the_segment() -> None:
    utility = StubUtility()
    with pytest.raises(TypeError, match="segment 0 of cands") as caught:
        select(["the cat sat", "the cat sat down"], utility)  # type: ignore[list-item]
    assert "string" in str(caught.value)
    assert utility.pairs == []


def test_select_rejects_a_string_as_a_later_segment() -> None:
    with pytest.raises(TypeError, match="segment 1 of cands"):
        select([["a", "b"], "cd"], StubUtility())  # type: ignore[list-item]


@pytest.mark.parametrize("refs", [["abc"], [["r"], "abc"]])
def test_select_rejects_a_string_as_a_pseudo_reference_segment(
    refs: list[object],
) -> None:
    cands = [["only"], ["a", "b"]][: len(refs)]
    utility = StubUtility()
    index = len(refs) - 1
    with pytest.raises(TypeError, match=f"segment {index} of pseudo_refs"):
        select(cands, utility, refs)  # type: ignore[arg-type]
    assert utility.pairs == []


def test_select_rejects_a_string_for_the_whole_of_cands_or_pseudo_refs() -> None:
    with pytest.raises(TypeError, match="cands"):
        select("ab", StubUtility())  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="pseudo_refs"):
        select([["a", "b"]], StubUtility(), "r")  # type: ignore[arg-type]


def test_select_rejects_a_generator_for_cands_instead_of_returning_nothing() -> None:
    with pytest.raises(TypeError, match="cands"):
        select((segment for segment in [["a", "b"]]), StubUtility())  # type: ignore[arg-type]


def test_select_rejects_a_generator_for_pseudo_refs() -> None:
    with pytest.raises(TypeError, match="pseudo_refs"):
        select([["a", "b"]], StubUtility(), (ref for ref in [["r"]]))  # type: ignore[arg-type]


def test_select_rejects_a_generator_as_a_segment() -> None:
    with pytest.raises(TypeError, match="segment 0 of cands"):
        select([(c for c in "ab")], StubUtility())  # type: ignore[list-item]
