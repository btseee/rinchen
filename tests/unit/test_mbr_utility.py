"""The MBR utilities: `ChrfUtility` wraps `rinchen.metrics.bleu_chrf.chrf` and
`NegatedUtility` flips the sign of any utility."""

from __future__ import annotations

import pytest

from rinchen.mbr.utility import ChrfUtility, NegatedUtility
from rinchen.metrics.bleu_chrf import chrf


class TableUtility:
    """Returns fixed scores per (hyp, ref) pair and records every call."""

    def __init__(self, table: dict[tuple[str, str], float]) -> None:
        self._table = table
        self.calls: list[tuple[str, str]] = []

    def score(self, hyp: str, ref: str) -> float:
        self.calls.append((hyp, ref))
        return self._table[(hyp, ref)]


def test_chrf_utility_scores_identical_strings_100() -> None:
    assert ChrfUtility().score("abc", "abc") == 100.0


def test_chrf_utility_scores_disjoint_strings_0() -> None:
    assert ChrfUtility().score("abc", "xyz") == 0.0


def test_chrf_utility_is_not_symmetric_and_takes_the_hypothesis_first() -> None:
    utility = ChrfUtility()
    assert utility.score("abc", "abcd") == pytest.approx(68.86227544910179)
    assert utility.score("abcd", "abc") == pytest.approx(89.84375)


@pytest.mark.parametrize(
    ("hyp", "ref"),
    [
        ("the cat sat", "the cat sat down"),
        ("the cat sat down", "the cat sat"),
        ("a dog ran", "the cat sat"),
        ("", "something"),
    ],
)
def test_chrf_utility_returns_what_chrf_returns(hyp: str, ref: str) -> None:
    assert ChrfUtility().score(hyp, ref) == chrf(hyp, ref)


@pytest.mark.parametrize(
    ("score", "negated"), [(3.5, -3.5), (-2.0, 2.0), (0.0, 0.0), (100.0, -100.0)]
)
def test_negated_utility_negates_the_inner_score(score: float, negated: float) -> None:
    inner = TableUtility({("hyp", "ref"): score})
    assert NegatedUtility(inner).score("hyp", "ref") == negated


def test_negated_utility_passes_hypothesis_and_reference_in_order() -> None:
    inner = TableUtility({("hyp", "ref"): 1.0})
    NegatedUtility(inner).score("hyp", "ref")
    assert inner.calls == [("hyp", "ref")]


def test_negated_utility_wraps_chrf() -> None:
    assert NegatedUtility(ChrfUtility()).score("abc", "abc") == -100.0
    assert NegatedUtility(ChrfUtility()).score("abc", "xyz") == 0.0
