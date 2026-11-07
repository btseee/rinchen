"""Minimum Bayes risk (MBR) selection: of several candidate translations of one
source, pick the one that agrees most, on average, with the others, or with
separately supplied pseudo-references, under a `Utility`.

Per segment of `n` candidates and `m` references the utility is called `n * m`
times; with the candidates as their own references that is `n * (n - 1)` calls,
so the cost is O(n^2) per segment. That suits the handful of candidates a
generator returns, not hundreds.
"""

from __future__ import annotations

from collections.abc import Sequence
from statistics import fmean

from rinchen.mbr.utility import Utility


def select(
    cands: Sequence[Sequence[str]],
    utility: Utility,
    pseudo_refs: Sequence[Sequence[str]] | None = None,
) -> list[int]:
    """For each segment, the index of the candidate with the highest expected
    utility: the mean of `utility.score(candidate, reference)` over the
    segment's references.

    The references of segment `i` are `pseudo_refs[i]`, all of them scored
    against every candidate, or, without `pseudo_refs`, the other candidates of
    `cands[i]`. A candidate is never scored against its own index, but an equal
    string at another index is a reference like any other. Every candidate of a
    segment is averaged over the same number of references, so the mean ranks
    them as their sum would.

    The first of equal best candidates wins. A segment with one candidate
    returns 0 and the utility is not called for it. Raises `ValueError`, before
    the utility is called at all, for a segment without candidates, for
    `pseudo_refs` that do not have one entry per segment, and for a segment of
    several candidates whose reference list is empty. Pure and deterministic
    when the utility is.
    """
    _check_inputs(cands, pseudo_refs)
    winners = []
    for index, candidates in enumerate(cands):
        if len(candidates) == 1:
            winners.append(0)
        elif pseudo_refs is None:
            winners.append(_best_among_each_other(candidates, utility))
        else:
            winners.append(_best_against(candidates, pseudo_refs[index], utility))
    return winners


def _check_inputs(
    cands: Sequence[Sequence[str]], pseudo_refs: Sequence[Sequence[str]] | None
) -> None:
    if pseudo_refs is not None and len(pseudo_refs) != len(cands):
        raise ValueError(
            f"pseudo_refs has {len(pseudo_refs)} segments but cands has {len(cands)}"
        )
    for index, candidates in enumerate(cands):
        if not candidates:
            raise ValueError(f"segment {index} has no candidates")
        if pseudo_refs is not None and len(candidates) > 1 and not pseudo_refs[index]:
            raise ValueError(f"segment {index} has no references")


def _best_among_each_other(candidates: Sequence[str], utility: Utility) -> int:
    expected = [
        fmean(
            utility.score(candidate, other)
            for other_index, other in enumerate(candidates)
            if other_index != index
        )
        for index, candidate in enumerate(candidates)
    ]
    return _first_maximum(expected)


def _best_against(
    candidates: Sequence[str], references: Sequence[str], utility: Utility
) -> int:
    expected = [
        fmean(utility.score(candidate, reference) for reference in references)
        for candidate in candidates
    ]
    return _first_maximum(expected)


def _first_maximum(values: Sequence[float]) -> int:
    best = 0
    for index in range(1, len(values)):
        if values[index] > values[best]:
            best = index
    return best
