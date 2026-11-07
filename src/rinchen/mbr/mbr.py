"""Minimum Bayes risk (MBR) selection: of several candidate translations of one
source, pick the one that agrees most, on average, with the others, or with
separately supplied pseudo-references, under a `Utility`.

Per segment of `n` candidates and `m` references the utility is called `n * m`
times; with the candidates as their own references that is `n * (n - 1)` calls,
so the cost is O(n^2) per segment. That suits the handful of candidates a
generator returns, not hundreds.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from statistics import fmean

from rinchen.mbr.utility import Utility

_SHOWN_CHARACTERS = 60


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
    returns 0 and the utility is not called for it. Pure and deterministic when
    the utility is.

    Raises, before the utility is called at all: `TypeError` for `cands` or
    `pseudo_refs` that is not a sequence (a generator) or is a string, and for
    a segment that is not a sequence (a string, which is what a flat list of
    strings holds); `ValueError` for a segment without candidates, for
    `pseudo_refs` that do not have one entry per segment, and for a segment of
    several candidates whose reference list is empty.

    Raises `ValueError` during scoring, so after the earlier segments were
    scored, for a candidate whose expected utility is NaN or whose utilities
    cannot be averaged (+inf with -inf, or a sum beyond the float range). The
    message names the segment, the candidate index and the first 60 characters
    of its text, and an error from the averaging is its `__cause__`. An
    expected utility of +inf or -inf is accepted and ranks as the extreme it is.
    """
    _check_inputs(cands, pseudo_refs)
    winners = []
    for index, candidates in enumerate(cands):
        if len(candidates) == 1:
            winners.append(0)
        elif pseudo_refs is None:
            winners.append(_best_against_the_others(index, candidates, utility))
        else:
            winners.append(
                _best_against(index, candidates, pseudo_refs[index], utility)
            )
    return winners


def _check_inputs(
    cands: Sequence[Sequence[str]], pseudo_refs: Sequence[Sequence[str]] | None
) -> None:
    _require_sequence(cands, "cands", "segments")
    if pseudo_refs is not None:
        _require_sequence(pseudo_refs, "pseudo_refs", "segments")
        if len(pseudo_refs) != len(cands):
            raise ValueError(
                f"pseudo_refs has {len(pseudo_refs)} segments "
                f"but cands has {len(cands)}"
            )
    for index, candidates in enumerate(cands):
        _require_sequence(candidates, f"segment {index} of cands", "strings")
        if pseudo_refs is not None:
            _require_sequence(
                pseudo_refs[index], f"segment {index} of pseudo_refs", "strings"
            )
        if not candidates:
            raise ValueError(f"segment {index} has no candidates")
        if pseudo_refs is not None and len(candidates) > 1 and not pseudo_refs[index]:
            raise ValueError(f"segment {index} has no references")


def _require_sequence(value: object, what: str, items: str) -> None:
    if isinstance(value, str):
        raise TypeError(f"{what} must be a sequence of {items}, not a string")
    if not isinstance(value, Sequence):
        raise TypeError(
            f"{what} must be a sequence of {items}, not a {type(value).__name__}"
        )


def _best_against_the_others(
    segment: int, candidates: Sequence[str], utility: Utility
) -> int:
    expected = []
    for index, candidate in enumerate(candidates):
        others = [*candidates[:index], *candidates[index + 1 :]]
        expected.append(_expected_utility(segment, index, candidate, others, utility))
    return _first_maximum(expected)


def _best_against(
    segment: int,
    candidates: Sequence[str],
    references: Sequence[str],
    utility: Utility,
) -> int:
    expected = [
        _expected_utility(segment, index, candidate, references, utility)
        for index, candidate in enumerate(candidates)
    ]
    return _first_maximum(expected)


def _expected_utility(
    segment: int,
    index: int,
    candidate: str,
    references: Sequence[str],
    utility: Utility,
) -> float:
    """The mean utility of `candidate` over `references`. Raises `ValueError`
    for a NaN mean or utilities that cannot be averaged, with the position of
    the candidate in the message."""
    scores = [utility.score(candidate, reference) for reference in references]
    try:
        expected = fmean(scores)
    except (ValueError, OverflowError) as error:
        raise ValueError(
            f"{_describe(segment, index, candidate)} has utilities that "
            f"cannot be averaged: {error}"
        ) from error
    if math.isnan(expected):
        raise ValueError(
            f"{_describe(segment, index, candidate)} has a NaN expected utility"
        )
    return expected


def _describe(segment: int, index: int, candidate: str) -> str:
    shown = candidate[:_SHOWN_CHARACTERS]
    if len(candidate) > _SHOWN_CHARACTERS:
        shown += "..."
    return f"segment {segment}, candidate {index} ({shown!r})"


def _first_maximum(values: Sequence[float]) -> int:
    best = 0
    for index in range(1, len(values)):
        if values[index] > values[best]:
            best = index
    return best
