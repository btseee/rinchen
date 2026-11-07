"""Metrics that score a script conversion against reference spellings."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ConversionScores:
    """Character error rate and exact-match share over `n` converted words."""

    cer: float
    word_accuracy: float
    n: int


def edit_distance(a: str, b: str) -> int:
    """Levenshtein distance between `a` and `b`, counted over code points."""
    previous = list(range(len(b) + 1))
    for row, char_a in enumerate(a, start=1):
        current = [row]
        for column, char_b in enumerate(b, start=1):
            current.append(
                min(
                    previous[column] + 1,
                    current[column - 1] + 1,
                    previous[column - 1] + (char_a != char_b),
                )
            )
        previous = current
    return previous[-1]


def _require_paired(references: Sequence[str], hypotheses: Sequence[str]) -> None:
    if len(references) != len(hypotheses):
        raise ValueError(
            f"{len(references)} references but {len(hypotheses)} hypotheses"
        )
    if not references:
        raise ValueError("references and hypotheses are empty")


def cer(references: Sequence[str], hypotheses: Sequence[str]) -> float:
    """Total edits over total reference characters, not the mean of per-word rates.

    Raises ValueError when the sequences differ in length, are empty, or the
    references hold no character at all: the rate is undefined, not zero.
    """
    _require_paired(references, hypotheses)
    total_characters = sum(len(reference) for reference in references)
    if total_characters == 0:
        raise ValueError("the references have no characters")
    total_edits = sum(
        edit_distance(reference, hypothesis)
        for reference, hypothesis in zip(references, hypotheses, strict=True)
    )
    return total_edits / total_characters


def word_accuracy(references: Sequence[str], hypotheses: Sequence[str]) -> float:
    """Share of hypotheses that equal their reference exactly."""
    _require_paired(references, hypotheses)
    exact = sum(
        reference == hypothesis
        for reference, hypothesis in zip(references, hypotheses, strict=True)
    )
    return exact / len(references)


def score_conversions(
    references: Sequence[str], hypotheses: Sequence[str]
) -> ConversionScores:
    """`cer` and `word_accuracy` of the same conversions, with their count."""
    return ConversionScores(
        cer=cer(references, hypotheses),
        word_accuracy=word_accuracy(references, hypotheses),
        n=len(references),
    )


def length_ratio_quantiles(
    pairs: Sequence[tuple[str, str]], quantiles: Sequence[float]
) -> dict[float, float]:
    """Quantiles of script length over Cyrillic length across `pairs`.

    Each pair is `(cyrillic, script)` and its ratio is `len(script) / len(cyrillic)`
    in characters. A quantile `q` in [0, 1] is read off the sorted ratios by linear
    interpolation between the two nearest ranks, the default of `numpy.quantile`.
    Raises ValueError for no pairs, a pair with an empty Cyrillic side, or a
    quantile outside [0, 1].
    """
    if not pairs:
        raise ValueError("no pairs to measure")
    for quantile in quantiles:
        if not 0.0 <= quantile <= 1.0:
            raise ValueError(f"quantile {quantile} is outside [0, 1]")
    ratios: list[float] = []
    for cyrillic, script in pairs:
        if not cyrillic:
            raise ValueError("a pair has a zero-length Cyrillic side")
        ratios.append(len(script) / len(cyrillic))
    values = np.quantile(np.asarray(ratios, dtype=np.float64), list(quantiles))
    return {
        quantile: float(value)
        for quantile, value in zip(quantiles, values, strict=True)
    }
