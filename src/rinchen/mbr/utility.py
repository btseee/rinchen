"""Utility metrics for minimum Bayes risk selection. A `Utility` scores a
hypothesis against a reference, higher meaning better; `ChrfUtility` is
sentence-level chrF2 and `NegatedUtility` turns a lower-is-better metric (such
as MetricX) into a higher-is-better one."""

from __future__ import annotations

from typing import Protocol

from rinchen.metrics.bleu_chrf import chrf


class Utility(Protocol):
    def score(self, hyp: str, ref: str) -> float:
        """How good `hyp` is when `ref` is taken as the truth; higher is better."""
        ...


class ChrfUtility:
    """Sentence-level chrF2, in [0, 100]. chrF weighs recall above precision,
    so `score(a, b)` and `score(b, a)` differ."""

    def score(self, hyp: str, ref: str) -> float:
        return chrf(hyp, ref)


class NegatedUtility:
    """The wrapped utility with its sign flipped."""

    def __init__(self, inner: Utility) -> None:
        self._inner = inner

    def score(self, hyp: str, ref: str) -> float:
        return -self._inner.score(hyp, ref)
