"""Sentence-level chrF via sacrebleu."""

from __future__ import annotations

from sacrebleu.metrics import CHRF

_CHRF = CHRF()


def chrf(hyp: str, ref: str) -> float:
    """Sentence-level chrF2 (sacrebleu defaults). Returns a float in [0, 100]."""
    return float(_CHRF.sentence_score(hyp, [ref]).score)
