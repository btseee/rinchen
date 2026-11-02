# tests/unit/test_metrics_bleu_chrf.py
from __future__ import annotations

from rinchen.metrics.bleu_chrf import chrf


def test_chrf_identical_strings_is_100():
    assert chrf("Sain baina uu", "Sain baina uu") == 100.0


def test_chrf_returns_plain_float_in_range():
    score = chrf("Sain baina uu", "Goodbye forever")
    assert isinstance(score, float)
    assert 0.0 <= score <= 100.0


def test_chrf_dissimilar_strings_below_threshold():
    # Used by the adjudication trigger (chrf < 40) -- confirm a genuinely
    # dissimilar pair actually lands below that threshold, not just "low".
    score = chrf("Sain baina uu", "The weather today is sunny and warm")
    assert score < 40.0
