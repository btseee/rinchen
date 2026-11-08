"""chrF and BLEU via sacrebleu: sentence-level chrF, and corpus-level chrF and BLEU
with the signatures that say how they were computed."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from sacrebleu.metrics import BLEU, CHRF

_CHRF = CHRF()


def chrf(hyp: str, ref: str) -> float:
    """Sentence-level chrF2 (sacrebleu defaults). Returns a float in [0, 100]."""
    return float(_CHRF.sentence_score(hyp, [ref]).score)


@dataclass(frozen=True)
class CorpusScores:
    """Corpus-level scores in [0, 100] and the sacrebleu signature of each metric
    (the settings and the version that produced the score)."""

    chrf: float
    bleu: float
    chrf_signature: str
    bleu_signature: str


def corpus_scores(hypotheses: Sequence[str], references: Sequence[str]) -> CorpusScores:
    """Corpus chrF2 and BLEU (13a tokeniser) of `hypotheses` against one reference
    each, with sacrebleu's defaults. The n-gram counts are added up over the whole
    corpus before a score is taken, which is not the mean of the sentence scores. A
    hypothesis may be an empty string.

    Both scores are at most 100.0: sacrebleu reports 100.00000000000004 for BLEU of
    identical text, which is float rounding, and the scores are documented in [0, 100].
    A corpus whose references are all empty scores 0.0 on both metrics, whatever the
    hypotheses, since sacrebleu has no n-gram to count (even empty hypotheses do not
    count as a match).

    Raises:
        ValueError: no sentences, or a different number of hypotheses and references.
    """
    if len(hypotheses) != len(references):
        raise ValueError(
            f"{len(hypotheses)} hypotheses but {len(references)} references: "
            "scoring needs one reference per hypothesis"
        )
    if not hypotheses:
        raise ValueError("at least one sentence is needed to score a corpus")
    # A new metric each call: a signature is only known after the metric has scored,
    # and it must describe this call.
    chrf_metric = CHRF()
    bleu_metric = BLEU()
    chrf_score = chrf_metric.corpus_score(list(hypotheses), [list(references)])
    bleu_score = bleu_metric.corpus_score(list(hypotheses), [list(references)])
    return CorpusScores(
        chrf=min(100.0, float(chrf_score.score)),
        bleu=min(100.0, float(bleu_score.score)),
        chrf_signature=str(chrf_metric.get_signature()),
        bleu_signature=str(bleu_metric.get_signature()),
    )
