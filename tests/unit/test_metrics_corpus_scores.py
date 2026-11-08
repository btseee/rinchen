"""Tests for `rinchen.metrics.bleu_chrf.corpus_scores`: corpus-level chrF2 and BLEU
(sacrebleu defaults: character n-grams up to 6 with beta 2, BLEU with the 13a
tokeniser) and the signatures of the two metrics.

The sentence-level `chrf` has its own tests in `test_metrics_bleu_chrf.py`.
"""

from __future__ import annotations

import dataclasses
from types import SimpleNamespace

import pytest
import sacrebleu

from rinchen.metrics import bleu_chrf
from rinchen.metrics.bleu_chrf import CorpusScores, corpus_scores

# Four words or more, so that BLEU has a 4-gram to count.
CYRILLIC_HYPOTHESES = [
    "Монгол улсын нийслэл Улаанбаатар хот юм",
    "Би маргааш найзтайгаа уулзана",
    "Өнөөдөр цаг агаар сайхан байна",
]
CYRILLIC_REFERENCES = [
    "Монгол улсын нийслэл Улаанбаатар хот мөн",
    "Би маргааш найзтайгаа уулзах болно",
    "Өнөөдөр цаг агаар үнэхээр сайхан байна",
]


def test_identical_text_scores_100_on_both_metrics() -> None:
    text = ["the cat sat on the mat today", "a dog barked at the moon last night"]

    scores = corpus_scores(text, text)

    assert scores.chrf == pytest.approx(100.0)
    assert scores.bleu == pytest.approx(100.0)


def test_identical_text_never_scores_above_100() -> None:
    # sacrebleu's BLEU of identical text is 100.00000000000004 (float rounding).
    text = ["the cat sat on the mat today", "a dog barked at the moon last night"]

    scores = corpus_scores(text, text)

    assert scores.bleu == 100.0
    assert scores.chrf == 100.0


class _FixedScore:
    """A sacrebleu metric that scores every corpus `score`."""

    def __init__(self, score: float) -> None:
        self._score = score

    def corpus_score(self, hypotheses: object, references: object) -> SimpleNamespace:
        return SimpleNamespace(score=self._score)

    def get_signature(self) -> str:
        return "fixed"


@pytest.mark.parametrize(
    ("reported", "returned"),
    [(100.00000000000004, 100.0), (100.0, 100.0), (73.5, 73.5), (0.0, 0.0)],
)
def test_each_metric_is_clamped_to_100_and_otherwise_returned_as_sacrebleu_has_it(
    monkeypatch: pytest.MonkeyPatch, reported: float, returned: float
) -> None:
    # sacrebleu can report a hair above 100; no real input is known to do it for
    # chrF, so the clamp of each metric is pinned with a stand-in for sacrebleu's.
    monkeypatch.setattr(bleu_chrf, "CHRF", lambda: _FixedScore(reported))
    monkeypatch.setattr(bleu_chrf, "BLEU", lambda: _FixedScore(reported))

    scores = corpus_scores(["a"], ["a"])

    assert scores.chrf == returned
    assert scores.bleu == returned


def test_a_corpus_whose_references_are_all_empty_scores_zero() -> None:
    # sacrebleu has no n-gram to count, so the score is 0.0 even for empty
    # hypotheses, which a reader might expect to match.
    for hypotheses in (["a b c d"], [""]):
        scores = corpus_scores(hypotheses, [""])

        assert scores.chrf == 0.0
        assert scores.bleu == 0.0


def test_identical_cyrillic_text_scores_100_on_both_metrics() -> None:
    scores = corpus_scores(CYRILLIC_REFERENCES, CYRILLIC_REFERENCES)

    assert scores.chrf == pytest.approx(100.0)
    assert scores.bleu == pytest.approx(100.0)


def test_text_with_nothing_in_common_scores_zero_on_both_metrics() -> None:
    scores = corpus_scores(["aaa bbb ccc ddd eee"], ["xyz uvw rst qpo nml"])

    assert scores.chrf == 0.0
    assert scores.bleu == 0.0


def test_one_word_changed_in_a_sentence_of_seven_scores_the_hand_computed_bleu() -> (
    None
):
    scores = corpus_scores(
        ["the cat sat on the mat today"], ["the cat sat on a mat today"]
    )

    # Matching n-grams over those of the hypothesis: 6/7, 4/6, 2/5, 1/4; the lengths
    # are equal, so there is no brevity penalty: exp((ln 6/7 + ln 4/6 + ln 2/5 +
    # ln 1/4) / 4) = 0.48892.
    assert scores.bleu == pytest.approx(48.89, abs=0.005)
    # chrF2 as sacrebleu 2.6.0 reports it for the same pair.
    assert scores.chrf == pytest.approx(77.087, abs=0.0005)


def test_the_scores_are_those_of_sacrebleus_corpus_functions() -> None:
    scores = corpus_scores(CYRILLIC_HYPOTHESES, CYRILLIC_REFERENCES)

    assert (
        scores.chrf
        == sacrebleu.corpus_chrf(CYRILLIC_HYPOTHESES, [CYRILLIC_REFERENCES]).score
    )
    assert (
        scores.bleu
        == sacrebleu.corpus_bleu(CYRILLIC_HYPOTHESES, [CYRILLIC_REFERENCES]).score
    )
    assert scores.chrf == pytest.approx(82.864, abs=0.0005)
    assert scores.bleu == pytest.approx(51.549, abs=0.0005)


def test_the_scores_are_for_the_corpus_and_not_the_mean_of_the_sentences() -> None:
    hypotheses = ["the cat sat on the mat today", "a b c d e"]
    references = ["the cat sat on a mat today", "a b c d e"]

    scores = corpus_scores(hypotheses, references)

    # The n-gram counts are added up before the precisions are taken: 6+5 of 7+5
    # unigrams, 4+4 of 6+4, 2+3 of 5+3, 1+2 of 4+2: exp((ln 11/12 + ln 8/10 +
    # ln 5/8 + ln 3/6) / 4) = 0.6919; the mean of the two sentence scores (48.9 and
    # 100) would be 74.4.
    assert scores.bleu == pytest.approx(69.19, abs=0.01)


def test_hypotheses_may_be_empty_strings() -> None:
    scores = corpus_scores(["", "a b c d"], ["x y z w", "a b c d"])

    # BLEU: every n-gram of the hypothesis (4 words) is right, so the precisions are
    # 1; the reference has 8 words, so the brevity penalty is exp(1 - 8/4) = 0.3679.
    assert scores.bleu == pytest.approx(36.788, abs=0.001)
    # chrF2: the precision of every n is 1 and the recall 1/2 (4 of 8 characters, 3
    # of 6 bigrams, ...), so F2 = 5 * 1 * 0.5 / (4 * 1 + 0.5) = 0.5556.
    assert scores.chrf == pytest.approx(55.556, abs=0.001)


def test_all_hypotheses_empty_scores_zero() -> None:
    scores = corpus_scores(["", ""], ["x y z w", "a b c d"])

    assert (scores.chrf, scores.bleu) == (0.0, 0.0)


def test_the_scores_are_plain_floats() -> None:
    scores = corpus_scores(["a b c d"], ["a b c d"])

    assert type(scores.chrf) is float
    assert type(scores.bleu) is float


def test_the_signatures_name_the_default_chrf2_and_the_13a_bleu() -> None:
    scores = corpus_scores(CYRILLIC_HYPOTHESES, CYRILLIC_REFERENCES)

    # chrF: character n-grams up to 6, word n-grams 0 (chrF, not chrF++), spaces
    # ignored, case kept, one reference; BLEU: 13a tokeniser, exponential smoothing.
    assert scores.chrf_signature.startswith(
        "nrefs:1|case:mixed|eff:yes|nc:6|nw:0|space:no|version:"
    )
    assert scores.bleu_signature.startswith(
        "nrefs:1|case:mixed|eff:no|tok:13a|smooth:exp|version:"
    )
    assert scores.chrf_signature.endswith(sacrebleu.__version__)
    assert scores.bleu_signature.endswith(sacrebleu.__version__)


def test_the_signatures_do_not_depend_on_the_text_and_do_not_change_between_calls() -> (
    None
):
    first = corpus_scores(CYRILLIC_HYPOTHESES, CYRILLIC_REFERENCES)
    second = corpus_scores(["a b c d"], ["a b c e"])
    third = corpus_scores(CYRILLIC_HYPOTHESES, CYRILLIC_REFERENCES)

    assert first.chrf_signature == second.chrf_signature == third.chrf_signature
    assert first.bleu_signature == second.bleu_signature == third.bleu_signature
    assert first == third


def test_the_result_is_a_frozen_record_of_four_fields() -> None:
    scores = corpus_scores(["a b c d"], ["a b c d"])

    assert isinstance(scores, CorpusScores)
    assert [field.name for field in dataclasses.fields(scores)] == [
        "chrf",
        "bleu",
        "chrf_signature",
        "bleu_signature",
    ]
    with pytest.raises(dataclasses.FrozenInstanceError):
        scores.chrf = 0.0  # type: ignore[misc]


def test_a_different_number_of_hypotheses_and_references_is_an_error() -> None:
    with pytest.raises(ValueError, match="2 hypotheses.*1 references"):
        corpus_scores(["a b c d", "e f g h"], ["a b c d"])


def test_no_sentences_at_all_is_an_error() -> None:
    with pytest.raises(ValueError, match="at least one"):
        corpus_scores([], [])
