"""Property tests of the metrics from integer statistics (D13, T7): whatever the segments, the
sum of the per-segment statistics scores exactly what sacrebleu scores over the whole corpus,
the order of the segments changes nothing, and the statistics of two halves add up to those of
the whole. Under the `no_network` guard like every test that touches sacrebleu."""

from __future__ import annotations

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from sacrebleu.metrics import BLEU, CHRF

from rinchen.metrics.registry import COUNT_METRICS
from rinchen.metrics.segment_stats import (
    mean_from_statistics,
    mean_statistics,
    score_from_statistics,
    segment_statistics,
    statistics_metric,
    totals_of,
)
from tests.support.no_network import no_network  # noqa: F401 - the fixture of the mark

pytestmark = pytest.mark.usefixtures("no_network")

# The guard only patches the socket module, and the same patch holds for every example.
SETTINGS = settings(
    max_examples=60, suppress_health_check=[HealthCheck.function_scoped_fixture]
)

# ASCII with the characters the metrics treat specially: spaces, punctuation, digits, and both
# cases (the metrics are case-sensitive).
TEXT = st.text(alphabet="abcABC xyz.,!?0123456789'-", max_size=40)
WORD = st.sampled_from(
    ["the", "The", "cat", "Cat", "dog", "sat", "ran", "on", "On", "mat", "park"]
)
# At least four words, so that BLEU has a 4-gram to count.
SENTENCE = st.lists(WORD, min_size=4, max_size=9).map(" ".join)


@st.composite
def near_pair(draw: st.DrawFn) -> tuple[str, str]:
    """A hypothesis and a reference that share most words, in both cases: the pairs on which
    BLEU has n-grams to count (random text has none) and the case of a word matters."""
    reference = draw(st.lists(WORD, min_size=4, max_size=9))
    hypothesis = [draw(st.one_of(st.just(word), WORD)) for word in reference]
    hypothesis = draw(
        st.sampled_from([hypothesis, hypothesis[:-1], [*hypothesis, draw(WORD)]])
    )
    return " ".join(hypothesis), " ".join(reference)


CORPUS = st.lists(
    st.one_of(st.tuples(TEXT, TEXT), near_pair()), min_size=1, max_size=12
)


def sacrebleu_score(name: str, hypotheses: list[str], references: list[str]) -> float:
    metric = (
        BLEU() if name == "bleu" else CHRF(word_order=2 if name == "chrf_pp" else 0)
    )
    score = metric.corpus_score(hypotheses, [references]).score
    return min(100.0, float(score))


def split(pairs: list[tuple[str, str]]) -> tuple[list[str], list[str]]:
    return [hypothesis for hypothesis, _ in pairs], [
        reference for _, reference in pairs
    ]


@SETTINGS
@given(pairs=CORPUS, name=st.sampled_from(COUNT_METRICS))
def test_the_sum_of_the_statistics_scores_what_sacrebleu_scores_over_the_corpus(
    pairs: list[tuple[str, str]], name: str
) -> None:
    hypotheses, references = split(pairs)
    statistics = segment_statistics(
        statistics_metric(name, "eng_Latn"), hypotheses, references
    )

    assert score_from_statistics(
        statistics.metric, totals_of(statistics.rows)
    ) == sacrebleu_score(name, hypotheses, references)


@SETTINGS
@given(
    sentences=st.lists(SENTENCE, min_size=1, max_size=8),
    name=st.sampled_from(COUNT_METRICS),
)
def test_text_that_equals_its_reference_scores_exactly_100(
    sentences: list[str], name: str
) -> None:
    statistics = segment_statistics(
        statistics_metric(name, "eng_Latn"), sentences, sentences
    )

    assert score_from_statistics(statistics.metric, totals_of(statistics.rows)) == 100.0


@SETTINGS
@given(pairs=CORPUS, name=st.sampled_from(COUNT_METRICS))
def test_a_score_is_a_number_from_0_to_100(
    pairs: list[tuple[str, str]], name: str
) -> None:
    statistics = segment_statistics(statistics_metric(name, "eng_Latn"), *split(pairs))

    assert (
        0.0
        <= score_from_statistics(statistics.metric, totals_of(statistics.rows))
        <= 100.0
    )


@SETTINGS
@given(data=st.data(), pairs=CORPUS, name=st.sampled_from(COUNT_METRICS))
def test_shuffling_the_segments_changes_neither_the_totals_nor_the_score(
    data: st.DataObject, pairs: list[tuple[str, str]], name: str
) -> None:
    shuffled = data.draw(st.permutations(pairs))
    metric = statistics_metric(name, "eng_Latn")

    before = segment_statistics(metric, *split(pairs))
    after = segment_statistics(metric, *split(shuffled))

    assert totals_of(before.rows) == totals_of(after.rows)
    assert score_from_statistics(
        metric, totals_of(before.rows)
    ) == score_from_statistics(metric, totals_of(after.rows))


@SETTINGS
@given(
    pairs=CORPUS.filter(lambda corpus: len(corpus) >= 2),
    name=st.sampled_from(COUNT_METRICS),
)
def test_the_statistics_of_two_halves_add_up_to_those_of_the_whole(
    pairs: list[tuple[str, str]], name: str
) -> None:
    metric = statistics_metric(name, "eng_Latn")
    cut = len(pairs) // 2

    whole = totals_of(segment_statistics(metric, *split(pairs)).rows)
    first = totals_of(segment_statistics(metric, *split(pairs[:cut])).rows)
    second = totals_of(segment_statistics(metric, *split(pairs[cut:])).rows)

    assert whole == tuple(a + b for a, b in zip(first, second, strict=True))


@SETTINGS
@given(
    pairs=CORPUS, extra=TEXT, name=st.sampled_from(COUNT_METRICS), longer=st.booleans()
)
def test_a_list_longer_or_shorter_by_one_is_a_value_error(
    pairs: list[tuple[str, str]], extra: str, name: str, longer: bool
) -> None:
    hypotheses, references = split(pairs)
    metric = statistics_metric(name, "eng_Latn")
    off_by_one = [*hypotheses, extra] if longer else hypotheses[:-1]

    with pytest.raises(ValueError):
        segment_statistics(metric, off_by_one, references)


@SETTINGS
@given(
    thousandths=st.lists(st.integers(-1_000_000, 1_000_000), min_size=1, max_size=30)
)
def test_scores_of_three_decimals_keep_every_micro_unit_and_their_mean(
    thousandths: list[int],
) -> None:
    # k / 1000 * 1e6 is not always the integer k * 1000 in floats (1.001 gives 1000999.99...):
    # the units are rounded, so the sum is exact.
    scores = [count / 1000 for count in thousandths]

    totals = totals_of(mean_statistics(scores))

    assert totals == (sum(thousandths) * 1000, len(scores))
    assert mean_from_statistics(totals) == pytest.approx(sum(scores) / len(scores))
