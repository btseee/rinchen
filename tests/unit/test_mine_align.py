"""The vecalign-style sentence aligner (B6) against planted alignments.

The planted data is built from a gold alignment of every type (1-1, 1-2, 2-1,
2-2, 1-3, 3-1 and both deletions) with the real `HashEmbedder`: a sentence is a
bag of concept words plus one or two noise words, so a span embeds as the sum
of its sentences. The F1 and recall below are computed here with plain
set arithmetic over exact `(src indexes, tgt indexes)` pairs, independent of
the aligner.
"""

from __future__ import annotations

import random
import time
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st
from numpy.typing import NDArray

from rinchen.mine.align import Alignment, align
from tests.support.embedders import HashEmbedder

Pair = tuple[tuple[int, ...], tuple[int, ...]]

SEED = 11
NOISE_POOL_SIZE = 40
SYLLABLES = (
    "ba",
    "ko",
    "ti",
    "ru",
    "sen",
    "lum",
    "dor",
    "fi",
    "na",
    "gar",
    "mi",
    "tho",
    "vel",
    "ju",
    "pra",
    "xo",
)
# (source sentences, target sentences) -> how many alignments of that shape.
PLANNED_SHAPES = {
    (1, 1): 94,
    (1, 2): 10,
    (2, 1): 10,
    (2, 2): 4,
    (1, 3): 3,
    (3, 1): 3,
    (1, 0): 6,
    (0, 1): 6,
}
SENTENCES_PER_SIDE = 150
MIN_F1 = 0.95
MIN_ONE_TO_ONE_RECALL = 0.97
MAX_DIAGONAL_BASELINE_F1 = 0.5
ALIGN_SECONDS = 2.0
DEFAULT_SKIP_COST = 0.9


@dataclass(frozen=True)
class Planted:
    src: list[str]
    tgt: list[str]
    gold: list[Pair]
    noise: frozenset[str]


class _Vocabulary:
    """Pseudo-words that are unique across the whole planted corpus."""

    def __init__(self, rng: random.Random) -> None:
        self._rng = rng
        self._used: set[str] = set()

    def fresh(self) -> str:
        while True:
            word = "".join(self._rng.choice(SYLLABLES) for _ in range(4))
            if word not in self._used:
                self._used.add(word)
                return word


def _side_sentences(
    concept: Sequence[str], count: int, noise: Sequence[str], rng: random.Random
) -> list[str]:
    """`count` sentences that share the concept words out (at least two each)
    and each add one or two noise words; no sentences for `count == 0`."""
    if count == 0:
        return []
    words = list(concept)
    rng.shuffle(words)
    sizes = [2] * count
    for _ in range(len(words) - 2 * count):
        sizes[rng.randrange(count)] += 1
    sentences = []
    start = 0
    for size in sizes:
        sentence = words[start : start + size] + rng.sample(noise, rng.randint(1, 2))
        rng.shuffle(sentence)
        sentences.append(" ".join(sentence) + ".")
        start += size
    return sentences


def build_planted_alignments() -> Planted:
    rng = random.Random(SEED)
    vocabulary = _Vocabulary(rng)
    noise = [vocabulary.fresh() for _ in range(NOISE_POOL_SIZE)]
    shapes = [shape for shape, count in PLANNED_SHAPES.items() for _ in range(count)]
    rng.shuffle(shapes)
    src: list[str] = []
    tgt: list[str] = []
    gold: list[Pair] = []
    for src_count, tgt_count in shapes:
        concept = [
            vocabulary.fresh()
            for _ in range(3 * max(src_count, tgt_count) + rng.randint(2, 4))
        ]
        gold.append(
            (
                tuple(range(len(src), len(src) + src_count)),
                tuple(range(len(tgt), len(tgt) + tgt_count)),
            )
        )
        src += _side_sentences(concept, src_count, noise, rng)
        tgt += _side_sentences(concept, tgt_count, noise, rng)
    return Planted(src=src, tgt=tgt, gold=gold, noise=frozenset(noise))


def _pairs(alignments: Sequence[Alignment]) -> list[Pair]:
    return [(alignment.src, alignment.tgt) for alignment in alignments]


def _f1(predicted: Sequence[Pair], gold: Sequence[Pair]) -> float:
    matched = len(set(predicted) & set(gold))
    if matched == 0:
        return 0.0
    precision = matched / len(predicted)
    recall = matched / len(gold)
    return 2 * precision * recall / (precision + recall)


def _recall_of_shape(
    predicted: Sequence[Pair], gold: Sequence[Pair], shape: tuple[int, int]
) -> float:
    wanted = [pair for pair in gold if (len(pair[0]), len(pair[1])) == shape]
    return len(set(predicted) & set(wanted)) / len(wanted)


def _shape(alignment: Alignment) -> tuple[int, int]:
    return len(alignment.src), len(alignment.tgt)


def _deletions(alignments: Sequence[Alignment]) -> list[Alignment]:
    return [a for a in alignments if not a.src or not a.tgt]


def _assert_partition(
    alignments: Sequence[Alignment], src_count: int, tgt_count: int, max_k: int
) -> None:
    """Every sentence of both sides appears once, in order, in consecutive runs."""
    src_cursor = tgt_cursor = 0
    for alignment in alignments:
        assert alignment.src == tuple(
            range(src_cursor, src_cursor + len(alignment.src))
        )
        assert alignment.tgt == tuple(
            range(tgt_cursor, tgt_cursor + len(alignment.tgt))
        )
        assert alignment.src or alignment.tgt
        assert len(alignment.src) + len(alignment.tgt) <= max_k
        if not alignment.src or not alignment.tgt:
            assert len(alignment.src) + len(alignment.tgt) == 1
        src_cursor += len(alignment.src)
        tgt_cursor += len(alignment.tgt)
    assert (src_cursor, tgt_cursor) == (src_count, tgt_count)


class SpyEmbedder:
    """A `HashEmbedder` that records the texts of every `embed` call."""

    def __init__(self) -> None:
        self.calls: list[list[str]] = []
        self._inner = HashEmbedder()

    def embed(self, texts: Sequence[str]) -> NDArray[np.float32]:
        self.calls.append(list(texts))
        return self._inner.embed(texts)


class DroppingEmbedder:
    """Returns one row fewer than asked for on its `bad_call`-th call."""

    def __init__(self, bad_call: int) -> None:
        self._bad_call = bad_call
        self._calls = 0
        self._inner = HashEmbedder()

    def embed(self, texts: Sequence[str]) -> NDArray[np.float32]:
        self._calls += 1
        vectors = self._inner.embed(texts)
        return vectors[:-1] if self._calls == self._bad_call else vectors


def _distinct_sentences(count: int) -> list[str]:
    return [f"first{i} second{i} third{i}" for i in range(count)]


@pytest.fixture(scope="module")
def planted() -> Planted:
    return build_planted_alignments()


@dataclass(frozen=True)
class Timed:
    alignments: list[Alignment]
    seconds: float


@pytest.fixture(scope="module")
def aligned(planted: Planted) -> Timed:
    started = time.perf_counter()
    alignments = align(planted.src, planted.tgt, HashEmbedder())
    return Timed(alignments, time.perf_counter() - started)


# --- the planted data ----------------------------------------------------------


def test_planted_data_has_every_alignment_type(planted: Planted) -> None:
    shapes = Counter((len(src), len(tgt)) for src, tgt in planted.gold)

    assert dict(shapes) == PLANNED_SHAPES
    assert len(planted.src) == len(planted.tgt) == SENTENCES_PER_SIDE
    assert [i for src, _ in planted.gold for i in src] == list(range(len(planted.src)))
    assert [j for _, tgt in planted.gold for j in tgt] == list(range(len(planted.tgt)))


def test_planted_sentences_all_carry_noise_words(planted: Planted) -> None:
    for sentence in planted.src + planted.tgt:
        words = set(sentence.removesuffix(".").split())
        assert words & planted.noise, sentence


def test_planted_data_is_two_runs_equal() -> None:
    first, second = build_planted_alignments(), build_planted_alignments()

    assert first == second


def test_planted_data_defeats_a_diagonal_baseline(planted: Planted) -> None:
    diagonal = [((i,), (i,)) for i in range(SENTENCES_PER_SIDE)]

    assert _f1(diagonal, planted.gold) < MAX_DIAGONAL_BASELINE_F1


# --- quality gates ---------------------------------------------------------------


def test_aligner_reaches_f1_0_95_on_planted_alignments(
    planted: Planted, aligned: Timed
) -> None:
    predicted = _pairs(aligned.alignments)

    assert _f1(predicted, planted.gold) >= MIN_F1
    assert _recall_of_shape(predicted, planted.gold, (1, 1)) >= MIN_ONE_TO_ONE_RECALL


def test_aligning_150_sentences_per_side_is_fast(aligned: Timed) -> None:
    assert aligned.seconds < ALIGN_SECONDS


def test_alignment_of_planted_data_covers_every_sentence_in_order(
    planted: Planted, aligned: Timed
) -> None:
    _assert_partition(aligned.alignments, len(planted.src), len(planted.tgt), 4)


def test_alignment_cost_of_a_deletion_is_the_skip_cost(aligned: Timed) -> None:
    deletions = _deletions(aligned.alignments)

    assert deletions
    assert {d.cost for d in deletions} == {DEFAULT_SKIP_COST}


# --- structural properties --------------------------------------------------------


_WORDS = st.sampled_from(["red", "green", "blue", "cat", "dog", "sun", "moon", "sea"])
_SENTENCE = st.lists(_WORDS, min_size=1, max_size=4).map(" ".join)


@given(
    src=st.lists(_SENTENCE, max_size=9),
    tgt=st.lists(_SENTENCE, max_size=9),
    max_k=st.integers(min_value=2, max_value=5),
)
def test_alignment_covers_every_sentence_exactly_once_and_in_order(
    src: list[str], tgt: list[str], max_k: int
) -> None:
    alignments = align(src, tgt, HashEmbedder(), max_k=max_k)

    _assert_partition(alignments, len(src), len(tgt), max_k)


def test_identical_sides_align_one_to_one() -> None:
    sentences = _distinct_sentences(20)

    alignments = align(sentences, sentences, HashEmbedder())

    assert _pairs(alignments) == [((i,), (i,)) for i in range(20)]
    assert all(alignment.cost < 0.1 for alignment in alignments)


def test_align_is_deterministic() -> None:
    planted = build_planted_alignments()
    small_src, small_tgt = planted.src[:40], planted.tgt[:40]

    first = align(small_src, small_tgt, HashEmbedder())
    second = align(small_src, small_tgt, HashEmbedder())

    assert first == second


# --- edge cases ---------------------------------------------------------------------


def test_two_empty_sides_give_no_alignments() -> None:
    spy = SpyEmbedder()

    assert align([], [], spy) == []
    assert spy.calls == []


@pytest.mark.parametrize(
    ("src", "tgt", "expected"),
    [
        (["a", "b"], [], [((0,), ()), ((1,), ())]),
        ([], ["a", "b", "c"], [((), (0,)), ((), (1,)), ((), (2,))]),
    ],
)
def test_one_empty_side_gives_only_deletions_without_embedding(
    src: list[str], tgt: list[str], expected: list[Pair]
) -> None:
    spy = SpyEmbedder()

    alignments = align(src, tgt, spy, skip_cost=0.5)

    assert _pairs(alignments) == expected
    assert [alignment.cost for alignment in alignments] == [0.5] * len(expected)
    assert spy.calls == []


def test_single_sentence_each_aligns_when_they_are_the_same_text() -> None:
    alignments = align(["red moon sea"], ["red moon sea"], HashEmbedder())

    assert _pairs(alignments) == [((0,), (0,))]
    assert alignments[0].cost == pytest.approx(0.0, abs=1e-4)


def test_single_unrelated_sentences_are_both_deleted_source_first() -> None:
    alignments = align(["red moon sea"], ["cat dog sun"], HashEmbedder())

    assert _pairs(alignments) == [((0,), ()), ((), (0,))]


@pytest.mark.parametrize("max_k", [1, 0, -3])
def test_max_k_below_two_raises(max_k: int) -> None:
    with pytest.raises(ValueError, match="max_k"):
        align(["a b"], ["a b"], HashEmbedder(), max_k=max_k)


def test_max_k_below_two_raises_even_for_empty_inputs() -> None:
    with pytest.raises(ValueError, match="max_k"):
        align([], [], HashEmbedder(), max_k=1)


def test_margin_k_below_one_raises() -> None:
    with pytest.raises(ValueError, match="margin_k"):
        align(["a b"], ["a b"], HashEmbedder(), margin_k=0)


@pytest.mark.parametrize("bad", [-0.1, float("nan")])
@pytest.mark.parametrize("name", ["skip_cost", "merge_penalty"])
def test_negative_or_nan_costs_raise(name: str, bad: float) -> None:
    with pytest.raises(ValueError, match=name):
        align(["a b"], ["a b"], HashEmbedder(), **{name: bad})


def test_max_k_three_forbids_two_to_two(planted: Planted, aligned: Timed) -> None:
    assert any(_shape(a) == (2, 2) for a in aligned.alignments)

    limited = align(planted.src, planted.tgt, HashEmbedder(), max_k=3)

    assert not any(_shape(a) == (2, 2) for a in limited)
    assert not any(len(a.src) + len(a.tgt) > 3 for a in limited)
    _assert_partition(limited, len(planted.src), len(planted.tgt), 3)


def test_max_k_two_allows_only_one_to_one_and_deletions(planted: Planted) -> None:
    limited = align(planted.src[:60], planted.tgt[:60], HashEmbedder(), max_k=2)

    assert {_shape(a) for a in limited} <= {(1, 1), (1, 0), (0, 1)}
    _assert_partition(limited, 60, 60, 2)


def test_larger_max_k_finds_the_one_to_three_alignments(
    planted: Planted, aligned: Timed
) -> None:
    gold_shapes = {(len(src), len(tgt)) for src, tgt in planted.gold}
    found = {_shape(a) for a in aligned.alignments}

    assert {(1, 3), (3, 1)} <= gold_shapes
    assert {(1, 3), (3, 1)} <= found


def test_skip_cost_controls_deletions(planted: Planted, aligned: Timed) -> None:
    default_deletions = len(_deletions(aligned.alignments))

    forbidden = align(planted.src, planted.tgt, HashEmbedder(), skip_cost=1000.0)
    cheap = align(planted.src, planted.tgt, HashEmbedder(), skip_cost=0.05)

    assert _deletions(forbidden) == []
    assert len(_deletions(cheap)) > default_deletions


def test_merge_penalty_controls_how_many_sentences_are_grouped(
    planted: Planted, aligned: Timed
) -> None:
    def grouped(alignments: Sequence[Alignment]) -> int:
        return sum(len(a.src) + len(a.tgt) > 2 for a in alignments)

    plain = align(planted.src, planted.tgt, HashEmbedder(), merge_penalty=0.0)
    strict = align(planted.src, planted.tgt, HashEmbedder(), merge_penalty=5.0)

    assert grouped(strict) == 0
    assert grouped(plain) > grouped(aligned.alignments) > 0


# --- the embedder contract -----------------------------------------------------------


def test_embedder_called_once_per_side_with_every_span() -> None:
    src = ["red sun", "green sea", "blue moon", "cat dog"]
    tgt = ["dog cat", "moon blue", "sea green"]
    spy = SpyEmbedder()

    align(src, tgt, spy, max_k=4)

    def spans(sentences: list[str]) -> list[str]:
        return [
            " ".join(sentences[i : i + size])
            for size in (1, 2, 3)
            for i in range(len(sentences) - size + 1)
        ]

    assert len(spy.calls) == 2
    assert sorted(spy.calls[0]) == sorted(spans(src))
    assert sorted(spy.calls[1]) == sorted(spans(tgt))
    assert "red sun green sea" in spy.calls[0]


def test_spans_stop_at_max_k_minus_one() -> None:
    spy = SpyEmbedder()

    align(["a b", "c d", "e f", "g h"], ["a b", "c d", "e f", "g h"], spy, max_k=2)

    assert [len(call) for call in spy.calls] == [4, 4]


@pytest.mark.parametrize("bad_call", [1, 2])
def test_embedder_returning_the_wrong_row_count_raises(bad_call: int) -> None:
    sentences = ["red sun", "green sea", "blue moon"]

    with pytest.raises(ValueError, match=r"expected 6 rows, got 5"):
        align(sentences, sentences, DroppingEmbedder(bad_call))
