"""The vecalign-style sentence aligner (B6): quality gates on the planted
alignments of `tests.support.planted_alignments` (real `HashEmbedder`, a gold
alignment of every type), structural properties, edge cases, the embedder
contract and a brute-force check of the cost formula and the DP."""

from __future__ import annotations

import itertools
import random
from collections.abc import Iterator, Sequence

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st
from numpy.typing import NDArray

from rinchen.mine.align import Alignment, align
from tests.support.embedders import HashEmbedder
from tests.support.planted_alignments import (
    Pair,
    Planted,
    build_planted_alignments,
    f1_score,
    pairs_of,
    recall_of_shape,
)

MIN_F1 = 0.95
MIN_ONE_TO_ONE_RECALL = 0.97
DEFAULT_SKIP_COST = 0.9


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


@pytest.fixture(scope="module")
def aligned(planted: Planted) -> list[Alignment]:
    return align(planted.src, planted.tgt, HashEmbedder())


# --- quality gates ---------------------------------------------------------------


def test_aligner_reaches_f1_0_95_on_planted_alignments(
    planted: Planted, aligned: list[Alignment]
) -> None:
    predicted = pairs_of(aligned)

    assert f1_score(predicted, planted.gold) >= MIN_F1
    assert recall_of_shape(predicted, planted.gold, (1, 1)) >= MIN_ONE_TO_ONE_RECALL


def test_alignment_of_planted_data_covers_every_sentence_in_order(
    planted: Planted, aligned: list[Alignment]
) -> None:
    _assert_partition(aligned, len(planted.src), len(planted.tgt), 4)


def test_alignment_cost_of_a_deletion_is_the_skip_cost(
    aligned: list[Alignment],
) -> None:
    deletions = _deletions(aligned)

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

    assert pairs_of(alignments) == [((i,), (i,)) for i in range(20)]
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

    assert pairs_of(alignments) == expected
    assert [alignment.cost for alignment in alignments] == [0.5] * len(expected)
    assert spy.calls == []


def test_single_sentence_each_aligns_when_they_are_the_same_text() -> None:
    alignments = align(["red moon sea"], ["red moon sea"], HashEmbedder())

    assert pairs_of(alignments) == [((0,), (0,))]
    assert alignments[0].cost == pytest.approx(0.0, abs=1e-4)


def test_single_unrelated_sentences_are_both_deleted_source_first() -> None:
    alignments = align(["red moon sea"], ["cat dog sun"], HashEmbedder())

    assert pairs_of(alignments) == [((0,), ()), ((), (0,))]


def test_single_matching_pair_aligns_without_being_identical() -> None:
    alignments = align(
        ["alpha beta gamma delta"], ["alpha beta gamma epsilon"], HashEmbedder()
    )

    assert pairs_of(alignments) == [((0,), (0,))]


def test_one_source_sentence_that_joins_two_targets_aligns_one_to_two() -> None:
    alignments = align(
        ["alpha beta gamma delta oak"],
        ["alpha beta elm", "gamma delta fir"],
        HashEmbedder(),
    )

    assert pairs_of(alignments) == [((0,), (0, 1))]


def test_two_source_sentences_that_split_one_target_align_two_to_one() -> None:
    alignments = align(
        ["alpha beta elm", "gamma delta fir"],
        ["alpha beta gamma delta oak"],
        HashEmbedder(),
    )

    assert pairs_of(alignments) == [((0, 1), (0,))]


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


@pytest.mark.parametrize("bad", [-0.1, float("nan"), float("inf")])
@pytest.mark.parametrize("name", ["skip_cost", "merge_penalty"])
def test_costs_that_are_not_finite_and_non_negative_raise(
    name: str, bad: float
) -> None:
    with pytest.raises(ValueError, match=f"{name} must be finite and non-negative"):
        align(["a b"], ["a b"], HashEmbedder(), **{name: bad})


def test_max_k_three_forbids_two_to_two(
    planted: Planted, aligned: list[Alignment]
) -> None:
    assert any(_shape(a) == (2, 2) for a in aligned)

    limited = align(planted.src, planted.tgt, HashEmbedder(), max_k=3)

    assert not any(_shape(a) == (2, 2) for a in limited)
    assert not any(len(a.src) + len(a.tgt) > 3 for a in limited)
    _assert_partition(limited, len(planted.src), len(planted.tgt), 3)


def test_max_k_two_allows_only_one_to_one_and_deletions(planted: Planted) -> None:
    limited = align(planted.src[:60], planted.tgt[:60], HashEmbedder(), max_k=2)

    assert {_shape(a) for a in limited} <= {(1, 1), (1, 0), (0, 1)}
    _assert_partition(limited, 60, 60, 2)


def test_larger_max_k_finds_the_one_to_three_alignments(
    planted: Planted, aligned: list[Alignment]
) -> None:
    gold_shapes = {(len(src), len(tgt)) for src, tgt in planted.gold}
    found = {_shape(a) for a in aligned}

    assert {(1, 3), (3, 1)} <= gold_shapes
    assert {(1, 3), (3, 1)} <= found


def test_skip_cost_controls_deletions(
    planted: Planted, aligned: list[Alignment]
) -> None:
    default_deletions = len(_deletions(aligned))

    forbidden = align(planted.src, planted.tgt, HashEmbedder(), skip_cost=1000.0)
    cheap = align(planted.src, planted.tgt, HashEmbedder(), skip_cost=0.05)

    assert _deletions(forbidden) == []
    assert len(_deletions(cheap)) > default_deletions


def test_merge_penalty_controls_how_many_sentences_are_grouped(
    planted: Planted, aligned: list[Alignment]
) -> None:
    def grouped(alignments: Sequence[Alignment]) -> int:
        return sum(len(a.src) + len(a.tgt) > 2 for a in alignments)

    plain = align(planted.src, planted.tgt, HashEmbedder(), merge_penalty=0.0)
    strict = align(planted.src, planted.tgt, HashEmbedder(), merge_penalty=5.0)

    assert grouped(strict) == 0
    assert grouped(plain) > grouped(aligned) > 0


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


# --- the cost formula and the DP against a brute-force search ---------------------

_ORACLE_WORDS = ("red", "green", "blue", "cat", "dog", "sun", "moon", "sea", "ant")


class _Oracle:
    """The edge weights and the optimum, re-derived without the aligner's
    batching: every text is embedded on its own and every segmentation is
    enumerated. Each sentence carries a tag word nobody else has, so no two
    spans are identical and float32 noise cannot decide a cost."""

    def __init__(
        self,
        src: Sequence[str],
        tgt: Sequence[str],
        *,
        skip_cost: float,
        margin_k: int,
        merge_penalty: float,
    ) -> None:
        self.src, self.tgt = src, tgt
        self.skip_cost, self.margin_k, self.merge_penalty = (
            skip_cost,
            margin_k,
            merge_penalty,
        )
        self._embedder = HashEmbedder()
        self._vectors: dict[str, NDArray[np.float64]] = {}

    def _distance(self, x: str, y: str) -> float:
        for text in (x, y):
            if text not in self._vectors:
                self._vectors[text] = self._embedder.embed([text])[0].astype(np.float64)
        return min(max(1.0 - float(self._vectors[x] @ self._vectors[y]), 0.0), 2.0)

    def _margin(self, x: str, others: Sequence[str]) -> float:
        nearest = sorted(self._distance(x, other) for other in others)[: self.margin_k]
        # Fewer neighbours than margin_k: the missing ones are unrelated (1.0).
        nearest += [1.0] * (self.margin_k - len(nearest))
        return sum(nearest) / self.margin_k

    @staticmethod
    def _spans(sentences: Sequence[str], size: int) -> list[str]:
        return [
            " ".join(sentences[start : start + size])
            for start in range(len(sentences) - size + 1)
        ]

    def weight(self, src_start: int, a: int, tgt_start: int, b: int) -> float:
        if a == 0 or b == 0:
            return self.skip_cost
        x = " ".join(self.src[src_start : src_start + a])
        y = " ".join(self.tgt[tgt_start : tgt_start + b])
        margin = 0.5 * (
            self._margin(x, self._spans(self.tgt, b))
            + self._margin(y, self._spans(self.src, a))
        )
        cost = self._distance(x, y) / max(margin, 1e-6)
        return cost * (a + b) + self.merge_penalty * (a + b - 2)

    def segmentations(
        self, i: int, j: int, max_k: int
    ) -> Iterator[list[tuple[int, int, int, int]]]:
        if (i, j) == (len(self.src), len(self.tgt)):
            yield []
            return
        steps = [(1, 0), (0, 1)] + [
            (a, b)
            for a, b in itertools.product(range(1, max_k), repeat=2)
            if a + b <= max_k
        ]
        for a, b in steps:
            if i + a <= len(self.src) and j + b <= len(self.tgt):
                for rest in self.segmentations(i + a, j + b, max_k):
                    yield [(i, a, j, b), *rest]

    def total(self, segmentation: Sequence[tuple[int, int, int, int]]) -> float:
        return sum(self.weight(*step) for step in segmentation)


def _oracle_sentences(rng: random.Random, side: str, count: int) -> list[str]:
    return [
        " ".join([*rng.sample(_ORACLE_WORDS, rng.randint(1, 3)), f"{side}{i}"])
        for i in range(count)
    ]


def test_aligner_matches_a_brute_force_search_over_all_segmentations() -> None:
    rng = random.Random(21)
    for _ in range(30):
        src = _oracle_sentences(rng, "s", rng.randint(1, 6))
        tgt = _oracle_sentences(rng, "t", rng.randint(1, 6))
        max_k = rng.randint(2, 5)
        # margin_k == 1 is left out: with one neighbour a pair that is its own
        # nearest neighbour costs 1.0 whatever its distance, and for
        # near-identical spans the 1e-6 floor and float32 rounding, not the
        # formula, decide the cost.
        options = {
            "skip_cost": rng.choice([0.5, 0.9, 1.5]),
            "margin_k": rng.randint(2, 3),
            "merge_penalty": rng.choice([0.0, 0.3]),
        }
        oracle = _Oracle(src, tgt, **options)

        alignments = align(src, tgt, HashEmbedder(), max_k=max_k, **options)

        chosen = [
            (a.src[0] if a.src else 0, len(a.src), a.tgt[0] if a.tgt else 0, len(a.tgt))
            for a in alignments
        ]
        best = min(oracle.total(s) for s in oracle.segmentations(0, 0, max_k))
        assert oracle.total(chosen) == pytest.approx(best, rel=1e-4, abs=1e-6)
        assert sum(a.cost for a in alignments) == pytest.approx(
            best, rel=1e-4, abs=1e-6
        )
        for alignment, step in zip(alignments, chosen, strict=True):
            assert alignment.cost == pytest.approx(
                oracle.weight(*step), rel=1e-4, abs=1e-6
            )
