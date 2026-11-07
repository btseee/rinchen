"""Self-test of the planted sentence alignments used by the aligner tests."""

from __future__ import annotations

from collections import Counter

import pytest

from tests.support.planted_alignments import (
    PLANNED_SHAPES,
    SENTENCES_PER_SIDE,
    Planted,
    build_planted_alignments,
    f1_score,
    recall_of_shape,
)

MAX_DIAGONAL_BASELINE_F1 = 0.5


@pytest.fixture(scope="module")
def planted() -> Planted:
    return build_planted_alignments()


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


def test_another_seed_gives_other_sentences_with_the_same_shapes() -> None:
    first, other = build_planted_alignments(), build_planted_alignments(seed=12)

    assert first.src != other.src
    assert Counter((len(s), len(t)) for s, t in other.gold) == PLANNED_SHAPES


def test_planted_data_defeats_a_diagonal_baseline(planted: Planted) -> None:
    diagonal = [((i,), (i,)) for i in range(SENTENCES_PER_SIDE)]

    assert f1_score(diagonal, planted.gold) < MAX_DIAGONAL_BASELINE_F1


def test_f1_and_recall_count_exact_index_tuple_matches() -> None:
    gold = [((0,), (0,)), ((1, 2), (1,)), ((3,), ()), ((), (2,))]
    predicted = [((0,), (0,)), ((1,), (1,)), ((2,), ()), ((3,), ()), ((), (2,))]

    # 3 of 5 predicted and 3 of 4 gold pairs match exactly.
    assert f1_score(predicted, gold) == pytest.approx(2 * 0.6 * 0.75 / (0.6 + 0.75))
    assert f1_score(predicted, []) == 0.0
    assert recall_of_shape(predicted, gold, (1, 1)) == 1.0
    assert recall_of_shape(predicted, gold, (2, 1)) == 0.0
    assert recall_of_shape(predicted, gold, (1, 0)) == 1.0
