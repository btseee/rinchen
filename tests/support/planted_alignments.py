"""Planted sentence alignments for the aligner tests: two sentence lists built
from a gold alignment of every type (1-1, 1-2, 2-1, 2-2, 1-3, 3-1 and both
deletions), for the real `HashEmbedder`.

A sentence is a bag of concept words plus one or two noise words from a shared
pool, so a span embeds as the sum of its sentences and the noise keeps true
pairs well below cosine 1. `f1_score` and `recall_of_shape` score a result with
plain set arithmetic over exact `(src indexes, tgt indexes)` pairs, independent
of the aligner.
"""

from __future__ import annotations

import random
from collections.abc import Sequence
from dataclasses import dataclass

from rinchen.mine.align import Alignment

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


def build_planted_alignments(seed: int = SEED) -> Planted:
    rng = random.Random(seed)
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


def pairs_of(alignments: Sequence[Alignment]) -> list[Pair]:
    return [(alignment.src, alignment.tgt) for alignment in alignments]


def f1_score(predicted: Sequence[Pair], gold: Sequence[Pair]) -> float:
    matched = len(set(predicted) & set(gold))
    if matched == 0:
        return 0.0
    precision = matched / len(predicted)
    recall = matched / len(gold)
    return 2 * precision * recall / (precision + recall)


def recall_of_shape(
    predicted: Sequence[Pair], gold: Sequence[Pair], shape: tuple[int, int]
) -> float:
    wanted = [pair for pair in gold if (len(pair[0]), len(pair[1])) == shape]
    return len(set(predicted) & set(wanted)) / len(wanted)
