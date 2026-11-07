"""Vecalign-style sentence alignment (B6): align two lists of sentences, one
per side of a document pair, into 1-1, 1-2, 2-1, 2-2, 1-3 and 3-1 groups plus
single-sentence deletions (1-0 and 0-1), by dynamic programming over sentence
embeddings.

Every span of 1..`max_k - 1` consecutive sentences, joined by one space, is
embedded as one text (one `embed` call per side). The cost of pairing a source
span `x` with a target span `y` is margin-normalised cosine distance:

    cost(x, y) = (1 - cos(x, y)) / (0.5 * (knn_x + knn_y))

where `knn_x` is the mean of the `margin_k` smallest `1 - cos(x, y')` over the
target spans `y'` of the same size as `y`, and `knn_y` the same over the source
spans of the size of `x` (fewer than `margin_k` spans: the missing neighbours
count as unrelated, at distance 1.0; the denominator is floored at 1e-6). A pairing of `a` source with `b` target
sentences weighs

    cost * (a + b) + merge_penalty * (a + b - 2)

a deletion weighs `skip_cost`, and the cheapest path covers every sentence of
both sides once, in order. Transitions `(a, b)` need `a + b <= max_k`; `a = 0`
or `b = 0` only for one sentence.

The `merge_penalty` term departs from the plain vecalign weighting
`cost * (a + b)` (`merge_penalty=0.0`). On the planted test data, which embeds
additively, the plain weighting merges neighbouring 1-1 pairs into one 2-2: the
cosine of a concatenation lets a weak pair ride on a strong neighbour, so one
2-2 beats the sum of its two 1-1 pairs. That gave F1 0.906 on the test data
(mean 0.897 over 20 seeds, min 0.849), and no `skip_cost` reached 0.95. The
penalty is a prior for simple alignments: each sentence beyond 1-1 costs 0.3
extra, which gives F1 0.989 (mean 0.993, min 0.978) at the default
`skip_cost` of 0.9. Retune it on real embeddings before trusting it there.

Ties in the DP go to the first transition in this order: 1-1, then the other
pairings by `a + b` and then `a`, then 0-1, then 1-0. So a pairing beats a
deletion, a smaller pairing beats a larger one, and where a source-only and a
target-only deletion are adjacent the source-only one comes first.

Limitations. The DP is unbanded: time and memory are O(n * m * |T|) for `n`
source and `m` target sentences and `|T|` transitions, so it suits one document
pair, not a whole corpus. Measured with `max_k=4` and a cheap additive embedder
(peak resident memory of the process): n = m = 600 takes 0.48 s and 109 MB,
1500 takes 2.7 s and 468 MB, 2500 takes 6.9 s and 1.1 GB; the embedder itself
comes on top.

Padding with unrelated neighbours keeps a lone pair from being judged against
itself, but it is lenient: with `skip_cost` 0.9 a side with fewer than
`margin_k` candidates accepts a pair whenever its cosine distance is below
about 0.87. That is permissive for a strong multilingual encoder, so downstream
filters must judge such pairs.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from rinchen.mine.embed import Embedder

_MARGIN_FLOOR = 1e-6
_MAX_COSINE_DISTANCE = 2.0
_DELETIONS = [(0, 1), (1, 0)]

Transition = tuple[int, int]
Step = tuple[int, int, float]


@dataclass(frozen=True, slots=True)
class Alignment:
    """One group of source sentence indexes aligned with one group of target
    indexes. An empty side marks a deletion. `cost` is the DP edge weight, so
    the costs of all alignments of a result add up to the cost of its path."""

    src: tuple[int, ...]
    tgt: tuple[int, ...]
    cost: float


def align(
    src: Sequence[str],
    tgt: Sequence[str],
    embed: Embedder,
    max_k: int = 4,
    *,
    skip_cost: float = 0.9,
    margin_k: int = 4,
    merge_penalty: float = 0.3,
) -> list[Alignment]:
    """Align `src` with `tgt`; the result covers every sentence of both sides
    exactly once, in order. An empty side yields only deletions and the
    embedder is not called. Raises `ValueError` for `max_k < 2`,
    `margin_k < 1`, a negative or NaN `skip_cost` or `merge_penalty`, or an
    embedder that returns the wrong number of rows."""
    _check_parameters(max_k, skip_cost, margin_k, merge_penalty)
    if not src or not tgt:
        return _only_deletions(len(src), len(tgt), skip_cost)
    src_vectors = _embed_spans(src, max_k - 1, embed)
    tgt_vectors = _embed_spans(tgt, max_k - 1, embed)
    weights = _pairing_weights(src_vectors, tgt_vectors, max_k, margin_k, merge_penalty)
    return _cheapest_path(len(src), len(tgt), weights, max_k, skip_cost)


def _check_parameters(
    max_k: int, skip_cost: float, margin_k: int, merge_penalty: float
) -> None:
    if max_k < 2:
        raise ValueError(f"max_k must be at least 2, got {max_k}")
    if margin_k < 1:
        raise ValueError(f"margin_k must be at least 1, got {margin_k}")
    for name, value in (("skip_cost", skip_cost), ("merge_penalty", merge_penalty)):
        if not (math.isfinite(value) and value >= 0.0):
            raise ValueError(f"{name} must be finite and non-negative, got {value}")


def _only_deletions(
    src_count: int, tgt_count: int, skip_cost: float
) -> list[Alignment]:
    return [Alignment((i,), (), skip_cost) for i in range(src_count)] + [
        Alignment((), (j,), skip_cost) for j in range(tgt_count)
    ]


def _embed_spans(
    sentences: Sequence[str], max_size: int, embed: Embedder
) -> dict[int, NDArray[np.float32]]:
    """Embed every span of 1..`max_size` consecutive sentences in one call;
    the result maps a span size to the matrix of its spans in start order."""
    sizes = range(1, min(max_size, len(sentences)) + 1)
    texts_by_size = {
        size: [
            " ".join(sentences[start : start + size])
            for start in range(len(sentences) - size + 1)
        ]
        for size in sizes
    }
    texts = [text for size in sizes for text in texts_by_size[size]]
    vectors = np.asarray(embed.embed(texts), dtype=np.float32)
    if vectors.ndim != 2:
        raise ValueError(f"embedder must return a 2-D array, got shape {vectors.shape}")
    if vectors.shape[0] != len(texts):
        raise ValueError(
            f"embedder must return one row per text: expected {len(texts)} rows, "
            f"got {vectors.shape[0]}"
        )
    matrices = {}
    offset = 0
    for size in sizes:
        count = len(texts_by_size[size])
        matrices[size] = vectors[offset : offset + count]
        offset += count
    return matrices


def _pairings(max_k: int) -> list[Transition]:
    """The `(a, b)` pairings with `a, b >= 1` and `a + b <= max_k`, in the
    order that settles ties."""
    pairs = [(a, b) for a in range(1, max_k) for b in range(1, max_k) if a + b <= max_k]
    return sorted(pairs, key=lambda pair: (pair[0] + pair[1], pair[0]))


def _pairing_weights(
    src_vectors: dict[int, NDArray[np.float32]],
    tgt_vectors: dict[int, NDArray[np.float32]],
    max_k: int,
    margin_k: int,
    merge_penalty: float,
) -> dict[Transition, NDArray[np.float64]]:
    """Per pairing `(a, b)`, the matrix of edge weights: entry `[i, j]` is for
    source span `i..i+a` with target span `j..j+b`. Pairings that need a span
    longer than a whole side are absent."""
    weights = {}
    for a, b in _pairings(max_k):
        if a in src_vectors and b in tgt_vectors:
            similarity = src_vectors[a] @ tgt_vectors[b].T
            distance = np.clip(
                1.0 - similarity.astype(np.float64), 0.0, _MAX_COSINE_DISTANCE
            )
            weights[(a, b)] = _margin_normalised(distance, margin_k) * (
                a + b
            ) + merge_penalty * (a + b - 2)
    return weights


def _margin_normalised(
    distance: NDArray[np.float64], margin_k: int
) -> NDArray[np.float64]:
    src_margin = _mean_of_smallest(distance, margin_k, axis=1)
    tgt_margin = _mean_of_smallest(distance, margin_k, axis=0)
    margin = 0.5 * (src_margin[:, np.newaxis] + tgt_margin[np.newaxis, :])
    return distance / np.maximum(margin, _MARGIN_FLOOR)


def _mean_of_smallest(
    distance: NDArray[np.float64], k: int, *, axis: int
) -> NDArray[np.float64]:
    """Mean of the `k` smallest values along `axis`. With fewer than `k`
    candidates the missing neighbours count as unrelated, at distance 1.0."""
    missing = k - distance.shape[axis]
    if missing > 0:
        padding_shape = list(distance.shape)
        padding_shape[axis] = missing
        distance = np.concatenate([distance, np.ones(padding_shape)], axis=axis)
    smallest = np.take(np.partition(distance, k - 1, axis=axis), range(k), axis=axis)
    return smallest.mean(axis=axis)


def _cheapest_path(
    src_count: int,
    tgt_count: int,
    weights: dict[Transition, NDArray[np.float64]],
    max_k: int,
    skip_cost: float,
) -> list[Alignment]:
    """Fill `best[i][j]`, the cheapest way to align the first `i` source and
    first `j` target sentences, then walk the recorded steps back from the end."""
    pairings = [pair for pair in _pairings(max_k) if pair in weights]
    transitions = [*pairings, *_DELETIONS]
    best = [[math.inf] * (tgt_count + 1) for _ in range(src_count + 1)]
    steps: list[list[Step | None]] = [
        [None] * (tgt_count + 1) for _ in range(src_count + 1)
    ]
    best[0][0] = 0.0
    for i in range(src_count + 1):
        # The weights of the spans that end at source sentence `i`, as lists.
        rows = {(a, b): weights[(a, b)][i - a].tolist() for a, b in pairings if i >= a}
        for j in range(tgt_count + 1):
            if i == 0 and j == 0:
                continue
            cell_best = math.inf
            cell_step: Step | None = None
            for a, b in transitions:
                if i < a or j < b:
                    continue
                edge = rows[(a, b)][j - b] if a and b else skip_cost
                total = best[i - a][j - b] + edge
                if total < cell_best:
                    cell_best, cell_step = total, (a, b, edge)
            best[i][j], steps[i][j] = cell_best, cell_step
    return _walk_back(steps, src_count, tgt_count)


def _walk_back(
    steps: list[list[Step | None]], src_count: int, tgt_count: int
) -> list[Alignment]:
    path: list[Alignment] = []
    i, j = src_count, tgt_count
    while (i, j) != (0, 0):
        step = steps[i][j]
        assert step is not None, "a deletion step reaches every cell"
        a, b, edge = step
        path.append(Alignment(tuple(range(i - a, i)), tuple(range(j - b, j)), edge))
        i, j = i - a, j - b
    path.reverse()
    return path
