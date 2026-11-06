"""Deterministic fake sentence embedders for the filter and mining tests.

`ConceptEmbedder` maps exact texts to concept ids, so a test decides which
sentence pairs are translations. `HashEmbedder` embeds a bag of hashed tokens,
so the embedding of a concatenation is the sum of the parts, which is what the
sentence aligner's overlap spans rely on. Both seed from `hashlib`, never the
builtin `hash()`, so results are the same in every process.
"""

from __future__ import annotations

import hashlib
import string
import unicodedata
from collections.abc import Mapping, Sequence
from functools import cache

import numpy as np
from numpy.typing import NDArray

_SEED_BYTES = 8


def _seed_of(text: str) -> int:
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return int.from_bytes(digest[:_SEED_BYTES], "big")


def _gaussian(seed: int, dim: int) -> NDArray[np.float64]:
    return np.random.default_rng(seed).standard_normal(dim)


def _unit_rows(vectors: NDArray[np.float64]) -> NDArray[np.float32]:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return (vectors / norms).astype(np.float32)


def concept_vector(concept_id: int, dim: int = 64) -> NDArray[np.float32]:
    """The unit vector of a concept: Gaussian, from a generator seeded by the id."""
    return _unit_rows(_gaussian(concept_id, dim)[np.newaxis, :])[0]


class ConceptEmbedder:
    """Texts of one concept get one vector; unrelated concepts are near
    orthogonal. A text outside the mapping gets a vector seeded by a hash of
    the text, so it is stable and unrelated to every concept."""

    def __init__(self, concepts: Mapping[str, int], dim: int = 64) -> None:
        self._concepts = dict(concepts)
        self._dim = dim

    def embed(self, texts: Sequence[str]) -> NDArray[np.float32]:
        if not texts:
            return np.empty((0, self._dim), dtype=np.float32)
        return _unit_rows(np.stack([self._vector(text) for text in texts]))

    def _vector(self, text: str) -> NDArray[np.float64]:
        concept = self._concepts.get(text)
        seed = _seed_of(text) if concept is None else concept
        return _gaussian(seed, self._dim)


def _is_punctuation(char: str) -> bool:
    return char in string.punctuation or unicodedata.category(char).startswith("P")


def _strip_edge_punctuation(token: str) -> str:
    start, end = 0, len(token)
    while start < end and _is_punctuation(token[start]):
        start += 1
    while end > start and _is_punctuation(token[end - 1]):
        end -= 1
    return token[start:end]


def _tokens(text: str) -> list[str]:
    stripped = (_strip_edge_punctuation(token) for token in text.lower().split())
    return [token for token in stripped if token]


@cache
def _token_vector(token: str, dim: int) -> NDArray[np.float64]:
    return _gaussian(_seed_of(token), dim)


class HashEmbedder:
    """Normalised sum of seeded Gaussian vectors, one per lower-cased
    whitespace token with edge punctuation removed. Text without tokens gets
    the vector of the empty token, so it is fixed and never zero."""

    def __init__(self, dim: int = 128) -> None:
        self._dim = dim

    def embed(self, texts: Sequence[str]) -> NDArray[np.float32]:
        if not texts:
            return np.empty((0, self._dim), dtype=np.float32)
        return _unit_rows(np.stack([self._vector(text) for text in texts]))

    def _vector(self, text: str) -> NDArray[np.float64]:
        tokens = _tokens(text) or [""]
        return np.sum([_token_vector(token, self._dim) for token in tokens], axis=0)
