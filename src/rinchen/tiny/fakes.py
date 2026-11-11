"""The planted stand-ins of `--qe-fake`: a sentence encoder and a quality scorer that need no model
(D15, spec P6).

`ConceptEncoder` stands in for LaBSE. It is not a trained network: a text is read as a BAG OF
CONCEPTS, each concept is one fixed pseudo-random 64-dimensional vector seeded by the CRC-32 of its
name, and the bag is summed in the order of the concepts' names and normalised. The concepts are the
nouns (the three words of a row and the traditional spelling of its Khalkha word), the entities (each
spelling and its traditional spelling), digit strings (the group and decimal separators are not part of
a number) and the numbered placeholders; any other word names none and is ignored. A Khalkha case
suffix glued to a known name is cut off by trying the longest known prefix of three characters or more.
Nothing is random at run time and nothing depends on the hash seed of the process, so a text has the
same vector in every process.

`ConceptQE` stands in for the MetricX scorer: `25 * max(0, 1 - cosine)` of the two sides, on MetricX's
scale, where lower is better. Neither ranks a real system; they give the filter something that can
tell a pair that says one thing from a pair that says two.
"""

from __future__ import annotations

import re
import zlib
from collections.abc import Mapping, Sequence
from functools import cache
from typing import Final

import numpy as np
from numpy.typing import NDArray

from rinchen.mine.embed import Embedder
from rinchen.script_conv.rules import transliterate

DIMENSIONS: Final = 64
_MIN_PREFIX: Final = 3
_QE_SCALE: Final = 25.0
_EMPTY_CONCEPT: Final = "<empty>"
_PLACEHOLDER_OPEN: Final = chr(0x27E6)
# A numbered placeholder, a number (digits, then a separator and one to three digits, as often as it
# goes) or a run of any other characters that are not white space or one of . , ? ! : ;
_TOKEN: Final = re.compile(r"\u27e6\d+\u27e7|\d+(?:[ .,]\d{1,3})*|[^\s.,?!:;]+")
_NOT_A_DIGIT: Final = re.compile(r"\D")


@cache
def _vector(concept: str) -> NDArray[np.float32]:
    """The fixed vector of a concept, before it is summed and normalised (read-only: it is shared)."""
    rng = np.random.default_rng(zlib.crc32(concept.encode("utf-8")))
    vector = rng.standard_normal(DIMENSIONS).astype(np.float32)
    vector.setflags(write=False)
    return vector


def _unit(vector: NDArray[np.float32]) -> NDArray[np.float32]:
    norm = float(np.linalg.norm(vector))
    if norm == 0.0:
        return _unit(_vector(_EMPTY_CONCEPT))
    return (vector / norm).astype(np.float32)


class ConceptEncoder:
    """An `Embedder` of bags of concepts. `nouns` are rows whose first column is the Khalkha word
    (the others are the same noun in other languages) and `entities` map an id to a language to the
    spellings of the name; both come from the fixtures."""

    def __init__(
        self,
        nouns: Sequence[Sequence[str]],
        entities: Mapping[str, Mapping[str, Sequence[str]]],
    ) -> None:
        concept_of: dict[str, str] = {}
        for index, row in enumerate(nouns):
            for word in row:
                concept_of[word.lower()] = f"n{index}"
            concept_of[transliterate(row[0])] = f"n{index}"
        for entity_id, by_language in entities.items():
            for spellings in by_language.values():
                for spelling in spellings:
                    concept_of[spelling.lower()] = f"e:{entity_id}"
                    concept_of[transliterate(spelling)] = f"e:{entity_id}"
        self._concept_of = concept_of

    def embed(self, texts: Sequence[str]) -> NDArray[np.float32]:
        """One unit row of 64 dimensions per text, in order; a text with no concept has the one fixed
        vector of the empty bag."""
        rows = np.zeros((len(texts), DIMENSIONS), dtype=np.float32)
        for index, text in enumerate(texts):
            rows[index] = self._embed_one(text)
        return rows

    def _embed_one(self, text: str) -> NDArray[np.float32]:
        total = np.zeros(DIMENSIONS, dtype=np.float32)
        concepts = sorted(self._concepts(text))
        if not concepts:
            return _unit(_vector(_EMPTY_CONCEPT))
        for concept in concepts:
            total += _vector(concept)
        return _unit(total)

    def _concepts(self, text: str) -> list[str]:
        found: list[str] = []
        for token in _TOKEN.findall(text):
            if token[0].isdecimal():
                found.append("d:" + _NOT_A_DIGIT.sub("", token))
            elif token.startswith(_PLACEHOLDER_OPEN):
                found.append("p:" + token)
            elif (concept := self._word_concept(token)) is not None:
                found.append(concept)
        return found

    def _word_concept(self, token: str) -> str | None:
        """The concept of a word, or of the longest known prefix of it of at least three characters
        (a case suffix glued to a name); None when the word names none."""
        concept = self._concept_of.get(token.lower())
        cut = len(token) - 1
        while concept is None and cut >= _MIN_PREFIX:
            concept = self._concept_of.get(token[:cut].lower())
            cut -= 1
        return concept


class ConceptQE:
    """A `QEScorer` over the cosine of the two sides' bags; it runs on the CPU."""

    requires_cuda: bool = False

    def __init__(self, encoder: Embedder) -> None:
        self._encoder = encoder

    def score(self, sources: Sequence[str], translations: Sequence[str]) -> list[float]:
        """`25 * max(0, 1 - cosine)` per pair, in order; 0 is a pair that says one thing.

        Raises:
            ValueError: the two lists differ in length.
        """
        if len(sources) != len(translations):
            raise ValueError(
                f"got {len(sources)} sources and {len(translations)} translations"
            )
        cosine = np.einsum(
            "ij,ij->i", self._encoder.embed(sources), self._encoder.embed(translations)
        )
        return [_QE_SCALE * max(0.0, 1.0 - float(value)) for value in cosine]
