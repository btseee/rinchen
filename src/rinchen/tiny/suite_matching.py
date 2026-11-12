"""Finding the entities and the glossary terms of a sentence by WHOLE TOKENS (D15, spec P2, brief G5.1).

A sentence is cut into tokens (white-space separated words, normalised to NFC, casefolded, with the
marks `. , ? ! : ;` taken off the edges of a word), and a name or a noun is found when a token, or for a
name of several words a run of tokens, is that spelling. A spelling that is a piece of a longer word is
not found: the prototype matched by substring and found the Russian person `Bat` in the city name
`Sukhe-Bator`.

The one allowance is a Khalkha case suffix glued to a name (the city `Ulaanbaatar` with a case ending
after it): a Khalkha token that is no known spelling is read as the longest known single-word spelling
of three characters or more that is a proper prefix of it. Nothing else is inflected: not an English
name, not a Russian one, and no noun of a glossary.

The matchers know no fixture file and no suite type: they are given the tables the fixtures hold and
answer with plain strings, so that the module imports nothing of `rinchen.eval`.
"""

from __future__ import annotations

import unicodedata
from collections.abc import Hashable, Iterable, Mapping, Sequence
from typing import Final

from rinchen.config.eval_suites import HUB_LANGUAGE

MIN_PREFIX: Final = 3
EDGE_MARKS: Final = ".,?!:;"


def fold(text: str) -> str:
    """`text` in NFC and without case."""
    return unicodedata.normalize("NFC", text).casefold()


def tokens_of(text: str) -> list[str]:
    """The folded tokens of `text`: its white-space separated words with the marks of `EDGE_MARKS`
    taken off their edges, a word that is only such marks left out."""
    tokens = (fold(word.strip(EDGE_MARKS)) for word in text.split())
    return [token for token in tokens if token]


class SpanIndex[Key: Hashable]:
    """Spellings, each a run of whole tokens, that stand for a key. When two spellings are the same
    run the first one given keeps it. With `case_suffix` a one-token word that is no spelling is read
    as the longest known one-token spelling of `MIN_PREFIX` characters or more that it starts with."""

    def __init__(
        self, spellings: Iterable[tuple[str, Key]], *, case_suffix: bool
    ) -> None:
        self._runs: dict[tuple[str, ...], Key] = {}
        for spelling, key in spellings:
            self._runs.setdefault(tuple(tokens_of(spelling)), key)
        self._longest = max((len(run) for run in self._runs), default=0)
        self._case_suffix = case_suffix

    def keys_in(self, text: str) -> list[Key]:
        """The keys whose spelling is in `text`, each once, in the order they first appear."""
        tokens = tokens_of(text)
        found: list[Key] = []
        position = 0
        while position < len(tokens):
            hit = self._at(tokens, position)
            if hit is None:
                position += 1
                continue
            key, width = hit
            if key not in found:
                found.append(key)
            position += width
        return found

    def _at(self, tokens: Sequence[str], position: int) -> tuple[Key, int] | None:
        for width in range(min(self._longest, len(tokens) - position), 0, -1):
            key = self._runs.get(tuple(tokens[position : position + width]))
            if key is not None:
                return key, width
        if self._case_suffix:
            word = tokens[position]
            for cut in range(len(word) - 1, MIN_PREFIX - 1, -1):
                key = self._runs.get((word[:cut],))
                if key is not None:
                    return key, 1
        return None


class EntityFinder:
    """The ids of the entities a sentence names. `entities` maps an id to a language to the
    spellings of the name in it, as the fixtures hold them."""

    def __init__(self, entities: Mapping[str, Mapping[str, Sequence[str]]]) -> None:
        languages = {language for by in entities.values() for language in by}
        self._indexes = {
            language: SpanIndex(
                (
                    (spelling, entity)
                    for entity, by_language in entities.items()
                    for spelling in by_language.get(language, ())
                ),
                case_suffix=language == HUB_LANGUAGE,
            )
            for language in languages
        }

    def ids_in(self, text: str, language: str) -> tuple[str, ...]:
        """The ids of the entities named in `text`, written in `language`, each once, in the order of
        the sentence; none for a language the table has no spelling in."""
        index = self._indexes.get(language)
        return () if index is None else tuple(index.keys_in(text))


class TermFinder:
    """The glossary terms of a sentence. `nouns` are rows with one word for each of `languages`, in
    that order: the same noun in each language."""

    def __init__(
        self, nouns: Sequence[Sequence[str]], languages: Sequence[str]
    ) -> None:
        if any(len(row) != len(languages) for row in nouns):
            raise ValueError("a noun row does not have the width of the languages")
        self._nouns = [tuple(row) for row in nouns]
        self._column = {language: index for index, language in enumerate(languages)}
        self._indexes = {
            language: SpanIndex(
                ((row[column], number) for number, row in enumerate(self._nouns)),
                case_suffix=False,
            )
            for language, column in self._column.items()
        }

    def pairs_in(
        self, text: str, source: str, target: str
    ) -> tuple[tuple[str, str], ...]:
        """`(word in source, word in target)` of each noun that `text`, written in `source`, holds as
        a whole token, each once, in the order of the sentence.

        Raises:
            ValueError: `source` or `target` is a language the table does not have.
        """
        if source not in self._column or target not in self._column:
            raise ValueError("the table has no word for that language")
        pairs = []
        for number in self._indexes[source].keys_in(text):
            row = self._nouns[number]
            pairs.append((row[self._column[source]], row[self._column[target]]))
        return tuple(pairs)
