"""The span metrics of D12: is a name or a term PRESENT in what a model wrote?

A spelling is present in a text when, after NFC and casefolding of both, it stands at a
boundary (no letter, mark or digit directly before it) and is followed by at most four
letters or marks (a case suffix: Khalkha `Улаанбаатарт`, `Улаанбаатараас`) and then
something that is not a letter, mark or digit. Runs of spaces inside a spelling match any
run of whitespace. The four is a prior; a short name is a prefix of a longer one (`Бат`
matches `Батор`), a known over-acceptance. Nothing is transliterated: a spelling is what
the table or the glossary lists."""

from __future__ import annotations

import unicodedata
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import lru_cache
from typing import Final

import regex

from rinchen.errors import DataError

MAX_SUFFIX_LETTERS: Final = 4

_WORD_CHARACTER = r"[\p{L}\p{M}\p{N}]"
_SUFFIX_CHARACTER = r"[\p{L}\p{M}]"
_WHITESPACE_RUN = regex.compile(r"\s+")
_PATTERN_CACHE_SIZE = 4096


@dataclass(frozen=True)
class EntityTable:
    """The accepted spellings of every entity of a suite: `id -> language -> spellings`."""

    spellings: Mapping[str, Mapping[str, tuple[str, ...]]]

    def spellings_of(self, entity_id: str, language: str) -> tuple[str, ...]:
        found = self.spellings.get(entity_id, {}).get(language)
        if not found:
            raise DataError(
                f"entity {entity_id!r} has no spelling for language {language!r}"
            )
        return found


@dataclass(frozen=True, slots=True)
class SpanOutcome:
    """How many spans of a row were expected, how many of them were found, and how many
    were left out of the count (a term whose source side is not in the source)."""

    expected: int
    found: int
    excluded: int


def span_present(
    text: str, spellings: Sequence[str], *, max_suffix: int = MAX_SUFFIX_LETTERS
) -> bool:
    """Whether any of `spellings` is present in `text`. No spelling, a blank spelling or a
    negative allowance is a `ValueError`, and a bare string is a `TypeError` (it would be
    read as a list of letters)."""
    if isinstance(spellings, str):
        raise TypeError("spellings must be a sequence of strings, not one string")
    if max_suffix < 0:
        raise ValueError("max_suffix must not be negative")
    return _pattern(tuple(spellings), max_suffix).search(_folded(text)) is not None


def score_entities(
    hypothesis: str, entity_ids: Sequence[str], tgt_lang: str, table: EntityTable
) -> SpanOutcome:
    """One mention per entity id of the row, found when one of its spellings in
    `tgt_lang` is present in the hypothesis."""
    found = sum(
        span_present(hypothesis, table.spellings_of(entity_id, tgt_lang))
        for entity_id in entity_ids
    )
    return SpanOutcome(expected=len(entity_ids), found=found, excluded=0)


def score_terms(
    source: str, hypothesis: str, terms: Sequence[tuple[str, str]]
) -> SpanOutcome:
    """One mention per glossary pair whose source term is present in the source, found
    when its target term is present in the hypothesis. A pair whose source term is absent
    is excluded and counted, never dropped."""
    expected = found = excluded = 0
    for source_term, target_term in terms:
        if not span_present(source, [source_term]):
            excluded += 1
            continue
        expected += 1
        if span_present(hypothesis, [target_term]):
            found += 1
    return SpanOutcome(expected=expected, found=found, excluded=excluded)


def _folded(text: str) -> str:
    return unicodedata.normalize("NFC", text).casefold()


@lru_cache(maxsize=_PATTERN_CACHE_SIZE)
def _pattern(spellings: tuple[str, ...], max_suffix: int) -> regex.Pattern[str]:
    if not spellings:
        raise ValueError("a span needs at least one spelling")
    alternatives = []
    for spelling in spellings:
        words = [word for word in _WHITESPACE_RUN.split(_folded(spelling)) if word]
        if not words:
            raise ValueError("a spelling must not be blank")
        alternatives.append(r"\s+".join(regex.escape(word) for word in words))
    return regex.compile(
        rf"(?<!{_WORD_CHARACTER})(?:{'|'.join(alternatives)})"
        rf"{_SUFFIX_CHARACTER}{{0,{max_suffix}}}(?!{_WORD_CHARACTER})"
    )
