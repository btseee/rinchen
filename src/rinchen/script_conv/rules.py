"""Rule baseline for Cyrillic to traditional-script conversion.

A word is looked up in the lexicon of verified spellings first. A word that is not
there is spelled by the phonographic fallback in `rinchen.script_conv.fallback`,
whose names are re-exported here so the plan's public names resolve from this
module. The lexicon is package data, loaded when this module is imported; the
fallback reads no file, so the fixture builder can regenerate a lost lexicon
without importing this module.

Input must be NFC (see `rinchen.text.normalize.normalize`).
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from importlib import resources
from types import MappingProxyType
from typing import Literal

from rinchen.errors import DataError
from rinchen.script_conv.alphabet import CYRILLIC_TO_SCRIPT_PUNCTUATION
from rinchen.script_conv.fallback import (
    FALLBACK_LETTERS,
    development_split,
    transliterate_fallback,
)
from rinchen.script_conv.segments import segment

__all__ = [
    "FALLBACK_LETTERS",
    "LEXICON",
    "WordResult",
    "development_split",
    "transliterate",
    "transliterate_fallback",
    "transliterate_word",
]

WordSource = Literal["lexicon", "fallback"]


@dataclass(frozen=True)
class WordResult:
    """A converted word and where its spelling came from."""

    text: str
    source: WordSource


_LEXICON_FILE = "lexicon.tsv"
_LEXICON_HEADER = ("cyrillic", "script")


def _parse_lexicon(text: str) -> dict[str, str]:
    """The word -> script pairs of a lexicon file's `text`.

    Raises DataError, naming the file and the 1-based line, for an empty file, a
    wrong header, a row that is not two tab-separated fields, an empty field, a blank
    line, a key with an upper-case letter (lookup lower-cases the word, so such a key
    could never be found) or a repeated key.
    """
    if not text:
        raise DataError(f"{_LEXICON_FILE}:1: the file is empty")
    lines = text.split("\n")
    if lines[-1] == "":
        lines.pop()
    if tuple(lines[0].split("\t")) != _LEXICON_HEADER:
        raise DataError(f"{_LEXICON_FILE}:1: expected header 'cyrillic<TAB>script'")
    entries: dict[str, str] = {}
    for number, line in enumerate(lines[1:], start=2):
        if not line:
            raise DataError(f"{_LEXICON_FILE}:{number}: blank line")
        fields = line.split("\t")
        if len(fields) != 2:
            raise DataError(
                f"{_LEXICON_FILE}:{number}: expected 2 tab-separated fields,"
                f" found {len(fields)}"
            )
        cyrillic, script = fields
        if not cyrillic or not script:
            raise DataError(f"{_LEXICON_FILE}:{number}: empty field")
        if cyrillic != cyrillic.lower():
            raise DataError(
                f"{_LEXICON_FILE}:{number}: key {cyrillic!r} is not lower-case"
            )
        if cyrillic in entries:
            raise DataError(f"{_LEXICON_FILE}:{number}: duplicate word {cyrillic!r}")
        entries[cyrillic] = script
    return entries


def _load_lexicon() -> Mapping[str, str]:
    try:
        text = (
            resources.files("rinchen.script_conv")
            .joinpath(_LEXICON_FILE)
            .read_text("utf-8")
        )
    except FileNotFoundError as error:
        raise DataError(f"{_LEXICON_FILE}: the package data file is missing") from error
    except UnicodeDecodeError as error:
        raise DataError(f"{_LEXICON_FILE}: the file is not valid UTF-8") from error
    return MappingProxyType(_parse_lexicon(text))


# Verified Cyrillic word -> traditional-script spelling; the development half of
# the reference (see `development_split`).
LEXICON: Mapping[str, str] = _load_lexicon()


def transliterate_word(word: str) -> WordResult:
    """The spelling of `word`: the verified lexicon entry if there is one, else the fallback.

    Lookup is on the lower-cased word, so `Монгол` and `МОНГОЛ` find the entry for
    `монгол`.
    """
    verified = LEXICON.get(word.lower())
    if verified is not None:
        return WordResult(verified, "lexicon")
    return WordResult(transliterate_fallback(word), "fallback")


# A full stop or comma that does not sit directly between two ASCII letters or digits.
# Inside a number, a domain name or an abbreviation such as `a.b` it is not
# sentence punctuation. The neighbours are looked up within one "other" segment, which
# is right: a neighbour outside the segment is Cyrillic or traditional script.
_SENTENCE_PUNCTUATION = re.compile(r"(?<![0-9A-Za-z])[.,]|[.,](?![0-9A-Za-z])")


def _script_punctuation(match: re.Match[str]) -> str:
    return CYRILLIC_TO_SCRIPT_PUNCTUATION[match.group()]


def transliterate(text: str) -> str:
    """Convert the Cyrillic words of `text` and leave everything else in place.

    Cyrillic runs go through `transliterate_word`. In the stretches between words, a
    full stop or comma becomes the script's own mark unless it sits directly between
    two ASCII letters or digits (`3.5`, `1,000`, `iPhone.com` stay; `5.`, `.5`, `...`
    and `т.м.` change). Digits, Latin runs, spaces, other punctuation and text already
    in traditional script pass through unchanged.
    """
    pieces: list[str] = []
    for piece in segment(text):
        if piece.kind == "cyrillic":
            pieces.append(transliterate_word(piece.text).text)
        elif piece.kind == "other":
            pieces.append(_SENTENCE_PUNCTUATION.sub(_script_punctuation, piece.text))
        else:
            pieces.append(piece.text)
    return "".join(pieces)
