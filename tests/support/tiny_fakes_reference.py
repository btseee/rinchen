"""An independent model of the planted encoder of the tiny pipeline (D15, `rinchen.tiny.fakes`), for
the differential and the property tests.

The encoder is a bag of concepts: each concept is a fixed pseudo-random 64-dimensional vector seeded
by the CRC-32 of its NAME, the bag is summed and normalised. The names are the ones the measured
numbers of the filter rest on (`n<row>` for a noun, `e:<id>` for an entity, `d:<digits>` for a
number, `p:<token>` for a placeholder), so a test that pins them pins the measurements. Here the text
is read by a character loop and not by the encoder's regular expression, the lexicon is rebuilt from
the fixture tables, and the vectors are summed in float64 by numpy; the encoder and this module agree
only if both read the same language.

The generators of random texts build a text from lexicon units (the words of the three languages and
their traditional-script spellings, names with a Khalkha case suffix glued on, numbers in the
conventions of the fixtures, placeholders and unknown words) and say nothing of the concepts: the
model is what knows them.
"""

from __future__ import annotations

import random
import zlib
from collections.abc import Mapping, Sequence
from typing import Final

import numpy as np

from rinchen.script_conv.rules import transliterate

DIMENSIONS: Final = 64
PLACEHOLDER_OPEN: Final = chr(0x27E6)
PLACEHOLDER_CLOSE: Final = chr(0x27E7)
MONGOLIAN_COMMA: Final = chr(0x1802)
MONGOLIAN_FULL_STOP: Final = chr(0x1803)
SEPARATOR_CHARACTERS: Final = ".,?!:;"
# Khalkha case suffixes (ablative, genitive, dative), written as code points.
SUFFIXES: Final = ("\u0430\u0430\u0441", "\u044b\u043d", "\u0434")
NUMBERS: Final = ("42", "7", "3.5", "0,5", "1,250.50", "1 250,50", "2024")


def vector(concept: str) -> np.ndarray:
    """The fixed vector of a concept, before it is summed and normalised."""
    rng = np.random.default_rng(zlib.crc32(concept.encode("utf-8")))
    return rng.standard_normal(DIMENSIONS)


def concept_spellings(
    nouns: Sequence[Sequence[str]],
    entities: Mapping[str, Mapping[str, Sequence[str]]],
) -> dict[str, tuple[str, ...]]:
    """Every spelling a concept has in the fixtures, by the name of the concept: the three words
    of a noun and the traditional spelling of its Khalkha word; each spelling of an entity (the ones
    of one word) and the traditional spelling of each."""
    spellings: dict[str, tuple[str, ...]] = {}
    for index, row in enumerate(nouns):
        spellings[f"n{index}"] = (*row, transliterate(row[0]))
    for entity_id, by_language in entities.items():
        names = [
            name
            for language_names in by_language.values()
            for name in language_names
            if " " not in name
        ]
        spellings[f"e:{entity_id}"] = (*names, *(transliterate(n) for n in names))
    return spellings


def lexicon(
    nouns: Sequence[Sequence[str]],
    entities: Mapping[str, Mapping[str, Sequence[str]]],
) -> dict[str, str]:
    """The lower-cased spelling of each word, by the concept it names (a later row overwrites an
    earlier one), as the encoder builds it from the same tables."""
    table: dict[str, str] = {}
    for index, row in enumerate(nouns):
        for word in row:
            table[word.lower()] = f"n{index}"
        table[transliterate(row[0])] = f"n{index}"
    for entity_id, by_language in entities.items():
        for names in by_language.values():
            for name in names:
                table[name.lower()] = f"e:{entity_id}"
                table[transliterate(name)] = f"e:{entity_id}"
    return table


def tokens(text: str) -> list[str]:
    """The numbers (digits, then a separator and one to three digits, as often as it goes), the
    placeholders and the runs of other characters, in order; whitespace and the separators
    `. , ? ! : ;` are between tokens and in none."""
    found: list[str] = []
    i, n = 0, len(text)
    while i < n:
        char = text[i]
        if char.isspace() or char in SEPARATOR_CHARACTERS:
            i += 1
            continue
        if char == PLACEHOLDER_OPEN:
            j = i + 1
            while j < n and text[j].isdecimal():
                j += 1
            if j > i + 1 and j < n and text[j] == PLACEHOLDER_CLOSE:
                found.append(text[i : j + 1])
                i = j + 1
                continue
        if char.isdecimal():
            j = i
            while j < n and text[j].isdecimal():
                j += 1
            while j + 1 < n and text[j] in " .," and text[j + 1].isdecimal():
                k = j + 1
                while k < n and k - (j + 1) < 3 and text[k].isdecimal():
                    k += 1
                j = k
            found.append(text[i:j])
            i = j
            continue
        j = i
        while j < n and not text[j].isspace() and text[j] not in SEPARATOR_CHARACTERS:
            j += 1
        found.append(text[i:j])
        i = j
    return found


def concepts(text: str, table: Mapping[str, str]) -> list[str]:
    """The concepts of a text: the digits of a number, a placeholder, a word of the lexicon, or the
    word cut back to its longest known prefix of three characters or more; any other word names
    none."""
    found: list[str] = []
    for token in tokens(text):
        if token[0].isdecimal():
            found.append("d:" + "".join(c for c in token if c.isdecimal()))
        elif token.startswith(PLACEHOLDER_OPEN):
            found.append("p:" + token)
        else:
            named = table.get(token.lower())
            cut = len(token) - 1
            while named is None and cut >= 3:
                named = table.get(token[:cut].lower())
                cut -= 1
            if named is not None:
                found.append(named)
    return found


def embedding(text: str, table: Mapping[str, str]) -> np.ndarray:
    """The unit vector of a text, in float64; a text with no concept has the fixed empty vector."""
    bag = concepts(text, table)
    total = sum((vector(c) for c in bag), np.zeros(DIMENSIONS))
    if not bag:
        total = vector("<empty>")
    return total / np.linalg.norm(total)


def cosine(a: str, b: str, table: Mapping[str, str]) -> float:
    return float(embedding(a, table) @ embedding(b, table))


def unit_pools(
    nouns: Sequence[Sequence[str]],
    entities: Mapping[str, Mapping[str, Sequence[str]]],
) -> dict[str, list[str]]:
    """The units a random text is made of, by kind."""
    spellings = concept_spellings(nouns, entities)
    words = [word for group in spellings.values() for word in group]
    return {
        "word": words,
        "suffixed": [word for word in words if len(word) >= 3],
        "number": list(NUMBERS),
        "placeholder": [f"{PLACEHOLDER_OPEN}{k}{PLACEHOLDER_CLOSE}" for k in range(10)],
        "unknown": ["zzz", "qwerty", "lorem", "ipsum", "xx", "a1b2"],
        "glued": [MONGOLIAN_COMMA, MONGOLIAN_FULL_STOP],
    }


def random_text(rnd: random.Random, pools: Mapping[str, Sequence[str]]) -> str:
    """A text of one to seven units, separated by spaces or by the separator characters, now and
    then with a Khalkha suffix or a Mongolian punctuation mark glued to a word."""
    parts: list[str] = []
    for _ in range(rnd.randint(1, 7)):
        kind = rnd.choices(
            ["word", "suffixed", "number", "placeholder", "unknown"],
            weights=[8, 3, 3, 2, 2],
        )[0]
        unit = rnd.choice(pools[kind])
        if kind == "suffixed":
            unit += rnd.choice(SUFFIXES)
        elif kind == "word" and rnd.random() < 0.15:
            unit += rnd.choice(pools["glued"])
        if rnd.random() < 0.3:
            unit = unit.upper() if rnd.random() < 0.5 else unit.capitalize()
        parts.append(unit)
        parts.append(rnd.choice([" ", " ", ", ", ". ", " - ", "? ", ": "]))
    return "".join(parts[:-1]) + rnd.choice(["", ".", "!"])
