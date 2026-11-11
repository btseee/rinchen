"""Content scans of the tiny pipeline's texts (D15), shared by the tests of its fixtures,
configurations, reports and documents.

The project's content rules (CLAUDE.md) exclude two subjects and three names from every file
the pipeline reads or writes. The excluded words are spelled here as decimal code points, and
the two names are split, so that this module holds none of them and passes its own scan. A hit
names the line and the number of the word or name and never the text itself.

`character_class` is the audit of the TSV fixtures: it sorts every character into the few
classes a fixture may hold and gives anything else a class of its own, named by code point.
"""

from __future__ import annotations

from collections import Counter
from typing import Final

PLACEHOLDER_OPEN: Final = chr(0x27E6)
PLACEHOLDER_CLOSE: Final = chr(0x27E7)
EM_DASH: Final = chr(0x2014)

# Lower-case stems, one string of decimal code points per word.
_WORD_CODE_POINTS: Final = (
    "108 105 99 101 110",
    "99 111 112 121 114 105 103 104 116",
    "108 101 103 97 108",
    "108 97 119 115 117 105 116",
    "108 97 119 102 117 108",
    "108 97 119 121 101 114",
    "116 114 97 100 101 109 97 114 107",
    "112 97 116 101 110 116",
    "108 105 97 98 105 108 105 116 121",
    "119 97 114 114 97 110 116 121",
    "116 101 114 109 115 32 111 102 32 117 115 101",
    "116 101 114 109 115 32 111 102 32 115 101 114 118 105 99 101",
    "115 112 100 120",
    "97 116 116 111 114 110 101 121",
    "114 111 121 97 108 116",
    "101 117 108 97",
    "100 109 99 97",
    "106 117 114 105 115 100 105 99 116 105 111 110",
)
# The two names the project's content rules refuse by name (a Latin spelling of Mongolian has
# no name to scan for: the script checks of the tests cover it).
_NAME_PARTS: Final = (("khk_", "Mong"), ("FLO", "RES"))


def excluded_words() -> tuple[str, ...]:
    """The lower-case stems of the excluded subjects."""
    return tuple(
        "".join(chr(int(code)) for code in points.split())
        for points in _WORD_CODE_POINTS
    )


def excluded_names() -> tuple[str, ...]:
    """The lower-case names the project never uses."""
    return tuple("".join(parts).lower() for parts in _NAME_PARTS)


def hits(text: str) -> list[tuple[int, str]]:
    """`(line number, label)` of every excluded word or name `text` holds, case ignored.

    The label is `word N` or `name N`, the number of the entry in the lists above.
    """
    words, names = excluded_words(), excluded_names()
    found: list[tuple[int, str]] = []
    for number, line in enumerate(text.lower().splitlines(), start=1):
        found += [(number, f"word {n}") for n, word in enumerate(words) if word in line]
        found += [(number, f"name {n}") for n, name in enumerate(names) if name in line]
    return found


def character_class(char: str) -> str:
    """The class of one character of a fixture: `ascii` (printable, tab or line feed),
    `cyrillic`, `mongolian`, `placeholder bracket`, `em dash`, or `other U+XXXX`."""
    code = ord(char)
    if char in "\t\n" or 0x20 <= code <= 0x7E:
        return "ascii"
    if 0x0400 <= code <= 0x04FF:
        return "cyrillic"
    if 0x1800 <= code <= 0x18AF:
        return "mongolian"
    if char in (PLACEHOLDER_OPEN, PLACEHOLDER_CLOSE):
        return "placeholder bracket"
    if char == EM_DASH:
        return "em dash"
    return f"other U+{code:04X}"


def character_classes(text: str) -> Counter[str]:
    """How many characters of each class `text` holds."""
    return Counter(character_class(char) for char in text)
