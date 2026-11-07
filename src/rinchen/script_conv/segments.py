"""Splits text into Cyrillic, traditional-script and other runs.

The converter turns Cyrillic runs into traditional script one word at a time and
passes everything else through, so the split has to keep a traditional-script word
whole: it is cut at a plain space, never at the NNBSP that precedes a separated
suffix. Concatenating the segments reproduces the input exactly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import regex

from rinchen.script_conv.alphabet import FVS, JOINERS, LETTERS, MVS, NIRUGU, NNBSP

SegmentKind = Literal["cyrillic", "mongolian", "other"]


@dataclass(frozen=True)
class Segment:
    """A maximal run of text of one `kind`."""

    text: str
    kind: SegmentKind


def _character_class(characters: frozenset[str]) -> str:
    # Escaped by code point so the pattern keeps the invisible characters legible.
    return "[" + "".join(f"\\u{ord(char):04x}" for char in sorted(characters)) + "]"


_RUN_CHARACTER = _character_class(LETTERS | FVS | JOINERS | {MVS, NIRUGU})
_LETTER = _character_class(LETTERS)
_NNBSP = f"\\u{ord(NNBSP):04x}"

# A Cyrillic letter is a letter of Script=Cyrillic; the combining marks that
# directly follow it belong to the same segment. Apostrophes and hyphens are not
# letters, so they end the run.
_CYRILLIC_RUN = r"(?:(?=\p{L})\p{Script=Cyrillic}\p{M}*)+"

# An NNBSP joins a run only between a run character and a letter, so the gap
# before a separated suffix stays inside its word.
_MONGOLIAN_RUN = rf"{_RUN_CHARACTER}+(?:{_NNBSP}(?={_LETTER}){_RUN_CHARACTER}+)*"

_RUNS = regex.compile(rf"(?P<cyrillic>{_CYRILLIC_RUN})|(?P<mongolian>{_MONGOLIAN_RUN})")


def segment(text: str) -> tuple[Segment, ...]:
    """Split `text` into Cyrillic, Mongolian and other segments, in order.

    A Cyrillic segment is a maximal run of Cyrillic letters with their combining
    marks. A Mongolian segment is a maximal run of Mongolian letters, free variation
    selectors, vowel separators, nirugu strokes and joiners, plus an NNBSP that sits
    between such a character and a letter. Every other stretch, however long, is
    one "other" segment, so a leading, trailing or doubled NNBSP, or one next to a
    space, lands there.
    """
    segments: list[Segment] = []
    position = 0
    for run in _RUNS.finditer(text):
        if run.start() > position:
            segments.append(Segment(text[position : run.start()], "other"))
        kind: SegmentKind = "cyrillic" if run.lastgroup == "cyrillic" else "mongolian"
        segments.append(Segment(run.group(), kind))
        position = run.end()
    if position < len(text):
        segments.append(Segment(text[position:], "other"))
    return tuple(segments)
