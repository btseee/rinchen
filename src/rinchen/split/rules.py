"""Rule-based sentence splitting, one rule family per script.

`split` routes a language code to its rule family. A family finds the offsets
where a sentence ends; `_parts_between` turns those offsets into the stripped,
non-empty parts. A new family registers its splitter in `_SPLITTERS`.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable
from itertools import pairwise

from rinchen.errors import ConfigError

# Quotes and brackets that close a sentence; they stay with the sentence they close.
_CLOSERS = "\"'\u00bb\u201d\u2019)]\uff09\u300d\u300f"

# U+1803 MONGOLIAN FULL STOP and U+1809 MONGOLIAN MANCHU FULL STOP.
# U+1802 MONGOLIAN COMMA is absent on purpose: a comma never ends a sentence.
_MN_MONG_TERMINATORS = "\u1803\u1809?!"
# U+3002 IDEOGRAPHIC FULL STOP, U+FF01 FULLWIDTH EXCLAMATION MARK,
# U+FF1F FULLWIDTH QUESTION MARK.
_CJK_TERMINATORS = "\u3002\uff01\uff1f"


def _boundary_after(terminators: str) -> re.Pattern[str]:
    """Match a run of `terminators` plus closers, only when more text follows.

    The quantifiers are possessive so that a run is never matched in part: at
    the end of the text, backtracking would otherwise cut "?!" between its
    characters, because the "!" counts as text that follows the "?".
    """
    return re.compile(
        rf"[{re.escape(terminators)}]++[{re.escape(_CLOSERS)}]*+(?=\s*\S)"
    )


_MN_MONG_BOUNDARY = _boundary_after(_MN_MONG_TERMINATORS)
_CJK_BOUNDARY = _boundary_after(_CJK_TERMINATORS)


def _parts_between(text: str, cut_offsets: Iterable[int]) -> list[str]:
    bounds = [0, *cut_offsets, len(text)]
    pieces = (text[start:end].strip() for start, end in pairwise(bounds))
    return [piece for piece in pieces if piece]


def _split_at_boundaries(boundary: re.Pattern[str], text: str) -> list[str]:
    return _parts_between(text, (match.end() for match in boundary.finditer(text)))


def _split_mn_mong(text: str) -> list[str]:
    return _split_at_boundaries(_MN_MONG_BOUNDARY, text)


def _split_cjk(text: str) -> list[str]:
    return _split_at_boundaries(_CJK_BOUNDARY, text)


_SPLITTERS: dict[str, Callable[[str], list[str]]] = {
    "mn_Mong": _split_mn_mong,
    "zho_Hans": _split_cjk,
    "jpn_Jpan": _split_cjk,
}


def split(text: str, lang: str) -> list[str]:
    """Split `text` into sentences using the rules for `lang`.

    Parts are stripped, non-empty and in order; joined, they equal `text` minus
    its whitespace. Empty or whitespace-only text gives `[]`. Raises ConfigError
    if `lang` has no splitting rules.
    """
    splitter = _SPLITTERS.get(lang)
    if splitter is None:
        raise ConfigError(
            f"no sentence-splitting rules for language {lang!r}; "
            f"expected one of {sorted(_SPLITTERS)}"
        )
    return splitter(text)
