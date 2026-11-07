"""Code points of the traditional Mongolian script, as the converter uses them.

Every constant is built from a code point number so the invisible and the
Mongolian characters stay reviewable in the source.
"""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType

# U+1820..U+1842: the letters Khalkha text uses. Todo, Sibe and Manchu letters
# (U+1843 and up) are deliberately outside the set.
LETTERS: frozenset[str] = frozenset(
    chr(code_point) for code_point in range(0x1820, 0x1843)
)

# Free variation selectors one to four (U+180B, U+180C, U+180D, U+180F).
FVS: frozenset[str] = frozenset(
    chr(code_point) for code_point in (0x180B, 0x180C, 0x180D, 0x180F)
)

# The two `rinchen.text.normalize.normalize` languages the converter reads and writes.
CYRILLIC_LANGUAGE = "khk_Cyrl"
SCRIPT_LANGUAGE = "mn_Mong"

MVS = chr(0x180E)  # Mongolian vowel separator
NNBSP = chr(0x202F)  # narrow no-break space, the gap before a separated suffix
NIRUGU = chr(0x180A)  # Mongolian nirugu, the stroke that joins or extends a word
JOINERS: frozenset[str] = frozenset({chr(0x200C), chr(0x200D)})  # ZWNJ, ZWJ

CYRILLIC_TO_SCRIPT_PUNCTUATION: Mapping[str, str] = MappingProxyType(
    {".": chr(0x1803), ",": chr(0x1802)}
)
SCRIPT_TO_CYRILLIC_PUNCTUATION: Mapping[str, str] = MappingProxyType(
    {script: cyrillic for cyrillic, script in CYRILLIC_TO_SCRIPT_PUNCTUATION.items()}
)
