"""Unicode script ratios over the letters of a text."""

from __future__ import annotations

from collections import Counter
from enum import Enum

import regex


class Script(str, Enum):
    """Scripts D3 tells apart; each value is the `regex` ``\\p{Script=...}`` name."""

    CYRILLIC = "Cyrillic"
    MONGOLIAN = "Mongolian"
    LATIN = "Latin"
    HAN = "Han"
    HANGUL = "Hangul"
    HIRAGANA = "Hiragana"
    KATAKANA = "Katakana"


_LETTER = regex.compile(r"\p{L}")
_SCRIPT_PATTERNS = {
    script: regex.compile(rf"\p{{Script={script.value}}}") for script in Script
}


def _script_of_letter(letter: str) -> Script | None:
    for script, pattern in _SCRIPT_PATTERNS.items():
        if pattern.match(letter):
            return script
    return None


def script_of(text: str) -> dict[Script, float]:
    """Share of each script among all letters of `text`.

    Digits, punctuation, marks and whitespace are not letters and are ignored.
    A letter of a script outside `Script` (e.g. Greek) counts in the
    denominator only, so the shares sum to at most 1.0. Text without letters
    yields an empty dict; scripts with no letters are absent.
    """
    letter_count = 0
    script_counts: Counter[Script] = Counter()
    for match in _LETTER.finditer(text):
        letter_count += 1
        script = _script_of_letter(match.group())
        if script is not None:
            script_counts[script] += 1
    return {script: count / letter_count for script, count in script_counts.items()}
