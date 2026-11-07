"""Phonographic fallback: the data-free half of the rule baseline.

A word that is not in the lexicon is spelled by one Cyrillic-letter to script-letter
map plus five contextual rules. Each rule was kept only because it raised the
exact-match rate on the development half of the Wiktionary reference by at least
0.5 percentage points, and each is stated in one sentence in the comment above it.

The fallback is a measured scaffold, not an orthography. The reference spells words
the dictionary way, which is not letter for letter, so the fallback matches only a
minority of them; its held-out figure is in the runbook. It never emits a joiner, a
variation selector, a vowel separator or an NNBSP: those appear only in lexicon
spellings, and a suffix gap is not guessed (the spec's ruling 3).

Input must be NFC (see `rinchen.text.normalize.normalize`). This module reads no
file, so the fixture builder can import it to regenerate the lexicon.

Mongolian letters are built from code points so they stay reviewable in the source.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from types import MappingProxyType


def development_split(word: str) -> bool:
    """Whether `word` belongs to the development half of the reference.

    The split is the parity of the SHA-256 of the word's UTF-8 bytes, so it needs no
    stored list and cannot drift. Rule development reads only this half; the other
    half is the held-out half that measures the fallback.
    """
    return int(hashlib.sha256(word.encode("utf-8")).hexdigest(), 16) % 2 == 0


_A = chr(0x1820)
_E = chr(0x1821)
_I = chr(0x1822)
_O = chr(0x1823)
_U = chr(0x1824)
_OE = chr(0x1825)
_UE = chr(0x1826)
_NA = chr(0x1828)
_BA = chr(0x182A)
_PA = chr(0x182B)
_QA = chr(0x182C)
_GA = chr(0x182D)
_MA = chr(0x182E)
_LA = chr(0x182F)
_SA = chr(0x1830)
_SHA = chr(0x1831)
_TA = chr(0x1832)
_DA = chr(0x1833)
_CHA = chr(0x1834)
_JA = chr(0x1835)
_YA = chr(0x1836)
_RA = chr(0x1837)
_FA = chr(0x1839)
_KA = chr(0x183A)

# One entry per letter of the Khalkha Cyrillic alphabet. The choices that are not the
# plain letter-for-letter ones were each tried on the development half:
# в -> ba, й and ь -> i, ж and з -> ja, ц and ч -> cha. ъ is dropped; a letter outside
# this map is dropped too (see `transliterate_fallback`).
_LETTER_MAP: Mapping[str, str] = MappingProxyType(
    {
        "а": _A,
        "э": _E,
        "и": _I,
        "о": _O,
        "у": _U,
        "ө": _OE,
        "ү": _UE,
        "ы": _I,
        "й": _I,
        "ь": _I,
        "ъ": "",
        "я": _YA + _A,
        "ю": _YA + _U,
        "е": _YA + _E,
        "ё": _YA + _O,
        "б": _BA,
        "в": _BA,
        "п": _PA,
        "ф": _FA,
        "м": _MA,
        "н": _NA,
        "л": _LA,
        "р": _RA,
        "с": _SA,
        "ш": _SHA,
        "щ": _SHA + _CHA,
        "т": _TA,
        "д": _DA,
        "ц": _CHA,
        "ч": _CHA,
        "ж": _JA,
        "з": _JA,
        "г": _GA,
        "х": _QA,
        "к": _KA,
    }
)

# The script characters the fallback can emit; a lexicon spelling with any other
# character (a variation selector, a vowel separator, an NNBSP, a rarer letter) is
# one the fallback cannot reproduce.
FALLBACK_LETTERS: frozenset[str] = frozenset("".join(_LETTER_MAP.values()))

_VOWEL_LETTERS = frozenset("аэиоуөүыяюеё")
# Vowel harmony: a word is a front-vowel word when it holds one of these letters.
_FRONT_VOWEL_LETTERS = frozenset("эөүе")
_DOUBLING_VOWEL_LETTERS = frozenset("аэоуөү")


@dataclass
class _Slot:
    """One Cyrillic letter and the script text it currently becomes."""

    letter: str
    script: str


_Rule = Callable[[list[_Slot], bool], None]


# After the word's first vowel letter, о is written U and ө is written UE.
def _later_round_vowels_are_high(slots: list[_Slot], front: bool) -> None:
    first_vowel = next(
        (i for i, slot in enumerate(slots) if slot.letter in _VOWEL_LETTERS),
        len(slots),
    )
    for slot in slots[first_vowel + 1 :]:
        if slot.letter == "о":
            slot.script = _U
        elif slot.letter == "ө":
            slot.script = _UE


# The second of two equal vowel letters (аа, ээ, оо, уу, өө, үү) is written GA and then
# its own vowel.
def _doubled_vowel_gets_ga(slots: list[_Slot], front: bool) -> None:
    for index in range(1, len(slots)):
        letter = slots[index].letter
        is_pair = (
            letter in _DOUBLING_VOWEL_LETTERS and slots[index - 1].letter == letter
        )
        starts_run = index < 2 or slots[index - 2].letter != letter
        if is_pair and starts_run:
            slots[index].script = _GA + slots[index].script


# A word-final х takes U in a back-vowel word and UE in a front-vowel word.
def _final_kh_takes_a_rounded_vowel(slots: list[_Slot], front: bool) -> None:
    if slots and slots[-1].letter == "х":
        slots[-1].script += _UE if front else _U


# A word-final ч or ж takes I.
def _final_ch_or_zh_takes_i(slots: list[_Slot], front: bool) -> None:
    if slots and slots[-1].letter in {"ч", "ж"}:
        slots[-1].script += _I


# A word-final т, ц or з takes A in a back-vowel word and E in a front-vowel word.
def _final_t_ts_z_takes_a_or_e(slots: list[_Slot], front: bool) -> None:
    if slots and slots[-1].letter in {"т", "ц", "з"}:
        slots[-1].script += _E if front else _A


# Applied in this order: the high-vowel rule runs before the doubling rule, which
# prefixes whatever script text the vowel has by then.
_CONTEXT_RULES: tuple[_Rule, ...] = (
    _later_round_vowels_are_high,
    _doubled_vowel_gets_ga,
    _final_kh_takes_a_rounded_vowel,
    _final_ch_or_zh_takes_i,
    _final_t_ts_z_takes_a_or_e,
)


def transliterate_fallback(word: str) -> str:
    """The lexicon-free phonographic spelling of `word`.

    `word` is a run of Cyrillic letters in NFC, as the segmenter hands it over; it is
    lower-cased first. A character that is not in the letter map, such as the Kazakh
    letter қ, is dropped, and nothing counts or reports it. The output holds only
    letters of the script, so it always passes `validate_traditional`.
    """
    slots = [
        _Slot(letter, _LETTER_MAP[letter])
        for letter in word.lower()
        if letter in _LETTER_MAP
    ]
    front = any(slot.letter in _FRONT_VOWEL_LETTERS for slot in slots)
    for rule in _CONTEXT_RULES:
        rule(slots, front)
    return "".join(slot.script for slot in slots)
