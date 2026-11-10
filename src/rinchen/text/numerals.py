"""Numerals read as ASCII digits, shared by D4's `numbers` stage and D12's numbers metric.

Every Unicode decimal digit is read as its ASCII digit and every run of CJK numerals as the
digits of its value, so Mongolian digits, fullwidth digits and `<ten><two>` all come out as
plain digits. The module imports the standard library only: it sits below every layer that
reads numbers."""

from __future__ import annotations

import re
import unicodedata

_DIGIT_VALUE = {
    chr(0x3007): 0,
    chr(0x96F6): 0,
    chr(0x4E00): 1,
    chr(0x4E8C): 2,
    chr(0x4E09): 3,
    chr(0x56DB): 4,
    chr(0x4E94): 5,
    chr(0x516D): 6,
    chr(0x4E03): 7,
    chr(0x516B): 8,
    chr(0x4E5D): 9,
}
_MULTIPLIER = {chr(0x5341): 10, chr(0x767E): 100, chr(0x5343): 1000}
_TEN_THOUSAND = chr(0x4E07)
_MAGNITUDES = frozenset([*_MULTIPLIER, _TEN_THOUSAND])
_NUMERAL_RUN = re.compile(
    "[" + re.escape("".join([*_DIGIT_VALUE, *_MULTIPLIER, _TEN_THOUSAND])) + "]+"
)


def fold_numerals(text: str) -> str:
    """`text` with every run of CJK numerals replaced by the digits it spells (nothing for
    a lone digit character, which is more often 'a' or 'one' than a number) and every
    character that has a Unicode decimal value replaced by its ASCII digit. Every other
    character is kept."""
    plain = _NUMERAL_RUN.sub(lambda run: _numeral_digits(run.group()), text)
    return "".join(
        str(value) if (value := unicodedata.decimal(char, None)) is not None else char
        for char in plain
    )


def _numeral_digits(run: str) -> str:
    """The digits of the number a run of CJK numerals writes; nothing for a single digit
    character, which is more often 'a' or 'one' than a number."""
    if len(run) == 1 and run in _DIGIT_VALUE:
        return ""
    if _MAGNITUDES.isdisjoint(run):
        return "".join(str(_DIGIT_VALUE[char]) for char in run)
    return str(_summed_value(run))


def _summed_value(run: str) -> int:
    """Value of a run with multipliers, as in `<one><hundred><two><ten><three>`
    for 123. A multiplier without a digit before it counts that digit as one."""
    total = section = digit = 0
    for char in run:
        if char in _DIGIT_VALUE:
            digit = _DIGIT_VALUE[char]
        elif char == _TEN_THOUSAND:
            total += ((section + digit) or 1) * 10_000
            section = digit = 0
        else:
            section += (digit or 1) * _MULTIPLIER[char]
            digit = 0
    return total + section + digit
