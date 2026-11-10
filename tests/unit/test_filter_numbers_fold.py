"""D12 moved the numeral folding of D4's `numbers` stage to `rinchen.text.numerals`. D4 must
behave exactly as before: `digit_multiset` equals a frozen copy of its pre-move body (kept
below with its own constants) for every text, and the stage module holds no private copy of
the folding and imports nothing private."""

from __future__ import annotations

import ast
import re
import unicodedata
from collections import Counter
from pathlib import Path

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from rinchen.filter import numbers
from rinchen.filter.numbers import digit_multiset

# --- the pre-move body of rinchen.filter.numbers, frozen -----------------------------

_OLD_DIGIT_VALUE = {
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
_OLD_MULTIPLIER = {chr(0x5341): 10, chr(0x767E): 100, chr(0x5343): 1000}
_OLD_TEN_THOUSAND = chr(0x4E07)
_OLD_MAGNITUDES = frozenset([*_OLD_MULTIPLIER, _OLD_TEN_THOUSAND])
_OLD_NUMERAL_RUN = re.compile(
    "["
    + re.escape("".join([*_OLD_DIGIT_VALUE, *_OLD_MULTIPLIER, _OLD_TEN_THOUSAND]))
    + "]+"
)


def old_digit_multiset(text: str) -> Counter[str]:
    plain = _OLD_NUMERAL_RUN.sub(lambda run: _old_numeral_digits(run.group()), text)
    return Counter(
        str(value)
        for char in plain
        if (value := unicodedata.decimal(char, None)) is not None
    )


def _old_numeral_digits(run: str) -> str:
    if len(run) == 1 and run in _OLD_DIGIT_VALUE:
        return ""
    if _OLD_MAGNITUDES.isdisjoint(run):
        return "".join(str(_OLD_DIGIT_VALUE[char]) for char in run)
    return str(_old_summed_value(run))


def _old_summed_value(run: str) -> int:
    total = section = digit = 0
    for char in run:
        if char in _OLD_DIGIT_VALUE:
            digit = _OLD_DIGIT_VALUE[char]
        elif char == _OLD_TEN_THOUSAND:
            total += ((section + digit) or 1) * 10_000
            section = digit = 0
        else:
            section += (digit or 1) * _OLD_MULTIPLIER[char]
            digit = 0
    return total + section + digit


# --- the alphabets ---------------------------------------------------------------------

DIGIT_SCRIPTS = "".join(
    chr(base + digit)
    for base in (0x30, 0x660, 0xFF10, 0x1810, 0x966)
    for digit in range(10)
)
CJK_NUMERALS = "".join(
    chr(code)
    for code in (
        *(0x3007, 0x96F6, 0x4E00, 0x4E8C, 0x4E09, 0x56DB, 0x4E94, 0x516D),
        *(0x4E03, 0x516B, 0x4E5D, 0x5341, 0x767E, 0x5343, 0x4E07),
    )
)
SEPARATORS_AND_LETTERS = ".,:/- %abcXYZ" + "".join(map(chr, (0x430, 0x431, 0x4E8)))
NUMERAL_ALPHABET = st.sampled_from(
    list(DIGIT_SCRIPTS + CJK_NUMERALS * 3 + SEPARATORS_AND_LETTERS)
)


@settings(max_examples=300)
@given(st.text(max_size=80))
def test_digit_multiset_equals_the_frozen_pre_move_body_for_any_text(
    text: str,
) -> None:
    assert digit_multiset(text) == old_digit_multiset(text)


@settings(max_examples=300)
@given(st.text(alphabet=NUMERAL_ALPHABET, max_size=40))
def test_digit_multiset_equals_the_frozen_pre_move_body_for_numeral_text(
    text: str,
) -> None:
    assert digit_multiset(text) == old_digit_multiset(text)


@pytest.mark.parametrize(
    "text",
    [
        "",
        "1,250.50",
        chr(0x4E8C) + chr(0x3007) + chr(0x4E8C) + chr(0x56DB),
        chr(0x4E09) + chr(0x6708),
        chr(0x4E07) + chr(0x4E00),
        chr(0x1811) + chr(0x1812) + "3",
    ],
)
def test_digit_multiset_keeps_the_values_the_d4_tests_pin(text: str) -> None:
    assert digit_multiset(text) == old_digit_multiset(text)


def test_the_stage_module_holds_no_private_copy_of_the_folding() -> None:
    for name in (
        "_DIGIT_VALUE",
        "_MULTIPLIER",
        "_TEN_THOUSAND",
        "_MAGNITUDES",
        "_NUMERAL_RUN",
        "_numeral_digits",
        "_summed_value",
    ):
        assert not hasattr(numbers, name), name


def test_the_stage_module_imports_nothing_private() -> None:
    source = Path(numbers.__file__).read_text(encoding="utf-8")

    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.ImportFrom):
            assert not any(alias.name.startswith("_") for alias in node.names)


def test_the_stage_module_takes_the_folding_from_text_numerals() -> None:
    source = Path(numbers.__file__).read_text(encoding="utf-8")

    imported = {
        (node.module, alias.name)
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.ImportFrom)
        for alias in node.names
    }

    assert ("rinchen.text.numerals", "fold_numerals") in imported
