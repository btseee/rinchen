"""`rinchen.text.numerals.fold_numerals`: the numeral folding D4 wrote and D12 shares. Every
run of CJK numerals becomes the digits it spells (a lone digit character, which is more
often "a" or "one" than a number, becomes nothing) and every character with a Unicode
decimal value becomes its ASCII digit; every other character stays. The module sits at
the bottom of the layers: it imports the standard library only."""

from __future__ import annotations

import ast
import unicodedata
from pathlib import Path

import pytest
from hypothesis import given
from hypothesis import strategies as st

import rinchen.text.numerals
from rinchen.text.numerals import fold_numerals

ARABIC_INDIC = "".join(chr(0x660 + digit) for digit in range(10))
FULLWIDTH = "".join(chr(0xFF10 + digit) for digit in range(10))
MONGOLIAN = "".join(chr(0x1810 + digit) for digit in range(10))
DEVANAGARI = "".join(chr(0x966 + digit) for digit in range(10))

CIRCLE = chr(0x3007)
ZERO = chr(0x96F6)
ONE = chr(0x4E00)
TWO = chr(0x4E8C)
THREE = chr(0x4E09)
FOUR = chr(0x56DB)
TEN = chr(0x5341)
HUNDRED = chr(0x767E)
THOUSAND = chr(0x5343)
TEN_THOUSAND = chr(0x4E07)
MONTH = chr(0x6708)
CJK_NUMERALS = "".join(
    [CIRCLE, ZERO, ONE, TWO, THREE, FOUR]
    + [chr(code) for code in (0x4E94, 0x516D, 0x4E03, 0x516B, 0x4E5D)]
    + [TEN, HUNDRED, THOUSAND, TEN_THOUSAND]
)


@pytest.mark.parametrize(
    "text",
    [
        "",
        "no digit at all",
        "Page 12, 3.5 and 2024.",
        "1,250.50 / 10:30 / -5",
        "Ном 1 250 байна.",
        MONTH,
    ],
)
def test_text_without_a_foreign_digit_is_returned_equal(text: str) -> None:
    assert fold_numerals(text) == text


@pytest.mark.parametrize(
    ("text", "folded"),
    [
        (ARABIC_INDIC[1:4], "123"),
        (FULLWIDTH[1:3], "12"),
        (MONGOLIAN[1:3], "12"),
        (DEVANAGARI[4:6], "45"),
        (ARABIC_INDIC + FULLWIDTH + MONGOLIAN, "0123456789" * 3),
        (f"a{ARABIC_INDIC[1]}b{FULLWIDTH[2]}c", "a1b2c"),
    ],
    ids=["arabic-indic", "fullwidth", "mongolian", "devanagari", "all", "inside-text"],
)
def test_every_decimal_digit_becomes_its_ascii_digit(text: str, folded: str) -> None:
    assert fold_numerals(text) == folded


@pytest.mark.parametrize(
    ("text", "folded"),
    [
        (TWO + CIRCLE + TWO + FOUR, "2024"),
        (ONE + HUNDRED + TWO + TEN + THREE, "123"),
        (THREE + MONTH, MONTH),
        (TEN_THOUSAND + ONE, "10001"),
        (TEN, "10"),
        (TWO + TEN, "20"),
        (ONE + THOUSAND + TEN_THOUSAND, "10000000"),
        (f"in {TWO}{CIRCLE}{TWO}{FOUR} and 12", "in 2024 and 12"),
        (f"{ONE} {TWO}", " "),
    ],
    ids=[
        "digits-in-order",
        "summed-with-magnitudes",
        "a-lone-digit-is-deleted",
        "ten-thousand-and-one",
        "a-lone-magnitude-is-its-value",
        "twenty",
        "stacked-magnitudes",
        "inside-text",
        "two-lone-digits-are-both-deleted",
    ],
)
def test_a_run_of_cjk_numerals_becomes_the_digits_it_spells(
    text: str, folded: str
) -> None:
    assert fold_numerals(text) == folded


SUPERSCRIPT_TWO = chr(0xB2)
CIRCLED_FIVE = chr(0x2464)
ROMAN_TWELVE = chr(0x216B)
VULGAR_HALF = chr(0xBD)
CIRCLED_ONE = chr(0x2460)


@pytest.mark.parametrize(
    "character",
    [SUPERSCRIPT_TWO, CIRCLED_FIVE, ROMAN_TWELVE, VULGAR_HALF, CIRCLED_ONE],
)
def test_a_character_with_a_digit_value_but_no_decimal_value_is_left_alone(
    character: str,
) -> None:
    # Superscripts, circled digits and fractions are not decimal digits.
    assert unicodedata.decimal(character, None) is None

    assert fold_numerals(f"a{character}b") == f"a{character}b"


@given(st.text(max_size=60))
def test_the_result_holds_no_decimal_digit_but_the_ascii_ones_and_no_cjk_numeral(
    text: str,
) -> None:
    folded = fold_numerals(text)

    for character in folded:
        value = unicodedata.decimal(character, None)
        assert value is None or character == str(value)
        assert character not in CJK_NUMERALS


@given(st.text(max_size=60))
def test_folding_twice_changes_nothing(text: str) -> None:
    once = fold_numerals(text)

    assert fold_numerals(once) == once


def test_the_module_imports_the_standard_library_only() -> None:
    source = Path(rinchen.text.numerals.__file__).read_text(encoding="utf-8")

    imported: set[str] = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            assert node.module is not None
            imported.add(node.module)

    assert imported <= {"__future__", "re", "unicodedata"}
