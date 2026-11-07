"""Validator that keeps forbidden controls out of traditional-script output.

Mongolian and invisible characters are written as `chr()` calls so the exact
code points stay reviewable.
"""

from __future__ import annotations

import hashlib
import json
import re
from importlib import resources
from pathlib import Path

import pytest

from rinchen.script_conv.alphabet import FVS, LETTERS, MVS, NNBSP
from rinchen.script_conv.validate import (
    VARIANT_TABLE,
    VIOLATION_CODES,
    Violation,
    validate_traditional,
)

A = chr(0x1820)
NA = chr(0x1828)
FVS_SECOND = chr(0x180B)
FVS_THIRD = chr(0x180C)
FVS_FIFTH = chr(0x180F)
ZWJ = chr(0x200D)
ZWNJ = chr(0x200C)
CYRILLIC_A = chr(0x0430)
# Mongolian "ongol" (the word MONGOL) and the genitive suffix after an NNBSP.
MONGOL = "".join(
    chr(code_point) for code_point in (0x182E, 0x1823, 0x1829, 0x182D, 0x1823, 0x182F)
)
SUFFIX_UN = chr(0x1824) + chr(0x1828)

UCD_URL = "https://www.unicode.org/Public/18.0.0/ucd/StandardizedVariants.txt"
REPO_ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = REPO_ROOT / "manifests" / "ucd_standardized_variants.json"
SHA256_HEX = re.compile(r"[0-9a-f]{64}")


def _located(text: str) -> list[tuple[int, str]]:
    return [
        (violation.index, violation.code) for violation in validate_traditional(text)
    ]


def _table_text() -> str:
    return (
        resources.files("rinchen.script_conv")
        .joinpath("standardized_variants.tsv")
        .read_text("utf-8")
    )


def _header() -> dict[str, str]:
    header: dict[str, str] = {}
    for line in _table_text().splitlines():
        if not line.startswith("#"):
            break
        key, separator, value = line[1:].partition(":")
        if separator:
            header[key.strip()] = value.strip()
    return header


def test_violation_codes_are_the_five_documented_ones() -> None:
    assert VIOLATION_CODES == {
        "joiner",
        "foreign_character",
        "orphan_selector",
        "misplaced_nnbsp",
        "invalid_variant",
    }


def test_empty_string_has_no_violations() -> None:
    assert validate_traditional("") == ()


def test_violation_carries_index_code_and_detail() -> None:
    violation = validate_traditional(A + ZWJ + A)[0]
    assert isinstance(violation, Violation)
    assert (violation.index, violation.code) == (1, "joiner")
    assert violation.detail


def test_violation_is_frozen() -> None:
    violation = Violation(index=0, code="joiner", detail="x")
    with pytest.raises(AttributeError):
        violation.index = 1  # type: ignore[misc]


def test_every_letter_alone_is_valid() -> None:
    for letter in LETTERS:
        assert validate_traditional(letter) == ()


# --- joiner -----------------------------------------------------------------


def test_zwj_is_a_joiner_violation() -> None:
    assert _located(A + ZWJ + A) == [(1, "joiner")]


def test_zwnj_is_a_joiner_violation() -> None:
    assert _located(ZWNJ + A) == [(0, "joiner")]


def test_text_without_joiners_has_no_joiner_violation() -> None:
    assert _located(A + NA) == []


# --- foreign_character ------------------------------------------------------


def test_latin_run_digits_and_ascii_punctuation_are_allowed() -> None:
    assert validate_traditional(A + "X9") == ()
    assert validate_traditional(A + " iPhone-15, ok (2024)!?") == ()


def test_script_marks_for_comma_and_full_stop_are_allowed() -> None:
    assert validate_traditional(A + chr(0x1802) + " " + A + chr(0x1803)) == ()


def test_nirugu_is_allowed() -> None:
    assert validate_traditional(A + chr(0x180A) + A) == ()


def test_cyrillic_letter_is_foreign() -> None:
    assert _located(A + CYRILLIC_A) == [(1, "foreign_character")]


@pytest.mark.parametrize(
    "foreign",
    [
        chr(0x1843),  # first Todo letter, outside U+1820..U+1842
        chr(0x1810),  # Mongolian digit zero; digits are ASCII only
        chr(0x1800),  # Mongolian birga
        chr(0x4E2D),  # a Han ideograph
        "\n",
        "\t",
        chr(0x00A0),  # NBSP is not NNBSP
    ],
)
def test_other_characters_are_foreign(foreign: str) -> None:
    assert _located(A + foreign) == [(1, "foreign_character")]


def test_foreign_detail_names_the_code_point() -> None:
    (violation,) = validate_traditional(CYRILLIC_A)
    assert "U+0430" in violation.detail


# --- orphan_selector --------------------------------------------------------


def test_fvs_at_start_is_orphan() -> None:
    assert _located(FVS_SECOND + A) == [(0, "orphan_selector")]


def test_fvs_after_space_or_after_another_selector_is_orphan() -> None:
    assert _located(A + " " + FVS_SECOND) == [(2, "orphan_selector")]
    assert _located(A + FVS_SECOND + FVS_SECOND) == [(2, "orphan_selector")]


def test_mvs_not_followed_by_a_letter_is_orphan() -> None:
    assert _located(A + MVS + " ") == [(1, "orphan_selector")]
    assert _located(A + MVS) == [(1, "orphan_selector")]


def test_mvs_not_after_a_letter_is_orphan() -> None:
    assert _located(MVS + A) == [(0, "orphan_selector")]
    assert _located(" " + MVS + A) == [(1, "orphan_selector")]


def test_mvs_between_two_letters_is_valid() -> None:
    assert validate_traditional(A + MVS + A) == ()


def test_fvs_after_a_letter_is_not_orphan() -> None:
    assert _located(A + FVS_SECOND) == []


# --- misplaced_nnbsp --------------------------------------------------------


def test_nnbsp_before_a_space_is_misplaced() -> None:
    assert _located(A + NNBSP + " ") == [(1, "misplaced_nnbsp")]


def test_nnbsp_at_start_is_misplaced() -> None:
    assert _located(NNBSP + A) == [(0, "misplaced_nnbsp")]


def test_nnbsp_at_end_is_misplaced() -> None:
    assert _located(A + NNBSP) == [(1, "misplaced_nnbsp")]


def test_nnbsp_next_to_a_space_or_digit_is_misplaced() -> None:
    assert _located(" " + NNBSP + A) == [(1, "misplaced_nnbsp")]
    assert _located("9" + NNBSP + A) == [(1, "misplaced_nnbsp")]


def test_nnbsp_between_two_letters_is_valid() -> None:
    assert validate_traditional(MONGOL + NNBSP + SUFFIX_UN) == ()


def test_nnbsp_after_a_stem_final_variant_is_valid() -> None:
    # The word-final letter may carry its own selector before the suffix gap.
    assert validate_traditional(A + FVS_SECOND + NNBSP + A) == ()


# --- invalid_variant --------------------------------------------------------


def test_unlisted_letter_selector_pair_is_invalid() -> None:
    assert chr(0x180F) not in VARIANT_TABLE[A]
    assert _located(A + FVS_FIFTH) == [(1, "invalid_variant")]


def test_invalid_variant_detail_names_both_code_points() -> None:
    (violation,) = validate_traditional(A + FVS_FIFTH)
    assert "U+1820" in violation.detail
    assert "U+180F" in violation.detail


def test_listed_letter_selector_pair_is_valid() -> None:
    assert validate_traditional(A + FVS_SECOND) == ()


def test_every_table_pair_is_valid_and_every_other_pair_is_invalid() -> None:
    for letter in LETTERS:
        for selector in FVS:
            expected = (
                []
                if selector in VARIANT_TABLE.get(letter, frozenset())
                else [(1, "invalid_variant")]
            )
            assert _located(letter + selector) == expected, (
                hex(ord(letter)),
                hex(ord(selector)),
            )


# --- several violations -----------------------------------------------------


def test_violations_come_in_text_order() -> None:
    text = ZWJ + A + CYRILLIC_A + FVS_FIFTH + " " + NNBSP
    assert _located(text) == [
        (0, "joiner"),
        (2, "foreign_character"),
        (3, "orphan_selector"),
        (5, "misplaced_nnbsp"),
    ]


def test_every_reported_code_is_a_documented_code() -> None:
    text = ZWJ + CYRILLIC_A + A + FVS_FIFTH + FVS_SECOND + FVS_SECOND + MVS + NNBSP
    assert {
        violation.code for violation in validate_traditional(text)
    } <= VIOLATION_CODES


# --- variant table ----------------------------------------------------------


def test_table_keys_are_letters_and_values_are_selectors() -> None:
    assert VARIANT_TABLE
    for base, selectors in VARIANT_TABLE.items():
        assert base in LETTERS
        assert selectors
        assert selectors <= FVS


def test_table_matches_ucd_lines_for_letter_a_and_qa() -> None:
    # StandardizedVariants.txt 18.0.0: "1820 180B/180C/180D" and "182C 180B..180F".
    assert VARIANT_TABLE[A] == {chr(0x180B), chr(0x180C), chr(0x180D)}
    assert chr(0x180F) in VARIANT_TABLE[chr(0x182C)]


def test_table_holds_no_todo_sibe_or_manchu_base() -> None:
    assert all(base in LETTERS for base in VARIANT_TABLE)
    assert chr(0x1844) not in VARIANT_TABLE


def test_table_is_read_only() -> None:
    with pytest.raises(TypeError):
        VARIANT_TABLE[A] = frozenset()  # type: ignore[index]


def test_table_rows_are_tab_separated_hex_pairs() -> None:
    rows = [
        line for line in _table_text().splitlines() if line and not line.startswith("#")
    ]
    assert rows
    assert all(re.fullmatch(r"[0-9A-F]{4}\t[0-9A-F]{4}", row) for row in rows)
    assert len(rows) == len(set(rows))


def test_table_header_records_source_version_and_sha256() -> None:
    header = _header()
    assert header["source_url"] == UCD_URL
    assert header["ucd_version"] == "18.0.0"
    assert SHA256_HEX.fullmatch(header["source_sha256"])
    assert re.fullmatch(r"\d{4}-\d{2}-\d{2}", header["retrieved"])


def test_manifest_carries_the_same_hashes_as_the_table() -> None:
    manifest = json.loads(MANIFEST_PATH.read_text("utf-8"))
    header = _header()
    assert manifest["source_url"] == header["source_url"] == UCD_URL
    assert manifest["ucd_version"] == header["ucd_version"] == "18.0.0"
    assert manifest["source_sha256"] == header["source_sha256"]
    assert manifest["retrieved"] == header["retrieved"]
    table_bytes = _table_text().encode("utf-8")
    assert manifest["derived_sha256"] == hashlib.sha256(table_bytes).hexdigest()
