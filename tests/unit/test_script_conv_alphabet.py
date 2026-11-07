"""Code point sets of the Cyrillic <-> traditional-script converter.

Mongolian and invisible characters are written as `chr()` calls so the exact
code points stay reviewable.
"""

from __future__ import annotations

import pytest

from rinchen.script_conv.alphabet import (
    CYRILLIC_LANGUAGE,
    CYRILLIC_TO_SCRIPT_PUNCTUATION,
    FVS,
    JOINERS,
    LETTERS,
    MVS,
    NIRUGU,
    NNBSP,
    SCRIPT_LANGUAGE,
    SCRIPT_TO_CYRILLIC_PUNCTUATION,
)
from rinchen.text.normalize import normalize

LETTER_A = chr(0x1820)
LAST_KHALKHA_LETTER = chr(0x1842)
FIRST_TODO_LETTER = chr(0x1843)


def test_the_two_language_codes_are_the_normalizer_languages_of_the_converter() -> None:
    assert CYRILLIC_LANGUAGE == "khk_Cyrl"
    assert SCRIPT_LANGUAGE == "mn_Mong"
    # `normalize` raises for a language it does not know.
    assert normalize("ab", CYRILLIC_LANGUAGE) == "ab"
    assert normalize(LETTER_A, SCRIPT_LANGUAGE) == LETTER_A


def test_letters_cover_u1820_to_u1842() -> None:
    assert len(LETTERS) == 35
    assert LETTERS == {chr(code_point) for code_point in range(0x1820, 0x1843)}
    assert LETTER_A in LETTERS
    assert LAST_KHALKHA_LETTER in LETTERS


def test_letters_exclude_todo_sibe_and_manchu_letters() -> None:
    assert FIRST_TODO_LETTER not in LETTERS


def test_fvs_are_the_four_free_variation_selectors() -> None:
    assert FVS == {chr(0x180B), chr(0x180C), chr(0x180D), chr(0x180F)}
    assert len(FVS) == 4


def test_single_code_point_constants() -> None:
    assert MVS == chr(0x180E)
    assert NNBSP == chr(0x202F)
    assert NIRUGU == chr(0x180A)
    assert JOINERS == {chr(0x200C), chr(0x200D)}


def test_the_sets_are_disjoint() -> None:
    groups = [LETTERS, FVS, JOINERS, {MVS}, {NNBSP}, {NIRUGU}]
    members = [char for group in groups for char in group]
    assert len(members) == len(set(members))


def test_cyrillic_punctuation_maps_to_script_marks() -> None:
    assert CYRILLIC_TO_SCRIPT_PUNCTUATION["."] == chr(0x1803)
    assert CYRILLIC_TO_SCRIPT_PUNCTUATION[","] == chr(0x1802)
    assert set(CYRILLIC_TO_SCRIPT_PUNCTUATION) == {".", ","}


def test_script_punctuation_is_the_inverse() -> None:
    assert SCRIPT_TO_CYRILLIC_PUNCTUATION[chr(0x1803)] == "."
    assert SCRIPT_TO_CYRILLIC_PUNCTUATION[chr(0x1802)] == ","
    for cyrillic, script in CYRILLIC_TO_SCRIPT_PUNCTUATION.items():
        assert SCRIPT_TO_CYRILLIC_PUNCTUATION[script] == cyrillic
    assert len(SCRIPT_TO_CYRILLIC_PUNCTUATION) == len(CYRILLIC_TO_SCRIPT_PUNCTUATION)


def test_punctuation_tables_are_read_only() -> None:
    with pytest.raises(TypeError):
        CYRILLIC_TO_SCRIPT_PUNCTUATION["!"] = chr(0x1803)  # type: ignore[index]
    with pytest.raises(TypeError):
        SCRIPT_TO_CYRILLIC_PUNCTUATION[chr(0x1803)] = "!"  # type: ignore[index]
