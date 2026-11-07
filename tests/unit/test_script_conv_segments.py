"""Text segmenter that splits Cyrillic, traditional-script and other runs.

Mongolian and invisible characters are written as `chr()` calls so the exact
code points stay reviewable.
"""

from __future__ import annotations

import pytest

from rinchen.script_conv.alphabet import NNBSP
from rinchen.script_conv.segments import Segment, segment

CYRILLIC = "cyrillic"
MONGOLIAN = "mongolian"
OTHER = "other"

A = chr(0x1820)
NA = chr(0x1828)
FVS_SECOND = chr(0x180B)
MVS = chr(0x180E)
NIRUGU = chr(0x180A)
ZWJ = chr(0x200D)
ZWNJ = chr(0x200C)
CYRILLIC_A = chr(0x0430)  # looks like Latin "a", so it is spelled by code point
COMBINING_ACUTE = chr(0x0301)
CYRILLIC_THOUSANDS_SIGN = chr(0x0482)  # a Cyrillic-block symbol, not a letter
RIGHT_SINGLE_QUOTE = chr(0x2019)
# Mongolian "ongol" (the word MONGOL), the genitive suffix and the word NOM.
MONGOL = "".join(
    chr(code_point) for code_point in (0x182E, 0x1823, 0x1829, 0x182D, 0x1823, 0x182F)
)
SUFFIX_UN = chr(0x1824) + chr(0x1828)
NOM = "".join(chr(code_point) for code_point in (0x1828, 0x1823, 0x182E))


def _kinds(text: str) -> list[str]:
    return [piece.kind for piece in segment(text)]


def _pairs(text: str) -> list[tuple[str, str]]:
    return [(piece.text, piece.kind) for piece in segment(text)]


def test_empty_text_has_no_segments() -> None:
    assert segment("") == ()


def test_segment_is_a_frozen_value() -> None:
    assert Segment("Монгол", CYRILLIC) == Segment("Монгол", CYRILLIC)
    with pytest.raises(AttributeError):
        Segment("Монгол", CYRILLIC).text = "x"  # type: ignore[misc]


def test_cyrillic_words_are_split_by_the_space() -> None:
    assert segment("Монгол улс") == (
        Segment("Монгол", CYRILLIC),
        Segment(" ", OTHER),
        Segment("улс", CYRILLIC),
    )


@pytest.mark.parametrize("word", ["МОНГОЛ", "Монгол", "Үнэн", "өдөр"])
def test_a_cyrillic_word_of_any_case_is_one_segment(word: str) -> None:
    assert segment(word) == (Segment(word, CYRILLIC),)


def test_a_separated_suffix_stays_in_its_mongolian_segment() -> None:
    text = MONGOL + NNBSP + SUFFIX_UN + " " + NOM
    assert segment(text) == (
        Segment(MONGOL + NNBSP + SUFFIX_UN, MONGOLIAN),
        Segment(" ", OTHER),
        Segment(NOM, MONGOLIAN),
    )


def test_a_selector_before_the_nnbsp_stays_in_the_segment() -> None:
    text = MONGOL + FVS_SECOND + NNBSP + SUFFIX_UN
    assert segment(text) == (Segment(text, MONGOLIAN),)


@pytest.mark.parametrize(
    "joiner_text",
    [A + ZWJ + NA, A + ZWNJ + NA],
    ids=["zwj", "zwnj"],
)
def test_a_joiner_between_letters_stays_in_the_mongolian_segment(
    joiner_text: str,
) -> None:
    assert segment(joiner_text) == (Segment(joiner_text, MONGOLIAN),)


def test_a_vowel_separator_and_a_nirugu_stay_in_the_mongolian_segment() -> None:
    text = A + MVS + NA + NIRUGU + A
    assert segment(text) == (Segment(text, MONGOLIAN),)


def test_a_leading_nnbsp_is_other() -> None:
    assert segment(NNBSP + MONGOL) == (
        Segment(NNBSP, OTHER),
        Segment(MONGOL, MONGOLIAN),
    )


def test_a_trailing_nnbsp_is_other() -> None:
    assert segment(MONGOL + NNBSP) == (
        Segment(MONGOL, MONGOLIAN),
        Segment(NNBSP, OTHER),
    )


def test_a_doubled_nnbsp_is_other() -> None:
    assert segment(MONGOL + NNBSP + NNBSP + SUFFIX_UN) == (
        Segment(MONGOL, MONGOLIAN),
        Segment(NNBSP + NNBSP, OTHER),
        Segment(SUFFIX_UN, MONGOLIAN),
    )


@pytest.mark.parametrize(
    "gap",
    [" " + NNBSP, NNBSP + " "],
    ids=["space-then-nnbsp", "nnbsp-then-space"],
)
def test_an_nnbsp_next_to_a_space_is_other(gap: str) -> None:
    assert segment(MONGOL + gap + SUFFIX_UN) == (
        Segment(MONGOL, MONGOLIAN),
        Segment(gap, OTHER),
        Segment(SUFFIX_UN, MONGOLIAN),
    )


def test_an_nnbsp_after_cyrillic_is_other() -> None:
    assert segment("Монгол" + NNBSP + SUFFIX_UN) == (
        Segment("Монгол", CYRILLIC),
        Segment(NNBSP, OTHER),
        Segment(SUFFIX_UN, MONGOLIAN),
    )


def test_an_nnbsp_before_a_non_letter_is_other() -> None:
    assert segment(MONGOL + NNBSP + FVS_SECOND) == (
        Segment(MONGOL, MONGOLIAN),
        Segment(NNBSP, OTHER),
        Segment(FVS_SECOND, MONGOLIAN),
    )


def test_an_nnbsp_before_cyrillic_is_other() -> None:
    assert segment(MONGOL + NNBSP + "ын") == (
        Segment(MONGOL, MONGOLIAN),
        Segment(NNBSP, OTHER),
        Segment("ын", CYRILLIC),
    )


def test_digits_glued_to_cyrillic_are_split() -> None:
    assert segment("2024оны") == (
        Segment("2024", OTHER),
        Segment("оны", CYRILLIC),
    )


def test_a_hyphen_splits_cyrillic_words() -> None:
    assert segment("н-р") == (
        Segment("н", CYRILLIC),
        Segment("-", OTHER),
        Segment("р", CYRILLIC),
    )


def test_latin_glued_to_cyrillic_is_split() -> None:
    assert segment("iPhone-ыг") == (
        Segment("iPhone-", OTHER),
        Segment("ыг", CYRILLIC),
    )


@pytest.mark.parametrize("apostrophe", ["'", RIGHT_SINGLE_QUOTE])
def test_an_apostrophe_splits_cyrillic_words(apostrophe: str) -> None:
    assert segment("д" + apostrophe + "Артаньян") == (
        Segment("д", CYRILLIC),
        Segment(apostrophe, OTHER),
        Segment("Артаньян", CYRILLIC),
    )


def test_a_combining_mark_stays_in_its_cyrillic_segment() -> None:
    word = "н" + CYRILLIC_A + COMBINING_ACUTE + "м"
    assert segment(word) == (Segment(word, CYRILLIC),)


def test_a_combining_mark_without_a_cyrillic_letter_is_other() -> None:
    assert _pairs("x" + COMBINING_ACUTE + "ы") == [
        ("x" + COMBINING_ACUTE, OTHER),
        ("ы", CYRILLIC),
    ]
    assert _pairs(COMBINING_ACUTE + "ы") == [(COMBINING_ACUTE, OTHER), ("ы", CYRILLIC)]
    assert _pairs(A + COMBINING_ACUTE) == [(A, MONGOLIAN), (COMBINING_ACUTE, OTHER)]


def test_a_cyrillic_block_symbol_is_other() -> None:
    assert _pairs("ы" + CYRILLIC_THOUSANDS_SIGN) == [
        ("ы", CYRILLIC),
        (CYRILLIC_THOUSANDS_SIGN, OTHER),
    ]


def test_other_scripts_are_other() -> None:
    assert _pairs("Ωx" + chr(0x4E2D)) == [("Ωx" + chr(0x4E2D), OTHER)]


def test_cyrillic_and_mongolian_without_a_gap_are_two_segments() -> None:
    assert _kinds("Монгол" + MONGOL) == [CYRILLIC, MONGOLIAN]
    assert _kinds(MONGOL + "Монгол") == [MONGOLIAN, CYRILLIC]


def test_a_mixed_line_is_classified_run_by_run() -> None:
    text = "Монгол: " + MONGOL + NNBSP + SUFFIX_UN + ", 2024"
    assert _pairs(text) == [
        ("Монгол", CYRILLIC),
        (": ", OTHER),
        (MONGOL + NNBSP + SUFFIX_UN, MONGOLIAN),
        (", 2024", OTHER),
    ]
