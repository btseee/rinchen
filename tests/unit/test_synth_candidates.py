"""`clean_candidates`, `has_alphanumeric` and `is_utf8_encodable`: what every
generator backend does to the candidates it returns before the selection stage
sees them."""

from __future__ import annotations

import pytest

from rinchen.synth._candidates import (
    clean_candidates,
    has_alphanumeric,
    is_utf8_encodable,
)

NNBSP = chr(0x202F)
NON_BMP = chr(0x1F600)
ZERO_WIDTH_SPACE = chr(0x200B)
BYTE_ORDER_MARK = chr(0xFEFF)
HIGH_SURROGATE = chr(0xD83D)
LOW_SURROGATE = chr(0xDE00)


def test_candidates_are_stripped_in_first_seen_order_without_duplicates() -> None:
    result = clean_candidates(["  one  ", "two", "one", "\ntwo\n", "three", "t  w"])

    assert result == ["one", "two", "three", "t  w"]


def test_a_text_with_no_candidates_gives_an_empty_list() -> None:
    assert clean_candidates([]) == []


@pytest.mark.parametrize(
    "blank",
    [
        "",
        " \n\t",
        f"{NNBSP}",
        "...",
        "-- !?",
        ZERO_WIDTH_SPACE,
        BYTE_ORDER_MARK,
        f"{ZERO_WIDTH_SPACE} ,",
    ],
    ids=[
        "empty",
        "whitespace",
        "NNBSP",
        "dots",
        "punctuation",
        "zero-width space",
        "byte order mark",
        "invisible and punctuation",
    ],
)
def test_a_candidate_without_a_letter_or_digit_is_dropped(blank: str) -> None:
    assert clean_candidates([blank, "kept"]) == ["kept"]


def test_hard_but_encodable_text_is_kept() -> None:
    text = f"Өдрийн{NNBSP}мэнд {NON_BMP}\nsecond line"

    assert clean_candidates([text]) == [text]


@pytest.mark.parametrize(
    "unencodable",
    [
        pytest.param(f"bad {HIGH_SURROGATE}", id="lone high surrogate"),
        pytest.param(f"{LOW_SURROGATE} bad", id="lone low surrogate"),
        pytest.param(HIGH_SURROGATE, id="only a surrogate"),
        pytest.param(f"a{HIGH_SURROGATE}{LOW_SURROGATE}b", id="split surrogate pair"),
        pytest.param(f"text{chr(0xDFFF)}", id="last surrogate code point"),
    ],
)
def test_a_candidate_utf8_cannot_encode_is_dropped_like_an_empty_one(
    unencodable: str,
) -> None:
    assert clean_candidates(["first", unencodable, "last"]) == ["first", "last"]


def test_a_text_whose_candidates_are_all_unencodable_gets_an_empty_list() -> None:
    assert clean_candidates([f"x{HIGH_SURROGATE}", HIGH_SURROGATE]) == []


@pytest.mark.parametrize(
    ("text", "encodable"),
    [
        ("", True),
        ("plain ASCII", True),
        (f"Өдөр{NNBSP}{NON_BMP}", True),
        (chr(0), True),
        (chr(0xD7FF), True),
        (chr(0xE000), True),
        (chr(0xD800), False),
        (chr(0xDBFF), False),
        (chr(0xDC00), False),
        (chr(0xDFFF), False),
        (f"fine {chr(0xD800)}", False),
    ],
)
def test_is_utf8_encodable_is_false_only_for_surrogate_code_points(
    text: str, encodable: bool
) -> None:
    assert is_utf8_encodable(text) is encodable


@pytest.mark.parametrize(
    "text",
    ["", " \n\t", "...", ZERO_WIDTH_SPACE, BYTE_ORDER_MARK, f"{NNBSP}-!"],
)
def test_text_without_a_letter_or_digit_has_no_alphanumeric_character(
    text: str,
) -> None:
    assert has_alphanumeric(text) is False


@pytest.mark.parametrize("text", ["a", "7", "Өдөр", "  x  ", "-- 3 --", "Ж"])
def test_a_letter_or_a_digit_is_an_alphanumeric_character(text: str) -> None:
    assert has_alphanumeric(text) is True
