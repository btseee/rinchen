"""The `numbers` stage: both sides must carry the same multiset of digits once
every kind of digit is read as an ASCII digit."""

from __future__ import annotations

from collections import Counter

import polars as pl
import pytest

from rinchen.corpus.schema import CORPUS_SCHEMA, corpus_frame
from rinchen.filter import numbers
from rinchen.filter.numbers import digit_multiset
from tests.support.planted_corpus import PlantedCorpus, build_planted_corpus

# CJK numerals, built with chr so the source holds no look-alike characters.
CIRCLE = chr(0x3007)
ZERO = chr(0x96F6)
ONE = chr(0x4E00)
TWO = chr(0x4E8C)
THREE = chr(0x4E09)
FOUR = chr(0x56DB)
FIVE = chr(0x4E94)
TEN = chr(0x5341)
HUNDRED = chr(0x767E)
THOUSAND = chr(0x5343)
TEN_THOUSAND = chr(0x4E07)

MONGOLIAN_ONE_TWO = chr(0x1811) + chr(0x1812)
FULLWIDTH_ONE = chr(0xFF11)
ARABIC_INDIC_ONE = chr(0x0661)


def _lazy(*rows: tuple[str, str]) -> pl.LazyFrame:
    return corpus_frame(
        [
            {
                "src_lang": "khk_Cyrl",
                "tgt_lang": "eng_Latn",
                "src": src,
                "tgt": tgt,
                "origin": "human",
                "corpus": "unit",
                "split": "train",
            }
            for src, tgt in rows
        ]
    ).lazy()


@pytest.fixture(scope="module")
def planted() -> PlantedCorpus:
    return build_planted_corpus()


# --- digit_multiset ---------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "digits"),
    [
        (TWO + CIRCLE + TWO + FOUR, "2024"),
        (TWO + ZERO + TWO + FOUR, "2024"),
        (TEN + TWO, "12"),
        (TWO + TEN, "20"),
        (TWO + TEN + FIVE, "25"),
        (ONE + HUNDRED + TWO + TEN + THREE, "123"),
        (ONE + HUNDRED + CIRCLE + FIVE, "105"),
        (THREE + THOUSAND, "3000"),
        (ONE + TEN_THOUSAND + TWO + THOUSAND, "12000"),
        (TEN, "10"),
        (HUNDRED, "100"),
        (TEN_THOUSAND, "10000"),
        (TWO + TEN_THOUSAND, "20000"),
        (ONE + THOUSAND + TEN_THOUSAND, "10000000"),
        (ONE, ""),
        (FIVE, ""),
        (CIRCLE, ""),
    ],
)
def test_cjk_numeral_runs_count_as_the_digits_of_their_value(
    text: str, digits: str
) -> None:
    assert digit_multiset(text) == Counter(digits)


def test_a_cjk_run_ends_at_the_first_character_that_is_not_a_numeral() -> None:
    text = f"({TEN}{TWO}) {FIVE} 7"

    assert digit_multiset(text) == Counter("127")


@pytest.mark.parametrize(
    ("text", "digits"),
    [
        ("", ""),
        ("no digits at all.", ""),
        ("1,5", "15"),
        ("1.5", "15"),
        ("12.03.2024", "12032024"),
        ("12 000", "12000"),
        ("11", "11"),
        (MONGOLIAN_ONE_TWO, "12"),
        (FULLWIDTH_ONE, "1"),
        (ARABIC_INDIC_ONE, "1"),
        (chr(0x00B2), ""),
    ],
)
def test_every_unicode_decimal_digit_counts_as_its_ascii_digit(
    text: str, digits: str
) -> None:
    assert digit_multiset(text) == Counter(digits)


@pytest.mark.parametrize(
    ("src", "tgt"),
    [
        ("1,5", "1.5"),
        ("12.03.2024", "2024-03-12"),
        (MONGOLIAN_ONE_TWO, "12"),
        (FULLWIDTH_ONE, "1"),
        (ARABIC_INDIC_ONE, "1"),
        ("", ""),
        (TWO + CIRCLE + TWO + FOUR, "2024"),
    ],
)
def test_equal_multisets_of_digits_match(src: str, tgt: str) -> None:
    assert digit_multiset(src) == digit_multiset(tgt)


@pytest.mark.parametrize(("src", "tgt"), [("12", "13"), ("11", "1"), ("5", "")])
def test_different_multisets_of_digits_differ(src: str, tgt: str) -> None:
    assert digit_multiset(src) != digit_multiset(tgt)


# --- apply ---------------------------------------------------------------------------


def test_scores_and_reasons_on_a_small_frame() -> None:
    frame = _lazy(
        ("цена 1,5", "price 1.5"),
        ("2024 он", "year 2025"),
        ("", ""),
        (MONGOLIAN_ONE_TWO, "twelve 12"),
        ("12", "13"),
    )

    result = numbers.apply(frame).collect()

    assert result["scores"].struct.field("num_ok").to_list() == [
        True,
        False,
        True,
        True,
        False,
    ]
    assert result["filter_reason"].to_list() == [None, "numbers", None, None, "numbers"]
    assert result.schema == CORPUS_SCHEMA


@pytest.mark.parametrize(
    ("src", "tgt", "num_ok"),
    [(None, None, True), (None, "no digits", True), (None, "12", False)],
)
def test_null_text_is_read_as_empty(
    src: str | None, tgt: str | None, num_ok: bool
) -> None:
    frame = _lazy(("x", "y")).with_columns(
        pl.lit(src, dtype=pl.String).alias("src"),
        pl.lit(tgt, dtype=pl.String).alias("tgt"),
    )

    result = numbers.apply(frame).collect()

    assert result["scores"].struct.field("num_ok").to_list() == [num_ok]
    assert result["filter_reason"].to_list() == [None if num_ok else "numbers"]


def test_an_earlier_reason_is_kept() -> None:
    frame = _lazy(("1", "2")).with_columns(pl.lit("script_src").alias("filter_reason"))

    result = numbers.apply(frame).collect()

    assert result["filter_reason"].to_list() == ["script_src"]
    assert result["scores"].struct.field("num_ok").to_list() == [False]


def test_zero_rows_keep_the_corpus_schema() -> None:
    result = numbers.apply(_lazy()).collect()

    assert result.height == 0
    assert result.schema == CORPUS_SCHEMA


def test_planted_corpus_flags_exactly_the_number_rows(planted: PlantedCorpus) -> None:
    result = numbers.apply(planted.frame.lazy()).collect()

    flagged = {
        row["id"]: row["filter_reason"]
        for row in result.iter_rows(named=True)
        if row["filter_reason"] is not None
    }

    assert set(flagged) == planted.isolated_failures["numbers"]
    assert set(flagged.values()) == {"numbers"}
    assert len(flagged) == 3
    assert result.schema == CORPUS_SCHEMA
