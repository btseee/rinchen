"""The `script_ratio` stage: the share of each side's letters that belong to
the scripts of its language, and the `script_src` / `script_tgt` verdicts."""

from __future__ import annotations

from collections import Counter

import polars as pl
import pytest

from rinchen.config.filters import ScriptRatioConfig
from rinchen.corpus.schema import CORPUS_SCHEMA, corpus_frame
from rinchen.errors import DataError
from rinchen.filter import script_ratio
from rinchen.filter.reasons import with_scores
from rinchen.filter.script_ratio import script_share
from rinchen.lang.scripts import Script
from tests.support.planted_corpus import PlantedCorpus, build_planted_corpus

CONFIG = ScriptRatioConfig()
HAN = chr(0x65E5) + chr(0x672C)
HIRAGANA = chr(0x3042) + chr(0x3044)
KATAKANA = chr(0x30A2) + chr(0x30A4)


def _lazy(*rows: tuple[str, str], src_lang: str = "khk_Cyrl") -> pl.LazyFrame:
    return corpus_frame(
        [
            {
                "src_lang": src_lang,
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


def _score(result: pl.DataFrame, field: str) -> list[float | None]:
    return result["scores"].struct.field(field).to_list()


@pytest.fixture(scope="module")
def planted() -> PlantedCorpus:
    return build_planted_corpus()


# --- script_share -------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "scripts", "expected"),
    [
        ("Монгол", {Script.CYRILLIC}, 1.0),
        ("Монгол hello", {Script.CYRILLIC}, 6 / 11),
        ("Монгол hello", {Script.LATIN}, 5 / 11),
        ("Монгол hello", {Script.CYRILLIC, Script.LATIN}, 1.0),
        ("Монгол hello", [Script.CYRILLIC, Script.CYRILLIC], 6 / 11),
        ("hello", {Script.CYRILLIC}, 0.0),
        ("Монгол αβγ", {Script.CYRILLIC}, 6 / 9),
        ("αβγ", {Script.CYRILLIC}, 0.0),
        (
            HAN + HIRAGANA + KATAKANA,
            {Script.HAN, Script.HIRAGANA, Script.KATAKANA},
            1.0,
        ),
        (HAN + HIRAGANA + KATAKANA, {Script.HAN}, 1 / 3),
    ],
)
def test_script_share_is_the_share_of_letters_in_the_given_scripts(
    text: str, scripts: set[Script] | list[Script], expected: float
) -> None:
    assert script_share(text, scripts) == pytest.approx(expected)


@pytest.mark.parametrize("text", ["12 !?", "", "   ", "1,5 - 2024"])
def test_text_without_letters_has_a_vacuous_share_of_one(text: str) -> None:
    assert script_share(text, {Script.CYRILLIC}) == 1.0


# --- apply: scores and reasons --------------------------------------------------


def test_scores_and_reasons_on_a_four_row_frame() -> None:
    frame = _lazy(
        ("Монгол улс", "Mongolia"),
        ("hello world", "hello"),
        ("Монгол", "Монгол"),
        ("hello", "Монгол"),
    )

    result = script_ratio.apply(frame, CONFIG).collect()

    assert _score(result, "script_src") == [1.0, 0.0, 1.0, 0.0]
    assert _score(result, "script_tgt") == [1.0, 1.0, 0.0, 0.0]
    # A row failing on both sides gets the source reason.
    assert result["filter_reason"].to_list() == [
        None,
        "script_src",
        "script_tgt",
        "script_src",
    ]
    assert result.schema == CORPUS_SCHEMA


def test_a_share_exactly_at_min_ratio_passes() -> None:
    brand_in_cyrillic = "ж" * 18 + "IT"  # 18 of 20 letters: 0.9
    frame = _lazy((brand_in_cyrillic, "text"))

    default = script_ratio.apply(frame, CONFIG).collect()
    stricter = script_ratio.apply(frame, ScriptRatioConfig(min_ratio=0.95)).collect()

    assert _score(default, "script_src") == pytest.approx([0.9])
    assert default["filter_reason"].to_list() == [None]
    assert stricter["filter_reason"].to_list() == ["script_src"]


def test_japanese_accepts_han_hiragana_and_katakana() -> None:
    frame = _lazy((HAN + HIRAGANA + KATAKANA, "text"), src_lang="jpn_Jpan")

    result = script_ratio.apply(frame, CONFIG).collect()

    assert _score(result, "script_src") == [1.0]
    assert result["filter_reason"].to_list() == [None]


def test_the_katakana_long_vowel_mark_does_not_lower_the_japanese_share() -> None:
    long_vowel = chr(0x30FC)
    coffee_please = (  # katakana, long-vowel marks, then hiragana
        chr(0x30B3)
        + long_vowel
        + chr(0x30D2)
        + long_vowel
        + chr(0x3092)
        + chr(0x304F)
        + chr(0x3060)
        + chr(0x3055)
        + chr(0x3044)
    )
    frame = _lazy((coffee_please, "coffee, please"), src_lang="jpn_Jpan")

    result = script_ratio.apply(frame, CONFIG).collect()

    assert _score(result, "script_src") == [1.0]
    assert result["filter_reason"].to_list() == [None]


def test_empty_and_whitespace_sides_are_vacuously_in_script() -> None:
    frame = _lazy(("", "   "), ("12", "1,5"))

    result = script_ratio.apply(frame, CONFIG).collect()

    assert _score(result, "script_src") == [1.0, 1.0]
    assert _score(result, "script_tgt") == [1.0, 1.0]
    assert result["filter_reason"].to_list() == [None, None]


def test_an_earlier_reason_is_kept() -> None:
    frame = _lazy(("hello", "text")).with_columns(
        pl.lit("lid_src").alias("filter_reason")
    )

    result = script_ratio.apply(frame, CONFIG).collect()

    assert result["filter_reason"].to_list() == ["lid_src"]
    assert _score(result, "script_src") == [0.0]


def test_scores_already_in_the_struct_survive() -> None:
    frame = with_scores(_lazy(("Монгол", "text")), labse=pl.lit(0.25))

    result = script_ratio.apply(frame, CONFIG).collect()

    assert _score(result, "labse") == [0.25]
    assert _score(result, "script_src") == [1.0]


def test_null_text_is_read_as_empty_and_in_script() -> None:
    frame = _lazy(("x", "y")).with_columns(
        pl.lit(None, dtype=pl.String).alias("src"),
        pl.lit(None, dtype=pl.String).alias("tgt"),
    )

    result = script_ratio.apply(frame, CONFIG).collect()

    assert _score(result, "script_src") == [1.0]
    assert _score(result, "script_tgt") == [1.0]
    assert result["filter_reason"].to_list() == [None]


@pytest.mark.parametrize("side", ["src_lang", "tgt_lang"])
def test_a_null_language_code_raises_data_error(side: str) -> None:
    frame = _lazy(("Монгол", "text")).with_columns(
        pl.lit(None, dtype=pl.String).alias(side)
    )

    with pytest.raises(DataError, match="None"):
        script_ratio.apply(frame, CONFIG).collect()


@pytest.mark.parametrize("side", ["src_lang", "tgt_lang"])
def test_an_unknown_language_code_raises_data_error_naming_it(side: str) -> None:
    frame = _lazy(("Монгол", "text")).with_columns(pl.lit("xx_Latn").alias(side))

    with pytest.raises(DataError, match="xx_Latn"):
        script_ratio.apply(frame, CONFIG).collect()


def test_zero_rows_keep_the_corpus_schema() -> None:
    result = script_ratio.apply(_lazy(), CONFIG).collect()

    assert result.height == 0
    assert result.schema == CORPUS_SCHEMA


# --- apply: the planted corpus --------------------------------------------------


def test_planted_corpus_flags_exactly_the_script_rows(planted: PlantedCorpus) -> None:
    result = script_ratio.apply(planted.frame.lazy(), CONFIG).collect()

    flagged = {
        row["id"]: row["filter_reason"]
        for row in result.iter_rows(named=True)
        if row["filter_reason"] is not None
    }

    assert set(flagged) == planted.isolated_failures["script_ratio"]
    assert flagged == {row_id: planted.expected_reason[row_id] for row_id in flagged}
    assert Counter(flagged.values()) == {"script_src": 2, "script_tgt": 2}
    assert result.schema == CORPUS_SCHEMA
