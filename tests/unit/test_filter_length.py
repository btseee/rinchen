"""The `length` stage: the weighted character ratio and, when a SentencePiece
model is configured, the token-count ratio of the two sides."""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest
import sentencepiece

from rinchen.config.filters import Bounds, DirectionBounds, LengthConfig
from rinchen.corpus.schema import CORPUS_SCHEMA, corpus_frame
from rinchen.errors import ConfigError
from rinchen.filter import length
from rinchen.filter.length import effective_length
from rinchen.lang.scripts import Script
from tests.support.planted_corpus import (
    PlantedCorpus,
    build_planted_corpus,
    tiny_spm_model,
)

DEFAULT_CONFIG = LengthConfig()
WEIGHTS = DEFAULT_CONFIG.char_weights
HAN = chr(0x65E5) + chr(0x672C)
HANGUL = chr(0xD55C) + chr(0xAE00)
HIRAGANA = chr(0x3042) + chr(0x3044)
KATAKANA = chr(0x30A2) + chr(0x30A4)
NO_BREAK_SPACE = chr(0x00A0)
SOH = chr(1)  # removed by the nmt_nfkc normalisation, so it encodes to no token
STX = chr(2)

DIRECTION = "khk_Cyrl->eng_Latn"
CHAR_REASONS = {"len_ratio_char"}


def _row(
    src: str, tgt: str, *, src_lang: str = "khk_Cyrl", tgt_lang: str = "eng_Latn"
) -> dict[str, str]:
    return {
        "src_lang": src_lang,
        "tgt_lang": tgt_lang,
        "src": src,
        "tgt": tgt,
        "origin": "human",
        "corpus": "unit",
        "split": "train",
    }


def _lazy(*pairs: tuple[str, str]) -> pl.LazyFrame:
    return corpus_frame([_row(src, tgt) for src, tgt in pairs]).lazy()


def _reasons(result: pl.DataFrame) -> dict[str, str]:
    return {
        row["id"]: row["filter_reason"]
        for row in result.iter_rows(named=True)
        if row["filter_reason"] is not None
    }


@pytest.fixture(scope="module")
def planted() -> PlantedCorpus:
    return build_planted_corpus()


@pytest.fixture(scope="module")
def tiny_model(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return tiny_spm_model(tmp_path_factory.mktemp("length_spm"))


@pytest.fixture(scope="module")
def normalising_char_model(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A character-level model whose normaliser drops control characters."""
    directory = tmp_path_factory.mktemp("length_char_spm")
    corpus = directory / "corpus.txt"
    corpus.write_text("abc abc bca\ncab acb bac\n" * 20, encoding="utf-8")
    sentencepiece.SentencePieceTrainer.train(
        input=str(corpus),
        model_prefix=str(directory / "char"),
        vocab_size=12,
        model_type="char",
        normalization_rule_name="nmt_nfkc",
        minloglevel=2,
    )
    return directory / "char.model"


# --- effective_length --------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("abc d", 4.0),
        ("", 0.0),
        (" \t\n" + NO_BREAK_SPACE, 0.0),
        (HAN, 5.0),
        (HANGUL, 4.0),
        (HIRAGANA, 3.0),
        (KATAKANA, 3.0),
        ("Монгол", 6.0),
        ("αβγ", 3.0),
        ("12 !?", 4.0),
        (HAN + "a1,", 8.0),
    ],
)
def test_effective_length_weighs_each_non_space_character_by_its_script(
    text: str, expected: float
) -> None:
    assert effective_length(text, WEIGHTS) == expected


def test_effective_length_uses_the_given_weights_only() -> None:
    assert effective_length("ab" + HAN, {Script.LATIN: 3.0}) == 8.0
    assert effective_length("ab" + HAN, {}) == 4.0


# --- apply: the character ratio ----------------------------------------------------


@pytest.mark.parametrize(
    ("src", "tgt", "ratio", "reason"),
    [
        ("abcd efgh", "wxyz stuv", 1.0, None),
        ("ab", "abcde", 2.5, "len_ratio_char"),
        ("abcdef", "ab", 1 / 3, "len_ratio_char"),
        ("ab", "abcd", 2.0, None),
        ("abcd", "ab", 0.5, None),
        ("  ", "abc", None, "len_ratio_char"),
        ("abc", "", None, "len_ratio_char"),
        ("", " ", None, "len_ratio_char"),
        (HAN, "abcde", 1.0, None),
    ],
)
def test_char_ratio_is_scored_and_judged_against_the_default_bounds(
    src: str, tgt: str, ratio: float | None, reason: str | None
) -> None:
    result = length.apply(_lazy((src, tgt)), DEFAULT_CONFIG).collect()

    score = result["scores"].struct.field("len_ratio").to_list()
    assert score == ([None] if ratio is None else pytest.approx([ratio]))
    assert result["filter_reason"].to_list() == [reason]
    assert result.schema == CORPUS_SCHEMA


def _bounded(bounds: Bounds, *, by_direction: bool) -> LengthConfig:
    if by_direction:
        return LengthConfig(
            by_direction={DIRECTION: DirectionBounds(char_ratio=bounds)}
        )
    return LengthConfig(char_ratio=bounds)


@pytest.mark.parametrize("by_direction", [False, True], ids=["default", "direction"])
@pytest.mark.parametrize(
    ("bounds", "tgt_length", "reason"),
    [
        # float32(0.7) is below 0.7 and float32(1.1) is above 1.1, so a
        # comparison against the float64 bound would flag the two inside rows.
        (Bounds(min=0.7, max=2.0), 7, None),
        (Bounds(min=0.7, max=2.0), 6, "len_ratio_char"),
        (Bounds(min=0.5, max=1.1), 11, None),
        (Bounds(min=0.5, max=1.1), 12, "len_ratio_char"),
    ],
)
def test_a_ratio_equal_to_a_bound_stays_inside_it_in_float32(
    bounds: Bounds, tgt_length: int, reason: str | None, by_direction: bool
) -> None:
    frame = _lazy(("abcdefghij", "k" * tgt_length))

    result = length.apply(frame, _bounded(bounds, by_direction=by_direction)).collect()

    assert result["filter_reason"].to_list() == [reason]


def test_an_earlier_reason_is_kept() -> None:
    frame = _lazy(("ab", "abcde")).with_columns(
        pl.lit("lid_src").alias("filter_reason")
    )

    result = length.apply(frame, DEFAULT_CONFIG).collect()

    assert result["filter_reason"].to_list() == ["lid_src"]


def test_a_language_missing_from_the_registry_is_not_an_error() -> None:
    frame = corpus_frame([_row("ab", "abcde", src_lang="zzz_Latn")]).lazy()

    result = length.apply(frame, DEFAULT_CONFIG).collect()

    assert result["filter_reason"].to_list() == ["len_ratio_char"]


@pytest.mark.parametrize("column", [["src"], ["tgt"], ["src", "tgt"]])
def test_null_text_has_no_ratio_and_fails_the_char_check(column: list[str]) -> None:
    frame = _lazy(("abcd", "abcd")).with_columns(
        *(pl.lit(None, dtype=pl.String).alias(name) for name in column)
    )

    result = length.apply(frame, DEFAULT_CONFIG).collect()

    assert result["scores"].struct.field("len_ratio").to_list() == [None]
    assert result["filter_reason"].to_list() == ["len_ratio_char"]


@pytest.mark.parametrize("side", ["src_lang", "tgt_lang"])
def test_a_null_language_code_falls_back_to_the_default_bounds(side: str) -> None:
    cfg = _bounded(Bounds(min=0.9, max=1.1), by_direction=True)
    frame = _lazy(("abcd", "abcdef"))  # ratio 1.5: outside only the direction bounds

    overridden = length.apply(frame, cfg).collect()
    unknown = length.apply(
        frame.with_columns(pl.lit(None, dtype=pl.String).alias(side)), cfg
    ).collect()

    assert overridden["filter_reason"].to_list() == ["len_ratio_char"]
    assert unknown["filter_reason"].to_list() == [None]


def test_zero_rows_keep_the_corpus_schema() -> None:
    result = length.apply(_lazy(), DEFAULT_CONFIG).collect()

    assert result.height == 0
    assert result.schema == CORPUS_SCHEMA


def test_zero_rows_keep_the_corpus_schema_with_a_model(tiny_model: Path) -> None:
    result = length.apply(_lazy(), LengthConfig(spm_model=tiny_model)).collect()

    assert result.height == 0
    assert result.schema == CORPUS_SCHEMA


# --- apply: per-direction bounds -----------------------------------------------------


def test_a_direction_override_changes_the_verdict_for_that_direction_only() -> None:
    cfg = LengthConfig(
        by_direction={DIRECTION: DirectionBounds(char_ratio=Bounds(min=0.9, max=1.1))}
    )
    frame = corpus_frame(
        [
            _row("abcd", "abcdef"),
            _row("abcd", "abcdef", src_lang="eng_Latn", tgt_lang="khk_Cyrl"),
            _row("abcd", "abcd"),
        ]
    ).lazy()

    default = length.apply(frame, DEFAULT_CONFIG).collect()
    overridden = length.apply(frame, cfg).collect()

    assert default["filter_reason"].to_list() == [None, None, None]
    assert overridden["filter_reason"].to_list() == ["len_ratio_char", None, None]


# --- apply: the planted corpus -------------------------------------------------------


def test_planted_corpus_without_a_model_flags_only_the_char_rows(
    planted: PlantedCorpus,
) -> None:
    result = length.apply(planted.frame.lazy(), DEFAULT_CONFIG).collect()

    flagged = _reasons(result)

    char_rows = {
        row_id
        for row_id in planted.isolated_failures["length"]
        if planted.expected_reason[row_id] == "len_ratio_char"
    }
    assert set(flagged) == char_rows
    assert len(char_rows) == 3
    assert set(flagged.values()) == CHAR_REASONS
    assert result["scores"].struct.field("len_ratio").null_count() == 0


def test_planted_corpus_with_the_tiny_model_flags_the_ledger_rows(
    planted: PlantedCorpus, tiny_model: Path
) -> None:
    cfg = LengthConfig(spm_model=tiny_model)

    result = length.apply(planted.frame.lazy(), cfg).collect()

    flagged = _reasons(result)
    assert set(flagged) == planted.isolated_failures["length"]
    assert flagged == {row_id: planted.expected_reason[row_id] for row_id in flagged}
    assert sorted(flagged.values()) == ["len_ratio_char"] * 3 + ["len_ratio_token"] * 2
    assert result.schema == CORPUS_SCHEMA


def test_direction_overrides_replace_the_char_and_token_bounds_individually(
    planted: PlantedCorpus, tiny_model: Path
) -> None:
    wide = Bounds(min=0.01, max=1000.0)
    by_reason = {
        "len_ratio_char": {
            row_id
            for row_id in planted.isolated_failures["length"]
            if planted.expected_reason[row_id] == "len_ratio_char"
        },
        "len_ratio_token": {
            row_id
            for row_id in planted.isolated_failures["length"]
            if planted.expected_reason[row_id] == "len_ratio_token"
        },
    }
    wide_tokens = LengthConfig(
        spm_model=tiny_model,
        by_direction={DIRECTION: DirectionBounds(token_ratio=wide)},
    )
    wide_chars = LengthConfig(
        spm_model=tiny_model,
        by_direction={DIRECTION: DirectionBounds(char_ratio=wide)},
    )

    tokens_ignored = _reasons(length.apply(planted.frame.lazy(), wide_tokens).collect())
    chars_ignored = _reasons(length.apply(planted.frame.lazy(), wide_chars).collect())

    assert set(tokens_ignored) == by_reason["len_ratio_char"]
    assert set(chars_ignored) == by_reason["len_ratio_token"]


def test_a_row_failing_both_checks_gets_the_char_reason(
    planted: PlantedCorpus, tiny_model: Path
) -> None:
    fragmenting = next(
        row
        for row in planted.frame.iter_rows(named=True)
        if planted.expected_reason[row["id"]] == "len_ratio_token"
    )
    frame = _lazy((fragmenting["src"], fragmenting["tgt"] * 2))

    result = length.apply(frame, LengthConfig(spm_model=tiny_model)).collect()

    assert result["filter_reason"].to_list() == ["len_ratio_char"]


@pytest.mark.parametrize(
    ("src", "tgt"), [("ab", SOH + STX), (SOH + STX, "ab")], ids=["tgt", "src"]
)
def test_a_side_with_no_tokens_fails_the_token_check(
    normalising_char_model: Path, src: str, tgt: str
) -> None:
    cfg = LengthConfig(spm_model=normalising_char_model)

    result = length.apply(_lazy((src, tgt)), cfg).collect()

    assert result["scores"].struct.field("len_ratio").to_list() == [1.0]
    assert result["filter_reason"].to_list() == ["len_ratio_token"]


@pytest.mark.parametrize("name", ["absent.model", "."], ids=["missing", "directory"])
def test_a_model_path_that_is_not_a_file_is_a_config_error(
    tmp_path: Path, name: str
) -> None:
    cfg = LengthConfig(spm_model=tmp_path / name)

    with pytest.raises(ConfigError, match="not a file"):
        length.apply(_lazy(("ab", "ab")), cfg).collect()


def test_a_file_that_is_not_a_model_is_a_config_error(tmp_path: Path) -> None:
    broken = tmp_path / "broken.model"
    broken.write_bytes(b"not a sentencepiece model")

    with pytest.raises(ConfigError, match="broken.model"):
        length.apply(_lazy(("ab", "ab")), LengthConfig(spm_model=broken)).collect()


def test_token_check_is_skipped_exactly_when_no_spm_model_is_configured(
    tiny_model: Path,
) -> None:
    assert length.token_check_skip_reason(LengthConfig()) == (
        "token ratio skipped (no spm_model)"
    )
    assert length.token_check_skip_reason(LengthConfig(spm_model=tiny_model)) is None
