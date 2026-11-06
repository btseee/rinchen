"""The `length` stage: the two sides of a pair must be about as long as each
other, measured in script-weighted characters and, when a SentencePiece model
is configured, in tokens. Writes `scores.len_ratio`."""

from __future__ import annotations

import functools
from collections.abc import Mapping
from pathlib import Path

import polars as pl
import sentencepiece

from rinchen.config.filters import Bounds, LengthConfig
from rinchen.errors import ConfigError
from rinchen.filter.reasons import FilterReason, flag, with_scores
from rinchen.lang.scripts import Script, script_of

_TOKEN_RATIO = "_token_ratio"
NO_SPM_MODEL_REASON = "token ratio skipped (no spm_model)"


@functools.lru_cache(maxsize=65_536)
def _script_of_char(char: str) -> Script | None:
    return next(iter(script_of(char)), None)


def effective_length(text: str, weights: Mapping[Script, float]) -> float:
    """Sum of the weights of the non-whitespace characters of `text`. A
    character weighs 1.0 unless it is a letter of a script listed in
    `weights`."""
    total = 0.0
    for char in text:
        if not char.isspace():
            script = _script_of_char(char)
            total += 1.0 if script is None else weights.get(script, 1.0)
    return total


def token_check_skip_reason(cfg: LengthConfig) -> str | None:
    """Why the token-ratio check will not run, or None when it will: it needs
    `cfg.spm_model`."""
    return NO_SPM_MODEL_REASON if cfg.spm_model is None else None


def apply(df: pl.LazyFrame, cfg: LengthConfig) -> pl.LazyFrame:
    """Write `scores.len_ratio` and reject rows with `len_ratio_char` (ratio
    missing or out of bounds), then, when `cfg.spm_model` is set, with
    `len_ratio_token`. Raises ConfigError for a model that cannot be loaded."""
    processor = _load_model(cfg.spm_model) if cfg.spm_model is not None else None
    char_overrides = {
        key: bounds.char_ratio
        for key, bounds in cfg.by_direction.items()
        if bounds.char_ratio is not None
    }
    token_overrides = {
        key: bounds.token_ratio
        for key, bounds in cfg.by_direction.items()
        if bounds.token_ratio is not None
    }

    scored = with_scores(df, len_ratio=_char_ratio(cfg.char_weights))
    flagged = _flag_outside(
        scored,
        pl.col("scores").struct.field("len_ratio"),
        cfg.char_ratio,
        char_overrides,
        FilterReason.LEN_RATIO_CHAR,
    )
    if processor is None:
        return flagged
    counted = flagged.with_columns(_token_ratio(processor).alias(_TOKEN_RATIO))
    return _flag_outside(
        counted,
        pl.col(_TOKEN_RATIO),
        cfg.token_ratio,
        token_overrides,
        FilterReason.LEN_RATIO_TOKEN,
    ).drop(_TOKEN_RATIO)


def _load_model(path: Path) -> sentencepiece.SentencePieceProcessor:
    # SentencePiece aborts the interpreter on a directory, so only files pass.
    if not path.is_file():
        raise ConfigError(f"length.spm_model is not a file: {path}")
    try:
        return sentencepiece.SentencePieceProcessor(model_file=str(path))
    except (OSError, RuntimeError) as exc:
        raise ConfigError(
            f"length.spm_model {path} is not a SentencePiece model: {exc}"
        ) from exc


def _char_ratio(weights: Mapping[Script, float]) -> pl.Expr:
    def ratio_batch(batch: pl.Series) -> pl.Series:
        return pl.Series(
            [
                _ratio(
                    effective_length(row["tgt"], weights),
                    effective_length(row["src"], weights),
                )
                for row in batch
            ],
            dtype=pl.Float32,
        )

    return _text_pair().map_batches(
        ratio_batch, return_dtype=pl.Float32, is_elementwise=True
    )


def _token_ratio(processor: sentencepiece.SentencePieceProcessor) -> pl.Expr:
    def ratio_batch(batch: pl.Series) -> pl.Series:
        src_counts = map(len, processor.encode(batch.struct.field("src").to_list()))
        tgt_counts = map(len, processor.encode(batch.struct.field("tgt").to_list()))
        return pl.Series(
            [_ratio(tgt, src) for src, tgt in zip(src_counts, tgt_counts, strict=True)],
            dtype=pl.Float32,
        )

    return _text_pair().map_batches(
        ratio_batch, return_dtype=pl.Float32, is_elementwise=True
    )


def _text_pair() -> pl.Expr:
    return pl.struct(pl.col("src").fill_null(""), pl.col("tgt").fill_null(""))


def _ratio(tgt: float, src: float) -> float | None:
    return tgt / src if src and tgt else None


def _flag_outside(
    df: pl.LazyFrame,
    ratio: pl.Expr,
    default: Bounds,
    overrides: Mapping[str, Bounds],
    reason: FilterReason,
) -> pl.LazyFrame:
    """Flag rows whose ratio is missing or outside the bounds of their
    direction. Bounds and ratio are both Float32, so a ratio equal to a bound
    stays inside it."""
    direction = pl.concat_str("src_lang", "tgt_lang", separator="->")
    low = _bound(direction, default.min, {k: b.min for k, b in overrides.items()})
    high = _bound(direction, default.max, {k: b.max for k, b in overrides.items()})
    return flag(df, ratio.is_null() | (ratio < low) | (ratio > high), reason)


def _bound(
    direction: pl.Expr, default: float, overrides: Mapping[str, float]
) -> pl.Expr:
    if not overrides:
        return pl.lit(default, dtype=pl.Float32)
    return direction.replace_strict(overrides, default=default, return_dtype=pl.Float32)
