"""The `script_ratio` stage: how much of each side is written in the scripts of
its language. Writes `scores.script_src` / `scores.script_tgt` and rejects a
row whose share is below `min_ratio`."""

from __future__ import annotations

from collections.abc import Collection, Mapping

import polars as pl

from rinchen.config.filters import ScriptRatioConfig
from rinchen.config.languages import load_languages
from rinchen.errors import DataError
from rinchen.filter.reasons import FilterReason, flag, with_scores
from rinchen.lang.scripts import Script, script_of


def script_share(text: str, scripts: Collection[Script]) -> float:
    """Share of the letters of `text` that belong to `scripts`.

    Letters of any other script count in the denominator only. Text without
    letters is vacuously in script and scores 1.0."""
    shares = script_of(text)
    if shares:
        return sum(shares.get(script, 0.0) for script in set(scripts))
    # No share at all: no letters, or only letters of scripts D3 does not name.
    return 0.0 if any(char.isalpha() for char in text) else 1.0


def apply(df: pl.LazyFrame, cfg: ScriptRatioConfig) -> pl.LazyFrame:
    """Score both sides, then reject with `script_src` first and `script_tgt`
    second. Raises DataError, when collected, for a language code that is not
    in the registry."""
    registry = {entry.code: frozenset(entry.scripts) for entry in load_languages()}
    scored = with_scores(
        df,
        script_src=_side_share("src", "src_lang", registry),
        script_tgt=_side_share("tgt", "tgt_lang", registry),
    )
    floor = pl.lit(cfg.min_ratio, dtype=pl.Float32)
    src_share = pl.col("scores").struct.field("script_src")
    tgt_share = pl.col("scores").struct.field("script_tgt")
    src_flagged = flag(scored, src_share < floor, FilterReason.SCRIPT_SRC)
    return flag(src_flagged, tgt_share < floor, FilterReason.SCRIPT_TGT)


def _side_share(
    text_column: str, lang_column: str, registry: Mapping[str, frozenset[Script]]
) -> pl.Expr:
    def share_batch(batch: pl.Series) -> pl.Series:
        return pl.Series(
            [_share_in_language(row["text"], row["lang"], registry) for row in batch],
            dtype=pl.Float32,
        )

    return pl.struct(
        pl.col(text_column).fill_null("").alias("text"),
        pl.col(lang_column).alias("lang"),
    ).map_batches(share_batch, return_dtype=pl.Float32, is_elementwise=True)


def _share_in_language(
    text: str, lang: str | None, registry: Mapping[str, frozenset[Script]]
) -> float:
    scripts = registry.get(lang) if lang is not None else None
    if scripts is None:
        raise DataError(
            f"language code {lang!r} is not in the language registry "
            "(configs/languages.yaml)"
        )
    return script_share(text, scripts)
