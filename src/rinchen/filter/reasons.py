"""The rejection vocabulary and the two LazyFrame helpers every filter stage
uses to record it: `flag` for `filter_reason`, `with_scores` for `scores`."""

from __future__ import annotations

from enum import StrEnum

import polars as pl

from rinchen.corpus.schema import SCORES_DTYPE

_SCORE_FIELDS = SCORES_DTYPE.to_schema()


class FilterReason(StrEnum):
    SCRIPT_SRC = "script_src"
    SCRIPT_TGT = "script_tgt"
    LID_SRC = "lid_src"
    LID_TGT = "lid_tgt"
    LEN_RATIO_CHAR = "len_ratio_char"
    LEN_RATIO_TOKEN = "len_ratio_token"
    NUMBERS = "numbers"
    DUPLICATE = "duplicate"
    LABSE = "labse"
    QE = "qe"


def flag(df: pl.LazyFrame, failed: pl.Expr, reason: FilterReason) -> pl.LazyFrame:
    """Set `filter_reason` to `reason` where `failed` is true and no reason is
    set yet, so the first failing stage keeps the row. A null `failed` is not
    a failure."""
    first_failure = failed.fill_null(False) & pl.col("filter_reason").is_null()
    return df.with_columns(
        pl.when(first_failure)
        .then(pl.lit(reason.value, dtype=pl.String))
        .otherwise(pl.col("filter_reason"))
        .alias("filter_reason")
    )


def with_scores(df: pl.LazyFrame, **scores: pl.Expr) -> pl.LazyFrame:
    """Replace or fill the named fields of the `scores` struct and keep the
    rest. Each expression is cast to the field's declared type, so the frame
    stays valid against `CORPUS_SCHEMA`. Raises ValueError for a name that is
    not a field of `SCORES_DTYPE`."""
    unknown = sorted(set(scores) - set(_SCORE_FIELDS))
    if unknown:
        raise ValueError(
            f"not score fields: {', '.join(unknown)}; "
            f"valid fields: {', '.join(_SCORE_FIELDS)}"
        )
    if not scores:
        return df
    # A struct that is null as a whole (a freshly built corpus) would stay
    # null through with_fields, so start from a struct of null fields.
    unscored = pl.struct(
        [pl.lit(None, dtype=dtype).alias(name) for name, dtype in _SCORE_FIELDS.items()]
    )
    current = (
        pl.when(pl.col("scores").is_null()).then(unscored).otherwise(pl.col("scores"))
    )
    updated = current.struct.with_fields(
        *(expr.cast(_SCORE_FIELDS[name]).alias(name) for name, expr in scores.items())
    )
    return df.with_columns(updated.alias("scores"))
