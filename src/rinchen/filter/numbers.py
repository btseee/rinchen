"""The `numbers` stage: both sides must carry the same multiset of digits.

Every Unicode decimal digit is read as its ASCII digit and every run of CJK
numerals as the digits of its value, so `1,5` and `1.5`, Mongolian digits and
`<ten><two>` all compare by the digits they spell."""

from __future__ import annotations

from collections import Counter

import polars as pl

from rinchen.filter.reasons import FilterReason, flag, with_scores
from rinchen.text.numerals import fold_numerals

_ASCII_DIGITS = frozenset("0123456789")


def digit_multiset(text: str) -> Counter[str]:
    """The ASCII digits `text` spells, counted. Separators and every other
    character are ignored."""
    return Counter(char for char in fold_numerals(text) if char in _ASCII_DIGITS)


def apply(df: pl.LazyFrame) -> pl.LazyFrame:
    """Write `scores.num_ok` and reject rows whose two sides spell different
    digits."""

    def same_digits(batch: pl.Series) -> pl.Series:
        return pl.Series(
            [digit_multiset(row["src"]) == digit_multiset(row["tgt"]) for row in batch],
            dtype=pl.Boolean,
        )

    num_ok = pl.struct(
        pl.col("src").fill_null(""), pl.col("tgt").fill_null("")
    ).map_batches(same_digits, return_dtype=pl.Boolean, is_elementwise=True)
    scored = with_scores(df, num_ok=num_ok)
    return flag(scored, ~pl.col("scores").struct.field("num_ok"), FilterReason.NUMBERS)
