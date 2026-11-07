"""The `minhash` stage: rejects a row whose `src + TAB + tgt` shares at least
`threshold` of its character shingles with an earlier kept row, as `duplicate`.

The first copy is kept. A kept row is one with no `filter_reason` that this
stage did not reject itself, so a row rejected earlier, here or in another
stage, never shadows a later one."""

from __future__ import annotations

from collections.abc import Sequence

import polars as pl
from datasketch import MinHash, MinHashLSH

from rinchen.config.filters import MinHashConfig
from rinchen.filter.reasons import FilterReason, flag


def duplicate_flags(
    texts: Sequence[str], alive: Sequence[bool], cfg: MinHashConfig
) -> list[bool]:
    """For each text, whether it duplicates an earlier kept text, one that is
    alive and was not itself flagged.

    A text that is not alive is never flagged and never compared against.
    Raises ValueError when `texts` and `alive` differ in length."""
    if len(texts) != len(alive):
        raise ValueError(
            "texts and alive must have the same length, "
            f"got {len(texts)} and {len(alive)}"
        )
    index = MinHashLSH(threshold=cfg.threshold, num_perm=cfg.num_perm)
    flags: list[bool] = []
    for position, (text, is_alive) in enumerate(zip(texts, alive, strict=True)):
        if not is_alive:
            flags.append(False)
            continue
        signature = _signature(text, cfg)
        is_duplicate = bool(index.query(signature))
        if not is_duplicate:
            index.insert(position, signature)
        flags.append(is_duplicate)
    return flags


def apply(df: pl.LazyFrame, cfg: MinHashConfig) -> pl.LazyFrame:
    """Reject near-duplicate rows as `duplicate`. A null `src` or `tgt` reads
    as empty text; the stage writes no score."""

    def duplicates(batch: pl.Series) -> pl.Series:
        rows = batch.to_list()
        return pl.Series(
            duplicate_flags(
                [f"{row['src']}\t{row['tgt']}" for row in rows],
                [row["reason"] is None for row in rows],
                cfg,
            ),
            dtype=pl.Boolean,
        )

    # Not elementwise: whether a row is a duplicate depends on the rows before
    # it, so the function has to see the whole column in row order, never a
    # chunk of it or one row at a time.
    is_duplicate = pl.struct(
        pl.col("src").fill_null("").alias("src"),
        pl.col("tgt").fill_null("").alias("tgt"),
        pl.col("filter_reason").alias("reason"),
    ).map_batches(duplicates, return_dtype=pl.Boolean)
    return flag(df, is_duplicate, FilterReason.DUPLICATE)


def _signature(text: str, cfg: MinHashConfig) -> MinHash:
    signature = MinHash(num_perm=cfg.num_perm, seed=cfg.seed)
    signature.update_batch(
        [shingle.encode("utf-8") for shingle in _shingles(text, cfg.ngram)]
    )
    return signature


def _shingles(text: str, length: int) -> set[str]:
    """The `length`-character substrings of `text`; a text shorter than that is
    one shingle of its own."""
    if len(text) < length:
        return {text}
    return {text[start : start + length] for start in range(len(text) - length + 1)}
