"""The `labse` stage: the two sides of a pair must embed close together.
Writes `scores.labse`, the cosine of the sentence embeddings, and rejects a row
whose cosine is below `min_cosine`. Every row gets a score; an empty side is
embedded like any other text."""

from __future__ import annotations

import numpy as np
import polars as pl

from rinchen.config.filters import LabseConfig
from rinchen.filter.reasons import FilterReason, flag, with_scores
from rinchen.mine.embed import Embedder, SentenceTransformerEmbedder


def apply(
    df: pl.LazyFrame, cfg: LabseConfig, *, embedder: Embedder | None = None
) -> pl.LazyFrame:
    """Write `scores.labse` and reject rows below `cfg.min_cosine` as `labse`;
    a NaN score is rejected too. Without `embedder`, a LaBSE embedder is built
    from `cfg`; it loads its model only when a row has to be scored.

    The stage embeds each side of all rows in one call, so it holds two
    `n x d` float32 matrices at once, about 6 KB per row for LaBSE."""
    resolved = (
        embedder
        if embedder is not None
        else SentenceTransformerEmbedder(
            cfg.model, batch_size=cfg.batch_size, device=cfg.device
        )
    )
    scored = with_scores(df, labse=_cosine(resolved))
    floor = pl.lit(cfg.min_cosine, dtype=pl.Float32)
    labse_score = pl.col("scores").struct.field("labse")
    # Polars sorts NaN above every number, so `<` alone would keep a NaN row.
    return flag(
        scored, labse_score.is_nan() | (labse_score < floor), FilterReason.LABSE
    )


def _cosine(embedder: Embedder) -> pl.Expr:
    def cosine_batch(batch: pl.Series) -> pl.Series:
        src_vectors = embedder.embed(batch.struct.field("src").to_list())
        tgt_vectors = embedder.embed(batch.struct.field("tgt").to_list())
        # Unit vectors have a cosine in [-1, 1]; float32 rounding can push the
        # dot product of two near-identical vectors a hair past it.
        return pl.Series(
            np.clip(np.einsum("ij,ij->i", src_vectors, tgt_vectors), -1.0, 1.0),
            dtype=pl.Float32,
        )

    # Not elementwise: embedding row by row would defeat the embedder's own
    # batching, so the function has to see the whole column in one call.
    return pl.struct(
        pl.col("src").fill_null("").alias("src"),
        pl.col("tgt").fill_null("").alias("tgt"),
    ).map_batches(cosine_batch, return_dtype=pl.Float32)
