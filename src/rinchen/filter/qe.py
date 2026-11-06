"""The `qe` stage: a translation that a reference-free quality estimator rates
poorly is rejected. Writes `scores.metricx_qe` (lower is better) or
`scores.cometkiwi` (higher is better), by `cfg.scorer`. The scorer is injected
and used for inference only; the pipeline asks `skip_reason` first and runs the
stage only when it returns None."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

import polars as pl

from rinchen.config.filters import QeConfig
from rinchen.errors import UpstreamError
from rinchen.filter.reasons import FilterReason, flag, with_scores

NO_SCORER_REASON = "no QE scorer configured"
NO_CUDA_REASON = "no CUDA device"


class QEScorer(Protocol):
    requires_cuda: bool

    def score(self, sources: Sequence[str], translations: Sequence[str]) -> list[float]:
        """One quality score per pair, in order, in the scale of the scorer."""
        ...


def skip_reason(
    cfg: QeConfig, scorer: QEScorer | None, *, cuda_available: bool | None = None
) -> str | None:
    """Why the stage cannot run, or None when it can. `cuda_available` is looked
    up from torch only when it is None and `scorer` requires CUDA. `cfg` is
    unused today; it keeps every stage's skip check on one signature."""
    if scorer is None:
        return NO_SCORER_REASON
    if not scorer.requires_cuda:
        return None
    available = cuda_is_available() if cuda_available is None else cuda_available
    return None if available else NO_CUDA_REASON


def apply(df: pl.LazyFrame, cfg: QeConfig, *, scorer: QEScorer) -> pl.LazyFrame:
    """Score every row with `scorer`, `cfg.batch_size` rows per call, and reject
    as `qe` the rows whose score is worse than the configured threshold: above
    `max_metricx_qe`, or below `min_cometkiwi`. A score exactly at the threshold
    passes; a NaN score is rejected."""
    if cfg.scorer == "metricx_qe":
        scored = with_scores(df, metricx_qe=_score(scorer, cfg.batch_size))
        score = pl.col("scores").struct.field("metricx_qe")
        worse_than_threshold = score > pl.lit(cfg.max_metricx_qe, dtype=pl.Float32)
    else:
        scored = with_scores(df, cometkiwi=_score(scorer, cfg.batch_size))
        score = pl.col("scores").struct.field("cometkiwi")
        worse_than_threshold = score < pl.lit(cfg.min_cometkiwi, dtype=pl.Float32)
    # Polars sorts NaN above every number, so only a "greater than" test would
    # catch a NaN score without the explicit check.
    return flag(scored, score.is_nan() | worse_than_threshold, FilterReason.QE)


def _score(scorer: QEScorer, batch_size: int) -> pl.Expr:
    def score_batch(batch: pl.Series) -> pl.Series:
        sources = batch.struct.field("src")
        translations = batch.struct.field("tgt")
        scores: list[float] = []
        # Slices become Python lists one chunk at a time, so the column is
        # never held as strings twice.
        for start in range(0, len(sources), batch_size):
            chunk_sources = sources.slice(start, batch_size).to_list()
            chunk_translations = translations.slice(start, batch_size).to_list()
            chunk_scores = scorer.score(chunk_sources, chunk_translations)
            if len(chunk_scores) != len(chunk_sources):
                raise UpstreamError(
                    f"QE scorer returned {len(chunk_scores)} scores "
                    f"for {len(chunk_sources)} rows"
                )
            scores.extend(chunk_scores)
        return pl.Series(scores, dtype=pl.Float32)

    # Not elementwise: scoring row by row would defeat the scorer's own
    # batching, so the function has to see the whole column in one call.
    return pl.struct(
        pl.col("src").fill_null("").alias("src"),
        pl.col("tgt").fill_null("").alias("tgt"),
    ).map_batches(score_batch, return_dtype=pl.Float32)


def cuda_is_available() -> bool:
    """Whether torch sees a CUDA device. Imports torch on the call."""
    import torch

    return torch.cuda.is_available()
