"""The `qe` stage: a translation that a reference-free quality estimator rates
poorly is rejected. Writes `scores.metricx_qe` (lower is better) or
`scores.cometkiwi` (higher is better), by `cfg.scorer`. The scorer is injected
and used for inference only; the pipeline asks `skip_reason` first and runs the
stage only when it returns None."""

from __future__ import annotations

import os
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal, Protocol

import polars as pl

from rinchen.config.filters import QeConfig
from rinchen.errors import UpstreamError
from rinchen.filter.reasons import FilterReason, flag, with_scores

NO_SCORER_REASON = "no QE scorer configured"
NO_CUDA_REASON = "no CUDA device"
EOLE_METRICX_QE_MODEL = "eole-nlp/metricx-24-hybrid-large-v2p6-eole"


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
    available = _cuda_is_available() if cuda_available is None else cuda_available
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


@dataclass(frozen=True)
class _LoadedMetricX:
    model: Any
    tokenizer: Any


class EoleMetricXQEScorer:
    """MetricX-24 hybrid in quality-estimation mode (source and translation, no
    reference), run by Eole's scorer model on the GPU. Scores run from 0 to 25
    and lower is better.

    `model` is a pre-converted Eole MetricX directory or Hugging Face repo; a
    repo is downloaded on first use to `~/.cache/eole/hf/`, about 6 GB. The
    model, the Eole imports and torch load on the first `score`, so importing
    this module stays cheap. `compute_dtype` defaults to `fp32`, Eole's own
    default."""

    requires_cuda = True

    def __init__(
        self,
        model: str = EOLE_METRICX_QE_MODEL,
        *,
        compute_dtype: Literal["fp32", "bf16", "fp16"] = "fp32",
    ) -> None:
        self._model_name = model
        self._compute_dtype = compute_dtype
        self._loaded: _LoadedMetricX | None = None

    def score(self, sources: Sequence[str], translations: Sequence[str]) -> list[float]:
        """Per-pair scores, all pairs in one forward batch. Eole's own scorer
        class only returns the corpus mean and reloads the model on every
        call, so this drives the same model and helpers directly."""
        if not sources:
            return []
        from eole.utils.scorer import (
            build_template_inputs,
            encode_scorer_inputs,
            predict_scores_with_oom_retry,
        )

        loaded = self._loaded_model()
        texts = build_template_inputs(
            list(sources),
            list(translations),
            None,
            loaded.model.input_templates,
            input_mode="qe",
        )
        input_ids = encode_scorer_inputs(texts, loaded.tokenizer, loaded.model)
        rows = [{"input_ids": ids} for ids in input_ids]
        scores: list[float] = predict_scores_with_oom_retry(
            loaded.model, rows, len(rows)
        )
        return scores

    def _loaded_model(self) -> _LoadedMetricX:
        if self._loaded is None:
            self._loaded = self._load()
        return self._loaded

    def _load(self) -> _LoadedMetricX:
        """Raises UpstreamError when the model cannot be found, downloaded or
        read, is not a MetricX QE model, or CUDA is not usable."""
        import torch
        from eole.constants import TORCH_DTYPES
        from eole.models.hf_resolver import resolve_preconverted_eole_hf_repo
        from eole.models.model import EncoderDecoderScoringModel
        from eole.utils.scorer import build_scorer_sentencepiece_transform

        try:
            model_dir = (
                self._model_name
                if os.path.isdir(self._model_name)
                else resolve_preconverted_eole_hf_repo(self._model_name)
            )
            if model_dir is None:
                raise UpstreamError(
                    f"{self._model_name!r} is neither a directory nor a "
                    "pre-converted Eole repo on Hugging Face"
                )
            model = EncoderDecoderScoringModel.from_model_dir(
                model_dir,
                device=torch.device("cuda"),
                compute_dtype=TORCH_DTYPES[self._compute_dtype],
            )
            if getattr(model, "scoring_type", None) != "metricx":
                raise UpstreamError(
                    f"{self._model_name!r} is not a MetricX scoring model"
                )
            model.validate_input_mode("qe")
            tokenizer = build_scorer_sentencepiece_transform(model, model_dir)
        except (OSError, RuntimeError, ValueError) as exc:
            raise UpstreamError(
                f"could not load MetricX QE model {self._model_name!r}: {exc}"
            ) from exc
        return _LoadedMetricX(model=model, tokenizer=tokenizer)


def _score(scorer: QEScorer, batch_size: int) -> pl.Expr:
    def score_batch(batch: pl.Series) -> pl.Series:
        sources = batch.struct.field("src").to_list()
        translations = batch.struct.field("tgt").to_list()
        scores: list[float] = []
        for start in range(0, len(sources), batch_size):
            chunk_sources = sources[start : start + batch_size]
            chunk_scores = scorer.score(
                chunk_sources, translations[start : start + batch_size]
            )
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


def _cuda_is_available() -> bool:
    import torch

    return torch.cuda.is_available()
