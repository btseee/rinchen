"""The ordered filter pipeline: runs the stages a `FilterConfig` lists, in that
order, over a corpus frame. The order is the precedence of `filter_reason`: a
row keeps the reason of the first stage it fails. This module is the only one
that knows every stage; it holds no filtering rule of its own."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import polars as pl

from rinchen.config.filters import FilterConfig, StageName
from rinchen.corpus.schema import validate_corpus_schema
from rinchen.filter import labse, length, lid, minhash, numbers, qe, script_ratio
from rinchen.filter.qe import QEScorer
from rinchen.filter.reasons import FilterReason
from rinchen.mine.embed import Embedder

_OK = "ok"
_SKIPPED_PREFIX = "skipped: "
_LENGTH_WITHOUT_SPM_STATUS = "partial: token ratio skipped (no spm_model)"


# Compared by identity: the frame's own `==` is elementwise, not a bool.
@dataclass(frozen=True, slots=True, eq=False)
class PipelineResult:
    """The scored corpus with the outcome of the run.

    `frame` holds every input row, in order, with `scores` and
    `filter_reason` filled in. `reason_counts` counts the rejected rows by
    reason and names every `FilterReason`, 0 when unused; a reason that was
    already on the input and is not a `FilterReason` is counted too. Kept rows
    are not counted. `stage_status` maps each configured stage, in order, to
    `ok`, `partial: <what was skipped>` or `skipped: <reason>`."""

    frame: pl.DataFrame
    reason_counts: dict[str, int]
    stage_status: dict[str, str]


@dataclass(frozen=True, slots=True)
class _Models:
    embedder: Embedder | None
    qe_scorer: QEScorer | None


@dataclass(frozen=True, slots=True)
class _Stage:
    apply: Callable[[pl.LazyFrame, FilterConfig, _Models], pl.LazyFrame]
    status: Callable[[FilterConfig, _Models], str]


def run_pipeline(
    df: pl.LazyFrame,
    cfg: FilterConfig,
    *,
    embedder: Embedder | None = None,
    qe_scorer: QEScorer | None = None,
) -> PipelineResult:
    """Run `cfg.stages` over `df` in order.

    Raises DataError when `df` is not a corpus table; each stage's own errors
    (a missing LID model, a QE failure) pass through. A stage whose status is
    `skipped` is not applied and adds no scores. `embedder` serves the `labse`
    stage, which builds the LaBSE embedder from `cfg` when it is None;
    `qe_scorer` serves `qe`, which is skipped without one.

    The frame is collected after every stage, so the LID model, LaBSE and the
    QE model are never in memory together and an error surfaces at the stage
    that caused it."""
    validate_corpus_schema(df.collect_schema(), source="filter pipeline input")
    models = _Models(embedder=embedder, qe_scorer=qe_scorer)
    frame = df.collect()
    stage_status: dict[str, str] = {}
    for name in cfg.stages:
        stage = _STAGES[name]
        status = stage.status(cfg, models)
        stage_status[name] = status
        if not status.startswith(_SKIPPED_PREFIX):
            frame = stage.apply(frame.lazy(), cfg, models).collect()
    return PipelineResult(
        frame=frame,
        reason_counts=_count_reasons(frame),
        stage_status=stage_status,
    )


def kept_rows(frame: pl.DataFrame) -> pl.DataFrame:
    """The rows of a pipeline result that no stage rejected."""
    return frame.filter(pl.col("filter_reason").is_null())


def _count_reasons(frame: pl.DataFrame) -> dict[str, int]:
    counts = dict.fromkeys((reason.value for reason in FilterReason), 0)
    found = frame.get_column("filter_reason").drop_nulls().value_counts()
    counts.update(sorted(found.iter_rows()))
    return counts


def _always_ok(cfg: FilterConfig, models: _Models) -> str:
    return _OK


def _length_status(cfg: FilterConfig, models: _Models) -> str:
    return _OK if cfg.length.spm_model is not None else _LENGTH_WITHOUT_SPM_STATUS


def _qe_status(cfg: FilterConfig, models: _Models) -> str:
    reason = qe.skip_reason(cfg.qe, models.qe_scorer)
    return _OK if reason is None else f"{_SKIPPED_PREFIX}{reason}"


def _apply_qe(df: pl.LazyFrame, cfg: FilterConfig, models: _Models) -> pl.LazyFrame:
    scorer = models.qe_scorer
    assert scorer is not None  # `_qe_status` skips the stage without a scorer
    return qe.apply(df, cfg.qe, scorer=scorer)


_STAGES: dict[StageName, _Stage] = {
    "script_ratio": _Stage(
        lambda df, cfg, models: script_ratio.apply(df, cfg.script_ratio), _always_ok
    ),
    "numbers": _Stage(lambda df, cfg, models: numbers.apply(df), _always_ok),
    "length": _Stage(
        lambda df, cfg, models: length.apply(df, cfg.length), _length_status
    ),
    "lid": _Stage(lambda df, cfg, models: lid.apply(df, cfg.lid), _always_ok),
    "minhash": _Stage(
        lambda df, cfg, models: minhash.apply(df, cfg.minhash), _always_ok
    ),
    "labse": _Stage(
        lambda df, cfg, models: labse.apply(df, cfg.labse, embedder=models.embedder),
        _always_ok,
    ),
    "qe": _Stage(_apply_qe, _qe_status),
}
