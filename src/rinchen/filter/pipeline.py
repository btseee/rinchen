"""The ordered filter pipeline: runs the stages a `FilterConfig` lists, in that
order, over a corpus frame. The order is the precedence of `filter_reason`: a
row keeps the reason of the first stage it fails. This module is the only one
that knows every stage; it holds no filtering rule of its own.

The corpus is held in memory, as the input and again as each stage's result:
budget two to three times its size."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from functools import partial

import polars as pl

from rinchen.config.filters import FilterConfig, StageName
from rinchen.corpus.schema import validate_corpus_schema
from rinchen.filter import labse, length, lid, minhash, numbers, qe, script_ratio
from rinchen.filter.qe import QEScorer
from rinchen.filter.reasons import FilterReason
from rinchen.mine.embed import Embedder

_OK = "ok"
_PARTIAL_PREFIX = "partial: "
_SKIPPED_PREFIX = "skipped: "


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
class _Plan:
    """What one stage does in this run: its status and, unless it is skipped,
    the function that applies it."""

    status: str
    apply: Callable[[pl.LazyFrame], pl.LazyFrame] | None


_Planner = Callable[[FilterConfig, _Models], _Plan]


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
    that caused it. The whole corpus is in memory, the input plus the current
    stage's result: budget two to three times its size."""
    validate_corpus_schema(df.collect_schema(), source="filter pipeline input")
    models = _Models(embedder=embedder, qe_scorer=qe_scorer)
    frame = df.collect()
    stage_status: dict[str, str] = {}
    for name in cfg.stages:
        plan = _PLANNERS[name](cfg, models)
        stage_status[name] = plan.status
        if plan.apply is not None:
            frame = plan.apply(frame.lazy()).collect()
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


def _runs(apply: Callable[[pl.LazyFrame], pl.LazyFrame]) -> _Plan:
    return _Plan(_OK, apply)


def _skipped(reason: str) -> _Plan:
    return _Plan(f"{_SKIPPED_PREFIX}{reason}", None)


def _plan_length(cfg: FilterConfig, models: _Models) -> _Plan:
    reason = length.token_check_skip_reason(cfg.length)
    status = _OK if reason is None else f"{_PARTIAL_PREFIX}{reason}"
    return _Plan(status, partial(length.apply, cfg=cfg.length))


def _plan_qe(cfg: FilterConfig, models: _Models) -> _Plan:
    scorer = models.qe_scorer
    if scorer is None:
        return _skipped(qe.NO_SCORER_REASON)
    reason = qe.skip_reason(cfg.qe, scorer)
    if reason is not None:
        return _skipped(reason)
    return _runs(partial(qe.apply, cfg=cfg.qe, scorer=scorer))


_PLANNERS: dict[StageName, _Planner] = {
    "script_ratio": lambda cfg, models: _runs(
        partial(script_ratio.apply, cfg=cfg.script_ratio)
    ),
    "numbers": lambda cfg, models: _runs(numbers.apply),
    "length": _plan_length,
    "lid": lambda cfg, models: _runs(partial(lid.apply, cfg=cfg.lid)),
    "minhash": lambda cfg, models: _runs(partial(minhash.apply, cfg=cfg.minhash)),
    "labse": lambda cfg, models: _runs(
        partial(labse.apply, cfg=cfg.labse, embedder=models.embedder)
    ),
    "qe": _plan_qe,
}
