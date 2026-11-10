"""Scoring one direction of a suite (D13, spec T7 to T9): the value, the interval and the signature
of every metric, from the outputs of a system and the references of the suite.

The statistics of every segment (`rinchen.metrics.segment_stats`: sacrebleu's integers) are
taken ONCE per count metric and grouped into the units of the bootstrap: the document when every
row of the direction has one (the sentences of an article stay together), else the segment.
The value is the score of the column sums, the interval the percentile interval of the scores of
the resamples (`rinchen.eval.significance.estimate`); every metric of the direction uses the same
draws. The tokenizer of BLEU follows the TARGET language of the view. The text of an output is
scored as it is whatever its status, and the statuses are counted: a failure the system flagged
stays visible beside the metric.

COMET and MetricX are not computed here. Per-segment scores handed in (`segment_scores`, read by
`rinchen.eval.segment_scores`) make their mean the value and are bootstrapped through integer
micro-units with the same machinery; a metric without scores is `None` with the reason
`NO_SCORER_REASON`, never 0.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Final

import numpy as np
from numpy.typing import NDArray

from rinchen.eval.bootstrap import (
    MIN_RESAMPLES,
    RNG_NAME,
    BootstrapSettings,
    group_rows,
)
from rinchen.eval.field_rules import (
    problem,
    require_choice,
    require_integer,
)
from rinchen.eval.outputs import Output
from rinchen.eval.significance import Scorer, estimate
from rinchen.eval.suites import DirectionView
from rinchen.metrics.registry import (
    COUNT_METRICS,
    METRIC_NAMES,
    NO_SCORER_REASON,
    SEGMENT_SCORE_METRICS,
    CountMetricName,
)
from rinchen.metrics.segment_stats import (
    StatisticsMetric,
    mean_from_statistics,
    mean_statistics,
    score_from_statistics,
    segment_statistics,
    statistics_metric,
)

__all__ = [
    "INTERVAL_KIND",
    "UNIT_NAMES",
    "BootstrapRecord",
    "DirectionRows",
    "DirectionScores",
    "MetricRows",
    "direction_rows",
    "score_direction",
]

INTERVAL_KIND: Final = "percentile"
UNIT_NAMES: Final = ("document", "segment")
_MAX_SEED: Final = 2**64 - 1


@dataclass(frozen=True, slots=True)
class BootstrapRecord:
    """What the intervals of a result are: `resamples` draws from `seed` over `units` units of
    the kind `unit` (`document` or `segment`), a `interval` of the kind `percentile` at level
    `1 - alpha`, the stream of draws named `rng`.

    Raises:
        ValueError: a field that the bootstrap cannot have (the message names it, never its
            value).
    """

    resamples: int
    seed: int
    alpha: float
    interval: str
    rng: str
    unit: str
    units: int

    def __post_init__(self) -> None:
        require_integer(self.resamples, "bootstrap.resamples", minimum=MIN_RESAMPLES)
        require_integer(self.seed, "bootstrap.seed", maximum=_MAX_SEED)
        if not isinstance(self.alpha, float) or not 0 < self.alpha < 0.5:
            raise problem("bootstrap.alpha", "must be a number above 0 and below 0.5")
        if self.interval != INTERVAL_KIND:
            raise problem("bootstrap.interval", f"must be {INTERVAL_KIND}")
        if self.rng != RNG_NAME:
            raise problem("bootstrap.rng", f"must be {RNG_NAME}")
        require_choice(self.unit, "bootstrap.unit", UNIT_NAMES)
        require_integer(self.units, "bootstrap.units", minimum=1)

    def to_dict(self) -> dict[str, object]:
        return {
            "resamples": self.resamples,
            "seed": self.seed,
            "alpha": self.alpha,
            "interval": self.interval,
            "rng": self.rng,
            "unit": self.unit,
            "units": self.units,
        }


@dataclass(frozen=True, slots=True)
class DirectionScores:
    """The scores of one direction: its `n_segments`, the `statuses` of its outputs counted,
    the six `metrics` (None for one that was not measured, its reason in `unmeasured`), the
    interval `ci95` of each (`[low, high]`, None without a value or with fewer than two
    units), the `signatures` of the three count metrics and the `bootstrap` record. Every
    mapping holds the six metrics in the order of `METRIC_NAMES` and is read-only."""

    n_segments: int
    statuses: Mapping[str, int]
    metrics: Mapping[str, float | None]
    ci95: Mapping[str, list[float] | None]
    signatures: Mapping[str, str]
    unmeasured: Mapping[str, str]
    bootstrap: BootstrapRecord

    def __post_init__(self) -> None:
        for name in ("statuses", "metrics", "ci95", "signatures", "unmeasured"):
            object.__setattr__(self, name, MappingProxyType(dict(getattr(self, name))))


@dataclass(frozen=True, slots=True)
class MetricRows:
    """The integer statistics of a count metric over the units of a direction: `rows` is a
    `(units, width)` int64 array, `signature` the one sacrebleu reported for the text."""

    metric: StatisticsMetric
    rows: NDArray[np.int64]
    signature: str


@dataclass(frozen=True, slots=True)
class DirectionRows:
    """The statistics of every count metric of a direction, grouped by `unit` (`document` or
    `segment`) into the rows whose `labels` name the units in order of first appearance."""

    unit: str
    labels: tuple[str, ...]
    metrics: Mapping[CountMetricName, MetricRows]


def direction_rows(view: DirectionView, outputs: Sequence[Output]) -> DirectionRows:
    """The statistics of the three count metrics of `outputs` (in the order of the rows of
    `view`) against the references of `view`, grouped by the view's documents when it has them.
    The tokenizer of BLEU is chosen by the TARGET language of the view.

    Raises:
        ValueError: no segment, or a different number of outputs than segments (the message
            gives the counts and never a text).
    """
    if len(outputs) != len(view.ids):
        raise ValueError(
            f"{len(outputs)} outputs for {len(view.ids)} segments: a direction needs "
            "one output for each"
        )
    hypotheses = [output.text for output in outputs]
    found: dict[CountMetricName, MetricRows] = {}
    labels: tuple[str, ...] = ()
    for name in COUNT_METRICS:
        metric = statistics_metric(name, view.tgt_lang)
        statistics = segment_statistics(metric, hypotheses, view.references)
        rows, labels = group_rows(statistics.rows, view.docs)
        found[name] = MetricRows(metric, rows, statistics.signature)
    unit = "segment" if view.docs is None else "document"
    return DirectionRows(unit, labels, MappingProxyType(found))


def score_direction(
    view: DirectionView,
    outputs: Sequence[Output],
    settings: BootstrapSettings,
    *,
    segment_scores: Mapping[str, Sequence[float]] | None = None,
) -> DirectionScores:
    """Score `outputs` over `view`: BLEU, chrF and chrF++ from the statistics and, for each
    metric of `segment_scores` (per-segment scores in the order of the rows of `view`), the
    mean with its interval. A metric without scores is None with `NO_SCORER_REASON`.

    Raises:
        ValueError: no segment, outputs or scores of another number than the segments, a name in
            `segment_scores` that takes no segment scores, or a score that is not finite or is
            larger than 1000 in magnitude. A message gives counts and indices, never a text.
    """
    rows = direction_rows(view, outputs)
    given = _checked_scores(segment_scores, len(view.ids))
    estimates: dict[str, tuple[float, list[float] | None]] = {}
    for counted_name in COUNT_METRICS:
        counted = rows.metrics[counted_name]
        result = estimate(counted.rows, _scorer_of(counted.metric), settings)
        estimates[counted_name] = (result.value, _as_list(result.interval))
    for given_name, scores in given.items():
        means, _ = group_rows(mean_statistics(scores), view.docs)
        result = estimate(means, mean_from_statistics, settings)
        estimates[given_name] = (result.value, _as_list(result.interval))
    unmeasured: dict[str, str] = {
        name: NO_SCORER_REASON for name in SEGMENT_SCORE_METRICS if name not in given
    }
    bootstrap = BootstrapRecord(
        settings.resamples,
        settings.seed,
        settings.alpha,
        INTERVAL_KIND,
        RNG_NAME,
        rows.unit,
        len(rows.labels),
    )
    return DirectionScores(
        n_segments=len(view.ids),
        statuses=dict(sorted(Counter(output.status for output in outputs).items())),
        metrics={name: _value(estimates, name) for name in METRIC_NAMES},
        ci95={name: _interval(estimates, name) for name in METRIC_NAMES},
        signatures={name: rows.metrics[name].signature for name in COUNT_METRICS},
        unmeasured=unmeasured,
        bootstrap=bootstrap,
    )


def _scorer_of(metric: StatisticsMetric) -> Scorer:
    def score(totals: Sequence[int]) -> float:
        return score_from_statistics(metric, totals)

    return score


def _checked_scores(
    segment_scores: Mapping[str, Sequence[float]] | None, segments: int
) -> Mapping[str, Sequence[float]]:
    given = segment_scores or {}
    for name, scores in given.items():
        if name not in SEGMENT_SCORE_METRICS:
            raise ValueError(
                "segment scores can only be given for "
                f"{', '.join(SEGMENT_SCORE_METRICS)}"
            )
        if len(scores) != segments:
            raise ValueError(
                f"{len(scores)} scores for {segments} segments: a metric needs one "
                "score for each segment"
            )
    return given


def _as_list(interval: tuple[float, float] | None) -> list[float] | None:
    return None if interval is None else list(interval)


def _value(
    estimates: Mapping[str, tuple[float, list[float] | None]], name: str
) -> float | None:
    return estimates[name][0] if name in estimates else None


def _interval(
    estimates: Mapping[str, tuple[float, list[float] | None]], name: str
) -> list[float] | None:
    return estimates[name][1] if name in estimates else None
