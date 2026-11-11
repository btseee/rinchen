"""The names of the metrics an evaluation result reports (D13, T7 and T8), the order they come
in, which of them are computed from counts and which from scores handed in, and which way is
better. A table every layer above reads: it imports nothing but the errors.

`bleu`, `chrf` and `chrf_pp` (chrF++) are COUNT metrics: sacrebleu's statistics of every
segment, added up, give the score (`rinchen.metrics.segment_stats`). `comet22`, `metricx24` and
`metricx24_qe` are SEGMENT SCORE metrics: no scorer is built, they are `null` in every result
with `NO_SCORER_REASON` unless per-segment scores are handed in. MetricX scores are lower-is-better.
"""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType
from typing import Final, Literal

from rinchen.errors import ConfigError

CountMetricName = Literal["bleu", "chrf", "chrf_pp"]
MetricName = Literal["bleu", "chrf", "chrf_pp", "comet22", "metricx24", "metricx24_qe"]

# The order of the `metrics` of a result.
METRIC_NAMES: Final[tuple[MetricName, ...]] = (
    "bleu",
    "chrf",
    "chrf_pp",
    "comet22",
    "metricx24",
    "metricx24_qe",
)
COUNT_METRICS: Final[tuple[CountMetricName, ...]] = ("bleu", "chrf", "chrf_pp")
SEGMENT_SCORE_METRICS: Final[tuple[MetricName, ...]] = (
    "comet22",
    "metricx24",
    "metricx24_qe",
)
HIGHER_IS_BETTER: Final[Mapping[MetricName, bool]] = MappingProxyType(
    {
        "bleu": True,
        "chrf": True,
        "chrf_pp": True,
        "comet22": True,
        "metricx24": False,
        "metricx24_qe": False,
    }
)

NO_SCORER_REASON: Final = (
    "no segment scores were given: the COMET and MetricX scorers are not built"
)


def require_metric(name: str) -> MetricName:
    """`name` as a metric name.

    Raises:
        ConfigError: `name` is none of the six; the message names them.
    """
    for metric in METRIC_NAMES:
        if name == metric:
            return metric
    raise ConfigError(
        f"unknown metric {name!r}: the metrics are {', '.join(METRIC_NAMES)}"
    )


def require_count_metric(name: str) -> CountMetricName:
    """`name` as a count metric (one that has per-segment statistics).

    Raises:
        ConfigError: `name` is not `bleu`, `chrf` or `chrf_pp`; the message names them.
    """
    for metric in COUNT_METRICS:
        if name == metric:
            return metric
    raise ConfigError(
        f"{name!r} is not a metric computed from segment statistics: "
        f"the count metrics are {', '.join(COUNT_METRICS)}"
    )
