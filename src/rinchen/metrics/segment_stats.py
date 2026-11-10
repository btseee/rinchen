"""BLEU, chrF and chrF++ from per-segment integer statistics, and the mean seam of the neural
metrics (D13, T7 and T8).

sacrebleu extracts, for every segment, a row of integers (`STATISTICS_WIDTH` of them: lengths and
n-gram counts), and a corpus score is its own score-from-statistics of the SUM of the rows. So the
statistics are taken once and every score, a bootstrap resample's included, is a score of a sum of
integers, equal with `==` to what sacrebleu scores over the whole corpus and clamped to 100.0 as
`rinchen.metrics.bleu_chrf.corpus_scores` clamps it (BLEU of identical text is
100.00000000000004). A corpus score is never the mean of sentence scores.

The two statistics methods are PRIVATE to the pinned `sacrebleu==2.6.0`
(`_extract_corpus_statistics`, `_compute_score_from_stats`): the tests compare them with the
public `corpus_score`, so a new version that changes them fails loudly.

BLEU's tokenizer follows the TARGET language: `zh` for Chinese, `char` for Japanese and Korean
(no MeCab here), `13a` for the rest. Only `ALLOWED_TOKENIZERS` may be built. `flores200`,
`flores101`, `spm` and `spBLEU-1K` download a SentencePiece model on their first use and
`ja-mecab` and `ko-mecab` need MeCab: a metric with one of them is refused by name, and the
refusal happens in `StatisticsMetric` itself, before sacrebleu is asked to build anything.

The mean seam: a score handed in for a segment (COMET, MetricX) becomes an integer count of
micro-units, `(round(score * MICRO), 1)`, so that a mean is a sum of integers divided once and
the bootstrap can resample it exactly like the count metrics.

Importing this module loads sacrebleu (`bleu_chrf` already does) and the language registry
(`rinchen.config.languages`: pydantic and PyYAML), and no model, dataset or hub library.
"""

from __future__ import annotations

import functools
import math
import operator
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Final

from sacrebleu.metrics import BLEU, CHRF

from rinchen.config.languages import known_language_codes
from rinchen.errors import ConfigError, UpstreamError
from rinchen.metrics.registry import CountMetricName, require_count_metric

ALLOWED_TOKENIZERS: Final = frozenset({"13a", "zh", "char", "intl", "none"})
BLEU_TOKENIZER_BY_TARGET: Final[Mapping[str, str]] = MappingProxyType(
    {"zho_Hans": "zh", "jpn_Jpan": "char", "kor_Hang": "char"}
)
DEFAULT_TOKENIZER: Final = "13a"
STATISTICS_WIDTH: Final[Mapping[str, int]] = MappingProxyType(
    {"bleu": 10, "chrf": 18, "chrf_pp": 24}
)
MICRO: Final = 1_000_000

# A COMET or MetricX score is a few units at most; a larger one is a corrupt file, and a bound
# keeps a sum of micro-units far from the 64-bit limit.
_MAX_ABS_SCORE: Final = 1000


@dataclass(frozen=True, slots=True)
class StatisticsMetric:
    """A metric with per-segment statistics. `tokenizer` is BLEU's and is `None` for the chrF
    metrics, which have none. A tokenizer outside `ALLOWED_TOKENIZERS` cannot be held, however
    the metric is built."""

    name: CountMetricName
    tokenizer: str | None

    def __post_init__(self) -> None:
        require_count_metric(self.name)
        if self.name != "bleu":
            if self.tokenizer is not None:
                raise ConfigError(
                    f"{self.name} has no tokenizer: it works on characters"
                )
        elif self.tokenizer not in ALLOWED_TOKENIZERS:
            raise ConfigError(
                f"BLEU tokenizer {self.tokenizer!r} is not allowed: the tokenizers are "
                f"{', '.join(sorted(ALLOWED_TOKENIZERS))} (flores200, flores101, spm and "
                "spBLEU-1K download a model, ja-mecab and ko-mecab need MeCab)"
            )


@dataclass(frozen=True, slots=True)
class SegmentStatistics:
    """The statistics of every segment of a corpus, in order, and the signature of the metric
    as sacrebleu reports it once it has read them (BLEU's names the tokenizer)."""

    metric: StatisticsMetric
    rows: tuple[tuple[int, ...], ...]
    signature: str


def statistics_metric(name: str, tgt_lang: str) -> StatisticsMetric:
    """The metric `name` for a corpus whose target language is `tgt_lang`: BLEU with the
    tokenizer of that language, or a chrF metric.

    Raises:
        ConfigError: `name` is not a count metric, or `tgt_lang` is not in the language
            registry (`khk_Mong` and any other code outside it are refused).
    """
    metric_name = require_count_metric(name)
    if tgt_lang not in known_language_codes():
        raise ConfigError(
            f"target language {tgt_lang!r} is not in the language registry"
        )
    if metric_name != "bleu":
        return StatisticsMetric(metric_name, None)
    tokenizer = BLEU_TOKENIZER_BY_TARGET.get(tgt_lang, DEFAULT_TOKENIZER)
    return StatisticsMetric(metric_name, tokenizer)


def metric_with_tokenizer(name: str, tokenizer: str) -> StatisticsMetric:
    """The metric `name` with an explicit BLEU tokenizer.

    Raises:
        ConfigError: `name` is not a count metric, a chrF metric was given a tokenizer, or the
            tokenizer is not in `ALLOWED_TOKENIZERS` (raised before sacrebleu is asked).
    """
    return StatisticsMetric(require_count_metric(name), tokenizer)


def _build(name: CountMetricName, tokenizer: str | None) -> BLEU | CHRF:
    if name == "bleu":
        return BLEU(tokenize=tokenizer)
    return CHRF(word_order=2 if name == "chrf_pp" else 0)


@functools.cache
def _scorer(name: CountMetricName, tokenizer: str | None) -> BLEU | CHRF:
    """The sacrebleu metric that turns a vector of totals into a score; one per metric and
    tokenizer, as it holds nothing a score depends on."""
    return _build(name, tokenizer)


def segment_statistics(
    metric: StatisticsMetric, hypotheses: Sequence[str], references: Sequence[str]
) -> SegmentStatistics:
    """The statistics of every segment of `hypotheses` against one reference each, with a fresh
    sacrebleu metric (a signature describes the call that made it, and is only known once the
    metric has read text).

    Raises:
        ValueError: no segment, or a different number of hypotheses and references (the message
            gives the counts and never a text).
        UpstreamError: sacrebleu returned statistics of another shape than the pinned version.
    """
    if len(hypotheses) != len(references):
        raise ValueError(
            f"{len(hypotheses)} hypotheses but {len(references)} references: "
            "statistics need one reference per hypothesis"
        )
    if not hypotheses:
        raise ValueError("at least one segment is needed to take statistics")
    scorer = _build(metric.name, metric.tokenizer)
    extracted = scorer._extract_corpus_statistics(list(hypotheses), [list(references)])
    rows = tuple(tuple(int(cell) for cell in row) for row in extracted)
    width = STATISTICS_WIDTH[metric.name]
    if len(rows) != len(hypotheses) or any(len(row) != width for row in rows):
        raise UpstreamError(
            f"sacrebleu did not return {width} integers for each of the "
            f"{len(hypotheses)} segments of {metric.name}: its statistics changed"
        )
    return SegmentStatistics(metric, rows, str(scorer.get_signature()))


def totals_of(rows: Sequence[Sequence[int]]) -> tuple[int, ...]:
    """The column sums of `rows`.

    Raises:
        ValueError: no row, or rows of different widths.
    """
    if not rows:
        raise ValueError("at least one row is needed to add statistics up")
    width = len(rows[0])
    if any(len(row) != width for row in rows):
        raise ValueError(
            f"rows of different width cannot be added up (first row: {width})"
        )
    return tuple(sum(column) for column in zip(*rows, strict=True))


def score_from_statistics(metric: StatisticsMetric, totals: Sequence[int]) -> float:
    """The corpus score of `totals`, the statistics of a corpus (or of a resample) added up:
    sacrebleu's own score-from-statistics of a copy of the integers, at most 100.0. `totals` is
    never changed.

    Raises:
        ValueError: `totals` does not have the width of the metric.
    """
    cells = [operator.index(total) for total in totals]
    width = STATISTICS_WIDTH[metric.name]
    if len(cells) != width:
        raise ValueError(
            f"{metric.name} has {width} statistics, {len(cells)} were given"
        )
    score = _scorer(metric.name, metric.tokenizer)._compute_score_from_stats(cells)
    return min(100.0, float(score.score))


def mean_statistics(scores: Sequence[float]) -> tuple[tuple[int, int], ...]:
    """The statistics of the segment scores of a mean-type metric: `(micro-units, 1)` for each.

    Raises:
        ValueError: a score is not finite or is larger than 1000 in magnitude (the message
            gives its index).
    """
    rows: list[tuple[int, int]] = []
    for index, score in enumerate(scores):
        if not math.isfinite(score) or abs(score) > _MAX_ABS_SCORE:
            raise ValueError(
                f"score {index} is not a finite number of at most {_MAX_ABS_SCORE} "
                "in magnitude"
            )
        rows.append((round(score * MICRO), 1))
    return tuple(rows)


def mean_from_statistics(totals: Sequence[int]) -> float:
    """The mean of the scores whose `(micro-units, count)` statistics add up to `totals`.

    Raises:
        ValueError: `totals` is not a sum and a count (width 2), or the count is below 1.
    """
    if len(totals) != 2:
        raise ValueError(f"a mean has a sum and a count, not a width of {len(totals)}")
    total, count = totals
    if count < 1:
        raise ValueError(f"a mean needs a count of at least 1, not {count}")
    return total / (count * MICRO)
