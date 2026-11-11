"""The cells and tables of the Markdown report of D13 (spec T14), shared by the sections of
`rinchen.eval.report_markdown`: text as one line, a table row, a table with its separator, the first
twelve hex digits of a digest, the label of a metric, a value with its fixed format and a count with
its noun. Every format is fixed, so that the same report renders to the same bytes anywhere."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Final

from rinchen.metrics.registry import MetricName

__all__ = [
    "LABELS",
    "TABLE_ORDER",
    "inline",
    "label",
    "number",
    "plural",
    "row",
    "short",
    "table",
]

LABELS: Final[Mapping[str, str]] = {
    "chrf_pp": "chrF++",
    "chrf": "chrF",
    "bleu": "BLEU",
    "comet22": "COMET-22",
    "metricx24": "MetricX-24",
    "metricx24_qe": "MetricX-24-QE",
}
# The headline first, then the other count metrics, then the metrics that need segment scores.
TABLE_ORDER: Final[tuple[MetricName, ...]] = (
    "chrf_pp",
    "chrf",
    "bleu",
    "comet22",
    "metricx24",
    "metricx24_qe",
)
_DIGEST_SHOWN: Final = 12


def inline(text: str) -> str:
    """Free text for a cell or a line: one line, no table separator."""
    return " ".join(text.split()).replace("|", "\\|")


def row(cells: Sequence[str]) -> str:
    return "| " + " | ".join(cells) + " |"


def table(header: Sequence[str], rows: Sequence[Sequence[str]]) -> list[str]:
    return [
        row(header),
        row(["---"] * len(header)),
        *(row(cells) for cells in rows),
        "",
    ]


def short(digest: str | None) -> str:
    return "n/a" if digest is None else digest[:_DIGEST_SHOWN]


def label(metric: str) -> str:
    return LABELS[metric]


def number(value: Any) -> str:
    """A value of a target or a block: whole numbers as they are, others with four decimals."""
    if value is None:
        return "n/a"
    if isinstance(value, int):
        return str(value)
    return format(value, ".4f")


def plural(count: int, word: str) -> str:
    return f"{count} {word}" if count == 1 else f"{count} {word}s"
