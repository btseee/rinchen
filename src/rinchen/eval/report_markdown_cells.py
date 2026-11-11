"""The cells and tables of the Markdown report of D13 (spec T14), shared by the sections of
`rinchen.eval.report_markdown`: text as one line, a table row, a table with its separator, the first
twelve hex digits of a digest, the label of a metric, a value with its fixed format and a count with
its noun; and the three cells that keep a figure honest: the counts of the outputs that were not `ok`
(`status_cell`), the dataset and revision of a suite (`source_cells`) and the verdict of a comparison
in words (`verdict`). Every format is fixed, so that the same report renders to the same bytes
anywhere."""

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
    "source_cells",
    "status_cell",
    "table",
    "verdict",
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
NO_SOURCE_RECORD: Final = "n/a (no source record)"
UNPINNED: Final = "unpinned"
FEWER_THAN_TWO_UNITS: Final = "n/a (fewer than two units)"


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


def status_cell(statuses: Mapping[str, int], segments: int) -> str:
    """The outputs that were `ok` of the `segments` of a direction, `7 of 10`, and, when any was
    not, the others by name, `7 of 10 (1 empty, 2 truncated)`: a failure the system flagged stays
    beside the metric."""
    ok = statuses.get("ok", 0)
    others = ", ".join(
        f"{count} {status}"
        for status, count in sorted(statuses.items())
        if status != "ok"
    )
    return f"{ok} of {segments} ({others})" if others else f"{ok} of {segments}"


def source_cells(
    dataset: str | None, revision: str | None, pinned: bool
) -> tuple[str, str]:
    """The dataset and the revision a suite was built from: the dataset id and the revision, or
    `unpinned` when none was pinned (`<revision> (unpinned)` when one was recorded and not pinned),
    and `n/a (no source record)` for both when the suite has no dataset (a fixture, a human set or
    a local directory)."""
    if dataset is None:
        return NO_SOURCE_RECORD, NO_SOURCE_RECORD
    if revision is None:
        return dataset, UNPINNED
    return dataset, revision if pinned else f"{revision} ({UNPINNED})"


def verdict(
    p_value: float | None, delta: float, higher_is_better: bool, alpha: float
) -> str:
    """The verdict of a comparison of a system with its baseline, in words: `better` (a p-value
    below `alpha` with the difference in the system's favour), `worse` (below `alpha` with the
    difference against it) and `no significant difference` otherwise; `n/a (<reason>)` without a
    p-value. The p-value decides, never the interval."""
    if p_value is None:
        return FEWER_THAN_TWO_UNITS
    if p_value < alpha and delta != 0:
        return "better" if (delta > 0) == higher_is_better else "worse"
    return "no significant difference"
