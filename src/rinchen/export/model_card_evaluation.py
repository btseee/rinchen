"""The sections of the model card that carry figures (D13, spec T16): the evaluation of the system, its
paired comparisons, the robustness block and the release targets. Each is a pure function of the parsed
report (`rinchen.eval.report_read.ReportView`) and returns lines of Markdown.

The rules of the card are kept here. A table has a row for each direction the system was MEASURED in
and for no other; every other direction the suite expects or the model was trained for is named in a
line `n/a (<reason>)`, the reason being `not in the suite`, `not trained` or the state the report
gives; a metric that was not measured is `n/a (<reason>)` with the reason the result carries, never a
blank or a zero; a number is one of the report's, printed with a fixed format; a comparison row always
has its p-value (or why there is none) and the verdict, which is the p-value's; nothing is said of a
system that is not the one the card is about.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from rinchen.eval.report_markdown_cells import (
    FEWER_THAN_TWO_UNITS,
    LABELS,
    TABLE_ORDER,
    inline,
    label,
    number,
    plural,
    short,
    source_cells,
    status_cell,
    table,
    verdict,
)
from rinchen.eval.report_read import ReportView
from rinchen.eval.report_view import ComparisonView, RobustnessView, SuiteView
from rinchen.eval.result import DirectionResult
from rinchen.eval.robustness_input import BLOCK_KEYS
from rinchen.export.record import ModelRecord
from rinchen.metrics.registry import COUNT_METRICS, SEGMENT_SCORE_METRICS

__all__ = [
    "comparisons_section",
    "evaluation_section",
    "robustness_section",
    "targets_section",
    "unscored_neural_metrics",
]

_NOT_IN_THE_SUITE = "not in the suite"
_NOT_TRAINED = "not trained"
_NO_RUN = "no run"
_FEWER_THAN_TWO_UNITS = FEWER_THAN_TWO_UNITS
# The count metrics in the order of the report's tables: the headline first.
_COUNT_COLUMNS = tuple(metric for metric in TABLE_ORDER if metric in COUNT_METRICS)
# The reasons a direction is listed under, in the order the card lists them; any other (a state of
# the report) follows, sorted.
_REASON_ORDER = (_NOT_IN_THE_SUITE, _NOT_TRAINED)


def unscored_neural_metrics(results: Sequence[DirectionResult]) -> list[str]:
    """The metrics that need segment scores and have no value in any of `results`, in the
    registry's order."""
    return [
        metric
        for metric in SEGMENT_SCORE_METRICS
        if all(result.metrics[metric] is None for result in results)
    ]


def evaluation_section(
    record: ModelRecord,
    report: ReportView,
    system: str,
    results: Sequence[DirectionResult],
) -> list[str]:
    """The evaluation of `system`: how the figures were made, then a block for each suite in
    which the system has a run, then the suites in which it has none."""
    lines = ["## Evaluation", "", *_how_it_was_measured(report, system), ""]
    ran_in = {run.suite for run in report.runs if run.system == system}
    for suite in report.suites:
        if suite.name in ran_in:
            of_suite = [r for r in results if r.suite == suite.name]
            lines += _suite_block(record, report, system, suite, of_suite)
    without_a_run = [suite.name for suite in report.suites if suite.name not in ran_in]
    if without_a_run:
        lines += [
            (
                "Suites of the report in which this system has no run: "
                f"{', '.join(without_a_run)}."
            ),
            "",
        ]
    return [*lines, *comparisons_section(report, system)]


def _how_it_was_measured(report: ReportView, system: str) -> list[str]:
    entry = next(entry for entry in report.systems if entry.name == system)
    options = (
        ", ".join(
            f"{name}={_plain(value)}" for name, value in sorted(entry.options.items())
        )
        if entry.options
        else "n/a (none recorded)"
    )
    bootstrap = report.bootstrap
    level = format(round((1 - bootstrap.alpha) * 100, 6), "g")
    return [
        (
            f"The figures are those of the system `{system}` in `report.json`: every number "
            "below is a number of that report."
        ),
        "",
        f"- Decoding settings of the run: {options}.",
        "- A cell is a corpus score with its interval: `value [low, high]`.",
        (
            f"- Intervals: {bootstrap.resamples:,} resamples, seed {bootstrap.seed}, "
            f"{bootstrap.interval} {level} %."
        ),
    ]


def _plain(value: object) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


# ---- one suite ----------------------------------------------------------------------------------------


def _suite_block(
    record: ModelRecord,
    report: ReportView,
    system: str,
    suite: SuiteView,
    results: Sequence[DirectionResult],
) -> list[str]:
    by_direction = {result.direction: result for result in results}
    dataset, revision = source_cells(
        suite.source.dataset, suite.source.revision, suite.source.pinned
    )
    lines = [
        f"### Suite {suite.name}",
        "",
        f"- kind: {suite.kind}",
        f"- split: {suite.split}",
        f"- rows: {suite.rows}",
        f"- dataset: {dataset}",
        f"- revision: {revision}",
        *_signature_lines(results),
        "",
    ]
    if by_direction:
        columns = [*_COUNT_COLUMNS, *_scored_neural(results)]
        rows = [
            [
                suite_direction.direction,
                str(by_direction[suite_direction.direction].n_segments),
                _outputs_ok(by_direction[suite_direction.direction]),
                _units(by_direction[suite_direction.direction]),
                *(
                    _cell(by_direction[suite_direction.direction], metric)
                    for metric in columns
                ),
            ]
            for suite_direction in suite.directions
            if suite_direction.direction in by_direction
        ]
        header = [
            "direction",
            "segments",
            "outputs ok",
            "resampling units",
            *(label(m) for m in columns),
        ]
        lines += table(header, rows)
    for reason, directions in _not_measured(
        record, report, system, suite, by_direction
    ):
        lines += [f"n/a ({reason}): {', '.join(directions)}.", ""]
    lines += _unscored_lines(results)
    return lines


def _scored_neural(results: Sequence[DirectionResult]) -> list[str]:
    missing = unscored_neural_metrics(results)
    return [metric for metric in SEGMENT_SCORE_METRICS if metric not in missing]


def _signature_lines(results: Sequence[DirectionResult]) -> list[str]:
    lines: list[str] = []
    for metric, field in (
        ("bleu", "bleu_signature"),
        ("chrf", "chrf_signature"),
        ("chrf_pp", "chrf_pp_signature"),
    ):
        signatures = sorted({getattr(result, field) for result in results})
        if signatures:
            shown = ", ".join(f"`{signature}`" for signature in signatures)
            lines.append(f"- {label(metric)} signature: {shown}")
    return lines


def _outputs_ok(result: DirectionResult) -> str:
    return status_cell(result.statuses, result.n_segments)


def _units(result: DirectionResult) -> str:
    return plural(result.bootstrap.units, f"{result.bootstrap.unit} unit")


def _cell(result: DirectionResult, metric: str) -> str:
    value = result.metrics[metric]
    if value is None:
        return f"n/a ({inline(result.unmeasured[metric])})"
    interval = result.ci95[metric]
    spread = (
        f"[{interval[0]:.2f}, {interval[1]:.2f}]"
        if interval is not None
        else f"[{_FEWER_THAN_TWO_UNITS}]"
    )
    return f"{value:.2f} {spread}"


def _not_measured(
    record: ModelRecord,
    report: ReportView,
    system: str,
    suite: SuiteView,
    measured: Mapping[str, DirectionResult],
) -> list[tuple[str, list[str]]]:
    """The directions of the suite and of the model that were not measured, grouped by the reason,
    each direction once in the order the suite lists them and then the model's own."""
    trained = set(record.directions)
    declared = {entry.direction for entry in suite.directions if entry.rows is not None}
    states = {
        row.direction: row.state
        for row in report.coverage
        if row.suite == suite.name and row.system == system
    }
    listed = [entry.direction for entry in suite.directions]
    expected = [*listed, *(d for d in record.directions if d not in listed)]
    groups: dict[str, list[str]] = {}
    for direction in expected:
        if direction in measured:
            continue
        if direction not in trained:
            reason = _NOT_TRAINED
        elif direction not in declared:
            reason = _NOT_IN_THE_SUITE
        else:
            reason = states.get(direction, _NO_RUN).replace("_", " ")
        groups.setdefault(reason, []).append(direction)
    first = [
        (reason, groups.pop(reason)) for reason in _REASON_ORDER if reason in groups
    ]
    return [*first, *sorted(groups.items())]


def _unscored_lines(results: Sequence[DirectionResult]) -> list[str]:
    """`<metrics>: n/a (<reason>).` for the metrics that need segment scores and have none, the
    metrics that share a reason on one line."""
    by_reasons: dict[tuple[str, ...], list[str]] = {}
    for metric in unscored_neural_metrics(results):
        reasons = tuple(sorted({inline(r.unmeasured[metric]) for r in results}))
        by_reasons.setdefault(reasons, []).append(LABELS[metric])
    return [
        line
        for reasons, labels in by_reasons.items()
        for line in (f"{', '.join(labels)}: n/a ({'; '.join(reasons)}).", "")
    ]


# ---- paired comparisons ---------------------------------------------------------------------------------


def comparisons_section(report: ReportView, system: str) -> list[str]:
    """The comparisons in which `system` is the candidate, with the verdict of each."""
    mine = [c for c in report.comparisons if c.system == system]
    lines = ["### Paired comparisons", ""]
    if not mine:
        return [
            *lines,
            "n/a (no comparison of the report has this system as the candidate)",
            "",
        ]
    alpha = format(report.bootstrap.alpha, "g")
    header = [
        "suite",
        "direction",
        "metric",
        "baseline",
        "delta",
        "interval",
        "p",
        "verdict",
    ]
    return [
        *lines,
        (
            f"{report.note_multiple_comparisons}. The verdict compares this system with the "
            "baseline and is the p-value's, not the interval's: `better` is a p-value below "
            f"{alpha} with the difference in this system's favour, `worse` is a p-value below "
            f"{alpha} with the difference against it, and `no significant difference` is "
            "anything else."
        ),
        "",
        *table(
            header,
            [
                _comparison_row(comparison, report.bootstrap.alpha)
                for comparison in mine
            ],
        ),
    ]


def _comparison_row(comparison: ComparisonView, alpha: float) -> list[str]:
    interval = comparison.ci95
    return [
        comparison.suite,
        comparison.direction,
        label(comparison.metric),
        comparison.baseline,
        f"{comparison.delta:+.2f}",
        _FEWER_THAN_TWO_UNITS
        if interval is None
        else f"[{interval[0]:+.2f}, {interval[1]:+.2f}]",
        _FEWER_THAN_TWO_UNITS
        if comparison.p_value is None
        else f"{comparison.p_value:.3f}",
        verdict(
            comparison.p_value,
            comparison.delta,
            comparison.higher_is_better,
            alpha,
        ),
    ]


# ---- robustness -----------------------------------------------------------------------------------------


def robustness_section(report: ReportView, system: str) -> list[str]:
    """D12's six values of `system` with the units each rests on and where they were measured, or
    the statement that no run was attached."""
    lines = ["## Robustness", ""]
    entry: RobustnessView | None = report.robustness.get(system)
    if entry is None:
        return [*lines, "n/a (no robustness run was attached)", ""]
    source = entry.source
    binding = (
        "its model digest" if source["binding"] == "model_digest" else "its name only"
    )
    rows = [
        [
            key,
            _robustness_value(entry.block[key]),
            f"{entry.coverage[key]['scored']} of {entry.coverage[key]['total']}",
        ]
        for key in BLOCK_KEYS
    ]
    return [
        *lines,
        (
            f"Measured on the robustness suite `{source['suite']}` (sha256 "
            f"{short(source['suite_sha256'])}, suite kind not recorded), not on the suites "
            "above; bound to the system "
            f"by {binding}. A value is a share of the units that were scored: read it with "
            "its counts."
        ),
        "",
        *table(["value", "result", "scored of total"], rows),
    ]


def _robustness_value(value: float | None) -> str:
    return "n/a (no scored unit)" if value is None else format(value, ".4f")


# ---- release targets ------------------------------------------------------------------------------------


def targets_section(report: ReportView) -> list[str]:
    """The release targets as the report measured them."""
    rows = [
        [
            target.id,
            target.status,
            "n/a (no threshold)"
            if target.threshold is None
            else format(target.threshold, "g"),
            _target_value(target.value, target.basis),
        ]
        for target in report.targets
    ]
    return [
        "## Release targets",
        "",
        "A target is a measurement: no command fails because one is not met.",
        "",
        *table(["target", "status", "threshold", "value"], rows),
    ]


def _target_value(value: float | None, basis: str) -> str:
    if value is None:
        return f"n/a ({inline(basis)})"
    return f"{number(value)} ({inline(basis)})"
