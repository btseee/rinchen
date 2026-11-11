"""The score tables and the paired comparisons of the Markdown report of D13 (spec T14), sections of
`rinchen.eval.report_markdown`: for each suite and metric a table with a row for every direction the
report expects and, for every system, a column of scores, each cell `value [low, high]` or `n/a
(<state or reason>)`, and a column `<system> ok` with the outputs that were `ok` (`10 of 10`, or `6
of 10 (1 empty, 3 truncated)`): a failure the system flagged stays beside the metric. And for each
suite with a baseline its comparisons with the delta, the interval, the p-value and the verdict in
words (`better`, `worse` or `no significant difference`)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Final

from rinchen.eval.report_format import ABSENT_STATE
from rinchen.eval.report_markdown_cells import (
    TABLE_ORDER,
    label,
    reason_text,
    status_cell,
    table,
    verdict,
)
from rinchen.metrics.registry import COUNT_METRICS, HIGHER_IS_BETTER

__all__ = ["comparisons_section", "scores_section"]

_NOT_IN_THE_SUITE: Final = "n/a (not in the suite)"


def scores_section(report: Mapping[str, Any]) -> list[str]:
    lines = [
        "## Scores",
        "",
        (
            "A cell is the corpus score with its interval, `value [low, high]`; the column beside "
            "it counts the outputs that were `ok` and names the others. A system that did not run "
            "a direction has `n/a` and the state; a direction the suite does not declare is `n/a "
            "(not in the suite)`."
        ),
        "",
    ]
    for suite in report["suites"]:
        lines += _scores_of(report, suite)
    return lines


def _systems_of(report: Mapping[str, Any], suite: str) -> list[str]:
    return sorted({run["system"] for run in report["runs"] if run["suite"] == suite})


def _scores_of(report: Mapping[str, Any], suite: Mapping[str, Any]) -> list[str]:
    name = suite["name"]
    results = {
        (r["system"], r["direction"]): r
        for r in report["results"]
        if r["suite"] == name
    }
    states = {
        (c["system"], c["direction"]): c["state"]
        for c in report["coverage"]
        if c["suite"] == name
    }
    systems = _systems_of(report, name)
    lines = ["### " + name, ""]
    unit = _unit_line(results.values())
    if unit:
        lines += [unit, ""]
    for metric in TABLE_ORDER:
        if metric not in COUNT_METRICS and not any(
            r["metrics"][metric] is not None for r in results.values()
        ):
            continue
        suffix = "" if HIGHER_IS_BETTER[metric] else " (lower is better)"
        rows = [
            [
                entry["direction"],
                *(
                    cell
                    for system in systems
                    for cell in (
                        _score_cell(
                            metric,
                            results.get((system, entry["direction"])),
                            states.get((system, entry["direction"])),
                            entry["rows"] is None,
                        ),
                        _status_cell(results.get((system, entry["direction"]))),
                    )
                ),
            ]
            for entry in suite["directions"]
        ]
        header = ["direction", *(c for s in systems for c in (s, f"{s} ok"))]
        lines += [f"#### {label(metric)}{suffix}", ""]
        lines += table(header, rows)
    return lines


def _unit_line(results: Any) -> str:
    units = {(r["bootstrap"]["unit"], r["bootstrap"]["units"]) for r in results}
    if not units:
        return ""
    if len(units) > 1:
        return "Resampling unit: mixed (see report.json)."
    ((unit, count),) = units
    return f"Resampling unit: {unit} ({count} units in each direction)."


def _score_cell(
    metric: str,
    result: Mapping[str, Any] | None,
    state: str | None,
    undeclared: bool,
) -> str:
    if undeclared:
        return _NOT_IN_THE_SUITE
    if result is None:
        return f"n/a ({state or 'no run'})"
    value = result["metrics"][metric]
    if value is None:
        return f"n/a ({reason_text(result['unmeasured'][metric])})"
    interval = result["ci95"][metric]
    spread = "[n/a]" if interval is None else f"[{interval[0]:.2f}, {interval[1]:.2f}]"
    return f"{value:.2f} {spread}"


def _status_cell(result: Mapping[str, Any] | None) -> str:
    if result is None:
        return "n/a"
    return status_cell(result["statuses"], result["n_segments"])


def comparisons_section(report: Mapping[str, Any]) -> list[str]:
    note = report["note_multiple_comparisons"]
    lines = ["## Paired comparisons", "", f"{note}.", ""]
    sections: list[str] = []
    alpha = report["bootstrap"]["alpha"]
    for suite in report["suites"]:
        name = suite["name"]
        held = [c for c in report["comparisons"] if c["suite"] == name]
        absent = [
            c
            for c in report["coverage"]
            if c["suite"] == name and c["state"] == ABSENT_STATE
        ]
        if held:
            sections += _comparison_table(name, held, alpha)
        elif absent:
            sections += [
                "### " + name,
                "",
                (
                    f"The baseline {absent[0]['system']} has no run on this suite: nothing is "
                    "compared."
                ),
                "",
            ]
    if not sections:
        sections = [
            (
                "No paired comparison was made: no baseline was named, or no other system ran "
                "with it."
            ),
            "",
        ]
    return lines + sections


def _comparison_table(
    suite: str, comparisons: Sequence[Mapping[str, Any]], alpha: float
) -> list[str]:
    rows = [
        [
            c["direction"],
            c["system"],
            label(c["metric"]),
            f"{c['delta']:+.2f}",
            _interval(c["ci95"]),
            "n/a" if c["p_value"] is None else f"{c['p_value']:.3f}",
            verdict(c["p_value"], c["delta"], c["higher_is_better"], alpha),
        ]
        for c in comparisons
    ]
    header = ["direction", "system", "metric", "delta", "interval", "p", "verdict"]
    shown = format(alpha, "g")
    return [
        "### " + suite,
        "",
        (
            f"Baseline: {comparisons[0]['baseline']}. The verdict is the p-value's, not the "
            f"interval's: `better` is a p-value below {shown} with the difference in the "
            f"system's favour, `worse` is a p-value below {shown} with the difference against "
            "it, and `no significant difference` is anything else."
        ),
        "",
        *table(header, rows),
    ]


def _interval(interval: Sequence[float] | None) -> str:
    if interval is None:
        return "n/a"
    return f"[{interval[0]:+.2f}, {interval[1]:+.2f}]"
