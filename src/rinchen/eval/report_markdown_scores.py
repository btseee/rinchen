"""The score tables and the paired comparisons of the Markdown report of D13 (spec T14), sections of
`rinchen.eval.report_markdown`: for each suite and metric a table with a row for every direction the
report expects and a column for every system, each cell `value [low, high]` or `n/a (<state or
reason>)`; and for each suite with a baseline its comparisons with the delta, the interval, the
p-value and a star where the verdict is significant."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Final

from rinchen.eval.report_format import ABSENT_STATE
from rinchen.eval.report_markdown_cells import (
    TABLE_ORDER,
    inline,
    label,
    table,
)
from rinchen.metrics.registry import COUNT_METRICS, HIGHER_IS_BETTER

__all__ = ["comparisons_section", "scores_section"]

_NOT_IN_THE_SUITE: Final = "n/a (not in the suite)"


def scores_section(report: Mapping[str, Any]) -> list[str]:
    lines = [
        "## Scores",
        "",
        (
            "A cell is the corpus score with its interval, `value [low, high]`. A system that did "
            "not run a direction has `n/a` and the state; a direction the suite does not declare "
            "is `n/a (not in the suite)`."
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
                    _score_cell(
                        metric,
                        results.get((system, entry["direction"])),
                        states.get((system, entry["direction"])),
                        entry["rows"] is None,
                    )
                    for system in systems
                ),
            ]
            for entry in suite["directions"]
        ]
        lines += [f"#### {label(metric)}{suffix}", ""]
        lines += table(["direction", *systems], rows)
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
        return f"n/a ({inline(result['unmeasured'][metric])})"
    interval = result["ci95"][metric]
    spread = "[n/a]" if interval is None else f"[{interval[0]:.2f}, {interval[1]:.2f}]"
    return f"{value:.2f} {spread}"


def comparisons_section(report: Mapping[str, Any]) -> list[str]:
    note = report["note_multiple_comparisons"]
    lines = ["## Paired comparisons", "", f"{note}.", ""]
    sections: list[str] = []
    alpha = format(report["bootstrap"]["alpha"], "g")
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
    suite: str, comparisons: Sequence[Mapping[str, Any]], alpha: str
) -> list[str]:
    rows = [
        [
            c["direction"],
            c["system"],
            label(c["metric"]),
            f"{c['delta']:+.2f}",
            _interval(c["ci95"]),
            "n/a" if c["p_value"] is None else f"{c['p_value']:.3f}",
            "*" if c["significant"] else "",
        ]
        for c in comparisons
    ]
    header = ["direction", "system", "metric", "delta", "interval", "p", "significant"]
    return [
        "### " + suite,
        "",
        (
            f"Baseline: {comparisons[0]['baseline']}. A star marks a system significantly "
            f"better than the baseline: a p-value below {alpha} with the difference in its "
            "favour."
        ),
        "",
        *table(header, rows),
    ]


def _interval(interval: Sequence[float] | None) -> str:
    if interval is None:
        return "n/a"
    return f"[{interval[0]:+.2f}, {interval[1]:+.2f}]"
