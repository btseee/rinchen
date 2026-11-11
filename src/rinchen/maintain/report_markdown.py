"""The Markdown summary of a maintenance report (D14, spec M11): what the nightly job copies to its
run page, rendered from the report alone.

The summary is a heading with the overall state and the cadence, then blocks separated by one
blank line: the table of checks (one row per doctor check and one per other step), the drift
section (the step's state and reason, the table of at most `MAX_DRIFT_ROWS` entries and a line that
counts the entries left in `maintain_report.json`, the sentence of the blind spot), the table of
the parity reruns and, for each failing step, one `What to do` line that names the section of the
runbook. It holds line feeds only and ends in one. Every cell is escaped: a line break becomes a
space and a backslash, a pipe and a backtick are escaped, so that nothing can break a table.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any, Final

from rinchen.maintain.report import Report, Step, overall_state

MAX_DRIFT_ROWS: Final = 200
RUNBOOK: Final = "docs/runbooks/14_maintenance.md"
BLIND_SPOT: Final = (
    "The drift table is what uv could take within the constraints of pyproject.toml; "
    f"a package pinned with == never appears in it ({RUNBOOK})."
)

_LINE_BREAKS: Final = re.compile(r"[\r\n]+")
_NONE: Final = "-"


def _cell(text: object) -> str:
    flat = _LINE_BREAKS.sub(" ", str(text))
    return flat.replace("\\", "\\\\").replace("|", "\\|").replace("`", "\\`")


def _table(header: Sequence[str], rows: Sequence[Sequence[object]]) -> str:
    lines = [header, ["---"] * len(header), *rows]
    return "\n".join(
        "| " + " | ".join(_cell(cell) for cell in row) + " |" for row in lines
    )


def _step(report: Report, step_id: str) -> Step:
    return next(step for step in report.steps if step.id == step_id)


def _or_none(value: object) -> object:
    return _NONE if value is None else value


def _checks_section(report: Report) -> str:
    rows: list[list[object]] = []
    for step in report.steps:
        checks = step.detail.get("checks") if step.id == "doctor" else None
        if checks:
            rows += [
                [step.id, check["id"], check["state"], check["reason"] or ""]
                for check in checks
            ]
        else:
            rows.append([step.id, _NONE, step.state, step.reason or ""])
    return "## Checks\n\n" + _table(["step", "check", "state", "reason"], rows)


def _state_line(step: Step) -> str:
    reason = "" if step.reason is None else f" Reason: {_cell(step.reason)}."
    return f"State: {step.state}.{reason}"


def _count_summary(entries: Sequence[Mapping[str, Any]]) -> str:
    changes = Counter(entry["change"] for entry in entries)
    classes = Counter(entry["class"] for entry in entries)
    return (
        f"{len(entries)} entries ({changes['update']} update, {changes['add']} add, "
        f"{changes['remove']} remove; {classes['drift']} drift, {classes['blocked']} blocked)"
    )


def _drift_rows(entries: Sequence[Mapping[str, Any]]) -> list[list[object]]:
    return [
        [
            entry["package"],
            entry["change"],
            _or_none(entry["locked"]),
            _or_none(entry["available"]),
            entry["class"],
            ", ".join(entry["requires"]) or _NONE,
        ]
        for entry in entries[:MAX_DRIFT_ROWS]
    ]


def _drift_section(step: Step) -> str:
    entries: Sequence[Mapping[str, Any]] = step.detail.get("entries", [])
    blocks = ["## Dependency drift"]
    if entries:
        blocks.append(f"State: {step.state}. {_count_summary(entries)}.")
        header = ["package", "change", "locked", "available", "class", "needs"]
        blocks.append(_table(header, _drift_rows(entries)))
        if len(entries) > MAX_DRIFT_ROWS:
            more = len(entries) - MAX_DRIFT_ROWS
            blocks.append(f"{more} more entries are in maintain_report.json")
    else:
        blocks.append(_state_line(step))
    blocks.append(BLIND_SPOT)
    return "\n\n".join(blocks)


def _rerun_row(item: Mapping[str, Any]) -> list[object]:
    tests = item["tests"]
    counters = (
        [_NONE] * 3
        if tests is None
        else [tests["passed"], tests["failed"], tests["skipped"]]
    )
    return [item["id"], item["state"], *counters, item["reason"] or ""]


def _reruns_section(step: Step) -> str:
    items: Sequence[Mapping[str, Any]] = step.detail.get("items", [])
    heading = "## Parity reruns"
    if not items:
        return f"{heading}\n\n{_state_line(step)}"
    header = ["id", "state", "passed", "failed", "skipped", "reason"]
    return f"{heading}\n\n" + _table(header, [_rerun_row(item) for item in items])


def _failures_section(failed: Sequence[str]) -> str:
    lines = [
        f"- What to do ({step_id}): {RUNBOOK}#failure-{step_id}" for step_id in failed
    ]
    return "## Failures\n\n" + "\n".join(lines)


def render_markdown(report: Report) -> str:
    """The summary of `report`: line feeds only and one final line feed."""
    blocks = [
        f"# Maintenance check: {overall_state(report)} ({report.cadence} cadence)",
        _checks_section(report),
        _drift_section(_step(report, "drift")),
        _reruns_section(_step(report, "reruns")),
    ]
    failed = [step.id for step in report.steps if step.state == "fail"]
    if failed:
        blocks.append(_failures_section(failed))
    return "\n\n".join(blocks) + "\n"
