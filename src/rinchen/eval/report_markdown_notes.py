"""The robustness and not-measured sections of the Markdown report of D13 (spec T14), sections of
`rinchen.eval.report_markdown`: D12's block of each system with the units each value rests on and
where it came from, and the list of what was not measured (the metrics without a scorer, the
directions a suite lacks or a system did not run, the share of outputs in the wrong language)."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any

from rinchen.eval.report_format import ABSENT_STATE
from rinchen.eval.report_markdown_cells import inline, plural, short, table
from rinchen.eval.robustness_input import BLOCK_KEYS
from rinchen.metrics.registry import METRIC_NAMES

__all__ = ["not_measured_section", "robustness_section"]


def robustness_section(report: Mapping[str, Any]) -> list[str]:
    lines = ["## Robustness", ""]
    if not report["robustness"]:
        return [*lines, "No robustness run was given.", ""]
    lines += [
        (
            "A value is a share of the units that were scored: read it with its counts. The "
            "retention is a ratio of two chrF scores (see report.json). The kind of the "
            "robustness suite is not recorded in D12's block, so the evidence banner above "
            "does not count it."
        ),
        "",
    ]
    for system, entry in sorted(report["robustness"].items()):
        rows = [
            [
                key,
                _robustness_value(entry["block"][key]),
                str(entry["coverage"][key]["scored"]),
                str(entry["coverage"][key]["excluded"]),
                str(entry["coverage"][key]["total"]),
            ]
            for key in BLOCK_KEYS
        ]
        source = entry["source"]
        binding = "its model digest" if source["binding"] == "model_digest" else "name"
        lines += [
            "### " + system,
            "",
            *table(["value", "result", "scored", "excluded", "total"], rows),
            (
                f"Measured on the suite {source['suite']} ({short(source['suite_sha256'])}, "
                f"suite kind not recorded), bound to the system by {binding}."
            ),
            "",
        ]
    return lines


def _robustness_value(value: float | None) -> str:
    return "n/a (no scored unit)" if value is None else format(value, ".4f")


def not_measured_section(report: Mapping[str, Any]) -> list[str]:
    lines = ["## Not measured", ""]
    results = report["results"]
    for metric in METRIC_NAMES:
        reasons = Counter(
            r["unmeasured"][metric] for r in results if metric in r["unmeasured"]
        )
        for reason, count in sorted(reasons.items()):
            lines.append(
                f"- {metric}: {inline(reason)} ({count} of {len(results)} results)"
            )
    for suite in report["suites"]:
        missing = sum(entry["rows"] is None for entry in suite["directions"])
        if missing:
            verb = "is" if missing == 1 else "are"
            lines.append(
                f"- {suite['name']}: {plural(missing, 'expected direction')} {verb} "
                "not in the suite."
            )
    lines += _directions_that_did_not_run(report["coverage"])
    lines += [
        "- The share of outputs in the wrong language: no language identification is run.",
        "",
    ]
    return lines


def _directions_that_did_not_run(coverage: Sequence[Mapping[str, Any]]) -> list[str]:
    states: dict[tuple[str, str], Counter[str]] = {}
    for row in coverage:
        if row["state"] not in ("run", ABSENT_STATE):
            states.setdefault((row["suite"], row["system"]), Counter())[
                row["state"]
            ] += 1
    return [
        f"- {system} on {suite}: {plural(sum(counts.values()), 'direction')} did not run "
        f"({', '.join(f'{state}: {count}' for state, count in sorted(counts.items()))})."
        for (suite, system), counts in sorted(states.items())
    ]
