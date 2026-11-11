"""The Markdown report of D13 (spec T14): `render_markdown` is a PURE function of the parsed report,
so one document gives the same bytes in every process, locale and time zone (it reads nothing of
the environment, iterates only lists and sorted keys, and formats numbers with fixed formats).

The sections come in a fixed order: the title with the evidence banner, the systems, the suites (with
the dataset and the revision of each), the scores (a table for each suite and metric, a row for every
direction the report expects and, for every system, a column of scores and a column of the outputs
that were `ok`), the paired comparisons, the robustness block with its counts, what was not measured,
the release targets and the provenance. A score is `value [low, high]` with two decimals, a p-value
has three and a delta a sign; whatever was not measured is `n/a (<state or reason>)` and never a
number; a comparison has a verdict in words (`better`, `worse` or `no significant difference`). The
banner is a function of the kinds of suites the figures come from: a fixture is generated and says
nothing about translation quality. No line holds a timestamp or a path.

The score and comparison sections are in `report_markdown_scores`, the robustness and not-measured
sections in `report_markdown_notes` and the cells they share in `report_markdown_cells`.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any, Final

from rinchen.eval.report_markdown_cells import (
    inline,
    label,
    number,
    short,
    source_cells,
    table,
)
from rinchen.eval.report_markdown_notes import not_measured_section, robustness_section
from rinchen.eval.report_markdown_scores import comparisons_section, scores_section

__all__ = ["render_markdown"]

FIXTURE_BANNER: Final = (
    "These figures come from generated fixture suites. They test the tooling and say "
    "nothing about translation quality."
)


def render_markdown(report: Mapping[str, Any]) -> str:
    """The Markdown of `report`, the parsed `report.json`: sections in a fixed order, LF line
    ends, one final LF."""
    lines: list[str] = []
    for section in (
        _title,
        _systems,
        _suites,
        scores_section,
        comparisons_section,
        robustness_section,
        not_measured_section,
        _targets,
        _provenance,
    ):
        lines.extend(section(report))
    return "\n".join(lines).rstrip("\n") + "\n"


def _title(report: Mapping[str, Any]) -> list[str]:
    return ["# Evaluation report", "", _banner(report["evidence"]), ""]


def _banner(evidence: Mapping[str, Any]) -> str:
    if evidence["fixture_only"]:
        return FIXTURE_BANNER
    kinds = evidence["kinds"]
    noun = "kind" if len(kinds) == 1 else "kinds"
    opening = f"Evidence: suites of the {noun} {', '.join(kinds)}."
    if "fixture" in kinds:
        return (
            f"{opening} The suites of the kind fixture are generated: they test the "
            "tooling and say nothing about translation quality."
        )
    return (
        f"{opening} Each figure is a measurement of the system named, on the suite "
        "named, and says nothing about other text."
    )


def _systems(report: Mapping[str, Any]) -> list[str]:
    rows = [
        [
            system["name"],
            system["kind"],
            short(system["model_sha256"]),
            short(system["manifest_sha256"]),
            _options(system["options"]),
        ]
        for system in report["systems"]
    ]
    header = ["system", "kind", "model", "manifest", "options"]
    return ["## Systems", "", *table(header, rows)]


def _options(options: Mapping[str, Any]) -> str:
    if not options:
        return "none"
    return inline(
        ", ".join(
            f"{name}={value if isinstance(value, str) else json.dumps(value)}"
            for name, value in sorted(options.items())
        )
    )


def _declared(suite: Mapping[str, Any]) -> int:
    return sum(entry["rows"] is not None for entry in suite["directions"])


def _suites(report: Mapping[str, Any]) -> list[str]:
    rows = [
        [
            suite["name"],
            suite["kind"],
            suite["benchmark"],
            suite["split"],
            suite["purpose"],
            str(suite["rows"]),
            str(_declared(suite)),
            *source_cells(
                suite["source"]["dataset"],
                suite["source"]["revision"],
                suite["source"]["pinned"],
            ),
            short(suite["sha256"]),
        ]
        for suite in report["suites"]
    ]
    header = [
        "suite",
        "kind",
        "benchmark",
        "split",
        "purpose",
        "rows",
        "directions",
        "dataset",
        "revision",
        "sha256",
    ]
    lines = ["## Suites", "", *table(header, rows)]
    for suite in report["suites"]:
        missing = suite["source"]["missing_languages"]
        if missing:
            lines += [f"The suite {suite['name']} lacks {', '.join(missing)}.", ""]
    return lines


def _targets(report: Mapping[str, Any]) -> list[str]:
    rows = [
        [
            target["id"],
            target["status"],
            number(target["value"]),
            "n/a" if target["threshold"] is None else format(target["threshold"], "g"),
            inline(target["basis"]),
        ]
        for target in report["targets"]
    ]
    return [
        "## Release targets",
        "",
        "A target is a measurement: no command fails because one is not met.",
        "",
        *table(["target", "status", "value", "threshold", "basis"], rows),
    ]


def _provenance(report: Mapping[str, Any]) -> list[str]:
    bootstrap = report["bootstrap"]
    level = format(1 - bootstrap["alpha"], "g")
    lines = [
        "## Provenance",
        "",
        (
            f"- Bootstrap: {bootstrap['resamples']} resamples from the seed {bootstrap['seed']}, "
            f"{bootstrap['interval']} intervals at the level {level} (alpha "
            f"{format(bootstrap['alpha'], 'g')}), draws {bootstrap['rng']}."
        ),
    ]
    for metric, signature in (
        ("bleu", "bleu_signature"),
        ("chrf", "chrf_signature"),
        ("chrf_pp", "chrf_pp_signature"),
    ):
        signatures = sorted({result[signature] for result in report["results"]})
        if signatures:
            lines.append(f"- {label(metric)} signature: {'; '.join(signatures)}")
    lines.append("")
    rows = [
        [run["suite"], run["system"], short(run["run_sha256"])]
        for run in report["runs"]
    ]
    return [*lines, *table(["suite", "system", "run.json"], rows)]
