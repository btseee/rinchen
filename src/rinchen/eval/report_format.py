"""The format of the report of D13 (spec T14), `rinchen.report/1`: what the builder
(`rinchen.eval.report`), the Markdown (`rinchen.eval.report_markdown`) and the strict reader
(`rinchen.eval.report_read`) share, so that none of them needs another.

The report is one JSON document with these sections, in this order:

    schema, evidence, systems, suites, runs, coverage, results, robustness, comparisons, targets,
    bootstrap, note_multiple_comparisons

`evidence` says which kinds of suite the figures come from; `systems` and `suites` describe what was
run and over what (a suite lists a row for every direction the report expects, with the number of
rows it has or null when the suite does not declare it); `runs` names each run by suite, system and
the digest of its `run.json`; `coverage` says what became of every direction of every system (a
state, and why when it did not run); `results` are the results of the directions that ran without
their robustness fields; `robustness` holds, for each system that carries D12's block, the block
WITH its coverage and its source (never the block alone); `comparisons` are the paired comparisons
against the baseline, recomputed from the suite and the outputs; `targets` are the release targets
as measurements; `bootstrap` says what the intervals are and `note_multiple_comparisons` that none
is corrected.

The bytes of a report are a function of its inputs: sorted keys, two-space indent, one final LF, no
timestamp, and no absolute path in any string (a report can be uploaded; it names no local file).
`require_no_absolute_path` is the check the builder and the reader both make.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any, Final

from rinchen.eval.field_rules import document_bytes, problem

__all__ = [
    "ABSENT_REASON",
    "ABSENT_STATE",
    "MULTIPLE_COMPARISONS_NOTE",
    "REPORT_SCHEMA",
    "SECTIONS",
    "count_intervals",
    "evidence_label",
    "multiple_comparisons_note",
    "report_to_bytes",
    "require_no_absolute_path",
]

REPORT_SCHEMA: Final = "rinchen.report/1"
# The sections of the format, `schema` first; the order they are checked and named in.
SECTIONS: Final = (
    "schema",
    "evidence",
    "systems",
    "suites",
    "runs",
    "coverage",
    "results",
    "robustness",
    "comparisons",
    "targets",
    "bootstrap",
    "note_multiple_comparisons",
)
# What the coverage says of the baseline in a suite where it has no run.
ABSENT_STATE: Final = "absent"
ABSENT_REASON: Final = "baseline not in this suite"
MULTIPLE_COMPARISONS_NOTE: Final = (
    "{intervals} intervals and {comparisons} paired comparisons are reported with no "
    "correction for multiple comparisons"
)
# A POSIX path from a root, a UNC path or a drive path, at the start of a string or after a space,
# a quote, a bracket, `=`, `,` or `;` (a slash inside a word, `and/or`, `12/16` or an address
# `http://host/x`, is not a local path).
_ABSOLUTE_PATH: Final = re.compile(
    r"(?:^|[\s\"'(\[=,;])(?:/[A-Za-z0-9._~-]|\\\\|[A-Za-z]:[\\/])"
)


def report_to_bytes(report: Mapping[str, Any]) -> bytes:
    """The UTF-8 bytes of a report: sorted keys, two-space indent, text as it is, one final LF.

    Raises:
        ValueError: a value is not a finite number (JSON has no NaN).
    """
    return document_bytes(report)


def multiple_comparisons_note(intervals: int, comparisons: int) -> str:
    """The sentence that says how many intervals and paired comparisons the report holds and
    that none is corrected for multiple comparisons."""
    return MULTIPLE_COMPARISONS_NOTE.format(
        intervals=intervals, comparisons=comparisons
    )


def count_intervals(
    results: Sequence[Mapping[str, Any]], comparisons: Sequence[Mapping[str, Any]]
) -> int:
    """The intervals a report holds: every interval of every result (the metrics that have none
    have `null`) and of every comparison."""
    in_results = sum(
        interval is not None
        for result in results
        for interval in result["ci95"].values()
    )
    return in_results + sum(
        comparison["ci95"] is not None for comparison in comparisons
    )


def evidence_label(kinds: Sequence[str]) -> str:
    """`fixture` when every suite is generated, `measured` when none is and `mixed` otherwise."""
    if "fixture" not in kinds:
        return "measured"
    return "fixture" if all(kind == "fixture" for kind in kinds) else "mixed"


def require_no_absolute_path(report: Mapping[str, Any]) -> None:
    """No string of the report, a key included, is an absolute local path.

    Raises:
        ValueError: a section holds one; the message names the section and the index of the
            entry (`coverage.3`) and never the string.
    """
    for section in SECTIONS:
        value = report.get(section)
        if isinstance(value, list):
            for index, entry in enumerate(value):
                if _holds_a_path(entry):
                    raise _path_problem(f"{section}.{index}")
        elif _holds_a_path(value):
            raise _path_problem(section)


def _path_problem(field: str) -> ValueError:
    return problem(field, "holds an absolute path: a report names no local file")


def _holds_a_path(node: object) -> bool:
    if isinstance(node, str):
        return _ABSOLUTE_PATH.search(node) is not None
    if isinstance(node, Mapping):
        return any(
            _holds_a_path(key) or _holds_a_path(value) for key, value in node.items()
        )
    if isinstance(node, list | tuple):
        return any(_holds_a_path(item) for item in node)
    return False
