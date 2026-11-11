"""The report documents of the release tests (D13, Task 9): the golden report of Task 8 with a
system of kind `rinchen` bound to an export, and the variants the card and the stage need (a
robustness block attached, suites of another kind, a direction that did not run, a system that
is a candidate in a comparison).

Everything starts from `tests.support.eval_report_golden.golden_report_document`, which is written
out from literals with nothing of `rinchen` in it; the run digests are those of the committed
`tests/fixtures/eval/golden_report.json`. A document is a plain dict: `report_bytes` writes it as
`rinchen eval report` does and `report_from_bytes` reads it back, so a variant that the strict
reader would refuse fails here and not in a test of the card.
"""

from __future__ import annotations

import copy
import json
from collections.abc import Mapping
from typing import Any, Final

from rinchen.eval.report_format import report_to_bytes
from rinchen.eval.report_markdown import render_markdown
from rinchen.eval.report_read import report_from_bytes
from rinchen.eval.report_view import ReportView
from tests.support.eval_report_golden import (
    FIRST,
    GOLDEN_REPORT_JSON,
    SECOND,
    golden_report_document,
)

SYSTEM: Final = "alpha"
OTHER: Final = "beta"
SUITE: Final = "golden"
REPO_ID: Final = "rinchen-test/rinchen-tiny"
# What `rinchen eval run --model` records for an engine system.
ENGINE_OPTIONS: Final[Mapping[str, str | int | bool]] = {
    "beam_size": 4,
    "device": "cpu",
    "max_decoding_length": 256,
    "script_mask": "retry",
    "threads": 1,
}
# A robustness block of the shape of D12's (numbers 892 of 960 as D12's toy measured it; the rest
# are values of the test).
ROBUSTNESS_BLOCK: Final[Mapping[str, float]] = {
    "numbers_acc": 0.9291666666666667,
    "placeholder_acc": 0.9983333333333333,
    "term_acc": 1.0,
    "script_purity": 1.0,
    "entity_acc": 0.9312,
    "mong_source_retention": 0.9976,
}
ROBUSTNESS_SUITE: Final = "robust-main"
# The digests and sizes the golden card is rendered with: the card is a function of its inputs, and
# the digests of a fabricated export hold the temporary directory it was made in (its manifest
# lists the paths), so the golden test hands the card these fixed values and not the disk's.
GOLDEN_MODEL_SHA256: Final = "0123456789abcdef" * 4
GOLDEN_EXPORT_MANIFEST_SHA256: Final = "fedcba9876543210" * 4
GOLDEN_REPORT_SHA256: Final = "00112233445566778899aabbccddeeff" * 2
GOLDEN_FILE_SIZES: Final[Mapping[str, int]] = {
    "model.bin": 30,
    "shared_vocabulary.json": 2048,
    "config.json": 312,
    "spm.model": 128,
    "rinchen.json": 1089,
    "report.json": 14894,
    "report.md": 6610,
}


def golden_runs() -> dict[str, str]:
    """The digest of each golden run's `run.json`, from the committed golden report."""
    golden = json.loads(GOLDEN_REPORT_JSON.read_text(encoding="utf-8"))
    return {run["system"]: run["run_sha256"] for run in golden["runs"]}


def golden_document() -> dict[str, Any]:
    return golden_report_document(golden_runs())


def bound_document(
    model_sha256: str,
    manifest_sha256: str,
    *,
    system: str = SYSTEM,
    options: Mapping[str, str | int | bool] | None = None,
) -> dict[str, Any]:
    """The golden report in which `system` is a Rinchen export with these digests (its results
    say so too: `manifest_sha` is the digest of the export's manifest)."""
    document = golden_document()
    recorded = dict(ENGINE_OPTIONS if options is None else options)
    for entry in document["systems"]:
        if entry["name"] == system:
            entry.update(
                kind="rinchen",
                model_sha256=model_sha256,
                manifest_sha256=manifest_sha256,
                options=recorded,
            )
    for result in document["results"]:
        if result["system"] == system:
            result["manifest_sha"] = manifest_sha256
            result["provenance"]["system"] = {
                "kind": "rinchen",
                "name": system,
                "model_sha256": model_sha256,
                "manifest_sha256": manifest_sha256,
                "options": recorded,
                "outputs_sha256": None,
            }
    return document


def with_robustness(
    document: dict[str, Any],
    system: str = SYSTEM,
    *,
    binding: str = "model_digest",
    block: Mapping[str, float | None] | None = None,
) -> dict[str, Any]:
    """`document` with D12's block of `system` attached, with its coverage and its source."""
    counts = {
        "numbers_acc": (960, 0),
        "placeholder_acc": (600, 0),
        "term_acc": (1044, 12),
        "script_purity": (900, 0),
        "entity_acc": (480, 0),
        "mong_source_retention": (100, 4),
    }
    coverage: dict[str, dict[str, float | int | None]] = {
        key: {"scored": scored, "excluded": excluded, "total": scored + excluded}
        for key, (scored, excluded) in counts.items()
    }
    coverage["mong_source_retention"].update(chrf_cyrl=97.5, chrf_mong=97.26)
    document["robustness"][system] = {
        "block": dict(ROBUSTNESS_BLOCK if block is None else block),
        "coverage": coverage,
        "source": {
            "suite": ROBUSTNESS_SUITE,
            "suite_sha256": "5" * 64,
            "robustness_sha256": "6" * 64,
            "report_sha256": "7" * 64,
            "binding": binding,
        },
    }
    return document


def as_benchmark(document: dict[str, Any]) -> dict[str, Any]:
    """The report's suite as a benchmark (kind `benchmark`): evidence `measured`."""
    suite = document["suites"][0]
    suite["kind"] = "benchmark"
    suite["benchmark"] = "flores_plus"
    document["evidence"] = {"kinds": ["benchmark"], "fixture_only": False}
    return document


def with_a_human_suite(document: dict[str, Any]) -> dict[str, Any]:
    """Another suite, of kind `human`, that no run scored: evidence `mixed` (kinds `fixture` and
    `human`) and a suite in which no system has a run."""
    other = copy.deepcopy(document["suites"][0])
    other.update(
        name="golden-human", kind="human", benchmark="golden-human", sha256="8" * 64
    )
    other["split"] = "test"
    document["suites"].append(other)
    document["evidence"] = {"kinds": ["fixture", "human"], "fixture_only": False}
    return document


def with_a_run_in_another_suite(
    document: dict[str, Any],
    system: str,
    *,
    name: str = "golden-bench",
    kind: str = "benchmark",
) -> dict[str, Any]:
    """A second suite of `kind` in which ONLY `system` has a run, with the results of its first
    suite (as a copy): the report's evidence now includes `kind`, and so does the evidence of
    `system` and of no other system."""
    other = copy.deepcopy(document["suites"][0])
    other.update(name=name, kind=kind, sha256="8" * 64)
    other["benchmark"] = "flores_plus" if kind == "benchmark" else name
    document["suites"].append(other)
    for result in [r for r in document["results"] if r["system"] == system]:
        copied = copy.deepcopy(result)
        copied["suite"] = name
        copied["provenance"]["suite_sha256"] = "8" * 64
        document["results"].append(copied)
    document["runs"].append({"suite": name, "system": system, "run_sha256": "9" * 64})
    kinds = sorted({suite["kind"] for suite in document["suites"]})
    document["evidence"] = {"kinds": kinds, "fixture_only": kinds == ["fixture"]}
    return with_the_note_of(document)


def with_a_direction_that_did_not_run(
    document: dict[str, Any], direction: str, state: str, reason: str
) -> dict[str, Any]:
    """`direction` declared by the suite (10 rows) and, for every system, a coverage row of that
    `state` with its `reason` (no result)."""
    for entry in document["suites"][0]["directions"]:
        if entry["direction"] == direction:
            entry["rows"] = 10
    for system in document["systems"]:
        document["coverage"].append(
            {
                "suite": SUITE,
                "system": system["name"],
                "direction": direction,
                "state": state,
                "reason": reason,
            }
        )
    return document


def without_results_of(document: dict[str, Any], system: str) -> dict[str, Any]:
    """`document` in which `system` has no result (its runs and coverage are dropped)."""
    document["results"] = [r for r in document["results"] if r["system"] != system]
    document["runs"] = [r for r in document["runs"] if r["system"] != system]
    document["coverage"] = [c for c in document["coverage"] if c["system"] != system]
    document["comparisons"] = [
        c for c in document["comparisons"] if system not in (c["system"], c["baseline"])
    ]
    return with_the_note_of(document)


def with_the_note_of(document: dict[str, Any]) -> dict[str, Any]:
    """Make the multiple-comparisons note the one the document's counts give (after a change to
    its results or comparisons)."""
    intervals = sum(
        interval is not None
        for result in document["results"]
        for interval in result["ci95"].values()
    ) + sum(c["ci95"] is not None for c in document["comparisons"])
    document["note_multiple_comparisons"] = (
        f"{intervals} intervals and {len(document['comparisons'])} paired comparisons are "
        "reported with no correction for multiple comparisons"
    )
    return document


def with_comet_scores(
    document: dict[str, Any], system: str = SYSTEM, *, direction: str | None = None
) -> dict[str, Any]:
    """`document` in which the results of `system` (only those of `direction` when it is given)
    have a COMET-22 score with its interval (and so no reason in `unmeasured` for it)."""
    for result in document["results"]:
        if result["system"] == system and direction in (None, result["direction"]):
            result["metrics"]["comet22"] = 0.8512
            result["ci95"]["comet22"] = [0.7931, 0.9087]
            del result["unmeasured"]["comet22"]
    return with_the_note_of(document)


def with_one_unit(document: dict[str, Any], index: int = 0) -> dict[str, Any]:
    """The comparison `index` made over one unit only: no interval, no p-value, not significant."""
    comparison = document["comparisons"][index]
    comparison.update(
        units=1, ci95=None, p_value=None, better_fraction=None, significant=False
    )
    return with_the_note_of(document)


def report_bytes(document: Mapping[str, Any]) -> bytes:
    return report_to_bytes(document)


def markdown_bytes(document: Mapping[str, Any]) -> bytes:
    return render_markdown(document).encode("utf-8")


def view_of(document: Mapping[str, Any]) -> ReportView:
    """The strict reader's view of `document`."""
    return report_from_bytes(report_bytes(document))


def a_candidate_in_two_comparisons(document: dict[str, Any]) -> dict[str, Any]:
    """Make `alpha` a candidate against `beta` in both directions (chrF++ and BLEU): the
    comparisons of the golden report with the two systems swapped, with the delta negated and
    the interval mirrored (a difference that goes against `alpha`, so not significant)."""
    swapped = []
    for comparison in document["comparisons"]:
        mirrored = copy.deepcopy(comparison)
        mirrored["system"], mirrored["baseline"] = SYSTEM, OTHER
        mirrored["delta"] = -comparison["delta"]
        low, high = comparison["ci95"]
        mirrored["ci95"] = [-high, -low]
        mirrored["better_fraction"] = 0.0
        mirrored["significant"] = False
        swapped.append(mirrored)
    document["comparisons"] = [*document["comparisons"], *swapped]
    return with_the_note_of(document)


__all__ = [
    "ENGINE_OPTIONS",
    "FIRST",
    "GOLDEN_EXPORT_MANIFEST_SHA256",
    "GOLDEN_FILE_SIZES",
    "GOLDEN_MODEL_SHA256",
    "GOLDEN_REPORT_SHA256",
    "OTHER",
    "REPO_ID",
    "ROBUSTNESS_BLOCK",
    "ROBUSTNESS_SUITE",
    "SECOND",
    "SUITE",
    "SYSTEM",
    "a_candidate_in_two_comparisons",
    "as_benchmark",
    "bound_document",
    "golden_document",
    "golden_runs",
    "markdown_bytes",
    "report_bytes",
    "view_of",
    "with_a_direction_that_did_not_run",
    "with_a_human_suite",
    "with_a_run_in_another_suite",
    "with_comet_scores",
    "with_one_unit",
    "with_robustness",
    "with_the_note_of",
    "without_results_of",
]
