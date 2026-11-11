"""The golden report of D14 (Task 4): `tests/fixtures/maintain/golden_report.json` and
`golden_report.md` are the report of the golden run written byte for byte, and what `run_check`,
`report_to_bytes` and `render_markdown` make of that run must equal them.

The golden run is the nightly run of the shipped configuration over the sample lock of
`tests/support/uv_outputs.py`, with six doctor passes and the GPU skip, every installed version
equal to the lock's, the HAND-WRITTEN dry run of five updates, one add and one remove (its lines
are UNMEASURED) and the reruns `numerics` (1,110 passed) and `parity_tiny` (13 passed). The JSON
was cross-checked with an independent rendering: `expected_document` below is the same document
written by hand from the rules of the spec, and `json.dumps` of it is the file's bytes. The
Markdown was cross-checked by a second renderer written apart from the code and read once by a
person; here it is pinned as it renders and by its digest. After an intended change of the shipped
pin rules or reruns (they name themselves in the table) or of the layout, regenerate both files
from `report_to_bytes` and `render_markdown` over `golden_run` and check the JSON against
`expected_document` again before committing them.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from rinchen.maintain.report import report_document, report_to_bytes
from rinchen.maintain.report_markdown import render_markdown
from tests.support.maintain_check import (
    CPU_CHECKS,
    GPU_CHECK,
    SECRET_ENVIRONMENT,
    golden_run,
    make_inputs,
)
from tests.support.maintain_fakes import JUNIT_HOST, JUNIT_PATH
from tests.support.maintain_scans import closed_reason, reasons_of, violations
from tests.support.no_network import no_network  # noqa: F401 - the fixture of the mark

pytestmark = pytest.mark.usefixtures("no_network")

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "maintain"
GOLDEN_JSON = FIXTURES / "golden_report.json"
GOLDEN_MD = FIXTURES / "golden_report.md"
# `sha256sum` of the two files, taken from the files and not from the code.
GOLDEN_JSON_SHA256 = "ac7f08b1270119b75286e55153e77e38044c4c5730db0c0adcdacd7b4126eeec"
GOLDEN_MD_SHA256 = "7e7774f8f0dc3b26287aca55851bfdae9e058719fabfb49e76924c94b0f6d07a"
MAX_BYTES = 8 * 1024


def expected_document() -> dict[str, Any]:
    """The report of the golden run, written by hand from the rules of the spec."""
    inputs = make_inputs()

    def drift(
        package: str,
        change: str,
        locked: str | None,
        available: str | None,
        rule: str | None,
        requires: list[str],
    ) -> dict[str, Any]:
        return {
            "package": package,
            "change": change,
            "locked": locked,
            "available": available,
            "class": "drift" if rule is None else "blocked",
            "rule": rule,
            "requires": requires,
        }

    def rerun(
        rerun_id: str,
        kind: str,
        cadence: str,
        state: str,
        reason: str | None,
        passed: int | None,
    ) -> dict[str, Any]:
        tests = (
            None
            if passed is None
            else {"passed": passed, "failed": 0, "errors": 0, "skipped": 0}
        )
        return {
            "id": rerun_id,
            "kind": kind,
            "cadence": cadence,
            "state": state,
            "reason": reason,
            "tests": tests,
            "figures": None,
        }

    not_due = "not due at the nightly cadence"
    return {
        "schema": "rinchen.maintain/1",
        "cadence": "nightly",
        "state": "drift",
        "inputs": {
            "config": inputs.config_sha256,
            "lock": inputs.lock_sha256,
            "pyproject": inputs.pyproject_sha256,
        },
        "lock": {"packages": 6, "names": 6, "format": {"version": 1, "revision": 5}},
        "steps": [
            {
                "id": "doctor",
                "state": "ok",
                "reason": None,
                "detail": {
                    "checks": [
                        *(
                            {"id": check, "state": "ok", "reason": None}
                            for check in CPU_CHECKS
                        ),
                        {
                            "id": GPU_CHECK,
                            "state": "skipped",
                            "reason": "a GPU check: maintain check never asks a device question",
                        },
                    ]
                },
            },
            {"id": "lock_file", "state": "ok", "reason": None, "detail": {}},
            {"id": "lock_consistent", "state": "ok", "reason": None, "detail": {}},
            {
                "id": "environment",
                "state": "ok",
                "reason": None,
                "detail": {
                    "installed": 6,
                    "mismatched": [],
                    "mismatched_more": 0,
                    "extra": [],
                    "extra_more": 0,
                    "not_installed": 0,
                },
            },
            {
                "id": "drift",
                "state": "drift",
                "reason": None,
                "detail": {
                    "counts": {
                        "update": 5,
                        "add": 1,
                        "remove": 1,
                        "drift": 3,
                        "blocked": 4,
                    },
                    "warnings": 0,
                    "entries": [
                        drift(
                            "huggingface-hub",
                            "update",
                            "2.3.0",
                            "2.3.1",
                            "hub-client",
                            ["evaluation_tiny", "gate"],
                        ),
                        drift(
                            "numpy",
                            "update",
                            "2.5.4",
                            "2.5.5",
                            "sacrebleu-numpy",
                            ["numerics", "evaluation_tiny"],
                        ),
                        drift(
                            "nvidia-cudnn-cu13",
                            "update",
                            "9.24.0.43",
                            "9.25.0.10",
                            "torch-family",
                            ["parity_tiny", "robustness_tiny", "gate"],
                        ),
                        drift("rich", "update", "15.0.0", "15.0.1", None, []),
                        drift(
                            "safetensors",
                            "update",
                            "0.8.0",
                            "0.8.1",
                            "transformers-stack",
                            ["gate"],
                        ),
                        drift("sortedcontainers", "remove", "2.4.0", None, None, []),
                        drift("tomli-w", "add", None, "1.2.0", None, []),
                    ],
                },
            },
            {
                "id": "reruns",
                "state": "ok",
                "reason": None,
                "detail": {
                    "items": [
                        rerun("numerics", "pytest", "nightly", "ok", None, 1110),
                        rerun("parity_tiny", "pytest", "nightly", "ok", None, 13),
                        rerun(
                            "parity_export",
                            "export_parity",
                            "nightly",
                            "skipped",
                            "no export was given",
                            None,
                        ),
                        rerun(
                            "robustness_tiny",
                            "pytest",
                            "weekly",
                            "skipped",
                            not_due,
                            None,
                        ),
                        rerun(
                            "evaluation_tiny",
                            "pytest",
                            "weekly",
                            "skipped",
                            not_due,
                            None,
                        ),
                        rerun(
                            "serving_tiny", "pytest", "weekly", "skipped", not_due, None
                        ),
                    ]
                },
            },
        ],
    }


# ---- the golden files ---------------------------------------------------------------------------------


def test_the_report_of_the_golden_run_is_the_golden_json_byte_for_byte(
    tmp_path: Path,
) -> None:
    report = golden_run(tmp_path).result.report

    assert report_to_bytes(report) == GOLDEN_JSON.read_bytes()


def test_the_markdown_of_the_golden_run_is_the_golden_markdown_byte_for_byte(
    tmp_path: Path,
) -> None:
    report = golden_run(tmp_path).result.report

    assert render_markdown(report).encode("utf-8") == GOLDEN_MD.read_bytes()


def test_the_golden_json_is_the_hand_written_document_rendered_independently() -> None:
    expected = json.dumps(
        expected_document(), sort_keys=True, indent=2, ensure_ascii=False
    )

    assert GOLDEN_JSON.read_bytes() == f"{expected}\n".encode()


def test_the_document_of_the_golden_run_equals_the_hand_written_one_value_by_value(
    tmp_path: Path,
) -> None:
    document = report_document(golden_run(tmp_path).result.report)

    assert document == expected_document()


def test_the_golden_files_are_the_ones_whose_digests_are_pinned_and_are_small() -> None:
    assert hashlib.sha256(GOLDEN_JSON.read_bytes()).hexdigest() == GOLDEN_JSON_SHA256
    assert hashlib.sha256(GOLDEN_MD.read_bytes()).hexdigest() == GOLDEN_MD_SHA256
    assert 0 < len(GOLDEN_JSON.read_bytes()) <= MAX_BYTES
    assert 0 < len(GOLDEN_MD.read_bytes()) <= MAX_BYTES


def test_the_golden_json_has_the_sorted_two_space_form_and_one_final_line_feed() -> (
    None
):
    data = GOLDEN_JSON.read_bytes()

    assert data.endswith(b"}\n") and not data.endswith(b"\n\n")
    assert b"\r" not in data
    assert (
        data
        == (
            json.dumps(json.loads(data), sort_keys=True, indent=2, ensure_ascii=False)
            + "\n"
        ).encode()
    )


def test_the_golden_markdown_has_line_feeds_only_and_one_final_one() -> None:
    text = GOLDEN_MD.read_text(encoding="utf-8")

    assert text.endswith("\n") and not text.endswith("\n\n")
    assert "\r" not in text
    assert text.startswith("# Maintenance check: drift (nightly cadence)\n")


# ---- the content rules ---------------------------------------------------------------------------------


def secrets() -> list[str]:
    return [*SECRET_ENVIRONMENT.values(), JUNIT_HOST, JUNIT_PATH, "/srv/example"]


@pytest.mark.parametrize("path", [GOLDEN_JSON, GOLDEN_MD], ids=lambda path: path.name)
def test_the_golden_files_hold_no_path_timestamp_host_output_line_or_environment_value(
    path: Path,
) -> None:
    assert violations(path.read_text(encoding="utf-8"), secrets()) == []


def test_every_reason_of_the_golden_report_is_a_closed_reason(tmp_path: Path) -> None:
    reasons = list(reasons_of(report_document(golden_run(tmp_path).result.report)))

    assert reasons
    assert all(closed_reason(reason) for reason in reasons)


def test_the_scans_see_what_they_are_for() -> None:
    text = "at /home/x on 2026-01-02 host hostname Resolved 3 packages swordfish-0451"

    assert violations(text, ["swordfish-0451"]) == [
        "path /home",
        "timestamp",
        "host hostname",
        "output Resolved",
        "secret 0",
    ]
    assert violations("") == []
    assert not closed_reason("uv exited with code")
    assert closed_reason("uv exited with code 1") and closed_reason(
        "not due at the weekly cadence"
    )
    assert not closed_reason("not due at the hourly cadence")


def test_two_runs_over_the_same_observations_give_equal_bytes(tmp_path: Path) -> None:
    first = golden_run(tmp_path / "a")
    second = golden_run(tmp_path / "b")

    assert report_to_bytes(first.result.report) == report_to_bytes(second.result.report)
    assert render_markdown(first.result.report) == render_markdown(second.result.report)
    assert first.result.logs == second.result.logs
