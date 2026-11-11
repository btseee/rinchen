"""No text of a suite or of an output comes out of `rinchen eval report` (D13, Task 8): not in the
report it writes, and not in any message of any way it fails.

A marker is planted in every sentence of the suite and in every text of both systems' outputs, the
runs are made by `rinchen eval run`, and then the report is asked for in each way it can fail: a dev
run, a suite that was not given, a duplicate run, a file that changed, a result that is not a result,
a baseline that is no system, an output directory that is an input's, a foreign manifest and an
output that cannot be written. The marker is in no standard output, no standard error and no log line,
and a report that is written holds none of it. (The messages themselves are scanned at the source:
`tests/unit/test_eval_error_messages.py` and `test_eval_error_messages_d13.py`.)
"""

from __future__ import annotations

import json
import logging
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
from click.testing import Result

from rinchen.eval.suites import Suite
from tests.support.distill_cli import write_foreign_manifest
from tests.support.eval_cli import invoke, make_workspace, outputs_file
from tests.support.eval_golden import GOLDEN_IDS
from tests.support.eval_report_cli import (
    FIRST,
    SECOND,
    ReportWorkspace,
    copy_run,
    invoke_report,
    rewrite_run_json,
    sha256_of,
)
from tests.support.no_network import no_network  # noqa: F401 - the fixture of the mark
from tests.support.robust_cli import fail_replace

pytestmark = pytest.mark.usefixtures("no_network")

MARKER = "SECRET-MARKER-77"


@pytest.fixture
def workspace(tmp_path: Path) -> ReportWorkspace:
    """The golden runs over a suite and outputs whose every text holds the marker."""
    ws = make_workspace(tmp_path)
    document = json.loads(ws.suite.read_text("utf-8"))
    for row in document["rows"]:
        row["text"]["eng_Latn"] = f"{MARKER} english {row['id']}"
        row["text"]["khk_Cyrl"] = f"{MARKER} khalkha {row['id']}"
    ws.suite.write_text(
        json.dumps(document, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        "utf-8",
    )
    for name, path in (("alpha", ws.alpha), ("beta", ws.beta)):
        outputs_file(
            path,
            {
                FIRST: [f"{MARKER} {name} one {i}" for i in GOLDEN_IDS],
                SECOND: [f"{MARKER} {name} two {i}" for i in GOLDEN_IDS],
            },
        )
        made = invoke(ws.hypotheses_args(hypotheses=path))
        assert made.exit_code == 0, made.output
    return ReportWorkspace(
        ws, ws.out_dir / "golden" / "alpha", ws.out_dir / "golden" / "beta"
    )


def assert_no_marker(result: Result, caplog: pytest.LogCaptureFixture) -> None:
    assert MARKER not in result.stdout
    assert MARKER not in result.stderr
    assert MARKER not in caplog.text


def other_suite(workspace: ReportWorkspace, name: str, **changes: Any) -> Path:
    suite = Suite.from_bytes(workspace.suite.read_bytes())
    path = workspace.ws.root / "in" / f"{name}.json"
    path.write_bytes(replace(suite, name=name, **changes).to_bytes())
    return path


def test_a_report_that_is_written_holds_no_text_of_a_suite_or_of_an_output(
    workspace: ReportWorkspace, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.DEBUG)

    result = invoke_report(workspace)

    assert result.exit_code == 0, result.output
    assert_no_marker(result, caplog)
    for name in ("report.json", "report.md", "manifest.json"):
        assert MARKER not in (workspace.report_out / name).read_text("utf-8")


def test_a_dev_run_prints_no_text(
    workspace: ReportWorkspace, caplog: pytest.LogCaptureFixture
) -> None:
    dev = other_suite(workspace, "golden", split="dev")
    out = workspace.ws.root / "dev-runs"
    made = invoke(workspace.ws.hypotheses_args(suite=dev, out_dir=out))
    assert made.exit_code == 0, made.output
    caplog.set_level(logging.DEBUG)

    result = invoke_report(
        workspace, runs=[out / "golden" / "alpha"], suites=[dev], baseline=None
    )

    assert result.exit_code == 3
    assert_no_marker(result, caplog)


def test_a_suite_that_was_not_given_prints_no_text(
    workspace: ReportWorkspace, caplog: pytest.LogCaptureFixture
) -> None:
    unrelated = other_suite(workspace, "unrelated")
    caplog.set_level(logging.DEBUG)

    result = invoke_report(workspace, suites=[unrelated])

    assert result.exit_code == 3
    assert_no_marker(result, caplog)


def test_a_duplicate_run_prints_no_text(
    workspace: ReportWorkspace, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    twin = copy_run(workspace.alpha, tmp_path / "twin")
    caplog.set_level(logging.DEBUG)

    result = invoke_report(workspace, runs=[workspace.alpha, twin, workspace.beta])

    assert result.exit_code == 3
    assert_no_marker(result, caplog)


@pytest.mark.parametrize("name", [f"{FIRST}/result.json", "hypotheses.jsonl"])
def test_a_file_that_changed_prints_its_name_and_no_text(
    workspace: ReportWorkspace, caplog: pytest.LogCaptureFixture, name: str
) -> None:
    path = workspace.alpha / name
    path.write_bytes(path.read_bytes() + f"{MARKER}\n".encode())
    caplog.set_level(logging.DEBUG)

    result = invoke_report(workspace)

    assert result.exit_code == 3
    assert name in result.stderr
    assert_no_marker(result, caplog)


def test_outputs_that_are_not_the_suites_print_no_text(
    workspace: ReportWorkspace, caplog: pytest.LogCaptureFixture
) -> None:
    path = workspace.alpha / "hypotheses.jsonl"
    path.write_text(
        f'{{"direction": "{FIRST}", "id": "x", "status": "ok", "text": "{MARKER}"}}\n',
        "utf-8",
    )
    rewrite_run_json(
        workspace.alpha,
        lambda document: document["files"].update(
            {"hypotheses.jsonl": sha256_of(path)}
        ),
    )
    caplog.set_level(logging.DEBUG)

    result = invoke_report(workspace)

    assert result.exit_code == 3
    assert "not an evaluation" in result.stderr or "outputs" in result.stderr
    assert_no_marker(result, caplog)


def test_a_result_that_is_not_a_result_prints_no_text(
    workspace: ReportWorkspace, caplog: pytest.LogCaptureFixture
) -> None:
    path = workspace.alpha / FIRST / "result.json"
    path.write_text(json.dumps({"schema": MARKER, "system": MARKER}), "utf-8")
    rewrite_run_json(
        workspace.alpha,
        lambda document: document["files"].update(
            {f"{FIRST}/result.json": sha256_of(path)}
        ),
    )
    caplog.set_level(logging.DEBUG)

    result = invoke_report(workspace)

    assert result.exit_code == 3
    assert "not an evaluation result" in result.stderr
    assert_no_marker(result, caplog)


def test_a_baseline_that_is_no_system_prints_no_text_of_a_suite(
    workspace: ReportWorkspace, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.DEBUG)

    result = invoke_report(workspace, baseline="nobody")

    assert result.exit_code == 2
    assert_no_marker(result, caplog)


def test_an_output_directory_that_is_an_input_prints_no_text(
    workspace: ReportWorkspace, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.DEBUG)

    result = invoke_report(workspace, out_dir=workspace.beta)

    assert result.exit_code == 2
    assert_no_marker(result, caplog)


def test_a_foreign_manifest_prints_no_text(
    workspace: ReportWorkspace, caplog: pytest.LogCaptureFixture
) -> None:
    write_foreign_manifest(workspace.report_out)
    caplog.set_level(logging.DEBUG)

    result = invoke_report(workspace)

    assert result.exit_code == 3
    assert_no_marker(result, caplog)


def test_an_output_that_cannot_be_written_prints_no_text(
    workspace: ReportWorkspace,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    fail_replace(monkeypatch, "report.json")
    caplog.set_level(logging.DEBUG)

    result = invoke_report(workspace)

    assert result.exit_code == 1
    assert_no_marker(result, caplog)
