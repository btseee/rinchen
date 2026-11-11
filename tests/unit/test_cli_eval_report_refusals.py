"""`rinchen eval report`: what it refuses, with the exit code and the one-line error the command
documents (D13, Task 8). 2 is a bad option, an output directory that is an input's, a file that
changed after it was hashed, a baseline that names no system, a suite that no run used; 3 an
unreadable or invalid input (a run directory without a manifest or with the manifest of another
stage, a manifest that does not vouch for `run.json`, a file that is not the one `run.json` lists,
a suite that was not given, two runs of one system over one suite, a run of a `dev` suite, a
foreign manifest in the output directory); 1 an output that cannot be written.

A refusal writes NOTHING: the output directory is not made (or, when it exists, is as it was), the
run directories are untouched and standard output is empty.
"""

from __future__ import annotations

import json
import shutil
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
from click.testing import Result

import rinchen.cli.eval_report_inputs as inputs_module
from rinchen.eval.suites import Suite
from tests.support.distill_cli import write_foreign_manifest
from tests.support.eval_cli import invoke
from tests.support.eval_report_cli import (
    FIRST,
    ReportWorkspace,
    copy_run,
    invoke_report,
    make_report_workspace,
    rewrite_manifest,
    rewrite_run_json,
    sha256_of,
)
from tests.support.launch_cli import assert_one_error, snapshot
from tests.support.no_network import no_network  # noqa: F401 - the fixture of the mark
from tests.support.robust_cli import fail_replace

pytestmark = pytest.mark.usefixtures("no_network")


@pytest.fixture
def workspace(tmp_path: Path) -> ReportWorkspace:
    return make_report_workspace(tmp_path)


def assert_refused(
    workspace: ReportWorkspace, result: Result, code: int, start: str
) -> None:
    assert_one_error(result, code, start)
    assert result.stdout == ""
    assert not workspace.report_out.exists()


def edited_suite(
    workspace: ReportWorkspace, name: str, change: Callable[[dict[str, Any]], None]
) -> Path:
    document = json.loads(workspace.suite.read_text("utf-8"))
    change(document)
    path = workspace.ws.root / "in" / name
    path.write_text(
        json.dumps(document, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        "utf-8",
    )
    return path


# ---- the options -------------------------------------------------------------------------------


def test_no_run_is_a_usage_error(workspace: ReportWorkspace) -> None:
    result = invoke_report(workspace, runs=[])

    assert result.exit_code == 2
    assert "--run" in result.stderr
    assert not workspace.report_out.exists()


def test_no_suite_is_a_usage_error(workspace: ReportWorkspace) -> None:
    result = invoke_report(workspace, suites=[])

    assert result.exit_code == 2
    assert "--suite" in result.stderr
    assert not workspace.report_out.exists()


def test_a_baseline_that_names_no_system_is_a_config_error(
    workspace: ReportWorkspace,
) -> None:
    result = invoke_report(workspace, baseline="nobody")

    assert_refused(
        workspace, result, 2, "the baseline 'nobody' is not a system of any run"
    )


def test_a_missing_metrics_configuration_is_a_config_error(
    workspace: ReportWorkspace, tmp_path: Path
) -> None:
    result = invoke_report(workspace, "--metrics-config", str(tmp_path / "nope.yaml"))

    assert_refused(workspace, result, 2, "eval metrics config not found")


def test_an_invalid_metrics_configuration_is_a_config_error(
    workspace: ReportWorkspace, tmp_path: Path
) -> None:
    bad = tmp_path / "bad.yaml"
    bad.write_text("bootstrap: 3\n", "utf-8")

    result = invoke_report(workspace, "--metrics-config", str(bad))

    assert_refused(workspace, result, 2, "invalid eval metrics config")


def test_a_run_scored_with_other_bootstrap_settings_than_the_configuration_is_refused(
    workspace: ReportWorkspace, tmp_path: Path
) -> None:
    config = tmp_path / "config.yaml"
    config.write_text(
        workspace.ws.metrics.read_text("utf-8").replace(
            "resamples: 1000", "resamples: 500"
        ),
        "utf-8",
    )

    result = invoke_report(workspace, "--metrics-config", str(config))

    assert_refused(
        workspace,
        result,
        3,
        "the run of the system 'alpha' over the suite 'golden' was scored with other "
        "bootstrap settings",
    )


def test_an_output_directory_that_is_a_run_directory_is_a_config_error(
    workspace: ReportWorkspace,
) -> None:
    before = snapshot(workspace.alpha)

    result = invoke_report(workspace, out_dir=workspace.alpha)

    assert_one_error(
        result, 2, f"--out-dir {workspace.alpha} is the directory of the input file"
    )
    assert result.stdout == ""
    assert snapshot(workspace.alpha) == before


def test_an_output_directory_that_holds_a_suite_file_is_a_config_error(
    workspace: ReportWorkspace,
) -> None:
    result = invoke_report(workspace, out_dir=workspace.suite.parent)

    assert_one_error(
        result, 2, f"--out-dir {workspace.suite.parent} is the directory of the input"
    )
    assert result.stdout == ""


def test_a_foreign_manifest_in_the_output_directory_is_a_data_error_and_stays(
    workspace: ReportWorkspace,
) -> None:
    kept = write_foreign_manifest(workspace.report_out)
    before = snapshot(workspace.report_out)

    result = invoke_report(workspace)

    assert_one_error(result, 3, str(workspace.report_out))
    assert (
        "holds the manifest of stage 'train_docify', not of 'eval_report'"
        in result.stderr
    )
    assert result.stdout == ""
    assert (workspace.report_out / "manifest.json").read_bytes() == kept
    assert snapshot(workspace.report_out) == before


def test_an_output_that_cannot_be_written_is_exit_one_and_leaves_no_manifest(
    workspace: ReportWorkspace, monkeypatch: pytest.MonkeyPatch
) -> None:
    fail_replace(monkeypatch, "report.md")

    result = invoke_report(workspace)

    assert_one_error(result, 1, "cannot write the evaluation report under")
    assert result.stdout == ""
    assert not (workspace.report_out / "manifest.json").exists()


def test_a_report_that_would_hold_a_path_is_refused_before_anything_is_written(
    workspace: ReportWorkspace, monkeypatch: pytest.MonkeyPatch
) -> None:
    import rinchen.cli.eval_report as command

    real = command.build_report

    def leaking(*args: Any, **kwargs: Any) -> dict[str, Any]:
        report = real(*args, **kwargs)
        report["systems"][0]["options"] = {"model_dir": "/home/user/model"}
        return report

    monkeypatch.setattr(command, "build_report", leaking)

    result = invoke_report(workspace)

    assert_refused(workspace, result, 3, "not an evaluation report: field 'systems.0'")
    assert "absolute path" in result.stderr
    assert "/home/user" not in result.stderr


# ---- the run directories -----------------------------------------------------------------------


def test_a_run_directory_without_a_manifest_is_a_data_error(
    workspace: ReportWorkspace,
) -> None:
    (workspace.alpha / "manifest.json").unlink()

    result = invoke_report(workspace)

    assert_refused(
        workspace,
        result,
        3,
        f"run manifest not found: {workspace.alpha / 'manifest.json'}",
    )


def test_a_run_directory_without_a_run_record_is_a_data_error(
    workspace: ReportWorkspace,
) -> None:
    (workspace.alpha / "run.json").unlink()

    result = invoke_report(workspace)

    assert_refused(
        workspace, result, 3, f"run record not found: {workspace.alpha / 'run.json'}"
    )


def test_a_directory_that_is_not_a_run_is_a_data_error(
    workspace: ReportWorkspace, tmp_path: Path
) -> None:
    result = invoke_report(workspace, runs=[tmp_path / "in"])

    assert_refused(workspace, result, 3, "run manifest not found")


def test_a_run_directory_whose_manifest_is_of_another_stage_is_a_data_error(
    workspace: ReportWorkspace,
) -> None:
    rewrite_manifest(workspace.alpha, lambda m: setattr(m, "stage", "eval_robustness"))

    result = invoke_report(workspace)

    assert_refused(
        workspace,
        result,
        3,
        f"the manifest of the run directory {workspace.alpha} is not the manifest of an "
        "eval_run stage",
    )


def test_a_run_directory_whose_manifest_does_not_say_the_run_finished_is_refused(
    workspace: ReportWorkspace,
) -> None:
    rewrite_manifest(
        workspace.alpha, lambda m: setattr(m, "status", {"eval": "partial"})
    )

    result = invoke_report(workspace)

    assert_refused(
        workspace,
        result,
        3,
        f"the manifest of the run directory {workspace.alpha} does not say that the run "
        "finished",
    )


def test_a_manifest_that_does_not_list_run_json_is_a_data_error(
    workspace: ReportWorkspace,
) -> None:
    def drop(manifest: Any) -> None:
        manifest.outputs = [
            i for i in manifest.outputs if not i.path.endswith("run.json")
        ]

    rewrite_manifest(workspace.alpha, drop)

    result = invoke_report(workspace)

    assert_refused(
        workspace,
        result,
        3,
        f"the manifest of the run directory {workspace.alpha} does not vouch for run.json "
        "as it is now",
    )


def test_a_run_json_that_changed_after_the_manifest_vouched_for_it_is_a_data_error(
    workspace: ReportWorkspace,
) -> None:
    path = workspace.alpha / "run.json"
    path.write_bytes(path.read_bytes() + b" ")

    result = invoke_report(workspace)

    assert_refused(
        workspace,
        result,
        3,
        f"the manifest of the run directory {workspace.alpha} does not vouch for run.json "
        "as it is now",
    )


@pytest.mark.parametrize("name", [f"{FIRST}/result.json", "hypotheses.jsonl"])
def test_a_file_whose_bytes_changed_is_a_data_error_naming_it(
    workspace: ReportWorkspace, name: str
) -> None:
    path = workspace.alpha / name
    path.write_bytes(path.read_bytes() + b"\n")

    result = invoke_report(workspace)

    assert_refused(
        workspace,
        result,
        3,
        f"{name} of the run directory {workspace.alpha} is not the file that run.json lists",
    )


def test_a_file_that_run_json_lists_and_has_gone_is_a_data_error(
    workspace: ReportWorkspace,
) -> None:
    (workspace.beta / f"{FIRST}/result.json").unlink()

    result = invoke_report(workspace)

    assert_refused(
        workspace,
        result,
        3,
        f"run file not found: {workspace.beta / FIRST / 'result.json'}",
    )


def test_a_run_json_that_is_not_a_run_is_a_data_error(
    workspace: ReportWorkspace,
) -> None:
    rewrite_run_json(workspace.alpha, lambda document: document.update(schema="x"))

    result = invoke_report(workspace)

    assert_refused(workspace, result, 3, "not an evaluation run: field 'schema'")


def test_a_result_that_is_not_a_result_is_a_data_error(
    workspace: ReportWorkspace,
) -> None:
    path = workspace.alpha / FIRST / "result.json"
    path.write_bytes(b"{}\n")

    def lists_the_new_bytes(document: dict[str, Any]) -> None:
        document["files"][f"{FIRST}/result.json"] = sha256_of(path)

    rewrite_run_json(workspace.alpha, lists_the_new_bytes)

    result = invoke_report(workspace)

    assert_refused(workspace, result, 3, "not an evaluation result")


def test_a_result_of_another_system_than_its_run_is_a_data_error(
    workspace: ReportWorkspace,
) -> None:
    # beta's result file in alpha's run directory, listed by digest: it says it is beta's
    shutil.copyfile(
        workspace.beta / FIRST / "result.json", workspace.alpha / FIRST / "result.json"
    )

    def lists_the_copy(document: dict[str, Any]) -> None:
        document["files"][f"{FIRST}/result.json"] = sha256_of(
            workspace.alpha / FIRST / "result.json"
        )

    rewrite_run_json(workspace.alpha, lists_the_copy)

    result = invoke_report(workspace)

    assert_refused(
        workspace,
        result,
        3,
        "the run of the system 'alpha' over the suite 'golden', direction "
        f"{FIRST}: the result is of another system",
    )


# ---- the suites --------------------------------------------------------------------------------


def test_a_suite_that_was_not_given_is_a_data_error_naming_the_run(
    workspace: ReportWorkspace, tmp_path: Path
) -> None:
    other = edited_suite(workspace, "other.json", lambda d: d.update(name="other"))

    result = invoke_report(workspace, suites=[other])

    assert_refused(
        workspace,
        result,
        3,
        "no --suite file matches the run of the system 'alpha' over the suite 'golden'",
    )


def test_a_suite_file_that_does_not_exist_is_a_data_error(
    workspace: ReportWorkspace, tmp_path: Path
) -> None:
    result = invoke_report(workspace, suites=[tmp_path / "nope.json"])

    assert_refused(workspace, result, 3, "suite file not found")


def test_a_suite_file_that_is_not_a_suite_is_a_data_error(
    workspace: ReportWorkspace, tmp_path: Path
) -> None:
    junk = tmp_path / "junk.json"
    junk.write_text("[]", "utf-8")

    result = invoke_report(workspace, suites=[workspace.suite, junk])

    assert_refused(workspace, result, 3, "not an evaluation suite")


def test_a_suite_that_no_run_scored_is_a_config_error(
    workspace: ReportWorkspace,
) -> None:
    unused = edited_suite(workspace, "unused.json", lambda d: d.update(name="unused"))

    result = invoke_report(workspace, suites=[workspace.suite, unused])

    assert_refused(workspace, result, 2, "no run scored the suite 'unused'")


def test_two_runs_of_one_system_over_one_suite_are_a_data_error(
    workspace: ReportWorkspace, tmp_path: Path
) -> None:
    twin = copy_run(workspace.alpha, tmp_path / "twin")

    result = invoke_report(workspace, runs=[workspace.alpha, twin, workspace.beta])

    assert_refused(
        workspace,
        result,
        3,
        "two runs of the system 'alpha' over the suite 'golden'",
    )


def test_two_files_of_one_suite_given_twice_are_not_a_second_suite(
    workspace: ReportWorkspace, tmp_path: Path
) -> None:
    again = tmp_path / "same-bytes.json"
    again.write_bytes(workspace.suite.read_bytes())

    result = invoke_report(workspace, suites=[workspace.suite, again])

    assert result.exit_code == 0, result.output


# ---- the dev guard -----------------------------------------------------------------------------


@pytest.fixture
def dev_run(workspace: ReportWorkspace, tmp_path: Path) -> tuple[Path, Path]:
    """A run of the golden suite as a `dev` suite (a fixture may be of any split): `eval run`
    accepts it (its purpose is `selection`), and `eval report` must not."""
    suite = Suite.from_bytes(workspace.suite.read_bytes())
    dev = replace(suite, split="dev")
    path = tmp_path / "dev_suite.json"
    path.write_bytes(dev.to_bytes())
    out = tmp_path / "dev-runs"
    made = invoke(workspace.ws.hypotheses_args(suite=path, out_dir=out))
    assert made.exit_code == 0, made.output
    return out / "golden" / "alpha", path


def test_a_run_of_a_dev_suite_is_refused_naming_the_suite(
    workspace: ReportWorkspace, dev_run: tuple[Path, Path]
) -> None:
    run, suite = dev_run

    result = invoke_report(workspace, runs=[run], suites=[suite], baseline=None)

    assert_refused(workspace, result, 3, "the suite 'golden' is a dev suite")
    assert "choosing a model" in result.stderr


def test_a_run_of_a_dev_suite_is_refused_even_when_its_suite_was_not_given(
    workspace: ReportWorkspace, dev_run: tuple[Path, Path]
) -> None:
    run, _ = dev_run

    result = invoke_report(workspace, runs=[run], baseline=None)

    assert_refused(workspace, result, 3, "the suite 'golden' is a dev suite")


def test_a_dev_run_beside_good_runs_refuses_the_whole_report(
    workspace: ReportWorkspace, dev_run: tuple[Path, Path]
) -> None:
    run, suite = dev_run

    result = invoke_report(
        workspace,
        runs=[workspace.alpha, workspace.beta, run],
        suites=[workspace.suite, suite],
    )

    assert_refused(workspace, result, 3, "the suite 'golden' is a dev suite")


# ---- hash before read --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name",
    [
        "run.json",
        "manifest.json",
        "result.json",
        "hypotheses.jsonl",
        "golden_suite.json",
        "metrics.yaml",
    ],
)
def test_a_file_that_changes_after_it_was_hashed_is_refused_and_never_parsed(
    workspace: ReportWorkspace, monkeypatch: pytest.MonkeyPatch, name: str
) -> None:
    real = inputs_module.verified_content
    changed: list[Path] = []

    def late(file: Path, digest: str) -> bytes:
        if file.name == name and not changed:
            changed.append(file)
            file.write_bytes(file.read_bytes() + b" ")
        return real(file, digest)

    monkeypatch.setattr(inputs_module, "verified_content", late)

    # the copy of the configuration of the workspace: the shipped one is never touched
    result = invoke_report(workspace, "--metrics-config", str(workspace.ws.metrics))

    assert changed, "the file was never read through verified_content"
    assert_refused(workspace, result, 2, f"{changed[0]} changed after it was hashed")


def test_a_run_that_names_a_dev_suite_by_its_digest_is_refused_though_it_says_it_reports(
    workspace: ReportWorkspace, tmp_path: Path
) -> None:
    # A forged `run.json`: the purpose says `report` but the suite file it names is a dev suite.
    dev = replace(Suite.from_bytes(workspace.suite.read_bytes()), split="dev")
    dev_file = tmp_path / "forged_dev_suite.json"
    dev_file.write_bytes(dev.to_bytes())
    rewrite_run_json(
        workspace.alpha,
        lambda document: document["suite"].update(sha256=sha256_of(dev_file)),
    )

    result = invoke_report(workspace, suites=[workspace.suite, dev_file], baseline=None)

    assert_refused(workspace, result, 3, "the suite 'golden' is a dev suite")
