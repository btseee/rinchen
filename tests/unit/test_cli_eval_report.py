"""`rinchen eval report` over the golden runs (D13, Task 8): what it prints and writes for a report
that was measured, with the runs `rinchen eval run` wrote for `alpha` and `beta`.

The report and the Markdown it writes are the golden files byte for byte, so the command, the
library test and the independent rendering all agree. The command reads a run through its
`manifest.json`, `run.json` and the files `run.json` names with their digests (taken before any is
read), so a run directory that was moved still works. What it refuses is in
`test_cli_eval_report_refusals.py`.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from rinchen.eval.suites import Suite
from rinchen.manifest import store
from tests.support.eval_cli import invoke, robustness_directories
from tests.support.eval_report import shipped_config
from tests.support.eval_report_cli import (
    FIRST,
    SECOND,
    ReportWorkspace,
    copy_run,
    invoke_report,
    listing,
    make_report_workspace,
    sha256_of,
)
from tests.support.eval_report_golden import GOLDEN_REPORT_JSON, GOLDEN_REPORT_MD
from tests.support.no_network import no_network  # noqa: F401 - the fixture of the mark

pytestmark = pytest.mark.usefixtures("no_network")


@pytest.fixture
def workspace(tmp_path: Path) -> ReportWorkspace:
    return make_report_workspace(tmp_path)


def lines_for(out: Path) -> list[str]:
    return [
        f"report: {out / 'report.json'}",
        f"markdown: {out / 'report.md'}",
        f"manifest: {out / 'manifest.json'}",
        "targets: beats_baseline unmeasured",
        "targets: off_target unmeasured",
        "targets: numbers unmeasured",
        "targets: placeholders unmeasured",
    ]


def test_the_command_prints_the_three_paths_and_a_line_for_every_target(
    workspace: ReportWorkspace,
) -> None:
    result = invoke_report(workspace)

    assert result.exit_code == 0, result.output
    assert result.stderr == ""
    assert result.stdout.splitlines() == lines_for(workspace.report_out)


def test_the_command_writes_the_two_reports_and_the_manifest_and_nothing_else(
    workspace: ReportWorkspace,
) -> None:
    assert invoke_report(workspace).exit_code == 0

    assert listing(workspace.report_out) == [
        "manifest.json",
        "report.json",
        "report.md",
    ]


def test_the_report_and_the_markdown_are_the_golden_files_byte_for_byte(
    workspace: ReportWorkspace,
) -> None:
    assert invoke_report(workspace).exit_code == 0

    assert (
        workspace.report_out / "report.json"
    ).read_bytes() == GOLDEN_REPORT_JSON.read_bytes()
    assert (
        workspace.report_out / "report.md"
    ).read_bytes() == GOLDEN_REPORT_MD.read_bytes()


def test_two_runs_of_the_command_give_equal_bytes(
    workspace: ReportWorkspace, tmp_path: Path
) -> None:
    assert invoke_report(workspace).exit_code == 0
    again = tmp_path / "again"
    assert invoke_report(workspace, out_dir=again).exit_code == 0

    for name in ("report.json", "report.md"):
        assert (again / name).read_bytes() == (workspace.report_out / name).read_bytes()


def test_running_it_again_into_the_same_directory_replaces_the_files(
    workspace: ReportWorkspace,
) -> None:
    assert invoke_report(workspace).exit_code == 0
    first = (workspace.report_out / "report.json").read_bytes()

    again = invoke_report(workspace)

    assert again.exit_code == 0, again.output
    assert (workspace.report_out / "report.json").read_bytes() == first
    assert listing(workspace.report_out) == [
        "manifest.json",
        "report.json",
        "report.md",
    ]


def test_the_order_of_the_runs_on_the_command_line_changes_nothing(
    workspace: ReportWorkspace, tmp_path: Path
) -> None:
    assert invoke_report(workspace).exit_code == 0
    reversed_out = tmp_path / "reversed"

    result = invoke_report(
        workspace, runs=[workspace.beta, workspace.alpha], out_dir=reversed_out
    )

    assert result.exit_code == 0, result.output
    assert (
        reversed_out / "report.json"
    ).read_bytes() == GOLDEN_REPORT_JSON.read_bytes()
    assert (reversed_out / "report.md").read_bytes() == GOLDEN_REPORT_MD.read_bytes()


def test_a_run_directory_that_was_moved_still_works(
    workspace: ReportWorkspace, tmp_path: Path
) -> None:
    moved_alpha = copy_run(workspace.alpha, tmp_path / "elsewhere" / "a")
    moved_beta = copy_run(workspace.beta, tmp_path / "elsewhere" / "b")
    out = tmp_path / "moved-report"

    result = invoke_report(workspace, runs=[moved_alpha, moved_beta], out_dir=out)

    assert result.exit_code == 0, result.output
    assert (out / "report.json").read_bytes() == GOLDEN_REPORT_JSON.read_bytes()


def test_the_suite_is_matched_by_its_digest_and_not_by_its_name_or_path(
    workspace: ReportWorkspace, tmp_path: Path
) -> None:
    renamed = tmp_path / "another-name.json"
    renamed.write_bytes(workspace.suite.read_bytes())

    result = invoke_report(workspace, suites=[renamed])

    assert result.exit_code == 0, result.output
    assert (
        workspace.report_out / "report.json"
    ).read_bytes() == GOLDEN_REPORT_JSON.read_bytes()


def test_a_suite_that_is_not_in_canonical_form_is_matched_by_the_bytes_of_the_file(
    workspace: ReportWorkspace, tmp_path: Path
) -> None:
    # The run named the SHA-256 of the file it read: only that file matches, whatever it parses to.
    document = json.loads(workspace.suite.read_text("utf-8"))
    spaced = tmp_path / "spaced.json"
    spaced.write_text(json.dumps(document, ensure_ascii=False) + "\n", "utf-8")

    result = invoke_report(workspace, suites=[spaced])

    assert result.exit_code == 3, result.output
    assert not workspace.report_out.exists()


# ---- the manifest ------------------------------------------------------------------------------


def manifest_of(workspace: ReportWorkspace) -> Any:
    manifest = store.read(workspace.report_out)
    assert manifest is not None
    return manifest


def test_the_manifest_is_of_the_stage_and_says_what_it_measured(
    workspace: ReportWorkspace,
) -> None:
    assert invoke_report(workspace).exit_code == 0

    manifest = manifest_of(workspace)

    assert manifest.stage == "eval_report"
    assert manifest.status == {"report": "ok", "evidence": "fixture"}
    assert manifest.metrics == {
        "results": 4.0,
        "comparisons": 4.0,
        "systems": 2.0,
        "suites": 1.0,
    }
    assert manifest.seed == shipped_config().bootstrap.seed
    assert [Path(item.path).name for item in manifest.outputs] == [
        "report.json",
        "report.md",
    ]
    for item in manifest.outputs:
        assert item.sha256 == sha256_of(Path(item.path))
        assert Path(item.path).parent == workspace.report_out


def test_the_manifest_lists_every_file_it_read_with_the_digest_it_has(
    workspace: ReportWorkspace,
) -> None:
    assert invoke_report(workspace).exit_code == 0

    inputs = manifest_of(workspace).inputs

    listed = {item.path: item.sha256 for item in inputs}
    assert len(inputs) == len(listed) == 12
    for path, digest in listed.items():
        assert digest == sha256_of(Path(path)), path
    names = sorted(Path(path).name for path in listed)
    assert names == sorted(
        [
            "golden_suite.json",
            "metrics.yaml",
            *["manifest.json"] * 2,
            *["run.json"] * 2,
            *["hypotheses.jsonl"] * 2,
            *["result.json"] * 4,
        ]
    )
    assert Path(inputs[0].path) == workspace.suite
    assert Path(inputs[0].path).name == "golden_suite.json"
    for run in (workspace.alpha, workspace.beta):
        assert str(run / "run.json") in listed
        assert str(run / FIRST / "result.json") in listed
        assert str(run / SECOND / "result.json") in listed
        assert str(run / "hypotheses.jsonl") in listed


def test_the_manifest_digest_of_the_options_depends_on_the_baseline(
    workspace: ReportWorkspace, tmp_path: Path
) -> None:
    assert invoke_report(workspace).exit_code == 0
    first = manifest_of(workspace).config_sha256
    other = tmp_path / "other"
    assert invoke_report(workspace, out_dir=other, baseline="beta").exit_code == 0

    assert store.read(other).config_sha256 != first  # type: ignore[union-attr]
    assert len(first) == 64


def test_the_manifest_is_written_last(
    workspace: ReportWorkspace, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tests.support.robust_cli import record_replaces

    names = record_replaces(monkeypatch)

    assert invoke_report(workspace).exit_code == 0

    assert names[-3:] == ["report.json", "report.md", "manifest.json"]


# ---- the options -------------------------------------------------------------------------------


def test_without_a_baseline_there_is_no_comparison_and_the_golden_differs_only_there(
    workspace: ReportWorkspace,
) -> None:
    result = invoke_report(workspace, baseline=None)

    assert result.exit_code == 0, result.output
    document = json.loads((workspace.report_out / "report.json").read_text("utf-8"))
    assert document["comparisons"] == []
    assert len(document["results"]) == 4


def test_the_baseline_of_the_metrics_configuration_stands_in_for_the_option(
    workspace: ReportWorkspace, tmp_path: Path
) -> None:
    config = tmp_path / "config.yaml"
    config.write_text(
        workspace.ws.metrics.read_text("utf-8").replace(
            "baseline_system: null", "baseline_system: alpha"
        ),
        "utf-8",
    )

    result = invoke_report(workspace, "--metrics-config", str(config), baseline=None)

    assert result.exit_code == 0, result.output
    document = json.loads((workspace.report_out / "report.json").read_text("utf-8"))
    assert {c["baseline"] for c in document["comparisons"]} == {"alpha"}
    # that configuration holds no target, so no `targets:` line is printed
    assert not any(line.startswith("targets:") for line in result.stdout.splitlines())


def test_the_report_of_a_run_that_attached_a_robustness_block_carries_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # the block of D12's run of the same system, attached by `rinchen eval run --robustness`
    inner = make_report_workspace(tmp_path)
    directories = robustness_directories(inner.ws, monkeypatch)
    out = tmp_path / "with-robustness"
    made = invoke(
        inner.ws.hypotheses_args(
            "--robustness", str(directories.named_alpha), out_dir=out
        )
    )
    assert made.exit_code == 0, made.output

    result = invoke_report(
        inner, runs=[out / "golden" / "alpha"], out_dir=tmp_path / "rep", baseline=None
    )

    assert result.exit_code == 0, result.output
    document = json.loads((tmp_path / "rep" / "report.json").read_text("utf-8"))
    entry = document["robustness"]["alpha"]
    assert sorted(entry) == ["block", "coverage", "source"]
    assert entry["source"]["binding"] == "name"
    assert sorted(entry["block"]) == [
        "entity_acc",
        "mong_source_retention",
        "numbers_acc",
        "placeholder_acc",
        "script_purity",
        "term_acc",
    ]


# ---- the files of an earlier report -------------------------------------------------------------


def test_the_files_the_old_manifest_lists_go_and_no_other_file_is_touched(
    workspace: ReportWorkspace,
) -> None:
    assert invoke_report(workspace).exit_code == 0
    out = workspace.report_out
    (out / "stale.md").write_text("an output of an earlier version", "utf-8")
    (out / "notes.txt").write_text("the operator's own file", "utf-8")
    manifest = store.read(out)
    assert manifest is not None
    manifest.outputs = [
        *manifest.outputs,
        manifest.outputs[0].model_copy(update={"path": str(out / "stale.md")}),
    ]
    store.write(out, manifest)

    result = invoke_report(workspace)

    assert result.exit_code == 0, result.output
    assert listing(out) == ["manifest.json", "notes.txt", "report.json", "report.md"]
    assert (out / "notes.txt").read_text("utf-8") == "the operator's own file"


def test_a_manifest_that_names_a_path_outside_the_directory_removes_nothing_outside_it(
    workspace: ReportWorkspace, tmp_path: Path
) -> None:
    assert invoke_report(workspace).exit_code == 0
    out = workspace.report_out
    outside = tmp_path / "outside.txt"
    outside.write_text("not ours", "utf-8")
    manifest = store.read(out)
    assert manifest is not None
    manifest.outputs = [
        manifest.outputs[0].model_copy(update={"path": str(outside)}),
        *manifest.outputs,
    ]
    store.write(out, manifest)

    result = invoke_report(workspace)

    assert result.exit_code == 0, result.output
    assert outside.read_text("utf-8") == "not ours"


def test_the_help_names_the_options(workspace: ReportWorkspace) -> None:
    result = invoke(["eval", "report", "--help"])

    assert result.exit_code == 0
    for option in ("--run", "--suite", "--out-dir", "--baseline", "--metrics-config"):
        assert option in result.stdout


# ---- the evidence ------------------------------------------------------------------------------


def benchmark_runs(workspace: ReportWorkspace, tmp_path: Path) -> tuple[Path, Path]:
    """A run of `alpha` over a benchmark suite (the golden one as FLORES+ devtest, named
    `bench`) made by `rinchen eval run`: its suite file and its run directory."""
    suite = Suite.from_bytes(workspace.suite.read_bytes())
    bench = replace(suite, name="bench", kind="benchmark", benchmark="flores_plus")
    path = tmp_path / "bench_suite.json"
    path.write_bytes(bench.to_bytes())
    out = tmp_path / "bench-runs"
    made = invoke(workspace.ws.hypotheses_args(suite=path, out_dir=out))
    assert made.exit_code == 0, made.output
    return path, out / "bench" / "alpha"


def test_a_benchmark_suite_makes_the_evidence_measured_and_the_banner_says_so(
    workspace: ReportWorkspace, tmp_path: Path
) -> None:
    suite, run = benchmark_runs(workspace, tmp_path)

    result = invoke_report(workspace, runs=[run], suites=[suite], baseline=None)

    assert result.exit_code == 0, result.output
    assert manifest_of(workspace).status == {"report": "ok", "evidence": "measured"}
    banner = (workspace.report_out / "report.md").read_text("utf-8").splitlines()[2]
    assert banner.startswith("Evidence: suites of the kind benchmark. ")


def test_a_benchmark_beside_a_fixture_makes_the_evidence_mixed(
    workspace: ReportWorkspace, tmp_path: Path
) -> None:
    suite, run = benchmark_runs(workspace, tmp_path)

    result = invoke_report(
        workspace,
        runs=[workspace.alpha, run],
        suites=[workspace.suite, suite],
        baseline=None,
    )

    assert result.exit_code == 0, result.output
    assert manifest_of(workspace).status == {"report": "ok", "evidence": "mixed"}
    document = json.loads((workspace.report_out / "report.json").read_text("utf-8"))
    assert document["evidence"] == {
        "kinds": ["benchmark", "fixture"],
        "fixture_only": False,
    }
