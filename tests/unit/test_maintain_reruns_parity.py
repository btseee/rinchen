"""The parity rerun of `rinchen.maintain.reruns` (D14, spec M8): `rinchen export parity` over an
export, a checkpoint and corpus files that the operator gives.

Against the recording `FakeRunner` of `tests/support/maintain_fakes.py`, so that no process
starts: a test scripts the exit code and the `parity.json` the command would have left and reads
the state, the reason and the three figures back. Without an export the rerun is skipped and
nothing runs; exit 0 needs the figures; exit 4 is a measured miss and keeps them; any other code,
a timeout and a command that did not start are failures; a `parity.json` of an earlier run never
counts; paths that reach the child are absolute.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from rinchen.config.maintain import RerunConfig
from rinchen.errors import ConfigError
from rinchen.maintain.commands import HIDDEN_ENVIRONMENT, CommandResult
from rinchen.maintain.reruns import (
    ExportInputs,
    RerunResult,
    parity_argv,
    run_parity_rerun,
)
from rinchen.maintain.states import (
    FAIL_NO_FIGURES,
    SKIP_NO_EXPORT,
    fail_command_exit,
    fail_timeout,
)
from tests.support.maintain_fakes import FakeRunner, parity_rerun, pytest_rerun
from tests.support.maintain_project import PASSES_A, make_project
from tests.support.no_network import no_network  # noqa: F401 - the fixture of the mark

pytestmark = pytest.mark.usefixtures("no_network")

PARITY_JSON = {
    "exact": 200,
    "sentences": 200,
    "mean_abs_delta": 7.19e-07,
    "max_abs_delta": 3.43e-06,
    "passes": True,
    "quantization": "float32",
    "placement": "src",
}
FIGURES = {"exact": 200, "sentences": 200, "mean_abs_delta": 7.19e-07}


# ---- the parity rerun ---------------------------------------------------------------------------------


def inputs_in(tmp_path: Path) -> ExportInputs:
    return ExportInputs(
        export=tmp_path / "export",
        checkpoint=tmp_path / "ckpt",
        corpora=(tmp_path / "a.parquet", tmp_path / "b.parquet"),
    )


def parity_in(
    tmp_path: Path,
    runner: FakeRunner,
    *,
    with_inputs: bool = True,
    rerun: RerunConfig | None = None,
) -> tuple[RerunResult, Path, Path]:
    project = make_project(tmp_path)
    out = tmp_path / "out"
    result = run_parity_rerun(
        parity_rerun() if rerun is None else rerun,
        runner=runner,
        python="/opt/python/bin/python",
        project_dir=project,
        out_dir=out,
        parent_env={"KEEP": "me"},
        inputs=inputs_in(tmp_path) if with_inputs else None,
    )
    return result, project, out


def parity_file(out: Path) -> Path:
    return out / "parity_export" / "parity.json"


def test_without_an_export_the_rerun_is_skipped_and_nothing_runs(
    tmp_path: Path,
) -> None:
    runner = FakeRunner()

    result, _, out = parity_in(tmp_path, runner, with_inputs=False)

    assert (result.state, result.reason) == ("skipped", SKIP_NO_EXPORT)
    assert (result.id, result.kind, result.cadence) == (
        "parity",
        "export_parity",
        "nightly",
    )
    assert (result.tests, result.figures) == (None, None)
    assert runner.calls == []
    assert not out.exists()


def test_the_parity_command_is_run_in_the_project_with_the_hidden_environment(
    tmp_path: Path,
) -> None:
    rerun = parity_rerun(timeout=640)
    runner = FakeRunner().script(
        CommandResult(0, False),
        files={parity_file(tmp_path / "out"): json.dumps(PARITY_JSON).encode()},
    )

    result, project, out = parity_in(tmp_path, runner, rerun=rerun)

    (call,) = runner.calls
    assert call.cwd == project
    assert call.timeout == 640
    assert call.argv == tuple(
        parity_argv(
            "/opt/python/bin/python", rerun, inputs_in(tmp_path), out / "parity_export"
        )
    )
    assert call.argv[call.argv.index("--timeout") + 1] == "610"
    assert call.log == out / "logs" / "parity.log"
    assert call.env == {"KEEP": "me", **HIDDEN_ENVIRONMENT}
    assert (result.state, result.reason) == ("ok", None)


def passing_runner(out: Path, document: object = PARITY_JSON) -> FakeRunner:
    return FakeRunner().script(
        CommandResult(0, False),
        files={parity_file(out): json.dumps(document).encode()},
    )


def test_exit_zero_with_the_figures_is_ok_and_carries_them(tmp_path: Path) -> None:
    out = tmp_path / "out"

    result, _, _ = parity_in(tmp_path, passing_runner(out))

    assert (result.state, result.reason) == ("ok", None)
    assert result.figures == FIGURES
    assert [type(value) for value in (result.figures or {}).values()] == [
        int,
        int,
        float,
    ]
    assert result.tests is None


def test_the_figures_that_are_not_the_three_are_left_out(tmp_path: Path) -> None:
    out = tmp_path / "out"

    result, _, _ = parity_in(tmp_path, passing_runner(out, PARITY_JSON))

    assert set(result.figures or {}) == {"exact", "sentences", "mean_abs_delta"}


def test_a_mean_delta_written_as_an_integer_is_a_float(tmp_path: Path) -> None:
    out = tmp_path / "out"
    document = {**PARITY_JSON, "mean_abs_delta": 0}

    result, _, _ = parity_in(tmp_path, passing_runner(out, document))

    assert result.figures is not None
    assert result.figures["mean_abs_delta"] == 0.0
    assert type(result.figures["mean_abs_delta"]) is float


def test_exit_four_is_a_measured_miss_and_keeps_the_figures_when_they_are_there(
    tmp_path: Path,
) -> None:
    out = tmp_path / "out"
    document = {**PARITY_JSON, "exact": 150}
    runner = FakeRunner().script(
        CommandResult(4, False), files={parity_file(out): json.dumps(document).encode()}
    )

    result, _, _ = parity_in(tmp_path, runner)

    assert (result.state, result.reason) == ("fail", fail_command_exit(4))
    assert result.figures == {**FIGURES, "exact": 150}


def test_exit_four_without_figures_is_a_failure_with_none(tmp_path: Path) -> None:
    result, _, _ = parity_in(tmp_path, FakeRunner().script(CommandResult(4, False)))

    assert (result.state, result.reason, result.figures) == (
        "fail",
        fail_command_exit(4),
        None,
    )


@pytest.mark.parametrize("code", [1, 2, 3, 130, -9])
def test_any_other_exit_code_is_a_failure(tmp_path: Path, code: int) -> None:
    result, _, _ = parity_in(tmp_path, FakeRunner().script(CommandResult(code, False)))

    assert (result.state, result.reason) == ("fail", fail_command_exit(code))


def test_exit_zero_without_a_parity_json_is_a_failure(tmp_path: Path) -> None:
    result, _, _ = parity_in(tmp_path, FakeRunner().script(CommandResult(0, False)))

    assert (result.state, result.reason, result.figures) == (
        "fail",
        FAIL_NO_FIGURES,
        None,
    )


@pytest.mark.parametrize(
    "content",
    [
        b"not json",
        b"",
        b"\xff\xfe",
        b"[]",
        b"null",
        b'"text"',
        json.dumps({"exact": 200, "sentences": 200}).encode(),
        json.dumps({"exact": 200, "mean_abs_delta": 0.1}).encode(),
        json.dumps({"sentences": 200, "mean_abs_delta": 0.1}).encode(),
        json.dumps({"exact": "200", "sentences": 200, "mean_abs_delta": 0.1}).encode(),
        json.dumps({"exact": 200.0, "sentences": 200, "mean_abs_delta": 0.1}).encode(),
        json.dumps({"exact": True, "sentences": 200, "mean_abs_delta": 0.1}).encode(),
        json.dumps({"exact": 200, "sentences": True, "mean_abs_delta": 0.1}).encode(),
        json.dumps({"exact": 200, "sentences": 200, "mean_abs_delta": True}).encode(),
        json.dumps({"exact": 200, "sentences": 200, "mean_abs_delta": "0.1"}).encode(),
        json.dumps({"exact": 200, "sentences": 200, "mean_abs_delta": None}).encode(),
        json.dumps({"exact": -1, "sentences": 200, "mean_abs_delta": 0.1}).encode(),
        json.dumps({"exact": 200, "sentences": -2, "mean_abs_delta": 0.1}).encode(),
        json.dumps({"exact": 200, "sentences": 200, "mean_abs_delta": -0.1}).encode(),
        b'{"exact": 200, "sentences": 200, "mean_abs_delta": NaN}',
        b'{"exact": 200, "sentences": 200, "mean_abs_delta": Infinity}',
    ],
    ids=[
        "not json",
        "empty",
        "not utf-8",
        "a list",
        "null",
        "a string",
        "no mean",
        "no sentence count",
        "no exact count",
        "exact as text",
        "exact as a float",
        "exact as a bool",
        "sentences as a bool",
        "mean as a bool",
        "mean as text",
        "mean as null",
        "a negative exact",
        "a negative sentence count",
        "a negative mean",
        "a mean that is NaN",
        "a mean that is infinite",
    ],
)
def test_a_parity_json_that_does_not_hold_the_three_figures_is_no_figures(
    tmp_path: Path, content: bytes
) -> None:
    out = tmp_path / "out"
    runner = FakeRunner().script(
        CommandResult(0, False), files={parity_file(out): content}
    )

    result, _, _ = parity_in(tmp_path, runner)

    assert (result.state, result.reason, result.figures) == (
        "fail",
        FAIL_NO_FIGURES,
        None,
    )


def test_a_timeout_is_a_failure_and_the_figures_are_not_read(tmp_path: Path) -> None:
    out = tmp_path / "out"
    runner = FakeRunner().script(
        CommandResult(None, True),
        files={parity_file(out): json.dumps(PARITY_JSON).encode()},
    )

    result, _, _ = parity_in(tmp_path, runner, rerun=parity_rerun(timeout=700))

    assert (result.state, result.reason, result.figures) == (
        "fail",
        fail_timeout(700),
        None,
    )


def test_a_command_that_did_not_start_is_a_failure(tmp_path: Path) -> None:
    runner = FakeRunner().script(CommandResult(None, False, False))

    result, _, _ = parity_in(tmp_path, runner)

    assert (result.state, result.reason) == ("fail", fail_command_exit(-1))


def test_a_parity_json_of_an_earlier_run_is_removed_and_never_counts(
    tmp_path: Path,
) -> None:
    out = tmp_path / "out"
    stale = parity_file(out)
    stale.parent.mkdir(parents=True)
    stale.write_text(json.dumps(PARITY_JSON))
    existed_at_the_call: list[bool] = []

    def look(argv: object) -> bool:
        existed_at_the_call.append(stale.exists())
        return True

    ok_run = FakeRunner().script(CommandResult(0, False), when=look)

    result, _, _ = parity_in(tmp_path, ok_run)

    assert existed_at_the_call == [False]
    assert (result.state, result.reason, result.figures) == (
        "fail",
        FAIL_NO_FIGURES,
        None,
    )


def test_relative_paths_are_made_absolute_for_the_child(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project = make_project(tmp_path)
    monkeypatch.chdir(tmp_path)
    runner = FakeRunner().script(CommandResult(0, False))

    run_parity_rerun(
        parity_rerun(),
        runner=runner,
        python="py",
        project_dir=project,
        out_dir=Path("out"),
        parent_env={},
        inputs=ExportInputs(
            Path("export"), Path("ckpt"), (Path("a.parquet"), Path("b.parquet"))
        ),
    )

    (call,) = runner.calls
    argv = list(call.argv)
    given = {
        "--checkpoint": tmp_path / "ckpt",
        "--export": tmp_path / "export",
        "--out-dir": tmp_path / "out" / "parity_export",
    }
    for option, path in given.items():
        assert argv[argv.index(option) + 1] == str(path)
    corpora = [argv[i + 1] for i, arg in enumerate(argv) if arg == "--corpus"]
    assert corpora == [str(tmp_path / "a.parquet"), str(tmp_path / "b.parquet")]
    assert call.log == tmp_path / "out" / "logs" / "parity.log"


def test_a_parity_rerun_is_refused_for_a_rerun_of_another_kind(tmp_path: Path) -> None:
    runner = FakeRunner()

    with pytest.raises(ConfigError, match="demo"):
        parity_in(tmp_path, runner, rerun=pytest_rerun(PASSES_A, id="demo"))

    assert runner.calls == []
