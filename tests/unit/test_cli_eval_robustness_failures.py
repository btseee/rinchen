"""`rinchen eval robustness`: what a run that fails after the options and the inputs were accepted
does and leaves behind. An error of the engine has its exit code (2 for a target the model cannot
write, 3, 4 for CTranslate2) and leaves nothing and the engine closed; an interrupt is 130; a file
that cannot be written is exit 1 with no manifest and the old results gone; and the files appear
in the order hypotheses, result, report, with the manifest LAST.

What a run refuses before any of that is in `test_cli_eval_robustness_refusals.py`.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any

import pytest

from rinchen.errors import ConfigError, DataError, UpstreamError
from rinchen.manifest import store
from tests.support.launch_cli import assert_one_error, snapshot
from tests.support.robust_cli import (
    RobustWorkspace,
    echo_answers,
    fail_replace,
    install,
    invoke,
    make_workspace,
    record_replaces,
)


@pytest.fixture
def workspace(tmp_path: Path) -> RobustWorkspace:
    return make_workspace(tmp_path)


# --- the engine --------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("error", "code"),
    [
        (ConfigError("the model does not write the target"), 2),
        (DataError("a text holds a lone surrogate"), 3),
        (UpstreamError("CTranslate2 failed"), 4),
    ],
)
def test_an_error_of_the_engine_has_its_exit_code_and_leaves_nothing_and_the_engine_closed(
    workspace: RobustWorkspace,
    monkeypatch: pytest.MonkeyPatch,
    error: Exception,
    code: int,
) -> None:
    engine = workspace.engine(error=error)
    install(monkeypatch, engine)

    result = invoke(workspace.model_args())

    assert_one_error(result, code, str(error))
    assert result.stdout == ""
    assert engine.closed == 1
    assert not workspace.out_dir.exists()


@pytest.mark.parametrize(
    ("error", "code"),
    [(ConfigError("no such device"), 2), (DataError("no such file"), 3)],
)
def test_an_engine_that_cannot_be_opened_is_refused_with_nothing_written(
    workspace: RobustWorkspace,
    monkeypatch: pytest.MonkeyPatch,
    error: Exception,
    code: int,
) -> None:
    seams = install(monkeypatch, workspace.engine(), open_error=error)

    result = invoke(workspace.model_args())

    assert_one_error(result, code, str(error))
    assert seams.events == ["open"]
    assert not workspace.out_dir.exists()


def test_a_converter_that_cannot_be_loaded_is_refused_and_the_engine_is_never_opened(
    workspace: RobustWorkspace, monkeypatch: pytest.MonkeyPatch
) -> None:
    seams = install(
        monkeypatch, workspace.engine(), load_error=DataError("converter model missing")
    )

    result = invoke(workspace.model_args(*workspace.converter_args()))

    assert_one_error(result, 3, "converter model missing")
    assert seams.events == ["open_converter"]
    assert not workspace.out_dir.exists()


def test_an_engine_that_answers_fewer_translations_than_requests_is_an_upstream_error(
    workspace: RobustWorkspace, monkeypatch: pytest.MonkeyPatch
) -> None:
    engine = workspace.engine(answer=lambda rows: echo_answers(rows)[:-1])
    install(monkeypatch, engine)

    result = invoke(workspace.model_args())

    assert_one_error(result, 4, "the engine answered 32 translations for 33 requests")
    assert engine.closed == 1
    assert not workspace.out_dir.exists()


@pytest.mark.parametrize("which", ["model_sha256", "manifest_sha256"])
def test_an_engine_that_loaded_other_files_than_the_ones_hashed_is_refused(
    workspace: RobustWorkspace, monkeypatch: pytest.MonkeyPatch, which: str
) -> None:
    engine = workspace.engine()
    engine.info = dataclasses.replace(engine.info, **{which: "f" * 64})
    install(monkeypatch, engine)

    result = invoke(workspace.model_args())

    assert_one_error(result, 2, "the engine loaded")
    assert engine.closed == 1
    assert engine.calls == []
    assert not workspace.out_dir.exists()


def test_an_interrupt_during_the_run_exits_130_closes_the_engine_and_writes_nothing(
    workspace: RobustWorkspace, monkeypatch: pytest.MonkeyPatch
) -> None:
    engine = workspace.engine(error=KeyboardInterrupt())
    install(monkeypatch, engine)

    result = invoke(workspace.model_args())

    assert result.exit_code == 130, result.output
    assert engine.closed == 1
    assert not workspace.out_dir.exists()


def test_an_interrupt_leaves_the_results_of_the_last_run_as_they_were(
    workspace: RobustWorkspace, monkeypatch: pytest.MonkeyPatch
) -> None:
    install(monkeypatch, workspace.engine())
    assert invoke(workspace.model_args()).exit_code == 0
    before = snapshot(workspace.out_dir)
    install(monkeypatch, workspace.engine(error=KeyboardInterrupt()))

    result = invoke(workspace.model_args())

    assert result.exit_code == 130
    assert snapshot(workspace.out_dir) == before


def test_a_failed_engine_run_leaves_the_last_results_and_their_manifest_untouched(
    workspace: RobustWorkspace, monkeypatch: pytest.MonkeyPatch
) -> None:
    install(monkeypatch, workspace.engine())
    assert invoke(workspace.model_args()).exit_code == 0
    before = snapshot(workspace.out_dir)
    install(monkeypatch, workspace.engine(error=UpstreamError("CTranslate2 failed")))

    result = invoke(workspace.model_args())

    assert result.exit_code == 4
    assert snapshot(workspace.out_dir) == before


# --- the writes --------------------------------------------------------------------------------


@pytest.mark.parametrize("name", ["hypotheses.jsonl", "robustness.json", "report.json"])
def test_a_file_that_cannot_be_written_exits_1_with_no_manifest_and_the_old_results_gone(
    workspace: RobustWorkspace, monkeypatch: pytest.MonkeyPatch, name: str
) -> None:
    install(monkeypatch, workspace.engine())
    assert invoke(workspace.model_args()).exit_code == 0
    assert workspace.manifest.is_file()
    engine = workspace.engine()
    install(monkeypatch, engine)
    fail_replace(monkeypatch, name)

    result = invoke(workspace.model_args())

    assert_one_error(
        result, 1, f"cannot write the robustness result under {workspace.out_dir}"
    )
    assert "disk full" in result.stderr
    assert result.stdout == ""
    assert not workspace.manifest.exists()
    assert not (workspace.out_dir / name).exists()
    assert engine.closed == 1


def test_a_manifest_that_cannot_be_written_exits_1_and_leaves_none(
    workspace: RobustWorkspace, monkeypatch: pytest.MonkeyPatch
) -> None:
    install(monkeypatch, workspace.engine())
    fail_replace(monkeypatch, "manifest.json")

    result = invoke(workspace.model_args())

    assert_one_error(
        result, 1, f"cannot write the robustness result under {workspace.out_dir}"
    )
    assert result.stdout == ""
    assert not workspace.manifest.exists()
    assert workspace.result.is_file()


def test_a_failed_second_run_leaves_no_manifest_and_a_good_one_replaces_every_file(
    workspace: RobustWorkspace, monkeypatch: pytest.MonkeyPatch
) -> None:
    install(monkeypatch, workspace.engine())
    assert invoke(workspace.model_args()).exit_code == 0
    first = store.read(workspace.out_dir)
    assert first is not None
    with monkeypatch.context() as failing:
        fail_replace(failing, "report.json")
        assert invoke(workspace.model_args()).exit_code == 1
    assert not workspace.manifest.exists()

    result = invoke(workspace.model_args("--beam-size", "2"))

    assert result.exit_code == 0, result.output
    second = store.read(workspace.out_dir)
    assert second is not None
    assert second.config_sha256 != first.config_sha256
    assert sorted(path.name for path in workspace.out_dir.iterdir()) == [
        "hypotheses.jsonl",
        "manifest.json",
        "report.json",
        "robustness.json",
    ]


def test_a_hypotheses_run_removes_the_hypotheses_file_of_an_earlier_model_run(
    workspace: RobustWorkspace, monkeypatch: pytest.MonkeyPatch
) -> None:
    install(monkeypatch, workspace.engine())
    assert invoke(workspace.model_args()).exit_code == 0
    assert workspace.hypotheses_out.is_file()

    result = invoke(workspace.hypotheses_args())

    assert result.exit_code == 0, result.output
    assert not workspace.hypotheses_out.exists()


# --- the manifest is last ----------------------------------------------------------------------


def test_the_files_appear_in_the_order_hypotheses_result_report_and_the_manifest_last(
    workspace: RobustWorkspace, monkeypatch: pytest.MonkeyPatch
) -> None:
    install(monkeypatch, workspace.engine())
    names = record_replaces(monkeypatch)

    result = invoke(workspace.model_args())

    assert result.exit_code == 0, result.output
    assert names == [
        "hypotheses.jsonl",
        "robustness.json",
        "report.json",
        "manifest.json",
    ]


def test_a_hypotheses_run_writes_no_hypotheses_file_and_the_manifest_last(
    workspace: RobustWorkspace, monkeypatch: pytest.MonkeyPatch
) -> None:
    names = record_replaces(monkeypatch)

    result = invoke(workspace.hypotheses_args())

    assert result.exit_code == 0, result.output
    assert names == ["robustness.json", "report.json", "manifest.json"]


def test_the_manifest_of_the_last_run_is_removed_before_the_first_file_is_written(
    workspace: RobustWorkspace, monkeypatch: pytest.MonkeyPatch
) -> None:
    install(monkeypatch, workspace.engine())
    assert invoke(workspace.model_args()).exit_code == 0
    seen: list[bool] = []
    real = Path.replace

    def replacing(self: Path, target: Any) -> Any:
        seen.append(workspace.manifest.exists())
        return real(self, target)

    monkeypatch.setattr(Path, "replace", replacing)

    assert invoke(workspace.model_args()).exit_code == 0

    assert seen[:3] == [False, False, False]
