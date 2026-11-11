"""The device is hidden for the whole run of `rinchen maintain check` (D14 fix wave 1, finding M8).

The six doctor checks that the command makes run IN this process, and a child process gets
`CUDA_VISIBLE_DEVICES=""` from `child_environment`: the checks in process saw the caller's device.
The command now sets the variable to the empty text for the whole of its own run and puts the
caller's value back (or removes the variable) when the run ends, whatever the exit. These tests
run the command in process through typer's runner, so a value left behind would be seen by the
next test: each case ends by comparing the variable with what it was.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from pathlib import Path

import pytest

from rinchen.cli import maintain
from rinchen.cli.maintain_doctor import device_hidden
from rinchen.maintain.check_types import DoctorFinding
from tests.support.maintain_check import finding, golden_doctor
from tests.support.maintain_cli import install, invoke, make_workspace
from tests.support.no_network import no_network  # noqa: F401 - the fixture of the mark

pytestmark = pytest.mark.usefixtures("no_network")

VARIABLE = "CUDA_VISIBLE_DEVICES"
UNSET = None
START_STATES = ["0", "0,1", "", UNSET]


def start_with(monkeypatch: pytest.MonkeyPatch, value: str | None) -> None:
    if value is None:
        monkeypatch.delenv(VARIABLE, raising=False)
    else:
        monkeypatch.setenv(VARIABLE, value)


class SpyingDoctor:
    """A doctor seam that records the variable as it sees it, then reports the golden findings
    (or the ones it was given)."""

    def __init__(self, findings: Sequence[DoctorFinding] | None = None) -> None:
        self.seen: list[str | None] = []
        self.findings = list(golden_doctor().findings if findings is None else findings)

    def __call__(self) -> Sequence[DoctorFinding]:
        self.seen.append(os.environ.get(VARIABLE))
        return self.findings


class InterruptedDoctor(SpyingDoctor):
    def __call__(self) -> Sequence[DoctorFinding]:
        super().__call__()
        raise KeyboardInterrupt


# ---- the helper ---------------------------------------------------------------------------------------


@pytest.mark.parametrize("start", START_STATES)
def test_the_block_sees_an_empty_device_list_and_the_start_is_restored(
    monkeypatch: pytest.MonkeyPatch, start: str | None
) -> None:
    start_with(monkeypatch, start)

    with device_hidden():
        assert os.environ[VARIABLE] == ""

    assert os.environ.get(VARIABLE) == start
    assert (VARIABLE in os.environ) == (start is not None)


@pytest.mark.parametrize("start", START_STATES)
def test_the_start_is_restored_when_the_block_raises(
    monkeypatch: pytest.MonkeyPatch, start: str | None
) -> None:
    start_with(monkeypatch, start)

    with pytest.raises(RuntimeError), device_hidden():
        raise RuntimeError("the block failed")

    assert os.environ.get(VARIABLE) == start
    assert (VARIABLE in os.environ) == (start is not None)


def test_a_block_that_changes_the_variable_is_still_undone(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    start_with(monkeypatch, "2")

    with device_hidden():
        os.environ[VARIABLE] = "5"

    assert os.environ[VARIABLE] == "2"


def test_nested_blocks_restore_in_order(monkeypatch: pytest.MonkeyPatch) -> None:
    start_with(monkeypatch, "1")

    with device_hidden():
        with device_hidden():
            assert os.environ[VARIABLE] == ""
        assert os.environ[VARIABLE] == ""

    assert os.environ[VARIABLE] == "1"


def test_no_other_variable_is_touched(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("KEEP_ME", "yes")
    before = {name: value for name, value in os.environ.items() if name != VARIABLE}

    with device_hidden():
        during = {name: value for name, value in os.environ.items() if name != VARIABLE}

    assert before == during
    assert os.environ["KEEP_ME"] == "yes"


# ---- the command --------------------------------------------------------------------------------------


@pytest.mark.parametrize("start", START_STATES)
def test_the_doctor_checks_run_with_the_device_hidden_and_the_caller_gets_it_back(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, start: str | None
) -> None:
    ws = make_workspace(tmp_path)
    doctor = SpyingDoctor()
    install(monkeypatch, doctor=doctor)
    start_with(monkeypatch, start)

    result = invoke(ws.args())

    assert result.exit_code == 0, result.output
    assert doctor.seen == [""]
    assert os.environ.get(VARIABLE) == start
    assert (VARIABLE in os.environ) == (start is not None)


@pytest.mark.parametrize("start", START_STATES)
def test_the_installed_versions_and_the_runner_are_made_with_the_device_hidden(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, start: str | None
) -> None:
    ws = make_workspace(tmp_path)
    seen: list[tuple[str, str | None]] = []
    runner = install(monkeypatch)
    real_installed = maintain.installed

    def installed() -> object:
        seen.append(("installed", os.environ.get(VARIABLE)))
        return real_installed()

    monkeypatch.setattr(maintain, "installed", installed)
    monkeypatch.setattr(
        maintain,
        "make_runner",
        lambda: seen.append(("runner", os.environ.get(VARIABLE))) or runner,
    )
    start_with(monkeypatch, start)

    result = invoke(ws.args())

    assert result.exit_code == 0, result.output
    assert seen == [("runner", ""), ("installed", "")]


@pytest.mark.parametrize("start", START_STATES)
def test_the_value_is_back_after_a_failed_run_exit_4(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, start: str | None
) -> None:
    ws = make_workspace(tmp_path)
    doctor = SpyingDoctor([finding("D-PY", "fail")])
    install(monkeypatch, doctor=doctor)
    start_with(monkeypatch, start)

    result = invoke(ws.args())

    assert result.exit_code == 4, result.output
    assert doctor.seen == [""]
    assert os.environ.get(VARIABLE) == start
    assert (VARIABLE in os.environ) == (start is not None)


@pytest.mark.parametrize("start", START_STATES)
def test_the_value_is_back_after_a_usage_error_exit_2(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, start: str | None
) -> None:
    ws = make_workspace(tmp_path)
    install(monkeypatch)
    start_with(monkeypatch, start)

    result = invoke(ws.args("--cadence", "monthly"))

    assert result.exit_code == 2, result.output
    assert os.environ.get(VARIABLE) == start
    assert (VARIABLE in os.environ) == (start is not None)


@pytest.mark.parametrize("start", START_STATES)
def test_the_value_is_back_after_an_interrupt_exit_130(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, start: str | None
) -> None:
    ws = make_workspace(tmp_path)
    doctor = InterruptedDoctor()
    install(monkeypatch, doctor=doctor)
    start_with(monkeypatch, start)

    result = invoke(ws.args())

    assert result.exit_code == 130, result.output
    assert doctor.seen == [""]
    assert os.environ.get(VARIABLE) == start
    assert (VARIABLE in os.environ) == (start is not None)
