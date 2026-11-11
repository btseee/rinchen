"""Tests for `rinchen.maintain.commands` (D14, spec M10): the runner that every process of a
maintenance check goes through, the environment its children get and the bounded read of a log.

The runner tests start real children (the test interpreter running a few lines) because what they
pin is process behaviour: the exit code is handed back, the combined log, no shell, no standard
input, a session of its own, the timeout, and that a timeout or an interrupt takes away the whole
process group, a grandchild included. Every child is bounded by a timeout and runs with the GPU
hidden; none starts `uv`, the project's own tests, a model or a network connection.
"""

from __future__ import annotations

import dataclasses
import importlib
import os
import signal
import subprocess
import sys
import threading
import time
from collections.abc import Iterator, Mapping
from pathlib import Path
from types import MappingProxyType
from typing import Any

import pytest

from rinchen.maintain.commands import (
    HIDDEN_ENVIRONMENT,
    LOG_LIMIT,
    UV_ENVIRONMENT,
    CommandResult,
    CommandRunner,
    SubprocessRunner,
    child_environment,
    read_log,
)
from tests.support.maintain_project import kill_quietly, read_pids, wait_until_gone
from tests.support.no_network import no_network  # noqa: F401 - the fixture of the mark

pytestmark = pytest.mark.usefixtures("no_network")
process_test = pytest.mark.subprocess  # the test starts real child processes
linux_only = pytest.mark.skipif(
    sys.platform != "linux", reason="needs process groups, fork and /proc"
)

# Forks a grandchild, records both pids once it is set up and sleeps. The arguments are the pid
# file, the seconds to sleep (far longer than any timeout below, so a survivor is a defect and not
# luck) and whether the grandchild and the leader ignore SIGTERM ("1") or take its default ("0").
CHILD_WITH_GRANDCHILD = """
import os, signal, sys, time
pid_file, lifetime = sys.argv[1], float(sys.argv[2])
grandchild_ignores, leader_ignores = sys.argv[3] == "1", sys.argv[4] == "1"
grandchild = os.fork()
if grandchild == 0:
    if grandchild_ignores:
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
    time.sleep(lifetime)
    os._exit(0)
if leader_ignores:
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
with open(pid_file + ".tmp", "w") as handle:
    handle.write(f"{os.getpid()} {grandchild}")
os.replace(pid_file + ".tmp", pid_file)
print("child started", flush=True)
time.sleep(lifetime)
"""


def python(code: str, *args: str) -> list[str]:
    return [sys.executable, "-c", code, *args]


def with_grandchild(
    pid_file: Path,
    *,
    lifetime: int = 60,
    grandchild_ignores: bool = False,
    leader_ignores: bool = False,
) -> list[str]:
    return python(
        CHILD_WITH_GRANDCHILD,
        str(pid_file),
        str(lifetime),
        "1" if grandchild_ignores else "0",
        "1" if leader_ignores else "0",
    )


def log_of(tmp_path: Path) -> Path:
    return tmp_path / "logs" / "child.log"


def run(
    tmp_path: Path,
    argv: list[str],
    *,
    timeout: int = 30,
    env: Mapping[str, str] | None = None,
    cwd: Path | None = None,
) -> CommandResult:
    return SubprocessRunner().run(
        argv,
        cwd=tmp_path if cwd is None else cwd,
        env=child_environment(os.environ) if env is None else env,
        log=log_of(tmp_path),
        timeout=timeout,
    )


def survivors(pids: list[int], seconds: float = 5.0) -> list[int]:
    return [pid for pid in pids if not wait_until_gone(pid, seconds)]


# ---- the result and the interface ------------------------------------------------------------------


def test_a_result_is_frozen_and_slotted_and_started_by_default() -> None:
    result = CommandResult(exit_code=0, timed_out=False)

    assert result.started is True
    assert not hasattr(result, "__dict__")
    with pytest.raises(dataclasses.FrozenInstanceError):
        result.exit_code = 1  # type: ignore[misc]
    assert CommandResult(None, True) == CommandResult(
        exit_code=None, timed_out=True, started=True
    )
    assert CommandResult(None, False, False) != CommandResult(None, False)


def test_the_subprocess_runner_is_a_command_runner() -> None:
    runner: CommandRunner = SubprocessRunner()

    assert callable(runner.run)


# ---- the environment -------------------------------------------------------------------------------


def test_the_hidden_environment_is_the_four_switches_and_read_only() -> None:
    assert dict(HIDDEN_ENVIRONMENT) == {
        "CUDA_VISIBLE_DEVICES": "",
        "HF_HUB_OFFLINE": "1",
        "HF_DATASETS_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
    }
    with pytest.raises(TypeError):
        HIDDEN_ENVIRONMENT["CUDA_VISIBLE_DEVICES"] = "0"  # type: ignore[index]


def test_the_uv_environment_is_the_three_switches_and_read_only() -> None:
    assert dict(UV_ENVIRONMENT) == {
        "NO_COLOR": "1",
        "UV_NO_PROGRESS": "1",
        "UV_PYTHON_DOWNLOADS": "never",
    }
    with pytest.raises(TypeError):
        UV_ENVIRONMENT["NO_COLOR"] = "0"  # type: ignore[index]


def test_a_child_environment_is_the_parent_with_the_hidden_switches() -> None:
    parent = {"PATH": "/usr/bin", "KEEP": "me"}

    child = child_environment(parent)

    assert child == {"PATH": "/usr/bin", "KEEP": "me", **HIDDEN_ENVIRONMENT}
    assert "NO_COLOR" not in child
    assert not set(UV_ENVIRONMENT) & set(child)


def test_a_child_environment_for_uv_adds_the_uv_switches() -> None:
    child = child_environment({"PATH": "/usr/bin"}, uv=True)

    assert child == {
        "PATH": "/usr/bin",
        **HIDDEN_ENVIRONMENT,
        **UV_ENVIRONMENT,
    }


def test_the_hidden_switches_win_over_the_parents_values() -> None:
    parent = {
        "CUDA_VISIBLE_DEVICES": "0,1",
        "HF_HUB_OFFLINE": "0",
        "HF_DATASETS_OFFLINE": "0",
        "TRANSFORMERS_OFFLINE": "0",
        "NO_COLOR": "0",
    }

    child = child_environment(parent, uv=True)

    assert child["CUDA_VISIBLE_DEVICES"] == ""
    assert (
        child["HF_HUB_OFFLINE"],
        child["HF_DATASETS_OFFLINE"],
        child["TRANSFORMERS_OFFLINE"],
    ) == ("1", "1", "1")
    assert child["NO_COLOR"] == "1"


def test_a_child_environment_is_a_new_dict_and_the_parent_is_never_modified() -> None:
    parent = {"CUDA_VISIBLE_DEVICES": "0", "KEEP": "me"}

    child = child_environment(parent, uv=True)
    child["EXTRA"] = "x"

    assert child is not parent
    assert parent == {"CUDA_VISIBLE_DEVICES": "0", "KEEP": "me"}


def test_a_read_only_parent_is_accepted() -> None:
    parent = MappingProxyType({"KEEP": "me"})

    assert child_environment(parent)["KEEP"] == "me"


@process_test
def test_the_child_sees_the_hidden_switches_and_no_uv_switch(tmp_path: Path) -> None:
    code = (
        "import os\n"
        "print(repr(os.environ['CUDA_VISIBLE_DEVICES']), os.environ['HF_HUB_OFFLINE'],\n"
        "      os.environ['HF_DATASETS_OFFLINE'], os.environ['TRANSFORMERS_OFFLINE'],\n"
        "      'NO_COLOR' in os.environ, 'UV_NO_PROGRESS' in os.environ)\n"
    )
    parent = {"CUDA_VISIBLE_DEVICES": "0"}

    result = run(tmp_path, python(code), env=child_environment(parent))

    assert result == CommandResult(0, False)

    assert log_of(tmp_path).read_text() == "'' 1 1 1 False False\n"


@process_test
def test_the_child_sees_the_uv_switches_only_for_uv(tmp_path: Path) -> None:
    code = (
        "import os\n"
        "print(os.environ['NO_COLOR'], os.environ['UV_NO_PROGRESS'],\n"
        "      os.environ['UV_PYTHON_DOWNLOADS'], repr(os.environ['CUDA_VISIBLE_DEVICES']))\n"
    )

    run(tmp_path, python(code), env=child_environment({}, uv=True))

    assert log_of(tmp_path).read_text() == "1 1 never ''\n"


@process_test
def test_the_runner_hands_the_child_the_environment_it_was_given_and_no_other(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("RINCHEN_PARENT_ONLY", "leaks")
    code = (
        "import os; print(os.environ.get('RINCHEN_PARENT_ONLY'), os.environ['GIVEN'])"
    )

    run(tmp_path, python(code), env=child_environment({"GIVEN": "yes"}))

    assert log_of(tmp_path).read_text() == "None yes\n"


# ---- the exit code, the log and no shell ----------------------------------------------------------


@process_test
def test_the_exit_code_of_the_child_is_handed_back(tmp_path: Path) -> None:
    assert run(tmp_path, python("raise SystemExit(3)")) == CommandResult(3, False)
    assert run(tmp_path, python("pass")) == CommandResult(0, False)


@process_test
@linux_only
def test_a_child_ended_by_a_signal_gives_the_negative_signal_number(
    tmp_path: Path,
) -> None:
    code = "import os, signal; os.kill(os.getpid(), signal.SIGKILL)"

    assert run(tmp_path, python(code)) == CommandResult(-9, False)


@process_test
def test_standard_output_and_error_go_to_one_log_in_order(tmp_path: Path) -> None:
    code = (
        "import sys\n"
        "print('to stdout', flush=True)\n"
        "print('to stderr', file=sys.stderr, flush=True)\n"
        "print('to stdout again')\n"
    )

    run(tmp_path, python(code))

    assert log_of(tmp_path).read_text() == "to stdout\nto stderr\nto stdout again\n"


@process_test
def test_a_hundred_kilobytes_land_in_the_log_whole_and_the_read_is_bounded(
    tmp_path: Path,
) -> None:
    run(tmp_path, python("print('x' * 99_999)"))

    assert log_of(tmp_path).stat().st_size == 100_000
    assert len(read_log(log_of(tmp_path), limit=65536)) == 65536
    assert len(read_log(log_of(tmp_path))) == LOG_LIMIT == 65536
    assert len(read_log(log_of(tmp_path), limit=200_000)) == 100_000


@process_test
def test_the_log_is_created_with_its_directory_and_replaced_when_it_exists(
    tmp_path: Path,
) -> None:
    log = tmp_path / "does" / "not" / "exist" / "child.log"

    result = SubprocessRunner().run(
        python("print('first')"),
        cwd=tmp_path,
        env=child_environment(os.environ),
        log=log,
        timeout=30,
    )
    assert (result, log.read_text()) == (CommandResult(0, False), "first\n")

    SubprocessRunner().run(
        python("print('second')"),
        cwd=tmp_path,
        env=child_environment(os.environ),
        log=log,
        timeout=30,
    )
    assert log.read_text() == "second\n"


@process_test
def test_a_missing_executable_did_not_start_and_leaves_no_log_text(
    tmp_path: Path,
) -> None:
    log_of(tmp_path).parent.mkdir()
    log_of(tmp_path).write_text("left over from an earlier run")

    result = run(tmp_path, [str(tmp_path / "no-such-program"), "--version"])

    assert result == CommandResult(None, False, False)
    assert log_of(tmp_path).read_text() == ""


@process_test
def test_a_file_that_cannot_be_executed_did_not_start(tmp_path: Path) -> None:
    program = tmp_path / "not-executable"
    program.write_text("#!/bin/sh\nexit 0\n")
    program.chmod(0o644)

    assert run(tmp_path, [str(program)]) == CommandResult(None, False, False)


@process_test
def test_the_arguments_are_never_given_to_a_shell(tmp_path: Path) -> None:
    code = "import sys; print(sys.argv[1:])"
    arguments = ["; echo x", "$(echo y)", "`echo z`", "*", "a b", "$HOME"]

    run(tmp_path, python(code, *arguments))

    assert log_of(tmp_path).read_text() == f"{arguments}\n"


@pytest.fixture
def blocking_standard_input() -> Iterator[None]:
    """File descriptor 0 is, for the test, the read end of a pipe that nobody writes to: a child
    that inherited it would block on a read. (Under pytest descriptor 0 is the null device, which
    ends at once and would hide a child that inherits it.)"""
    read_end, write_end = os.pipe()
    saved = os.dup(0)
    os.dup2(read_end, 0)
    try:
        yield
    finally:
        os.dup2(saved, 0)
        for descriptor in (saved, read_end, write_end):
            os.close(descriptor)


@process_test
@pytest.mark.timeout(60)
@pytest.mark.usefixtures("blocking_standard_input")
def test_the_child_has_no_standard_input(tmp_path: Path) -> None:
    started = time.monotonic()

    result = run(
        tmp_path, python("import sys; print(repr(sys.stdin.read()))"), timeout=20
    )

    assert result == CommandResult(0, False)
    assert log_of(tmp_path).read_text() == "''\n"
    assert time.monotonic() - started < 15


@process_test
def test_the_child_runs_in_the_working_directory_it_is_given(tmp_path: Path) -> None:
    work = tmp_path / "work"
    work.mkdir()

    run(tmp_path, python("import os; print(os.getcwd())"), cwd=work)

    assert log_of(tmp_path).read_text() == f"{os.path.realpath(work)}\n"


@process_test
@linux_only
def test_the_child_leads_a_session_of_its_own(tmp_path: Path) -> None:
    # The session of its own is what lets a timeout signal the whole group and nothing else.
    run(tmp_path, python("import os; print(os.getpgid(0) == os.getpid())"))

    assert log_of(tmp_path).read_text() == "True\n"


@process_test
def test_the_text_a_child_wrote_before_a_timeout_stays_in_the_log(
    tmp_path: Path,
) -> None:
    code = "import time; print('before', flush=True); time.sleep(60)"

    result = run(tmp_path, python(code), timeout=2)

    assert result == CommandResult(None, True)
    assert log_of(tmp_path).read_text() == "before\n"


# ---- the timeout and the interrupt ---------------------------------------------------------------


@process_test
@linux_only
@pytest.mark.timeout(60)
def test_a_timeout_kills_the_child_and_its_grandchild(tmp_path: Path) -> None:
    pid_file = tmp_path / "pids"
    started = time.monotonic()

    try:
        result = run(tmp_path, with_grandchild(pid_file), timeout=2)
        elapsed = time.monotonic() - started
        pids = read_pids(pid_file)
        left = survivors(pids, 1.0)
    finally:
        if pid_file.exists():
            kill_quietly(read_pids(pid_file))

    assert result == CommandResult(None, True)
    assert elapsed >= 2, "the runner stopped the child before its bound"
    assert elapsed < 6, "the runner waited for the child instead of killing it"
    assert left == [], f"processes still running after the timeout: {left}"


@process_test
@linux_only
@pytest.mark.timeout(60)
def test_a_child_that_ignores_sigterm_is_killed_after_the_grace(tmp_path: Path) -> None:
    pid_file = tmp_path / "pids"
    started = time.monotonic()

    try:
        result = run(
            tmp_path,
            with_grandchild(pid_file, grandchild_ignores=True, leader_ignores=True),
            timeout=2,
        )
        elapsed = time.monotonic() - started
        left = survivors(read_pids(pid_file), 1.0)
    finally:
        if pid_file.exists():
            kill_quietly(read_pids(pid_file))

    assert result == CommandResult(None, True)
    assert elapsed >= 3.8, "the runner did not give SIGTERM its 2 s grace"
    assert elapsed < 12, "the runner did not follow SIGTERM with SIGKILL"
    assert left == [], f"processes still running after the kill: {left}"


@process_test
@linux_only
@pytest.mark.timeout(60)
def test_a_grandchild_that_ignores_sigterm_is_killed_when_the_leader_has_exited(
    tmp_path: Path,
) -> None:
    # The leader dies of SIGTERM at once; what it left behind must still be killed.
    pid_file = tmp_path / "pids"

    try:
        result = run(
            tmp_path, with_grandchild(pid_file, grandchild_ignores=True), timeout=2
        )
        left = survivors(read_pids(pid_file), 1.0)
    finally:
        if pid_file.exists():
            kill_quietly(read_pids(pid_file))

    assert result == CommandResult(None, True)
    assert left == [], f"processes still running after the timeout: {left}"


CHILD_THAT_HANDLES_SIGTERM = """
import signal, sys, time
def on_term(signum, frame):
    open(sys.argv[1], "w").write("asked to end")
    sys.exit(0)
signal.signal(signal.SIGTERM, on_term)
open(sys.argv[1] + ".ready", "w").close()
time.sleep(60)
"""


@process_test
@linux_only
@pytest.mark.timeout(60)
def test_a_timeout_first_asks_the_group_to_end_with_sigterm(tmp_path: Path) -> None:
    marker = tmp_path / "marker"

    result = run(tmp_path, python(CHILD_THAT_HANDLES_SIGTERM, str(marker)), timeout=2)

    assert result == CommandResult(None, True)
    assert marker.read_text() == "asked to end"


@process_test
@linux_only
@pytest.mark.timeout(60)
@pytest.mark.parametrize(
    ("interrupt", "arguments"),
    [(KeyboardInterrupt, ()), (SystemExit, (143,))],
    ids=["KeyboardInterrupt", "SystemExit"],
)
def test_an_interrupt_in_the_wait_kills_the_group_before_it_propagates(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    interrupt: type[BaseException],
    arguments: tuple[Any, ...],
) -> None:
    pid_file = tmp_path / "pids"
    original_wait = subprocess.Popen.wait
    interrupted: list[bool] = []

    def interrupt_once(
        self: subprocess.Popen[bytes], timeout: float | None = None
    ) -> int:
        if not interrupted:
            interrupted.append(True)
            deadline = time.monotonic() + 10
            while not pid_file.exists() and time.monotonic() < deadline:
                time.sleep(0.02)
            raise interrupt(*arguments)
        return original_wait(self, timeout)

    monkeypatch.setattr(subprocess.Popen, "wait", interrupt_once)

    try:
        with pytest.raises(interrupt):
            run(tmp_path, with_grandchild(pid_file), timeout=300)
        left = survivors(read_pids(pid_file))
    finally:
        if pid_file.exists():
            kill_quietly(read_pids(pid_file))

    assert left == [], f"processes still running after the interrupt: {left}"


@process_test
@linux_only
@pytest.mark.timeout(60)
@pytest.mark.usefixtures("default_signal_handlers")
def test_a_real_sigint_interrupts_the_wait_while_the_child_runs(tmp_path: Path) -> None:
    # Once polars is imported (every `rinchen` command has it) a SIGINT does not wake a blocking
    # wait that has no timeout; a wait with a timeout is woken (docs/VERIFY.md, D-POLARS-SIGINT).
    # The child lives 8 s, so a wait that is not woken by the signal fails the assertion below.
    importlib.import_module("polars")
    pid_file = tmp_path / "pids"
    signalled_at: list[float] = []
    stop = threading.Event()

    def send_sigint_once_the_child_runs() -> None:
        while not pid_file.exists():  # a loaded machine may start the child late
            if stop.wait(0.02):
                return
        signalled_at.append(time.monotonic())
        os.kill(os.getpid(), signal.SIGINT)

    interrupter = threading.Timer(0.4, send_sigint_once_the_child_runs)
    interrupter.start()

    try:
        with pytest.raises(KeyboardInterrupt):
            run(tmp_path, with_grandchild(pid_file, lifetime=8), timeout=300)
        interrupted_after = time.monotonic() - signalled_at[0]
        left = survivors(read_pids(pid_file))
    finally:
        stop.set()
        interrupter.cancel()
        interrupter.join()
        if pid_file.exists():
            kill_quietly(read_pids(pid_file))

    assert interrupted_after < 2.0, (
        f"the interrupt was raised {interrupted_after:.2f} s after the signal"
    )
    assert left == [], f"processes still running after the interrupt: {left}"


@process_test
@pytest.mark.timeout(60)
def test_the_wait_is_in_slices_so_that_it_ends_when_the_child_does(
    tmp_path: Path,
) -> None:
    started = time.monotonic()

    result = run(tmp_path, python("pass"), timeout=300)

    assert result == CommandResult(0, False)
    assert time.monotonic() - started < 20


# ---- the bounded read of a log -----------------------------------------------------------------------


def test_a_missing_log_reads_as_the_empty_text(tmp_path: Path) -> None:
    assert read_log(tmp_path / "logs" / "absent.log") == ""


def test_a_log_is_decoded_as_utf_8_with_replacement(tmp_path: Path) -> None:
    log = tmp_path / "child.log"
    log.write_bytes(b"a\xffb \xd0\x96\n")

    assert read_log(log) == "a" + chr(0xFFFD) + "b " + chr(0x0416) + "\n"


def test_a_log_is_cut_at_the_limit_and_a_short_one_is_whole(tmp_path: Path) -> None:
    log = tmp_path / "child.log"
    log.write_bytes(b"0123456789")

    assert read_log(log, limit=4) == "0123"
    assert read_log(log, limit=10) == "0123456789"
    assert read_log(log, limit=11) == "0123456789"
    assert read_log(log, limit=0) == ""


def test_a_negative_limit_is_refused_and_not_read_as_no_limit(tmp_path: Path) -> None:
    log = tmp_path / "child.log"
    log.write_bytes(b"0123456789")

    with pytest.raises(ValueError, match="limit"):
        read_log(log, limit=-1)
