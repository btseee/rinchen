"""The reader of the processes a gate run leaves (`tests/support/tiny_gate_processes.py`), tested over a
made-up `/proc` (D15 Task 9): no process is started and no signal is sent. A process belongs to a run when
its environment holds the run's marker or when it is in the session the run was started in; a zombie
has ended and is never counted; only the processes that belong to the run are ever signalled.
"""

from __future__ import annotations

import signal
from pathlib import Path
from typing import Final

from tests.support.tiny_gate_processes import (
    RunProcess,
    kill_the_run,
    left_after_the_end,
    processes_of_the_run,
    wait_for_command,
)

MARKER: Final = "RINCHEN_GATE_RUN=abc123"
OTHER_MARKER: Final = "RINCHEN_GATE_RUN=zzz999"


def _process(
    proc: Path,
    pid: int,
    *,
    session: int,
    environment: list[str] | None,
    command: list[str],
    state: str = "S",
    name: str = "worker",
) -> None:
    directory = proc / str(pid)
    directory.mkdir(parents=True)
    # `pid (name) state ppid pgrp session ...`: the name may hold spaces and parentheses.
    (directory / "stat").write_text(
        f"{pid} ({name}) {state} 1 {pid} {session} 0 -1 4194560\n", encoding="utf-8"
    )
    if environment is not None:
        (directory / "environ").write_bytes(
            b"\0".join(item.encode() for item in [*environment, ""])
        )
    (directory / "cmdline").write_bytes(
        b"\0".join(item.encode() for item in [*command, ""])
    )


def _proc(tmp_path: Path) -> Path:
    proc = tmp_path / "proc"
    proc.mkdir()
    (proc / "self").mkdir()
    (proc / "cpuinfo").write_text("not a process\n", encoding="utf-8")
    return proc


def test_a_process_with_the_marker_belongs_to_the_run_wherever_its_session_is(
    tmp_path: Path,
) -> None:
    proc = _proc(tmp_path)
    _process(proc, 10, session=10, environment=["A=1", MARKER], command=["rinchen"])
    _process(
        proc, 11, session=11, environment=["A=1", MARKER], command=["eole", "train"]
    )

    found = processes_of_the_run("abc123", proc=proc)

    assert found == [
        RunProcess(10, 10, "rinchen"),
        RunProcess(11, 11, "eole train"),
    ]


def test_a_process_of_another_run_or_of_another_user_is_not_the_runs(
    tmp_path: Path,
) -> None:
    proc = _proc(tmp_path)
    _process(proc, 20, session=20, environment=["A=1", OTHER_MARKER], command=["x"])
    _process(proc, 21, session=21, environment=["A=1"], command=["x"])
    _process(proc, 22, session=22, environment=None, command=["unreadable"])

    assert processes_of_the_run("abc123", proc=proc) == []


def test_the_marker_must_match_whole_and_not_as_a_prefix(tmp_path: Path) -> None:
    proc = _proc(tmp_path)
    _process(
        proc, 30, session=30, environment=["RINCHEN_GATE_RUN=abc1234"], command=["x"]
    )

    assert processes_of_the_run("abc123", proc=proc) == []


def test_a_process_of_the_session_of_the_run_belongs_to_it_without_the_marker(
    tmp_path: Path,
) -> None:
    proc = _proc(tmp_path)
    _process(proc, 40, session=40, environment=["A=1"], command=["sh", "-c", "x"])
    _process(proc, 41, session=40, environment=None, command=["child"])
    _process(proc, 42, session=99, environment=["A=1"], command=["other"])

    found = processes_of_the_run("abc123", session=40, proc=proc)

    assert [item.pid for item in found] == [40, 41]


def test_a_zombie_has_ended_and_is_not_counted(tmp_path: Path) -> None:
    proc = _proc(tmp_path)
    _process(
        proc,
        50,
        session=50,
        environment=["A=1", MARKER],
        command=["gone"],
        state="Z",
    )

    assert processes_of_the_run("abc123", session=50, proc=proc) == []


def test_a_name_with_spaces_and_a_parenthesis_does_not_hide_the_session(
    tmp_path: Path,
) -> None:
    proc = _proc(tmp_path)
    _process(
        proc,
        60,
        session=60,
        environment=None,
        command=["x"],
        name="a b) S 1 2 3",
    )

    assert [
        item.pid for item in processes_of_the_run("none", session=60, proc=proc)
    ] == [60]


def test_the_run_is_signalled_by_pid_and_only_its_own_processes(tmp_path: Path) -> None:
    proc = _proc(tmp_path)
    _process(proc, 70, session=70, environment=[MARKER], command=["mine"])
    _process(proc, 71, session=71, environment=[OTHER_MARKER], command=["not mine"])
    _process(proc, 72, session=72, environment=["A=1"], command=["not mine either"])
    sent: list[tuple[int, int]] = []

    killed = kill_the_run(
        "abc123", proc=proc, kill=lambda pid, number: sent.append((pid, number))
    )

    assert killed == [70]
    assert sent == [(70, signal.SIGKILL)]


def test_a_process_that_is_gone_when_it_is_signalled_is_not_an_error(
    tmp_path: Path,
) -> None:
    proc = _proc(tmp_path)
    _process(proc, 80, session=80, environment=[MARKER], command=["mine"])

    def vanished(pid: int, number: int) -> None:
        raise ProcessLookupError(pid)

    assert kill_the_run("abc123", proc=proc, kill=vanished) == [80]


def test_the_wait_for_a_command_hands_back_the_processes_whose_command_holds_the_needle(
    tmp_path: Path,
) -> None:
    proc = _proc(tmp_path)
    _process(
        proc,
        100,
        session=100,
        environment=[MARKER],
        command=["python", "-m", "rinchen"],
    )
    _process(
        proc,
        101,
        session=101,
        environment=[MARKER],
        command=["python", "bin/eole", "train"],
    )
    _process(
        proc,
        102,
        session=102,
        environment=[OTHER_MARKER],
        command=["bin/eole", "train"],
    )

    found = wait_for_command("abc123", "bin/eole", seconds=5.0, proc=proc)

    assert [item.pid for item in found] == [101]


def test_the_wait_for_a_command_gives_up_when_none_appears(tmp_path: Path) -> None:
    proc = _proc(tmp_path)
    _process(proc, 110, session=110, environment=[MARKER], command=["python"])

    assert wait_for_command("abc123", "bin/eole", seconds=0.2, proc=proc) == []


def test_the_check_ends_at_once_when_nothing_is_left(tmp_path: Path) -> None:
    proc = _proc(tmp_path)

    assert left_after_the_end("abc123", proc=proc) == []


def test_the_check_hands_back_what_is_still_there_when_the_grace_is_over(
    tmp_path: Path,
) -> None:
    proc = _proc(tmp_path)
    _process(proc, 90, session=90, environment=[MARKER], command=["stays"])

    left = left_after_the_end("abc123", grace=0.2, proc=proc)

    assert [item.pid for item in left] == [90]
