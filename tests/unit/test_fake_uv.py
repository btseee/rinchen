"""The `uv` shim of the maintenance gate (`tests/support/fake_uv.py`), tested on its own with a recorder in
place of the real uv: nothing here starts the real uv, a model or a network (D14, Task 7).

The shim is what keeps the one online step of a night local, so its behaviour is pinned exactly: the dry run
is answered from its sample in either order of the flags and never forwarded; `lock --check`, `--version`
and `run` are forwarded with the flags of the gate (and the caller's leading `--frozen` and `--no-sync` taken
out, since uv refuses a flag given twice); every other list, and every list that holds an upgrade, a sync,
an add or a pip, exits 97 printing `refused` and reaches no program.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from tests.support.fake_uv import CALL_LOG, REFUSED_EXIT, read_calls, write_uv_shim
from tests.support.uv_outputs import CHANGES_HAND_WRITTEN, OFFLINE_FAILURE_MEASURED

SHIM_TIMEOUT_SECONDS = 30


def _recorder(directory: Path) -> tuple[Path, Path]:
    """A program that records its arguments to a file and exits 0: the stand-in for the real uv."""
    directory.mkdir(parents=True, exist_ok=True)
    log = directory / "recorded.jsonl"
    program = directory / "recorder"
    program.write_text(
        f"#!{sys.executable} -I\nimport json, os, sys\n"
        f"entry = [sys.argv[1:], os.environ.get('UV_OFFLINE')]\n"
        f"open({str(log)!r}, 'a').write(json.dumps(entry) + '\\n')\n",
        encoding="utf-8",
    )
    os.chmod(program, 0o755)
    return program, log


@dataclass(frozen=True)
class ShimRun:
    returncode: int
    stdout: str
    stderr: str
    calls: list[dict[str, Any]]
    recorded: list[list[Any]]


def _call_shim(
    directory: Path,
    arguments: Sequence[str],
    *,
    output: str = CHANGES_HAND_WRITTEN,
    exit_code: int = 0,
) -> ShimRun:
    program, log = _recorder(directory / "real")
    shim = write_uv_shim(
        directory / "bin",
        real_uv=str(program),
        project=Path("/the/project"),
        dry_run_output=output,
        dry_run_exit=exit_code,
    )
    completed = subprocess.run(
        [str(shim), *arguments],
        capture_output=True,
        text=True,
        timeout=SHIM_TIMEOUT_SECONDS,
        check=False,
        env={"PATH": os.defpath},
    )
    recorded = (
        [json.loads(line) for line in log.read_text().splitlines()]
        if log.exists()
        else []
    )
    return ShimRun(
        completed.returncode,
        completed.stdout,
        completed.stderr,
        read_calls(shim.parent),
        recorded,
    )


@pytest.mark.parametrize(
    "flags", [["--upgrade", "--dry-run"], ["--dry-run", "--upgrade"]]
)
@pytest.mark.parametrize(
    ("output", "exit_code"), [(CHANGES_HAND_WRITTEN, 0), (OFFLINE_FAILURE_MEASURED, 1)]
)
def test_the_shim_answers_the_dry_run_from_its_sample_and_never_forwards_it(
    tmp_path: Path, flags: list[str], output: str, exit_code: int
) -> None:
    run = _call_shim(tmp_path, ["lock", *flags], output=output, exit_code=exit_code)

    assert run.stderr == output
    assert run.stdout == ""
    assert run.returncode == exit_code
    assert run.recorded == []
    assert [(c["forwarded"], c["refused"], c["final"]) for c in run.calls] == [
        (False, False, None)
    ]


@pytest.mark.parametrize(
    ("arguments", "final"),
    [
        (["lock", "--check"], ["lock", "--check", "--offline"]),
        (["lock", "--offline", "--check"], ["lock", "--check", "--offline"]),
        (["--version"], ["--version"]),
        (
            [
                "run",
                "--frozen",
                "--no-sync",
                "rinchen",
                "maintain",
                "check",
                "--out-dir",
                "o",
            ],
            [
                "run",
                "--project",
                "/the/project",
                "--no-sync",
                "--frozen",
                "--offline",
                "rinchen",
                "maintain",
                "check",
                "--out-dir",
                "o",
            ],
        ),
        (
            ["run", "--no-sync", "--frozen", "rinchen", "x"],
            [
                "run",
                "--project",
                "/the/project",
                "--no-sync",
                "--frozen",
                "--offline",
                "rinchen",
                "x",
            ],
        ),
        (
            ["run", "rinchen", "x"],
            [
                "run",
                "--project",
                "/the/project",
                "--no-sync",
                "--frozen",
                "--offline",
                "rinchen",
                "x",
            ],
        ),
    ],
)
def test_the_shim_forwards_the_allowed_calls_with_the_flags_of_the_gate(
    tmp_path: Path, arguments: list[str], final: list[str]
) -> None:
    run = _call_shim(tmp_path, arguments)

    assert run.returncode == 0, run.stderr
    # The shim started with no UV_OFFLINE of its own: it is the shim that makes the call offline.
    assert run.recorded == [[final, "1"]]
    assert [(c["forwarded"], c["refused"]) for c in run.calls] == [(True, False)]
    assert run.calls[0]["argv"] == arguments
    assert run.calls[0]["final"][1:] == final
    flags = [token for token in final if token.startswith("--")]
    assert len(flags) == len(set(flags))


@pytest.mark.parametrize(
    "arguments",
    [
        [],
        ["lock"],
        ["lock", "--upgrade"],
        ["lock", "--dry-run"],
        ["lock", "--upgrade", "--dry-run", "--offline"],
        ["lock", "--check", "--upgrade-package", "numpy"],
        ["lock", "--check", "--upgrade-package=numpy"],
        ["lock", "--check", "-U"],
        ["lock", "--check", "-P", "numpy"],
        ["lock", "--check", "--upgrade"],
        ["lock", "--check", "sync"],
        ["sync"],
        ["sync", "--frozen"],
        ["add", "numpy"],
        ["pip", "install", "numpy"],
        ["tree"],
        ["self", "update"],
        ["--version", "--verbose"],
        ["run"],
        ["run", "--frozen"],
        ["run", "--frozen", "--frozen", "rinchen"],
        ["run", "--no-sync", "--no-sync", "rinchen"],
        ["run", "--python", "3.12", "rinchen"],
        ["run", "--with", "x", "rinchen"],
        ["run", "--upgrade", "rinchen"],
        ["run", "rinchen", "--upgrade"],
        ["run", "rinchen", "--upgrade-package", "x"],
        ["run", "rinchen", "-U"],
        ["run", "rinchen", "-P", "numpy"],
        ["run", "rinchen", "sync"],
        ["run", "rinchen", "pip", "install"],
        ["run", "rinchen", "add", "x"],
    ],
    ids=lambda arguments: " ".join(arguments) or "no arguments",
)
def test_the_shim_refuses_every_other_list_and_every_upgrade_sync_add_or_pip(
    tmp_path: Path, arguments: list[str]
) -> None:
    run = _call_shim(tmp_path, arguments)

    assert run.returncode == REFUSED_EXIT == 97
    assert run.stderr == "refused\n"
    assert run.stdout == ""
    assert run.recorded == []
    assert [(c["forwarded"], c["refused"], c["final"]) for c in run.calls] == [
        (False, True, None)
    ]


def test_the_shim_is_an_executable_with_a_call_log_and_imports_nothing_that_reaches_a_network(
    tmp_path: Path,
) -> None:
    shim = write_uv_shim(
        tmp_path, real_uv="/x/uv", project=tmp_path, dry_run_output="x\n"
    )
    source = shim.read_text("utf-8")
    imported = set(re.findall(r"^import (\w+)$", source, re.MULTILINE))

    assert shim.stat().st_mode & 0o777 == 0o755
    assert (tmp_path / CALL_LOG).read_text() == ""
    assert imported == {"json", "os", "sys"}
    assert source.startswith(f"#!{sys.executable} -I\n")
