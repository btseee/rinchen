"""The workspace and the helpers of the tests of `rinchen maintain check` (D14, Task 5).

A `Workspace` is a project on disk (`tests.support.maintain_project.make_project`: a
`pyproject.toml`, a `uv.lock` that holds the packages of `LOCKED`, and the files the shipped
reruns select), a copy of the shipped configuration and an output directory that does not exist.
`install` replaces the three seams of the command module (the runner, the doctor and the
installed versions) with doubles, so that a run starts no process and imports no model: the
runner is a `FakeRunner` scripted for the lock check, the dry run, `uv --version` and the reruns.

`LOCKED` is the lock of `tests.support.uv_outputs.SAMPLE_LOCKED` and one name that has two
versions (`twin`), so that the manifest's `tool_versions` has a value to join. `snapshot` reads a
directory tree to `{relative path: bytes}` and `identity` the bytes and the modification time of
one file, for the tests that look at what a run left alone.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import pytest
from click.testing import Result
from typer.testing import CliRunner

from rinchen.cli import maintain, maintain_output
from rinchen.cli.main import app
from rinchen.maintain.check_types import DoctorFinding
from rinchen.maintain.commands import CommandResult
from tests.support.maintain_check import (
    SHIPPED_CONFIG,
    golden_doctor,
    rerun_call,
    shipped_config,
    uv_call,
)
from tests.support.maintain_fakes import FakeRunner, junit_xml
from tests.support.maintain_project import make_project
from tests.support.uv_outputs import (
    CHANGES_HAND_WRITTEN,
    NO_CHANGE_MEASURED,
    SAMPLE_LOCKED,
)

UV: Final = "/opt/uv-fake/bin/uv"
UV_VERSION: Final = "0.13.7"
UV_VERSION_TEXT: Final = f"uv {UV_VERSION} (x86_64-unknown-linux-gnu)\n"
CODE_VERSION: Final = "abc1234"
LOCKED: Final[Mapping[str, tuple[str, ...]]] = {**SAMPLE_LOCKED, "twin": ("1.0", "2.0")}
PARITY_FIGURES: Final = {"exact": 200, "sentences": 200, "mean_abs_delta": 7.19e-07}

_runner = CliRunner()


def sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def lock_text(versions: Mapping[str, Sequence[str]]) -> bytes:
    """A `uv.lock` of format version 1 with one `[[package]]` table for each name and version."""
    tables = [
        f'[[package]]\nname = "{name}"\nversion = "{version}"\n'
        for name, found in versions.items()
        for version in found
    ]
    return ("version = 1\nrevision = 5\n\n" + "\n".join(tables)).encode()


@dataclass(frozen=True)
class Workspace:
    """A project, a configuration file and an output directory that does not exist."""

    root: Path
    project: Path
    config: Path
    out: Path

    def args(
        self,
        *extra: str,
        uv: str | None = UV,
        out_dir: Path | None = None,
        project_dir: Path | None = None,
        config: Path | None = None,
    ) -> list[str]:
        """`maintain check` with the paths of the workspace, `--uv UV` unless `uv` is None, and
        `extra`."""
        arguments = [
            "maintain",
            "check",
            "--out-dir",
            str(self.out if out_dir is None else out_dir),
            "--project-dir",
            str(self.project if project_dir is None else project_dir),
            "--config",
            str(self.config if config is None else config),
        ]
        if uv is not None:
            arguments += ["--uv", uv]
        return [*arguments, *extra]

    def written(self) -> dict[str, bytes]:
        """Every file of the output directory, by its path relative to it."""
        return snapshot(self.out)


def make_workspace(
    root: Path,
    *,
    lock: bytes | None = None,
    config_text: str | None = None,
) -> Workspace:
    """The workspace in `root`; `lock` (the bytes of `LOCKED` by default) and `config_text` (the
    shipped configuration by default) replace the files."""
    project = make_project(
        root, config=shipped_config(), lock=lock_text(LOCKED) if lock is None else lock
    )
    config = root / "config" / "maintain.yaml"
    config.parent.mkdir()
    if config_text is None:
        shutil.copyfile(SHIPPED_CONFIG, config)
    else:
        config.write_text(config_text, encoding="utf-8")
    return Workspace(root, project, config, root / "out")


def version_call(argv: Sequence[str]) -> bool:
    return list(argv[1:]) == ["--version"]


def night_runner(
    *,
    drift_text: str = CHANGES_HAND_WRITTEN,
    numerics: tuple[int, bytes] = (0, junit_xml(1110, 0, 0, 0)),
    parity_tiny: tuple[int, bytes] = (0, junit_xml(13, 0, 0, 0)),
    version_text: str = UV_VERSION_TEXT,
) -> FakeRunner:
    """The commands of a nightly run: the lock check (uv's measured no-change output), the dry
    run (`drift_text`), the two nightly pytest reruns (an exit code and a JUnit report each) and
    `uv --version`."""
    runner = (
        FakeRunner()
        .script(
            CommandResult(0, False),
            log_text=NO_CHANGE_MEASURED,
            when=uv_call("--check"),
        )
        .script(
            CommandResult(0, False),
            log_text=drift_text,
            when=uv_call("--upgrade", "--dry-run"),
        )
    )
    for rerun_id, (exit_code, junit) in (
        ("numerics", numerics),
        ("parity_tiny", parity_tiny),
    ):
        runner.script(
            CommandResult(exit_code, False), junit=junit, when=rerun_call(rerun_id)
        )
    return runner.script(
        CommandResult(0, False), log_text=version_text, when=version_call
    )


def cli_runner() -> FakeRunner:
    """The golden nightly run: uv finds seven changes and every rerun passes."""
    return night_runner()


def quiet_runner() -> FakeRunner:
    """A night in which uv finds nothing to change."""
    return night_runner(drift_text=NO_CHANGE_MEASURED)


def parity_script(runner: FakeRunner, out: Path, *, exit_code: int = 0) -> FakeRunner:
    """Script the parity command to write a `parity.json` of `PARITY_FIGURES` and exit."""
    runner.script(
        CommandResult(exit_code, False),
        files={
            out / "parity_export" / "parity.json": json.dumps(PARITY_FIGURES).encode()
        },
        when=lambda argv: list(argv[2:5]) == ["rinchen", "export", "parity"],
    )
    return runner


def install(
    monkeypatch: pytest.MonkeyPatch,
    *,
    runner: FakeRunner | None = None,
    doctor: Callable[[], Sequence[DoctorFinding]] | None = None,
    installed: Callable[[], Mapping[str, Sequence[str]]] | None = None,
    versions: Mapping[str, Sequence[str]] = LOCKED,
    constant_code_version: bool = True,
) -> FakeRunner:
    """Replace the seams of the command: the runner (`cli_runner()` by default), the doctor (six
    passes and the GPU skip) and the installed versions (`versions`, the lock's own, by default).
    The code version of the manifest is a constant, so that a test never asks git, unless
    `constant_code_version` is False (a test of the code version itself). Returns the runner."""
    chosen = cli_runner() if runner is None else runner
    monkeypatch.setattr(maintain, "make_runner", lambda: chosen)
    monkeypatch.setattr(
        maintain, "doctor_findings", golden_doctor() if doctor is None else doctor
    )
    monkeypatch.setattr(
        maintain, "installed", (lambda: versions) if installed is None else installed
    )
    if constant_code_version:
        monkeypatch.setattr(maintain_output, "code_version", lambda: CODE_VERSION)
    return chosen


def invoke(arguments: Sequence[str]) -> Result:
    """Run the `rinchen` app on these arguments through typer's runner."""
    return _runner.invoke(app, list(arguments))


def snapshot(root: Path) -> dict[str, bytes]:
    """`{relative path: bytes}` of every file under `root` (empty when it does not exist)."""
    if not root.exists():
        return {}
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def identity(path: Path) -> tuple[bytes, int]:
    """The bytes and the modification time of a file: what 'untouched' means."""
    return path.read_bytes(), path.stat().st_mtime_ns


def manifest_of(out: Path) -> dict[str, Any]:
    loaded = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert isinstance(loaded, dict)
    return loaded


def lines_of(result: Result) -> list[str]:
    """The lines the command printed to standard output."""
    return result.stdout.splitlines()
