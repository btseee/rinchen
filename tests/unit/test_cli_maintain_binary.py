"""`rinchen maintain` through the real binary (`python -m rinchen`) and through the registry of
commands (D14, spec M3): the help, the registration and every kind of refusal.

A refusal ends the command before the doctor, the lock check or any rerun starts, so it needs no
`uv`, no model and no network, and the real command can be run for it. Every refusal here is run
with `--no-drift --no-reruns` and a `uv` that does not exist, so that a command that wrongly got
past its checks would still start nothing but the doctor (which is bounded by the timeout of the
subprocess). The success path is never run for real here: its commands are the fake runner's in the
other modules of this task.
"""

from __future__ import annotations

import subprocess
import sys
from collections.abc import Callable
from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

from rinchen.cli.main import app
from rinchen.maintain.commands import child_environment
from rinchen.manifest import store
from rinchen.manifest.model import Manifest
from tests.support.maintain_check import SHIPPED_CONFIG
from tests.support.maintain_cli import Workspace, lock_text, make_workspace, snapshot
from tests.support.maintain_project import parent_environment
from tests.support.no_network import no_network  # noqa: F401 - the fixture of the mark

pytestmark = pytest.mark.usefixtures("no_network")

TIMEOUT_SECONDS = 180
OPTIONS = (
    "--out-dir",
    "--cadence",
    "--config",
    "--project-dir",
    "--uv",
    "--no-drift",
    "--no-reruns",
    "--rerun",
    "--fail-on-drift",
    "--export",
    "--checkpoint",
    "--corpus",
)


def binary(*arguments: str, cwd: Path) -> subprocess.CompletedProcess[str]:
    """`python -m rinchen <arguments>` in `cwd`, bounded, with the device hidden."""
    return subprocess.run(
        [sys.executable, "-m", "rinchen", *arguments],
        cwd=cwd,
        env=child_environment(parent_environment()),
        capture_output=True,
        text=True,
        timeout=TIMEOUT_SECONDS,
        check=False,
        stdin=subprocess.DEVNULL,
    )


# ---- help and registration ------------------------------------------------------------------------------


def test_the_root_help_lists_the_maintain_group() -> None:
    result = CliRunner().invoke(app, ["--help"])

    assert result.exit_code == 0
    assert "maintain" in result.output


def test_the_group_help_lists_the_check_command() -> None:
    result = CliRunner().invoke(app, ["maintain", "--help"])

    assert result.exit_code == 0
    assert "check" in result.output


def test_the_group_without_a_command_shows_its_help() -> None:
    result = CliRunner().invoke(app, ["maintain"])

    assert "check" in result.output


def test_the_check_help_lists_every_option() -> None:
    result = CliRunner().invoke(
        app, ["maintain", "check", "--help"], terminal_width=200
    )

    assert result.exit_code == 0
    for option in OPTIONS:
        assert option in result.output, option


def test_the_check_help_says_what_the_exit_codes_are() -> None:
    result = CliRunner().invoke(
        app, ["maintain", "check", "--help"], terminal_width=200
    )

    for word in ("Exit", "codes:", "130", "interrupted", "--fail-on-drift"):
        assert word in result.output, word


def test_the_real_binary_shows_the_help_of_the_group_and_of_the_command(
    tmp_path: Path,
) -> None:
    group = binary("maintain", "--help", cwd=tmp_path)
    command = binary("maintain", "check", "--help", cwd=tmp_path)

    assert group.returncode == 0, group.stderr
    assert "check" in group.stdout
    assert command.returncode == 0, command.stderr
    for option in OPTIONS:
        assert option in command.stdout, option


# ---- the refusals, through the real binary --------------------------------------------------------------


def args(ws: Workspace, *extra: str, **options: Path) -> list[str]:
    """The command line of a refusal: `--no-drift --no-reruns` and a `uv` that does not exist."""
    return ws.args(*extra, "--no-drift", "--no-reruns", uv="/nonexistent/uv", **options)


def refuse_cadence(ws: Workspace) -> list[str]:
    return args(ws, "--cadence", "monthly")


def refuse_half_an_export(ws: Workspace) -> list[str]:
    return args(ws, "--export", str(ws.root / "export"))


def refuse_unknown_rerun(ws: Workspace) -> list[str]:
    return args(ws, "--rerun", "nosuch")


def refuse_missing_config(ws: Workspace) -> list[str]:
    return args(ws, config=ws.root / "nowhere.yaml")


def refuse_bad_config(ws: Workspace) -> list[str]:
    document = yaml.safe_load(SHIPPED_CONFIG.read_text(encoding="utf-8"))
    document["drift"]["uv_args"] = ["lock", "--upgrade"]
    ws.config.write_text(yaml.safe_dump(document), encoding="utf-8")
    return args(
        ws,
    )


def refuse_missing_lock(ws: Workspace) -> list[str]:
    (ws.project / "uv.lock").unlink()
    return args(
        ws,
    )


def refuse_invalid_lock(ws: Workspace) -> list[str]:
    (ws.project / "uv.lock").write_bytes(b"version = [\n")
    return args(
        ws,
    )


def refuse_lock_version_2(ws: Workspace) -> list[str]:
    (ws.project / "uv.lock").write_bytes(
        lock_text({"a": ("1",)}).replace(b"version = 1", b"version = 2", 1)
    )
    return args(
        ws,
    )


def refuse_missing_pyproject(ws: Workspace) -> list[str]:
    (ws.project / "pyproject.toml").unlink()
    return args(
        ws,
    )


def refuse_out_dir_is_the_project(ws: Workspace) -> list[str]:
    return args(ws, out_dir=ws.project)


def refuse_foreign_manifest(ws: Workspace) -> list[str]:
    ws.out.mkdir()
    store.write(
        ws.out,
        Manifest(
            stage="export_parity",
            created="2026-01-01T00:00:00+00:00",
            code_version="abc1234",
            config_sha256="0" * 64,
            seed=-1,
        ),
    )
    return args(
        ws,
    )


REFUSALS: list[tuple[Callable[[Workspace], list[str]], int, str]] = [
    (refuse_cadence, 2, "--cadence"),
    (refuse_half_an_export, 2, "--export, --checkpoint and --corpus"),
    (refuse_unknown_rerun, 2, "nosuch"),
    (refuse_missing_config, 2, "maintain config"),
    (refuse_bad_config, 2, "drift.uv_args"),
    (refuse_out_dir_is_the_project, 2, "--out-dir"),
    (refuse_missing_lock, 3, "uv.lock"),
    (refuse_invalid_lock, 3, "uv.lock"),
    (refuse_lock_version_2, 3, "uv.lock"),
    (refuse_missing_pyproject, 3, "pyproject.toml"),
    (refuse_foreign_manifest, 3, "export_parity"),
]


@pytest.mark.parametrize(
    ("prepare", "exit_code", "message"),
    REFUSALS,
    ids=[refusal[0].__name__.removeprefix("refuse_") for refusal in REFUSALS],
)
def test_the_real_binary_refuses_and_writes_nothing(
    tmp_path: Path,
    prepare: Callable[[Workspace], list[str]],
    exit_code: int,
    message: str,
) -> None:
    ws = make_workspace(tmp_path)
    arguments = prepare(ws)
    before = snapshot(ws.root)

    result = binary(*arguments, cwd=tmp_path)

    assert result.returncode == exit_code, result.stderr
    assert message in result.stderr
    assert result.stdout == ""
    assert snapshot(ws.root) == before
