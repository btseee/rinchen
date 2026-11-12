"""The check of the `uv` shim before the recipe of the D15 gate starts (`tests/support/tiny_gate.py`,
`require_shim`, called by `start_guarded` for every command that is given a shim directory).

The recipe's body calls `uv run ...`; the gate puts the directory of the shim of `tests/support/fake_uv.py` first on
`PATH` so that nothing but `run` ever reaches the real `uv` (which would lock, sync and ask an index). A directory
without the shim, a shim that cannot be executed, a relative directory (the child runs somewhere else) or a `PATH`
the caller overrode would let the child fall through to the real `uv` in silence, so each is refused by a failed
assertion BEFORE a process exists. `subprocess.Popen` of the helper's module is replaced by a function that fails
the test with an exception of its own when it is called, so that a refusal that did not come first cannot pass as
the refusal the tests look for.
"""

from __future__ import annotations

import os
import stat
from pathlib import Path
from typing import Any

import pytest

from tests.support import tiny_gate
from tests.support.tiny_gate import make_shim, require_shim, run_recipe, start_guarded


class StartedAProcess(Exception):
    """The helper reached `Popen`: no refusal came before it."""


@pytest.fixture(autouse=True)
def nothing_starts(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []

    def refuse(argv: list[str], **kwargs: Any) -> Any:
        calls.append({"argv": argv, **kwargs})
        raise StartedAProcess

    monkeypatch.setattr(tiny_gate.subprocess, "Popen", refuse)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    return calls


def _executable(path: Path, *, mode: int = 0o755) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("#!/bin/sh\n", encoding="utf-8")
    path.chmod(mode)
    return path


def test_a_directory_without_the_shim_is_refused_before_anything_starts(
    tmp_path: Path, nothing_starts: list[dict[str, Any]]
) -> None:
    empty = tmp_path / "bin"
    empty.mkdir()

    with pytest.raises(AssertionError, match="not an executable file"):
        run_recipe("the recipe body", cwd=tmp_path, shim_dir=empty)

    assert nothing_starts == []


def test_a_directory_that_does_not_exist_is_refused_before_anything_starts(
    tmp_path: Path, nothing_starts: list[dict[str, Any]]
) -> None:
    with pytest.raises(AssertionError, match="not an executable file"):
        run_recipe("the recipe body", cwd=tmp_path, shim_dir=tmp_path / "absent")

    assert nothing_starts == []


def test_a_shim_that_cannot_be_executed_is_refused_before_anything_starts(
    tmp_path: Path, nothing_starts: list[dict[str, Any]]
) -> None:
    _executable(tmp_path / "bin" / "uv", mode=0o644)

    with pytest.raises(AssertionError, match="not an executable file"):
        run_recipe("the recipe body", cwd=tmp_path, shim_dir=tmp_path / "bin")

    assert nothing_starts == []


def test_a_directory_named_uv_is_not_a_shim(
    tmp_path: Path, nothing_starts: list[dict[str, Any]]
) -> None:
    (tmp_path / "bin" / "uv").mkdir(parents=True)

    with pytest.raises(AssertionError, match="not an executable file"):
        run_recipe("the recipe body", cwd=tmp_path, shim_dir=tmp_path / "bin")

    assert nothing_starts == []


def test_a_relative_directory_is_refused_because_the_child_runs_somewhere_else(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    nothing_starts: list[dict[str, Any]],
) -> None:
    _executable(tmp_path / "bin" / "uv")
    monkeypatch.chdir(tmp_path)

    with pytest.raises(AssertionError, match="absolute"):
        run_recipe("the recipe body", cwd=tmp_path, shim_dir=Path("bin"))

    assert nothing_starts == []


def test_a_path_the_caller_overrode_so_that_the_shim_is_not_found_is_refused(
    tmp_path: Path, nothing_starts: list[dict[str, Any]]
) -> None:
    _executable(tmp_path / "bin" / "uv")

    with pytest.raises(AssertionError, match="is not the shim"):
        run_recipe(
            "the recipe body",
            cwd=tmp_path,
            shim_dir=tmp_path / "bin",
            env_extra={"PATH": str(tmp_path / "elsewhere")},
        )

    assert nothing_starts == []


def test_another_uv_ahead_of_the_shim_on_the_path_is_refused(
    tmp_path: Path, nothing_starts: list[dict[str, Any]]
) -> None:
    _executable(tmp_path / "bin" / "uv")
    _executable(tmp_path / "other" / "uv")

    with pytest.raises(AssertionError, match="is not the shim"):
        run_recipe(
            "the recipe body",
            cwd=tmp_path,
            shim_dir=tmp_path / "bin",
            env_extra={"PATH": f"{tmp_path / 'other'}{os.pathsep}{tmp_path / 'bin'}"},
        )

    assert nothing_starts == []


def test_the_refusal_also_holds_for_any_other_command_given_a_shim_directory(
    tmp_path: Path, nothing_starts: list[dict[str, Any]]
) -> None:
    with pytest.raises(AssertionError, match="not an executable file"):
        start_guarded(["a-program"], cwd=tmp_path, shim_dir=tmp_path / "bin")

    assert nothing_starts == []


def test_the_shim_of_the_gate_is_accepted_and_the_command_then_starts(
    tmp_path: Path, nothing_starts: list[dict[str, Any]]
) -> None:
    shim_dir = make_shim(tmp_path / "bin")

    with pytest.raises(StartedAProcess):
        run_recipe("the recipe body", cwd=tmp_path, shim_dir=shim_dir)

    assert len(nothing_starts) == 1
    assert nothing_starts[0]["env"]["PATH"].split(os.pathsep)[0] == str(shim_dir)


def test_a_command_given_no_shim_directory_is_not_asked_for_one(
    tmp_path: Path, nothing_starts: list[dict[str, Any]]
) -> None:
    with pytest.raises(StartedAProcess):
        start_guarded(["a-program"], cwd=tmp_path)

    assert len(nothing_starts) == 1


def test_the_check_reads_the_path_it_is_given(tmp_path: Path) -> None:
    shim = _executable(tmp_path / "bin" / "uv")

    require_shim(tmp_path / "bin", f"{shim.parent}{os.pathsep}/usr/bin")

    with pytest.raises(AssertionError, match="is not the shim"):
        require_shim(tmp_path / "bin", "/usr/bin")


def test_a_shim_reached_through_a_link_is_the_shim(tmp_path: Path) -> None:
    real = _executable(tmp_path / "real" / "uv")
    link_dir = tmp_path / "bin"
    link_dir.mkdir()
    (link_dir / "uv").symlink_to(real)

    require_shim(link_dir, f"{link_dir}{os.pathsep}/usr/bin")


def test_the_mode_of_the_shim_the_gate_writes_is_executable(tmp_path: Path) -> None:
    shim = make_shim(tmp_path / "bin") / "uv"

    assert stat.S_IMODE(shim.stat().st_mode) & stat.S_IXUSR
