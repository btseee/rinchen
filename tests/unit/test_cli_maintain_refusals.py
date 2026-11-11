"""The refusals of `rinchen maintain check` (D14, spec M3 and M12): each input that is refused ends
the command with the exit code of the ruling, a message that names the place and never quotes a
file, and NOTHING written: no output directory, no manifest, no log, no change anywhere under the
workspace (the whole tree is read before and after), and the runner and the doctor are never
reached.

Exit codes: 2 for a bad option, an invalid configuration, a `--rerun` that names no rerun, a file
that changed after it was hashed and an output directory that is the project's or an input's; 3 for
a `pyproject.toml` or `uv.lock` that is missing or invalid and for the manifest of another stage in
the output directory; 1 for an output directory that cannot be made.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import pytest
import yaml

from rinchen.cli import maintain_inputs
from rinchen.manifest import store
from rinchen.manifest.model import Manifest
from tests.support.maintain_check import SHIPPED_CONFIG, Doctor
from tests.support.maintain_cli import (
    Workspace,
    install,
    invoke,
    lock_text,
    make_workspace,
    snapshot,
)
from tests.support.maintain_fakes import FakeRunner
from tests.support.no_network import no_network  # noqa: F401 - the fixture of the mark

pytestmark = pytest.mark.usefixtures("no_network")

MARKER = "SECRET-FILE-MARKER-61"


@pytest.fixture
def ws(tmp_path: Path) -> Workspace:
    return make_workspace(tmp_path)


class Untouched:
    """The doubles of a refused run and what the run was not allowed to reach."""

    def __init__(self, ws: Workspace, monkeypatch: pytest.MonkeyPatch) -> None:
        self.ws = ws
        self.runner = FakeRunner()
        self.doctor = Doctor([])
        install(monkeypatch, runner=self.runner, doctor=self.doctor)
        self.before = snapshot(ws.root)

    def refused(self, arguments: list[str], exit_code: int, message: str) -> str:
        """Run, check the exit code, the message and that nothing was reached or written;
        return the error text."""
        result = invoke(arguments)

        assert result.exit_code == exit_code, result.output
        assert message in result.stderr, result.stderr
        assert MARKER not in result.output
        assert snapshot(self.ws.root) == self.before
        assert not self.ws.out.exists()
        assert self.runner.calls == []
        assert self.doctor.calls == 0
        assert result.stdout == ""
        return result.stderr


@pytest.fixture
def untouched(ws: Workspace, monkeypatch: pytest.MonkeyPatch) -> Untouched:
    return Untouched(ws, monkeypatch)


def configuration(**drift: Any) -> str:
    """The shipped configuration with these keys of `drift` replaced."""
    document = yaml.safe_load(SHIPPED_CONFIG.read_text(encoding="utf-8"))
    document["drift"].update(drift)
    return yaml.safe_dump(document)


# ---- the options --------------------------------------------------------------------------------------


def test_a_cadence_that_is_neither_nightly_nor_weekly_is_a_bad_option(
    ws: Workspace, untouched: Untouched
) -> None:
    untouched.refused(ws.args("--cadence", "monthly"), 2, "--cadence")


@pytest.mark.parametrize(
    "given",
    [
        ["--export", "e"],
        ["--checkpoint", "c"],
        ["--corpus", "f"],
        ["--export", "e", "--checkpoint", "c"],
        ["--export", "e", "--corpus", "f"],
        ["--checkpoint", "c", "--corpus", "f"],
    ],
    ids=lambda given: "+".join(item.lstrip("-") for item in given[::2]),
)
def test_the_export_the_checkpoint_and_the_corpus_come_together_or_not_at_all(
    ws: Workspace, untouched: Untouched, given: list[str]
) -> None:
    untouched.refused(
        ws.args(*given), 2, "--export, --checkpoint and --corpus must be given together"
    )


def test_a_rerun_that_the_configuration_does_not_hold_is_named(
    ws: Workspace, untouched: Untouched
) -> None:
    error = untouched.refused(ws.args("--rerun", "nosuch"), 2, "'nosuch'")

    assert "numerics" in error


def test_one_unknown_rerun_among_known_ones_is_refused(
    ws: Workspace, untouched: Untouched
) -> None:
    untouched.refused(
        ws.args("--rerun", "numerics", "--rerun", "nosuch"), 2, "'nosuch'"
    )


def test_a_very_long_rerun_name_is_cut_in_the_message(
    ws: Workspace, untouched: Untouched
) -> None:
    error = untouched.refused(ws.args("--rerun", "x" * 500), 2, "--rerun")

    assert "x" * 100 not in error


# ---- the configuration --------------------------------------------------------------------------------


def test_a_configuration_whose_drift_arguments_lack_the_dry_run_is_refused_by_field(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ws = make_workspace(
        tmp_path, config_text=configuration(uv_args=["lock", "--upgrade"])
    )
    untouched = Untouched(ws, monkeypatch)

    untouched.refused(ws.args(), 2, "drift.uv_args")


def test_a_configuration_with_a_flag_that_could_write_the_lock_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ws = make_workspace(
        tmp_path,
        config_text=configuration(uv_args=["lock", "--dry-run", "--upgrade-package"]),
    )
    untouched = Untouched(ws, monkeypatch)

    untouched.refused(ws.args(), 2, "drift.uv_args")


def test_a_missing_configuration_is_a_bad_input(
    ws: Workspace, untouched: Untouched
) -> None:
    untouched.refused(ws.args(config=ws.root / "nowhere.yaml"), 2, "maintain config")


def test_a_configuration_that_is_not_yaml_is_refused_without_quoting_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ws = make_workspace(tmp_path, config_text=f"{MARKER}: [unclosed\n")
    untouched = Untouched(ws, monkeypatch)

    untouched.refused(ws.args(), 2, "invalid maintain config")


def test_a_configuration_with_an_unknown_key_is_refused_by_name(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    document = yaml.safe_load(SHIPPED_CONFIG.read_text(encoding="utf-8"))
    document["surprise"] = MARKER
    ws = make_workspace(tmp_path, config_text=yaml.safe_dump(document))
    untouched = Untouched(ws, monkeypatch)

    untouched.refused(ws.args(), 2, "surprise")


# ---- the project --------------------------------------------------------------------------------------


def test_a_missing_lock_is_a_data_error(ws: Workspace, untouched: Untouched) -> None:
    (ws.project / "uv.lock").unlink()
    untouched.before = snapshot(ws.root)

    untouched.refused(ws.args(), 3, "uv.lock")


def test_a_missing_pyproject_is_a_data_error(
    ws: Workspace, untouched: Untouched
) -> None:
    (ws.project / "pyproject.toml").unlink()
    untouched.before = snapshot(ws.root)

    untouched.refused(ws.args(), 3, "pyproject.toml")


@pytest.mark.parametrize(
    "content",
    [
        f"{MARKER} = [\n".encode(),
        b"\xff\xfe not utf-8",
        b"version = 2\nrevision = 5\n",
        b'version = 1\n[[package]]\nname = "a"\n',
        b'version = 1\n[[package]]\nversion = "1"\n',
        b'version = 1\npackage = "x"\n',
    ],
    ids=["not-toml", "not-utf8", "format-2", "no-version", "no-name", "not-a-list"],
)
def test_an_invalid_lock_is_a_data_error_that_quotes_nothing(
    ws: Workspace, untouched: Untouched, content: bytes
) -> None:
    (ws.project / "uv.lock").write_bytes(content)
    untouched.before = snapshot(ws.root)

    untouched.refused(ws.args(), 3, "uv.lock")


def test_a_pyproject_that_is_not_toml_is_a_data_error(
    ws: Workspace, untouched: Untouched
) -> None:
    (ws.project / "pyproject.toml").write_text(f"{MARKER} = [\n")
    untouched.before = snapshot(ws.root)

    untouched.refused(ws.args(), 3, "pyproject.toml")


def test_a_project_directory_that_does_not_exist_is_a_data_error(
    ws: Workspace, untouched: Untouched
) -> None:
    untouched.refused(ws.args(project_dir=ws.root / "nowhere"), 3, "pyproject.toml")


# ---- the output directory -----------------------------------------------------------------------------


def test_the_output_directory_is_not_the_project_directory(
    ws: Workspace, untouched: Untouched
) -> None:
    untouched.refused(ws.args(out_dir=ws.project), 2, "--out-dir")


def test_the_output_directory_is_not_the_project_directory_however_it_is_spelled(
    ws: Workspace, untouched: Untouched
) -> None:
    spelled = ws.project / ".." / ws.project.name

    untouched.refused(ws.args(out_dir=spelled), 2, "--out-dir")


def test_the_output_directory_is_not_the_directory_of_the_configuration(
    ws: Workspace, untouched: Untouched
) -> None:
    untouched.refused(ws.args(out_dir=ws.config.parent), 2, "--out-dir")


def test_the_output_directory_is_not_the_directory_of_a_linked_input(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ws = make_workspace(tmp_path)
    shelf = tmp_path / "shelf"
    shelf.mkdir()
    real = shelf / "uv.lock"
    shutil.move(ws.project / "uv.lock", real)
    (ws.project / "uv.lock").symlink_to(real)
    untouched = Untouched(ws, monkeypatch)

    untouched.refused(ws.args(out_dir=shelf), 2, "--out-dir")


def test_a_foreign_manifest_in_the_output_directory_is_refused_and_nothing_is_touched(
    ws: Workspace, monkeypatch: pytest.MonkeyPatch
) -> None:
    ws.out.mkdir()
    (ws.out / "keep.txt").write_text("the other stage's file")
    (ws.out / "maintain_report.json").write_text("old")
    store.write(
        ws.out,
        Manifest(
            stage="export_parity",
            created="2026-01-01T00:00:00+00:00",
            code_version="abc1234",
            config_sha256="0" * 64,
            seed=-1,
            outputs=[],
        ),
    )
    runner, doctor = FakeRunner(), Doctor([])
    install(monkeypatch, runner=runner, doctor=doctor)
    before = snapshot(ws.root)

    result = invoke(ws.args())

    assert result.exit_code == 3, result.output
    assert "export_parity" in result.stderr
    assert snapshot(ws.root) == before
    assert runner.calls == [] and doctor.calls == 0


def test_a_foreign_manifest_is_refused_before_the_files_are_parsed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The configuration is invalid (exit 2 if it were read): the refusal that comes first wins.
    ws = make_workspace(
        tmp_path, config_text=configuration(uv_args=["lock", "--upgrade"])
    )
    ws.out.mkdir()
    store.write(
        ws.out,
        Manifest(
            stage="export_parity",
            created="2026-01-01T00:00:00+00:00",
            code_version="abc1234",
            config_sha256="0" * 64,
            seed=-1,
            outputs=[],
        ),
    )
    install(monkeypatch, runner=FakeRunner(), doctor=Doctor([]))

    result = invoke(ws.args())

    assert result.exit_code == 3, result.output
    assert "export_parity" in result.stderr


def test_an_output_directory_that_cannot_be_made_is_exit_1_with_nothing_run(
    ws: Workspace, monkeypatch: pytest.MonkeyPatch
) -> None:
    blocker = ws.root / "a-file"
    blocker.write_text("not a directory")
    runner, doctor = FakeRunner(), Doctor([])
    install(monkeypatch, runner=runner, doctor=doctor)
    before = snapshot(ws.root)

    result = invoke(ws.args(out_dir=blocker / "out"))

    assert result.exit_code == 1, result.output
    assert "cannot write" in result.stderr
    assert snapshot(ws.root) == before
    assert runner.calls == [] and doctor.calls == 0


# ---- a file that changes between its hashing and its reading ------------------------------------------


@pytest.mark.parametrize("which", ["pyproject", "lock", "config"])
def test_a_file_that_changes_after_it_was_hashed_is_exit_2_and_nothing_is_written(
    ws: Workspace,
    untouched: Untouched,
    monkeypatch: pytest.MonkeyPatch,
    which: str,
) -> None:
    changing = {
        "pyproject": ws.project / "pyproject.toml",
        "lock": ws.project / "uv.lock",
        "config": ws.config,
    }[which]
    real = maintain_inputs.hash_input

    def hashing(path: Path, **options: Any) -> Any:
        entry = real(path, **options)
        if path == changing:
            path.write_bytes(path.read_bytes() + b"\n# changed after the hash\n")
        return entry

    monkeypatch.setattr(maintain_inputs, "hash_input", hashing)
    untouched.before = snapshot(ws.root)

    result = invoke(ws.args())

    assert result.exit_code == 2, result.output
    assert "changed after it was hashed" in result.stderr
    assert not ws.out.exists()
    assert untouched.runner.calls == [] and untouched.doctor.calls == 0
    assert set(snapshot(ws.root)) == set(untouched.before)


def test_a_lock_that_is_replaced_by_another_one_after_its_hash_is_not_read(
    ws: Workspace, untouched: Untouched, monkeypatch: pytest.MonkeyPatch
) -> None:
    real = maintain_inputs.hash_input

    def hashing(path: Path, **options: Any) -> Any:
        entry = real(path, **options)
        if path.name == "uv.lock":
            path.write_bytes(lock_text({"swapped": ("9.9",)}))
        return entry

    monkeypatch.setattr(maintain_inputs, "hash_input", hashing)

    result = invoke(ws.args())

    assert result.exit_code == 2, result.output
    assert not ws.out.exists()
