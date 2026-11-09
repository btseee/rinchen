"""Tests for how `export_checkpoint` makes its output directory (D10): the converter refuses a
directory that exists, so it writes into a private directory THIS call made next to the output
(a child of it that does not exist yet), the two files beside the model are written there, and
the result is renamed onto the output directory. A failure removes the private directory and
only that: a directory someone else made at the output path in the meantime is refused and
never touched, which a cleanup that removes "the output directory" could not promise.

The converter is `tests.support.distill_exports.fake_converter`: nothing starts a process.
The staging itself moved to `rinchen.export.ct2.convert` in D11 (`test_export_convert*.py`);
these tests keep D10's expectations of it through `export_checkpoint`.
"""

from __future__ import annotations

import errno
import os
from pathlib import Path
from typing import Any

import pytest

from rinchen.errors import ConfigError, UpstreamError
from rinchen.export import ct2
from rinchen.export.eole_ct2 import export_log_path
from rinchen.export.record import ModelRecord
from rinchen.train.distill_export import export_checkpoint
from tests.support.distill_cli import STATED_DIRECTIONS
from tests.support.distill_exports import FakeConverter, fake_converter
from tests.support.step_dirs import (
    FabricatedRun,
    eole_model_section,
    make_finished_run,
    tree_digest,
    write_spm_model,
)


@pytest.fixture
def teacher(tmp_path: Path) -> FabricatedRun:
    return make_finished_run(
        tmp_path / "teacher",
        model=eole_model_section(encoder_layers=2, decoder_layers=1),
    )


@pytest.fixture
def spm_model(tmp_path: Path) -> Path:
    return write_spm_model(tmp_path / "spm")


@pytest.fixture
def converter(monkeypatch: pytest.MonkeyPatch) -> FakeConverter:
    return fake_converter(monkeypatch)


@pytest.fixture
def out_dir(tmp_path: Path) -> Path:
    return tmp_path / "export"


def export(teacher: FabricatedRun, target: Path, spm_model: Path) -> ModelRecord:
    return export_checkpoint(
        teacher.final,
        target,
        spm_model=spm_model,
        tag_placement="src",
        directions=STATED_DIRECTIONS,
    ).record


def names(directory: Path) -> set[str]:
    return {path.name for path in directory.iterdir()}


# ---- where the converter writes -------------------------------------------------------


def test_the_converter_writes_into_a_private_directory_beside_the_output_not_into_it(
    teacher: FabricatedRun,
    spm_model: Path,
    converter: FakeConverter,
    out_dir: Path,
) -> None:
    export(teacher, out_dir, spm_model)

    (call,) = converter.calls
    assert call.out_dir != out_dir
    assert call.out_dir.parent.parent == out_dir.parent
    assert call.out_dir.parent.name.startswith(".export.")
    assert call.out_dir.name == "export"
    assert call.checkpoint == teacher.final
    assert call.log_path == export_log_path(out_dir)


def test_the_converter_is_told_to_name_the_output_directory_in_its_messages(
    teacher: FabricatedRun,
    spm_model: Path,
    converter: FakeConverter,
    out_dir: Path,
) -> None:
    export(teacher, out_dir, spm_model)

    (call,) = converter.calls
    assert call.reported_dir == out_dir


def test_a_successful_export_leaves_the_directory_and_its_log_and_nothing_else(
    teacher: FabricatedRun,
    spm_model: Path,
    converter: FakeConverter,
    out_dir: Path,
    tmp_path: Path,
) -> None:
    before = names(tmp_path)

    export(teacher, out_dir, spm_model)

    assert names(tmp_path) - before == {"export", "export.export.log"}
    assert names(out_dir) == {
        "model.bin",
        "shared_vocabulary.json",
        "config.json",
        "spm.model",
        "rinchen.json",
        "manifest.json",
    }
    assert export_log_path(out_dir).read_text(encoding="utf-8") == "converter log\n"


def test_a_missing_parent_directory_of_the_output_is_made(
    teacher: FabricatedRun, spm_model: Path, converter: FakeConverter, tmp_path: Path
) -> None:
    target = tmp_path / "deep" / "er" / "export"

    export(teacher, target, spm_model)

    assert (target / "model.bin").is_file()
    assert names(target.parent) == {"export", "export.export.log"}


def test_the_private_directory_of_another_call_is_not_touched(
    teacher: FabricatedRun,
    spm_model: Path,
    converter: FakeConverter,
    out_dir: Path,
    tmp_path: Path,
) -> None:
    # What a killed run leaves behind cannot be told from a run in progress.
    stale = tmp_path / ".export.stale"
    stale.mkdir()
    (stale / "model.bin").write_bytes(b"theirs")

    export(teacher, out_dir, spm_model)

    assert (stale / "model.bin").read_bytes() == b"theirs"


# ---- someone else made the output path while the converter ran -------------------------


def racing_converter(
    monkeypatch: pytest.MonkeyPatch, make_theirs: Any
) -> FakeConverter:
    inner = FakeConverter()

    def racing(ckpt_dir: Path, built: Path, **options: Any) -> Path:
        make_theirs()
        return inner(ckpt_dir, built, **options)

    monkeypatch.setattr(ct2, "export_ct2", racing)
    return inner


def test_a_directory_made_by_someone_else_while_the_converter_ran_is_refused_and_kept(
    teacher: FabricatedRun,
    spm_model: Path,
    out_dir: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def theirs() -> None:
        out_dir.mkdir()
        (out_dir / "theirs.txt").write_text("keep", encoding="utf-8")

    inner = racing_converter(monkeypatch, theirs)
    before = names(tmp_path)

    with pytest.raises(ConfigError, match=f"{out_dir} exists"):
        export(teacher, out_dir, spm_model)

    assert len(inner.calls) == 1
    assert names(out_dir) == {"theirs.txt"}
    assert (out_dir / "theirs.txt").read_text(encoding="utf-8") == "keep"
    # Only the converter's log is left of this call; the private directory is gone.
    assert names(tmp_path) - before == {"export", "export.export.log"}


def test_a_file_made_at_the_output_path_while_the_converter_ran_is_refused_and_kept(
    teacher: FabricatedRun,
    spm_model: Path,
    out_dir: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    racing_converter(
        monkeypatch, lambda: out_dir.write_text("theirs", encoding="utf-8")
    )

    with pytest.raises(ConfigError, match=f"{out_dir} exists"):
        export(teacher, out_dir, spm_model)

    assert out_dir.read_text(encoding="utf-8") == "theirs"
    assert not any(name.startswith(".export.") for name in names(tmp_path))


# ---- a failure removes what this call made, and only that --------------------------------


@pytest.mark.parametrize(
    "error",
    [
        UpstreamError("the converter failed"),
        KeyboardInterrupt(),
        OSError(errno.ENOSPC, "No space left on device"),
    ],
    ids=["failed", "interrupted", "disk full"],
)
def test_a_converter_that_fails_leaves_its_log_and_no_directory_of_any_kind(
    teacher: FabricatedRun,
    spm_model: Path,
    out_dir: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    error: BaseException,
) -> None:
    fake_converter(monkeypatch, error=error)
    before = names(tmp_path)
    digest = tree_digest(teacher.run_dir)

    with pytest.raises(type(error)):
        export(teacher, out_dir, spm_model)

    assert names(tmp_path) - before == {"export.export.log"}
    assert tree_digest(teacher.run_dir) == digest


@pytest.mark.parametrize("name", ["model.bin", "shared_vocabulary.json", "config.json"])
def test_a_converter_that_leaves_a_file_out_names_the_output_directory(
    teacher: FabricatedRun,
    spm_model: Path,
    out_dir: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    name: str,
) -> None:
    fake_converter(monkeypatch, skip=(name,))
    before = names(tmp_path)

    with pytest.raises(UpstreamError) as raised:
        export(teacher, out_dir, spm_model)

    message = str(raised.value)
    assert f"left no {name} in {out_dir} " in message
    assert f"the converter's log is {export_log_path(out_dir)}" in message
    assert names(tmp_path) - before == {"export.export.log"}


class ExitingZeroConverter:
    """Stands in for the process the real `export_ct2` runs (`run_logged`): it exits zero
    after it made the directory it was told to write to and put `model_bytes` in it as
    `model.bin` (nothing when None). `built` is that directory."""

    def __init__(self, model_bytes: bytes | None) -> None:
        self.model_bytes = model_bytes
        self.built: list[Path] = []

    def __call__(self, command: Any, *, log_path: Path, **options: Any) -> None:
        built = Path(command[command.index("--output_dir") + 1])
        self.built.append(built)
        log_path.write_text("last words of the converter\n", encoding="utf-8")
        built.mkdir()
        if self.model_bytes is not None:
            (built / "model.bin").write_bytes(self.model_bytes)


@pytest.mark.parametrize("model_bytes", [None, b""], ids=["missing", "empty"])
def test_a_converter_that_exits_zero_without_a_model_names_the_output_not_the_private_path(
    teacher: FabricatedRun,
    spm_model: Path,
    out_dir: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    model_bytes: bytes | None,
) -> None:
    # The real `export_ct2` runs here, over a private directory that no longer exists when
    # the message is read: the message names the directory the caller asked for.
    process = ExitingZeroConverter(model_bytes)
    monkeypatch.setattr("rinchen.export.eole_ct2.run_logged", process)
    before = names(tmp_path)

    with pytest.raises(UpstreamError) as raised:
        export(teacher, out_dir, spm_model)

    message = str(raised.value)
    (private,) = process.built
    assert private != out_dir
    assert str(private.parent) not in message
    assert f"left no model.bin in {out_dir}, or an empty one" in message
    assert f"log: {export_log_path(out_dir)};" in message
    assert message.endswith("last words of the converter")
    assert names(tmp_path) - before == {"export.export.log"}


def test_a_rename_that_fails_removes_the_private_directory_and_makes_no_output(
    teacher: FabricatedRun,
    spm_model: Path,
    converter: FakeConverter,
    out_dir: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    before = names(tmp_path)

    def cross_device(source: Any, target: Any) -> None:
        raise OSError(errno.EXDEV, "Invalid cross-device link")

    with monkeypatch.context() as patched:
        patched.setattr(os, "rename", cross_device)
        with pytest.raises(OSError, match="cross-device"):
            export(teacher, out_dir, spm_model)

    assert names(tmp_path) - before == {"export.export.log"}


def test_an_interrupt_right_after_the_rename_removes_the_output_this_call_made(
    teacher: FabricatedRun,
    spm_model: Path,
    converter: FakeConverter,
    out_dir: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The last thing the call does once the output is in place is to remove its now empty
    # private directory. An interrupt there must not leave an export that no manifest will
    # ever vouch for in the way of the next attempt.
    real = Path.rmdir

    def interrupted(self: Path) -> None:
        if self.name.startswith(".export."):
            raise KeyboardInterrupt
        real(self)

    before = names(tmp_path)
    with monkeypatch.context() as patched:
        patched.setattr(Path, "rmdir", interrupted)
        with pytest.raises(KeyboardInterrupt):
            export(teacher, out_dir, spm_model)

    assert names(tmp_path) - before == {"export.export.log"}


def test_a_private_directory_that_cannot_be_removed_after_the_rename_does_not_fail_the_export(
    teacher: FabricatedRun,
    spm_model: Path,
    converter: FakeConverter,
    out_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def busy(self: Path) -> None:
        raise OSError(errno.EBUSY, "Device or resource busy")

    with monkeypatch.context() as patched:
        patched.setattr(Path, "rmdir", busy)
        record = export(teacher, out_dir, spm_model)

    assert record.tag_placement == "src"
    assert (out_dir / "rinchen.json").is_file()
