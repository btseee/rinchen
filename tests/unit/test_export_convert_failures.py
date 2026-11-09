"""Tests for what happens when `convert` (`rinchen.export.ct2`) fails half way (D11).

The converter refuses a directory that exists, so it writes into a private directory THIS
call made beside the output (a child of it that does not exist yet), the files beside the
model and the manifest are written there, and the result is renamed onto the output
directory. A failure removes the private directory and only that: a directory someone else
made at the output path in the meantime is refused and never touched, which a cleanup that
removes "the output directory" could not promise. The converter's log stays beside the
output.

The converter is `tests.support.model_exports.fake_convert_converter`, or the real
`export_ct2` over a fake process (`run_logged`): nothing starts a process.
"""

from __future__ import annotations

import errno
import json
import os
from pathlib import Path
from typing import Any

import pytest

from rinchen.errors import ConfigError, UpstreamError
from rinchen.export import ct2
from rinchen.export.ct2 import ConvertResult, ExportFacts, convert
from rinchen.export.eole_ct2 import export_log_path
from rinchen.manifest import store
from tests.support.model_exports import (
    checkpoint_inputs,
    export_facts,
    fake_convert_converter,
)
from tests.support.step_dirs import SOURCE_VOCABULARY, SPM_BYTES, sha256


@pytest.fixture(autouse=True)
def no_git(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ct2, "code_version", lambda: "abc1234")


@pytest.fixture
def checkpoint(tmp_path: Path) -> Path:
    path = tmp_path / "run" / "ckpt" / "step_200"
    path.mkdir(parents=True)
    return path


@pytest.fixture
def out_dir(tmp_path: Path) -> Path:
    return tmp_path / "export"


def names(directory: Path) -> set[str]:
    return {path.name for path in directory.iterdir()}


def run_convert(
    checkpoint: Path, out_dir: Path, *, using: ExportFacts | None = None, **options: Any
) -> ConvertResult:
    return convert(
        checkpoint,
        out_dir,
        facts=export_facts() if using is None else using,
        spm_bytes=SPM_BYTES,
        inputs=checkpoint_inputs(checkpoint),
        **options,
    )


def only_the_log_is_left(tmp_path: Path, before: set[str]) -> bool:
    return names(tmp_path) - before == {"export.export.log"}


# ---- the converter fails -----------------------------------------------------------------


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
    checkpoint: Path,
    out_dir: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    error: BaseException,
) -> None:
    fake_convert_converter(monkeypatch, error=error)
    before = names(tmp_path)

    with pytest.raises(type(error)):
        run_convert(checkpoint, out_dir)

    assert only_the_log_is_left(tmp_path, before)
    assert export_log_path(out_dir).read_text(encoding="utf-8") == "converter log\n"


def test_a_converter_that_fails_after_a_partial_model_leaves_no_trace_of_it(
    checkpoint: Path,
    out_dir: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_convert_converter(
        monkeypatch, error=UpstreamError("killed"), partial_model=b"half a model"
    )
    before = names(tmp_path)

    with pytest.raises(UpstreamError, match="killed"):
        run_convert(checkpoint, out_dir)

    assert only_the_log_is_left(tmp_path, before)
    assert not any(tmp_path.rglob("model.bin"))


@pytest.mark.parametrize("name", ["model.bin", "shared_vocabulary.json", "config.json"])
def test_a_converter_that_leaves_a_file_out_names_the_output_directory(
    checkpoint: Path,
    out_dir: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    name: str,
) -> None:
    fake_convert_converter(monkeypatch, skip=(name,))
    before = names(tmp_path)

    with pytest.raises(UpstreamError) as raised:
        run_convert(checkpoint, out_dir)

    message = str(raised.value)
    assert f"left no {name} in {out_dir} " in message
    assert f"the converter's log is {export_log_path(out_dir)}" in message
    assert only_the_log_is_left(tmp_path, before)


class ExitingZeroConverter:
    """Stands in for the process the real `export_ct2` runs (`run_logged`): it exits zero
    after it made the directory it was told to write to and put `model_bytes` in it as
    `model.bin` (nothing when None). `built` is that directory, `commands` what it was
    told."""

    def __init__(self, model_bytes: bytes | None) -> None:
        self.model_bytes = model_bytes
        self.built: list[Path] = []
        self.commands: list[list[str]] = []

    def __call__(self, command: Any, *, log_path: Path, **options: Any) -> None:
        built = Path(command[command.index("--output_dir") + 1])
        self.built.append(built)
        self.commands.append(list(command))
        log_path.write_text("last words of the converter\n", encoding="utf-8")
        built.mkdir()
        if self.model_bytes is not None:
            (built / "model.bin").write_bytes(self.model_bytes)


@pytest.mark.parametrize("model_bytes", [None, b""], ids=["missing", "empty"])
def test_a_converter_that_exits_zero_without_a_model_names_the_output_not_the_private_path(
    checkpoint: Path,
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
        run_convert(checkpoint, out_dir)

    message = str(raised.value)
    (private,) = process.built
    assert private != out_dir
    assert str(private.parent) not in message
    assert f"left no model.bin in {out_dir}, or an empty one" in message
    assert f"log: {export_log_path(out_dir)};" in message
    assert message.endswith("last words of the converter")
    assert only_the_log_is_left(tmp_path, before)


@pytest.mark.parametrize("quantization", ["float32", "float16", "int8", "int8_float16"])
def test_the_quantization_is_on_the_command_line_of_the_real_converter_call(
    checkpoint: Path,
    out_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
    quantization: str,
) -> None:
    process = ExitingZeroConverter(None)
    monkeypatch.setattr("rinchen.export.eole_ct2.run_logged", process)

    with pytest.raises(UpstreamError):
        run_convert(checkpoint, out_dir, using=export_facts(quantization=quantization))

    (command,) = process.commands
    assert command[command.index("--quantization") + 1] == quantization
    assert command.count("--quantization") == 1
    assert command[command.index("--model_path") + 1] == str(checkpoint)


# ---- the converter's vocabulary is not the checkpoint's ----------------------------------


def vocabulary_of(count: int) -> bytes:
    return json.dumps([f"piece{index}" for index in range(count)]).encode("utf-8")


@pytest.mark.parametrize("count", [0, 1, 123, 125, 1000])
def test_a_converter_whose_vocabulary_has_another_length_is_refused_with_nothing_left(
    checkpoint: Path,
    out_dir: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    count: int,
) -> None:
    fake_convert_converter(
        monkeypatch, replace={"shared_vocabulary.json": vocabulary_of(count)}
    )
    before = names(tmp_path)

    with pytest.raises(UpstreamError) as raised:
        run_convert(checkpoint, out_dir)

    message = str(raised.value)
    assert f"{out_dir / 'shared_vocabulary.json'} has {count} entries" in message
    assert f"vocabulary has {len(SOURCE_VOCABULARY)}:" in message
    assert only_the_log_is_left(tmp_path, before)


@pytest.mark.parametrize(
    "content",
    [b"not json", b'{"a": 1}', b'["a", 2]', b""],
    ids=["not json", "object", "non-string", "empty"],
)
def test_a_converter_vocabulary_that_is_not_a_list_of_strings_is_refused(
    checkpoint: Path,
    out_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
    content: bytes,
) -> None:
    fake_convert_converter(monkeypatch, replace={"shared_vocabulary.json": content})

    with pytest.raises(UpstreamError) as raised:
        run_convert(checkpoint, out_dir)

    assert f"{out_dir / 'shared_vocabulary.json'} is not a JSON list of strings" in str(
        raised.value
    )
    assert not out_dir.exists()


def test_a_converter_vocabulary_that_cannot_be_read_is_refused(
    checkpoint: Path,
    out_dir: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A directory where the file should be: `is_file` passes nothing, so the converter
    # "left no" file; the unreadable case is a file that vanishes after the check.
    real = Path.read_bytes

    def unreadable(self: Path) -> bytes:
        if self.name == "shared_vocabulary.json":
            raise OSError(errno.EIO, "Input/output error")
        return real(self)

    fake_convert_converter(monkeypatch)
    before = names(tmp_path)
    with monkeypatch.context() as patched:
        patched.setattr(Path, "read_bytes", unreadable)
        with pytest.raises(UpstreamError, match="cannot read .*shared_vocabulary.json"):
            run_convert(checkpoint, out_dir)

    assert only_the_log_is_left(tmp_path, before)


# ---- writing the files and the manifest -------------------------------------------------


@pytest.mark.parametrize(
    "error",
    [OSError(errno.ENOSPC, "No space left on device"), KeyboardInterrupt()],
    ids=["disk full", "interrupted"],
)
def test_a_manifest_that_cannot_be_written_leaves_no_directory(
    checkpoint: Path,
    out_dir: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    error: BaseException,
) -> None:
    fake_convert_converter(monkeypatch)

    def failing(stage_dir: Path, manifest: object) -> Path:
        raise error

    before = names(tmp_path)
    with monkeypatch.context() as patched:
        patched.setattr(store, "write", failing)
        with pytest.raises(type(error)):
            run_convert(checkpoint, out_dir)

    assert only_the_log_is_left(tmp_path, before)


def test_a_file_that_cannot_be_written_beside_the_model_leaves_no_directory(
    checkpoint: Path,
    out_dir: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_convert_converter(monkeypatch)

    def full(path: Path, content: bytes, **options: Any) -> None:
        raise OSError(errno.ENOSPC, "No space left on device")

    before = names(tmp_path)
    with monkeypatch.context() as patched:
        patched.setattr(ct2, "replace_file", full)
        with pytest.raises(OSError, match="No space"):
            run_convert(checkpoint, out_dir)

    assert only_the_log_is_left(tmp_path, before)


def test_a_rename_that_fails_removes_the_private_directory_and_makes_no_output(
    checkpoint: Path,
    out_dir: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_convert_converter(monkeypatch)
    before = names(tmp_path)

    def cross_device(source: Any, target: Any) -> None:
        raise OSError(errno.EXDEV, "Invalid cross-device link")

    with monkeypatch.context() as patched:
        patched.setattr(os, "rename", cross_device)
        with pytest.raises(OSError, match="cross-device"):
            run_convert(checkpoint, out_dir)

    assert only_the_log_is_left(tmp_path, before)


def test_an_interrupt_right_after_the_rename_removes_the_output_this_call_made(
    checkpoint: Path,
    out_dir: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The last thing the call does once the output is in place is to remove its now empty
    # private directory. An interrupt there must not leave an export in the way of the
    # next attempt: the one who pressed Ctrl-C wanted nothing.
    fake_convert_converter(monkeypatch)
    real = Path.rmdir

    def interrupted(self: Path) -> None:
        if self.name.startswith(".export."):
            raise KeyboardInterrupt
        real(self)

    before = names(tmp_path)
    with monkeypatch.context() as patched:
        patched.setattr(Path, "rmdir", interrupted)
        with pytest.raises(KeyboardInterrupt):
            run_convert(checkpoint, out_dir)

    assert only_the_log_is_left(tmp_path, before)


def test_a_private_directory_that_cannot_be_removed_after_the_rename_does_not_fail_the_export(
    checkpoint: Path,
    out_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_convert_converter(monkeypatch)

    def busy(self: Path) -> None:
        raise OSError(errno.EBUSY, "Device or resource busy")

    with monkeypatch.context() as patched:
        patched.setattr(Path, "rmdir", busy)
        result = run_convert(checkpoint, out_dir)

    assert isinstance(result, ConvertResult)
    assert (out_dir / "manifest.json").is_file()
    assert (out_dir / "rinchen.json").is_file()


# ---- someone else made the output path while the converter ran ---------------------------


def test_a_directory_made_by_someone_else_while_the_converter_ran_is_refused_and_kept(
    checkpoint: Path,
    out_dir: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def theirs() -> None:
        out_dir.mkdir()
        (out_dir / "theirs.txt").write_text("keep", encoding="utf-8")

    inner = fake_convert_converter(monkeypatch, during=theirs)
    before = names(tmp_path)

    with pytest.raises(ConfigError, match=f"{out_dir} exists"):
        run_convert(checkpoint, out_dir)

    assert len(inner.calls) == 1
    assert names(out_dir) == {"theirs.txt"}
    assert (out_dir / "theirs.txt").read_text(encoding="utf-8") == "keep"
    # Only the converter's log is left of this call; the private directory is gone.
    assert names(tmp_path) - before == {"export", "export.export.log"}


def test_a_file_made_at_the_output_path_while_the_converter_ran_is_refused_and_kept(
    checkpoint: Path,
    out_dir: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_convert_converter(
        monkeypatch,
        during=lambda: out_dir.write_text("theirs", encoding="utf-8"),
    )

    with pytest.raises(ConfigError, match=f"{out_dir} exists"):
        run_convert(checkpoint, out_dir)

    assert out_dir.read_text(encoding="utf-8") == "theirs"
    assert not any(name.startswith(".export.") for name in names(tmp_path))


def test_an_empty_directory_made_at_the_output_path_while_the_converter_ran_is_refused_and_kept(
    checkpoint: Path,
    out_dir: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A rename would replace an empty directory without a word: only the check just before
    # it keeps what someone else made.
    fake_convert_converter(monkeypatch, during=out_dir.mkdir)

    with pytest.raises(ConfigError, match=f"{out_dir} exists"):
        run_convert(checkpoint, out_dir)

    assert out_dir.is_dir()
    assert names(out_dir) == set()
    assert not any(name.startswith(".export.") for name in names(tmp_path))


def test_a_directory_that_takes_the_path_in_the_instant_of_the_rename_is_kept(
    checkpoint: Path,
    out_dir: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_convert_converter(monkeypatch)

    def too_late(source: Any, target: Any) -> None:
        Path(target).mkdir()
        (Path(target) / "theirs.txt").write_text("keep", encoding="utf-8")
        raise OSError(errno.ENOTEMPTY, "Directory not empty")

    with monkeypatch.context() as patched:
        patched.setattr(os, "rename", too_late)
        with pytest.raises(ConfigError, match=f"{out_dir} exists"):
            run_convert(checkpoint, out_dir)

    assert names(out_dir) == {"theirs.txt"}
    assert not any(name.startswith(".export.") for name in names(tmp_path))


def test_the_output_of_an_earlier_successful_call_is_never_removed_by_a_later_failure(
    checkpoint: Path,
    out_dir: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_convert_converter(monkeypatch)
    run_convert(checkpoint, out_dir)
    digest = sha256(out_dir / "model.bin")
    fake_convert_converter(monkeypatch, error=UpstreamError("again"))

    with pytest.raises(ConfigError, match=f"{out_dir} exists"):
        run_convert(checkpoint, out_dir)

    assert sha256(out_dir / "model.bin") == digest
    assert (out_dir / "manifest.json").is_file()


# ---- the arguments the caller gave stay as they were ----------------------------------


def test_what_the_caller_passed_is_not_changed(
    checkpoint: Path, out_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake_convert_converter(monkeypatch)
    given: list[Any] = checkpoint_inputs(checkpoint)
    snapshot = [item.model_copy() for item in given]
    given_facts: ExportFacts = export_facts()

    convert(
        checkpoint,
        out_dir,
        facts=given_facts,
        spm_bytes=SPM_BYTES,
        inputs=given,
    )

    assert given == snapshot
    assert given_facts == export_facts()
