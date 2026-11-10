"""Shared setup of the `rinchen eval robustness` tests (D12, Task 7).

The command touches the model and the converter in two places only, `open_engine` and
`load_converter` of `rinchen.cli.eval_robustness`; `install` replaces both by seams that record
their calls and hand out a `FakeRobustEngine` (and a converter of the test's own). Nothing here
loads a model, imports `ctranslate2` or touches a GPU.

A `RobustWorkspace` is the files of one run under a temporary directory: copies of the golden
suite and the golden hypotheses (`tests.support.robust_golden`), a fabricated export that
`verify_export` vouches for, the files of a converter and an output directory that does not exist.
`FakeRobustEngine` answers the requests of the golden suite by row order from the golden
hypotheses (`answers_from_golden`) or with a function of the test's own.

`record_replaces` and `fail_replace` see every file the command writes by name: a file appears
by a rename (`replace_file`, the manifest's store), so the renames are the writes.
"""

from __future__ import annotations

import hashlib
import shutil
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final

import pytest
from click.testing import Result
from typer.testing import CliRunner

from rinchen.cli.main import app
from rinchen.infer.engine_types import (
    DecodeOptions,
    LoadOptions,
    ModelInfo,
    Row,
    Translation,
)
from tests.support.model_exports import WrittenModel, write_model_export
from tests.support.robust_golden import (
    GOLDEN_HYPOTHESES,
    GOLDEN_SUITE,
    golden_hypotheses,
)
from tests.support.translate_cli import translation_of

runner = CliRunner()

COMMAND_MODULE: Final = "rinchen.cli.eval_robustness"
RESULT_FILES: Final = ("robustness.json", "report.json", "hypotheses.jsonl")
MANIFEST_FILE: Final = "manifest.json"
CONVERTER_FILES: Final = ("model.bin", "shared_vocabulary.json", "config.json")
CONVERTER_SPM: Final = "char.model"

Answer = Callable[[Sequence[Row]], list[Translation]]


def answers_from_golden(rows: Sequence[Row]) -> list[Translation]:
    """The golden hypotheses, in request order, as the engine's translations of `rows`: the
    rows must be exactly the 33 requests of the golden suite."""
    hypotheses = list(golden_hypotheses().values())
    return [
        translation_of(row, hypothesis.text, hypothesis.status)
        for row, hypothesis in zip(rows, hypotheses, strict=True)
    ]


def echo_answers(rows: Sequence[Row]) -> list[Translation]:
    """Every text translated as itself, `ok`."""
    return [translation_of(row, row.text, "ok") for row in rows]


def model_info(written: WrittenModel, *, device: str = "cpu") -> ModelInfo:
    """What a loaded engine says about the fabricated export `written`."""
    record = written.record
    return ModelInfo(
        name=record.name,
        quantization=record.quantization,
        device=device,
        compute_type="float32",
        placement=record.tag_placement,
        directions=record.directions,
        targets=tuple(sorted({d.split("-")[1] for d in record.directions})),
        document_level=record.document_level,
        model_sha256=written.sha256["model.bin"],
        manifest_sha256=written.manifest_sha256,
    )


@dataclass(frozen=True)
class RowsCall:
    """One call of `translate`: the rows and the options it was given."""

    rows: list[Row]
    options: DecodeOptions | None


@dataclass
class FakeRobustEngine:
    """The engine as the command sees it: `info`, `translate` and `close`. `answer` turns
    the rows of a call into translations; `error` is raised by `translate` instead of
    answering; `events` says in what order things happened."""

    info: ModelInfo
    answer: Answer = answers_from_golden
    error: BaseException | None = None
    calls: list[RowsCall] = field(default_factory=list)
    closed: int = 0
    events: list[str] = field(default_factory=list)

    def translate(
        self, rows: Sequence[Row], options: DecodeOptions | None = None
    ) -> list[Translation]:
        self.events.append("translate")
        self.calls.append(RowsCall(list(rows), options))
        if self.error is not None:
            raise self.error
        return self.answer(rows)

    def close(self) -> None:
        self.closed += 1
        self.events.append("close")


@dataclass(frozen=True)
class OpenCall:
    """What the command asked `open_engine` for."""

    model: Path
    load: LoadOptions
    converter: Any


@dataclass(frozen=True)
class LoadCall:
    """What the command asked `load_converter` for."""

    ct2_dir: Path
    spm_model: Path
    device: str


@dataclass
class Seams:
    """The two seams: their calls, the engine and the converter they hand out, and the
    events of both (`open_converter`, `open`, then what the engine saw)."""

    engine: Any
    converter: Any
    opens: list[OpenCall] = field(default_factory=list)
    loads: list[LoadCall] = field(default_factory=list)
    events: list[str] = field(default_factory=list)


def install(
    monkeypatch: pytest.MonkeyPatch,
    engine: Any,
    *,
    converter: Any = None,
    open_error: BaseException | None = None,
    load_error: BaseException | None = None,
) -> Seams:
    """Replace `open_engine` and `load_converter` of the command by seams that record their
    calls and hand out `engine` and `converter`, or raise the given error. The engine's own
    `events` list is the seams' list."""
    seams = Seams(engine, converter)
    if hasattr(engine, "events"):
        seams.events = engine.events

    def open_engine(model: Path, load: LoadOptions, converter: Any) -> Any:
        seams.events.append("open")
        seams.opens.append(OpenCall(model, load, converter))
        if open_error is not None:
            raise open_error
        return seams.engine

    def load_converter(ct2_dir: Path, spm_model: Path, device: str) -> Any:
        seams.events.append("open_converter")
        seams.loads.append(LoadCall(ct2_dir, spm_model, device))
        if load_error is not None:
            raise load_error
        return seams.converter

    monkeypatch.setattr(f"{COMMAND_MODULE}.open_engine", open_engine)
    monkeypatch.setattr(f"{COMMAND_MODULE}.load_converter", load_converter)
    return seams


def sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@dataclass(frozen=True)
class RobustWorkspace:
    """The files of one run; `out_dir` does not exist."""

    root: Path
    suite: Path
    hypotheses: Path
    export: WrittenModel
    converter_ct2: Path
    converter_spm: Path
    out_dir: Path

    @property
    def manifest(self) -> Path:
        return self.out_dir / MANIFEST_FILE

    @property
    def result(self) -> Path:
        return self.out_dir / "robustness.json"

    @property
    def report(self) -> Path:
        return self.out_dir / "report.json"

    @property
    def hypotheses_out(self) -> Path:
        return self.out_dir / "hypotheses.jsonl"

    def engine(self, **options: Any) -> FakeRobustEngine:
        """A fake engine whose `info` is that of the fabricated export."""
        return FakeRobustEngine(info=model_info(self.export), **options)

    def hypotheses_args(
        self,
        *extra: str,
        suite: Path | None = None,
        hypotheses: Path | None = None,
        out_dir: Path | None = None,
    ) -> list[str]:
        """`eval robustness --suite S --hypotheses H --out-dir O` and `extra`."""
        return [
            "eval",
            "robustness",
            "--suite",
            str(self.suite if suite is None else suite),
            "--hypotheses",
            str(self.hypotheses if hypotheses is None else hypotheses),
            "--out-dir",
            str(self.out_dir if out_dir is None else out_dir),
            *extra,
        ]

    def model_args(
        self,
        *extra: str,
        suite: Path | None = None,
        model: Path | None = None,
        out_dir: Path | None = None,
    ) -> list[str]:
        """`eval robustness --suite S --model M --out-dir O` and `extra`."""
        return [
            "eval",
            "robustness",
            "--suite",
            str(self.suite if suite is None else suite),
            "--model",
            str(self.export.directory if model is None else model),
            "--out-dir",
            str(self.out_dir if out_dir is None else out_dir),
            *extra,
        ]

    def converter_args(self) -> list[str]:
        return [
            "--script-converter-ct2",
            str(self.converter_ct2),
            "--script-converter-spm",
            str(self.converter_spm),
        ]


def make_workspace(root: Path, *, quantization: str = "float32") -> RobustWorkspace:
    """The golden suite and hypotheses, an export of `quantization` and a converter under
    `root`."""
    inputs = root / "in"
    inputs.mkdir(parents=True)
    suite = shutil.copyfile(GOLDEN_SUITE, inputs / GOLDEN_SUITE.name)
    hypotheses = shutil.copyfile(GOLDEN_HYPOTHESES, inputs / GOLDEN_HYPOTHESES.name)
    ct2 = root / "converter" / "ct2"
    ct2.mkdir(parents=True)
    for name in CONVERTER_FILES:
        (ct2 / name).write_bytes(f"converter file {name}".encode())
    spm = root / "converter" / CONVERTER_SPM
    spm.write_bytes(b"converter character model")
    return RobustWorkspace(
        root=root,
        suite=Path(suite),
        hypotheses=Path(hypotheses),
        export=write_model_export(root / "model", quantization=quantization),
        converter_ct2=ct2,
        converter_spm=spm,
        out_dir=root / "out",
    )


def invoke(args: Sequence[str], *, env: dict[str, str] | None = None) -> Result:
    """Run `rinchen` with `args` in this process; the result keeps standard output and
    standard error apart (`result.stdout`, `result.stderr`)."""
    return runner.invoke(app, list(args), env=env)


def record_replaces(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Record the name of every file that appears by a rename, in order."""
    names: list[str] = []
    real = Path.replace

    def replacing(self: Path, target: Any) -> Any:
        names.append(Path(target).name)
        return real(self, target)

    monkeypatch.setattr(Path, "replace", replacing)
    return names


def fail_replace(
    monkeypatch: pytest.MonkeyPatch, name: str, error: OSError | None = None
) -> None:
    """Make the file `name` impossible to write: its rename raises `error`."""
    real = Path.replace

    def replacing(self: Path, target: Any) -> Any:
        if Path(target).name == name:
            raise OSError("disk full") if error is None else error
        return real(self, target)

    monkeypatch.setattr(Path, "replace", replacing)


def written(directory: Path) -> dict[str, bytes]:
    """Every file under `directory` (its bytes), by relative path."""
    return {
        path.relative_to(directory).as_posix(): path.read_bytes()
        for path in sorted(directory.rglob("*"))
        if path.is_file()
    }
