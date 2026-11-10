"""Shared setup of the tests of `rinchen export parity` (D11): a workspace that holds the inputs
of one run, the command line it is run with, and a stand-in for `strict_parity` that the
command calls through its module-level name. No model, process or GPU is involved.

The workspace is a corpus of dev, train and test rows of the four directions of the toy
(`DEV_ROWS` dev rows each, whose texts say `dev`), a fabricated export of those four
directions whose record names the weights of a fabricated checkpoint (`write_checkpoint`), and
an output directory that does not exist.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import pytest
from click.testing import Result
from typer.testing import CliRunner

from rinchen.cli import export_parity
from rinchen.cli.main import app
from rinchen.export.record import CheckpointRecord
from rinchen.export.verified import VerifiedExport
from rinchen.train.export_parity import ParityReport, ParitySource
from tests.support.corpus_rows import corpus_row, write_rows
from tests.support.fake_parity_eole import (
    CONFIG_SHA256,
    VOCAB_SHA256,
    WEIGHTS_SHA256,
    write_checkpoint,
)
from tests.support.model_exports import (
    CHECKPOINT_MANIFEST_SHA256,
    TOY_DIRECTIONS,
    WrittenModel,
    write_model_export,
)

runner = CliRunner()

DEV_ROWS = 3
# `rinchen-` and the first 12 hex characters of the SHA-256 of the fabricated `model.bin`.
EXPORT_NAME = "rinchen-07148803f29a"
MODEL_SHA256 = "07148803f29a4819c336f2449a35416f5fe125662a8008d5d060e975fe2f6756"


def invoke(args: Sequence[str]) -> Result:
    """Run `rinchen` with `args` in this process."""
    return runner.invoke(app, list(args))


def perfect_report(sources: Sequence[ParitySource], **changes: object) -> ParityReport:
    """A report that says every source agreed, with the float32 numbers of the spike; any
    field in `changes` replaces its value."""
    fields: dict[str, object] = {
        "sentences": len(sources),
        "exact": len(sources),
        "exact_share": 1.0,
        "mean_abs_delta": 8.3e-07,
        "max_abs_delta": 3.5e-06,
        "quantization": "float32",
        "placement": "src",
        "gated": True,
        "mismatches": (),
    }
    return ParityReport(**(fields | changes))  # type: ignore[arg-type]


@dataclass(frozen=True)
class ParityCall:
    """What the command asked `strict_parity` for."""

    checkpoint: Path
    export: VerifiedExport
    sources: tuple[ParitySource, ...]
    out_dir: Path
    threads: int
    timeout: float | None
    weights_sha256: str | None


@dataclass
class FakeStrictParity:
    """Stands in for `strict_parity` in the command: records its calls and returns `report`
    (a function of the sources, by default `perfect_report`), or raises `error`. `during` is
    called with the call first, as something else that happens while the run is in
    progress."""

    report: Callable[[Sequence[ParitySource]], ParityReport] = perfect_report
    error: BaseException | None = None
    during: Callable[[ParityCall], None] | None = None
    calls: list[ParityCall] = field(default_factory=list)

    def __call__(
        self,
        checkpoint: Path,
        export: VerifiedExport,
        sources: Sequence[ParitySource],
        *,
        out_dir: Path,
        threads: int = 2,
        timeout: float | None = None,
        engine: object = None,
        processor: object = None,
        weights_sha256: str | None = None,
    ) -> ParityReport:
        call = ParityCall(
            checkpoint=checkpoint,
            export=export,
            sources=tuple(sources),
            out_dir=out_dir,
            threads=threads,
            timeout=timeout,
            weights_sha256=weights_sha256,
        )
        self.calls.append(call)
        if self.during is not None:
            self.during(call)
        if self.error is not None:
            raise self.error
        return self.report(sources)


def install_parity(
    monkeypatch: pytest.MonkeyPatch, **options: object
) -> FakeStrictParity:
    """Make the run `rinchen export parity` makes a `FakeStrictParity(**options)`."""
    fake = FakeStrictParity(**options)  # type: ignore[arg-type]
    monkeypatch.setattr(export_parity, "strict_parity", fake)
    return fake


def corpus_rows() -> list[dict[str, str]]:
    """The rows of the workspace corpus: dev, train and test rows of the four directions."""
    rows = []
    for direction in TOY_DIRECTIONS:
        src_lang, tgt_lang = direction.split("-")
        for split in ("dev", "train", "test"):
            count = DEV_ROWS if split == "dev" else 2
            for number in range(count):
                rows.append(
                    corpus_row(
                        f"{src_lang} {split} {number}",
                        f"{tgt_lang} {split} {number}",
                        src_lang=src_lang,
                        tgt_lang=tgt_lang,
                        split=split,
                    )
                )
    return rows


@dataclass(frozen=True)
class ParityWorkspace:
    """The files of one `rinchen export parity` test; `out_dir` does not exist."""

    root: Path
    corpus: Path
    export: WrittenModel
    checkpoint: Path
    out_dir: Path

    @property
    def manifest(self) -> Path:
        return self.out_dir / "manifest.json"

    @property
    def result(self) -> Path:
        return self.out_dir / "parity.json"

    @property
    def weights(self) -> Path:
        return self.checkpoint / "model.00.safetensors"

    def args(
        self,
        *extra: str,
        corpora: Sequence[Path] | None = None,
        checkpoint: Path | None = None,
        export: Path | None = None,
        out_dir: Path | None = None,
    ) -> list[str]:
        """The command line with every required option, each with the workspace's value
        unless given, then `extra`."""
        chosen = (self.corpus,) if corpora is None else tuple(corpora)
        return [
            "export",
            "parity",
            "--checkpoint",
            str(self.checkpoint if checkpoint is None else checkpoint),
            "--export",
            str(self.export.directory if export is None else export),
            *(part for path in chosen for part in ("--corpus", str(path))),
            "--out-dir",
            str(self.out_dir if out_dir is None else out_dir),
            *extra,
        ]


def make_workspace(root: Path, *, quantization: str = "float32") -> ParityWorkspace:
    """The corpus, a fabricated export of `quantization` and its checkpoint under `root`."""
    root.mkdir(parents=True, exist_ok=True)
    export = write_model_export(
        root / "export",
        quantization=quantization,
        checkpoint=CheckpointRecord(
            kind="step",
            step=200,
            weights_sha256=WEIGHTS_SHA256,
            config_sha256=CONFIG_SHA256,
            vocab_sha256=VOCAB_SHA256,
            manifest_sha256=CHECKPOINT_MANIFEST_SHA256,
            manifest_stage="train_launch",
        ),
    )
    return ParityWorkspace(
        root=root,
        corpus=write_rows(root / "corpus", "toy", corpus_rows()),
        export=export,
        checkpoint=write_checkpoint(root / "ckpt"),
        out_dir=root / "out",
    )
