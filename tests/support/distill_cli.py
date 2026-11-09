"""Shared setup of the tests of `rinchen distill export` and `rinchen distill generate` (D10):
workspaces that hold the inputs of one command, the command line each is run with, and the
seams the commands call replaced by fakes. No model, converter, process or GPU is involved.

`ExportWorkspace` is a checkpoint a run of ours finished (or an averaged model), its
SentencePiece model and an output directory that does not exist; the converter is
`tests.support.distill_exports.fake_converter`.

`GenerateWorkspace` is the planted corpus of `tests.support.planted_kd`, a complete export
(`write_export`) as the teacher and an output directory that does not exist. The planted
sources are four English and three Russian texts, so the output of a run with
`PLANTED_SCRIPT` is worked out by hand in `tests.support.planted_kd` (4 KD rows, all of them
English to Mongolian).

The two seams of `rinchen.cli.distill_generate` are `build_teacher` and `build_qe_scorer`;
`install_teacher` and `install_scorer` replace them and record what they were asked for.
`install_ct2_teacher` installs a REAL `TeacherTranslator` over a scripted CTranslate2, for
the tests that need its tag checks and its call options.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest
import yaml
from click.testing import Result
from typer.testing import CliRunner

from rinchen.cli import distill_generate
from rinchen.cli.main import app
from rinchen.config.synth import SelectionConfig
from rinchen.filter.qe import QEScorer
from rinchen.train.distill_export import VerifiedExport
from rinchen.train.distill_teacher import TeacherOptions, TeacherTranslator
from tests.support.corpus_rows import write_rows
from tests.support.distill_exports import WrittenExport, write_export
from tests.support.distill_teachers import END, tag_processor
from tests.support.fake_ct2 import FakeProcessor, ScriptedCt2
from tests.support.planted_kd import (
    PLANTED_SCRIPT,
    ScriptedTeacher,
    planted_training_rows,
)
from tests.support.planted_synth import FakeScorer
from tests.support.step_dirs import (
    FabricatedRun,
    eole_model_section,
    make_averaged_dir,
    make_finished_run,
    write_spm_model,
)

runner = CliRunner()

# `rinchen-` and the first 12 hex characters of the SHA-256 of the fabricated `model.bin`.
DEFAULT_NAME = "rinchen-07148803f29a"


def invoke(args: Sequence[str]) -> Result:
    """Run `rinchen` with `args` in this process."""
    return runner.invoke(app, list(args))


def run_ok(args: Sequence[str]) -> Result:
    """Run `rinchen` with `args`; the run must have exited 0."""
    result = invoke(args)
    assert result.exit_code == 0, result.output
    return result


def written(directory: Path) -> dict[str, bytes]:
    """Every file under `directory` (its bytes), by relative path."""
    return {
        path.relative_to(directory).as_posix(): path.read_bytes()
        for path in sorted(directory.rglob("*"))
        if path.is_file()
    }


@dataclass(frozen=True, slots=True)
class ExportWorkspace:
    """The files of one `rinchen distill export` test. `checkpoint` and `spm_model` exist;
    `out_dir` does not."""

    root: Path
    run: FabricatedRun | None
    checkpoint: Path
    spm_model: Path
    out_dir: Path

    @property
    def manifest(self) -> Path:
        return self.out_dir / "manifest.json"

    @property
    def record(self) -> Path:
        return self.out_dir / "export.json"

    @property
    def log(self) -> Path:
        return Path(f"{self.out_dir}.export.log")

    def args(
        self,
        *extra: str,
        checkpoint: Path | None = None,
        spm_model: Path | None = None,
        tag_placement: str | None = "src",
        out_dir: Path | None = None,
    ) -> list[str]:
        """The command line of `rinchen distill export` with every required option, each
        with the workspace's value unless given (`tag_placement=None` leaves it out), then
        `extra`."""
        return [
            "distill",
            "export",
            "--checkpoint",
            str(self.checkpoint if checkpoint is None else checkpoint),
            "--spm-model",
            str(self.spm_model if spm_model is None else spm_model),
            *([] if tag_placement is None else ["--tag-placement", tag_placement]),
            "--out-dir",
            str(self.out_dir if out_dir is None else out_dir),
            *extra,
        ]

    def written(self) -> dict[str, bytes]:
        return written(self.out_dir)


def make_export_workspace(
    root: Path, *, averaged: bool = False, **options: Any
) -> ExportWorkspace:
    """A finished run (its final step is the checkpoint) or, with `averaged`, an averaged
    model, of a 2-encoder, 1-decoder model unless `model` says otherwise; `options` go to
    `make_finished_run` or `make_averaged_dir`."""
    root.mkdir(parents=True, exist_ok=True)
    model = options.pop("model", eole_model_section(encoder_layers=2, decoder_layers=1))
    run: FabricatedRun | None = None
    if averaged:
        checkpoint = make_averaged_dir(root / "averaged", model=model, **options)
    else:
        run = make_finished_run(root / "teacher", model=model, **options)
        checkpoint = run.final
    return ExportWorkspace(
        root=root,
        run=run,
        checkpoint=checkpoint,
        spm_model=write_spm_model(root / "spm"),
        out_dir=root / "export",
    )


@dataclass(frozen=True, slots=True)
class GenerateWorkspace:
    """The files of one `rinchen distill generate` test. The corpus files and the teacher
    exist; `out_dir` does not."""

    root: Path
    corpora: tuple[Path, ...]
    teacher: Path
    export: WrittenExport
    out_dir: Path

    @property
    def kd_corpus(self) -> Path:
        return self.out_dir / "kd.parquet"

    @property
    def counts(self) -> Path:
        return self.out_dir / "counts.json"

    @property
    def manifest(self) -> Path:
        return self.out_dir / "manifest.json"

    def args(
        self,
        *extra: str,
        corpora: Sequence[Path] | None = None,
        teacher: Path | None = None,
        out_dir: Path | None = None,
    ) -> list[str]:
        """The command line of `rinchen distill generate` with every required option, each
        with the workspace's value unless given, then `extra`."""
        chosen = self.corpora if corpora is None else tuple(corpora)
        return [
            "distill",
            "generate",
            *(part for path in chosen for part in ("--corpus", str(path))),
            "--teacher",
            str(self.teacher if teacher is None else teacher),
            "--out-dir",
            str(self.out_dir if out_dir is None else out_dir),
            *extra,
        ]

    def counts_document(self) -> dict[str, Any]:
        return json.loads(self.counts.read_text(encoding="utf-8"))

    def shard_directories(self) -> list[Path]:
        return sorted(self.out_dir.glob("shards-*"))

    def shard_files(self) -> list[str]:
        """The shard files under the output directory, as `<direction>/<file>`."""
        return sorted(
            path.relative_to(directory).as_posix()
            for directory in self.shard_directories()
            for path in directory.rglob("shard-*.parquet")
        )

    def written(self) -> dict[str, bytes]:
        return written(self.out_dir)


def make_generate_workspace(
    root: Path,
    *,
    rows: Sequence[Mapping[str, str]] | None = None,
    **export_options: Any,
) -> GenerateWorkspace:
    """The corpus under `root/corpus` (the planted files, or one file of `rows`), a
    fabricated export under `root/export` (`export_options` go to `write_export`) and a
    missing `root/out`."""
    if rows is None:
        corpora = planted_training_rows(root / "corpus").paths
    else:
        (root / "corpus").mkdir(parents=True, exist_ok=True)
        corpora = (write_rows(root / "corpus", "rows", rows),)
    export = write_export(root / "export", **export_options)
    return GenerateWorkspace(
        root=root,
        corpora=corpora,
        teacher=export.directory,
        export=export,
        out_dir=root / "out",
    )


def write_selection_config(path: Path, **selection: Any) -> Path:
    """A synth config file whose `selection` section is `selection` (everything else is
    the default)."""
    path.write_text(yaml.safe_dump({"selection": selection}), encoding="utf-8")
    return path


class CliTeacher(ScriptedTeacher):
    """The teacher the command builds, scripted: `ScriptedTeacher` (by default with
    `PLANTED_SCRIPT`) that also records each check of directions it was asked for, with
    the number of translate calls made before it."""

    def __init__(
        self, script: Mapping[str, Sequence[str]] | None = None, **options: Any
    ) -> None:
        super().__init__(PLANTED_SCRIPT if script is None else script, **options)
        self.checked: list[tuple[tuple[str, ...], int]] = []

    def check_directions(self, directions: Sequence[str]) -> None:
        self.checked.append((tuple(directions), len(self.calls)))


@dataclass
class TeacherSeam:
    """What `install_teacher` installed: the teacher the command gets and the
    `(export, options)` of every build."""

    teacher: Any
    built: list[tuple[VerifiedExport, TeacherOptions]] = field(default_factory=list)


def install_teacher(
    monkeypatch: pytest.MonkeyPatch, teacher: Any | None = None
) -> TeacherSeam:
    """Make `build_teacher` return `teacher` (a `CliTeacher` of the planted script by
    default) and record its arguments."""
    seam = TeacherSeam(CliTeacher() if teacher is None else teacher)

    def build(export: VerifiedExport, options: TeacherOptions) -> Any:
        seam.built.append((export, options))
        return seam.teacher

    monkeypatch.setattr(distill_generate, "build_teacher", build)
    return seam


def answer_from(
    script: Mapping[str, Sequence[str]],
) -> Callable[[tuple[str, ...]], list[tuple[list[str], float]]]:
    """A script for `ScriptedCt2` that translates the source line `[tag, *pieces]` (the
    tag in the source, placement `src`) by looking up the text of its pieces in `script`
    and answering with the pieces of each candidate and the end token."""
    processor = FakeProcessor()

    def answer(line: tuple[str, ...]) -> list[tuple[list[str], float]]:
        text = processor.decode(list(line[1:]))
        return [
            (processor.encode(candidate) + [END], -0.5) for candidate in script[text]
        ]

    return answer


def install_ct2_teacher(
    monkeypatch: pytest.MonkeyPatch,
    script: Mapping[str, Sequence[str]] = PLANTED_SCRIPT,
    *,
    ct2: ScriptedCt2 | None = None,
) -> ScriptedCt2:
    """Make `build_teacher` build a real `TeacherTranslator` over `ct2` (a `ScriptedCt2`
    that answers `script` by default) with a processor that knows the tags of English,
    Mongolian and Russian; return the scripted library, which records its calls."""
    library = ScriptedCt2(answer_from(script)) if ct2 is None else ct2

    def build(export: VerifiedExport, options: TeacherOptions) -> TeacherTranslator:
        return TeacherTranslator(
            export, options, translator=library, processor=tag_processor()
        )

    monkeypatch.setattr(distill_generate, "build_teacher", build)
    return library


class ReleasingScorer(FakeScorer):
    """A `FakeScorer` that counts how often it was released."""

    def __init__(
        self,
        scores: Mapping[tuple[str, str], float] | None = None,
        **options: Any,
    ) -> None:
        super().__init__({} if scores is None else scores, **options)
        self.released = 0

    def release(self) -> None:
        self.released += 1


@dataclass
class ScorerSeam:
    """What `install_scorer` installed: the scorer the command gets (None: none exists)
    and the selection config of every build."""

    scorer: QEScorer | None
    built: list[SelectionConfig] = field(default_factory=list)


def install_scorer(
    monkeypatch: pytest.MonkeyPatch, scorer: QEScorer | None
) -> ScorerSeam:
    """Make `build_qe_scorer` return `scorer` and record its argument."""
    seam = ScorerSeam(scorer)

    def build(cfg: SelectionConfig) -> QEScorer | None:
        seam.built.append(cfg)
        return seam.scorer

    monkeypatch.setattr(distill_generate, "build_qe_scorer", build)
    return seam
