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

`PrepareWorkspace` (`rinchen distill prepare`) is the toy corpus and the toy KD file of
`tests.support.planted_student` with a base config of the tiny student; `kd_with_exclusions`
is a KD file with one row of every reason a KD row is left out. `EvaluateWorkspace`
(`rinchen distill evaluate`) is the same corpus with a fabricated checkpoint directory;
`OraclePredict` stands in for `eole_run.predict_tokens`, which the command calls through its
module-level name (`install_predictor` replaces it): it answers each dev line with the
reference of that line, so the scores of a run are worked out by hand.
"""

from __future__ import annotations

import json
import shutil
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import polars as pl
import pytest
import sentencepiece
import yaml
from click.testing import Result
from typer.testing import CliRunner

from rinchen.cli import distill_evaluate, distill_generate
from rinchen.cli.main import app
from rinchen.config.synth import SelectionConfig
from rinchen.corpus.schema import corpus_frame
from rinchen.filter.qe import QEScorer
from rinchen.manifest import store
from rinchen.manifest.model import Manifest
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
    write_kd_files,
)
from tests.support.planted_student import BASE, Inputs
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


def write_foreign_manifest(directory: Path, stage: str = "train_docify") -> bytes:
    """Write the manifest of another stage into `directory` (made when missing) and return its
    bytes: what a command must refuse to replace."""
    directory.mkdir(parents=True, exist_ok=True)
    store.write(
        directory,
        Manifest(
            stage=stage,
            created="2026-01-01T00:00:00+00:00",
            code_version="test",
            config_sha256="0" * 64,
            seed=1,
        ),
    )
    return store.manifest_path(directory).read_bytes()


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


# --- rinchen distill prepare ---------------------------------------------------------------------

# The options that give the toy the weights of `tests.support.planted_student` (22, 6, 22, 6,
# 15, 7, 15, 7 for the originals).
TOY_OPTIONS = ("--bt-ratio", "0.5", "--temperature", "0.5", "--dev-max-lines", "100")
# The sources of the six rows `kd_with_exclusions` adds, each followed by 1 to 6.
KD_EXTRA_SOURCE = "A source that only the planted exclusions have"


@dataclass(frozen=True, slots=True)
class PrepareWorkspace:
    """The files of one `rinchen distill prepare` test. `corpus` and `spm_model` are the
    toy's, shared and never written; `kd` (with its manifest beside it) and `base` are
    copies of this test's own; `out_dir` does not exist."""

    root: Path
    corpus: Path
    kd: Path
    spm_model: Path
    base: Path
    out_dir: Path

    @property
    def manifest(self) -> Path:
        return self.out_dir / "manifest.json"

    @property
    def config(self) -> Path:
        return self.out_dir / "config.yaml"

    @property
    def inputs(self) -> tuple[Path, ...]:
        """The files a run hashes, in the order of its manifest."""
        return (
            self.base,
            self.corpus,
            self.kd,
            self.kd.parent / "manifest.json",
            self.spm_model,
        )

    def args(
        self,
        *extra: str,
        corpora: Sequence[Path] | None = None,
        kd_files: Sequence[Path] | None = None,
        spm_model: Path | None = None,
        config: Path | None = None,
        out_dir: Path | None = None,
    ) -> list[str]:
        """The command line of `rinchen distill prepare` with every required option, each
        with the workspace's value unless given, then `extra`."""
        chosen = (self.corpus,) if corpora is None else tuple(corpora)
        kd = (self.kd,) if kd_files is None else tuple(kd_files)
        return [
            "distill",
            "prepare",
            *(part for path in chosen for part in ("--corpus", str(path))),
            *(part for path in kd for part in ("--kd", str(path))),
            "--spm-model",
            str(self.spm_model if spm_model is None else spm_model),
            "--config",
            str(self.base if config is None else config),
            "--out-dir",
            str(self.out_dir if out_dir is None else out_dir),
            *extra,
        ]

    def written(self) -> dict[str, bytes]:
        return written(self.out_dir)


def make_prepare_workspace(
    root: Path, inputs: Inputs, *, kd: Path | None = None
) -> PrepareWorkspace:
    """The toy corpus and SentencePiece model of `inputs` (not copied), a copy of the KD
    file with its manifest under `root/kd` (the toy's, or the one in `kd`), a copy of the
    tiny student's base config and a missing `root/out`."""
    root.mkdir(parents=True, exist_ok=True)
    source = inputs.kd if kd is None else kd
    shutil.copytree(source.parent, root / "kd")
    base = root / "base.yaml"
    shutil.copyfile(BASE, base)
    return PrepareWorkspace(
        root=root,
        corpus=inputs.corpus,
        kd=root / "kd" / source.name,
        spm_model=inputs.spm,
        base=base,
        out_dir=root / "out",
    )


def kd_with_exclusions(directory: Path, inputs: Inputs) -> Path:
    """The toy KD file (3,600 rows) with six more rows that must not train, one for each
    reason of `KD_EXCLUSION_REASONS`: a dev row, a filtered row, a row of another origin, a
    row with a blank target, a row whose target holds U+2028 and a row of a direction the
    original corpora do not train (English to Russian). Written with its manifest under
    `directory`; returns the file."""
    kd = pl.read_parquet(inputs.kd)
    row = {
        "src_lang": "eng_Latn",
        "tgt_lang": "khk_Cyrl",
        "origin": "ft",
        "corpus": "ft:toy",
        "split": "train",
    }
    extra = corpus_frame(
        [
            {**row, "src": f"{KD_EXTRA_SOURCE} 1", "tgt": "Нэг", "split": "dev"},
            {**row, "src": f"{KD_EXTRA_SOURCE} 2", "tgt": "Хоёр"},
            {**row, "src": f"{KD_EXTRA_SOURCE} 3", "tgt": "Гурав", "origin": "bt"},
            {**row, "src": f"{KD_EXTRA_SOURCE} 4", "tgt": "   "},
            {**row, "src": f"{KD_EXTRA_SOURCE} 5", "tgt": f"Дөрөв{chr(0x2028)}тав"},
            {**row, "src": f"{KD_EXTRA_SOURCE} 6", "tgt": "Зургаа"}
            | {"tgt_lang": "rus_Cyrl"},
        ]
    ).with_columns(
        pl.when(pl.col("src") == f"{KD_EXTRA_SOURCE} 2")
        .then(pl.lit("length"))
        .otherwise(pl.col("filter_reason"))
        .alias("filter_reason")
    )
    return write_kd_files(directory, pl.concat([kd, extra]))


# --- rinchen distill evaluate --------------------------------------------------------------------

CHECKPOINT_FILES = {
    "config.json": b'{"model": {"encoder": {"layers": 2}}}\n',
    "vocab.json": b'{"src": ["<unk>"]}\n',
    "model.00.safetensors": b"weights of a fabricated checkpoint",
}
_TAG_LANGUAGE = {
    "<2khk_Cyrl>": "khk_Cyrl",
    "<2eng_Latn>": "eng_Latn",
    "<2rus_Cyrl>": "rus_Cyrl",
}


def write_checkpoint(directory: Path) -> Path:
    """A directory with the three files `rinchen distill evaluate` hashes (and no manifest:
    a checkpoint of another model may be evaluated). Returns it."""
    directory.mkdir(parents=True, exist_ok=True)
    for name, content in CHECKPOINT_FILES.items():
        (directory / name).write_bytes(content)
    return directory


@dataclass(frozen=True, slots=True)
class EvaluateWorkspace:
    """The files of one `rinchen distill evaluate` test. `corpus` and `spm_model` are the
    toy's (shared, never written), `checkpoint` is this test's own; `out_dir` does not
    exist."""

    root: Path
    corpus: Path
    spm_model: Path
    checkpoint: Path
    out_dir: Path

    @property
    def manifest(self) -> Path:
        return self.out_dir / "manifest.json"

    @property
    def result(self) -> Path:
        return self.out_dir / "result.json"

    @property
    def checkpoint_files(self) -> tuple[Path, ...]:
        return tuple(self.checkpoint / name for name in CHECKPOINT_FILES)

    def args(
        self,
        *extra: str,
        corpora: Sequence[Path] | None = None,
        checkpoint: Path | None = None,
        spm_model: Path | None = None,
        tag_placement: str | None = "src",
        out_dir: Path | None = None,
    ) -> list[str]:
        """The command line of `rinchen distill evaluate` with every required option, each
        with the workspace's value unless given (`tag_placement=None` leaves it out), then
        `extra`."""
        chosen = (self.corpus,) if corpora is None else tuple(corpora)
        return [
            "distill",
            "evaluate",
            "--checkpoint",
            str(self.checkpoint if checkpoint is None else checkpoint),
            "--spm-model",
            str(self.spm_model if spm_model is None else spm_model),
            *(part for path in chosen for part in ("--corpus", str(path))),
            *([] if tag_placement is None else ["--tag-placement", tag_placement]),
            "--out-dir",
            str(self.out_dir if out_dir is None else out_dir),
            *extra,
        ]

    def written(self) -> dict[str, bytes]:
        return written(self.out_dir)


def make_evaluate_workspace(root: Path, inputs: Inputs) -> EvaluateWorkspace:
    """The toy corpus and SentencePiece model of `inputs`, a fabricated checkpoint under
    `root/ckpt` and a missing `root/out`."""
    return EvaluateWorkspace(
        root=root,
        corpus=inputs.corpus,
        spm_model=inputs.spm,
        checkpoint=write_checkpoint(root / "ckpt"),
        out_dir=root / "out",
    )


@dataclass(frozen=True, slots=True)
class PredictCall:
    """What the command asked the prediction for."""

    ckpt_dir: Path
    source_tokens: tuple[str, ...]
    out_dir: Path
    forced_prefix_tokens: tuple[str, ...] | None
    threads: int
    timeout: float | None
    stem: str
    stale_files: tuple[Path, ...]


class OraclePredict:
    """Stands in for `eole_run.predict_tokens`: each source line is answered with the
    reference of that dev line (tokenised with the model), so every direction scores chrF
    100.0 and BLEU 100.0, except where the test makes it differ: the target languages in
    `silent` get an empty hypothesis (only the forced tag, with placement `tgt`), the lines
    in `leaks` (numbers from 0) start with the tag `<2eng_Latn>` as text, and `fail` is
    raised instead of an answer. `during` is called with the call, before the answer.
    Every call is recorded in `calls`."""

    def __init__(
        self,
        spm_model: Path,
        dev_rows: Sequence[Mapping[str, str]],
        *,
        silent: Sequence[str] = (),
        leaks: Sequence[int] = (),
        fail: BaseException | None = None,
        during: Callable[[PredictCall], None] | None = None,
    ) -> None:
        self._processor = sentencepiece.SentencePieceProcessor(
            model_file=str(spm_model)
        )
        self._references = {
            (row["tgt_lang"], row["src"]): row["tgt"] for row in dev_rows
        }
        self._silent = set(silent)
        self._leaks = set(leaks)
        self._fail = fail
        self._during = during
        self.calls: list[PredictCall] = []

    def __call__(
        self,
        ckpt_dir: Path,
        source_tokens: Sequence[str],
        *,
        out_dir: Path,
        forced_prefix_tokens: Sequence[str] | None = None,
        threads: int = 2,
        timeout: float | None = None,
        stem: str = "eole",
        stale_files: Sequence[Path] = (),
    ) -> list[str]:
        call = PredictCall(
            ckpt_dir=ckpt_dir,
            source_tokens=tuple(source_tokens),
            out_dir=out_dir,
            forced_prefix_tokens=(
                None if forced_prefix_tokens is None else tuple(forced_prefix_tokens)
            ),
            threads=threads,
            timeout=timeout,
            stem=stem,
            stale_files=tuple(stale_files),
        )
        self.calls.append(call)
        if self._during is not None:
            self._during(call)
        if self._fail is not None:
            raise self._fail
        return [self._answer(call, index) for index in range(len(source_tokens))]

    def _answer(self, call: PredictCall, index: int) -> str:
        pieces = call.source_tokens[index].split(" ")
        if call.forced_prefix_tokens is None:
            tag, pieces = pieces[0], pieces[1:]
        else:
            tag = call.forced_prefix_tokens[index]
        language = _TAG_LANGUAGE[tag]
        text = self._processor.decode(pieces)
        hypothesis = (
            "" if language in self._silent else self._references[(language, text)]
        )
        if index in self._leaks:
            hypothesis = f"{_LEAKED_TAG} {hypothesis}"
        answer = self._processor.encode(hypothesis, out_type=str) if hypothesis else []
        if call.forced_prefix_tokens is not None:
            answer = [tag, *answer]
        return " ".join(answer)


_LEAKED_TAG = "<2eng_Latn>"


def install_predictor(
    monkeypatch: pytest.MonkeyPatch, predictor: OraclePredict | Callable[..., Any]
) -> None:
    """Make the prediction `rinchen distill evaluate` runs `predictor`."""
    monkeypatch.setattr(distill_evaluate, "predict_tokens", predictor)
