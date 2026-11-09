"""The stages of the D10 STOP gate (`tests/integration/test_distill_tiny_student.py`): each
function runs one step of the distillation through the real `rinchen` binary (the real Eole
and CTranslate2, on the CPU) and returns a snapshot of what the gate's tests look at.

The module that uses them hides every device (`CUDA_VISIBLE_DEVICES=""`) before it calls
any of them, and keeps the order: teacher, export, generation, preparation, student,
evaluation, student export, smoke check; the second student and the refusals take what the
first stages made.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import sentencepiece
import yaml

from rinchen.config.languages import direction_tag
from rinchen.config.loader import REPO_ROOT
from rinchen.corpus.readers import scan_corpus
from rinchen.manifest import store
from rinchen.manifest.model import Manifest
from rinchen.train.ablation_layout import variant_paths
from rinchen.train.checkpoints import step_directories
from rinchen.train.curves import TrainCurve, parse_train_log
from rinchen.train.eole_cfg import dump_config, read_config
from rinchen.train.eole_run import predict_tokens
from tests.support.distill_gate import (
    ENGINE_TIMEOUT_SECONDS,
    GENERATION_LIMIT,
    NO_RATIO_CAP,
    SUBPROCESS_TIMEOUT_SECONDS,
    THREADS,
    Invocation,
    ct2_greedy,
    read_json,
    run_rinchen,
    sha256_of,
    succeeded,
)
from tests.support.distill_gate_refusals import digest_of
from tests.support.tiny_parallel import ToyData, build_toy
from tests.support.tiny_variant import ToyInputs, build_toy_inputs

TEACHER_BASE: Final = REPO_ROOT / "configs" / "eole" / "teacher_tiny.yaml"
STUDENT_BASE: Final = REPO_ROOT / "configs" / "eole" / "student_tiny.yaml"
VARIANT_ID: Final = "t0.5_bt0.5_src_s1234"
DEV_LINES: Final = 100
STEPS: Final = 2000
SAVE_EVERY: Final = 500
SECOND_STUDENT_STEPS: Final = 200
SMOKE_SENTENCES_PER_DIRECTION: Final = 50
WEIGHTS: Final = "model.00.safetensors"
DIRECTIONS: Final = (
    "eng_Latn-khk_Cyrl",
    "khk_Cyrl-eng_Latn",
    "khk_Cyrl-rus_Cyrl",
    "rus_Cyrl-khk_Cyrl",
)
EXPORT_FILES: Final = (
    "model.bin",
    "shared_vocabulary.json",
    "config.json",
    "spm.model",
    "rinchen.json",
)
SMOKE_SENTENCE: Final = "This is the star."


# ---- shared helpers --------------------------------------------------------------------


@dataclass(frozen=True)
class Workspace:
    root: Path
    inputs: ToyInputs
    toy: ToyData
    processor: sentencepiece.SentencePieceProcessor


def make_workspace(root: Path) -> Workspace:
    """The toy corpus file (train, back-translated and dev rows; no test row), the toy
    SentencePiece model and the toy data, whose test split only the test process reads."""
    inputs = build_toy_inputs(root / "in")
    return Workspace(
        root=root,
        inputs=inputs,
        toy=build_toy(),
        processor=sentencepiece.SentencePieceProcessor(
            model_file=str(inputs.spm_model)
        ),
    )


def manifest_of(directory: Path) -> Manifest:
    manifest = store.read(directory)
    assert manifest is not None, directory
    return manifest


def listed_digest(manifest: Manifest, name: str) -> str:
    """The SHA-256 that `manifest` lists for the one output called `name`."""
    digests = [item.sha256 for item in manifest.outputs if Path(item.path).name == name]
    assert len(digests) == 1, (name, digests)
    return digests[0]


def launch(config: Path, run_dir: Path) -> Invocation:
    return run_rinchen(
        "train",
        "launch",
        "--config",
        config,
        "--run-dir",
        run_dir,
        "--threads",
        THREADS,
        "--timeout",
        str(ENGINE_TIMEOUT_SECONDS),
    )


def attempt_log(run_dir: Path) -> str:
    logs = sorted((run_dir / "launch").glob("attempt-*.log"))
    assert len(logs) == 1, logs
    return logs[0].read_text(encoding="utf-8", errors="replace")


def complete_steps(model_path: Path) -> list[int]:
    return [step.step for step in step_directories(model_path) if step.problem is None]


def prepare(
    workspace: Workspace,
    spm_model: Path,
    out_dir: Path,
    kd_file: Path,
    mix: str,
    *more: str,
) -> Invocation:
    return run_rinchen(
        "distill",
        "prepare",
        "--corpus",
        workspace.inputs.corpus,
        "--kd",
        kd_file,
        "--spm-model",
        spm_model,
        "--config",
        STUDENT_BASE,
        "--mix",
        mix,
        "--bt-ratio",
        "0.5",
        "--temperature",
        "0.5",
        "--tag-placement",
        "src",
        "--dev-max-lines",
        str(DEV_LINES),
        "--out-dir",
        out_dir,
        *more,
    )


# ---- T: the tiny teacher ---------------------------------------------------------------


@dataclass(frozen=True)
class Teacher:
    run_dir: Path
    averaged: Path
    prepare_seconds: float
    launch: Invocation
    average: Invocation
    log: str
    curve: TrainCurve
    saved_steps: list[int]


def train_teacher(workspace: Workspace) -> Teacher:
    """`teacher_tiny.yaml` with 2,000 steps (a save every 500, all four kept), placement
    `src`, through `train ablate --prepare-only`, `train launch` and `train average`."""
    base = read_config(TEACHER_BASE, "teacher base config")
    base["training"].update(
        train_steps=STEPS,
        valid_steps=SAVE_EVERY,
        save_checkpoint_steps=SAVE_EVERY,
        keep_checkpoint=4,
    )
    base_copy = workspace.root / "teacher_base.yaml"
    dump_config(base, base_copy)
    matrix = workspace.root / "matrix.yaml"
    matrix.write_text(
        yaml.safe_dump(
            {
                "name": "distill_gate_teacher",
                "base": str(base_copy),
                "axes": {
                    "tag_placement": ["src"],
                    "temperature": [0.5],
                    "bt_ratio": [0.5],
                },
                "seeds": [1234],
                "dev_max_lines": DEV_LINES,
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    out_dir = workspace.root / "teacher_out"
    prepared = succeeded(
        run_rinchen(
            "train",
            "ablate",
            "--matrix",
            matrix,
            "--corpus",
            workspace.inputs.corpus,
            "--spm-model",
            workspace.inputs.spm_model,
            "--out-dir",
            out_dir,
            "--prepare-only",
            "--threads",
            THREADS,
        )
    )
    paths = variant_paths(out_dir, VARIANT_ID)
    trained = succeeded(launch(paths.config, paths.run_dir))
    average = succeeded(
        run_rinchen(
            "train",
            "average",
            "--checkpoint-dir",
            paths.run_dir / "ckpt",
            "--last",
            "2",
        )
    )
    log = attempt_log(paths.run_dir)
    return Teacher(
        run_dir=paths.run_dir,
        averaged=paths.run_dir / "averaged",
        prepare_seconds=prepared.seconds,
        launch=trained,
        average=average,
        log=log,
        curve=parse_train_log(log),
        saved_steps=complete_steps(paths.run_dir / "ckpt"),
    )


# ---- X, Q: an export -------------------------------------------------------------------


@dataclass(frozen=True)
class Exported:
    invocation: Invocation
    directory: Path
    record: dict[str, Any]
    manifest: Manifest
    converter_log: Path


def export_checkpoint(
    workspace: Workspace, checkpoint: Path, out_dir: Path
) -> Exported:
    """`rinchen distill export` of `checkpoint` with the toy model and placement `src`."""
    invocation = succeeded(
        run_rinchen(
            "distill",
            "export",
            "--checkpoint",
            checkpoint,
            "--spm-model",
            workspace.inputs.spm_model,
            "--tag-placement",
            "src",
            "--out-dir",
            out_dir,
        )
    )
    return Exported(
        invocation=invocation,
        directory=out_dir,
        record=read_json(out_dir / "rinchen.json"),
        manifest=manifest_of(out_dir),
        converter_log=out_dir.parent / f"{out_dir.name}.export.log",
    )


def assert_model_record_facts(exported: Exported, directions: tuple[str, ...]) -> None:
    """What D11 added to the record of an export of the toy model (and to its manifest): the
    schema, the name made from `model.bin`, the `directions` the export derived from the
    saved config of the checkpoint, a float32 sentence model, and how it was settled."""
    record = exported.record
    assert record["schema"] == "rinchen.model/1"
    assert (
        record["name"] == f"rinchen-{sha256_of(exported.directory / 'model.bin')[:12]}"
    )
    assert record["directions"] == list(directions)
    assert record["quantization"] == "float32"
    assert record["document_level"] is False
    assert exported.manifest.status["quantization"] == "float32"
    assert exported.manifest.status["directions_check"] == "derived"
    assert exported.manifest.status["document_level_check"] == "derived"


# ---- G: the KD corpus ------------------------------------------------------------------


@dataclass(frozen=True)
class Generated:
    invocation: Invocation
    directory: Path
    frame: Any
    counts: dict[str, Any]
    manifest: Manifest
    rerun: Invocation
    rerun_frame: Any
    rerun_counts: dict[str, Any]


def _kd_table(directory: Path) -> Any:
    return scan_corpus(directory / "kd.parquet").collect()


def generate_kd(workspace: Workspace, teacher_export: Exported) -> Generated:
    """`rinchen distill generate` over every authentic training source (beam 4, n 1), then
    the same command again into a copy of the output directory."""
    directory = workspace.root / "kd"
    arguments = [
        "distill",
        "generate",
        "--corpus",
        workspace.inputs.corpus,
        "--teacher",
        teacher_export.directory,
        "--beam-size",
        "4",
        "--n",
        "1",
        "--seed",
        "1234",
    ]
    invocation = succeeded(run_rinchen(*arguments, "--out-dir", directory))
    frame = _kd_table(directory)
    counts = read_json(directory / "counts.json")
    manifest = manifest_of(directory)
    # The rerun works in a copy: the shards are found by their fingerprint, not by the
    # directory, and the original is what the next stages read.
    rerun_directory = workspace.root / "kd_rerun"
    shutil.copytree(directory, rerun_directory)
    rerun = succeeded(run_rinchen(*arguments, "--out-dir", rerun_directory))
    return Generated(
        invocation=invocation,
        directory=directory,
        frame=frame,
        counts=counts,
        manifest=manifest,
        rerun=rerun,
        rerun_frame=_kd_table(rerun_directory),
        rerun_counts=read_json(rerun_directory / "counts.json"),
    )


# ---- P: the student's data and config --------------------------------------------------


@dataclass(frozen=True)
class Prepared:
    invocation: Invocation
    out_dir: Path
    config: dict[str, Any]
    manifest: Manifest
    three_to_one: Invocation
    three_to_one_out_dir: Path
    three_to_one_over_the_run: Invocation
    digest_before_dry_run: str
    digest_after_dry_run: str


def prepare_student_run(workspace: Workspace, kd: Generated) -> Prepared:
    """`rinchen distill prepare --mix 1:1`, and two dry runs of `--mix 3:1`: one into a
    directory that does not exist and one over the prepared student."""
    kd_file = kd.directory / "kd.parquet"
    spm_model = workspace.inputs.spm_model
    dry_dir = workspace.root / "student_dry"
    three_to_one = succeeded(
        prepare(workspace, spm_model, dry_dir, kd_file, "3:1", "--dry-run")
    )
    out_dir = workspace.root / "student"
    invocation = succeeded(prepare(workspace, spm_model, out_dir, kd_file, "1:1"))
    before = digest_of(out_dir)
    over_the_run = succeeded(
        prepare(workspace, spm_model, out_dir, kd_file, "3:1", "--dry-run")
    )
    return Prepared(
        invocation=invocation,
        out_dir=out_dir,
        config=read_config(out_dir / "config.yaml", "student config"),
        manifest=manifest_of(out_dir),
        three_to_one=three_to_one,
        three_to_one_out_dir=dry_dir,
        three_to_one_over_the_run=over_the_run,
        digest_before_dry_run=before,
        digest_after_dry_run=digest_of(out_dir),
    )


# ---- S: the student's training ---------------------------------------------------------


@dataclass(frozen=True)
class Student:
    run_dir: Path
    final_step: Path
    launch: Invocation
    log: str
    curve: TrainCurve
    saved_steps: list[int]
    saved_config: dict[str, Any]
    effective_config: dict[str, Any]
    launch_record: dict[str, Any]
    launch_manifest: Manifest


def train_student(prepared: Prepared) -> Student:
    """`rinchen train launch` of the prepared student: 2,000 steps from random init."""
    run_dir = prepared.out_dir / "run"
    trained = succeeded(launch(prepared.out_dir / "config.yaml", run_dir))
    final_step = run_dir / "ckpt" / f"step_{STEPS}"
    log = attempt_log(run_dir)
    return Student(
        run_dir=run_dir,
        final_step=final_step,
        launch=trained,
        log=log,
        curve=parse_train_log(log),
        saved_steps=complete_steps(run_dir / "ckpt"),
        saved_config=read_json(final_step / "config.json"),
        effective_config=read_config(run_dir / "launch" / "train-1.yaml", "config"),
        launch_record=read_json(run_dir / "launch.json"),
        launch_manifest=manifest_of(run_dir),
    )


# ---- E: the evaluation -----------------------------------------------------------------


@dataclass(frozen=True)
class Evaluated:
    invocation: Invocation
    out_dir: Path
    result: dict[str, Any]
    manifest: Manifest


def evaluate_checkpoint(workspace: Workspace, checkpoint: Path, name: str) -> Evaluated:
    """`rinchen distill evaluate` of `checkpoint` on the first 100 dev lines of each
    direction, into `<root>/<name>`."""
    out_dir = workspace.root / name
    invocation = succeeded(
        run_rinchen(
            "distill",
            "evaluate",
            "--checkpoint",
            checkpoint,
            "--spm-model",
            workspace.inputs.spm_model,
            "--corpus",
            workspace.inputs.corpus,
            "--tag-placement",
            "src",
            "--out-dir",
            out_dir,
            "--max-lines",
            str(DEV_LINES),
        )
    )
    return Evaluated(
        invocation=invocation,
        out_dir=out_dir,
        result=read_json(out_dir / "result.json"),
        manifest=manifest_of(out_dir),
    )


@dataclass(frozen=True)
class Evaluations:
    """The averaged teacher, the student's final step and the `ckpt` directory of the
    student's run (Eole links the newest step's files into it)."""

    teacher: Evaluated
    student: Evaluated
    student_run_directory: Evaluated


# ---- Q: CTranslate2 against Eole -------------------------------------------------------


@dataclass(frozen=True)
class Smoke:
    sentences: int
    equal_pieces: int
    equal_text: int
    star_text: str
    first_mismatches: list[tuple[str, str]]


def smoke_check(
    workspace: Workspace, student: Student, student_export: Exported
) -> Smoke:
    """200 dev sentences (50 per direction) greedily through CTranslate2 with nothing of
    D10 but the exported files, and through Eole's own prediction of the checkpoint."""
    processor = workspace.processor
    sources: list[list[str]] = []
    for direction in DIRECTIONS:
        dev = workspace.inputs.dev[direction]
        tag = direction_tag(dev.tgt_lang)
        sources += [
            [tag, *processor.encode(text, out_type=str)]
            for text in dev.src[:SMOKE_SENTENCES_PER_DIRECTION]
        ]
    sentence = [
        direction_tag("khk_Cyrl"),
        *processor.encode(SMOKE_SENTENCE, out_type=str),
    ]
    translated = ct2_greedy(student_export.directory, [*sources, sentence])
    eole_lines = predict_tokens(
        student.final_step,
        [" ".join(source) for source in sources],
        out_dir=workspace.root / "smoke_eole",
        threads=int(THREADS),
        timeout=SUBPROCESS_TIMEOUT_SECONDS,
        stem="smoke",
        max_length=GENERATION_LIMIT,
        max_length_ratio=NO_RATIO_CAP,
    )
    eole = [line.split(" ") if line else [] for line in eole_lines]
    pairs = list(zip(translated[: len(sources)], eole, strict=True))
    return Smoke(
        sentences=len(sources),
        equal_pieces=sum(left == right for left, right in pairs),
        equal_text=sum(
            processor.decode(left) == processor.decode(right) for left, right in pairs
        ),
        star_text=processor.decode(translated[-1]),
        first_mismatches=[
            (processor.decode(left), processor.decode(right))
            for left, right in pairs
            if left != right
        ][:3],
    )


# ---- V: a second student with its own SentencePiece model ------------------------------


@dataclass(frozen=True)
class SecondStudent:
    spm_model: Path
    prepare: Invocation
    out_dir: Path
    config: dict[str, Any]
    run_dir: Path
    launch: Invocation
    log: str
    curve: TrainCurve


def train_second_student(
    workspace: Workspace, kd: Generated, spm_model: Path
) -> SecondStudent:
    """`distill prepare` with `spm_model` and a 200-step launch of a scratch copy of the
    rendered config (a short schedule and a run directory of its own)."""
    out_dir = workspace.root / "student500"
    prepared = succeeded(
        prepare(workspace, spm_model, out_dir, kd.directory / "kd.parquet", "1:1")
    )
    rendered = read_config(out_dir / "config.yaml", "student config")
    run_dir = workspace.root / "student500_short" / "run"
    short = dict(rendered)
    short["src_vocab"] = str(run_dir / "vocab.src")
    short["training"] = {
        **rendered["training"],
        "model_path": str(run_dir / "ckpt"),
        "train_steps": SECOND_STUDENT_STEPS,
        "valid_steps": SECOND_STUDENT_STEPS,
        "save_checkpoint_steps": SECOND_STUDENT_STEPS,
    }
    config = workspace.root / "student500_short" / "config.yaml"
    config.parent.mkdir(parents=True)
    dump_config(short, config)
    trained = succeeded(launch(config, run_dir))
    log = attempt_log(run_dir)
    return SecondStudent(
        spm_model=spm_model,
        prepare=prepared,
        out_dir=out_dir,
        config=rendered,
        run_dir=run_dir,
        launch=trained,
        log=log,
        curve=parse_train_log(log),
    )
