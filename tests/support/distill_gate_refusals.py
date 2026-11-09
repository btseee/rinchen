"""The refusals of the D10 STOP gate: `rinchen distill` commands that must exit 2 or 3
before Eole or the converter is started, with the real binary, each on a damaged copy of
what the pipeline made (a flipped byte, a missing manifest, a rewritten vocabulary).

`run_refusals` builds the damaged inputs under a scratch directory, runs every command
once and returns, for each, what it printed and whether the inputs and the output
directory it was given are as they were. The module that uses it states which exit code
and which words each refusal must have.
"""

from __future__ import annotations

import shutil
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

from rinchen.train.eole_cfg import dump_config, read_config
from tests.support.distill_gate import (
    Invocation,
    replace_vocabulary_entry,
    run_rinchen,
    sha256_of,
)
from tests.support.step_dirs import flip_one_byte, tree_digest

WEIGHTS: Final = "model.00.safetensors"
# The tag whose entry is taken out of the vocabulary of a teacher export. The name it is
# given instead is in no vocabulary and in no direction.
MISSING_TAG: Final = "<2eng_Latn>"
STAND_IN_ENTRY: Final = "gate_stand_in_entry"


@dataclass(frozen=True)
class RefusalInputs:
    """What the refusals start from. Nothing here is written by them: the damaged copies
    are made under `scratch`."""

    scratch: Path
    corpus: Path
    spm_model: Path
    other_spm_model: Path
    student_base: Path
    teacher_run: Path
    teacher_export: Path
    kd_directory: Path
    student_step: Path


@dataclass(frozen=True)
class Refusal:
    """One command that must be refused: what it printed, whether every input it was
    given is byte for byte as before, what it left in its output directory and whether
    the converter's log (which only a started converter writes) appeared beside it."""

    invocation: Invocation
    inputs_unchanged: bool
    out_dir_files: list[str]
    converter_logs: list[str]


def digest_of(path: Path) -> str:
    """One digest of a file or of everything under a directory."""
    return tree_digest(path) if path.is_dir() else sha256_of(path)


def _refused(
    args: Sequence[str | Path], inputs: Sequence[Path], out_dir: Path
) -> Refusal:
    before = [digest_of(path) for path in inputs]
    invocation = run_rinchen(*args)
    after = [digest_of(path) for path in inputs]
    files = (
        sorted(str(p.relative_to(out_dir)) for p in out_dir.rglob("*"))
        if out_dir.is_dir()
        else []
    )
    return Refusal(
        invocation=invocation,
        inputs_unchanged=before == after,
        out_dir_files=files,
        converter_logs=sorted(p.name for p in out_dir.parent.glob("*.export.log")),
    )


def _copy(source: Path, destination: Path) -> Path:
    shutil.copytree(source, destination, symlinks=True)
    return destination


def _export_args(checkpoint: Path, spm: Path, out_dir: Path) -> list[str | Path]:
    return [
        "distill",
        "export",
        "--checkpoint",
        checkpoint,
        "--spm-model",
        spm,
        "--tag-placement",
        "src",
        "--out-dir",
        out_dir,
    ]


def _generate_args(
    source: RefusalInputs, teacher: Path, out_dir: Path, *more: str
) -> list[str | Path]:
    return [
        "distill",
        "generate",
        "--corpus",
        source.corpus,
        "--teacher",
        teacher,
        "--out-dir",
        out_dir,
        *more,
    ]


def _prepare_args(
    source: RefusalInputs, kd_file: Path, base: Path, out_dir: Path, mix: str = "1:1"
) -> list[str | Path]:
    return [
        "distill",
        "prepare",
        "--corpus",
        source.corpus,
        "--kd",
        kd_file,
        "--spm-model",
        source.spm_model,
        "--config",
        base,
        "--mix",
        mix,
        "--bt-ratio",
        "0.5",
        "--temperature",
        "0.5",
        "--tag-placement",
        "src",
        "--dev-max-lines",
        "100",
        "--out-dir",
        out_dir,
    ]


def _base_with(
    source: RefusalInputs, name: str, edit: Callable[[dict[str, Any]], None]
) -> Path:
    cfg = read_config(source.student_base, "student base config")
    edit(cfg)
    path = source.scratch / "bases" / f"{name}.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    dump_config(cfg, path)
    return path


def _kd_copy(source: RefusalInputs, name: str, *, with_manifest: bool) -> Path:
    directory = source.scratch / "kd_copies" / name
    directory.mkdir(parents=True)
    shutil.copy2(source.kd_directory / "kd.parquet", directory / "kd.parquet")
    if with_manifest:
        shutil.copy2(source.kd_directory / "manifest.json", directory / "manifest.json")
    return directory


def run_refusals(source: RefusalInputs) -> dict[str, Refusal]:
    """Run every refusal of the gate once; the keys are the names the tests use."""
    out = source.scratch / "out"
    results: dict[str, Refusal] = {}

    # distill export ---------------------------------------------------------------
    alone = _copy(
        source.teacher_run / "ckpt" / "step_2000",
        source.scratch / "export_sources" / "alone" / "step_2000",
    )
    results["export_a_step_without_its_run"] = _refused(
        _export_args(alone, source.spm_model, out / "e1"), [alone], out / "e1"
    )
    run_copy = _copy(source.teacher_run, source.scratch / "export_sources" / "flipped")
    flipped = run_copy / "ckpt" / "step_2000"
    flip_one_byte(flipped / WEIGHTS)
    results["export_a_flipped_weights_byte"] = _refused(
        _export_args(flipped, source.spm_model, out / "e2"), [run_copy], out / "e2"
    )
    non_final = source.teacher_run / "ckpt" / "step_1500"
    results["export_a_non_final_step"] = _refused(
        _export_args(non_final, source.spm_model, out / "e3"),
        [source.teacher_run],
        out / "e3",
    )
    averaged = source.teacher_run / "averaged"
    results["export_another_sentencepiece_model"] = _refused(
        _export_args(averaged, source.other_spm_model, out / "e4"),
        [averaged, source.other_spm_model],
        out / "e4",
    )
    existing = out / "e5"
    existing.mkdir(parents=True)
    (existing / "kept.txt").write_text("kept\n", encoding="utf-8")
    results["export_into_an_existing_directory"] = _refused(
        _export_args(averaged, source.spm_model, existing),
        [averaged, existing],
        existing,
    )

    # distill generate -------------------------------------------------------------
    copies = source.scratch / "export_copies"
    damaged = _copy(source.teacher_export, copies / "model_bin_flipped")
    flip_one_byte(damaged / "model.bin")
    results["generate_a_model_with_a_flipped_byte"] = _refused(
        _generate_args(source, damaged, out / "g1"), [damaged], out / "g1"
    )
    unvouched = _copy(source.teacher_export, copies / "no_manifest")
    (unvouched / "manifest.json").unlink()
    results["generate_an_export_without_its_manifest"] = _refused(
        _generate_args(source, unvouched, out / "g2"), [unvouched], out / "g2"
    )
    untagged = _copy(source.teacher_export, copies / "tag_missing")
    replace_vocabulary_entry(untagged, MISSING_TAG, STAND_IN_ENTRY)
    results["generate_a_vocabulary_without_a_tag"] = _refused(
        _generate_args(source, untagged, out / "g3"), [untagged], out / "g3"
    )
    results["generate_n_above_the_beam"] = _refused(
        _generate_args(
            source, source.teacher_export, out / "g4", "--n", "5", "--beam-size", "4"
        ),
        [source.teacher_export],
        out / "g4",
    )

    # distill prepare --------------------------------------------------------------
    flipped_kd = _kd_copy(source, "flipped", with_manifest=True)
    flip_one_byte(flipped_kd / "kd.parquet")
    results["prepare_a_kd_file_with_a_flipped_byte"] = _refused(
        _prepare_args(
            source, flipped_kd / "kd.parquet", source.student_base, out / "p1"
        ),
        [flipped_kd],
        out / "p1",
    )
    no_manifest = _kd_copy(source, "no_manifest", with_manifest=False)
    results["prepare_a_kd_file_without_its_manifest"] = _refused(
        _prepare_args(
            source, no_manifest / "kd.parquet", source.student_base, out / "p2"
        ),
        [no_manifest],
        out / "p2",
    )
    kd_file = source.kd_directory / "kd.parquet"
    results["prepare_a_mix_of_zero"] = _refused(
        _prepare_args(source, kd_file, source.student_base, out / "p3", mix="0:1"),
        [source.kd_directory],
        out / "p3",
    )

    def add_train_from(cfg: dict[str, Any]) -> None:
        cfg["training"]["train_from"] = str(source.teacher_run / "averaged")

    with_train_from = _base_with(source, "train_from", add_train_from)
    results["prepare_a_base_with_train_from"] = _refused(
        _prepare_args(source, kd_file, with_train_from, out / "p4"),
        [source.kd_directory, with_train_from],
        out / "p4",
    )

    def state_a_small_vocabulary(cfg: dict[str, Any]) -> None:
        cfg["src_vocab_size"] = 100

    small_vocabulary = _base_with(source, "small_vocabulary", state_a_small_vocabulary)
    results["prepare_a_stated_vocabulary_size_below_the_pieces"] = _refused(
        _prepare_args(source, kd_file, small_vocabulary, out / "p5"),
        [source.kd_directory, small_vocabulary],
        out / "p5",
    )

    # distill evaluate -------------------------------------------------------------
    bracketed = _copy(
        source.student_step, source.scratch / "bracketed[1]" / "step_2000"
    )
    results["evaluate_a_checkpoint_path_with_a_bracket"] = _refused(
        [
            "distill",
            "evaluate",
            "--checkpoint",
            bracketed,
            "--spm-model",
            source.spm_model,
            "--corpus",
            source.corpus,
            "--tag-placement",
            "src",
            "--out-dir",
            out / "v1",
        ],
        [bracketed],
        out / "v1",
    )
    return results
