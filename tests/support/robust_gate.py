"""Stages and helpers of the D12 robustness gate (`tests/integration/test_robustness_gate.py`):
train the robustness toy's model once, convert it, write the suites of the gate and run the real
`rinchen eval robustness` binary over them.

Everything runs on the CPU: the session fixtures of `tests/integration/conftest.py` set
`CUDA_VISIBLE_DEVICES=""` before they call any of this, Eole trains with `gpu_ranks: []`, the
command's device is `cpu` (its default; `auto` is refused) and `run_robustness` refuses to start a
command in a process that does not hide the devices. Nothing here asks a device question.

The model is the tiny teacher of `train_teacher` (2/2 layers, d64, the tag in the source) trained
from random init on `tests/support/robust_toy.py` for `ROBUST_TEACHER_STEPS` steps through
`train ablate --prepare-only`, `train launch` and `train average`, and converted by
`rinchen export convert` to float32. Training is deterministic here (the design spike's two
independent trainings gave the same `model.bin`), so the figures the gate reads are exact. A gate
figure under its floor is answered by raising `ROBUST_TEACHER_STEPS`, never by lowering the floor.
"""

from __future__ import annotations

import json
import os
import shutil
from collections import defaultdict
from collections.abc import Collection
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import sentencepiece

from rinchen.eval.robust_suite import write_suite
from tests.support.d11_gate import convert_export
from tests.support.distill_gate import Invocation, read_json, run_rinchen
from tests.support.distill_gate_stages import (
    Exported,
    Teacher,
    Workspace,
    train_teacher,
)
from tests.support.robust_suites import (
    build_main_suite,
    build_mong_output_suite,
    build_term_control_suite,
    build_unseen_placeholder_suite,
)
from tests.support.robust_toy import build_robust_toy, robust_toy_inputs
from tests.support.tiny_parallel import build_toy

# 3,000 steps (the spike's first toy) did not teach dates, two-number rows or entities; 6,000 did.
ROBUST_TEACHER_STEPS: Final = 6000
RESULT_FILE: Final = "robustness.json"
REPORT_FILE: Final = "report.json"
HYPOTHESES_FILE: Final = "hypotheses.jsonl"
MANIFEST_FILE: Final = "manifest.json"
# The groups of a hypotheses file that `rotate_hypotheses` moves: (section, arm).
MONG_ARM: Final = ("mong", "mong")
_GPU_HIDDEN: Final = "CUDA_VISIBLE_DEVICES"


# ---- the model -------------------------------------------------------------------------------


def robust_workspace(root: Path) -> Workspace:
    """The robustness toy's corpus file (train and dev rows; no test row) and SentencePiece model
    under `root`, as the stages of D10 and D11 take them. `toy` is the D7 toy that
    `Workspace` also holds: no stage used here reads it."""
    inputs = robust_toy_inputs(root / "in")
    return Workspace(
        root=root,
        inputs=inputs,
        toy=build_toy(),
        processor=sentencepiece.SentencePieceProcessor(
            model_file=str(inputs.spm_model)
        ),
    )


def train_robust_teacher(workspace: Workspace) -> Teacher:
    """The tiny teacher trained on the robustness toy for `ROBUST_TEACHER_STEPS` steps."""
    return train_teacher(workspace, placement="src", steps=ROBUST_TEACHER_STEPS)


def export_robust(workspace: Workspace, teacher: Teacher, out_dir: Path) -> Exported:
    """`rinchen export convert` of the averaged weights of `teacher` at float32."""
    return convert_export(workspace, teacher.averaged, out_dir, placement="src")


# ---- the suites ------------------------------------------------------------------------------


@dataclass(frozen=True)
class Suites:
    """The four suite files of the gate."""

    main: Path
    term_control: Path
    unseen_placeholders: Path
    mong_output: Path


def write_suites(directory: Path) -> Suites:
    """Write the main suite and the three control suites of the robustness toy into
    `directory` (made when missing)."""
    directory.mkdir(parents=True, exist_ok=True)
    toy = build_robust_toy()
    suites = Suites(
        main=directory / "main.json",
        term_control=directory / "term_control.json",
        unseen_placeholders=directory / "unseen_placeholders.json",
        mong_output=directory / "mong_output.json",
    )
    write_suite(build_main_suite(toy), suites.main)
    write_suite(build_term_control_suite(toy), suites.term_control)
    write_suite(build_unseen_placeholder_suite(), suites.unseen_placeholders)
    write_suite(build_mong_output_suite(toy), suites.mong_output)
    return suites


# ---- the command -----------------------------------------------------------------------------


def run_robustness(*args: str | Path) -> Invocation:
    """`rinchen eval robustness <args>` through the real binary, to its end. The process must
    hide every device (the gate's session fixture does): a command started in one that does not
    is refused before it starts."""
    hidden = os.environ.get(_GPU_HIDDEN)
    assert hidden == "", f"{_GPU_HIDDEN} must be the empty string, not {hidden!r}"
    return run_rinchen("eval", "robustness", *args)


@dataclass(frozen=True)
class Run:
    """A command that exited 0: what it printed and the files it wrote."""

    invocation: Invocation
    out_dir: Path
    result: dict[str, Any]
    report: dict[str, Any]

    @property
    def block(self) -> dict[str, float | None]:
        """The six values of `report.json`."""
        return self.report["robustness"]

    @property
    def sections(self) -> dict[str, Any]:
        return self.result["sections"]


def read_result(out_dir: Path) -> dict[str, Any]:
    """`robustness.json` of a run."""
    return read_json(out_dir / RESULT_FILE)


def measured(invocation: Invocation, out_dir: Path) -> Run:
    """The `Run` of a command that must have exited 0."""
    assert invocation.returncode == 0, invocation.stderr
    return Run(
        invocation=invocation,
        out_dir=out_dir,
        result=read_result(out_dir),
        report=read_json(out_dir / REPORT_FILE),
    )


def model_run(suite: Path, model: Path, out_dir: Path, *options: str | Path) -> Run:
    """The suite scored with what the engine of the export `model` writes for it."""
    return measured(
        run_robustness(
            "--suite", suite, "--model", model, "--out-dir", out_dir, *options
        ),
        out_dir,
    )


def hypotheses_run(
    suite: Path, hypotheses: Path, out_dir: Path, *options: str | Path
) -> Run:
    """The suite scored with the hypotheses in the file `hypotheses`."""
    return measured(
        run_robustness(
            "--suite",
            suite,
            "--hypotheses",
            hypotheses,
            "--out-dir",
            out_dir,
            *options,
        ),
        out_dir,
    )


# ---- the controls ----------------------------------------------------------------------------


def rotate_hypotheses(
    path_in: Path,
    path_out: Path,
    k: int = 7,
    *,
    groups: Collection[tuple[str, str]] | None = None,
) -> None:
    """Write `path_in`, a hypotheses file, to `path_out` with the texts of every group moved
    by `k` rows: inside the (section, arm) group the line `i` gets the text of the line
    `i + k` (the last lines take the first ones). The keys and the statuses stay where they
    are, so every request is still answered; only what the answer says is another row's. The
    `groups` named (pairs of section and arm) are the only ones moved; `None` moves all.
    The file keeps the writer's form: sorted keys, one JSON object per line."""
    lines = [json.loads(line) for line in path_in.read_text("utf-8").splitlines()]
    members: defaultdict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for line in lines:
        members[(line["section"], line["arm"])].append(line)
    for group, group_lines in members.items():
        if groups is not None and group not in groups:
            continue
        size = len(group_lines)
        texts = [line["text"] for line in group_lines]
        for position, line in enumerate(group_lines):
            line["text"] = texts[(position + k) % size]
    path_out.write_text(
        "".join(
            json.dumps(line, sort_keys=True, ensure_ascii=False) + "\n"
            for line in lines
        ),
        encoding="utf-8",
        newline="\n",
    )


def copy_export_with_a_flipped_byte(export: Path, out_dir: Path) -> None:
    """A copy of the export directory `export` at `out_dir` whose `model.bin` has one byte
    changed (the manifest still lists the old digest)."""
    shutil.copytree(export, out_dir)
    weights = out_dir / "model.bin"
    data = bytearray(weights.read_bytes())
    data[len(data) // 2] ^= 0xFF
    weights.write_bytes(bytes(data))


def suite_with_an_unknown_key(path_in: Path, path_out: Path) -> None:
    """The suite file `path_in` with one key no row of the format has added to its first
    row of the first section."""
    document = json.loads(path_in.read_text("utf-8"))
    first = next(iter(document["sections"].values()))
    first[0]["unknown_key"] = "x"
    path_out.write_text(
        json.dumps(document, ensure_ascii=False), encoding="utf-8", newline="\n"
    )
