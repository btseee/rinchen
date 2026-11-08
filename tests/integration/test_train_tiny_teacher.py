"""The D8 STOP gate: a tiny teacher is trained for 200 CPU steps, finished, resumed after
a damaged checkpoint, extended, refused when its config changed, averaged and
interrupted by a real SIGTERM, all through the real `rinchen train` commands with the
real Eole. Nothing here touches a GPU: the launcher hides every one from Eole and the
config trains with `gpu_ranks: []`.

The matrix is a scratch copy this module writes: one variant (the direction tag on the
source, temperature 0.5, bt ratio 0.5, seed 1234, 100 dev lines per direction) over
the checked-in `configs/eole/teacher_tiny.yaml`, on the toy corpora of
`tests/support/tiny_parallel.py`. Each command is a subprocess of the `rinchen` binary
beside the interpreter, bounded by `SUBPROCESS_TIMEOUT_SECONDS`.

Module fixtures run the expensive commands once, in the order the scenarios need, and
take a snapshot of what the tests look at, so no test depends on the order the tests
run in:

    A  prepare only, then launch the 200-step run (about 35 s)
    B  launch it again: it is already complete and nothing is touched
    C  cut the weights of step_200 in half and launch: resumed from step_150
    D  a separate run: 100 steps, then 200 (resumed from step_100), then 50 (refused)
    E  a config with another learning rate is refused and Eole is not started
    F  average the last three checkpoints and translate the dev sets with the result
    G  SIGTERM the launcher of a fresh run, mid-run and before its first checkpoint;
       the next launch finishes it
    H  a launch into a run that another launch holds is refused (the run lock)

The toy sentences are templated and a 200-step model learns little of them: the
training loss and the validation perplexity are the gate, never a translation score.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import signal
import subprocess
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
import yaml

from rinchen.config.loader import REPO_ROOT
from rinchen.errors import ConfigError
from rinchen.manifest import store
from rinchen.manifest.model import Manifest
from rinchen.train.ablation_layout import variant_paths
from rinchen.train.checkpoints import is_complete, step_directories
from rinchen.train.curves import TrainCurve, loss_gate, parse_train_log
from rinchen.train.eole_cfg import dump_config, read_config
from rinchen.train.evaluate import DevResult, evaluate_variant, hypothesis_path
from rinchen.train.run_lock import holding_the_run
from tests.support.tiny_parallel import TOY_DIRECTIONS
from tests.support.tiny_variant import ToyInputs, build_toy_inputs

pytestmark = pytest.mark.subprocess

# A hung Eole must fail the test: the whole 200-step run takes about 35 s.
SUBPROCESS_TIMEOUT_SECONDS = 900
# Handed to `rinchen train launch --timeout`: it bounds each Eole command inside.
ENGINE_TIMEOUT_SECONDS = 600
RINCHEN = str(Path(sys.executable).with_name("rinchen"))
BASE_CONFIG = REPO_ROOT / "configs" / "eole" / "teacher_tiny.yaml"
VARIANT_ID = "t0.5_bt0.5_src_s1234"
DEV_LINES = 100
FINAL_STEP = 200
SAVED_STEPS = (50, 100, 150, 200)
REPORT_STEPS = tuple(range(10, FINAL_STEP + 1, 10))
WEIGHTS = "model.00.safetensors"
STEP_FILES = (WEIGHTS, "optimizer.pt", "config.json", "vocab.json")
# A launch that finds the run finished only reads it.
MAX_NOTHING_TO_DO_SECONDS = 30.0
# The learning rate of the config of scenario E; the base config trains with 0.5.
CHANGED_LEARNING_RATE = 0.25
# How long after the first training report the SIGTERM comes, and how long the
# launcher has to die after it.
# Seconds after the first training report at which the SIGTERM comes. The first
# checkpoint is saved about 5 s after that report: 8 s is mid-run, with a checkpoint
# on disk on an unloaded machine; half a second is before any checkpoint exists.
SIGTERM_DELAYS = {"before_the_first_checkpoint": 0.5, "mid_run": 8.0}
FIRST_REPORT_TIMEOUT_SECONDS = 300
MAX_SECONDS_TO_DIE = 10.0
# Killing the whole group of Eole ends every process of it within milliseconds. A
# data-loader worker whose parent alone was killed lives on until its next parent check
# (seconds later), so a short grace tells the two apart.
ORPHAN_GRACE_SECONDS = 3.0
SIGTERM_EXIT_CODE = 128 + signal.SIGTERM
# `UpstreamError` of a command that failed, as the CLI exits.
UPSTREAM_EXIT_CODE = 4
# The group of a killed leader is swept before the launcher returns; this is only the
# kernel's time to tear the processes down.
SWEEP_GRACE_SECONDS = 0.5
POLL_SECONDS = 0.1
_REPORT_LINE = re.compile(r"Step\s+[0-9]+/\s*[0-9]+;")

# What a resume from step_150 must not touch, as paths in a run directory.
KEPT = (
    "ckpt/step_50/",
    "ckpt/step_100/",
    "ckpt/step_150/",
    "launch/attempt-1.log",
    "launch/train-1.yaml",
    "launch/build_vocab.log",
    "launch.json",
    "vocab.src",
)

Tree = dict[str, tuple[Any, ...]]


@dataclass(frozen=True)
class Invocation:
    """One `rinchen` command: what it printed and how long it took."""

    returncode: int
    stdout: list[str]
    stderr: str
    seconds: float

    @property
    def facts(self) -> dict[str, str]:
        """The `key: value` lines of `train launch`, `train average` or
        `train ablate --prepare-only`."""
        return dict(line.split(": ", 1) for line in self.stdout)


def _rinchen(*args: str) -> Invocation:
    started = time.perf_counter()
    completed = subprocess.run(
        [RINCHEN, *args],
        capture_output=True,
        timeout=SUBPROCESS_TIMEOUT_SECONDS,
        check=False,
    )
    return Invocation(
        returncode=completed.returncode,
        stdout=completed.stdout.decode("utf-8").splitlines(),
        stderr=completed.stderr.decode("utf-8", "replace"),
        seconds=time.perf_counter() - started,
    )


def _launch_arguments(config: Path, run_dir: Path) -> list[str]:
    return [
        "train",
        "launch",
        "--config",
        str(config),
        "--run-dir",
        str(run_dir),
        "--threads",
        "2",
        "--timeout",
        str(ENGINE_TIMEOUT_SECONDS),
    ]


def _launch(config: Path, run_dir: Path) -> Invocation:
    return _rinchen(*_launch_arguments(config, run_dir))


def _ok(invocation: Invocation) -> Invocation:
    assert invocation.returncode == 0, invocation.stderr
    return invocation


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _tree(root: Path) -> Tree:
    """Every entry under `root`, links not followed: a directory, a link and where it
    points, or a file with its size, modification time and SHA-256. Two equal trees
    mean nothing was written, replaced or touched."""
    entries: Tree = {}
    for directory, dirnames, filenames in os.walk(root, followlinks=False):
        for name in [*dirnames, *filenames]:
            path = Path(directory) / name
            relative = str(path.relative_to(root))
            if path.is_symlink():
                entries[relative] = ("link", os.readlink(path))
            elif path.is_dir():
                entries[relative] = ("dir",)
            else:
                status = path.stat()
                entries[relative] = (
                    "file",
                    status.st_size,
                    status.st_mtime_ns,
                    _sha256(path),
                )
    return entries


def _disk_bytes(path: Path) -> int:
    """`du -sb`: the apparent size of everything under `path`."""
    output = subprocess.run(
        ["du", "-sb", str(path)], capture_output=True, text=True, check=True
    ).stdout
    return int(output.split()[0])


def _logs_report(log: str, step: int, total: int) -> bool:
    """Whether Eole logged the report line of `step` of `total`. It prints
    `Step 160/  200;` (`Step %2d/%5d`), so the blanks are not fixed."""
    return re.search(rf"\bStep\s+{step}/\s*{total};", log) is not None


def _attempt_number(log: Path) -> int:
    return int(log.stem.split("-")[1])


def _attempt_logs(run_dir: Path) -> list[Path]:
    return sorted((run_dir / "launch").glob("attempt-*.log"), key=_attempt_number)


@dataclass(frozen=True)
class RunPaths:
    """Where a run keeps its files, as `rinchen train ablate` lays them out."""

    run_dir: Path

    @property
    def model_path(self) -> Path:
        return self.run_dir / "ckpt"

    def step(self, step: int) -> Path:
        return self.model_path / f"step_{step}"


@dataclass(frozen=True)
class RunState:
    """A run directory right after a launch that ran or found the run finished:
    what the command printed and what was on disk."""

    invocation: Invocation
    steps: dict[int, str | None]
    root_links: dict[str, str]
    log: str
    curve: TrainCurve
    launch_json: bytes
    manifest: Manifest | None
    digests: dict[str, str]
    tree: Tree

    @property
    def facts(self) -> dict[str, str]:
        return self.invocation.facts


def _state(run_dir: Path, invocation: Invocation) -> RunState:
    paths = RunPaths(run_dir)
    log = _attempt_logs(run_dir)[-1].read_text(encoding="utf-8", errors="replace")
    manifest = store.read(run_dir)
    return RunState(
        invocation=invocation,
        steps={
            directory.step: is_complete(directory.path)
            for directory in step_directories(paths.model_path)
        },
        root_links={
            link.name: os.path.realpath(link)
            for link in sorted(paths.model_path.iterdir())
            if link.is_symlink()
        },
        log=log,
        curve=parse_train_log(log),
        launch_json=(run_dir / "launch.json").read_bytes(),
        manifest=manifest,
        digests=(
            {}
            if manifest is None
            else {item.path: _sha256(Path(item.path)) for item in manifest.outputs}
        ),
        tree=_tree(run_dir),
    )


def _truncate_to_half(path: Path) -> int:
    """Cut `path` to half its size and return the size it had."""
    size = path.stat().st_size
    with path.open("r+b") as handle:
        handle.truncate(size // 2)
    return size


def _config_copy(
    source: Path, destination: Path, run: RunPaths, **training: Any
) -> Path:
    """A copy of the rendered config at `source` for the run `run`: `model_path`
    and `src_vocab` are rewritten into the run's directory, `training` values are laid
    over the training section. The corpora stay where they are; they are only read."""
    cfg = read_config(source, "Eole config")
    cfg["src_vocab"] = str(run.run_dir / "vocab.src")
    cfg["training"]["model_path"] = str(run.model_path)
    cfg["training"].update(training)
    destination.parent.mkdir(parents=True, exist_ok=True)
    dump_config(cfg, destination)
    return destination


@dataclass(frozen=True)
class Workspace:
    root: Path
    inputs: ToyInputs
    matrix: Path
    out_dir: Path


@pytest.fixture(scope="module")
def workspace(tmp_path_factory: pytest.TempPathFactory) -> Workspace:
    root = tmp_path_factory.mktemp("train_teacher")
    inputs = build_toy_inputs(root / "in")
    matrix = root / "matrix.yaml"
    matrix.write_text(
        yaml.safe_dump(
            {
                "name": "teacher_tiny_gate",
                "base": str(BASE_CONFIG),
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
    return Workspace(root, inputs, matrix, root / "out")


@dataclass(frozen=True)
class Prepared:
    """The result of `rinchen train ablate --prepare-only`."""

    invocation: Invocation
    config_path: Path
    run: RunPaths
    config: dict[str, Any]
    files_after: list[str]


@pytest.fixture(scope="module")
def prepared(workspace: Workspace) -> Prepared:
    invocation = _ok(
        _rinchen(
            "train",
            "ablate",
            "--matrix",
            str(workspace.matrix),
            "--corpus",
            str(workspace.inputs.corpus),
            "--spm-model",
            str(workspace.inputs.spm_model),
            "--out-dir",
            str(workspace.out_dir),
            "--prepare-only",
            "--threads",
            "2",
        )
    )
    paths = variant_paths(workspace.out_dir, VARIANT_ID)
    return Prepared(
        invocation=invocation,
        config_path=paths.config,
        run=RunPaths(paths.run_dir),
        config=read_config(paths.config, "Eole config"),
        files_after=sorted(
            str(path.relative_to(workspace.out_dir))
            for path in workspace.out_dir.rglob("*")
        ),
    )


@pytest.fixture(scope="module")
def first_launch(prepared: Prepared) -> RunState:
    """Scenario A: the 200-step run, launched from nothing."""
    invocation = _ok(_launch(prepared.config_path, prepared.run.run_dir))
    return _state(prepared.run.run_dir, invocation)


@dataclass(frozen=True)
class Rerun:
    """Scenario B: the state before and after launching a finished run again."""

    state: RunState
    tree_before: Tree


@pytest.fixture(scope="module")
def rerun(prepared: Prepared, first_launch: RunState) -> Rerun:
    tree_before = _tree(prepared.run.run_dir)
    invocation = _ok(_launch(prepared.config_path, prepared.run.run_dir))
    return Rerun(_state(prepared.run.run_dir, invocation), tree_before)


@dataclass(frozen=True)
class Resumed:
    """Scenario C: the state after the weights of step_200 were cut in half."""

    state: RunState
    weights_size: int
    problem_before: str | None


@pytest.fixture(scope="module")
def resumed(prepared: Prepared, first_launch: RunState, rerun: Rerun) -> Resumed:
    weights = prepared.run.step(FINAL_STEP) / WEIGHTS
    size = _truncate_to_half(weights)
    problem = is_complete(prepared.run.step(FINAL_STEP))
    invocation = _ok(_launch(prepared.config_path, prepared.run.run_dir))
    return Resumed(_state(prepared.run.run_dir, invocation), size, problem)


@dataclass(frozen=True)
class Extension:
    """Scenario D: a separate run of 100 steps, extended to 200, then lowered to 50."""

    run: RunPaths
    to_100: RunState
    to_200: RunState
    lowered: Invocation
    tree_before_lowering: Tree
    tree_after_lowering: Tree


@pytest.fixture(scope="module")
def extension(workspace: Workspace, prepared: Prepared) -> Extension:
    run = RunPaths(workspace.root / "separate" / "run")
    configs = workspace.root / "separate"
    to_100 = _state(
        run.run_dir,
        _ok(
            _launch(
                _config_copy(
                    prepared.config_path,
                    configs / "steps-100.yaml",
                    run,
                    train_steps=100,
                ),
                run.run_dir,
            )
        ),
    )
    config_200 = _config_copy(
        prepared.config_path, configs / "steps-200.yaml", run, train_steps=200
    )
    to_200 = _state(run.run_dir, _ok(_launch(config_200, run.run_dir)))
    tree_before = _tree(run.run_dir)
    lowered = _launch(
        _config_copy(
            prepared.config_path, configs / "steps-50.yaml", run, train_steps=50
        ),
        run.run_dir,
    )
    return Extension(run, to_100, to_200, lowered, tree_before, _tree(run.run_dir))


@dataclass(frozen=True)
class Refused:
    """Scenario E: a launch of the finished run of A with another learning rate."""

    same_values: Invocation
    changed: Invocation
    tree_before: Tree
    tree_after: Tree


@pytest.fixture(scope="module")
def refused(
    workspace: Workspace, prepared: Prepared, first_launch: RunState, resumed: Resumed
) -> Refused:
    configs = workspace.root / "refused"
    # The control: the same values written by another hand (a rewritten file with
    # the same settings) are the same run, so the refusal below is for the rate.
    same = _launch(
        _config_copy(prepared.config_path, configs / "same.yaml", prepared.run),
        prepared.run.run_dir,
    )
    changed_config = _config_copy(
        prepared.config_path,
        configs / "changed.yaml",
        prepared.run,
        learning_rate=CHANGED_LEARNING_RATE,
    )
    tree_before = _tree(prepared.run.run_dir)
    changed = _launch(changed_config, prepared.run.run_dir)
    return Refused(same, changed, tree_before, _tree(prepared.run.run_dir))


@dataclass(frozen=True)
class Averaged:
    """Scenario F: `rinchen train average --last 3` on the run of A, the dev sets
    translated with the result, and a second average into the same directory."""

    invocation: Invocation
    directory: Path
    document: dict[str, Any]
    manifest: Manifest | None
    result: DevResult
    eval_dir: Path
    eval_seconds: float
    repeated: Invocation
    tree_before_repeat: Tree
    tree_after_repeat: Tree


@pytest.fixture(scope="module")
def averaged(
    workspace: Workspace, prepared: Prepared, resumed: Resumed, refused: Refused
) -> Averaged:
    invocation = _ok(
        _rinchen(
            "train",
            "average",
            "--checkpoint-dir",
            str(prepared.run.model_path),
            "--last",
            "3",
        )
    )
    directory = prepared.run.run_dir / "averaged"
    eval_dir = workspace.root / "averaged_eval"
    started = time.perf_counter()
    result = evaluate_variant(
        directory,
        workspace.inputs.dev,
        spm_model=workspace.inputs.spm_model,
        placement="src",
        out_dir=eval_dir,
        threads=2,
        timeout=SUBPROCESS_TIMEOUT_SECONDS,
    )
    eval_seconds = time.perf_counter() - started
    tree_before = _tree(directory)
    repeated = _rinchen(
        "train",
        "average",
        "--checkpoint-dir",
        str(prepared.run.model_path),
        "--last",
        "3",
    )
    return Averaged(
        invocation=invocation,
        directory=directory,
        document=json.loads((directory / "averaged_from.json").read_text("utf-8")),
        manifest=store.read(directory),
        result=result,
        eval_dir=eval_dir,
        eval_seconds=eval_seconds,
        repeated=repeated,
        tree_before_repeat=tree_before,
        tree_after_repeat=_tree(directory),
    )


# ---------------------------------------------------------------------------------
# Scenario A: prepare only, then the 200-step launch.


def test_prepare_only_writes_the_config_and_trains_nothing(
    workspace: Workspace, prepared: Prepared, record_property: Any
) -> None:
    record_property("seconds", round(prepared.invocation.seconds, 1))

    first = prepared.invocation.stdout[0]
    assert first == f"prepared {VARIANT_ID}: {prepared.config_path}"
    assert prepared.config_path.is_file()
    assert not prepared.run.run_dir.exists()
    assert [
        name for name in prepared.files_after if name.endswith(".safetensors")
    ] == []
    assert [name for name in prepared.files_after if "step_" in name] == []
    # The early-stopping code path is off in the tiny config: the gate must not be able
    # to end by it.
    assert prepared.config["training"].get("early_stopping", 0) == 0
    assert prepared.config["training"]["gpu_ranks"] == []


def test_the_first_launch_completes_the_200_steps_from_nothing(
    prepared: Prepared, first_launch: RunState, record_property: Any
) -> None:
    run = prepared.run
    record_property("seconds", round(first_launch.invocation.seconds, 1))
    record_property("run_dir_bytes", _disk_bytes(run.run_dir))
    record_property(
        "step_dir_bytes", {step: _disk_bytes(run.step(step)) for step in SAVED_STEPS}
    )

    facts = first_launch.facts
    assert facts["status"] == "completed"
    assert facts["steps"] == str(FINAL_STEP)
    assert facts["resumed from"] == "none"
    assert facts["attempt"] == "1"
    assert facts["checkpoint"] == str(run.step(FINAL_STEP))
    assert facts["log"] == str(run.run_dir / "launch" / "attempt-1.log")
    assert facts["command.sh"] == str(run.run_dir / "command.sh")
    assert facts["manifest"] == str(run.run_dir / "manifest.json")
    assert "incomplete step directories skipped" not in facts
    for name in ("launch.json", "command.sh", "launch/attempt-1.log", "launch/lock"):
        assert (run.run_dir / name).is_file(), name


def test_command_sh_is_a_script_bash_accepts(
    prepared: Prepared, first_launch: RunState
) -> None:
    script = prepared.run.run_dir / "command.sh"

    checked = subprocess.run(
        ["bash", "-n", str(script)], capture_output=True, check=False
    )

    assert checked.returncode == 0, checked.stderr
    text = script.read_text(encoding="utf-8")
    assert text.startswith("#!/usr/bin/env bash\n")
    assert "export CUDA_VISIBLE_DEVICES=''\n" in text
    assert "train -config" in text


def test_the_four_step_directories_are_complete_and_the_root_links_follow_step_200(
    prepared: Prepared, first_launch: RunState
) -> None:
    run = prepared.run

    assert first_launch.steps == dict.fromkeys(SAVED_STEPS)
    for step in SAVED_STEPS:
        assert set(STEP_FILES) <= {path.name for path in run.step(step).iterdir()}
    # Eole's own links in the checkpoint directory, not what a resume reads.
    assert set(first_launch.root_links) >= set(STEP_FILES)
    for name in STEP_FILES:
        assert first_launch.root_links[name] == os.path.realpath(
            run.step(FINAL_STEP) / name
        )


def test_the_loss_falls_by_at_least_30_percent_without_a_nan(
    first_launch: RunState, record_property: Any
) -> None:
    curve = first_launch.curve
    first, last = curve.reports[0], curve.reports[-1]
    drop = 1 - last.xent / first.xent
    record_property("xent_first", first.xent)
    record_property("xent_last", last.xent)
    record_property("xent_drop_percent", round(100 * drop, 1))
    record_property("validation_perplexities", list(curve.validations))

    assert curve.non_finite_step is None
    assert all(math.isfinite(value) for report in curve.reports for value in report[2:])
    assert loss_gate(curve) is None
    assert last.xent <= 0.7 * first.xent
    assert curve.validations[-1] < curve.validations[0]


def test_the_log_has_a_report_every_10_steps_and_a_validation_every_save(
    first_launch: RunState,
) -> None:
    curve = first_launch.curve

    assert tuple(report.step for report in curve.reports) == REPORT_STEPS
    assert {report.total for report in curve.reports} == {FINAL_STEP}
    assert curve.saved_steps == SAVED_STEPS
    assert len(curve.validations) == len(SAVED_STEPS)
    assert (curve.runs, curve.unparsed_reports) == (1, 0)
    assert not curve.early_stopped
    assert "Traceback" not in first_launch.log
    assert " ERROR]" not in first_launch.log


def test_the_launch_manifest_hashes_match_the_files(
    prepared: Prepared, first_launch: RunState
) -> None:
    manifest = first_launch.manifest
    assert manifest is not None
    step = prepared.run.step(FINAL_STEP)

    assert manifest.stage == "train_launch"
    assert manifest.status == {"launch": "ok"}
    assert [Path(item.path) for item in manifest.outputs] == [
        *(step / name for name in STEP_FILES),
        prepared.run.run_dir / "launch.json",
        prepared.run.run_dir / "command.sh",
    ]
    assert [item.sha256 for item in manifest.outputs] == [
        _sha256(Path(item.path)) for item in manifest.outputs
    ]
    assert [item.path for item in manifest.inputs] == [str(prepared.config_path)]
    assert manifest.inputs[0].sha256 == _sha256(prepared.config_path)
    metrics = manifest.metrics
    assert (metrics["steps"], metrics["attempt"]) == (FINAL_STEP, 1)
    assert metrics["early_stopped"] == 0.0
    assert "resumed_from" not in metrics
    curve = first_launch.curve
    assert metrics["xent_first"] == curve.reports[0].xent
    assert metrics["xent_last"] == curve.reports[-1].xent
    assert metrics["valid_ppl_first"] == curve.validations[0]
    assert metrics["valid_ppl_last"] == curve.validations[-1]


# ---------------------------------------------------------------------------------
# Scenario B: launching a finished run again changes nothing.


def test_a_second_launch_finds_the_run_finished_and_touches_nothing(
    prepared: Prepared, rerun: Rerun, record_property: Any
) -> None:
    record_property("seconds", round(rerun.state.invocation.seconds, 1))

    facts = rerun.state.facts
    assert facts["status"] == "already_complete"
    assert facts["steps"] == str(FINAL_STEP)
    assert facts["resumed from"] == "none"
    assert facts["attempt"] == "0"
    assert facts["log"] == "none"
    assert facts["command.sh"] == "none"
    assert rerun.state.invocation.seconds < MAX_NOTHING_TO_DO_SECONDS
    # Every file, its size, its modification time and its bytes, the links, and the
    # manifest and the launch record with them.
    assert rerun.state.tree == rerun.tree_before
    assert [log.name for log in _attempt_logs(prepared.run.run_dir)] == [
        "attempt-1.log"
    ]


# ---------------------------------------------------------------------------------
# Scenario C: a cut weights file is skipped and the run resumes from step_150.


def test_cutting_the_weights_of_step_200_makes_the_launch_resume_from_step_150(
    prepared: Prepared, resumed: Resumed, record_property: Any
) -> None:
    record_property("seconds", round(resumed.state.invocation.seconds, 1))

    assert resumed.problem_before is not None
    assert resumed.problem_before.startswith(WEIGHTS)
    facts = resumed.state.facts
    assert facts["status"] == "completed"
    assert facts["steps"] == str(FINAL_STEP)
    assert facts["resumed from"] == "150"
    assert facts["attempt"] == "2"
    assert facts["incomplete step directories skipped"] == str([FINAL_STEP])
    assert resumed.state.steps == dict.fromkeys(SAVED_STEPS)
    weights = prepared.run.step(FINAL_STEP) / WEIGHTS
    assert weights.stat().st_size == resumed.weights_size


def test_the_resumed_attempt_logs_its_first_report_at_step_160(
    prepared: Prepared, resumed: Resumed
) -> None:
    curve = resumed.state.curve

    assert _logs_report(resumed.state.log, 160, FINAL_STEP)
    assert tuple(report.step for report in curve.reports) == tuple(range(160, 201, 10))
    assert {report.total for report in curve.reports} == {FINAL_STEP}
    assert curve.saved_steps == (FINAL_STEP,)
    assert len(curve.validations) == 1
    assert curve.non_finite_step is None
    assert not curve.early_stopped
    assert [log.name for log in _attempt_logs(prepared.run.run_dir)] == [
        "attempt-1.log",
        "attempt-2.log",
    ]
    config = read_config(prepared.run.run_dir / "launch" / "train-2.yaml", "config")
    assert config["training"]["train_from"] == str(prepared.run.step(150))
    first = read_config(prepared.run.run_dir / "launch" / "train-1.yaml", "config")
    assert "train_from" not in first["training"]


def test_the_resume_leaves_the_earlier_files_of_the_run_as_they_were(
    first_launch: RunState, resumed: Resumed
) -> None:
    def kept(tree: Tree) -> Tree:
        return {name: entry for name, entry in tree.items() if name.startswith(KEPT)}

    assert kept(first_launch.tree)
    assert kept(resumed.state.tree) == kept(first_launch.tree)
    assert resumed.state.launch_json == first_launch.launch_json
    assert resumed.state.root_links == first_launch.root_links


def test_the_manifest_is_refreshed_and_matches_the_files_again(
    prepared: Prepared, first_launch: RunState, resumed: Resumed
) -> None:
    manifest, before = resumed.state.manifest, first_launch.manifest
    assert manifest is not None and before is not None

    assert resumed.state.digests == {
        item.path: _sha256(Path(item.path)) for item in manifest.outputs
    }
    weights = str(prepared.run.step(FINAL_STEP) / WEIGHTS)
    old = {item.path: item.sha256 for item in before.outputs}
    new = {item.path: item.sha256 for item in manifest.outputs}
    # A resume restarts the data stream (spec finding 1): the retrained step is not
    # the one that was cut.
    assert new[weights] != old[weights]
    assert manifest.metrics["resumed_from"] == 150.0
    assert manifest.metrics["attempt"] == 2.0
    assert manifest.metrics["incomplete_skipped"] == 1.0
    assert manifest.metrics["steps"] == float(FINAL_STEP)


# ---------------------------------------------------------------------------------
# Scenario D: a separate run, extended from 100 to 200 steps.


def test_a_100_step_run_completes_at_step_100(extension: Extension) -> None:
    state = extension.to_100

    assert state.facts["status"] == "completed"
    assert state.facts["steps"] == "100"
    assert state.facts["resumed from"] == "none"
    assert state.steps == {50: None, 100: None}
    assert state.curve.reports[-1].step == 100
    assert state.curve.reports[-1].total == 100


def test_raising_train_steps_to_200_resumes_from_step_100_and_completes(
    extension: Extension, record_property: Any
) -> None:
    record_property("seconds_to_100", round(extension.to_100.invocation.seconds, 1))
    record_property("seconds_to_200", round(extension.to_200.invocation.seconds, 1))
    state = extension.to_200

    assert state.facts["status"] == "completed"
    assert state.facts["steps"] == str(FINAL_STEP)
    assert state.facts["resumed from"] == "100"
    assert state.facts["attempt"] == "2"
    assert _logs_report(state.log, 110, FINAL_STEP)
    assert tuple(report.step for report in state.curve.reports) == tuple(
        range(110, 201, 10)
    )
    assert state.steps == dict.fromkeys(SAVED_STEPS)
    # train_steps is the one setting a run may change: the record is the same.
    assert state.launch_json == extension.to_100.launch_json
    assert state.manifest is not None
    assert state.manifest.metrics["resumed_from"] == 100.0


def test_lowering_train_steps_below_the_checkpoints_is_refused_naming_both_numbers(
    extension: Extension,
) -> None:
    lowered = extension.lowered

    assert lowered.returncode == 2, lowered.stderr
    assert "50" in lowered.stderr
    assert str(FINAL_STEP) in lowered.stderr
    assert "train_steps" in lowered.stderr
    assert extension.tree_after_lowering == extension.tree_before_lowering


# ---------------------------------------------------------------------------------
# Scenario E: a changed learning rate is refused before Eole starts.


def test_the_same_values_rewritten_are_the_same_run(refused: Refused) -> None:
    assert refused.same_values.returncode == 0, refused.same_values.stderr
    assert refused.same_values.facts["status"] == "already_complete"


def test_another_learning_rate_is_refused_and_eole_is_not_started(
    prepared: Prepared, refused: Refused
) -> None:
    changed = refused.changed

    assert changed.returncode == 2, changed.stderr
    assert "not the one this run was started with" in changed.stderr
    assert "launch.json" in changed.stderr
    assert refused.tree_after == refused.tree_before
    assert [log.name for log in _attempt_logs(prepared.run.run_dir)] == [
        "attempt-1.log",
        "attempt-2.log",
    ]
    assert sorted(
        directory.step for directory in step_directories(prepared.run.model_path)
    ) == list(SAVED_STEPS)


# ---------------------------------------------------------------------------------
# Scenario F: average the last three checkpoints and translate with the result.


def test_average_writes_the_model_and_records_the_three_steps(
    prepared: Prepared, averaged: Averaged
) -> None:
    facts = averaged.invocation.facts

    assert facts["averaged"] == "3 checkpoints (steps 100, 150, 200)"
    assert facts["output"] == str(averaged.directory)
    assert averaged.directory == prepared.run.run_dir / "averaged"
    assert [source["step"] for source in averaged.document["sources"]] == [
        100,
        150,
        200,
    ]
    assert [Path(source["path"]) for source in averaged.document["sources"]] == [
        prepared.run.step(step) for step in (100, 150, 200)
    ]
    for source in averaged.document["sources"]:
        assert source["sha256"] == _sha256(Path(source["path"]) / WEIGHTS)
    weights = averaged.directory / WEIGHTS
    assert averaged.document["weights"] == {"file": WEIGHTS, "sha256": _sha256(weights)}
    assert facts["weights sha256"] == _sha256(weights)
    assert not (averaged.directory / "optimizer.pt").exists()
    for name in ("config.json", "vocab.json"):
        assert (averaged.directory / name).read_bytes() == (
            prepared.run.step(FINAL_STEP) / name
        ).read_bytes()
    assert _sha256(weights) != _sha256(prepared.run.step(FINAL_STEP) / WEIGHTS)


def test_the_average_manifest_hashes_match_the_files(
    prepared: Prepared, averaged: Averaged
) -> None:
    manifest = averaged.manifest
    assert manifest is not None

    assert manifest.stage == "train_average"
    assert manifest.status == {"average": "ok"}
    assert [Path(item.path) for item in manifest.inputs] == [
        prepared.run.step(step) / WEIGHTS for step in (100, 150, 200)
    ]
    for item in [*manifest.inputs, *manifest.outputs]:
        assert item.sha256 == _sha256(Path(item.path)), item.path
    assert str(averaged.directory / WEIGHTS) in {item.path for item in manifest.outputs}
    assert manifest.metrics["first_step"] == 100.0
    assert manifest.metrics["last_step"] == 200.0


def test_the_dev_sets_are_translated_with_the_averaged_model(
    averaged: Averaged, record_property: Any
) -> None:
    result = averaged.result
    record_property("eval_seconds", round(averaged.eval_seconds, 1))
    record_property(
        "averaged_chrf",
        {name: round(scores.chrf, 2) for name, scores in result.directions.items()},
    )
    record_property("averaged_tag_leaks", result.tag_leaks)

    # No quality threshold, on purpose: a 200-step model translates the templated
    # toy sentences poorly, and averaging checkpoints taken in the steep part of the
    # loss curve can make it worse than the last one (spec finding 4). What is
    # checked is that the averaged files are a model `eole predict` can use.
    assert list(result.directions) == sorted(TOY_DIRECTIONS)
    for direction, scores in result.directions.items():
        assert scores.lines == DEV_LINES, direction
        assert math.isfinite(scores.chrf) and 0 <= scores.chrf <= 100, direction
        assert math.isfinite(scores.bleu) and 0 <= scores.bleu <= 100, direction
        hypotheses = hypothesis_path(averaged.eval_dir, direction)
        assert len(hypotheses.read_text("utf-8").splitlines()) == DEV_LINES
    assert math.isfinite(result.chrf_mean) and 0 <= result.chrf_mean <= 100
    assert math.isfinite(result.chrf_min) and 0 <= result.chrf_min <= 100
    assert isinstance(result.tag_leaks, int)


def test_a_second_average_into_the_same_directory_is_refused(
    averaged: Averaged,
) -> None:
    assert averaged.repeated.returncode == 2, averaged.repeated.stderr
    assert str(averaged.directory) in averaged.repeated.stderr
    assert averaged.tree_after_repeat == averaged.tree_before_repeat


# ---------------------------------------------------------------------------------
# Scenario G: a real SIGTERM to the launcher.


def _process_table() -> dict[int, tuple[int, str, str]]:
    """pid -> (parent pid, state, start time in clock ticks) of every process."""
    table: dict[int, tuple[int, str, str]] = {}
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            text = (entry / "stat").read_text(encoding="utf-8")
        except OSError:
            continue  # it ended while the table was read
        # The name between the parentheses can hold spaces and parentheses.
        fields = text[text.rindex(")") + 2 :].split()
        table[int(entry.name)] = (int(fields[1]), fields[0], fields[19])
    return table


def _descendants(root: int) -> dict[int, str]:
    """pid -> start time of every process below `root` (its children, theirs, ...)."""
    table = _process_table()
    found: dict[int, str] = {}
    pending = [root]
    while pending:
        parent = pending.pop()
        for pid, (ppid, _state, started) in table.items():
            if ppid == parent and pid not in found:
                found[pid] = started
                pending.append(pid)
    return found


def _still_running(processes: dict[int, str]) -> list[int]:
    """The pids of `processes` that are alive (the same process, not a zombie)."""
    table = _process_table()
    return [
        pid
        for pid, started in processes.items()
        if pid in table and table[pid][2] == started and table[pid][1] != "Z"
    ]


def _wait_for(condition: Callable[[], bool], seconds: float) -> bool:
    """Poll `condition` until it holds (True) or `seconds` have passed (False)."""
    deadline = time.monotonic() + seconds
    while not condition():
        if time.monotonic() >= deadline:
            return False
        time.sleep(POLL_SECONDS)
    return True


def _group_gone(leader: int) -> bool:
    """Whether no process is left in the process group `leader`."""
    try:
        os.killpg(leader, 0)
    except ProcessLookupError:
        return True
    return False


def _start_launcher(
    config: Path, run: RunPaths, output: Path
) -> subprocess.Popen[bytes]:
    """`rinchen train launch` in a session of its own, its output in `output`."""
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as sink:
        return subprocess.Popen(
            [RINCHEN, *_launch_arguments(config, run.run_dir)],
            stdout=sink,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
        )


def _wait_for_the_first_report(
    process: subprocess.Popen[bytes], run: RunPaths, output: Path
) -> None:
    """Return once the launcher's first attempt has logged a training report."""
    log = run.run_dir / "launch" / "attempt-1.log"
    reported = _wait_for(
        lambda: (
            process.poll() is not None
            or (
                log.is_file()
                and _REPORT_LINE.search(log.read_text("utf-8", "replace")) is not None
            )
        ),
        FIRST_REPORT_TIMEOUT_SECONDS,
    )
    assert reported, f"no training report in {FIRST_REPORT_TIMEOUT_SECONDS} s"
    assert process.poll() is None, output.read_text("utf-8", "replace")


def _stop_launcher(process: subprocess.Popen[bytes]) -> None:
    """Leave nothing running: SIGTERM the launcher if it still runs (it kills Eole's
    group), SIGKILL it when it does not end, then SIGKILL what is left below it."""
    below = _descendants(process.pid) if process.poll() is None else {}
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
    for pid in _still_running(below):
        os.kill(pid, signal.SIGKILL)


@dataclass(frozen=True)
class Terminated:
    """Scenario G: the launcher killed by SIGTERM and the launch that followed."""

    case: str
    run: RunPaths
    returncode: int
    seconds_to_die: float
    descendants: dict[int, str]
    survivors: list[int]
    group_gone: bool
    child_groups_gone: bool
    lock_free: bool
    manifest_left: bool
    complete_steps_before: list[int]
    incomplete_steps_before: list[int]
    stdout_before: str
    after: RunState


@pytest.fixture(scope="module", params=list(SIGTERM_DELAYS))
def terminated(
    request: pytest.FixtureRequest, workspace: Workspace, prepared: Prepared
) -> Terminated:
    case: str = request.param
    return _interrupted_launch(
        workspace, prepared, case=case, delay=SIGTERM_DELAYS[case]
    )


def _interrupted_launch(
    workspace: Workspace, prepared: Prepared, *, case: str, delay: float
) -> Terminated:
    """Launch a fresh run in a session of its own, send SIGTERM to the launcher
    `delay` seconds after its first training report, see what is left, and launch
    the same config again."""
    run = RunPaths(workspace.root / case / "run")
    config = _config_copy(
        prepared.config_path, workspace.root / case / "config.yaml", run
    )
    output = workspace.root / case / "launch.out"
    process = _start_launcher(config, run, output)
    try:
        _wait_for_the_first_report(process, run, output)
        try:
            process.wait(timeout=delay)
        except subprocess.TimeoutExpired:
            pass
        else:
            pytest.fail("the launch ended before the SIGTERM")
        descendants = _descendants(process.pid)
        table = _process_table()
        child_pids = [pid for pid in descendants if table[pid][0] == process.pid]
        sent = time.perf_counter()
        process.send_signal(signal.SIGTERM)
        returncode = process.wait(timeout=SUBPROCESS_TIMEOUT_SECONDS)
        seconds_to_die = time.perf_counter() - sent
    except BaseException:
        _stop_launcher(process)
        raise

    _wait_for(
        lambda: not _still_running(descendants) and _group_gone(process.pid),
        ORPHAN_GRACE_SECONDS,
    )
    survivors = _still_running(descendants)
    launcher_group_gone = _group_gone(process.pid)
    child_groups_gone = all(_group_gone(pid) for pid in child_pids)
    for pid in survivors:  # the test reports them; they must not outlive it
        os.kill(pid, signal.SIGKILL)
    lock_free = True
    try:
        with holding_the_run(run.run_dir / "launch" / "lock"):
            pass
    except ConfigError:
        lock_free = False
    steps = step_directories(run.model_path)
    manifest_left = (run.run_dir / "manifest.json").exists()
    after = _state(run.run_dir, _ok(_launch(config, run.run_dir)))
    return Terminated(
        case=case,
        run=run,
        returncode=returncode,
        seconds_to_die=seconds_to_die,
        descendants=descendants,
        survivors=survivors,
        group_gone=launcher_group_gone,
        child_groups_gone=child_groups_gone,
        lock_free=lock_free,
        manifest_left=manifest_left,
        complete_steps_before=[step.step for step in steps if step.complete],
        incomplete_steps_before=[step.step for step in steps if not step.complete],
        stdout_before=output.read_text("utf-8", "replace"),
        after=after,
    )


@pytest.mark.skipif(sys.platform != "linux", reason="reads /proc")
def test_sigterm_ends_the_launcher_with_143_and_leaves_no_process_of_the_run(
    terminated: Terminated, record_property: Any
) -> None:
    record_property("seconds_to_die", round(terminated.seconds_to_die, 2))
    record_property("processes_below_the_launcher", len(terminated.descendants))

    assert terminated.returncode == SIGTERM_EXIT_CODE, terminated.stdout_before
    assert terminated.seconds_to_die < MAX_SECONDS_TO_DIE
    # Eole and the data-loader workers it started, none of which may outlive the launch.
    assert len(terminated.descendants) >= 1
    assert terminated.survivors == []
    assert terminated.group_gone
    assert terminated.child_groups_gone
    assert terminated.lock_free


@pytest.mark.skipif(sys.platform != "linux", reason="reads /proc")
def test_the_launch_after_the_sigterm_finishes_the_run(
    terminated: Terminated, record_property: Any
) -> None:
    after, run = terminated.after, terminated.run
    newest = max(terminated.complete_steps_before, default=None)
    record_property(
        "case", "fresh start" if newest is None else f"resumed from step {newest}"
    )
    record_property("complete_steps_at_the_kill", terminated.complete_steps_before)
    record_property("incomplete_steps_at_the_kill", terminated.incomplete_steps_before)

    # The launch record is written before Eole starts, so it exists in both cases; a
    # record without a complete step is a fresh start, otherwise the newest complete
    # step is the resume point.
    if terminated.case == "before_the_first_checkpoint":
        assert terminated.complete_steps_before == []
        assert terminated.incomplete_steps_before == []
    assert (run.run_dir / "launch.json").is_file()
    assert after.facts["status"] == "completed"
    assert after.facts["steps"] == str(FINAL_STEP)
    assert after.facts["resumed from"] == ("none" if newest is None else str(newest))
    assert after.facts["attempt"] == "2"
    assert after.steps[FINAL_STEP] is None
    assert is_complete(run.step(FINAL_STEP)) is None
    # Whatever the killed attempt left in the root links, the finished run's follow
    # the final step.
    for name in STEP_FILES:
        assert after.root_links[name] == os.path.realpath(run.step(FINAL_STEP) / name)
    assert after.curve.non_finite_step is None
    assert after.manifest is not None
    assert after.digests == {
        item.path: _sha256(Path(item.path)) for item in after.manifest.outputs
    }


@pytest.mark.skipif(sys.platform != "linux", reason="reads /proc")
def test_the_killed_launch_left_no_manifest_for_a_run_it_did_not_finish(
    terminated: Terminated,
) -> None:
    assert not terminated.manifest_left


# ---------------------------------------------------------------------------------
# Scenario H: a second launch into a run that is being launched is refused.

# What the running launch wrote before Eole started; a refused launch must not touch it.
LAUNCHER_FILES = ("launch.json", "command.sh", "launch/train-1.yaml")


@dataclass(frozen=True)
class Rivals:
    """Scenario H: the launches tried while a launch ran in the same run directory."""

    lock: Path
    launch: Invocation
    write_only: Invocation
    files_before: dict[str, bytes]
    files_after: dict[str, bytes]
    first_still_running: bool


@pytest.fixture(scope="module")
def rivals(workspace: Workspace, prepared: Prepared) -> Rivals:
    run = RunPaths(workspace.root / "rivals" / "run")
    config = _config_copy(
        prepared.config_path, workspace.root / "rivals" / "config.yaml", run
    )
    output = workspace.root / "rivals" / "launch.out"
    process = _start_launcher(config, run, output)
    try:
        _wait_for_the_first_report(process, run, output)

        def files() -> dict[str, bytes]:
            return {name: (run.run_dir / name).read_bytes() for name in LAUNCHER_FILES}

        before = files()
        launch = _launch(config, run.run_dir)
        write_only = _rinchen(*_launch_arguments(config, run.run_dir), "--write-only")
        return Rivals(
            lock=run.run_dir / "launch" / "lock",
            launch=launch,
            write_only=write_only,
            files_before=before,
            files_after=files(),
            first_still_running=process.poll() is None,
        )
    finally:
        _stop_launcher(process)


@pytest.mark.skipif(sys.platform != "linux", reason="reads /proc")
def test_a_second_launch_into_a_run_being_launched_is_refused_naming_the_lock(
    rivals: Rivals,
) -> None:
    for rival in (rivals.launch, rivals.write_only):
        assert rival.returncode == 2, rival.stderr
        assert "another launch holds this run" in rival.stderr
        assert str(rivals.lock) in rival.stderr
    assert rivals.first_still_running
    assert rivals.files_after == rivals.files_before


# ---------------------------------------------------------------------------------
# Scenario I: the `eole train` leader killed on its own (a crash or an OOM-kill).


@dataclass(frozen=True)
class LeaderKilled:
    """Scenario I: `eole train` killed by SIGKILL while the launcher waited for it, and
    the launch that followed."""

    returncode: int
    output: str
    group_before: dict[int, str]
    survivors: list[int]
    group_gone: bool
    lock_free: bool
    manifest_left: bool
    after: RunState


@pytest.fixture(scope="module")
def leader_killed(workspace: Workspace, prepared: Prepared) -> LeaderKilled:
    run = RunPaths(workspace.root / "leader_killed" / "run")
    config = _config_copy(
        prepared.config_path, workspace.root / "leader_killed" / "config.yaml", run
    )
    output = workspace.root / "leader_killed" / "launch.out"
    process = _start_launcher(config, run, output)
    try:
        _wait_for_the_first_report(process, run, output)
        table = _process_table()
        (leader,) = [
            pid for pid in _descendants(process.pid) if table[pid][0] == process.pid
        ]
        group_before = {
            pid: started
            for pid, started in _descendants(process.pid).items()
            if os.getpgid(pid) == leader
        }
        os.kill(leader, signal.SIGKILL)  # the leader only, as the OOM killer does
        returncode = process.wait(timeout=SUBPROCESS_TIMEOUT_SECONDS)
    except BaseException:
        _stop_launcher(process)
        raise

    _wait_for(
        lambda: not _still_running(group_before) and _group_gone(leader),
        SWEEP_GRACE_SECONDS,
    )
    survivors = _still_running(group_before)
    group_gone = _group_gone(leader)
    for pid in survivors:  # the test reports them; they must not outlive it
        os.kill(pid, signal.SIGKILL)
    lock_free = True
    try:
        with holding_the_run(run.run_dir / "launch" / "lock"):
            pass
    except ConfigError:
        lock_free = False
    return LeaderKilled(
        returncode=returncode,
        output=output.read_text("utf-8", "replace"),
        group_before=group_before,
        survivors=survivors,
        group_gone=group_gone,
        lock_free=lock_free,
        manifest_left=(run.run_dir / "manifest.json").exists(),
        after=_state(run.run_dir, _ok(_launch(config, run.run_dir))),
    )


@pytest.mark.skipif(sys.platform != "linux", reason="reads /proc")
def test_a_killed_eole_leader_fails_the_launch_and_takes_its_whole_group_with_it(
    leader_killed: LeaderKilled,
) -> None:
    assert leader_killed.returncode == UPSTREAM_EXIT_CODE, leader_killed.output
    assert "eole train failed: killed by signal 9" in leader_killed.output
    # Besides the leader: the data-loader workers and the resource tracker.
    assert len(leader_killed.group_before) >= 2
    assert leader_killed.survivors == []
    assert leader_killed.group_gone
    assert leader_killed.lock_free
    assert not leader_killed.manifest_left


@pytest.mark.skipif(sys.platform != "linux", reason="reads /proc")
def test_the_launch_after_a_killed_eole_leader_finishes_the_run(
    leader_killed: LeaderKilled,
) -> None:
    after = leader_killed.after

    assert after.facts["status"] == "completed"
    assert after.facts["steps"] == str(FINAL_STEP)
    assert after.facts["attempt"] == "2"
    assert after.curve.non_finite_step is None
