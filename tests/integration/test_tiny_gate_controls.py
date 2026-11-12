"""The D15 STOP gate, the controls (R7 to R12): every threshold of the pipeline FAILS when its precondition is
broken, and every refusal and every interrupt of the command behaves as the design says, through the real
`rinchen` binary and the real Eole, CTranslate2 and SentencePiece on the CPU with every device hidden. The
recipe and what it wrote (R1 to R3, R5, R6 and R13) are in `test_tiny_gate.py`, the determinism (R4) in
`test_tiny_gate_determinism.py` and the hygiene (R14) in `test_tiny_gate_hygiene.py`.

Every command starts through `tests/support/tiny_gate.py`, which refuses to start without the empty device list
and the offline switches, starts the command in a session of its own and bounds every wait. The model-level
controls reuse the finished run of the recipe (`tiny_run_dir`, read only: an export is copied before a byte of
it is flipped) and ONE control run of their own, the profile with the master plan's 400 steps; the processes of
a run are found by the marker of their environment and by the session they started in, and a run that was
interrupted or timed out must leave none.

    R7  `--config` of the profile with `train.steps: 400`: exit 4, verdict `fail`, `train_chrf` `fail` under 50.0,
        `placeholder_acc` `fail`, `script_purity` `pass` (an undertrained model still writes the right script: the
        purity threshold has a control of its own, R9), the report and the manifest written
    R8  one flipped byte of an export: `rinchen eval robustness --model` and `rinchen eval run --model` exit 2 and
        write nothing; a threshold over a missing value is `fail`
    R9  each of the other thresholds fails when its figure does, the figure measured by the real command and read
        by the pipeline's own reader: placeholders no training row holds (`placeholder_acc`), answers in the wrong
        script (`script_purity`), the checkpoint of another model (`export parity` exits 2: the parity stage fails),
        a parity whose mean |delta| is 2e-3 (`parity_delta`) and a hypotheses file whose texts are rotated by 7
        rows (`int8_chrf_drop`); the same replay over the unchanged report passes every threshold. Two things the
        real run showed about the brief's recipe: a traditional-script output asked of a model with no converter
        is REFUSED by the engine (exit 2, nothing written) even with the mask off, so it makes no figure, and the
        purity control is a hypotheses file of copied inputs instead; with the converter attached the same request
        is pure
    R10 `--budget-seconds 5`: exit 4 inside 30 s (the brief says 60), the stage in progress `fail` on a timeout that the budget set,
        every later stage skipped, the report and the manifest written, no process of the run left. The wall-time
        threshold is NOT the one that fails: each command gets at most what is left of the budget, so the run ends
        inside it (measured: `pass`)
    R11 SIGTERM (143), SIGINT (130) and SIGHUP (129) sent once the log of `train launch` exists and the trainer
        has run for 3 s, into a directory that holds the manifest and the documents of an earlier run: they are
        gone, no process of the run is left (the trainer and its two helpers included), the command ends within
        a second of the signal; SIGHUP at the moment the first `eole` shows (the vocabulary build) the same; the
        same output directory is then run to the end and writes the `report.json` of the recipe. FOUND BY THIS
        GATE: before `rinchen.tiny.graceful_runner` an interrupt killed `train launch` with SIGKILL, and the
        Eole that leads its own session kept running (the vocabulary build 5 to 15 s, `eole train` 30 s and
        more, after the command had ended with its code); a first version of this control signalled at the
        spawn of the vocabulary build and passed by the luck of a build that ended within its 10 s wait: the
        process table is now read at once after the command has ended and been reaped, with a grace of 1.5 s for a
        process that is being torn down (`left_after_the_end`), so an orphan that lives 2 s or more fails it
    R12 a manifest of another stage in the output directory (exit 3, the file untouched), a foreign entry (exit 2,
        untouched), `--profile full` (exit 2, nothing written) and `--device auto` (typer's own exit 2)

Measured on this machine (CPU, 20 cores shared with other work, Eole 0.6.4, CTranslate2 4.8.4, 2026-11-12; a time
moves by a third between runs, a count, a score or a digest never does):

    R7  400 steps: exit 4 after 102.2 s; `train_chrf` 16.83 (the least of five directions), `placeholder_acc`
           0.0420, `script_purity` 1.0; every stage `ok` (the spike measured 16.83 and 0.0420 too)
    R9  placeholders of the indices 10 and up: `placeholder_acc` 0.0; wrong script: `script_purity` 0.5 (600 of
           600 answers copied, the half whose source is not in the script of its target); the mongolian-output
           suite with the converter: 1.0; rotated hypotheses: chrF 21.09 against 99.96
    R10 exit 4 after 5.4 s; the `converter` stage `fail` (`timed out after 1 s`), the next stages skipped; the
           manifest's seconds add up to 4.46 and `wall_time` is `pass`
    R11 after the fix, with the machine busy with other work (the signal came 72 to 108 s into the run, 46 to
           50 s on a quiet one): three to five processes of the run alive at the signal (the command, `train
           launch`, the trainer and its two helpers); the command exits 0.5 to 1.0 s later with the code of the
           signal, 0.7 s after a SIGHUP at the spawn of the vocabulary build, nothing left alive; the rerun
           takes 148 s. Before the fix: `eole train` and its helpers alive 30 s and more after the exit
           (probe of the real command), the vocabulary build 5 to 15 s (4 of 4)

UNMEASURED: GitHub Actions, a 4-core runner, anything on a GPU. The toy ranks nothing: every figure here is the
figure of a templated grammar of 37 nouns.
"""

from __future__ import annotations

import json
import re
import signal
from collections.abc import Callable
from pathlib import Path
from typing import Any, Final

import pytest

from rinchen.manifest import store
from rinchen.manifest.model import Manifest
from rinchen.tiny.commands import read_parity
from rinchen.tiny.eval_readers import read_robustness_report, read_run_result
from rinchen.tiny.states import (
    SKIP_EARLIER_FAILED,
    StageRecord,
    fail_exit,
)
from rinchen.tiny.thresholds import ThresholdInputs
from tests.support.robust_gate import copy_export_with_a_flipped_byte
from tests.support.tiny_gate import (
    COMMAND_TIMEOUT_SECONDS,
    MAX_WALL_TIME_SECONDS,
    MIN_PLACEHOLDER_ACC,
    MIN_SCRIPT_PURITY,
    MIN_TRAIN_CHRF,
    RecipeRun,
    read_manifest,
    read_report,
    run_rinchen,
    run_tiny,
    threshold_of,
)
from tests.support.tiny_gate_controls import (
    ROTATION,
    answer_with_the_source,
    averaged_directory,
    converter_options,
    export_directory,
    judged,
    replayed,
    report_inputs,
    rotate_texts,
    stage_path,
    verdicts,
    with_changes,
    write_control_suites,
)
from tests.support.tiny_gate_runs import (
    INTERRUPTS,
    STALE_FILES,
    BudgetRun,
    ControlRun,
    Interrupted,
    interrupt_run,
    run_with_budget,
    run_with_profile,
)
from tests.support.tiny_profiles import control_profile

pytestmark = [pytest.mark.subprocess, pytest.mark.usefixtures("tiny_gpu_hidden")]

MASTER_PLAN_STEPS: Final = 400
THRESHOLD_IDS: Final = (
    "stages",
    "train_chrf",
    "parity_exact",
    "parity_delta",
    "int8_chrf_drop",
    "script_purity",
    "placeholder_acc",
    "wall_time",
)
# A run with a budget of five seconds ends by itself in about five seconds. The brief allows a minute; the bound
# here is 30 s because the converter stage alone takes 40 s: a command that waited for the child it stopped to
# end by itself would not be inside it.
BUDGET_SECONDS: Final = 5
BUDGET_RUN_BOUND: Final = 30
# A command that is sent a signal ends within a fraction of a second (0.2 to 0.4 s measured); the training
# that it stops would take 40 s more to end by itself.
EXIT_PROMPTLY_SECONDS: Final = 10
TIMEOUT_REASON: Final = re.compile(r"timed out after [0-9]+ s")
HYPOTHESES: Final = "hypotheses.jsonl"
MEMORISED_SUITE: Final = "tiny-memorised-eng"
INT8_SYSTEM: Final = "tiny-int8"
DIRECTION: Final = "eng_Latn-khk_Cyrl"
RESAMPLES: Final = "100"
BEAM: Final = "1"


def _empty_directory(root: Path, name: str) -> Path:
    directory = root / name
    directory.mkdir()
    return directory


def _tree(directory: Path) -> list[str]:
    """What a directory holds, as relative names (a refusal must leave it as it was)."""
    return sorted(
        path.relative_to(directory).as_posix() for path in directory.rglob("*")
    )


# =================================================================================
# R7: a model that learned nothing.


@pytest.fixture(scope="module")
def learned_nothing(tmp_path_factory: pytest.TempPathFactory) -> ControlRun:
    """The pipeline with the master plan's 400 steps (the spike measured the least chrF at 16.83)."""
    root = tmp_path_factory.mktemp("tiny_gate_400_steps")
    return run_with_profile(root, control_profile(root, train_steps=MASTER_PLAN_STEPS))


def test_a_run_of_the_master_plans_400_steps_exits_four_with_everything_written(
    learned_nothing: ControlRun, record_property: Any
) -> None:
    run = learned_nothing
    record_property("seconds", round(run.seconds, 1))

    assert run.exit_code == 4, run.stderr[-3000:]
    assert "verdict: fail" in run.stdout
    for name in ("report.json", "report.md", "manifest.json"):
        assert (run.out_dir / name).is_file(), name
    assert read_manifest(run.out_dir)["metrics"]["train.steps"] == MASTER_PLAN_STEPS


def test_the_run_of_400_steps_fails_the_chrf_and_the_placeholders_and_not_the_script(
    learned_nothing: ControlRun, record_property: Any
) -> None:
    report = read_report(learned_nothing.out_dir)
    chrf = threshold_of(report, "train_chrf")
    placeholders = threshold_of(report, "placeholder_acc")
    record_property("train_chrf", chrf["value"])
    record_property("placeholder_acc", placeholders["value"])

    assert report["verdict"] == "fail"
    assert chrf["verdict"] == "fail"
    assert chrf["value"] < MIN_TRAIN_CHRF
    assert placeholders["verdict"] == "fail"
    assert placeholders["value"] < MIN_PLACEHOLDER_ACC
    # An undertrained model still writes the right script: purity needs the control of R9.
    purity = threshold_of(report, "script_purity")
    assert purity["verdict"] == "pass"
    assert purity["value"] >= MIN_SCRIPT_PURITY


def test_the_stages_of_the_run_of_400_steps_all_ran_and_only_the_figures_failed(
    learned_nothing: ControlRun,
) -> None:
    report = read_report(learned_nothing.out_dir)

    assert [stage["state"] for stage in report["stages"]] == ["ok"] * 15
    assert threshold_of(report, "stages")["verdict"] == "pass"
    assert report["blocks"]["train"]["steps"] == MASTER_PLAN_STEPS


# =================================================================================
# R8: a damaged export.


@pytest.fixture(scope="module")
def damaged_export(
    tiny_run_dir: RecipeRun, tmp_path_factory: pytest.TempPathFactory
) -> Path:
    """A copy of the int8 export of the recipe's run with one byte of `model.bin` changed."""
    damaged = tmp_path_factory.mktemp("tiny_gate_damaged") / "int8"
    copy_export_with_a_flipped_byte(
        export_directory(tiny_run_dir.out_dir, "int8"), damaged
    )
    return damaged


def test_the_robustness_command_refuses_a_damaged_export_and_writes_nothing(
    tiny_run_dir: RecipeRun, damaged_export: Path, tmp_path: Path
) -> None:
    out_dir = tmp_path / "robustness"
    suite = stage_path(tiny_run_dir.out_dir, "robustness", "suites", "tiny-robust.json")

    run = run_rinchen(
        [
            "eval",
            "robustness",
            "--suite",
            suite,
            "--model",
            damaged_export,
            "--out-dir",
            out_dir,
            "--beam-size",
            BEAM,
            *converter_options(tiny_run_dir.out_dir),
        ],
        cwd=tmp_path,
    )

    assert run.returncode == 2, run.stderr[-2000:]
    assert run.stdout == []
    assert not out_dir.exists()


def test_the_evaluation_command_refuses_a_damaged_export_and_writes_nothing(
    tiny_run_dir: RecipeRun, damaged_export: Path, tmp_path: Path
) -> None:
    out_dir = tmp_path / "runs"
    suite = stage_path(
        tiny_run_dir.out_dir, "evaluate", "suites", f"{MEMORISED_SUITE}.json"
    )

    run = run_rinchen(
        [
            "eval",
            "run",
            "--suite",
            suite,
            "--model",
            damaged_export,
            "--system",
            INT8_SYSTEM,
            "--beam-size",
            BEAM,
            "--resamples",
            RESAMPLES,
            "--out-dir",
            out_dir,
        ],
        cwd=tmp_path,
    )

    assert run.returncode == 2, run.stderr[-2000:]
    assert run.stdout == []
    assert not out_dir.exists()


def test_every_threshold_over_a_missing_value_is_a_fail() -> None:
    nothing = ThresholdInputs(
        stages=[],
        memorised_chrf={},
        parity=None,
        robustness=None,
        elapsed_seconds=float("nan"),
    )

    results = judged(nothing)

    assert [result.id for result in results] == list(THRESHOLD_IDS)
    assert {result.id: result.verdict for result in results} == dict.fromkeys(
        THRESHOLD_IDS, "fail"
    )
    assert [result.value for result in results[1:]] == [None] * 7


# =================================================================================
# R9: every other threshold fails when its figure does.


def test_the_unchanged_report_replays_to_the_verdicts_it_has(
    tiny_run_dir: RecipeRun, tiny_report: dict[str, Any]
) -> None:
    results = replayed(tiny_run_dir.out_dir)

    assert verdicts(results) == dict.fromkeys(THRESHOLD_IDS, "pass")
    assert verdicts(results) == {
        item["id"]: item["verdict"] for item in tiny_report["thresholds"]
    }
    for result in results:
        if result.id != "wall_time":
            assert result.value == threshold_of(tiny_report, result.id)["value"]


@pytest.fixture(scope="module")
def control_suites(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path]:
    return write_control_suites(tmp_path_factory.mktemp("tiny_gate_control_suites"))


def _robustness_values(
    tiny_run_dir: RecipeRun, suite: Path, out_dir: Path, *options: str | Path
) -> dict[str, float | None]:
    """The six values `rinchen eval robustness` measures for the int8 export over `suite`, read by the
    pipeline's own reader."""
    run = run_rinchen(
        [
            "eval",
            "robustness",
            "--suite",
            suite,
            "--model",
            export_directory(tiny_run_dir.out_dir, "int8"),
            "--out-dir",
            out_dir,
            "--beam-size",
            BEAM,
            *options,
        ],
        cwd=out_dir.parent,
    )
    assert run.returncode == 0, run.stderr[-2000:]
    return read_robustness_report(out_dir / "report.json")["values"]


def test_placeholders_no_training_row_holds_make_the_placeholder_threshold_fail(
    tiny_run_dir: RecipeRun,
    control_suites: tuple[Path, Path],
    tmp_path: Path,
    record_property: Any,
) -> None:
    values = _robustness_values(
        tiny_run_dir,
        control_suites[0],
        tmp_path / "unseen",
        *converter_options(tiny_run_dir.out_dir),
    )
    accuracy = values["placeholder_acc"]
    record_property("placeholder_acc", accuracy)
    inputs = report_inputs(tiny_run_dir.out_dir)
    assert inputs.robustness is not None

    assert accuracy is not None
    assert accuracy < MIN_PLACEHOLDER_ACC
    judged_inputs = with_changes(
        inputs, robustness={**inputs.robustness, "placeholder_acc": accuracy}
    )
    results = judged(judged_inputs)
    assert verdicts(results)["placeholder_acc"] == "fail"
    assert [
        name for name, verdict in verdicts(results).items() if verdict == "fail"
    ] == ["placeholder_acc"]


def test_a_traditional_script_asked_of_a_model_with_no_converter_is_refused_even_with_no_mask(
    tiny_run_dir: RecipeRun,
    control_suites: tuple[Path, Path],
    tmp_path: Path,
) -> None:
    """The brief's control (the traditional script asked without a converter, the mask off) cannot make
    a figure: the engine refuses the request first, with exit 2 and nothing written."""
    out_dir = tmp_path / "mong_output"

    run = run_rinchen(
        [
            "eval",
            "robustness",
            "--suite",
            control_suites[1],
            "--model",
            export_directory(tiny_run_dir.out_dir, "int8"),
            "--out-dir",
            out_dir,
            "--beam-size",
            BEAM,
            "--script-mask",
            "off",
        ],
        cwd=tmp_path,
    )

    assert run.returncode == 2, run.stderr[-2000:]
    assert run.stdout == []
    assert "script converter" in run.stderr
    assert not out_dir.exists()


def test_the_same_request_with_the_converter_attached_is_pure_and_passes_the_threshold(
    tiny_run_dir: RecipeRun,
    control_suites: tuple[Path, Path],
    tmp_path: Path,
    record_property: Any,
) -> None:
    values = _robustness_values(
        tiny_run_dir,
        control_suites[1],
        tmp_path / "mong_output",
        *converter_options(tiny_run_dir.out_dir),
    )
    purity = values["script_purity"]
    record_property("script_purity_with_the_converter", purity)

    assert purity is not None
    assert purity >= MIN_SCRIPT_PURITY


def test_answers_in_the_wrong_script_make_the_purity_threshold_fail(
    tiny_run_dir: RecipeRun, tmp_path: Path, record_property: Any
) -> None:
    """A system that copies its input writes the wrong script for every row whose source is not in the
    script of its target: the robustness command scores such a file for the run's own suite, and the
    pipeline's reader and threshold judge the figure."""
    suite = stage_path(tiny_run_dir.out_dir, "robustness", "suites", "tiny-robust.json")
    copied = tmp_path / HYPOTHESES
    changed = answer_with_the_source(
        suite,
        stage_path(tiny_run_dir.out_dir, "robustness", HYPOTHESES),
        copied,
        section="script",
    )
    out_dir = tmp_path / "copied"

    run = run_rinchen(
        [
            "eval",
            "robustness",
            "--suite",
            suite,
            "--hypotheses",
            copied,
            "--out-dir",
            out_dir,
        ],
        cwd=tmp_path,
    )

    assert run.returncode == 0, run.stderr[-2000:]
    purity = read_robustness_report(out_dir / "report.json")["values"]["script_purity"]
    record_property("script_purity", purity)
    record_property("rows_copied", changed)
    inputs = report_inputs(tiny_run_dir.out_dir)
    assert inputs.robustness is not None

    assert changed == 600
    assert purity is not None
    assert purity < MIN_SCRIPT_PURITY
    results = judged(
        with_changes(inputs, robustness={**inputs.robustness, "script_purity": purity})
    )
    assert [
        name for name, verdict in verdicts(results).items() if verdict == "fail"
    ] == ["script_purity"]


def test_the_checkpoint_of_another_model_is_refused_and_the_parity_stage_would_fail(
    tiny_run_dir: RecipeRun, learned_nothing: ControlRun, tmp_path: Path
) -> None:
    out_dir = tmp_path / "parity"

    run = run_rinchen(
        [
            "export",
            "parity",
            "--checkpoint",
            averaged_directory(learned_nothing.out_dir),
            "--export",
            export_directory(tiny_run_dir.out_dir, "float32"),
            "--corpus",
            stage_path(tiny_run_dir.out_dir, "filter", "train.parquet"),
            "--out-dir",
            out_dir,
            "--max-lines",
            "50",
            "--threads",
            "2",
            "--timeout",
            "120",
        ],
        cwd=tmp_path,
    )

    assert run.returncode == 2, run.stderr[-2000:]
    inputs = report_inputs(tiny_run_dir.out_dir)
    records = [
        StageRecord(record.id, "fail", fail_exit(run.returncode))
        if record.id == "parity"
        else record
        for record in inputs.stages
    ]
    results = judged(with_changes(inputs, stages=records))
    assert verdicts(results)["stages"] == "fail"


def test_a_parity_with_a_mean_delta_of_two_thousandths_makes_the_delta_threshold_fail(
    tiny_run_dir: RecipeRun, tmp_path: Path
) -> None:
    parity_file = stage_path(tiny_run_dir.out_dir, "parity", "float32", "parity.json")
    document = json.loads(parity_file.read_text(encoding="utf-8"))
    document["mean_abs_delta"] = 2e-3
    doctored = tmp_path / "parity.json"
    doctored.write_text(json.dumps(document), encoding="utf-8")
    figures = read_parity(doctored)
    inputs = report_inputs(tiny_run_dir.out_dir)

    results = judged(
        with_changes(
            inputs,
            parity={
                "exact_share": figures["exact_share"],
                "mean_abs_delta": figures["mean_abs_delta"],
            },
        )
    )

    assert [
        name for name, verdict in verdicts(results).items() if verdict == "fail"
    ] == ["parity_delta"]


def test_hypotheses_with_their_texts_rotated_make_the_quantisation_threshold_fail(
    tiny_run_dir: RecipeRun, tmp_path: Path, record_property: Any
) -> None:
    run_dir = stage_path(
        tiny_run_dir.out_dir, "evaluate", "runs", MEMORISED_SUITE, INT8_SYSTEM
    )
    suite = stage_path(
        tiny_run_dir.out_dir, "evaluate", "suites", f"{MEMORISED_SUITE}.json"
    )
    rotated = tmp_path / HYPOTHESES
    rotate_texts(run_dir / HYPOTHESES, rotated, ROTATION)
    out_dir = tmp_path / "runs"

    run = run_rinchen(
        [
            "eval",
            "run",
            "--suite",
            suite,
            "--hypotheses",
            rotated,
            "--system",
            INT8_SYSTEM,
            "--resamples",
            RESAMPLES,
            "--out-dir",
            out_dir,
        ],
        cwd=tmp_path,
    )

    assert run.returncode == 0, run.stderr[-2000:]
    result = read_run_result(
        out_dir / MEMORISED_SUITE / INT8_SYSTEM / DIRECTION / "result.json",
        suite=MEMORISED_SUITE,
        system=INT8_SYSTEM,
        direction=DIRECTION,
    )
    chrf = result["metrics"]["chrf"]
    record_property("rotated_chrf", chrf)
    inputs = report_inputs(tiny_run_dir.out_dir)
    table = {name: dict(systems) for name, systems in inputs.memorised_chrf.items()}
    assert table[DIRECTION]["int8"] > chrf
    table[DIRECTION]["int8"] = chrf

    results = judged(with_changes(inputs, memorised_chrf=table))

    drop = {result.id: result for result in results}["int8_chrf_drop"]
    assert drop.verdict == "fail"
    assert drop.value is not None and drop.value > 1.0
    assert DIRECTION in drop.detail
    # The rotated figure is the int8 figure of the least direction too, so `train_chrf` fails with it.
    failing = [name for name, verdict in verdicts(results).items() if verdict == "fail"]
    assert failing == ["train_chrf", "int8_chrf_drop"]


# =================================================================================
# R10: the wall-time budget.


@pytest.fixture(scope="module")
def budget_run(tmp_path_factory: pytest.TempPathFactory) -> BudgetRun:
    """The command with a budget of five seconds, and what of it is left alive after it ended."""
    return run_with_budget(tmp_path_factory.mktemp("tiny_gate_budget"), BUDGET_SECONDS)


def test_a_budget_of_five_seconds_ends_the_run_with_exit_four_and_everything_written(
    budget_run: BudgetRun, record_property: Any
) -> None:
    run = budget_run.run
    record_property("seconds", round(run.seconds, 1))

    assert run.exit_code == 4, run.stderr[-2000:]
    assert run.seconds < BUDGET_RUN_BOUND
    assert "verdict: fail" in run.stdout
    for name in ("report.json", "report.md", "manifest.json"):
        assert (run.out_dir / name).is_file(), name


def test_the_stage_in_progress_fails_and_every_later_stage_is_skipped(
    budget_run: BudgetRun, record_property: Any
) -> None:
    report = read_report(budget_run.run.out_dir)
    stages = report["stages"]
    states = [stage["state"] for stage in stages]
    first_failure = states.index("fail")
    reason = stages[first_failure]["reason"]
    record_property("failed_stage", stages[first_failure]["id"])
    record_property("failed_reason", reason)
    record_property("wall_time_verdict", threshold_of(report, "wall_time")["verdict"])
    seconds = sum(
        value
        for name, value in read_manifest(budget_run.run.out_dir)["metrics"].items()
        if name.startswith("seconds.")
    )
    record_property("seconds_in_the_manifest", round(seconds, 2))

    assert states.count("fail") == 1
    assert all(state == "ok" for state in states[:first_failure])
    # The stage that was running when the budget ran out was killed on a timeout that the budget set: no
    # command is given more than what is left of it, so the run ends inside (or hardly past) the budget and
    # the wall-time threshold does not tell the overrun: the failed stage does.
    assert TIMEOUT_REASON.fullmatch(reason), reason
    assert int(reason.split()[3]) <= BUDGET_SECONDS
    for stage in stages[first_failure + 1 :]:
        assert (stage["state"], stage["reason"]) == ("skipped", SKIP_EARLIER_FAILED)
    assert threshold_of(report, "stages")["verdict"] == "fail"
    assert report["verdict"] == "fail"
    assert (
        read_manifest(budget_run.run.out_dir)["status"][stages[first_failure]["id"]]
        == "fail"
    )


def test_a_run_that_ran_out_of_time_leaves_no_process_behind(
    budget_run: BudgetRun,
) -> None:
    assert budget_run.left == []


# =================================================================================
# R11: interrupts, and the rerun after a kill.


@pytest.fixture(scope="module")
def interrupted(
    tmp_path_factory: pytest.TempPathFactory, tiny_run_dir: RecipeRun
) -> Callable[..., Interrupted]:
    """An interrupted run for a signal (during the training, or at the spawn of the first `eole`), made the
    first time a test asks for it, in a directory of its own."""
    made: dict[tuple[str, bool], Interrupted] = {}

    def run(number: signal.Signals, *, at_the_spawn: bool = False) -> Interrupted:
        key = (number.name, at_the_spawn)
        if key not in made:
            where = "spawn" if at_the_spawn else "training"
            made[key] = interrupt_run(
                tmp_path_factory.mktemp(f"tiny_gate_{number.name}_{where}"),
                number,
                tiny_run_dir.out_dir,
                at_the_spawn=at_the_spawn,
            )
        return made[key]

    return run


def test_a_signal_the_moment_the_vocabulary_build_starts_leaves_nothing_of_the_run_alive(
    interrupted: Callable[..., Interrupted], record_property: Any
) -> None:
    """The first real run of this gate sent the signal as soon as `eole` showed and found the vocabulary
    build running for 5 to 15 s after the command had ended: an interrupt killed `train launch` with
    SIGKILL, which cannot stop the Eole that leads a session of its own."""
    run = interrupted(signal.SIGHUP, at_the_spawn=True)
    record_property("seconds_after_the_signal", round(run.seconds_after_the_signal, 1))

    assert run.exit_code == 129
    assert run.seconds_after_the_signal < EXIT_PROMPTLY_SECONDS
    assert run.trainers >= 1
    assert run.left == []
    for name in STALE_FILES:
        assert not (run.out_dir / name).exists(), name


@pytest.mark.parametrize(
    ("number", "exit_code"), INTERRUPTS, ids=[number.name for number, _ in INTERRUPTS]
)
def test_a_signal_during_the_training_ends_the_command_with_its_code_and_no_manifest(
    interrupted: Callable[..., Interrupted],
    number: signal.Signals,
    exit_code: int,
    record_property: Any,
) -> None:
    run = interrupted(number)
    record_property(
        "seconds_to_the_signal", round(run.seconds - run.seconds_after_the_signal, 1)
    )
    record_property("seconds_after_the_signal", round(run.seconds_after_the_signal, 1))
    record_property("processes_at_the_signal", run.at_the_signal)

    assert run.exit_code == exit_code
    # It did not wait for the training it was stopping to end by itself.
    assert run.seconds_after_the_signal < EXIT_PROMPTLY_SECONDS
    # The old manifest and documents went first, and nothing of this run took their place.
    for name in STALE_FILES:
        assert not (run.out_dir / name).exists(), name
    # The training was running: the command, `train launch` and the trainer were alive.
    assert run.trainers >= 1
    assert run.at_the_signal >= 3


@pytest.mark.parametrize(
    "number", [number for number, _ in INTERRUPTS], ids=lambda n: n.name
)
def test_an_interrupted_command_leaves_no_process_of_its_own_alive(
    interrupted: Callable[..., Interrupted], number: signal.Signals
) -> None:
    assert interrupted(number).left == []


def test_the_directory_of_a_killed_run_is_run_to_the_end_by_a_rerun_from_nothing(
    interrupted: Callable[..., Interrupted],
    tiny_run_dir: RecipeRun,
    record_property: Any,
) -> None:
    killed = interrupted(signal.SIGTERM)
    assert (killed.out_dir / "07_train").is_dir()  # what the killed run left
    cwd = killed.out_dir.parent / "cwd"

    run = run_tiny(["--out-dir", killed.out_dir], cwd=cwd)
    record_property("seconds", round(run.seconds, 1))

    assert run.returncode == 0, run.stderr[-3000:]
    assert run.seconds <= MAX_WALL_TIME_SECONDS
    assert (killed.out_dir / "report.json").read_bytes() == (
        tiny_run_dir.out_dir / "report.json"
    ).read_bytes()
    assert (killed.out_dir / "manifest.json").is_file()


# =================================================================================
# R12: refusals.


def _refusal(tmp_path: Path, args: list[str | Path]) -> tuple[int, list[str], str]:
    cwd = _empty_directory(tmp_path, "cwd")
    run = run_tiny(args, cwd=cwd, timeout=COMMAND_TIMEOUT_SECONDS)
    assert list(cwd.iterdir()) == [], "a refusal wrote into the working directory"
    return run.returncode, run.stdout, run.stderr


def test_the_manifest_of_another_stage_is_refused_with_exit_three_and_left_alone(
    tmp_path: Path,
) -> None:
    out_dir = _empty_directory(tmp_path, "out")
    store.write(
        out_dir,
        Manifest(
            stage="eval_robustness",
            created="2026-11-12T00:00:00+00:00",
            code_version="abc1234",
            config_sha256="0" * 64,
            seed=0,
        ),
    )
    before = (out_dir / "manifest.json").read_bytes()

    code, stdout, _ = _refusal(tmp_path, ["--out-dir", out_dir])

    assert code == 3
    assert stdout == []
    assert (out_dir / "manifest.json").read_bytes() == before
    assert _tree(out_dir) == ["manifest.json"]


def test_a_foreign_entry_is_refused_with_exit_two_and_left_alone(
    tmp_path: Path,
) -> None:
    out_dir = _empty_directory(tmp_path, "out")
    (out_dir / "stray.txt").write_text("not the command's\n", encoding="utf-8")

    code, stdout, stderr = _refusal(tmp_path, ["--out-dir", out_dir])

    assert code == 2
    assert stdout == []
    assert "stray.txt" not in stderr
    assert _tree(out_dir) == ["stray.txt"]
    assert (out_dir / "stray.txt").read_text(encoding="utf-8") == "not the command's\n"


def test_the_full_profile_is_refused_with_exit_two_and_nothing_is_written(
    tmp_path: Path,
) -> None:
    out_dir = tmp_path / "out"

    code, stdout, stderr = _refusal(
        tmp_path, ["--profile", "full", "--out-dir", out_dir]
    )

    assert code == 2
    assert stdout == []
    assert "full" in stderr
    assert not out_dir.exists()


def test_a_device_option_does_not_exist_and_typer_refuses_it_with_exit_two(
    tmp_path: Path,
) -> None:
    out_dir = tmp_path / "out"

    code, stdout, _ = _refusal(
        tmp_path, ["--device", "au" + "to", "--out-dir", out_dir]
    )

    assert code == 2
    assert stdout == []
    assert not out_dir.exists()
