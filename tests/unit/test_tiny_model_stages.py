"""The CLI stages of the tiny pipeline over a recording fake runner: what every stage does with its
commands (D15, spec P2, P3, P12, brief G4.4). The facts of each stage are in
`test_tiny_model_stage_facts.py`.

The stages are `spm`, `converter`, `synth`, `prepare`, `train`, `average`, `export` and `parity`. No
process starts and no network is used (the `no_network` guard; `run_synth` only binds a listener on the
loopback, which the guard allows). `tests/support/tiny_run.py` plays each command: it checks the inputs
the real command would need and writes the files it writes, from the recordings of the spike, so a stage
that wires a path wrongly fails on it; the stages run over the real output of the corpus and filter
stages. The fake that connects to the listener, and the proof that the listener is gone, are in
`test_tiny_synth_stage.py`, which cannot use the guard.
"""

from __future__ import annotations

import dataclasses
import math
import os
import socket
from pathlib import Path
from typing import Any, Final

import pytest

from rinchen.maintain.commands import child_environment
from rinchen.tiny import step_runner
from rinchen.tiny.states import FAIL_INPUT, fail_check, fail_exit, is_closed_reason
from tests.support import tiny_run
from tests.support.no_network import no_network  # noqa: F401 - the fixture of the mark
from tests.support.tiny_help import check_argument_list
from tests.support.tiny_run import (
    COMMANDS_OF,
    MODEL_STAGES,
    RecordingRunner,
    at,
    changed_config,
    context,
    copy_inputs,
    ready_for,
    recorded_text,
    run_one,
    stage_function,
    tree,
)
from tests.support.tiny_stage_inputs import shipped_config

pytestmark = pytest.mark.usefixtures("no_network")

# The fixtures of `tests/support/tiny_run.py`, named here so that pytest finds them.
library_outputs = tiny_run.library_outputs
out_dir = tiny_run.out_dir

LOGS: Final = {
    "spm": ("spm-train.log", "spm-check.log"),
    "converter": (
        "converter-build.log",
        "converter-train.log",
        "converter-export.log",
        "converter-eval.log",
    ),
    "synth": ("synth-run.log",),
    "prepare": ("prepare-ablate.log",),
    "train": ("train-launch.log",),
    "average": ("average-average.log",),
    "export": ("export-float32.log", "export-int8.log"),
    "parity": ("parity-float32.log",),
}
ALL_KEYS: Final = [(stage, key) for stage in MODEL_STAGES for key in COMMANDS_OF[stage]]
KEY_IDS: Final = [f"{stage}:{key}" for stage, key in ALL_KEYS]
FACT_KEYS: Final = [pair for pair in ALL_KEYS if pair[1] != "export parity"]
FACT_IDS: Final = [
    ids
    for ids, (_, key) in zip(KEY_IDS, ALL_KEYS, strict=True)
    if key != "export parity"
]


def commands_up_to(stage: str, key: str) -> list[str]:
    keys = list(COMMANDS_OF[stage])
    return keys[: keys.index(key) + 1]


# ---- the budget of a command -------------------------------------------------------------------


@pytest.mark.parametrize(
    ("stage_seconds", "remaining", "expected"),
    [
        (600, 1e9, 600),
        (600, 1200.0, 600),
        (600, 600.0, 600),
        (600, 599.9, 599),
        (600, 100.0, 100),
        (30, 7.9, 7),
        (30, 2.0, 2),
        (30, 1.5, 1),
        (30, 1.0, 1),
        (30, 0.999, 1),
        (30, 0.4, 1),
        (30, 0.0, 1),
        (30, -50.0, 1),
        (30, -math.inf, 1),
        (30, math.inf, 30),
        (30, math.nan, 30),
        (1, 1e9, 1),
    ],
)
def test_a_command_gets_the_stage_timeout_or_what_is_left_and_at_least_one_second(
    stage_seconds: int, remaining: float, expected: int
) -> None:
    timeout = step_runner.command_timeout(stage_seconds, remaining)

    assert timeout == expected
    assert type(timeout) is int


# ---- the commands of each stage -----------------------------------------------------------------


@pytest.mark.parametrize("stage", MODEL_STAGES)
def test_each_stage_runs_its_commands_in_order_and_is_ok(
    out_dir: Path, stage: str
) -> None:
    result, runner = run_one(out_dir, stage)

    assert result.record.id == stage
    assert result.record.state == "ok"
    assert result.record.reason is None
    assert result.record.detail == result.facts
    assert runner.keys == list(COMMANDS_OF[stage])


def test_the_record_holds_its_own_copy_of_the_facts(out_dir: Path) -> None:
    result, _ = run_one(out_dir, "synth")

    result.facts["counts"]["selected"] = -1

    assert result.record.detail is not result.facts
    assert result.record.detail["counts"]["selected"] == 64


@pytest.mark.parametrize("stage", MODEL_STAGES)
def test_each_call_is_a_valid_call_of_the_real_command_with_absolute_paths(
    tmp_path: Path, library_outputs: Path, monkeypatch: pytest.MonkeyPatch, stage: str
) -> None:
    monkeypatch.chdir(tmp_path)
    relative = Path("out")
    copy_inputs(library_outputs, relative)

    _, runner = run_one(relative, stage)

    assert runner.calls
    for call in runner.calls:
        assert check_argument_list(call.argv, call.argv[3:5]) == [], call.key


@pytest.mark.parametrize("stage", MODEL_STAGES)
def test_each_command_runs_in_the_directory_of_its_stage_and_logs_there(
    out_dir: Path, stage: str
) -> None:
    _, runner = run_one(out_dir, stage)
    directory = at(out_dir, stage)

    assert [call.cwd for call in runner.calls] == [directory] * len(runner.calls)
    assert [call.log for call in runner.calls] == [
        directory / "logs" / n for n in LOGS[stage]
    ]
    assert all(call.log.is_file() for call in runner.calls)


@pytest.mark.parametrize("stage", MODEL_STAGES)
def test_a_stage_writes_only_into_its_own_directory_and_changes_nothing_that_is_there(
    out_dir: Path, stage: str
) -> None:
    ctx, _ = ready_for(out_dir, stage)
    before = tree(out_dir)

    stage_function(stage)(ctx)

    after = tree(out_dir)
    prefix = at(out_dir, stage).relative_to(out_dir).as_posix() + "/"
    assert set(after) - set(before)
    assert all(name.startswith(prefix) for name in set(after) - set(before))
    assert set(before) <= set(after)
    assert all(after[name] == data for name, data in before.items())


def test_the_three_stages_of_the_training_directory_write_distinct_files(
    out_dir: Path,
) -> None:
    seen: dict[str, set[str]] = {}
    before = set(tree(out_dir))
    ctx = context(out_dir)
    for stage in ("spm", "converter", "synth", "prepare", "train", "average"):
        stage_function(stage)(ctx)
        now = set(tree(out_dir))
        seen[stage] = now - before
        before = now

    assert seen["prepare"] and seen["train"] and seen["average"]
    assert not seen["prepare"] & seen["train"]
    assert not seen["train"] & seen["average"]
    assert not seen["prepare"] & seen["average"]
    for stage in ("prepare", "train", "average"):
        assert all(name.startswith("07_train/") for name in seen[stage])


@pytest.mark.parametrize("stage", MODEL_STAGES)
def test_a_child_gets_no_secret_and_a_hidden_device(
    out_dir: Path, monkeypatch: pytest.MonkeyPatch, stage: str
) -> None:
    for name in ("HF_TOKEN", "MY_SECRET", "DB_PASSWORD", "SERVICE_API_KEY"):
        monkeypatch.setenv(name, "value")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    monkeypatch.setenv("KEPT_VARIABLE", "kept")

    _, runner = run_one(out_dir, stage)

    expected = child_environment(os.environ)
    suffixes = ("_TOKEN", "_SECRET", "_PASSWORD", "_API_KEY")
    for call in runner.calls:
        assert dict(call.env) == expected
        assert call.env["CUDA_VISIBLE_DEVICES"] == ""
        assert call.env["HF_HUB_OFFLINE"] == "1"
        assert call.env["HF_DATASETS_OFFLINE"] == "1"
        assert call.env["KEPT_VARIABLE"] == "kept"
        assert not [name for name in call.env if name.endswith(suffixes)]


# ---- the timeout of every call --------------------------------------------------------------------


@pytest.mark.parametrize("stage", MODEL_STAGES)
@pytest.mark.parametrize(
    "remaining", [1e9, 1200.0, 100.0, 7.9, 1.0, 0.4, 0.0, -50.0, math.inf]
)
def test_every_call_gets_at_most_the_stage_timeout_and_what_is_left_and_at_least_one(
    out_dir: Path, stage: str, remaining: float
) -> None:
    ctx, runner = ready_for(out_dir, stage, remaining=lambda: remaining)

    result = stage_function(stage)(ctx)

    expected = max(1, math.floor(min(ctx.config.timeouts[stage], remaining)))
    assert result.record.state == "ok"
    assert [call.timeout for call in runner.calls] == [expected] * len(
        COMMANDS_OF[stage]
    )
    assert all(type(call.timeout) is int for call in runner.calls)


@pytest.mark.parametrize("stage", ["spm", "converter", "export"])
def test_the_budget_is_read_again_before_each_command(
    out_dir: Path, stage: str
) -> None:
    readings = iter([100.0, 40.0, 3.4, 0.2, 0.1, 0.1])
    seen: list[float] = []

    def clock() -> float:
        value = next(readings)
        seen.append(value)
        return value

    ctx, runner = ready_for(out_dir, stage, remaining=clock)
    stage_function(stage)(ctx)

    assert len(seen) == len(runner.calls) == len(COMMANDS_OF[stage])
    seconds = ctx.config.timeouts[stage]
    assert [call.timeout for call in runner.calls] == [
        max(1, math.floor(min(seconds, value))) for value in seen
    ]
    assert len({call.timeout for call in runner.calls}) == len(runner.calls)


@pytest.mark.parametrize("stage", MODEL_STAGES)
def test_the_stage_timeout_of_the_profile_is_the_bound(
    out_dir: Path, stage: str
) -> None:
    timeouts = dict(shipped_config().timeouts)
    timeouts[stage] = 5

    ctx, runner = ready_for(out_dir, stage, config=changed_config(timeouts=timeouts))
    stage_function(stage)(ctx)

    assert [call.timeout for call in runner.calls] == [5] * len(COMMANDS_OF[stage])


@pytest.mark.parametrize("stage", ["converter", "train", "parity"])
def test_the_bound_the_commands_are_given_is_the_bound_the_runner_enforces(
    out_dir: Path, stage: str
) -> None:
    ctx, runner = ready_for(out_dir, stage, remaining=lambda: 9.5)

    stage_function(stage)(ctx)

    bounded = [call for call in runner.calls if "--timeout" in call.argv]
    assert bounded
    for call in bounded:
        assert call.argv[call.argv.index("--timeout") + 1] == str(call.timeout) == "9"


@pytest.mark.parametrize("stage", ["converter", "prepare", "train", "parity"])
def test_the_threads_of_the_context_are_the_threads_of_each_command(
    out_dir: Path, stage: str
) -> None:
    ctx, runner = ready_for(out_dir, stage)

    stage_function(stage)(dataclasses.replace(ctx, threads=3))

    counted = [call for call in runner.calls if "--threads" in call.argv]
    assert counted
    for call in counted:
        assert call.argv[call.argv.index("--threads") + 1] == "3"


# ---- a command that fails ----------------------------------------------------------------------------


@pytest.mark.parametrize(("stage", "key"), ALL_KEYS, ids=KEY_IDS)
def test_a_command_that_exits_4_makes_the_stage_fail_with_that_exit_code(
    out_dir: Path, stage: str, key: str
) -> None:
    result, runner = run_one(out_dir, stage, fail={key: 4})

    assert result.record.state == "fail"
    assert result.record.reason == "exit code 4" == fail_exit(4)
    assert is_closed_reason(result.record.reason)
    assert result.record.id == stage
    assert runner.keys == commands_up_to(stage, key)


@pytest.mark.parametrize("code", [1, 2, 3, 130, 255, -9, -15])
def test_the_exit_code_is_the_one_the_command_gave(out_dir: Path, code: int) -> None:
    result, _ = run_one(out_dir, "average", fail={"train average": code})

    assert result.record.reason == f"exit code {code}"
    assert is_closed_reason(result.record.reason)


@pytest.mark.parametrize(("stage", "key"), ALL_KEYS, ids=KEY_IDS)
def test_a_command_that_fails_with_an_empty_log_still_fails_the_stage(
    out_dir: Path, stage: str, key: str
) -> None:
    result, _ = run_one(out_dir, stage, fail={key: 4}, logs={key: ""})

    assert (result.record.state, result.record.reason) == ("fail", "exit code 4")


@pytest.mark.parametrize("stage", MODEL_STAGES)
def test_a_command_that_exits_0_with_a_log_that_reads_like_a_failure_is_a_success(
    out_dir: Path, stage: str
) -> None:
    text = "error: failed\nexit code 4\nTraceback (most recent call last)\n"
    logs = {key: text for key in COMMANDS_OF[stage] if key != "convert-script eval"}

    result, _ = run_one(out_dir, stage, logs=logs)

    assert result.record.state == "ok"
    assert result.record.reason is None


@pytest.mark.parametrize(("stage", "key"), FACT_KEYS, ids=FACT_IDS)
def test_a_command_that_ends_in_failure_leaves_no_facts(
    out_dir: Path, stage: str, key: str
) -> None:
    result, _ = run_one(out_dir, stage, fail={key: 4})

    assert result.facts == {}
    assert result.record.detail == {}


@pytest.mark.parametrize(("stage", "key"), ALL_KEYS, ids=KEY_IDS)
def test_a_command_that_runs_past_its_bound_makes_the_stage_fail_with_that_bound(
    out_dir: Path, stage: str, key: str
) -> None:
    result, runner = run_one(out_dir, stage, time_out={key})

    assert result.record.state == "fail"
    assert result.record.reason == f"timed out after {runner.calls[-1].timeout} s"
    assert is_closed_reason(result.record.reason)
    assert runner.keys == commands_up_to(stage, key)


@pytest.mark.parametrize(("stage", "key"), ALL_KEYS, ids=KEY_IDS)
def test_a_command_that_could_not_start_makes_the_stage_fail_with_a_closed_check(
    out_dir: Path, stage: str, key: str
) -> None:
    result, runner = run_one(out_dir, stage, not_started={key})

    assert result.record.state == "fail"
    assert result.record.reason == fail_check("command_not_started")
    assert is_closed_reason(result.record.reason)
    assert runner.keys == commands_up_to(stage, key)


@pytest.mark.parametrize("stage", MODEL_STAGES)
def test_an_exception_of_the_runner_is_not_swallowed(out_dir: Path, stage: str) -> None:
    ctx, _ = ready_for(
        out_dir, stage, raises={COMMANDS_OF[stage][0]: RuntimeError("boom")}
    )

    with pytest.raises(RuntimeError, match="boom"):
        stage_function(stage)(ctx)


# ---- results come from files, never from printed text ---------------------------------------------


def facts_of(out: Path, **options: Any) -> dict[str, Any]:
    ctx = context(out, runner=RecordingRunner(**options))
    return {stage: stage_function(stage)(ctx).facts for stage in MODEL_STAGES}


def test_no_stage_reads_what_a_command_printed_except_the_converter_evaluation(
    tmp_path: Path, library_outputs: Path
) -> None:
    quiet, noisy = tmp_path / "quiet", tmp_path / "noisy"
    copy_inputs(library_outputs, quiet)
    copy_inputs(library_outputs, noisy)

    facts_quiet = facts_of(quiet)
    facts_noisy = facts_of(noisy, noisy=True)

    assert facts_noisy == facts_quiet
    assert all(facts_quiet.values())


@pytest.mark.parametrize(
    "text",
    [
        "",
        "not the five lines\n",
        "cer 0.0285\n",
        recorded_text("conv_eval.log") + "extra 1\n",
    ],
    ids=["empty", "other text", "one line", "six lines"],
)
def test_the_converter_evaluation_is_read_by_shape_and_a_log_that_is_not_five_lines_fails(
    out_dir: Path, text: str
) -> None:
    result, runner = run_one(out_dir, "converter", logs={"convert-script eval": text})

    assert result.record.state == "fail"
    assert result.record.reason == FAIL_INPUT
    assert result.facts == {}
    assert runner.keys == list(COMMANDS_OF["converter"])


def test_a_converter_log_longer_than_the_runner_keeps_is_not_read_in_part(
    out_dir: Path,
) -> None:
    text = recorded_text("conv_eval.log") + "x" * (70 * 1024)

    result, _ = run_one(out_dir, "converter", logs={"convert-script eval": text})

    assert result.record.reason == FAIL_INPUT


def test_a_parity_file_the_reader_refuses_fails_the_stage_on_its_input(
    out_dir: Path,
) -> None:
    result, runner = run_one(out_dir, "parity", parity={"passes": "yes"})

    assert (result.record.state, result.record.reason) == ("fail", FAIL_INPUT)
    assert result.facts == {}
    assert runner.keys == ["export parity"]


# ---- the one listener -----------------------------------------------------------------------------


def test_synth_binds_the_one_listener_on_the_loopback_and_no_other_stage_binds_anything(
    out_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bound: dict[str, list[Any]] = {}
    original = socket.socket.bind
    current = ["start"]

    def spy(self: socket.socket, address: Any) -> None:
        bound.setdefault(current[0], []).append(address)
        original(self, address)

    monkeypatch.setattr(socket.socket, "bind", spy)
    ctx = context(out_dir)
    for stage in MODEL_STAGES:
        current[0] = stage
        stage_function(stage)(ctx)

    assert list(bound) == ["synth"]
    assert bound["synth"] == [("127.0.0.1", 0)]
