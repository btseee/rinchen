"""The pipeline of `rinchen tiny run` (D15 Task 7, brief G7.1, spec P2, P4 and P12): the fifteen stages in the
order of `STAGE_IDS`, each inside `device_hidden()`, the first failure ending the run, a spent budget as a
failure of the next stage, an exception of a stage as a failure whose record holds no text of it, and the
dry run that lists the commands.

The stage functions are fakes (`tests/support/tiny_cli.py`: each says what it does and for how many seconds
of a fake clock), except in the tests of the timeout of a child and of the dry run, which run the real stage
functions over the recording fake runners of Tasks 4 and 5 and compare what they started with what the plan
lists.
"""

from __future__ import annotations

import ast
import inspect
import itertools
import os
import shlex
import sys
import time
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import pytest

from rinchen.config.tiny import STAGE_IDS
from rinchen.errors import ConfigError, DataError
from rinchen.tiny import pipeline
from rinchen.tiny.context import StageContext, StageResult
from rinchen.tiny.pipeline import PipelineRun, plan_lines, run, stage_functions
from rinchen.tiny.states import (
    FAIL_BUDGET,
    FAIL_INPUT,
    SKIP_EARLIER_FAILED,
    StageRecord,
    fail_check,
    fail_exit,
)
from tests.support import tiny_run
from tests.support.no_network import no_network  # noqa: F401 - the fixture of the mark
from tests.support.tiny_cli import (
    DATA_ERROR_TEXT,
    SECRET,
    FakeClock,
    Outcome,
    Recording,
)
from tests.support.tiny_pipeline_runs import (
    LOCK,
    SEED,
    fakes,
    make_plan,
    real_functions,
)
from tests.support.tiny_report_inputs import golden_results
from tests.support.tiny_run import VARIANT, RecordingRunner, stage_function
from tests.support.tiny_run_eval import EvalRunner
from tests.support.tiny_stage_inputs import shipped_config, shipped_fixtures

pytestmark = pytest.mark.usefixtures("no_network")

# The fixtures of `tests/support/tiny_run.py`, named here so that pytest finds them.
library_outputs = tiny_run.library_outputs
out_dir = tiny_run.out_dir


def run_fakes(
    directory: Path,
    outcomes: Mapping[str, Outcome] | None = None,
    *,
    budget: int = 1200,
) -> tuple[PipelineRun, Recording, FakeClock]:
    clock = FakeClock()
    recording = Recording()
    functions = fakes(outcomes, clock, recording)
    result = run(make_plan(directory, budget=budget), clock, functions=functions)
    return result, recording, clock


# ---- the table ---------------------------------------------------------------------------------------------


def test_the_table_has_the_fifteen_stages_in_the_order_of_the_ids() -> None:
    assert tuple(stage_functions(SEED, LOCK)) == STAGE_IDS


def test_the_table_binds_the_seed_and_the_lock_where_a_stage_needs_them(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: list[tuple[str, tuple[Any, ...], dict[str, Any]]] = []

    def recorder(name: str) -> Callable[..., StageResult]:
        def record(*args: Any, **kwargs: Any) -> StageResult:
            seen.append((name, args[1:], kwargs))
            return golden_results()[0]

        return record

    names = ("environment", "spm", "synth", "prepare", "serve")
    for name in names:
        monkeypatch.setattr(pipeline, f"run_{name}", recorder(name))

    table = stage_functions(SEED, LOCK)
    for stage_id in names:
        table[stage_id](object())  # type: ignore[arg-type]

    bound = {name: (args, kwargs) for name, args, kwargs in seen}
    assert bound["environment"] == ((), {"lock_bytes": LOCK})
    assert bound["spm"] == ((), {"seed": SEED})
    assert bound["synth"] == ((), {"seed": SEED})
    assert bound["prepare"] == ((), {"seed": SEED})
    # The service is checked over the real export: no engine is handed to it.
    assert bound["serve"] == ((), {})


def test_the_pipeline_has_no_branch_on_the_name_of_a_stage() -> None:
    source = Path(pipeline.__file__).read_text(encoding="utf-8")
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Compare):
            operands = [node.left, *node.comparators]
            named = [
                operand
                for operand in operands
                if isinstance(operand, ast.Constant) and operand.value in STAGE_IDS
            ]
            assert not named, node.lineno
        assert not isinstance(node, ast.Match)


# ---- the order and the stop --------------------------------------------------------------------------------


def test_the_stages_run_in_order_and_each_is_called_once(tmp_path: Path) -> None:
    result, recording, _ = run_fakes(tmp_path)

    assert recording.stages == list(STAGE_IDS)
    assert [item.record.id for item in result.results] == list(STAGE_IDS)
    assert [item.record.state for item in result.results] == ["ok"] * 15
    assert isinstance(result, PipelineRun)


def test_the_context_carries_the_plan(tmp_path: Path) -> None:
    _, recording, _ = run_fakes(tmp_path, budget=900)

    ctx = recording.contexts[0]
    assert isinstance(ctx, StageContext)
    assert ctx.out_dir == tmp_path
    assert ctx.threads == 2
    assert ctx.qe_fake is True
    assert ctx.config == shipped_config()
    assert ctx.fixtures == shipped_fixtures()


@pytest.mark.parametrize("failing", list(STAGE_IDS))
def test_the_first_failure_ends_the_run_and_the_rest_are_skipped(
    failing: str, tmp_path: Path
) -> None:
    result, recording, _ = run_fakes(tmp_path, {failing: Outcome("fail", exit_code=4)})

    index = STAGE_IDS.index(failing)
    assert recording.stages == list(STAGE_IDS[: index + 1])
    states = [item.record.state for item in result.results]
    assert states == ["ok"] * index + ["fail"] + ["skipped"] * (14 - index)
    assert result.results[index].record.reason == fail_exit(4)
    for later in result.results[index + 1 :]:
        assert later.record.reason == SKIP_EARLIER_FAILED
        assert later.record.detail == {}
        assert later.facts == {}


def test_a_skip_of_a_stage_by_itself_does_not_end_the_run(tmp_path: Path) -> None:
    result, recording, _ = run_fakes(tmp_path, {"fetch": Outcome("skipped")})

    assert recording.stages == list(STAGE_IDS)
    assert result.results[1].record.state == "skipped"
    assert result.results[2].record.state == "ok"


def test_the_facts_of_a_stage_that_failed_are_kept(tmp_path: Path) -> None:
    functions = fakes()

    def parity(ctx: StageContext) -> StageResult:
        return StageResult(
            StageRecord("parity", "fail", fail_exit(4)), {"exact_share": 0.5}
        )

    functions["parity"] = parity
    result = run(make_plan(tmp_path), FakeClock(), functions=functions)

    assert result.results[11].facts == {"exact_share": 0.5}
    assert result.results[12].record.reason == SKIP_EARLIER_FAILED


# ---- an exception of a stage -------------------------------------------------------------------------------


def test_a_stage_that_raises_a_rinchen_error_is_an_input_failure_without_its_text(
    tmp_path: Path,
) -> None:
    result, recording, _ = run_fakes(tmp_path, {"fetch": Outcome("data_error")})

    record = result.results[1].record
    assert (record.state, record.reason) == ("fail", FAIL_INPUT)
    assert recording.stages == ["environment", "fetch"]
    assert DATA_ERROR_TEXT not in repr(result.results)


def test_a_stage_that_raises_anything_else_is_a_failed_check_without_its_text(
    tmp_path: Path,
) -> None:
    result, _, _ = run_fakes(tmp_path, {"corpus": Outcome("crash")})

    record = result.results[2].record
    assert (record.state, record.reason) == ("fail", fail_check("stage_raised"))
    assert SECRET not in repr(result.results)
    assert result.results[3].record.reason == SKIP_EARLIER_FAILED


@pytest.mark.parametrize("error", [DataError("x"), ConfigError("x")])
def test_every_rinchen_error_is_an_input_failure(
    error: Exception, tmp_path: Path
) -> None:
    functions = fakes()

    def raising(ctx: StageContext) -> StageResult:
        raise error

    functions["fetch"] = raising
    result = run(make_plan(tmp_path), FakeClock(), functions=functions)

    assert result.results[1].record.reason == FAIL_INPUT


def test_a_stage_that_raises_leaves_a_log_of_the_class_and_not_the_text_of_a_crash(
    tmp_path: Path,
) -> None:
    run_fakes(tmp_path, {"fetch": Outcome("data_error")})
    run_fakes(tmp_path / "second", {"corpus": Outcome("crash")})

    logged = (tmp_path / "logs" / "fetch-raised.log").read_text(encoding="utf-8")
    assert "DataError" in logged
    assert DATA_ERROR_TEXT in logged
    crashed = (tmp_path / "second" / "logs" / "corpus-raised.log").read_text(
        encoding="utf-8"
    )
    assert "RuntimeError" in crashed
    assert SECRET not in crashed
    assert 'File "' in crashed


def test_a_log_that_cannot_be_written_does_not_hide_the_failure(tmp_path: Path) -> None:
    blocked = tmp_path / "blocked"
    blocked.write_text("a file where the output directory would go", encoding="utf-8")

    result, _, _ = run_fakes(blocked, {"fetch": Outcome("crash")})

    assert result.results[1].record.reason == fail_check("stage_raised")


def test_a_stage_that_returns_another_stages_record_is_a_failed_check(
    tmp_path: Path,
) -> None:
    functions = fakes()
    functions["fetch"] = lambda ctx: golden_results()[0]

    result = run(make_plan(tmp_path), FakeClock(), functions=functions)

    assert result.results[1].record.reason == fail_check("stage_raised")


def test_a_stage_that_returns_nothing_is_a_failed_check(tmp_path: Path) -> None:
    functions = fakes()
    functions["fetch"] = lambda ctx: None  # type: ignore[return-value]

    result = run(make_plan(tmp_path), FakeClock(), functions=functions)

    assert result.results[1].record.reason == fail_check("stage_raised")


@pytest.mark.parametrize("signal_like", [KeyboardInterrupt(), SystemExit(143)])
def test_an_interrupt_and_an_exit_pass_through_untouched(
    signal_like: BaseException, tmp_path: Path
) -> None:
    functions = fakes()

    def interrupted(ctx: StageContext) -> StageResult:
        raise signal_like

    functions["spm"] = interrupted
    with pytest.raises(type(signal_like)) as raised:
        run(make_plan(tmp_path), FakeClock(), functions=functions)

    assert raised.value is signal_like


def test_the_functions_must_be_exactly_the_fifteen_stages(tmp_path: Path) -> None:
    short = fakes()
    del short["average"]
    with pytest.raises(ValueError, match="fifteen"):
        run(make_plan(tmp_path), FakeClock(), functions=short)

    more = fakes()
    more["extra"] = more["average"]
    with pytest.raises(ValueError, match="fifteen"):
        run(make_plan(tmp_path), FakeClock(), functions=more)


# ---- the clock and the budget ------------------------------------------------------------------------------


def test_the_remaining_budget_falls_with_the_clock(tmp_path: Path) -> None:
    clock = FakeClock()
    remaining: dict[str, float] = {}
    functions = fakes({stage: Outcome("ok", 10.0) for stage in STAGE_IDS}, clock)

    def watch(
        stage_id: str, function: Callable[[StageContext], StageResult]
    ) -> Callable[[StageContext], StageResult]:
        def watching(ctx: StageContext) -> StageResult:
            remaining[stage_id] = ctx.remaining_seconds()
            return function(ctx)

        return watching

    for stage_id, function in list(functions.items()):
        functions[stage_id] = watch(stage_id, function)

    run(make_plan(tmp_path, budget=1200), clock, functions=functions)

    assert remaining == {
        stage: 1200.0 - 10.0 * index for index, stage in enumerate(STAGE_IDS)
    }


def test_the_remaining_budget_is_read_afresh_inside_a_stage(tmp_path: Path) -> None:
    clock = FakeClock()
    functions = fakes(clock=clock)
    readings: list[float] = []

    def reads_twice(ctx: StageContext) -> StageResult:
        readings.append(ctx.remaining_seconds())
        clock.advance(7.0)
        readings.append(ctx.remaining_seconds())
        return golden_results()[0]

    functions["environment"] = reads_twice
    run(make_plan(tmp_path, budget=100), clock, functions=functions)

    assert readings == [100.0, 93.0]


def test_a_spent_budget_fails_the_next_stage_and_ends_the_run(tmp_path: Path) -> None:
    outcomes = {stage: Outcome("ok", 6.0) for stage in STAGE_IDS}

    result, recording, _ = run_fakes(tmp_path, outcomes, budget=10)

    assert recording.stages == ["environment", "fetch"]
    third = result.results[2].record
    assert (third.state, third.reason) == ("fail", FAIL_BUDGET)
    assert [item.record.reason for item in result.results[3:]] == [
        SKIP_EARLIER_FAILED
    ] * 12
    assert result.seconds["environment"] == 6.0
    assert result.seconds["fetch"] == 6.0
    assert result.seconds["corpus"] == 0.0


def test_a_budget_that_is_exactly_gone_is_spent_and_one_that_is_not_is_not(
    tmp_path: Path,
) -> None:
    outcomes = {stage: Outcome("ok", 6.0) for stage in STAGE_IDS}

    exact, exact_seen, _ = run_fakes(tmp_path / "exact", outcomes, budget=12)
    left, left_seen, _ = run_fakes(tmp_path / "left", outcomes, budget=13)

    assert exact_seen.stages == ["environment", "fetch"]
    assert exact.results[2].record.reason == FAIL_BUDGET
    assert left_seen.stages[2] == "corpus"
    assert left.results[2].record.state == "ok"


class Gone:
    """A clock that reads 0 at the start of a run and a day later every time after that."""

    def __init__(self) -> None:
        self.reads = 0

    def __call__(self) -> float:
        self.reads += 1
        return 0.0 if self.reads == 1 else 86_400.0


def test_a_run_whose_budget_is_gone_before_the_first_stage_starts_no_stage(
    tmp_path: Path,
) -> None:
    recording = Recording()
    functions = fakes(recording=recording)

    result = run(make_plan(tmp_path, budget=1), Gone(), functions=functions)

    assert recording.stages == []
    assert result.results[0].record.reason == FAIL_BUDGET
    assert result.results[0].record.state == "fail"
    assert all(item.record.reason == SKIP_EARLIER_FAILED for item in result.results[1:])


def test_the_seconds_and_the_elapsed_time_come_from_the_clock(tmp_path: Path) -> None:
    outcomes = {"fetch": Outcome("ok", 4.0), "filter": Outcome("ok", 2.5)}

    result, _, clock = run_fakes(tmp_path, outcomes)

    assert result.seconds["fetch"] == 4.0
    assert result.seconds["filter"] == 2.5
    assert result.seconds["environment"] == 1.0
    assert tuple(result.seconds) == STAGE_IDS
    assert result.elapsed == clock.now - 1000.0 == 4.0 + 2.5 + 13.0


def test_the_elapsed_time_includes_the_stage_that_failed(tmp_path: Path) -> None:
    result, _, clock = run_fakes(tmp_path, {"corpus": Outcome("fail", 9.0)})

    assert result.elapsed == 1.0 + 1.0 + 9.0
    assert result.seconds["corpus"] == 9.0
    assert result.seconds["filter"] == 0.0
    assert clock.now == 1011.0


def test_the_default_clock_is_the_monotonic_one_and_a_run_with_it_takes_time(
    tmp_path: Path,
) -> None:
    assert inspect.signature(run).parameters["clock"].default is time.monotonic

    result = run(make_plan(tmp_path), functions=fakes())

    assert result.elapsed >= 0.0
    assert [item.record.state for item in result.results] == ["ok"] * 15


# ---- the timeout of a child --------------------------------------------------------------------------------


def run_the_spm_stage(
    directory: Path, left: float, runner: RecordingRunner
) -> PipelineRun | None:
    """The real `spm` stage, as the pipeline calls it, with `left` seconds of the budget left when it
    starts (the fake stage before it uses the rest); the stage after it ends the run."""
    clock = FakeClock()
    outcomes = {stage: Outcome("ok", 0.0) for stage in ("fetch", "corpus", "filter")}
    outcomes["environment"] = Outcome("ok", 1000.0 - left)
    functions = fakes(outcomes, clock)
    functions["spm"] = stage_function("spm", SEED)

    def end_of_the_test(ctx: StageContext) -> StageResult:
        raise KeyboardInterrupt

    functions["converter"] = end_of_the_test
    try:
        return run(
            make_plan(directory, budget=1000, runner=runner), clock, functions=functions
        )
    except KeyboardInterrupt:
        return None


@pytest.mark.parametrize(
    ("left", "wanted"),
    [(900.0, 120), (7.5, 7), (0.5, 1)],
    ids=["plenty", "short", "almost-gone"],
)
def test_every_command_of_a_stage_gets_the_least_of_its_timeout_and_the_budget_left(
    left: float, wanted: int, out_dir: Path
) -> None:
    runner = RecordingRunner()

    run_the_spm_stage(out_dir, left, runner)

    assert runner.keys == ["spm train", "spm check"]
    assert [call.timeout for call in runner.calls] == [wanted, wanted]


def test_a_command_that_ends_badly_is_a_failed_stage_with_its_exit_code(
    out_dir: Path,
) -> None:
    runner = RecordingRunner(fail={"spm train": 2})

    result = run_the_spm_stage(out_dir, 900.0, runner)

    assert result is not None
    assert result.results[4].record.reason == fail_exit(2)
    assert result.results[5].record.reason == SKIP_EARLIER_FAILED


# ---- the device --------------------------------------------------------------------------------------------


def test_a_stage_runs_with_the_device_hidden_and_the_variable_is_put_back(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    seen: list[str | None] = []
    functions = fakes()

    def looking(ctx: StageContext) -> StageResult:
        seen.append(os.environ.get("CUDA_VISIBLE_DEVICES"))
        return golden_results()[0]

    functions["environment"] = looking
    run(make_plan(tmp_path), FakeClock(), functions=functions)

    assert seen == [""]
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "0"


def test_the_variable_is_put_back_after_an_exception_and_removed_when_there_was_none(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    seen: list[str | None] = []
    functions = fakes()

    def failing(ctx: StageContext) -> StageResult:
        seen.append(os.environ.get("CUDA_VISIBLE_DEVICES"))
        raise RuntimeError("a stage that fails")

    functions["environment"] = failing
    run(make_plan(tmp_path), FakeClock(), functions=functions)

    assert seen == [""]
    assert "CUDA_VISIBLE_DEVICES" not in os.environ


def test_the_variable_is_put_back_when_an_interrupt_leaves_the_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1")
    functions = fakes()

    def interrupted(ctx: StageContext) -> StageResult:
        raise KeyboardInterrupt

    functions["environment"] = interrupted
    with pytest.raises(KeyboardInterrupt):
        run(make_plan(tmp_path), FakeClock(), functions=functions)

    assert os.environ["CUDA_VISIBLE_DEVICES"] == "1"


# ---- the dry run -------------------------------------------------------------------------------------------

COMMANDS_PER_STAGE = {
    "spm": 2,
    "converter": 4,
    "synth": 1,
    "prepare": 1,
    "train": 1,
    "average": 1,
    "export": 2,
    "parity": 1,
    "robustness": 1,
    "evaluate": 10,
}


def stage_blocks(lines: list[str]) -> dict[str, list[str]]:
    """The command lines under each stage line (a command line is indented)."""
    blocks: dict[str, list[str]] = {}
    current = ""
    for line in lines:
        if line.startswith(" "):
            blocks[current].append(line.strip())
        else:
            current = line.split(":", 1)[0]
            blocks[current] = []
    return blocks


def test_the_plan_lists_the_fifteen_stages_in_order_and_the_commands_of_each(
    tmp_path: Path,
) -> None:
    blocks = stage_blocks(plan_lines(make_plan(tmp_path)))

    assert list(blocks) == list(STAGE_IDS)
    assert {
        stage: len(found) for stage, found in blocks.items() if found
    } == COMMANDS_PER_STAGE


def test_the_plan_without_the_stand_ins_has_one_command_less_in_the_evaluation(
    tmp_path: Path,
) -> None:
    blocks = stage_blocks(plan_lines(make_plan(tmp_path, qe_fake=False)))

    assert len(blocks["evaluate"]) == 9
    assert not any("metricx24_qe" in line for line in blocks["evaluate"])


def test_every_command_line_is_a_call_of_the_real_binary_with_no_forbidden_word(
    tmp_path: Path,
) -> None:
    forbidden = (
        "au" + "to",
        "cu" + "da",
        "--up" + "load",
        "--pub" + "lic",
        "fe" + "tch",
    )
    for stage, commands in stage_blocks(plan_lines(make_plan(tmp_path))).items():
        for command in commands:
            words = shlex.split(command)
            assert words[:3] == [sys.executable, "-m", "rinchen"], stage
            assert not [
                word
                for word in words
                if word.lower() in forbidden or word.startswith("--dev" + "ice")
            ], stage


def test_the_plan_writes_nothing_and_starts_nothing(tmp_path: Path) -> None:
    directory = tmp_path / "out"
    runner = RecordingRunner()

    plan_lines(make_plan(directory, runner=runner))

    assert not directory.exists()
    assert runner.calls == []


def test_the_plan_names_the_variant_the_preparation_will_make_as_a_placeholder(
    tmp_path: Path,
) -> None:
    blocks = stage_blocks(plan_lines(make_plan(tmp_path)))

    assert "<variant>" in blocks["train"][0]
    assert VARIANT not in " ".join(itertools.chain.from_iterable(blocks.values()))


def test_the_timeout_of_a_command_in_the_plan_is_the_timeout_of_its_stage(
    tmp_path: Path,
) -> None:
    blocks = stage_blocks(plan_lines(make_plan(tmp_path)))

    timeouts = shipped_config().timeouts
    assert shlex.split(blocks["train"][0])[-1] == str(timeouts["train"])
    assert shlex.split(blocks["parity"][0])[-1] == str(timeouts["parity"])


def normalised(argv: list[str]) -> list[str]:
    """The words of a command with the id of the variant that the run made as the plan writes it."""
    return [word.replace(VARIANT, "<variant>") for word in argv]


@pytest.mark.parametrize("qe_fake", [True, False], ids=["with-stand-ins", "without"])
def test_the_plan_is_what_a_run_starts_command_for_command(
    qe_fake: bool, tmp_path: Path
) -> None:
    """The drift guard: the real stage functions, over the runner that plays every command, start the
    commands the plan lists, in the order it lists them and with the arguments it lists."""
    runner = EvalRunner()
    plan = make_plan(tmp_path / "out", runner=runner, qe_fake=qe_fake)

    result = run(plan, FakeClock(), functions=real_functions())

    assert [item.record.state for item in result.results] == ["ok"] * 15
    started = [normalised(list(call.argv)) for call in runner.calls]
    listed = [
        normalised(shlex.split(command))
        for commands in stage_blocks(plan_lines(plan)).values()
        for command in commands
    ]
    assert started == listed
    assert len(started) == sum(COMMANDS_PER_STAGE.values()) - (not qe_fake)
