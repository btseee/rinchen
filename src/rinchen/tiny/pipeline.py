"""The pipeline of `rinchen tiny run`: the fifteen stages in order, each in its turn, until one fails or the
wall-time budget is spent (D15, spec P2, P4 and P12).

`stage_functions` is the one place the table of the stage functions is built, in the order of `STAGE_IDS`:
each is a function of the `StageContext` alone, the two kinds of datum a run adds (the seed, which the
stages that train or generate need, and the bytes of `uv.lock`, which the environment stage compares with
the machine) bound to the ones that take them. Nothing here branches on the id of a stage or on the name of
a profile: the order is the order of the ids, and what a stage does is the stage's own.

`run` calls each function inside `device_hidden()`, with a clock of the budget that falls as the injected
clock moves. The first stage that comes back `fail` ends the run, and so does a budget that is spent before
a stage starts (the stage is `fail` for `the wall-time budget is spent` and is not called); every stage after
it is recorded `skipped` for an earlier failure, with no facts. A stage that raises is recorded as a failure
whose record holds no text of the exception: a `RinchenError` is a failed input, anything else a failed
check named `stage_raised`, and the class of the exception, the frames and, for a `RinchenError` (its message
names a place and holds no text of a file), its message are in `logs/<stage>-raised.log` of the output
directory when that can be written. A result that is not the stage's own, and an `ok` record that leaves out
a part of its stage (the report lists every part, so one dropped silently would read as health), are the
same failed check. An interrupt, `SystemExit` and the exit a signal makes pass through.

The seconds of each stage (0.0 for a stage that did not run) and the elapsed time are the manifest's and the
verdict of the `wall_time` threshold: they are in no document the report holds.
"""

from __future__ import annotations

import contextlib
import shlex
import time
import traceback
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Final

from rinchen.config.tiny import STAGE_IDS, TinyConfig
from rinchen.errors import RinchenError
from rinchen.maintain.commands import CommandRunner, device_hidden
from rinchen.tiny.context import StageContext, StageResult
from rinchen.tiny.data_stages import (
    run_corpus,
    run_environment,
    run_fetch,
    run_filter,
)
from rinchen.tiny.dry_run import stage_commands
from rinchen.tiny.eval_stages import run_evaluate, run_robustness, run_serve
from rinchen.tiny.fixtures import Fixtures
from rinchen.tiny.model_stages import run_converter, run_spm, run_synth
from rinchen.tiny.model_stages_rest import (
    run_average,
    run_export,
    run_parity,
    run_prepare,
    run_train,
)
from rinchen.tiny.states import (
    FAIL_BUDGET,
    FAIL_INPUT,
    SKIP_EARLIER_FAILED,
    StageRecord,
    fail_check,
    unlisted_parts,
)
from rinchen.utils.part_file import replace_file

__all__ = [
    "PipelinePlan",
    "PipelineRun",
    "StageFunction",
    "plan_lines",
    "run",
    "stage_functions",
]

StageFunction = Callable[[StageContext], StageResult]
LOGS_DIRECTORY: Final = "logs"
RAISED_CHECK: Final = "stage_raised"


@dataclass(frozen=True, slots=True)
class PipelinePlan:
    """What a run needs: the profile and the parsed fixtures, the (absolute) output directory, the
    runner of the commands, the stand-in switch, the threads, the wall-time budget in seconds, the seed
    and the bytes of `uv.lock` that the stages bound to them are given."""

    config: TinyConfig
    fixtures: Fixtures
    out_dir: Path
    runner: CommandRunner
    qe_fake: bool
    threads: int
    budget_seconds: int
    seed: int
    lock_bytes: bytes


@dataclass(frozen=True, slots=True)
class PipelineRun:
    """What a run came to: the result of each of the fifteen stages in order, the seconds each took
    (0.0 for a stage that did not run) and the seconds from the start of the run to its end."""

    results: tuple[StageResult, ...]
    seconds: Mapping[str, float]
    elapsed: float


def stage_functions(seed: int, lock_bytes: bytes) -> dict[str, StageFunction]:
    """The fifteen stage functions by id, in the order of `STAGE_IDS`, each a function of the context
    alone. The service is checked over the export the run made: it is handed no engine."""
    return {
        "environment": partial(run_environment, lock_bytes=lock_bytes),
        "fetch": run_fetch,
        "corpus": run_corpus,
        "filter": run_filter,
        "spm": partial(run_spm, seed=seed),
        "converter": run_converter,
        "synth": partial(run_synth, seed=seed),
        "prepare": partial(run_prepare, seed=seed),
        "train": run_train,
        "average": run_average,
        "export": run_export,
        "parity": run_parity,
        "serve": run_serve,
        "robustness": run_robustness,
        "evaluate": run_evaluate,
    }


def _log_raised(out_dir: Path, stage_id: str, text: str) -> None:
    """Keep what is known of an exception of a stage in `logs/<stage>-raised.log`; a log that cannot be
    written is not worth hiding the failure for."""
    with contextlib.suppress(OSError):
        directory = out_dir / LOGS_DIRECTORY
        directory.mkdir(parents=True, exist_ok=True)
        replace_file(directory / f"{stage_id}-raised.log", text.encode("utf-8"))


def _described(error: BaseException) -> str:
    """The class of the exception and its frames; the message too for a `RinchenError`, whose messages
    name places and counts and never a text of a file."""
    head = type(error).__name__
    if isinstance(error, RinchenError):
        head = f"{head}: {error}"
    return head + "\n" + "".join(traceback.format_tb(error.__traceback__))


def _failed(stage_id: str, reason: str) -> StageResult:
    return StageResult(StageRecord(stage_id, "fail", reason), {})


def _called(stage_id: str, function: StageFunction, ctx: StageContext) -> StageResult:
    """The result of the stage's function, or the failure an exception of it comes to."""
    try:
        with device_hidden():
            result = function(ctx)
    except RinchenError as error:
        _log_raised(ctx.out_dir, stage_id, _described(error))
        return _failed(stage_id, FAIL_INPUT)
    except Exception as error:  # noqa: BLE001 -- any crash of a stage is a failed check, logged
        _log_raised(ctx.out_dir, stage_id, _described(error))
        return _failed(stage_id, fail_check(RAISED_CHECK))
    if not isinstance(result, StageResult) or result.record.id != stage_id:
        _log_raised(
            ctx.out_dir, stage_id, "the function did not return its own result\n"
        )
        return _failed(stage_id, fail_check(RAISED_CHECK))
    if result.record.state == "ok" and unlisted_parts(result.record):
        _log_raised(
            ctx.out_dir,
            stage_id,
            f"the function's ok result leaves out {len(unlisted_parts(result.record))} "
            "of the parts of its stage\n",
        )
        return _failed(stage_id, fail_check(RAISED_CHECK))
    return result


def run(
    plan: PipelinePlan,
    clock: Callable[[], float] = time.monotonic,
    *,
    functions: Mapping[str, StageFunction] | None = None,
) -> PipelineRun:
    """Run the stages of `plan` in order.

    `functions` replaces the table of `stage_functions` (the tests of the command and of the pipeline
    give fakes); it must hold exactly the fifteen stages. The budget is read before each stage, from
    `clock`, and again by whatever the stage asks of `StageContext.remaining_seconds`.

    Raises:
        ValueError: `functions` is not exactly the fifteen stages (nothing has run).
    """
    table = (
        stage_functions(plan.seed, plan.lock_bytes) if functions is None else functions
    )
    if set(table) != set(STAGE_IDS):
        raise ValueError("the stage functions are not exactly the fifteen stages")
    started = clock()

    def remaining() -> float:
        return plan.budget_seconds - (clock() - started)

    ctx = StageContext(
        config=plan.config,
        fixtures=plan.fixtures,
        out_dir=plan.out_dir,
        runner=plan.runner,
        qe_fake=plan.qe_fake,
        threads=plan.threads,
        remaining_seconds=remaining,
    )
    results: list[StageResult] = []
    seconds: dict[str, float] = {}
    ended = False
    for stage_id in STAGE_IDS:
        seconds[stage_id] = 0.0
        if ended:
            results.append(
                StageResult(StageRecord(stage_id, "skipped", SKIP_EARLIER_FAILED), {})
            )
            continue
        begun = clock()
        if plan.budget_seconds - (begun - started) <= 0:
            results.append(_failed(stage_id, FAIL_BUDGET))
            ended = True
            continue
        result = _called(stage_id, table[stage_id], ctx)
        seconds[stage_id] = clock() - begun
        results.append(result)
        ended = result.record.state == "fail"
    return PipelineRun(tuple(results), seconds, clock() - started)


def plan_lines(plan: PipelinePlan) -> list[str]:
    """The dry run: a line for each stage, `<id>: in process` or `<id>: N commands`, and under a stage
    with commands one indented line for each, the argument list as a shell would read it. Nothing is
    read, written or started."""
    lines = []
    for stage_id, commands in stage_commands(
        plan.config,
        plan.out_dir,
        threads=plan.threads,
        seed=plan.seed,
        qe_fake=plan.qe_fake,
    ).items():
        what = (
            f"{len(commands)} command{'' if len(commands) == 1 else 's'}"
            if commands
            else "in process"
        )
        lines.append(f"{stage_id}: {what}")
        lines.extend(f"  {shlex.join(argv)}" for argv in commands)
    return lines
