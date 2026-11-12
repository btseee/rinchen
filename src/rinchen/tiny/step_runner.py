"""Running the commands of one stage of the tiny pipeline (D15, spec P3, P4 and P12).

A stage that runs commands is a function of a `StageContext`; this module is what every such stage shares.
`open_stage` makes the directory of the stage, `StageCommands.run` runs one command through the runner of
the context and `run_stage` turns whatever the stage did into the `StageResult` the report lists.

A command runs in the directory of its stage, with the environment of `child_environment` (no secret,
the device hidden, the Hugging Face libraries offline), its output in `logs/<stage>-<step>.log` of that
directory, and for at most `command_timeout` seconds: the stage's timeout of the profile or what is left
of the wall-time budget, whichever is less, at least one second; the budget is read again before each
command. A command that ends any way but exit code 0 ends the stage as a failure with a reason of the
closed list (`exit code N`, `timed out after S s`, or the check that it could not be started), whatever
its log says. A stage reads its results from files: an input that is missing or is not what its command
writes is `DataError`, which `run_stage` turns into the failure `an input is missing or differs`.
"""

from __future__ import annotations

import copy
import hashlib
import math
import os
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, NoReturn

from rinchen.config.tiny import StageId
from rinchen.errors import DataError
from rinchen.maintain.commands import child_environment
from rinchen.tiny.context import StageContext, StageResult, stage_dir
from rinchen.tiny.states import (
    FAIL_INPUT,
    StageRecord,
    fail_check,
    fail_exit,
    fail_timeout,
)

NOT_STARTED: Final = fail_check("command_not_started")
LOGS: Final = "logs"
_CHUNK: Final = 1 << 20


def command_timeout(stage_seconds: int, remaining: float) -> int:
    """The seconds a command may run: `stage_seconds` or `remaining`, whichever is less, rounded down,
    and at least 1. A `remaining` that is not a number leaves `stage_seconds`."""
    bound = min(float(stage_seconds), remaining)
    return 1 if bound < 1 else math.floor(bound)


class StageStopped(Exception):
    """A stage that ends before its end: the record it came to and the facts it still has."""

    def __init__(
        self, record: StageRecord, facts: Mapping[str, Any] | None = None
    ) -> None:
        super().__init__("the stage was stopped")
        self.record = record
        self.facts = dict(facts or {})


@dataclass(frozen=True, slots=True)
class StageCommands:
    """The commands of one stage: its context, its id and its directory (absolute and made)."""

    ctx: StageContext
    stage_id: StageId
    directory: Path

    def log_path(self, step: str) -> Path:
        return self.directory / LOGS / f"{self.stage_id}-{step}.log"

    def timeout(self) -> int:
        """The bound of the next command, from the budget as it is now."""
        seconds = self.ctx.config.timeouts[self.stage_id]
        return command_timeout(seconds, self.ctx.remaining_seconds())

    def run_bounded(self, step: str, build: Callable[[int], Sequence[str]]) -> None:
        """Run the command `build(timeout)` makes, so that a command that is told its own bound is
        told the one the runner enforces.

        Raises:
            StageStopped: the command did not end with exit code 0.
        """
        timeout = self.timeout()
        result = self.ctx.runner.run(
            build(timeout),
            cwd=self.directory,
            env=child_environment(os.environ),
            log=self.log_path(step),
            timeout=timeout,
        )
        if result.timed_out:
            self._stop(fail_timeout(timeout))
        if not result.started or result.exit_code is None:
            self._stop(NOT_STARTED)
        if result.exit_code != 0:
            self._stop(fail_exit(result.exit_code))

    def run(self, step: str, argv: Sequence[str]) -> None:
        """Run `argv` (see `run_bounded`)."""
        self.run_bounded(step, lambda _timeout: argv)

    def _stop(self, reason: str) -> NoReturn:
        raise StageStopped(StageRecord(self.stage_id, "fail", reason))


def open_stage(ctx: StageContext, stage_id: StageId) -> StageCommands:
    """The commands of `stage_id`, with its directory under the absolute output directory made.

    Raises:
        ValueError: the stage writes no directory.
    """
    directory = stage_dir(ctx.out_dir.resolve(), stage_id)
    if directory is None:
        raise ValueError("the stage writes no directory")
    directory.mkdir(parents=True, exist_ok=True)
    return StageCommands(ctx, stage_id, directory)


def path_in(ctx: StageContext, stage_id: StageId, *parts: str) -> Path:
    """A path under the directory of the stage `stage_id` (absolute, like every path a command gets)."""
    directory = stage_dir(ctx.out_dir.resolve(), stage_id)
    if directory is None:
        raise ValueError("the stage writes no directory")
    return directory.joinpath(*parts)


def run_stage(
    ctx: StageContext,
    stage_id: StageId,
    body: Callable[[StageCommands], Mapping[str, Any]],
) -> StageResult:
    """Run `body` for the stage: its facts make an `ok` record, a `StageStopped` is the record it
    carries and a `DataError` (an input that is missing or not what it should be) is the failure
    `an input is missing or differs`. Nothing else is caught."""
    stage = open_stage(ctx, stage_id)
    try:
        facts = dict(body(stage))
    except StageStopped as stopped:
        return StageResult(stopped.record, stopped.facts)
    except DataError:
        return StageResult(StageRecord(stage_id, "fail", FAIL_INPUT), {})
    return StageResult(StageRecord(stage_id, "ok", detail=copy.deepcopy(facts)), facts)


def read_text(path: Path) -> str:
    """The UTF-8 text of a file.

    Raises:
        DataError: the file is missing, cannot be read or is not UTF-8 (the message names the file).
    """
    try:
        return path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as error:
        raise DataError(
            f"{path.name}: the file is missing or cannot be read"
        ) from error


def file_digest(path: Path) -> str:
    """The SHA-256 of a file, in hex.

    Raises:
        DataError: the file is missing or cannot be read.
    """
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            while chunk := handle.read(_CHUNK):
                digest.update(chunk)
    except OSError as error:
        raise DataError(
            f"{path.name}: the file is missing or cannot be read"
        ) from error
    return digest.hexdigest()
