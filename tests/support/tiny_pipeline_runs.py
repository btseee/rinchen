"""Plans and stage functions for the tests that run the pipeline of `rinchen tiny run` (D15 Task 7).

`make_plan` is a `PipelinePlan` over the shipped profile and the committed fixtures, `fakes` the fifteen
fake stage functions of `tests/support/tiny_cli.py`, and `real_functions` the real ones with two stand-ins:
the environment stage (it compares this machine with the lock, which is not what these tests are about) and
the service stage (it loads a real export, which the recording runner cannot make). Run over the runner
that plays every command (`tiny_run_eval.EvalRunner`) they start the commands a real run starts, and write
the files the stages of Tasks 3 to 5 write, in the paths the command lists in the manifest.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, Final

from rinchen.config.loader import REPO_ROOT
from rinchen.tiny.context import StageContext, StageResult
from rinchen.tiny.pipeline import PipelinePlan, stage_functions
from tests.support.tiny_cli import FakeClock, Outcome, Recording, fake_table
from tests.support.tiny_run import RecordingRunner
from tests.support.tiny_stage_inputs import shipped_config, shipped_fixtures

SEED: Final = 1234
LOCK: Final = b"version = 1\n"
StageFunction = Callable[[StageContext], StageResult]


def make_plan(
    directory: Path, *, budget: int = 1200, runner: Any = None, **changes: Any
) -> PipelinePlan:
    """A plan over the shipped profile writing into `directory`; `changes` replace fields."""
    plan = PipelinePlan(
        config=shipped_config(),
        fixtures=shipped_fixtures(),
        out_dir=directory,
        runner=runner if runner is not None else RecordingRunner(),
        qe_fake=True,
        threads=2,
        budget_seconds=budget,
        seed=SEED,
        lock_bytes=LOCK,
    )
    return dataclasses.replace(plan, **changes)


def fakes(
    outcomes: Mapping[str, Outcome] | None = None,
    clock: FakeClock | None = None,
    recording: Recording | None = None,
) -> dict[str, StageFunction]:
    """The fifteen fake stage functions (`tests/support/tiny_cli.py`)."""
    return fake_table(outcomes or {}, clock or FakeClock(), recording or Recording())(
        SEED, LOCK
    )


def real_table(seed: int, lock_bytes: bytes) -> dict[str, StageFunction]:
    """The real stage functions bound to `seed` and `lock_bytes`, but for the environment and the
    service, which are the fakes."""
    table = stage_functions(seed, lock_bytes)
    stand_ins = fakes()
    table["environment"] = stand_ins["environment"]
    table["serve"] = stand_ins["serve"]
    return table


def real_functions() -> dict[str, StageFunction]:
    """`real_table` for the seed of the shipped profile and the lock of the repository."""
    return real_table(SEED, REPO_ROOT.joinpath("uv.lock").read_bytes())
