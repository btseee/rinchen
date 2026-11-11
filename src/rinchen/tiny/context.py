"""What a stage of `rinchen tiny run` is handed and what it hands back, and the directories of the
output (D15, spec P2 and P11).

A stage gets a `StageContext` (the profile, the parsed fixtures, the output directory, the runner of
its commands, the stand-in switch, the threads and the clock of the wall-time budget) and returns a
`StageResult`: the record the report lists and the `facts` it observed, from which the blocks of the
report are built. The `facts` are numbers and closed identifiers read from files the stage's commands
wrote, never lines of another program's output.

The directories a run writes under `--out-dir` are a closed list of eleven: the stages that write
files use these, `prepare`, `train` and `average` share `07_train`, and `environment` and `serve`
write none. The names are numbered in the order of the stages, so a renamed or reordered one moves
every path of the report; a test pins them.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Final

from rinchen.config.tiny import STAGE_IDS, TinyConfig
from rinchen.maintain.commands import CommandRunner
from rinchen.tiny.fixtures import Fixtures
from rinchen.tiny.states import StageRecord

STAGE_DIRECTORIES: Final[Mapping[str, str]] = MappingProxyType(
    {
        "fetch": "01_fetch",
        "corpus": "02_corpus",
        "filter": "03_filter",
        "spm": "04_spm",
        "converter": "05_converter",
        "synth": "06_synth",
        "prepare": "07_train",
        "train": "07_train",
        "average": "07_train",
        "export": "08_export",
        "parity": "09_parity",
        "robustness": "10_robustness",
        "evaluate": "11_evaluate",
    }
)


@dataclass(frozen=True, slots=True)
class StageContext:
    """The inputs of every stage. `remaining_seconds` is the wall-time budget that is left, read
    before each stage and each command."""

    config: TinyConfig
    fixtures: Fixtures
    out_dir: Path
    runner: CommandRunner
    qe_fake: bool
    threads: int
    remaining_seconds: Callable[[], float]


@dataclass(frozen=True, slots=True)
class StageResult:
    """What a stage hands back: its record and the facts the report's blocks are built from."""

    record: StageRecord
    facts: Mapping[str, Any]


def stage_dir(out_dir: Path, stage_id: str) -> Path | None:
    """The directory of a stage under `out_dir`, or `None` for a stage that writes none.

    Raises:
        ValueError: `stage_id` is none of the fifteen stages (the message does not quote it).
    """
    if stage_id not in STAGE_IDS:
        raise ValueError("the stage is none of the fifteen stages")
    name = STAGE_DIRECTORIES.get(stage_id)
    return None if name is None else out_dir / name
