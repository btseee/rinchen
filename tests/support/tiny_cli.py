"""The doubles and helpers of the tests of `rinchen tiny run` (D15 Task 7).

The command has three seams, module-level names of `rinchen.cli.tiny` that the tests replace (the
precedent of `rinchen maintain check`): `make_runner` (the runner of every command of a stage),
`make_stage_functions` (the table of the fifteen stage functions, bound to the seed and the bytes of the
lock the command hashed) and `clock` (the monotonic clock of the wall-time budget). `install` replaces
them, and the code version of the manifest, with doubles:

- `Outcome` says what a fake stage does: `ok` (the stage's result of the golden run), `skipped` (an own
  skip with a closed reason), `fail` (a failed record `exit code N`), `data_error` and `crash` (the
  function raises, with a text the record must not hold), each after `seconds` of the fake clock;
- `FakeClock` is the clock they advance, so that the elapsed time and the budget are exact;
- `Recording` keeps what the fakes saw: the stages called in order, the seed and the lock bytes the table
  was bound to and the runner the table was given.

`invoke` runs the `rinchen` app through typer's runner. `snapshot` and `identity` read what a directory
holds, for the tests that look at what a refusal left alone.
"""

from __future__ import annotations

import json
import shutil
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import pytest
from click.testing import Result
from typer.testing import CliRunner

from rinchen.config.loader import REPO_ROOT
from rinchen.config.tiny import STAGE_IDS
from rinchen.errors import DataError
from rinchen.tiny.context import StageContext, StageResult
from rinchen.tiny.states import StageRecord, fail_exit
from tests.support.tiny_report_inputs import golden_results

CODE_VERSION: Final = "abc1234"
# The text of a crash: no file the command writes may hold it (only the class of the exception and its
# frames are logged). The message of a `DataError` names a place, and may be in the log of the stage and
# in nothing else.
SECRET: Final = "text-of-an-input-that-no-record-may-hold"
DATA_ERROR_TEXT: Final = "the fixture file differs from its pin"

_runner = CliRunner()
StageFunction = Callable[[StageContext], StageResult]


class FakeClock:
    """A clock that moves only when it is told to."""

    def __init__(self, start: float = 1000.0) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


@dataclass(frozen=True)
class Outcome:
    """What a fake stage does after `seconds` of the fake clock."""

    kind: str = "ok"
    seconds: float = 1.0
    reason: str = "an earlier stage failed"
    exit_code: int = 4


OK = Outcome()


class Recording:
    """What the fake table and the fake stages saw."""

    def __init__(self) -> None:
        self.stages: list[str] = []
        self.seeds: list[int] = []
        self.lock_bytes: list[bytes] = []
        self.runners: list[object] = []
        self.contexts: list[StageContext] = []
        # The seconds of the budget that were left when each stage started.
        self.remaining: list[float] = []


def _function(
    stage_id: str, outcome: Outcome, clock: FakeClock, recording: Recording
) -> StageFunction:
    golden = {result.record.id: result for result in golden_results()}

    def play(ctx: StageContext) -> StageResult:
        recording.stages.append(stage_id)
        recording.contexts.append(ctx)
        recording.remaining.append(ctx.remaining_seconds())
        clock.advance(outcome.seconds)
        if outcome.kind == "data_error":
            raise DataError(DATA_ERROR_TEXT)
        if outcome.kind == "crash":
            raise RuntimeError(SECRET)
        if outcome.kind == "skipped":
            return StageResult(StageRecord(stage_id, "skipped", outcome.reason), {})
        if outcome.kind == "no_facts":
            # `ok`, but it measured nothing: the thresholds that read its figures are missing.
            return StageResult(golden[stage_id].record, {})
        if outcome.kind == "fail":
            return StageResult(
                StageRecord(stage_id, "fail", fail_exit(outcome.exit_code)), {}
            )
        assert outcome.kind == "ok"
        return golden[stage_id]

    return play


def fake_table(
    outcomes: Mapping[str, Outcome], clock: FakeClock, recording: Recording
) -> Callable[[int, bytes], dict[str, StageFunction]]:
    """The replacement of `tiny.make_stage_functions`: fifteen fake stages that do what `outcomes` say
    (`OK` for a stage not named) and remember the seed and the lock bytes they were bound to."""

    def make(seed: int, lock_bytes: bytes) -> dict[str, StageFunction]:
        recording.seeds.append(seed)
        recording.lock_bytes.append(lock_bytes)
        return {
            stage_id: _function(stage_id, outcomes.get(stage_id, OK), clock, recording)
            for stage_id in STAGE_IDS
        }

    return make


def install(
    monkeypatch: pytest.MonkeyPatch,
    *,
    outcomes: Mapping[str, Outcome] | None = None,
    clock: FakeClock | None = None,
    runner: object | None = None,
    table: Callable[[int, bytes], dict[str, StageFunction]] | None = None,
) -> tuple[Recording, FakeClock]:
    """Replace the three seams of the command and the code version of its manifest. The table is
    `fake_table(outcomes)` unless `table` is given; the runner is the fake one unless `runner` is.
    Returns what the fakes will record and the clock."""
    from rinchen.cli import tiny, tiny_output

    recording = Recording()
    used_clock = clock if clock is not None else FakeClock()
    used_runner = runner if runner is not None else object()

    def make_runner() -> object:
        recording.runners.append(used_runner)
        return used_runner

    monkeypatch.setattr(tiny, "make_runner", make_runner)
    monkeypatch.setattr(
        tiny,
        "make_stage_functions",
        table
        if table is not None
        else fake_table(outcomes or {}, used_clock, recording),
    )
    monkeypatch.setattr(tiny, "clock", used_clock)
    monkeypatch.setattr(tiny_output, "code_version", lambda: CODE_VERSION)
    return recording, used_clock


def watch_first_stage(
    monkeypatch: pytest.MonkeyPatch, observe: Callable[[StageContext], None]
) -> tuple[Recording, FakeClock]:
    """`install` with the fake stages, the first of which calls `observe(ctx)` before it does its work:
    what a test looks at from inside the run, at the moment the first stage starts."""
    from rinchen.cli import tiny

    recording, clock = install(monkeypatch)
    inner = tiny.make_stage_functions

    def table(seed: int, lock_bytes: bytes) -> dict[str, StageFunction]:
        functions = dict(inner(seed, lock_bytes))
        first = functions[STAGE_IDS[0]]

        def watching(ctx: StageContext) -> StageResult:
            observe(ctx)
            return first(ctx)

        functions[STAGE_IDS[0]] = watching
        return functions

    monkeypatch.setattr(tiny, "make_stage_functions", table)
    return recording, clock


def invoke(arguments: Sequence[str]) -> Result:
    """Run the `rinchen` app on these arguments through typer's runner."""
    from rinchen.cli.main import app

    return _runner.invoke(app, list(arguments))


def go(
    monkeypatch: pytest.MonkeyPatch,
    out: Path,
    *extra: str,
    outcomes: Mapping[str, Outcome] | None = None,
    config: Path | None = None,
) -> tuple[Result, Recording, FakeClock]:
    """Run `tiny run --out-dir OUT extra` over the fake stages: the result, what the fakes saw and the
    clock."""
    recording, clock = install(monkeypatch, outcomes=outcomes)
    return invoke(run_args(out, *extra, config=config)), recording, clock


def report_of(out: Path) -> dict[str, Any]:
    loaded = json.loads((out / "report.json").read_text(encoding="utf-8"))
    assert isinstance(loaded, dict)
    return loaded


def run_args(out: Path, *extra: str, config: Path | None = None) -> list[str]:
    """`tiny run --out-dir OUT [--config CONFIG] extra`."""
    arguments = ["tiny", "run", "--out-dir", str(out)]
    if config is not None:
        arguments += ["--config", str(config)]
    return [*arguments, *extra]


def snapshot(root: Path) -> dict[str, bytes]:
    """`{relative path: bytes}` of every file under `root` (empty when it does not exist)."""
    if not root.exists():
        return {}
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def identity(root: Path) -> dict[str, tuple[bytes, int]]:
    """The bytes and the modification time of every file under `root`: what 'untouched' means."""
    if not root.exists():
        return {}
    return {
        path.relative_to(root).as_posix(): (path.read_bytes(), path.stat().st_mtime_ns)
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def manifest_of(out: Path) -> dict[str, Any]:
    loaded = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert isinstance(loaded, dict)
    return loaded


def entries_of(out: Path) -> list[str]:
    """The names at the top level of `out`, sorted."""
    return sorted(path.name for path in out.iterdir())


def lines_of(result: Result) -> list[str]:
    """The lines the command printed to standard output."""
    return result.stdout.splitlines()


# ---- the files of the inputs, copied so that a test may change them ----------------------------------------


@dataclass(frozen=True)
class Workspace:
    """Copies of every file the command hashes, and a control profile that names them."""

    config: Path
    fixtures: Path
    base: Path
    serve: Path
    words: Path
    profile: Path
    lock: Path


def make_workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Workspace:
    """The fixtures, the Eole base config, the serve config, the words file, the profile file and the lock
    copied under `tmp_path`, a control profile (`--config`) that names the copies, and the two module
    constants of `rinchen.cli.tiny_inputs` that name the profile file and the lock pointed at the copies."""
    from rinchen.cli import tiny_inputs
    from tests.support.tiny_profiles import control_profile, copy_fixtures

    fixtures = copy_fixtures(tmp_path / "fixtures")
    base = tmp_path / "base.yaml"
    serve = tmp_path / "serve.yaml"
    words = tmp_path / "words.tsv"
    profiles = tmp_path / "profiles"
    profiles.mkdir()
    lock = tmp_path / "uv.lock"
    for source, copy in (
        (REPO_ROOT / "configs" / "eole" / "tiny.yaml", base),
        (REPO_ROOT / "configs" / "serve" / "tiny.yaml", serve),
        (REPO_ROOT / "tests" / "fixtures" / "script_conv" / "tiny_words.tsv", words),
        (REPO_ROOT / "configs" / "profiles" / "tiny.yaml", profiles / "tiny.yaml"),
        (REPO_ROOT / "uv.lock", lock),
    ):
        shutil.copyfile(source, copy)
    monkeypatch.setattr(tiny_inputs, "PROFILES_DIRECTORY", profiles)
    monkeypatch.setattr(tiny_inputs, "LOCK_PATH", lock)
    config = control_profile(
        tmp_path,
        fixtures_directory=str(fixtures),
        train_base=str(base),
        serve_config=str(serve),
        converter_pairs=str(words),
    )
    return Workspace(config, fixtures, base, serve, words, profiles / "tiny.yaml", lock)


def append_a_line(path: Path) -> None:
    """Change a file the way an edit between two readings would."""
    path.write_bytes(path.read_bytes() + b"\n# changed\n")


def changing_after_hashing(
    monkeypatch: pytest.MonkeyPatch, target: Path, change: Callable[[Path], None]
) -> None:
    """Hash `target` as it is and then change it, the way a file changes between the hashing of the
    inputs and the reading of them (the hashing function of `rinchen.cli.tiny_inputs` is wrapped)."""
    from rinchen.cli import tiny_inputs

    real = tiny_inputs.hash_input

    def hashing(path: Path, **keywords: Any) -> Any:
        entry = real(path, **keywords)
        if path == target:
            change(path)
        return entry

    monkeypatch.setattr(tiny_inputs, "hash_input", hashing)
