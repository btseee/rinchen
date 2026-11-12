"""Helpers of the D15 STOP gate (`tests/integration/test_tiny_gate*.py`): run the body of the `tiny` recipe
and the real `rinchen` binary on the CPU, read what they wrote and hold the floors.

Every command of the gate starts through ONE function, `start_guarded`, which REFUSES to start unless the
environment it would hand the child holds the switches below (`ENVIRONMENT_GUARDS`: the empty device list,
the Hub and the datasets offline, uv offline) and no variable ending in `_TOKEN`, `_SECRET`, `_PASSWORD` or
`_API_KEY`, and unless the test process itself hides every device (`tiny_gpu_hidden`, a session fixture of
`tests/integration/conftest.py`, sets it). The refusal is a failed assertion BEFORE a process exists. A child
starts in a session of its own with its input closed, and every wait is bounded: on a timeout the whole
process group of the child is taken away, and then every process that carries the run's marker (the stages
start children of their own, in sessions of their own); nothing else is ever signalled, and only by pid.

The only programs these helpers start are `sh` (the recipe's body, with the `uv` shim of
`tests/support/fake_uv.py` first on `PATH`) and `python -m rinchen` (the pipeline and, in the controls, the
commands of `eval` and `export`); the `uv` the recipe finds is the shim, which forwards `run` as `uv run
--project <repository> --no-sync --frozen --offline` and refuses everything else. A start that is given a
shim directory is refused, before a process exists, unless the directory is absolute and holds an executable
`uv` and the `uv` that the `PATH` of the child leads to is that file (`require_shim`): a missing shim would let
the recipe fall through to the real `uv`.

The honest limit: the gate cannot prove the absence of a socket in a child. What keeps it off the network
is the shim, the offline switches and the empty device list; the one socket the pipeline opens is the
loopback listener of its synthetic-data stage (`rinchen.tiny.fakes`).

THE FLOORS below are literals of this module and are never lowered: a miss of a floor is answered with more
`train.steps` in `configs/tiny/tiny.yaml`, never by editing a number.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

from rinchen.config.loader import REPO_ROOT
from tests.support import tiny_gate_processes as processes
from tests.support.distill_gate import Invocation
from tests.support.fake_uv import write_uv_shim
from tests.support.maintain_gate import absolute_paths, real_uv, timestamps
from tests.support.maintain_project import parent_environment
from tests.support.tiny_gate_processes import MARKER_VARIABLE
from tests.support.uv_outputs import NO_CHANGE_MEASURED

__all__ = [
    "COMMAND_TIMEOUT_SECONDS",
    "ENVIRONMENT_GUARDS",
    "JUSTFILE",
    "MARKER_VARIABLE",
    "MAX_INT8_CHRF_DROP",
    "MAX_PARITY_DELTA",
    "MAX_WALL_TIME_SECONDS",
    "MIN_PARITY_EXACT",
    "MIN_PLACEHOLDER_ACC",
    "MIN_SCRIPT_PURITY",
    "MIN_TRAIN_CHRF",
    "RECIPE_TIMEOUT_SECONDS",
    "RecipeRun",
    "Started",
    "absolute_paths",
    "duration_keys",
    "finish",
    "fixture_text_hits",
    "fixture_texts",
    "guarded_environment",
    "leaked_values",
    "make_shim",
    "part_states",
    "read_manifest",
    "read_report",
    "recipe_command",
    "require_guards",
    "require_shim",
    "run_recipe",
    "run_rinchen",
    "run_the_recipe",
    "run_tiny",
    "secret_like_strings",
    "send_to_group",
    "stage_of",
    "start_guarded",
    "start_rinchen",
    "start_tiny",
    "stop_run",
    "threshold_of",
    "timestamps",
    "wait_for_file",
]

# ---- the floors: literals, never lowered ----------------------------------------------------------

# The master plan's thresholds, as `configs/tiny/tiny.yaml` states them and `report.json` judges them.
MIN_TRAIN_CHRF: Final = 50.0
MIN_PARITY_EXACT: Final = 0.95
MAX_PARITY_DELTA: Final = 1e-3
MAX_INT8_CHRF_DROP: Final = 1.0
MIN_SCRIPT_PURITY: Final = 1.0
MIN_PLACEHOLDER_ACC: Final = 1.0
# The master plan's 20 minutes on a 4-core CPU: the budget of the profile.
MAX_WALL_TIME_SECONDS: Final = 1200

# ---- what the gate runs ---------------------------------------------------------------------------

JUSTFILE: Final = REPO_ROOT / "justfile"
RECIPE: Final = "tiny"
SHIM_NAME: Final = "uv"
# One run of the recipe: 120 to 158 s at the spike; the bound is the budget and a margin for a slow runner.
RECIPE_TIMEOUT_SECONDS: Final = 1500
# One command of a control (`eval robustness`, `export parity`, a refusal): seconds, bounded generously.
COMMAND_TIMEOUT_SECONDS: Final = 300
_GRACE_SECONDS: Final = 5
_POLL_SECONDS: Final = 0.2
_SECRET_SUFFIXES: Final = ("_TOKEN", "_SECRET", "_PASSWORD", "_API_KEY")
# The switches every child of the gate has: the device list is empty, the Hub and the datasets are
# offline, uv is offline. `TRANSFORMERS_OFFLINE` is the D14 gate's fifth.
ENVIRONMENT_GUARDS: Final = {
    "CUDA_VISIBLE_DEVICES": "",
    "HF_HUB_OFFLINE": "1",
    "HF_DATASETS_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
    "UV_OFFLINE": "1",
}


# ---- the environment of a child ---------------------------------------------------------------------


def _is_secret(name: str) -> bool:
    return name.upper().endswith(_SECRET_SUFFIXES)


def guarded_environment(
    *, env_extra: Mapping[str, str] | None = None, shim_dir: Path | None = None
) -> dict[str, str]:
    """The environment of a command of the gate: the test process's without the variables that steer a
    pytest and without any secret, the guards, a marker of its own and, when `shim_dir` is given, that
    directory first on `PATH`. `env_extra` is added last and may not name a guard, the marker or a secret."""
    environment = {
        name: value
        for name, value in parent_environment().items()
        if not _is_secret(name)
    }
    environment.update(ENVIRONMENT_GUARDS)
    environment[MARKER_VARIABLE] = uuid.uuid4().hex
    if shim_dir is not None:
        environment["PATH"] = f"{shim_dir}{os.pathsep}{environment.get('PATH', '')}"
    for name, value in (env_extra or {}).items():
        assert name not in ENVIRONMENT_GUARDS and name != MARKER_VARIABLE, (
            f"{name} is protected: the gate sets it"
        )
        assert not _is_secret(name), f"{name} would hand a secret to the child"
        environment[name] = value
    return environment


def require_guards(environment: Mapping[str, str]) -> None:
    """Refuse (a failed assertion) unless this process hides every device, `environment` holds every guard
    with its value and no secret would reach the child."""
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == "", (
        "CUDA_VISIBLE_DEVICES must be the empty string in the test process: use tiny_gpu_hidden"
    )
    for name, value in ENVIRONMENT_GUARDS.items():
        assert environment.get(name) == value, (
            f"{name} must be {value!r} for every command of the gate"
        )
    secrets = [name for name in environment if _is_secret(name)]
    assert not secrets, f"{', '.join(secrets)} would reach the child"


def require_shim(shim_dir: Path, path: str) -> None:
    """Refuse (a failed assertion) unless `shim_dir` is absolute and holds an executable `uv` and the first
    `uv` on `path`, the `PATH` the child will get, is that file. A directory without the shim would let the
    recipe fall through to the real `uv`, which locks, syncs and asks an index."""
    shim = shim_dir / SHIM_NAME
    assert shim_dir.is_absolute(), (
        f"{shim_dir} must be absolute: the child runs in another directory"
    )
    assert shim.is_file() and os.access(shim, os.X_OK), (
        f"{shim} is not an executable file"
    )
    found = shutil.which(SHIM_NAME, path=path)
    assert found is not None and Path(found).samefile(shim), (
        f"the {SHIM_NAME} the child would find is not the shim in {shim_dir}"
    )


# ---- the recipe ------------------------------------------------------------------------------------


def recipe_command(justfile: Path) -> str:
    """The body of the `tiny` recipe of `justfile`: its one command line, without the indent.

    Raises:
        AssertionError: there is no recipe `tiny` without a parameter or a dependency, or its body is
            not exactly one line (a recipe of several lines runs each in a shell of its own).
    """
    lines = justfile.read_text(encoding="utf-8").splitlines()
    assert f"{RECIPE}:" in lines, (
        f"the justfile has no recipe `{RECIPE}` without a parameter or a dependency"
    )
    body: list[str] = []
    for line in lines[lines.index(f"{RECIPE}:") + 1 :]:
        if line.strip() and not line[0].isspace():
            break
        if line.strip():
            body.append(line.strip())
    assert len(body) == 1, f"the `{RECIPE}` recipe is not one command line"
    return body[0]


# ---- starting and finishing a command ----------------------------------------------------------------


@dataclass(frozen=True)
class Started:
    """A command of the gate that is running: its process (it leads its own session), the marker its
    environment carries and the clock reading at its start."""

    process: subprocess.Popen[bytes]
    marker: str
    began: float


def start_guarded(
    argv: Sequence[str | Path],
    *,
    cwd: Path,
    env_extra: Mapping[str, str] | None = None,
    shim_dir: Path | None = None,
) -> Started:
    """Start `argv` in `cwd`, in a session of its own with its input closed, with the guards. The caller
    ends it with `finish`. This is the one place a command of the gate is started. With `shim_dir` the
    shim is checked first (`require_shim`), against the `PATH` the child is about to get."""
    environment = guarded_environment(env_extra=env_extra, shim_dir=shim_dir)
    require_guards(environment)
    if shim_dir is not None:
        require_shim(shim_dir, environment["PATH"])
    process = subprocess.Popen(
        [str(part) for part in argv],
        cwd=cwd,
        env=environment,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
    )
    return Started(process, environment[MARKER_VARIABLE], time.perf_counter())


def stop_run(started: Started) -> None:
    """SIGKILL the process group of the child this gate started, then every process that carries its
    marker, and reap the child. Nothing else is signalled."""
    try:
        os.killpg(started.process.pid, signal.SIGKILL)
    except ProcessLookupError, PermissionError:
        pass
    processes.kill_the_run(started.marker)
    started.process.wait(timeout=_GRACE_SECONDS)


def send_to_group(started: Started, number: signal.Signals) -> None:
    """Send `number` to the process group of the child this gate started (the child leads it: its own
    children start sessions of their own and are not in it)."""
    os.killpg(started.process.pid, number)


def finish(started: Started, *, timeout: int) -> Invocation:
    """Wait for the command to end (at most `timeout` seconds, then it is stopped and the wait fails) and
    describe how it ended."""
    process = started.process
    try:
        stdout, stderr = process.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        stop_run(started)
        raise
    finally:
        # A child that ended leaves no group behind; one that was left running is taken away.
        if process.poll() is None:
            stop_run(started)
    assert process.returncode is not None
    return Invocation(
        returncode=process.returncode,
        stdout=stdout.decode("utf-8", "replace").splitlines(),
        stderr=stderr.decode("utf-8", "replace"),
        seconds=time.perf_counter() - started.began,
    )


def wait_for_file(path: Path, started: Started, *, seconds: float) -> None:
    """Return once `path` exists. Fails when the command ended first or `seconds` pass."""
    deadline = time.monotonic() + seconds
    while not path.exists():
        assert started.process.poll() is None, (
            "the command ended before the file existed"
        )
        assert time.monotonic() < deadline, "the file did not exist in time"
        time.sleep(_POLL_SECONDS)


def start_rinchen(
    args: Sequence[str | Path], *, cwd: Path, env_extra: Mapping[str, str] | None = None
) -> Started:
    """Start `python -m rinchen <args>` with the interpreter of the tests, guarded."""
    return start_guarded(
        [sys.executable, "-m", "rinchen", *args], cwd=cwd, env_extra=env_extra
    )


def start_tiny(
    args: Sequence[str | Path], *, cwd: Path, env_extra: Mapping[str, str] | None = None
) -> Started:
    """Start `python -m rinchen tiny run <args>`, guarded."""
    return start_rinchen(["tiny", "run", *args], cwd=cwd, env_extra=env_extra)


def run_rinchen(
    args: Sequence[str | Path],
    *,
    cwd: Path,
    env_extra: Mapping[str, str] | None = None,
    timeout: int = COMMAND_TIMEOUT_SECONDS,
) -> Invocation:
    """`python -m rinchen <args>` to its end, guarded and bounded."""
    return finish(start_rinchen(args, cwd=cwd, env_extra=env_extra), timeout=timeout)


def run_tiny(
    args: Sequence[str | Path],
    *,
    cwd: Path,
    env_extra: Mapping[str, str] | None = None,
    timeout: int = RECIPE_TIMEOUT_SECONDS,
) -> Invocation:
    """`python -m rinchen tiny run <args>` to its end, guarded and bounded."""
    return finish(start_tiny(args, cwd=cwd, env_extra=env_extra), timeout=timeout)


def run_recipe(
    command: str,
    *,
    cwd: Path,
    env_extra: Mapping[str, str] | None = None,
    timeout: int = RECIPE_TIMEOUT_SECONDS,
    shim_dir: Path,
) -> Invocation:
    """`sh -c <command>` in `cwd` (a temporary directory, so that the recipe's relative `artifacts/tiny`
    lands there) with the directory of the `uv` shim first on `PATH`."""
    started = start_guarded(
        ["sh", "-c", command], cwd=cwd, env_extra=env_extra, shim_dir=shim_dir
    )
    return finish(started, timeout=timeout)


@dataclass(frozen=True)
class RecipeRun:
    """One run of the body of the `tiny` recipe: what it printed, where it ran, the directory of its
    shim (whose call log is the proof that nothing but `run` was asked of uv) and the load average of
    the machine (1, 5 and 15 minutes) when it started and when it ended."""

    invocation: Invocation
    cwd: Path
    shim_dir: Path
    load_start: tuple[float, float, float]
    load_end: tuple[float, float, float]

    @property
    def out_dir(self) -> Path:
        """The recipe's relative `artifacts/tiny`."""
        return self.cwd / "artifacts" / "tiny"


def make_shim(directory: Path) -> Path:
    """Write the `uv` shim of `tests/support/fake_uv.py` into `directory` (made) with an empty call log
    and return the directory: the real uv, which the shim forwards `run` to, only ever sees
    `run --project <repository> --no-sync --frozen --offline ...`."""
    shim = write_uv_shim(
        directory,
        real_uv=real_uv(),
        project=REPO_ROOT,
        dry_run_output=NO_CHANGE_MEASURED,
    )
    return shim.parent


def run_the_recipe(
    root: Path,
    *,
    shim_dir: Path,
    env_extra: Mapping[str, str] | None = None,
    timeout: int = RECIPE_TIMEOUT_SECONDS,
) -> RecipeRun:
    """The body of the justfile's `tiny` recipe, run through `sh` in `root/cwd` (made) with the shim in
    `shim_dir` first on `PATH`."""
    cwd = root / "cwd"
    cwd.mkdir()
    load_start = os.getloadavg()
    invocation = run_recipe(
        recipe_command(JUSTFILE),
        cwd=cwd,
        env_extra=env_extra,
        timeout=timeout,
        shim_dir=shim_dir,
    )
    return RecipeRun(invocation, cwd, shim_dir, load_start, os.getloadavg())


# ---- reading what a run wrote -----------------------------------------------------------------------


def _document(path: Path) -> dict[str, Any]:
    document = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(document, dict), f"{path.name} is not an object"
    return document


def read_report(out_dir: Path) -> dict[str, Any]:
    """`report.json` of a run."""
    return _document(out_dir / "report.json")


def read_manifest(out_dir: Path) -> dict[str, Any]:
    """`manifest.json` of a run."""
    return _document(out_dir / "manifest.json")


def stage_of(report: Mapping[str, Any], stage_id: str) -> dict[str, Any]:
    """The entry of the stage `stage_id` in a report (exactly one is listed)."""
    (found,) = [stage for stage in report["stages"] if stage["id"] == stage_id]
    assert isinstance(found, dict)
    return found


def threshold_of(report: Mapping[str, Any], threshold_id: str) -> dict[str, Any]:
    """The entry of the threshold `threshold_id` in a report (exactly one is listed)."""
    (found,) = [item for item in report["thresholds"] if item["id"] == threshold_id]
    assert isinstance(found, dict)
    return found


def part_states(stage: Mapping[str, Any]) -> dict[str, str]:
    """The state of each part of a stage entry, by part id."""
    return {part["id"]: part["state"] for part in stage["parts"]}


# ---- the content scans -------------------------------------------------------------------------------

# A key is a duration or a date when one of its words is one of these.
_DURATION_WORDS: Final = frozenset(
    {
        "seconds",
        "second",
        "secs",
        "duration",
        "elapsed",
        "timestamp",
        "time",
        "date",
        "created",
        "started",
        "finished",
        "ms",
        "millis",
    }
)
# The shape of a credential. A word such as `token` is not one (the report speaks of a token ratio).
_SECRET_SHAPES: Final = (
    ("a Hub token", re.compile(r"\bhf_[A-Za-z0-9]{20,}")),
    ("an API key", re.compile(r"\bsk-[A-Za-z0-9]{20,}")),
    ("a GitHub token", re.compile(r"\bghp_[A-Za-z0-9]{20,}")),
    ("a cloud key", re.compile(r"\bAKIA[0-9A-Z]{16}\b")),
    ("a private key", re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----")),
    (
        "a secret variable",
        re.compile(r"\b[A-Z][A-Z0-9_]*(?:_TOKEN|_SECRET|_PASSWORD|_API_KEY)\s*[=:]"),
    ),
)
_NOT_TEXT_COLUMNS: Final = frozenset(
    {"id", "split", "kind", "expect", "src_lang", "tgt_lang"}
)
_SENTENCE_LENGTH: Final = 10
_WORD_LENGTH: Final = 4
_LEAK_LENGTH: Final = 4


def duration_keys(document: object, prefix: str = "") -> list[str]:
    """The paths of the keys of a JSON document that name a duration or a date, sorted (`a.b[0].key`)."""
    found: list[str] = []
    if isinstance(document, Mapping):
        for key, value in document.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            words = re.split(r"[_\-\s]+", str(key).lower())
            if _DURATION_WORDS.intersection(words):
                found.append(path)
            found.extend(duration_keys(value, path))
    elif isinstance(document, list):
        for index, item in enumerate(document):
            found.extend(duration_keys(item, f"{prefix}[{index}]"))
    return sorted(found)


def leaked_values(
    text: str, values: Mapping[str, str], *, minimum: int = _LEAK_LENGTH
) -> list[str]:
    """The names, in the order of `values`, of the values that `text` holds. A value shorter than
    `minimum` characters says nothing and is not looked for. The value itself is never returned."""
    return [
        name
        for name, value in values.items()
        if len(value) >= minimum and value in text
    ]


def secret_like_strings(text: str) -> list[str]:
    """What kinds of credential a text holds by their shape (never the credential)."""
    return [kind for kind, shape in _SECRET_SHAPES if shape.search(text)]


def fixture_texts(directory: Path) -> tuple[str, ...]:
    """The texts of the fixture files in `directory`, sorted and without repeats: every cell of a text
    column that is a sentence (ten characters or more, with a space) or a word that is not ASCII (four
    characters or more). The id, split, kind and reason columns are not texts."""
    found: set[str] = set()
    for path in sorted(directory.glob("*.tsv")):
        rows = path.read_text(encoding="utf-8").splitlines()
        header = rows[0].split("\t")
        for row in rows[1:]:
            for name, cell in zip(header, row.split("\t"), strict=False):
                if name in _NOT_TEXT_COLUMNS:
                    continue
                if (len(cell) >= _SENTENCE_LENGTH and " " in cell) or (
                    not cell.isascii() and len(cell) >= _WORD_LENGTH
                ):
                    found.add(cell)
    return tuple(sorted(found))


def fixture_text_hits(text: str, texts: Sequence[str]) -> int:
    """How many of `texts` a document holds (a count: no text is ever returned)."""
    return sum(1 for candidate in texts if candidate in text)
