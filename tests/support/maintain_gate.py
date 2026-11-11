"""Helpers of the D14 STOP gate (`tests/integration/test_maintain_gate.py` and
`test_maintain_gate_controls.py`): run the real `rinchen maintain check` binary, and the command of the
nightly workflow itself, read what they wrote and hold the floors.

Every command runs in a process that hides every device. `run_maintain` and `run_workflow_command` REFUSE to
start unless the environment they would hand the child holds `CUDA_VISIBLE_DEVICES` as the empty string
(`d11_gpu_hidden`, a session fixture of `tests/integration/conftest.py`, sets it), and every child gets the
Hugging Face offline switches, `UV_OFFLINE=1` and no token. Each command is bounded by `COMMAND_TIMEOUT_SECONDS`, starts in a
session of its own and, on a timeout, takes the whole process group of that child away (nothing else is ever
signalled). The only programs these helpers start are `sh` (the workflow's command), the `rinchen` binary and, in
the one place a test must read it, the real `uv --version`; the `uv` the workflow's command finds is the shim of
`tests/support/fake_uv.py`.

The honest limit: the gate cannot prove the absence of a socket in a child process. What keeps the run off the
network is the shim (it forwards `lock --check --offline`, `--version` and `run --offline` and answers the one
call of a night that would reach an index, `lock --upgrade --dry-run`, from a hand-written sample), the
offline switches and the empty device list; the gate checks the shim's call log and that `uv.lock` and
`pyproject.toml` kept their bytes and their modification times.

THE FLOORS below are literals of this module and are never lowered: a miss of a floor is answered with more
steps of the toy, never by editing a number. The measured values they stand under are in the docstring of the
gate module.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import signal
import subprocess
import time
import tomllib
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import yaml

from rinchen.config.loader import REPO_ROOT
from tests.support.distill_gate import RINCHEN, Invocation
from tests.support.maintain_project import make_project, parent_environment
from tests.support.nightly_workflow import load, maintain_step_run
from tests.support.workflow_expression import Expression

__all__ = [
    "COMMAND_TIMEOUT_SECONDS",
    "MAX_MEAN_ABS_DELTA",
    "MIN_EXACT_SHARE",
    "MIN_NUMERICS_PASSED",
    "PARITY_TINY_PASSED",
    "REPOSITORY",
    "WORKFLOW",
    "LockTable",
    "absolute_paths",
    "file_identity",
    "finish",
    "gate_environment",
    "lock_table",
    "project_files_unchanged",
    "read_manifest",
    "read_report",
    "real_uv",
    "rerun_item",
    "run_maintain",
    "run_workflow_command",
    "scratch_project",
    "start_maintain",
    "step_of",
    "stop_group",
    "timestamps",
    "workflow_command",
    "write_config",
]

# ---- the floors: literals, never lowered ----------------------------------------------------------

# The numerics rerun: at least this many tests pass (1,110 at the spike; the configuration's floor).
MIN_NUMERICS_PASSED: Final = 1100
# The Eole against CTranslate2 rerun on the toy: exactly this many tests pass (13 at the spike).
PARITY_TINY_PASSED: Final = 13
# The parity of an operator's export: the command's own thresholds, held again over the report's figures.
MIN_EXACT_SHARE: Final = 0.95
MAX_MEAN_ABS_DELTA: Final = 1e-3

# ---- what the gate runs ---------------------------------------------------------------------------

REPOSITORY: Final = REPO_ROOT
WORKFLOW: Final = REPOSITORY / ".github" / "workflows" / "nightly.yml"
# One command of a night: the doctor (15 s), the lock steps, the toy's two trainings and every rerun.
COMMAND_TIMEOUT_SECONDS: Final = 1500
# `UV_OFFLINE` makes every uv a command starts offline, whichever way it finds it: the shim adds `--offline`
# to what it forwards, and this variable covers a uv that is reached by another route.
_OFFLINE: Final = {
    "CUDA_VISIBLE_DEVICES": "",
    "HF_HUB_OFFLINE": "1",
    "HF_DATASETS_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
    "UV_OFFLINE": "1",
}
_GRACE_SECONDS: Final = 5
# An absolute path: a slash that starts a token (not the slash of `a/b`, `1/2` or `n/a`) followed by at
# least one more component, or a Windows drive.
_ABSOLUTE_PATH: Final = re.compile(
    r"(?<![\w./:~-])/(?:[\w.+@=-]+/)*[\w.+@=-]+|\b[A-Za-z]:\\"
)
_TIMESTAMP: Final = re.compile(r"\d{4}-\d{2}-\d{2}|\d{2}:\d{2}:\d{2}")


# ---- the environment and the commands -------------------------------------------------------------


def real_uv() -> str:
    """The real uv the shim forwards to: the one on this process's `PATH`."""
    found = shutil.which("uv")
    assert found is not None, "the real uv must be on PATH: the gate forwards to it"
    return found


def gate_environment(
    *, tmpdir: Path | None = None, extra: Mapping[str, str] | None = None
) -> dict[str, str]:
    """The environment of a command of the gate: the test process's without the variables that steer
    a pytest started from it and without a token, the device hidden and the Hub offline. `tmpdir`
    (when given) becomes `TMPDIR`, so that a pytest the command starts keeps its temporary files
    out of the test process's; `extra` is added last and may not change a protected variable."""
    environment = parent_environment()
    environment.pop("HF_TOKEN", None)
    environment.update(_OFFLINE)
    if tmpdir is not None:
        tmpdir.mkdir(parents=True, exist_ok=True)
        environment["TMPDIR"] = str(tmpdir)
    for name, value in (extra or {}).items():
        assert name not in _OFFLINE and name != "HF_TOKEN", f"{name} is protected"
        environment[name] = value
    return environment


def _require_hidden_device(environment: Mapping[str, str]) -> None:
    """Refuse to start a command unless this process and the environment of the child both hide
    every device (the session fixture `d11_gpu_hidden` hides them for the process)."""
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == "", (
        "CUDA_VISIBLE_DEVICES must be the empty string in the test process: use d11_gpu_hidden"
    )
    assert environment.get("CUDA_VISIBLE_DEVICES") == "", (
        "CUDA_VISIBLE_DEVICES must be the empty string for every command of the gate"
    )


def start_maintain(
    args: Sequence[str | Path], *, cwd: Path, env: Mapping[str, str] | None = None
) -> subprocess.Popen[bytes]:
    """Start `rinchen maintain check <args>` in `cwd`, in a session of its own, and return it.
    The caller finishes it with `finish`."""
    environment = dict(gate_environment() if env is None else env)
    _require_hidden_device(environment)
    return _start([RINCHEN, "maintain", "check", *map(str, args)], cwd, environment)


def _start(
    argv: Sequence[str], cwd: Path, environment: Mapping[str, str]
) -> subprocess.Popen[bytes]:
    return subprocess.Popen(
        list(argv),
        cwd=cwd,
        env=dict(environment),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
    )


def stop_group(process: subprocess.Popen[bytes]) -> None:
    """SIGKILL the process group of a child this gate started (it leads its own session) and reap it."""
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError, PermissionError:
        pass
    process.wait(timeout=_GRACE_SECONDS)


def finish(
    process: subprocess.Popen[bytes],
    *,
    timeout: int = COMMAND_TIMEOUT_SECONDS,
    started: float | None = None,
) -> Invocation:
    """Wait for `process` to end (at most `timeout` seconds, then its whole group is killed and the
    wait fails) and describe how it ended."""
    began = time.perf_counter() if started is None else started
    try:
        stdout, stderr = process.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        stop_group(process)
        raise
    finally:
        # A child that ended leaves no group behind; one that was left running is taken away.
        if process.poll() is None:
            stop_group(process)
    return Invocation(
        returncode=process.returncode,
        stdout=stdout.decode("utf-8", "replace").splitlines(),
        stderr=stderr.decode("utf-8", "replace"),
        seconds=time.perf_counter() - began,
    )


def run_maintain(
    args: Sequence[str | Path],
    *,
    cwd: Path,
    env: Mapping[str, str] | None = None,
    timeout: int = COMMAND_TIMEOUT_SECONDS,
) -> Invocation:
    """`rinchen maintain check <args>` through the real binary, to its end, in `cwd`."""
    started = time.perf_counter()
    return finish(
        start_maintain(args, cwd=cwd, env=env), timeout=timeout, started=started
    )


def workflow_command(workflow: Path, cadence: str) -> str:
    """The `run:` string of the step of `workflow` that runs the check, after asserting that the
    job's own expression gives `cadence` for a manual dispatch that chose it (so that `$CADENCE` of
    `run_workflow_command` is what the workflow would hand the command)."""
    document = load(workflow.read_text("utf-8"))
    (job,) = document["jobs"].values()
    chosen = Expression(
        job["env"]["CADENCE"], {"github.event.inputs.cadence": cadence}
    ).value()
    assert chosen == cadence, "the workflow's CADENCE expression gives another cadence"
    return maintain_step_run(document)


def run_workflow_command(
    shim_dir: Path,
    command: str,
    *,
    cwd: Path,
    cadence: str,
    tmpdir: Path | None = None,
    extra: Mapping[str, str] | None = None,
    timeout: int = COMMAND_TIMEOUT_SECONDS,
) -> Invocation:
    """`sh -c <command>` in `cwd` (a temporary directory, so that the command's relative
    `--out-dir` lands there), with `CADENCE` set and the directory of the `uv` shim first on `PATH`."""
    environment = gate_environment(tmpdir=tmpdir, extra=extra)
    environment["CADENCE"] = cadence
    environment["PATH"] = f"{shim_dir}{os.pathsep}{environment['PATH']}"
    _require_hidden_device(environment)
    started = time.perf_counter()
    return finish(
        _start(["sh", "-c", command], cwd, environment),
        timeout=timeout,
        started=started,
    )


# ---- reading what a run wrote ---------------------------------------------------------------------


def read_report(out_dir: Path) -> dict[str, Any]:
    document = json.loads((out_dir / "maintain_report.json").read_text("utf-8"))
    assert isinstance(document, dict)
    return document


def read_manifest(out_dir: Path) -> dict[str, Any]:
    document = json.loads((out_dir / "manifest.json").read_text("utf-8"))
    assert isinstance(document, dict)
    return document


def step_of(report: Mapping[str, Any], step_id: str) -> dict[str, Any]:
    (found,) = [step for step in report["steps"] if step["id"] == step_id]
    assert isinstance(found, dict)
    return found


def rerun_item(report: Mapping[str, Any], rerun_id: str) -> dict[str, Any]:
    (found,) = [
        item
        for item in step_of(report, "reruns")["detail"]["items"]
        if item["id"] == rerun_id
    ]
    assert isinstance(found, dict)
    return found


def file_identity(path: Path) -> tuple[str, int]:
    """The SHA-256 of a file's bytes and its modification time: what 'untouched' means."""
    return hashlib.sha256(path.read_bytes()).hexdigest(), path.stat().st_mtime_ns


@contextmanager
def project_files_unchanged(root: Path = REPOSITORY) -> Iterator[None]:
    """Inside the block nothing may touch `pyproject.toml` or `uv.lock` of the project at `root` (the
    repository by default): the bytes and the modification times they have on entry are the ones they
    have on exit (the check only reads them)."""
    names = ("pyproject.toml", "uv.lock")
    before = {name: file_identity(root / name) for name in names}
    yield
    after = {name: file_identity(root / name) for name in names}
    changed = [name for name in names if after[name] != before[name]]
    assert not changed, f"{', '.join(changed)} changed while a command of the gate ran"


@dataclass(frozen=True)
class LockTable:
    """What `uv.lock` says, read independently of `rinchen.maintain.lock`: the `[[package]]` lines,
    the distinct versions of each name (sorted) and the format."""

    package_lines: int
    versions: Mapping[str, tuple[str, ...]]
    format_version: int
    revision: int


def lock_table(lock: Path) -> LockTable:
    text = lock.read_text("utf-8")
    document = tomllib.loads(text)
    versions: dict[str, list[str]] = {}
    for entry in document["package"]:
        versions.setdefault(entry["name"], []).append(entry["version"])
    return LockTable(
        package_lines=len(re.findall(r"^\[\[package\]\]$", text, re.MULTILINE)),
        # A name locked twice at one version (`pyonmttok`: one wheel per platform) has one version.
        versions={name: tuple(sorted(set(found))) for name, found in versions.items()},
        format_version=document["version"],
        revision=document["revision"],
    )


# ---- the content scans ----------------------------------------------------------------------------


def absolute_paths(text: str) -> list[str]:
    """The absolute paths a text holds (POSIX paths with at least two components, Windows drives)."""
    return _ABSOLUTE_PATH.findall(text)


def timestamps(text: str) -> list[str]:
    """The dates and the times of day a text holds."""
    return _TIMESTAMP.findall(text)


# ---- a project and a configuration for the controls -----------------------------------------------


def scratch_project(tmp_path: Path, tests: Mapping[str, str] | None = None) -> Path:
    """A project under `tmp_path` that the real checks accept: the repository's `pyproject.toml` and
    `uv.lock` (copies, so the lock check, which needs the project's own files, and the environment
    check, which needs the lock the installed packages come from, both pass), a link to the repository's
    `vendor` directory (the lock names wheels there) and `tests/` made of `tests`."""
    project = make_project(tmp_path, tests=tests)
    for name in ("pyproject.toml", "uv.lock"):
        (project / name).write_bytes((REPOSITORY / name).read_bytes())
    (project / "vendor").symlink_to(REPOSITORY / "vendor")
    return project


def write_config(path: Path, reruns: Sequence[Mapping[str, Any]]) -> Path:
    """The shipped configuration with its reruns replaced by `reruns` (and no pin rules, which name
    the shipped reruns), written to `path`."""
    shipped = REPOSITORY / "configs" / "maintain" / "maintain.yaml"
    document = yaml.safe_load(shipped.read_text("utf-8"))
    document["reruns"] = [dict(rerun) for rerun in reruns]
    document["drift"]["rules"] = []
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
    return path
