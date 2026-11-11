"""Scenarios for the tests of `rinchen.maintain.check.run_check` (D14, Task 4): the seams of a run
built from doubles, so that no process starts, and the one scripted run the golden report is made
from.

Everything `run_check` takes as an argument is made here with a known answer: a doctor that
records how often it is called, installed versions equal to a lock, a clock that moves by a fixed
step, a parent environment holding values that no report may repeat, and a `FakeRunner` scripted
for the lock check, the dry run and the reruns by what each command is (a `uv` argument list, a
JUnit option naming a rerun), so that a test's script does not depend on the order of the calls
and the order is asserted by itself.

The golden scenario is the nightly run of the shipped configuration over the lock of
`tests/support/uv_outputs.py`: six doctor passes and the GPU skip, every installed version equal
to the lock's, the HAND-WRITTEN dry run of five updates, one add and one remove, and the reruns
`numerics` (1,110 passed) and `parity_tiny` (13 passed). The three digests are fixed strings, so
that a change of the shipped files does not move the golden bytes except through the rules and
the reruns the report names.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Literal

import yaml

from rinchen.config.loader import REPO_ROOT
from rinchen.config.maintain import MaintainConfig, load_maintain_config
from rinchen.maintain.check import (
    CheckOptions,
    CheckResult,
    CheckSeams,
    DoctorFinding,
    Inputs,
    run_check,
)
from rinchen.maintain.commands import CommandResult
from rinchen.maintain.lock import LockFile
from rinchen.maintain.reruns import ExportInputs
from tests.support.maintain_fakes import FakeRunner, junit_xml
from tests.support.maintain_project import make_project
from tests.support.uv_outputs import (
    CHANGES_HAND_WRITTEN,
    NO_CHANGE_MEASURED,
    SAMPLE_LOCKED,
    lock_for_samples,
)

UV: Final = "uv"
PYTHON: Final = "python"
SHIPPED_CONFIG: Final = REPO_ROOT / "configs" / "maintain" / "maintain.yaml"
CPU_CHECKS: Final = (
    "D-PY",
    "D-WHEELS",
    "D-EOLE-PY",
    "D-EOLE-CLI",
    "D-CT2-CONV",
    "D-CT2-IMPORT",
)
GPU_CHECK: Final = "D-TORCH-CUDA"
# The messages of the fake doctor carry a path and an exception text: the report may hold neither.
DOCTOR_MESSAGE: Final = (
    "import eole failed: No module named /srv/example/site-packages/eole"
)
SECRET_ENVIRONMENT: Final[Mapping[str, str]] = {
    "RINCHEN_TEST_SECRET": "swordfish-0451",
    "HOME": "/home/ghost-user",
    "CI_TOKEN": "tok-7f3a9c21",
}
# The statistics of the shipped reruns' floors: a JUnit report that just meets each.
FLOORS: Final[Mapping[str, int]] = {
    "numerics": 1110,
    "parity_tiny": 13,
    "robustness_tiny": 31,
    "evaluation_tiny": 99,
    "serving_tiny": 55,
}


def sha256_of(label: str) -> str:
    """A well-formed digest that stands for the file called `label`."""
    return hashlib.sha256(label.encode("ascii")).hexdigest()


def shipped_config() -> MaintainConfig:
    return load_maintain_config(SHIPPED_CONFIG.read_bytes())


def config_with(**changes: Any) -> MaintainConfig:
    """The shipped table with these top-level keys replaced (as plain data, so that they are
    validated again)."""
    document = yaml.safe_load(SHIPPED_CONFIG.read_text(encoding="utf-8"))
    document.update(changes)
    return MaintainConfig.model_validate(document)


class Doctor:
    """A doctor seam that reports these findings and counts its calls."""

    def __init__(self, findings: Sequence[DoctorFinding]) -> None:
        self.findings = tuple(findings)
        self.calls = 0

    def __call__(self) -> Sequence[DoctorFinding]:
        self.calls += 1
        return self.findings


class Clock:
    """A clock that moves forward by `step` seconds each time it is read."""

    def __init__(self, step: float = 0.25) -> None:
        self.now = 0.0
        self.step = step

    def __call__(self) -> float:
        self.now += self.step
        return self.now


def finding(
    check_id: str, status: Literal["pass", "fail", "skip"] = "pass"
) -> DoctorFinding:
    return DoctorFinding(check_id, status, DOCTOR_MESSAGE)


def golden_doctor() -> Doctor:
    """The doctor of the golden run: six passes and the GPU check, which skips by itself."""
    return Doctor(
        [*(finding(check_id) for check_id in CPU_CHECKS), finding(GPU_CHECK, "skip")]
    )


def make_inputs(
    config: MaintainConfig | None = None, lock: LockFile | None = None
) -> Inputs:
    return Inputs(
        config=shipped_config() if config is None else config,
        lock=lock_for_samples() if lock is None else lock,
        config_sha256=sha256_of("maintain.yaml of the golden run"),
        lock_sha256=sha256_of("uv.lock of the golden run"),
        pyproject_sha256=sha256_of("pyproject.toml of the golden run"),
    )


def make_options(
    *,
    cadence: str = "nightly",
    drift: bool = True,
    reruns: bool = True,
    selected: tuple[str, ...] = (),
    export: ExportInputs | None = None,
) -> CheckOptions:
    return CheckOptions(
        cadence=cadence, drift=drift, reruns=reruns, selected=selected, export=export
    )


def make_seams(
    runner: FakeRunner,
    *,
    doctor: Callable[[], Sequence[DoctorFinding]] | None = None,
    installed: Callable[[], Mapping[str, Sequence[str]]] | None = None,
    uv: str | None = UV,
    parent_env: Mapping[str, str] | None = None,
) -> CheckSeams:
    return CheckSeams(
        runner=runner,
        uv=uv,
        python=PYTHON,
        doctor=golden_doctor() if doctor is None else doctor,
        installed=(lambda: SAMPLE_LOCKED) if installed is None else installed,
        clock=Clock(),
        parent_env=SECRET_ENVIRONMENT if parent_env is None else parent_env,
    )


# ---- scripting the runner by what a command is --------------------------------------------------


def uv_call(*flags: str) -> Callable[[Sequence[str]], bool]:
    """Matches the argument list `[uv, "lock", *flags]`."""
    wanted = ["lock", *flags]
    return lambda argv: list(argv[1:]) == wanted


def rerun_call(rerun_id: str) -> Callable[[Sequence[str]], bool]:
    """Matches the pytest command whose JUnit option names `rerun_id`, or, for `parity_export`,
    the parity command."""
    if rerun_id == "parity_export":
        return lambda argv: list(argv[2:5]) == ["rinchen", "export", "parity"]
    return lambda argv: any(arg.endswith(f"/junit/{rerun_id}.xml") for arg in argv)


def rerun_id_of(argv: Sequence[str]) -> str:
    """The rerun a command of a run belongs to: the name in its JUnit option, or the parity
    command's."""
    if list(argv[2:5]) == ["rinchen", "export", "parity"]:
        return "parity_export"
    (option,) = [arg for arg in argv if arg.startswith("--junitxml=")]
    return Path(option).stem


def script_passing_reruns(runner: FakeRunner, *rerun_ids: str) -> FakeRunner:
    """Script each pytest rerun to exit 0 with a JUnit report that just meets its floor."""
    for rerun_id in rerun_ids:
        runner.script(
            CommandResult(0, False),
            junit=junit_xml(FLOORS[rerun_id], 0, 0, 0),
            when=rerun_call(rerun_id),
        )
    return runner


def golden_runner() -> FakeRunner:
    """The lock check, the dry run and the two nightly pytest reruns of the golden run."""
    runner = (
        FakeRunner()
        .script(
            CommandResult(0, False),
            log_text=NO_CHANGE_MEASURED,
            when=uv_call("--check"),
        )
        .script(
            CommandResult(0, False),
            log_text=CHANGES_HAND_WRITTEN,
            when=uv_call("--upgrade", "--dry-run"),
        )
    )
    return script_passing_reruns(runner, "numerics", "parity_tiny")


@dataclass(frozen=True, slots=True)
class Run:
    """A run of `run_check` and what it ran against."""

    result: CheckResult
    runner: FakeRunner
    project: Path
    out: Path


def execute(
    tmp_path: Path,
    runner: FakeRunner,
    *,
    options: CheckOptions | None = None,
    seams: CheckSeams | None = None,
    inputs: Inputs | None = None,
) -> Run:
    """`run_check` over a fixture project in `tmp_path`, which holds the shipped targets."""
    inputs = make_inputs() if inputs is None else inputs
    project = make_project(tmp_path, config=inputs.config)
    out = tmp_path / "out"
    result = run_check(
        inputs,
        make_options() if options is None else options,
        make_seams(runner) if seams is None else seams,
        project_dir=project,
        out_dir=out,
    )
    return Run(result, runner, project, out)


def golden_run(tmp_path: Path) -> Run:
    return execute(tmp_path, golden_runner())


def step_of(document: Mapping[str, object], step_id: str) -> dict[str, object]:
    """The step `step_id` of a report document (`report_document` of a report)."""
    steps = document["steps"]
    assert isinstance(steps, list)
    (found,) = [step for step in steps if step["id"] == step_id]
    return found
