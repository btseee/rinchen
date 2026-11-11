"""The rules of `.github/workflows/nightly.yml` as a function of its text (D14, Task 6, spec M14).

`findings(text, ci_text, config)` says what is wrong with a workflow, one short sentence each, and
nothing for a good one; the readers and the constants it judges by are here too, for
`tests/unit/test_nightly_workflow.py`, which asserts them on the file and on broken copies of it.
The trigger is looked up under the key `True` because PyYAML reads the key `on` as a boolean (it
does so in `ci.yml` too). The cadence is an expression of GitHub Actions that
`tests/support/workflow_expression.py` evaluates for the four triggers.
"""

from __future__ import annotations

import functools
import math
import re
import shlex
from collections.abc import Mapping
from typing import Any, Final

import yaml
from typer.testing import CliRunner

from rinchen.cli.main import app
from rinchen.config.maintain import MaintainConfig
from tests.support.workflow_expression import Expression

USES: Final = re.compile(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+@v\d+(?:\.\d+)*")

JOB: Final = "maintain"
NIGHTLY_CRON: Final = "17 3 * * *"
SUNDAY_CRON: Final = "47 4 * * 0"
CADENCE_CHOICE: Final = {
    "type": "choice",
    "options": ["nightly", "weekly"],
    "default": "nightly",
}
# `uv run` alone locks the project first: a `pyproject.toml` that moved without the lock would be
# repaired before the check could see it. `--frozen` keeps the lock as it is and `--no-sync` leaves
# the environment the previous step made, so the check reads the stack as it is.
UV_RUN: Final = "uv run --frozen --no-sync"
CHECK_PREFIX: Final = f"{UV_RUN} rinchen maintain check"
SYNC_RUN: Final = "uv sync --frozen --group dev"
MAINTAIN_RUN: Final = f'{CHECK_PREFIX} --out-dir maintain-out --cadence "$CADENCE"'
SUMMARY_RUN: Final = 'cat maintain-out/maintain_report.md >> "$GITHUB_STEP_SUMMARY"'
SUMMARY_IF: Final = (
    "${{ always() && hashFiles('maintain-out/maintain_report.md') != '' }}"
)
REPORT_FILE: Final = "hashFiles('maintain-out/maintain_report.md')"
# The steps of the spec, in order, without their names and without the version of an action (the
# version is checked against `ci.yml`, not repeated here).
EXPECTED_STEPS: Final = [
    {"uses": "actions/checkout"},
    {"uses": "astral-sh/setup-uv"},
    {"uses": "actions/setup-python", "with": {"python-version": "3.14"}},
    {"uses": "extractions/setup-just"},
    {"run": SYNC_RUN},
    {"run": MAINTAIN_RUN},
    {"run": "just tiny"},
    {"if": SUMMARY_IF, "run": SUMMARY_RUN},
]
TOP_KEYS: Final = {"name", True, "permissions", "jobs"}
JOB_KEYS: Final = {"runs-on", "timeout-minutes", "env", "steps"}
# Said by a `run:` of a workflow that must not be said: an upload, a push, a commit, a pull request
# or any other command of `gh`, the lock file or a package changing, an echo of a variable, a dump
# of the environment, a trace of the commands.
FORBIDDEN_IN_RUN: Final = (
    "--upload",
    "git push",
    "git commit",
    "gh ",
    "uv lock",
    "uv add",
    "uv pip",
    "--upgrade",
    "echo $",
    "printenv",
    "set -x",
    "env |",
)
# Mentioned nowhere, comments included (looked up in lower case).
FORBIDDEN_WORDS: Final = ("docker", "gpu", "cuda", "nvidia", "self-hosted")
# The seconds of the doctor, the checkout and the set-up that no rerun accounts for, and the margin.
DOCTOR_AND_SETUP_SECONDS: Final = 300
MARGIN_MINUTES: Final = 10
# One entry of the Options panel of the help starts a line: the panel's left border, a space, a star
# for a required option, two spaces and the option (the prose of the help never starts a line so).
PANEL_BORDER: Final = chr(0x2502)
OPTION_ENTRY: Final = re.compile(
    rf"^{PANEL_BORDER} [* ] {{2}}(--[a-z][a-z-]*)", re.MULTILINE
)


# The four triggers of the workflow and the cadence each must give: the context of a manual dispatch
# holds the input and no schedule, the context of a schedule holds the cron string that fired.
CADENCE_CASES: Final[list[tuple[dict[str, str], str]]] = [
    ({"github.event.inputs.cadence": "weekly"}, "weekly"),
    ({"github.event.inputs.cadence": "nightly"}, "nightly"),
    ({"github.event.schedule": SUNDAY_CRON}, "weekly"),
    ({"github.event.schedule": NIGHTLY_CRON}, "nightly"),
]


# ---- reading a workflow ---------------------------------------------------------------------------------------------


def load(text: str) -> dict[Any, Any]:
    document = yaml.safe_load(text)
    assert isinstance(document, dict)
    return document


def steps_of(document: dict[Any, Any]) -> list[dict[str, Any]]:
    return [step for job in document["jobs"].values() for step in job["steps"]]


def uses_of(document: dict[Any, Any]) -> list[str]:
    return [step["uses"] for step in steps_of(document) if "uses" in step]


def runs_of(document: dict[Any, Any]) -> list[str]:
    return [step["run"] for step in steps_of(document) if "run" in step]


def pinned(text: str) -> dict[str, str]:
    """The version of each action a workflow uses, by action."""
    return {
        action: version
        for action, _, version in (use.partition("@") for use in uses_of(load(text)))
    }


def shape_of(step: Mapping[str, Any]) -> dict[str, Any]:
    """A step without its name and without the version of its action."""
    shape = {key: value for key, value in step.items() if key != "name"}
    if "uses" in shape:
        shape["uses"] = shape["uses"].partition("@")[0]
    return shape


def options_in(help_text: str) -> frozenset[str]:
    """The options the Options panel of a help text lists (not the ones its prose mentions)."""
    return frozenset(OPTION_ENTRY.findall(help_text))


@functools.cache
def check_options() -> frozenset[str]:
    """The options of `rinchen maintain check`, read from its help."""
    result = CliRunner().invoke(
        app, ["maintain", "check", "--help"], terminal_width=200
    )
    assert result.exit_code == 0
    return options_in(result.output)


def minimum_timeout(config: MaintainConfig) -> int:
    """The least `timeout-minutes` of the job: every rerun, the lock check, the drift step and the
    set-up, in whole minutes, and a margin."""
    seconds = (
        sum(rerun.timeout_seconds for rerun in config.reruns)
        + config.lock_check.timeout_seconds
        + config.drift.timeout_seconds
        + DOCTOR_AND_SETUP_SECONDS
    )
    return math.ceil(seconds / 60) + MARGIN_MINUTES


# ---- the scan -------------------------------------------------------------------------------------------------------


def has_the_cadence_choice(dispatch: object) -> bool:
    inputs = dispatch.get("inputs") if isinstance(dispatch, dict) else None
    if not isinstance(inputs, dict) or set(inputs) != {"cadence"}:
        return False
    entry = inputs["cadence"]
    return (
        isinstance(entry, dict)
        and {key: entry.get(key) for key in CADENCE_CHOICE} == CADENCE_CHOICE
    )


def trigger_problems(trigger: object) -> list[str]:
    if not isinstance(trigger, dict) or set(trigger) != {
        "schedule",
        "workflow_dispatch",
    }:
        return ["the trigger is not the schedule and the manual dispatch"]
    problems = []
    schedule = trigger["schedule"] if isinstance(trigger["schedule"], list) else []
    crons = [entry.get("cron") for entry in schedule if isinstance(entry, dict)]
    if crons != [NIGHTLY_CRON, SUNDAY_CRON]:
        problems.append("the crons are not the nightly one and the Sunday one")
    if not has_the_cadence_choice(trigger["workflow_dispatch"]):
        problems.append(
            "the manual dispatch has no cadence choice of nightly and weekly"
        )
    return problems


def permission_problems(document: dict[Any, Any]) -> list[str]:
    problems = []
    if document.get("permissions") != {"contents": "read"}:
        problems.append("the permissions are not contents: read")
    problems += [
        f"job {name} sets its own permissions"
        for name, job in document["jobs"].items()
        if "permissions" in job
    ]
    return problems


def key_problems(document: dict[Any, Any]) -> list[str]:
    problems = []
    if set(document) != TOP_KEYS:
        problems.append(
            "the workflow holds keys other than name, on, permissions and jobs"
        )
    problems += [
        f"job {name} holds keys other than runs-on, timeout-minutes, env and steps"
        for name, job in document["jobs"].items()
        if set(job) != JOB_KEYS
    ]
    return problems


def timeout_problems(name: str, timeout: object, minimum: int) -> list[str]:
    if not isinstance(timeout, int) or isinstance(timeout, bool):
        return [f"job {name} has no timeout"]
    if timeout < minimum:
        return [f"job {name} has a timeout below the weekly bound of {minimum} minutes"]
    return []


def cadence_problems(env: object) -> list[str]:
    """The environment of the job is the cadence and the cadence it gives is the trigger's."""
    if not isinstance(env, dict) or set(env) != {"CADENCE"}:
        return ["the job sets an environment other than CADENCE"]
    try:
        chosen = [
            Expression(env["CADENCE"], context).value() for context, _ in CADENCE_CASES
        ]
    except ValueError, AttributeError:
        chosen = []
    if chosen != [cadence for _, cadence in CADENCE_CASES]:
        return ["the CADENCE expression does not give the cadence of the trigger"]
    return []


def job_problems(jobs: dict[str, Any], config: MaintainConfig) -> list[str]:
    problems = []
    if list(jobs) != [JOB]:
        problems.append(f"the workflow does not hold the one job {JOB}")
    minimum = minimum_timeout(config)
    for name, job in jobs.items():
        if job.get("runs-on") != "ubuntu-24.04":
            problems.append(f"job {name} is not on ubuntu-24.04")
        problems += timeout_problems(name, job.get("timeout-minutes"), minimum)
    if JOB in jobs:
        problems += cadence_problems(jobs[JOB].get("env"))
    return problems


def action_problems(document: dict[Any, Any], ci_text: str) -> list[str]:
    problems = []
    known = pinned(ci_text)
    for use in uses_of(document):
        action, _, version = use.partition("@")
        if not USES.fullmatch(use):
            problems.append("an action is not pinned to a version tag")
        elif known.get(action) != version:
            problems.append("an action is not at the version ci.yml uses")
        if action == "actions/upload-artifact":
            problems.append("a step uploads an artifact")
        if "cache" in action:
            problems.append("a step caches")
    return problems


def check_problems(run: str) -> list[str]:
    """The command of the check: the one of the spec, naming only options the command has."""
    problems = []
    if run != MAINTAIN_RUN:
        problems.append(
            'the check does not get --out-dir maintain-out and --cadence "$CADENCE"'
        )
    try:
        tokens = shlex.split(run)
    except ValueError:
        return [*problems, "the check is not a well-formed shell line"]
    command_tokens = (
        tokens[tokens.index("check") + 1 :] if "check" in tokens else tokens
    )
    problems += [
        f"the check is given an option the command lacks: {token}"
        for token in command_tokens
        if token.startswith("--") and token not in check_options()
    ]
    return problems


def summary_problems(step: dict[str, Any]) -> list[str]:
    condition = str(step.get("if", ""))
    problems = []
    if "always()" not in condition:
        problems.append("the summary step does not run when an earlier step failed")
    if REPORT_FILE not in condition:
        problems.append("the summary step does not wait for the report")
    return problems


def run_problems(run: str) -> list[str]:
    problems = []
    if "uv sync" in run and run != SYNC_RUN:
        problems.append(f"a uv sync is not {SYNC_RUN}")
    problems += [
        f"a run holds {forbidden!r}"
        for forbidden in FORBIDDEN_IN_RUN
        if forbidden in run
    ]
    if run.startswith(CHECK_PREFIX):
        problems += check_problems(run)
    return problems


def step_problems(document: dict[Any, Any]) -> list[str]:
    problems = []
    maintain_steps = document["jobs"].get(JOB, {}).get("steps", [])
    if [shape_of(step) for step in maintain_steps] != EXPECTED_STEPS:
        problems.append("the steps are not the ones of the nightly workflow, in order")
    for step in steps_of(document):
        run = step.get("run")
        if run is None:
            continue
        problems += run_problems(run)
        if "GITHUB_STEP_SUMMARY" in run:
            problems += summary_problems(step)
    return problems


def text_problems(text: str) -> list[str]:
    problems = []
    if "secrets." in text:
        problems.append("a step names a secret")
    lowered = text.lower()
    problems += [f"it mentions {word}" for word in FORBIDDEN_WORDS if word in lowered]
    return problems


def findings(text: str, ci_text: str, config: MaintainConfig) -> list[str]:
    """What is wrong with the workflow `text`, one short sentence each (none for a good one)."""
    document = load(text)
    return [
        *trigger_problems(document.get(True)),
        *permission_problems(document),
        *key_problems(document),
        *job_problems(document["jobs"], config),
        *action_problems(document, ci_text),
        *step_problems(document),
        *text_problems(text),
    ]


def maintain_step_run(document: dict[Any, Any]) -> str:
    """The `run:` of the one step that runs the check."""
    (run,) = [run for run in runs_of(document) if run.startswith(CHECK_PREFIX)]
    return run
