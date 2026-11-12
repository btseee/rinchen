"""The rules of the files of D15's Task 8 as a function of their texts (spec P16): the two jobs of
`.github/workflows/ci.yml`, the `tiny` recipe of the justfile, the ignore rule of `artifacts/`, and the
two things the other workflows owe the new job (the same version of every action in the three
workflows, a nightly timeout that covers the pipeline). The workflows are never run here (the runs on
the runner are recorded in the comment above the job tiny), and the rules only say what their text must
be.

`findings(files, tiny_config, maintain_config)` says what is wrong with a set of files, one short
sentence each, and nothing for a good set; the readers and the constants it judges by are here too,
for `tests/unit/test_ci_workflow.py`, which asserts them on the files and on broken copies of them
(`tests/support/ci_workflow_breaks.py`). The rules of D14's scan of `nightly.yml` stay in
`tests/support/nightly_workflow.py` and are reused from there (the readers, the words and runs no
workflow says, the secret rule); here is only what D15 adds to them.

PyYAML reads the key `on` as the boolean `True`, so the trigger is looked up under `True`.
"""

from __future__ import annotations

import functools
import math
import shlex
from collections.abc import Mapping
from dataclasses import dataclass, fields
from typing import Any, Final

import typer.main
import yaml

from rinchen.cli.main import app
from rinchen.config.maintain import MaintainConfig
from rinchen.config.tiny import TinyConfig
from tests.support.nightly_workflow import (
    SYNC_RUN,
    USES,
    load,
    minimum_timeout,
    run_problems,
    text_problems,
)

GATE_JOB: Final = "gate"
TINY_JOB: Final = "tiny"
NIGHTLY_JOB: Final = "maintain"
RUNNER: Final = "ubuntu-24.04"
TINY_TIMEOUT_MINUTES: Final = 45
# The least timeout of the tiny job: the budget of the pipeline in whole minutes, and ten minutes
# for the checkout, the set-up and the sync.
TINY_MARGIN_MINUTES: Final = 10
TINY_RUN: Final = "just tiny"
TINY_RECIPE: Final = "uv run --frozen --no-sync rinchen tiny run --profile tiny"
STUB_TEXTS: Final = ("stub until D15", "not implemented until D15")
REPORT_JSON: Final = "artifacts/tiny/report.json"
REPORT_MARKDOWN: Final = "artifacts/tiny/report.md"
UPLOAD_PATHS: Final = [REPORT_JSON, REPORT_MARKDOWN]
UPLOAD_NAME: Final = "tiny-report"
UPLOAD_OPTIONS: Final = {"name", "path", "if-no-files-found"}
SUMMARY_RUN: Final = f'cat {REPORT_MARKDOWN} >> "$GITHUB_STEP_SUMMARY"'
SUMMARY_REPORT: Final = f"hashFiles('{REPORT_MARKDOWN}')"
SUMMARY_IF: Final = f"${{{{ always() && {SUMMARY_REPORT} != '' }}}}"
IGNORE_COMMENT: Final = "# The output directory of the tiny pipeline (just tiny)"
IGNORE_LINE: Final = "/artifacts/"
CI_TRIGGER: Final = {"push": {"branches": ["main"]}, "pull_request": None}
CI_TOP_KEYS: Final = {"name", True, "jobs"}
GATE_JOB_KEYS: Final = {"runs-on", "steps"}
TINY_JOB_KEYS: Final = {"runs-on", "timeout-minutes", "permissions", "steps"}
# Said by a `run:` of the recipe that the recipe must not say.
RECIPE_FORBIDDEN: Final = (
    "--upload",
    "docker",
    "nvidia",
    "uv lock",
    "--upgrade",
    "uv add",
    "uv pip",
)
# Mentioned nowhere in the file (looked up in lower case): the Hugging Face hub and a model download.
HUB_WORDS: Final = (
    "huggingface",
    "hf_token",
    "hf_home",
    "hf download",
    "snapshot_download",
    "from_pretrained",
)
# Said by no `run:` of the tiny job: a download by hand.
DOWNLOADS: Final = ("curl ", "wget ", "git clone", "git lfs")

# The steps of the job `gate` as they were before the pipeline got its own job (without the version
# of an action: the versions are compared across the three workflows), in order.
GATE_STEPS: Final[list[dict[str, Any]]] = [
    {"uses": "actions/checkout"},
    {"name": "Install uv", "uses": "astral-sh/setup-uv"},
    {
        "name": "Set up Python 3.14",
        "uses": "actions/setup-python",
        "with": {"python-version": "3.14"},
    },
    {"name": "Install just", "uses": "extractions/setup-just"},
    {"name": "Sync dependencies", "run": SYNC_RUN},
    {"name": "Gate (ruff format, ruff check, mypy, fast tests)", "run": "just gate"},
    {
        "name": "Doctor (environment checks must all pass or skip, never fail)",
        "run": "just doctor",
    },
    {
        "name": "Upload coverage",
        "if": "always()",
        "uses": "actions/upload-artifact",
        "with": {
            "name": "coverage",
            "path": "coverage.xml",
            "if-no-files-found": "ignore",
        },
    },
]
# The steps of the job `tiny`, in order: the set-up of `gate`, the sync, the pipeline, the upload of
# the two files of the report and the copy of the Markdown to the run page.
TINY_STEPS: Final[list[dict[str, Any]]] = [
    *GATE_STEPS[:5],
    {"name": "Tiny pipeline", "run": TINY_RUN},
    {
        "name": "Upload the tiny report",
        "if": "always()",
        "uses": "actions/upload-artifact",
        "with": {
            "name": UPLOAD_NAME,
            "path": "\n".join(UPLOAD_PATHS),
            "if-no-files-found": "ignore",
        },
    },
    {
        "name": "Show the report on the run page",
        "if": SUMMARY_IF,
        "run": SUMMARY_RUN,
    },
]
# The steps of the nightly job by name (None for the checkout): the old ones with the stub renamed.
NIGHTLY_STEP_NAMES: Final = [
    None,
    "Install uv",
    "Set up Python 3.14",
    "Install just",
    "Sync dependencies",
    "Maintenance check (doctor, lock, dependency drift, parity reruns)",
    "Tiny pipeline",
    "Show the report on the run page",
]


@dataclass(frozen=True)
class Files:
    """The texts of the files the rules judge; a broken copy replaces one of them."""

    ci: str
    nightly: str
    release: str
    justfile: str
    gitignore: str

    def replacing(self, name: str, text: str) -> Files:
        texts = {field.name: getattr(self, field.name) for field in fields(self)}
        return Files(**{**texts, name: text})


LABELS: Final = {
    "ci": "ci.yml",
    "nightly": "nightly.yml",
    "release": "release.yml",
    "justfile": "justfile",
    "gitignore": ".gitignore",
}


# ---- reading --------------------------------------------------------------------------------------------------------


def load_ci(text: str) -> dict[Any, Any]:
    """The document of a workflow (its trigger under the key `True`)."""
    return load(text)


def tiny_job(workflow: Mapping[Any, Any]) -> dict[str, Any]:
    """The job `tiny` of a workflow document."""
    job = workflow["jobs"][TINY_JOB]
    assert isinstance(job, dict)
    return job


def minimum_tiny_timeout(config: TinyConfig) -> int:
    """The least `timeout-minutes` of the job `tiny`: the budget of the pipeline in whole minutes
    and ten minutes of set-up."""
    return math.ceil(config.budget_seconds / 60) + TINY_MARGIN_MINUTES


def nightly_minimum_timeout(tiny: TinyConfig, maintain: MaintainConfig) -> int:
    """The least `timeout-minutes` of the nightly job: the least of the check (D14) and the
    budget of the pipeline, which `just tiny` runs after it."""
    return minimum_timeout(maintain) + math.ceil(tiny.budget_seconds / 60)


def read(text: str) -> dict[Any, Any] | None:
    """The document of `text`, or None when it is not a YAML mapping."""
    try:
        document = yaml.safe_load(text)
    except yaml.YAMLError:
        return None
    return document if isinstance(document, dict) else None


def job_steps(job: object) -> list[dict[str, Any]]:
    steps = job.get("steps") if isinstance(job, dict) else None
    if not isinstance(steps, list):
        return []
    return [step for step in steps if isinstance(step, dict)]


def uses_in(document: dict[Any, Any] | None) -> list[str]:
    """The `uses:` of every step of every job of a workflow document, in order."""
    jobs = document.get("jobs") if document else None
    if not isinstance(jobs, dict):
        return []
    return [
        step["uses"]
        for job in jobs.values()
        for step in job_steps(job)
        if isinstance(step.get("uses"), str)
    ]


def identity_of(step: Mapping[str, Any]) -> dict[str, Any]:
    """A step as it is, without the version of its action and with its upload paths as one line each."""
    identity = dict(step)
    uses = identity.get("uses")
    if isinstance(uses, str):
        identity["uses"] = uses.partition("@")[0]
    options = identity.get("with")
    if isinstance(options, dict) and isinstance(options.get("path"), str):
        lines = [line.strip() for line in options["path"].splitlines()]
        identity["with"] = {**options, "path": "\n".join(filter(None, lines))}
    return identity


def runs_in(steps: list[dict[str, Any]]) -> list[str]:
    return [step["run"] for step in steps if isinstance(step.get("run"), str)]


@functools.cache
def tiny_run_options() -> frozenset[str]:
    """The options of `rinchen tiny run`, read from the command object (nothing is invoked)."""
    # Typer ships its own command classes, so the groups are used by their `commands` and not by type.
    group: Any = typer.main.get_command(app)
    run = group.commands["tiny"].commands["run"]
    return frozenset(
        opt for param in run.params for opt in param.opts if opt.startswith("--")
    )


def recipe_body(text: str, name: str) -> list[str] | None:
    """The indented lines under the one recipe `name` of a justfile, or None unless it has one."""
    lines = text.splitlines()
    starts = [
        number for number, line in enumerate(lines) if line.rstrip() == f"{name}:"
    ]
    if len(starts) != 1:
        return None
    body = []
    for line in lines[starts[0] + 1 :]:
        if not line.strip():
            continue
        if line[0] not in " \t":
            break
        body.append(line.strip())
    return body


# ---- the justfile and the ignore rule -------------------------------------------------------------------------------


def recipe_problems(text: str) -> list[str]:
    body = recipe_body(text, TINY_JOB)
    if body is None:
        return ["the justfile does not hold exactly one tiny recipe"]
    problems = []
    if body != [TINY_RECIPE]:
        problems.append(f"the tiny recipe is not {TINY_RECIPE}")
    joined = "\n".join(body).lower()
    problems += [
        f"the tiny recipe holds {word!r}" for word in RECIPE_FORBIDDEN if word in joined
    ]
    try:
        tokens = shlex.split(joined)
    except ValueError:
        return [*problems, "the tiny recipe is not a well-formed shell line"]
    after = tokens[tokens.index("rinchen") :] if "rinchen" in tokens else []
    problems += [
        f"the tiny recipe is given an option the command lacks: {token}"
        for token in after
        if token.startswith("--") and token not in tiny_run_options()
    ]
    return problems


def ignore_problems(text: str) -> list[str]:
    lines = text.splitlines()
    problems = []
    if IGNORE_LINE not in lines:
        return ["the output directory of the tiny pipeline is not ignored at the root"]
    at = lines.index(IGNORE_LINE)
    if at == 0 or lines[at - 1] != IGNORE_COMMENT:
        problems.append(
            "the ignore rule of the output directory has no comment above it"
        )
    if any(line.startswith("!") and "artifacts" in line for line in lines):
        problems.append(
            "a rule of .gitignore takes a path of the output directory back"
        )
    return problems


# ---- ci.yml ---------------------------------------------------------------------------------------------------------


def gate_problems(job: object) -> list[str]:
    if not isinstance(job, dict):
        return ["job gate is not a mapping"]
    steps = job_steps(job)
    problems = []
    if job.get("runs-on") != RUNNER:
        problems.append(f"job gate is not on {RUNNER}")
    if set(job) != GATE_JOB_KEYS:
        problems.append("job gate holds keys other than runs-on and steps")
    if [identity_of(step) for step in steps] != GATE_STEPS:
        problems.append(
            "the steps of job gate are not the ones it had before the pipeline job, in order"
        )
    if any(TINY_RUN in run for run in runs_in(steps)):
        problems.append("job gate runs the tiny pipeline")
    return problems


def timeout_problems(timeout: object, config: TinyConfig) -> list[str]:
    if not isinstance(timeout, int) or isinstance(timeout, bool):
        return ["job tiny has no timeout"]
    problems = []
    minimum = minimum_tiny_timeout(config)
    if timeout < minimum:
        problems.append(f"job tiny has a timeout below the bound of {minimum} minutes")
    if timeout != TINY_TIMEOUT_MINUTES:
        problems.append(
            f"job tiny has a timeout other than {TINY_TIMEOUT_MINUTES} minutes"
        )
    return problems


def upload_problems(steps: list[dict[str, Any]]) -> list[str]:
    uploads = [
        step
        for step in steps
        if str(step.get("uses", "")).partition("@")[0] == "actions/upload-artifact"
    ]
    if len(uploads) != 1:
        return ["job tiny does not hold exactly one upload step"]
    upload = identity_of(uploads[0])
    options = upload["with"] if isinstance(upload.get("with"), dict) else {}
    problems = []
    if upload.get("if") != "always()":
        problems.append("the upload of job tiny does not run after a failure")
    if set(options) != UPLOAD_OPTIONS:
        problems.append(
            "the upload of job tiny sets options other than name, path and if-no-files-found"
        )
    if options.get("name") != UPLOAD_NAME:
        problems.append(f"the upload of job tiny is not named {UPLOAD_NAME}")
    if str(options.get("path", "")).splitlines() != UPLOAD_PATHS:
        problems.append("the upload of job tiny is not the two files of the report")
    if options.get("if-no-files-found") != "ignore":
        problems.append("the upload of job tiny does not ignore missing files")
    return problems


def summary_problems(steps: list[dict[str, Any]]) -> list[str]:
    writers = [
        step for step in steps if "GITHUB_STEP_SUMMARY" in str(step.get("run", ""))
    ]
    if len(writers) != 1:
        return ["job tiny does not hold exactly one step that writes the run page"]
    condition = str(writers[0].get("if", ""))
    problems = []
    if writers[0].get("run") != SUMMARY_RUN:
        problems.append("the summary step of job tiny does not copy report.md")
    if "always()" not in condition:
        problems.append(
            "the summary step of job tiny does not run when an earlier step failed"
        )
    if SUMMARY_REPORT not in condition:
        problems.append("the summary step of job tiny does not wait for the report")
    return problems


def step_problems(steps: list[dict[str, Any]]) -> list[str]:
    problems = []
    if [identity_of(step) for step in steps] != TINY_STEPS:
        problems.append(
            "the steps of job tiny are not the ones of the pipeline, in order"
        )
    if runs_in(steps).count(TINY_RUN) != 1:
        problems.append(f"job tiny does not run {TINY_RUN} exactly once")
    if any("continue-on-error" in step for step in steps):
        problems.append("a step of job tiny ignores a failure")
    if any("env" in step for step in steps):
        problems.append("a step of job tiny sets an environment")
    if any(_caches(step) for step in steps):
        problems.append("a step of job tiny caches")
    if any(command in run for run in runs_in(steps) for command in DOWNLOADS):
        problems.append("a step of job tiny downloads something")
    return problems + upload_problems(steps) + summary_problems(steps)


def _caches(step: Mapping[str, Any]) -> bool:
    options = step.get("with")
    names = list(options) if isinstance(options, dict) else []
    return any(
        "cache" in str(text)
        for text in (step.get("uses", ""), step.get("run", ""), *names)
    )


def job_problems(job: object, config: TinyConfig) -> list[str]:
    if not isinstance(job, dict):
        return ["job tiny is not a mapping"]
    problems = []
    if job.get("runs-on") != RUNNER:
        problems.append(f"job tiny is not on {RUNNER}")
    if set(job) != TINY_JOB_KEYS:
        problems.append(
            "job tiny holds keys other than runs-on, timeout-minutes, permissions and steps"
        )
    if job.get("permissions") != {"contents": "read"}:
        problems.append("job tiny does not have the permissions contents: read")
    if "needs" in job:
        problems.append("job tiny waits for another job")
    if "continue-on-error" in job:
        problems.append("job tiny ignores a failure")
    if "env" in job:
        problems.append("job tiny sets an environment")
    problems += timeout_problems(job.get("timeout-minutes"), config)
    return problems + step_problems(job_steps(job))


def ci_problems(text: str, config: TinyConfig) -> list[str]:
    document = read(text)
    if document is None:
        return ["the file is not a YAML mapping"]
    jobs = document.get("jobs") if isinstance(document.get("jobs"), dict) else {}
    problems = []
    if document.get(True) != CI_TRIGGER:
        problems.append("the trigger is not a push to main and pull requests")
    if set(document) != CI_TOP_KEYS:
        problems.append("the workflow holds keys other than name, on and jobs")
    if list(jobs) != [GATE_JOB, TINY_JOB]:
        problems.append("the jobs are not gate and tiny, in order")
    if GATE_JOB in jobs:
        problems += gate_problems(jobs[GATE_JOB])
    if TINY_JOB in jobs:
        problems += job_problems(jobs[TINY_JOB], config)
    for job in jobs.values():
        for run in runs_in(job_steps(job)):
            problems += run_problems(run)
    problems += text_problems(text)
    if any(word in text.lower() for word in HUB_WORDS):
        problems.append("the file reaches the Hugging Face hub or downloads a model")
    return problems


# ---- the other workflows --------------------------------------------------------------------------------------------


def nightly_problems(
    text: str, tiny: TinyConfig, maintain: MaintainConfig
) -> list[str]:
    document = read(text)
    jobs = (
        document.get("jobs")
        if document and isinstance(document.get("jobs"), dict)
        else {}
    )
    job = jobs.get(NIGHTLY_JOB)
    if not isinstance(job, dict):
        return ["the workflow does not hold the job maintain"]
    problems = []
    if [step.get("name") for step in job_steps(job)] != NIGHTLY_STEP_NAMES:
        problems.append(
            "the names of the steps are not the old ones with the tiny pipeline renamed"
        )
    timeout = job.get("timeout-minutes")
    minimum = nightly_minimum_timeout(tiny, maintain)
    if not isinstance(timeout, int) or isinstance(timeout, bool) or timeout < minimum:
        problems.append(
            f"job maintain has a timeout below the check plus the pipeline, {minimum} minutes"
        )
    return problems


def version_problems(texts: Mapping[str, str]) -> list[str]:
    """Every action of the workflows is pinned to a version tag, and each at one version in all."""
    versions: dict[str, set[str]] = {}
    problems = []
    for label, text in texts.items():
        for use in uses_in(read(text)):
            action, _, version = use.partition("@")
            if USES.fullmatch(use):
                versions.setdefault(action, set()).add(version)
            else:
                problems.append(f"{label}: an action is not pinned to a version tag")
    problems += [
        f"the workflows name more than one version of {action}"
        for action, seen in sorted(versions.items())
        if len(seen) > 1
    ]
    return problems


# ---- the whole -------------------------------------------------------------------------------------------------------


def findings(files: Files, tiny: TinyConfig, maintain: MaintainConfig) -> list[str]:
    """What is wrong with the set of files, one short sentence each (none for a good set)."""
    found = [f"ci.yml: {sentence}" for sentence in ci_problems(files.ci, tiny)]
    found += [
        f"nightly.yml: {sentence}"
        for sentence in nightly_problems(files.nightly, tiny, maintain)
    ]
    found += version_problems(
        {"ci.yml": files.ci, "nightly.yml": files.nightly, "release.yml": files.release}
    )
    found += [
        f"{LABELS[name]} still names the stub of the pipeline"
        for name in LABELS
        if any(stub in getattr(files, name) for stub in STUB_TEXTS)
    ]
    found += recipe_problems(files.justfile)
    found += ignore_problems(files.gitignore)
    return found
