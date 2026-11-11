"""`.github/workflows/release.yml` read as data (D13, Task 10, spec T19; the workflow is never run here:
UNMEASURED). A tag starts the checks and the build and nothing uploads: the trigger is the push of a
tag `v*` and nothing else, the permissions are `contents: read`, every job is on `ubuntu-24.04` with a
timeout, every action is pinned like `ci.yml` pins it (the same action at the same version tag), no
step names a secret or prints the environment, no `run:` uploads to the Hub, nothing mentions
docker, a GPU or a self-hosted runner, and the steps are the gate, the build and the dry-run test.

PyYAML reads the key `on` as the boolean `True` (it does so in `ci.yml` too), so the trigger is
looked up under `True`. The scan is a function of the text of a workflow, `findings`, and is itself
tested on workflows that break each rule.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Final

import pytest
import yaml

from tests.support.no_network import no_network  # noqa: F401 - the fixture of the mark

pytestmark = pytest.mark.usefixtures("no_network")

ROOT: Final = Path(__file__).resolve().parents[2]
WORKFLOWS: Final = ROOT / ".github" / "workflows"
RELEASE: Final = WORKFLOWS / "release.yml"
CI: Final = WORKFLOWS / "ci.yml"
USES: Final = re.compile(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+@v\d+(?:\.\d+)*")
DRY_RUN_STEP: Final = "uv run pytest tests/integration/test_release_dry_run.py -q"
# Said by a `run:` of a workflow that must not be said: the upload, an echo of a variable, a dump
# of the environment, a trace of the commands.
FORBIDDEN_IN_RUN: Final = ("--upload", "echo $", "printenv", "set -x", "env |")
# Mentioned nowhere, comments included (looked up in lower case).
FORBIDDEN_WORDS: Final = ("docker", "gpu", "cuda", "nvidia", "self-hosted")


def load(text: str) -> dict[Any, Any]:
    document = yaml.safe_load(text)
    assert isinstance(document, dict)
    return document


def uses_of(document: dict[Any, Any]) -> list[str]:
    return [
        step["uses"]
        for job in document["jobs"].values()
        for step in job["steps"]
        if "uses" in step
    ]


def runs_of(document: dict[Any, Any]) -> list[str]:
    return [
        step["run"]
        for job in document["jobs"].values()
        for step in job["steps"]
        if "run" in step
    ]


def pinned(text: str) -> dict[str, str]:
    """The version of each action a workflow uses, by action."""
    return {
        action: version
        for action, _, version in (use.partition("@") for use in uses_of(load(text)))
    }


def findings(text: str, ci_text: str) -> list[str]:
    """What is wrong with the workflow `text`, one short sentence each (none for a good one)."""
    problems: list[str] = []
    document = load(text)
    if document.get(True) != {"push": {"tags": ["v*"]}}:
        problems.append("the trigger is not the push of a tag v*")
    if document.get("permissions") != {"contents": "read"}:
        problems.append("the permissions are not contents: read")
    known = pinned(ci_text)
    for use in uses_of(document):
        action, _, version = use.partition("@")
        if not USES.fullmatch(use):
            problems.append("an action is not pinned to a version tag")
        elif known.get(action) != version:
            problems.append("an action is not at the version ci.yml uses")
    for name, job in document["jobs"].items():
        if job.get("runs-on") != "ubuntu-24.04":
            problems.append(f"job {name} is not on ubuntu-24.04")
        timeout = job.get("timeout-minutes")
        if not isinstance(timeout, int) or isinstance(timeout, bool) or timeout <= 0:
            problems.append(f"job {name} has no timeout")
        if "permissions" in job:
            problems.append(f"job {name} sets its own permissions")
    if "secrets." in text:
        problems.append("a step names a secret")
    for run in runs_of(document):
        problems += [
            f"a run holds {forbidden!r}"
            for forbidden in FORBIDDEN_IN_RUN
            if forbidden in run
        ]
    lowered = text.lower()
    problems += [f"it mentions {word}" for word in FORBIDDEN_WORDS if word in lowered]
    return problems


@pytest.fixture(scope="module")
def release_text() -> str:
    return RELEASE.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def ci_text() -> str:
    return CI.read_text(encoding="utf-8")


# ---- the workflow ---------------------------------------------------------------------------------------------------


def test_the_workflow_parses_and_breaks_none_of_the_rules(
    release_text: str, ci_text: str
) -> None:
    assert findings(release_text, ci_text) == []


def test_the_trigger_is_the_push_of_a_tag_v_star_and_nothing_else(
    release_text: str,
) -> None:
    document = load(release_text)

    assert "on" not in document, "PyYAML reads the key `on` as True"
    assert document[True] == {"push": {"tags": ["v*"]}}


def test_the_permissions_are_exactly_contents_read(release_text: str) -> None:
    document = load(release_text)

    assert document["permissions"] == {"contents": "read"}
    assert all("permissions" not in job for job in document["jobs"].values())


def test_every_action_is_pinned_to_a_version_tag_and_to_the_version_ci_uses(
    release_text: str, ci_text: str
) -> None:
    release_uses = uses_of(load(release_text))
    known = pinned(ci_text)

    assert release_uses
    for use in release_uses:
        assert USES.fullmatch(use), use
        action, _, version = use.partition("@")
        assert known[action] == version, use


def test_every_job_is_on_ubuntu_24_04_with_a_timeout(release_text: str) -> None:
    jobs = load(release_text)["jobs"]

    assert jobs
    for job in jobs.values():
        assert job["runs-on"] == "ubuntu-24.04"
        assert isinstance(job["timeout-minutes"], int) and job["timeout-minutes"] > 0


def test_no_step_names_a_secret_or_prints_the_environment_or_uploads(
    release_text: str,
) -> None:
    assert "secrets." not in release_text
    for run in runs_of(load(release_text)):
        for forbidden in FORBIDDEN_IN_RUN:
            assert forbidden not in run, forbidden


def test_nothing_mentions_docker_a_gpu_or_a_self_hosted_runner(
    release_text: str,
) -> None:
    lowered = release_text.lower()

    for word in FORBIDDEN_WORDS:
        assert word not in lowered, word


def test_the_steps_are_the_gate_the_build_and_the_dry_run_of_the_upload(
    release_text: str,
) -> None:
    runs = runs_of(load(release_text))

    assert "just gate" in runs
    assert "uv build" in runs
    assert DRY_RUN_STEP in runs
    assert runs.index("just gate") < runs.index("uv build") < runs.index(DRY_RUN_STEP)


def test_the_dry_run_step_names_a_test_module_that_exists() -> None:
    assert (ROOT / "tests" / "integration" / "test_release_dry_run.py").is_file()


def test_the_workflow_runs_nothing_that_needs_the_hub_token(release_text: str) -> None:
    lowered = release_text.lower()

    assert "hf_token" not in lowered and "huggingface" not in lowered


# ---- the scan, on workflows that break one rule each ------------------------------------------------------------------


def broken(release_text: str, old: str, new: str) -> str:
    assert release_text.count(old) == 1, old
    return release_text.replace(old, new)


BREAKS: Final[dict[str, tuple[str, str, str]]] = {
    "an action on a branch": (
        "uses: actions/checkout@v7",
        "uses: actions/checkout@main",
        "an action is not pinned to a version tag",
    ),
    "an action at another version": (
        "uses: actions/checkout@v7",
        "uses: actions/checkout@v6",
        "an action is not at the version ci.yml uses",
    ),
    "an action ci.yml does not use": (
        "uses: actions/checkout@v7",
        "uses: someone/else@v1",
        "an action is not at the version ci.yml uses",
    ),
    "a secret in a run": (
        "run: just doctor",
        "run: echo ${{ secrets.HF_TOKEN }}",
        "a step names a secret",
    ),
    "an upload": (
        "run: uv build",
        "run: rinchen release hub --stage s --repo-id a/b --out-dir o --upload",
        "a run holds '--upload'",
    ),
    "an echo of a variable": (
        "run: uv build",
        "run: echo $HOME",
        "a run holds 'echo $'",
    ),
    "a dump of the environment": (
        "run: uv build",
        "run: printenv",
        "a run holds 'printenv'",
    ),
    "a trace of the commands": (
        "run: uv build",
        "run: set -x && uv build",
        "a run holds 'set -x'",
    ),
    "a pipe from the environment": (
        "run: uv build",
        "run: env | sort",
        "a run holds 'env |'",
    ),
    "write permission": (
        "permissions:\n  contents: read",
        "permissions:\n  contents: write",
        "the permissions are not contents: read",
    ),
    "a branch trigger": (
        'tags:\n      - "v*"',
        "branches: [main]",
        "the trigger is not the push of a tag v*",
    ),
    "a pull request trigger": (
        "on:\n  push:",
        "on:\n  pull_request:\n  push:",
        "the trigger is not the push of a tag v*",
    ),
    "another runner": (
        "runs-on: ubuntu-24.04",
        "runs-on: ubuntu-latest",
        "job verify-and-build is not on ubuntu-24.04",
    ),
    "a self-hosted runner": (
        "runs-on: ubuntu-24.04",
        "runs-on: [self-hosted]",
        "it mentions self-hosted",
    ),
    "no timeout": (
        "    timeout-minutes: 90\n",
        "",
        "job verify-and-build has no timeout",
    ),
    "a container image": (
        "- name: Build the distributions",
        "- name: Build the docker image",
        "it mentions docker",
    ),
    "a graphics card": (
        "- name: Install just",
        "- name: Install just for the GPU",
        "it mentions gpu",
    ),
    "a job with its own permissions": (
        "    timeout-minutes: 90\n",
        "    timeout-minutes: 90\n    permissions:\n      contents: write\n",
        "job verify-and-build sets its own permissions",
    ),
}


@pytest.mark.parametrize("case", sorted(BREAKS))
def test_the_scan_finds_a_workflow_that_breaks_a_rule(
    release_text: str, ci_text: str, case: str
) -> None:
    old, new, expected = BREAKS[case]

    found = findings(broken(release_text, old, new), ci_text)

    assert expected in found
