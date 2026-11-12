"""The `tiny` job of `.github/workflows/ci.yml`, the `tiny` recipe, the ignore rule of `artifacts/` and what
the other two workflows owe them, read as data (D15, Task 8, spec P16; the workflows are never run here
and neither is a recipe: the runs on the runner are recorded in the comment above the job tiny). The recipe is the one command of the pipeline; `ci.yml`
has two jobs, `gate` with the steps it had before (without the stub) and `tiny` with its own timeout, its
read-only permissions, the set-up of `gate`, the pipeline, ONE upload of the two files of the report and
the copy of `report.md` to the run page; every action of the three workflows is at one version; the
nightly job keeps its steps (the stub renamed) and its timeout covers the check and the pipeline; and git
ignores the output directory of the pipeline.

PyYAML reads the key `on` as the boolean `True`, so the trigger is looked up under `True`. The rules are a
function of the texts, `findings` (`tests/support/ci_workflow.py`), and are themselves tested here on
copies that break each rule (`tests/support/ci_workflow_breaks.py`); the rules of D14's scan of the
nightly workflow stay in `tests/unit/test_nightly_workflow.py`.
"""

from __future__ import annotations

import math
import os
import shlex
import subprocess
from pathlib import Path
from typing import Final

import pytest
import yaml

from rinchen.cli.tiny import DEFAULT_OUT_DIR
from rinchen.cli.tiny_output import REPORT_FILE, SUMMARY_FILE
from rinchen.config.maintain import MaintainConfig, load_maintain_config
from rinchen.config.tiny import TinyConfig
from tests.support.ci_workflow import (
    GATE_STEPS,
    IGNORE_COMMENT,
    IGNORE_LINE,
    LABELS,
    NIGHTLY_JOB,
    NIGHTLY_STEP_NAMES,
    RECIPE_FORBIDDEN,
    STUB_TEXTS,
    SUMMARY_IF,
    SUMMARY_REPORT,
    SUMMARY_RUN,
    TINY_RECIPE,
    TINY_RUN,
    TINY_STEPS,
    TINY_TIMEOUT_MINUTES,
    UPLOAD_NAME,
    UPLOAD_PATHS,
    Files,
    findings,
    identity_of,
    job_steps,
    load_ci,
    minimum_tiny_timeout,
    nightly_minimum_timeout,
    recipe_body,
    tiny_job,
    tiny_run_options,
    uses_in,
)
from tests.support.ci_workflow_breaks import BREAKS
from tests.support.ci_workflow_edits import (
    JOBS,
    PIPELINE_STEP,
    TINY_STEPS_SENTENCE,
    broken,
    replace_occurrence,
    sentence_of,
)
from tests.support.nightly_workflow import (
    FORBIDDEN_IN_RUN,
    FORBIDDEN_WORDS,
    SYNC_RUN,
    minimum_timeout,
    runs_of,
)
from tests.support.nightly_workflow import (
    findings as nightly_findings,
)
from tests.support.no_network import no_network  # noqa: F401 - the fixture of the mark
from tests.support.tiny_stage_inputs import shipped_config

pytestmark = pytest.mark.usefixtures("no_network")

ROOT: Final = Path(__file__).resolve().parents[2]
WORKFLOWS: Final = ROOT / ".github" / "workflows"
SHIPPED_MAINTAIN_CONFIG: Final = ROOT / "configs" / "maintain" / "maintain.yaml"
NIGHTLY_STEPS_SENTENCE: Final = (
    "the steps are not the ones of the nightly workflow, in order"
)


@pytest.fixture(scope="module")
def files() -> Files:
    return Files(
        ci=(WORKFLOWS / "ci.yml").read_text(encoding="utf-8"),
        nightly=(WORKFLOWS / "nightly.yml").read_text(encoding="utf-8"),
        release=(WORKFLOWS / "release.yml").read_text(encoding="utf-8"),
        justfile=(ROOT / "justfile").read_text(encoding="utf-8"),
        gitignore=(ROOT / ".gitignore").read_text(encoding="utf-8"),
    )


@pytest.fixture(scope="module")
def tiny() -> TinyConfig:
    return shipped_config()


@pytest.fixture(scope="module")
def maintain() -> MaintainConfig:
    return load_maintain_config(SHIPPED_MAINTAIN_CONFIG.read_bytes())


def tiny_steps_of(files: Files) -> list[dict[str, object]]:
    return job_steps(tiny_job(load_ci(files.ci)))


# ---- the files as a whole -------------------------------------------------------------------------------------------


def test_the_files_break_none_of_the_rules(
    files: Files, tiny: TinyConfig, maintain: MaintainConfig
) -> None:
    assert findings(files, tiny, maintain) == []


@pytest.mark.parametrize("name", sorted(LABELS))
def test_every_file_is_plain_ascii(files: Files, name: str) -> None:
    assert getattr(files, name).isascii()


# ---- G8.1: the recipe -----------------------------------------------------------------------------------------------


def test_the_tiny_recipe_is_the_one_command_of_the_pipeline(files: Files) -> None:
    assert recipe_body(files.justfile, "tiny") == [TINY_RECIPE]


def test_the_recipe_runs_the_tiny_command_of_the_tiny_profile_through_a_frozen_uv_run(
    files: Files,
) -> None:
    (command,) = recipe_body(files.justfile, "tiny") or []
    tokens = shlex.split(command)

    assert tokens[: tokens.index("rinchen")] == ["uv", "run", "--frozen", "--no-sync"]
    assert tokens[tokens.index("rinchen") :] == [
        "rinchen",
        "tiny",
        "run",
        "--profile",
        "tiny",
    ]


@pytest.mark.parametrize("word", RECIPE_FORBIDDEN)
def test_the_recipe_holds_none_of_the_words_it_never_says(
    files: Files, word: str
) -> None:
    assert word not in "\n".join(recipe_body(files.justfile, "tiny") or []).lower()


def test_the_recipe_planted_words_are_the_ones_of_the_brief() -> None:
    assert set(RECIPE_FORBIDDEN) == {
        "--upload",
        "docker",
        "nvidia",
        "uv lock",
        "--upgrade",
        "uv add",
        "uv pip",
    }


def test_the_options_of_the_recipe_are_options_of_the_command(files: Files) -> None:
    (command,) = recipe_body(files.justfile, "tiny") or []
    options = {token for token in shlex.split(command) if token.startswith("--")}

    assert {"--profile"} <= options
    assert options - {"--frozen", "--no-sync"} <= tiny_run_options()


def test_the_options_of_the_command_are_read_from_the_command_and_not_from_prose() -> (
    None
):
    options = tiny_run_options()

    assert {"--profile", "--out-dir", "--dry-run", "--config"} <= options
    assert all(option.startswith("--") for option in options)
    assert "--upload" not in options


def test_the_justfile_holds_one_tiny_recipe_and_no_stub(files: Files) -> None:
    lines = files.justfile.splitlines()

    assert lines.count("tiny:") == 1
    assert not any(stub in files.justfile for stub in STUB_TEXTS)


# ---- G8.2: ci.yml ---------------------------------------------------------------------------------------------------


def test_ci_holds_the_two_jobs_gate_and_tiny_in_that_order(files: Files) -> None:
    document = load_ci(files.ci)

    assert document["name"] == "CI"
    assert list(document["jobs"]) == ["gate", "tiny"]


def test_the_trigger_is_the_push_to_main_and_the_pull_requests_as_before(
    files: Files,
) -> None:
    document = load_ci(files.ci)

    assert "on" not in document, "PyYAML reads the key `on` as True"
    assert document[True] == {"push": {"branches": ["main"]}, "pull_request": None}
    assert set(document) == {"name", True, "jobs"}


def test_the_gate_job_keeps_its_steps_and_runs_no_pipeline(files: Files) -> None:
    job = load_ci(files.ci)["jobs"]["gate"]

    assert set(job) == {"runs-on", "steps"}
    assert job["runs-on"] == "ubuntu-24.04"
    assert [identity_of(step) for step in job_steps(job)] == GATE_STEPS
    assert not any(TINY_RUN in run for run in runs_of({"jobs": {"gate": job}}))


def test_the_comment_above_the_tiny_job_records_its_measured_runs(files: Files) -> None:
    lines = files.ci.splitlines()
    head = lines.index("  tiny:")
    comment = []
    for line in reversed(lines[:head]):
        if not line.startswith("  #"):
            break
        comment.insert(0, line.removeprefix("  #").strip())
    text = " ".join(comment)

    assert "UNMEASURED" not in text
    for fact in (
        "43244678285",
        "43248539643",
        "3 min 19 s",
        "191.646 s",
        "tiny-report",
    ):
        assert fact in text, fact


def test_the_tiny_job_has_the_steps_of_the_pipeline_in_order(files: Files) -> None:
    steps = tiny_steps_of(files)

    assert [identity_of(step) for step in steps] == TINY_STEPS
    assert [step["name"] for step in steps[1:]] == [
        "Install uv",
        "Set up Python 3.14",
        "Install just",
        "Sync dependencies",
        "Tiny pipeline",
        "Upload the tiny report",
        "Show the report on the run page",
    ]


def test_the_tiny_job_is_on_ubuntu_24_04_with_read_permission_and_no_other_key(
    files: Files,
) -> None:
    job = tiny_job(load_ci(files.ci))

    assert job["runs-on"] == "ubuntu-24.04"
    assert job["permissions"] == {"contents": "read"}
    assert set(job) == {"runs-on", "timeout-minutes", "permissions", "steps"}


def test_the_tiny_timeout_is_45_and_covers_the_bound_of_the_shipped_configuration(
    files: Files, tiny: TinyConfig
) -> None:
    timeout = tiny_job(load_ci(files.ci))["timeout-minutes"]

    assert isinstance(timeout, int) and not isinstance(timeout, bool)
    assert timeout == TINY_TIMEOUT_MINUTES == 45
    assert timeout >= minimum_tiny_timeout(tiny) == 30


@pytest.mark.parametrize(
    ("budget_seconds", "minutes"),
    [(1, 11), (60, 11), (61, 12), (1200, 30), (1201, 31), (2100, 45), (2101, 46)],
    ids=lambda value: str(value),
)
def test_the_minimum_timeout_is_the_budget_in_whole_minutes_plus_ten(
    tiny: TinyConfig, budget_seconds: int, minutes: int
) -> None:
    config = tiny.model_copy(update={"budget_seconds": budget_seconds})

    assert minimum_tiny_timeout(config) == minutes


def test_the_scan_finds_the_timeout_that_a_longer_budget_outgrows(
    files: Files, tiny: TinyConfig, maintain: MaintainConfig
) -> None:
    longer = tiny.model_copy(update={"budget_seconds": 2101})

    found = findings(files, longer, maintain)

    assert "ci.yml: job tiny has a timeout below the bound of 46 minutes" in found
    assert "nightly.yml" not in " ".join(found), "150 minutes still cover 104 + 36"


def test_neither_job_waits_for_the_other(files: Files) -> None:
    jobs = load_ci(files.ci)["jobs"]

    assert all("needs" not in job and "if" not in job for job in jobs.values())


def test_the_tiny_job_uploads_the_two_files_of_the_report_once_and_ignores_a_missing_one(
    files: Files,
) -> None:
    uploads = [
        step
        for step in tiny_steps_of(files)
        if str(step.get("uses", "")).startswith("actions/upload-artifact@")
    ]

    (upload,) = uploads
    assert upload["if"] == "always()"
    assert upload["with"]["name"] == UPLOAD_NAME == "tiny-report"
    assert upload["with"]["path"].split() == UPLOAD_PATHS
    assert upload["with"]["if-no-files-found"] == "ignore"
    assert set(upload["with"]) == {"name", "path", "if-no-files-found"}


def test_the_files_of_the_upload_are_the_report_of_the_default_output_directory() -> (
    None
):
    assert UPLOAD_PATHS == [
        (DEFAULT_OUT_DIR / REPORT_FILE).as_posix(),
        (DEFAULT_OUT_DIR / SUMMARY_FILE).as_posix(),
    ]


def test_the_summary_step_runs_always_and_only_when_the_report_exists(
    files: Files,
) -> None:
    (step,) = [
        step
        for step in tiny_steps_of(files)
        if "GITHUB_STEP_SUMMARY" in str(step.get("run", ""))
    ]

    assert step["run"] == SUMMARY_RUN
    assert step["if"] == SUMMARY_IF
    assert "always()" in step["if"]
    assert SUMMARY_REPORT in step["if"]


def test_no_text_of_ci_names_a_secret_an_environment_a_cache_or_a_download(
    files: Files,
) -> None:
    lowered = files.ci.lower()
    jobs = load_ci(files.ci)["jobs"]

    assert "secrets." not in files.ci
    assert "huggingface" not in lowered and "hf_" not in lowered
    assert "cache" not in lowered and "curl" not in lowered and "wget" not in lowered
    assert all("env" not in job for job in jobs.values())
    for job in jobs.values():
        assert all("env" not in step for step in job_steps(job))


def test_the_only_uv_sync_of_each_job_is_the_frozen_one_of_the_dev_group(
    files: Files,
) -> None:
    for job in load_ci(files.ci)["jobs"].values():
        syncs = [run for run in runs_of({"jobs": {"job": job}}) if "uv sync" in run]
        assert syncs == [SYNC_RUN]


def test_no_run_of_ci_uploads_pushes_upgrades_changes_the_lock_or_prints_the_environment(
    files: Files,
) -> None:
    for run in runs_of(load_ci(files.ci)):
        for forbidden in FORBIDDEN_IN_RUN:
            assert forbidden not in run, forbidden


def test_nothing_in_ci_mentions_docker_a_gpu_or_a_self_hosted_runner(
    files: Files,
) -> None:
    lowered = files.ci.lower()

    for word in FORBIDDEN_WORDS:
        assert word not in lowered, word


def test_the_tiny_job_uses_the_actions_of_the_gate_at_the_same_versions(
    files: Files,
) -> None:
    jobs = load_ci(files.ci)["jobs"]
    gate_uses = uses_in({"jobs": {"gate": jobs["gate"]}})
    tiny_uses = uses_in({"jobs": {"tiny": jobs["tiny"]}})

    assert len(tiny_uses) == len(gate_uses) == 5
    assert tiny_uses == gate_uses, "the set-up and the upload are the gate's"


# ---- G8.3: the versions of the actions ------------------------------------------------------------------------------


def test_every_action_of_the_three_workflows_is_pinned_and_at_one_version(
    files: Files,
) -> None:
    versions: dict[str, set[str]] = {}
    for text in (files.ci, files.nightly, files.release):
        for use in uses_in(yaml.safe_load(text)):
            action, at, version = use.partition("@")
            assert at and version.startswith("v") and version[1:2].isdigit(), use
            versions.setdefault(action, set()).add(version)

    assert versions
    assert {action: len(seen) for action, seen in versions.items()} == dict.fromkeys(
        versions, 1
    )
    assert {
        "actions/checkout",
        "astral-sh/setup-uv",
        "actions/setup-python",
        "extractions/setup-just",
        "actions/upload-artifact",
    } <= set(versions)


# ---- G8.4: the nightly workflow -------------------------------------------------------------------------------------


def test_the_nightly_pipeline_step_is_named_tiny_pipeline_and_the_other_steps_keep_their_names(
    files: Files,
) -> None:
    steps = job_steps(load_ci(files.nightly)["jobs"][NIGHTLY_JOB])

    assert [step.get("name") for step in steps] == NIGHTLY_STEP_NAMES
    assert steps[6] == {"name": "Tiny pipeline", "run": TINY_RUN}


def test_no_workflow_or_recipe_names_the_stub(files: Files) -> None:
    for name in LABELS:
        for stub in STUB_TEXTS:
            assert stub not in getattr(files, name), (name, stub)


def test_the_nightly_timeout_covers_the_check_and_the_budget_of_the_pipeline(
    files: Files, tiny: TinyConfig, maintain: MaintainConfig
) -> None:
    timeout = load_ci(files.nightly)["jobs"][NIGHTLY_JOB]["timeout-minutes"]
    check = minimum_timeout(maintain)
    pipeline = math.ceil(tiny.budget_seconds / 60)

    # The figures of the records: 104 minutes for the check, 20 for the pipeline, 150 in the file.
    assert (check, pipeline, timeout) == (104, 20, 150)
    assert timeout >= check + pipeline == nightly_minimum_timeout(tiny, maintain)


@pytest.mark.parametrize(
    ("budget_seconds", "minutes"),
    [(1200, 124), (1201, 125), (60, 105), (3600, 164)],
    ids=lambda value: str(value),
)
def test_the_nightly_minimum_adds_the_budget_in_whole_minutes_to_the_check(
    tiny: TinyConfig, maintain: MaintainConfig, budget_seconds: int, minutes: int
) -> None:
    config = tiny.model_copy(update={"budget_seconds": budget_seconds})

    assert nightly_minimum_timeout(config, maintain) == minutes


def test_a_nightly_timeout_at_the_bound_is_enough_and_one_under_it_is_not(
    files: Files, tiny: TinyConfig, maintain: MaintainConfig
) -> None:
    at_the_bound = replace_occurrence(
        files.nightly, "timeout-minutes: 150", "timeout-minutes: 124", None
    )

    assert findings(files.replacing("nightly", at_the_bound), tiny, maintain) == []


def test_a_nightly_without_the_pipeline_step_is_still_found_by_the_scan_of_the_nightly(
    files: Files, maintain: MaintainConfig
) -> None:
    without = replace_occurrence(files.nightly, PIPELINE_STEP, "", None)

    found = nightly_findings(without, files.ci, maintain)

    assert NIGHTLY_STEPS_SENTENCE in found


# ---- G8.5: the scan, on copies that break one rule each ---------------------------------------------------------------


def test_the_table_holds_at_least_sixty_cases_and_each_expects_one_sentence() -> None:
    assert len(BREAKS) >= 60
    assert all(case.expected for case in BREAKS.values())


@pytest.mark.parametrize("case", sorted(BREAKS))
def test_the_scan_finds_a_set_of_files_that_breaks_a_rule(
    files: Files, tiny: TinyConfig, maintain: MaintainConfig, case: str
) -> None:
    change = BREAKS[case]
    copy = broken(files, change)

    found = findings(copy, tiny, maintain)

    assert getattr(copy, change.file) != getattr(files, change.file)
    assert sentence_of(change, tiny, maintain) in found


# A command for each text a `run:` must not hold, written out apart from FORBIDDEN_IN_RUN so that a
# text dropped from the scan is a test that fails and not a case that disappears.
PLANTED_IN_A_RUN: Final = {
    "--upload": "rinchen release hub --stage s --repo-id a/b --out-dir o --upload",
    "git push": "git push origin HEAD",
    "git commit": "git commit -am x",
    "gh ": "gh pr create",
    "uv lock": "uv lock --check",
    "uv add": "uv add requests",
    "uv pip": "uv pip install requests",
    "--upgrade": "uv lock --upgrade",
    "echo $": "echo $HOME",
    "printenv": "printenv",
    "set -x": "set -x && just tiny",
    "env |": "env | sort",
}


def test_every_text_a_run_must_not_hold_has_a_command_that_plants_it() -> None:
    assert set(PLANTED_IN_A_RUN) == set(FORBIDDEN_IN_RUN)
    for forbidden, command in PLANTED_IN_A_RUN.items():
        assert forbidden in command


@pytest.mark.parametrize("forbidden", sorted(PLANTED_IN_A_RUN))
def test_the_scan_finds_each_text_a_run_must_not_hold(
    files: Files, tiny: TinyConfig, maintain: MaintainConfig, forbidden: str
) -> None:
    run = f"        run: {PLANTED_IN_A_RUN[forbidden]}\n"
    planted = replace_occurrence(files.ci, "        run: just tiny\n", run, None)

    found = findings(files.replacing("ci", planted), tiny, maintain)

    assert f"ci.yml: a run holds {forbidden!r}" in found


@pytest.mark.parametrize("word", ["docker", "gpu", "cuda", "nvidia", "self-hosted"])
def test_the_scan_finds_each_word_that_must_not_be_mentioned_even_in_a_comment(
    files: Files, tiny: TinyConfig, maintain: MaintainConfig, word: str
) -> None:
    planted = replace_occurrence(
        files.ci, "# The tiny pipeline", f"# The {word.upper()} pipeline", None
    )

    found = findings(files.replacing("ci", planted), tiny, maintain)

    assert f"ci.yml: it mentions {word}" in found


def test_the_scan_finds_the_steps_of_the_tiny_job_in_another_order(
    files: Files, tiny: TinyConfig, maintain: MaintainConfig
) -> None:
    document = yaml.safe_load(files.ci)
    steps = document["jobs"]["tiny"]["steps"]
    runs = [step.get("run") for step in steps]
    sync, pipeline = runs.index(SYNC_RUN), runs.index(TINY_RUN)
    steps[sync], steps[pipeline] = steps[pipeline], steps[sync]
    text = yaml.safe_dump(document, sort_keys=False)

    found = findings(files.replacing("ci", text), tiny, maintain)

    assert found == [TINY_STEPS_SENTENCE]


def test_the_scan_finds_the_jobs_in_another_order(
    files: Files, tiny: TinyConfig, maintain: MaintainConfig
) -> None:
    document = yaml.safe_load(files.ci)
    document["jobs"] = {
        "tiny": document["jobs"]["tiny"],
        "gate": document["jobs"]["gate"],
    }
    text = yaml.safe_dump(document, sort_keys=False)

    found = findings(files.replacing("ci", text), tiny, maintain)

    assert found == [JOBS]


# ---- G8.6: the ignore rule ---------------------------------------------------------------------------------------------


def check_ignore(path: str) -> subprocess.CompletedProcess[str]:
    """`git check-ignore -v --no-index <path>` in the repository: read only, no network, no index."""
    environment = {
        name: value for name, value in os.environ.items() if not name.startswith("GIT_")
    }
    return subprocess.run(
        ["git", "-C", str(ROOT), "check-ignore", "-v", "--no-index", path],
        capture_output=True,
        text=True,
        stdin=subprocess.DEVNULL,
        env=environment,
        timeout=30,
        check=False,
    )


def test_the_ignore_rule_is_the_comment_and_the_directory_anchored_to_the_root(
    files: Files,
) -> None:
    lines = files.gitignore.splitlines()

    assert lines.count(IGNORE_LINE) == 1
    assert lines[lines.index(IGNORE_LINE) - 1] == IGNORE_COMMENT
    assert IGNORE_LINE == "/artifacts/"


@pytest.mark.subprocess
@pytest.mark.parametrize("name", [REPORT_FILE, SUMMARY_FILE, "manifest.json"])
def test_git_ignores_the_files_the_pipeline_writes_into_its_output_directory(
    name: str,
) -> None:
    result = check_ignore((DEFAULT_OUT_DIR / name).as_posix())

    rule, _, path = result.stdout.rstrip("\n").partition("\t")
    source, _, rest = rule.partition(":")
    _line, _, pattern = rest.partition(":")
    assert result.returncode == 0, result.stderr
    assert (source, pattern) == (".gitignore", IGNORE_LINE)
    assert path == (DEFAULT_OUT_DIR / name).as_posix()


@pytest.mark.subprocess
@pytest.mark.parametrize(
    "path",
    ["docs/artifacts/report.json", "tests/unit/test_ci_workflow.py", "artifact/x"],
)
def test_git_does_not_ignore_other_paths(path: str) -> None:
    result = check_ignore(path)

    assert result.returncode == 1, result.stdout
    assert result.stdout == ""
