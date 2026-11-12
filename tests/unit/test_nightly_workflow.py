"""`.github/workflows/nightly.yml` read as data (D14, Task 6, spec M14; the workflow is never run
here: UNMEASURED). A schedule starts the maintenance check and nothing else changes: the trigger is
the two schedules (every night, and Sunday) and the manual dispatch with a cadence choice, the
permissions are `contents: read`, the one job is on `ubuntu-24.04` with a timeout that covers the
weekly bounds of `configs/maintain/maintain.yaml`, every action is pinned like `ci.yml` pins it (the
same action at the same version tag), the steps are the eight of the spec and no more, the command
of the check names only options the command has, and no step names a secret, prints the environment,
uploads, caches, pushes, upgrades or touches the lock file; nothing mentions docker, a GPU or a
self-hosted runner.

PyYAML reads the key `on` as the boolean `True` (it does so in `ci.yml` too), so the trigger is
looked up under `True`. The scan is a function of the text of a workflow, `findings`
(`tests/support/nightly_workflow.py`, which holds the readers and the constants too), and is itself
tested here on workflows that break each rule. The cadence is an expression of GitHub Actions;
`tests/support/workflow_expression.py` models the three operators it uses (`||`, `&&`, `==`) and
evaluates it for the four triggers.
"""

from __future__ import annotations

import re
import shlex
from pathlib import Path
from typing import Final

import pytest
import yaml

from rinchen.config.maintain import MaintainConfig, load_maintain_config
from tests.support.nightly_workflow import (
    CADENCE_CASES,
    CADENCE_CHOICE,
    EXPECTED_STEPS,
    FORBIDDEN_IN_RUN,
    FORBIDDEN_WORDS,
    JOB,
    MAINTAIN_RUN,
    NIGHTLY_CRON,
    PANEL_BORDER,
    REPORT_FILE,
    SUMMARY_IF,
    SUMMARY_RUN,
    SUNDAY_CRON,
    SYNC_RUN,
    TOP_KEYS,
    USES,
    check_options,
    findings,
    load,
    maintain_step_run,
    minimum_timeout,
    options_in,
    pinned,
    runs_of,
    shape_of,
    steps_of,
    uses_of,
)
from tests.support.no_network import no_network  # noqa: F401 - the fixture of the mark
from tests.support.workflow_expression import Expression

pytestmark = pytest.mark.usefixtures("no_network")

ROOT: Final = Path(__file__).resolve().parents[2]
WORKFLOWS: Final = ROOT / ".github" / "workflows"
NIGHTLY: Final = WORKFLOWS / "nightly.yml"
CI: Final = WORKFLOWS / "ci.yml"
SHIPPED_CONFIG: Final = ROOT / "configs" / "maintain" / "maintain.yaml"


@pytest.fixture(scope="module")
def nightly_text() -> str:
    return NIGHTLY.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def ci_text() -> str:
    return CI.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def config() -> MaintainConfig:
    return load_maintain_config(SHIPPED_CONFIG.read_bytes())


# ---- the workflow ---------------------------------------------------------------------------------------------------


def test_the_workflow_parses_and_breaks_none_of_the_rules(
    nightly_text: str, ci_text: str, config: MaintainConfig
) -> None:
    assert findings(nightly_text, ci_text, config) == []


def test_the_workflow_is_plain_ascii(nightly_text: str) -> None:
    assert nightly_text.isascii()


def test_the_workflow_is_named_nightly_and_holds_the_one_job_maintain(
    nightly_text: str,
) -> None:
    document = load(nightly_text)

    assert document["name"] == "Nightly"
    assert list(document["jobs"]) == [JOB]
    assert set(document) == TOP_KEYS


def test_the_trigger_is_the_two_schedules_and_the_manual_dispatch_and_nothing_else(
    nightly_text: str,
) -> None:
    document = load(nightly_text)
    trigger = document[True]

    assert "on" not in document, "PyYAML reads the key `on` as True"
    assert set(trigger) == {"schedule", "workflow_dispatch"}
    assert trigger["schedule"] == [{"cron": NIGHTLY_CRON}, {"cron": SUNDAY_CRON}]
    dispatch = trigger["workflow_dispatch"]
    assert set(dispatch) == {"inputs"}
    assert set(dispatch["inputs"]) == {"cadence"}
    cadence = dispatch["inputs"]["cadence"]
    assert {key: cadence[key] for key in CADENCE_CHOICE} == CADENCE_CHOICE


def test_the_first_cron_is_every_night_and_the_second_is_sunday(
    nightly_text: str,
) -> None:
    crons = [entry["cron"] for entry in load(nightly_text)[True]["schedule"]]

    assert [len(cron.split()) for cron in crons] == [5, 5]
    minute, hour, day, month, weekday = crons[0].split()
    assert minute.isdigit() and hour.isdigit()
    assert (day, month, weekday) == ("*", "*", "*")
    assert crons[1].split()[4] == "0"


def test_the_permissions_are_exactly_contents_read(nightly_text: str) -> None:
    document = load(nightly_text)

    assert document["permissions"] == {"contents": "read"}
    assert all("permissions" not in job for job in document["jobs"].values())


def test_every_action_is_pinned_to_a_version_tag_and_to_the_version_ci_uses(
    nightly_text: str, ci_text: str
) -> None:
    uses = uses_of(load(nightly_text))
    known = pinned(ci_text)

    assert [use.partition("@")[0] for use in uses] == [
        "actions/checkout",
        "astral-sh/setup-uv",
        "actions/setup-python",
        "extractions/setup-just",
    ]
    for use in uses:
        assert USES.fullmatch(use), use
        action, _, version = use.partition("@")
        assert known[action] == version, use


def test_the_job_is_on_ubuntu_24_04(nightly_text: str) -> None:
    assert load(nightly_text)["jobs"][JOB]["runs-on"] == "ubuntu-24.04"


def test_the_timeout_covers_the_weekly_bounds_of_the_shipped_configuration(
    nightly_text: str, config: MaintainConfig
) -> None:
    timeout = load(nightly_text)["jobs"][JOB]["timeout-minutes"]

    assert isinstance(timeout, int) and not isinstance(timeout, bool)
    assert timeout >= minimum_timeout(config)


@pytest.mark.parametrize(
    ("last_rerun_seconds", "minutes"),
    [(900, 67), (901, 68)],
    ids=["exactly 57 minutes", "one second over 57 minutes"],
)
def test_the_minimum_timeout_adds_every_bound_the_set_up_and_the_margin(
    last_rerun_seconds: int, minutes: int
) -> None:
    reruns = [
        {"kind": "export_parity", "timeout_seconds": 600, "cadence": "nightly"},
        {"kind": "export_parity", "timeout_seconds": 1200, "cadence": "weekly"},
        {
            "kind": "export_parity",
            "timeout_seconds": last_rerun_seconds,
            "cadence": "weekly",
        },
    ]
    config = MaintainConfig.model_validate(
        {
            "skipped_doctor_checks": {},
            "lock_check": {"uv_args": ["lock", "--check"], "timeout_seconds": 120},
            "drift": {
                "uv_args": ["lock", "--upgrade", "--dry-run"],
                "timeout_seconds": 300,
                "rules": [],
            },
            "reruns": [
                {"id": f"rerun_{number}", "why": "parity", **rerun}
                for number, rerun in enumerate(reruns)
            ],
        }
    )

    # 600 + 1200 + 900 + 120 + 300 + 300 = 3420 s = 57 min, so 57 whole minutes and the margin of
    # 10; one second more is 58 whole minutes.
    assert minimum_timeout(config) == minutes


def test_the_options_of_the_check_command_are_read_from_its_help() -> None:
    options = check_options()

    assert {"--out-dir", "--cadence", "--no-reruns", "--help"} <= options
    assert all(re.fullmatch(r"--[a-z][a-z-]*", option) for option in options)


def test_an_option_is_an_entry_of_the_options_panel_and_not_a_mention_in_prose() -> (
    None
):
    border = PANEL_BORDER
    help_text = "\n".join(
        [
            "Prose that mentions --not-an-option and, wrapped,",
            "--also-not-an-option at the start of a line.",
            f"{border} *  --out-dir  <path>  Where; unlike --prose-option, a column.   {border}",
            f"{border}    --cadence  <str>   Which; see",
            f"{border}                       --wrapped-mention in the column.{border}",
            f"{border}    --help             Show this message and exit.{border}",
        ]
    )

    assert options_in(help_text) == {"--out-dir", "--cadence", "--help"}


def test_the_check_runs_with_an_output_directory_and_the_cadence_variable(
    nightly_text: str,
) -> None:
    run = maintain_step_run(load(nightly_text))
    tokens = shlex.split(run)
    uv_options = [
        token for token in tokens[: tokens.index("rinchen")] if token.startswith("--")
    ]
    options = [
        token for token in tokens[tokens.index("check") :] if token.startswith("--")
    ]

    assert run == MAINTAIN_RUN
    assert uv_options == ["--frozen", "--no-sync"], "uv neither locks nor syncs"
    assert options == ["--out-dir", "--cadence"]
    assert set(options) <= check_options()
    assert tokens[tokens.index("--out-dir") + 1] == "maintain-out"
    assert tokens[tokens.index("--cadence") + 1] == "$CADENCE"
    assert '"$CADENCE"' in run, "the variable is quoted for the shell"


def test_the_steps_are_the_eight_of_the_spec_in_order(nightly_text: str) -> None:
    steps = load(nightly_text)["jobs"][JOB]["steps"]

    assert [shape_of(step) for step in steps] == EXPECTED_STEPS


def test_just_tiny_runs_after_the_check_so_that_the_pipeline_is_picked_up_without_an_edit(
    nightly_text: str,
) -> None:
    runs = runs_of(load(nightly_text))

    assert runs.index(MAINTAIN_RUN) < runs.index("just tiny") < runs.index(SUMMARY_RUN)


def test_the_summary_step_runs_always_and_only_when_the_report_exists(
    nightly_text: str,
) -> None:
    (step,) = [
        step for step in steps_of(load(nightly_text)) if step.get("run") == SUMMARY_RUN
    ]

    assert step["if"] == SUMMARY_IF
    assert "always()" in step["if"]
    assert REPORT_FILE in step["if"]


def test_the_cadence_is_one_expression_over_both_crons_and_both_cadences(
    nightly_text: str,
) -> None:
    env = load(nightly_text)["jobs"][JOB]["env"]

    assert set(env) == {"CADENCE"}
    for text in (SUNDAY_CRON, "weekly", "nightly"):
        assert text in env["CADENCE"], text


def test_the_cadence_expression_gives_the_cadence_of_each_trigger(
    nightly_text: str,
) -> None:
    expression = load(nightly_text)["jobs"][JOB]["env"]["CADENCE"]

    chosen = [Expression(expression, context).value() for context, _ in CADENCE_CASES]

    assert chosen == ["weekly", "nightly", "weekly", "nightly"]


# ---- what the workflow never does -----------------------------------------------------------------------------------


def test_no_text_of_the_file_names_a_secret(nightly_text: str) -> None:
    assert "secrets." not in nightly_text


def test_no_run_uploads_pushes_upgrades_changes_the_lock_or_prints_the_environment(
    nightly_text: str,
) -> None:
    for run in runs_of(load(nightly_text)):
        for forbidden in FORBIDDEN_IN_RUN:
            assert forbidden not in run, forbidden


def test_the_only_uv_sync_is_the_frozen_one_of_the_dev_group(nightly_text: str) -> None:
    syncs = [run for run in runs_of(load(nightly_text)) if "uv sync" in run]

    assert syncs == [SYNC_RUN]


def test_no_step_uploads_an_artifact_or_caches(nightly_text: str) -> None:
    for use in uses_of(load(nightly_text)):
        assert use.partition("@")[0] != "actions/upload-artifact"
        assert "cache" not in use
    for step in steps_of(load(nightly_text)):
        assert not any("cache" in key for key in step.get("with", {})), step


def test_nothing_mentions_docker_a_gpu_or_a_self_hosted_runner(
    nightly_text: str,
) -> None:
    lowered = nightly_text.lower()

    for word in FORBIDDEN_WORDS:
        assert word not in lowered, word


# ---- the scan, on workflows that break one rule each ------------------------------------------------------------------


def broken(nightly_text: str, old: str, new: str) -> str:
    assert nightly_text.count(old) == 1, old
    return nightly_text.replace(old, new)


STEPS: Final = "the steps are not the ones of the nightly workflow, in order"
TRIGGER: Final = "the trigger is not the schedule and the manual dispatch"
CRONS: Final = "the crons are not the nightly one and the Sunday one"
CADENCE: Final = "the CADENCE expression does not give the cadence of the trigger"
DISPATCH: Final = "the manual dispatch has no cadence choice of nightly and weekly"
CADENCE_INPUT: Final = (
    '      cadence:\n        description: "nightly or weekly"\n'
    "        type: choice\n        options: [nightly, weekly]\n"
    "        default: nightly\n"
)
JOB_KEYS_SENTENCE: Final = (
    "job maintain holds keys other than runs-on, timeout-minutes, env and steps"
)
RUN_JUST_TINY: Final = "run: just tiny"


def expression(choice: str, dispatch: str = "github.event.inputs.cadence || ") -> str:
    """The expression of the job's CADENCE: the dispatch input, else `choice` by the schedule."""
    on_sunday = f"github.event.schedule == '{SUNDAY_CRON}'"
    return "${{ " + dispatch + "(" + on_sunday + " && " + choice + ") }}"


def after_tiny(step: str) -> str:
    """The `just tiny` step followed by one more step."""
    return f"{RUN_JUST_TINY}\n\n      - {step}"


EXPRESSION: Final = expression("'weekly' || 'nightly'")
CHECK_SENTENCE: Final = (
    'the check does not get --out-dir maintain-out and --cadence "$CADENCE"'
)

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
        RUN_JUST_TINY,
        "run: echo ${{ secrets.HF_TOKEN }}",
        "a step names a secret",
    ),
    "a sync without --frozen": (
        f"run: {SYNC_RUN}",
        "run: uv sync --group dev",
        f"a uv sync is not {SYNC_RUN}",
    ),
    "a check through a uv run that locks the project": (
        f"run: {MAINTAIN_RUN}",
        f"run: {MAINTAIN_RUN.replace(' --frozen --no-sync', '')}",
        "the steps are not the ones of the nightly workflow, in order",
    ),
    "a check through a uv run that syncs the environment": (
        f"run: {MAINTAIN_RUN}",
        f"run: {MAINTAIN_RUN.replace(' --no-sync', '')}",
        "the steps are not the ones of the nightly workflow, in order",
    ),
    "write permission": (
        "permissions:\n  contents: read",
        "permissions:\n  contents: write",
        "the permissions are not contents: read",
    ),
    "a pull request trigger": (
        "on:\n  schedule:",
        "on:\n  pull_request:\n  schedule:",
        TRIGGER,
    ),
    "a wrong cron": ('- cron: "47 4 * * 0"', '- cron: "47 4 * * 1"', CRONS),
    "a nightly cron that fires every hour": (
        '- cron: "17 3 * * *"',
        '- cron: "17 * * * *"',
        CRONS,
    ),
    "no schedule": (
        '  schedule:\n    - cron: "17 3 * * *"\n    - cron: "47 4 * * 0"\n',
        "  schedule:\n",
        CRONS,
    ),
    "a schedule entry that is a string": (
        '    - cron: "47 4 * * 0"',
        '    - "47 4 * * 0"',
        CRONS,
    ),
    "a dispatch without inputs": (
        f"    inputs:\n{CADENCE_INPUT}",
        "",
        DISPATCH,
    ),
    "a cadence input that is not a mapping": (
        CADENCE_INPUT,
        "      cadence: nightly\n",
        DISPATCH,
    ),
    "a cadence that is a number": (EXPRESSION, "7", CADENCE),
    "a dispatch without the choice": ("type: choice", "type: string", DISPATCH),
    "another runner": (
        "runs-on: ubuntu-24.04",
        "runs-on: ubuntu-latest",
        "job maintain is not on ubuntu-24.04",
    ),
    "a self-hosted runner": (
        "runs-on: ubuntu-24.04",
        "runs-on: [self-hosted]",
        "it mentions self-hosted",
    ),
    "no timeout": (
        "    timeout-minutes: 150\n",
        "",
        "job maintain has no timeout",
    ),
    "a timeout of 20 minutes": (
        "timeout-minutes: 150",
        "timeout-minutes: 20",
        "job maintain has a timeout below the weekly bound of {minimum} minutes",
    ),
    "a timeout that is not a number": (
        "timeout-minutes: 150",
        "timeout-minutes: true",
        "job maintain has no timeout",
    ),
    "an upload-artifact step": (
        RUN_JUST_TINY,
        after_tiny("uses: actions/upload-artifact@v7"),
        "a step uploads an artifact",
    ),
    "a cache step": (
        RUN_JUST_TINY,
        after_tiny("uses: actions/cache@v4"),
        "a step caches",
    ),
    "the word docker in a name": (
        "- name: Install just",
        "- name: Install just for docker",
        "it mentions docker",
    ),
    "a job with its own permissions": (
        "    timeout-minutes: 150\n",
        "    timeout-minutes: 150\n    permissions:\n      contents: write\n",
        "job maintain sets its own permissions",
    ),
    "a job that ignores a failure": (
        "    timeout-minutes: 150\n",
        "    timeout-minutes: 150\n    continue-on-error: true\n",
        JOB_KEYS_SENTENCE,
    ),
    "a second job": (
        "jobs:\n  maintain:",
        "jobs:\n  check:",
        "the workflow does not hold the one job maintain",
    ),
    "an environment at the top": (
        "permissions:\n  contents: read",
        "permissions:\n  contents: read\n\nenv:\n  SOMETHING: value",
        "the workflow holds keys other than name, on, permissions and jobs",
    ),
    "another variable in the job": (
        "    env:\n      CADENCE:",
        "    env:\n      OTHER: value\n      CADENCE:",
        "the job sets an environment other than CADENCE",
    ),
    "the cadence option removed from the command": (
        ' --cadence "$CADENCE"',
        "",
        CHECK_SENTENCE,
    ),
    "the cadence variable not quoted": (
        '--cadence "$CADENCE"',
        "--cadence $CADENCE",
        CHECK_SENTENCE,
    ),
    "an unclosed quote in the check": (
        '--cadence "$CADENCE"',
        '--cadence "$CADENCE',
        "the check is not a well-formed shell line",
    ),
    "an input besides the cadence": (
        "        default: nightly\n",
        "        default: nightly\n      other:\n        type: string\n",
        DISPATCH,
    ),
    "an option the command does not have": (
        "--cadence ",
        "--cadance ",
        "the check is given an option the command lacks: --cadance",
    ),
    "a step that ignores a failure": (
        RUN_JUST_TINY,
        f"{RUN_JUST_TINY}\n        continue-on-error: true",
        STEPS,
    ),
    "another Python": ('python-version: "3.14"', 'python-version: "3.13"', STEPS),
    "a step added": (
        RUN_JUST_TINY,
        after_tiny("run: just doctor"),
        STEPS,
    ),
    "no tiny pipeline step": (
        "      - name: Tiny pipeline\n        run: just tiny\n\n",
        "",
        STEPS,
    ),
    "a summary that waits for nothing": (
        SUMMARY_IF,
        "${{ always() }}",
        "the summary step does not wait for the report",
    ),
    "a summary that is skipped after a failure": (
        "${{ always() && hashFiles",
        "${{ hashFiles",
        "the summary step does not run when an earlier step failed",
    ),
    "a cadence expression that inverts the cadences": (
        EXPRESSION,
        expression("'nightly' || 'weekly'"),
        CADENCE,
    ),
    "a cadence that ignores the dispatch input": (
        EXPRESSION,
        expression("'weekly' || 'nightly'", dispatch=""),
        CADENCE,
    ),
    "a cadence that is weekly every night": (
        EXPRESSION,
        expression("'weekly' || 'weekly'"),
        CADENCE,
    ),
    "a cadence expression nobody can read": (
        EXPRESSION,
        "${{ github.event.inputs.cadence ?? 'nightly' }}",
        CADENCE,
    ),
}


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
    nightly_text: str, ci_text: str, config: MaintainConfig, forbidden: str
) -> None:
    planted = broken(nightly_text, RUN_JUST_TINY, f"run: {PLANTED_IN_A_RUN[forbidden]}")

    found = findings(planted, ci_text, config)

    assert f"a run holds {forbidden!r}" in found


@pytest.mark.parametrize("word", ["docker", "gpu", "cuda", "nvidia", "self-hosted"])
def test_the_scan_finds_each_word_that_must_not_be_mentioned_even_in_a_comment(
    nightly_text: str, ci_text: str, config: MaintainConfig, word: str
) -> None:
    planted = broken(nightly_text, "# A schedule starts", f"# A {word.upper()} starts")

    found = findings(planted, ci_text, config)

    assert f"it mentions {word}" in found


@pytest.mark.parametrize("case", sorted(BREAKS))
def test_the_scan_finds_a_workflow_that_breaks_a_rule(
    nightly_text: str, ci_text: str, config: MaintainConfig, case: str
) -> None:
    old, new, expected = BREAKS[case]

    found = findings(broken(nightly_text, old, new), ci_text, config)

    assert expected.format(minimum=minimum_timeout(config)) in found


def test_the_scan_finds_the_steps_in_another_order(
    nightly_text: str, ci_text: str, config: MaintainConfig
) -> None:
    document = load(nightly_text)
    steps = document["jobs"][JOB]["steps"]
    runs = [step.get("run") for step in steps]
    sync, check = runs.index(SYNC_RUN), runs.index(MAINTAIN_RUN)
    steps[sync], steps[check] = steps[check], steps[sync]

    found = findings(yaml.safe_dump(document, sort_keys=False), ci_text, config)

    assert found == [STEPS]
