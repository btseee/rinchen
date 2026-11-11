"""The parity reruns of `rinchen.maintain.reruns` (D14, spec M8): the argument lists, the decision
of a pytest rerun from the exit code and the JUnit counters, and the parity rerun over an export.

Everything here runs against the recording `FakeRunner` of `tests/support/maintain_fakes.py`, so
no process starts: a test scripts the exit code, the log and the files a command would have
left (a JUnit report, a `parity.json`) and reads the state, the reason and the figures back. The
rerun over the real pytest is in `test_maintain_junit.py`. The decision is also compared, over a
grid of exit codes and counters that a real pytest never produces together (exit 0 with a
failure in the report), with a small independent model of the rule of M8.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest

from rinchen.config.loader import REPO_ROOT
from rinchen.config.maintain import RerunConfig, load_maintain_config
from rinchen.errors import ConfigError
from rinchen.maintain.commands import HIDDEN_ENVIRONMENT, CommandResult
from rinchen.maintain.reruns import (
    ExportInputs,
    RerunResult,
    TestCounts,
    parity_argv,
    pytest_argv,
    run_pytest_rerun,
)
from rinchen.maintain.states import (
    FAIL_NO_JUNIT,
    fail_floor,
    fail_pytest_exit,
    fail_skipped,
    fail_tests,
    fail_timeout,
)
from tests.support.maintain_fakes import (
    FakeRunner,
    junit_xml,
    parity_rerun,
    pytest_rerun,
)
from tests.support.maintain_project import (
    EMPTY_TARGET,
    PASSES_A,
    PASSES_B,
    make_project,
)
from tests.support.no_network import no_network  # noqa: F401 - the fixture of the mark

pytestmark = pytest.mark.usefixtures("no_network")

SHIPPED = REPO_ROOT / "configs" / "maintain" / "maintain.yaml"
NUMERICS_TARGETS = [
    "tests/unit/test_metrics_segment_stats.py",
    "tests/unit/test_metrics_corpus_scores.py",
    "tests/unit/test_metrics_bleu_chrf.py",
    "tests/property/test_metrics_segment_stats_property.py",
    "tests/unit/test_eval_bootstrap.py",
    "tests/unit/test_eval_bootstrap_reference.py",
    "tests/unit/test_eval_bootstrap_determinism.py",
    "tests/property/test_eval_bootstrap_property.py",
    "tests/unit/test_eval_golden.py",
    "tests/unit/test_eval_report_golden.py",
    "tests/unit/test_eval_suite_golden.py",
    "tests/unit/test_eval_significance.py",
    "tests/unit/test_eval_robust_numbers.py",
    "tests/unit/test_eval_robust_placeholders.py",
    "tests/unit/test_eval_robust_script.py",
    "tests/golden",
]


def shipped_rerun(rerun_id: str) -> RerunConfig:
    config = load_maintain_config(SHIPPED.read_bytes())
    return next(rerun for rerun in config.reruns if rerun.id == rerun_id)


# ---- the argument lists ----------------------------------------------------------------------------


def test_the_pytest_argument_list_of_the_shipped_numerics_rerun_is_the_interface_literally() -> (
    None
):
    argv = pytest_argv(
        "/usr/bin/python3.14",
        shipped_rerun("numerics"),
        Path("/work/out/junit/numerics.xml"),
    )

    assert argv == [
        "/usr/bin/python3.14",
        "-m",
        "pytest",
        "-q",
        "-m",
        "not network",
        "-p",
        "no:cacheprovider",
        "--junitxml=/work/out/junit/numerics.xml",
        *NUMERICS_TARGETS,
    ]


def test_the_pytest_argument_list_is_a_new_list_each_time() -> None:
    rerun = pytest_rerun(PASSES_A)

    first = pytest_argv("py", rerun, Path("/j.xml"))
    first.append("--changed")

    assert pytest_argv("py", rerun, Path("/j.xml"))[-1] == PASSES_A


def test_the_parity_argument_list_of_the_shipped_rerun_is_the_interface_literally() -> (
    None
):
    inputs = ExportInputs(
        export=Path("/data/export"),
        checkpoint=Path("/data/ckpt"),
        corpora=(Path("/data/a.parquet"), Path("/data/b.parquet")),
    )

    argv = parity_argv(
        "/usr/bin/python3.14",
        shipped_rerun("parity_export"),
        inputs,
        Path("/work/out/parity_export"),
    )

    assert argv == [
        "/usr/bin/python3.14",
        "-m",
        "rinchen",
        "export",
        "parity",
        "--checkpoint",
        "/data/ckpt",
        "--export",
        "/data/export",
        "--corpus",
        "/data/a.parquet",
        "--corpus",
        "/data/b.parquet",
        "--out-dir",
        "/work/out/parity_export",
        "--max-lines",
        "50",
        "--threads",
        "2",
        "--timeout",
        "570",
    ]


@pytest.mark.parametrize(
    ("bound", "given"),
    [(600, "570"), (100, "70"), (31, "1"), (30, "1"), (1, "1")],
)
def test_the_parity_command_gets_the_bound_less_thirty_seconds_and_never_less_than_one(
    bound: int, given: str
) -> None:
    inputs = ExportInputs(Path("/e"), Path("/c"), (Path("/f"),))

    argv = parity_argv("py", parity_rerun(timeout=bound), inputs, Path("/o"))

    assert argv[-2:] == ["--timeout", given]


# ---- the results ---------------------------------------------------------------------------------------


def test_a_result_is_frozen_and_holds_counters_and_figures_that_default_to_none() -> (
    None
):
    result = RerunResult("demo", "pytest", "nightly", "skipped", "a reason")

    assert (result.tests, result.figures) == (None, None)
    with pytest.raises(dataclasses.FrozenInstanceError):
        result.state = "ok"  # type: ignore[misc]
    assert dataclasses.astuple(TestCounts(1, 2, 3, 4)) == (1, 2, 3, 4)
    assert (
        TestCounts.__test__ is False
    )  # pytest must not collect the counters as a test


# ---- a pytest rerun over a fake runner ---------------------------------------------------------------


def rerun_with(
    tmp_path: Path,
    runner: FakeRunner,
    rerun: RerunConfig | None = None,
    *,
    python: str = "/opt/python/bin/python",
    env: dict[str, str] | None = None,
) -> tuple[RerunResult, Path, Path]:
    project = make_project(tmp_path)
    out = tmp_path / "out"
    result = run_pytest_rerun(
        pytest_rerun(PASSES_A, PASSES_B, min_passed=2) if rerun is None else rerun,
        runner=runner,
        python=python,
        project_dir=project,
        out_dir=out,
        parent_env={"KEEP": "me"} if env is None else env,
    )
    return result, project, out


def test_the_runner_gets_the_project_the_hidden_environment_the_bound_and_the_list(
    tmp_path: Path,
) -> None:
    rerun = pytest_rerun(PASSES_A, PASSES_B, min_passed=2, timeout=77)
    runner = FakeRunner().script(CommandResult(0, False), junit=junit_xml(2, 0, 0, 0))

    result, project, out = rerun_with(
        tmp_path,
        runner,
        rerun,
        env={"KEEP": "me", "CUDA_VISIBLE_DEVICES": "0", "NO_COLOR": "0"},
    )

    (call,) = runner.calls
    assert call.cwd == project
    assert call.timeout == 77
    assert call.argv == tuple(
        pytest_argv("/opt/python/bin/python", rerun, out / "junit" / "demo.xml")
    )
    assert call.log == out / "logs" / "demo.log"
    assert call.env == {
        "KEEP": "me",
        "NO_COLOR": "0",  # the uv switches are for uv alone
        **HIDDEN_ENVIRONMENT,
    }
    assert (result.state, result.reason) == ("ok", None)
    assert result.tests == TestCounts(2, 0, 0, 0)


def test_the_result_names_the_rerun_and_holds_no_path(tmp_path: Path) -> None:
    runner = FakeRunner().script(
        CommandResult(1, False), junit=junit_xml(3, 1, 0, 0), log_text="/home/x/secret"
    )

    result, _, out = rerun_with(tmp_path, runner)

    assert (result.id, result.kind, result.cadence) == ("demo", "pytest", "nightly")
    assert str(tmp_path) not in repr(result)
    assert str(out) not in repr(result)


# (what the runner returned, the report it wrote, the floor, the allowance) -> the decision
J = junit_xml
ROWS = [
    (
        "all as hoped",
        CommandResult(0, False),
        J(5, 0, 0, 0),
        5,
        0,
        "ok",
        None,
        (5, 0, 0, 0),
    ),
    (
        "more than the floor",
        CommandResult(0, False),
        J(9, 0, 0, 1),
        5,
        1,
        "ok",
        None,
        (8, 0, 0, 1),
    ),
    (
        "exit 0 against a failure in the report",
        CommandResult(0, False),
        J(5, 1, 0, 0),
        1,
        0,
        "fail",
        fail_tests(1, 0),
        (4, 1, 0, 0),
    ),
    (
        "exit 0 against errors in the report",
        CommandResult(0, False),
        J(5, 0, 2, 0),
        1,
        0,
        "fail",
        fail_tests(0, 2),
        (3, 0, 2, 0),
    ),
    (
        "a failure and an error",
        CommandResult(1, False),
        J(5, 1, 1, 0),
        1,
        0,
        "fail",
        fail_tests(1, 1),
        (3, 1, 1, 0),
    ),
    (
        "exit 0 below the floor",
        CommandResult(0, False),
        J(3, 0, 0, 0),
        5,
        0,
        "fail",
        fail_floor(3, 5),
        (3, 0, 0, 0),
    ),
    (
        "exit 0 above the allowance",
        CommandResult(0, False),
        J(5, 0, 0, 2),
        3,
        1,
        "fail",
        fail_skipped(2, 1),
        (3, 0, 0, 2),
    ),
    (
        "exit 1 against a clean report",
        CommandResult(1, False),
        J(5, 0, 0, 0),
        5,
        0,
        "fail",
        fail_pytest_exit(1),
        (5, 0, 0, 0),
    ),
    (
        "exit 2",
        CommandResult(2, False),
        J(5, 0, 0, 0),
        5,
        0,
        "fail",
        fail_pytest_exit(2),
        (5, 0, 0, 0),
    ),
    (
        "exit 4, nothing collected",
        CommandResult(4, False),
        J(0, 0, 0, 0),
        1,
        0,
        "fail",
        fail_floor(0, 1),
        (0, 0, 0, 0),
    ),
    (
        "exit 5, nothing collected",
        CommandResult(5, False),
        J(0, 0, 0, 0),
        1,
        0,
        "fail",
        fail_floor(0, 1),
        (0, 0, 0, 0),
    ),
    (
        "killed by a signal",
        CommandResult(-9, False),
        J(5, 0, 0, 0),
        5,
        0,
        "fail",
        fail_pytest_exit(-9),
        (5, 0, 0, 0),
    ),
    (
        "a failure outranks the floor",
        CommandResult(1, False),
        J(5, 2, 0, 0),
        9,
        0,
        "fail",
        fail_tests(2, 0),
        (3, 2, 0, 0),
    ),
    (
        "the floor outranks the allowance",
        CommandResult(0, False),
        J(5, 0, 0, 4),
        5,
        0,
        "fail",
        fail_floor(1, 5),
        (1, 0, 0, 4),
    ),
    (
        "the allowance outranks the exit code",
        CommandResult(1, False),
        J(7, 0, 0, 2),
        5,
        1,
        "fail",
        fail_skipped(2, 1),
        (5, 0, 0, 2),
    ),
    (
        "a timeout",
        CommandResult(None, True),
        None,
        5,
        0,
        "fail",
        fail_timeout(60),
        None,
    ),
    (
        "a timeout over a report that was written",
        CommandResult(None, True),
        J(5, 0, 0, 0),
        5,
        0,
        "fail",
        fail_timeout(60),
        None,
    ),
    (
        "no process",
        CommandResult(None, False, False),
        None,
        5,
        0,
        "fail",
        fail_pytest_exit(-1),
        None,
    ),
    ("no report", CommandResult(0, False), None, 5, 0, "fail", FAIL_NO_JUNIT, None),
    (
        "a report that is not XML",
        CommandResult(0, False),
        b"garbage",
        5,
        0,
        "fail",
        FAIL_NO_JUNIT,
        None,
    ),
    (
        "a report that is nonsense",
        CommandResult(0, False),
        J(1, 2, 0, 0),
        1,
        0,
        "fail",
        FAIL_NO_JUNIT,
        None,
    ),
]


@pytest.mark.parametrize(
    ("outcome", "report", "floor", "allowance", "state", "reason", "counts"),
    [row[1:] for row in ROWS],
    ids=[row[0] for row in ROWS],
)
def test_the_decision_follows_the_exit_code_and_the_counters_in_order(
    tmp_path: Path,
    outcome: CommandResult,
    report: bytes | None,
    floor: int,
    allowance: int,
    state: str,
    reason: str | None,
    counts: tuple[int, int, int, int] | None,
) -> None:
    runner = FakeRunner().script(outcome, junit=report)
    rerun = pytest_rerun(PASSES_A, min_passed=floor, max_skipped=allowance, timeout=60)

    result, _, _ = rerun_with(tmp_path, runner, rerun)

    assert (result.state, result.reason) == (state, reason)
    assert result.tests == (None if counts is None else TestCounts(*counts))
    assert result.figures is None


def test_a_stale_report_from_an_earlier_run_is_removed_and_never_counts(
    tmp_path: Path,
) -> None:
    project = make_project(tmp_path)
    out = tmp_path / "out"
    stale = out / "junit" / "demo.xml"
    stale.parent.mkdir(parents=True)
    stale.write_bytes(junit_xml(99, 0, 0, 0))
    existed_at_the_call: list[bool] = []

    def look(argv: object) -> bool:
        existed_at_the_call.append(stale.exists())
        return True

    runner = FakeRunner().script(CommandResult(0, False), when=look)

    result = run_pytest_rerun(
        pytest_rerun(PASSES_A, min_passed=1),
        runner=runner,
        python="py",
        project_dir=project,
        out_dir=out,
        parent_env={},
    )

    assert existed_at_the_call == [False]
    assert (result.state, result.reason, result.tests) == ("fail", FAIL_NO_JUNIT, None)
    assert not stale.exists()


def test_a_missing_target_fails_before_the_runner_is_called(tmp_path: Path) -> None:
    runner = FakeRunner()
    rerun = pytest_rerun(PASSES_A, "tests/no_such_file.py", min_passed=1)

    result, _, out = rerun_with(tmp_path, runner, rerun)

    assert (result.state, result.reason, result.tests) == (
        "fail",
        "a target does not exist",
        None,
    )
    assert runner.calls == []
    assert not out.exists()


def test_every_target_is_looked_for_and_the_first_one_that_exists_is_not_enough(
    tmp_path: Path,
) -> None:
    runner = FakeRunner()

    result, _, _ = rerun_with(
        tmp_path, runner, pytest_rerun("tests/no_such_dir", PASSES_A, min_passed=1)
    )

    assert result.reason == "a target does not exist"
    assert runner.calls == []


def test_a_directory_target_that_exists_is_a_target(tmp_path: Path) -> None:
    runner = FakeRunner().script(CommandResult(0, False), junit=junit_xml(1, 0, 0, 0))

    result, _, _ = rerun_with(
        tmp_path, runner, pytest_rerun(EMPTY_TARGET, min_passed=1)
    )

    assert result.state == "ok"


def test_a_relative_output_directory_is_made_absolute_for_the_child(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The child runs in the project directory: a relative path would name another place there.
    project = make_project(tmp_path)
    monkeypatch.chdir(tmp_path)
    runner = FakeRunner().script(CommandResult(0, False), junit=junit_xml(1, 0, 0, 0))

    result = run_pytest_rerun(
        pytest_rerun(PASSES_A, min_passed=1),
        runner=runner,
        python="py",
        project_dir=project,
        out_dir=Path("out"),
        parent_env={},
    )

    (call,) = runner.calls
    assert f"--junitxml={tmp_path / 'out' / 'junit' / 'demo.xml'}" in call.argv
    assert call.log == tmp_path / "out" / "logs" / "demo.log"
    assert result.state == "ok"
    assert (tmp_path / "out" / "junit" / "demo.xml").is_file()


def test_a_pytest_rerun_is_refused_for_a_rerun_of_another_kind(tmp_path: Path) -> None:
    runner = FakeRunner()

    with pytest.raises(ConfigError, match="demo"):
        rerun_with(tmp_path, runner, parity_rerun(id="demo"))

    assert runner.calls == []


def test_a_pytest_rerun_is_refused_for_a_config_of_the_other_kind_that_has_a_floor(
    tmp_path: Path,
) -> None:
    # The loader never builds one (an export_parity rerun has no floor); the guard is of the
    # function, which does not rely on the loader.
    rerun = RerunConfig.model_construct(
        id="demo",
        kind="export_parity",
        cadence="nightly",
        timeout_seconds=60,
        why="x",
        targets=[PASSES_A],
        min_passed=1,
        max_skipped=0,
    )
    runner = FakeRunner()

    with pytest.raises(ConfigError, match="demo"):
        rerun_with(tmp_path, runner, rerun)

    assert runner.calls == []


# ---- the decision against a model --------------------------------------------------------------------


def model(
    exit_code: int | None,
    timed_out: bool,
    counts: tuple[int, int, int, int] | None,
    floor: int,
    allowance: int,
) -> tuple[str, str | None]:
    """The rule of M8 as a list of conditions in order; the first that holds decides."""
    if timed_out:
        return "fail", fail_timeout(60)
    if exit_code is None:
        return "fail", fail_pytest_exit(-1)
    if counts is None or counts[0] - counts[1] - counts[2] - counts[3] < 0:
        return "fail", FAIL_NO_JUNIT  # no report, or one that counts more than it ran
    tests, failures, errors, skipped = counts
    passed = tests - failures - errors - skipped
    conditions = [
        (failures + errors > 0, fail_tests(failures, errors)),
        (passed < floor, fail_floor(passed, floor)),
        (skipped > allowance, fail_skipped(skipped, allowance)),
        (exit_code != 0, fail_pytest_exit(exit_code)),
    ]
    return next(
        (("fail", reason) for holds, reason in conditions if holds), ("ok", None)
    )


def grid_of_counts() -> list[tuple[int, int, int, int] | None]:
    """No report, and every report of these counters (some of them count more than they ran)."""
    grid: list[tuple[int, int, int, int] | None] = [None]
    for tests in (0, 3, 8):
        for failures in (0, 1):
            for errors in (0, 1):
                for skipped in (0, 2, 9):
                    grid.append((tests, failures, errors, skipped))
    return grid


def test_the_decision_equals_the_model_over_a_grid_of_outcomes(tmp_path: Path) -> None:
    project = make_project(tmp_path)
    out = (
        tmp_path / "out"
    )  # one directory for all: each run must not see the last one's report
    exit_codes = (0, 1, 2, 4, 5, -9, None)
    floors = (1, 3, 8)
    allowances = (0, 2)
    checked = 0
    for exit_code in exit_codes:
        for timed_out in (False, True):
            for counts in grid_of_counts():
                for floor in floors:
                    for allowance in allowances:
                        report = None if counts is None else junit_xml(*counts)
                        runner = FakeRunner().script(
                            CommandResult(exit_code, timed_out), junit=report
                        )
                        rerun = pytest_rerun(
                            PASSES_A, min_passed=floor, max_skipped=allowance
                        )
                        result = run_pytest_rerun(
                            rerun,
                            runner=runner,
                            python="py",
                            project_dir=project,
                            out_dir=out,
                            parent_env={},
                        )
                        expected = model(exit_code, timed_out, counts, floor, allowance)
                        assert (result.state, result.reason) == expected, (
                            exit_code,
                            timed_out,
                            counts,
                            floor,
                            allowance,
                        )
                        checked += 1
    assert checked == (
        len(exit_codes) * 2 * len(grid_of_counts()) * len(floors) * len(allowances)
    )
