"""The doubles of the D14 tests (`tests/support/maintain_fakes.py` and `maintain_project.py`),
tested as the contract that Tasks 4, 5 and 7 build on: the recording fake runner (scripted outcomes
by order or by predicate, each used once, the log replaced, files written, an unscripted call is
a failure), the JUnit document it can write, and the project on disk (its files, the targets of a
configuration, the lock and the tests it can be given).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from rinchen.config.loader import REPO_ROOT
from rinchen.config.maintain import load_maintain_config
from rinchen.maintain.commands import CommandResult
from rinchen.maintain.lock import read_lock
from rinchen.maintain.reruns import TestCounts, read_junit
from tests.support.maintain_fakes import (
    JUNIT_HOST,
    JUNIT_PATH,
    FakeRunner,
    UnscriptedCall,
    junit_path_of,
    junit_xml,
    parity_rerun,
    pytest_rerun,
)
from tests.support.maintain_project import (
    EMPTY_TARGET,
    PASSES_A,
    SAMPLE_LOCK,
    STANDARD_TESTS,
    make_project,
    parent_environment,
)
from tests.support.no_network import no_network  # noqa: F401 - the fixture of the mark
from tests.support.uv_outputs import SAMPLE_LOCKED

pytestmark = pytest.mark.usefixtures("no_network")

SHIPPED = REPO_ROOT / "configs" / "maintain" / "maintain.yaml"

OK = CommandResult(0, False)
FAILED = CommandResult(1, False)


def call(
    runner: FakeRunner, tmp_path: Path, *argv: str, log: str = "a.log"
) -> CommandResult:
    return runner.run(
        argv, cwd=tmp_path, env={"K": "v"}, log=tmp_path / "logs" / log, timeout=7
    )


# ---- the fake runner -------------------------------------------------------------------------------


def test_outcomes_without_a_predicate_are_played_in_order_and_each_once(
    tmp_path: Path,
) -> None:
    runner = FakeRunner().script(OK).script(FAILED)

    assert [call(runner, tmp_path, "x"), call(runner, tmp_path, "y")] == [OK, FAILED]
    assert runner.unused() == 0
    with pytest.raises(UnscriptedCall):
        call(runner, tmp_path, "z")


def test_a_predicate_picks_the_outcome_whatever_the_order(tmp_path: Path) -> None:
    runner = FakeRunner()
    runner.script(FAILED, when=lambda argv: argv[0] == "second")
    runner.script(OK, when=lambda argv: argv[0] == "first")

    assert call(runner, tmp_path, "first") == OK
    assert call(runner, tmp_path, "second") == FAILED
    assert runner.unused() == 0


def test_a_call_takes_the_first_unused_outcome_that_accepts_it(tmp_path: Path) -> None:
    runner = FakeRunner()
    runner.script(OK, when=lambda argv: argv[0] == "a")
    runner.script(FAILED)
    runner.script(CommandResult(2, False), when=lambda argv: argv[0] == "a")

    results = [call(runner, tmp_path, "a"), call(runner, tmp_path, "a")]

    assert results == [OK, FAILED]
    assert runner.unused() == 1


def test_a_call_that_no_outcome_accepts_is_a_failure_that_names_no_more_than_the_command(
    tmp_path: Path,
) -> None:
    runner = FakeRunner().script(OK, when=lambda argv: argv[0] == "known")

    with pytest.raises(UnscriptedCall, match="no outcome was scripted"):
        call(runner, tmp_path, "unknown", "arg")

    assert runner.unused() == 1
    assert issubclass(UnscriptedCall, AssertionError)


def test_every_call_is_recorded_as_it_was_made(tmp_path: Path) -> None:
    runner = FakeRunner().script(OK)
    env = {"K": "v"}

    runner.run(
        ["prog", "--flag"], cwd=tmp_path, env=env, log=tmp_path / "l.log", timeout=7
    )
    env["K"] = "changed afterwards"

    (recorded,) = runner.calls
    assert recorded.argv == ("prog", "--flag")
    assert recorded.cwd == tmp_path
    assert recorded.env == {"K": "v"}
    assert recorded.log == tmp_path / "l.log"
    assert recorded.timeout == 7
    assert runner.argvs == [("prog", "--flag")]


def test_a_call_is_recorded_even_when_nothing_was_scripted(tmp_path: Path) -> None:
    runner = FakeRunner()

    with pytest.raises(UnscriptedCall):
        call(runner, tmp_path, "x")

    assert runner.argvs == [("x",)]


def test_the_log_is_written_with_its_directory_and_replaced(tmp_path: Path) -> None:
    log = tmp_path / "deep" / "logs" / "a.log"
    log.parent.mkdir(parents=True)
    log.write_text("from an earlier run")
    runner = FakeRunner().script(OK, log_text="the new text").script(OK)

    runner.run(["x"], cwd=tmp_path, env={}, log=log, timeout=1)
    assert log.read_text() == "the new text"
    runner.run(["y"], cwd=tmp_path, env={}, log=tmp_path / "other" / "b.log", timeout=1)
    assert (tmp_path / "other" / "b.log").read_text() == ""


def test_the_scripted_files_are_written_with_their_directories(tmp_path: Path) -> None:
    target = tmp_path / "out" / "nested" / "parity.json"
    runner = FakeRunner().script(OK, files={target: b'{"a": 1}'})

    call(runner, tmp_path, "x")

    assert json.loads(target.read_text()) == {"a": 1}


def test_a_scripted_report_is_written_to_the_junit_option_of_the_call(
    tmp_path: Path,
) -> None:
    junit = tmp_path / "out" / "junit" / "demo.xml"
    runner = FakeRunner().script(OK, junit=junit_xml(3, 0, 0, 1))

    call(runner, tmp_path, "python", "-m", "pytest", f"--junitxml={junit}", "t.py")

    assert read_junit(junit.read_bytes()) == TestCounts(2, 0, 0, 1)


def test_a_scripted_report_needs_exactly_one_junit_option(tmp_path: Path) -> None:
    runner = FakeRunner().script(OK, junit=junit_xml(1, 0, 0, 0))

    with pytest.raises(AssertionError, match="junitxml"):
        call(runner, tmp_path, "python", "-m", "pytest")

    assert junit_path_of(["a", "--junitxml=/x/y.xml"]) == Path("/x/y.xml")
    with pytest.raises(AssertionError):
        junit_path_of(["--junitxml=/a.xml", "--junitxml=/b.xml"])


# ---- the JUnit document ----------------------------------------------------------------------------


@pytest.mark.parametrize(
    "counters", [(0, 0, 0, 0), (3, 0, 0, 1), (9, 2, 1, 3), (5, 5, 0, 0), (4, 0, 4, 0)]
)
def test_junit_xml_reads_back_as_the_counters_it_was_made_from(
    counters: tuple[int, int, int, int],
) -> None:
    tests, failures, errors, skipped = counters

    counts = read_junit(junit_xml(*counters))

    assert counts == TestCounts(
        tests - failures - errors - skipped, failures, errors, skipped
    )


def test_junit_xml_carries_what_a_report_must_never_copy() -> None:
    data = junit_xml(3, 1, 1, 1).decode()

    assert JUNIT_PATH in data
    assert JUNIT_HOST in data
    assert "timestamp=" in data


def test_the_rerun_builders_make_valid_reruns() -> None:
    pytest_one = pytest_rerun(PASSES_A, min_passed=4, max_skipped=2, timeout=90, id="x")
    parity_one = parity_rerun(id="p", timeout=700, cadence="weekly")

    assert (
        pytest_one.id,
        pytest_one.kind,
        pytest_one.min_passed,
        pytest_one.max_skipped,
    ) == (
        "x",
        "pytest",
        4,
        2,
    )
    assert pytest_one.targets == [PASSES_A]
    assert (parity_one.kind, parity_one.cadence, parity_one.timeout_seconds) == (
        "export_parity",
        "weekly",
        700,
    )


# ---- the project on disk ---------------------------------------------------------------------------


def test_a_project_holds_a_pyproject_the_sample_lock_and_the_standard_tests(
    tmp_path: Path,
) -> None:
    project = make_project(tmp_path)

    assert project == tmp_path / "project"
    assert (project / "pyproject.toml").read_text().startswith("[project]")
    assert (project / "uv.lock").read_bytes() == SAMPLE_LOCK.read_bytes()
    assert read_lock((project / "uv.lock").read_bytes()).entries > 0
    for relative, source in STANDARD_TESTS.items():
        assert (project / relative).read_text() == source
    assert (project / EMPTY_TARGET).is_dir()
    assert list((project / EMPTY_TARGET).iterdir()) == []


def test_the_versions_the_drift_sample_names_are_the_project_locks() -> None:
    # The maintenance gate runs the hand-written dry run against the repository's own uv.lock:
    # a locked version of the sample that the lock does not hold fails that gate.
    project_lock = read_lock((REPO_ROOT / "uv.lock").read_bytes())

    for name, versions in SAMPLE_LOCKED.items():
        assert project_lock.versions.get(name) == versions, name


def test_a_project_can_be_given_another_lock_and_other_tests(tmp_path: Path) -> None:
    project = make_project(
        tmp_path,
        lock=b"version = 2\n",
        tests={"tests/test_only.py": "def test_x():\n    pass\n"},
    )

    assert (project / "uv.lock").read_bytes() == b"version = 2\n"
    assert (project / "tests" / "test_only.py").is_file()
    assert not (project / PASSES_A).exists()


def test_a_project_without_tests_has_only_the_empty_directory(tmp_path: Path) -> None:
    project = make_project(tmp_path, tests={})

    assert sorted(path.name for path in (project / "tests").iterdir()) == ["empty"]


def test_the_targets_of_a_configuration_exist_in_the_project(tmp_path: Path) -> None:
    config = load_maintain_config(
        (
            Path(__file__).resolve().parents[2] / "configs/maintain/maintain.yaml"
        ).read_bytes()
    )

    project = make_project(tmp_path, config=config)

    for rerun in config.reruns:
        for target in rerun.targets:
            path = project / target
            assert path.exists(), target
            if target.endswith(".py"):
                assert path.is_file()
                assert path.read_bytes() == b""
            else:
                assert path.is_dir()
                assert list(path.iterdir()) == []


def test_a_target_that_the_tests_already_wrote_is_not_replaced(tmp_path: Path) -> None:
    config = load_maintain_config(
        (
            Path(__file__).resolve().parents[2] / "configs/maintain/maintain.yaml"
        ).read_bytes()
    )
    shipped_target = "tests/unit/test_metrics_segment_stats.py"

    project = make_project(
        tmp_path, config=config, tests={shipped_target: "def test_kept():\n    pass\n"}
    )

    assert (project / shipped_target).read_text() == "def test_kept():\n    pass\n"


def test_the_parent_environment_leaves_out_what_steers_a_pytest(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PYTEST_ADDOPTS", "-x")
    monkeypatch.setenv("PYTEST_CURRENT_TEST", "a::b")
    monkeypatch.setenv("COV_CORE_SOURCE", "src")
    monkeypatch.setenv("RINCHEN_KEEP", "me")

    environment = parent_environment()

    assert "PYTEST_ADDOPTS" not in environment
    assert "PYTEST_CURRENT_TEST" not in environment
    assert "COV_CORE_SOURCE" not in environment
    assert environment["RINCHEN_KEEP"] == "me"
