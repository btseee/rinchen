"""The JUnit reader of `rinchen.maintain.reruns` and the outcome of a pytest rerun over a
SCRATCH project run by the real pytest (D14, spec M8 and finding F12).

`read_junit` takes the four counters of every `testsuite` and nothing else of the file: the file
holds an absolute path, a host name and a timestamp that no report may copy. A document it cannot
read is refused with a `DataError` that names the problem and quotes none of the bytes.

The rerun tests run `python -m pytest` through the real `SubprocessRunner` over the tiny test
files of `tests/support/maintain_project.py`, written under `tmp_path`: a project of their own
that never selects the repository's `tests/`. They cover the outcomes of F12 (all passing, a
failure and an error, nothing collected, a missing file) and the rest of the decision: the floor,
the skip allowance, a target that does not exist, a timeout that must take the sleeping test and
its grandchild away. Every child is bounded; none starts `uv`, a model or a network connection.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pytest

from rinchen.errors import DataError
from rinchen.maintain.commands import SubprocessRunner
from rinchen.maintain.reruns import (
    RerunResult,
    TestCounts,
    read_junit,
    run_pytest_rerun,
)
from rinchen.maintain.states import (
    FAIL_TARGET,
    fail_floor,
    fail_skipped,
    fail_tests,
    fail_timeout,
)
from tests.support.maintain_fakes import (
    JUNIT_HOST,
    JUNIT_PATH,
    junit_xml,
    pytest_rerun,
)
from tests.support.maintain_project import (
    EMPTY_TARGET,
    ERRORS,
    FAILS,
    HIDDEN_ENVIRONMENT_TEST,
    NETWORK_ONLY,
    PASSES_A,
    PASSES_B,
    PIDS_FILE,
    SKIPS,
    SLEEPS,
    kill_quietly,
    make_project,
    parent_environment,
    read_pids,
    wait_until_gone,
)
from tests.support.no_network import no_network  # noqa: F401 - the fixture of the mark

pytestmark = pytest.mark.usefixtures("no_network")
process_test = pytest.mark.subprocess  # the test starts real child processes
linux_only = pytest.mark.skipif(
    sys.platform != "linux", reason="needs process groups and /proc"
)

SENTINEL = "SENTINELVALUE"


def document(*suites: str, root: str = "testsuites") -> bytes:
    """A JUnit document: the suites inside a root element, as pytest writes them."""
    return (
        f'<?xml version="1.0" encoding="utf-8"?><{root} name="pytest tests">'
        f"{''.join(suites)}</{root}>"
    ).encode()


def suite(
    tests: int | str = 3,
    failures: int | str = 0,
    errors: int | str = 0,
    skipped: int | str = 0,
    *,
    inner: str = "",
    drop: str | None = None,
) -> str:
    """A `testsuite` element with these counters (a counter named in `drop` is left out)."""
    counters = {
        "errors": errors,
        "failures": failures,
        "skipped": skipped,
        "tests": tests,
    }
    attributes = " ".join(
        f'{name}="{value}"' for name, value in counters.items() if name != drop
    )
    return (
        f'<testsuite name="pytest" {attributes} time="0.014" '
        f'timestamp="2026-01-02T03:04:05.678901+00:00" hostname="{JUNIT_HOST}">'
        f"{inner}</testsuite>"
    )


# ---- the four documents of F12, as pytest wrote them ------------------------------------------------

ALL_PASSING = (
    b'<?xml version="1.0" encoding="utf-8"?><testsuites name="pytest tests">'
    b'<testsuite name="pytest" errors="0" failures="0" skipped="1" tests="3" time="0.014" '
    b'timestamp="2026-01-02T03:04:05.678901+00:00" hostname="build-host-7">'
    b'<testcase classname="tests.test_a" name="test_one" time="0.000" />'
    b'<testcase classname="tests.test_a" name="test_two" time="0.000" />'
    b'<testcase classname="tests.test_a" name="test_three" time="0.000">'
    b'<skipped type="pytest.skip" message="nope">'
    b"/srv/example-runner/work/project/tests/test_example.py:17: nope</skipped>"
    b"</testcase></testsuite></testsuites>"
)
A_FAILURE_AND_AN_ERROR = (
    b'<?xml version="1.0" encoding="utf-8"?><testsuites name="pytest tests">'
    b'<testsuite name="pytest" errors="1" failures="1" skipped="1" tests="5" time="0.016" '
    b'timestamp="2026-01-02T03:04:05.678901+00:00" hostname="build-host-7">'
    b'<testcase classname="tests.test_a" name="test_fail" time="0.000">'
    b'<failure message="assert 0">def test_fail():\n&gt;       assert 0\nE       assert 0</failure>'
    b'</testcase><testcase classname="tests.test_a" name="test_error" time="0.000">'
    b'<error message="failed on setup">RuntimeError: x</error></testcase>'
    b"</testsuite></testsuites>"
)
NOTHING_COLLECTED = (
    b'<?xml version="1.0" encoding="utf-8"?><testsuites name="pytest tests">'
    b'<testsuite name="pytest" errors="0" failures="0" skipped="0" tests="0" time="0.010" '
    b'timestamp="2026-01-02T03:04:05.678901+00:00" hostname="build-host-7" />'
    b"</testsuites>"
)
A_FILE_THAT_DOES_NOT_EXIST = NOTHING_COLLECTED.replace(b'time="0.010"', b'time="0.005"')


@pytest.mark.parametrize(
    ("data", "expected"),
    [
        (ALL_PASSING, TestCounts(passed=2, failed=0, errors=0, skipped=1)),
        (A_FAILURE_AND_AN_ERROR, TestCounts(passed=2, failed=1, errors=1, skipped=1)),
        (NOTHING_COLLECTED, TestCounts(passed=0, failed=0, errors=0, skipped=0)),
        (
            A_FILE_THAT_DOES_NOT_EXIST,
            TestCounts(passed=0, failed=0, errors=0, skipped=0),
        ),
    ],
    ids=["all passing", "a failure and an error", "nothing collected", "no such file"],
)
def test_the_four_documents_of_pytest_give_their_counters(
    data: bytes, expected: TestCounts
) -> None:
    assert read_junit(data) == expected


def test_the_counters_of_several_suites_are_summed() -> None:
    data = document(
        suite(4, failures=1, skipped=1), suite(3, errors=2), suite(0), suite(10)
    )

    assert read_junit(data) == TestCounts(passed=13, failed=1, errors=2, skipped=1)


def test_a_testsuite_root_is_read_as_it_is() -> None:
    data = (
        b'<testsuite errors="0" failures="1" skipped="0" tests="5">'
        b'<testcase name="a" /></testsuite>'
    )

    assert read_junit(data) == TestCounts(passed=4, failed=1, errors=0, skipped=0)


def test_only_the_outermost_suites_count_so_that_a_nested_one_is_not_added_twice() -> (
    None
):
    nested = suite(100, failures=100)
    data = document(suite(2, inner=nested), suite(3, inner=f"<x>{nested}</x>"))
    root_with_a_nested_suite = (
        f'<testsuite tests="2" failures="0" errors="0" skipped="0">{nested}</testsuite>'
    ).encode()

    assert read_junit(data) == TestCounts(passed=5, failed=0, errors=0, skipped=0)
    assert read_junit(root_with_a_nested_suite) == TestCounts(2, 0, 0, 0)


def test_an_element_named_alike_is_not_a_suite() -> None:
    alike = (
        '<testsuites tests="90" failures="0" errors="0" skipped="0" />'
        '<testsuite_extra tests="91" failures="0" errors="0" skipped="0" />'
        '<TestSuite tests="92" failures="0" errors="0" skipped="0" />'
    )

    data = document(suite(2), alike)

    assert read_junit(data) == TestCounts(passed=2, failed=0, errors=0, skipped=0)


def test_a_junit_document_from_junit_xml_gives_the_counters_it_was_made_from() -> None:
    assert read_junit(junit_xml(7, 2, 1, 3)) == TestCounts(
        passed=1, failed=2, errors=1, skipped=3
    )


def test_a_path_and_a_host_name_in_the_file_never_appear_in_a_result() -> None:
    counts = read_junit(junit_xml(5, 1, 1, 1))

    shown = repr(counts) + str(counts)
    assert JUNIT_PATH not in shown
    assert JUNIT_HOST not in shown
    assert "2026" not in shown
    assert JUNIT_PATH.encode() in junit_xml(5, 1, 1, 1)  # the file did hold them


def test_the_counters_are_plain_integers_of_the_result() -> None:
    counts = read_junit(ALL_PASSING)

    assert all(
        type(value) is int
        for value in (counts.passed, counts.failed, counts.errors, counts.skipped)
    )


REFUSED = {
    "not xml": b"this is not xml " + SENTINEL.encode(),
    "empty": b"",
    "binary": b"\x00\x01\xff\xfe",
    "an unknown encoding": (
        b'<?xml version="1.0" encoding="no-such-codec"?><testsuites>'
        + suite(3).encode()
        + b"</testsuites>"
    ),
    "invalid bytes in the declared encoding": (
        b'<?xml version="1.0" encoding="utf-8"?><testsuites name="\xff\xfe">'
        + suite(3).encode()
        + b"</testsuites>"
    ),
    "truncated": ALL_PASSING[:150],
    "another root": f'<foo hostname="{SENTINEL}"/>'.encode(),
    "a root that is not a suite": f"<testcase>{SENTINEL}</testcase>".encode(),
    "another root that holds a valid suite": f"<foo>{suite(3)}</foo>".encode(),
    "another root in the position of the suites": f"<run><testsuites>{suite(3)}</testsuites></run>".encode(),
    "no suite": f"<testsuites>{SENTINEL}</testsuites>".encode(),
    "no suite but another child": f"<testsuites><testcase>{SENTINEL}</testcase></testsuites>".encode(),
    "a suite that is not a child of the root": document(
        f"<wrapper>{suite(3)}</wrapper>"
    ),
    "no tests counter": document(suite(drop="tests")),
    "no failures counter": document(suite(drop="failures")),
    "no errors counter": document(suite(drop="errors")),
    "no skipped counter": document(suite(drop="skipped")),
    "a counter of the second suite is missing": document(
        suite(2), suite(drop="errors")
    ),
    "a counter that is not a number": document(suite(tests=SENTINEL)),
    "an empty counter": document(suite(tests="")),
    "a decimal counter": document(suite(tests="1.0")),
    "a negative counter": document(suite(failures="-1")),
    "a signed counter": document(suite(tests="+3")),
    "a counter with a space": document(suite(tests=" 3")),
    "a counter in other digits": document(suite(tests=chr(0x663))),
    "a counter with a separator": document(suite(tests="1_0")),
    "a counter of absurd size": document(suite(tests="9" * 30)),
    "more failures and errors than tests": document(suite(1, failures=1, errors=1)),
    "more skipped than tests": document(suite(1, skipped=2)),
    "a document type with an entity": (
        b'<!DOCTYPE testsuites [<!ENTITY a "aaaa">]><testsuites>'
        + suite(3).encode()
        + b"</testsuites>"
    ),
}


@pytest.mark.parametrize("data", REFUSED.values(), ids=REFUSED.keys())
def test_a_document_that_cannot_be_read_is_refused_without_quoting_it(
    data: bytes,
) -> None:
    with pytest.raises(DataError) as raised:
        read_junit(data)

    message = str(raised.value)
    assert "JUnit" in message
    for secret in (SENTINEL, JUNIT_HOST, JUNIT_PATH, "2026", "<", ">"):
        assert secret not in message


# ---- a rerun over the real pytest in a scratch project ---------------------------------------------


def rerun_of(
    tmp_path: Path,
    targets: list[str],
    *,
    min_passed: int,
    max_skipped: int = 0,
    timeout: int = 120,
) -> tuple[Path, Path, RerunResult]:
    """A scratch project and the result of its rerun of `targets`, through the real runner."""
    project = make_project(tmp_path)
    out = tmp_path / "out"
    result = run_pytest_rerun(
        pytest_rerun(
            *targets, min_passed=min_passed, max_skipped=max_skipped, timeout=timeout
        ),
        runner=SubprocessRunner(),
        python=sys.executable,
        project_dir=project,
        out_dir=out,
        parent_env=parent_environment(),
    )
    return project, out, result


# (what the project holds, the floor, the allowance, the state, the reason, the counters)
CASES = [
    ([PASSES_A, PASSES_B], 2, 0, "ok", None, (2, 0, 0, 0)),
    ([PASSES_A, PASSES_B, FAILS], 2, 0, "fail", fail_tests(1, 0), (2, 1, 0, 0)),
    ([PASSES_A, PASSES_B, ERRORS], 2, 0, "fail", fail_tests(0, 1), (2, 0, 1, 0)),
    ([PASSES_A, PASSES_B, SKIPS], 2, 0, "fail", fail_skipped(1, 0), (2, 0, 0, 1)),
    ([PASSES_A, PASSES_B, SKIPS], 2, 1, "ok", None, (2, 0, 0, 1)),
    ([PASSES_A, PASSES_B], 3, 0, "fail", fail_floor(2, 3), (2, 0, 0, 0)),
    ([EMPTY_TARGET], 1, 0, "fail", fail_floor(0, 1), (0, 0, 0, 0)),
    ([NETWORK_ONLY], 1, 0, "fail", fail_floor(0, 1), (0, 0, 0, 0)),
    ([PASSES_A, FAILS, ERRORS, SKIPS], 1, 0, "fail", fail_tests(1, 1), (1, 1, 1, 1)),
    ([HIDDEN_ENVIRONMENT_TEST], 1, 0, "ok", None, (1, 0, 0, 0)),
]
CASE_IDS = [
    "two passing files",
    "a failure: the exit code is 1 and the file is still read",
    "an error",
    "a skip above the allowance",
    "a skip within the allowance",
    "a floor above the passes",
    "an empty directory: exit 5",
    "only a network test: deselected, exit 5",
    "a failure, an error and a skip",
    "the child has the device and the hub hidden",
]


@process_test
@pytest.mark.timeout(120)
@pytest.mark.parametrize(
    ("targets", "floor", "allowance", "state", "reason", "counts"),
    CASES,
    ids=CASE_IDS,
)
def test_a_rerun_over_the_real_pytest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    targets: list[str],
    floor: int,
    allowance: int,
    state: str,
    reason: str | None,
    counts: tuple[int, int, int, int],
) -> None:
    monkeypatch.setenv(
        "CUDA_VISIBLE_DEVICES", "0"
    )  # the rerun must hide it from its child

    project, out, result = rerun_of(
        tmp_path, targets, min_passed=floor, max_skipped=allowance
    )

    assert (result.state, result.reason) == (state, reason)
    assert result.tests == TestCounts(*counts)
    assert result.figures is None
    assert (result.id, result.kind, result.cadence) == ("demo", "pytest", "nightly")
    assert (out / "junit" / "demo.xml").is_file()
    assert (out / "logs" / "demo.log").is_file()
    assert not (project / ".pytest_cache").exists()  # -p no:cacheprovider


@process_test
def test_a_missing_target_fails_before_anything_runs(tmp_path: Path) -> None:
    _, out, result = rerun_of(
        tmp_path, [PASSES_A, "tests/no_such_file.py"], min_passed=1
    )

    assert (result.state, result.reason) == ("fail", FAIL_TARGET)
    assert result.tests is None
    assert not out.exists(), "something was written although nothing was to run"


@process_test
@linux_only
@pytest.mark.timeout(120)
def test_a_sleeping_test_is_cut_on_the_timeout_and_its_processes_are_gone(
    tmp_path: Path,
) -> None:
    project = make_project(tmp_path)
    started = time.monotonic()

    try:
        result = run_pytest_rerun(
            pytest_rerun(SLEEPS, min_passed=1, timeout=5),
            runner=SubprocessRunner(),
            python=sys.executable,
            project_dir=project,
            out_dir=tmp_path / "out",
            parent_env=parent_environment(),
        )
        elapsed = time.monotonic() - started
        pids = read_pids(project / PIDS_FILE)
        left = [pid for pid in pids if not wait_until_gone(pid, 1.0)]
    finally:
        if (project / PIDS_FILE).exists():
            kill_quietly(read_pids(project / PIDS_FILE))

    assert (result.state, result.reason) == ("fail", fail_timeout(5))
    assert result.tests is None
    assert elapsed < 20
    assert len(pids) == 2
    assert left == [], f"processes still running after the timeout: {left}"
