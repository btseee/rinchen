"""Why a pytest rerun has no counters (D14 fix wave 1, deferred minor of Task 4).

A rerun that wrote no JUnit report is `fail` with "no JUnit report was written". The same words were
the reason when the file EXISTED and could not be read as a report (not well-formed XML, an unknown
encoding, the wrong root, a counter that is no number, more failures than tests, a file that is a
directory), which sends the operator looking for a report that is on disk. A file that is missing keeps
`FAIL_NO_JUNIT`; a file that is there and unusable is `FAIL_BAD_JUNIT`, "the JUnit report could not be
read". Neither carries counters, and both fail the step.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import pytest

from rinchen.maintain import states
from rinchen.maintain.commands import CommandResult
from rinchen.maintain.report import report_document
from rinchen.maintain.reruns import RerunResult, run_pytest_rerun
from rinchen.maintain.states import FAIL_BAD_JUNIT, FAIL_NO_JUNIT
from tests.support.maintain_check import (
    execute,
    make_seams,
    rerun_call,
    script_passing_reruns,
    step_of,
    uv_call,
)
from tests.support.maintain_fakes import FakeRunner, junit_xml, pytest_rerun
from tests.support.maintain_project import PASSES_A, make_project
from tests.support.maintain_scans import closed_reason
from tests.support.no_network import no_network  # noqa: F401 - the fixture of the mark
from tests.support.uv_outputs import NO_CHANGE_MEASURED

pytestmark = pytest.mark.usefixtures("no_network")

HEADER = b'<?xml version="1.0" encoding="utf-8"?>'
SUITE = b'<testsuite name="pytest" errors="0" failures="0" skipped="0" tests="3"/>'
BAD_REPORTS: Mapping[str, bytes] = {
    "empty": b"",
    "garbage": b"garbage",
    "truncated": junit_xml(3, 0, 0, 0)[:-30],
    "not xml but a json text": b'{"tests": 3}',
    "unknown encoding": b'<?xml version="1.0" encoding="no-such-codec"?>' + SUITE,
    "a doctype": HEADER + b'<!DOCTYPE testsuite SYSTEM "x.dtd">' + SUITE,
    "another root": HEADER + b"<report><testsuite/></report>",
    "no testsuite": HEADER + b'<testsuites name="pytest tests"></testsuites>',
    "a counter that is no number": HEADER
    + b'<testsuite errors="0" failures="0" skipped="0" tests="many"/>',
    "a counter that is missing": HEADER
    + b'<testsuite errors="0" failures="0" tests="3"/>',
    "a negative counter": HEADER
    + b'<testsuite errors="0" failures="0" skipped="-1" tests="3"/>',
    "more failures than tests": junit_xml(1, 2, 0, 0),
}


def rerun_over(
    tmp_path: Path, runner: object, *, min_passed: int = 1
) -> tuple[RerunResult, Path]:
    project = make_project(tmp_path)
    out = tmp_path / "out"
    result = run_pytest_rerun(
        pytest_rerun(PASSES_A, min_passed=min_passed),
        runner=runner,  # type: ignore[arg-type]
        python="py",
        project_dir=project,
        out_dir=out,
        parent_env={},
    )
    return result, out


class WritingRunner:
    """A runner whose command exits 0 and leaves `data` at the JUnit path (or what `make` makes
    there)."""

    def __init__(self, data: bytes | None = None, *, directory: bool = False) -> None:
        self.data = data
        self.directory = directory

    def run(
        self,
        argv: Sequence[str],
        *,
        cwd: Path,
        env: Mapping[str, str],
        log: Path,
        timeout: int,
    ) -> CommandResult:
        (junit,) = [arg for arg in argv if arg.startswith("--junitxml=")]
        path = Path(junit.removeprefix("--junitxml="))
        path.parent.mkdir(parents=True, exist_ok=True)
        if self.directory:
            path.mkdir()
        elif self.data is not None:
            path.write_bytes(self.data)
        return CommandResult(0, False)


# ---- the two reasons -----------------------------------------------------------------------------------


def test_the_two_reasons_are_closed_and_say_different_things() -> None:
    assert FAIL_NO_JUNIT == "no JUnit report was written"
    assert FAIL_BAD_JUNIT == "the JUnit report could not be read"
    assert FAIL_BAD_JUNIT in states.ALL_REASONS and FAIL_NO_JUNIT in states.ALL_REASONS
    assert closed_reason(FAIL_BAD_JUNIT) and closed_reason(FAIL_NO_JUNIT)
    assert "written" not in FAIL_BAD_JUNIT


def test_a_report_that_was_never_written_is_no_report(tmp_path: Path) -> None:
    result, _ = rerun_over(tmp_path, FakeRunner().script(CommandResult(0, False)))

    assert (result.state, result.reason, result.tests) == ("fail", FAIL_NO_JUNIT, None)


@pytest.mark.parametrize("name", list(BAD_REPORTS))
def test_a_report_that_is_there_and_unusable_is_a_report_that_could_not_be_read(
    tmp_path: Path, name: str
) -> None:
    result, out = rerun_over(tmp_path, WritingRunner(BAD_REPORTS[name]))

    assert (out / "junit" / "demo.xml").is_file()
    assert (result.state, result.reason, result.tests) == ("fail", FAIL_BAD_JUNIT, None)
    assert result.figures is None


def test_a_report_that_is_a_directory_could_not_be_read(tmp_path: Path) -> None:
    result, out = rerun_over(tmp_path, WritingRunner(directory=True))

    assert (out / "junit" / "demo.xml").is_dir()
    assert (result.state, result.reason, result.tests) == ("fail", FAIL_BAD_JUNIT, None)


def test_a_good_report_is_still_read(tmp_path: Path) -> None:
    result, _ = rerun_over(tmp_path, WritingRunner(junit_xml(3, 0, 0, 0)), min_passed=3)

    assert (result.state, result.reason) == ("ok", None)
    assert result.tests is not None and result.tests.passed == 3


@pytest.mark.parametrize(
    ("outcome", "reason"),
    [
        (CommandResult(None, True), states.fail_timeout(60)),
        (CommandResult(None, False, False), states.fail_pytest_exit(-1)),
    ],
)
def test_a_timeout_and_a_command_that_did_not_start_keep_their_own_reasons(
    tmp_path: Path, outcome: CommandResult, reason: str
) -> None:
    runner = FakeRunner().script(outcome, junit=BAD_REPORTS["garbage"])

    result, _ = rerun_over(tmp_path, runner)

    assert (result.state, result.reason) == ("fail", reason)


def test_the_reason_reaches_the_report_of_the_check(tmp_path: Path) -> None:
    runner = (
        FakeRunner()
        .script(CommandResult(0, False), when=uv_call("--check"))
        .script(
            CommandResult(0, False),
            log_text=NO_CHANGE_MEASURED,
            when=uv_call("--upgrade", "--dry-run"),
        )
        .script(
            CommandResult(0, False),
            junit=BAD_REPORTS["garbage"],
            when=rerun_call("numerics"),
        )
    )
    script_passing_reruns(runner, "parity_tiny")

    run = execute(tmp_path, runner, seams=make_seams(runner))

    document = report_document(run.result.report)
    items = step_of(document, "reruns")["detail"]["items"]  # type: ignore[index]
    by_id = {item["id"]: (item["state"], item["reason"]) for item in items}
    assert by_id["numerics"] == ("fail", FAIL_BAD_JUNIT)
    assert by_id["parity_tiny"] == ("ok", None)
    assert document["state"] == "fail"
