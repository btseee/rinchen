"""Doubles for the D14 tests: a command runner that plays scripted outcomes and records what it
was asked to run (no process starts, so no test of the library needs `uv`, a network, a model or
a Python child), and the JUnit document that a real pytest writes.

`FakeRunner` stands in for `rinchen.maintain.commands.SubprocessRunner`. A test scripts the
outcomes it expects, each with the `CommandResult` to return, the text the command would have
written to its log and the files it would have written (a JUnit report, a `parity.json`); a call
takes the first outcome not yet used whose predicate accepts its argument list (an outcome with
no predicate accepts any, so a list of them plays by order). Like the real runner it replaces the
log of every call, and it raises `UnscriptedCall` for a call nothing was scripted for, so that a
command the library runs without the test knowing is a failure and never a silent pass.

`junit_xml` writes what pytest writes for a run with these counters, including the things a
report must never copy: an absolute path (in the text of a skipped test), a host name and a
timestamp. `JUNIT_PATH` and `JUNIT_HOST` are those values, for the tests that look for them.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from rinchen.config.maintain import RerunConfig
from rinchen.maintain.commands import CommandResult

JUNIT_PATH: Final = "/srv/example-runner/work/project/tests/test_example.py"
JUNIT_HOST: Final = "build-host-7"
JUNIT_TIMESTAMP: Final = "2026-01-02T03:04:05.678901+00:00"
_JUNIT_OPTION: Final = "--junitxml="


class UnscriptedCall(AssertionError):
    """The code under test ran a command that the test did not script."""


@dataclass(frozen=True, slots=True)
class Call:
    """One call of `run`, as it was made."""

    argv: tuple[str, ...]
    cwd: Path
    env: Mapping[str, str]
    log: Path
    timeout: int


@dataclass(slots=True)
class _Outcome:
    result: CommandResult
    log_text: str
    files: Mapping[Path, bytes]
    junit: bytes | None
    when: Callable[[Sequence[str]], bool] | None
    used: bool = False


class FakeRunner:
    """A `CommandRunner` that plays scripted outcomes."""

    def __init__(self) -> None:
        self.calls: list[Call] = []
        self._outcomes: list[_Outcome] = []

    def script(
        self,
        result: CommandResult,
        *,
        log_text: str = "",
        files: Mapping[Path, bytes] | None = None,
        junit: bytes | None = None,
        when: Callable[[Sequence[str]], bool] | None = None,
    ) -> FakeRunner:
        """Add an outcome: the call that takes it gets `result`, its log is `log_text`, each of
        `files` is written, `junit` is written to the path of the call's `--junitxml=` option,
        and `when` (any call when None) decides which calls may take it. Returns the runner."""
        self._outcomes.append(_Outcome(result, log_text, files or {}, junit, when))
        return self

    def run(
        self,
        argv: Sequence[str],
        *,
        cwd: Path,
        env: Mapping[str, str],
        log: Path,
        timeout: int,
    ) -> CommandResult:
        call = Call(tuple(argv), cwd, dict(env), log, timeout)
        self.calls.append(call)
        outcome = next(
            (
                scripted
                for scripted in self._outcomes
                if not scripted.used
                and (scripted.when is None or scripted.when(call.argv))
            ),
            None,
        )
        if outcome is None:
            raise UnscriptedCall(
                f"no outcome was scripted for the command {call.argv[:4]}"
            )
        outcome.used = True
        log.parent.mkdir(parents=True, exist_ok=True)
        log.write_text(outcome.log_text, encoding="utf-8")
        for path, data in outcome.files.items():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
        if outcome.junit is not None:
            written = junit_path_of(call.argv)
            written.parent.mkdir(parents=True, exist_ok=True)
            written.write_bytes(outcome.junit)
        return outcome.result

    @property
    def argvs(self) -> list[tuple[str, ...]]:
        """The argument list of every call, in order."""
        return [call.argv for call in self.calls]

    def unused(self) -> int:
        """How many scripted outcomes no call took."""
        return sum(1 for outcome in self._outcomes if not outcome.used)


def pytest_rerun(
    *targets: str,
    id: str = "demo",
    min_passed: int = 1,
    max_skipped: int = 0,
    timeout: int = 60,
    cadence: str = "nightly",
) -> RerunConfig:
    """A rerun of kind `pytest` over these targets."""
    return RerunConfig.model_validate(
        {
            "id": id,
            "kind": "pytest",
            "cadence": cadence,
            "timeout_seconds": timeout,
            "why": "a rerun of the tests",
            "targets": list(targets),
            "min_passed": min_passed,
            "max_skipped": max_skipped,
        }
    )


def parity_rerun(
    *, id: str = "parity", timeout: int = 600, cadence: str = "nightly"
) -> RerunConfig:
    """A rerun of kind `export_parity`."""
    return RerunConfig.model_validate(
        {
            "id": id,
            "kind": "export_parity",
            "cadence": cadence,
            "timeout_seconds": timeout,
            "why": "the parity command over an export",
        }
    )


def junit_path_of(argv: Sequence[str]) -> Path:
    """The path of the `--junitxml=` option of a pytest argument list."""
    options = [arg for arg in argv if arg.startswith(_JUNIT_OPTION)]
    if len(options) != 1:
        raise AssertionError("the argument list needs exactly one --junitxml= option")
    return Path(options[0].removeprefix(_JUNIT_OPTION))


def junit_xml(tests: int, failures: int, errors: int, skipped: int) -> bytes:
    """The report pytest writes for a run with these four counters: one `testsuite` in a
    `testsuites` root, a `testcase` for each passed, failed, erroring and skipped test (as many
    passes as the counters leave), a host name, a timestamp and, in the first skipped test, an
    absolute path."""
    passed = max(tests - failures - errors - skipped, 0)
    cases = [
        f'<testcase classname="tests.test_example" name="test_pass_{n}" time="0.000" />'
        for n in range(passed)
    ]
    cases += [
        f'<testcase classname="tests.test_example" name="test_fail_{n}" time="0.000">'
        f'<failure message="assert 0">{JUNIT_PATH}:{n}: AssertionError</failure></testcase>'
        for n in range(failures)
    ]
    cases += [
        f'<testcase classname="tests.test_example" name="test_error_{n}" time="0.000">'
        f'<error message="failed on setup">{JUNIT_PATH}:{n}: RuntimeError</error></testcase>'
        for n in range(errors)
    ]
    cases += [
        f'<testcase classname="tests.test_example" name="test_skip_{n}" time="0.000">'
        f'<skipped type="pytest.skip" message="skipped">{JUNIT_PATH}:{n}: skipped</skipped></testcase>'
        for n in range(skipped)
    ]
    suite = (
        f'<testsuite name="pytest" errors="{errors}" failures="{failures}" '
        f'skipped="{skipped}" tests="{tests}" time="0.016" timestamp="{JUNIT_TIMESTAMP}" '
        f'hostname="{JUNIT_HOST}">{"".join(cases)}</testsuite>'
    )
    return (
        f'<?xml version="1.0" encoding="utf-8"?><testsuites name="pytest tests">{suite}</testsuites>'
    ).encode()
