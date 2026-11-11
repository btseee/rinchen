"""The parity reruns of a maintenance check (D14, spec M8): the cheap, decisive checks that the
project's own tests and commands already are, asked again under a changed stack.

A rerun of kind `pytest` runs `python -m pytest` over a few files of `tests/` from the project
directory, with the interpreter that runs the check, and decides from the exit code AND the
counters of the JUnit report it asked for. It is `ok` only when the exit code is 0, nothing failed
or errored, at least `min_passed` tests passed and no more than `max_skipped` were skipped: a
selection that shrank or collected nothing (exit 5) is a failure and never a pass, so a golden
that was deleted cannot turn a rerun green. A rerun of kind `export_parity` runs `rinchen export
parity` over an export the operator gives and reads its exit code and the three figures of the
`parity.json` it wrote; it is skipped without one.

The counters are the only thing read from the JUnit file: it holds an absolute path, a host name
and a timestamp that no report may copy. A rerun's result names the rerun, a state, one reason of
`rinchen.maintain.states` and the counters or figures; the log (`logs/<id>.log`) is the only place
a program's own words land. Paths that reach the child are made absolute first: the child runs in
the project directory, which is not the directory the operator's relative paths mean.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, TypeGuard
from xml.parsers import expat

from rinchen.config.maintain import RerunConfig
from rinchen.errors import ConfigError, DataError
from rinchen.maintain.commands import CommandRunner, child_environment
from rinchen.maintain.states import (
    FAIL_BAD_JUNIT,
    FAIL_NO_FIGURES,
    FAIL_NO_JUNIT,
    FAIL_TARGET,
    SKIP_NO_EXPORT,
    State,
    fail_command_exit,
    fail_floor,
    fail_pytest_exit,
    fail_skipped,
    fail_tests,
    fail_timeout,
)

LOGS_DIR: Final = "logs"
JUNIT_DIR: Final = "junit"
PARITY_DIR: Final = "parity_export"
PARITY_FILE: Final = "parity.json"

_COUNTERS: Final = ("tests", "failures", "errors", "skipped")
_COUNT: Final = re.compile(r"[0-9]{1,9}")
_PARITY_MAX_LINES: Final = "50"
_PARITY_THREADS: Final = "2"
# What the parity command is left of the bound for its own set-up and the CTranslate2 run: its
# `--timeout` is the limit of the Eole run alone.
_PARITY_RESERVE_SECONDS: Final = 30


@dataclass(frozen=True, slots=True)
class TestCounts:
    """The counters of a JUnit report: `passed` is `tests - failures - errors - skipped`."""

    __test__ = False  # a counter, and no test class, whatever pytest makes of the name

    passed: int
    failed: int
    errors: int
    skipped: int


@dataclass(frozen=True, slots=True)
class ExportInputs:
    """What `rinchen export parity` needs: an export, the checkpoint it was made from and the
    corpus files whose dev rows are translated."""

    export: Path
    checkpoint: Path
    corpora: tuple[Path, ...]


@dataclass(frozen=True, slots=True)
class RerunResult:
    """How a rerun ended: the state with its reason (`None` for `ok`), the counters of a pytest
    rerun whose report was read and the figures of a parity rerun whose `parity.json` was."""

    id: str
    kind: str
    cadence: str
    state: State
    reason: str | None
    tests: TestCounts | None = None
    figures: Mapping[str, float | int] | None = None


# ---- the JUnit report ---------------------------------------------------------------------------------


class _Suites:
    """Collects the attributes of the outermost `testsuite` elements: the root, or the children
    of a `testsuites` root. Nothing else of the document is looked at."""

    def __init__(self) -> None:
        self.depth = 0
        self.root = ""
        self.found: list[dict[str, str]] = []

    def start(self, name: str, attributes: dict[str, str]) -> None:
        self.depth += 1
        if self.depth == 1:
            if name not in ("testsuites", "testsuite"):
                raise DataError(
                    "the JUnit report's root is neither testsuites nor testsuite"
                )
            self.root = name
        if name == "testsuite" and self.depth == (1 if self.root == "testsuite" else 2):
            self.found.append(attributes)

    def end(self, name: str) -> None:
        self.depth -= 1

    def refuse_doctype(self, *declaration: object) -> None:
        raise DataError("the JUnit report has a document type declaration")


def read_junit(data: bytes) -> TestCounts:
    """The counters of the JUnit report in `data`, the counters of several `testsuite` elements
    summed.

    Raises:
        DataError: `data` is not well-formed XML, has another root than `testsuites` or
            `testsuite`, holds no `testsuite`, has a suite without one of the counters `tests`,
            `failures`, `errors` and `skipped` as a plain count, or counts more failures,
            errors and skipped tests than tests; the message names the problem and quotes none
            of the bytes.
    """
    suites = _Suites()
    parser = expat.ParserCreate()
    parser.StartElementHandler = suites.start
    parser.EndElementHandler = suites.end
    parser.StartDoctypeDeclHandler = suites.refuse_doctype
    try:
        parser.Parse(data, True)
    except (expat.ExpatError, LookupError) as error:  # LookupError: an unknown encoding
        raise DataError("the JUnit report is not well-formed XML") from error
    if not suites.found:
        raise DataError("the JUnit report holds no testsuite")
    totals = dict.fromkeys(_COUNTERS, 0)
    for position, attributes in enumerate(suites.found, start=1):
        for name in _COUNTERS:
            value = attributes.get(name, "")
            if not _COUNT.fullmatch(value):
                raise DataError(
                    f"the JUnit report: testsuite {position} has no valid {name} count"
                )
            totals[name] += int(value)
    passed = totals["tests"] - totals["failures"] - totals["errors"] - totals["skipped"]
    if passed < 0:
        raise DataError(
            "the JUnit report counts more failures, errors and skipped tests than tests"
        )
    return TestCounts(
        passed=passed,
        failed=totals["failures"],
        errors=totals["errors"],
        skipped=totals["skipped"],
    )


# ---- the argument lists ----------------------------------------------------------------------------


def pytest_argv(python: str, rerun: RerunConfig, junit: Path) -> list[str]:
    """The command of a pytest rerun: quiet, the `network` tests left out, no cache directory
    written into the project, the JUnit report at `junit` and the targets in file order."""
    return [
        python,
        "-m",
        "pytest",
        "-q",
        "-m",
        "not network",
        "-p",
        "no:cacheprovider",
        f"--junitxml={junit}",
        *rerun.targets,
    ]


def parity_argv(
    python: str, rerun: RerunConfig, inputs: ExportInputs, out_dir: Path
) -> list[str]:
    """The command of a parity rerun. Its `--timeout` is the bound less the reserve, and at least
    one second."""
    corpora = [arg for corpus in inputs.corpora for arg in ("--corpus", str(corpus))]
    return [
        python,
        "-m",
        "rinchen",
        "export",
        "parity",
        "--checkpoint",
        str(inputs.checkpoint),
        "--export",
        str(inputs.export),
        *corpora,
        "--out-dir",
        str(out_dir),
        "--max-lines",
        _PARITY_MAX_LINES,
        "--threads",
        _PARITY_THREADS,
        "--timeout",
        str(max(1, rerun.timeout_seconds - _PARITY_RESERVE_SECONDS)),
    ]


# ---- the reruns -------------------------------------------------------------------------------------


def run_pytest_rerun(
    rerun: RerunConfig,
    *,
    runner: CommandRunner,
    python: str,
    project_dir: Path,
    out_dir: Path,
    parent_env: Mapping[str, str],
) -> RerunResult:
    """Run a rerun of kind `pytest` in `project_dir` and decide, in this order and stopping at
    the first that holds: a timeout; a process that did not start; no JUnit report, or one that
    cannot be read (two reasons); failures or errors; fewer passed tests than the floor; more skipped tests than allowed; a
    non-zero exit code; else `ok`.

    A target that does not exist under `project_dir` is a failure before anything runs. The
    report `<out_dir>/junit/<id>.xml` is removed before the run, so a stale one never counts, and
    the log is `<out_dir>/logs/<id>.log`.

    Raises:
        ConfigError: `rerun` is not of kind `pytest`.
    """
    floor = rerun.min_passed
    if rerun.kind != "pytest" or floor is None:
        raise ConfigError(f"rerun {rerun.id}: is not a pytest rerun with a floor")
    if any(not (project_dir / target).exists() for target in rerun.targets):
        return _result(rerun, "fail", FAIL_TARGET)
    out = out_dir.absolute()
    junit = out / JUNIT_DIR / f"{rerun.id}.xml"
    junit.unlink(missing_ok=True)
    outcome = runner.run(
        pytest_argv(python, rerun, junit),
        cwd=project_dir,
        env=child_environment(parent_env),
        log=out / LOGS_DIR / f"{rerun.id}.log",
        timeout=rerun.timeout_seconds,
    )
    if outcome.timed_out:
        return _result(rerun, "fail", fail_timeout(rerun.timeout_seconds))
    if outcome.exit_code is None:
        return _result(rerun, "fail", fail_pytest_exit(-1))
    counts = _read_counts(junit)
    if isinstance(counts, str):
        return _result(rerun, "fail", counts)
    reason = _pytest_failure(counts, outcome.exit_code, floor, rerun.max_skipped)
    return _result(rerun, "ok" if reason is None else "fail", reason, tests=counts)


def run_parity_rerun(
    rerun: RerunConfig,
    *,
    runner: CommandRunner,
    python: str,
    project_dir: Path,
    out_dir: Path,
    parent_env: Mapping[str, str],
    inputs: ExportInputs | None,
) -> RerunResult:
    """Run a rerun of kind `export_parity` over `inputs`; without them it is skipped and nothing
    runs. Exit 0 with the three figures of `<out_dir>/parity_export/parity.json` is `ok` (without
    them a failure). Any other exit code (4 is a measured miss) is a failure that carries the
    figures when the file holds them; a timeout and a command that did not start are failures
    too. A `parity.json` of an earlier run is removed before the run.

    Raises:
        ConfigError: `rerun` is not of kind `export_parity`.
    """
    if rerun.kind != "export_parity":
        raise ConfigError(f"rerun {rerun.id}: is not an export_parity rerun")
    if inputs is None:
        return _result(rerun, "skipped", SKIP_NO_EXPORT)
    out = out_dir.absolute()
    parity_dir = out / PARITY_DIR
    (parity_dir / PARITY_FILE).unlink(missing_ok=True)
    outcome = runner.run(
        parity_argv(python, rerun, _absolute(inputs), parity_dir),
        cwd=project_dir,
        env=child_environment(parent_env),
        log=out / LOGS_DIR / f"{rerun.id}.log",
        timeout=rerun.timeout_seconds,
    )
    if outcome.timed_out:
        return _result(rerun, "fail", fail_timeout(rerun.timeout_seconds))
    if outcome.exit_code is None:
        return _result(rerun, "fail", fail_command_exit(-1))
    figures = _read_figures(parity_dir / PARITY_FILE)
    if outcome.exit_code != 0:
        reason = fail_command_exit(outcome.exit_code)
        return _result(rerun, "fail", reason, figures=figures)
    if figures is None:
        return _result(rerun, "fail", FAIL_NO_FIGURES)
    return _result(rerun, "ok", None, figures=figures)


def _result(
    rerun: RerunConfig,
    state: State,
    reason: str | None,
    *,
    tests: TestCounts | None = None,
    figures: Mapping[str, float | int] | None = None,
) -> RerunResult:
    return RerunResult(
        id=rerun.id,
        kind=rerun.kind,
        cadence=rerun.cadence,
        state=state,
        reason=reason,
        tests=tests,
        figures=figures,
    )


def _absolute(inputs: ExportInputs) -> ExportInputs:
    return ExportInputs(
        export=inputs.export.absolute(),
        checkpoint=inputs.checkpoint.absolute(),
        corpora=tuple(corpus.absolute() for corpus in inputs.corpora),
    )


def _read_counts(junit: Path) -> TestCounts | str:
    """The counters of the report at `junit`, or the reason there are none: `FAIL_NO_JUNIT` when no
    report was written, `FAIL_BAD_JUNIT` when there is one that cannot be read as a JUnit report
    (not well-formed XML, an unknown encoding, no counters, a file that is no file)."""
    try:
        data = junit.read_bytes()
    except FileNotFoundError:
        return FAIL_NO_JUNIT
    except OSError:
        return FAIL_BAD_JUNIT
    try:
        return read_junit(data)
    except DataError:
        return FAIL_BAD_JUNIT


def _pytest_failure(
    counts: TestCounts, exit_code: int, floor: int, allowed_skips: int
) -> str | None:
    """Why a pytest rerun whose report was read is not `ok`, or `None` when it is."""
    if counts.failed + counts.errors > 0:
        return fail_tests(counts.failed, counts.errors)
    if counts.passed < floor:
        return fail_floor(counts.passed, floor)
    if counts.skipped > allowed_skips:
        return fail_skipped(counts.skipped, allowed_skips)
    if exit_code != 0:
        return fail_pytest_exit(exit_code)
    return None


def _read_figures(path: Path) -> dict[str, float | int] | None:
    """`exact`, `sentences` and `mean_abs_delta` of the parity result at `path`, or `None` when
    it is missing, is not JSON or holds anything else than counts and a finite delta."""
    try:
        document: Any = json.loads(path.read_bytes())
    except OSError, ValueError:
        return None
    if not isinstance(document, dict):
        return None
    exact, sentences = document.get("exact"), document.get("sentences")
    delta = document.get("mean_abs_delta")
    if not (_is_count(exact) and _is_count(sentences)):
        return None
    if isinstance(delta, bool) or not isinstance(delta, int | float):
        return None
    if not math.isfinite(delta) or delta < 0:
        return None
    return {"exact": exact, "sentences": sentences, "mean_abs_delta": float(delta)}


def _is_count(value: object) -> TypeGuard[int]:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0
