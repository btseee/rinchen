"""Differential check of `rinchen tiny run` (D15 Task 7): over 500 random scenarios the command does what an
independent oracle says it must.

A scenario is what the fifteen fake stages do (each `ok`, `skipped` by itself, `fail`, raising a
`DataError` or raising something else) for how many seconds of a fake clock, under which budget and with
which options (`tests/support/tiny_oracle.py`, written before the command, restates the closed reasons as
literals and reads only the files on disk). The oracle derives the stages that ran and the record each one
came to, the verdict and the exit code, the bytes of `report.json` and `report.md`, the keys and values of
`manifest.json` and the printed lines, and `check` lists what differs. The test counts the scenarios that
differ and reports the count; it is 0.

The draw has to reach the endings that matter (a pass, a failed stage, a spent budget, a stage that raises,
a skipped stage, a run slower than its budget), and the oracle has to see a difference when there is one:
the second test alters what a run wrote, in five ways, and `check` names each.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pytest

from tests.support.no_network import no_network  # noqa: F401 - the fixture of the mark
from tests.support.tiny_cli import install, invoke
from tests.support.tiny_oracle import (
    BUDGET,
    Scenario,
    check,
    describe,
    draw,
    expect,
)

pytestmark = pytest.mark.usefixtures("no_network")

CASES = 500


def passing_seed() -> int:
    """The first seed whose scenario is a passing run."""
    return next(seed for seed in range(CASES) if expect(draw(seed)).verdict == "pass")


def run_scenario(
    scenario: Scenario, out: Path, monkeypatch: pytest.MonkeyPatch
) -> list[str]:
    recording, clock = install(monkeypatch, outcomes=scenario.table)
    result = invoke(scenario.arguments(out))
    return check(
        scenario,
        out,
        exit_code=result.exit_code,
        stdout=result.stdout,
        stderr=result.stderr,
        recording=recording,
        clock=clock,
    )


def test_the_command_equals_the_oracle_on_500_scenarios(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    mismatches: dict[int, list[str]] = {}
    for seed in range(CASES):
        scenario = draw(seed)
        problems = run_scenario(
            scenario, tmp_path / f"s{seed:03d}" / "out", monkeypatch
        )
        if problems:
            mismatches[seed] = [describe(scenario.outcomes), *problems]
    assert mismatches == {}, f"{len(mismatches)} of {CASES} scenarios differ"


def test_the_draw_reaches_the_endings_that_matter() -> None:
    endings: Counter[str] = Counter()
    for seed in range(CASES):
        scenario = draw(seed)
        expected = expect(scenario)
        endings[f"verdict {expected.verdict}"] += 1
        endings[f"exit {expected.exit_code}"] += 1
        reasons = [reason for _, reason in expected.states.values()]
        endings["budget spent"] += BUDGET in reasons
        endings["too slow"] += expected.total > scenario.budget
        kinds = {outcome.kind for outcome in scenario.outcomes[: len(expected.ran)]}
        for kind in ("skipped", "fail", "data_error", "crash"):
            endings[kind] += kind in kinds
        endings["qe flag"] += scenario.qe_flag is not None
        endings["threads"] += scenario.threads is not None
        endings["every stage ran"] += len(expected.ran) == 15
    assert endings["verdict pass"] >= 60
    assert endings["verdict fail"] >= 250
    assert endings["budget spent"] >= 20
    assert endings["too slow"] >= 10
    assert endings["every stage ran"] >= 100
    for kind in ("skipped", "fail", "data_error", "crash", "qe flag", "threads"):
        assert endings[kind] >= 40, kind


def test_the_draw_is_a_function_of_its_seed() -> None:
    assert draw(17) == draw(17)
    assert draw(17) != draw(18)


def alter_report(out: Path) -> None:
    path = out / "report.json"
    path.write_bytes(
        path.read_bytes().replace(b'"verdict": "pass"', b'"verdict": "fail"')
    )


def alter_markdown(out: Path) -> None:
    (out / "report.md").write_text("# changed\n", encoding="utf-8")


def drop_a_status(out: Path) -> None:
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    del manifest["status"]["spm"]
    (out / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def change_a_second(out: Path) -> None:
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    manifest["metrics"]["seconds.corpus"] += 1.0
    (out / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def leave_a_part_file(out: Path) -> None:
    (out / "report.json.0123.part").write_bytes(b"x")


@pytest.mark.parametrize(
    ("alteration", "named"),
    [
        (alter_report, "report.json"),
        (alter_markdown, "report.md"),
        (drop_a_status, "manifest status"),
        (change_a_second, "manifest metrics"),
        (leave_a_part_file, "the output directory holds"),
    ],
    ids=["report", "markdown", "status", "seconds", "part"],
)
def test_the_oracle_names_a_difference_in_what_a_run_wrote(
    alteration, named: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    scenario = draw(passing_seed())
    out = tmp_path / "out"
    recording, clock = install(monkeypatch, outcomes=scenario.table)
    result = invoke(scenario.arguments(out))
    assert result.exit_code == 0

    alteration(out)

    problems = check(
        scenario,
        out,
        exit_code=result.exit_code,
        stdout=result.stdout,
        stderr=result.stderr,
        recording=recording,
        clock=clock,
    )
    assert any(named in problem for problem in problems), problems


def test_the_oracle_names_a_wrong_exit_code_and_a_wrong_stage_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    scenario = draw(passing_seed())
    out = tmp_path / "out"
    recording, clock = install(monkeypatch, outcomes=scenario.table)
    result = invoke(scenario.arguments(out))
    recording.stages.reverse()

    problems = check(
        scenario,
        out,
        exit_code=4,
        stdout=result.stdout,
        stderr=result.stderr,
        recording=recording,
        clock=clock,
    )

    assert "exit code" in problems
    assert "stages called" in problems
