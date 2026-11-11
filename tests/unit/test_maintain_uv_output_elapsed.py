"""The elapsed time of uv's `Resolved <n> packages in <elapsed>` line (D14 fix wave 1, deferred
minors of Task 3).

uv prints that line with its own formatter: milliseconds below a second (`1ms`, `0ms`), seconds with
two decimals from a second (`2.31s`) and minutes with two-digit seconds from a minute (`1m 02s`).
The last has a space, which made the line two tokens and one unrecognised line (a failed night); the
parser now takes exactly those three spellings. The three are read from uv's source in memory and
`1m 02s` and `3.21s` are HAND-WRITTEN and UNMEASURED (`tests/support/uv_outputs.py` tags every
spelling). Anything else is still unrecognised and counted, so that a format that moved fails loudly
and is never skipped as boilerplate: that includes a spelling that is not uv's, `1m02s`, `0.5 s`,
`-1s` or a word.
"""

from __future__ import annotations

import pytest

from rinchen.maintain import states
from rinchen.maintain.drift import drift_outcome
from rinchen.maintain.uv_output import DryRun, parse_dry_run
from tests.support.no_network import no_network  # noqa: F401 - the fixture of the mark
from tests.support.uv_outputs import (
    ELAPSED_SPELLINGS,
    NO_CHANGE_MEASURED,
    lock_for_samples,
    resolved_line,
)

pytestmark = pytest.mark.usefixtures("no_network")

NOTHING = DryRun((), False, 0, 0)
ONE_UNRECOGNISED = DryRun((), False, 0, 1)
POSITIVE = "No lockfile changes detected"
ACCEPTED = [
    "0ms",
    "1ms",
    "15ms",
    "999ms",
    "007ms",
    "0.00s",
    "1.11s",
    "2.31s",
    "3.21s",
    "10.99s",
    "123.45s",
    "0m 00s",
    "1m 02s",
    "59m 59s",
    "100m 05s",
]
REFUSED = [
    "1m02s",
    "0.5 s",
    "-1s",
    "+1s",
    "banana",
    "1s",
    "3m",
    "5",
    "ms",
    "s",
    "1.5s",
    "1.234s",
    ".50s",
    "1.s",
    "1m 2s",
    "1m 002s",
    "1m 02",
    "1m  02s",
    "1 m 02s",
    "1m\t02s",
    "m 02s",
    "-1m 02s",
    "1m -2s",
    "1m 02s extra",
    "1ms extra",
    "1 ms",
    "1e3ms",
    "1,5s",
    "1ms.",
    chr(0x0663) + "ms",
    "1m " + chr(0x0660) + chr(0x0662) + "s",
    "1" * 40 + "ms ms",
]


# ---- the line --------------------------------------------------------------------------------------------


@pytest.mark.parametrize("elapsed", ACCEPTED)
def test_the_three_spellings_of_uv_are_boilerplate(elapsed: str) -> None:
    assert parse_dry_run(resolved_line(elapsed) + "\n") == NOTHING


@pytest.mark.parametrize("elapsed", REFUSED)
def test_any_other_elapsed_is_one_unrecognised_line(elapsed: str) -> None:
    assert parse_dry_run(resolved_line(elapsed) + "\n") == ONE_UNRECOGNISED


@pytest.mark.parametrize(("elapsed", "origin"), ELAPSED_SPELLINGS)
def test_every_recorded_spelling_is_accepted_and_says_where_it_comes_from(
    elapsed: str, origin: str
) -> None:
    assert parse_dry_run(resolved_line(elapsed) + "\n") == NOTHING
    assert origin.startswith(("MEASURED", "HAND-WRITTEN", "SECOND-HAND"))


def test_the_recorded_spellings_are_the_five_named_by_the_task() -> None:
    assert [elapsed for elapsed, _ in ELAPSED_SPELLINGS] == [
        "1ms",
        "2.31s",
        "1.11s",
        "3.21s",
        "1m 02s",
    ]
    tags = dict(ELAPSED_SPELLINGS)
    assert tags["1ms"].startswith("MEASURED")
    assert tags["1.11s"].startswith("SECOND-HAND")
    assert "UNMEASURED" in tags["3.21s"] and "UNMEASURED" in tags["1m 02s"]
    assert "UNMEASURED" not in tags["1ms"] and "UNMEASURED" not in tags["2.31s"]


@pytest.mark.parametrize("count", [0, 1, 2, 187, 12345])
def test_the_count_and_the_plural_are_still_checked(count: int) -> None:
    noun = "package" if count == 1 else "packages"
    line = f"Resolved {count} {noun} in 1m 02s\n"

    assert parse_dry_run(line) == NOTHING


@pytest.mark.parametrize(
    "line",
    [
        "Resolved packages in 1m 02s",
        "Resolved x packages in 1m 02s",
        "Resolved -1 packages in 1m 02s",
        "Resolved 187 packages 1m 02s",
        "Resolved 187 packages in",
        "resolved 187 packages in 1m 02s",
        "Resolved 187 packages in 1m 02s " + chr(0x0663),
    ],
)
def test_the_rest_of_the_line_is_still_checked(line: str) -> None:
    assert parse_dry_run(line + "\n") == ONE_UNRECOGNISED


def test_the_line_may_have_spaces_and_tabs_at_its_ends_and_a_carriage_return() -> None:
    line = "  \tResolved 187 packages in 1m 02s \t\r\n"

    assert parse_dry_run(line) == NOTHING


def test_an_escape_sequence_in_the_line_is_removed_first() -> None:
    line = "\x1b[1mResolved 187 packages in 1m 02s\x1b[0m\n"

    assert parse_dry_run(line) == NOTHING


# ---- the night ----------------------------------------------------------------------------------------------


@pytest.mark.parametrize(("elapsed", "origin"), ELAPSED_SPELLINGS)
def test_a_dry_run_with_no_change_is_ok_whatever_uv_spelled_the_time(
    elapsed: str, origin: str
) -> None:
    text = f"{resolved_line(elapsed)}\n{POSITIVE}\n"

    outcome = drift_outcome(
        exit_code=0,
        timed_out=False,
        text=text,
        lock=lock_for_samples(),
        rules=[],
        timeout_seconds=300,
    )

    assert (outcome.state, outcome.reason) == ("ok", None), origin


def test_the_two_token_form_in_the_measured_output_is_ok() -> None:
    text = NO_CHANGE_MEASURED.replace("1ms", "1m 02s")
    assert "1m 02s" in text

    outcome = drift_outcome(
        exit_code=0,
        timed_out=False,
        text=text,
        lock=lock_for_samples(),
        rules=[],
        timeout_seconds=300,
    )

    assert outcome.state == "ok"


def test_an_elapsed_that_is_not_uv_s_fails_the_step_loudly() -> None:
    text = f"{resolved_line('1m02s')}\n{POSITIVE}\n"

    outcome = drift_outcome(
        exit_code=0,
        timed_out=False,
        text=text,
        lock=lock_for_samples(),
        rules=[],
        timeout_seconds=300,
    )

    assert (outcome.state, outcome.reason) == ("fail", states.fail_unrecognised(1))
