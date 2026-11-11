"""The size of a log that the drift step judges (D14 fix wave 1, finding M1).

`uv lock --upgrade --dry-run` writes its text to a log and the drift step decides from that text.
A log is read whole or not at all: a log of at most `LOG_LIMIT` bytes is judged, and a longer one
is a failure of the step with its own closed reason and is never parsed, because the first
`LOG_LIMIT` bytes of a long table can end on a line boundary and look like a complete one (the
lines after the cut, an `error:` among them, would be lost and the table would be called `drift`).
The two probes of the review (a table of 3,000 updates and an error line after the 64 KiB mark,
and a cut that lands on a whole line) are the last two tests here.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from rinchen.maintain import states
from rinchen.maintain.check_steps import DRIFT_LOG, drift_step
from rinchen.maintain.commands import LOG_LIMIT, CommandResult, read_complete_log
from rinchen.maintain.drift import drift_outcome
from rinchen.maintain.report import Step
from rinchen.maintain.states import fail_log_too_long
from tests.support.maintain_check import shipped_config, uv_call
from tests.support.maintain_fakes import FakeRunner
from tests.support.maintain_scans import closed_reason
from tests.support.no_network import no_network  # noqa: F401 - the fixture of the mark
from tests.support.uv_outputs import lock_of

pytestmark = pytest.mark.usefixtures("no_network")

TABLE_SIZE = 3000
NAMES = {f"pkg{index}": ("1.0.0",) for index in range(TABLE_SIZE)}


def log_of(tmp_path: Path) -> Path:
    return tmp_path / "logs" / "drift.log"


def update_table(count: int) -> str:
    return "".join(f"Update pkg{index} v1.0.0 -> v1.0.1\n" for index in range(count))


def padded_to(text: str, size: int) -> str:
    """`text` with the spaces that make it exactly `size` bytes put after its first line's text
    (the parser trims them: a line is judged without the spaces at its ends)."""
    first, rest = text.split("\n", 1)
    missing = size - len(text.encode())
    assert missing >= 0
    return first + " " * missing + "\n" + rest


def dry_run_step(text: str, *, tmp_path: Path, exit_code: int = 0) -> Step:
    runner = FakeRunner().script(
        CommandResult(exit_code, False),
        log_text=text,
        when=uv_call("--upgrade", "--dry-run"),
    )
    return drift_step(
        shipped_config(),
        lock_of(NAMES),
        "uv",
        runner,
        {},
        tmp_path,
        tmp_path / "out",
        enabled=True,
    )


# ---- reading a log whole ---------------------------------------------------------------------------


def test_the_limit_is_one_constant_of_64_kib() -> None:
    assert LOG_LIMIT == 64 * 1024


@pytest.mark.parametrize("size", [0, 1, 4, LOG_LIMIT - 1, LOG_LIMIT])
def test_a_log_of_at_most_the_limit_is_read_whole(tmp_path: Path, size: int) -> None:
    log = log_of(tmp_path)
    log.parent.mkdir()
    log.write_bytes(b"a" * size)

    assert read_complete_log(log, limit=max(size, 4)) == "a" * size


@pytest.mark.parametrize("size", [LOG_LIMIT + 1, LOG_LIMIT + 2, 3 * LOG_LIMIT])
def test_a_log_of_more_than_the_limit_is_not_read_at_all(
    tmp_path: Path, size: int
) -> None:
    log = log_of(tmp_path)
    log.parent.mkdir()
    log.write_bytes(b"a" * size)

    assert read_complete_log(log) is None


def test_the_default_limit_is_the_constant(tmp_path: Path) -> None:
    log = log_of(tmp_path)
    log.parent.mkdir()
    log.write_bytes(b"a" * LOG_LIMIT)
    assert read_complete_log(log) == "a" * LOG_LIMIT

    log.write_bytes(b"a" * (LOG_LIMIT + 1))
    assert read_complete_log(log) is None


def test_the_limit_counts_bytes_and_not_characters(tmp_path: Path) -> None:
    log = log_of(tmp_path)
    log.parent.mkdir()
    log.write_text(chr(0x0416) * 6, encoding="utf-8")  # 6 characters, 12 bytes

    assert read_complete_log(log, limit=12) == chr(0x0416) * 6
    assert read_complete_log(log, limit=11) is None


def test_a_missing_log_is_the_empty_text(tmp_path: Path) -> None:
    assert read_complete_log(log_of(tmp_path)) == ""


def test_bytes_that_are_not_utf_8_are_replaced_and_still_count_as_bytes(
    tmp_path: Path,
) -> None:
    log = log_of(tmp_path)
    log.parent.mkdir()
    log.write_bytes(b"a\xffb")

    assert read_complete_log(log, limit=3) == "a" + chr(0xFFFD) + "b"
    assert read_complete_log(log, limit=2) is None


def test_a_negative_limit_is_refused(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="negative"):
        read_complete_log(log_of(tmp_path), limit=-1)


# ---- the reason -----------------------------------------------------------------------------------


def test_the_reason_names_the_limit_in_bytes() -> None:
    assert fail_log_too_long(65536) == "the output of uv is longer than 65536 bytes"
    assert fail_log_too_long(1) == "the output of uv is longer than 1 bytes"


def test_the_reason_is_a_closed_one() -> None:
    assert closed_reason(fail_log_too_long(LOG_LIMIT))
    assert "the output of uv is longer than <limit> bytes" in states.ALL_REASONS


# ---- the decision ---------------------------------------------------------------------------------


def test_a_text_that_was_too_long_is_a_failure_with_the_reason() -> None:
    outcome = drift_outcome(
        exit_code=0,
        timed_out=False,
        text=None,
        lock=lock_of(NAMES),
        rules=[],
        timeout_seconds=300,
        log_limit=1000,
    )

    assert (outcome.state, outcome.reason, outcome.entries, outcome.warnings) == (
        "fail",
        fail_log_too_long(1000),
        (),
        0,
    )


@pytest.mark.parametrize(
    ("exit_code", "timed_out", "reason"),
    [
        (None, True, states.fail_timeout(300)),
        (None, False, states.FAIL_UV_MISSING),
        (2, False, states.fail_uv_exit(2)),
    ],
)
def test_the_exit_of_uv_decides_before_the_length_of_its_text(
    exit_code: int | None, timed_out: bool, reason: str
) -> None:
    outcome = drift_outcome(
        exit_code=exit_code,
        timed_out=timed_out,
        text=None,
        lock=lock_of(NAMES),
        rules=[],
        timeout_seconds=300,
        log_limit=1000,
    )

    assert (outcome.state, outcome.reason) == ("fail", reason)


# ---- the step --------------------------------------------------------------------------------------


def test_a_log_of_exactly_the_limit_is_judged(tmp_path: Path) -> None:
    text = padded_to(update_table(2000), LOG_LIMIT)
    assert len(text.encode()) == LOG_LIMIT

    step = dry_run_step(text, tmp_path=tmp_path)

    assert (step.state, step.reason) == ("drift", None)
    assert len(step.detail["entries"]) == 2000


def test_a_log_of_one_byte_more_is_a_failure_and_nothing_of_it_is_a_table(
    tmp_path: Path,
) -> None:
    text = padded_to(update_table(2000), LOG_LIMIT + 1)
    assert len(text.encode()) == LOG_LIMIT + 1

    step = dry_run_step(text, tmp_path=tmp_path)

    assert (step.state, step.reason) == ("fail", fail_log_too_long(LOG_LIMIT))
    assert step.detail == {}


def test_the_drift_log_is_where_the_step_reads(tmp_path: Path) -> None:
    step = dry_run_step(update_table(1), tmp_path=tmp_path)

    assert step.state == "drift"
    assert (tmp_path / "out" / "logs" / DRIFT_LOG).is_file()


def test_the_probe_of_a_long_table_with_an_error_line_after_the_mark(
    tmp_path: Path,
) -> None:
    text = (
        update_table(TABLE_SIZE) + "error: something went wrong after the 64 KiB mark\n"
    )
    assert len(text.encode()) > LOG_LIMIT

    step = dry_run_step(text, tmp_path=tmp_path)

    assert (step.state, step.reason) == ("fail", fail_log_too_long(LOG_LIMIT))


def test_the_probe_of_a_cut_that_lands_on_a_whole_line(tmp_path: Path) -> None:
    # The first LOG_LIMIT bytes are a clean table that ends on a line boundary, and an `error:` line
    # follows. Reading only the prefix called this `drift`, with the error line lost.
    table = update_table(2000)
    table = padded_to(table, LOG_LIMIT)
    text = table + "error: boom\n"
    assert text.encode()[:LOG_LIMIT].endswith(b"\n")

    step = dry_run_step(text, tmp_path=tmp_path)

    assert (step.state, step.reason) == ("fail", fail_log_too_long(LOG_LIMIT))
    assert step.detail == {}


def test_a_long_log_of_a_failed_run_keeps_the_reason_of_its_exit_code(
    tmp_path: Path,
) -> None:
    step = dry_run_step(
        update_table(TABLE_SIZE) + "error: boom\n", tmp_path=tmp_path, exit_code=2
    )

    assert (step.state, step.reason) == ("fail", states.fail_uv_exit(2))
