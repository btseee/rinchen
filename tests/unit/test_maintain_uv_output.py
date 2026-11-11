"""The parser of a `uv lock ... --dry-run` output (D14, spec M6): every non-blank line is exactly
one of a boilerplate line, a warning, the positive line `No lockfile changes detected`, a change
line, or UNRECOGNISED, and the parser never raises.

The three recorded outputs are in `tests/support/uv_outputs.py`: two MEASURED (nothing to change,
an offline failure) and one HAND-WRITTEN (the lines of an update, an add and a remove: their real
format is UNMEASURED, so the grammar is positive and every other line is counted, never skipped).
The line is the text between `\\n` characters; the escape sequences `ESC [ digits ; letter` and one
trailing `\\r` are removed and the spaces and tabs at both ends are trimmed; the tokens of a line
are separated by exactly one space.
"""

from __future__ import annotations

import dataclasses

import pytest

from rinchen.maintain.uv_output import Change, DryRun, parse_dry_run
from tests.support.no_network import no_network  # noqa: F401 - the fixture of the mark
from tests.support.uv_outputs import (
    CHANGES_HAND_WRITTEN,
    NO_CHANGE_MEASURED,
    OFFLINE_FAILURE_MEASURED,
)

pytestmark = pytest.mark.usefixtures("no_network")

ESC = chr(0x1B)
NBSP = chr(0xA0)
LINE_SEPARATOR = chr(0x2028)
NOTHING = DryRun(changes=(), no_changes=False, warnings=0, unrecognised=0)

SAMPLE_CHANGES = (
    Change("update", "numpy", "2.5.4", "2.5.5"),
    Change("update", "rich", "15.0.0", "15.0.1"),
    Change("update", "huggingface-hub", "2.3.0", "2.3.1"),
    Change("update", "safetensors", "0.8.0", "0.8.1"),
    Change("update", "nvidia-cudnn-cu13", "9.24.0.43", "9.25.0.10"),
    Change("add", "tomli-w", None, "1.2.0"),
    Change("remove", "sortedcontainers", "2.4.0", None),
)


def only_change(line: str) -> Change:
    """The one change a text of one line parses to."""
    run = parse_dry_run(line + "\n")
    assert run.unrecognised == 0, line
    assert len(run.changes) == 1, line
    return run.changes[0]


def unrecognised_count(line: str) -> int:
    return parse_dry_run(line + "\n").unrecognised


# --- the three recorded outputs ---------------------------------------------------------------


def test_the_measured_no_change_output_is_the_positive_line_alone() -> None:
    assert parse_dry_run(NO_CHANGE_MEASURED) == DryRun(
        changes=(), no_changes=True, warnings=0, unrecognised=0
    )


def test_the_hand_written_output_gives_its_seven_changes_in_the_order_of_the_text() -> (
    None
):
    run = parse_dry_run(CHANGES_HAND_WRITTEN)
    assert run.changes == SAMPLE_CHANGES
    assert (run.no_changes, run.warnings, run.unrecognised) == (False, 0, 0)


def test_the_measured_offline_failure_leaves_its_error_cause_and_hint_lines_unrecognised() -> (
    None
):
    run = parse_dry_run(OFFLINE_FAILURE_MEASURED)
    assert run == DryRun(changes=(), no_changes=False, warnings=0, unrecognised=4)


@pytest.mark.parametrize("sample", [OFFLINE_FAILURE_MEASURED, CHANGES_HAND_WRITTEN])
def test_blank_lines_are_never_counted(sample: str) -> None:
    padded = "\n\n" + sample.replace("\n", "\n\n") + "\n \t \n"
    assert parse_dry_run(padded) == parse_dry_run(sample)


@pytest.mark.parametrize(
    "text", ["", "\n", "\n\n\n", "   \n\t\n", "\r\n\r\n", " \t ", ESC + "[0m\n"]
)
def test_an_empty_text_or_only_blank_lines_is_no_result_at_all(text: str) -> None:
    assert parse_dry_run(text) == NOTHING


def test_the_value_objects_are_frozen_and_slotted() -> None:
    change = SAMPLE_CHANGES[0]
    run = parse_dry_run(NO_CHANGE_MEASURED)
    for value, field in ((change, "name"), (run, "warnings")):
        with pytest.raises(dataclasses.FrozenInstanceError):
            setattr(value, field, "x")
        assert not hasattr(value, "__dict__")


def test_a_change_holds_an_old_version_for_an_update_and_a_remove_and_a_new_one_for_an_update_and_an_add() -> (
    None
):
    update, add, remove = SAMPLE_CHANGES[0], SAMPLE_CHANGES[5], SAMPLE_CHANGES[6]
    assert (update.old, update.new) == ("2.5.4", "2.5.5")
    assert (add.old, add.new) == (None, "1.2.0")
    assert (remove.old, remove.new) == ("2.4.0", None)


# --- the spelling of a change line --------------------------------------------------------------

SPELLINGS = [
    ("Update numpy v2.5.4 -> v2.5.5", Change("update", "numpy", "2.5.4", "2.5.5")),
    ("Updated numpy v2.5.4 -> v2.5.5", Change("update", "numpy", "2.5.4", "2.5.5")),
    ("Update numpy 2.5.4 -> 2.5.5", Change("update", "numpy", "2.5.4", "2.5.5")),
    ("Updated numpy 2.5.4 -> v2.5.5", Change("update", "numpy", "2.5.4", "2.5.5")),
    ("Update numpy v2.5.4 -> 2.5.5", Change("update", "numpy", "2.5.4", "2.5.5")),
    ("Add tomli-w v1.2.0", Change("add", "tomli-w", None, "1.2.0")),
    ("Added tomli-w v1.2.0", Change("add", "tomli-w", None, "1.2.0")),
    ("Add tomli-w 1.2.0", Change("add", "tomli-w", None, "1.2.0")),
    ("Added tomli-w 1.2.0", Change("add", "tomli-w", None, "1.2.0")),
    ("Remove tomli-w v1.2.0", Change("remove", "tomli-w", "1.2.0", None)),
    ("Removed tomli-w v1.2.0", Change("remove", "tomli-w", "1.2.0", None)),
    ("Remove tomli-w 1.2.0", Change("remove", "tomli-w", "1.2.0", None)),
    ("Removed tomli-w 1.2.0", Change("remove", "tomli-w", "1.2.0", None)),
]


@pytest.mark.parametrize(("line", "expected"), SPELLINGS)
def test_both_tenses_and_a_missing_v_give_one_change(
    line: str, expected: Change
) -> None:
    assert only_change(line) == expected


@pytest.mark.parametrize(
    ("line", "expected"),
    [
        (
            "Update Huggingface_Hub v2.3.0 -> v2.3.1",
            Change("update", "huggingface-hub", "2.3.0", "2.3.1"),
        ),
        ("Add Zope.Interface v7.0", Change("add", "zope-interface", None, "7.0")),
        ("Add a__b--c..d v1", Change("add", "a-b-c-d", None, "1")),
        (
            "Remove NVIDIA_cudnn_CU13 v9.24",
            Change("remove", "nvidia-cudnn-cu13", "9.24", None),
        ),
        ("Add 1abc v1", Change("add", "1abc", None, "1")),
        ("Add x v1", Change("add", "x", None, "1")),
    ],
)
def test_a_name_is_put_in_pep_503_form(line: str, expected: Change) -> None:
    assert only_change(line) == expected


@pytest.mark.parametrize(
    ("line", "expected"),
    [
        (
            "Update torch v2.14.1+cu130 -> v2.15.0+cu130",
            Change("update", "torch", "2.14.1+cu130", "2.15.0+cu130"),
        ),
        ("Update x v1.0rc1 -> v1.0", Change("update", "x", "1.0rc1", "1.0")),
        ("Update x 1.0rc1 -> 1.0rc2", Change("update", "x", "1.0rc1", "1.0rc2")),
        ("Update x v1!2.0 -> v1!2.1", Change("update", "x", "1!2.0", "1!2.1")),
        ("Add x v1.0.post1", Change("add", "x", None, "1.0.post1")),
        ("Add x v1.0.dev3+local.7", Change("add", "x", None, "1.0.dev3+local.7")),
        ("Remove x v2026.11.11", Change("remove", "x", "2026.11.11", None)),
        ("Add x v1_0-beta", Change("add", "x", None, "1_0-beta")),
        ("Add x v7", Change("add", "x", None, "7")),
    ],
)
def test_a_local_label_a_pre_release_and_an_epoch_are_part_of_the_version(
    line: str, expected: Change
) -> None:
    assert only_change(line) == expected


# --- the noise around a line --------------------------------------------------------------------

LINE = "Update numpy v2.5.4 -> v2.5.5"
LINE_CHANGE = Change("update", "numpy", "2.5.4", "2.5.5")
BOLD, GREEN, RESET = ESC + "[1m", ESC + "[32m", ESC + "[0m"
ANSI_PLACEMENTS = [
    BOLD + LINE + RESET,
    BOLD + "Update" + RESET + " numpy v2.5.4 -> v2.5.5",
    "Update " + GREEN + "numpy" + RESET + " v2.5.4 -> v2.5.5",
    "Update numpy " + GREEN + "v2.5.4" + RESET + " -> " + BOLD + "v2.5.5" + RESET,
    "Update numpy v2.5.4 " + GREEN + "->" + RESET + " v2.5.5",
    "Update nu" + BOLD + "mpy" + RESET + " v2.5.4 -> v2.5.5",
    "Update numpy v2.5.4 -> v2.5." + GREEN + "5" + RESET,
    RESET + LINE,
    LINE + RESET + RESET,
    GREEN + BOLD + LINE + RESET,
    ESC + "[1;32m" + LINE + ESC + "[0m",
    ESC + "[m" + LINE,
    "Update numpy v2.5.4 -> v2.5.5" + ESC + "[K",
]


@pytest.mark.parametrize("line", ANSI_PLACEMENTS)
def test_escape_sequences_anywhere_in_a_change_line_are_removed(line: str) -> None:
    assert only_change(line) == LINE_CHANGE


@pytest.mark.parametrize(
    "line",
    [
        "Using CPython 3.14.10 interpreter at: /opt/python/bin/python3",
        "Resolved 187 packages in 1ms",
        "warning: a thing",
        "No lockfile changes detected",
    ],
)
def test_escape_sequences_around_the_other_classes_of_line_are_removed(
    line: str,
) -> None:
    plain = parse_dry_run(line + "\n")
    assert plain.unrecognised == 0
    assert parse_dry_run(BOLD + line + RESET + "\n") == plain
    assert parse_dry_run(GREEN + line[:3] + RESET + line[3:] + "\n") == plain


@pytest.mark.parametrize(
    "text",
    [
        ESC,
        ESC + "[1",
        ESC + "[",
        ESC + "[1;",
        ESC + "1m",
        ESC + "[1m" + ESC,
        ESC + "[?25l",
    ],
)
def test_a_stray_escape_that_is_not_a_complete_sequence_is_an_unrecognised_line(
    text: str,
) -> None:
    assert unrecognised_count(text) == 1


@pytest.mark.parametrize(
    "sample", [NO_CHANGE_MEASURED, OFFLINE_FAILURE_MEASURED, CHANGES_HAND_WRITTEN]
)
def test_crlf_line_ends_change_nothing(sample: str) -> None:
    assert parse_dry_run(sample.replace("\n", "\r\n")) == parse_dry_run(sample)


def test_a_last_line_without_a_line_end_is_read() -> None:
    assert parse_dry_run(CHANGES_HAND_WRITTEN.rstrip("\n")) == parse_dry_run(
        CHANGES_HAND_WRITTEN
    )
    assert parse_dry_run("No lockfile changes detected").no_changes


@pytest.mark.parametrize(
    "line",
    [
        "  " + LINE,
        LINE + "   ",
        "\t" + LINE + "\t",
        " \t " + LINE + " \t ",
        LINE + " \r",
        "  " + BOLD + LINE + RESET + "  ",
    ],
)
def test_spaces_and_tabs_at_both_ends_of_a_line_are_trimmed(line: str) -> None:
    assert only_change(line) == LINE_CHANGE


@pytest.mark.parametrize(
    "line",
    [
        "Update  numpy v2.5.4 -> v2.5.5",
        "Update numpy  v2.5.4 -> v2.5.5",
        "Update numpy v2.5.4  -> v2.5.5",
        "Update numpy v2.5.4 ->  v2.5.5",
        "Update\tnumpy v2.5.4 -> v2.5.5",
        "Update numpy v2.5.4 ->\tv2.5.5",
        "Update" + NBSP + "numpy v2.5.4 -> v2.5.5",
        "Update numpy v2.5.4 -> v2.5.5" + NBSP,
        "Add\ttomli-w v1.2.0",
        "Add tomli-w  v1.2.0",
        "No  lockfile changes detected",
        "No lockfile\tchanges detected",
    ],
)
def test_tokens_are_separated_by_exactly_one_space(line: str) -> None:
    assert unrecognised_count(line) == 1


def test_a_carriage_return_inside_a_line_makes_it_unrecognised_not_two_lines() -> None:
    run = parse_dry_run(LINE + "\r" + LINE + "\n")
    assert (run.changes, run.unrecognised) == ((), 1)


def test_a_unicode_line_separator_inside_a_line_makes_it_unrecognised_not_two_lines() -> (
    None
):
    run = parse_dry_run(LINE + LINE_SEPARATOR + LINE + "\n")
    assert (run.changes, run.unrecognised) == ((), 1)


def test_only_one_trailing_carriage_return_is_removed() -> None:
    assert unrecognised_count(LINE + "\r\r") == 1


# --- warnings, the boilerplate and the positive line --------------------------------------------


def test_a_warning_is_counted_and_is_never_a_change() -> None:
    run = parse_dry_run("warning: x\n" + LINE + "\nwarning: y z\nwarning:\n")
    assert run == DryRun(
        changes=(LINE_CHANGE,), no_changes=False, warnings=3, unrecognised=0
    )


def test_a_warning_that_looks_like_a_change_is_still_only_a_warning() -> None:
    run = parse_dry_run("warning: Update numpy v1 -> v2\n")
    assert (run.changes, run.warnings, run.unrecognised) == ((), 1, 0)


@pytest.mark.parametrize(
    "line",
    [
        "Warning: x",
        "warning:x",
        "warn: x",
        "warning x",
        "WARNING: x",
        "warning;: x",
        "warning:\tx",
    ],
)
def test_a_line_that_is_not_exactly_a_warning_is_unrecognised(line: str) -> None:
    assert parse_dry_run(line + "\n") == DryRun((), False, 0, 1)


@pytest.mark.parametrize(
    "line",
    [
        "Using CPython 3.14.10 interpreter at: /opt/python/bin/python3",
        "Using CPython 3.14.10 interpreter at: C:\\Python314\\python.exe",
        "Using CPython 3.14.10+freethreaded interpreter at: /a b/c",
        "Resolved 187 packages in 1ms",
        "Resolved 1 package in 2.31s",
        "Resolved 0 packages in 0.00s",
        "Resolved 12 packages in 3m",
    ],
)
def test_the_boilerplate_lines_are_skipped_and_not_counted(line: str) -> None:
    assert parse_dry_run(line + "\n") == NOTHING


@pytest.mark.parametrize(
    "line",
    [
        "Using CPython interpreter at: /x",
        "Using CPython 3.14.10 interpreter at:",
        "Using CPython 3.14.10 interpreter at: ",
        "Using CPython 3.14.10 interpreter /x",
        "Using Python 3.14.10 interpreter at: /x",
        "using CPython 3.14.10 interpreter at: /x",
        "Using CPython 3.14.10 interpreter at /x",
        "Resolved packages in 1ms",
        "Resolved x packages in 1ms",
        "Resolved -1 packages in 1ms",
        "Resolved 1.5 packages in 1ms",
        "Resolved 187 packages 1ms",
        "Resolved 187 packages in",
        "Resolved 187 packages in 1 ms",
        "Resolved 187 package in 1ms extra",
        "Resolved " + chr(0x0663) + " packages in 1ms",
        "resolved 187 packages in 1ms",
    ],
)
def test_a_boilerplate_line_with_a_missing_or_malformed_part_is_unrecognised(
    line: str,
) -> None:
    assert parse_dry_run(line + "\n") == DryRun((), False, 0, 1)


@pytest.mark.parametrize(
    "line",
    [
        "No lockfile changes detected",
        "  No lockfile changes detected  ",
        BOLD + "No lockfile changes detected" + RESET,
    ],
)
def test_the_positive_line_sets_no_changes(line: str) -> None:
    assert parse_dry_run(line + "\n") == DryRun((), True, 0, 0)


@pytest.mark.parametrize(
    "line",
    [
        "No lockfile changes detected.",
        "no lockfile changes detected",
        "No lockfile changes detected!",
        "No lockfile change detected",
        "No lockfile changes",
        "Lockfile changes detected",
        "Lockfile changes detected:",
        "No changes detected",
    ],
)
def test_a_line_that_is_not_exactly_the_positive_line_is_unrecognised(
    line: str,
) -> None:
    assert parse_dry_run(line + "\n") == DryRun((), False, 0, 1)


def test_the_positive_line_and_change_lines_are_both_reported() -> None:
    run = parse_dry_run("No lockfile changes detected\n" + CHANGES_HAND_WRITTEN)
    assert run.no_changes is True
    assert run.changes == SAMPLE_CHANGES


# --- what is not a change line -------------------------------------------------------------------

NOT_A_CHANGE = [
    "Frobnicated numpy",
    "Update numpy v2.5.4",
    "Updated numpy 2.5.4",
    "Update numpy",
    "Update",
    "Update numpy -> v2.5.5",
    "Update numpy v2.5.4 ->",
    "Update numpy v2.5.4 -> ",
    "Update numpy v2.5.4 => v2.5.5",
    "Update numpy v2.5.4 " + chr(0x2192) + " v2.5.5",
    "Update numpy v2.5.4 <- v2.5.5",
    "Update numpy v2.5.5 -> v2.5.4 -> v2.5.3",
    "Update numpy v2.5.4 -> v2.5.5 extra",
    "Update numpy v2.5.4 -> v2.5.5 (marker)",
    "Add tomli-w",
    "Add tomli-w v1.0 -> v2.0",
    "Added tomli-w 1.0 -> 2.0",
    "Remove x v1 -> v2",
    "Removed x v1 -> v2",
    "Remove x",
    "update numpy v2.5.4 -> v2.5.5",
    "UPDATE numpy v2.5.4 -> v2.5.5",
    "Updating numpy v2.5.4 -> v2.5.5",
    "Upgrade numpy v2.5.4 -> v2.5.5",
    "Delete numpy v2.5.4",
    "Update numpy vv2.5.4 -> v2.5.5",
    "Update numpy v -> v2.5.5",
    "Update numpy v2.5.4 -> v",
    "Update numpy vx -> v1",
    "Update numpy v.1 -> v1",
    "Update numpy v-1 -> v1",
    "Update numpy v2.5.4> -> v2.5.5",
    "Update numpy v2.5.4, -> v2.5.5",
    "Update numpy v2.5.4 -> v2.5.5;",
    "Update numpy v2.5.4 -> v2.5.5" + chr(0x00E9),
    "Update numpy v" + chr(0x0662) + ".5.4 -> v2.5.5",
    "Update -numpy v2.5.4 -> v2.5.5",
    "Update numpy- v2.5.4 -> v2.5.5",
    "Update .numpy v2.5.4 -> v2.5.5",
    "Update numpy. v2.5.4 -> v2.5.5",
    "Update _numpy v2.5.4 -> v2.5.5",
    "Update numpy[extra] v2.5.4 -> v2.5.5",
    "Update numpy==2.5.4 v2.5.4 -> v2.5.5",
    "Update " + chr(0x043D) + "umpy v2.5.4 -> v2.5.5",
    "Add " + chr(0x212A) + "eras v1",
    "Add x v1 y",
    "numpy v2.5.4 -> v2.5.5",
    "v2.5.4 -> v2.5.5",
    "-> v2.5.5",
    "error: No solution found",
    "  cause: Because x",
    "hint: try again",
    "Lockfile changes detected",
    "Using CPython",
    "x",
    "-",
    ".",
    "0",
]


@pytest.mark.parametrize("line", NOT_A_CHANGE)
def test_a_line_outside_the_grammar_is_counted_never_skipped_and_never_a_change(
    line: str,
) -> None:
    assert parse_dry_run(line + "\n") == DryRun((), False, 0, 1)


def test_an_unknown_line_anywhere_among_the_sample_is_counted_and_the_changes_still_parse() -> (
    None
):
    lines = CHANGES_HAND_WRITTEN.splitlines()
    for position in range(len(lines) + 1):
        text = "\n".join([*lines[:position], "Frobnicated numpy", *lines[position:]])
        run = parse_dry_run(text)
        assert run.changes == SAMPLE_CHANGES, position
        assert (run.unrecognised, run.warnings, run.no_changes) == (1, 0, False), (
            position
        )


def test_every_unknown_line_is_counted_once() -> None:
    run = parse_dry_run("a\nb\n\nc\nUpdate numpy\n")
    assert run.unrecognised == 4


def test_the_same_change_twice_is_two_changes_the_parser_does_not_judge() -> None:
    run = parse_dry_run(LINE + "\n" + LINE + "\n")
    assert run.changes == (LINE_CHANGE, LINE_CHANGE)


# --- it reads any text ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        "\x00",
        "\x00" * 1000 + "\n",
        chr(0xD800),
        chr(0xDFFF) + "\n" + LINE,
        chr(0x10FFFF),
        "\n" * 100_000,
        "x" * 200_000,
        "Update " + "a" * 100_000 + " v1 -> v2",
        "Update " + "a-" * 50_000 + " v1 -> v2",
        "Update a v" + "1" * 100_000 + " -> v2",
        "Update a v" + "1." * 50_000 + " -> v2",
        "Update " + "a" * 100_000 + "!",
        "Using CPython " + "x" * 100_000,
        "Resolved " + "1" * 100_000 + " packages in 1ms",
        ESC + "[" + "1;" * 50_000,
        (ESC + "[1m") * 50_000,
        "\r" * 10_000,
        " " * 100_000 + LINE,
    ],
)
def test_no_text_makes_the_parser_raise_or_hang(text: str) -> None:
    run = parse_dry_run(text)
    assert isinstance(run, DryRun)
    assert run.unrecognised >= 0


def test_a_very_long_valid_line_is_still_read() -> None:
    name = "a" * 100_000
    version = "1" * 100_000
    run = parse_dry_run(f"Update {name} v{version} -> v{version}2\n")
    assert run.unrecognised == 0
    assert run.changes == (Change("update", name, version, version + "2"),)


def test_parsing_is_pure_and_repeatable() -> None:
    assert parse_dry_run(CHANGES_HAND_WRITTEN) == parse_dry_run(CHANGES_HAND_WRITTEN)
