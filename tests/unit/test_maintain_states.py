"""The states and the closed lists of reasons of `rinchen.maintain.states` (D14, spec M2).

A step ends in one of four states and, when it is not `ok`, a reason from a CLOSED list written
once in that module: a report never carries free text from a program's output. The list is pinned
here string by string, so a reworded reason, a new one that was not added to `ALL_REASONS` and a
template that does not match its function are each a failing test.
"""

from __future__ import annotations

import inspect
import re
from collections.abc import Callable
from typing import Any

import pytest

from rinchen.maintain import states
from rinchen.maintain.states import (
    ALL_REASONS,
    FAIL_BAD_JUNIT,
    FAIL_CONTRADICTION,
    FAIL_DOCTOR,
    FAIL_NO_FIGURES,
    FAIL_NO_JUNIT,
    FAIL_NO_RESULT,
    FAIL_TARGET,
    FAIL_UV_MISSING,
    SKIP_DOCTOR_SELF,
    SKIP_ENVIRONMENT,
    SKIP_GPU_CHECK,
    SKIP_NO_DRIFT,
    SKIP_NO_EXPORT,
    SKIP_NO_RERUNS,
    SKIP_NOT_SELECTED,
    SKIP_REASONS,
    STATES,
    fail_command_exit,
    fail_cross_check,
    fail_environment,
    fail_floor,
    fail_log_too_long,
    fail_pytest_exit,
    fail_skipped,
    fail_tests,
    fail_timeout,
    fail_unrecognised,
    fail_uv_exit,
    skip_not_due,
)
from tests.support.no_network import no_network  # noqa: F401 - the fixture of the mark

pytestmark = pytest.mark.usefixtures("no_network")

# Every reason, in the order of the interface: the constants and the templates of the functions.
EXPECTED_REASONS = (
    "a GPU check: maintain check never asks a device question",
    "disabled by --no-drift",
    "disabled by --no-reruns",
    "no export was given",
    "a doctor or environment check failed",
    "reported by the doctor itself",
    "not due at the <cadence> cadence",
    "not selected with --rerun",
    "uv was not found",
    "uv exited with code <n>",
    "timed out after <s> s",
    "the output of uv has <n> lines this version does not recognise",
    "the output of uv is longer than <limit> bytes",
    "uv printed no recognisable result",
    "uv printed a no-change line next to change lines",
    "uv reported a locked version that differs from uv.lock for <n> packages",
    "<m> packages differ from the lock, <e> are not in it",
    "the check failed",
    "pytest exited with code <n>",
    "<f> failed and <e> errors",
    "<p> passed, fewer than the floor <m>",
    "<k> skipped, more than the allowed <m>",
    "no JUnit report was written",
    "the JUnit report could not be read",
    "a target does not exist",
    "the command exited with code <n>",
    "the command wrote no readable parity.json",
)

# Where each template holds placeholders, and which.
EXPECTED_PLACEHOLDERS = {
    "not due at the <cadence> cadence": ["cadence"],
    "uv exited with code <n>": ["n"],
    "timed out after <s> s": ["s"],
    "the output of uv has <n> lines this version does not recognise": ["n"],
    "the output of uv is longer than <limit> bytes": ["limit"],
    "uv reported a locked version that differs from uv.lock for <n> packages": ["n"],
    "<m> packages differ from the lock, <e> are not in it": ["m", "e"],
    "pytest exited with code <n>": ["n"],
    "<f> failed and <e> errors": ["f", "e"],
    "<p> passed, fewer than the floor <m>": ["p", "m"],
    "<k> skipped, more than the allowed <m>": ["k", "m"],
    "the command exited with code <n>": ["n"],
}


def test_the_states_are_the_four_of_the_spec() -> None:
    assert STATES == ("ok", "drift", "fail", "skipped")


def test_the_reasons_are_the_closed_list_of_the_interface() -> None:
    assert ALL_REASONS == EXPECTED_REASONS


def test_the_constants_hold_the_strings_of_the_interface() -> None:
    assert SKIP_GPU_CHECK == EXPECTED_REASONS[0]
    assert SKIP_NO_DRIFT == "disabled by --no-drift"
    assert SKIP_NO_RERUNS == "disabled by --no-reruns"
    assert SKIP_NO_EXPORT == "no export was given"
    assert SKIP_ENVIRONMENT == "a doctor or environment check failed"
    assert SKIP_DOCTOR_SELF == "reported by the doctor itself"
    assert SKIP_NOT_SELECTED == "not selected with --rerun"
    assert FAIL_UV_MISSING == "uv was not found"
    assert FAIL_NO_RESULT == "uv printed no recognisable result"
    assert FAIL_CONTRADICTION == "uv printed a no-change line next to change lines"
    assert FAIL_DOCTOR == "the check failed"
    assert FAIL_NO_JUNIT == "no JUnit report was written"
    assert FAIL_BAD_JUNIT == "the JUnit report could not be read"
    assert FAIL_TARGET == "a target does not exist"
    assert FAIL_NO_FIGURES == "the command wrote no readable parity.json"


def test_the_fixed_skip_reasons_are_the_six_constants() -> None:
    assert SKIP_REASONS == (
        SKIP_GPU_CHECK,
        SKIP_NO_DRIFT,
        SKIP_NO_RERUNS,
        SKIP_NO_EXPORT,
        SKIP_ENVIRONMENT,
        SKIP_DOCTOR_SELF,
    )


def test_no_reason_is_listed_twice() -> None:
    assert len(set(ALL_REASONS)) == len(ALL_REASONS)


@pytest.mark.parametrize("reason", ALL_REASONS)
def test_a_reason_is_one_short_line_without_a_path_separator(reason: str) -> None:
    assert reason == reason.strip()
    assert reason.isascii()
    assert reason.isprintable()
    assert "/" not in reason
    assert "\\" not in reason
    assert 0 < len(reason) <= 100


@pytest.mark.parametrize("reason", ALL_REASONS)
def test_a_template_holds_its_placeholders_only_where_listed(reason: str) -> None:
    found = re.findall(r"<([a-z]+)>", reason)
    assert found == EXPECTED_PLACEHOLDERS.get(reason, [])
    # Nothing else in the line is angle-bracketed and no number is written into a template.
    assert reason.count("<") == len(found)
    assert reason.count(">") == len(found)
    assert not re.search(r"[0-9]", reason)


Call = tuple[Callable[..., str], tuple[Any, ...], str]

FUNCTION_CASES: list[Call] = [
    (skip_not_due, ("nightly",), "not due at the nightly cadence"),
    (skip_not_due, ("weekly",), "not due at the weekly cadence"),
    (fail_uv_exit, (0,), "uv exited with code 0"),
    (fail_uv_exit, (12345,), "uv exited with code 12345"),
    (fail_timeout, (0,), "timed out after 0 s"),
    (fail_timeout, (12345,), "timed out after 12345 s"),
    (
        fail_unrecognised,
        (0,),
        "the output of uv has 0 lines this version does not recognise",
    ),
    (
        fail_unrecognised,
        (12345,),
        "the output of uv has 12345 lines this version does not recognise",
    ),
    (fail_log_too_long, (0,), "the output of uv is longer than 0 bytes"),
    (fail_log_too_long, (65536,), "the output of uv is longer than 65536 bytes"),
    (
        fail_cross_check,
        (0,),
        "uv reported a locked version that differs from uv.lock for 0 packages",
    ),
    (
        fail_cross_check,
        (12345,),
        "uv reported a locked version that differs from uv.lock for 12345 packages",
    ),
    (fail_environment, (0, 0), "0 packages differ from the lock, 0 are not in it"),
    (
        fail_environment,
        (12, 12345),
        "12 packages differ from the lock, 12345 are not in it",
    ),
    (
        fail_environment,
        (12345, 12),
        "12345 packages differ from the lock, 12 are not in it",
    ),
    (fail_pytest_exit, (0,), "pytest exited with code 0"),
    (fail_pytest_exit, (12345,), "pytest exited with code 12345"),
    (fail_tests, (0, 0), "0 failed and 0 errors"),
    (fail_tests, (12, 12345), "12 failed and 12345 errors"),
    (fail_tests, (12345, 12), "12345 failed and 12 errors"),
    (fail_floor, (0, 0), "0 passed, fewer than the floor 0"),
    (fail_floor, (12, 12345), "12 passed, fewer than the floor 12345"),
    (fail_floor, (12345, 12), "12345 passed, fewer than the floor 12"),
    (fail_skipped, (0, 0), "0 skipped, more than the allowed 0"),
    (fail_skipped, (12, 12345), "12 skipped, more than the allowed 12345"),
    (fail_skipped, (12345, 12), "12345 skipped, more than the allowed 12"),
    (fail_command_exit, (0,), "the command exited with code 0"),
    (fail_command_exit, (12345,), "the command exited with code 12345"),
]


@pytest.mark.parametrize(("function", "args", "expected"), FUNCTION_CASES)
def test_each_function_fills_the_placeholders_of_its_template(
    function: Callable[..., str], args: tuple[Any, ...], expected: str
) -> None:
    assert function(*args) == expected


def test_the_two_values_of_a_function_are_not_interchangeable() -> None:
    assert fail_environment(1, 2) != fail_environment(2, 1)
    assert fail_tests(1, 2) != fail_tests(2, 1)
    assert fail_floor(1, 2) != fail_floor(2, 1)
    assert fail_skipped(1, 2) != fail_skipped(2, 1)


def test_every_function_of_the_module_has_its_template_in_the_closed_list() -> None:
    # A function that fills a template, called with a marker per placeholder, must give a string
    # of ALL_REASONS once the markers are put back: nothing is built outside the list.
    functions = [
        member
        for name, member in inspect.getmembers(states, inspect.isfunction)
        if name.startswith(("skip_", "fail_")) and member.__module__ == states.__name__
    ]
    templates = [reason for reason in ALL_REASONS if "<" in reason]
    assert len(functions) == len(templates) == len(EXPECTED_PLACEHOLDERS)
    for function in functions:
        arity = len(inspect.signature(function).parameters)
        markers = [f"@{index}@" for index in range(arity)]
        text = function(*markers)
        pattern = re.compile(r"@\d@")
        shape = pattern.sub("<>", text)
        matching = [
            template
            for template in templates
            if re.sub(r"<[a-z]+>", "<>", template) == shape
        ]
        assert len(matching) == 1, function.__name__


def test_every_constant_of_the_module_is_in_the_closed_list() -> None:
    constants = {
        name: value
        for name, value in vars(states).items()
        if name.startswith(("SKIP_", "FAIL_")) and isinstance(value, str)
    }
    assert len(constants) == 7 + 8
    for name, value in constants.items():
        assert value in ALL_REASONS, name
