"""The states and the closed lists of reasons of `rinchen.tiny.states` (D15, spec P4).

A stage ends in one of three states and, unless it is `ok`, in a reason that is one of the strings
written once in that module: a report never carries a line of another program's output, an exception
text or a path. The list is pinned here string by string, so a reworded reason, a new one that was
not added to `ALL_REASONS` and a template that does not match its function are each a failing test.
A `StageRecord` refuses what the closed lists forbid when it is built.
"""

from __future__ import annotations

import dataclasses
import inspect
import math
import re
from collections.abc import Callable
from typing import Any

import pytest

from rinchen.config.tiny import STAGE_IDS
from rinchen.tiny import states
from rinchen.tiny.states import (
    ALL_REASONS,
    FAIL_BUDGET,
    FAIL_INPUT,
    PART_IDS,
    SKIP_EARLIER_FAILED,
    SKIP_NO_LABSE,
    SKIP_NO_LID,
    SKIP_NO_NEURAL,
    SKIP_NO_QE,
    SKIP_REASONS,
    SKIP_TOKEN_RATIO,
    STATES,
    PartRecord,
    StageRecord,
    fail_check,
    fail_environment,
    fail_exit,
    fail_timeout,
    is_closed_reason,
    is_fail_reason,
    is_skip_reason,
    unlisted_parts,
)
from tests.support.no_network import no_network  # noqa: F401 - the fixture of the mark

pytestmark = pytest.mark.usefixtures("no_network")

# Every reason, in the order of the interface: the skip constants, then the fail reasons.
EXPECTED_REASONS = (
    "no language-identification model: GlotLID is a download",
    "LaBSE needs a model that was not downloaded: --qe-fake runs the planted stand-in",
    "the QE scorer needs a model that was not downloaded: --qe-fake runs the planted stand-in",
    "no neural metric scorer: only the seam of score files was built",
    "the token-ratio check needs a SentencePiece model, which does not exist yet",
    "an earlier stage failed",
    "exit code <n>",
    "timed out after <s> s",
    "the wall-time budget is spent",
    "a check of the stage failed: <check>",
    "an input is missing or differs",
    "<m> packages differ from the lock, <e> are not in it",
)
EXPECTED_PLACEHOLDERS = {
    "exit code <n>": ["n"],
    "timed out after <s> s": ["s"],
    "a check of the stage failed: <check>": ["check"],
    "<m> packages differ from the lock, <e> are not in it": ["m", "e"],
}


def test_the_stage_ids_are_the_fifteen_of_the_spec_in_order() -> None:
    assert STAGE_IDS == (
        "environment",
        "fetch",
        "corpus",
        "filter",
        "spm",
        "converter",
        "synth",
        "prepare",
        "train",
        "average",
        "export",
        "parity",
        "serve",
        "robustness",
        "evaluate",
    )


def test_the_states_are_the_three_of_the_spec() -> None:
    assert STATES == ("ok", "skipped", "fail")


def test_the_reasons_are_the_closed_list_of_the_interface() -> None:
    assert ALL_REASONS == EXPECTED_REASONS


def test_the_constants_hold_the_strings_of_the_interface() -> None:
    assert SKIP_NO_LID == EXPECTED_REASONS[0]
    assert SKIP_NO_LABSE == EXPECTED_REASONS[1]
    assert SKIP_NO_QE == EXPECTED_REASONS[2]
    assert SKIP_NO_NEURAL == EXPECTED_REASONS[3]
    assert SKIP_TOKEN_RATIO == EXPECTED_REASONS[4]
    assert SKIP_EARLIER_FAILED == "an earlier stage failed"
    assert FAIL_BUDGET == "the wall-time budget is spent"
    assert FAIL_INPUT == "an input is missing or differs"


def test_the_skip_reasons_are_the_six_constants_in_order() -> None:
    assert SKIP_REASONS == (
        SKIP_NO_LID,
        SKIP_NO_LABSE,
        SKIP_NO_QE,
        SKIP_NO_NEURAL,
        SKIP_TOKEN_RATIO,
        SKIP_EARLIER_FAILED,
    )
    assert ALL_REASONS[:6] == SKIP_REASONS


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
    assert reason.count("<") == len(found)
    assert reason.count(">") == len(found)
    assert not re.search(r"[0-9]", reason)


Call = tuple[Callable[..., str], tuple[Any, ...], str]

FUNCTION_CASES: list[Call] = [
    (fail_exit, (0,), "exit code 0"),
    (fail_exit, (4,), "exit code 4"),
    (fail_exit, (-9,), "exit code -9"),
    (fail_exit, (12345,), "exit code 12345"),
    (fail_timeout, (0,), "timed out after 0 s"),
    (fail_timeout, (600,), "timed out after 600 s"),
    (fail_check, ("planted_noise",), "a check of the stage failed: planted_noise"),
    (fail_check, ("a1",), "a check of the stage failed: a1"),
    (fail_environment, (0, 0), "0 packages differ from the lock, 0 are not in it"),
    (fail_environment, (3, 12), "3 packages differ from the lock, 12 are not in it"),
    (fail_environment, (12, 3), "12 packages differ from the lock, 3 are not in it"),
]


@pytest.mark.parametrize(("function", "args", "expected"), FUNCTION_CASES)
def test_each_function_fills_the_placeholders_of_its_template(
    function: Callable[..., str], args: tuple[Any, ...], expected: str
) -> None:
    assert function(*args) == expected


def test_the_two_values_of_the_environment_reason_are_not_interchangeable() -> None:
    assert fail_environment(1, 2) != fail_environment(2, 1)


def test_every_function_of_the_module_has_its_template_in_the_closed_list() -> None:
    # A function that fills a template, called with a marker per placeholder, must give a string
    # of ALL_REASONS once the markers are put back: nothing is built outside the list.
    functions = [
        member
        for name, member in inspect.getmembers(states, inspect.isfunction)
        if name.startswith("fail_") and member.__module__ == states.__name__
    ]
    templates = [reason for reason in ALL_REASONS if "<" in reason]
    assert len(functions) == len(templates) == len(EXPECTED_PLACEHOLDERS)
    for function in functions:
        arity = len(inspect.signature(function).parameters)
        text = function(*[f"@{index}@" for index in range(arity)])
        shape = re.sub(r"@\d@", "<>", text)
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
    assert len(constants) == 6 + 2
    for name, value in constants.items():
        assert value in ALL_REASONS, name


# ---- is_closed_reason, is_skip_reason, is_fail_reason ----------------------------------------


@pytest.mark.parametrize("reason", SKIP_REASONS)
def test_a_skip_reason_is_a_skip_reason_and_closed_and_no_fail_reason(
    reason: str,
) -> None:
    assert is_skip_reason(reason)
    assert is_closed_reason(reason)
    assert not is_fail_reason(reason)


@pytest.mark.parametrize(
    "reason",
    [
        FAIL_BUDGET,
        FAIL_INPUT,
        "exit code 3",
        "exit code -9",
        "timed out after 120 s",
        "a check of the stage failed: planted_noise",
        "3 packages differ from the lock, 0 are not in it",
    ],
)
def test_a_fail_reason_is_a_fail_reason_and_closed_and_no_skip_reason(
    reason: str,
) -> None:
    assert is_fail_reason(reason)
    assert is_closed_reason(reason)
    assert not is_skip_reason(reason)


@pytest.mark.parametrize(
    "reason",
    [
        "",
        " ",
        "exit code",
        "exit code ",
        "exit code -",
        "timed out after  s",
        "exit code four",
        "exit code 4 ",
        "exit code 4\n",
        "exit code 4.0",
        "exit code 1e3",
        "exit code \u0664",
        " exit code 4",
        "timed out after 1 s and more",
        "timed out after -1.5 s",
        "a check of the stage failed: ",
        "a check of the stage failed: Planted",
        "a check of the stage failed: planted noise",
        "a check of the stage failed: ../etc",
        "a check of the stage failed: " + "a" * 33,
        "a check of the stage failed: 1abc",
        "a check of the stage failed: x\n",
        "an earlier stage failed.",
        "AN EARLIER STAGE FAILED",
        "an input is missing",
        "x packages differ from the lock, 0 are not in it",
        "/home/someone/project",
        "Traceback (most recent call last):",
        "<n>",
        "exit code <n>",
    ],
)
def test_free_text_and_near_misses_are_no_reason(reason: str) -> None:
    assert not is_closed_reason(reason)
    assert not is_skip_reason(reason)
    assert not is_fail_reason(reason)


def test_a_check_id_of_the_longest_closed_shape_is_a_reason() -> None:
    assert is_fail_reason("a check of the stage failed: a" + "b" * 31)


# ---- the records -------------------------------------------------------------------------------


def test_an_ok_record_has_no_reason() -> None:
    record = StageRecord("fetch", "ok")
    assert (record.id, record.state, record.reason, record.parts) == (
        "fetch",
        "ok",
        None,
        (),
    )
    assert record.detail == {}


def test_a_skipped_record_has_a_skip_reason() -> None:
    record = StageRecord("filter", "skipped", SKIP_EARLIER_FAILED)
    assert record.reason == SKIP_EARLIER_FAILED


def test_a_failed_record_has_a_fail_reason() -> None:
    record = StageRecord("train", "fail", fail_exit(4), {"exit": 4})
    assert record.reason == "exit code 4"
    assert record.detail == {"exit": 4}


@pytest.mark.parametrize("stage_id", STAGE_IDS)
def test_every_stage_id_is_accepted(stage_id: str) -> None:
    assert StageRecord(stage_id, "ok").id == stage_id


@pytest.mark.parametrize(
    "stage_id", ["", "tiny", "Fetch", "fetch ", "fetch\n", "07_train", "robustnes"]
)
def test_an_id_outside_the_stage_ids_is_refused(stage_id: str) -> None:
    with pytest.raises(ValueError, match="id") as caught:
        StageRecord(stage_id, "ok")
    assert stage_id.strip() not in str(caught.value) or not stage_id.strip()


@pytest.mark.parametrize("state", ["drift", "OK", "failed", "skip", "", None, 1, True])
def test_a_state_outside_the_three_is_refused(state: Any) -> None:
    with pytest.raises(ValueError, match="state"):
        StageRecord("fetch", state)


def test_a_skipped_record_without_a_reason_is_refused() -> None:
    with pytest.raises(ValueError, match="reason"):
        StageRecord("fetch", "skipped")


def test_a_failed_record_without_a_reason_is_refused() -> None:
    with pytest.raises(ValueError, match="reason"):
        StageRecord("fetch", "fail")


@pytest.mark.parametrize("reason", [SKIP_NO_LID, "exit code 1", "free text", ""])
def test_an_ok_record_with_a_reason_is_refused(reason: str) -> None:
    with pytest.raises(ValueError, match="reason"):
        StageRecord("fetch", "ok", reason)


@pytest.mark.parametrize(
    "reason", ["free text", "exit code 1", FAIL_BUDGET, "", "an earlier stage failed "]
)
def test_a_skipped_record_needs_one_of_the_skip_reasons(reason: str) -> None:
    with pytest.raises(ValueError, match="reason"):
        StageRecord("fetch", "skipped", reason)


@pytest.mark.parametrize(
    "reason", ["free text", SKIP_EARLIER_FAILED, SKIP_NO_LID, "", "exit code x"]
)
def test_a_failed_record_needs_one_of_the_fail_reasons(reason: str) -> None:
    with pytest.raises(ValueError, match="reason"):
        StageRecord("fetch", "fail", reason)


def test_a_record_is_frozen() -> None:
    record = StageRecord("fetch", "ok")
    with pytest.raises(dataclasses.FrozenInstanceError):
        record.state = "fail"  # type: ignore[misc]


def test_a_message_names_no_value_it_refuses() -> None:
    with pytest.raises(ValueError) as caught:
        StageRecord("fetch", "skipped", "ZZ-planted-reason")
    assert "ZZ-planted-reason" not in str(caught.value)
    with pytest.raises(ValueError) as caught:
        StageRecord("ZZ-planted-id", "ok")
    assert "ZZ-planted-id" not in str(caught.value)


# ---- the parts -----------------------------------------------------------------------------------


def test_the_stages_that_have_parts_are_the_filter_and_the_evaluation() -> None:
    assert dict(PART_IDS) == {
        "filter": (
            "script_ratio",
            "numbers",
            "length",
            "lid",
            "minhash",
            "labse",
            "qe",
        ),
        "evaluate": ("memorised", "heldout", "neural_seam", "report"),
    }


def test_a_filter_record_lists_its_parts_with_their_states() -> None:
    parts = (
        PartRecord("script_ratio", "ok"),
        PartRecord("lid", "skipped", SKIP_NO_LID),
        PartRecord("qe", "skipped", SKIP_NO_QE),
    )
    record = StageRecord("filter", "ok", parts=parts)
    assert record.parts == parts


def test_a_failed_stage_may_have_a_failed_part() -> None:
    record = StageRecord(
        "filter",
        "fail",
        fail_check("planted_noise"),
        parts=(PartRecord("numbers", "fail", fail_check("planted_noise")),),
    )
    assert record.parts[0].state == "fail"


def test_an_ok_stage_with_a_failed_part_is_refused() -> None:
    with pytest.raises(ValueError, match="part"):
        StageRecord(
            "filter", "ok", parts=(PartRecord("numbers", "fail", fail_exit(1)),)
        )


@pytest.mark.parametrize("state", ["drift", "OK", "", None, 3])
def test_a_part_with_a_fourth_state_is_refused(state: Any) -> None:
    with pytest.raises(ValueError, match="state"):
        PartRecord("numbers", state)


def test_a_skipped_part_needs_a_skip_reason_and_a_failed_part_a_fail_reason() -> None:
    with pytest.raises(ValueError, match="reason"):
        PartRecord("lid", "skipped")
    with pytest.raises(ValueError, match="reason"):
        PartRecord("lid", "skipped", "free text")
    with pytest.raises(ValueError, match="reason"):
        PartRecord("lid", "skipped", fail_exit(1))
    with pytest.raises(ValueError, match="reason"):
        PartRecord("lid", "fail")
    with pytest.raises(ValueError, match="reason"):
        PartRecord("lid", "fail", SKIP_NO_LID)
    with pytest.raises(ValueError, match="reason"):
        PartRecord("lid", "ok", SKIP_NO_LID)


def test_a_part_is_frozen() -> None:
    part = PartRecord("numbers", "ok")
    with pytest.raises(dataclasses.FrozenInstanceError):
        part.state = "fail"  # type: ignore[misc]


def test_a_part_outside_the_list_of_its_stage_is_refused() -> None:
    with pytest.raises(ValueError, match="part"):
        StageRecord("filter", "ok", parts=(PartRecord("memorised", "ok"),))
    with pytest.raises(ValueError, match="part"):
        StageRecord("evaluate", "ok", parts=(PartRecord("lid", "ok"),))


@pytest.mark.parametrize(
    "stage_id", [stage for stage in STAGE_IDS if stage not in PART_IDS]
)
def test_a_stage_without_parts_refuses_one(stage_id: str) -> None:
    with pytest.raises(ValueError, match="part"):
        StageRecord(stage_id, "ok", parts=(PartRecord("numbers", "ok"),))


def test_parts_given_as_a_list_are_kept_as_a_tuple_and_compare_equal() -> None:
    part = PartRecord("numbers", "ok")
    from_list = StageRecord("filter", "ok", parts=[part])  # type: ignore[arg-type]
    assert isinstance(from_list.parts, tuple)
    assert from_list == StageRecord("filter", "ok", parts=(part,))


def test_a_part_listed_twice_is_refused() -> None:
    with pytest.raises(ValueError, match="part"):
        StageRecord(
            "filter",
            "ok",
            parts=(PartRecord("numbers", "ok"), PartRecord("numbers", "ok")),
        )


def test_the_unlisted_parts_are_those_of_the_stage_the_record_does_not_list() -> None:
    record = StageRecord(
        "filter", "ok", parts=(PartRecord("qe", "ok"), PartRecord("numbers", "ok"))
    )
    assert unlisted_parts(record) == (
        "script_ratio",
        "length",
        "lid",
        "minhash",
        "labse",
    )


@pytest.mark.parametrize("stage_id", list(PART_IDS))
def test_a_record_that_lists_every_part_in_any_order_has_none_unlisted(
    stage_id: str,
) -> None:
    parts = tuple(PartRecord(part, "ok") for part in reversed(PART_IDS[stage_id]))
    assert unlisted_parts(StageRecord(stage_id, "ok", parts=parts)) == ()


@pytest.mark.parametrize("stage_id", list(PART_IDS))
def test_a_record_with_no_part_lists_none_of_them_whatever_its_state(
    stage_id: str,
) -> None:
    for record in (
        StageRecord(stage_id, "ok"),
        StageRecord(stage_id, "skipped", SKIP_EARLIER_FAILED),
        StageRecord(stage_id, "fail", fail_exit(1)),
    ):
        assert unlisted_parts(record) == PART_IDS[stage_id]


@pytest.mark.parametrize(
    "stage_id", [stage for stage in STAGE_IDS if stage not in PART_IDS]
)
def test_a_stage_that_has_no_parts_has_none_unlisted(stage_id: str) -> None:
    assert unlisted_parts(StageRecord(stage_id, "ok")) == ()


def test_a_part_of_another_type_is_refused() -> None:
    with pytest.raises(ValueError, match="part"):
        StageRecord("filter", "ok", parts=(("numbers", "ok", None),))  # type: ignore[arg-type]


# ---- the detail ------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "detail",
    [
        {},
        {"rows": 3, "share": 0.5, "ok": True, "none": None, "text": "a"},
        {"nested": {"a": [1, 2, {"b": [None]}]}, "tuple": (1, "x")},
    ],
)
def test_a_json_compatible_detail_is_kept(detail: dict[str, Any]) -> None:
    assert StageRecord("fetch", "ok", detail=detail).detail == detail


@pytest.mark.parametrize(
    "detail",
    [
        {"x": {1, 2}},
        {"x": b"bytes"},
        {"x": math.nan},
        {"x": math.inf},
        {"x": -math.inf},
        {"x": object()},
        {1: "x"},
        {"x": {2: "y"}},
        {"x": [{"y": {3}}]},
        {"x": complex(1, 1)},
        {"x": range(3)},
        {"x": ("a", {1: 2})},
    ],
)
def test_a_detail_that_is_not_json_compatible_is_refused(detail: Any) -> None:
    with pytest.raises(ValueError, match="detail"):
        StageRecord("fetch", "ok", detail=detail)


def test_a_detail_that_is_not_a_mapping_is_refused() -> None:
    with pytest.raises(ValueError, match="detail"):
        StageRecord("fetch", "ok", detail=[("a", 1)])  # type: ignore[arg-type]


def test_a_detail_nested_without_end_is_refused_not_followed() -> None:
    cycle: dict[str, Any] = {}
    cycle["again"] = cycle
    with pytest.raises(ValueError, match="detail"):
        StageRecord("fetch", "ok", detail=cycle)
