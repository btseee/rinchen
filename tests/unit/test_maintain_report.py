"""The report of `rinchen maintain check` (D14, spec M2, M3 and M11): the aggregation of states, the
exit code, the document and its bytes.

The aggregation is tested exhaustively against a rule written another way: `group_state` over every
tuple of up to four states (121), `overall_state` over every combination of the six steps' states
in which only the drift step may be `drift` (five steps of three states and one of four: 3 ** 5 * 4
= 972; a `Step` refuses `drift` anywhere else, so the other 3,124 of the 4 ** 6 combinations
cannot be built). The document is the schema, the cadence, the
state, the digests, the lock's figures and the six steps in order; its bytes are sorted, indented
by two spaces, not ASCII-escaped and end in one line feed.
"""

from __future__ import annotations

import dataclasses
import itertools
import json
from types import MappingProxyType

import pytest

from rinchen.maintain.report import (
    SCHEMA,
    STEP_IDS,
    Report,
    Step,
    exit_code_of,
    group_state,
    overall_state,
    report_document,
    report_to_bytes,
)
from rinchen.maintain.states import STATES
from tests.support.maintain_reports import report_of
from tests.support.no_network import no_network  # noqa: F401 - the fixture of the mark

pytestmark = pytest.mark.usefixtures("no_network")

NON_DRIFT = ("ok", "fail", "skipped")
RANK = {"skipped": 0, "ok": 1, "fail": 2}


# ---- the aggregation, exhaustively ----------------------------------------------------------------


def test_the_step_ids_are_the_six_steps_of_the_check_in_order() -> None:
    assert STEP_IDS == (
        "doctor",
        "lock_file",
        "lock_consistent",
        "environment",
        "drift",
        "reruns",
    )
    assert SCHEMA == "rinchen.maintain/1"


def test_group_state_over_every_tuple_of_up_to_four_states_is_the_worst_rank() -> None:
    checked = 0
    for length in range(5):
        for states in itertools.product(NON_DRIFT, repeat=length):
            worst = max((RANK[state] for state in states), default=0)
            expected = {2: "fail", 1: "ok", 0: "skipped"}[worst]

            assert group_state(states) == expected, states  # type: ignore[arg-type]
            checked += 1
    assert checked == 121


def test_a_group_has_no_drift_state() -> None:
    with pytest.raises(ValueError, match="drift"):
        group_state(["ok", "drift"])


def test_overall_state_over_every_combination_of_step_states() -> None:
    checked = 0
    for others in itertools.product(NON_DRIFT, repeat=5):
        for drift in STATES:
            by_id = dict(
                zip([i for i in STEP_IDS if i != "drift"], others, strict=True)
            )
            by_id["drift"] = drift
            if "fail" in by_id.values():
                expected = "fail"
            elif drift == "drift":
                expected = "drift"
            else:
                expected = "ok"

            report = report_of(by_id)

            assert overall_state(report) == expected, by_id
            assert report_document(report)["state"] == expected
            checked += 1
    assert checked == 972


@pytest.mark.parametrize("step_id", [i for i in STEP_IDS if i != "drift"])
def test_only_the_drift_step_may_be_in_the_drift_state(step_id: str) -> None:
    with pytest.raises(ValueError, match="drift"):
        Step(step_id, "drift", None, {})


def test_a_step_refuses_a_state_that_is_none_of_the_four() -> None:
    with pytest.raises(ValueError, match="state"):
        Step("doctor", "passed", None, {})  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "ids",
    [
        STEP_IDS[:-1],
        STEP_IDS[1:],
        (*STEP_IDS[1:], STEP_IDS[0]),
        (*STEP_IDS, "doctor"),
        (),
    ],
)
def test_a_report_must_list_every_step_once_in_order(ids: tuple[str, ...]) -> None:
    steps = tuple(Step(step_id, "ok", None, {}) for step_id in ids)

    with pytest.raises(ValueError, match="steps"):
        Report("nightly", {}, {}, steps)


# ---- the exit code ---------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("states", "fail_on_drift", "expected"),
    [
        ({}, False, 0),
        ({}, True, 0),
        ({"drift": "drift"}, False, 0),
        ({"drift": "drift"}, True, 4),
        ({"doctor": "fail"}, False, 4),
        ({"doctor": "fail"}, True, 4),
        ({"drift": "fail"}, False, 4),
        ({"reruns": "fail", "drift": "drift"}, False, 4),
        ({"drift": "skipped", "reruns": "skipped"}, True, 0),
    ],
)
def test_the_exit_code_is_four_for_a_failure_or_for_drift_when_asked(
    states: dict[str, str], fail_on_drift: bool, expected: int
) -> None:
    assert exit_code_of(report_of(states), fail_on_drift=fail_on_drift) == expected


# ---- the document and its bytes -------------------------------------------------------------------


def test_the_document_lists_the_schema_the_inputs_the_lock_and_the_steps_in_order() -> (
    None
):
    report = report_of(
        {"drift": "drift"},
        doctor={"checks": [{"id": "D-PY", "state": "ok", "reason": None}]},
    )

    document = report_document(report)

    assert set(document) == {"schema", "cadence", "state", "inputs", "lock", "steps"}
    assert document["schema"] == "rinchen.maintain/1"
    assert document["cadence"] == "nightly"
    assert document["state"] == "drift"
    assert document["inputs"] == {
        "config": "a" * 64,
        "lock": "b" * 64,
        "pyproject": "c" * 64,
    }
    assert document["lock"] == {
        "packages": 6,
        "names": 5,
        "format": {"version": 1, "revision": 5},
    }
    steps = document["steps"]
    assert isinstance(steps, list)
    assert [step["id"] for step in steps] == list(STEP_IDS)
    assert all(set(step) == {"id", "state", "reason", "detail"} for step in steps)
    assert steps[0]["detail"] == {
        "checks": [{"id": "D-PY", "state": "ok", "reason": None}]
    }
    assert steps[4]["state"] == "drift"


def test_a_step_with_a_reason_and_one_without_are_written_as_given() -> None:
    steps = tuple(
        Step(
            step_id,
            "skipped" if step_id == "drift" else "ok",
            "disabled by --no-drift" if step_id == "drift" else None,
            {},
        )
        for step_id in STEP_IDS
    )
    report = Report("nightly", {}, {}, steps)

    document = report_document(report)

    assert [step["reason"] for step in document["steps"]] == [  # type: ignore[attr-defined]
        None,
        None,
        None,
        None,
        "disabled by --no-drift",
        None,
    ]


def test_the_document_turns_mappings_and_tuples_into_plain_json_values() -> None:
    detail = MappingProxyType({"rows": ({"a": MappingProxyType({"b": (1, 2)})},)})

    document = report_document(report_of(reruns=detail))

    step = document["steps"][5]  # type: ignore[index]
    assert step["detail"] == {"rows": [{"a": {"b": [1, 2]}}]}
    assert type(step["detail"]) is dict
    assert type(step["detail"]["rows"][0]["a"]) is dict
    json.dumps(document)


def test_the_bytes_are_sorted_indented_by_two_and_end_in_one_line_feed() -> None:
    report = report_of({"drift": "drift"}, reruns={"z": 1, "a": {"y": 2, "b": 3}})

    data = report_to_bytes(report)

    text = data.decode("utf-8")
    assert text.endswith("}\n") and not text.endswith("\n\n")
    assert "\r" not in text
    assert text.startswith('{\n  "cadence": "nightly",\n  "inputs": {\n    "config": ')
    assert (
        text
        == json.dumps(
            report_document(report), sort_keys=True, indent=2, ensure_ascii=False
        )
        + "\n"
    )
    assert json.loads(data) == report_document(report)
    assert text.index('"a": {') < text.index('"z": 1')
    assert text.index('"b": 3') < text.index('"y": 2')


def test_the_bytes_are_not_ascii_escaped() -> None:
    detail = {"name": chr(0x043C) + chr(0x043E) + chr(0x043D)}

    text = report_to_bytes(report_of(reruns=detail)).decode("utf-8")

    assert chr(0x043C) + chr(0x043E) + chr(0x043D) in text
    assert "\\u" not in text


def test_the_bytes_refuse_a_value_that_is_not_a_number_of_json() -> None:
    with pytest.raises(ValueError, match="Out of range"):
        report_to_bytes(report_of(reruns={"x": float("nan")}))


def test_equal_observations_give_equal_bytes() -> None:
    assert report_to_bytes(report_of({"doctor": "fail"})) == report_to_bytes(
        report_of({"doctor": "fail"})
    )
    assert report_to_bytes(report_of({"doctor": "fail"})) != report_to_bytes(
        report_of()
    )


def test_the_report_and_its_steps_are_frozen() -> None:
    report = report_of()

    with pytest.raises(dataclasses.FrozenInstanceError):
        report.cadence = "weekly"  # type: ignore[misc]
    with pytest.raises(dataclasses.FrozenInstanceError):
        report.steps[0].state = "fail"  # type: ignore[misc]
    assert not hasattr(report, "__dict__")
    assert not hasattr(report.steps[0], "__dict__")
