"""`rinchen.tiny.report.parse_report`: the strict reader of `report.json` (D15 Task 6, G6.4).

A report is read back to a `Report` only when it is the document `to_bytes` writes: every key of the
closed layout and no other at any level, every state one of the three, every reason one of the closed
strings of `rinchen.tiny.states` (also the check identifiers that later tasks added), every number
finite, no text that looks like a path or a time, no repeated key, and the bytes in the canonical form
(sorted keys, two-space indent, one final line feed, nothing after it). A refusal is a `DataError`
that names the place and quotes nothing the document held.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from rinchen.errors import DataError
from rinchen.tiny.context import StageResult
from rinchen.tiny.report import Report, parse_report, to_bytes
from rinchen.tiny.report_schema import DOCUMENT
from rinchen.tiny.report_shapes import ShapeError
from rinchen.tiny.states import (
    FAIL_BUDGET,
    FAIL_INPUT,
    SKIP_EARLIER_FAILED,
    SKIP_NO_LID,
    PartRecord,
    StageRecord,
    fail_check,
    fail_environment,
    fail_exit,
    fail_timeout,
)
from tests.support.tiny_report_build import build
from tests.support.tiny_report_inputs import (
    facts_of,
    failed,
    golden_results,
    replaced,
    stopped_after,
    without_neural_part,
)
from tests.support.tiny_report_walk import (
    GOLDEN_JSON,
    closed_places,
    dumps,
    golden_document,
    place_of,
    with_key,
    with_value,
)

SENTINEL = "ZZ-secret-value-ZZ"


def golden_bytes() -> bytes:
    return GOLDEN_JSON.read_bytes()


def refusal(data: bytes) -> str:
    with pytest.raises(DataError) as raised:
        parse_report(data)
    assert SENTINEL not in str(raised.value)
    return str(raised.value)


def refusal_of(document: Any) -> str:
    return refusal(dumps(document))


def test_the_golden_report_parses_and_writes_back_to_the_same_bytes() -> None:
    parsed = parse_report(golden_bytes())
    assert isinstance(parsed, Report)
    assert to_bytes(parsed) == golden_bytes()


def reports() -> list[Report]:
    parity = dict(facts_of(golden_results(), "parity"))
    parity.update(exact=60, exact_share=60 / 70, passes=False)
    return [
        build(),
        build(stopped_after(9, failing=True)),
        build(stopped_after(0)),
        build(replaced(golden_results(), "parity", failed("parity", facts=parity))),
        build(replaced(golden_results(), "evaluate", without_neural_part())),
        build(elapsed=4000.0, qe_fake=False, profile="control"),
    ]


@pytest.mark.parametrize("index", range(6))
def test_every_report_the_builder_makes_parses_back_to_an_equal_report(
    index: int,
) -> None:
    report = reports()[index]
    data = to_bytes(report)
    parsed = parse_report(data)
    assert parsed == report
    assert to_bytes(parsed) == data
    assert parsed.verdict == report.verdict


def test_a_bad_byte_string_is_a_data_error_that_says_what_kind() -> None:
    assert "UTF-8" in refusal(b"\xff\xfe")
    assert refusal(b"").startswith("report: ")
    assert refusal(b"[]").startswith("report: ")
    assert refusal(b"null").startswith("report: ")
    assert refusal(b"{").startswith("report: ")
    assert "line" in refusal(b'{"a": }\n')


def test_a_document_nested_without_limit_is_refused_and_not_a_crash() -> None:
    assert refusal(b"[" * 100_000).startswith("report: ")


# ---- an unknown or a missing key at every level ------------------------------------------------------


def test_every_closed_mapping_refuses_a_key_that_is_not_in_the_layout() -> None:
    document = golden_document()
    places = list(closed_places(DOCUMENT, document))
    assert len(places) > 60
    for path in places:
        message = refusal_of(with_key(document, path, SENTINEL, 1))
        assert message.startswith(f"{place_of(path)}: "), path


@pytest.mark.parametrize(
    "path",
    [
        (),
        ("inputs",),
        ("stages", 0),
        ("stages", 3, "parts", 0),
        ("thresholds", 0),
        ("blocks",),
        ("blocks", "filter"),
        ("blocks", "filter", "planted"),
        ("blocks", "robustness", "coverage", "numbers_acc"),
        ("blocks", "int8", "agreement"),
    ],
)
def test_an_unknown_key_is_counted_and_never_quoted(
    path: tuple[str | int, ...],
) -> None:
    document = with_key(golden_document(), path, SENTINEL, 1)
    message = refusal_of(document)
    assert "1 keys that are none of the expected ones" in message


# ---- the states and the reasons ----------------------------------------------------------------------


def test_a_state_outside_the_three_is_refused_for_a_stage_and_for_a_part() -> None:
    for path in (("stages", 0, "state"), ("stages", 3, "parts", 0, "state")):
        message = refusal_of(with_value(golden_document(), path, "warn"))
        assert message.startswith(place_of(path))
        assert "ok, skipped, fail" in message


def test_a_skipped_stage_without_a_reason_is_refused() -> None:
    document = with_value(golden_document(), ("stages", 0, "state"), "skipped")
    message = refusal_of(document)
    assert message.startswith("report.stages[0]: ")
    assert "reason" in message


def test_a_failed_stage_without_a_reason_is_refused() -> None:
    document = with_value(golden_document(), ("stages", 1, "state"), "fail")
    assert refusal_of(document).startswith("report.stages[1]: ")


def test_a_skipped_part_without_a_reason_is_refused() -> None:
    document = with_value(golden_document(), ("stages", 3, "parts", 3, "reason"), None)
    message = refusal_of(document)
    assert message.startswith("report.stages[3].parts[3]: ")


@pytest.mark.parametrize(
    ("position", "kept"), [(3, 0), (3, 6), (14, 0), (14, 1), (14, 3)]
)
def test_an_ok_stage_that_lists_only_some_of_its_parts_is_refused(
    position: int, kept: int
) -> None:
    document = golden_document()
    parts = document["stages"][position]["parts"]
    message = refusal_of(
        with_value(document, ("stages", position, "parts"), parts[:kept])
    )
    assert message.startswith(f"report.stages[{position}]: ")
    assert "every one of its parts" in message


def test_a_reason_on_an_ok_stage_is_refused() -> None:
    document = with_value(golden_document(), ("stages", 0, "reason"), SKIP_NO_LID)
    assert refusal_of(document).startswith("report.stages[0]: ")


@pytest.mark.parametrize(
    "reason",
    ["the machine was busy", "exit code four", "an earlier stage failed.", SENTINEL],
)
def test_a_reason_that_is_not_one_of_the_closed_strings_is_refused(reason: str) -> None:
    for path in (("stages", 0, "reason"), ("stages", 3, "parts", 3, "reason")):
        message = refusal_of(with_value(golden_document(), path, reason))
        assert message.startswith(place_of(path)), path
        assert "closed" in message


def test_a_skip_reason_on_a_failure_and_a_failure_reason_on_a_skip_are_refused() -> (
    None
):
    document = with_value(golden_document(), ("stages", 0, "state"), "fail")
    document = with_value(document, ("stages", 0, "reason"), SKIP_EARLIER_FAILED)
    assert refusal_of(document).startswith("report.stages[0]: ")
    other = with_value(
        golden_document(), ("stages", 3, "parts", 3, "reason"), fail_exit(4)
    )
    assert refusal_of(other).startswith("report.stages[3].parts[3]: ")


@pytest.mark.parametrize(
    "reason",
    [
        fail_exit(4),
        fail_timeout(600),
        FAIL_BUDGET,
        FAIL_INPUT,
        fail_environment(2, 1),
        fail_check("planted_noise"),
        fail_check("stage_skipped"),
        fail_check("command_not_started"),
        fail_check("service_probe"),
        fail_check("service_translate"),
        fail_check("service_refusal"),
        fail_check("service_metrics"),
        fail_check("neural_scores"),
    ],
)
def test_every_closed_failure_reason_including_the_later_check_identifiers_is_read(
    reason: str,
) -> None:
    record = StageRecord("fetch", "fail", reason)
    report = build(replaced(golden_results(), "fetch", StageResult(record, {})))
    parsed = parse_report(to_bytes(report))
    assert parsed.stages[1].reason == reason
    assert parsed == report


@pytest.mark.parametrize(
    "reason",
    [
        "a check of the stage failed: Bad-Id",
        "a check of the stage failed: 9lives",
        "a check of the stage failed: " + "x" * 33,
        "a check of the stage failed: planted noise",
        "exit code ",
        "exit code 4x",
        "timed out after s s",
    ],
)
def test_a_failure_reason_of_the_wrong_shape_is_refused(reason: str) -> None:
    document = with_value(golden_document(), ("stages", 1, "state"), "fail")
    document = with_value(document, ("stages", 1, "reason"), reason)
    assert refusal_of(document).startswith("report.stages[1]")


# ---- the thresholds ----------------------------------------------------------------------------------


def test_a_threshold_verdict_outside_pass_and_fail_is_refused() -> None:
    message = refusal_of(
        with_value(golden_document(), ("thresholds", 0, "verdict"), "warn")
    )
    assert message.startswith("report.thresholds[0].verdict: ")
    assert "pass, fail" in message


def test_a_threshold_verdict_that_does_not_follow_from_its_value_is_refused() -> None:
    document = with_value(golden_document(), ("thresholds", 1, "verdict"), "fail")
    assert refusal_of(document).startswith("report.thresholds[1]: ")
    document = with_value(golden_document(), ("thresholds", 1, "value"), 10.0)
    assert refusal_of(document).startswith("report.thresholds[1]: ")


def test_the_thresholds_must_be_the_eight_in_order() -> None:
    document = golden_document()
    document["thresholds"][0], document["thresholds"][1] = (
        document["thresholds"][1],
        document["thresholds"][0],
    )
    assert "in order" in refusal_of(document)
    document = golden_document()
    document["stages"][0], document["stages"][1] = (
        document["stages"][1],
        document["stages"][0],
    )
    assert "in order" in refusal_of(document)


def test_the_overall_verdict_must_follow_from_the_stages_and_the_thresholds() -> None:
    document = with_value(golden_document(), ("verdict",), "fail")
    assert refusal_of(document).startswith("report.verdict: ")
    failing = build(stopped_after(9, failing=True))
    forged = json.loads(to_bytes(failing))
    forged["verdict"] = "pass"
    assert refusal_of(forged).startswith("report.verdict: ")


def test_a_failed_stage_with_every_threshold_passing_cannot_be_called_a_pass() -> None:
    document = golden_document()
    document["stages"][0].update(state="fail", reason=fail_exit(4))
    assert refusal_of(document).startswith("report.verdict: ")


# ---- numbers, texts and bytes ------------------------------------------------------------------------


@pytest.mark.parametrize(
    "literal", [b"NaN", b"Infinity", b"-Infinity", b"1e999", b"-1e999"]
)
def test_a_number_that_is_not_finite_is_refused(literal: bytes) -> None:
    data = golden_bytes().replace(b"99.35849111913797", literal)
    message = refusal(data)
    assert message.startswith("report")


def test_a_nan_in_a_block_and_in_a_detail_is_refused_too() -> None:
    data = golden_bytes().replace(b"0.0285", b"NaN")
    assert "NaN" in refusal(data)


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("blocks", "synth", "qe_status"), "skipped: /tmp/work"),
        (("blocks", "synth", "qe_status"), "C:\\Users\\someone"),
        (("blocks", "synth", "qe_status"), "relative/path"),
        (("blocks", "synth", "qe_status"), "~/work"),
        (("stages", 0, "detail", "note"), "/home/someone/x"),
        (("stages", 0, "detail", "note"), "a\\b"),
        (("thresholds", 1, "detail"), "see /etc/hosts"),
    ],
)
def test_a_string_that_looks_like_a_path_is_refused_by_place(
    path: tuple[str | int, ...], value: str
) -> None:
    message = refusal_of(with_value(golden_document(), path, value))
    assert message.startswith(place_of(path))
    assert "path" in message
    assert value not in message


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("blocks", "train", "variant"), "2026-11-11"),
        (("blocks", "synth", "qe_status"), "run of 2026-11-11T00:00:00Z"),
        (("blocks", "synth", "qe_status"), "took 12:30"),
        (("stages", 0, "detail", "note"), "at 9:05"),
        (("thresholds", 1, "detail"), "T00:00:00"),
    ],
)
def test_a_string_shaped_like_a_date_or_a_time_is_refused_by_place(
    path: tuple[str | int, ...], value: str
) -> None:
    message = refusal_of(with_value(golden_document(), path, value))
    assert message.startswith(place_of(path))
    assert "time" in message


@pytest.mark.parametrize(
    "key", ["seconds", "elapsed_seconds", "duration", "timestamp", "wall_time"]
)
def test_a_detail_key_that_names_a_duration_is_refused(key: str) -> None:
    document = with_key(golden_document(), ("stages", 0, "detail"), key, 3)
    message = refusal_of(document)
    assert message.startswith("report.stages[0].detail: ")


def test_a_text_that_is_not_ascii_is_refused() -> None:
    document = with_value(
        golden_document(), ("blocks", "synth", "qe_status"), "skipped: " + chr(0x41D)
    )
    assert "ASCII" in refusal_of(document)


def test_a_bom_a_crlf_a_tab_or_leading_space_is_refused() -> None:
    data = golden_bytes()
    assert refusal(b"\xef\xbb\xbf" + data)
    assert refusal(data.replace(b"\n", b"\r\n"))
    assert refusal(b" " + data)
    assert refusal(data.replace(b"  ", b"\t", 1))


def test_a_trailing_byte_is_refused() -> None:
    data = golden_bytes()
    assert refusal(data + b"x")
    assert "canonical" in refusal(data + b"\n")
    assert "canonical" in refusal(data[:-1])
    assert refusal(data + b"{}")
    assert refusal(data + data)


def test_a_document_that_is_not_in_the_canonical_form_is_refused_even_when_it_is_valid_json() -> (
    None
):
    document = golden_document()
    compact = (
        json.dumps(document, sort_keys=True, separators=(",", ":")).encode() + b"\n"
    )
    assert "canonical" in refusal(compact)
    four = (json.dumps(document, sort_keys=True, indent=4) + "\n").encode()
    assert "canonical" in refusal(four)
    shuffled = (
        json.dumps(dict(reversed(list(document.items()))), indent=2) + "\n"
    ).encode()
    assert "canonical" in refusal(shuffled)
    inner = golden_document()
    inner["blocks"]["filter"] = dict(reversed(list(inner["blocks"]["filter"].items())))
    assert "canonical" in refusal((json.dumps(inner, indent=2) + "\n").encode())


def test_a_repeated_key_is_refused_wherever_it_is() -> None:
    data = golden_bytes()
    doubled = data.replace(
        b'  "profile": "tiny",\n', b'  "profile": "tiny",\n  "profile": "tiny",\n', 1
    )
    assert doubled != data
    assert "repeats a key" in refusal(doubled)
    inner = data.replace(
        b'      "files": 7\n', b'      "files": 7,\n      "files": 7\n', 1
    )
    assert inner != data
    assert "repeats a key" in refusal(inner)


def test_the_wrong_schema_name_is_refused_by_place() -> None:
    document = with_value(golden_document(), ("schema",), "rinchen.tiny/2")
    message = refusal_of(document)
    assert message.startswith("report.schema: ")
    assert "rinchen.tiny/2" not in message


def test_the_fixed_list_of_unmeasured_things_is_part_of_the_schema() -> None:
    document = with_value(
        golden_document(),
        ("unmeasured",),
        ["a real corpus: the fixtures are templated sentences"],
    )
    assert refusal_of(document).startswith("report.unmeasured: ")


def test_the_parser_leaves_the_error_message_free_of_every_value_it_was_given() -> None:
    document = with_value(
        golden_document(), ("blocks", "train", "variant"), SENTINEL + "/x"
    )
    refusal_of(document)
    document = with_key(
        golden_document(), ("blocks", "filter", "reasons"), SENTINEL + " bad", 1
    )
    refusal_of(document)
    document = with_value(golden_document(), ("stages", 0, "reason"), SENTINEL)
    refusal_of(document)


def test_a_shape_error_from_a_hand_made_report_is_a_value_error_and_a_parse_error_is_a_data_error() -> (
    None
):
    assert issubclass(ShapeError, ValueError)
    assert not issubclass(DataError, ShapeError)


def test_the_filter_parts_are_kept_with_their_reasons_through_a_round_trip() -> None:
    parsed = parse_report(golden_bytes())
    filtering = parsed.stages[3]
    assert [part.id for part in filtering.parts][3] == "lid"
    assert filtering.parts[3] == PartRecord("lid", "skipped", SKIP_NO_LID)
    assert filtering.detail["token_ratio_check"].startswith("the token-ratio check")
