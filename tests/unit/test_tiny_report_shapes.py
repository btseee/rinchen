"""The closed shapes of the tiny pipeline's report (D15 Task 6, spec P10): `rinchen.tiny.report_shapes`.

A shape checks a plain JSON value and says where it is wrong, never what it holds: the place is built
from the names of the schema and from positions, and a key or a text that a document brought is counted
or left out. Every leaf is held to the same text rules (printable ASCII, no path, no time of day, no
date), so that no shape lets a path, a timestamp or a line of another program's output through.
"""

from __future__ import annotations

import math
from typing import Any

import pytest

from rinchen.tiny.report_shapes import (
    BOOL,
    COUNT,
    DIGEST,
    MAX_DEPTH,
    MAX_TEXT,
    NAME,
    NONNEG,
    NUMBER,
    SCORE,
    SHARE,
    TEXT,
    Detail,
    Keys,
    Nullable,
    Seq,
    ShapeError,
    Table,
    choice,
    equals,
    plain,
    text_problem,
)

SENTINEL = "ZZ-secret-value-ZZ"
CYRILLIC = "".join(chr(code) for code in (0x41D, 0x43E, 0x43C))


def refused(shape: Any, value: object) -> str:
    """The message of the refusal of `value` by `shape`, or the failure of the test."""
    with pytest.raises(ShapeError) as raised:
        shape.check(value, "doc")
    return str(raised.value)


@pytest.mark.parametrize("value", [0, 1, 7, 10**12])
def test_a_count_is_a_whole_number_of_zero_or_more(value: int) -> None:
    COUNT.check(value, "doc")


@pytest.mark.parametrize("value", [-1, 1.0, 0.5, True, False, "1", None, [1]])
def test_a_count_refuses_everything_else(value: object) -> None:
    assert refused(COUNT, value).startswith("doc: ")


@pytest.mark.parametrize("value", [0, -3, 2.5, 1e-06, -0.0548, 100])
def test_a_number_is_finite(value: float) -> None:
    NUMBER.check(value, "doc")


@pytest.mark.parametrize(
    "value", [math.nan, math.inf, -math.inf, True, False, "1.0", None, {}]
)
def test_a_number_refuses_what_is_not_a_finite_number(value: object) -> None:
    assert refused(NUMBER, value).startswith("doc: ")


@pytest.mark.parametrize(
    ("shape", "low", "high"),
    [(NONNEG, 0, None), (SHARE, 0, 1), (SCORE, 0, 100)],
    ids=["nonneg", "share", "score"],
)
def test_the_ranged_numbers_hold_their_bounds_inclusively(
    shape: Any, low: float, high: float | None
) -> None:
    shape.check(low, "doc")
    shape.check(float(low), "doc")
    assert refused(shape, low - 0.001)
    if high is not None:
        shape.check(high, "doc")
        assert refused(shape, high + 0.001)
    else:
        shape.check(10**9, "doc")


@pytest.mark.parametrize("value", [True, False])
def test_a_flag_is_a_boolean(value: bool) -> None:
    BOOL.check(value, "doc")


@pytest.mark.parametrize("value", [0, 1, "true", None])
def test_a_flag_refuses_a_number_or_a_text(value: object) -> None:
    assert refused(BOOL, value)


@pytest.mark.parametrize(
    "value",
    ["int8", "eng_Latn-khk_Cyrl", "t1_bt0.2_src_s1234", "tiny-int8-qe", "a", "9lives"],
)
def test_a_name_is_a_short_identifier(value: str) -> None:
    NAME.check(value, "doc")


@pytest.mark.parametrize(
    "value",
    ["", "a b", "-x", ".x", "a" * 65, CYRILLIC, "a\nb", "a/b", "a\\b", None, 3],
)
def test_a_name_refuses_what_is_not_one(value: object) -> None:
    assert refused(NAME, value)


def test_a_digest_is_64_lower_case_hex_characters() -> None:
    DIGEST.check("0123456789abcdef" * 4, "doc")
    for bad in ("A" * 64, "a" * 63, "a" * 65, "g" * 64, "", None, 5):
        assert refused(DIGEST, bad)


@pytest.mark.parametrize(
    "value",
    [
        "ok",
        "skipped: --no-qe",
        "least of 5 directions, mn_Mong-eng_Latn, int8 export",
        "x" * MAX_TEXT,
    ],
)
def test_a_text_is_printable_ascii(value: str) -> None:
    TEXT.check(value, "doc")


@pytest.mark.parametrize(
    "value",
    [
        "",
        "x" * (MAX_TEXT + 1),
        "tab\there",
        "line\nfeed",
        "bell" + chr(7),
        CYRILLIC,
        "caf" + chr(0xE9),
        None,
        4,
    ],
)
def test_a_text_refuses_what_is_not_printable_ascii_or_is_too_long(
    value: object,
) -> None:
    assert refused(TEXT, value)


@pytest.mark.parametrize(
    "value",
    [
        "/home/someone",
        "relative/path",
        "a\\b",
        "C:\\Users",
        "~/work",
        "2026-11-11",
        "x 2026-11-11T00:00:00Z",
        "12:30",
        "at 9:05 sharp",
        "T00:00:00",
    ],
)
def test_a_text_leaf_refuses_a_path_or_a_time(value: str) -> None:
    message = refused(TEXT, value)
    assert "path" in message or "time" in message
    assert text_problem(value) is not None


def test_a_name_that_looks_like_a_date_is_refused_as_one() -> None:
    assert "time" in refused(NAME, "2026-11-11")


@pytest.mark.parametrize(
    "value",
    [
        "ok",
        "exit code 4",
        "least of 5 directions",
        "1e-06",
        "50.0",
        ">= 0.95",
        "<= 1200",
        "a-b_c.d",
    ],
)
def test_the_text_rules_let_the_report_s_own_texts_through(value: str) -> None:
    assert text_problem(value) is None


def test_a_choice_names_the_members_and_never_the_value() -> None:
    shape = choice("pass", "fail")
    shape.check("pass", "doc")
    message = refused(shape, SENTINEL)
    assert "pass" in message
    assert "fail" in message
    assert SENTINEL not in message
    assert refused(shape, None)


def test_equals_accepts_only_that_value_and_does_not_quote_it() -> None:
    shape = equals("rinchen.tiny/1")
    shape.check("rinchen.tiny/1", "doc.schema")
    assert refused(shape, "rinchen.tiny/2").startswith("doc: ")
    assert "rinchen.tiny/2" not in refused(shape, "rinchen.tiny/2")
    assert refused(shape, 1)


def test_equals_takes_neither_a_boolean_for_a_number_nor_a_float_for_a_whole_number() -> (
    None
):
    equals(1).check(1, "doc")
    assert refused(equals(1), True)
    assert refused(equals(True), 1)
    assert refused(equals(1), 1.0)


def test_equals_holds_a_list_to_the_same_items_in_the_same_order() -> None:
    shape = equals(["a", "b"])
    shape.check(["a", "b"], "doc")
    assert refused(shape, ["b", "a"])
    assert refused(shape, ["a"])
    assert refused(shape, ("a", "b"))


SHAPE = Keys({"count": COUNT, "name": NAME, "nested": Keys({"flag": BOOL})})
GOOD = {"count": 3, "name": "x", "nested": {"flag": True}}


def test_keys_accept_exactly_their_keys() -> None:
    SHAPE.check(GOOD, "doc")


def test_keys_name_the_place_of_a_wrong_value() -> None:
    assert refused(SHAPE, {**GOOD, "count": -1}).startswith("doc.count: ")
    assert refused(SHAPE, {**GOOD, "nested": {"flag": 1}}).startswith(
        "doc.nested.flag: "
    )


def test_keys_name_a_missing_key_by_its_own_name() -> None:
    message = refused(SHAPE, {"count": 1, "nested": {"flag": True}})
    assert message.startswith("doc: ")
    assert "name" in message


def test_keys_count_the_unknown_keys_and_do_not_quote_them() -> None:
    message = refused(SHAPE, {**GOOD, SENTINEL: 1, "another": 2})
    assert message.startswith("doc: ")
    assert "2" in message
    assert SENTINEL not in message
    assert "another" not in message


@pytest.mark.parametrize("value", [None, [], "x", 3, [("count", 1)]])
def test_keys_refuse_what_is_not_a_mapping(value: object) -> None:
    assert refused(SHAPE, value).startswith("doc: ")


def test_keys_refuse_a_key_that_is_not_text() -> None:
    message = refused(SHAPE, {**GOOD, 1: 2})
    assert message.startswith("doc: ")
    assert "not text" in message


def test_a_table_holds_names_of_one_shape() -> None:
    shape = Table(NAME, COUNT)
    shape.check({}, "doc")
    shape.check({"ok": 3, "empty": 0}, "doc")
    assert refused(shape, {"ok": -1}).startswith("doc.ok: ")
    assert refused(shape, {"not a name": 1}).startswith("doc: ")
    assert "not a name" not in refused(shape, {"not a name": 1})
    assert refused(shape, [])


def test_a_table_may_demand_a_least_size() -> None:
    shape = Table(NAME, COUNT, minimum=1)
    shape.check({"a": 1}, "doc")
    assert refused(shape, {}).startswith("doc: ")


def test_a_table_names_the_place_of_a_key_it_cannot_name_by_a_position() -> None:
    message = refused(Table(NAME, COUNT), {"a/b": 1})
    assert "a/b" not in message


def test_a_sequence_names_the_position_of_a_wrong_item() -> None:
    shape = Seq(COUNT)
    shape.check([], "doc")
    shape.check([1, 2, 3], "doc")
    assert refused(shape, [1, -2, 3]).startswith("doc[1]: ")
    assert refused(shape, (1, 2)).startswith("doc: ")
    assert refused(shape, "12")


def test_a_sequence_holds_its_length_when_it_has_one() -> None:
    shape = Seq(COUNT, length=2)
    shape.check([1, 2], "doc")
    assert "1 entries" in refused(shape, [1])
    assert "3 entries" in refused(shape, [1, 2, 3])


def test_a_sequence_may_demand_a_least_length() -> None:
    shape = Seq(COUNT, minimum=1)
    shape.check([1], "doc")
    assert refused(shape, [])


def test_a_nullable_shape_takes_null_or_its_own_value() -> None:
    shape = Nullable(COUNT)
    shape.check(None, "doc")
    shape.check(4, "doc")
    assert refused(shape, -4).startswith("doc: ")
    assert refused(shape, "x")


JSON_VALUES = [
    None,
    True,
    3,
    -2.5,
    "ok",
    [],
    {},
    [1, [2, [3]]],
    {
        "installed": 186,
        "reasons": {"duplicate": 4},
        "languages": ["eng_Latn", "khk_Cyrl"],
    },
    {
        "token_ratio_check": "the token-ratio check needs a SentencePiece model, which does not exist yet"
    },
]


@pytest.mark.parametrize("value", JSON_VALUES)
def test_a_detail_is_free_json_made_of_safe_parts(value: object) -> None:
    Detail().check(value, "doc")


def deep(levels: int) -> object:
    value: object = 1
    for _ in range(levels):
        value = {"k": value}
    return value


def test_the_limits_are_those_of_the_spec() -> None:
    # The longest text the report holds is the detail of a threshold whose figures are all missing: the
    # `int8_chrf_drop` one names ten figures and is about 350 characters long.
    assert MAX_TEXT == 512
    assert MAX_DEPTH == 8


def test_a_detail_is_not_nested_without_limit() -> None:
    Detail().check(deep(MAX_DEPTH - 1), "doc")
    assert "deeply" in refused(Detail(), deep(MAX_DEPTH + 30))


@pytest.mark.parametrize(
    "value",
    [
        {"a": math.nan},
        [math.inf],
        {"a": (1, 2)},
        {"a": {1, 2}},
        {"a": b"x"},
        {"a/b": 1},
        {"a b": 1},
        {1: 2},
        {"a": "/etc/hosts"},
        {"a": "2026-11-11"},
        {"a": CYRILLIC},
        {"a": "x" * (MAX_TEXT + 1)},
    ],
)
def test_a_detail_refuses_what_json_cannot_hold_or_the_report_must_not(
    value: object,
) -> None:
    assert refused(Detail(), value).startswith("doc")


@pytest.mark.parametrize(
    "key",
    [
        "seconds",
        "duration",
        "elapsed",
        "elapsed_seconds",
        "wall_time",
        "timestamp",
        "date",
        "ms",
        "time_s",
    ],
)
def test_a_detail_has_no_key_that_names_a_duration_or_a_date(key: str) -> None:
    message = refused(Detail(), {key: 1})
    assert "time" in message or "duration" in message


@pytest.mark.parametrize(
    "key", ["timeout", "runtime_files", "datasets", "mismatched", "msg_count"]
)
def test_a_detail_keeps_keys_that_only_contain_those_letters(key: str) -> None:
    Detail().check({key: 1}, "doc")


def test_a_detail_names_the_place_and_never_the_text() -> None:
    message = refused(Detail(), {"a": {"b": [1, SENTINEL + "/x"]}})
    assert message.startswith("doc.a.b[1]: ")
    assert SENTINEL not in message


def test_plain_makes_json_types_without_touching_its_argument() -> None:
    original = {"a": (1, 2), "b": [{"c": (3,)}], "d": "x"}
    made = plain(original)
    assert made == {"a": [1, 2], "b": [{"c": [3]}], "d": "x"}
    assert type(made["a"]) is list
    assert original["a"] == (1, 2)
    assert plain(5) == 5
    assert plain(None) is None


def test_plain_turns_any_mapping_into_a_dict() -> None:
    from types import MappingProxyType

    made = plain({"a": MappingProxyType({"b": 1})})
    assert type(made["a"]) is dict
    assert made == {"a": {"b": 1}}
