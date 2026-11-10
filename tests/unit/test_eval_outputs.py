"""The outputs file of D13 (spec T6): what a system wrote for each row of the directions of a suite,
one JSON object per line (`direction`, `id`, `status`, `text`), in the order of the suite's
directions and rows.

The golden files `tests/fixtures/eval/golden_outputs_alpha.jsonl` and `golden_outputs_beta.jsonl`
hold what the two golden systems wrote for the golden suite. Their expected text is built here
with the standard library's `json` from the constants of the golden corpus, never with the code
under test, and their size and digest are literals taken with `sha256sum`. Every refusal names the
line (and the direction and the id when they are well formed) and never what the file held: each
refusing test plants `MARKER` in the value it is about, so a message that echoed it would show."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest

from rinchen.errors import DataError
from rinchen.eval import outputs
from rinchen.eval.outputs import (
    MAX_OUTPUTS_BYTES,
    Output,
    OutputKey,
    outputs_from_bytes,
    outputs_to_bytes,
    write_outputs,
)
from tests.support.eval_fakes import (
    GOLDEN_OUTPUTS_FILES,
    GOLDEN_TEXTS,
    golden_outputs,
    golden_suite,
    no_network,  # noqa: F401 - the fixture of the mark
)
from tests.support.eval_golden import GOLDEN_DIRECTIONS, GOLDEN_IDS

pytestmark = pytest.mark.usefixtures("no_network")

MARKER = "SECRET-MARKER-91"
PREFIX = "not an outputs file:"
FIRST, SECOND = GOLDEN_DIRECTIONS
STATUSES = ("ok", "empty", "too_long", "truncated", "no_output")
# sha256sum of the committed files and their sizes.
GOLDEN_FILE_FACTS = {
    "alpha": (
        "0708a74060c85c5d579d55d907a36ff7bb0ce419c4ab558d1cb16314c5aca2da",
        2049,
    ),
    "beta": (
        "fcf6f29435de72e35b151e6bb22f854455c1604bc237130551cf26fcdd46ceef",
        2166,
    ),
}


def line_of(direction: str, row_id: str, text: str, /, **extra: Any) -> str:
    document: dict[str, Any] = {
        "direction": direction,
        "id": row_id,
        "status": "ok",
        "text": text,
    }
    document.update(extra)
    return json.dumps(document, ensure_ascii=False)


def golden_lines(system: str = "alpha", direction: str | None = None) -> list[str]:
    """The lines of the golden file of `system`, written with `json` here."""
    return [
        line_of(d, row_id, text)
        for d in GOLDEN_DIRECTIONS
        if direction in (None, d)
        for row_id, text in zip(GOLDEN_IDS, GOLDEN_TEXTS[system][d], strict=True)
    ]


def as_file(lines: Sequence[str]) -> bytes:
    return "".join(f"{line}\n" for line in lines).encode("utf-8")


def refused(data: bytes) -> str:
    with pytest.raises(DataError) as info:
        outputs_from_bytes(data, golden_suite())
    message = str(info.value)
    assert message.startswith(PREFIX), message
    assert MARKER not in message, "a refusal quoted what the file held"
    return message


def names(message: str, *fragments: str) -> None:
    for fragment in fragments:
        assert fragment in message, f"{fragment!r} is not in {message!r}"


# ---- the output ----------------------------------------------------------------------------


@pytest.mark.parametrize("status", STATUSES)
def test_an_output_is_a_text_and_one_of_the_five_statuses_of_the_engine(
    status: str,
) -> None:
    output = Output("Үнэ", status)  # type: ignore[arg-type]

    assert (output.text, output.status) == ("Үнэ", status)


@pytest.mark.parametrize("status", ["weird", "OK", "", "partial"])
def test_an_output_with_another_status_is_a_value_error(status: str) -> None:
    with pytest.raises(ValueError, match="status"):
        Output("x", status)  # type: ignore[arg-type]


@pytest.mark.parametrize("text", [None, 3, b"x", ["x"]])
def test_an_output_whose_text_is_not_a_string_is_a_type_error(text: object) -> None:
    with pytest.raises(TypeError, match="text"):
        Output(text, "ok")  # type: ignore[arg-type]


def test_an_output_is_frozen() -> None:
    output = Output("x", "ok")

    with pytest.raises(AttributeError):
        output.text = "y"  # type: ignore[misc]


def test_a_key_is_a_direction_and_an_id() -> None:
    key = OutputKey(FIRST, "g01")

    assert (key.direction, key.id) == (FIRST, "g01")
    assert key == OutputKey(FIRST, "g01")
    assert key != OutputKey(SECOND, "g01")


# ---- the golden files ------------------------------------------------------------------------


@pytest.mark.parametrize("system", ["alpha", "beta"])
def test_the_golden_outputs_round_trip_byte_for_byte(system: str) -> None:
    data = GOLDEN_OUTPUTS_FILES[system].read_bytes()

    read = outputs_from_bytes(data, golden_suite())

    assert read == golden_outputs(system)
    assert outputs_to_bytes(read) == data


@pytest.mark.parametrize("system", ["alpha", "beta"])
def test_the_committed_file_is_the_document_the_constants_of_the_golden_corpus_make(
    system: str,
) -> None:
    assert GOLDEN_OUTPUTS_FILES[system].read_bytes() == as_file(golden_lines(system))


@pytest.mark.parametrize("system", ["alpha", "beta"])
def test_the_committed_files_are_pinned_by_size_and_digest(system: str) -> None:
    data = GOLDEN_OUTPUTS_FILES[system].read_bytes()
    digest, size = GOLDEN_FILE_FACTS[system]

    assert len(data) == size
    assert hashlib.sha256(data).hexdigest() == digest
    assert len(data) < 3000


def test_the_golden_outputs_hold_both_directions_in_the_order_of_the_suite() -> None:
    read = outputs_from_bytes(
        GOLDEN_OUTPUTS_FILES["alpha"].read_bytes(), golden_suite()
    )

    assert [key.direction for key in read] == [FIRST] * 10 + [SECOND] * 10
    assert [key.id for key in read] == list(GOLDEN_IDS) * 2
    assert {output.status for output in read.values()} == {"ok"}


def test_the_last_sentence_of_alpha_is_empty_and_still_ok() -> None:
    read = outputs_from_bytes(
        GOLDEN_OUTPUTS_FILES["alpha"].read_bytes(), golden_suite()
    )

    assert read[OutputKey(FIRST, "g10")] == Output("", "ok")
    assert read[OutputKey(SECOND, "g10")] == Output("", "ok")


# ---- what the reader accepts ---------------------------------------------------------------


def test_the_result_is_ordered_by_the_directions_and_rows_of_the_suite_whatever_the_lines() -> (
    None
):
    lines = golden_lines("beta")

    read = outputs_from_bytes(as_file(lines[::-1]), golden_suite())

    assert list(read) == list(golden_outputs("beta"))
    assert read == golden_outputs("beta")


def test_a_file_with_one_direction_gives_only_its_ten_keys() -> None:
    data = as_file(golden_lines("alpha", FIRST))

    read = outputs_from_bytes(data, golden_suite())

    assert list(read) == [OutputKey(FIRST, row_id) for row_id in GOLDEN_IDS]
    assert read[OutputKey(FIRST, "g02")].text == GOLDEN_TEXTS["alpha"][FIRST][1]


def test_the_second_direction_alone_is_accepted_too() -> None:
    read = outputs_from_bytes(as_file(golden_lines("beta", SECOND)), golden_suite())

    assert {key.direction for key in read} == {SECOND}
    assert len(read) == 10


def test_an_empty_file_is_an_empty_set_of_outputs() -> None:
    assert outputs_from_bytes(b"", golden_suite()) == {}
    assert outputs_to_bytes({}) == b""


def test_a_missing_status_reads_ok() -> None:
    lines = golden_lines("alpha", FIRST)
    document = json.loads(lines[0])
    del document["status"]
    lines[0] = json.dumps(document, ensure_ascii=False)

    read = outputs_from_bytes(as_file(lines), golden_suite())

    assert read[OutputKey(FIRST, "g01")].status == "ok"


@pytest.mark.parametrize("status", STATUSES)
def test_every_status_of_the_engine_is_read_as_written(status: str) -> None:
    lines = golden_lines("alpha", FIRST)
    lines[3] = line_of(FIRST, "g04", "partial", status=status)

    read = outputs_from_bytes(as_file(lines), golden_suite())

    assert read[OutputKey(FIRST, "g04")] == Output("partial", status)  # type: ignore[arg-type]


def test_the_last_line_needs_no_final_line_feed() -> None:
    data = as_file(golden_lines("alpha", FIRST)).rstrip(b"\n")

    assert len(outputs_from_bytes(data, golden_suite())) == 10


def test_a_line_separator_inside_a_text_does_not_split_the_line() -> None:
    separator = chr(0x2028)
    lines = golden_lines("alpha", FIRST)
    lines[0] = line_of(FIRST, "g01", f"a{separator}b")

    read = outputs_from_bytes(as_file(lines), golden_suite())

    assert read[OutputKey(FIRST, "g01")].text == f"a{separator}b"


def test_the_order_of_the_keys_inside_a_line_does_not_matter() -> None:
    lines = golden_lines("alpha", FIRST)
    lines[0] = (
        '{"text": "the cat sat on the mat", "id": "g01", "direction": "' + FIRST + '"}'
    )

    read = outputs_from_bytes(as_file(lines), golden_suite())

    assert read[OutputKey(FIRST, "g01")] == Output("the cat sat on the mat", "ok")


# ---- every refusal -------------------------------------------------------------------------


def test_a_direction_with_nine_rows_is_refused_naming_the_direction_and_one_missing() -> (
    None
):
    lines = golden_lines("alpha", FIRST)
    del lines[9]

    message = refused(as_file(lines))

    names(message, f"direction '{FIRST}'", "lacks 1 of 10 rows", "id 'g10'")


def test_a_direction_missing_most_rows_counts_them() -> None:
    message = refused(as_file(golden_lines("alpha", SECOND)[:4]))

    names(message, f"direction '{SECOND}'", "lacks 6 of 10 rows", "id 'g05'")


def test_the_incomplete_direction_is_found_in_the_order_of_the_suite() -> None:
    lines = golden_lines("alpha")
    del lines[0]  # the first direction lacks g01
    del lines[-1]  # the second lacks g10

    message = refused(as_file(lines))

    names(message, f"direction '{FIRST}'", "id 'g01'")


def test_a_complete_direction_beside_an_incomplete_one_is_still_refused() -> None:
    lines = golden_lines("alpha")
    del lines[-1]

    message = refused(as_file(lines))

    names(message, f"direction '{SECOND}'", "lacks 1 of 10 rows")


def test_an_extra_id_is_refused_naming_the_line_the_direction_and_the_id() -> None:
    lines = golden_lines("alpha", FIRST)
    lines.append(line_of(FIRST, "g11", MARKER))

    message = refused(as_file(lines))

    names(message, "line 11", f"direction '{FIRST}'", "id 'g11'", "is not a row")


def test_an_id_that_is_not_well_formed_is_not_named() -> None:
    lines = golden_lines("alpha", FIRST)
    lines[2] = line_of(FIRST, f"{MARKER}!", "x")

    message = refused(as_file(lines))

    names(message, "line 3", f"direction '{FIRST}'", "is not a row")
    assert "id '" not in message


def test_a_well_formed_direction_the_suite_lacks_is_named() -> None:
    lines = golden_lines("alpha", FIRST)
    lines[0] = line_of("eng_Latn-rus_Cyrl", "g01", MARKER)

    message = refused(as_file(lines))

    names(message, "line 1", "direction 'eng_Latn-rus_Cyrl'", "not a direction")


def test_a_direction_that_is_not_well_formed_is_not_named() -> None:
    lines = golden_lines("alpha", FIRST)
    lines[0] = line_of(f"{MARKER}-x", "g01", "x")

    message = refused(as_file(lines))

    names(message, "line 1", "not a direction")
    assert "direction '" not in message


def test_khk_mong_is_no_direction_of_the_suite_whatever_the_file_says() -> None:
    lines = golden_lines("alpha", FIRST)
    lines[0] = line_of("khk_Mong-eng_Latn", "g01", "x")

    message = refused(as_file(lines))

    names(message, "line 1", "not a direction of the suite")


def test_a_repeated_request_is_refused_naming_the_second_line() -> None:
    lines = golden_lines("alpha", FIRST)
    lines.append(lines[4])

    message = refused(as_file(lines))

    names(message, "line 11", f"direction '{FIRST}'", "id 'g05'", "more than once")


def test_the_same_id_in_the_other_direction_is_not_a_repeat() -> None:
    assert len(outputs_from_bytes(as_file(golden_lines("beta")), golden_suite())) == 20


@pytest.mark.parametrize("status", ["weird", "", "OK", " ok"])
def test_a_status_outside_the_five_is_refused(status: str) -> None:
    lines = golden_lines("alpha", FIRST)
    lines[1] = line_of(FIRST, "g02", "x", status=status)

    message = refused(as_file(lines))

    names(message, "line 2", f"direction '{FIRST}'", "id 'g02'", "'status'")
    assert "weird" not in message


@pytest.mark.parametrize("status", [3, None, ["ok"], {"a": 1}, True])
def test_a_status_that_is_not_a_string_is_refused(status: object) -> None:
    lines = golden_lines("alpha", FIRST)
    lines[1] = line_of(FIRST, "g02", "x", status=status)

    names(refused(as_file(lines)), "line 2", "'status'")


@pytest.mark.parametrize("text", [3, 1.5, None, True, ["x"], {"a": MARKER}])
def test_a_text_that_is_not_a_string_is_refused(text: object) -> None:
    lines = golden_lines("alpha", FIRST)
    lines[0] = line_of(FIRST, "g01", text)  # type: ignore[arg-type]

    message = refused(as_file(lines))

    names(message, "line 1", f"direction '{FIRST}'", "id 'g01'", "'text'", "string")


@pytest.mark.parametrize("field", ["direction", "id"])
@pytest.mark.parametrize("value", [3, None, ["g01"]])
def test_a_direction_or_an_id_that_is_not_a_string_is_refused(
    field: str, value: object
) -> None:
    lines = golden_lines("alpha", FIRST)
    lines[0] = line_of(FIRST, "g01", "x", **{field: value})

    names(refused(as_file(lines)), "line 1", f"'{field}'", "string")


@pytest.mark.parametrize("field", ["direction", "id", "text"])
def test_a_missing_field_is_refused_by_name(field: str) -> None:
    lines = golden_lines("alpha", FIRST)
    document = json.loads(lines[0])
    del document[field]
    lines[0] = json.dumps(document, ensure_ascii=False)

    names(refused(as_file(lines)), "line 1", f"'{field}'", "missing")


def test_a_field_outside_the_four_is_refused_and_never_quoted() -> None:
    lines = golden_lines("alpha", FIRST)
    lines[0] = line_of(FIRST, "g01", "x", **{MARKER: "y"})

    names(
        refused(as_file(lines)),
        "line 1",
        f"direction '{FIRST}'",
        "not part of the format",
    )


def test_a_repeated_key_is_refused() -> None:
    lines = golden_lines("alpha", FIRST)
    lines[0] = lines[0].replace('"id"', '"id": "g99", "id"', 1)

    names(refused(as_file(lines)), "line 1", "more than once")


def test_a_lone_surrogate_in_a_text_is_refused_without_quoting() -> None:
    lines = golden_lines("alpha", FIRST)
    lines[0] = (
        '{"direction": "'
        + FIRST
        + '", "id": "g01", "status": "ok", "text": "'
        + MARKER
        + '\\ud800"}'
    )

    message = refused(as_file(lines))

    names(message, "line 1", "id 'g01'", "'text'", "lone surrogate")


@pytest.mark.parametrize("line", ["", "   ", "not json", "{", "[1, 2]", '"text"', "3"])
def test_a_line_that_is_not_a_json_object_is_refused(line: str) -> None:
    lines = golden_lines("alpha", FIRST)
    lines.insert(2, line)

    message = refused(as_file(lines))

    names(message, "line 3")
    assert "{" not in message


def test_a_blank_line_in_the_middle_is_not_skipped() -> None:
    lines = golden_lines("alpha", FIRST)
    lines.insert(5, "")

    names(refused(as_file(lines)), "line 6", "not valid JSON")


def test_a_file_that_is_not_utf8_is_refused() -> None:
    data = as_file(golden_lines("alpha", FIRST))[:-1] + b"\xff\n"

    names(refused(data), "not UTF-8")


def test_a_json_object_nested_too_deep_is_refused_not_crashed() -> None:
    lines = golden_lines("alpha", FIRST)
    lines[0] = "[" * 100_000

    names(refused(as_file(lines)), "line 1")


def test_a_file_over_the_cap_is_refused_before_it_is_parsed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(outputs, "MAX_OUTPUTS_BYTES", 64)
    data = b"\xff" * 65  # also not UTF-8: the cap comes first

    message = refused(data)

    names(message, "larger than")
    assert "UTF-8" not in message


def test_a_file_at_the_cap_is_read(monkeypatch: pytest.MonkeyPatch) -> None:
    data = as_file(golden_lines("alpha", FIRST))
    monkeypatch.setattr(outputs, "MAX_OUTPUTS_BYTES", len(data))

    assert len(outputs_from_bytes(data, golden_suite())) == 10


def test_the_cap_is_128_mebibytes() -> None:
    assert MAX_OUTPUTS_BYTES == 128 * 2**20


@pytest.mark.parametrize(
    "document",
    [
        # every field holds the marker in a shape that is no direction and no id
        {
            "direction": f"{MARKER}!",
            "id": f"{MARKER}!",
            "status": MARKER,
            "text": MARKER,
        },
        # the marker as a key, as the value of an unknown field, as a status
        {"direction": FIRST, "id": "g01", "status": "ok", "text": "x", MARKER: MARKER},
        {"direction": FIRST, "id": "g01", "status": MARKER, "text": MARKER},
        # the marker as a text of the wrong type, and with a lone surrogate
        {"direction": FIRST, "id": "g01", "status": "ok", "text": [MARKER]},
        {"direction": f"{MARKER}!", "id": "g01", "status": "ok", "text": MARKER},
        {"direction": FIRST, "id": f"{MARKER}!", "status": "ok", "text": MARKER},
    ],
)
def test_no_refusal_quotes_a_field_of_the_line_it_refuses(
    document: dict[str, Any],
) -> None:
    lines = golden_lines("alpha", FIRST)
    lines[0] = json.dumps(document, ensure_ascii=False)

    message = refused(as_file(lines))

    names(message, "line 1")


def test_a_marker_in_every_text_of_a_refused_file_is_in_no_message() -> None:
    lines = [line_of(FIRST, row_id, MARKER) for row_id in GOLDEN_IDS]
    lines.append(line_of(FIRST, "g03", MARKER))

    message = refused(as_file(lines))

    names(message, "line 11", "more than once")


# ---- the writer ----------------------------------------------------------------------------


def test_the_writer_makes_sorted_keys_unescaped_text_and_one_lf_per_line() -> None:
    data = outputs_to_bytes({OutputKey(FIRST, "g01"): Output('Муур"сууж', "truncated")})

    assert (
        data
        == (
            '{"direction": "khk_Cyrl-eng_Latn", "id": "g01", "status": "truncated", '
            '"text": "Муур\\"сууж"}\n'
        ).encode()
    )


def test_the_writer_keeps_the_order_of_the_mapping() -> None:
    reverse = dict(reversed(list(golden_outputs("alpha").items())))

    data = outputs_to_bytes(reverse)

    assert data.decode().split("\n")[0].count("g10") == 1
    assert data.decode().split("\n")[0].startswith(f'{{"direction": "{SECOND}"')


def test_a_text_that_holds_a_line_feed_stays_on_its_line() -> None:
    data = outputs_to_bytes({OutputKey(FIRST, "g01"): Output("a\nb\r", "ok")})

    assert data.count(b"\n") == 1
    assert b"a\\nb\\r" in data


def test_the_writer_refuses_a_lone_surrogate_naming_the_row_and_not_the_text() -> None:
    data = {OutputKey(FIRST, "g01"): Output(f"{MARKER}{chr(0xD800)}", "ok")}

    with pytest.raises(DataError) as info:
        outputs_to_bytes(data)

    message = str(info.value)
    assert message.startswith(PREFIX)
    names(message, f"direction '{FIRST}'", "id 'g01'", "lone surrogate")
    assert MARKER not in message


def test_the_writer_names_no_id_that_is_not_well_formed() -> None:
    data = {OutputKey(FIRST, f"{MARKER}!"): Output(chr(0xD800), "ok")}

    with pytest.raises(DataError) as info:
        outputs_to_bytes(data)

    assert MARKER not in str(info.value)


def test_write_outputs_returns_the_digest_of_the_bytes_it_wrote(tmp_path: Path) -> None:
    path = tmp_path / "hypotheses.jsonl"

    digest = write_outputs(golden_outputs("alpha"), path)

    data = GOLDEN_OUTPUTS_FILES["alpha"].read_bytes()
    assert path.read_bytes() == data
    assert digest == hashlib.sha256(data).hexdigest()
    assert sorted(item.name for item in tmp_path.iterdir()) == ["hypotheses.jsonl"]


def test_write_outputs_replaces_a_file_whole(tmp_path: Path) -> None:
    path = tmp_path / "hypotheses.jsonl"
    path.write_bytes(b"old\n")

    write_outputs(golden_outputs("beta"), path)

    assert path.read_bytes() == GOLDEN_OUTPUTS_FILES["beta"].read_bytes()


def test_a_refused_write_leaves_the_old_file_and_no_part(tmp_path: Path) -> None:
    path = tmp_path / "hypotheses.jsonl"
    path.write_bytes(b"old\n")
    bad = {OutputKey(FIRST, "g01"): Output(chr(0xD800), "ok")}

    with pytest.raises(DataError):
        write_outputs(bad, path)

    assert path.read_bytes() == b"old\n"
    assert sorted(item.name for item in tmp_path.iterdir()) == ["hypotheses.jsonl"]


def test_a_write_that_fails_at_the_rename_leaves_the_old_file_whole_and_no_part(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "hypotheses.jsonl"
    path.write_bytes(b"old\n")

    def failing_replace(self: Path, target: object) -> Path:
        raise OSError("the disk is gone")

    monkeypatch.setattr(Path, "replace", failing_replace)

    with pytest.raises(OSError, match="disk"):
        write_outputs(golden_outputs("alpha"), path)

    assert path.read_bytes() == b"old\n"
    assert sorted(item.name for item in tmp_path.iterdir()) == ["hypotheses.jsonl"]


def test_write_outputs_refuses_what_the_reader_would_not_accept_by_size(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "hypotheses.jsonl"
    monkeypatch.setattr(outputs, "MAX_OUTPUTS_BYTES", 100)

    with pytest.raises(DataError, match="larger than"):
        write_outputs(golden_outputs("alpha"), path)

    assert not path.exists()
