"""The reading, the writing and the summary of `rinchen translate` (D11, Task 5): what the
command does with a stream of bytes, apart from the engine and the options.

Input is UTF-8 read strictly and split at LF only; output is LF only, flushed whole chunks at a
time; the summary is the text the command writes to stderr. The literals are the contract.
"""

from __future__ import annotations

import io
from typing import Any

import pytest

from rinchen.cli import translate_io
from rinchen.cli.translate_io import (
    Output,
    Summary,
    document_line,
    document_status,
    read_blocks,
    read_lines,
    translation_line,
)
from rinchen.errors import DataError
from rinchen.infer.engine_types import (
    ENGINE_COUNTERS,
    DocumentResult,
    Status,
    Translation,
)
from tests.support.translate_cli import NO_STATS

LS = chr(0x2028)
PS = chr(0x2029)
NEL = chr(0x85)
BAD_UTF8 = b"\xff"
# The three bytes of a lone surrogate in a UTF-8 look-alike: Python's strict decoder refuses them.
SURROGATE_BYTES = b"\xed\xa0\x80"


def lines_of(content: bytes) -> list[str]:
    return list(read_lines(io.BytesIO(content)))


def blocks_of(content: bytes) -> list[list[str]]:
    return list(read_blocks(io.BytesIO(content)))


def translation(
    text: str,
    status: Status = "ok",
    *,
    alternatives: tuple[str, ...] = (),
    purity: float | None = 1.0,
) -> Translation:
    return Translation(
        text=text,
        status=status,
        alternatives=alternatives,
        src="eng_Latn",
        tgt="khk_Cyrl",
        script_purity=purity,
        retried=False,
        zero_shot=False,
    )


# --- read_lines: the lines of the input -----------------------------------------------------


def test_lines_are_split_at_lf_and_the_last_one_needs_no_lf() -> None:
    assert lines_of(b"a\nb\nc") == ["a", "b", "c"]


def test_a_final_lf_does_not_make_a_last_empty_line() -> None:
    assert lines_of(b"a\nb\n") == ["a", "b"]


def test_an_empty_line_at_the_end_is_a_line() -> None:
    assert lines_of(b"a\n\n") == ["a", ""]


def test_an_empty_input_has_no_lines() -> None:
    assert lines_of(b"") == []


def test_a_lone_lf_is_one_empty_line() -> None:
    assert lines_of(b"\n") == [""]


def test_one_trailing_cr_is_dropped_and_no_other() -> None:
    assert lines_of(b"a\r\nb\r\r\nc\rd\r\ne\r") == ["a", "b\r", "c\rd", "e"]


@pytest.mark.parametrize(
    "separator",
    [LS, PS, NEL, chr(0x0B), chr(0x0C), chr(0x1C), chr(0x1D), chr(0x1E)],
    ids=["LS", "PS", "NEL", "VT", "FF", "FS", "GS", "RS"],
)
def test_only_lf_splits_a_line(separator: str) -> None:
    content = f"one{separator}two\nthree\n".encode()

    assert lines_of(content) == [f"one{separator}two", "three"]


def test_the_text_is_the_decoded_utf_8_of_the_line() -> None:
    assert lines_of("Сайн байна уу\n中文\n".encode()) == ["Сайн байна уу", "中文"]


def test_invalid_utf_8_is_a_data_error_that_names_the_line_and_the_offset() -> None:
    reading = read_lines(io.BytesIO(b"fine\nab" + BAD_UTF8 + b"c\n"))

    assert next(reading) == "fine"
    with pytest.raises(DataError) as caught:
        next(reading)
    message = str(caught.value)
    assert (
        message
        == "input line 2 is not valid UTF-8: invalid byte at offset 2 of the line"
    )


def test_the_message_of_a_decoding_error_quotes_no_text() -> None:
    with pytest.raises(DataError) as caught:
        lines_of(b"secret" + BAD_UTF8)

    assert "secret" not in str(caught.value)


def test_the_bytes_of_a_lone_surrogate_are_refused() -> None:
    with pytest.raises(DataError, match="input line 1 is not valid UTF-8"):
        lines_of(SURROGATE_BYTES)


def test_a_truncated_sequence_at_the_end_is_refused() -> None:
    with pytest.raises(DataError, match="input line 2 is not valid UTF-8"):
        lines_of(b"a\n" + "Сайн".encode()[:-1])


def test_lines_are_read_one_at_a_time_as_they_are_asked_for() -> None:
    class Counting(io.BytesIO):
        reads = 0

        def readline(self, size: int | None = -1, /) -> bytes:
            Counting.reads += 1
            return super().readline(size)

    stream = Counting(b"a\nb\nc\n")
    reading = read_lines(stream)

    assert next(reading) == "a"
    assert Counting.reads == 1
    assert next(reading) == "b"
    assert Counting.reads == 2


def test_a_stream_that_fails_to_read_is_a_data_error_naming_the_line() -> None:
    class Failing(io.BytesIO):
        def readline(self, size: int | None = -1, /) -> bytes:
            raise OSError("disk gone")

    with pytest.raises(DataError, match="cannot read the input at line 1: disk gone"):
        list(read_lines(Failing(b"a\n")))


# --- read_blocks: the documents of the input ------------------------------------------------


def test_blocks_are_separated_by_blank_lines() -> None:
    assert blocks_of(b"a\nb\n\nc\n") == [["a", "b"], ["c"]]


def test_runs_of_blank_lines_and_the_edges_make_no_empty_block() -> None:
    assert blocks_of(b"\n\na\n\n\n\nb\n\n\n") == [["a"], ["b"]]


def test_a_line_of_spaces_is_a_blank_line() -> None:
    assert blocks_of(b"a\n   \nb\n") == [["a"], ["b"]]


def test_a_last_block_needs_no_lf() -> None:
    assert blocks_of(b"a\n\nb") == [["a"], ["b"]]


def test_an_input_of_blank_lines_has_no_block() -> None:
    assert blocks_of(b"\n \n\n") == []
    assert blocks_of(b"") == []


def test_blocks_keep_the_rules_of_lines() -> None:
    assert blocks_of(b"a\r\nb\r\n\r\nc\r\n") == [["a", "b"], ["c"]]
    with pytest.raises(DataError, match="input line 3 is not valid UTF-8"):
        blocks_of(b"a\n\n" + BAD_UTF8)


# --- the output lines -----------------------------------------------------------------------


def test_a_text_line_is_the_translation() -> None:
    assert translation_line(translation("Сайн байна уу"), "text") == "Сайн байна уу"


def test_a_text_line_of_a_translation_without_text_is_empty() -> None:
    assert translation_line(translation("", "empty", purity=None), "text") == ""


@pytest.mark.parametrize(
    "separator",
    ["\n", "\r", chr(0x0B), chr(0x0C), chr(0x1C), chr(0x1D), chr(0x1E), NEL, LS, PS],
)
def test_a_text_line_never_holds_a_line_break(separator: str) -> None:
    line = translation_line(
        translation(f"a{separator}b{separator}{separator}c"), "text"
    )

    assert line == "a b c"


def test_a_jsonl_line_is_one_object_with_a_fixed_order_of_keys() -> None:
    answer = translation("Сайн", alternatives=("Сайхан", "Мэнд"), purity=0.5)

    assert translation_line(answer, "jsonl") == (
        '{"text":"Сайн","status":"ok","alternatives":["Сайхан","Мэнд"],'
        '"src":"eng_Latn","tgt":"khk_Cyrl","script_purity":0.5}'
    )


def test_a_jsonl_line_of_a_problem_says_its_status_and_a_null_purity() -> None:
    line = translation_line(translation("", "too_long", purity=None), "jsonl")

    assert line == (
        '{"text":"","status":"too_long","alternatives":[],'
        '"src":"eng_Latn","tgt":"khk_Cyrl","script_purity":null}'
    )


def test_a_jsonl_line_escapes_quotes_and_line_breaks() -> None:
    line = translation_line(translation('say "hi"\\\n'), "jsonl")

    assert line.startswith('{"text":"say \\"hi\\"\\\\\\n","status":"ok"')
    assert "\n" not in line


def test_a_jsonl_line_holds_no_unescaped_line_break_character() -> None:
    answer = translation(f"a{LS}b{PS}c{NEL}d", alternatives=(f"e{LS}f",))

    line = translation_line(answer, "jsonl")

    assert not any(separator in line for separator in (LS, PS, NEL, "\n", "\r"))
    assert line.startswith('{"text":"a\\u2028b\\u2029c\\u0085d"')


# --- the output lines of a document ---------------------------------------------------------


def document(*translations: Translation, text: str = "joined") -> DocumentResult:
    return DocumentResult(text=text, translations=translations, stats=NO_STATS)


def test_the_text_line_of_a_document_is_its_text() -> None:
    # A target written without spaces: the text is the sentences joined by nothing.
    result = document(translation("你"), translation("好"), text="你好")

    assert (
        document_line(result, src="eng_Latn", tgt="zho_Hans", output_format="text")
        == "你好"
    )


def test_a_document_line_never_holds_a_line_break() -> None:
    result = document(translation("a"), text=f"a{LS}b")

    assert document_line(
        result, src="eng_Latn", tgt="khk_Cyrl", output_format="text"
    ) == ("a b")


def test_the_jsonl_line_of_a_document_has_the_keys_of_a_sentence() -> None:
    result = document(
        translation("Нэг", purity=1.0), translation("Two", purity=0.25), text="Нэг Two"
    )

    line = document_line(result, src="eng_Latn", tgt="khk_Cyrl", output_format="jsonl")

    assert line == (
        '{"text":"Нэг Two","status":"ok","alternatives":[],'
        '"src":"eng_Latn","tgt":"khk_Cyrl","script_purity":0.25}'
    )


def test_the_purity_of_a_document_is_that_of_its_least_pure_sentence() -> None:
    result = document(
        translation("a", purity=None),
        translation("b", purity=0.75),
        translation("c", purity=0.5),
    )

    line = document_line(result, src="eng_Latn", tgt="khk_Cyrl", output_format="jsonl")

    assert line.endswith('"script_purity":0.5}')


def test_a_document_without_a_purity_has_a_null_one() -> None:
    result = document(translation("", "empty", purity=None), text="")

    line = document_line(result, src="eng_Latn", tgt="khk_Cyrl", output_format="jsonl")

    assert line == (
        '{"text":"","status":"empty","alternatives":[],'
        '"src":"eng_Latn","tgt":"khk_Cyrl","script_purity":null}'
    )


@pytest.mark.parametrize(
    ("statuses", "expected"),
    [
        ((), "empty"),
        (("empty",), "empty"),
        (("empty", "empty"), "empty"),
        (("ok",), "ok"),
        (("ok", "ok"), "ok"),
        (("ok", "empty"), "ok"),
        (("ok", "truncated"), "truncated"),
        (("truncated", "empty"), "truncated"),
        (("ok", "too_long", "truncated"), "too_long"),
        (("truncated", "no_output", "too_long"), "no_output"),
        (("no_output", "ok"), "no_output"),
    ],
)
def test_the_status_of_a_document_is_the_worst_of_its_sentences(
    statuses: tuple[Status, ...], expected: Status
) -> None:
    result = document(*(translation("x", status) for status in statuses))

    assert document_status(result) == expected


# --- the output stream ----------------------------------------------------------------------


class RecordingStream(io.BytesIO):
    """A stream that remembers what was written when it was flushed."""

    def __init__(self) -> None:
        super().__init__()
        self.flushed_at: list[bytes] = []

    def flush(self) -> None:
        self.flushed_at.append(self.getvalue())
        super().flush()


def test_output_writes_lf_terminated_utf_8_lines_and_flushes_each_write() -> None:
    stream = RecordingStream()
    output = Output(stream)

    output.write(["Сайн", ""])
    output.write(["x"])

    assert stream.getvalue() == "Сайн\n\nx\n".encode()
    assert stream.flushed_at == ["Сайн\n\n".encode(), "Сайн\n\nx\n".encode()]


def test_output_writes_no_carriage_return() -> None:
    stream = RecordingStream()

    Output(stream).write(["a", "b"])

    assert b"\r" not in stream.getvalue()


def test_output_writes_and_flushes_nothing_for_nothing() -> None:
    stream = RecordingStream()

    Output(stream).write([])

    assert stream.getvalue() == b""
    assert stream.flushed_at == []


def test_a_blank_line_goes_between_documents_across_writes() -> None:
    stream = RecordingStream()
    output = Output(stream, blank_between=True)

    output.write(["one", "two"])
    output.write(["three"])

    assert stream.getvalue() == b"one\n\ntwo\n\nthree\n"


def test_no_blank_line_is_written_before_the_first_document_or_after_the_last() -> None:
    stream = RecordingStream()

    Output(stream, blank_between=True).write(["only"])

    assert stream.getvalue() == b"only\n"


# --- the summary ----------------------------------------------------------------------------


def summary_of(
    statuses: list[Status],
    *,
    unit: Any = "lines",
    counters: dict[str, int] | None = None,
) -> Summary:
    summary = Summary(unit)
    for number, status in enumerate(statuses, start=1):
        summary.add(number, status)
    summary.counters = {} if counters is None else counters
    return summary


def test_a_summary_of_lines_that_are_all_ok_is_one_line() -> None:
    summary = summary_of(["ok", "ok"])

    assert summary.stderr_text() == (
        "translated 2 lines: ok 2, empty 0, too_long 0, truncated 0, no_output 0"
    )


def test_a_summary_names_the_first_lines_of_each_problem() -> None:
    statuses: list[Status] = ["ok", "empty", "too_long", "truncated", "no_output", "ok"]

    assert summary_of(statuses).stderr_text() == (
        "translated 6 lines: ok 2, empty 1, too_long 1, truncated 1, no_output 1\n"
        "empty: first lines 2\n"
        "too_long: first lines 3\n"
        "truncated: first lines 4\n"
        "no_output: first lines 5"
    )


def test_a_summary_keeps_only_the_first_five_line_numbers_of_a_status() -> None:
    summary = summary_of(["empty"] * 8)

    assert summary.stderr_text() == (
        "translated 8 lines: ok 0, empty 8, too_long 0, truncated 0, no_output 0\n"
        "empty: first lines 1, 2, 3, 4, 5"
    )


def test_a_summary_of_documents_counts_documents() -> None:
    summary = summary_of(["ok", "truncated"], unit="documents")

    assert summary.stderr_text() == (
        "translated 2 documents: ok 1, empty 0, too_long 0, truncated 1, no_output 0\n"
        "truncated: first documents 2"
    )


def test_a_summary_of_nothing_says_zero() -> None:
    assert summary_of([]).stderr_text() == (
        "translated 0 lines: ok 0, empty 0, too_long 0, truncated 0, no_output 0"
    )


def test_a_summary_adds_the_counters_that_are_not_zero_in_a_fixed_order() -> None:
    counters = {
        "sources": 9,
        "empty": 1,
        "script_impure": 2,
        "zero_shot": 3,
        "script_retries": 4,
        "script_retry_improved": 1,
        "script_conversions": 0,
        "converter_rule_fallbacks": 5,
        "converter_invalid_outputs": 6,
    }

    text = summary_of(["ok"], counters=counters).stderr_text()

    assert text.splitlines()[1:] == [
        (
            "counters: zero_shot 3, script_retries 4, script_retry_improved 1, "
            "script_impure 2, converter_rule_fallbacks 5, converter_invalid_outputs 6"
        )
    ]


def test_a_summary_without_a_counter_that_is_not_zero_has_no_counters_line() -> None:
    counters = {"sources": 9, "empty": 1, "windows": 4, "script_retries": 0}

    assert "counters" not in summary_of(["ok"], counters=counters).stderr_text()


def test_the_summary_counts_problems_and_lines() -> None:
    summary = summary_of(["ok", "empty", "ok", "truncated"])

    assert (summary.total, summary.problems) == (4, 2)


def test_a_status_the_summary_does_not_know_is_refused() -> None:
    with pytest.raises(ValueError, match="status"):
        Summary("lines").add(1, "fine")  # type: ignore[arg-type]


def test_the_summary_names_its_counters_as_the_engine_does() -> None:
    assert set(translate_io.SUMMARY_COUNTERS) <= set(ENGINE_COUNTERS)
