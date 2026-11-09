"""`rinchen translate` (D11, Task 5): what the command writes for what it reads.

The engine is a `FakeEngine` behind the seam `rinchen.cli.translate.open_engine`; its answer
depends on the text of the line (see `tests.support.translate_cli`), so each status is asked for
by the text a test writes. The real engine over its stand-ins is used at the end of the file.
The refusals and the exit codes are in `test_cli_translate_exits.py`.

The contract: one output line per input line (per document with `--doc`) whatever the
statuses, in order; stdout carries only translations; the summary goes to stderr; a problem is
an exit code of 0 unless `--strict`.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from click.testing import Result

from rinchen.infer.engine_types import DecodeOptions, LoadOptions, Row, Translation
from tests.support.engines import (
    ENG,
    KHK,
    MON,
    FakeConverter,
    engine_environment,  # noqa: F401 - the fixture of the tests that open the real engine
    scripted_engine,
)
from tests.support.translate_cli import (
    MODEL,
    NO_OUTPUT_TEXT,
    SRC,
    TGT,
    FakeEngine,
    Opened,
    install,
    invoke,
    probe_stdin,
    translate_args,
    translation_of,
)

LONG = "x" * 31
ALL_OK = "translated 2 lines: ok 2, empty 0, too_long 0, truncated 0, no_output 0"
# One of each status: ok, empty, too_long, truncated, no_output, ok.
MIXED_INPUT = f"hello\n\n{LONG}\ncut...\n{NO_OUTPUT_TEXT}\nworld\n"
MIXED_OUTPUT = "<hello>\n\n\n<cut...>\n\n<world>\n"
MIXED_SUMMARY = (
    "translated 6 lines: ok 2, empty 1, too_long 1, truncated 1, no_output 1\n"
    "empty: first lines 2\n"
    "too_long: first lines 3\n"
    "truncated: first lines 4\n"
    "no_output: first lines 5\n"
)


@pytest.fixture
def opened(monkeypatch: pytest.MonkeyPatch) -> Opened:
    return install(monkeypatch)


def run_ok(*args: str, stdin: bytes | str = b"", tgt: str | None = TGT) -> Result:
    result = invoke(translate_args(*args, tgt=tgt), stdin=stdin)
    assert result.exit_code == 0, result.stderr
    return result


def texts(opened: Opened) -> list[list[str]]:
    """The texts of the rows of each call of `translate`."""
    return [[row.text for row in call.rows] for call in opened.engine.calls]


# --- one output line per input line ---------------------------------------------------------


def test_a_translation_is_written_for_each_line_in_order(opened: Opened) -> None:
    result = run_ok(stdin="hello\nworld\n")

    assert result.stdout == "<hello>\n<world>\n"
    assert result.stderr == ALL_OK + "\n"


def test_every_status_gives_a_line_and_the_summary_says_which(opened: Opened) -> None:
    result = run_ok(stdin=MIXED_INPUT)

    assert result.stdout == MIXED_OUTPUT
    assert result.stderr == MIXED_SUMMARY


def test_an_empty_line_and_a_line_of_spaces_give_empty_lines(opened: Opened) -> None:
    result = run_ok(stdin="a\n\n   \nb\n")

    assert result.stdout == "<a>\n\n\n<b>\n"
    assert texts(opened) == [["a", "", "   ", "b"]]
    assert result.stderr == (
        "translated 4 lines: ok 2, empty 2, too_long 0, truncated 0, no_output 0\n"
        "empty: first lines 2, 3\n"
    )


def test_a_very_long_line_gives_an_empty_line_and_is_in_the_summary(
    opened: Opened,
) -> None:
    result = run_ok(stdin=f"a\n{LONG}\nb\n")

    assert result.stdout == "<a>\n\n<b>\n"
    assert result.stderr == (
        "translated 3 lines: ok 2, empty 0, too_long 1, truncated 0, no_output 0\n"
        "too_long: first lines 2\n"
    )


def test_a_final_line_without_lf_is_translated_and_the_output_ends_with_lf(
    opened: Opened,
) -> None:
    result = run_ok(stdin=b"a\nb")

    assert result.stdout_bytes == b"<a>\n<b>\n"


def test_a_final_empty_line_gives_a_final_empty_line(opened: Opened) -> None:
    result = run_ok(stdin=b"a\n\n")

    assert result.stdout_bytes == b"<a>\n\n"
    assert texts(opened) == [["a", ""]]


def test_crlf_input_is_read_as_lf_and_written_as_lf(opened: Opened) -> None:
    result = run_ok(stdin=b"a\r\nb\r\n")

    assert texts(opened) == [["a", "b"]]
    assert result.stdout_bytes == b"<a>\n<b>\n"


def test_a_line_separator_inside_a_line_does_not_split_it(opened: Opened) -> None:
    separators = chr(0x2028) + chr(0x2029) + chr(0x85) + chr(0x0B) + chr(0x0C)

    result = run_ok(stdin=f"a{separators}b\nc\n")

    assert texts(opened) == [[f"a{separators}b", "c"]]
    assert result.stdout.count("\n") == 2


def test_an_empty_input_gives_no_output_and_no_call(opened: Opened) -> None:
    result = run_ok(stdin=b"")

    assert result.stdout == ""
    assert opened.engine.calls == []
    assert result.stderr == (
        "translated 0 lines: ok 0, empty 0, too_long 0, truncated 0, no_output 0\n"
    )


def test_the_output_is_utf_8_with_no_carriage_return(opened: Opened) -> None:
    result = run_ok(stdin="Сайн байна уу\n中文\r\n")

    assert result.stdout_bytes == "<Сайн байна уу>\n<中文>\n".encode()


def test_a_line_break_in_an_answer_never_moves_the_next_line_up(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def answer(row: Row) -> Translation:
        return translation_of(row, f"{row.text}\nsplit", "ok")

    install(monkeypatch, FakeEngine(answer=answer))

    result = run_ok(stdin="a\nb\n")

    assert result.stdout == "a split\nb split\n"


# --- chunks ---------------------------------------------------------------------------------


def test_lines_are_translated_in_chunks_of_the_batch_size(opened: Opened) -> None:
    result = run_ok("--batch-size", "2", stdin="a\nb\nc\nd\ne\n")

    assert texts(opened) == [["a", "b"], ["c", "d"], ["e"]]
    assert result.stdout == "<a>\n<b>\n<c>\n<d>\n<e>\n"


def test_the_default_batch_size_is_32_lines(opened: Opened) -> None:
    run_ok(stdin="".join(f"line{number}\n" for number in range(70)))

    assert [len(call.rows) for call in opened.engine.calls] == [32, 32, 6]


def test_a_batch_size_above_the_number_of_lines_is_one_call(opened: Opened) -> None:
    run_ok("--batch-size", "100", stdin="a\nb\nc\n")

    assert texts(opened) == [["a", "b", "c"]]


def test_the_output_is_flushed_after_each_chunk(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    out = tmp_path / "out.txt"
    seen: dict[int, bytes] = {}

    def look(number: int) -> None:
        seen[number] = out.read_bytes()

    install(monkeypatch, FakeEngine(on_call=look))

    run_ok("--batch-size", "1", "--output", str(out), stdin="a\nb\nc\n")

    assert seen == {1: b"", 2: b"<a>\n", 3: b"<a>\n<b>\n"}
    assert out.read_bytes() == b"<a>\n<b>\n<c>\n"


def test_the_model_is_opened_before_the_input_is_read_and_each_chunk_is_read_when_needed(
    opened: Opened, monkeypatch: pytest.MonkeyPatch
) -> None:
    probe_stdin(monkeypatch, b"a\nb\nc\n", opened.events)

    result = invoke(translate_args("--batch-size", "2"))

    assert result.exit_code == 0, result.stderr
    assert opened.events == [
        "open",
        "read",
        "read",
        "translate",
        "read",
        "read",
        "translate",
    ]


def test_the_engine_is_opened_once_and_closed_after_the_run(opened: Opened) -> None:
    run_ok("--batch-size", "1", stdin="a\nb\n")

    assert len(opened.calls) == 1
    assert opened.engine.closed == 1


# --- what the engine is asked ---------------------------------------------------------------


def test_the_defaults_are_the_engines(opened: Opened) -> None:
    run_ok(stdin="a\n")

    (call,) = opened.calls
    assert call.model == Path(os.path.abspath(MODEL))
    assert call.load == LoadOptions()
    assert call.converter_paths is None
    (rows_call,) = opened.engine.calls
    assert rows_call.options == DecodeOptions()
    assert rows_call.rows == [Row("a", None, TGT, None)]


def test_every_option_reaches_the_engine(opened: Opened) -> None:

    run_ok(
        "--src",
        SRC,
        "--script",
        "Mong",
        "--beam-size",
        "6",
        "--n-best",
        "2",
        "--max-decoding-length",
        "40",
        "--max-source-pieces",
        "50",
        "--batch-size",
        "3",
        "--script-mask",
        "always",
        "--device",
        "cpu",
        "--compute-type",
        "int8",
        "--inter-threads",
        "2",
        "--intra-threads",
        "3",
        "--script-converter-ct2",
        "conv",
        "--script-converter-spm",
        "conv.model",
        tgt=KHK,
        stdin="a\nb\n",
    )

    (call,) = opened.calls
    assert call.load == LoadOptions(
        device="cpu", compute_type="int8", inter_threads=2, intra_threads=3
    )
    assert call.converter_paths == (
        Path(os.path.abspath("conv")),
        Path(os.path.abspath("conv.model")),
    )
    (rows_call,) = opened.engine.calls
    assert rows_call.options == DecodeOptions(
        beam_size=6,
        n=2,
        max_decoding_length=40,
        max_source_pieces=50,
        max_batch_size=3,
        script_mask="always",
    )
    assert rows_call.rows == [Row("a", SRC, KHK, "Mong"), Row("b", SRC, KHK, "Mong")]


def test_the_model_path_is_made_absolute(opened: Opened) -> None:
    run_ok(stdin="a\n")

    assert opened.calls[0].model.is_absolute()
    assert opened.calls[0].model == Path(os.path.abspath(MODEL))


# --- documents ------------------------------------------------------------------------------


def test_blocks_are_documents_and_the_output_is_a_line_per_document(
    opened: Opened,
) -> None:
    result = run_ok("--doc", "--src", SRC, stdin="one\ntwo\n\nthree\n")

    assert [call.documents for call in opened.engine.document_calls] == [
        ["one two", "three"]
    ]
    assert result.stdout == "<one two>\n\n<three>\n"
    assert result.stderr == (
        "translated 2 documents: ok 2, empty 0, too_long 0, truncated 0, no_output 0\n"
    )


def test_a_document_is_asked_for_with_the_languages_and_the_options(
    opened: Opened,
) -> None:
    run_ok(
        "--doc",
        "--src",
        SRC,
        "--script",
        "Cyrl",
        "--beam-size",
        "2",
        stdin="a\n",
        tgt=KHK,
    )

    (call,) = opened.engine.document_calls
    assert (call.src, call.tgt, call.script) == (SRC, KHK, "Cyrl")
    assert call.options == DecodeOptions(beam_size=2)
    assert opened.engine.calls == []


def test_runs_of_blank_lines_between_documents_make_one_separator(
    opened: Opened,
) -> None:
    result = run_ok("--doc", "--src", SRC, stdin="\n\none\n\n\n\ntwo\n\n\n")

    assert result.stdout == "<one>\n\n<two>\n"


def test_documents_are_translated_in_chunks_and_the_separator_goes_between_chunks(
    opened: Opened,
) -> None:
    result = run_ok("--doc", "--src", SRC, "--batch-size", "1", stdin="a\n\nb\n\nc\n")

    assert [call.documents for call in opened.engine.document_calls] == [
        ["a"],
        ["b"],
        ["c"],
    ]
    assert result.stdout == "<a>\n\n<b>\n\n<c>\n"


def test_a_document_that_is_not_ok_is_in_the_summary(opened: Opened) -> None:
    result = run_ok("--doc", "--src", SRC, stdin=f"fine\n\n{LONG}\n\ncut...\n")

    assert result.stdout == "<fine>\n\n\n\n<cut...>\n"
    assert result.stderr == (
        "translated 3 documents: ok 1, empty 0, too_long 1, truncated 1, no_output 0\n"
        "too_long: first documents 2\n"
        "truncated: first documents 3\n"
    )


def test_an_input_without_a_document_gives_no_output(opened: Opened) -> None:
    result = run_ok("--doc", "--src", SRC, stdin="\n\n  \n")

    assert result.stdout == ""
    assert opened.engine.document_calls == []


# --- the formats ----------------------------------------------------------------------------


def test_jsonl_writes_one_object_per_line(opened: Opened) -> None:
    result = run_ok("--format", "jsonl", "--src", SRC, stdin="hello\n\n")

    assert result.stdout == (
        '{"text":"<hello>","status":"ok","alternatives":[],'
        '"src":"eng_Latn","tgt":"khk_Cyrl","script_purity":1.0}\n'
        '{"text":"","status":"empty","alternatives":[],'
        '"src":"eng_Latn","tgt":"khk_Cyrl","script_purity":null}\n'
    )


def test_jsonl_of_a_text_line_without_a_source_language_says_und(
    opened: Opened,
) -> None:
    result = run_ok("--format", "jsonl", stdin="hello\n")

    assert '"src":"und"' in result.stdout


def test_jsonl_keeps_the_alternatives_the_text_format_drops(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def answer(row: Row) -> Translation:
        return translation_of(row, "Сайн", "ok", alternatives=("Мэнд",), purity=1.0)

    install(monkeypatch, FakeEngine(answer=answer))

    jsonl = run_ok("--format", "jsonl", "--n-best", "2", stdin="a\n")
    text = run_ok("--n-best", "2", stdin="a\n")

    assert '"text":"Сайн","status":"ok","alternatives":["Мэнд"]' in jsonl.stdout
    assert text.stdout == "Сайн\n"


def test_jsonl_of_documents_is_one_object_per_document_with_no_blank_line(
    opened: Opened,
) -> None:
    result = run_ok(
        "--doc", "--format", "jsonl", "--src", SRC, stdin="hello\n\nworld\n"
    )

    assert result.stdout == (
        '{"text":"<hello>","status":"ok","alternatives":[],'
        '"src":"eng_Latn","tgt":"khk_Cyrl","script_purity":1.0}\n'
        '{"text":"<world>","status":"ok","alternatives":[],'
        '"src":"eng_Latn","tgt":"khk_Cyrl","script_purity":1.0}\n'
    )


def test_jsonl_of_a_document_names_the_script_target(opened: Opened) -> None:
    result = run_ok(
        "--doc",
        "--format",
        "jsonl",
        "--src",
        SRC,
        "--script",
        "Mong",
        stdin="a\n",
        tgt=KHK,
    )

    assert '"tgt":"mn_Mong"' in result.stdout


# --- the summary and --strict ---------------------------------------------------------------


def test_stdout_holds_only_the_translations_and_stderr_only_the_summary(
    opened: Opened,
) -> None:
    result = invoke(translate_args(), stdin=MIXED_INPUT)

    assert result.stdout == MIXED_OUTPUT
    assert result.stderr == MIXED_SUMMARY


def test_problems_do_not_change_the_exit_code_without_strict(opened: Opened) -> None:
    result = invoke(translate_args(), stdin=MIXED_INPUT)

    assert result.exit_code == 0


def test_strict_exits_4_after_the_whole_output_and_the_summary_are_written(
    opened: Opened,
) -> None:
    result = invoke(translate_args("--strict"), stdin=MIXED_INPUT)

    assert result.exit_code == 4, result.output
    assert result.stdout == MIXED_OUTPUT
    assert result.stderr == (
        MIXED_SUMMARY + "error: --strict: 4 of 6 lines were not translated ok\n"
    )


def test_strict_counts_an_empty_line_as_a_problem(opened: Opened) -> None:
    result = invoke(translate_args("--strict"), stdin="a\n\nb\n")

    assert result.exit_code == 4
    assert result.stdout == "<a>\n\n<b>\n"
    assert result.stderr.endswith(
        "error: --strict: 1 of 3 lines were not translated ok\n"
    )


def test_strict_with_every_line_ok_exits_0(opened: Opened) -> None:
    result = invoke(translate_args("--strict"), stdin="a\nb\n")

    assert result.exit_code == 0
    assert result.stderr == ALL_OK + "\n"


def test_strict_with_documents_names_documents(opened: Opened) -> None:
    result = invoke(
        translate_args("--strict", "--doc", "--src", SRC), stdin=f"a\n\n{LONG}\n"
    )

    assert result.exit_code == 4
    assert result.stderr.endswith(
        "error: --strict: 1 of 2 documents were not translated ok\n"
    )


def test_the_engine_counters_that_matter_are_in_the_summary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install(
        monkeypatch,
        FakeEngine(
            counter_values={"script_retries": 2, "script_impure": 1, "sources": 9}
        ),
    )

    result = run_ok(stdin="a\nb\n")

    assert result.stderr == (ALL_OK + "\ncounters: script_retries 2, script_impure 1\n")


# --- the real engine behind the seam --------------------------------------------------------

# Cyrillic words the stand-in processor knows as pieces, so that the "translation" (the text
# itself) has the script of the target and no retry or purity warning is in the way.
ONE, THREE = "Энэ", "Энэ ном ус"


@pytest.mark.usefixtures("engine_environment")
class TestOverTheRealEngine:
    """The engine over a fabricated export and the stand-ins of the library (`echo`: the
    translation of a text is the text): the statuses are the engine's."""

    def test_the_lines_keep_their_places_whatever_the_engine_says(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        install(monkeypatch, scripted_engine(tmp_path).engine)

        result = invoke(
            translate_args("--src", ENG), stdin=f"{ONE}\n\n...\n{THREE}\nном\n"
        )

        assert result.exit_code == 0, result.stderr
        assert result.stdout == f"{ONE}\n\n\n{THREE}\nном\n"
        assert result.stderr == (
            "translated 5 lines: ok 3, empty 2, too_long 0, truncated 0, no_output 0\n"
            "empty: first lines 2, 3\n"
        )

    def test_the_source_piece_limit_makes_a_line_too_long(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        install(monkeypatch, scripted_engine(tmp_path).engine)

        result = invoke(
            translate_args("--src", ENG, "--max-source-pieces", "2", "--strict"),
            stdin=f"{ONE}\n{THREE}\n",
        )

        assert result.exit_code == 4
        assert result.stdout == f"{ONE}\n\n"
        assert "too_long: first lines 2" in result.stderr

    def test_jsonl_reports_the_engines_purity_and_languages(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        install(monkeypatch, scripted_engine(tmp_path).engine)

        result = invoke(
            translate_args("--format", "jsonl", "--src", ENG), stdin=f"{ONE}\n"
        )

        assert result.stdout == (
            f'{{"text":"{ONE}","status":"ok","alternatives":[],'
            '"src":"eng_Latn","tgt":"khk_Cyrl","script_purity":1.0}\n'
        )

    def test_the_engine_is_closed_when_the_run_is_over(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        scripted = scripted_engine(tmp_path)
        install(monkeypatch, scripted.engine)

        invoke(translate_args(), stdin=f"{ONE}\n")

        assert scripted.ct2.unloaded == 1

    def test_a_target_the_model_cannot_write_is_a_refusal_before_any_output(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        install(monkeypatch, scripted_engine(tmp_path).engine)

        result = invoke(translate_args(tgt="deu_Latn"), stdin=f"{ONE}\n")

        assert result.exit_code == 2
        assert result.stdout == ""
        assert result.stderr.startswith(
            "error: the model does not translate into 'deu_Latn'"
        )

    def test_the_traditional_script_is_made_by_the_converter_when_one_is_attached(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        converter = FakeConverter(prefix="[mong]", rule_fallbacks=2, invalid_outputs=1)
        scripted = scripted_engine(
            tmp_path, directions=("eng_Latn-khk_Cyrl",), converter=converter
        )
        install(monkeypatch, scripted.engine)

        result = invoke(
            translate_args(
                "--src", ENG, "--script", "Mong", "--format", "jsonl", tgt=KHK
            ),
            stdin=f"{ONE}\n",
        )

        assert result.exit_code == 0, result.stderr
        assert converter.calls == [(ONE, "Mong")]
        assert f'"text":"[mong]{ONE}","status":"ok"' in result.stdout
        assert f'"tgt":"{MON}"' in result.stdout
        assert result.stderr.endswith(
            "counters: script_impure 1, script_conversions 1, "
            "converter_rule_fallbacks 2, converter_invalid_outputs 1\n"
        )
