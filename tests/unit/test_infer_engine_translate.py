"""`Engine.translate` (D11): the call the library gets for each placement, the statuses of a
text, what happens to a hypothesis, the guards, and the errors of a request.

The library is `LiveCt2` (a scripted stand-in); the SentencePiece model is `FakeProcessor`,
which cuts a text into one piece per word with the dummy prefix. A text of Latin words into
`eng_Latn` has the script of its target, so the default script mask stays out of the way here
(the masking has its own tests).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from rinchen.errors import ConfigError, DataError, UpstreamError
from rinchen.infer.engine_types import ENGINE_COUNTERS, DecodeOptions, Row, Translation
from tests.support.engines import (
    END,
    ENG,
    ENGINE_PIECES,
    KHK,
    MONGOLIAN,
    NNBSP,
    SP,
    TAG_ENG,
    ScriptedEngine,
    engine_environment,  # noqa: F401 - the fixture of `pytestmark`
    hyp,
    scripted_engine,
)
from tests.support.fake_ct2 import FakeProcessor, Hypothesis, ScriptedCt2

OPTIONS = DecodeOptions()
SECRET = "the secret words of the request"

pytestmark = pytest.mark.usefixtures("engine_environment")


def eng(*texts: str, src: str | None = None) -> list[Row]:
    return [Row(text, src, ENG) for text in texts]


def answering(*hypotheses: Hypothesis) -> Any:
    """A script that answers every source with `hypotheses`."""
    return lambda line: list(hypotheses)


def counted(fixture: ScriptedEngine, **expected: int) -> None:
    """The counters are zero except the ones named."""
    assert fixture.engine.counters() == {
        **dict.fromkeys(ENGINE_COUNTERS, 0),
        **expected,
    }


# -- the call the library gets


def test_the_call_for_the_placement_src_puts_the_tag_in_the_source(
    tmp_path: Path,
) -> None:
    fixture = scripted_engine(tmp_path)

    result = fixture.engine.translate(eng("hello world"), OPTIONS)

    (call,) = fixture.ct2.calls
    assert call.source == [["<2eng_Latn>", SP + "hello", SP + "world"]]
    assert call.target_prefix is None
    options = {k: v for k, v in call.options.items() if k != "suppress_sequences"}
    assert options == {
        "beam_size": 4,
        "num_hypotheses": 1,
        "max_batch_size": 32,
        "max_decoding_length": 256,
        "return_end_token": True,
        "disable_unk": True,
    }
    assert result == [
        Translation(
            text="hello world",
            status="ok",
            alternatives=(),
            src="und",
            tgt=ENG,
            script_purity=1.0,
            retried=False,
            zero_shot=False,
        )
    ]


def test_the_call_for_the_placement_tgt_forces_the_tag_as_the_prefix(
    tmp_path: Path,
) -> None:
    fixture = scripted_engine(tmp_path, placement="tgt")

    result = fixture.engine.translate(eng("hello world", "world"), OPTIONS)

    (call,) = fixture.ct2.calls
    assert call.source == [[SP + "hello", SP + "world"], [SP + "world"]]
    assert call.target_prefix == [["<2eng_Latn>"], ["<2eng_Latn>"]]
    assert [translation.text for translation in result] == ["hello world", "world"]


def test_the_forced_prefix_is_stripped_once_so_a_tag_the_model_wrote_is_a_reserved_piece(
    tmp_path: Path,
) -> None:
    fixture = scripted_engine(
        tmp_path, placement="tgt", script=answering(hyp(TAG_ENG, SP + "x", END))
    )

    (result,) = fixture.engine.translate(eng("hello"), OPTIONS)

    assert (result.status, result.text) == ("no_output", "")
    counted(fixture, sources=1, no_output=1, reserved_symbol=1)


def test_a_hypothesis_that_does_not_start_with_the_forced_prefix_names_the_row_only(
    tmp_path: Path,
) -> None:
    # A plain scripted library does not put the forced prefix in front, as a broken one.
    plain = ScriptedCt2(lambda line: [hyp(SP + "x", END)])
    fixture = scripted_engine(tmp_path, placement="tgt", ct2=plain)

    with pytest.raises(UpstreamError) as excinfo:
        fixture.engine.translate(eng("  ", SECRET), OPTIONS)

    message = str(excinfo.value)
    assert "row 1" in message
    assert "<2eng_Latn>" in message
    assert "secret" not in message
    assert fixture.engine.counters() == dict.fromkeys(ENGINE_COUNTERS, 0)


def test_the_options_of_the_decode_reach_the_library(tmp_path: Path) -> None:
    fixture = scripted_engine(tmp_path)
    options = DecodeOptions(
        beam_size=2, n=2, max_decoding_length=64, max_source_pieces=9, max_batch_size=8
    )

    fixture.engine.translate(eng("hello"), options)

    call = fixture.ct2.calls[0]
    assert call.options["beam_size"] == 2
    assert call.options["num_hypotheses"] == 2
    assert call.options["max_decoding_length"] == 64
    assert call.options["max_batch_size"] == 8


def test_no_options_means_the_defaults(tmp_path: Path) -> None:
    fixture = scripted_engine(tmp_path)

    fixture.engine.translate(eng("hello"))

    assert fixture.ct2.calls[0].options["beam_size"] == 4


# -- the options and the rows


def test_the_decode_options_have_these_defaults() -> None:
    assert DecodeOptions() == DecodeOptions(
        beam_size=4,
        n=1,
        max_decoding_length=256,
        max_source_pieces=256,
        max_batch_size=32,
        script_mask="retry",
    )


@pytest.mark.parametrize(
    "options",
    [
        {"beam_size": 0},
        {"beam_size": True},
        {"beam_size": 2.0},
        {"n": 0},
        {"n": 5},
        {"max_decoding_length": 0},
        {"max_source_pieces": 0},
        {"max_source_pieces": 1024},
        {"max_batch_size": 0},
        {"max_batch_size": False},
        {"script_mask": "sometimes"},
        {"script_mask": None},
    ],
)
def test_decode_options_the_library_cannot_honour_are_config_errors(
    options: dict[str, Any],
) -> None:
    with pytest.raises(ConfigError):
        DecodeOptions(**options)


def test_the_longest_source_is_the_library_limit_less_the_tag() -> None:
    assert DecodeOptions(max_source_pieces=1023).max_source_pieces == 1023
    assert DecodeOptions(beam_size=8, n=8).n == 8


def test_a_row_that_is_not_a_row_is_a_type_error(tmp_path: Path) -> None:
    fixture = scripted_engine(tmp_path)

    with pytest.raises(TypeError, match="row 1"):
        fixture.engine.translate([*eng("hello"), "world"], OPTIONS)  # type: ignore[list-item]

    assert fixture.ct2.calls == []


def test_a_text_that_is_not_text_is_a_type_error(tmp_path: Path) -> None:
    fixture = scripted_engine(tmp_path)

    with pytest.raises(TypeError, match="row 0"):
        fixture.engine.translate([Row(5, None, ENG)], OPTIONS)  # type: ignore[arg-type]


# -- the statuses of a text


def test_blank_and_punctuation_only_texts_are_empty_and_never_sent(
    tmp_path: Path,
) -> None:
    fixture = scripted_engine(tmp_path)

    result = fixture.engine.translate(
        eng("hello", "   ", "...!?", "world", chr(0x200B)), OPTIONS
    )

    assert [(r.status, r.text) for r in result] == [
        ("ok", "hello"),
        ("empty", ""),
        ("empty", ""),
        ("ok", "world"),
        ("empty", ""),
    ]
    (call,) = fixture.ct2.calls
    assert call.source == [
        ["<2eng_Latn>", SP + "hello"],
        ["<2eng_Latn>", SP + "world"],
    ]
    counted(fixture, sources=5, empty=3)


def test_a_text_of_digits_only_is_a_text(tmp_path: Path) -> None:
    fixture = scripted_engine(tmp_path)

    (result,) = fixture.engine.translate(eng("2026"), OPTIONS)

    assert (result.status, result.text) == ("ok", "2026")


def test_a_text_with_more_pieces_than_the_limit_is_too_long_and_never_sent(
    tmp_path: Path,
) -> None:
    fixture = scripted_engine(tmp_path)
    options = DecodeOptions(max_source_pieces=4)

    result = fixture.engine.translate(eng("a b c d", "a b c d e", "hello"), options)

    assert [r.status for r in result] == ["ok", "too_long", "ok"]
    assert result[1].text == "" and result[1].script_purity is None
    (call,) = fixture.ct2.calls
    assert len(call.source) == 2
    counted(fixture, sources=3, too_long=1)


def test_when_nothing_can_be_sent_nothing_is_called(tmp_path: Path) -> None:
    fixture = scripted_engine(tmp_path)

    result = fixture.engine.translate(eng("", "!!!"), OPTIONS)

    assert [r.status for r in result] == ["empty", "empty"]
    assert fixture.ct2.calls == []


def test_a_text_the_model_cuts_into_no_piece_is_empty(tmp_path: Path) -> None:
    class NoPieces(FakeProcessor):
        def encode(self, text: str, out_type: type = str) -> list[str]:
            return [] if text == "zzz" else super().encode(text, out_type)

    fixture = scripted_engine(tmp_path, processor=NoPieces(ENGINE_PIECES))

    result = fixture.engine.translate(eng("zzz", "hello"), OPTIONS)

    assert [r.status for r in result] == ["empty", "ok"]
    assert len(fixture.ct2.calls[0].source) == 1


def test_no_rows_give_no_results_and_no_call(tmp_path: Path) -> None:
    fixture = scripted_engine(tmp_path)

    assert fixture.engine.translate([], OPTIONS) == []
    assert fixture.ct2.calls == []


def test_the_arguments_are_not_changed_and_every_row_has_one_result(
    tmp_path: Path,
) -> None:
    fixture = scripted_engine(tmp_path)
    rows = eng("hello", "", "world", "a b c d e f", "...")
    kept = list(rows)
    options = DecodeOptions(max_source_pieces=5)

    result = fixture.engine.translate(rows, options)

    assert rows == kept
    assert len(result) == 5
    assert [r.status for r in result] == ["ok", "empty", "ok", "too_long", "empty"]


# -- the hypotheses


def test_a_hypothesis_without_the_end_token_is_truncated_and_keeps_its_partial_text(
    tmp_path: Path,
) -> None:
    fixture = scripted_engine(tmp_path, script=answering(hyp(SP + "hello", SP + "wor")))

    (result,) = fixture.engine.translate(eng("hello"), OPTIONS)

    assert (result.status, result.text) == ("truncated", "hello wor")
    assert result.script_purity == 1.0
    counted(fixture, sources=1, truncated=1)


@pytest.mark.parametrize(
    "piece",
    [
        "<unk>",
        END,
        "<2khk_Cyrl>",
        "<bt>",
        "<doc>",
        "<sep>",
        "<blank>",
        "averyunlikelytoken0",
    ],
)
def test_a_hypothesis_with_a_piece_no_translation_holds_is_dropped_and_the_next_is_the_text(
    tmp_path: Path, piece: str
) -> None:
    fixture = scripted_engine(
        tmp_path,
        script=answering(
            hyp(SP + "bad", piece, SP + "more", END), hyp(SP + "good", END)
        ),
    )
    options = DecodeOptions(n=2)

    (result,) = fixture.engine.translate(eng("hello"), options)

    assert (result.status, result.text, result.alternatives) == ("ok", "good", ())
    counted(fixture, sources=1, reserved_symbol=1)


def test_a_row_all_of_whose_hypotheses_are_dropped_has_no_output(
    tmp_path: Path,
) -> None:
    fixture = scripted_engine(
        tmp_path,
        script=answering(hyp(SP + "bad", "<unk>", END), hyp("<bt>", END)),
    )

    (result,) = fixture.engine.translate(eng("hello"), DecodeOptions(n=2))

    assert result == Translation(
        text="",
        status="no_output",
        alternatives=(),
        src="und",
        tgt=ENG,
        script_purity=None,
        retried=False,
        zero_shot=False,
    )
    counted(fixture, sources=1, no_output=1, reserved_symbol=2)


def test_a_numbered_placeholder_is_text_the_model_may_write(tmp_path: Path) -> None:
    placeholder = chr(0x27E6) + "3" + chr(0x27E7)
    fixture = scripted_engine(
        tmp_path, script=answering(hyp(SP + "hello", placeholder, END))
    )

    (result,) = fixture.engine.translate(eng("hello"), OPTIONS)

    assert (result.status, result.text) == ("ok", "hello" + placeholder)
    counted(fixture, sources=1)


def test_a_byte_piece_is_ordinary_text(tmp_path: Path) -> None:
    fixture = scripted_engine(
        tmp_path, script=answering(hyp(SP + "hello", "<0x41>", END))
    )

    (result,) = fixture.engine.translate(eng("hello"), OPTIONS)

    assert result.status == "ok"
    counted(fixture, sources=1)


def test_n_hypotheses_give_distinct_alternatives_in_order(tmp_path: Path) -> None:
    fixture = scripted_engine(
        tmp_path,
        script=answering(
            hyp(SP + "a", END),
            hyp(SP + "b", END),
            hyp(SP + "a", END),
            hyp(SP + "c", END),
        ),
    )

    (result,) = fixture.engine.translate(eng("hello"), DecodeOptions(n=3))

    assert (result.text, result.alternatives) == ("a", ("b", "c"))


def test_the_status_follows_the_best_hypothesis(tmp_path: Path) -> None:
    cut_first = scripted_engine(
        tmp_path / "one",
        script=answering(hyp(SP + "cut"), hyp(SP + "whole", END)),
    )
    whole_first = scripted_engine(
        tmp_path / "two",
        script=answering(hyp(SP + "whole", END), hyp(SP + "cut")),
    )

    (first,) = cut_first.engine.translate(eng("hello"), DecodeOptions(n=2))
    (second,) = whole_first.engine.translate(eng("hello"), DecodeOptions(n=2))

    assert (first.status, first.text, first.alternatives) == (
        "truncated",
        "cut",
        ("whole",),
    )
    assert (second.status, second.text, second.alternatives) == (
        "ok",
        "whole",
        ("cut",),
    )


def test_a_repeated_text_keeps_the_status_of_its_first_hypothesis(
    tmp_path: Path,
) -> None:
    fixture = scripted_engine(
        tmp_path, script=answering(hyp(SP + "x"), hyp(SP + "x", END))
    )

    (result,) = fixture.engine.translate(eng("hello"), DecodeOptions(n=2))

    assert (result.status, result.text, result.alternatives) == ("truncated", "x", ())


def test_a_hypothesis_that_decodes_to_nothing_is_not_a_translation(
    tmp_path: Path,
) -> None:
    fixture = scripted_engine(
        tmp_path, script=answering(hyp(END), hyp(SP + "good", END))
    )

    (result,) = fixture.engine.translate(eng("hello"), DecodeOptions(n=2))

    assert (result.status, result.text, result.alternatives) == ("ok", "good", ())


def test_a_row_whose_only_hypothesis_decodes_to_nothing_has_no_output(
    tmp_path: Path,
) -> None:
    fixture = scripted_engine(tmp_path, script=answering(hyp(END)))

    (result,) = fixture.engine.translate(eng("hello"), OPTIONS)

    assert (result.status, result.text) == ("no_output", "")


@pytest.mark.parametrize(
    ("separator", "expected"),
    [
        ("\n", "a b"),
        ("\r\n", "a b"),
        (chr(0x2028), "a b"),
        (chr(0x2029), "a b"),
        (chr(0x0B), "a b"),
        (chr(0x0C), "a b"),
        (chr(0x1C), "a b"),
        (chr(0x1D), "a b"),
        (chr(0x1E), "a b"),
        (chr(0x85), "a b"),
        ("\n\n\n", "a b"),
    ],
)
def test_a_line_break_in_the_output_becomes_one_space_and_is_counted(
    tmp_path: Path, separator: str, expected: str
) -> None:
    fixture = scripted_engine(
        tmp_path, script=answering(hyp(SP + "a" + separator + "b", END))
    )

    (result,) = fixture.engine.translate(eng("hello"), OPTIONS)

    assert (result.status, result.text) == ("ok", expected)
    counted(fixture, sources=1, line_break=1)


def test_a_line_break_at_the_end_of_the_output_leaves_no_space(tmp_path: Path) -> None:
    fixture = scripted_engine(tmp_path, script=answering(hyp(SP + "a" + "\n", END)))

    (result,) = fixture.engine.translate(eng("hello"), OPTIONS)

    assert result.text == "a"
    counted(fixture, sources=1, line_break=1)


# -- what fails


@pytest.mark.parametrize("error", [RuntimeError, ValueError])
def test_a_failure_of_the_library_is_an_upstream_error_that_quotes_no_text(
    tmp_path: Path, error: type[Exception]
) -> None:
    fixture = scripted_engine(tmp_path)
    cause = error(f"failed on {SECRET}")
    fixture.ct2.error = cause

    with pytest.raises(UpstreamError) as excinfo:
        fixture.engine.translate(eng(SECRET), OPTIONS)

    assert excinfo.value.__cause__ is cause
    assert "secret" not in str(excinfo.value)
    assert error.__name__ in str(excinfo.value)
    assert fixture.engine.counters() == dict.fromkeys(ENGINE_COUNTERS, 0)


@pytest.mark.parametrize("offset", [1, -1])
def test_a_library_that_returns_another_number_of_results_is_an_upstream_error(
    tmp_path: Path, offset: int
) -> None:
    fixture = scripted_engine(tmp_path)
    fixture.ct2.result_offset = offset

    with pytest.raises(UpstreamError, match="2 sources"):
        fixture.engine.translate(eng("hello", "world"), OPTIONS)

    assert fixture.engine.counters() == dict.fromkeys(ENGINE_COUNTERS, 0)


def test_a_lone_surrogate_is_a_data_error_naming_the_row_before_anything_is_sent(
    tmp_path: Path,
) -> None:
    fixture = scripted_engine(tmp_path)

    with pytest.raises(DataError, match="row 1") as excinfo:
        fixture.engine.translate(eng("hello", "bad" + chr(0xD800)), OPTIONS)

    assert "bad" not in str(excinfo.value)
    assert fixture.ct2.calls == []
    assert fixture.engine.counters() == dict.fromkeys(ENGINE_COUNTERS, 0)


def test_a_failed_call_counts_nothing_even_when_part_of_it_was_counted(
    tmp_path: Path,
) -> None:
    fixture = scripted_engine(tmp_path)
    fixture.ct2.result_offset = 1

    with pytest.raises(UpstreamError):
        fixture.engine.translate(eng("", "!!", "hello"), OPTIONS)

    assert fixture.engine.counters() == dict.fromkeys(ENGINE_COUNTERS, 0)


# -- what the text is before it is cut into pieces


def test_the_processor_sees_the_normal_form_of_the_text(tmp_path: Path) -> None:
    fixture = scripted_engine(tmp_path)
    decomposed = "e" + chr(0x0301) + "ta"
    zero_width = "a" + chr(0x200B) + "b"

    fixture.engine.translate(eng(decomposed, zero_width, "x    y", src=KHK), OPTIONS)

    assert fixture.processor.encoded == [chr(0xE9) + "ta", "ab", "x y"]


def test_a_text_in_the_mongolian_script_keeps_its_narrow_no_break_space(
    tmp_path: Path,
) -> None:
    fixture = scripted_engine(tmp_path)
    text = MONGOLIAN + NNBSP + MONGOLIAN

    fixture.engine.translate([Row(text, None, ENG)], OPTIONS)
    fixture.engine.translate([Row(text, ENG, KHK)], OPTIONS)

    assert fixture.processor.encoded == [text, MONGOLIAN + " " + MONGOLIAN]


def test_the_source_language_of_a_result_is_the_one_given_or_the_script_hint(
    tmp_path: Path,
) -> None:
    fixture = scripted_engine(tmp_path)

    result = fixture.engine.translate(
        [
            Row("hello", None, ENG),
            Row(MONGOLIAN, None, ENG),
            Row("hello", KHK, ENG),
            Row("", None, ENG),
        ],
        OPTIONS,
    )

    assert [r.src for r in result] == ["und", "mn_Mong", KHK, "und"]


def test_results_keep_the_order_of_the_rows_whatever_was_sent(tmp_path: Path) -> None:
    fixture = scripted_engine(tmp_path)
    texts = ["one", "", "two", "...", "three", "four five six seven eight nine"]
    options = DecodeOptions(max_source_pieces=5)

    result = fixture.engine.translate(eng(*texts), options)

    assert [r.text for r in result] == ["one", "", "two", "", "three", ""]


def test_a_batch_is_sent_in_one_call(tmp_path: Path) -> None:
    fixture = scripted_engine(tmp_path)

    result = fixture.engine.translate(eng(*[f"w{n}" for n in range(40)]), OPTIONS)

    assert len(result) == 40
    assert len(fixture.ct2.calls) == 1
    assert len(fixture.ct2.calls[0].source) == 40


def test_the_counters_add_up_over_calls(tmp_path: Path) -> None:
    fixture = scripted_engine(tmp_path)

    fixture.engine.translate(eng("hello", ""), OPTIONS)
    fixture.engine.translate(eng("world", "!"), OPTIONS)

    counted(fixture, sources=4, empty=2)
