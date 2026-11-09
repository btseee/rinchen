"""`Engine.translate_documents` (D11, ruling T10): a document is cut into sentences, the
sentences into windows, a window is one call whose source is `[tag, "<doc>", *pieces]`, and a
window whose output does not line up (or was cut) is translated sentence by sentence.

The model of these tests copies a window with the `<sep>` pieces turned into the real symbol
and copies a sentence: what comes out is what went in, so the windows line up unless a test
says they must not.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import pytest

from rinchen.errors import ConfigError, DataError
from rinchen.infer.doc_chunker import ChunkStats
from rinchen.infer.engine_types import ENGINE_COUNTERS, DecodeOptions, Row
from tests.support.engines import (
    END,
    ENG,
    KHK,
    MON,
    RUS,
    SP,
    TAG_KHK,
    TAG_RUS,
    FakeConverter,
    ScriptedEngine,
    copy_windows,
    engine_environment,  # noqa: F401 - the fixture of `pytestmark`
    hyp,
    scripted_engine,
)
from tests.support.fake_ct2 import Hypothesis

SEP = "<sep>"
DOC = "Энэ ном. Энэ ус. Энэ гэр."
SENTENCES = ["Энэ ном.", "Энэ ус.", "Энэ гэр."]
WINDOW = [
    SP + "Энэ",
    SP + "ном.",
    SP + SEP,
    SP + "Энэ",
    SP + "ус.",
    SP + SEP,
    SP + "Энэ",
    SP + "гэр.",
]
OFF = DecodeOptions(script_mask="off")
LATIN = SP + "hello"

pytestmark = pytest.mark.usefixtures("engine_environment")


def is_window(line: tuple[str, ...]) -> bool:
    return len(line) > 1 and line[1] == "<doc>"


def documents(tmp_path: Path, **keywords: object) -> ScriptedEngine:
    keywords.setdefault("script", copy_windows)
    return scripted_engine(tmp_path, document_level=True, **keywords)  # type: ignore[arg-type]


def counted(fixture: ScriptedEngine, **expected: int) -> None:
    assert fixture.engine.counters() == {
        **dict.fromkeys(ENGINE_COUNTERS, 0),
        **expected,
    }


# -- a window


def test_a_window_is_one_call_whose_source_is_the_tag_the_doc_marker_and_the_pieces(
    tmp_path: Path,
) -> None:
    fixture = documents(tmp_path)

    (result,) = fixture.engine.translate_documents([DOC], KHK, RUS, OFF)

    (call,) = fixture.ct2.calls
    assert call.source == [[TAG_RUS, "<doc>", *WINDOW]]
    assert call.target_prefix is None
    assert call.options["return_end_token"] is True
    assert call.options["disable_unk"] is True
    assert [t.text for t in result.translations] == SENTENCES
    assert result.text == DOC
    assert result.stats == ChunkStats(
        sentences=3,
        chunks=1,
        windows=1,
        aligned=1,
        fallbacks=(),
        singles=0,
        oversize=0,
        empty=0,
    )
    counted(fixture, windows=1)


def test_a_window_is_asked_for_one_hypothesis_whatever_n_is(tmp_path: Path) -> None:
    fixture = documents(tmp_path)

    fixture.engine.translate_documents(
        [DOC], KHK, RUS, DecodeOptions(n=3, script_mask="off")
    )

    assert fixture.ct2.calls[0].options["num_hypotheses"] == 1
    assert fixture.ct2.calls[0].options["beam_size"] == 4


def test_a_translation_of_a_window_has_the_fields_of_a_translation(
    tmp_path: Path,
) -> None:
    fixture = documents(tmp_path)

    (result,) = fixture.engine.translate_documents([DOC], KHK, RUS, OFF)

    first = result.translations[0]
    assert (first.status, first.alternatives, first.src, first.tgt) == (
        "ok",
        (),
        KHK,
        RUS,
    )
    assert first.zero_shot is False and first.retried is False
    assert first.script_purity == 1.0


def test_the_window_of_a_document_that_the_model_cut_is_translated_sentence_by_sentence(
    tmp_path: Path,
) -> None:
    def cut(line: tuple[str, ...]) -> Sequence[Hypothesis]:
        return [hyp(*line[2:])] if is_window(line) else [hyp(*line[1:], END)]

    fixture = documents(tmp_path, script=cut)

    (result,) = fixture.engine.translate_documents([DOC], KHK, RUS, OFF)

    window, *sentences = fixture.ct2.calls
    assert window.source[0][1] == "<doc>"
    assert [call.source for call in sentences] == [
        [[TAG_RUS, SP + "Энэ", SP + "ном."]],
        [[TAG_RUS, SP + "Энэ", SP + "ус."]],
        [[TAG_RUS, SP + "Энэ", SP + "гэр."]],
    ]
    assert [t.text for t in result.translations] == SENTENCES
    assert [t.status for t in result.translations] == ["ok", "ok", "ok"]
    assert result.stats.fallbacks == (0,) and result.stats.aligned == 0
    counted(fixture, windows=1, window_fallbacks=1, sources=3)


def test_a_window_that_comes_back_with_another_number_of_segments_falls_back(
    tmp_path: Path,
) -> None:
    def merged(line: tuple[str, ...]) -> Sequence[Hypothesis]:
        if is_window(line):
            return [hyp(SP + "Энэ", SEP, SP + "ном", END)]
        return [hyp(*line[1:], END)]

    fixture = documents(tmp_path, script=merged)

    (result,) = fixture.engine.translate_documents([DOC], KHK, RUS, OFF)

    assert len(fixture.ct2.calls) == 4
    assert [t.text for t in result.translations] == SENTENCES
    counted(fixture, windows=1, window_fallbacks=1, sources=3)


def test_a_window_with_an_empty_segment_falls_back(tmp_path: Path) -> None:
    def holey(line: tuple[str, ...]) -> Sequence[Hypothesis]:
        if is_window(line):
            return [hyp(SP + "Энэ", SEP, SEP, SP + "ном", END)]
        return [hyp(*line[1:], END)]

    fixture = documents(tmp_path, script=holey)

    (result,) = fixture.engine.translate_documents([DOC], KHK, RUS, OFF)

    assert [t.text for t in result.translations] == SENTENCES
    counted(fixture, windows=1, window_fallbacks=1, sources=3)


def test_a_window_that_holds_a_reserved_piece_falls_back_and_is_counted(
    tmp_path: Path,
) -> None:
    def marker(line: tuple[str, ...]) -> Sequence[Hypothesis]:
        if is_window(line):
            return [hyp(SP + "Энэ", "<doc>", SEP, SP + "x", END)]
        return [hyp(*line[1:], END)]

    fixture = documents(tmp_path, script=marker)

    (result,) = fixture.engine.translate_documents([DOC], KHK, RUS, OFF)

    assert [t.text for t in result.translations] == SENTENCES
    counted(fixture, windows=1, window_fallbacks=1, sources=3, reserved_symbol=1)


def test_the_separator_is_text_the_model_may_write_in_a_window(
    tmp_path: Path,
) -> None:
    def other_words(line: tuple[str, ...]) -> Sequence[Hypothesis]:
        return [hyp(SP + "Энэ", SEP, SP + "ном", SEP, SP + "ус", END)]

    fixture = documents(tmp_path, script=other_words)

    (result,) = fixture.engine.translate_documents([DOC], KHK, RUS, OFF)

    # The window lines up, so nothing falls back and no sentence was decoded.
    assert [t.text for t in result.translations] == ["Энэ", "ном", "ус"]
    assert len(fixture.ct2.calls) == 1
    counted(fixture, windows=1)


def test_a_sentence_never_writes_the_separator(tmp_path: Path) -> None:
    fixture = documents(tmp_path, script=lambda line: [hyp(SP + "x", SEP, END)])

    (result,) = fixture.engine.translate([Row("hello", None, ENG)])

    assert (result.status, result.text) == ("no_output", "")


# -- the cut of a document


def test_a_small_budget_cuts_the_document_into_windows_and_single_sentences(
    tmp_path: Path,
) -> None:
    fixture = documents(tmp_path)
    options = DecodeOptions(max_source_pieces=7, script_mask="off")

    (result,) = fixture.engine.translate_documents([DOC], KHK, RUS, options)

    window, single = fixture.ct2.calls
    assert window.source[0][:2] == [TAG_RUS, "<doc>"]
    assert single.source == [[TAG_RUS, SP + "Энэ", SP + "гэр."]]
    assert [t.text for t in result.translations] == SENTENCES
    assert (result.stats.windows, result.stats.singles) == (1, 1)
    counted(fixture, windows=1, sources=1)


def test_a_sentence_that_does_not_fit_a_window_even_alone_is_translated_alone(
    tmp_path: Path,
) -> None:
    fixture = documents(tmp_path)
    # A window costs the tag and `<doc>` on top of its pieces: 2 + 2 is over 3.
    options = DecodeOptions(max_source_pieces=3, script_mask="off")

    (result,) = fixture.engine.translate_documents([DOC], KHK, RUS, options)

    assert [call.source[0][1] for call in fixture.ct2.calls] == [SP + "Энэ"] * 3
    assert [t.text for t in result.translations] == SENTENCES
    assert result.stats.oversize == 3
    counted(fixture, sources=3)


def test_a_sentence_over_the_source_limit_is_too_long_and_the_others_still_join(
    tmp_path: Path,
) -> None:
    fixture = documents(tmp_path)
    options = DecodeOptions(max_source_pieces=3, script_mask="off")

    (result,) = fixture.engine.translate_documents(
        ["Энэ. Энэ ном бол ус. Ус."], KHK, RUS, options
    )

    assert [t.status for t in result.translations] == ["ok", "too_long", "ok"]
    assert [t.text for t in result.translations] == ["Энэ.", "", "Ус."]
    assert result.text == "Энэ. Ус."
    counted(fixture, sources=3, too_long=1)


def test_a_document_with_no_sentence_is_an_empty_result(tmp_path: Path) -> None:
    fixture = documents(tmp_path)

    (result,) = fixture.engine.translate_documents(["  "], KHK, RUS, OFF)

    assert result.text == "" and result.translations == ()
    assert result.stats.sentences == 0
    assert fixture.ct2.calls == []


def test_several_documents_give_one_result_each_in_order(tmp_path: Path) -> None:
    fixture = documents(tmp_path)

    results = fixture.engine.translate_documents(
        ["Энэ ном. Энэ ус.", "", "Энэ гэр."], KHK, RUS, OFF
    )

    assert [r.text for r in results] == ["Энэ ном. Энэ ус.", "", "Энэ гэр."]
    assert fixture.engine.translate_documents([], KHK, RUS, OFF) == []


def test_the_translations_of_a_document_are_joined_by_a_space_and_by_nothing_for_han(
    tmp_path: Path,
) -> None:
    spaced = documents(tmp_path / "spaced")
    han = documents(
        tmp_path / "han", directions=("eng_Latn-zho_Hans", "zho_Hans-eng_Latn")
    )
    text = "Hello world. Hello book."

    (with_space,) = spaced.engine.translate_documents([text], ENG, KHK, OFF)
    (without,) = han.engine.translate_documents([text], ENG, "zho_Hans", OFF)

    assert with_space.text == "Hello world. Hello book."
    assert without.text == "Hello world.Hello book."
    assert [t.text for t in without.translations] == ["Hello world.", "Hello book."]


def test_the_source_of_a_window_is_normalised_before_it_is_cut(
    tmp_path: Path,
) -> None:
    fixture = documents(tmp_path)

    fixture.engine.translate_documents(["Энэ   ном.\n\nЭнэ ус."], KHK, RUS, OFF)

    assert "Энэ ном. <sep> Энэ ус." in fixture.processor.encoded
    assert all(
        "  " not in text and "\n" not in text for text in fixture.processor.encoded
    )


def test_a_sentence_that_fell_back_in_one_window_does_not_lend_its_text_to_another(
    tmp_path: Path,
) -> None:
    windows_seen: list[int] = []

    def script(line: tuple[str, ...]) -> Sequence[Hypothesis]:
        if not is_window(line):
            return [hyp(*line[1:], END)]
        windows_seen.append(1)
        if len(windows_seen) == 1:
            return [hyp(*line[2:])]  # cut: the first window falls back
        return [hyp(SP + "WIN1", SEP, SP + "WIN2", END)]

    fixture = documents(tmp_path, script=script)
    options = DecodeOptions(max_source_pieces=7, script_mask="off")

    (result,) = fixture.engine.translate_documents(
        ["Энэ ном. Энэ ном. Энэ ном. Энэ ном."], KHK, RUS, options
    )

    assert [t.text for t in result.translations] == [
        "Энэ ном.",
        "Энэ ном.",
        "WIN1",
        "WIN2",
    ]
    assert result.stats.fallbacks == (0,) and result.stats.aligned == 1


def test_a_document_is_normalised_before_it_is_cut_and_windowed(tmp_path: Path) -> None:
    fixture = documents(tmp_path)
    zero_width = chr(0x200B)

    fixture.engine.translate_documents([f"Энэ{zero_width} ном. Энэ ус."], KHK, RUS, OFF)

    assert fixture.processor.encoded
    assert all(zero_width not in text for text in fixture.processor.encoded)


# -- what a document model must be, and what a document must not hold


def test_a_sentence_model_refuses_documents_and_sends_nothing(tmp_path: Path) -> None:
    fixture = scripted_engine(tmp_path)

    with pytest.raises(ConfigError, match="document"):
        fixture.engine.translate_documents([DOC], KHK, RUS, OFF)

    assert fixture.ct2.calls == []


def test_a_model_with_the_tag_on_the_target_side_refuses_documents(
    tmp_path: Path,
) -> None:
    fixture = scripted_engine(tmp_path, document_level=True, placement="tgt")

    with pytest.raises(ConfigError, match="src"):
        fixture.engine.translate_documents([DOC], KHK, RUS, OFF)

    assert fixture.ct2.calls == []


def test_a_document_needs_its_source_language(tmp_path: Path) -> None:
    fixture = documents(tmp_path)

    with pytest.raises(ConfigError, match="src"):
        fixture.engine.translate_documents([DOC], None, ENG, OFF)

    assert fixture.ct2.calls == []


def test_the_rules_of_a_target_hold_for_a_document(tmp_path: Path) -> None:
    fixture = documents(tmp_path)

    with pytest.raises(ConfigError):
        fixture.engine.translate_documents([DOC], KHK, "deu_Latn", OFF)
    with pytest.raises(ConfigError):
        fixture.engine.translate_documents([DOC], KHK, KHK, OFF)
    with pytest.raises(ConfigError):
        fixture.engine.translate_documents([DOC], KHK, RUS, OFF, script="Mong")

    assert fixture.ct2.calls == []


def test_a_document_that_holds_the_separator_is_refused_without_quoting_it(
    tmp_path: Path,
) -> None:
    fixture = documents(tmp_path)
    secret = "private words"

    with pytest.raises(DataError) as excinfo:
        fixture.engine.translate_documents(
            [DOC, f"{secret} <sep> {secret}."], KHK, RUS, OFF
        )

    assert "document 1" in str(excinfo.value)
    assert "private" not in str(excinfo.value)
    assert fixture.ct2.calls == []
    assert fixture.engine.counters() == dict.fromkeys(ENGINE_COUNTERS, 0)


def test_a_document_with_a_lone_surrogate_is_refused_naming_it(tmp_path: Path) -> None:
    fixture = documents(tmp_path)

    with pytest.raises(DataError, match="document 1"):
        fixture.engine.translate_documents([DOC, "a" + chr(0xD800)], KHK, RUS, OFF)

    assert fixture.ct2.calls == []


# -- the script


def test_a_document_into_traditional_script_is_converted_sentence_by_sentence(
    tmp_path: Path,
) -> None:
    converter = FakeConverter(prefix=">")
    fixture = documents(tmp_path, converter=converter)

    (result,) = fixture.engine.translate_documents([DOC], RUS, KHK, OFF, script="Mong")

    assert fixture.ct2.calls[0].source[0][0] == TAG_KHK
    assert converter.calls == [(s, "Mong") for s in SENTENCES]
    assert [t.text for t in result.translations] == [f">{s}" for s in SENTENCES]
    assert result.text == " ".join(f">{s}" for s in SENTENCES)
    assert all(t.tgt == MON for t in result.translations)
    counters = fixture.engine.counters()
    assert counters["script_conversions"] == 3
    assert counters["converter_rule_fallbacks"] == 6
    assert counters["converter_invalid_outputs"] == 3


def test_a_sentence_that_fell_back_is_converted_once(tmp_path: Path) -> None:
    def cut(line: tuple[str, ...]) -> Sequence[Hypothesis]:
        return [hyp(*line[2:])] if is_window(line) else [hyp(*line[1:], END)]

    converter = FakeConverter(prefix=">")
    fixture = documents(tmp_path, script=cut, converter=converter)

    (result,) = fixture.engine.translate_documents([DOC], RUS, KHK, OFF, script="Mong")

    assert [t.text for t in result.translations] == [f">{s}" for s in SENTENCES]
    assert len(converter.calls) == 3


def pure_window(
    line: tuple[str, ...], suppressed: frozenset[str]
) -> Sequence[Hypothesis]:
    return [hyp(SP + "Энэ", SEP, SP + "ном", END)]


def test_the_separator_does_not_count_against_the_purity_of_a_window(
    tmp_path: Path,
) -> None:
    fixture = documents(tmp_path, answer=pure_window, script=None)

    (result,) = fixture.engine.translate_documents(
        ["Hello world. Hello book."], ENG, KHK, DecodeOptions(script_mask="retry")
    )

    assert len(fixture.ct2.calls) == 1
    assert [t.text for t in result.translations] == ["Энэ", "ном"]
    counted(fixture, windows=1)


def test_an_impure_window_is_decoded_again_with_the_list_that_lets_the_separator_through(
    tmp_path: Path,
) -> None:
    def leaky(
        line: tuple[str, ...], suppressed: frozenset[str]
    ) -> Sequence[Hypothesis]:
        if LATIN in suppressed:
            return [hyp(SP + "Энэ", SEP, SP + "ном", END)]
        return [hyp(LATIN, SEP, LATIN, END)]

    fixture = documents(tmp_path, answer=leaky, script=None)

    (result,) = fixture.engine.translate_documents(
        ["Hello world. Hello book."], ENG, KHK, DecodeOptions(script_mask="retry")
    )

    first, second = fixture.ct2.calls
    assert first.suppress_sequences == [
        ["<unk>"],
        ["<blank>"],
        ["<s>"],
        ["averyunlikelytoken0"],
    ]
    assert [SEP] not in second.suppress_sequences  # type: ignore[operator]
    assert ["<doc>"] in second.suppress_sequences  # type: ignore[operator]
    assert [LATIN] in second.suppress_sequences  # type: ignore[operator]
    assert [t.text for t in result.translations] == ["Энэ", "ном"]
    counted(fixture, windows=1, script_retries=1, script_retry_improved=1)


def test_the_list_of_a_window_is_a_list_of_its_own(tmp_path: Path) -> None:
    fixture = documents(tmp_path, answer=pure_window, script=None)

    fixture.engine.translate_documents(
        ["Hello world. Hello book."], ENG, KHK, DecodeOptions(script_mask="always")
    )
    fixture.engine.translate(
        [Row("hello", None, KHK)], DecodeOptions(script_mask="always")
    )

    window, sentence = fixture.ct2.calls
    assert [SEP] not in window.suppress_sequences  # type: ignore[operator]
    assert [SEP] in sentence.suppress_sequences  # type: ignore[operator]


def test_a_separator_hidden_by_a_zero_width_character_is_still_refused(
    tmp_path: Path,
) -> None:
    fixture = documents(tmp_path)
    hidden = "Энэ ном. <se" + chr(0x200B) + "p> Энэ ус."

    with pytest.raises(DataError, match="document 0"):
        fixture.engine.translate_documents([hidden], KHK, RUS, OFF)

    assert fixture.ct2.calls == []


def test_a_document_that_is_not_text_is_a_type_error(tmp_path: Path) -> None:
    fixture = documents(tmp_path)

    with pytest.raises(TypeError, match="document 1"):
        fixture.engine.translate_documents([DOC, 5], KHK, RUS, OFF)  # type: ignore[list-item]

    assert fixture.ct2.calls == []
