"""The teacher as a `Translator` (D10): what a call hands CTranslate2, what the hypotheses go
through before they are candidates, the sources that are not sent, the refusals of a call
and the counters.

No model runs: the export is the fabricated one of `tests.support.distill_exports`, the
CTranslate2 translator is `ScriptedCt2` and the SentencePiece model `FakeProcessor`;
`ctranslate2` itself is made unimportable for every test, so a test that reached the
library would fail instead of loading it. Nothing touches a GPU."""

from __future__ import annotations

import sys

import pytest

from rinchen.errors import ConfigError, DataError, UpstreamError
from rinchen.train.distill_teacher import TeacherOptions, TeacherTranslator
from tests.support.distill_teachers import (
    END,
    ENG,
    KHK,
    SP,
    TAG,
    config_bytes,
    export_with_files,
    hyp,
    scripted_teacher,
    tag_processor,
)
from tests.support.fake_ct2 import ScriptedCt2

COUNTER_KEYS = (
    "sources",
    "source_too_long",
    "empty_source",
    "truncated",
    "line_break",
    "reserved_symbol",
)
# The characters `str.splitlines` splits at, by code point.
LINE_BREAKS = [0x0A, 0x0D, 0x0B, 0x0C, 0x1C, 0x1D, 0x1E, 0x85, 0x2028, 0x2029]


@pytest.fixture(autouse=True)
def real_ctranslate2_is_unimportable(monkeypatch):
    monkeypatch.setitem(sys.modules, "ctranslate2", None)


# -- what a call hands CTranslate2


def test_a_source_tag_call_has_tagged_sources_no_prefix_and_the_documented_options(
    tmp_path,
):
    script = {
        (TAG, SP + "hello", SP + "world"): [hyp(SP + "sain", SP + "uu", END)],
        (TAG, SP + "good", SP + "night"): [hyp(SP + "ber", END)],
    }
    teacher, ct2 = scripted_teacher(tmp_path, script)

    result = teacher.translate(["hello world", "good night"], ENG, KHK)

    assert result == [["sain uu"], ["ber"]]
    (call,) = ct2.calls
    assert call.source == [
        [TAG, SP + "hello", SP + "world"],
        [TAG, SP + "good", SP + "night"],
    ]
    assert call.target_prefix is None
    assert call.options == {
        "beam_size": 4,
        "num_hypotheses": 1,
        "max_batch_size": 64,
        "max_decoding_length": 256,
        "return_end_token": True,
        "return_scores": False,
    }


def test_a_target_tag_call_has_untagged_sources_and_one_forced_prefix_each(tmp_path):
    script = {
        (SP + "hello", SP + "world"): [hyp(TAG, SP + "sain", END)],
        (SP + "good", SP + "night"): [hyp(TAG, SP + "ber", END)],
    }
    teacher, ct2 = scripted_teacher(tmp_path, script, placement="tgt")

    result = teacher.translate(["hello world", "good night"], ENG, KHK)

    assert result == [["sain"], ["ber"]]
    (call,) = ct2.calls
    assert call.source == [[SP + "hello", SP + "world"], [SP + "good", SP + "night"]]
    assert call.target_prefix == [[TAG], [TAG]]
    assert call.options["return_end_token"] is True


def test_the_tag_is_the_one_of_the_target_language(tmp_path):
    script = {("<2eng_Latn>", SP + "a"): [hyp(SP + "x", END)]}
    teacher, ct2 = scripted_teacher(tmp_path, script)

    teacher.translate(["a"], KHK, ENG)

    assert ct2.calls[0].source == [["<2eng_Latn>", SP + "a"]]


def test_the_options_the_caller_chose_reach_the_library(tmp_path):
    options = TeacherOptions(
        beam_size=6, n=3, max_decoding_length=99, batch_size=7, max_source_pieces=50
    )
    script = {(TAG, SP + "a"): [hyp(SP + "x", END)]}
    teacher, ct2 = scripted_teacher(tmp_path, script, options=options)

    teacher.translate(["a"], ENG, KHK)

    assert ct2.calls[0].options == {
        "beam_size": 6,
        "num_hypotheses": 3,
        "max_batch_size": 7,
        "max_decoding_length": 99,
        "return_end_token": True,
        "return_scores": False,
    }


def test_the_placement_is_the_one_the_export_records(tmp_path):
    script = {(SP + "a",): [hyp(TAG, SP + "x", END)]}
    teacher, ct2 = scripted_teacher(tmp_path, script, placement="tgt")

    teacher.translate(["a"], ENG, KHK)

    assert ct2.calls[0].target_prefix == [[TAG]]


def test_a_call_carries_every_text_in_one_batch(tmp_path):
    texts = [f"w{i}" for i in range(5)]
    script = {(TAG, SP + text): [hyp(SP + "x", END)] for text in texts}
    teacher, ct2 = scripted_teacher(
        tmp_path, script, options=TeacherOptions(batch_size=2)
    )

    teacher.translate(texts, ENG, KHK)

    assert len(ct2.calls) == 1
    assert len(ct2.calls[0].source) == 5


# -- what a hypothesis becomes


def test_a_source_tag_hypothesis_loses_its_end_token_and_is_decoded(tmp_path):
    script = {(TAG, SP + "a"): [hyp(SP + "hello", SP + "world", END)]}
    teacher, _ = scripted_teacher(tmp_path, script)

    assert teacher.translate(["a"], ENG, KHK) == [["hello world"]]


def test_a_target_tag_hypothesis_loses_its_forced_prefix_and_its_end_token(tmp_path):
    script = {(SP + "a",): [hyp(TAG, SP + "hello", END)]}
    teacher, _ = scripted_teacher(tmp_path, script, placement="tgt")

    assert teacher.translate(["a"], ENG, KHK) == [["hello"]]


def test_the_end_token_is_the_one_in_the_converters_config(tmp_path, monkeypatch):
    export = export_with_files(
        tmp_path, monkeypatch, **{"config.json": config_bytes(eos_token="<end>")}
    )
    script = {
        (TAG, SP + "a"): [hyp(SP + "kept", "<end>"), hyp(SP + "cut", END)],
    }
    teacher = TeacherTranslator(
        export,
        TeacherOptions(n=2),
        translator=ScriptedCt2(script),
        processor=tag_processor(),
    )

    assert teacher.translate(["a"], ENG, KHK) == [["kept"]]
    assert teacher.counters()["truncated"] == 1


def test_a_target_tag_hypothesis_that_does_not_start_with_the_tag_is_an_upstream_error(
    tmp_path,
):
    script = {
        (SP + "a",): [hyp(TAG, SP + "fine", END)],
        (SP + "b",): [hyp(SP + "leaky", SP + "secret", END)],
    }
    teacher, _ = scripted_teacher(tmp_path, script, placement="tgt")

    with pytest.raises(UpstreamError) as excinfo:
        teacher.translate(["a", "b"], ENG, KHK)

    message = str(excinfo.value)
    assert "source 1" in message
    assert TAG in message
    assert "leaky" not in message
    assert "secret" not in message


def test_the_index_in_an_upstream_error_is_the_position_in_the_texts_given(tmp_path):
    script = {(SP + "b",): [hyp(SP + "wrong", END)]}
    teacher, _ = scripted_teacher(
        tmp_path,
        script,
        placement="tgt",
        options=TeacherOptions(max_source_pieces=2),
    )

    with pytest.raises(UpstreamError, match="source 2"):
        teacher.translate(["p q r", "", "b"], ENG, KHK)


def test_an_empty_target_tag_hypothesis_is_an_upstream_error_naming_index_0(tmp_path):
    script = {(SP + "a",): [([], -1.0)]}
    teacher, _ = scripted_teacher(tmp_path, script, placement="tgt")

    with pytest.raises(UpstreamError, match="source 0"):
        teacher.translate(["a"], ENG, KHK)


def test_a_cut_hypothesis_is_dropped_and_counted_and_the_source_has_none(tmp_path):
    script = {(TAG, SP + "a"): [hyp(SP + "half", SP + "a")]}
    teacher, _ = scripted_teacher(tmp_path, script)

    assert teacher.translate(["a"], ENG, KHK) == [[]]
    assert teacher.counters()["truncated"] == 1
    assert teacher.counters()["line_break"] == 0


def test_a_cut_target_tag_hypothesis_is_dropped_too(tmp_path):
    script = {(SP + "a",): [hyp(TAG, SP + "half")]}
    teacher, _ = scripted_teacher(tmp_path, script, placement="tgt")

    assert teacher.translate(["a"], ENG, KHK) == [[]]
    assert teacher.counters()["truncated"] == 1


def test_an_empty_source_tag_hypothesis_has_no_end_token_and_counts_as_cut(tmp_path):
    script = {(TAG, SP + "a"): [([], -1.0)]}
    teacher, _ = scripted_teacher(tmp_path, script)

    assert teacher.translate(["a"], ENG, KHK) == [[]]
    assert teacher.counters()["truncated"] == 1


@pytest.mark.parametrize("code", LINE_BREAKS, ids=[f"U+{c:04X}" for c in LINE_BREAKS])
def test_a_hypothesis_with_a_line_break_is_dropped_and_counted(tmp_path, code):
    script = {
        (TAG, SP + "a"): [
            hyp(SP + "one", chr(code), SP + "two", END),
            hyp(SP + "plain", END),
        ]
    }
    teacher, _ = scripted_teacher(tmp_path, script, options=TeacherOptions(n=2))

    assert teacher.translate(["a"], ENG, KHK) == [["plain"]]
    assert teacher.counters()["line_break"] == 1
    assert teacher.counters()["truncated"] == 0


def test_a_line_break_at_the_edge_is_dropped_and_not_stripped_away(tmp_path):
    script = {(TAG, SP + "a"): [hyp(SP + "one", chr(0x0A), END)]}
    teacher, _ = scripted_teacher(tmp_path, script)

    assert teacher.translate(["a"], ENG, KHK) == [[]]
    assert teacher.counters()["line_break"] == 1


def test_a_cut_hypothesis_and_a_line_break_are_counted_under_their_own_reason(tmp_path):
    script = {
        (TAG, SP + "a"): [
            hyp(SP + "cut"),
            hyp(SP + "one", chr(0x0A), SP + "two", END),
            hyp(SP + "good", END),
        ]
    }
    teacher, _ = scripted_teacher(tmp_path, script, options=TeacherOptions(n=3))

    assert teacher.translate(["a"], ENG, KHK) == [["good"]]
    counters = teacher.counters()
    assert (counters["truncated"], counters["line_break"]) == (1, 1)


def test_candidates_are_stripped_distinct_and_never_blank(tmp_path):
    script = {
        (TAG, SP + "a"): [
            hyp(SP + "b", END),
            hyp(SP + "a", END),
            hyp(SP + "b", END),
            hyp(SP, END),
        ]
    }
    teacher, _ = scripted_teacher(
        tmp_path, script, options=TeacherOptions(beam_size=4, n=4)
    )

    assert teacher.translate(["a"], ENG, KHK) == [["b", "a"]]


def test_n_hypotheses_come_back_in_the_order_of_the_beam(tmp_path):
    script = {
        (TAG, SP + "a"): [
            hyp(SP + "first", END),
            hyp(SP + "second", END),
            hyp(SP + "third", END),
        ],
        (TAG, SP + "b"): [hyp(SP + "only", END)],
    }
    teacher, _ = scripted_teacher(tmp_path, script, options=TeacherOptions(n=3))

    assert teacher.translate(["a", "b"], ENG, KHK) == [
        ["first", "second", "third"],
        ["only"],
    ]


def test_the_results_come_back_in_the_order_of_the_texts(tmp_path):
    script = {
        (TAG, SP + "x"): [hyp(SP + "one", END)],
        (TAG, SP + "y"): [hyp(SP + "two", END)],
        (TAG, SP + "z"): [hyp(SP + "three", END)],
    }
    teacher, _ = scripted_teacher(tmp_path, script)

    assert teacher.translate(["z", "x", "y"], ENG, KHK) == [
        ["three"],
        ["one"],
        ["two"],
    ]


def test_the_same_call_twice_gives_the_same_lists(tmp_path):
    script = {(TAG, SP + "a"): [hyp(SP + "x", END), hyp(SP + "y", END)]}
    teacher, _ = scripted_teacher(tmp_path, script, options=TeacherOptions(n=2))

    first = teacher.translate(["a"], ENG, KHK)
    second = teacher.translate(["a"], ENG, KHK)

    assert first == second == [["x", "y"]]


# -- sources that are not sent


def test_a_source_over_the_piece_limit_is_not_sent_and_the_others_keep_their_place(
    tmp_path,
):
    script = {
        (TAG, SP + "a", SP + "b"): [hyp(SP + "one", END)],
        (TAG, SP + "c", SP + "d"): [hyp(SP + "two", END)],
    }
    teacher, ct2 = scripted_teacher(
        tmp_path, script, options=TeacherOptions(max_source_pieces=4)
    )

    result = teacher.translate(["a b", "p q r s t", "c d"], ENG, KHK)

    assert result == [["one"], [], ["two"]]
    (call,) = ct2.calls
    assert call.source == [[TAG, SP + "a", SP + "b"], [TAG, SP + "c", SP + "d"]]
    assert teacher.counters()["source_too_long"] == 1


def test_a_source_of_exactly_the_limit_is_sent(tmp_path):
    script = {(TAG, SP + "p", SP + "q", SP + "r", SP + "s"): [hyp(SP + "x", END)]}
    teacher, ct2 = scripted_teacher(
        tmp_path, script, options=TeacherOptions(max_source_pieces=4)
    )

    assert teacher.translate(["p q r s"], ENG, KHK) == [["x"]]
    assert teacher.counters()["source_too_long"] == 0
    assert len(ct2.calls) == 1


def test_the_forced_prefixes_follow_the_sources_that_are_sent(tmp_path):
    script = {(SP + "a",): [hyp(TAG, SP + "x", END)]}
    teacher, ct2 = scripted_teacher(
        tmp_path,
        script,
        placement="tgt",
        options=TeacherOptions(max_source_pieces=2),
    )

    assert teacher.translate(["p q r", "a", "p q r"], ENG, KHK) == [[], ["x"], []]
    assert ct2.calls[0].target_prefix == [[TAG]]


def test_when_every_source_is_too_long_the_library_is_not_called(tmp_path):
    teacher, ct2 = scripted_teacher(
        tmp_path, {}, options=TeacherOptions(max_source_pieces=1)
    )

    assert teacher.translate(["a b", "c d e"], ENG, KHK) == [[], []]
    assert ct2.calls == []
    assert teacher.counters()["source_too_long"] == 2
    assert teacher.counters()["sources"] == 2


def test_a_text_without_a_piece_is_not_sent_because_a_bare_tag_gets_a_made_up_sentence(
    tmp_path,
):
    script = {(TAG, SP + "a"): [hyp(SP + "x", END)]}
    teacher, ct2 = scripted_teacher(tmp_path, script)

    assert teacher.translate(["   ", "a", ""], ENG, KHK) == [[], ["x"], []]
    assert ct2.calls[0].source == [[TAG, SP + "a"]]
    assert teacher.counters()["empty_source"] == 2


def test_no_texts_give_no_lists_and_no_call(tmp_path):
    teacher, ct2 = scripted_teacher(tmp_path, {})

    assert teacher.translate([], ENG, KHK) == []
    assert ct2.calls == []
    assert teacher.counters() == dict.fromkeys(COUNTER_KEYS, 0)


# -- refusals of a call


def test_a_result_count_other_than_the_number_sent_is_an_upstream_error(tmp_path):
    script = {
        (TAG, SP + "a"): [hyp(SP + "x", END)],
        (TAG, SP + "b"): [hyp(SP + "y", END)],
    }
    teacher, ct2 = scripted_teacher(tmp_path, script)

    ct2.result_offset = -1
    with pytest.raises(UpstreamError, match=r"1 results for 2"):
        teacher.translate(["a", "b"], ENG, KHK)
    ct2.result_offset = 1
    with pytest.raises(UpstreamError, match=r"3 results for 2"):
        teacher.translate(["a", "b"], ENG, KHK)


@pytest.mark.parametrize("error", [RuntimeError("out of memory"), ValueError("bad")])
def test_a_library_failure_is_an_upstream_error_that_counts_the_sources(
    tmp_path, error
):
    teacher, ct2 = scripted_teacher(tmp_path, {})
    ct2.error = error

    with pytest.raises(UpstreamError, match="3 sources") as excinfo:
        teacher.translate(["a", "b", "c"], ENG, KHK)

    assert excinfo.value.__cause__ is error


def test_a_failed_call_adds_nothing_to_the_counters(tmp_path):
    teacher, ct2 = scripted_teacher(
        tmp_path, {}, options=TeacherOptions(max_source_pieces=1)
    )
    ct2.error = RuntimeError("boom")

    with pytest.raises(UpstreamError):
        teacher.translate(["a", "b c"], ENG, KHK)

    assert teacher.counters() == dict.fromkeys(COUNTER_KEYS, 0)


def test_a_lone_surrogate_is_a_data_error_naming_its_index_before_any_encoding(
    tmp_path,
):
    processor = tag_processor()
    teacher, ct2 = scripted_teacher(tmp_path, {}, processor=processor)

    with pytest.raises(DataError, match="text 1") as excinfo:
        teacher.translate(["fine", "bad" + chr(0xD800)], ENG, KHK)

    assert chr(0xD800) not in str(excinfo.value)
    assert processor.encoded == []
    assert ct2.calls == []


@pytest.mark.parametrize(
    ("src_lang", "tgt_lang", "named"),
    [("fra_Latn", KHK, "fra_Latn"), (ENG, "xx_Latn", "xx_Latn"), ("", KHK, "''")],
)
@pytest.mark.parametrize("texts", [["a"], []])
def test_a_language_outside_the_registry_is_a_config_error_even_without_texts(
    tmp_path, src_lang, tgt_lang, named, texts
):
    teacher, ct2 = scripted_teacher(tmp_path, {})

    with pytest.raises(ConfigError, match="registry") as excinfo:
        teacher.translate(texts, src_lang, tgt_lang)

    assert named in str(excinfo.value)
    assert ct2.calls == []


def test_a_target_whose_tag_is_not_in_the_vocabulary_is_refused_not_translated(
    tmp_path,
):
    teacher, ct2 = scripted_teacher(
        tmp_path, {}, tokens=("<unk>", "<blank>", "<s>", END, "a", "b")
    )

    with pytest.raises(ConfigError, match=TAG):
        teacher.translate(["a"], ENG, KHK)
    with pytest.raises(ConfigError, match=TAG):
        teacher.translate([], ENG, KHK)
    assert ct2.calls == []


# -- the counters


def test_the_counters_start_at_zero_with_exactly_these_keys(tmp_path):
    teacher, _ = scripted_teacher(tmp_path, {})

    assert teacher.counters() == {
        "sources": 0,
        "source_too_long": 0,
        "empty_source": 0,
        "truncated": 0,
        "line_break": 0,
        "reserved_symbol": 0,
    }


def test_the_counters_accumulate_over_calls_and_count_every_text_given(tmp_path):
    script = {
        (TAG, SP + "a"): [hyp(SP + "x", END)],
        (TAG, SP + "b"): [hyp(SP + "cut")],
    }
    teacher, _ = scripted_teacher(
        tmp_path, script, options=TeacherOptions(max_source_pieces=2)
    )

    teacher.translate(["a", "p q r"], ENG, KHK)
    teacher.translate(["b", ""], ENG, KHK)

    assert teacher.counters() == {
        "sources": 4,
        "source_too_long": 1,
        "empty_source": 1,
        "truncated": 1,
        "line_break": 0,
        "reserved_symbol": 0,
    }


def test_the_counters_returned_are_a_copy(tmp_path):
    teacher, _ = scripted_teacher(tmp_path, {})

    teacher.counters()["sources"] = 99

    assert teacher.counters()["sources"] == 0


def test_two_translators_do_not_share_counters(tmp_path):
    script = {(TAG, SP + "a"): [hyp(SP + "x", END)]}
    first, _ = scripted_teacher(tmp_path / "one", script)
    second, _ = scripted_teacher(tmp_path / "two", script)

    first.translate(["a"], ENG, KHK)

    assert first.counters()["sources"] == 1
    assert second.counters()["sources"] == 0
