"""The teacher drops, and counts, a hypothesis that holds a piece no translation holds (D10).

SentencePiece decodes the unknown piece to the text of U+2047 and every reserved symbol of the
project (a direction tag, `<bt>`, `<doc>`, a numbered placeholder) to its own letters, so such
a hypothesis passes the script and length rules of the selection and a student would learn to
emit it. The teacher counts it under `reserved_symbol` and gives no candidate for it. The pieces
that stand for bytes (a model with byte fallback) are ordinary and stay.

The CTranslate2 library is `ScriptedCt2` (nothing is loaded); the processor is `FakeProcessor`,
or a real SentencePiece model trained here, with byte fallback and the project's reserved
symbols, for the tests that need the library's own decoding.
"""

from __future__ import annotations

import sys
from collections.abc import Callable, Sequence
from pathlib import Path

import pytest
import sentencepiece

from rinchen.errors import UpstreamError
from rinchen.spm.symbols import user_defined_symbols
from rinchen.train.distill_teacher import COUNTER_KEYS, TeacherOptions
from tests.support.distill_teachers import (
    END,
    ENG,
    KHK,
    SP,
    TAG,
    hyp,
    scripted_teacher,
)
from tests.support.fake_ct2 import FakeProcessor, Hypothesis
from tests.support.step_dirs import SOURCE_VOCABULARY
from tests.support.tiny_parallel import build_toy

PLACEHOLDER = chr(0x27E6) + "7" + chr(0x27E7)
UNKNOWN_SURFACE = chr(0x2047)


@pytest.fixture(autouse=True)
def real_ctranslate2_is_unimportable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "ctranslate2", None)


class OtherUnknownPiece(FakeProcessor):
    """A processor whose unknown piece is not spelled `<unk>`."""

    def id_to_piece(self, piece_id: int) -> str:
        return (
            "<unknown-piece>"
            if piece_id == self.unk_id()
            else super().id_to_piece(piece_id)
        )


# ---- the counter ------------------------------------------------------------------------


def test_the_new_counter_is_one_of_the_keys_in_the_order_the_records_list_them() -> (
    None
):
    assert COUNTER_KEYS == (
        "sources",
        "source_too_long",
        "empty_source",
        "truncated",
        "line_break",
        "reserved_symbol",
    )


def test_a_hypothesis_with_the_unknown_piece_is_dropped_and_counted(
    tmp_path: Path,
) -> None:
    script = {
        (TAG, SP + "a"): [
            hyp(SP + "Ном", "<unk>", SP + "ба", END),
            hyp(SP + "good", END),
        ]
    }
    teacher, _ = scripted_teacher(tmp_path, script, options=TeacherOptions(n=2))

    assert teacher.translate(["a"], ENG, KHK) == [["good"]]
    counters = teacher.counters()
    assert counters["reserved_symbol"] == 1
    assert (counters["truncated"], counters["line_break"]) == (0, 0)


def test_the_unknown_piece_is_the_one_the_processor_names_not_a_spelling(
    tmp_path: Path,
) -> None:
    script = {
        (TAG, SP + "a"): [
            hyp(SP + "x", "<unknown-piece>", END),
            hyp(SP + "y", END),
        ]
    }
    teacher, _ = scripted_teacher(
        tmp_path,
        script,
        options=TeacherOptions(beam_size=2, n=2),
        processor=OtherUnknownPiece([TAG, "<2eng_Latn>", "<2rus_Cyrl>"]),
    )

    assert teacher.translate(["a"], ENG, KHK) == [["y"]]
    assert teacher.counters()["reserved_symbol"] == 1


@pytest.mark.parametrize("symbol", user_defined_symbols())
def test_every_reserved_symbol_of_the_project_makes_a_hypothesis_unusable(
    tmp_path: Path, symbol: str
) -> None:
    script = {(TAG, SP + "a"): [hyp(SP + "one", symbol, SP + "two", END)]}
    teacher, _ = scripted_teacher(tmp_path, script)

    assert teacher.translate(["a"], ENG, KHK) == [[]]
    assert teacher.counters()["reserved_symbol"] == 1


@pytest.mark.parametrize("position", [0, 1, 2], ids=["first", "middle", "last"])
def test_a_reserved_symbol_is_found_wherever_it_stands_in_the_hypothesis(
    tmp_path: Path, position: int
) -> None:
    pieces = [SP + "one", SP + "two"]
    pieces.insert(position, "<bt>")
    script = {(TAG, SP + "a"): [hyp(*pieces, END)]}
    teacher, _ = scripted_teacher(tmp_path, script)

    assert teacher.translate(["a"], ENG, KHK) == [[]]
    assert teacher.counters()["reserved_symbol"] == 1


def test_a_tag_of_another_direction_leaking_into_the_text_is_counted(
    tmp_path: Path,
) -> None:
    script = {(TAG, SP + "a"): [hyp(SP + "one", "<2eng_Latn>", END)]}
    teacher, _ = scripted_teacher(tmp_path, script)

    assert teacher.translate(["a"], ENG, KHK) == [[]]
    assert teacher.counters()["reserved_symbol"] == 1


def test_the_end_token_in_the_middle_of_a_hypothesis_is_counted(tmp_path: Path) -> None:
    script = {(TAG, SP + "a"): [hyp(SP + "one", END, SP + "two", END)]}
    teacher, _ = scripted_teacher(tmp_path, script)

    assert teacher.translate(["a"], ENG, KHK) == [[]]
    assert teacher.counters()["reserved_symbol"] == 1


def test_the_forced_tag_of_a_target_tag_export_is_not_a_leak_but_a_second_one_is(
    tmp_path: Path,
) -> None:
    script = {
        (SP + "a",): [
            hyp(TAG, SP + "fine", END),
            hyp(TAG, SP + "one", TAG, SP + "two", END),
        ]
    }
    teacher, _ = scripted_teacher(
        tmp_path, script, placement="tgt", options=TeacherOptions(beam_size=2, n=2)
    )

    assert teacher.translate(["a"], ENG, KHK) == [["fine"]]
    assert teacher.counters()["reserved_symbol"] == 1


def test_a_hypothesis_is_counted_under_one_reason_only(tmp_path: Path) -> None:
    script = {
        (TAG, SP + "a"): [
            hyp("<unk>", SP + "cut"),
            hyp(SP + "both", "<unk>", chr(0x0A), END),
            hyp(SP + "break", chr(0x0A), END),
            hyp(SP + "ok", END),
        ]
    }
    teacher, _ = scripted_teacher(
        tmp_path, script, options=TeacherOptions(beam_size=4, n=4)
    )

    assert teacher.translate(["a"], ENG, KHK) == [["ok"]]
    counters = teacher.counters()
    assert (
        counters["truncated"],
        counters["reserved_symbol"],
        counters["line_break"],
    ) == (
        1,
        1,
        1,
    )


def test_the_counter_adds_up_over_calls_and_a_failed_call_adds_nothing(
    tmp_path: Path,
) -> None:
    script = {
        (TAG, SP + "a"): [hyp(SP + "x", "<unk>", END)],
        (TAG, SP + "b"): [hyp(SP + "y", END)],
    }
    teacher, ct2 = scripted_teacher(tmp_path, script)

    teacher.translate(["a", "b"], ENG, KHK)
    teacher.translate(["a"], ENG, KHK)
    ct2.error = RuntimeError("out of memory")
    with pytest.raises(UpstreamError, match="out of memory"):
        teacher.translate(["a"], ENG, KHK)

    assert teacher.counters()["reserved_symbol"] == 2


def test_ordinary_pieces_that_look_like_markup_are_kept(tmp_path: Path) -> None:
    # Only the reserved symbols and the unknown piece are refused, not anything in angle
    # brackets.
    script = {(TAG, SP + "a"): [hyp(SP + "a", "<b>", SP + "c", END)]}
    teacher, _ = scripted_teacher(tmp_path, script)

    assert teacher.translate(["a"], ENG, KHK) == [["a<b> c"]]
    assert teacher.counters()["reserved_symbol"] == 0


# ---- the entries of the vocabulary that are no piece of the model ----------------------


FOREIGN_TOKENS = (
    "<unk>",
    "<blank>",
    "<s>",
    END,
    TAG,
    "<2eng_Latn>",
    "<2rus_Cyrl>",
    "held",
    "averyunlikelytoken0",
)


@pytest.mark.parametrize("foreign", ["<blank>", "averyunlikelytoken0"])
def test_a_vocabulary_entry_the_model_does_not_hold_is_dropped_and_counted(
    tmp_path: Path, foreign: str
) -> None:
    # Eole adds `<blank>` to the vocabulary and pads it with made-up tokens; the model could
    # emit one, and the processor would write it as its letters.
    script = {
        (TAG, SP + "a"): [
            hyp(SP + "x", foreign, END),
            hyp("held", END),
        ]
    }
    teacher, _ = scripted_teacher(
        tmp_path,
        script,
        options=TeacherOptions(beam_size=2, n=2),
        tokens=FOREIGN_TOKENS,
        processor=FakeProcessor([TAG, "<2eng_Latn>", "<2rus_Cyrl>", "held"]),
    )

    assert teacher.translate(["a"], ENG, KHK) == [["held"]]
    assert teacher.counters()["reserved_symbol"] == 1


def test_a_vocabulary_entry_the_model_holds_is_kept(tmp_path: Path) -> None:
    script = {(TAG, SP + "a"): [hyp(SP + "x", "held", END)]}
    teacher, _ = scripted_teacher(
        tmp_path,
        script,
        tokens=FOREIGN_TOKENS,
        processor=FakeProcessor([TAG, "<2eng_Latn>", "<2rus_Cyrl>", "held"]),
    )

    assert teacher.translate(["a"], ENG, KHK) == [["xheld"]]
    assert teacher.counters()["reserved_symbol"] == 0


def test_a_piece_outside_the_vocabulary_is_not_the_teachers_to_judge(
    tmp_path: Path,
) -> None:
    # CTranslate2 cannot emit it; what is decided is by the vocabulary the export holds.
    script = {(TAG, SP + "a"): [hyp(SP + "x", "stranger", END)]}
    teacher, _ = scripted_teacher(tmp_path, script, tokens=FOREIGN_TOKENS)

    assert teacher.translate(["a"], ENG, KHK) == [["xstranger"]]


# ---- against a real SentencePiece model with byte fallback ------------------------------


@pytest.fixture(scope="module")
def real_model(
    tmp_path_factory: pytest.TempPathFactory,
) -> sentencepiece.SentencePieceProcessor:
    directory = tmp_path_factory.mktemp("reserved_spm")
    text = directory / "text.txt"
    text.write_text(
        "".join(f"{row['src']}\n{row['tgt']}\n" for row in build_toy().train[:600]),
        encoding="utf-8",
    )
    sentencepiece.SentencePieceTrainer.train(
        input=str(text),
        model_prefix=str(directory / "byte"),
        vocab_size=500,
        user_defined_symbols=user_defined_symbols(),
        byte_fallback=True,
        hard_vocab_limit=False,
        minloglevel=2,
    )
    return sentencepiece.SentencePieceProcessor(
        model_file=str(directory / "byte.model")
    )


def answering(
    *hypotheses: Hypothesis,
) -> Callable[[tuple[str, ...]], Sequence[Hypothesis]]:
    """A script that gives every source the same hypotheses (the real model cuts a text into
    pieces a test does not want to spell)."""
    return lambda line: list(hypotheses)


def pieces_of(model: sentencepiece.SentencePieceProcessor, text: str) -> list[str]:
    return model.encode(text, out_type=str)


def test_the_library_decodes_the_unknown_piece_to_a_text_that_passes_for_a_sentence(
    real_model: sentencepiece.SentencePieceProcessor,
) -> None:
    # The premise of the rule, measured on the library: nothing in the decoded text says
    # a piece was unknown, but the surface mark.
    decoded = real_model.decode(
        [*pieces_of(real_model, "Сайн"), "<unk>", *pieces_of(real_model, "уу")]
    )

    assert UNKNOWN_SURFACE in decoded


def test_a_real_model_decodes_a_leaked_tag_to_literal_text_and_the_teacher_drops_it(
    tmp_path: Path, real_model: sentencepiece.SentencePieceProcessor
) -> None:
    leaked = [
        *pieces_of(real_model, "Сайн"),
        "<2eng_Latn>",
        *pieces_of(real_model, "уу"),
    ]
    assert "<2eng_Latn>" in real_model.decode(leaked)
    good = pieces_of(real_model, "Сайн уу")
    teacher, _ = scripted_teacher(
        tmp_path,
        answering(hyp(*leaked, END), hyp(*good, END)),
        options=TeacherOptions(beam_size=2, n=2),
        processor=real_model,
    )

    assert teacher.translate(["a"], ENG, KHK) == [[real_model.decode(good)]]
    assert teacher.counters()["reserved_symbol"] == 1


@pytest.mark.parametrize("foreign", ["<blank>", "averyunlikelytoken0"])
def test_a_real_model_writes_a_vocabulary_padding_token_as_letters_and_the_teacher_drops_it(
    tmp_path: Path, real_model: sentencepiece.SentencePieceProcessor, foreign: str
) -> None:
    pieces = [*pieces_of(real_model, "Сайн"), foreign, *pieces_of(real_model, "уу")]
    assert foreign in real_model.decode(pieces)
    tokens = (*SOURCE_VOCABULARY, "averyunlikelytoken0")
    teacher, _ = scripted_teacher(
        tmp_path, answering(hyp(*pieces, END)), processor=real_model, tokens=tokens
    )

    assert teacher.translate(["a"], ENG, KHK) == [[]]
    assert teacher.counters()["reserved_symbol"] == 1


@pytest.mark.parametrize("marker", ["<unk>", "<bt>", "<ft>", "<doc>", PLACEHOLDER])
def test_a_real_model_marker_inside_a_hypothesis_is_dropped(
    tmp_path: Path, real_model: sentencepiece.SentencePieceProcessor, marker: str
) -> None:
    pieces = [*pieces_of(real_model, "Сайн"), marker, *pieces_of(real_model, "уу")]
    teacher, _ = scripted_teacher(
        tmp_path, answering(hyp(*pieces, END)), processor=real_model
    )

    assert teacher.translate(["a"], ENG, KHK) == [[]]
    assert teacher.counters()["reserved_symbol"] == 1


def test_byte_pieces_of_a_byte_fallback_model_are_legitimate_and_stay(
    tmp_path: Path, real_model: sentencepiece.SentencePieceProcessor
) -> None:
    text = "Сайн " + chr(0x2605) + " уу"
    pieces = pieces_of(real_model, text)
    assert any(piece.startswith("<0x") for piece in pieces)
    # The pieces of the text are entries of the vocabulary, as they are of a real export.
    tokens = (*SOURCE_VOCABULARY, *pieces)
    teacher, _ = scripted_teacher(
        tmp_path,
        answering(hyp(*pieces, END)),
        processor=real_model,
        tokens=tokens,
    )

    assert teacher.translate(["a"], ENG, KHK) == [[text]]
    assert teacher.counters()["reserved_symbol"] == 0
