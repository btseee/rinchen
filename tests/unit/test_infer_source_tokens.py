"""Tests for `rinchen.infer.source_tokens`: where the direction tag goes for each placement,
and the pieces no translation of ours holds (the rule D10's teacher and the engine share).

The processor is `FakeProcessor` over the pieces a model of ours has; one test also reads the
pieces of a real SentencePiece model (the toy one) to see the rule agree with the library.
"""

from __future__ import annotations

import pytest
import sentencepiece

from rinchen.infer import source_tokens
from rinchen.infer.source_tokens import ct2_source_tokens, reserved_pieces
from rinchen.spm.symbols import STRUCTURAL_SYMBOLS, user_defined_symbols
from rinchen.train import distill_teacher
from tests.support.fake_ct2 import SPIECE, FakeProcessor
from tests.support.tiny_parallel import train_toy_spm

TAG = "<2khk_Cyrl>"
END = "</s>"
OPEN, CLOSE = chr(0x27E6), chr(0x27E7)
PLACEHOLDERS = [f"{OPEN}{index}{CLOSE}" for index in range(100)]
SYMBOLS = user_defined_symbols()
NON_PLACEHOLDERS = [symbol for symbol in SYMBOLS if symbol not in PLACEHOLDERS]
FOREIGN = ["<blank>", "averyunlikelytoken0"]
ORDINARY = SPIECE + "a"
BYTE = "<0x41>"

# The pieces of the fake model: the symbols of the project, an ordinary piece, a byte piece.
PIECES = ["<s>", END, *SYMBOLS, ORDINARY, BYTE]
VOCABULARY = frozenset({*PIECES, *FOREIGN})


@pytest.fixture
def processor() -> FakeProcessor:
    return FakeProcessor(PIECES)


# -- the source tokens of the two placements


def test_a_source_tag_goes_in_front_of_the_pieces_and_nothing_is_forced() -> None:
    assert ct2_source_tokens("src", TAG, [SPIECE + "a", SPIECE + "b"]) == (
        [TAG, SPIECE + "a", SPIECE + "b"],
        None,
    )


def test_a_target_tag_is_a_forced_prefix_and_the_source_stays_untagged() -> None:
    assert ct2_source_tokens("tgt", TAG, [SPIECE + "a", SPIECE + "b"]) == (
        [SPIECE + "a", SPIECE + "b"],
        [TAG],
    )


@pytest.mark.parametrize("placement", ["src", "tgt"])
def test_the_source_tokens_are_a_new_list_and_the_pieces_are_not_changed(
    placement: str,
) -> None:
    pieces = (SPIECE + "a",)

    tokens, _ = ct2_source_tokens(placement, TAG, pieces)  # type: ignore[arg-type]

    assert isinstance(tokens, list)
    assert pieces == (SPIECE + "a",)


def test_no_pieces_give_the_bare_tag_for_a_source_tag_and_nothing_for_a_target_tag() -> (
    None
):
    assert ct2_source_tokens("src", TAG, []) == ([TAG], None)
    assert ct2_source_tokens("tgt", TAG, []) == ([], [TAG])


@pytest.mark.parametrize("placement", ["both", "", "SRC", None])
def test_another_placement_is_a_value_error(placement: object) -> None:
    with pytest.raises(ValueError, match="placement"):
        ct2_source_tokens(placement, TAG, [SPIECE + "a"])  # type: ignore[arg-type]


def test_the_teacher_of_d10_uses_the_same_function() -> None:
    assert distill_teacher.ct2_source_tokens is source_tokens.ct2_source_tokens


# -- reserved_pieces: what a translation never holds


def test_the_unknown_piece_is_reserved(processor: FakeProcessor) -> None:
    assert "<unk>" in reserved_pieces(processor, END, VOCABULARY)


def test_the_unknown_piece_is_the_one_the_processor_names_not_a_spelling() -> None:
    class Renamed(FakeProcessor):
        def id_to_piece(self, piece_id: int) -> str:
            return (
                "<unknown-piece>"
                if piece_id == self.unk_id()
                else super().id_to_piece(piece_id)
            )

    reserved = reserved_pieces(Renamed(PIECES), END, VOCABULARY)

    assert "<unknown-piece>" in reserved
    assert "<unk>" not in reserved


def test_the_end_token_is_reserved_whatever_it_is_spelled(
    processor: FakeProcessor,
) -> None:
    assert END in reserved_pieces(processor, END, VOCABULARY)
    assert "<eos>" in reserved_pieces(processor, "<eos>", VOCABULARY)


@pytest.mark.parametrize("symbol", NON_PLACEHOLDERS)
def test_every_direction_tag_and_structural_symbol_is_reserved(
    processor: FakeProcessor, symbol: str
) -> None:
    assert symbol in reserved_pieces(processor, END, VOCABULARY)


def test_the_symbols_are_reserved_without_being_entries_of_the_vocabulary(
    processor: FakeProcessor,
) -> None:
    reserved = reserved_pieces(processor, END, frozenset({ORDINARY}))

    assert set(NON_PLACEHOLDERS) <= reserved


@pytest.mark.parametrize("entry", FOREIGN)
def test_a_vocabulary_entry_that_is_no_piece_of_the_model_is_reserved(
    processor: FakeProcessor, entry: str
) -> None:
    assert entry in reserved_pieces(processor, END, VOCABULARY)


def test_an_ordinary_piece_and_a_byte_piece_are_not_reserved(
    processor: FakeProcessor,
) -> None:
    reserved = reserved_pieces(processor, END, VOCABULARY)

    assert ORDINARY not in reserved
    assert BYTE not in reserved


def test_the_set_is_exactly_the_unknown_piece_the_end_the_symbols_and_the_foreign() -> (
    None
):
    reserved = reserved_pieces(FakeProcessor(PIECES), END, VOCABULARY)

    assert reserved == frozenset({"<unk>", END, *NON_PLACEHOLDERS, *FOREIGN})


# -- the placeholders and `allow`


def test_the_numbered_placeholders_are_never_reserved(processor: FakeProcessor) -> None:
    reserved = reserved_pieces(processor, END, VOCABULARY)

    assert len(PLACEHOLDERS) == 100
    assert not reserved & set(PLACEHOLDERS)


def test_the_placeholders_are_the_hundred_symbols_after_the_structural_ones() -> None:
    assert SYMBOLS[-100:] == PLACEHOLDERS
    assert SYMBOLS[-108:-100] == list(STRUCTURAL_SYMBOLS)


def test_a_symbol_in_allow_is_not_reserved(processor: FakeProcessor) -> None:
    default = reserved_pieces(processor, END, VOCABULARY)
    allowed = reserved_pieces(processor, END, VOCABULARY, allow=("<sep>",))

    assert "<sep>" in default
    assert allowed == default - {"<sep>"}


def test_allow_lets_through_the_symbols_it_names_and_no_others(
    processor: FakeProcessor,
) -> None:
    allowed = reserved_pieces(processor, END, VOCABULARY, allow=("<sep>", "<doc>"))

    assert not allowed & {"<sep>", "<doc>"}
    assert {"<bt>", TAG, *FOREIGN} <= allowed


def test_a_symbol_the_model_has_no_piece_for_stays_reserved_even_when_allowed() -> None:
    pieces = [piece for piece in PIECES if piece != "<sep>"]

    reserved = reserved_pieces(
        FakeProcessor(pieces), END, VOCABULARY | {"<sep>"}, allow=("<sep>",)
    )

    # An entry of the vocabulary that is no piece decodes to its letters: `allow` cannot help.
    assert "<sep>" in reserved


@pytest.mark.parametrize("allow", [("<sepp>",), (END,), (ORDINARY,), ("<sep>", "sep")])
def test_allow_names_reserved_symbols_and_nothing_else(
    processor: FakeProcessor, allow: tuple[str, ...]
) -> None:
    with pytest.raises(ValueError, match="allow"):
        reserved_pieces(processor, END, VOCABULARY, allow=allow)


@pytest.mark.parametrize("allow", ["<sep>", "sep"])
def test_allow_is_not_a_bare_string(processor: FakeProcessor, allow: str) -> None:
    with pytest.raises(TypeError, match="allow"):
        reserved_pieces(processor, END, VOCABULARY, allow=allow)


def test_a_placeholder_in_allow_is_accepted(processor: FakeProcessor) -> None:
    reserved = reserved_pieces(processor, END, VOCABULARY, allow=(PLACEHOLDERS[3],))

    assert PLACEHOLDERS[3] not in reserved


# -- the rule against the library


def test_the_rule_agrees_with_the_pieces_of_a_real_model(
    tmp_path_factory: pytest.TempPathFactory,
) -> None:
    model = train_toy_spm(tmp_path_factory.mktemp("source_tokens_spm"))
    real = sentencepiece.SentencePieceProcessor(model_file=str(model))
    pieces = [real.id_to_piece(index) for index in range(real.get_piece_size())]
    vocabulary = frozenset({*pieces, *FOREIGN})

    reserved = reserved_pieces(real, END, vocabulary)

    assert {"<unk>", END, "<sep>", TAG, "<bt>", *FOREIGN} <= reserved
    assert not reserved & {BYTE, "<0xE0>", "<s>", SPIECE, *PLACEHOLDERS}
    assert reserved == frozenset({"<unk>", END, *NON_PLACEHOLDERS, *FOREIGN})
