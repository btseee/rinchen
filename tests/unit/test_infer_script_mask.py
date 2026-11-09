"""Tests for `rinchen.infer.script_mask`: which scripts a target language must not write, how
pure a text is in its target's scripts, and the entries of `shared_vocabulary.json` that
CTranslate2 is told never to produce (`suppress_sequences`).

Most expectations are literals over a small fabricated vocabulary and `FakeProcessor`; a
second group reads the pieces of the toy SentencePiece model (real byte-fallback pieces, real
control pieces) so the rule is seen against the library's own answers.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import pytest
import sentencepiece

from rinchen.config.languages import load_languages
from rinchen.errors import ConfigError
from rinchen.infer.script_mask import (
    PURITY_FLOOR,
    allowed_scripts,
    forbidden_scripts,
    script_hint,
    script_purity,
    suppression_list,
)
from rinchen.lang.scripts import Script, counted_letters
from rinchen.spm.symbols import STRUCTURAL_SYMBOLS, user_defined_symbols
from tests.support.fake_ct2 import SPIECE, FakeProcessor
from tests.support.tiny_parallel import train_toy_spm

CYRILLIC, MONGOLIAN, LATIN = Script.CYRILLIC, Script.MONGOLIAN, Script.LATIN
HAN, HANGUL = Script.HAN, Script.HANGUL
HIRAGANA, KATAKANA = Script.HIRAGANA, Script.KATAKANA

END = "</s>"
# Letters of the Mongolian script (U+1820, U+1821, U+1822), a Mongolian format character
# (the free variation selector U+180B) and the katakana long-vowel mark (U+30FC, a letter of
# no script).
MONGOLIAN_WORD = chr(0x1820) + chr(0x1821) + chr(0x1822)
FREE_VARIATION_SELECTOR = chr(0x180B)
LONG_VOWEL_MARK = chr(0x30FC)
GREEK_WORD = chr(0x3B1) + chr(0x3B2)
HAN_WORD = chr(0x4E2D) + chr(0x6587)
KANA_WORD = chr(0x3072) + chr(0x3089)
PLACEHOLDERS = [chr(0x27E6) + str(index) + chr(0x27E7) for index in range(100)]
TAGS = [symbol for symbol in user_defined_symbols() if symbol.startswith("<2")]


def entries(listed: list[list[str]]) -> list[str]:
    """The single tokens of a suppression list, in order (every entry is one token)."""
    assert all(len(entry) == 1 for entry in listed)
    return [entry[0] for entry in listed]


# -- the scripts a target language writes and must not write


def test_a_cyrillic_target_forbids_every_other_script() -> None:
    assert forbidden_scripts("khk_Cyrl") == {
        MONGOLIAN,
        LATIN,
        HAN,
        HANGUL,
        HIRAGANA,
        KATAKANA,
    }


def test_a_traditional_script_target_forbids_cyrillic_and_the_rest() -> None:
    forbidden = forbidden_scripts("mn_Mong")

    assert CYRILLIC in forbidden
    assert MONGOLIAN not in forbidden
    assert forbidden == {CYRILLIC, LATIN, HAN, HANGUL, HIRAGANA, KATAKANA}


def test_a_target_of_three_scripts_forbids_none_of_them() -> None:
    forbidden = forbidden_scripts("jpn_Jpan")

    assert {LATIN, CYRILLIC, MONGOLIAN, HANGUL} <= forbidden
    assert not forbidden & {HAN, HIRAGANA, KATAKANA}
    assert forbidden == {LATIN, CYRILLIC, MONGOLIAN, HANGUL}


def test_the_allowed_scripts_are_the_registrys() -> None:
    assert allowed_scripts("khk_Cyrl") == {CYRILLIC}
    assert allowed_scripts("mn_Mong") == {MONGOLIAN}
    assert allowed_scripts("jpn_Jpan") == {HAN, HIRAGANA, KATAKANA}


def test_both_are_frozen_sets() -> None:
    assert isinstance(forbidden_scripts("khk_Cyrl"), frozenset)
    assert isinstance(allowed_scripts("khk_Cyrl"), frozenset)


@pytest.mark.parametrize("function", [forbidden_scripts, allowed_scripts])
def test_a_code_outside_the_registry_is_a_config_error(
    function: Callable[[str], frozenset[Script]],
) -> None:
    with pytest.raises(ConfigError, match="xx_Zzzz"):
        function("xx_Zzzz")


@pytest.mark.parametrize("code", [entry.code for entry in load_languages()])
def test_the_allowed_and_forbidden_scripts_of_a_language_split_every_script(
    code: str,
) -> None:
    allowed, forbidden = allowed_scripts(code), forbidden_scripts(code)

    assert allowed
    assert not allowed & forbidden
    assert allowed | forbidden == set(Script)


# -- the script of a text


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        (MONGOLIAN_WORD, "mn_Mong"),
        ("Энэ нь", "und"),
        ("Hello there", "und"),
        ("", "und"),
        ("123 !!", "und"),
        (FREE_VARIATION_SELECTOR, "und"),
        ("Энэ " + MONGOLIAN_WORD, "mn_Mong"),
        (HAN_WORD, "und"),
    ],
)
def test_the_script_hint_is_mongolian_script_or_undetermined(
    text: str, expected: str
) -> None:
    assert script_hint(text) == expected


# -- counted letters (the denominator of lang.scripts.script_of)


@pytest.mark.parametrize(
    ("text", "count"),
    [
        ("", 0),
        ("123 !? ", 0),
        ("Энэ abc 12", 6),
        (GREEK_WORD, 2),
        (LONG_VOWEL_MARK, 0),
        (MONGOLIAN_WORD + FREE_VARIATION_SELECTOR, 3),
    ],
)
def test_counted_letters_are_the_letters_script_of_divides_by(
    text: str, count: int
) -> None:
    assert counted_letters(text) == count


# -- script purity


def test_half_of_the_letters_in_the_scripts_is_a_purity_of_one_half() -> None:
    assert script_purity("Энэ abc", {CYRILLIC}) == 0.5


def test_a_text_with_no_letters_is_pure() -> None:
    assert script_purity("123 !!", {CYRILLIC}) == 1.0
    assert script_purity("", {CYRILLIC}) == 1.0
    assert script_purity(" \n ", {CYRILLIC}) == 1.0


def test_every_letter_in_one_of_several_scripts_is_pure() -> None:
    assert script_purity("Энэ", {CYRILLIC, LATIN}) == 1.0
    assert script_purity("Энэ abc", {CYRILLIC, LATIN}) == 1.0


def test_a_word_of_another_script_is_not_pure_at_all() -> None:
    assert script_purity(MONGOLIAN_WORD, {CYRILLIC}) == 0.0
    assert script_purity("abc", {CYRILLIC}) == 0.0


def test_punctuation_digits_and_marks_do_not_count() -> None:
    assert script_purity("12 Энэ, abc! 7" + FREE_VARIATION_SELECTOR, {CYRILLIC}) == 0.5


def test_a_letter_of_a_script_nobody_names_counts_against_the_text() -> None:
    assert script_purity(GREEK_WORD, {CYRILLIC}) == 0.0
    assert script_purity("Энэ" + GREEK_WORD, {CYRILLIC}) == 0.6


def test_a_letter_of_no_script_is_neutral() -> None:
    assert script_purity(LONG_VOWEL_MARK, {CYRILLIC}) == 1.0
    assert script_purity("Энэ" + LONG_VOWEL_MARK, {CYRILLIC}) == 1.0


def test_the_purity_of_a_text_in_no_scripts_is_zero_when_it_has_letters() -> None:
    assert script_purity("abc", set()) == 0.0


def test_the_purity_floor_is_the_documented_prior() -> None:
    assert PURITY_FLOOR == 0.98


# -- suppression_list over a fabricated vocabulary

# "ba" in Cyrillic, spelled by code point: the second letter looks like a Latin one.
CYRILLIC_PIECE = SPIECE + chr(0x431) + chr(0x430)
LATIN_PIECE = SPIECE + "Th"
LATIN_TAIL = "e"
MIXED_PIECE = SPIECE + "Э" + "x"  # a Cyrillic and a Latin letter
MONGOLIAN_PIECE = SPIECE + MONGOLIAN_WORD
HAN_PIECE = SPIECE + HAN_WORD
DIGIT_PIECE = SPIECE + "123"
BYTE_UPPER_A, BYTE_DIGIT, BYTE_TAIL = "<0x41>", "<0x30>", "<0xE0>"
TAG_KHK, TAG_ENG = "<2khk_Cyrl>", "<2eng_Latn>"

# The vocabulary of an Eole export: its own tokens first (`<blank>`, `<s>`, `</s>`, `<unk>`),
# the pieces of the model, and the made-up padding token at the end.
VOCABULARY = [
    "<blank>",
    "<s>",
    END,
    "<unk>",
    TAG_KHK,
    TAG_ENG,
    "<sep>",
    PLACEHOLDERS[0],
    CYRILLIC_PIECE,
    LATIN_PIECE,
    LATIN_TAIL,
    MIXED_PIECE,
    SPIECE,
    DIGIT_PIECE,
    MONGOLIAN_PIECE,
    HAN_PIECE,
    BYTE_UPPER_A,
    BYTE_DIGIT,
    BYTE_TAIL,
    "averyunlikelytoken0",
]
# Every entry but the padding tokens and the unknown piece is a piece of the model.
PROCESSOR_PIECES = [
    entry
    for entry in VOCABULARY
    if entry not in ("<blank>", "<unk>", "averyunlikelytoken0")
]
CONTROL_ENTRIES = ["<blank>", "<s>", "<unk>", "averyunlikelytoken0"]


@pytest.fixture
def processor() -> FakeProcessor:
    return FakeProcessor(PROCESSOR_PIECES)


def listed(
    processor: FakeProcessor,
    tgt: str,
    *,
    scripts: bool = True,
    allow: Sequence[str] = (),
) -> list[str]:
    return entries(
        suppression_list(
            VOCABULARY, processor, tgt, end_token=END, allow=allow, scripts=scripts
        )
    )


def test_a_cyrillic_target_lists_the_control_entries_the_symbols_and_the_foreign_pieces(
    processor: FakeProcessor,
) -> None:
    assert listed(processor, "khk_Cyrl") == [
        "<blank>",
        "<s>",
        "<unk>",
        TAG_KHK,
        TAG_ENG,
        "<sep>",
        LATIN_PIECE,
        LATIN_TAIL,
        MIXED_PIECE,
        MONGOLIAN_PIECE,
        HAN_PIECE,
        BYTE_UPPER_A,
        "averyunlikelytoken0",
    ]


def test_a_latin_target_lists_the_cyrillic_pieces_and_no_byte_piece(
    processor: FakeProcessor,
) -> None:
    assert listed(processor, "eng_Latn") == [
        "<blank>",
        "<s>",
        "<unk>",
        TAG_KHK,
        TAG_ENG,
        "<sep>",
        CYRILLIC_PIECE,
        MIXED_PIECE,
        MONGOLIAN_PIECE,
        HAN_PIECE,
        "averyunlikelytoken0",
    ]


def test_a_traditional_script_target_lists_cyrillic_latin_and_the_ascii_byte_pieces(
    processor: FakeProcessor,
) -> None:
    assert listed(processor, "mn_Mong") == [
        "<blank>",
        "<s>",
        "<unk>",
        TAG_KHK,
        TAG_ENG,
        "<sep>",
        CYRILLIC_PIECE,
        LATIN_PIECE,
        LATIN_TAIL,
        MIXED_PIECE,
        HAN_PIECE,
        BYTE_UPPER_A,
        "averyunlikelytoken0",
    ]


def test_the_end_token_is_never_listed_for_any_target(processor: FakeProcessor) -> None:
    for code in (entry.code for entry in load_languages()):
        assert END not in listed(processor, code)
        assert END not in listed(processor, code, scripts=False)


def test_the_end_token_is_the_one_the_caller_names() -> None:
    vocabulary = [*VOCABULARY, "<eos>"]
    other = FakeProcessor([*PROCESSOR_PIECES, "<eos>"])

    result = entries(suppression_list(vocabulary, other, "khk_Cyrl", end_token="<eos>"))

    assert "<eos>" not in result
    assert "</s>" in result  # a control piece, now that it is not the end


def test_the_off_mode_lists_the_control_entries_only(processor: FakeProcessor) -> None:
    for code in ("khk_Cyrl", "eng_Latn", "mn_Mong"):
        assert listed(processor, code, scripts=False) == CONTROL_ENTRIES


@pytest.mark.parametrize("code", ["khk_Cyrl", "eng_Latn", "mn_Mong", "jpn_Jpan"])
def test_the_control_entries_are_listed_for_every_target(
    processor: FakeProcessor, code: str
) -> None:
    assert set(CONTROL_ENTRIES) <= set(listed(processor, code))


def test_a_reserved_symbol_the_caller_allows_is_not_listed(
    processor: FakeProcessor,
) -> None:
    default = listed(processor, "khk_Cyrl")
    allowed = listed(processor, "khk_Cyrl", allow=("<sep>",))

    assert "<sep>" in default
    assert allowed == [entry for entry in default if entry != "<sep>"]


def test_a_placeholder_is_never_listed(processor: FakeProcessor) -> None:
    for code in ("khk_Cyrl", "eng_Latn", "mn_Mong"):
        assert PLACEHOLDERS[0] not in listed(processor, code)
        assert PLACEHOLDERS[0] not in listed(processor, code, scripts=False)


def test_a_piece_with_a_letter_of_a_forbidden_script_is_listed_by_that_letter(
    processor: FakeProcessor,
) -> None:
    result = listed(processor, "khk_Cyrl")

    assert LATIN_PIECE in result
    assert CYRILLIC_PIECE not in result


def test_the_dummy_prefix_alone_is_no_letter_of_any_script(
    processor: FakeProcessor,
) -> None:
    for code in ("khk_Cyrl", "eng_Latn", "mn_Mong", "jpn_Jpan"):
        assert SPIECE not in listed(processor, code)


def test_a_piece_of_digits_is_listed_for_no_target(processor: FakeProcessor) -> None:
    for code in ("khk_Cyrl", "eng_Latn", "mn_Mong", "jpn_Jpan"):
        assert DIGIT_PIECE not in listed(processor, code)


def test_a_byte_piece_of_an_ascii_letter_is_listed_when_latin_is_forbidden(
    processor: FakeProcessor,
) -> None:
    assert BYTE_UPPER_A in listed(processor, "khk_Cyrl")
    assert BYTE_UPPER_A in listed(processor, "mn_Mong")
    assert BYTE_UPPER_A not in listed(processor, "eng_Latn")


def test_the_other_byte_pieces_are_never_listed(processor: FakeProcessor) -> None:
    for code in ("khk_Cyrl", "eng_Latn", "mn_Mong", "jpn_Jpan"):
        assert not {BYTE_DIGIT, BYTE_TAIL} & set(listed(processor, code))


def test_an_entry_that_is_not_in_the_vocabulary_is_never_listed(
    processor: FakeProcessor,
) -> None:
    short = [CYRILLIC_PIECE, LATIN_PIECE]

    result = entries(suppression_list(short, processor, "khk_Cyrl", end_token=END))

    assert result == [LATIN_PIECE]


def test_the_list_follows_the_order_of_the_vocabulary_and_names_an_entry_once(
    processor: FakeProcessor,
) -> None:
    vocabulary = [LATIN_PIECE, "<blank>", LATIN_PIECE, LATIN_TAIL, "<blank>"]

    result = entries(suppression_list(vocabulary, processor, "khk_Cyrl", end_token=END))

    assert result == [LATIN_PIECE, "<blank>", LATIN_TAIL]


def test_every_call_gives_new_lists(processor: FakeProcessor) -> None:
    first = suppression_list(VOCABULARY, processor, "khk_Cyrl", end_token=END)
    first[0].append("changed")
    second = suppression_list(VOCABULARY, processor, "khk_Cyrl", end_token=END)

    assert second[0] == ["<blank>"]


def test_an_empty_vocabulary_lists_nothing(processor: FakeProcessor) -> None:
    assert suppression_list([], processor, "khk_Cyrl", end_token=END) == []


@pytest.mark.parametrize("scripts", [True, False])
def test_a_target_outside_the_registry_is_a_config_error(
    processor: FakeProcessor, scripts: bool
) -> None:
    with pytest.raises(ConfigError, match="xx_Zzzz"):
        suppression_list(
            VOCABULARY, processor, "xx_Zzzz", end_token=END, scripts=scripts
        )


def test_an_allow_that_names_no_reserved_symbol_is_refused(
    processor: FakeProcessor,
) -> None:
    with pytest.raises(ValueError, match="allow"):
        suppression_list(
            VOCABULARY, processor, "khk_Cyrl", end_token=END, allow=("<sepp>",)
        )


# -- suppression_list over the pieces of a real SentencePiece model


@pytest.fixture(scope="module")
def real(
    tmp_path_factory: pytest.TempPathFactory,
) -> sentencepiece.SentencePieceProcessor:
    model = train_toy_spm(tmp_path_factory.mktemp("script_mask_spm"))
    return sentencepiece.SentencePieceProcessor(model_file=str(model))


def eole_vocabulary(real: sentencepiece.SentencePieceProcessor) -> list[str]:
    pieces = [real.id_to_piece(index) for index in range(real.get_piece_size())]
    return ["<blank>", *pieces, "averyunlikelytoken0"]


def real_listed(
    real: sentencepiece.SentencePieceProcessor,
    tgt: str,
    *,
    scripts: bool = True,
    allow: Sequence[str] = (),
) -> list[str]:
    return entries(
        suppression_list(
            eole_vocabulary(real),
            real,
            tgt,
            end_token=END,
            allow=allow,
            scripts=scripts,
        )
    )


def test_the_toy_model_has_the_pieces_the_literals_below_name(
    real: sentencepiece.SentencePieceProcessor,
) -> None:
    pieces = set(eole_vocabulary(real))

    assert {LATIN_PIECE, CYRILLIC_PIECE, SPIECE, "<unk>", "<s>", END} <= pieces
    assert {f"<0x{value:02X}>" for value in range(256)} <= pieces


def test_a_cyrillic_target_lists_latin_pieces_and_the_ascii_letter_bytes_of_a_real_model(
    real: sentencepiece.SentencePieceProcessor,
) -> None:
    result = real_listed(real, "khk_Cyrl")

    assert {LATIN_PIECE, SPIECE + "the", LATIN_TAIL} <= set(result)
    letter_bytes = {
        f"<0x{value:02X}>" for value in (*range(0x41, 0x5B), *range(0x61, 0x7B))
    }
    assert len(letter_bytes) == 52
    assert {entry for entry in result if entry.startswith("<0x")} == letter_bytes


def test_a_cyrillic_target_lists_no_cyrillic_piece_and_no_other_byte_piece(
    real: sentencepiece.SentencePieceProcessor,
) -> None:
    result = set(real_listed(real, "khk_Cyrl"))

    assert not result & {CYRILLIC_PIECE, SPIECE + "Э", SPIECE, END, "<0x30>", "<0x80>"}
    assert not result & {"<0x40>", "<0x5B>", "<0x60>", "<0x7B>", "<0xE0>"}
    assert not result & set(PLACEHOLDERS)


def test_a_cyrillic_target_lists_the_control_pieces_and_the_foreign_entries_of_a_real_model(
    real: sentencepiece.SentencePieceProcessor,
) -> None:
    result = real_listed(real, "khk_Cyrl")

    assert {"<blank>", "<unk>", "<s>", "averyunlikelytoken0"} <= set(result)


def test_every_direction_tag_and_structural_symbol_of_a_real_model_is_listed(
    real: sentencepiece.SentencePieceProcessor,
) -> None:
    result = set(real_listed(real, "khk_Cyrl"))

    assert len(TAGS) == 10
    assert set(TAGS) <= result
    assert set(STRUCTURAL_SYMBOLS) <= result
    assert set(STRUCTURAL_SYMBOLS) - {"<sep>"} <= set(
        real_listed(real, "khk_Cyrl", allow=("<sep>",))
    )
    assert "<sep>" not in real_listed(real, "khk_Cyrl", allow=("<sep>",))


def test_a_latin_target_lists_cyrillic_pieces_and_no_byte_piece_of_a_real_model(
    real: sentencepiece.SentencePieceProcessor,
) -> None:
    result = real_listed(real, "eng_Latn")

    assert {CYRILLIC_PIECE, SPIECE + "Э"} <= set(result)
    assert not {entry for entry in result if entry.startswith("<0x")}
    assert not set(result) & {LATIN_PIECE, SPIECE + "the", LATIN_TAIL, SPIECE, END}


def test_the_off_mode_of_a_real_model_lists_four_entries_in_vocabulary_order(
    real: sentencepiece.SentencePieceProcessor,
) -> None:
    assert real_listed(real, "khk_Cyrl", scripts=False) == [
        "<blank>",
        "<unk>",
        "<s>",
        "averyunlikelytoken0",
    ]


def test_the_list_of_a_real_model_holds_only_entries_of_its_vocabulary(
    real: sentencepiece.SentencePieceProcessor,
) -> None:
    vocabulary = eole_vocabulary(real)

    for code in (entry.code for entry in load_languages()):
        result = real_listed(real, code)
        assert set(result) <= set(vocabulary)
        assert len(result) == len(set(result))
        assert END not in result
