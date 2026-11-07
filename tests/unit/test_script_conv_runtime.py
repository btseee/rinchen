"""`ScriptConverter` with a stub translator and a stub or tiny real SentencePiece model.

No CTranslate2 model is loaded here (the integration tests do that). Mongolian and
invisible characters are written as `chr()` calls so the exact code points stay
reviewable; Cyrillic words are written out.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest
import sentencepiece

from rinchen.errors import ConfigError, DataError
from rinchen.script_conv import runtime
from rinchen.script_conv.alphabet import NNBSP
from rinchen.script_conv.char_model import train_char_model
from rinchen.script_conv.rules import transliterate_word
from rinchen.script_conv.runtime import Conversion, ScriptConverter
from rinchen.script_conv.validate import validate_traditional
from tests.support.stub_runtime import (
    DUMMY_PREFIX,
    StubProcessor,
    StubTranslator,
)

TO_SCRIPT_TAG = "<2mn_Mong>"
TO_CYRILLIC_TAG = "<2mn_Cyrl>"

A = chr(0x1820)
NA = chr(0x1828)
FVS_FIRST = chr(0x180B)
ZWJ = chr(0x200D)
ZWNJ = chr(0x200C)
SCRIPT_COMMA = chr(0x1802)
SCRIPT_FULL_STOP = chr(0x1803)
MONGOL_SCRIPT = "".join(
    chr(code_point) for code_point in (0x182E, 0x1823, 0x1829, 0x182D, 0x1823, 0x182F)
)
ULS_SCRIPT = "".join(chr(code_point) for code_point in (0x1824, 0x182F, 0x1830))
NOM_SCRIPT = "".join(chr(code_point) for code_point in (0x1828, 0x1823, 0x182E))
SUFFIX_UN = chr(0x1824) + chr(0x1828)
NOM_WITH_SUFFIX = NOM_SCRIPT + NNBSP + SUFFIX_UN

TO_SCRIPT_ANSWERS = {
    "монгол": MONGOL_SCRIPT,
    "улс": ULS_SCRIPT,
    "ном": NOM_SCRIPT,
    "оны": NA + A,
    "н": A,
    "р": NA,
    "ыг": A + A,
}
TO_CYRILLIC_ANSWERS = {
    MONGOL_SCRIPT: "монгол",
    ULS_SCRIPT: "улс",
    NOM_SCRIPT: "ном",
    NOM_WITH_SUFFIX: "номын",
    A + NA: "ан",
}


def _converter(
    answers: Mapping[str, str],
) -> tuple[ScriptConverter, StubTranslator]:
    """A converter over a stub translator that raises KeyError for a word it was not
    given an answer for, so an unexpected word sent to the model fails the test."""
    translator = StubTranslator(answers.__getitem__)
    return ScriptConverter.from_parts(translator, StubProcessor()), translator


def _to_script() -> tuple[ScriptConverter, StubTranslator]:
    return _converter(TO_SCRIPT_ANSWERS)


def _to_cyrillic() -> tuple[ScriptConverter, StubTranslator]:
    return _converter(TO_CYRILLIC_ANSWERS)


def _call_options(translator: StubTranslator) -> list[dict[str, Any]]:
    return [options for _, options in translator.calls]


# ---- Conversion and construction --------------------------------------------------


def test_conversion_is_a_frozen_value_with_both_counts() -> None:
    assert Conversion("x", 1, 2) == Conversion("x", 1, 2)
    with pytest.raises(AttributeError):
        Conversion("x", 0, 0).text = "y"  # type: ignore[misc]


def test_from_parts_builds_a_converter_without_loading_a_model() -> None:
    converter = ScriptConverter.from_parts(
        StubTranslator(TO_SCRIPT_ANSWERS.__getitem__), StubProcessor()
    )

    assert isinstance(converter, ScriptConverter)


# ---- Mong: words and the text around them ----------------------------------------


def test_three_occurrences_of_a_word_cause_one_translated_entry_in_one_batch() -> None:
    converter, translator = _to_script()

    result = converter.convert("монгол монгол монгол", "Mong")

    assert len(translator.calls) == 1
    assert translator.words_sent == ["монгол"]
    assert result == Conversion(" ".join([MONGOL_SCRIPT] * 3), 0, 0)


def test_all_distinct_words_of_a_call_go_in_one_batch_in_order_of_appearance() -> None:
    converter, translator = _to_script()

    result = converter.convert("улс монгол улс ном", "Mong")

    assert len(translator.calls) == 1
    assert translator.words_sent == ["улс", "монгол", "ном"]
    assert result.text == f"{ULS_SCRIPT} {MONGOL_SCRIPT} {ULS_SCRIPT} {NOM_SCRIPT}"


def test_spaces_digits_latin_runs_hyphens_and_other_punctuation_stay() -> None:
    text = "  2024оны н-р iPhone-ыг (улс)!?;:\n\tномын"
    converter, _ = _converter({**TO_SCRIPT_ANSWERS, "номын": NOM_SCRIPT + SUFFIX_UN})

    result = converter.convert(text, "Mong")

    assert result.text == (
        f"  2024{NA + A} {A}-{NA} iPhone-{A + A} ({ULS_SCRIPT})!?;:\n\t"
        f"{NOM_SCRIPT + SUFFIX_UN}"
    )
    assert (result.rule_fallbacks, result.invalid_outputs) == (0, 0)


def test_full_stop_and_comma_become_the_script_marks_outside_numbers_and_latin() -> (
    None
):
    converter, _ = _to_script()

    result = converter.convert("монгол, улс. 3.5 a.b 1,000", "Mong")

    assert result.text == (
        f"{MONGOL_SCRIPT}{SCRIPT_COMMA} {ULS_SCRIPT}{SCRIPT_FULL_STOP} 3.5 a.b 1,000"
    )


def test_script_text_passes_through_while_the_cyrillic_word_beside_it_converts() -> (
    None
):
    converter, translator = _to_script()

    result = converter.convert(f"{NOM_WITH_SUFFIX} улс", "Mong")

    assert result.text == f"{NOM_WITH_SUFFIX} {ULS_SCRIPT}"
    assert translator.words_sent == ["улс"]


def test_mixed_case_words_reach_the_translator_lower_cased_and_only_once() -> None:
    converter, translator = _to_script()

    result = converter.convert("МОНГОЛ Монгол монгол", "Mong")

    assert translator.words_sent == ["монгол"]
    assert result.text == " ".join([MONGOL_SCRIPT] * 3)


def test_words_reach_the_translator_normalized() -> None:
    # И + combining breve is not NFC: normalization composes it into one letter.
    decomposed = "и" + chr(0x0306)
    composed = chr(0x0439)
    converter, translator = _converter({composed: A})

    result = converter.convert(decomposed, "Mong")

    assert translator.words_sent == [composed]
    assert result.text == A


def test_source_tokens_are_the_script_tag_then_the_pieces() -> None:
    converter, translator = _to_script()

    converter.convert("улс", "Mong")

    ((source, _),) = translator.calls
    assert source == [["<2mn_Mong>", DUMMY_PREFIX, "у", "л", "с"]]


def test_the_batch_is_decoded_greedily_with_a_limit_from_the_longest_word() -> None:
    converter, translator = _converter({"а": A, "улс": ULS_SCRIPT})

    converter.convert("а улс", "Mong")

    assert _call_options(translator) == [{"beam_size": 1, "max_decoding_length": 20}]


@pytest.mark.parametrize(
    ("length", "limit"),
    [(1, 12), (3, 20), (60, 248), (64, 256), (80, 256)],
)
def test_the_decoding_limit_is_four_times_the_longest_word_plus_eight_up_to_256(
    length: int, limit: int
) -> None:
    converter, translator = _converter({"а" * length: A})

    converter.convert("а" * length, "Mong")

    assert _call_options(translator)[0]["max_decoding_length"] == limit


def test_the_leading_dummy_prefix_does_not_become_a_space_in_the_word() -> None:
    converter, _ = _to_script()

    result = converter.convert("улс", "Mong")

    assert result.text == ULS_SCRIPT
    assert not result.text.startswith(" ")


@pytest.fixture(scope="module")
def spm_model(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return train_char_model(
        ["монгол", "улс", MONGOL_SCRIPT, ULS_SCRIPT, NOM_WITH_SUFFIX],
        tmp_path_factory.mktemp("runtime_model") / "chars",
        vocab_size=64,
    )


@pytest.fixture(scope="module")
def real_processor(spm_model: Path) -> sentencepiece.SentencePieceProcessor:
    return sentencepiece.SentencePieceProcessor(model_file=str(spm_model))


def test_the_stub_processor_encodes_and_decodes_like_the_real_model(
    real_processor: sentencepiece.SentencePieceProcessor,
) -> None:
    stub = StubProcessor()

    for text in ("монгол", NOM_WITH_SUFFIX):
        assert stub.encode(text, out_type=str) == real_processor.encode(
            text, out_type=str
        )
    for pieces in ([DUMMY_PREFIX, "м"], ["м", "о"], [DUMMY_PREFIX, "м", DUMMY_PREFIX]):
        assert stub.decode(pieces) == real_processor.decode(pieces)


def test_a_real_character_model_decodes_the_hypothesis_without_a_leading_space(
    real_processor: sentencepiece.SentencePieceProcessor,
) -> None:
    translator = StubTranslator(TO_SCRIPT_ANSWERS.__getitem__)
    converter = ScriptConverter.from_parts(translator, real_processor)

    result = converter.convert("монгол", "Mong")

    assert result.text == MONGOL_SCRIPT
    ((source, _),) = translator.calls
    assert source == [["<2mn_Mong>", DUMMY_PREFIX, *"монгол"]]


# ---- Mong: outputs the model got wrong -------------------------------------------

INVALID_SCRIPT_OUTPUTS = {
    "zero width joiner": A + ZWJ + NA,
    "zero width non-joiner": A + ZWNJ + NA,
    "cyrillic letters": "монгол",
    "selector with no letter": FVS_FIRST + A,
    "newline": A + "\n" + NA,
    "empty": "",
}


@pytest.mark.parametrize(
    "bad_output", INVALID_SCRIPT_OUTPUTS.values(), ids=list(INVALID_SCRIPT_OUTPUTS)
)
def test_an_invalid_output_is_replaced_by_the_rule_word_for_each_occurrence(
    bad_output: str,
) -> None:
    converter, _ = _converter({"улс": bad_output})
    rule_word = transliterate_word("улс").text
    assert rule_word
    assert validate_traditional(rule_word) == ()

    result = converter.convert("улс, улс", "Mong")

    assert result == Conversion(f"{rule_word}{SCRIPT_COMMA} {rule_word}", 2, 0)


def test_a_replaced_word_is_spelled_from_its_lower_cased_form() -> None:
    converter, _ = _converter({"улс": A + ZWJ})

    result = converter.convert("УЛС", "Mong")

    assert result == Conversion(transliterate_word("улс").text, 1, 0)


def test_a_replaced_word_is_spelled_from_its_normalized_form() -> None:
    # Normalization folds the old letter fita into the letter oe, which the rules can
    # spell and the fita itself they cannot.
    fita = chr(0x0473)
    oe = chr(0x04E9)
    assert transliterate_word(fita).text == ""
    converter, _ = _converter({oe: ""})

    result = converter.convert(fita, "Mong")

    assert result == Conversion(transliterate_word(oe).text, 1, 0)


def test_a_word_the_rules_cannot_spell_keeps_its_original_text_and_counts_invalid() -> (
    None
):
    dropped_letter_word = "Ъ"
    assert transliterate_word(dropped_letter_word).text == ""
    converter, _ = _converter({"ъ": ""})

    result = converter.convert(f"{dropped_letter_word} {dropped_letter_word}", "Mong")

    assert result == Conversion(f"{dropped_letter_word} {dropped_letter_word}", 0, 2)


def test_valid_outputs_are_not_counted_as_fallbacks() -> None:
    converter, _ = _to_script()

    result = converter.convert("монгол улс", "Mong")

    assert (result.rule_fallbacks, result.invalid_outputs) == (0, 0)


# ---- Mong: words the model never sees --------------------------------------------


def test_a_word_over_eighty_characters_is_spelled_by_the_rules_and_never_sent() -> None:
    long_word = "а" * 81
    rule_word = transliterate_word(long_word).text
    assert rule_word
    converter, translator = _to_script()

    result = converter.convert(f"{long_word} улс", "Mong")

    assert translator.words_sent == ["улс"]
    assert result == Conversion(f"{rule_word} {ULS_SCRIPT}", 1, 0)


def test_a_word_of_exactly_eighty_characters_is_sent() -> None:
    converter, translator = _converter({"а" * 80: A})

    result = converter.convert("а" * 80, "Mong")

    assert translator.words_sent == ["а" * 80]
    assert result == Conversion(A, 0, 0)


def test_a_word_that_normalizes_to_nothing_is_spelled_by_the_rules_and_never_sent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(runtime, "normalize", lambda text, lang: "")
    converter, translator = _to_script()

    result = converter.convert("улс", "Mong")

    assert translator.calls == []
    assert result == Conversion(transliterate_word("улс").text, 1, 0)


# ---- Mong: nothing to convert ----------------------------------------------------


@pytest.mark.parametrize("target", ["Mong", "Cyrl"])
@pytest.mark.parametrize("text", ["", " ", "\n\t ", NNBSP, "  \n  "])
def test_empty_and_whitespace_only_text_comes_back_unchanged_with_no_call(
    text: str, target: str
) -> None:
    converter, translator = _to_script()

    result = converter.convert(text, target)  # type: ignore[arg-type]

    assert result == Conversion(text, 0, 0)
    assert translator.calls == []


def test_text_with_no_cyrillic_word_comes_back_unchanged_with_no_call() -> None:
    converter, translator = _to_script()
    text = f"3.5, abc. {MONGOL_SCRIPT}{SCRIPT_COMMA} 1,000"

    result = converter.convert(text, "Mong")

    assert result == Conversion(text, 0, 0)
    assert translator.calls == []


# ---- Cache -----------------------------------------------------------------------


def test_a_second_call_with_the_same_words_sends_nothing_to_the_translator() -> None:
    converter, translator = _to_script()
    first = converter.convert("монгол улс", "Mong")

    second = converter.convert("улс монгол", "Mong")

    assert len(translator.calls) == 1
    assert second.text == f"{ULS_SCRIPT} {MONGOL_SCRIPT}"
    assert first.text == f"{MONGOL_SCRIPT} {ULS_SCRIPT}"


def test_a_call_sends_only_the_words_not_cached_yet() -> None:
    converter, translator = _to_script()
    converter.convert("монгол", "Mong")

    converter.convert("монгол улс", "Mong")

    assert _sent_per_call(translator) == [["монгол"], ["улс"]]


def test_a_cached_word_is_still_counted_as_a_fallback_each_time_it_occurs() -> None:
    converter, translator = _converter({"улс": ""})

    first = converter.convert("улс", "Mong")
    second = converter.convert("улс улс", "Mong")

    assert len(translator.calls) == 1
    assert (first.rule_fallbacks, second.rule_fallbacks) == (1, 2)


def test_the_cache_is_dropped_when_it_would_pass_its_bound(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(runtime, "_MAX_CACHED_WORDS", 2)
    converter, translator = _to_script()

    converter.convert("монгол улс", "Mong")  # fills the cache
    converter.convert("ном", "Mong")  # would pass the bound: the cache restarts
    converter.convert("ном", "Mong")  # cached again
    converter.convert("монгол", "Mong")  # was dropped

    assert _sent_per_call(translator) == [
        ["монгол", "улс"],
        ["ном"],
        ["монгол"],
    ]


def _sent_per_call(translator: StubTranslator) -> list[list[str]]:
    return [
        ["".join(tokens[2:]) for tokens in source] for source, _ in translator.calls
    ]


# ---- Cyrl ------------------------------------------------------------------------


def test_source_tokens_are_the_cyrillic_tag_then_the_pieces() -> None:
    converter, translator = _to_cyrillic()

    converter.convert(ULS_SCRIPT, "Cyrl")

    ((source, _),) = translator.calls
    assert source == [["<2mn_Cyrl>", DUMMY_PREFIX, *ULS_SCRIPT]]


def test_script_words_become_cyrillic_and_the_marks_map_back() -> None:
    converter, translator = _to_cyrillic()
    text = f"{MONGOL_SCRIPT}{SCRIPT_COMMA} {ULS_SCRIPT}{SCRIPT_FULL_STOP} 3.5 a.b"

    result = converter.convert(text, "Cyrl")

    assert result == Conversion("монгол, улс. 3.5 a.b", 0, 0)
    assert _call_options(translator) == [{"beam_size": 1, "max_decoding_length": 32}]


def test_cyrillic_words_stay_when_converting_to_cyrillic() -> None:
    converter, translator = _to_cyrillic()

    result = converter.convert(f"монгол {ULS_SCRIPT}", "Cyrl")

    assert result.text == "монгол улс"
    assert translator.words_sent == [ULS_SCRIPT]


def test_text_with_no_script_word_comes_back_unchanged_with_no_call() -> None:
    converter, translator = _to_cyrillic()
    text = "монгол, улс. 3.5"

    result = converter.convert(text, "Cyrl")

    assert result == Conversion(text, 0, 0)
    assert translator.calls == []


def test_a_script_word_with_a_suffix_gap_is_one_word_for_the_model() -> None:
    converter, translator = _to_cyrillic()

    result = converter.convert(f"{NOM_WITH_SUFFIX} {MONGOL_SCRIPT}", "Cyrl")

    assert translator.words_sent == [NOM_WITH_SUFFIX, MONGOL_SCRIPT]
    assert result.text == "номын монгол"


def test_a_script_word_is_sent_without_the_joiners_the_training_words_had_none_of() -> (
    None
):
    converter, translator = _to_cyrillic()

    result = converter.convert(A + ZWJ + NA, "Cyrl")

    assert translator.words_sent == [A + NA]
    assert result == Conversion("ан", 0, 0)


def test_a_run_of_selectors_without_a_letter_is_not_a_word() -> None:
    converter, translator = _to_cyrillic()

    result = converter.convert(f"{FVS_FIRST} {ULS_SCRIPT}", "Cyrl")

    assert translator.words_sent == [ULS_SCRIPT]
    assert result == Conversion(f"{FVS_FIRST} улс", 0, 0)


@pytest.mark.parametrize(
    "bad_output",
    ["", "uls", "улс1", "ул с", "ул-с", "улс.", "уlс", "улс\n", A + NA],
    ids=[
        "empty",
        "latin",
        "digit",
        "space",
        "hyphen",
        "full stop",
        "mixed script",
        "newline",
        "script letters",
    ],
)
def test_an_output_that_is_not_one_cyrillic_word_keeps_the_script_word_and_is_counted(
    bad_output: str,
) -> None:
    converter, _ = _converter({ULS_SCRIPT: bad_output})

    result = converter.convert(f"{ULS_SCRIPT} {ULS_SCRIPT}", "Cyrl")

    assert result == Conversion(f"{ULS_SCRIPT} {ULS_SCRIPT}", 0, 2)


def test_a_script_word_over_eighty_characters_is_kept_counted_and_never_sent() -> None:
    long_word = A * 81
    converter, translator = _to_cyrillic()

    result = converter.convert(f"{long_word} {ULS_SCRIPT}", "Cyrl")

    assert translator.words_sent == [ULS_SCRIPT]
    assert result == Conversion(f"{long_word} улс", 0, 1)


def test_a_failed_script_word_keeps_its_own_joiners() -> None:
    converter, _ = _converter({A + NA: ""})

    result = converter.convert(A + ZWJ + NA, "Cyrl")

    assert result == Conversion(A + ZWJ + NA, 0, 1)


# ---- Target and odd text ---------------------------------------------------------


@pytest.mark.parametrize("target", ["Latn", "mong", "", "Mong "])
@pytest.mark.parametrize("text", ["", "монгол"])
def test_an_unknown_target_raises_a_config_error(text: str, target: str) -> None:
    converter, translator = _to_script()

    with pytest.raises(ConfigError, match="target"):
        converter.convert(text, target)  # type: ignore[arg-type]

    assert translator.calls == []


@pytest.mark.parametrize("target", ["Mong", "Cyrl"])
def test_odd_text_never_raises(target: str) -> None:
    converter = ScriptConverter.from_parts(
        StubTranslator(lambda word: A + NA), StubProcessor()
    )
    odd = (
        f"монгол{chr(0xD800)}\x00{chr(0x2028)}{ZWJ}{NNBSP}{NNBSP}{FVS_FIRST}{A}{NNBSP}"
    )

    result = converter.convert(odd, target)  # type: ignore[arg-type]

    assert isinstance(result, Conversion)


# ---- Loading ---------------------------------------------------------------------


class _TranslatorFactory:
    """Stands in for `ctranslate2.Translator`: records how it was built."""

    def __init__(self, failure: Exception | None = None) -> None:
        self.failure = failure
        self.built: list[tuple[str, dict[str, Any]]] = []
        self.translator = StubTranslator(TO_SCRIPT_ANSWERS.__getitem__)

    def __call__(self, model_path: str, **kwargs: Any) -> StubTranslator:
        if self.failure is not None:
            raise self.failure
        self.built.append((model_path, kwargs))
        return self.translator


@pytest.fixture
def factory(monkeypatch: pytest.MonkeyPatch) -> _TranslatorFactory:
    stub = _TranslatorFactory()
    monkeypatch.setattr(runtime.ctranslate2, "Translator", stub)
    return stub


def test_the_constructor_loads_a_single_thread_cpu_translator_and_converts(
    factory: _TranslatorFactory, spm_model: Path, tmp_path: Path
) -> None:
    converter = ScriptConverter(tmp_path / "ct2", spm_model)

    result = converter.convert("монгол", "Mong")

    assert factory.built == [
        (str(tmp_path / "ct2"), {"device": "cpu", "intra_threads": 1})
    ]
    assert result.text == MONGOL_SCRIPT


def test_the_device_is_passed_to_the_translator(
    factory: _TranslatorFactory, spm_model: Path, tmp_path: Path
) -> None:
    ScriptConverter(tmp_path, spm_model, device="cuda")

    assert factory.built[0][1]["device"] == "cuda"


def test_a_missing_model_directory_is_a_data_error_chained_to_the_library_error(
    spm_model: Path, tmp_path: Path
) -> None:
    with pytest.raises(DataError, match="ct2") as raised:
        ScriptConverter(tmp_path / "ct2", spm_model)

    assert isinstance(raised.value.__cause__, RuntimeError)


def test_a_translator_that_fails_to_load_is_a_data_error_with_its_cause(
    monkeypatch: pytest.MonkeyPatch, spm_model: Path, tmp_path: Path
) -> None:
    failure = RuntimeError("no model.bin")
    monkeypatch.setattr(runtime.ctranslate2, "Translator", _TranslatorFactory(failure))

    with pytest.raises(DataError, match="no model.bin") as raised:
        ScriptConverter(tmp_path, spm_model)

    assert raised.value.__cause__ is failure


def test_a_missing_character_model_is_a_data_error_chained_to_the_library_error(
    factory: _TranslatorFactory, tmp_path: Path
) -> None:
    with pytest.raises(DataError, match="missing.model") as raised:
        ScriptConverter(tmp_path, tmp_path / "missing.model")

    assert isinstance(raised.value.__cause__, OSError)


def test_a_directory_as_the_character_model_is_a_data_error(
    factory: _TranslatorFactory, tmp_path: Path
) -> None:
    with pytest.raises(DataError, match="directory"):
        ScriptConverter(tmp_path, tmp_path)
