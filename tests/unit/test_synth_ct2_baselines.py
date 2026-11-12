"""`CT2Translator`: the tokens it feeds CTranslate2 for NLLB and MADLAD, what it
makes of the hypotheses, and how it loads its two models. No real model is
used: the translator and the SentencePiece processor are stubs that record
their calls, and `ctranslate2` itself is made unimportable for every test, so a
test that reached the real library would fail instead of loading it."""

from __future__ import annotations

import re
import subprocess
import sys
import types
from pathlib import Path

import pytest

from rinchen.config.synth import Ct2Config
from rinchen.errors import ConfigError, DataError, UpstreamError
from rinchen.synth import ct2_baselines as ct2_module
from rinchen.synth.ct2_baselines import MADLAD_CODES, NLLB_CODES, CT2Translator
from rinchen.synth.prompts import supported_directions

ENG, KHK, RUS = "eng_Latn", "khk_Cyrl", "rus_Cyrl"
SP = chr(0x2581)

# The nine registry codes a generator can write, and the tag MADLAD wants in
# front of the text for each. Written out so a change to the tables is a change
# to a literal here as well.
NINE_CODES = [
    "eng_Latn",
    "rus_Cyrl",
    "zho_Hans",
    "kor_Hang",
    "jpn_Jpan",
    "deu_Latn",
    "kaz_Cyrl",
    "tur_Latn",
    "khk_Cyrl",
]
MADLAD_TAGS = {
    "eng_Latn": "<2en>",
    "rus_Cyrl": "<2ru>",
    "zho_Hans": "<2zh>",
    "kor_Hang": "<2ko>",
    "jpn_Jpan": "<2ja>",
    "deu_Latn": "<2de>",
    "kaz_Cyrl": "<2kk>",
    "tur_Latn": "<2tr>",
    "khk_Cyrl": "<2mn>",
}


@pytest.fixture(autouse=True)
def real_ctranslate2_is_unimportable(monkeypatch):
    monkeypatch.setitem(sys.modules, "ctranslate2", None)


class WordProcessor:
    """One piece per space-separated word, each with the dummy prefix, so what a
    text and a tag encode to is readable in a literal. Records every encode.

    A simplification: real SentencePiece with a user-defined symbol tokenises
    "<2mn> text" as [SP, "<2mn>", SP, ...] (SP is the dummy prefix, U+2581),
    not as one SP + "<2mn>" piece."""

    def __init__(self):
        self.encoded: list[str] = []

    def encode(self, text, return_type=str):
        self.encoded.append(text)
        return [SP + word for word in text.split(" ")]

    def decode(self, tokens):
        return "".join(tokens).replace(SP, " ").removeprefix(" ")


class Result:
    """The shape of a `ctranslate2.TranslationResult` this code reads."""

    def __init__(self, hypotheses):
        self.hypotheses = hypotheses


class StubTranslator:
    """Answers every source line with `answer(tokens)`, a list of hypotheses, and
    records each call."""

    def __init__(self, answer=lambda tokens: [[SP + "x"]]):
        self.answer = answer
        self.calls: list[tuple[list[list[str]], dict]] = []

    def translate_batch(self, source, **kwargs):
        self.calls.append((source, kwargs))
        return [Result(self.answer(tokens)) for tokens in source]


class FixedResults:
    def __init__(self, results):
        self.results = results
        self.calls = 0

    def translate_batch(self, source, **kwargs):
        self.calls += 1
        return self.results


def _config(family="nllb", **fields):
    fields.setdefault("model_dir", Path("/models/ct2"))
    fields.setdefault("spm_model", Path("/models/ct2/sentencepiece.model"))
    return Ct2Config(family=family, **fields)


def _translator(family="nllb", answer=None, **fields):
    stub = StubTranslator(answer) if answer else StubTranslator()
    processor = WordProcessor()
    translator = CT2Translator(
        _config(family, **fields), translator=stub, processor=processor
    )
    return translator, stub, processor


def _only_call(stub):
    (call,) = stub.calls
    return call


def test_translate_takes_its_arguments_by_the_protocol_names():
    translator, _, _ = _translator()

    result = translator.translate(texts=["a"], src_lang=ENG, tgt_lang=KHK)

    assert result == [["x"]]


# -- the tables


def _registry_codes():
    return {code for pair in supported_directions() for code in pair}


def test_the_nllb_table_has_exactly_the_languages_a_job_can_name():
    assert set(NLLB_CODES) == _registry_codes()


def test_the_madlad_table_has_exactly_the_languages_a_job_can_name():
    assert set(MADLAD_CODES) == _registry_codes()


def test_the_literal_codes_of_these_tests_are_the_languages_a_job_can_name():
    assert set(NINE_CODES) == _registry_codes()
    assert set(MADLAD_TAGS) == _registry_codes()


@pytest.mark.parametrize("family", ["nllb", "madlad"])
@pytest.mark.parametrize(
    ("src_lang", "tgt_lang", "named"),
    [
        ("mn_Mong", KHK, "mn_Mong"),
        (ENG, "mn_Mong", "mn_Mong"),
        ("fra_Latn", KHK, "fra_Latn"),
        (ENG, "xx", "xx"),
    ],
)
def test_a_language_outside_the_table_raises_naming_it_and_the_family(
    family, src_lang, tgt_lang, named
):
    translator, stub, _ = _translator(family)

    with pytest.raises(ConfigError) as excinfo:
        translator.translate(["a"], src_lang, tgt_lang)

    assert named in str(excinfo.value)
    assert family in str(excinfo.value)
    assert stub.calls == []


def test_a_bad_direction_raises_even_when_there_is_no_text():
    translator, stub, _ = _translator("nllb")

    with pytest.raises(ConfigError, match="mn_Mong"):
        translator.translate([], "mn_Mong", KHK)

    assert stub.calls == []


# -- NLLB tokens


def test_nllb_source_is_the_source_code_then_the_pieces_then_end_of_sentence():
    translator, stub, processor = _translator("nllb")

    translator.translate(["hello world"], ENG, KHK)

    source, _ = _only_call(stub)
    assert source == [["eng_Latn", SP + "hello", SP + "world", "</s>"]]
    assert processor.encoded == ["hello world"]


def test_nllb_forces_the_target_code_as_a_prefix_of_every_text():
    translator, stub, _ = _translator("nllb")

    translator.translate(["a", "b", "c"], KHK, RUS)

    source, kwargs = _only_call(stub)
    assert [line[0] for line in source] == ["khk_Cyrl"] * 3
    assert kwargs["target_prefix"] == [["rus_Cyrl"], ["rus_Cyrl"], ["rus_Cyrl"]]


@pytest.mark.parametrize("code", NINE_CODES)
def test_nllb_starts_the_source_with_the_source_language_code(code):
    translator, stub, _ = _translator("nllb")
    other = ENG if code != ENG else KHK

    translator.translate(["a"], code, other)

    source, _ = _only_call(stub)
    assert source == [[code, SP + "a", "</s>"]]


@pytest.mark.parametrize("code", NINE_CODES)
def test_nllb_prefixes_the_target_with_the_target_language_code(code):
    translator, stub, _ = _translator("nllb")
    other = ENG if code != ENG else KHK

    translator.translate(["a"], other, code)

    _, kwargs = _only_call(stub)
    assert kwargs["target_prefix"] == [[code]]


def test_nllb_removes_the_leading_target_code_from_a_hypothesis():
    translator, _, _ = _translator(
        "nllb", answer=lambda tokens: [["khk_Cyrl", SP + "сайн", SP + "байна"]]
    )

    assert translator.translate(["hello"], ENG, KHK) == [["сайн байна"]]


def test_nllb_keeps_a_hypothesis_that_does_not_start_with_the_target_code():
    translator, _, _ = _translator("nllb", answer=lambda tokens: [[SP + "сайн"]])

    assert translator.translate(["hello"], ENG, KHK) == [["сайн"]]


def test_nllb_removes_only_the_leading_token_and_only_the_target_code():
    translator, _, _ = _translator(
        "nllb",
        answer=lambda tokens: [
            ["khk_Cyrl", SP + "a", "khk_Cyrl"],
            ["rus_Cyrl", SP + "b"],
        ],
    )

    assert translator.translate(["hello"], ENG, KHK) == [["akhk_Cyrl", "rus_Cyrl b"]]


def test_a_hypothesis_that_is_only_the_target_code_is_dropped():
    translator, _, _ = _translator(
        "nllb", answer=lambda tokens: [["khk_Cyrl"], ["khk_Cyrl", SP + "a"]]
    )

    assert translator.translate(["hello"], ENG, KHK) == [["a"]]


# -- MADLAD tokens


def test_madlad_encodes_the_target_tag_together_with_the_text():
    translator, stub, processor = _translator("madlad")

    translator.translate(["hello world"], ENG, KHK)

    source, _ = _only_call(stub)
    assert processor.encoded == ["<2mn> hello world"]
    assert source == [[SP + "<2mn>", SP + "hello", SP + "world", "</s>"]]


@pytest.mark.parametrize(("code", "tag"), MADLAD_TAGS.items())
def test_madlad_tags_each_target_language_with_its_own_tag(code, tag):
    translator, stub, processor = _translator("madlad")
    other = ENG if code != ENG else KHK

    translator.translate(["ab"], other, code)

    source, _ = _only_call(stub)
    assert processor.encoded == [f"{tag} ab"]
    assert source == [[SP + tag, SP + "ab", "</s>"]]


def test_madlad_gives_one_line_per_text_each_with_its_own_text_encoded():
    translator, stub, processor = _translator("madlad")

    translator.translate(["a b", "c"], RUS, ENG)

    source, _ = _only_call(stub)
    assert processor.encoded == ["<2en> a b", "<2en> c"]
    assert source == [
        [SP + "<2en>", SP + "a", SP + "b", "</s>"],
        [SP + "<2en>", SP + "c", "</s>"],
    ]


def test_madlad_does_not_use_the_source_language_in_the_tokens():
    first, first_stub, _ = _translator("madlad")
    second, second_stub, _ = _translator("madlad")

    first.translate(["a"], ENG, KHK)
    second.translate(["a"], RUS, KHK)

    assert _only_call(first_stub)[0] == _only_call(second_stub)[0]


def test_madlad_sends_no_target_prefix_at_all():
    translator, stub, _ = _translator("madlad")

    translator.translate(["a"], ENG, KHK)

    _, kwargs = _only_call(stub)
    assert "target_prefix" not in kwargs


def test_madlad_removes_nothing_from_a_hypothesis():
    translator, _, _ = _translator(
        "madlad", answer=lambda tokens: [["khk_Cyrl", SP + "сайн"]]
    )

    assert translator.translate(["hello"], ENG, KHK) == [["khk_Cyrl сайн"]]


# -- the call to the library


def test_the_search_and_batch_options_come_from_the_config():
    translator, stub, _ = _translator(
        "nllb",
        beam_size=5,
        n=3,
        batch_size=7,
        max_tokens=9,
        temperature=1.5,
        top_p=0.3,
    )

    translator.translate(["a", "b"], ENG, KHK)

    _, kwargs = _only_call(stub)
    assert kwargs == {
        "target_prefix": [["khk_Cyrl"], ["khk_Cyrl"]],
        "beam_size": 5,
        "num_hypotheses": 3,
        "max_batch_size": 7,
        "max_decoding_length": 9,
        "return_scores": False,
    }


def test_madlad_passes_the_same_options_without_a_prefix():
    translator, stub, _ = _translator(
        "madlad", beam_size=6, n=2, batch_size=3, max_tokens=33
    )

    translator.translate(["a"], ENG, KHK)

    _, kwargs = _only_call(stub)
    assert kwargs == {
        "beam_size": 6,
        "num_hypotheses": 2,
        "max_batch_size": 3,
        "max_decoding_length": 33,
        "return_scores": False,
    }


@pytest.mark.parametrize("family", ["nllb", "madlad"])
def test_the_default_config_allows_256_output_tokens(family):
    translator, stub, _ = _translator(family)

    translator.translate(["a"], ENG, KHK)

    _, kwargs = _only_call(stub)
    assert kwargs["max_decoding_length"] == 256


@pytest.mark.parametrize("family", ["nllb", "madlad"])
@pytest.mark.parametrize("max_tokens", [1, 512])
def test_the_longest_output_is_max_tokens_not_the_library_default(family, max_tokens):
    translator, stub, _ = _translator(family, max_tokens=max_tokens)

    translator.translate(["a"], ENG, KHK)

    _, kwargs = _only_call(stub)
    assert kwargs["max_decoding_length"] == max_tokens


@pytest.mark.parametrize("family", ["nllb", "madlad"])
def test_a_text_that_cannot_be_encoded_names_its_index_before_any_work(family):
    translator, stub, processor = _translator(family)

    with pytest.raises(DataError, match="text 1 ") as raised:
        translator.translate(["fine", "bad" + chr(0xD800), "ok"], ENG, KHK)

    assert "bad" not in str(raised.value)
    assert isinstance(raised.value.__cause__, UnicodeEncodeError)
    assert stub.calls == []
    assert processor.encoded == []


def test_the_direction_is_checked_before_the_texts_are_encoded():
    translator, stub, _ = _translator("nllb")

    with pytest.raises(ConfigError, match="mn_Mong"):
        translator.translate(["bad" + chr(0xD800)], "mn_Mong", KHK)

    assert stub.calls == []


def test_all_texts_go_to_the_library_in_one_batch():
    translator, stub, _ = _translator("nllb", batch_size=2)

    translator.translate(["a", "b", "c", "d", "e"], ENG, KHK)

    source, _ = _only_call(stub)
    assert len(source) == 5


def test_an_empty_text_list_makes_no_call_and_returns_an_empty_list():
    translator, stub, processor = _translator("nllb")

    assert translator.translate([], ENG, KHK) == []

    assert stub.calls == []
    assert processor.encoded == []


# -- what comes back


def test_each_text_gets_its_own_hypotheses_in_n_best_order_and_input_order():
    translator, _, _ = _translator(
        "nllb",
        answer=lambda tokens: [
            [tokens[1] + "/best"],
            [tokens[1] + "/second"],
            [tokens[1] + "/third"],
        ],
        n=3,
    )

    result = translator.translate(["gamma", "alpha", "beta"], ENG, KHK)

    assert result == [
        ["gamma/best", "gamma/second", "gamma/third"],
        ["alpha/best", "alpha/second", "alpha/third"],
        ["beta/best", "beta/second", "beta/third"],
    ]


def test_candidates_are_decoded_with_the_processor():
    translator, _, _ = _translator(
        "madlad", answer=lambda tokens: [[SP + "one", SP + "two"]]
    )

    assert translator.translate(["a"], ENG, KHK) == [["one two"]]


def test_candidates_are_stripped_and_empty_or_symbol_only_ones_are_dropped():
    zero_width = chr(0x200B)
    translator, _, _ = _translator(
        "madlad",
        answer=lambda tokens: [
            [SP + "ab" + SP],
            [],
            [SP],
            [SP + "..."],
            [SP + zero_width],
            [SP + "cd"],
        ],
        n=6,
        beam_size=6,
    )

    assert translator.translate(["a"], ENG, KHK) == [["ab", "cd"]]


def test_duplicates_are_dropped_keeping_the_first_in_order():
    translator, _, _ = _translator(
        "madlad",
        answer=lambda tokens: [
            [SP + "b"],
            [SP + "a"],
            [SP + "b" + SP],
            [SP + "a"],
        ],
        n=4,
    )

    assert translator.translate(["x"], ENG, KHK) == [["b", "a"]]


def test_a_text_with_only_unusable_hypotheses_gets_an_empty_list():
    translator, _, _ = _translator(
        "madlad", answer=lambda tokens: [[SP], [SP + "!"]], n=2
    )

    assert translator.translate(["a", "b"], ENG, KHK) == [[], []]


def test_a_result_without_hypotheses_gives_an_empty_list_for_that_text():
    translator = CT2Translator(
        _config("madlad"),
        translator=FixedResults([Result([]), Result([[SP + "a"]])]),
        processor=WordProcessor(),
    )

    assert translator.translate(["a", "b"], ENG, KHK) == [[], ["a"]]


@pytest.mark.parametrize("returned", [0, 1, 3])
def test_a_wrong_number_of_results_raises_upstream_error(returned):
    results = [Result([[SP + "a"]]) for _ in range(returned)]
    translator = CT2Translator(
        _config("nllb"), translator=FixedResults(results), processor=WordProcessor()
    )

    with pytest.raises(UpstreamError, match=f"{returned} results for 2 texts"):
        translator.translate(["a", "b"], ENG, KHK)


@pytest.mark.parametrize("failure", [RuntimeError("out of memory"), ValueError("bad")])
def test_a_library_failure_is_wrapped_with_its_cause_chained(failure):
    class Failing:
        def translate_batch(self, source, **kwargs):
            raise failure

    translator = CT2Translator(
        _config("nllb"), translator=Failing(), processor=WordProcessor()
    )

    with pytest.raises(UpstreamError, match="out of memory|bad") as excinfo:
        translator.translate(["a"], ENG, KHK)

    assert excinfo.value.__cause__ is failure


# -- loading the two models


def _real_paths(tmp_path, family="nllb", **fields):
    model_dir = tmp_path / "ct2"
    model_dir.mkdir()
    return _config(
        family,
        model_dir=model_dir,
        spm_model=tmp_path / "sentencepiece.model",
        **fields,
    )


def test_a_missing_model_directory_raises_data_error_naming_it(tmp_path):
    cfg = _config("nllb", model_dir=tmp_path / "absent")

    with pytest.raises(DataError, match="absent"):
        CT2Translator(cfg, processor=WordProcessor())


def test_a_model_directory_that_is_a_file_raises_data_error(tmp_path):
    model_dir = tmp_path / "model.bin"
    model_dir.write_bytes(b"x")

    with pytest.raises(DataError, match=r"model\.bin"):
        CT2Translator(_config("nllb", model_dir=model_dir), processor=WordProcessor())


# The three tests below pass no translator: the model directory exists, so a
# translator loaded before the SentencePiece file would reach the (blocked)
# library import and raise ImportError instead of the DataError.


def test_a_missing_spm_file_raises_data_error_naming_it(tmp_path):
    cfg = _real_paths(tmp_path)

    with pytest.raises(DataError, match=r"sentencepiece\.model"):
        CT2Translator(cfg)


def test_an_spm_path_that_is_a_directory_raises_data_error(tmp_path):
    cfg = _real_paths(tmp_path)
    cfg.spm_model.mkdir()

    with pytest.raises(DataError, match="directory"):
        CT2Translator(cfg)


def test_an_spm_file_that_is_not_a_model_raises_data_error(tmp_path):
    cfg = _real_paths(tmp_path)
    cfg.spm_model.write_bytes(b"not a sentencepiece model")

    with pytest.raises(DataError, match=r"sentencepiece\.model"):
        CT2Translator(cfg)


def test_the_processor_is_loaded_from_the_configured_spm_file(tmp_path, monkeypatch):
    cfg = _real_paths(tmp_path)
    loaded = []

    def load(path):
        loaded.append(path)
        return WordProcessor()

    monkeypatch.setattr(ct2_module, "load_char_processor", load)
    translator = CT2Translator(cfg, translator=StubTranslator())

    assert loaded == [cfg.spm_model]
    assert translator.translate(["a b"], RUS, ENG) == [["x"]]


class FakeCtranslate2:
    """Stands in for the `ctranslate2` module: `Translator(...)` records how it
    was called and returns `stub`, or raises `error`."""

    def __init__(self, stub=None, error=None):
        self.stub = stub or StubTranslator()
        self.error = error
        self.created: list[tuple[tuple, dict]] = []

    def Translator(self, *args, **kwargs):
        self.created.append((args, kwargs))
        if self.error is not None:
            raise self.error
        return self.stub


def _install(monkeypatch, fake):
    monkeypatch.setitem(
        sys.modules, "ctranslate2", types.SimpleNamespace(Translator=fake.Translator)
    )


@pytest.mark.parametrize("device", ["cpu", "cuda", "auto"])
def test_the_library_translator_is_made_from_the_model_directory_and_device(
    tmp_path, monkeypatch, device
):
    cfg = _real_paths(tmp_path, device=device)
    fake = FakeCtranslate2()
    _install(monkeypatch, fake)

    translator = CT2Translator(cfg, processor=WordProcessor())

    assert fake.created == [((str(cfg.model_dir),), {"device": device})]
    assert translator.translate(["a"], ENG, KHK) == [["x"]]
    assert len(fake.stub.calls) == 1


def test_a_device_the_library_rejects_raises_config_error_with_the_cause(
    tmp_path, monkeypatch
):
    error = ValueError("unsupported device cuda")
    _install(monkeypatch, FakeCtranslate2(error=error))
    cfg = _real_paths(tmp_path, device="cuda")

    with pytest.raises(ConfigError, match="cuda") as excinfo:
        CT2Translator(cfg, processor=WordProcessor())

    assert excinfo.value.__cause__ is error


@pytest.mark.parametrize(
    "error", [RuntimeError("Unable to open file 'model.bin'"), OSError("gone")]
)
def test_a_model_the_library_cannot_load_raises_data_error_with_the_cause(
    tmp_path, monkeypatch, error
):
    _install(monkeypatch, FakeCtranslate2(error=error))
    cfg = _real_paths(tmp_path)

    with pytest.raises(DataError, match=re.escape(str(cfg.model_dir))) as excinfo:
        CT2Translator(cfg, processor=WordProcessor())

    assert excinfo.value.__cause__ is error


@pytest.mark.subprocess
def test_importing_the_module_does_not_import_ctranslate2():
    code = (
        "import sys\n"
        "import rinchen.synth.ct2_baselines\n"
        "assert 'ctranslate2' not in sys.modules, 'ctranslate2 was imported'\n"
    )

    result = subprocess.run(
        [sys.executable, "-I", "-c", code],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )

    assert result.returncode == 0, result.stderr
