"""Which language a request is for (D11, ruling T8): the target and its tag, the `script`
option for Mongolian, the pairs the model was not trained for, and the script converter for a
model with no `mn_Mong` target. A request is all-or-nothing in its errors: one refused row
sends nothing.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from rinchen.errors import ConfigError, UpstreamError
from rinchen.infer.engine_types import ENGINE_COUNTERS, DecodeOptions, Row
from rinchen.script_conv.runtime import ScriptConverter
from tests.support.engines import (
    ENG,
    KHK,
    MON,
    MONGOLIAN,
    RUS,
    SP,
    TAG_ENG,
    TAG_KHK,
    TAG_MON,
    FakeConverter,
    ScriptedEngine,
    engine_environment,  # noqa: F401 - the fixture of `pytestmark`
    scripted_engine,
)
from tests.support.stub_runtime import StubProcessor, StubTranslator

OPTIONS = DecodeOptions(script_mask="off")
# The model that knows traditional script: both Mongolian targets.
BOTH = ("eng_Latn-khk_Cyrl", "eng_Latn-mn_Mong", "khk_Cyrl-eng_Latn")

pytestmark = pytest.mark.usefixtures("engine_environment")


def tags_sent(fixture: ScriptedEngine) -> list[str]:
    return [line[0] for call in fixture.ct2.calls for line in call.source]


# -- the target


def test_a_target_outside_the_registry_is_refused_naming_what_the_model_translates_into(
    tmp_path: Path,
) -> None:
    fixture = scripted_engine(tmp_path)

    with pytest.raises(ConfigError) as excinfo:
        fixture.engine.translate([Row("hello", None, "xx_Zzzz")], OPTIONS)

    assert "xx_Zzzz" in str(excinfo.value)
    for target in (ENG, KHK, RUS):
        assert target in str(excinfo.value)


def test_a_registry_language_the_model_was_not_trained_to_write_is_refused(
    tmp_path: Path,
) -> None:
    fixture = scripted_engine(tmp_path)

    with pytest.raises(ConfigError, match="deu_Latn"):
        fixture.engine.translate([Row("hello", None, "deu_Latn")], OPTIONS)


def test_the_same_language_as_source_and_target_is_refused(tmp_path: Path) -> None:
    fixture = scripted_engine(tmp_path)

    with pytest.raises(ConfigError, match=KHK):
        fixture.engine.translate([Row("hello", KHK, KHK)], OPTIONS)


def test_a_source_language_outside_the_registry_is_refused(tmp_path: Path) -> None:
    fixture = scripted_engine(tmp_path)

    with pytest.raises(ConfigError, match="xx_Zzzz.*language registry"):
        fixture.engine.translate([Row("hello", "xx_Zzzz", ENG)], OPTIONS)

    assert fixture.ct2.calls == []


def test_a_language_the_model_was_not_trained_to_read_is_a_hint_not_a_refusal(
    tmp_path: Path,
) -> None:
    # `deu_Latn` is in the registry and in no direction of the model.
    fixture = scripted_engine(tmp_path)

    (result,) = fixture.engine.translate([Row("hello", "deu_Latn", ENG)], OPTIONS)

    assert result.status == "ok" and result.zero_shot is True


def test_a_pair_the_model_was_not_trained_for_is_zero_shot_and_counted(
    tmp_path: Path,
) -> None:
    fixture = scripted_engine(tmp_path)

    result = fixture.engine.translate(
        [
            Row("hello", ENG, RUS),
            Row("hello", ENG, KHK),
            Row("hello", None, RUS),
            Row("", ENG, RUS),
        ],
        OPTIONS,
    )

    assert [r.zero_shot for r in result] == [True, False, False, True]
    assert fixture.engine.counters()["zero_shot"] == 2


def test_a_trained_pair_is_not_zero_shot(tmp_path: Path) -> None:
    fixture = scripted_engine(tmp_path)

    (result,) = fixture.engine.translate([Row("hello", RUS, KHK)], OPTIONS)

    assert result.zero_shot is False
    assert fixture.engine.counters()["zero_shot"] == 0


def test_a_request_with_one_refused_row_sends_nothing_and_counts_nothing(
    tmp_path: Path,
) -> None:
    fixture = scripted_engine(tmp_path)

    with pytest.raises(ConfigError):
        fixture.engine.translate(
            [Row("hello", None, ENG), Row("world", None, "deu_Latn")], OPTIONS
        )

    assert fixture.ct2.calls == []
    assert fixture.engine.counters() == dict.fromkeys(ENGINE_COUNTERS, 0)


def test_the_target_of_a_result_is_the_one_asked_for(tmp_path: Path) -> None:
    fixture = scripted_engine(tmp_path)

    result = fixture.engine.translate(
        [Row("hello", None, ENG), Row("hello", None, RUS)], OPTIONS
    )

    assert [r.tgt for r in result] == [ENG, RUS]
    assert tags_sent(fixture) == [TAG_ENG, "<2rus_Cyrl>"]


# -- the script option


def test_a_script_for_a_target_that_is_not_mongolian_is_refused(tmp_path: Path) -> None:
    fixture = scripted_engine(tmp_path)

    with pytest.raises(ConfigError, match="Mongolian"):
        fixture.engine.translate([Row("hello", None, ENG, "Cyrl")], OPTIONS)

    assert fixture.ct2.calls == []


@pytest.mark.parametrize("script", ["Latn", "cyrl", "", 5])
def test_a_script_that_is_neither_cyrl_nor_mong_is_refused(
    tmp_path: Path, script: object
) -> None:
    fixture = scripted_engine(tmp_path, directions=BOTH)

    with pytest.raises(ConfigError, match="Cyrl"):
        fixture.engine.translate([Row("hello", None, KHK, script)], OPTIONS)  # type: ignore[arg-type]


def test_mong_selects_the_traditional_script_target_when_the_model_has_one(
    tmp_path: Path,
) -> None:
    fixture = scripted_engine(tmp_path, directions=BOTH)

    (result,) = fixture.engine.translate([Row("hello", ENG, KHK, "Mong")], OPTIONS)

    assert tags_sent(fixture) == [TAG_MON]
    assert result.tgt == MON
    assert result.zero_shot is False


def test_cyrl_selects_the_cyrillic_target_even_when_mn_mong_was_asked_for(
    tmp_path: Path,
) -> None:
    fixture = scripted_engine(tmp_path, directions=BOTH)

    (result,) = fixture.engine.translate([Row("hello", ENG, MON, "Cyrl")], OPTIONS)

    assert tags_sent(fixture) == [TAG_KHK]
    assert result.tgt == KHK


def test_no_script_keeps_the_code_asked_for(tmp_path: Path) -> None:
    fixture = scripted_engine(tmp_path, directions=BOTH)

    result = fixture.engine.translate(
        [Row("hello", ENG, KHK), Row("hello", ENG, MON)], OPTIONS
    )

    assert tags_sent(fixture) == [TAG_KHK, TAG_MON]
    assert [r.tgt for r in result] == [KHK, MON]


def test_cyrl_for_a_model_with_no_cyrillic_target_is_refused(tmp_path: Path) -> None:
    fixture = scripted_engine(
        tmp_path, directions=("eng_Latn-mn_Mong", "mn_Mong-eng_Latn")
    )

    with pytest.raises(ConfigError, match=KHK):
        fixture.engine.translate([Row("hello", ENG, MON, "Cyrl")], OPTIONS)


# -- a model with no traditional script target and a converter


def cyrillic_only(tmp_path: Path, converter: FakeConverter | None) -> ScriptedEngine:
    return scripted_engine(tmp_path, converter=converter)


def test_mong_without_a_traditional_target_and_without_a_converter_is_refused(
    tmp_path: Path,
) -> None:
    fixture = cyrillic_only(tmp_path, None)

    with pytest.raises(ConfigError, match="converter"):
        fixture.engine.translate([Row("hello", ENG, KHK, "Mong")], OPTIONS)

    assert fixture.ct2.calls == []


def test_a_traditional_target_the_model_lacks_is_refused_without_a_converter(
    tmp_path: Path,
) -> None:
    fixture = cyrillic_only(tmp_path, None)

    with pytest.raises(ConfigError, match=MON):
        fixture.engine.translate([Row("hello", ENG, MON)], OPTIONS)


def test_mong_translates_into_cyrillic_and_converts_the_text_once(
    tmp_path: Path,
) -> None:
    converter = FakeConverter(prefix=">")
    fixture = cyrillic_only(tmp_path, converter)

    result = fixture.engine.translate(
        [Row("hello", ENG, KHK, "Mong"), Row("", ENG, KHK, "Mong")], OPTIONS
    )

    assert tags_sent(fixture) == [TAG_KHK]
    assert converter.calls == [("hello", "Mong")]
    assert [(r.text, r.status, r.tgt) for r in result] == [
        (">hello", "ok", MON),
        ("", "empty", MON),
    ]
    counters = fixture.engine.counters()
    assert counters["script_conversions"] == 1
    assert counters["converter_rule_fallbacks"] == 2
    assert counters["converter_invalid_outputs"] == 1


def test_a_traditional_target_the_model_lacks_is_converted_when_a_converter_is_attached(
    tmp_path: Path,
) -> None:
    converter = FakeConverter(prefix=">")
    fixture = cyrillic_only(tmp_path, converter)

    (result,) = fixture.engine.translate([Row("hello", ENG, MON)], OPTIONS)

    assert tags_sent(fixture) == [TAG_KHK]
    assert (result.text, result.tgt) == (">hello", MON)


def test_the_converter_is_not_used_when_the_model_has_the_traditional_target(
    tmp_path: Path,
) -> None:
    converter = FakeConverter()
    fixture = scripted_engine(tmp_path, directions=BOTH, converter=converter)

    fixture.engine.translate([Row("hello", ENG, KHK, "Mong")], OPTIONS)

    assert converter.calls == []
    assert tags_sent(fixture) == [TAG_MON]


def test_cyrillic_is_never_converted(tmp_path: Path) -> None:
    converter = FakeConverter()
    fixture = cyrillic_only(tmp_path, converter)

    fixture.engine.translate(
        [Row("hello", ENG, KHK), Row("hello", ENG, KHK, "Cyrl")], OPTIONS
    )

    assert converter.calls == []


def test_every_alternative_is_converted_with_the_text(tmp_path: Path) -> None:
    converter = FakeConverter(prefix=">")
    fixture = scripted_engine(
        tmp_path,
        converter=converter,
        script=lambda line: [([SP + "a", "</s>"], -1.0), ([SP + "b", "</s>"], -2.0)],
    )

    (result,) = fixture.engine.translate(
        [Row("hello", ENG, KHK, "Mong")], DecodeOptions(n=2, script_mask="off")
    )

    assert (result.text, result.alternatives) == (">a", (">b",))
    assert converter.calls == [("a", "Mong"), ("b", "Mong")]
    assert fixture.engine.counters()["script_conversions"] == 2


def test_the_purity_of_a_converted_text_is_that_of_the_traditional_script(
    tmp_path: Path,
) -> None:
    converter = FakeConverter(prefix=MONGOLIAN)
    fixture = cyrillic_only(tmp_path, converter)

    (result,) = fixture.engine.translate(
        [Row("hello", ENG, KHK, "Mong")], DecodeOptions(script_mask="off")
    )

    # The converter's prefix is Mongolian, the converted word Latin: 3 letters of 8.
    assert result.script_purity == pytest.approx(3 / 8)


def test_a_converter_that_fails_fails_the_request_and_counts_nothing(
    tmp_path: Path,
) -> None:
    class Failing(FakeConverter):
        def convert(self, text: str, target: str) -> SimpleNamespace:
            raise RuntimeError("no model")

    fixture = cyrillic_only(tmp_path, Failing())

    with pytest.raises(RuntimeError, match="no model"):
        fixture.engine.translate([Row("hello", ENG, KHK, "Mong")], OPTIONS)

    assert fixture.engine.counters() == dict.fromkeys(ENGINE_COUNTERS, 0)


class _NoHypothesis:
    hypotheses: tuple[list[str], ...] = ()


class _BrokenTranslator:
    """A CTranslate2 stand-in of the converter that fails in the two ways that put the word
    being converted in the message of D5's `UpstreamError`."""

    def __init__(self, how: str) -> None:
        self._how = how

    def translate_batch(
        self, source: list[list[str]], **kwargs: object
    ) -> list[object]:
        if self._how == "raises":
            raise RuntimeError("cannot decode the word " + "".join(source[0][2:]))
        return [_NoHypothesis() for _ in source]


@pytest.mark.parametrize("how", ["raises", "no_hypothesis"])
def test_a_failure_of_the_real_converter_is_an_upstream_error_that_does_not_quote_the_word(
    tmp_path: Path, how: str
) -> None:
    secret = "тусгаарлагдсан"
    converter = ScriptConverter.from_parts(_BrokenTranslator(how), StubProcessor())  # type: ignore[arg-type]
    fixture = scripted_engine(tmp_path, converter=converter)  # type: ignore[arg-type]

    with pytest.raises(UpstreamError) as raised:
        fixture.engine.translate([Row(secret, RUS, KHK, "Mong")], OPTIONS)

    assert str(raised.value) == "the script converter failed (UpstreamError)"
    assert isinstance(raised.value.__cause__, UpstreamError)
    assert secret in str(raised.value.__cause__)
    assert fixture.engine.counters() == dict.fromkeys(ENGINE_COUNTERS, 0)


def test_a_long_target_code_is_cut_in_the_message(tmp_path: Path) -> None:
    fixture = scripted_engine(tmp_path)

    with pytest.raises(ConfigError) as excinfo:
        fixture.engine.translate([Row("hello", None, "x" * 5000)], OPTIONS)

    assert len(str(excinfo.value)) < 400


def test_a_converter_does_not_make_a_language_the_model_does_not_write(
    tmp_path: Path,
) -> None:
    converter = FakeConverter()
    fixture = cyrillic_only(tmp_path, converter)

    with pytest.raises(ConfigError, match="deu_Latn"):
        fixture.engine.translate([Row("hello", ENG, "deu_Latn")], OPTIONS)

    assert converter.calls == [] and fixture.ct2.calls == []


def test_the_real_script_converter_of_d5_fits_what_the_engine_asks_of_one(
    tmp_path: Path,
) -> None:
    mongol = "".join(
        chr(code) for code in (0x182E, 0x1823, 0x1829, 0x182D, 0x1823, 0x182F)
    )
    converter = ScriptConverter.from_parts(
        StubTranslator({"монгол": mongol}.__getitem__), StubProcessor()
    )
    fixture = scripted_engine(tmp_path, converter=converter)  # type: ignore[arg-type]

    (result,) = fixture.engine.translate(
        [Row("монгол", RUS, KHK, "Mong")], DecodeOptions(script_mask="off")
    )

    assert (result.text, result.tgt, result.script_purity) == (mongol, MON, 1.0)
    assert fixture.engine.counters()["script_conversions"] == 1


# -- the request, checked before any text


def test_a_request_the_model_can_serve_passes_the_check_and_sends_nothing(
    tmp_path: Path,
) -> None:
    fixture = scripted_engine(tmp_path)

    fixture.engine.require_request(ENG, KHK)
    fixture.engine.require_request(None, KHK, "Cyrl")
    fixture.engine.require_request(None, RUS)

    assert fixture.ct2.calls == []
    assert fixture.engine.counters() == dict.fromkeys(ENGINE_COUNTERS, 0)


@pytest.mark.parametrize(
    ("src", "tgt", "script"),
    [
        (ENG, "deu_Latn", None),
        (ENG, "khk_Cyrll", None),
        (KHK, KHK, None),
        (ENG, RUS, "Mong"),
        (ENG, KHK, "Latn"),
        ("xxx_Latn", KHK, None),
        (ENG, MON, None),
    ],
)
def test_the_check_refuses_what_translate_refuses_with_the_same_message(
    tmp_path: Path, src: str, tgt: str, script: str | None
) -> None:
    fixture = scripted_engine(tmp_path)

    with pytest.raises(ConfigError) as from_the_check:
        fixture.engine.require_request(src, tgt, script)  # type: ignore[arg-type]
    with pytest.raises(ConfigError) as from_translate:
        fixture.engine.translate([Row("hello", src, tgt, script)], OPTIONS)  # type: ignore[arg-type]

    assert str(from_the_check.value) == str(from_translate.value)


def test_the_check_for_documents_needs_a_document_model_and_a_source(
    tmp_path: Path,
) -> None:
    sentences = scripted_engine(tmp_path / "sentences")
    documents = scripted_engine(tmp_path / "documents", document_level=True)

    with pytest.raises(ConfigError, match="does not translate documents"):
        sentences.engine.require_request(ENG, KHK, documents=True)
    with pytest.raises(ConfigError, match="needs its source language"):
        documents.engine.require_request(None, KHK, documents=True)
    with pytest.raises(ConfigError, match="does not translate into"):
        documents.engine.require_request(ENG, "deu_Latn", documents=True)
    documents.engine.require_request(ENG, KHK, documents=True)
    documents.engine.require_request(ENG, KHK)


def test_the_check_of_a_closed_engine_is_refused(tmp_path: Path) -> None:
    fixture = scripted_engine(tmp_path)
    fixture.engine.close()

    with pytest.raises(ConfigError, match="closed"):
        fixture.engine.require_request(ENG, KHK)
