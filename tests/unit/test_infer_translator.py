"""`rinchen.infer.translator.Translator` (D11, ruling T14): the facade over one engine. A text
gives a `Translation`, a sequence a list, the keywords reach the engine, and the context
manager closes it. The engine under it is the scripted one: `Engine.open` is made to open the
fabricated export with the stand-ins of the library and of SentencePiece.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from rinchen.errors import ConfigError
from rinchen.infer import translator as translator_module
from rinchen.infer.engine import Engine
from rinchen.infer.engine_types import DocumentResult, LoadOptions, Translation
from rinchen.infer.translator import Translator
from tests.support.engines import (
    ENG,
    ENGINE_PIECES,
    ENGINE_TOKENS,
    KHK,
    SP,
    LiveCt2,
    copy_windows,
    engine_environment,  # noqa: F401 - the fixture of `pytestmark`
)
from tests.support.fake_ct2 import FakeProcessor
from tests.support.model_exports import write_model_export

pytestmark = pytest.mark.usefixtures("engine_environment")


class Opened:
    """What `Engine.open` was called with, and the library it opened over."""

    def __init__(self) -> None:
        self.calls: list[tuple[Path, LoadOptions, Any]] = []
        self.ct2 = LiveCt2(copy_windows)


@pytest.fixture
def opened(monkeypatch: pytest.MonkeyPatch) -> Opened:
    record = Opened()
    real_open = Engine.open

    def open_with_stand_ins(
        directory: Path,
        load: LoadOptions | None = None,
        *,
        script_converter: Any = None,
        translator: Any = None,
        processor: Any = None,
    ) -> Engine:
        load = LoadOptions() if load is None else load
        record.calls.append((directory, load, script_converter))
        return real_open(
            directory,
            load,
            script_converter=script_converter,
            translator=record.ct2,
            processor=FakeProcessor(ENGINE_PIECES),
        )

    monkeypatch.setattr(Engine, "open", staticmethod(open_with_stand_ins))
    return record


def model(tmp_path: Path, **options: Any) -> Path:
    return write_model_export(
        tmp_path / "model", tokens=ENGINE_TOKENS, **options
    ).directory


def test_a_text_gives_a_translation_and_a_sequence_a_list(
    tmp_path: Path, opened: Opened
) -> None:
    with Translator(model(tmp_path)) as translator:
        one = translator.translate("hello", tgt=ENG)
        several = translator.translate(["hello", "world"], tgt=ENG)
        none = translator.translate([], tgt=ENG)

    assert isinstance(one, Translation) and one.text == "hello"
    assert [t.text for t in several] == ["hello", "world"]
    assert none == []


def test_the_keywords_reach_the_engine(tmp_path: Path, opened: Opened) -> None:
    with Translator(model(tmp_path)) as translator:
        translator.translate(
            "hello",
            tgt=ENG,
            src=KHK,
            beam_size=2,
            n=2,
            max_decoding_length=64,
            max_source_pieces=9,
            script_mask="always",
        )

    (call,) = opened.ct2.calls
    assert call.options["beam_size"] == 2
    assert call.options["num_hypotheses"] == 2
    assert call.options["max_decoding_length"] == 64
    # The list of a Latin target: the pieces of the other scripts are in it.
    assert [SP + "Энэ"] in call.options["suppress_sequences"]


def test_the_source_language_and_the_script_reach_the_engine(
    tmp_path: Path, opened: Opened
) -> None:
    with Translator(model(tmp_path)) as translator:
        result = translator.translate("hello", tgt=ENG, src=KHK)
        with pytest.raises(ConfigError, match="Mongolian"):
            translator.translate("hello", tgt=ENG, script="Cyrl")

    assert isinstance(result, Translation)
    assert (result.src, result.tgt) == (KHK, ENG)


def test_the_source_language_and_the_script_reach_the_engine_for_a_sequence_too(
    tmp_path: Path, opened: Opened
) -> None:
    with Translator(model(tmp_path)) as translator:
        results = translator.translate(["hello"], tgt=ENG, src=KHK)
        with pytest.raises(ConfigError, match="Mongolian"):
            translator.translate(["hello"], tgt=ENG, script="Cyrl")

    assert [(r.src, r.tgt) for r in results] == [(KHK, ENG)]


def test_the_load_options_reach_the_engine_and_the_directory_is_the_one_given(
    tmp_path: Path, opened: Opened
) -> None:
    directory = model(tmp_path)
    converter = object()

    # `cuda` is only named here: the engine is the stand-in of the fixture and no library loads.
    Translator(
        directory,
        device="cuda",
        compute_type="int8",
        inter_threads=2,
        intra_threads=3,
        script_converter=converter,  # type: ignore[arg-type]
    )

    ((seen, load, attached),) = opened.calls
    assert seen == directory
    assert load == LoadOptions(
        device="cuda", compute_type="int8", inter_threads=2, intra_threads=3
    )
    assert attached is converter


def test_a_refused_option_is_refused_before_the_engine_is_opened(
    tmp_path: Path, opened: Opened
) -> None:
    with pytest.raises(ConfigError):
        Translator(model(tmp_path), compute_type="auto")

    assert opened.calls == []


def test_the_defaults_are_cpu_one_thread_and_the_stored_type(
    tmp_path: Path, opened: Opened
) -> None:
    Translator(model(tmp_path))

    ((_, load, converter),) = opened.calls
    assert load == LoadOptions()
    assert converter is None


def test_the_context_manager_closes_the_engine(tmp_path: Path, opened: Opened) -> None:
    with Translator(model(tmp_path)) as translator:
        assert translator.translate("hello", tgt=ENG)

    assert opened.ct2.unloaded == 1
    with pytest.raises(ConfigError, match="closed"):
        translator.translate("hello", tgt=ENG)


def test_close_can_be_called_twice(tmp_path: Path, opened: Opened) -> None:
    translator = Translator(model(tmp_path))

    translator.close()
    translator.close()

    assert opened.ct2.unloaded == 1


def test_the_info_and_the_counters_are_the_engines(
    tmp_path: Path, opened: Opened
) -> None:
    with Translator(model(tmp_path)) as translator:
        translator.translate(["hello", ""], tgt=ENG)

        assert translator.info.targets == (ENG, KHK, "rus_Cyrl")
        assert translator.info.placement == "src"
        assert translator.counters()["sources"] == 2
        assert translator.counters()["empty"] == 1


def test_a_document_gives_a_document_result_and_documents_a_list(
    tmp_path: Path, opened: Opened
) -> None:
    directory = model(tmp_path, document_level=True)
    text = "Hello world. Hello book."

    with Translator(directory) as translator:
        one = translator.translate_document(text, tgt=KHK, src=ENG, script_mask="off")
        several = translator.translate_document(
            [text, ""], tgt=KHK, src=ENG, script_mask="off"
        )

    assert isinstance(one, DocumentResult)
    assert one.text == "Hello world. Hello book."
    assert [r.text for r in several] == ["Hello world. Hello book.", ""]


def test_the_keywords_of_a_document_reach_the_engine(
    tmp_path: Path, opened: Opened
) -> None:
    with Translator(model(tmp_path, document_level=True)) as translator:
        translator.translate_document(
            "Hello world. Hello book.",
            tgt=KHK,
            src=ENG,
            beam_size=2,
            max_decoding_length=32,
            script_mask="off",
        )

    (call,) = opened.ct2.calls
    assert call.options["beam_size"] == 2
    assert call.options["max_decoding_length"] == 32


def test_a_document_needs_its_source_language(tmp_path: Path, opened: Opened) -> None:
    translator = Translator(model(tmp_path, document_level=True))

    with pytest.raises(TypeError, match="src"):
        translator.translate_document("Hello.", tgt=KHK)  # type: ignore[call-overload]


def test_the_script_of_a_sequence_of_documents_reaches_the_engine(
    tmp_path: Path, opened: Opened
) -> None:
    translator = Translator(model(tmp_path, document_level=True))

    with pytest.raises(ConfigError, match="Mongolian"):
        translator.translate_document(
            ["Hello world."], tgt="rus_Cyrl", src=ENG, script="Cyrl"
        )


def test_the_facade_is_the_module_attribute_the_top_level_exports() -> None:
    assert translator_module.Translator is Translator
