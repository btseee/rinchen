"""`Engine` (D11, fix wave): a reserved symbol in the text of a request is refused.

SentencePiece keeps every reserved symbol of the project as one piece, wherever it stands in a
text, so a request that spells `<2eng_Latn>` hands the model a direction tag it did not ask for
(measured: asked for `khk_Cyrl`, "<2eng_Latn> This is the star." came back as Mongolian text of
another meaning and a purity of 1.0, the language silently changed), and `<sep>`, `<doc>` and
the structural tags steer the document model the same way. The engine refuses the request with
a `DataError` that names the row (or the document) and the symbol, never the text. The numbered
placeholders stay allowed (T7), and the `<sep>` and `<doc>` that the engine writes into a window
itself are not the request's text.

The processor is `SymbolProcessor`: it cuts a reserved symbol out as its own piece, as the real
SentencePiece model does.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import pytest

from rinchen.errors import DataError
from rinchen.infer.engine_types import ENGINE_COUNTERS, DecodeOptions, Row
from rinchen.spm.symbols import PLACEHOLDER_COUNT, user_defined_symbols
from tests.support.engines import (
    END,
    KHK,
    RUS,
    SP,
    ScriptedEngine,
    engine_environment,  # noqa: F401 - the fixture of `pytestmark`
    hyp,
    scripted_engine,
    symbol_processor,
)
from tests.support.fake_ct2 import Hypothesis

SECRET = "xylophone quartz"
SYMBOLS = [
    "<2eng_Latn>",
    "<2khk_Cyrl>",
    "<2mn_Mong>",
    "<sep>",
    "<doc>",
    "<bt>",
    "<ft>",
    "<t>",
    "</t>",
    "<r>",
    "</r>",
]
OFF = DecodeOptions(script_mask="off")
PLACEHOLDER = chr(0x27E6) + "1" + chr(0x27E7)

pytestmark = pytest.mark.usefixtures("engine_environment")


def engine(tmp_path: Path, **keywords: object) -> ScriptedEngine:
    return scripted_engine(tmp_path, processor=symbol_processor(), **keywords)  # type: ignore[arg-type]


def untouched(fixture: ScriptedEngine) -> None:
    """Nothing was sent to the library and nothing was counted."""
    assert fixture.ct2.calls == []
    assert fixture.engine.counters() == dict.fromkeys(ENGINE_COUNTERS, 0)


# -- rows


@pytest.mark.parametrize("symbol", SYMBOLS)
def test_a_row_with_a_reserved_symbol_is_refused_naming_the_row_and_not_the_text(
    tmp_path: Path, symbol: str
) -> None:
    fixture = engine(tmp_path)

    with pytest.raises(DataError) as excinfo:
        fixture.engine.translate(
            [Row("hello", None, KHK), Row(f"{SECRET} {symbol} {SECRET}", None, KHK)]
        )

    message = str(excinfo.value)
    assert "row 1" in message
    assert symbol in message
    assert "xylophone" not in message and "quartz" not in message
    untouched(fixture)


def test_a_reserved_symbol_inside_a_word_is_refused_too(tmp_path: Path) -> None:
    fixture = engine(tmp_path)

    with pytest.raises(DataError, match="row 0"):
        fixture.engine.translate([Row(f"{SECRET}<2eng_Latn>{SECRET}", None, KHK)])

    untouched(fixture)


def test_every_reserved_symbol_but_the_placeholders_is_refused_in_a_row(
    tmp_path: Path,
) -> None:
    fixture = engine(tmp_path)
    refused = user_defined_symbols()[:-PLACEHOLDER_COUNT]

    for symbol in refused:
        with pytest.raises(DataError, match="row 0"):
            fixture.engine.translate([Row(f"hello {symbol} world", None, KHK)])

    assert len(refused) > 10
    untouched(fixture)


def test_the_first_reserved_symbol_of_the_text_is_the_one_named(tmp_path: Path) -> None:
    fixture = engine(tmp_path)

    with pytest.raises(DataError) as excinfo:
        fixture.engine.translate([Row("hello <bt> world <2eng_Latn>", None, KHK)])

    assert "<bt>" in str(excinfo.value)
    assert "<2eng_Latn>" not in str(excinfo.value)


def test_a_row_that_is_also_too_long_is_refused_not_given_a_status(
    tmp_path: Path,
) -> None:
    fixture = engine(tmp_path)

    with pytest.raises(DataError, match="row 0"):
        fixture.engine.translate(
            [Row("a b c d <doc>", None, KHK)], DecodeOptions(max_source_pieces=2)
        )

    untouched(fixture)


def test_a_placeholder_is_text_the_model_may_read(tmp_path: Path) -> None:
    fixture = engine(tmp_path)

    (result,) = fixture.engine.translate(
        [Row(f"hello {PLACEHOLDER} world", None, KHK)], OFF
    )

    assert result.status == "ok"
    assert PLACEHOLDER in result.text
    assert fixture.ct2.calls[0].source == [
        ["<2khk_Cyrl>", SP + "hello", PLACEHOLDER, SP + "world"]
    ]


def test_text_that_only_looks_like_a_symbol_is_text(tmp_path: Path) -> None:
    fixture = engine(tmp_path)

    results = fixture.engine.translate(
        [
            Row("a <html> b", None, KHK),
            Row("x < 2eng_Latn > y", None, KHK),
            Row("<SEP> <Doc> </s> <unk>", None, KHK),
        ],
        OFF,
    )

    assert [result.status for result in results] == ["ok", "ok", "ok"]
    assert fixture.engine.counters()["sources"] == 3


def test_a_refused_row_does_not_stop_the_next_call(tmp_path: Path) -> None:
    fixture = engine(tmp_path)
    with pytest.raises(DataError):
        fixture.engine.translate([Row("hello <sep>", None, KHK)])

    (result,) = fixture.engine.translate([Row("hello", None, KHK)], OFF)

    assert result.status == "ok"


# -- documents

DOC = "Энэ ном. Энэ ус."


def documents(tmp_path: Path) -> ScriptedEngine:
    return engine(tmp_path, document_level=True, script=window_copy)


def window_copy(line: tuple[str, ...]) -> Sequence[Hypothesis]:
    """A window (`[tag, "<doc>", *pieces]`) is copied, its `<sep>` piece and all; a sentence
    is copied as the stand-in model does."""
    if len(line) > 1 and line[1] == "<doc>":
        return [hyp(*line[2:], END)]
    return [hyp(*line[1:], END)]


@pytest.mark.parametrize("symbol", SYMBOLS)
def test_a_document_with_a_reserved_symbol_is_refused_naming_the_document(
    tmp_path: Path, symbol: str
) -> None:
    fixture = documents(tmp_path)

    with pytest.raises(DataError) as excinfo:
        fixture.engine.translate_documents(
            [DOC, f"{SECRET} {symbol} {SECRET}."], KHK, RUS, OFF
        )

    message = str(excinfo.value)
    assert "document 1" in message
    assert symbol in message
    assert "xylophone" not in message and "quartz" not in message
    untouched(fixture)


def test_every_reserved_symbol_but_the_placeholders_is_refused_in_a_document(
    tmp_path: Path,
) -> None:
    fixture = documents(tmp_path)

    for symbol in user_defined_symbols()[:-PLACEHOLDER_COUNT]:
        with pytest.raises(DataError, match="document 0"):
            fixture.engine.translate_documents([f"Энэ {symbol} ном."], KHK, RUS, OFF)

    untouched(fixture)


def test_the_separator_and_the_doc_marker_the_engine_writes_are_not_the_requests_text(
    tmp_path: Path,
) -> None:
    fixture = documents(tmp_path)

    (result,) = fixture.engine.translate_documents([DOC], KHK, RUS, OFF)

    (window,) = fixture.ct2.calls
    assert window.source[0][:2] == ["<2rus_Cyrl>", "<doc>"]
    assert "<sep>" in window.source[0]
    assert [t.text for t in result.translations] == ["Энэ ном.", "Энэ ус."]


def test_a_placeholder_in_a_document_is_text(tmp_path: Path) -> None:
    fixture = documents(tmp_path)

    (result,) = fixture.engine.translate_documents(
        [f"Энэ {PLACEHOLDER} ном. Энэ ус."], KHK, RUS, OFF
    )

    assert [t.status for t in result.translations] == ["ok", "ok"]
