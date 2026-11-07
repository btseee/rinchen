"""`ScriptConverter` over the exported tiny model: the same checkpoint Eole decoded,
now through CTranslate2 and the runtime wrapper.

Every test uses the session fixtures of `conftest.py`, so the model is trained and
exported once. Each test builds its own converter, so its cache and its counts are its
own. The measured numbers go to the test report as properties (`word_exact_match`,
`rule_fallbacks`, `invalid_outputs`, ...), not to the output.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Literal

import pytest

from rinchen.script_conv.alphabet import JOINERS, NNBSP
from rinchen.script_conv.eval import cer
from rinchen.script_conv.runtime import ScriptConverter
from rinchen.script_conv.segments import segment
from rinchen.script_conv.validate import is_script_word
from tests.support.tiny_converter import TinyRun

MIN_WORD_EXACT_MATCH = 0.90
MIN_AGREEMENT_WITH_EOLE = 0.99
MAX_CER = 0.15
SCRIPT_COMMA = chr(0x1802)
SCRIPT_FULL_STOP = chr(0x1803)

RecordProperty = Callable[[str, object], None]


class Converted:
    """What converting a list of words gave: the words back and the counts summed."""

    def __init__(self) -> None:
        self.words: list[str] = []
        self.rule_fallbacks = 0
        self.invalid_outputs = 0


def _convert_in_groups(
    converter: ScriptConverter,
    words: Sequence[str],
    target: Literal["Mong", "Cyrl"],
    *,
    words_per_call: int,
) -> Converted:
    """Convert `words`, `words_per_call` of them joined by U+0020 in each call. The
    output is split on U+0020 again; a group that does not come back as as many words
    is recorded as empty words, so it counts against the exact-match rate."""
    converted = Converted()
    for start in range(0, len(words), words_per_call):
        group = words[start : start + words_per_call]
        result = converter.convert(" ".join(group), target)
        pieces = result.text.split(" ")
        converted.words.extend(
            pieces if len(pieces) == len(group) else [""] * len(group)
        )
        converted.rule_fallbacks += result.rule_fallbacks
        converted.invalid_outputs += result.invalid_outputs
    return converted


@pytest.fixture
def converter(tiny_run: TinyRun, tiny_ct2: Path) -> ScriptConverter:
    return ScriptConverter(tiny_ct2, tiny_run.spm_model)


def _exact_share(references: Sequence[str], outputs: Sequence[str]) -> float:
    return sum(
        reference == output
        for reference, output in zip(references, outputs, strict=True)
    ) / len(references)


@pytest.mark.subprocess
@pytest.mark.parametrize("words_per_call", [1, 3])
def test_training_words_convert_to_their_reference_script_spelling(
    tiny_run: TinyRun,
    converter: ScriptConverter,
    record_property: RecordProperty,
    words_per_call: int,
) -> None:
    sources = tiny_run.lines("cyrl2mong.train.src")
    references = tiny_run.lines("cyrl2mong.train.tgt")

    converted = _convert_in_groups(
        converter, sources, "Mong", words_per_call=words_per_call
    )

    exact = _exact_share(references, converted.words)
    record_property(f"word_exact_match_{words_per_call}_per_call", exact)
    record_property(
        f"rule_fallbacks_{words_per_call}_per_call", converted.rule_fallbacks
    )
    record_property(
        f"invalid_outputs_{words_per_call}_per_call", converted.invalid_outputs
    )
    assert exact >= MIN_WORD_EXACT_MATCH, f"exact match {exact:.4f}"
    assert not any(joiner in word for word in converted.words for joiner in JOINERS)
    assert all(is_script_word(word) for word in converted.words)


@pytest.mark.subprocess
@pytest.mark.parametrize(
    ("direction", "target"), [("cyrl2mong", "Mong"), ("mong2cyrl", "Cyrl")]
)
def test_the_runtime_agrees_with_eole_on_the_same_checkpoint(
    tiny_run: TinyRun,
    converter: ScriptConverter,
    record_property: RecordProperty,
    direction: str,
    target: Literal["Mong", "Cyrl"],
) -> None:
    sources = tiny_run.lines(f"{direction}.train.src")
    eole_lines = tiny_run.predictions[direction]

    converted = _convert_in_groups(converter, sources, target, words_per_call=1)

    agreement = _exact_share(eole_lines, converted.words)
    record_property(f"agreement_with_eole_{direction}", agreement)
    assert agreement >= MIN_AGREEMENT_WITH_EOLE, f"agreement {agreement:.4f}"


def _words_the_model_reproduces(
    tiny_run: TinyRun, direction: str
) -> list[tuple[str, str]]:
    """(source, reference) of the training words of `direction` that Eole's prediction
    reproduces exactly."""
    return [
        (source, reference)
        for source, reference, prediction in zip(
            tiny_run.lines(f"{direction}.train.src"),
            tiny_run.lines(f"{direction}.train.tgt"),
            tiny_run.predictions[direction],
            strict=True,
        )
        if prediction == reference
    ]


def _reproduced_words_with_a_suffix_gap(
    tiny_run: TinyRun, direction: str, gap_side: Literal["source", "reference"]
) -> list[tuple[str, str]]:
    """(source, reference) of the training words of `direction` whose `gap_side`
    holds an NNBSP and that Eole's prediction reproduces exactly.

    Fails, never skips: the fixture must hold such words, and the trained model must
    know at least one of them, or there is no right answer for the runtime to be held
    to. The check is on Eole's prediction, not on the runtime's, so a model that
    simply lost every gap word is reported as that, not as a runtime defect.
    """
    rows = list(
        zip(
            tiny_run.lines(f"{direction}.train.src"),
            tiny_run.lines(f"{direction}.train.tgt"),
            tiny_run.predictions[direction],
            strict=True,
        )
    )
    with_gap = [
        (source, reference, prediction)
        for source, reference, prediction in rows
        if NNBSP in (source if gap_side == "source" else reference)
    ]
    assert with_gap, f"the {direction} training fixture has no word with a suffix gap"
    reproduced = [
        (source, reference)
        for source, reference, prediction in with_gap
        if prediction == reference
    ]
    assert reproduced, (
        f"Eole reproduced none of the {len(with_gap)} {direction} words "
        "with a suffix gap"
    )
    return reproduced


@pytest.mark.subprocess
def test_a_script_word_with_a_suffix_gap_comes_back_as_one_word_with_its_gap(
    tiny_run: TinyRun, converter: ScriptConverter
) -> None:
    source, reference = _reproduced_words_with_a_suffix_gap(
        tiny_run, "cyrl2mong", "reference"
    )[0]

    alone = converter.convert(source, "Mong")
    in_a_line = converter.convert(f"{source} {source}", "Mong")

    assert alone.text == reference
    assert in_a_line.text == f"{reference} {reference}"
    assert [piece.kind for piece in segment(alone.text)] == ["mongolian"]


@pytest.mark.subprocess
def test_a_script_word_with_a_suffix_gap_is_converted_back_as_one_word(
    tiny_run: TinyRun, converter: ScriptConverter
) -> None:
    script, cyrillic = _reproduced_words_with_a_suffix_gap(
        tiny_run, "mong2cyrl", "source"
    )[0]

    result = converter.convert(f"{script} {script}", "Cyrl")

    assert result.text == f"{cyrillic} {cyrillic}"
    assert (result.rule_fallbacks, result.invalid_outputs) == (0, 0)


@pytest.mark.subprocess
def test_script_words_convert_back_to_cyrillic_with_a_low_character_error_rate(
    tiny_run: TinyRun, converter: ScriptConverter, record_property: RecordProperty
) -> None:
    sources = tiny_run.lines("mong2cyrl.train.src")
    references = tiny_run.lines("mong2cyrl.train.tgt")

    converted = _convert_in_groups(converter, sources, "Cyrl", words_per_call=1)

    error_rate = cer(references, converted.words)
    record_property("reverse_cer", error_rate)
    record_property("reverse_invalid_outputs", converted.invalid_outputs)
    assert converted.rule_fallbacks == 0
    assert error_rate <= MAX_CER, f"CER {error_rate:.4f}"


@pytest.mark.subprocess
def test_a_mixed_line_converts_its_words_and_marks_and_keeps_the_numbers(
    tiny_run: TinyRun, tiny_ct2: Path, converter: ScriptConverter
) -> None:
    reproduced = [
        (source, reference)
        for source, reference in _words_the_model_reproduces(tiny_run, "cyrl2mong")
        if converter.convert(source, "Mong").text == reference
    ]
    assert len(reproduced) >= 2, "fewer than two words are reproduced by both runtimes"
    (first_source, first), (second_source, second) = reproduced[:2]
    fresh = ScriptConverter(tiny_ct2, tiny_run.spm_model)

    result = fresh.convert(f"{first_source} 3.5, {second_source}.", "Mong")

    assert result.text == f"{first} 3.5{SCRIPT_COMMA} {second}{SCRIPT_FULL_STOP}"
    assert (result.rule_fallbacks, result.invalid_outputs) == (0, 0)
