"""The planted synthetic-data inputs describe themselves truthfully: the shape
of the monolingual frame, the scripted translator's call log and failures, the
fake scorer's pair lookup and the planted outcome of each row class."""

from __future__ import annotations

import math

import pytest

from rinchen.lang.scripts import Script, script_of
from rinchen.synth.select import Kind
from tests.support.planted_synth import (
    FakeScorer,
    ScriptedTranslator,
    planted_monolingual,
    planted_scorer,
    planted_script,
)

REAL_3 = "Монгол улс 003"


def test_the_planted_monolingual_frame_has_forty_cyrillic_rows_by_default() -> None:
    frame = planted_monolingual()

    assert frame.columns == ["id", "lang", "text"]
    assert frame.height == 40
    assert frame["id"][0] == "s000"
    assert frame["id"][39] == "s039"
    assert frame["text"][7] == "Монгол улс 007"
    assert frame["lang"].unique().to_list() == ["khk_Cyrl"]
    assert frame["text"].n_unique() == 40
    assert frame["id"].to_list() == sorted(frame["id"].to_list())
    assert all(script_of(text) == {Script.CYRILLIC: 1.0} for text in frame["text"])


def test_the_planted_monolingual_frame_takes_a_size_and_a_label() -> None:
    frame = planted_monolingual(3, lang="eng_Latn")

    assert frame["id"].to_list() == ["s000", "s001", "s002"]
    assert frame["lang"].to_list() == ["eng_Latn"] * 3


@pytest.mark.parametrize("size", [0, 1001])
def test_the_planted_monolingual_frame_rejects_a_size_outside_its_range(
    size: int,
) -> None:
    with pytest.raises(ValueError, match="between 1 and 1000"):
        planted_monolingual(size)


def test_the_planted_script_holds_the_documented_outcome_of_each_row_class() -> None:
    frame = planted_monolingual(8)

    script = planted_script(frame)

    assert list(script.values()) == [
        ["The state of Mongolia 000 stands", "The state of Mongolia 000"],
        [],
        ["", "Англи хэл 002"],
        [REAL_3],
        ["Mongolia is poor 004"],
        ["Mongolia is nan 005"],
        ["Mongolia is edge 006"],
        [
            "A completely different sentence 007",
            "Mongolia is a country 007",
            "Mongolia is a country 007 indeed",
            "Mongolia is the country 007",
        ],
    ]
    assert list(script) == frame["text"].to_list()


def test_the_planted_script_repeats_its_pattern_every_eight_rows() -> None:
    frame = planted_monolingual(40)

    script = planted_script(frame)

    sizes = [len(candidates) for candidates in script.values()]
    assert sizes == [2, 0, 2, 1, 1, 1, 1, 4] * 5
    assert script["Монгол улс 009"] == []
    assert script["Монгол улс 011"] == ["Монгол улс 011"]


def test_the_scripted_translator_answers_each_text_with_its_candidates() -> None:
    translator = ScriptedTranslator({"a": ["x", "y"], "b": []})

    assert translator.translate(["b", "a"], "khk_Cyrl", "eng_Latn") == [[], ["x", "y"]]


def test_the_scripted_translator_records_every_call_in_order() -> None:
    translator = ScriptedTranslator({"a": ["x"], "b": ["y"]})

    translator.translate(["a", "b"], "khk_Cyrl", "eng_Latn")
    translator.translate(("b",), "eng_Latn", "khk_Cyrl")

    assert translator.calls == [
        (("a", "b"), "khk_Cyrl", "eng_Latn"),
        (("b",), "eng_Latn", "khk_Cyrl"),
    ]


def test_the_scripted_translator_raises_on_the_scripted_calls_only() -> None:
    boom = RuntimeError("boom")
    translator = ScriptedTranslator({"a": ["x"]}, fail_on_call={1: boom})

    assert translator.translate(["a"], "khk_Cyrl", "eng_Latn") == [["x"]]
    with pytest.raises(RuntimeError, match="boom"):
        translator.translate(["a"], "khk_Cyrl", "eng_Latn")
    assert translator.translate(["a"], "khk_Cyrl", "eng_Latn") == [["x"]]
    assert len(translator.calls) == 3


def test_the_scripted_translator_rejects_an_unknown_text() -> None:
    translator = ScriptedTranslator({"a": ["x"]})

    with pytest.raises(KeyError, match="unknown"):
        translator.translate(["unknown"], "khk_Cyrl", "eng_Latn")


def test_the_scripted_translator_hands_out_copies_of_its_script() -> None:
    script = {"a": ["x"]}
    translator = ScriptedTranslator(script)

    translator.translate(["a"], "khk_Cyrl", "eng_Latn")[0].append("changed")

    assert translator.translate(["a"], "khk_Cyrl", "eng_Latn") == [["x"]]


def test_the_fake_scorer_rates_scripted_pairs_and_defaults_the_rest() -> None:
    scorer = FakeScorer({("a", "b"): 5.0, ("b", "a"): 6.0})

    assert scorer.score(["a", "b", "c"], ["b", "a", "b"]) == [5.0, 6.0, 1.0]


def test_the_fake_scorer_takes_a_default_and_records_its_calls() -> None:
    scorer = FakeScorer({}, default=0.9)

    scorer.score(["a"], ["b"])
    scorer.score(["c", "d"], ["e", "f"])

    assert scorer.calls == [(("a",), ("b",)), (("c", "d"), ("e", "f"))]
    assert scorer.score(["z"], ["y"]) == [0.9]


def test_the_fake_scorer_says_whether_it_needs_cuda() -> None:
    assert FakeScorer({}).requires_cuda is False
    assert FakeScorer({}, requires_cuda=True).requires_cuda is True


@pytest.mark.parametrize(
    ("kind", "pair"),
    [
        ("bt", ("Mongolia is poor 004", "Монгол улс 004")),
        ("ft", ("Монгол улс 004", "Mongolia is poor 004")),
    ],
)
def test_the_planted_scorer_rates_the_planted_pair_in_the_orientation_of_the_kind(
    kind: Kind, pair: tuple[str, str]
) -> None:
    scorer = planted_scorer(planted_monolingual(8), kind)

    assert scorer.score([pair[0]], [pair[1]]) == [9.0]


def test_the_planted_scorer_rates_rows_four_to_six_and_nothing_else() -> None:
    frame = planted_monolingual(16)
    scorer = planted_scorer(frame, "bt")
    script = planted_script(frame)

    rated = {
        index: score
        for index, real in enumerate(frame["text"])
        for candidate in script[real]
        if (score := scorer.score([candidate], [real])[0]) != 1.0
    }

    assert sorted(rated) == [4, 5, 6, 12, 13, 14]
    assert (rated[4], rated[6], rated[12], rated[14]) == (9.0, 8.0, 9.0, 8.0)
    assert math.isnan(rated[5])
    assert math.isnan(rated[13])


def test_the_planted_scorer_of_a_frame_without_rated_rows_gives_only_defaults() -> None:
    frame = planted_monolingual(1)

    assert planted_scorer(frame, "bt").score(["x"], ["y"]) == [1.0]
