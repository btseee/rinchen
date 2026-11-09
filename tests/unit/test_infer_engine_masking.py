"""Script masking in the engine (D11, ruling T9): the three modes, the retry of an impure
output with the suppression list, the purity each result reports, and the lists built once.

The model of these tests WANTS to write a Latin word into a Cyrillic target unless the call
suppresses the Latin piece (`leaky`): the way the toy leaks a language it never trained.
`LiveCt2.answer` sees what a call suppressed, so a test can tell what each call asked for.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import pytest

from rinchen.infer import engine as engine_module
from rinchen.infer.engine_types import ENGINE_COUNTERS, DecodeOptions, Row
from rinchen.infer.script_mask import suppression_list
from tests.support.engines import (
    END,
    ENG,
    KHK,
    RUS,
    SP,
    ScriptedEngine,
    engine_environment,  # noqa: F401 - the fixture of `pytestmark`
    hyp,
    scripted_engine,
)
from tests.support.fake_ct2 import Hypothesis

LATIN = SP + "hello"
CONTROL = [["<unk>"], ["<blank>"], ["<s>"], ["averyunlikelytoken0"]]
Answer = Callable[[tuple[str, ...], frozenset[str]], Sequence[Hypothesis]]

pytestmark = pytest.mark.usefixtures("engine_environment")


def mode(name: str, **options: Any) -> DecodeOptions:
    return DecodeOptions(script_mask=name, **options)  # type: ignore[arg-type]


def khk(*texts: str) -> list[Row]:
    return [Row(text, None, KHK) for text in texts]


def leaky(line: tuple[str, ...], suppressed: frozenset[str]) -> Sequence[Hypothesis]:
    """Writes the Latin piece unless the call suppresses it, then the Cyrillic one."""
    return [hyp(SP + "Энэ", END)] if LATIN in suppressed else [hyp(LATIN, END)]


def only_for(word: str, wanted: Answer) -> Answer:
    """`wanted` for the source that holds `word`; the echo of the source for the others."""

    def answer(
        line: tuple[str, ...], suppressed: frozenset[str]
    ) -> Sequence[Hypothesis]:
        if SP + word in line:
            return wanted(line, suppressed)
        return [hyp(*line[1:], END)]

    return answer


def counted(fixture: ScriptedEngine, **expected: int) -> None:
    assert fixture.engine.counters() == {
        **dict.fromkeys(ENGINE_COUNTERS, 0),
        **expected,
    }


# -- the three modes


def test_always_decodes_with_the_whole_list_of_the_target(tmp_path: Path) -> None:
    fixture = scripted_engine(tmp_path, answer=leaky)

    (result,) = fixture.engine.translate(khk("x"), mode("always"))

    (call,) = fixture.ct2.calls
    expected = suppression_list(
        fixture.vocabulary, fixture.processor, KHK, end_token=END
    )
    assert call.suppress_sequences == expected
    assert [LATIN] in expected and [SP + "Энэ"] not in expected
    assert ["<2eng_Latn>"] in expected and [END] not in expected
    assert call.disable_unk is True
    assert (result.text, result.script_purity, result.retried) == ("Энэ", 1.0, False)
    counted(fixture, sources=1)


def test_off_suppresses_only_what_is_never_text(tmp_path: Path) -> None:
    fixture = scripted_engine(tmp_path, answer=leaky)

    (result,) = fixture.engine.translate(khk("x"), mode("off"))

    (call,) = fixture.ct2.calls
    assert call.suppress_sequences == CONTROL
    assert call.disable_unk is True
    assert (result.text, result.script_purity, result.retried) == ("hello", 0.0, False)
    counted(fixture, sources=1, script_impure=1)


def test_retry_decodes_a_pure_output_once_without_a_script_entry(
    tmp_path: Path,
) -> None:
    fixture = scripted_engine(tmp_path)

    (result,) = fixture.engine.translate(khk("Энэ ном"), mode("retry"))

    (call,) = fixture.ct2.calls
    assert call.suppress_sequences == CONTROL
    assert call.disable_unk is True
    assert (result.text, result.script_purity, result.retried) == (
        "Энэ ном",
        1.0,
        False,
    )
    counted(fixture, sources=1)


def test_retry_is_the_default_mode(tmp_path: Path) -> None:
    fixture = scripted_engine(tmp_path, answer=leaky)

    (result,) = fixture.engine.translate(khk("x"))

    assert len(fixture.ct2.calls) == 2
    assert result.text == "Энэ"


# -- the retry


def test_an_impure_output_is_decoded_again_with_the_list_and_replaced_when_purer(
    tmp_path: Path,
) -> None:
    fixture = scripted_engine(tmp_path, answer=leaky)

    (result,) = fixture.engine.translate(khk("x"), mode("retry"))

    first, second = fixture.ct2.calls
    expected = suppression_list(
        fixture.vocabulary, fixture.processor, KHK, end_token=END
    )
    assert first.suppress_sequences == CONTROL
    assert second.suppress_sequences == expected
    assert second.source == first.source
    assert (result.text, result.script_purity, result.retried) == ("Энэ", 1.0, True)
    counted(fixture, sources=1, script_retries=1, script_retry_improved=1)


def test_only_the_impure_rows_are_decoded_again_and_keep_their_places(
    tmp_path: Path,
) -> None:
    fixture = scripted_engine(tmp_path, answer=only_for("x", leaky))

    result = fixture.engine.translate(khk("Энэ", "x", "ус"), mode("retry"))

    first, second = fixture.ct2.calls
    assert [line[1] for line in first.source] == [SP + "Энэ", SP + "x", SP + "ус"]
    assert second.source == [["<2khk_Cyrl>", SP + "x"]]
    assert [(r.text, r.retried) for r in result] == [
        ("Энэ", False),
        ("Энэ", True),
        ("ус", False),
    ]
    counted(fixture, sources=3, script_retries=1, script_retry_improved=1)


def mixed(first: Sequence[str], second: Sequence[str]) -> Answer:
    """The first decode says `first`, a decode that suppresses the Latin piece `second`."""

    def answer(
        line: tuple[str, ...], suppressed: frozenset[str]
    ) -> Sequence[Hypothesis]:
        return [hyp(*(second if LATIN in suppressed else first), END)]

    return answer


def test_a_retry_that_is_less_pure_replaces_nothing(tmp_path: Path) -> None:
    # 3 Cyrillic letters of 8 (0.375) against 0 of 10.
    answer = mixed([LATIN, SP + "Энэ"], [LATIN, SP + "world"])
    fixture = scripted_engine(tmp_path, answer=answer)

    (result,) = fixture.engine.translate(khk("x"), mode("retry"))

    assert len(fixture.ct2.calls) == 2
    assert (result.text, result.script_purity, result.retried) == (
        "hello Энэ",
        pytest.approx(3 / 8),
        True,
    )
    counted(fixture, sources=1, script_retries=1, script_impure=1)


def test_a_retry_that_is_as_pure_replaces_nothing(tmp_path: Path) -> None:
    answer = mixed([LATIN, SP + "Энэ"], [SP + "world", SP + "ном"])
    fixture = scripted_engine(tmp_path, answer=answer)

    (result,) = fixture.engine.translate(khk("x"), mode("retry"))

    assert result.text == "hello Энэ"
    counted(fixture, sources=1, script_retries=1, script_impure=1)


def test_a_retry_that_improves_but_stays_impure_replaces_and_is_counted_impure(
    tmp_path: Path,
) -> None:
    answer = mixed([LATIN, SP + "world"], [LATIN, SP + "Энэ", SP + "ном"])
    fixture = scripted_engine(tmp_path, answer=answer)

    (result,) = fixture.engine.translate(khk("x"), mode("retry"))

    assert result.text == "hello Энэ ном"
    assert result.script_purity == pytest.approx(6 / 11)
    counted(
        fixture, sources=1, script_retries=1, script_retry_improved=1, script_impure=1
    )


def test_a_retry_with_no_output_keeps_the_first_text(tmp_path: Path) -> None:
    def answer(
        line: tuple[str, ...], suppressed: frozenset[str]
    ) -> Sequence[Hypothesis]:
        return [hyp("<unk>", END)] if LATIN in suppressed else [hyp(LATIN, END)]

    fixture = scripted_engine(tmp_path, answer=answer)

    (result,) = fixture.engine.translate(khk("x"), mode("retry"))

    assert (result.text, result.status, result.retried) == ("hello", "ok", True)
    counted(fixture, sources=1, script_retries=1, script_impure=1, reserved_symbol=1)


def test_the_retry_replaces_the_alternatives_with_the_text(tmp_path: Path) -> None:
    def answer(
        line: tuple[str, ...], suppressed: frozenset[str]
    ) -> Sequence[Hypothesis]:
        if LATIN in suppressed:
            return [hyp(SP + "Энэ", END), hyp(SP + "ном", END)]
        return [hyp(LATIN, END), hyp(SP + "world", END)]

    fixture = scripted_engine(tmp_path, answer=answer)

    (result,) = fixture.engine.translate(khk("x"), mode("retry", n=2))

    assert (result.text, result.alternatives) == ("Энэ", ("ном",))


@pytest.mark.parametrize(
    ("cyrillic", "retried"),
    [(49, False), (48, True)],
    ids=["0.98-is-pure-enough", "below-0.98-is-retried"],
)
def test_the_purity_floor_is_0_98(tmp_path: Path, cyrillic: int, retried: bool) -> None:
    text = SP + "Э" * cyrillic + "a"

    def answer(
        line: tuple[str, ...], suppressed: frozenset[str]
    ) -> Sequence[Hypothesis]:
        return [hyp(text, END)]

    fixture = scripted_engine(tmp_path, answer=answer)

    (result,) = fixture.engine.translate(khk("x"), mode("retry"))

    assert result.retried is retried
    assert len(fixture.ct2.calls) == (2 if retried else 1)
    assert fixture.engine.counters()["script_impure"] == (1 if retried else 0)


def test_a_row_with_no_output_is_not_decoded_again(tmp_path: Path) -> None:
    def answer(
        line: tuple[str, ...], suppressed: frozenset[str]
    ) -> Sequence[Hypothesis]:
        return [hyp("<unk>", END)]

    fixture = scripted_engine(tmp_path, answer=answer)

    (result,) = fixture.engine.translate(khk("x"), mode("retry"))

    assert (result.status, result.retried, result.script_purity) == (
        "no_output",
        False,
        None,
    )
    assert len(fixture.ct2.calls) == 1


def test_a_text_with_no_letter_is_pure(tmp_path: Path) -> None:
    def answer(
        line: tuple[str, ...], suppressed: frozenset[str]
    ) -> Sequence[Hypothesis]:
        return [hyp(SP + "2026", END)]

    fixture = scripted_engine(tmp_path, answer=answer)

    (result,) = fixture.engine.translate(khk("x"), mode("retry"))

    assert (result.script_purity, result.retried) == (1.0, False)
    assert len(fixture.ct2.calls) == 1


# -- the lists of the calls


def test_rows_of_targets_with_other_scripts_are_decoded_one_call_per_list(
    tmp_path: Path,
) -> None:
    fixture = scripted_engine(tmp_path)

    result = fixture.engine.translate(
        [
            Row("Энэ", None, KHK),
            Row("hello", None, ENG),
            Row("ном", None, KHK),
            Row("ус", None, RUS),
        ],
        mode("always"),
    )

    cyrillic, latin = fixture.ct2.calls
    assert [line[0] for line in cyrillic.source] == [
        "<2khk_Cyrl>",
        "<2khk_Cyrl>",
        "<2rus_Cyrl>",
    ]
    assert [line[0] for line in latin.source] == ["<2eng_Latn>"]
    assert cyrillic.suppress_sequences == suppression_list(
        fixture.vocabulary, fixture.processor, KHK, end_token=END
    )
    assert latin.suppress_sequences == suppression_list(
        fixture.vocabulary, fixture.processor, ENG, end_token=END
    )
    assert cyrillic.suppress_sequences != latin.suppress_sequences
    assert [r.text for r in result] == ["Энэ", "hello", "ном", "ус"]


def test_rows_of_different_targets_share_one_call_when_nothing_script_specific_is_listed(
    tmp_path: Path,
) -> None:
    fixture = scripted_engine(tmp_path)

    fixture.engine.translate(
        [Row("Энэ", None, KHK), Row("hello", None, ENG)], mode("off")
    )

    (call,) = fixture.ct2.calls
    assert len(call.source) == 2


def test_the_second_pass_makes_one_call_per_list_too(tmp_path: Path) -> None:
    def answer(
        line: tuple[str, ...], suppressed: frozenset[str]
    ) -> Sequence[Hypothesis]:
        # Wants a Cyrillic word for the Latin target and a Latin one for the Cyrillic one.
        wrong = SP + "Энэ" if line[0] == "<2eng_Latn>" else LATIN
        right = LATIN if line[0] == "<2eng_Latn>" else SP + "Энэ"
        return [hyp(right if wrong in suppressed else wrong, END)]

    fixture = scripted_engine(tmp_path, answer=answer)

    result = fixture.engine.translate(
        [Row("a", None, KHK), Row("b", None, ENG)], mode("retry")
    )

    assert len(fixture.ct2.calls) == 3
    first, to_cyrillic, to_latin = fixture.ct2.calls
    assert first.suppress_sequences == CONTROL
    assert [line[0] for line in to_cyrillic.source] == ["<2khk_Cyrl>"]
    assert [line[0] for line in to_latin.source] == ["<2eng_Latn>"]
    assert [r.text for r in result] == ["Энэ", "hello"]
    counted(fixture, sources=2, script_retries=2, script_retry_improved=2)


def test_a_list_is_built_once_per_target_script_whatever_the_calls(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    built: list[tuple[str, bool]] = []
    real = engine_module.suppression_list

    def counting(*args: Any, **kwargs: Any) -> list[list[str]]:
        built.append((args[2], kwargs.get("scripts", True)))
        return real(*args, **kwargs)

    monkeypatch.setattr(engine_module, "suppression_list", counting)
    fixture = scripted_engine(tmp_path, answer=leaky)

    for _ in range(3):
        fixture.engine.translate(khk("x"), mode("retry"))
        fixture.engine.translate(khk("x", "y"), mode("always"))
        fixture.engine.translate([Row("x", None, RUS)], mode("always"))
        fixture.engine.translate(khk("x"), mode("off"))

    # `retry` and `off`: the control list; `always` and the second pass: the Cyrillic list.
    # Russian writes the scripts Mongolian does, so it shares the list of the Cyrillic target.
    assert sorted(built, key=str) == [(KHK, False), (KHK, True)]


def test_a_decode_that_never_asks_for_a_script_list_does_not_build_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    built: list[object] = []
    real = engine_module.suppression_list

    def counting(*args: Any, **kwargs: Any) -> list[list[str]]:
        built.append(kwargs.get("scripts", True))
        return real(*args, **kwargs)

    monkeypatch.setattr(engine_module, "suppression_list", counting)
    fixture = scripted_engine(tmp_path)

    fixture.engine.translate(khk("Энэ"), mode("retry"))

    assert built == [False]
