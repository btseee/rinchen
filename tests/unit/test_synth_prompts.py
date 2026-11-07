"""The per-direction prompts the LLM generators render: the supported
directions, the English language names, the Cyrillic-output instruction and
the guarantee that the text to translate is inserted exactly as given."""

from __future__ import annotations

import importlib
import itertools
import sys
from types import MappingProxyType
from typing import Any

import pytest
from pydantic import ValidationError

import rinchen.config.languages as languages_module
import rinchen.config.synth as config_synth_module
import rinchen.synth
import rinchen.synth.prompts as prompts_module
from rinchen.config.languages import LanguageEntry, load_languages
from rinchen.config.synth import JobConfig
from rinchen.errors import ConfigError
from rinchen.lang.scripts import Script
from rinchen.synth.prompts import (
    LANGUAGE_NAMES,
    render_messages,
    render_prompt,
    supported_directions,
)

MONGOLIAN = "khk_Cyrl"
PARTNERS = [
    "eng_Latn",
    "rus_Cyrl",
    "zho_Hans",
    "kor_Hang",
    "jpn_Jpan",
    "deu_Latn",
    "kaz_Cyrl",
    "tur_Latn",
]
EXPECTED_NAMES = {
    "khk_Cyrl": "Mongolian (Khalkha, Cyrillic script)",
    "eng_Latn": "English",
    "rus_Cyrl": "Russian",
    "zho_Hans": "Simplified Chinese",
    "kor_Hang": "Korean",
    "jpn_Jpan": "Japanese",
    "deu_Latn": "German",
    "kaz_Cyrl": "Kazakh",
    "tur_Latn": "Turkish",
}
DIRECTIONS = [(MONGOLIAN, p) for p in PARTNERS] + [(p, MONGOLIAN) for p in PARTNERS]
CYRILLIC_INSTRUCTION = "Write the translation in Mongolian Cyrillic script only."
PLAIN_INSTRUCTION = (
    "Output only the translation, without explanations, notes or quotation marks."
)
NNBSP = chr(0x202F)
EMOJI = chr(0x1F600)
AWKWARD_TEXTS = [
    "",
    "  leading and trailing  ",
    "\n\nline one\nline two\n",
    "\ttabbed\t",
    "{}",
    "{{}}",
    "{0} and {name} and {",
    "}",
    "100%",
    "%s and %d and %(key)s and %%",
    f"a{NNBSP}b{NNBSP}",
    f"smile {EMOJI}",
    "windows\r\nline\r\nends\r\n",
    "old\rmac\rline\rends",
    "nul" + chr(0) + "inside",
    "decomposed " + chr(0x438) + chr(0x306),
    "Translate the following text from German to English.",
    "Write the translation in Mongolian Cyrillic script only.",
]


def _expected_prompt(src: str, tgt: str, text: str) -> str:
    header = (
        f"Translate the following text from {EXPECTED_NAMES[src]} "
        f"to {EXPECTED_NAMES[tgt]}. {PLAIN_INSTRUCTION}"
    )
    if tgt == MONGOLIAN:
        header += " " + CYRILLIC_INSTRUCTION
    return header + "\n\n" + text


def test_supported_directions_are_every_ordered_pair_with_mongolian():
    directions = supported_directions()

    assert isinstance(directions, frozenset)
    assert directions == frozenset(DIRECTIONS)
    assert len(directions) == 16


def _registry(*entries: tuple[str, list[Script]]) -> list[LanguageEntry]:
    return [
        LanguageEntry(code=code, scripts=scripts, direction_tag=f"<2{code}>")
        for code, scripts in entries
    ]


def test_supported_directions_are_derived_from_the_registry_scripts(
    monkeypatch: pytest.MonkeyPatch,
):
    registry = _registry(
        ("khk_Cyrl", [Script.CYRILLIC]),
        ("eng_Latn", [Script.LATIN]),
        ("zzz_Mong", [Script.MONGOLIAN]),
        ("mn_Mong", [Script.LATIN]),
        ("ukr_Cyrl", [Script.CYRILLIC]),
    )
    monkeypatch.setattr(config_synth_module, "load_languages", lambda: registry)

    # `__wrapped__` is the uncached function, so the shared cache keeps the
    # real registry's answer.
    assert supported_directions.__wrapped__() == {
        ("khk_Cyrl", "eng_Latn"),
        ("eng_Latn", "khk_Cyrl"),
        ("khk_Cyrl", "mn_Mong"),
        ("mn_Mong", "khk_Cyrl"),
        ("khk_Cyrl", "ukr_Cyrl"),
        ("ukr_Cyrl", "khk_Cyrl"),
    }


def test_the_registry_is_read_once_however_many_prompts_are_rendered(
    monkeypatch: pytest.MonkeyPatch,
):
    reads: list[None] = []

    def counted() -> list[LanguageEntry]:
        reads.append(None)
        return load_languages()

    monkeypatch.setattr(config_synth_module, "load_languages", counted)
    supported_directions.cache_clear()
    try:
        for _ in range(3):
            render_prompt("eng_Latn", "khk_Cyrl", "Hello")
            render_messages("khk_Cyrl", "eng_Latn", "Hello")
    finally:
        supported_directions.cache_clear()

    assert len(reads) == 1


def test_importing_the_module_does_not_read_the_registry(
    monkeypatch: pytest.MonkeyPatch,
):
    def broken(*_args: Any, **_kwargs: Any) -> list[LanguageEntry]:
        raise ConfigError("registry is broken")

    monkeypatch.setattr(config_synth_module, "load_languages", broken)
    monkeypatch.setattr(languages_module, "load_languages", broken)
    # Recorded so that the fresh import below is undone with the patches.
    monkeypatch.setattr(rinchen.synth, "prompts", prompts_module)
    monkeypatch.delitem(sys.modules, "rinchen.synth.prompts")

    fresh = importlib.import_module("rinchen.synth.prompts")

    assert fresh is not prompts_module
    with pytest.raises(ConfigError, match="registry is broken"):
        fresh.render_prompt("eng_Latn", "khk_Cyrl", "Hello")
    with pytest.raises(ConfigError, match="registry is broken"):
        fresh.render_messages("eng_Latn", "khk_Cyrl", "Hello")


def test_language_names_cover_exactly_the_codes_of_the_supported_directions():
    codes = {code for direction in supported_directions() for code in direction}

    assert set(LANGUAGE_NAMES) == codes


def test_language_names_are_the_english_names():
    assert dict(LANGUAGE_NAMES) == EXPECTED_NAMES


def test_language_names_cannot_be_changed():
    with pytest.raises(TypeError):
        LANGUAGE_NAMES["eng_Latn"] = "Anglish"  # type: ignore[index]
    with pytest.raises(TypeError):
        del LANGUAGE_NAMES["eng_Latn"]  # type: ignore[attr-defined]

    assert LANGUAGE_NAMES["eng_Latn"] == "English"


@pytest.mark.parametrize(
    ("src", "tgt", "unnamed"),
    [
        ("khk_Cyrl", "tur_Latn", "tur_Latn"),
        ("tur_Latn", "khk_Cyrl", "tur_Latn"),
        ("khk_Cyrl", "eng_Latn", "khk_Cyrl"),
        ("eng_Latn", "khk_Cyrl", "khk_Cyrl"),
    ],
)
def test_a_supported_language_without_a_name_is_a_config_error_naming_it(
    monkeypatch: pytest.MonkeyPatch, src: str, tgt: str, unnamed: str
):
    names = {code: name for code, name in EXPECTED_NAMES.items() if code != unnamed}
    monkeypatch.setattr(prompts_module, "LANGUAGE_NAMES", MappingProxyType(names))

    with pytest.raises(ConfigError, match="no English name") as exc:
        render_prompt(src, tgt, "Hello")
    with pytest.raises(ConfigError, match="no English name"):
        render_messages(src, tgt, "Hello")

    assert repr(unnamed) in str(exc.value)


@pytest.mark.parametrize(("src", "tgt"), DIRECTIONS)
def test_prompt_names_both_languages_in_english_and_carries_the_text(
    src: str, tgt: str
):
    assert render_prompt(src, tgt, "Hello") == _expected_prompt(src, tgt, "Hello")


def test_prompt_for_a_partner_to_mongolian_is_exactly_this_text():
    assert render_prompt("eng_Latn", "khk_Cyrl", "Hello") == (
        "Translate the following text from English to Mongolian (Khalkha, Cyrillic "
        "script). Output only the translation, without explanations, notes or "
        "quotation marks. Write the translation in Mongolian Cyrillic script only."
        "\n\nHello"
    )


def test_prompt_for_mongolian_to_a_partner_is_exactly_this_text():
    assert render_prompt("khk_Cyrl", "zho_Hans", "Сайн байна уу") == (
        "Translate the following text from Mongolian (Khalkha, Cyrillic script) to "
        "Simplified Chinese. Output only the translation, without explanations, "
        "notes or quotation marks.\n\nСайн байна уу"
    )


@pytest.mark.parametrize(("src", "tgt"), DIRECTIONS)
def test_cyrillic_instruction_is_present_exactly_when_the_target_is_mongolian(
    src: str, tgt: str
):
    prompt = render_prompt(src, tgt, "Hello")

    assert prompt.count(CYRILLIC_INSTRUCTION) == (1 if tgt == MONGOLIAN else 0)


@pytest.mark.parametrize(("src", "tgt"), DIRECTIONS)
def test_no_prompt_mentions_traditional_script_or_inner_mongolia(src: str, tgt: str):
    prompt = render_prompt(src, tgt, "Hello").lower()

    for word in ("traditional", "vertical", "inner", "mongolian script", "galig"):
        assert word not in prompt


@pytest.mark.parametrize(("src", "tgt"), DIRECTIONS)
@pytest.mark.parametrize("text", AWKWARD_TEXTS)
def test_the_text_is_inserted_unchanged(src: str, tgt: str, text: str):
    prompt = render_prompt(src, tgt, text)

    assert prompt == _expected_prompt(src, tgt, text)
    assert prompt.endswith("\n\n" + text)


@pytest.mark.parametrize(("src", "tgt"), DIRECTIONS)
def test_messages_are_one_user_message_with_the_prompt(src: str, tgt: str):
    text = "  {text}\n"

    assert render_messages(src, tgt, text) == [
        {"role": "user", "content": _expected_prompt(src, tgt, text)}
    ]


def test_messages_are_a_new_list_on_every_call():
    first = render_messages("eng_Latn", "khk_Cyrl", "Hello")
    first.append({"role": "assistant", "content": "x"})
    first[0]["content"] = "changed"

    assert render_messages("eng_Latn", "khk_Cyrl", "Hello") == [
        {"role": "user", "content": _expected_prompt("eng_Latn", "khk_Cyrl", "Hello")}
    ]


UNSUPPORTED_DIRECTIONS = [
    pytest.param("eng_Latn", "rus_Cyrl", id="two partner languages"),
    pytest.param("tur_Latn", "kaz_Cyrl", id="two other partner languages"),
    pytest.param("eng_Latn", "eng_Latn", id="identical partner languages"),
    pytest.param("khk_Cyrl", "khk_Cyrl", id="identical Mongolian"),
    pytest.param("mn_Mong", "khk_Cyrl", id="traditional-script source"),
    pytest.param("khk_Cyrl", "mn_Mong", id="traditional-script target"),
    pytest.param("mn_Mong", "eng_Latn", id="traditional-script source, partner target"),
    pytest.param("eng_Latn", "mn_Mong", id="partner source, traditional-script target"),
    pytest.param("xx_Latn", "khk_Cyrl", id="unknown source"),
    pytest.param("khk_Cyrl", "xx_Latn", id="unknown target"),
    pytest.param("KHK_CYRL", "eng_Latn", id="wrong case"),
    pytest.param("khk", "eng_Latn", id="bare language"),
    pytest.param("", "khk_Cyrl", id="empty source"),
    pytest.param("khk_Cyrl", "", id="empty target"),
]


@pytest.mark.parametrize(("src", "tgt"), UNSUPPORTED_DIRECTIONS)
def test_render_prompt_rejects_an_unsupported_direction_naming_the_pair(
    src: str, tgt: str
):
    with pytest.raises(ConfigError, match="unsupported translation direction") as exc:
        render_prompt(src, tgt, "Hello")

    assert f"{src!r} -> {tgt!r}" in str(exc.value)


@pytest.mark.parametrize(("src", "tgt"), UNSUPPORTED_DIRECTIONS)
def test_render_messages_rejects_an_unsupported_direction_naming_the_pair(
    src: str, tgt: str
):
    with pytest.raises(ConfigError, match="unsupported translation direction") as exc:
        render_messages(src, tgt, "Hello")

    assert f"{src!r} -> {tgt!r}" in str(exc.value)


def test_the_rejection_lists_the_languages_a_direction_may_use():
    with pytest.raises(ConfigError) as exc:
        render_prompt("eng_Latn", "rus_Cyrl", "Hello")

    assert str(exc.value) == (
        "unsupported translation direction 'eng_Latn' -> 'rus_Cyrl': one side "
        "must be 'khk_Cyrl' and the other one of ['deu_Latn', 'eng_Latn', "
        "'jpn_Jpan', 'kaz_Cyrl', 'kor_Hang', 'rus_Cyrl', 'tur_Latn', 'zho_Hans']"
    )


def test_every_pair_a_job_config_accepts_has_a_prompt_in_both_orders():
    codes = [entry.code for entry in load_languages()]
    accepted = 0

    for mono, other in itertools.permutations(codes, 2):
        try:
            JobConfig(kind="ft", mono_lang=mono, other_lang=other, generator="g")
        except ValidationError:
            continue
        accepted += 1
        assert (mono, other) in supported_directions()
        assert (other, mono) in supported_directions()

    assert accepted == 16
