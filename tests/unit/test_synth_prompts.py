"""The per-direction prompts the LLM generators render: the supported
directions, the English language names, the Cyrillic-output instruction and
the guarantee that the text to translate is inserted exactly as given."""

from __future__ import annotations

import itertools

import pytest
from pydantic import ValidationError

from rinchen.config.languages import load_languages
from rinchen.config.synth import JobConfig
from rinchen.errors import ConfigError
from rinchen.lang.scripts import Script
from rinchen.synth.prompts import (
    LANGUAGE_NAMES,
    SUPPORTED_DIRECTIONS,
    render_messages,
    render_prompt,
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
    assert SUPPORTED_DIRECTIONS == frozenset(DIRECTIONS)
    assert len(SUPPORTED_DIRECTIONS) == 16


def test_supported_directions_follow_the_language_registry():
    # The partners are every registry language that is not Mongolian itself
    # and does not use Mongolian script.
    partners = {
        entry.code
        for entry in load_languages()
        if entry.code != MONGOLIAN and Script.MONGOLIAN not in entry.scripts
    }

    assert partners == set(PARTNERS)


def test_language_names_cover_exactly_the_codes_of_the_supported_directions():
    codes = {code for direction in SUPPORTED_DIRECTIONS for code in direction}

    assert set(LANGUAGE_NAMES) == codes


def test_language_names_are_the_english_names():
    assert dict(LANGUAGE_NAMES) == EXPECTED_NAMES


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

    message = str(exc.value)
    for code in ["khk_Cyrl", *PARTNERS]:
        assert repr(code) in message
    assert "mn_Mong" not in message


def test_every_pair_a_job_config_accepts_has_a_prompt_in_both_orders():
    codes = [entry.code for entry in load_languages()]
    accepted = 0

    for mono, other in itertools.permutations(codes, 2):
        try:
            JobConfig(kind="ft", mono_lang=mono, other_lang=other, generator="g")
        except ValidationError:
            continue
        accepted += 1
        assert (mono, other) in SUPPORTED_DIRECTIONS
        assert (other, mono) in SUPPORTED_DIRECTIONS

    assert accepted == 16
