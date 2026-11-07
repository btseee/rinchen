"""The prompts the LLM generators are given: one instruction per direction
between Mongolian (Cyrillic) and each partner language. The wording is ours,
not a published template of any model; it states the task plainly and, when
the target is Mongolian, asks for Cyrillic output only."""

from __future__ import annotations

import functools
from types import MappingProxyType

from rinchen.config.synth import MONGOLIAN, supported_partner_languages
from rinchen.errors import ConfigError

# One table, keyed by registry code. A language added to the registry has no
# name here, which `test_synth_prompts` reports and rendering refuses.
LANGUAGE_NAMES = MappingProxyType(
    {
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
)


@functools.cache
def supported_directions() -> frozenset[tuple[str, str]]:
    """Every ordered pair between Mongolian and a partner language of the
    registry. Read on first use, not at import, so a broken registry fails the
    call that needs it."""
    return frozenset(
        pair
        for partner in supported_partner_languages()
        for pair in ((MONGOLIAN, partner), (partner, MONGOLIAN))
    )


_TASK = "Translate the following text from "
_OUTPUT_RULE = (
    "Output only the translation, without explanations, notes or quotation marks."
)
_CYRILLIC_OUTPUT_INSTRUCTION = (
    "Write the translation in Mongolian Cyrillic script only."
)


def _language_name(code: str) -> str:
    name = LANGUAGE_NAMES.get(code)
    if name is None:
        raise ConfigError(f"no English name for language {code!r} in LANGUAGE_NAMES")
    return name


def render_prompt(src_lang: str, tgt_lang: str, text: str) -> str:
    """The instruction that makes a chat model translate `text`, which is
    appended exactly as given. Raises ConfigError for a direction other than
    Mongolian to or from a partner language."""
    directions = supported_directions()
    if (src_lang, tgt_lang) not in directions:
        partners = sorted({code for pair in directions for code in pair} - {MONGOLIAN})
        raise ConfigError(
            f"unsupported translation direction {src_lang!r} -> {tgt_lang!r}: "
            f"one side must be {MONGOLIAN!r} and the other one of {partners}"
        )
    instruction = (
        _TASK
        + _language_name(src_lang)
        + " to "
        + _language_name(tgt_lang)
        + ". "
        + _OUTPUT_RULE
    )
    if tgt_lang == MONGOLIAN:
        instruction += " " + _CYRILLIC_OUTPUT_INSTRUCTION
    return instruction + "\n\n" + text


def render_messages(src_lang: str, tgt_lang: str, text: str) -> list[dict[str, str]]:
    """The chat messages for `render_prompt`: one user message."""
    return [{"role": "user", "content": render_prompt(src_lang, tgt_lang, text)}]
