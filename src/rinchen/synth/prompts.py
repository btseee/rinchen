"""The prompts the LLM generators are given: one instruction per direction
between Mongolian (Cyrillic) and each partner language. The wording is ours,
not a published template of any model; it states the task plainly and, when
the target is Mongolian, asks for Cyrillic output only."""

from __future__ import annotations

from rinchen.config.languages import load_languages
from rinchen.errors import ConfigError
from rinchen.lang.scripts import Script

_MONGOLIAN = "khk_Cyrl"

# One table, keyed by registry code. A language added to the registry has no
# name here, which `test_synth_prompts` reports.
LANGUAGE_NAMES = {
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

_PARTNERS = tuple(
    entry.code
    for entry in load_languages()
    if entry.code != _MONGOLIAN and Script.MONGOLIAN not in entry.scripts
)

SUPPORTED_DIRECTIONS = frozenset(
    pair
    for partner in _PARTNERS
    for pair in ((_MONGOLIAN, partner), (partner, _MONGOLIAN))
)

_TASK = "Translate the following text from "
_OUTPUT_RULE = (
    "Output only the translation, without explanations, notes or quotation marks."
)
_CYRILLIC_OUTPUT_INSTRUCTION = (
    "Write the translation in Mongolian Cyrillic script only."
)


def render_prompt(src_lang: str, tgt_lang: str, text: str) -> str:
    """The instruction that makes a chat model translate `text`, which is
    appended exactly as given. Raises ConfigError for a direction other than
    Mongolian to or from a partner language."""
    if (src_lang, tgt_lang) not in SUPPORTED_DIRECTIONS:
        raise ConfigError(
            f"unsupported translation direction {src_lang!r} -> {tgt_lang!r}: "
            f"one side must be {_MONGOLIAN!r} and the other one of {list(_PARTNERS)}"
        )
    instruction = (
        _TASK
        + LANGUAGE_NAMES[src_lang]
        + " to "
        + LANGUAGE_NAMES[tgt_lang]
        + ". "
        + _OUTPUT_RULE
    )
    if tgt_lang == _MONGOLIAN:
        instruction += " " + _CYRILLIC_OUTPUT_INSTRUCTION
    return instruction + "\n\n" + text


def render_messages(src_lang: str, tgt_lang: str, text: str) -> list[dict[str, str]]:
    """The chat messages for `render_prompt`: one user message."""
    return [{"role": "user", "content": render_prompt(src_lang, tgt_lang, text)}]
