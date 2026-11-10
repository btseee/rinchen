"""The script purity metric of D12: how much of what a system wrote is in the script it was
asked for?

A row's output must be in the scripts the registry gives to its target language, except
that a row which names a script (`Mong` or `Cyrl`, allowed for `khk_Cyrl` and `mn_Mong` only,
as the engine's request is) is judged against that script's language whatever its target:
`Mong` against `mn_Mong`, `Cyrl` against `khk_Cyrl`. The purity is the one of the decode
check, `infer.script_mask.script_purity`, and the floor is its `PURITY_FLOOR`, a prior that
is inclusive (a text exactly on it passes). A text with no counted letter (empty, or only
digits, punctuation and placeholders) has no purity: unlike `script_purity`, which gives it
1.0, a robustness figure must not count silence as pure, so the caller counts it apart."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

from rinchen.errors import ConfigError
from rinchen.infer.script_mask import PURITY_FLOOR, allowed_scripts, script_purity
from rinchen.lang.scripts import Script, counted_letters

__all__ = ["PURITY_FLOOR", "ScriptOutcome", "score_script", "target_scripts"]

_LANGUAGE_OF_SCRIPT: Final = {"Mong": "mn_Mong", "Cyrl": "khk_Cyrl"}
_TARGETS_WITH_A_SCRIPT: Final = frozenset(_LANGUAGE_OF_SCRIPT.values())


@dataclass(frozen=True, slots=True)
class ScriptOutcome:
    """The purity of a text in the scripts it was asked for, and whether it is below the
    floor; both `None` for a text with no counted letter."""

    purity: float | None
    below_floor: bool | None


def target_scripts(tgt_lang: str, script: str | None) -> frozenset[Script]:
    """The scripts a row's output must be in.

    Raises:
        ConfigError: `tgt_lang` is not in the language registry, `script` is neither
            `Mong`, `Cyrl` nor `None`, or a script is given for a target other than
            `khk_Cyrl` and `mn_Mong`.
    """
    if script is None:
        return allowed_scripts(tgt_lang)
    if script not in _LANGUAGE_OF_SCRIPT:
        raise ConfigError("a script is Mong or Cyrl, or absent")
    if tgt_lang not in _TARGETS_WITH_A_SCRIPT:
        raise ConfigError(
            "a script is allowed only for the targets khk_Cyrl and mn_Mong"
        )
    return allowed_scripts(_LANGUAGE_OF_SCRIPT[script])


def score_script(text: str, tgt_lang: str, script: str | None = None) -> ScriptOutcome:
    """The purity of `text` in the scripts of the row, and whether it is below the floor.

    Raises:
        ConfigError: as `target_scripts`, whatever the text is.
    """
    scripts = target_scripts(tgt_lang, script)
    if counted_letters(text) == 0:
        return ScriptOutcome(purity=None, below_floor=None)
    purity = script_purity(text, scripts)
    return ScriptOutcome(purity=purity, below_floor=purity < PURITY_FLOOR)
