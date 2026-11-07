"""Checks that traditional-script text carries no forbidden control character.

The converter never emits ZWJ or ZWNJ, puts NNBSP only between letters, and
uses a variation selector only for a pair the Unicode Character Database lists.
`validate_traditional` reports every place where text breaks one of those rules.
"""

from __future__ import annotations

import string
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass
from importlib import resources
from types import MappingProxyType

from rinchen.errors import DataError
from rinchen.script_conv.alphabet import (
    FVS,
    JOINERS,
    LETTERS,
    MVS,
    NIRUGU,
    NNBSP,
    SCRIPT_TO_CYRILLIC_PUNCTUATION,
)

JOINER = "joiner"
FOREIGN_CHARACTER = "foreign_character"
ORPHAN_SELECTOR = "orphan_selector"
MISPLACED_NNBSP = "misplaced_nnbsp"
INVALID_VARIANT = "invalid_variant"
VIOLATION_CODES: frozenset[str] = frozenset(
    {JOINER, FOREIGN_CHARACTER, ORPHAN_SELECTOR, MISPLACED_NNBSP, INVALID_VARIANT}
)

_VARIANT_TABLE_FILE = "standardized_variants.tsv"

# Outside the script, ASCII text passes: a Latin run, digits and punctuation.
_ALLOWED = (
    LETTERS
    | {NIRUGU, " "}
    | frozenset(string.ascii_letters + string.digits + string.punctuation)
    | frozenset(SCRIPT_TO_CYRILLIC_PUNCTUATION)
)


@dataclass(frozen=True)
class Violation:
    """One broken rule at `index`, a code point offset into the validated text."""

    index: int
    code: str
    detail: str


def _load_variant_table() -> Mapping[str, frozenset[str]]:
    text = (
        resources.files("rinchen.script_conv")
        .joinpath(_VARIANT_TABLE_FILE)
        .read_text("utf-8")
    )
    selectors_by_base: defaultdict[str, set[str]] = defaultdict(set)
    for number, line in enumerate(text.splitlines(), start=1):
        if not line or line.startswith("#"):
            continue
        fields = line.split("\t")
        try:
            base, selector = (chr(int(field, 16)) for field in fields)
        except ValueError as error:
            raise DataError(
                f"{_VARIANT_TABLE_FILE}:{number}: expected 'base_hex<TAB>selector_hex'"
            ) from error
        if base not in LETTERS or selector not in FVS:
            raise DataError(
                f"{_VARIANT_TABLE_FILE}:{number}: not a Mongolian letter and selector pair"
            )
        selectors_by_base[base].add(selector)
    return MappingProxyType(
        {base: frozenset(selectors) for base, selectors in selectors_by_base.items()}
    )


# Base letter -> the selectors UCD StandardizedVariants.txt lists for it.
VARIANT_TABLE: Mapping[str, frozenset[str]] = _load_variant_table()


def _code_point(char: str) -> str:
    return f"U+{ord(char):04X}"


def _follows_letter(text: str, index: int) -> bool:
    return index > 0 and text[index - 1] in LETTERS


def _is_followed_by_letter(text: str, index: int) -> bool:
    return index + 1 < len(text) and text[index + 1] in LETTERS


def _check_fvs(text: str, index: int) -> Violation | None:
    if not _follows_letter(text, index):
        return Violation(
            index,
            ORPHAN_SELECTOR,
            f"{_code_point(text[index])} does not follow a letter",
        )
    base = text[index - 1]
    if text[index] not in VARIANT_TABLE.get(base, frozenset()):
        return Violation(
            index,
            INVALID_VARIANT,
            f"{_code_point(text[index])} is not a listed variant of {_code_point(base)}",
        )
    return None


def _check_mvs(text: str, index: int) -> Violation | None:
    if not _follows_letter(text, index):
        return Violation(
            index, ORPHAN_SELECTOR, f"{_code_point(MVS)} does not follow a letter"
        )
    if not _is_followed_by_letter(text, index):
        return Violation(
            index, ORPHAN_SELECTOR, f"{_code_point(MVS)} is not followed by a letter"
        )
    return None


def _ends_with_stem_letter(text: str, end: int) -> bool:
    """Whether `text[:end]` ends with a letter, which may carry its own selector."""
    last = end - 1
    if last >= 1 and text[last] in FVS:
        last -= 1
    return last >= 0 and text[last] in LETTERS


def _check_nnbsp(text: str, index: int) -> Violation | None:
    if _ends_with_stem_letter(text, index) and _is_followed_by_letter(text, index):
        return None
    return Violation(
        index, MISPLACED_NNBSP, f"{_code_point(NNBSP)} is not between two letters"
    )


def _check_character(text: str, index: int) -> Violation | None:
    char = text[index]
    if char in JOINERS:
        return Violation(index, JOINER, f"{_code_point(char)} is never emitted")
    if char in FVS:
        return _check_fvs(text, index)
    if char == MVS:
        return _check_mvs(text, index)
    if char == NNBSP:
        return _check_nnbsp(text, index)
    if char in _ALLOWED:
        return None
    return Violation(
        index,
        FOREIGN_CHARACTER,
        f"{_code_point(char)} is not allowed in traditional-script text",
    )


def validate_traditional(text: str) -> tuple[Violation, ...]:
    """Every violation in `text`, in index order; empty when the text is clean."""
    violations = (_check_character(text, index) for index in range(len(text)))
    return tuple(violation for violation in violations if violation is not None)
