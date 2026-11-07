"""Checks that traditional-script text carries no forbidden control character.

The converter never emits ZWJ or ZWNJ, puts NNBSP only between letters, and
uses a variation selector only for a pair the Unicode Character Database lists.
`validate_traditional` reports every place where text breaks one of those rules.

`validate_traditional` judges running text, so it lets ASCII letters, digits,
punctuation and spaces through. `is_script_word` has the stricter contract of one
converted word: letters, their selectors, the vowel separator, the stroke and the
suffix gap only, with at least one letter. Screening a converter's output or a
training pair's script side takes `is_script_word`.
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

# What one converted word may be made of: letters, their selectors, the vowel
# separator, the stroke and the suffix gap. `_ALLOWED` is wider because it checks
# running text.
_SCRIPT_WORD_CHARACTERS = LETTERS | FVS | {MVS, NIRUGU, NNBSP}

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


def _parse_variant_table(text: str) -> Mapping[str, frozenset[str]]:
    """Read the table file's text; a malformed or empty table is a `DataError`."""
    selectors_by_base: defaultdict[str, set[str]] = defaultdict(set)
    for number, line in enumerate(text.splitlines(), start=1):
        if not line or line.startswith("#"):
            continue
        where = f"{_VARIANT_TABLE_FILE}:{number}"
        try:
            base, selector = (chr(int(field, 16)) for field in line.split("\t"))
        except ValueError as error:
            raise DataError(
                f"{where}: expected 'base_hex<TAB>selector_hex', both below U+110000"
            ) from error
        if base not in LETTERS or selector not in FVS:
            raise DataError(f"{where}: not a Mongolian letter and selector pair")
        if selector in selectors_by_base[base]:
            raise DataError(f"{where}: duplicate row")
        selectors_by_base[base].add(selector)
    if not selectors_by_base:
        raise DataError(f"{_VARIANT_TABLE_FILE}: the table has no rows")
    return MappingProxyType(
        {base: frozenset(selectors) for base, selectors in selectors_by_base.items()}
    )


def _load_variant_table() -> Mapping[str, frozenset[str]]:
    return _parse_variant_table(
        resources.files("rinchen.script_conv")
        .joinpath(_VARIANT_TABLE_FILE)
        .read_text("utf-8")
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


def is_script_word(text: str) -> bool:
    """Whether `text` is one converted traditional-script word.

    It must pass `validate_traditional`, hold at least one letter, and be made of
    letters, free variation selectors, the vowel separator, the nirugu stroke and
    NNBSP only. A Cyrillic word can convert only to such a text, so ASCII letters,
    digits, punctuation (the script's own marks too) and spaces inside the word are
    defects even where running text may hold them.
    """
    return (
        not LETTERS.isdisjoint(text)
        and _SCRIPT_WORD_CHARACTERS.issuperset(text)
        and not validate_traditional(text)
    )
