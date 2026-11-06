"""Unit tests for `rinchen.spm.symbols` (the 118 reserved SentencePiece symbols).

The placeholder brackets are U+27E6 and U+27E7; tests assert the code points
directly so a look-alike character cannot pass unnoticed.
"""

from __future__ import annotations

from collections import Counter

import pytest

from rinchen.config.languages import load_languages
from rinchen.errors import ConfigError
from rinchen.spm import symbols
from rinchen.spm.symbols import STRUCTURAL_SYMBOLS, user_defined_symbols

LANGUAGE_COUNT = 10
STRUCTURAL_COUNT = 8
PLACEHOLDER_COUNT = 100
STRUCTURAL_START = LANGUAGE_COUNT
PLACEHOLDER_START = LANGUAGE_COUNT + STRUCTURAL_COUNT

# U+27E6 and U+27E7, spelled by name so the exact code points are reviewable.
PLACEHOLDER_OPEN = "\N{MATHEMATICAL LEFT WHITE SQUARE BRACKET}"
PLACEHOLDER_CLOSE = "\N{MATHEMATICAL RIGHT WHITE SQUARE BRACKET}"


def _placeholder(index: int) -> str:
    return f"{PLACEHOLDER_OPEN}{index}{PLACEHOLDER_CLOSE}"


def test_total_is_118() -> None:
    assert len(user_defined_symbols()) == 118


def test_all_symbols_unique() -> None:
    all_symbols = user_defined_symbols()
    assert len(set(all_symbols)) == len(all_symbols)


def test_language_tags_come_first_in_registry_order() -> None:
    expected = [entry.direction_tag for entry in load_languages()]
    assert user_defined_symbols()[:LANGUAGE_COUNT] == expected


@pytest.mark.parametrize(
    "tag",
    [entry.direction_tag for entry in load_languages()],
    ids=[entry.code for entry in load_languages()],
)
def test_every_direction_tag_present_exactly_once(tag: str) -> None:
    assert Counter(user_defined_symbols())[tag] == 1


@pytest.mark.parametrize("tag", ["<2khk_Cyrl>", "<2mn_Mong>"])
def test_mongolian_direction_tags_present_exactly_once(tag: str) -> None:
    assert Counter(user_defined_symbols())[tag] == 1


def test_structural_symbols_exact_and_ordered() -> None:
    expected = ("<bt>", "<ft>", "<t>", "</t>", "<r>", "</r>", "<doc>", "<sep>")
    assert STRUCTURAL_SYMBOLS == expected
    structural = user_defined_symbols()[STRUCTURAL_START:PLACEHOLDER_START]
    assert structural == list(expected)


def test_placeholders_zero_to_ninety_nine_only() -> None:
    all_symbols = user_defined_symbols()
    placeholders = all_symbols[PLACEHOLDER_START:]
    assert placeholders == [_placeholder(i) for i in range(PLACEHOLDER_COUNT)]
    assert _placeholder(0) in placeholders
    assert _placeholder(99) in placeholders
    assert _placeholder(100) not in all_symbols


def test_placeholder_brackets_are_the_mathematical_white_square_brackets() -> None:
    first_placeholder = user_defined_symbols()[PLACEHOLDER_START]
    assert ord(first_placeholder[0]) == 0x27E6
    assert ord(first_placeholder[-1]) == 0x27E7


def test_duplicate_symbols_raise_config_error_naming_the_duplicate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry = load_languages()
    repeated = registry[1]
    monkeypatch.setattr(symbols, "load_languages", lambda: [*registry, repeated])
    with pytest.raises(ConfigError, match=repeated.direction_tag):
        user_defined_symbols()
