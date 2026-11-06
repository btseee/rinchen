"""The reserved symbols every Rinchen SentencePiece model keeps as single pieces.

`user_defined_symbols` composes them in a fixed order, which fixes their piece
ids: the language-direction tags from the registry, then the structural tags,
then the numbered placeholders.
"""

from __future__ import annotations

from collections import Counter

from rinchen.config.languages import load_languages
from rinchen.errors import ConfigError

STRUCTURAL_SYMBOLS = ("<bt>", "<ft>", "<t>", "</t>", "<r>", "</r>", "<doc>", "<sep>")

PLACEHOLDER_COUNT = 100
# U+27E6 and U+27E7 MATHEMATICAL WHITE SQUARE BRACKETS: not ASCII brackets,
# so a placeholder cannot be mistaken for ordinary bracketed text.
_PLACEHOLDER_OPEN = chr(0x27E6)
_PLACEHOLDER_CLOSE = chr(0x27E7)


def _placeholders() -> list[str]:
    return [
        f"{_PLACEHOLDER_OPEN}{index}{_PLACEHOLDER_CLOSE}"
        for index in range(PLACEHOLDER_COUNT)
    ]


def user_defined_symbols() -> list[str]:
    """Return the reserved symbols in vocabulary order. Raises ConfigError if
    two of them are equal, as a registry edit that re-adds a tag would cause."""
    symbols = [
        *(entry.direction_tag for entry in load_languages()),
        *STRUCTURAL_SYMBOLS,
        *_placeholders(),
    ]
    duplicates = sorted(
        symbol for symbol, count in Counter(symbols).items() if count > 1
    )
    if duplicates:
        raise ConfigError(f"duplicate user-defined symbols: {duplicates}")
    return symbols
