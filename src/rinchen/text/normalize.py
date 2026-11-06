"""Language-aware text normalization applied before splitting and tokenizing.

Pipeline: NFC, zero-width cleanup, whitespace, quotes, Cyrillic homoglyph
repair, Fita fold, NFC again. Zero-width cleanup runs before whitespace so
that ``"a <ZWSP> b"`` collapses to one space, and the closing NFC pass
re-composes pairs that stripping or folding brought together; both keep
`normalize` idempotent.
"""

from __future__ import annotations

import re
import unicodedata

import regex

from rinchen.config.languages import known_language_codes
from rinchen.errors import ConfigError

_MN_MONG = "mn_Mong"

# ZWSP, ZWNJ, ZWJ, WORD JOINER, BOM; stripped for every language.
_ZERO_WIDTH_CHARS = "\u200b\u200c\u200d\u2060\ufeff"
# Mongolian free variation selectors and vowel separator (U+180B..U+180F):
# meaningful only in traditional Mongolian script, noise elsewhere.
_MONGOLIAN_FORMAT_CHARS = "".join(chr(code) for code in range(0x180B, 0x1810))
_ZERO_WIDTH_TABLE_MN_MONG = str.maketrans("", "", _ZERO_WIDTH_CHARS)
_ZERO_WIDTH_TABLE_OTHER = str.maketrans(
    "", "", _ZERO_WIDTH_CHARS + _MONGOLIAN_FORMAT_CHARS
)

_WHITESPACE_RUN = re.compile(r"\s+")
# In mn_Mong, U+202F NARROW NO-BREAK SPACE joins a suffix to its stem (it drives
# contextual shaping), so it is part of the word, not whitespace.
_WHITESPACE_RUN_KEEPING_NNBSP = re.compile(r"[^\S\u202f]+")

_DOUBLE_QUOTES = "\u201c\u201d\u201e\u201f\u00ab\u00bb"
_SINGLE_QUOTES = "\u2018\u2019\u201a\u201b"
_QUOTE_TABLE = str.maketrans(
    {
        **dict.fromkeys(map(ord, _DOUBLE_QUOTES), '"'),
        **dict.fromkeys(map(ord, _SINGLE_QUOTES), "'"),
    }
)

# Latin letters that look like Cyrillic ones: a e o p c x y, then uppercase.
_LATIN_LOOKALIKES = "aeopcxyAEOPCXY"
_CYRILLIC_COUNTERPARTS = (
    "\u0430\u0435\u043e\u0440\u0441\u0445\u0443"
    "\u0410\u0415\u041e\u0420\u0421\u0425\u0423"
)
_LATIN_TO_CYRILLIC = str.maketrans(_LATIN_LOOKALIKES, _CYRILLIC_COUNTERPARTS)
_CYRILLIC_LETTER = regex.compile(r"\p{Script=Cyrillic}")

# Fita U+0472/U+0473 looks like the Mongolian O-umlaut and stands in for it in
# noisy text; fold it into U+04E8/U+04E9.
# Kazakh U+04B0/U+04B1 and Greek theta U+0398/U+03B8 are different letters and
# are never folded.
_FITA_TO_O_UMLAUT = str.maketrans({0x0472: "\u04e8", 0x0473: "\u04e9"})


def _require_known_language(lang: str) -> None:
    known = known_language_codes()
    if lang not in known:
        raise ConfigError(
            f"unknown language code {lang!r}; expected one of {sorted(known)}"
        )


def _strip_zero_width(text: str, lang: str) -> str:
    table = _ZERO_WIDTH_TABLE_MN_MONG if lang == _MN_MONG else _ZERO_WIDTH_TABLE_OTHER
    return text.translate(table)


def _collapse_whitespace(text: str, lang: str) -> str:
    run = _WHITESPACE_RUN_KEEPING_NNBSP if lang == _MN_MONG else _WHITESPACE_RUN
    return run.sub(" ", text).strip(" ")


def _canonicalize_quotes(text: str) -> str:
    return text.translate(_QUOTE_TABLE)


def _is_cyrillic_majority(token: str) -> bool:
    letters = [char for char in token if char.isalpha()]
    cyrillic_count = sum(1 for char in letters if _CYRILLIC_LETTER.match(char))
    return 2 * cyrillic_count > len(letters)


def _repair_token_homoglyphs(token: str) -> str:
    if _is_cyrillic_majority(token):
        return token.translate(_LATIN_TO_CYRILLIC)
    return token


def _repair_homoglyphs(text: str) -> str:
    """Replace Latin lookalikes with Cyrillic letters inside Cyrillic-majority tokens.

    Tokens are the runs between plain spaces, so this runs after whitespace
    collapsing, when plain spaces are the only separators left.
    """
    return " ".join(_repair_token_homoglyphs(token) for token in text.split(" "))


def _fold_fita(text: str) -> str:
    return text.translate(_FITA_TO_O_UMLAUT)


def normalize(text: str, lang: str) -> str:
    """Normalize `text` for language `lang`.

    Meant for raw text, run before any reserved symbol (a ``<2xx>`` tag, ``<doc>``,
    ``<sep>``, ``<t>``, a placeholder) is inserted. Reserved symbols must be
    separated from text by whitespace: the homoglyph step works on space-delimited
    tokens, so a symbol glued to a Cyrillic word would share its token and have
    its Latin lookalike letters rewritten (``<sep>`` glued to a Cyrillic word
    becomes ``<s`` + Cyrillic ``e`` + ``r>``).

    Raises:
        ConfigError: `lang` is not in the language registry.
    """
    _require_known_language(lang)
    text = unicodedata.normalize("NFC", text)
    text = _strip_zero_width(text, lang)
    text = _collapse_whitespace(text, lang)
    text = _canonicalize_quotes(text)
    text = _repair_homoglyphs(text)
    text = _fold_fita(text)
    return unicodedata.normalize("NFC", text)
