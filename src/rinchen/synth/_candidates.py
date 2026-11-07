"""What every generator backend does to the candidates it returns, so the LLM
clients and the CTranslate2 baseline hand the selection stage the same kind of
list."""

from __future__ import annotations

from collections.abc import Iterable


def is_utf8_encodable(text: str) -> bool:
    """False for a text holding a surrogate code point, which no UTF-8 file can
    store. A server's JSON escape such as the one for U+D800 decodes to one."""
    try:
        text.encode("utf-8")
    except UnicodeEncodeError:
        return False
    return True


def clean_candidates(candidates: Iterable[str]) -> list[str]:
    """Stripped, in first-seen order, without duplicates. A candidate with no
    letter or digit at all (only whitespace, punctuation or invisible characters
    such as a zero-width space or a byte order mark) counts as empty and is
    dropped, and so does one that UTF-8 cannot encode: both are malformed
    generations, which leave the text with fewer candidates or none."""
    stripped = (candidate.strip() for candidate in candidates)
    return list(
        dict.fromkeys(
            candidate
            for candidate in stripped
            if any(char.isalnum() for char in candidate)
            and is_utf8_encodable(candidate)
        )
    )
