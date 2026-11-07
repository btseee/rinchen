"""What every generator backend does to the candidates it returns, so the LLM
clients and the CTranslate2 baseline hand the selection stage the same kind of
list."""

from __future__ import annotations

from collections.abc import Iterable


def clean_candidates(candidates: Iterable[str]) -> list[str]:
    """Stripped, in first-seen order, without duplicates. A candidate with no
    letter or digit at all (only whitespace, punctuation or invisible characters
    such as a zero-width space or a byte order mark) counts as empty and is
    dropped."""
    stripped = (candidate.strip() for candidate in candidates)
    return list(
        dict.fromkeys(
            candidate
            for candidate in stripped
            if any(char.isalnum() for char in candidate)
        )
    )
