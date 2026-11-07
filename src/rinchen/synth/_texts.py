"""What every generator backend checks about the source texts it is given,
before it does any work on them."""

from __future__ import annotations

from collections.abc import Sequence

from rinchen.errors import DataError


def require_encodable(texts: Sequence[str]) -> None:
    """A lone surrogate cannot go into a JSON body, and SentencePiece raises a
    bare `TypeError` on it. The error names the text by its index and never
    quotes it."""
    for index, text in enumerate(texts):
        try:
            text.encode()
        except UnicodeEncodeError as error:
            raise DataError(
                f"text {index} cannot be encoded as UTF-8: a lone surrogate at "
                f"position {error.start}"
            ) from error
