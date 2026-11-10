"""The documents the refusal tests of the suite reader (D13, spec T3) start from: a valid suite
of two rows and two directions as a fresh dict at every call, and the helpers that change it,
encode it and check what a refusal says.

Each test plants `MARKER` in the very value a refusal is about (a sentence, an id, a key, a file
name, a code), so a message that echoed the offending value would show it (`refused`)."""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

import pytest

from rinchen.errors import DataError
from rinchen.eval.suite_rules import SUITE_SCHEMA
from rinchen.eval.suites import Suite

MARKER = "SECRET-MARKER-83"
PREFIX = "not an evaluation suite:"
DIGEST = "ab" * 32
REVISION = "0123456789abcdef0123456789abcdef01234567"
Document = dict[str, Any]


def document() -> Document:
    """A valid fixture suite of two rows and the two directions between English and Khalkha,
    fresh at every call."""
    return {
        "schema": SUITE_SCHEMA,
        "name": "tiny",
        "kind": "fixture",
        "benchmark": "tiny",
        "split": "devtest",
        "source": {
            "dataset": None,
            "files": {},
            "missing_languages": [],
            "pinned": False,
            "revision": None,
        },
        "directions": ["khk_Cyrl-eng_Latn", "eng_Latn-khk_Cyrl"],
        "rows": [
            {
                "id": "r1",
                "doc": "d1",
                "text": {"eng_Latn": "The cat sat.", "khk_Cyrl": "Муур сууж байв."},
            },
            {
                "id": "r2",
                "doc": None,
                "text": {"eng_Latn": "A dog runs.", "khk_Cyrl": "Нохой гүйж байна."},
            },
        ],
    }


def encode(doc: Document, *, ensure_ascii: bool = False) -> bytes:
    return json.dumps(doc, ensure_ascii=ensure_ascii).encode("utf-8")


def changed(change: Callable[[Document], None]) -> bytes:
    doc = document()
    change(doc)
    return encode(doc)


def set_top(field: str, value: object) -> Callable[[Document], None]:
    def change(doc: Document) -> None:
        doc[field] = value

    return change


def set_source(field: str, value: object) -> Callable[[Document], None]:
    def change(doc: Document) -> None:
        doc["source"][field] = value

    return change


def set_row(field: str, value: object, index: int = 0) -> Callable[[Document], None]:
    def change(doc: Document) -> None:
        doc["rows"][index][field] = value

    return change


def set_sentence(
    language: str, value: object, index: int = 0
) -> Callable[[Document], None]:
    def change(doc: Document) -> None:
        doc["rows"][index]["text"][language] = value

    return change


def refused(data: bytes) -> str:
    with pytest.raises(DataError) as raised:
        Suite.from_bytes(data)
    message = str(raised.value)
    assert message.startswith(PREFIX), message
    assert MARKER not in message, "a refusal quoted the offending value"
    return message


def assert_names(message: str, *fragments: str) -> None:
    for fragment in fragments:
        assert fragment in message, f"{fragment!r} is not in {message!r}"


def at_row(index: int, field: str | None, row_id: str | None) -> tuple[str, ...]:
    """What a message about `field` of the row at `index` must name (the id only when the
    row had a well-formed one)."""
    named = [f"row index {index}"]
    if row_id is not None:
        named.append(f"(id '{row_id}')")
    if field is not None:
        named.append(f"'{field}'")
    return tuple(named)


def repeating(doc: Document, anchor: str, inserted: str) -> bytes:
    """The compact JSON of `doc` with `inserted` put after the one `anchor`, so a key is
    written twice."""
    text = json.dumps(doc, ensure_ascii=False)
    assert text.count(anchor) == 1, anchor
    return text.replace(anchor, anchor + inserted).encode("utf-8")


def accepted(change: Callable[[Document], None]) -> Suite:
    return Suite.from_bytes(changed(change))
