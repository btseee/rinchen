"""The documents the refusal tests of the suite reader (D12) start from: a valid suite with one
row in each section, as a fresh dict at every call, and the helpers that change it, encode it
and check what a refusal says.

Each test plants `MARKER` in the very value a refusal is about (a text, an id, a key, a code), so
a message that echoed the offending value would show it (`refused`)."""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

import pytest
from rinchen.eval.robust_suite import SUITE_SCHEMA, Suite

from rinchen.errors import DataError
from rinchen.spm.symbols import PLACEHOLDER_COUNT, user_defined_symbols

MARKER = "SECRET-MARKER-77"
MONGOLIAN = "".join(chr(code) for code in (0x1820, 0x1828, 0x1821))
PLACEHOLDER_5 = user_defined_symbols()[-PLACEHOLDER_COUNT:][5]
ROW_ID = {
    "numbers": "n1",
    "placeholders": "p1",
    "entities": "e1",
    "terms": "t1",
    "script": "s1",
    "mong": "m1",
}
COMMON = ("id", "src_lang", "tgt_lang", "source", "reference", "category")
EXTRA = {
    "numbers": (),
    "placeholders": (),
    "entities": ("entities",),
    "terms": ("terms",),
    "script": ("script",),
    "mong": ("source_mong",),
}
Document = dict[str, Any]


def document() -> Document:
    """A valid suite with one row in each section, fresh at every call."""
    return {
        "schema": SUITE_SCHEMA,
        "name": "tiny",
        "entities": {"bat": {"khk_Cyrl": ["Бат"], "eng_Latn": ["Bat"]}},
        "sections": {
            "numbers": [
                {
                    "id": "n1",
                    "src_lang": "eng_Latn",
                    "tgt_lang": "khk_Cyrl",
                    "source": "It costs 15.",
                    "reference": "Үнэ 15.",
                    "category": "int",
                }
            ],
            "placeholders": [
                {
                    "id": "p1",
                    "src_lang": "eng_Latn",
                    "tgt_lang": "khk_Cyrl",
                    "source": f"Where is {PLACEHOLDER_5}?",
                    "reference": f"{PLACEHOLDER_5} хаана байна вэ?",
                    "category": "noun",
                }
            ],
            "entities": [
                {
                    "id": "e1",
                    "src_lang": "eng_Latn",
                    "tgt_lang": "khk_Cyrl",
                    "source": "Bat is here.",
                    "reference": "Бат энд байна.",
                    "category": "person",
                    "entities": ["bat"],
                }
            ],
            "terms": [
                {
                    "id": "t1",
                    "src_lang": "eng_Latn",
                    "tgt_lang": "khk_Cyrl",
                    "source": "Where is the table?",
                    "reference": "Ширээ хаана байна вэ?",
                    "category": "glossary",
                    "terms": [{"src": "table", "tgt": "ширээ"}],
                }
            ],
            "script": [
                {
                    "id": "s1",
                    "src_lang": "eng_Latn",
                    "tgt_lang": "khk_Cyrl",
                    "source": "This is a book.",
                    "reference": None,
                    "category": "trained",
                    "script": None,
                }
            ],
            "mong": [
                {
                    "id": "m1",
                    "src_lang": "khk_Cyrl",
                    "tgt_lang": "eng_Latn",
                    "source": "Энэ бол ном.",
                    "reference": "This is a book.",
                    "category": "base",
                    "source_mong": MONGOLIAN,
                }
            ],
        },
    }


def encode(doc: Document) -> bytes:
    return json.dumps(doc, ensure_ascii=False).encode("utf-8")


def row_of(doc: Document, section: str, index: int = 0) -> Document:
    row: Document = doc["sections"][section][index]
    return row


def changed(change: Callable[[Document], None]) -> bytes:
    doc = document()
    change(doc)
    return encode(doc)


def set_row_field(
    section: str, field: str, value: object
) -> Callable[[Document], None]:
    def change(doc: Document) -> None:
        row_of(doc, section)[field] = value

    return change


def drop_row_field(section: str, field: str) -> Callable[[Document], None]:
    def change(doc: Document) -> None:
        del row_of(doc, section)[field]

    return change


def refused(data: bytes) -> str:
    with pytest.raises(DataError) as raised:
        Suite.from_bytes(data)
    message = str(raised.value)
    assert MARKER not in message, "a refusal quoted the offending value"
    return message


def assert_names(message: str, *fragments: str) -> None:
    for fragment in fragments:
        assert fragment in message, f"{fragment!r} is not in {message!r}"


def at(
    section: str, field: str | None, *, index: int = 0, named_id: bool = True
) -> tuple[str, ...]:
    """What a message about `field` of the row at `index` must name."""
    named = [f"'{section}'", f"index {index}"]
    if named_id:
        named.append(f"'{ROW_ID[section]}'")
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
