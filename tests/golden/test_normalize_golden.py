"""Golden gate for `rinchen.text.normalize.normalize`: 200 hand-derived rows.

Each data row of `tests/fixtures/golden/normalize_200.tsv` is
`input<TAB>expected<TAB>lang`. The file is pure ASCII so that invisible and
look-alike characters stay reviewable: every non-ASCII or control character is
a backslash-u (4 hex digits) or backslash-capital-U (8 hex digits) escape, and
a literal backslash is written as two backslashes. The `expected` values were
derived by hand from the normalization rule table, never from `normalize`
output, so a disagreement means a derivation error or an implementation bug.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import pytest

from rinchen.config.languages import known_language_codes
from rinchen.lang.scripts import Script, script_of
from rinchen.text.normalize import normalize

_GOLDEN_PATH = (
    Path(__file__).resolve().parents[1] / "fixtures" / "golden" / "normalize_200.tsv"
)
_CATEGORY_PREFIX = "# category:"
_EXPECTED_QUOTAS = {
    "nfc": 15,
    "zero_width": 20,
    "whitespace": 30,
    "quotes": 25,
    "homoglyph": 40,
    "fold": 20,
    "mn_mong_preservation": 30,
    "passthrough": 20,
}
_CYRILLIC_LANGS = frozenset({"khk_Cyrl", "rus_Cyrl", "kaz_Cyrl"})
_ESCAPE = re.compile(r"\\u([0-9a-fA-F]{4})|\\U([0-9a-fA-F]{8})|(\\\\)")


def _decode_escape(match: re.Match[str]) -> str:
    if match.group(3) is not None:
        return "\\"
    return chr(int(match.group(1) or match.group(2), 16))


def _decode(field: str) -> str:
    return _ESCAPE.sub(_decode_escape, field)


@dataclass(frozen=True, slots=True)
class GoldenRow:
    line_number: int
    category: str
    lang: str
    text: str
    expected: str


def _load_rows() -> list[GoldenRow]:
    rows: list[GoldenRow] = []
    category = ""
    lines = _GOLDEN_PATH.read_text(encoding="utf-8").split("\n")
    for line_number, raw_line in enumerate(lines, start=1):
        line = raw_line.rstrip("\r")
        if line.startswith(_CATEGORY_PREFIX):
            category = line.removeprefix(_CATEGORY_PREFIX).strip()
        if not line or line.startswith("#"):
            continue
        fields = line.split("\t")
        if len(fields) != 3:
            raise ValueError(
                f"{_GOLDEN_PATH.name}:{line_number}: expected 3 tab-separated "
                f"fields, found {len(fields)}"
            )
        text, expected, lang = fields
        rows.append(
            GoldenRow(
                line_number=line_number,
                category=category,
                lang=lang,
                text=_decode(text),
                expected=_decode(expected),
            )
        )
    return rows


ROWS = _load_rows()
_CYRILLIC_ROWS_WITH_LETTERS = [
    row
    for row in ROWS
    if row.lang in _CYRILLIC_LANGS and any(char.isalpha() for char in row.expected)
]


def _row_id(row: GoldenRow) -> str:
    return f"line-{row.line_number}"


def test_decode_expands_escapes_and_keeps_escaped_backslash_literal() -> None:
    backslash = chr(92)
    assert _decode(f"{backslash}u0041{backslash}U0001f600") == "A" + chr(0x1F600)
    assert _decode(f"{backslash}{backslash}u0041") == backslash + "u0041"


def test_golden_file_is_ascii_lf_with_well_formed_escapes() -> None:
    text = _GOLDEN_PATH.read_text(encoding="utf-8")
    assert text.isascii()
    assert "\r" not in text
    assert "\\" not in _ESCAPE.sub("", text)


@pytest.mark.parametrize("row", ROWS, ids=_row_id)
def test_golden_line(row: GoldenRow) -> None:
    assert normalize(row.text, row.lang) == row.expected


def test_golden_has_exactly_200_data_lines() -> None:
    assert len(ROWS) == 200


def test_every_category_meets_its_quota() -> None:
    assert Counter(row.category for row in ROWS) == Counter(_EXPECTED_QUOTAS)


def test_langs_are_registry_codes() -> None:
    assert {row.lang for row in ROWS} <= known_language_codes()


def test_passthrough_lines_cover_every_registry_language() -> None:
    covered = {row.lang for row in ROWS if row.category == "passthrough"}
    assert covered == known_language_codes()


@pytest.mark.parametrize("row", _CYRILLIC_ROWS_WITH_LETTERS, ids=_row_id)
def test_cyrillic_labeled_lines_are_cyrillic(row: GoldenRow) -> None:
    assert script_of(row.expected).get(Script.CYRILLIC, 0.0) >= 0.5
