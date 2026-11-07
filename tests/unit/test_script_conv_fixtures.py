"""Fixtures built from the Wiktionary reference: lexicon, rules golden, tiny words.

The reader in `tests.support.script_reference` is independent of the builder, so
a bug in the builder's selection shows up as a difference here.
"""

from __future__ import annotations

from collections import Counter
from importlib import resources
from pathlib import Path

import pytest
from build_script_conv_fixtures import build

from rinchen.config.loader import REPO_ROOT
from rinchen.script_conv.rules import LEXICON, transliterate_fallback
from rinchen.script_conv.validate import validate_traditional
from tests.support.script_reference import (
    FIXTURE_DIR,
    REFERENCE_COLUMNS,
    all_reference_rows,
    is_development_word,
    read_tsv_rows,
    valid_reference_pairs,
)

LEXICON_PATH = REPO_ROOT / "src" / "rinchen" / "script_conv" / "lexicon.tsv"
GOLDEN_PATH = FIXTURE_DIR / "rules_golden.tsv"
TINY_PATH = FIXTURE_DIR / "tiny_words.tsv"
REPORT_PATH = FIXTURE_DIR / "wiktionary_reference.report.md"

GOLDEN_ROWS = 100
GOLDEN_FALLBACK_TARGET = 70
TINY_MIN, TINY_MAX = 400, 700

# The two reference rows that break the validator (a doubled selector and a
# leading NNBSP) and so leave every derived fixture.
VALIDATOR_FAILURES = {"агентлаг", "руу"}


def _read_two_columns(path: Path, header: tuple[str, str]) -> list[tuple[str, str]]:
    lines = path.read_text(encoding="utf-8").split("\n")
    assert lines[-1] == "", f"{path.name} does not end with a newline"
    assert "\r" not in path.read_text(encoding="utf-8")
    assert tuple(lines[0].split("\t")) == header
    rows = [tuple(line.split("\t")) for line in lines[1:-1]]
    assert all(len(row) == 2 for row in rows), (
        f"{path.name} has a row of the wrong width"
    )
    return [(cyrillic, script) for cyrillic, script in rows]


def _golden_rows() -> list[tuple[str, str, str]]:
    rows = read_tsv_rows(GOLDEN_PATH)
    return [(row["cyrillic"], row["expected"], row["path"]) for row in rows]


def test_reference_is_committed_with_its_report() -> None:
    rows = all_reference_rows()

    assert tuple(rows[0]) == REFERENCE_COLUMNS
    assert len(rows) == 2029
    assert REPORT_PATH.read_text(encoding="utf-8").startswith(
        "# Wiktionary Khalkha Cyrillic to traditional-script reference"
    )


def test_two_reference_rows_fail_the_validator() -> None:
    failing = {
        row["cyrillic"]
        for row in all_reference_rows()
        if validate_traditional(row["traditional"])
    }

    assert failing == VALIDATOR_FAILURES


def test_build_is_deterministic(tmp_path: Path) -> None:
    reference = FIXTURE_DIR / "wiktionary_reference.tsv"
    for run in ("first", "second"):
        build(reference, tmp_path / run / "lexicon.tsv", tmp_path / run)

    names = sorted(path.name for path in (tmp_path / "first").iterdir())
    assert names == ["lexicon.tsv", "rules_golden.tsv", "tiny_words.tsv"]
    for name in names:
        assert (tmp_path / "first" / name).read_bytes() == (
            tmp_path / "second" / name
        ).read_bytes()


def test_build_reproduces_committed_files(tmp_path: Path) -> None:
    build(FIXTURE_DIR / "wiktionary_reference.tsv", tmp_path / "lexicon.tsv", tmp_path)

    assert (tmp_path / "lexicon.tsv").read_bytes() == LEXICON_PATH.read_bytes()
    assert (tmp_path / "rules_golden.tsv").read_bytes() == GOLDEN_PATH.read_bytes()
    assert (tmp_path / "tiny_words.tsv").read_bytes() == TINY_PATH.read_bytes()


def test_lexicon_is_exactly_the_development_rows_of_the_reference() -> None:
    expected = {
        cyrillic: script
        for cyrillic, script in valid_reference_pairs()
        if is_development_word(cyrillic)
    }

    rows = _read_two_columns(LEXICON_PATH, ("cyrillic", "script"))

    assert dict(rows) == expected
    assert len(rows) == len(expected)
    assert [cyrillic for cyrillic, _ in rows] == sorted(expected)
    assert not VALIDATOR_FAILURES & set(expected)


def test_lexicon_is_readable_as_package_data() -> None:
    packaged = (
        resources.files("rinchen.script_conv").joinpath("lexicon.tsv").read_bytes()
    )

    assert packaged == LEXICON_PATH.read_bytes()
    assert len(LEXICON) == len(_read_two_columns(LEXICON_PATH, ("cyrillic", "script")))


def test_golden_has_one_hundred_rows_with_two_known_paths() -> None:
    rows = _golden_rows()

    assert len(rows) == GOLDEN_ROWS
    assert len({cyrillic for cyrillic, _, _ in rows}) == GOLDEN_ROWS
    assert set(Counter(path for _, _, path in rows)) <= {"lexicon", "fallback"}
    assert tuple(read_tsv_rows(GOLDEN_PATH)[0]) == ("cyrillic", "expected", "path")


def test_golden_lexicon_rows_are_development_words_with_the_reference_spelling() -> (
    None
):
    reference = dict(valid_reference_pairs())

    rows = [row for row in _golden_rows() if row[2] == "lexicon"]

    assert len(rows) >= 30
    for cyrillic, expected, _ in rows:
        assert is_development_word(cyrillic), cyrillic
        assert expected == reference[cyrillic] == LEXICON[cyrillic], cyrillic


def test_golden_fallback_rows_are_held_out_words_the_fallback_gets_right() -> None:
    reference = dict(valid_reference_pairs())

    rows = [row for row in _golden_rows() if row[2] == "fallback"]

    for cyrillic, expected, _ in rows:
        assert not is_development_word(cyrillic), cyrillic
        assert cyrillic not in LEXICON, cyrillic
        assert expected == reference[cyrillic], cyrillic
        assert transliterate_fallback(cyrillic) == expected, cyrillic


def test_golden_fallback_count_is_seventy_or_every_word_the_fallback_gets_right() -> (
    None
):
    held_out_correct = [
        cyrillic
        for cyrillic, script in valid_reference_pairs()
        if not is_development_word(cyrillic)
        and transliterate_fallback(cyrillic) == script
    ]

    fallback_rows = [row for row in _golden_rows() if row[2] == "fallback"]

    assert len(fallback_rows) == min(GOLDEN_FALLBACK_TARGET, len(held_out_correct))
    assert [row[0] for row in fallback_rows] == sorted(held_out_correct)[
        : len(fallback_rows)
    ]


def test_golden_rows_are_sorted_within_each_path() -> None:
    for path in ("lexicon", "fallback"):
        words = [
            cyrillic for cyrillic, _, row_path in _golden_rows() if row_path == path
        ]
        assert words == sorted(words)


def test_tiny_words_are_the_first_reference_rows_by_sorted_word() -> None:
    reference = sorted(valid_reference_pairs())

    rows = _read_two_columns(TINY_PATH, ("cyrillic", "script"))

    assert TINY_MIN <= len(rows) <= TINY_MAX
    assert rows == reference[: len(rows)]
    assert len(rows) == 500
    assert not VALIDATOR_FAILURES & {cyrillic for cyrillic, _ in rows}


@pytest.mark.parametrize("path", [LEXICON_PATH, TINY_PATH])
def test_derived_script_columns_pass_the_validator(path: Path) -> None:
    for cyrillic, script in _read_two_columns(path, ("cyrillic", "script")):
        assert validate_traditional(script) == (), cyrillic
