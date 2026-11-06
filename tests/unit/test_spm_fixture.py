"""Tests for the tiny synthetic SentencePiece corpus fixture.

The fixture texts are deliberately SYNTHETIC pseudo-words, so these tests
check structure (reproducibility, script purity, normalization, digits) and
never meaning. Script purity is the mechanical guard against Latin-transliterated
("galig") Mongolian and against Inner Mongolian material: every file must be
written in the script(s) its registry language declares.
"""

from __future__ import annotations

import unicodedata
from pathlib import Path

import pytest
from build_spm_fixture import build

from rinchen.config.languages import known_language_codes, load_languages
from rinchen.config.loader import REPO_ROOT
from rinchen.lang.scripts import script_of

FIXTURE_DIR = REPO_ROOT / "tests" / "fixtures" / "spm" / "tiny_corpus"

LINE_COUNTS = {"khk_Cyrl": 400, "rus_Cyrl": 200, "eng_Latn": 100, "mn_Mong": 100}
FIXTURE_LANGUAGES = sorted(LINE_COUNTS)
LANGUAGES_WITH_DIGITS = ["khk_Cyrl", "rus_Cyrl", "eng_Latn"]

PURITY_FLOOR = 0.99


def _fixture_lines(code: str) -> list[str]:
    text = (FIXTURE_DIR / f"{code}.txt").read_text(encoding="utf-8")
    return text.splitlines()


def test_build_reproduces_committed_files(tmp_path: Path) -> None:
    build(tmp_path)

    assert sorted(p.name for p in tmp_path.iterdir()) == [
        f"{code}.txt" for code in FIXTURE_LANGUAGES
    ]
    for code in FIXTURE_LANGUAGES:
        built = (tmp_path / f"{code}.txt").read_bytes()
        committed = (FIXTURE_DIR / f"{code}.txt").read_bytes().replace(b"\r\n", b"\n")
        assert built == committed, f"{code}.txt differs from the builder's output"


def test_files_are_registry_languages() -> None:
    stems = {path.stem for path in FIXTURE_DIR.glob("*.txt")}

    assert stems == set(FIXTURE_LANGUAGES)
    assert stems <= known_language_codes()


@pytest.mark.parametrize("code", FIXTURE_LANGUAGES)
def test_each_file_is_script_pure(code: str) -> None:
    declared = next(e.scripts for e in load_languages() if e.code == code)
    ratios = script_of("\n".join(_fixture_lines(code)))

    declared_share = sum(ratios.get(script, 0.0) for script in declared)

    assert declared_share >= PURITY_FLOOR, ratios


@pytest.mark.parametrize("code", FIXTURE_LANGUAGES)
def test_lines_are_nonblank_nfc_and_tab_free(code: str) -> None:
    for number, line in enumerate(_fixture_lines(code), start=1):
        where = f"{code}.txt line {number}"
        assert line.strip(), f"{where} is blank"
        assert unicodedata.is_normalized("NFC", line), f"{where} is not NFC"
        assert "\t" not in line, f"{where} contains a tab"
        assert "\r" not in line, f"{where} contains a carriage return"


@pytest.mark.parametrize("code", LANGUAGES_WITH_DIGITS)
def test_files_contain_digits(code: str) -> None:
    text = (FIXTURE_DIR / f"{code}.txt").read_text(encoding="utf-8")

    assert any(char.isdigit() for char in text)


@pytest.mark.parametrize(("code", "expected"), sorted(LINE_COUNTS.items()))
def test_line_counts_match_the_plan(code: str, expected: int) -> None:
    assert len(_fixture_lines(code)) == expected
