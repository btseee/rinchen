# scripts/build_script_conv_fixtures.py
"""Derives the D5 rule-baseline files from the committed Wiktionary reference.

Reads tests/fixtures/script_conv/wiktionary_reference.tsv and writes three files:

- src/rinchen/script_conv/lexicon.tsv: every development-split row (the verified
  Cyrillic word -> script spelling pairs the rule baseline looks up first);
- tests/fixtures/script_conv/rules_golden.tsv: 100 rows, `cyrillic`, `expected`,
  `path`. The `fallback` rows are the first held-out words, in sorted order, that
  the lexicon-free fallback spells exactly like the reference, up to 70; the rest
  are `lexicon` rows (development words, those whose spelling the fallback cannot
  produce first), so the file always has exactly 100 rows;
- tests/fixtures/script_conv/tiny_words.tsv: the first 500 reference rows by sorted
  Cyrillic word, the training fixture of the tiny converter model.

Rows whose script form fails `validate_traditional` are dropped before anything
else. The output is a pure function of the reference, so rerunning it rewrites the
same bytes; tests/unit/test_script_conv_fixtures.py compares the committed files
with this output. The fallback rules were developed on the development half only;
this script reads the held-out half, so it must be run after they are frozen.
"""

from __future__ import annotations

import csv
import sys
from collections.abc import Iterable, Sequence
from pathlib import Path

from rinchen.script_conv.rules import (
    FALLBACK_LETTERS,
    development_split,
    transliterate_fallback,
)
from rinchen.script_conv.validate import validate_traditional

REPO_ROOT = Path(__file__).resolve().parent.parent
REFERENCE_PATH = (
    REPO_ROOT / "tests" / "fixtures" / "script_conv" / "wiktionary_reference.tsv"
)
LEXICON_PATH = REPO_ROOT / "src" / "rinchen" / "script_conv" / "lexicon.tsv"
FIXTURES_DIR = REPO_ROOT / "tests" / "fixtures" / "script_conv"

GOLDEN_ROWS = 100
GOLDEN_FALLBACK_ROWS = 70
TINY_ROWS = 500

LEXICON_HEADER = ("cyrillic", "script")
GOLDEN_HEADER = ("cyrillic", "expected", "path")

Pair = tuple[str, str]
GoldenRow = tuple[str, str, str]


def read_reference(path: Path) -> list[Pair]:
    """`(cyrillic, script)` pairs sorted by Cyrillic, validator failures dropped."""
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t", quoting=csv.QUOTE_NONE)
        pairs = [(row["cyrillic"], row["traditional"]) for row in reader]
    return sorted(pair for pair in pairs if not validate_traditional(pair[1]))


def _write_tsv(
    path: Path, header: Sequence[str], rows: Iterable[Sequence[str]]
) -> None:
    lines = ["\t".join(header)] + ["\t".join(row) for row in rows]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def _lexicon_candidates(development: Sequence[Pair]) -> list[Pair]:
    """Development rows in the order the golden takes them.

    A row whose script has a character the fallback cannot emit comes first, because
    it shows the lexicon doing something the fallback never would.
    """
    beyond_fallback = [p for p in development if set(p[1]) - FALLBACK_LETTERS]
    within_fallback = [p for p in development if not set(p[1]) - FALLBACK_LETTERS]
    return beyond_fallback + within_fallback


def _golden_rows(
    development: Sequence[Pair], held_out: Sequence[Pair]
) -> list[GoldenRow]:
    fallback_rows = [
        (cyrillic, script, "fallback")
        for cyrillic, script in held_out
        if transliterate_fallback(cyrillic) == script
    ][:GOLDEN_FALLBACK_ROWS]
    lexicon_count = GOLDEN_ROWS - len(fallback_rows)
    lexicon_rows = [
        (cyrillic, script, "lexicon")
        for cyrillic, script in _lexicon_candidates(development)[:lexicon_count]
    ]
    return sorted(lexicon_rows) + fallback_rows


def build(reference_path: Path, lexicon_path: Path, fixtures_dir: Path) -> None:
    """Write `lexicon_path` and, into `fixtures_dir`, the golden and tiny files."""
    pairs = read_reference(reference_path)
    development = [pair for pair in pairs if development_split(pair[0])]
    held_out = [pair for pair in pairs if not development_split(pair[0])]
    lexicon_path.parent.mkdir(parents=True, exist_ok=True)
    fixtures_dir.mkdir(parents=True, exist_ok=True)
    _write_tsv(lexicon_path, LEXICON_HEADER, development)
    _write_tsv(
        fixtures_dir / "rules_golden.tsv",
        GOLDEN_HEADER,
        _golden_rows(development, held_out),
    )
    _write_tsv(fixtures_dir / "tiny_words.tsv", LEXICON_HEADER, pairs[:TINY_ROWS])


def main() -> int:
    build(REFERENCE_PATH, LEXICON_PATH, FIXTURES_DIR)
    golden = (FIXTURES_DIR / "rules_golden.tsv").read_text(encoding="utf-8")
    fallback_rows = sum(line.endswith("\tfallback") for line in golden.splitlines())
    print(
        f"wrote {LEXICON_PATH}, {FIXTURES_DIR / 'rules_golden.tsv'}"
        f" ({fallback_rows} fallback rows of {GOLDEN_FALLBACK_ROWS} wanted)"
        f" and {FIXTURES_DIR / 'tiny_words.tsv'}"
    )
    return 0


if __name__ == "__main__":
    assert sys.version_info[:2] == (3, 14), sys.version
    raise SystemExit(main())
