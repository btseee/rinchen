"""Reads the committed Wiktionary reference list for the script-converter tests.

The reader is independent of `scripts/build_script_conv_fixtures.py`, so a test
that compares the builder's output with the reference does not share a bug with
the builder.
"""

from __future__ import annotations

import csv
import hashlib
from pathlib import Path

from rinchen.config.loader import REPO_ROOT
from rinchen.script_conv.validate import validate_traditional

FIXTURE_DIR = REPO_ROOT / "tests" / "fixtures" / "script_conv"
REFERENCE_PATH = FIXTURE_DIR / "wiktionary_reference.tsv"
REFERENCE_COLUMNS = (
    "cyrillic",
    "traditional",
    "part_of_speech",
    "page_url",
    "revision_id",
    "has_fvs_mvs_nnbsp",
    "source_templates",
)


def read_tsv_rows(path: Path) -> list[dict[str, str]]:
    """Rows of a UTF-8 TSV with a header, quotes left alone."""
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t", quoting=csv.QUOTE_NONE)
        return list(reader)


def all_reference_rows() -> list[dict[str, str]]:
    return read_tsv_rows(REFERENCE_PATH)


def valid_reference_pairs() -> list[tuple[str, str]]:
    """`(cyrillic, script)` of every reference row whose script passes the validator."""
    return [
        (row["cyrillic"], row["traditional"])
        for row in all_reference_rows()
        if not validate_traditional(row["traditional"])
    ]


def is_development_word(word: str) -> bool:
    """The spec's split: the sha256 of the word's UTF-8 bytes is even."""
    return int(hashlib.sha256(word.encode("utf-8")).hexdigest(), 16) % 2 == 0
