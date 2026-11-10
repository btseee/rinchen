"""The golden suite of D13: `tests/fixtures/eval/golden_suite.json` holds the ten sentences of
`tests.support.eval_golden` in English and Khalkha, five documents of two rows each, and the two
directions between the languages. The file is hand-checkable: its expected text is built here from
the constants of the golden corpus with the standard library's `json`, never with the code under
test, and the reader must give back the same suite and the writer the same bytes. The fixtures of
D13 together stay under 50 KB."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from rinchen.errors import ConfigError
from rinchen.eval.suites import Suite, read_suite
from tests.support.eval_golden import (
    GOLDEN_DIRECTIONS,
    GOLDEN_DOCS,
    GOLDEN_ENG,
    GOLDEN_IDS,
    GOLDEN_KHK,
)
from tests.support.no_network import no_network  # noqa: F401 - the fixture of the mark

pytestmark = pytest.mark.usefixtures("no_network")

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "eval"
GOLDEN_SUITE = FIXTURES / "golden_suite.json"
GOLDEN_SHA256 = "b707518d39133bd206092e6df28bf0bcf1e42c776ad5c16900995bf783ebc761"


def expected_document() -> dict[str, Any]:
    return {
        "schema": "rinchen.eval.suite/1",
        "name": "golden",
        "kind": "fixture",
        "benchmark": "golden",
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
                "id": GOLDEN_IDS[index],
                "doc": GOLDEN_DOCS[index],
                "text": {
                    "eng_Latn": GOLDEN_ENG[index],
                    "khk_Cyrl": GOLDEN_KHK[index],
                },
            }
            for index in range(10)
        ],
    }


def golden_bytes() -> bytes:
    return GOLDEN_SUITE.read_bytes()


def golden_suite() -> Suite:
    return Suite.from_bytes(golden_bytes())


def test_the_golden_file_is_the_document_the_constants_of_the_golden_corpus_make() -> (
    None
):
    expected = json.dumps(
        expected_document(), sort_keys=True, indent=2, ensure_ascii=False
    )

    assert golden_bytes() == f"{expected}\n".encode()


def test_the_golden_directions_are_those_of_the_golden_corpus() -> None:
    assert tuple(expected_document()["directions"]) == GOLDEN_DIRECTIONS


def test_the_golden_file_is_the_bytes_the_writer_makes_from_what_the_reader_reads() -> (
    None
):
    assert golden_suite().to_bytes() == golden_bytes()


def test_reading_the_golden_file_gives_the_digest_of_its_bytes() -> None:
    suite, digest = read_suite(GOLDEN_SUITE)

    assert digest == GOLDEN_SHA256 == hashlib.sha256(golden_bytes()).hexdigest()
    assert suite == golden_suite()


def test_the_golden_suite_has_the_fields_of_the_spec() -> None:
    suite = golden_suite()

    assert (suite.name, suite.kind, suite.benchmark, suite.split) == (
        "golden",
        "fixture",
        "golden",
        "devtest",
    )
    assert suite.purpose == "report"
    assert suite.directions == GOLDEN_DIRECTIONS
    assert suite.source.dataset is None and suite.source.revision is None
    assert suite.source.pinned is False
    assert dict(suite.source.files) == {} and suite.source.missing_languages == ()
    assert [row.id for row in suite.rows] == list(GOLDEN_IDS)
    assert [row.doc for row in suite.rows] == list(GOLDEN_DOCS)
    assert [row.text["eng_Latn"] for row in suite.rows] == list(GOLDEN_ENG)
    assert [row.text["khk_Cyrl"] for row in suite.rows] == list(GOLDEN_KHK)


def test_the_golden_suite_counts_ten_sentences_in_each_language() -> None:
    assert golden_suite().languages() == {"eng_Latn": 10, "khk_Cyrl": 10}


def test_the_golden_view_of_khalkha_to_english_reads_khalkha_and_scores_against_english() -> (
    None
):
    view = golden_suite().view("khk_Cyrl-eng_Latn")

    assert view.sources == GOLDEN_KHK
    assert view.references == GOLDEN_ENG
    assert view.ids == GOLDEN_IDS
    assert view.docs == GOLDEN_DOCS
    assert (view.src_lang, view.tgt_lang) == ("khk_Cyrl", "eng_Latn")


def test_the_golden_view_of_english_to_khalkha_is_the_mirror_image() -> None:
    view = golden_suite().view("eng_Latn-khk_Cyrl")

    assert view.sources == GOLDEN_ENG
    assert view.references == GOLDEN_KHK
    assert view.docs == GOLDEN_DOCS


def test_the_golden_suite_is_for_reporting_and_a_selection_cannot_read_it() -> None:
    with pytest.raises(ConfigError, match="devtest"):
        golden_suite().selection_view("khk_Cyrl-eng_Latn")


def test_the_fixtures_of_d13_stay_under_50_kb_together() -> None:
    sizes = [path.stat().st_size for path in FIXTURES.rglob("*") if path.is_file()]

    assert golden_bytes().endswith(b"}\n")
    assert len(golden_bytes()) == 2259
    assert sum(sizes) < 50 * 1024
