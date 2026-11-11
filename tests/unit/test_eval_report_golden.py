"""The golden report of D13 (Task 8): `tests/fixtures/eval/golden_report.json` and `golden_report.md`
are the report of the golden runs of `alpha` and `beta` (baseline `alpha`) written byte for byte,
and what `build_report`, `report_to_bytes` and `render_markdown` make of those runs must equal them.

The JSON was cross-checked with an independent rendering: `tests.support.eval_report_golden` builds
the same document from the literals of the golden tables, the reference derivations of this task and
the standard library only, and `json.dumps` of that document is the file's bytes. The Markdown was
read once by a person and is pinned as it renders. A floating point figure of BLEU is pinned to its
last digit: another C library's `exp` and `log` would fail it, which is the intended alarm
(`docs` of D14 name it); after an intended change of sacrebleu or of the layout, regenerate both files
from `build_report` and `render_markdown` over `tests.support.eval_report.golden_runs()` and check the
JSON against the independent rendering again before committing them.
"""

from __future__ import annotations

import hashlib
import json

import pytest

from rinchen.eval.report import build_report, report_to_bytes
from rinchen.eval.report_markdown import render_markdown
from rinchen.eval.report_read import report_from_bytes
from tests.support.eval_report import golden_runs, golden_suites, shipped_config
from tests.support.eval_report_golden import (
    FIXTURES,
    GOLDEN_REPORT_JSON,
    GOLDEN_REPORT_MD,
    golden_report_document,
)
from tests.support.no_network import no_network  # noqa: F401 - the fixture of the mark

pytestmark = pytest.mark.usefixtures("no_network")

# `sha256sum` of the two files, taken from the files and not from the code.
GOLDEN_JSON_SHA256 = "c8d2095fb0109e7682a3558261778031ca3d34fdce65f96f43f766755874ce30"
GOLDEN_MD_SHA256 = "a6b30c0dcceec9e08bc9f2b0a7ed22d38c722d645e00a9648087efde1d6f8a3f"


def golden_report() -> dict[str, object]:
    return build_report(
        golden_runs(), golden_suites(), baseline="alpha", config=shipped_config()
    )


def test_the_report_of_the_golden_runs_is_the_golden_json_byte_for_byte() -> None:
    assert report_to_bytes(golden_report()) == GOLDEN_REPORT_JSON.read_bytes()


def test_the_markdown_of_the_golden_report_is_the_golden_markdown_byte_for_byte() -> (
    None
):
    markdown = render_markdown(golden_report())

    assert markdown.encode() == GOLDEN_REPORT_MD.read_bytes()


def test_the_golden_json_is_the_independent_rendering_of_the_literals() -> None:
    runs = golden_runs()
    digests = {run.record.system.name: run.run_sha256 for run in runs}
    expected = json.dumps(
        golden_report_document(digests), sort_keys=True, indent=2, ensure_ascii=False
    )

    assert GOLDEN_REPORT_JSON.read_bytes() == f"{expected}\n".encode()


def test_the_report_equals_the_independent_document_value_by_value() -> None:
    runs = golden_runs()
    digests = {run.record.system.name: run.run_sha256 for run in runs}
    report = json.loads(report_to_bytes(golden_report()))

    expected = golden_report_document(digests)

    assert report == expected


def test_the_golden_files_are_the_ones_whose_digests_are_pinned() -> None:
    assert (
        hashlib.sha256(GOLDEN_REPORT_JSON.read_bytes()).hexdigest()
        == GOLDEN_JSON_SHA256
    )
    assert hashlib.sha256(GOLDEN_REPORT_MD.read_bytes()).hexdigest() == GOLDEN_MD_SHA256


def test_the_golden_markdown_is_under_eight_kilobytes() -> None:
    assert GOLDEN_REPORT_MD.stat().st_size < 8 * 1024


def test_the_fixtures_of_the_evaluation_stay_under_fifty_kilobytes_together() -> None:
    total = sum(path.stat().st_size for path in FIXTURES.iterdir() if path.is_file())

    assert total <= 50_000


def test_the_golden_json_is_a_report_the_reader_accepts() -> None:
    view = report_from_bytes(GOLDEN_REPORT_JSON.read_bytes())

    assert view.evidence.fixture_only is True
    assert len(view.results) == 4 and len(view.comparisons) == 4


def test_the_golden_files_hold_no_path_and_no_timestamp() -> None:
    for path in (GOLDEN_REPORT_JSON, GOLDEN_REPORT_MD):
        text = path.read_text(encoding="utf-8")
        assert "created" not in text
        assert "/tmp" not in text and "/home" not in text
        assert "T00:" not in text and "UTC" not in text
