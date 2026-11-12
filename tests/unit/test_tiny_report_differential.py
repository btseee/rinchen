"""Differential check of the tiny report (D15 Task 6): over 1,000 random reports, `to_bytes` writes
the bytes of `json.dumps(..., sort_keys=True, indent=2)` of a dictionary built independently from the
fields of the report, and `render` writes the text of a second renderer that shares no helper with it.

The reports are the golden run cut short, failed with or without facts, with changed figures and
dropped facts, a skipped neural part, another profile, flag and elapsed time (see
`tests/support/tiny_report_random.py`). The test counts the mismatches and reports the count; it is 0.
"""

from __future__ import annotations

from rinchen.tiny.report import parse_report, report_document, to_bytes
from rinchen.tiny.report_markdown import render
from tests.support.tiny_report_random import random_report
from tests.support.tiny_report_reference import (
    reference_bytes,
    reference_document,
    reference_markdown,
)

CASES = 1000


def test_the_bytes_and_the_markdown_equal_an_independent_statement_on_1000_random_reports() -> (
    None
):
    byte_mismatches: list[int] = []
    document_mismatches: list[int] = []
    markdown_mismatches: list[int] = []
    verdicts: dict[str, int] = {"pass": 0, "fail": 0}
    for seed in range(CASES):
        report = random_report(seed)
        verdicts[report.verdict] += 1
        if to_bytes(report) != reference_bytes(report):
            byte_mismatches.append(seed)
        if report_document(report) != reference_document(report):
            document_mismatches.append(seed)
        if render(report) != reference_markdown(report):
            markdown_mismatches.append(seed)
    assert byte_mismatches == [], f"{len(byte_mismatches)} byte mismatches of {CASES}"
    assert document_mismatches == [], (
        f"{len(document_mismatches)} document mismatches of {CASES}"
    )
    assert markdown_mismatches == [], (
        f"{len(markdown_mismatches)} Markdown mismatches of {CASES}"
    )
    # The draw reaches both verdicts often enough for the comparison to mean something.
    assert verdicts["pass"] >= 50
    assert verdicts["fail"] >= 500


def test_every_random_report_reads_back_to_itself_and_renders_alike() -> None:
    for seed in range(0, CASES, 4):
        report = random_report(seed)
        data = to_bytes(report)
        parsed = parse_report(data)
        assert parsed == report, seed
        assert to_bytes(parsed) == data, seed
        assert render(parsed) == render(report), seed


def test_the_random_reports_reach_every_kind_of_ending() -> None:
    reasons: set[str] = set()
    null_blocks: set[str] = set()
    states: set[str] = set()
    for seed in range(CASES):
        report = random_report(seed)
        states.update(record.state for record in report.stages)
        reasons.update(record.reason for record in report.stages if record.reason)
        null_blocks.update(
            name for name, block in report.blocks.items() if block is None
        )
    assert states == {"ok", "skipped", "fail"}
    assert len(reasons) >= 8
    assert null_blocks == set(random_report(0).blocks)
