"""Property test of the tiny pipeline's report (D15 Task 6, G6.5): over 300 generated sets of
observations (the golden run cut short, failed with or without facts, with other figures, a skipped neural
part, another profile, flag and elapsed time), the bytes of the report parse back to an equal report and
hold no path, time or duration, the Markdown rendered from the parsed bytes is the Markdown of the
report, every stage and every threshold is listed in order, every reason is a closed one and the
verdict and the exit code are what an independent statement of the rule gives.
"""

from __future__ import annotations

import json

from hypothesis import given, settings
from hypothesis import strategies as st

from rinchen.config.tiny import STAGE_IDS
from rinchen.tiny.report import exit_code, parse_report, to_bytes
from rinchen.tiny.report_markdown import render
from rinchen.tiny.states import PART_IDS, SKIP_EARLIER_FAILED, is_closed_reason
from rinchen.tiny.thresholds import THRESHOLD_IDS
from tests.support.tiny_report_random import random_report
from tests.support.tiny_report_scans import text_violations, violations

SEEDS = st.integers(min_value=0, max_value=2**32 - 1)


@given(SEEDS)
@settings(max_examples=300, deadline=None)
def test_the_bytes_parse_back_to_an_equal_report_and_hold_nothing_that_varies(
    seed: int,
) -> None:
    report = random_report(seed)
    data = to_bytes(report)
    parsed = parse_report(data)
    assert parsed == report
    assert to_bytes(parsed) == data
    assert violations(data) == []
    markdown = render(report)
    assert render(parsed) == markdown
    assert text_violations(markdown) == []


@given(SEEDS)
@settings(max_examples=300, deadline=None)
def test_every_stage_and_every_threshold_is_listed_and_every_reason_is_closed(
    seed: int,
) -> None:
    report = random_report(seed)
    assert [record.id for record in report.stages] == list(STAGE_IDS)
    assert [result.id for result in report.thresholds] == list(THRESHOLD_IDS)
    for record in report.stages:
        assert (record.reason is None) == (record.state == "ok")
        if record.reason is not None:
            assert is_closed_reason(record.reason)
        for part in record.parts:
            assert part.reason is None or is_closed_reason(part.reason)
        if record.state != "ok":
            assert [part.id for part in record.parts] == list(
                PART_IDS.get(record.id, ())
            )


@given(SEEDS)
@settings(max_examples=300, deadline=None)
def test_the_verdict_and_the_exit_code_are_what_the_rule_says(seed: int) -> None:
    report = random_report(seed)
    document = json.loads(to_bytes(report))
    failing = any(stage["state"] == "fail" for stage in document["stages"]) or any(
        threshold["verdict"] != "pass" for threshold in document["thresholds"]
    )
    assert document["verdict"] == ("fail" if failing else "pass")
    assert exit_code(report) == (4 if failing else 0)
    # A stage that did not run is never a pass by absence: it is listed, skipped for its reason.
    assert all(
        stage["reason"] == SKIP_EARLIER_FAILED
        for stage in document["stages"]
        if stage["state"] == "skipped"
    )


@given(SEEDS)
@settings(max_examples=300, deadline=None)
def test_the_wall_time_threshold_never_holds_a_duration(seed: int) -> None:
    document = json.loads(to_bytes(random_report(seed)))
    wall = document["thresholds"][-1]
    assert wall["id"] == "wall_time"
    assert wall["value"] is None
    assert wall["bound"] == "<= 1200"
    # A threshold without a value is a failure, and the wall time is the one that is not.
    for threshold in document["thresholds"][:-1]:
        assert threshold["value"] is not None or threshold["verdict"] == "fail"
