"""Builds the report of the golden run, or of a changed one, under the shipped profile (D15 Task 6).

This is the one support module that imports `rinchen.tiny.report`: the observations themselves
(`tests/support/tiny_report_inputs.py`) import nothing of it, so the tests of the blocks can run
before a report exists.
"""

from __future__ import annotations

from collections.abc import Sequence

from rinchen.config.tiny import TinyConfig
from rinchen.tiny.context import StageResult
from rinchen.tiny.report import Inputs, Report, build_report
from tests.support.tiny_report_inputs import (
    CONFIG_SHA256,
    FIXTURE_DIGESTS,
    GOLDEN_ELAPSED,
    LOCK_SHA256,
    facts_of,
    failed,
    golden_results,
    replaced,
    stopped_after,
    without_neural_part,
)
from tests.support.tiny_stage_inputs import shipped_config


def golden_inputs() -> Inputs:
    return Inputs(CONFIG_SHA256, dict(FIXTURE_DIGESTS), LOCK_SHA256)


def config() -> TinyConfig:
    return shipped_config()


VARIANT_NAMES = (
    "failed after nine",
    "stopped after two",
    "missed parity",
    "no neural part",
    "overrun",
    "nothing ran",
)


def variant_reports() -> dict[str, Report]:
    """Reports of runs that ended in other ways: a stop after a failure, an early stop, a missed parity
    gate with its figures, a skipped neural part, an overrun, a run in which nothing ran."""
    parity = dict(facts_of(golden_results(), "parity"))
    parity.update(exact=60, exact_share=60 / 70, passes=False)
    missed = replaced(golden_results(), "parity", failed("parity", facts=parity))
    no_neural = replaced(golden_results(), "evaluate", without_neural_part())
    return {
        "failed after nine": build(stopped_after(9, failing=True)),
        "stopped after two": build(stopped_after(2)),
        "missed parity": build(missed),
        "no neural part": build(no_neural),
        "overrun": build(elapsed=5000.0),
        "nothing ran": build([]),
    }


def build(
    results: Sequence[StageResult] | None = None,
    *,
    elapsed: float = GOLDEN_ELAPSED,
    qe_fake: bool = True,
    profile: str = "tiny",
) -> Report:
    """The report of `results` (the golden run by default) under the shipped profile."""
    chosen = golden_results() if results is None else results
    return build_report(profile, qe_fake, golden_inputs(), chosen, elapsed, config())
