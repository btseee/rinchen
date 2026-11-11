"""What a parsed report is (D13, spec T14): the frozen dataclasses of `rinchen.eval.report_read`, one
per section of `rinchen.report/1` (see `rinchen.eval.report_format`), so that the model card and the
release stage read a report through types and never through dictionaries.

A `ReportView` holds what a strict reader has checked and nothing it has not: `results` are the
`DirectionResult` of every direction that ran (without D12's block, which is in `robustness`, with
its coverage and its source), `coverage` says what became of every other direction and `comparisons`
and `targets` carry the verdicts the report reached. Every mapping is read-only.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from rinchen.eval.result import DirectionResult
from rinchen.eval.systems import Scalar

__all__ = [
    "BootstrapView",
    "ComparisonView",
    "CoverageView",
    "EvidenceView",
    "ReportView",
    "RobustnessView",
    "RunView",
    "SuiteDirectionView",
    "SuiteSourceView",
    "SuiteView",
    "SystemView",
    "TargetView",
]


@dataclass(frozen=True, slots=True)
class EvidenceView:
    """The kinds of suite the figures come from (sorted) and whether every one is a fixture."""

    kinds: tuple[str, ...]
    fixture_only: bool


@dataclass(frozen=True, slots=True)
class SystemView:
    """A system: its `name`, `kind`, the digests of its model and manifest (None without one)
    and its effective `options`."""

    name: str
    kind: str
    model_sha256: str | None
    manifest_sha256: str | None
    options: Mapping[str, Scalar]


@dataclass(frozen=True, slots=True)
class SuiteDirectionView:
    """A direction the report expects of a suite and the `rows` the suite has for it (None when
    the suite does not declare it)."""

    direction: str
    rows: int | None


@dataclass(frozen=True, slots=True)
class SuiteSourceView:
    """Where a suite came from: its dataset and revision (None for a local file), whether the
    revision was pinned and the registry languages the source lacked."""

    dataset: str | None
    revision: str | None
    pinned: bool
    missing_languages: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class SuiteView:
    """A suite the runs scored: what it is (`kind`, `benchmark`, `split`, `purpose`), the digest of
    its file, its `rows`, a `SuiteDirectionView` for every direction the report expects (the
    standard ones in their order, then the suite's others) and its `source`."""

    name: str
    kind: str
    benchmark: str
    split: str
    purpose: str
    sha256: str
    rows: int
    directions: tuple[SuiteDirectionView, ...]
    source: SuiteSourceView


@dataclass(frozen=True, slots=True)
class RunView:
    """A run, named by its suite, its system and the digest of its `run.json`."""

    suite: str
    system: str
    run_sha256: str


@dataclass(frozen=True, slots=True)
class CoverageView:
    """What became of a direction of a system in a suite: its `state` (a run state, or `absent`
    for a baseline that has no run in the suite) and the `reason` when it did not run."""

    suite: str
    system: str
    direction: str
    state: str
    reason: str | None


@dataclass(frozen=True, slots=True)
class RobustnessView:
    """D12's block of a system with its coverage and its source: the six values, the units each
    rests on and where the block came from."""

    block: Mapping[str, float | None]
    coverage: Mapping[str, Mapping[str, float | int | None]]
    source: Mapping[str, str]


@dataclass(frozen=True, slots=True)
class ComparisonView:
    """A paired comparison of `system` against `baseline` on `metric` in a direction of a suite:
    the `delta`, the percentile interval `ci95`, the two-sided `p_value`, the share of resamples in
    which the system was better and the verdict `significant`; the last four are None with fewer
    than two units (`units`) of the kind `unit`."""

    suite: str
    direction: str
    system: str
    baseline: str
    metric: str
    higher_is_better: bool
    delta: float
    ci95: tuple[float, float] | None
    p_value: float | None
    better_fraction: float | None
    significant: bool
    unit: str
    units: int


@dataclass(frozen=True, slots=True)
class TargetView:
    """A release target as measured: its `status` (`met`, `not_met` or `unmeasured`), the `value`
    found against the `threshold` and the `basis` sentence."""

    id: str
    description: str
    threshold: float | int | None
    value: float | int | None
    status: str
    basis: str


@dataclass(frozen=True, slots=True)
class BootstrapView:
    """What the intervals of the report are."""

    resamples: int
    seed: int
    alpha: float
    interval: str
    rng: str


@dataclass(frozen=True, slots=True)
class ReportView:
    """A parsed `report.json`, section by section."""

    evidence: EvidenceView
    systems: tuple[SystemView, ...]
    suites: tuple[SuiteView, ...]
    runs: tuple[RunView, ...]
    coverage: tuple[CoverageView, ...]
    results: tuple[DirectionResult, ...]
    robustness: Mapping[str, RobustnessView]
    comparisons: tuple[ComparisonView, ...]
    targets: tuple[TargetView, ...]
    bootstrap: BootstrapView
    note_multiple_comparisons: str
