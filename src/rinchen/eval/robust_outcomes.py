"""The shapes of a scored section of a robustness suite (D12, spec T13): a `RowOutcome` is
what one row came to (what the paired bootstrap of D13 resamples) and a `SectionResult` the
figures of a section, with the helpers `robust_sections` builds them with.

The unit of a section is a `row` (numbers, placeholders, script, mong) or a `mention` (entities
and terms: a row with two names counts two). A row or a mention that is not scored is counted
under a reason (`no_numbers`, `source_reference_disagree`, `no_placeholders`,
`absent_in_source`, `no_letters`), never dropped. The `value` of a section is `None` when nothing
was scored: there is no division, and no 1.0 or 0 that a reader could mistake for a measurement.
Every hypothesis of the section is counted by its status, the excluded rows' too, so a failure
the engine flagged itself stays visible beside the metric."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from types import MappingProxyType
from typing import Any, Final

from rinchen.eval.robust_hypotheses import Hypothesis
from rinchen.eval.robust_suite import RequestKey, SuiteRow

__all__ = [
    "Breakdown",
    "Details",
    "Hypotheses",
    "RowOutcome",
    "SectionResult",
    "Tally",
    "build_section",
    "direction",
    "excluded_outcome",
    "scored_outcome",
]

Hypotheses = Mapping[RequestKey, Hypothesis]
Details = Mapping[str, float | int | None]
Breakdown = Mapping[str, Mapping[str, int]]

_UNKNOWN_SOURCE: Final = "und"


@dataclass(frozen=True, slots=True)
class RowOutcome:
    """What one row came to: whether it was `scored` (else the `reason` it was not), whether
    it was `ok`, how many numbers, placeholders or mentions were `expected` and `found`, and a
    `value` (the purity of a script row, the sentence chrF of a mong row's `mong` arm). A field
    that does not apply is None."""

    id: str
    category: str
    direction: str
    scored: bool
    reason: str | None
    ok: bool | None
    expected: int | None
    found: int | None
    value: float | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SectionResult:
    """The figures of a section: its `unit` (`row` or `mention`), the number of `rows`, how
    many units were `scored` and `ok`, the `excluded` ones by reason (a reason that did not
    occur is absent), the headline `value` (None when nothing was scored), the `details` that
    depend on the section, the counts by category and by direction (`<src>-<tgt>`, `und` for
    an unknown source language: scored units only), the `statuses` of every hypothesis of the
    section and the `outcomes` of the rows in file order."""

    unit: str
    rows: int
    scored: int
    excluded: Mapping[str, int]
    ok: int
    value: float | None
    details: Details
    by_category: Breakdown
    by_direction: Breakdown
    statuses: Mapping[str, int]
    outcomes: tuple[RowOutcome, ...] = field(repr=False)

    def __post_init__(self) -> None:
        for name in ("excluded", "details", "statuses"):
            object.__setattr__(self, name, MappingProxyType(dict(getattr(self, name))))
        for name in ("by_category", "by_direction"):
            frozen = {
                key: MappingProxyType(dict(counts))
                for key, counts in getattr(self, name).items()
            }
            object.__setattr__(self, name, MappingProxyType(frozen))
        object.__setattr__(self, "outcomes", tuple(self.outcomes))

    @property
    def excluded_total(self) -> int:
        """The units left out of the value, under every reason."""
        return sum(self.excluded.values())

    @property
    def total(self) -> int:
        """The units the section holds, in its unit: those scored and those left out."""
        return self.scored + self.excluded_total

    def to_dict(self) -> dict[str, Any]:
        """The section as the document writes it (the outcomes go under `rows`)."""
        return {
            "unit": self.unit,
            "rows": self.rows,
            "scored": self.scored,
            "excluded": dict(self.excluded),
            "ok": self.ok,
            "value": self.value,
            "details": dict(self.details),
            "by_category": {k: dict(v) for k, v in self.by_category.items()},
            "by_direction": {k: dict(v) for k, v in self.by_direction.items()},
            "statuses": dict(self.statuses),
        }


def direction(row: SuiteRow) -> str:
    return f"{row.src_lang or _UNKNOWN_SOURCE}-{row.tgt_lang}"


def status_counts(section: str, hypotheses: Hypotheses) -> Counter[str]:
    return Counter(
        hypothesis.status
        for key, hypothesis in hypotheses.items()
        if key.section == section
    )


def excluded_outcome(row: SuiteRow, reason: str) -> RowOutcome:
    return RowOutcome(
        row.id, row.category, direction(row), False, reason, None, None, None, None
    )


def scored_outcome(
    row: SuiteRow,
    *,
    ok: bool,
    expected: int | None = None,
    found: int | None = None,
    value: float | None = None,
) -> RowOutcome:
    return RowOutcome(
        row.id, row.category, direction(row), True, None, ok, expected, found, value
    )


class Tally:
    """The units scored and ok in a section, and the same by category and by direction."""

    def __init__(self) -> None:
        self.scored = 0
        self.ok = 0
        self.by_category: dict[str, dict[str, int]] = {}
        self.by_direction: dict[str, dict[str, int]] = {}

    def add(self, row: SuiteRow, scored: int, ok: int) -> None:
        self.scored += scored
        self.ok += ok
        for table, name in (
            (self.by_category, row.category),
            (self.by_direction, direction(row)),
        ):
            counts = table.setdefault(name, {"scored": 0, "ok": 0})
            counts["scored"] += scored
            counts["ok"] += ok


def build_section(
    *,
    section: str,
    unit: str,
    rows: Sequence[SuiteRow],
    tally: Tally,
    excluded: Mapping[str, int],
    value: float | None,
    details: Details,
    hypotheses: Hypotheses,
    outcomes: list[RowOutcome],
) -> SectionResult:
    return SectionResult(
        unit=unit,
        rows=len(rows),
        scored=tally.scored,
        excluded={reason: count for reason, count in excluded.items() if count},
        ok=tally.ok,
        value=value,
        details=details,
        by_category=tally.by_category,
        by_direction=tally.by_direction,
        statuses=status_counts(section, hypotheses),
        outcomes=tuple(outcomes),
    )
