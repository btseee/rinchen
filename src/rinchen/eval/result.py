"""The result of D13 (spec T11): `result.json`, schema `rinchen.eval/1`, one per (suite, system,
direction) that ran, and the run record `run.json` that lists them (`rinchen.eval.run_record`,
re-exported here with the names of the files).

    {schema, system, suite, direction, n_segments, trained, statuses, metrics, bleu_signature,
     chrf_signature, chrf_pp_signature, unmeasured, ci95, bootstrap, robustness,
     robustness_coverage, robustness_source, manifest_sha, provenance}

Sorted keys, two-space indent, LF, NO timestamp and no path (the bytes of a result are a function
of its inputs; `created` lives in the manifest of the run). A metric with no value is `null` and
its reason is in `unmeasured`, never 0: every metric is either a number in `metrics` or a reason
in `unmeasured`. The robustness block of D12 comes WITH its coverage and its source (the suite
and the digests it came from, and how it was bound to the system), or none of the three. A
result's `manifest_sha` is the digest of the manifest that vouches for the system's model (null
for a baseline or a hypotheses file); `provenance` names the suite, the run's own outputs file and
the whole system record.

`DirectionResult` validates what it holds, whether a command builds it (`build_result`) or it is
read back (`result_from_bytes`): the statuses add up to the segments, the units fit the segments,
`ci95` agrees with the metrics and the bootstrap, and the system of the provenance is the system
of the result. A refusal names the field and never quotes a value.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Final, get_args

from rinchen.errors import DataError
from rinchen.eval.field_rules import (
    document_bytes,
    load_document,
    problem,
    require_digest,
    require_direction,
    require_fields,
    require_flag,
    require_id,
    require_integer,
    require_number,
    require_object,
    require_text,
    system_from,
)
from rinchen.eval.robustness_input import (
    RobustnessInput,
    bind,
    checked_block,
    checked_coverage,
    checked_source,
    source_record,
)
from rinchen.eval.run_record import (
    OUTPUTS_FILE,
    RESULT_FILE,
    RUN_FILE,
    RUN_SCHEMA,
    RunDirection,
    RunRecord,
    RunSettings,
    RunSuite,
    run_from_bytes,
)
from rinchen.eval.scoring import BootstrapRecord, DirectionScores
from rinchen.eval.systems import SystemRecord
from rinchen.infer.engine_types import Status
from rinchen.metrics.registry import METRIC_NAMES

__all__ = [
    "OUTPUTS_FILE",
    "RESULT_FILE",
    "RESULT_SCHEMA",
    "RUN_FILE",
    "RUN_SCHEMA",
    "DirectionResult",
    "Provenance",
    "RunDirection",
    "RunRecord",
    "RunSettings",
    "RunSuite",
    "build_result",
    "result_from_bytes",
    "run_from_bytes",
]

RESULT_SCHEMA: Final = "rinchen.eval/1"
_STATUSES: Final = get_args(Status)
_FIELDS: Final = (
    "system",
    "suite",
    "direction",
    "n_segments",
    "trained",
    "statuses",
    "metrics",
    "bleu_signature",
    "chrf_signature",
    "chrf_pp_signature",
    "unmeasured",
    "ci95",
    "bootstrap",
    "robustness",
    "robustness_coverage",
    "robustness_source",
    "manifest_sha",
    "provenance",
)
_BOOTSTRAP_FIELDS: Final = (
    "resamples",
    "seed",
    "alpha",
    "interval",
    "rng",
    "unit",
    "units",
)
_PROVENANCE_FIELDS: Final = ("suite_sha256", "outputs_sha256", "system")
_ROBUSTNESS_FIELDS: Final = ("robustness", "robustness_coverage", "robustness_source")


@dataclass(frozen=True, slots=True)
class Provenance:
    """Where a result comes from: the SHA-256 of the suite, of the run's own outputs file
    (`hypotheses.jsonl`, whatever the kind of system) and the whole record of the system."""

    suite_sha256: str
    outputs_sha256: str
    system: SystemRecord


@dataclass(frozen=True, slots=True)
class DirectionResult:
    """The result of one system over one direction of a suite (see the module's docstring for
    every field). The mappings are read-only after construction.

    Raises:
        ValueError: a field breaks a rule of the format (the message names it, never its
            value).
    """

    system: str
    suite: str
    direction: str
    n_segments: int
    trained: bool
    statuses: Mapping[str, int]
    metrics: Mapping[str, float | None]
    bleu_signature: str
    chrf_signature: str
    chrf_pp_signature: str
    unmeasured: Mapping[str, str]
    ci95: Mapping[str, list[float] | None]
    bootstrap: BootstrapRecord
    robustness: Mapping[str, float | None] | None
    robustness_coverage: Mapping[str, Mapping[str, float | int | None]] | None
    robustness_source: Mapping[str, str] | None
    manifest_sha: str | None
    provenance: Provenance

    def __post_init__(self) -> None:
        require_id(self.system, "system")
        require_id(self.suite, "suite")
        require_direction(self.direction, "direction")
        require_integer(self.n_segments, "n_segments", minimum=1)
        require_flag(self.trained, "trained")
        for name in ("bleu_signature", "chrf_signature", "chrf_pp_signature"):
            require_text(getattr(self, name), name)
        metrics = _checked_metrics(self.metrics)
        unmeasured = _checked_unmeasured(self.unmeasured, metrics)
        _check_bootstrap(self.bootstrap, self.n_segments)
        ci95 = _checked_intervals(self.ci95, metrics, self.bootstrap.units)
        statuses = _checked_statuses(self.statuses, self.n_segments)
        block, coverage, source = _checked_robustness(
            self.robustness, self.robustness_coverage, self.robustness_source
        )
        _check_provenance(self.provenance, self.system)
        require_digest(self.manifest_sha, "manifest_sha", nullable=True)
        if self.manifest_sha != self.provenance.system.manifest_sha256:
            raise problem(
                "manifest_sha",
                "must be the manifest digest of the system of the provenance",
            )
        for name, value in (
            ("statuses", statuses),
            ("metrics", metrics),
            ("unmeasured", unmeasured),
            ("ci95", ci95),
            ("robustness", block),
            ("robustness_coverage", coverage),
            ("robustness_source", source),
        ):
            object.__setattr__(self, name, value)

    def to_dict(self) -> dict[str, Any]:
        """The document as plain dicts, lists, numbers, strings, booleans and None."""
        return {
            "schema": RESULT_SCHEMA,
            "system": self.system,
            "suite": self.suite,
            "direction": self.direction,
            "n_segments": self.n_segments,
            "trained": self.trained,
            "statuses": dict(self.statuses),
            "metrics": dict(self.metrics),
            "bleu_signature": self.bleu_signature,
            "chrf_signature": self.chrf_signature,
            "chrf_pp_signature": self.chrf_pp_signature,
            "unmeasured": dict(self.unmeasured),
            "ci95": {
                name: None if interval is None else list(interval)
                for name, interval in self.ci95.items()
            },
            "bootstrap": self.bootstrap.to_dict(),
            "robustness": _plain(self.robustness),
            "robustness_coverage": _plain(self.robustness_coverage),
            "robustness_source": _plain(self.robustness_source),
            "manifest_sha": self.manifest_sha,
            "provenance": {
                "suite_sha256": self.provenance.suite_sha256,
                "outputs_sha256": self.provenance.outputs_sha256,
                "system": self.provenance.system.to_dict(),
            },
        }

    def to_bytes(self) -> bytes:
        """The UTF-8 bytes of `result.json`: sorted keys, two-space indent, one final LF."""
        return document_bytes(self.to_dict())


def build_result(
    *,
    scores: DirectionScores,
    system: SystemRecord,
    suite_name: str,
    suite_sha256: str,
    direction: str,
    outputs_sha256: str,
    trained: bool,
    robustness: RobustnessInput | None = None,
) -> DirectionResult:
    """The result of `system` over `direction` of the suite `suite_name` (whose file has the
    SHA-256 `suite_sha256`): `scores` as they are, `outputs_sha256` the digest of the run's own
    `hypotheses.jsonl`, and the robustness block with its coverage and its source when
    `robustness` is given, bound to `system` first.

    Raises:
        ConfigError: `robustness` cannot be bound to `system` (see `bind`).
        ValueError: a field breaks a rule of the format.
    """
    block = coverage = source = None
    if robustness is not None:
        block = robustness.block
        coverage = robustness.coverage
        source = source_record(robustness, bind(robustness, system))
    return DirectionResult(
        system=system.name,
        suite=suite_name,
        direction=direction,
        n_segments=scores.n_segments,
        trained=trained,
        statuses=scores.statuses,
        metrics=scores.metrics,
        bleu_signature=scores.signatures["bleu"],
        chrf_signature=scores.signatures["chrf"],
        chrf_pp_signature=scores.signatures["chrf_pp"],
        unmeasured=scores.unmeasured,
        ci95=scores.ci95,
        bootstrap=scores.bootstrap,
        robustness=block,
        robustness_coverage=coverage,
        robustness_source=source,
        manifest_sha=system.manifest_sha256,
        provenance=Provenance(suite_sha256, outputs_sha256, system),
    )


def result_from_bytes(data: bytes) -> DirectionResult:
    """The result in `data`, a `result.json`.

    Raises:
        DataError: `data` is not UTF-8, JSON or a result of this format; the message names the
            field and never a value.
    """
    document = load_document(data, "an evaluation result")
    try:
        return _result_of(document)
    except ValueError as error:
        raise DataError(f"not an evaluation result: {error}") from error


def _result_of(document: Mapping[str, Any]) -> DirectionResult:
    if document.get("schema") != RESULT_SCHEMA:
        raise problem("schema", "is not the first version of the result format")
    fields = require_fields(document, None, ("schema", *_FIELDS))
    provenance = require_fields(fields["provenance"], "provenance", _PROVENANCE_FIELDS)
    bootstrap = require_fields(fields["bootstrap"], "bootstrap", _BOOTSTRAP_FIELDS)
    return DirectionResult(
        **{
            name: fields[name]
            for name in _FIELDS
            if name not in ("bootstrap", "provenance")
        },
        bootstrap=BootstrapRecord(**bootstrap),
        provenance=Provenance(
            provenance["suite_sha256"],
            provenance["outputs_sha256"],
            system_from(provenance["system"], "provenance.system"),
        ),
    )


def _plain(value: Mapping[str, Any] | None) -> dict[str, Any] | None:
    if value is None:
        return None
    return {
        key: _plain(item) if isinstance(item, Mapping) else item
        for key, item in value.items()
    }


# ---- the checks of the fields ----------------------------------------------------------------


def _checked_statuses(raw: object, n_segments: int) -> Mapping[str, int]:
    statuses = require_object(raw, "statuses")
    counts: dict[str, int] = {}
    for status, count in statuses.items():
        if status not in _STATUSES:
            raise problem("statuses", "names a status the engine does not have")
        counts[status] = require_integer(count, "statuses", minimum=1)
    if sum(counts.values()) != n_segments:
        raise problem("statuses", "the counts must add up to the segments")
    return MappingProxyType(counts)


def _checked_metrics(raw: object) -> Mapping[str, float | None]:
    metrics = require_fields(raw, "metrics", METRIC_NAMES)
    return MappingProxyType(
        {
            name: require_number(metrics[name], f"metrics.{name}", nullable=True)
            for name in METRIC_NAMES
        }
    )


def _checked_unmeasured(
    raw: object, metrics: Mapping[str, float | None]
) -> Mapping[str, str]:
    unmeasured = require_object(raw, "unmeasured")
    reasons: dict[str, str] = {}
    for name, reason in unmeasured.items():
        if name not in METRIC_NAMES:
            raise problem("unmeasured", "has a field that is not part of the format")
        reasons[name] = require_text(reason, f"unmeasured.{name}")
    for name in METRIC_NAMES:
        if (metrics[name] is None) != (name in reasons):
            what = (
                "has a value and a reason"
                if name in reasons
                else "has no value and no reason"
            )
            raise problem("unmeasured", f"the metric {name!r} {what}")
    return MappingProxyType(reasons)


def _check_bootstrap(bootstrap: object, n_segments: int) -> None:
    if not isinstance(bootstrap, BootstrapRecord):
        raise problem("bootstrap", "must be a bootstrap record")
    if bootstrap.units > n_segments:
        raise problem("bootstrap.units", "cannot exceed the segments")
    if bootstrap.unit == "segment" and bootstrap.units != n_segments:
        raise problem(
            "bootstrap.units", "a segment unit has one unit for every segment"
        )


def _checked_intervals(
    raw: object, metrics: Mapping[str, float | None], units: int
) -> Mapping[str, list[float] | None]:
    intervals = require_fields(raw, "ci95", METRIC_NAMES)
    found: dict[str, list[float] | None] = {}
    for name in METRIC_NAMES:
        field = f"ci95.{name}"
        interval = _interval(intervals[name], field)
        if metrics[name] is None and interval is not None:
            raise problem(field, "must be null for a metric that has no value")
        if metrics[name] is not None and (interval is None) != (units < 2):
            raise problem(
                field,
                "must be null with fewer than two units, and an interval otherwise",
            )
        found[name] = interval
    return MappingProxyType(found)


def _interval(raw: object, field: str) -> list[float] | None:
    if raw is None:
        return None
    if not isinstance(raw, list | tuple) or len(raw) != 2:
        raise problem(field, "must be [low, high] or null")
    low = require_number(raw[0], field)
    high = require_number(raw[1], field)
    if low > high:
        raise problem(field, "has its low end above its high end")
    return [low, high]


def _checked_robustness(
    block: object, coverage: object, source: object
) -> tuple[Any, Any, Any]:
    trio = (block, coverage, source)
    if all(part is None for part in trio):
        return None, None, None
    for name, part in zip(_ROBUSTNESS_FIELDS, trio, strict=True):
        if part is None:
            raise problem(
                name,
                "is missing: a robustness block comes with its coverage and its source",
            )
    return (
        checked_block(block, "robustness"),
        checked_coverage(coverage, "robustness_coverage"),
        checked_source(source, "robustness_source"),
    )


def _check_provenance(provenance: object, system: str) -> None:
    if not isinstance(provenance, Provenance):
        raise problem("provenance", "must be a provenance record")
    require_digest(provenance.suite_sha256, "provenance.suite_sha256")
    require_digest(provenance.outputs_sha256, "provenance.outputs_sha256")
    if provenance.system.name != system:
        raise problem("provenance.system.name", "must be the system of the result")
