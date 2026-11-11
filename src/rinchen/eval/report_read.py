"""The strict reader of the report of D13 (spec T14): `report_from_bytes` turns the bytes of a
`report.json` into a `ReportView` (`rinchen.eval.report_view`), the one thing the model card and the
release stage read.

What it holds is what the builder writes and nothing else: the schema `rinchen.report/1`, exactly
the sections of the format and each of its type, every result a valid `DirectionResult` (the strict
reader of `result.json` is used for it), every robustness block with its coverage and its source,
every name and direction one the report lists, every comparison on a count metric whose `significant`
is the verdict the p-value (below the report's alpha) and the delta give, a `dev` suite never, the
evidence the kinds of the suites and the note the counts of what the report holds. No string, key
included, is an absolute path: a report can be uploaded and names no local file.

A refusal is a `DataError` that names the section and the entry (`comparisons.3.metric`) and NEVER
quotes a value or a key of the file. This module imports no reader of a suite file: a card never
needs one, so `report_read_rules` keeps its own copy of the few names of the suite format and a test
pins it equal. The descriptive sections (evidence, systems, suites, runs, coverage) are read by
`report_read_sections`; this module reads the rest and puts the views together.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from types import MappingProxyType
from typing import Any, Final

from rinchen.errors import DataError
from rinchen.eval import report_read_sections as sections
from rinchen.eval.bootstrap import MIN_RESAMPLES, RNG_NAME
from rinchen.eval.field_rules import (
    document_bytes,
    load_document,
    problem,
    require_choice,
    require_fields,
    require_flag,
    require_id,
    require_integer,
    require_number,
    require_object,
    require_text,
)
from rinchen.eval.report_format import (
    REPORT_SCHEMA,
    SECTIONS,
    count_intervals,
    multiple_comparisons_note,
    require_no_absolute_path,
)
from rinchen.eval.report_read_rules import (
    PURPOSES,
    SPLITS,
    SUITE_KINDS,
    Names,
    items,
    number_or_null,
)
from rinchen.eval.report_view import (
    BootstrapView,
    ComparisonView,
    ReportView,
    RobustnessView,
    RunView,
    TargetView,
)
from rinchen.eval.result import DirectionResult, result_from_bytes
from rinchen.eval.robustness_input import (
    checked_block,
    checked_coverage,
    checked_source,
)
from rinchen.eval.scoring import INTERVAL_KIND, UNIT_NAMES
from rinchen.eval.targets import TARGET_STATUSES
from rinchen.metrics.registry import COUNT_METRICS, HIGHER_IS_BETTER

__all__ = [
    "PURPOSES",
    "SPLITS",
    "SUITE_KINDS",
    "ReportView",
    "report_from_bytes",
]

_ROBUSTNESS_FIELDS: Final = ("robustness", "robustness_coverage", "robustness_source")
_MAX_SEED: Final = 2**64 - 1


def report_from_bytes(data: bytes) -> ReportView:
    """The report in `data`, the bytes of a `report.json`.

    Raises:
        DataError: `data` is not UTF-8, JSON or a report of this format; the message names the
            section and the entry and never a value.
    """
    document = load_document(data, "an evaluation report")
    try:
        return _report_of(document)
    except ValueError as error:
        raise DataError(f"not an evaluation report: {error}") from error


def _report_of(document: Mapping[str, Any]) -> ReportView:
    if document.get("schema") != REPORT_SCHEMA:
        raise problem("schema", "is not the first version of the report format")
    fields = require_fields(document, None, SECTIONS)
    systems = sections.systems(fields["systems"])
    suites = sections.suites(fields["suites"])
    evidence = sections.evidence(fields["evidence"], suites)
    names = Names(
        tuple(system.name for system in systems),
        {suite.name: suite for suite in suites},
    )
    runs = sections.runs(fields["runs"], names)
    coverage = sections.coverage(fields["coverage"], names)
    results = _results(fields["results"], names, runs)
    robustness = _robustness(fields["robustness"], names)
    bootstrap = _bootstrap(fields["bootstrap"])
    comparisons = _comparisons(fields["comparisons"], names, bootstrap.alpha)
    targets = _targets(fields["targets"])
    note = _note(fields["note_multiple_comparisons"], fields)
    require_no_absolute_path(document)
    return ReportView(
        evidence=evidence,
        systems=systems,
        suites=suites,
        runs=runs,
        coverage=coverage,
        results=results,
        robustness=robustness,
        comparisons=comparisons,
        targets=targets,
        bootstrap=bootstrap,
        note_multiple_comparisons=note,
    )


def _results(
    raw: object, names: Names, runs: Sequence[RunView]
) -> tuple[DirectionResult, ...]:
    results: list[DirectionResult] = []
    held = {(run.suite, run.system) for run in runs}
    for index, entry in enumerate(items(raw, "results")):
        where = f"results.{index}"
        document = require_object(entry, where)
        if any(name in document for name in _ROBUSTNESS_FIELDS):
            raise problem(where, "has a field that is not part of the format")
        try:
            result = result_from_bytes(
                document_bytes({**document, **dict.fromkeys(_ROBUSTNESS_FIELDS)})
            )
        except (DataError, ValueError) as error:
            raise problem(where, f"is not a valid result ({error})") from error
        suite = names.suite(result.suite, f"{where}.suite")
        names.declared(suite, result.direction, f"{where}.direction")
        if (result.suite, result.system) not in held:
            raise problem(f"{where}.system", "has no run in the report for this suite")
        results.append(result)
    return tuple(results)


def _robustness(raw: object, names: Names) -> Mapping[str, RobustnessView]:
    found: dict[str, RobustnessView] = {}
    for system, entry in require_object(raw, "robustness").items():
        require_id(system, "robustness")
        names.system(system, "robustness")
        where = f"robustness.{system}"
        fields = require_fields(entry, where, ("block", "coverage", "source"))
        found[system] = RobustnessView(
            checked_block(fields["block"], f"{where}.block"),
            checked_coverage(fields["coverage"], f"{where}.coverage"),
            checked_source(fields["source"], f"{where}.source"),
        )
    return MappingProxyType(found)


# ---- comparisons, targets, bootstrap, note ----------------------------------------------------------------


def _bootstrap(raw: object) -> BootstrapView:
    fields = require_fields(
        raw, "bootstrap", ("resamples", "seed", "alpha", "interval", "rng")
    )
    alpha = fields["alpha"]
    if not isinstance(alpha, float) or not 0 < alpha < 0.5:
        raise problem("bootstrap.alpha", "must be a number above 0 and below 0.5")
    if fields["interval"] != INTERVAL_KIND:
        raise problem("bootstrap.interval", f"must be {INTERVAL_KIND}")
    if fields["rng"] != RNG_NAME:
        raise problem("bootstrap.rng", f"must be {RNG_NAME}")
    return BootstrapView(
        require_integer(
            fields["resamples"], "bootstrap.resamples", minimum=MIN_RESAMPLES
        ),
        require_integer(fields["seed"], "bootstrap.seed", maximum=_MAX_SEED),
        alpha,
        INTERVAL_KIND,
        RNG_NAME,
    )


def _comparisons(raw: object, names: Names, alpha: float) -> tuple[ComparisonView, ...]:
    keys = (
        "suite",
        "direction",
        "system",
        "baseline",
        "metric",
        "higher_is_better",
        "delta",
        "ci95",
        "p_value",
        "better_fraction",
        "significant",
        "unit",
        "units",
    )
    comparisons: list[ComparisonView] = []
    for index, entry in enumerate(items(raw, "comparisons")):
        where = f"comparisons.{index}"
        fields = require_fields(entry, where, keys)
        suite = names.suite(fields["suite"], f"{where}.suite")
        metric = require_choice(fields["metric"], f"{where}.metric", COUNT_METRICS)
        higher_is_better = require_flag(
            fields["higher_is_better"], f"{where}.higher_is_better"
        )
        if higher_is_better != HIGHER_IS_BETTER[metric]:  # type: ignore[index]
            raise problem(
                f"{where}.higher_is_better", "must be which way the metric is better"
            )
        units = require_integer(fields["units"], f"{where}.units", minimum=1)
        delta = require_number(fields["delta"], f"{where}.delta")
        interval = _interval(fields["ci95"], f"{where}.ci95")
        p_value = _share(fields["p_value"], f"{where}.p_value")
        better = _share(fields["better_fraction"], f"{where}.better_fraction")
        for name, value in (
            ("ci95", interval),
            ("p_value", p_value),
            ("better_fraction", better),
        ):
            if (value is None) != (units < 2):
                raise problem(
                    f"{where}.{name}",
                    "is null exactly when the comparison has fewer than two units",
                )
        significant = require_flag(fields["significant"], f"{where}.significant")
        improved = delta > 0 if higher_is_better else delta < 0
        if significant != (p_value is not None and p_value < alpha and improved):
            raise problem(
                f"{where}.significant",
                "must be a p-value below alpha with the difference in favour of the system",
            )
        comparisons.append(
            ComparisonView(
                suite.name,
                names.declared(suite, fields["direction"], f"{where}.direction"),
                names.system(fields["system"], f"{where}.system"),
                names.system(fields["baseline"], f"{where}.baseline"),
                metric,
                higher_is_better,
                delta,
                interval,
                p_value,
                better,
                significant,
                require_choice(fields["unit"], f"{where}.unit", UNIT_NAMES),
                units,
            )
        )
    return tuple(comparisons)


def _interval(raw: object, field: str) -> tuple[float, float] | None:
    if raw is None:
        return None
    if not isinstance(raw, list) or len(raw) != 2:
        raise problem(field, "must be [low, high] or null")
    low, high = require_number(raw[0], field), require_number(raw[1], field)
    if low > high:
        raise problem(field, "has its low end above its high end")
    return (low, high)


def _share(raw: object, field: str) -> float | None:
    if raw is None:
        return None
    value = require_number(raw, field)
    if not 0 <= value <= 1:
        raise problem(field, "must be a number from 0 to 1 or null")
    return value


def _targets(raw: object) -> tuple[TargetView, ...]:
    targets: list[TargetView] = []
    for index, entry in enumerate(items(raw, "targets")):
        where = f"targets.{index}"
        fields = require_fields(
            entry, where, ("id", "description", "threshold", "value", "status", "basis")
        )
        targets.append(
            TargetView(
                require_id(fields["id"], f"{where}.id"),
                require_text(fields["description"], f"{where}.description"),
                number_or_null(fields["threshold"], f"{where}.threshold"),
                number_or_null(fields["value"], f"{where}.value"),
                require_choice(fields["status"], f"{where}.status", TARGET_STATUSES),
                require_text(fields["basis"], f"{where}.basis"),
            )
        )
    if len({target.id for target in targets}) != len(targets):
        raise problem("targets", "a target is listed twice")
    return tuple(targets)


def _note(raw: object, fields: Mapping[str, Any]) -> str:
    expected = multiple_comparisons_note(
        count_intervals(fields["results"], fields["comparisons"]),
        len(fields["comparisons"]),
    )
    if raw != expected:
        raise problem(
            "note_multiple_comparisons",
            "must be the sentence for the intervals and comparisons the report holds",
        )
    return expected
