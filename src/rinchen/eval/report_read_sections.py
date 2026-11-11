"""The descriptive sections of the report of D13 as the strict reader reads them
(`rinchen.eval.report_read`): the evidence, the systems, the suites, the runs and the coverage. Each
reader takes the raw section and returns its views or raises `ValueError` naming the field."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from types import MappingProxyType

from rinchen.eval.field_rules import (
    problem,
    require_choice,
    require_digest,
    require_direction,
    require_fields,
    require_flag,
    require_id,
    require_integer,
    require_number,
    require_object,
    require_text,
)
from rinchen.eval.report_read_rules import (
    COVERAGE_STATES,
    PURPOSES,
    SPLITS,
    SUITE_KINDS,
    SYSTEM_KINDS,
    Names,
    items,
    language,
    text_or_null,
)
from rinchen.eval.report_view import (
    CoverageView,
    EvidenceView,
    RunView,
    SuiteDirectionView,
    SuiteSourceView,
    SuiteView,
    SystemView,
)
from rinchen.eval.systems import Scalar

__all__ = ["coverage", "evidence", "runs", "suites", "systems"]


# ---- evidence, systems, suites ----------------------------------------------------------------------


def evidence(raw: object, suites: Sequence[SuiteView]) -> EvidenceView:
    fields = require_fields(raw, "evidence", ("kinds", "fixture_only"))
    listed = items(fields["kinds"], "evidence.kinds")
    kinds = tuple(
        require_choice(kind, "evidence.kinds", SUITE_KINDS) for kind in listed
    )
    if not kinds or list(kinds) != sorted(set(kinds)):
        raise problem(
            "evidence.kinds", "must be the kinds of the suites, sorted, each once"
        )
    fixture_only = require_flag(fields["fixture_only"], "evidence.fixture_only")
    if kinds != tuple(sorted({suite.kind for suite in suites})):
        raise problem("evidence.kinds", "must be the kinds of the suites of the report")
    if fixture_only != (kinds == ("fixture",)):
        raise problem(
            "evidence.fixture_only", "is true exactly when every suite is a fixture"
        )
    return EvidenceView(kinds, fixture_only)


def systems(raw: object) -> tuple[SystemView, ...]:
    names = ("name", "kind", "model_sha256", "manifest_sha256", "options")
    systems: list[SystemView] = []
    for index, entry in enumerate(items(raw, "systems")):
        where = f"systems.{index}"
        fields = require_fields(entry, where, names)
        systems.append(
            SystemView(
                require_id(fields["name"], f"{where}.name"),
                require_choice(fields["kind"], f"{where}.kind", SYSTEM_KINDS),
                require_digest(
                    fields["model_sha256"], f"{where}.model_sha256", nullable=True
                ),
                require_digest(
                    fields["manifest_sha256"], f"{where}.manifest_sha256", nullable=True
                ),
                _options(fields["options"], f"{where}.options"),
            )
        )
    if len({system.name for system in systems}) != len(systems):
        raise problem("systems", "a system is listed twice")
    return tuple(systems)


def _options(raw: object, field: str) -> Mapping[str, Scalar]:
    options: dict[str, Scalar] = {}
    for name, value in require_object(raw, field).items():
        if isinstance(value, bool | str | int):
            options[name] = value
        else:
            options[name] = require_number(value, field)
    return MappingProxyType(options)


def suites(raw: object) -> tuple[SuiteView, ...]:
    names = (
        "name",
        "kind",
        "benchmark",
        "split",
        "purpose",
        "sha256",
        "rows",
        "directions",
        "source",
    )
    suites: list[SuiteView] = []
    for index, entry in enumerate(items(raw, "suites")):
        where = f"suites.{index}"
        fields = require_fields(entry, where, names)
        split = require_choice(fields["split"], f"{where}.split", SPLITS)
        if split == "dev":
            raise problem(
                f"{where}.split",
                "a dev suite is for choosing a model and is never reported",
            )
        suites.append(
            SuiteView(
                name=require_id(fields["name"], f"{where}.name"),
                kind=require_choice(fields["kind"], f"{where}.kind", SUITE_KINDS),
                benchmark=require_id(fields["benchmark"], f"{where}.benchmark"),
                split=split,
                purpose=require_choice(fields["purpose"], f"{where}.purpose", PURPOSES),
                sha256=require_digest(fields["sha256"], f"{where}.sha256"),
                rows=require_integer(fields["rows"], f"{where}.rows", minimum=1),
                directions=_suite_directions(
                    fields["directions"], f"{where}.directions"
                ),
                source=_suite_source(fields["source"], f"{where}.source"),
            )
        )
        if suites[-1].purpose != "report":
            raise problem(f"{where}.purpose", "a reported suite has the purpose report")
    if len({suite.name for suite in suites}) != len(suites):
        raise problem("suites", "a suite is listed twice")
    return tuple(suites)


def _suite_directions(raw: object, field: str) -> tuple[SuiteDirectionView, ...]:
    directions: list[SuiteDirectionView] = []
    for index, entry in enumerate(items(raw, field)):
        where = f"{field}.{index}"
        fields = require_fields(entry, where, ("direction", "rows"))
        rows = fields["rows"]
        directions.append(
            SuiteDirectionView(
                require_direction(fields["direction"], f"{where}.direction"),
                None
                if rows is None
                else require_integer(rows, f"{where}.rows", minimum=1),
            )
        )
    if len({d.direction for d in directions}) != len(directions):
        raise problem(field, "a direction is listed twice")
    return tuple(directions)


def _suite_source(raw: object, field: str) -> SuiteSourceView:
    fields = require_fields(
        raw, field, ("dataset", "revision", "pinned", "missing_languages")
    )
    missing = tuple(
        language(code, f"{field}.missing_languages")
        for code in items(fields["missing_languages"], f"{field}.missing_languages")
    )
    return SuiteSourceView(
        text_or_null(fields["dataset"], f"{field}.dataset"),
        text_or_null(fields["revision"], f"{field}.revision"),
        require_flag(fields["pinned"], f"{field}.pinned"),
        missing,
    )


# ---- runs, coverage, results ----------------------------------------------------------------------------------


def runs(raw: object, names: Names) -> tuple[RunView, ...]:
    runs: list[RunView] = []
    for index, entry in enumerate(items(raw, "runs")):
        where = f"runs.{index}"
        fields = require_fields(entry, where, ("suite", "system", "run_sha256"))
        runs.append(
            RunView(
                names.suite(fields["suite"], f"{where}.suite").name,
                names.system(fields["system"], f"{where}.system"),
                require_digest(fields["run_sha256"], f"{where}.run_sha256"),
            )
        )
    if len({(run.suite, run.system) for run in runs}) != len(runs):
        raise problem("runs", "a run is listed twice")
    return tuple(runs)


def coverage(raw: object, names: Names) -> tuple[CoverageView, ...]:
    rows: list[CoverageView] = []
    for index, entry in enumerate(items(raw, "coverage")):
        where = f"coverage.{index}"
        fields = require_fields(
            entry, where, ("suite", "system", "direction", "state", "reason")
        )
        suite = names.suite(fields["suite"], f"{where}.suite")
        state = require_choice(fields["state"], f"{where}.state", COVERAGE_STATES)
        reason = fields["reason"]
        if (state == "run") != (reason is None):
            raise problem(f"{where}.reason", "is null exactly for a direction that ran")
        rows.append(
            CoverageView(
                suite.name,
                names.system(fields["system"], f"{where}.system"),
                names.declared(suite, fields["direction"], f"{where}.direction"),
                state,
                None if reason is None else require_text(reason, f"{where}.reason"),
            )
        )
    return tuple(rows)
