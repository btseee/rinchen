"""D12's robustness files as D13 takes them in (spec T12): the six values of the robustness block
WITH the coverage they rest on, and the digests that say whose they are.

`read_robustness` takes the BYTES the command hashed: the narrow `report.json` (schema
`rinchen.report.robustness/1`, the carrier of the block and of its `coverage`) and
`robustness.json` (schema `rinchen.robustness/1`). The report names the digest of
`robustness.json`, repeats the block and carries the coverage, so every part is checked against
the other file: the digest, the system and the suite the report names, the block value by value,
and the coverage against the sections (their `scored` and `excluded` counts and the two chrF
scores of the retention). A block without its coverage hides how many units each value left out,
so the two travel together everywhere D13 writes them.

`bind` says whether the block may sit in the result of a system. A `rinchen` system is bound by the
model and manifest digests of the robustness run (the block would describe another model
otherwise) and NEVER by its name; any other system is bound by the weaker name, which a result
records as such. A refusal names the two system names (ids only) and no digest.

The block and its coverage are D12's own shapes; `BLOCK_KEYS` is pinned against D12's
`robustness_block` by a test, so a change on that side fails here loudly.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Final, NamedTuple

from rinchen.errors import ConfigError, DataError
from rinchen.eval.field_rules import (
    load_document,
    problem,
    require_digest,
    require_fields,
    require_id,
    require_integer,
    require_number,
    require_object,
)
from rinchen.eval.robust_report import REPORT_SCHEMA, ROBUSTNESS_FILE
from rinchen.eval.robust_suite_rules import is_id
from rinchen.eval.robustness import RESULT_SCHEMA as ROBUSTNESS_SCHEMA
from rinchen.eval.systems import SystemRecord

__all__ = [
    "BINDINGS",
    "BLOCK_KEYS",
    "RobustnessInput",
    "bind",
    "checked_block",
    "checked_coverage",
    "checked_source",
    "read_robustness",
    "source_record",
]

# The six values of D12's block, in its order, and the section of the robustness suite each is
# the headline of.
BLOCK_KEYS: Final = (
    "numbers_acc",
    "placeholder_acc",
    "term_acc",
    "script_purity",
    "entity_acc",
    "mong_source_retention",
)
_SECTION_OF: Final = {
    "numbers_acc": "numbers",
    "placeholder_acc": "placeholders",
    "term_acc": "terms",
    "script_purity": "script",
    "entity_acc": "entities",
    "mong_source_retention": "mong",
}
_RETENTION: Final = "mong_source_retention"
_COUNTS: Final = ("scored", "excluded", "total")
_RETENTION_CHRF: Final = ("chrf_cyrl", "chrf_mong")
BINDINGS: Final = ("model_digest", "name")
_SOURCE_FIELDS: Final = (
    "suite",
    "suite_sha256",
    "robustness_sha256",
    "report_sha256",
    "binding",
)
_REPORT_FIELDS: Final = (
    "schema",
    "system",
    "suite",
    "robustness",
    "coverage",
    "robustness_file",
    "robustness_sha256",
)
_RESULT_FIELDS: Final = ("schema", "system", "suite", "robustness", "sections", "rows")
_REPORT_SYSTEM: Final = ("name", "model_sha256", "manifest_sha256")
_RESULT_SYSTEM: Final = (*_REPORT_SYSTEM, "hypotheses_sha256", "options")
_SUITE_FIELDS: Final = ("name", "sha256")
_FILE: Final = f"{ROBUSTNESS_FILE}:"

Block = Mapping[str, float | None]
Coverage = Mapping[str, Mapping[str, float | int | None]]


@dataclass(frozen=True, slots=True)
class RobustnessInput:
    """The block of six values and the coverage of each, of the robustness run of the system
    `system_name` (the digests of the model and of its manifest for a model run, `None` for a
    hypotheses file; `hypotheses_sha256` for a file) over the suite `suite_name` with the digest
    `suite_sha256`, and the SHA-256 of the two files they were read from."""

    block: Block
    coverage: Coverage
    system_name: str
    model_sha256: str | None
    manifest_sha256: str | None
    hypotheses_sha256: str | None
    suite_name: str
    suite_sha256: str
    robustness_sha256: str
    report_sha256: str


# ---- the shapes of the block, the coverage and the source ----------------------------------------


def checked_block(value: object, field: str) -> Block:
    """The six values of a block, each a finite number or null, read-only.

    Raises:
        ValueError: not exactly the six keys, or a value that is neither a number nor null.
    """
    block = require_fields(value, field, BLOCK_KEYS)
    return MappingProxyType(
        {
            key: require_number(block[key], f"{field}.{key}", nullable=True)
            for key in BLOCK_KEYS
        }
    )


def checked_coverage(value: object, field: str) -> Coverage:
    """The coverage of the six values: for each, the units `scored`, `excluded` and their
    `total` (the sum of the two), and for the retention the two chrF scores it is made of.

    Raises:
        ValueError: a key or a count that is missing, unknown, of another type or that does
            not add up.
    """
    coverage = require_fields(value, field, BLOCK_KEYS)
    return MappingProxyType(
        {key: _entry(coverage[key], f"{field}.{key}", key) for key in BLOCK_KEYS}
    )


def _entry(value: object, field: str, key: str) -> Mapping[str, float | int | None]:
    names = (*_COUNTS, *_RETENTION_CHRF) if key == _RETENTION else _COUNTS
    entry = require_fields(value, field, names)
    counts = {name: require_integer(entry[name], f"{field}.{name}") for name in _COUNTS}
    if counts["total"] != counts["scored"] + counts["excluded"]:
        raise problem(field, "the counts do not add up: total is scored plus excluded")
    found: dict[str, float | int | None] = dict(counts)
    if key == _RETENTION:
        for name in _RETENTION_CHRF:
            found[name] = require_number(entry[name], f"{field}.{name}", nullable=True)
    return MappingProxyType(found)


def checked_source(value: object, field: str) -> Mapping[str, str]:
    """The source of a block in a result: the suite and the three digests it came from and the
    binding to the system.

    Raises:
        ValueError: a field that is missing, unknown or of the wrong shape, or a binding that
            is neither `model_digest` nor `name`.
    """
    source = require_fields(value, field, _SOURCE_FIELDS)
    suite = require_id(source["suite"], f"{field}.suite")
    digests = {
        name: require_digest(source[name], f"{field}.{name}")
        for name in ("suite_sha256", "robustness_sha256", "report_sha256")
    }
    binding = source["binding"]
    if not isinstance(binding, str) or binding not in BINDINGS:
        raise problem(f"{field}.binding", f"must be one of {', '.join(BINDINGS)}")
    return MappingProxyType({"suite": suite, **digests, "binding": binding})


# ---- reading the two files -------------------------------------------------------------------


def read_robustness(report_bytes: bytes, robustness_bytes: bytes) -> RobustnessInput:
    """The block, its coverage and the digests, from the narrow report and `robustness.json`.

    Raises:
        DataError: either file is not what D12 writes, or the two disagree (the digest the
            report names, the system, the suite, the block, the coverage); the message names
            the field and never a value.
    """
    report_document = load_document(report_bytes, "a robustness report (report.json)")
    result_document = load_document(
        robustness_bytes, f"a robustness result ({ROBUSTNESS_FILE})"
    )
    robustness_sha256 = hashlib.sha256(robustness_bytes).hexdigest()
    try:
        return _read(
            report_document,
            result_document,
            robustness_sha256,
            hashlib.sha256(report_bytes).hexdigest(),
        )
    except ValueError as error:
        raise DataError(f"not a robustness input: {error}") from error


def _read(
    report_document: Mapping[str, Any],
    result_document: Mapping[str, Any],
    robustness_sha256: str,
    report_sha256: str,
) -> RobustnessInput:
    if report_document.get("schema") != REPORT_SCHEMA:
        raise problem("schema", "is not the first version of the robustness report")
    report = require_fields(report_document, None, _REPORT_FIELDS)
    if report["robustness_file"] != ROBUSTNESS_FILE:
        raise problem("robustness_file", f"must name {ROBUSTNESS_FILE}")
    named = require_digest(report["robustness_sha256"], "robustness_sha256")
    if named != robustness_sha256:
        raise problem(
            "robustness_sha256",
            f"is not the digest of {ROBUSTNESS_FILE}: the file changed after the "
            "report was written, or belongs to another run",
        )
    system = require_fields(report["system"], "system", _REPORT_SYSTEM)
    suite = require_fields(report["suite"], "suite", _SUITE_FIELDS)
    facts = _facts(system, suite, "")
    block = checked_block(report["robustness"], "robustness")
    coverage = checked_coverage(report["coverage"], "coverage")
    if result_document.get("schema") != ROBUSTNESS_SCHEMA:
        raise problem(
            f"{_FILE}schema", "is not the first version of the robustness result"
        )
    result = require_fields(result_document, _FILE, _RESULT_FIELDS)
    result_system = require_fields(result["system"], f"{_FILE}system", _RESULT_SYSTEM)
    result_suite = require_object(result["suite"], f"{_FILE}suite")
    agreed = _facts(
        {key: result_system[key] for key in _REPORT_SYSTEM},
        {key: result_suite.get(key) for key in _SUITE_FIELDS},
        _FILE,
    )
    for field, ours, theirs in zip(_Facts._fields, facts, agreed, strict=True):
        if ours != theirs:
            raise problem(
                _DOTTED[field], f"differs between the report and {ROBUSTNESS_FILE}"
            )
    _require_same_block(
        block, checked_block(result["robustness"], f"{_FILE}robustness")
    )
    _require_backed(coverage, require_object(result["sections"], f"{_FILE}sections"))
    return RobustnessInput(
        block=block,
        coverage=coverage,
        system_name=facts.system_name,
        model_sha256=facts.model_sha256,
        manifest_sha256=facts.manifest_sha256,
        hypotheses_sha256=require_digest(
            result_system["hypotheses_sha256"],
            f"{_FILE}system.hypotheses_sha256",
            nullable=True,
        ),
        suite_name=facts.suite_name,
        suite_sha256=facts.suite_sha256,
        robustness_sha256=robustness_sha256,
        report_sha256=report_sha256,
    )


class _Facts(NamedTuple):
    """The system and the suite a file names."""

    system_name: str
    model_sha256: str | None
    manifest_sha256: str | None
    suite_name: str
    suite_sha256: str


_DOTTED: Final = {
    "system_name": "system.name",
    "model_sha256": "system.model_sha256",
    "manifest_sha256": "system.manifest_sha256",
    "suite_name": "suite.name",
    "suite_sha256": "suite.sha256",
}


def _facts(system: Mapping[str, Any], suite: Mapping[str, Any], prefix: str) -> _Facts:
    return _Facts(
        require_id(system["name"], f"{prefix}system.name"),
        require_digest(
            system["model_sha256"], f"{prefix}system.model_sha256", nullable=True
        ),
        require_digest(
            system["manifest_sha256"], f"{prefix}system.manifest_sha256", nullable=True
        ),
        require_id(suite["name"], f"{prefix}suite.name"),
        require_digest(suite["sha256"], f"{prefix}suite.sha256"),
    )


def _require_same_block(report_block: Block, result_block: Block) -> None:
    for key in BLOCK_KEYS:
        if report_block[key] != result_block[key]:
            raise problem(
                f"robustness.{key}", f"differs between the report and {ROBUSTNESS_FILE}"
            )


def _require_backed(coverage: Coverage, sections: Mapping[str, Any]) -> None:
    """The coverage of the report must be what the sections of `robustness.json` count: the
    units of a section scored, those left out under any reason, and for the retention the two
    chrF scores of the `mong` section."""
    for key in BLOCK_KEYS:
        section = sections.get(_SECTION_OF[key])
        where = f"{_FILE}sections.{_SECTION_OF[key]}"
        scored, excluded = 0, 0
        details: Mapping[str, Any] = {}
        if section is not None:
            body = require_object(section, where)
            scored = require_integer(body.get("scored"), f"{where}.scored")
            excluded = sum(
                require_integer(count, f"{where}.excluded")
                for count in require_object(
                    body.get("excluded"), f"{where}.excluded"
                ).values()
            )
            details = require_object(body.get("details"), f"{where}.details")
        entry = coverage[key]
        if (entry["scored"], entry["excluded"], entry["total"]) != (
            scored,
            excluded,
            scored + excluded,
        ):
            raise problem(
                f"coverage.{key}", f"is not backed by the sections of {ROBUSTNESS_FILE}"
            )
        if key == _RETENTION:
            for name in _RETENTION_CHRF:
                backed = require_number(
                    details.get(name), f"{where}.details.{name}", nullable=True
                )
                if entry[name] != backed:
                    raise problem(
                        f"coverage.{key}.{name}",
                        f"differs from the section of {ROBUSTNESS_FILE}",
                    )


# ---- the binding to a system -----------------------------------------------------------------


def bind(inp: RobustnessInput, system: SystemRecord) -> str:
    """How the block is bound to `system`: `model_digest` for a `rinchen` system whose model and
    manifest digests are the robustness run's, `name` for any other system of the same name.

    Raises:
        ConfigError: a `rinchen` system whose digests are not the robustness run's (or a
            robustness run that had no model: the name of a model is no evidence), or another
            kind of system with another name. The message names the two system names when they
            are ids and no digest.
    """
    if system.kind == "rinchen":
        if inp.model_sha256 is None or inp.manifest_sha256 is None:
            raise ConfigError(
                "the robustness run was made without a model, so it cannot describe a "
                "rinchen system: such a system is bound by its model digest and never by "
                "its name"
            )
        if (system.model_sha256, system.manifest_sha256) != (
            inp.model_sha256,
            inp.manifest_sha256,
        ):
            raise ConfigError(
                "the robustness run is of another model: the model digest or the manifest "
                "digest of the system differs from the one the robustness run was made with"
            )
        return "model_digest"
    if system.name != inp.system_name:
        raise ConfigError(
            "the robustness run is of another system: it names "
            f"{_named(inp.system_name)} and this run {_named(system.name)}"
        )
    return "name"


def _named(name: str) -> str:
    return repr(name) if is_id(name) else "a name that is not an id"


def source_record(inp: RobustnessInput, binding: str) -> dict[str, str]:
    """What a result records of where its block came from: the suite and the digests of the
    robustness run and of its report, and how the block was bound to the system.

    Raises:
        ValueError: `binding` is neither `model_digest` nor `name`.
    """
    if binding not in BINDINGS:
        raise ValueError(f"the binding must be one of {', '.join(BINDINGS)}")
    return {
        "binding": binding,
        "report_sha256": inp.report_sha256,
        "robustness_sha256": inp.robustness_sha256,
        "suite": inp.suite_name,
        "suite_sha256": inp.suite_sha256,
    }
