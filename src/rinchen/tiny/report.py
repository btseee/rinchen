"""The report of a tiny run (D15, spec P4, P8, P10 and P12): `build_report` turns what the stages
observed into a `Report`, `to_bytes` writes it, `parse_report` reads it back strictly, and `exit_code`
says what the command exits with.

A `Report` lists the fifteen stages of the pipeline once each and in order: a stage that never ran is
`skipped` for its closed reason and so is every part of it, so that a missing stage is never a silent
drop. It lists the eight thresholds in the order of the table, each from the figures the stages
measured, those of a record that failed included (a parity that missed its gate has its figures, an
evaluation that stopped after its first part has the figures of that part); a figure that is not there
makes its threshold `fail` and says which. The `blocks` are the measurements, each null when its stage
never gave the figures it is built from (`rinchen.tiny.report_blocks`).

The document is built from observations only, so that two runs on the same machine write the same bytes
in two directories: the elapsed time of the run decides the verdict of the `wall_time` threshold and is
not in the document (the seconds are the manifest's, spec P12), and nothing in it is a timestamp, a
path, a line of another program's output or a value of the environment. A threshold whose value is
null is a `fail`, except `wall_time`, whose value is always null. The verdict is `pass` when no stage
failed and every threshold passed.

`parse_report` is strict: it reads only the canonical bytes (`to_bytes` of the report it returns equals
its argument), refuses each way a document can differ from the layout of `rinchen.tiny.report_schema`
with an error that names the place and quotes no value, and checks what the schema cannot: the stages
and thresholds are the fifteen and the eight in order, an `ok` stage lists every part of its stage, the
wall-time threshold holds no value, each threshold's verdict follows from its value and its bound, the
`stages` threshold is the one the stage records give, and the verdict follows from the stages and the
thresholds: a document that `build_report` could not have written is refused (the model, `Report`, holds
all of these, so no report of either origin can break them).
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any, Final

from rinchen.config.tiny import STAGE_IDS, TinyConfig
from rinchen.errors import DataError
from rinchen.tiny.context import StageResult
from rinchen.tiny.report_blocks import build_blocks
from rinchen.tiny.report_schema import DOCUMENT, INPUTS, SCHEMA, UNMEASURED
from rinchen.tiny.report_shapes import Detail, ShapeError, plain
from rinchen.tiny.states import (
    PART_IDS,
    SKIP_EARLIER_FAILED,
    PartRecord,
    StageRecord,
    unlisted_parts,
)
from rinchen.tiny.thresholds import (
    THRESHOLD_IDS,
    ThresholdInputs,
    ThresholdResult,
    Verdict,
    evaluate_thresholds,
    stages_threshold,
)

__all__ = [
    "EXIT_FAILED",
    "EXIT_PASSED",
    "UNMEASURED",
    "Inputs",
    "Report",
    "build_report",
    "exit_code",
    "parse_report",
    "report_document",
    "to_bytes",
]

EXIT_PASSED: Final = 0
EXIT_FAILED: Final = 4
# The one threshold whose figure is a duration: the report keeps its verdict and not its value.
WALL_TIME: Final = "wall_time"


@dataclass(frozen=True, slots=True)
class Inputs:
    """What the run read, by digest: the profile file, each of the seven fixture files by name, and
    `uv.lock`."""

    config_sha256: str
    fixtures: Mapping[str, str]
    lock_sha256: str

    def __post_init__(self) -> None:
        INPUTS.check(_inputs_document(self), "inputs")


def _inputs_document(inputs: Inputs) -> dict[str, Any]:
    return {
        "config_sha256": inputs.config_sha256,
        "fixtures": dict(inputs.fixtures),
        "lock_sha256": inputs.lock_sha256,
    }


def _follows(result: ThresholdResult) -> bool:
    """Whether the verdict of a threshold follows from its value and its bound (`>= 50.0`); a threshold
    without a value is a `fail`, and only `wall_time` may be a `pass` without one."""
    if result.value is None:
        return result.id == WALL_TIME or result.verdict == "fail"
    comparison, _, limit = result.bound.partition(" ")
    number = float(limit)
    passed = result.value >= number if comparison == ">=" else result.value <= number
    return passed == (result.verdict == "pass")


@dataclass(frozen=True, slots=True)
class Report:
    """The whole report: the profile and the stand-in switch, the digests of the inputs, the fifteen
    stages in order, the eight thresholds in order and the eleven blocks.

    Raises:
        ShapeError: (a `ValueError`) on construction when anything is not the layout of the schema,
            when the stages or the thresholds are not the fifteen and the eight in order, when an `ok`
            stage leaves out one of its parts, when the wall-time threshold holds a value, when a
            threshold's verdict does not follow from its value and its bound, or when the `stages`
            threshold is not the one `rinchen.tiny.thresholds.stages_threshold` gives for the
            stage records; the message names the place and holds no value.
    """

    profile: str
    qe_fake: bool
    inputs: Inputs
    stages: tuple[StageRecord, ...]
    thresholds: tuple[ThresholdResult, ...]
    blocks: Mapping[str, Any]

    def __post_init__(self) -> None:
        object.__setattr__(self, "stages", tuple(self.stages))
        object.__setattr__(self, "thresholds", tuple(self.thresholds))
        DOCUMENT.check(report_document(self), "report")
        if tuple(record.id for record in self.stages) != STAGE_IDS:
            raise ShapeError(
                "report.stages", "are not the fifteen stages, once each, in order"
            )
        if tuple(result.id for result in self.thresholds) != THRESHOLD_IDS:
            raise ShapeError(
                "report.thresholds", "are not the eight thresholds, once each, in order"
            )
        for position, record in enumerate(self.stages):
            if record.state == "ok" and unlisted_parts(record):
                raise ShapeError(
                    f"report.stages[{position}]",
                    "an ok stage does not list every one of its parts",
                )
        for position, result in enumerate(self.thresholds):
            if result.id == WALL_TIME and result.value is not None:
                raise ShapeError(
                    f"report.thresholds[{position}]",
                    "the wall-time threshold holds no value: the report holds no duration",
                )
            if not _follows(result):
                raise ShapeError(
                    f"report.thresholds[{position}]",
                    "the verdict does not follow from the value and the bound",
                )
        if self.thresholds[0] != stages_threshold(self.stages):
            raise ShapeError(
                "report.thresholds[0]",
                "is not the stages threshold that the stage records give",
            )

    @property
    def verdict(self) -> Verdict:
        """`pass` when no stage failed and every threshold passed, else `fail`."""
        failing = any(record.state == "fail" for record in self.stages) or any(
            result.verdict != "pass" for result in self.thresholds
        )
        return "fail" if failing else "pass"


def exit_code(report: Report) -> int:
    """0 for a report that passes, 4 for one that fails (a stage failed or a threshold was missed)."""
    return EXIT_PASSED if report.verdict == "pass" else EXIT_FAILED


def _stage_document(record: StageRecord) -> dict[str, Any]:
    return {
        "id": record.id,
        "state": record.state,
        "reason": record.reason,
        "detail": plain(record.detail),
        "parts": [
            {"id": part.id, "state": part.state, "reason": part.reason}
            for part in record.parts
        ],
    }


def report_document(report: Report) -> dict[str, Any]:
    """The document of `report.json` as plain JSON values."""
    return {
        "schema": SCHEMA,
        "profile": report.profile,
        "qe_fake": report.qe_fake,
        "verdict": report.verdict,
        "inputs": _inputs_document(report.inputs),
        "stages": [_stage_document(record) for record in report.stages],
        "thresholds": [
            {
                "id": result.id,
                "bound": result.bound,
                "value": result.value,
                "verdict": result.verdict,
                "detail": result.detail,
            }
            for result in report.thresholds
        ],
        "blocks": plain(report.blocks),
        "unmeasured": list(UNMEASURED),
    }


def to_bytes(report: Report) -> bytes:
    """The UTF-8 bytes of the document: sorted keys, two-space indent, no ASCII escapes, one final
    line feed."""
    text = json.dumps(
        report_document(report),
        sort_keys=True,
        indent=2,
        ensure_ascii=False,
        allow_nan=False,
    )
    return f"{text}\n".encode()


# ---- building ----------------------------------------------------------------------------------------


def _holdable(detail: Mapping[str, Any]) -> dict[str, Any]:
    """The detail of a record as the report can hold it: itself when it fits, else nothing."""
    document: dict[str, Any] = plain(detail)
    try:
        Detail().check(document, "detail")
    except ShapeError:
        return {}
    return document


def _parts(record: StageRecord) -> tuple[PartRecord, ...]:
    """The parts of a record as the report lists them: in the order of the stage's parts, and for a
    stage that did not end `ok` every part it did not judge skipped for its closed reason. An `ok`
    stage's parts are only put in order: one it left out is not made up, and the report refuses it."""
    wanted = PART_IDS.get(record.id, ())
    given = {part.id: part for part in record.parts}
    if record.state == "ok":
        return tuple(given[part_id] for part_id in wanted if part_id in given)
    return tuple(
        given.get(part_id) or PartRecord(part_id, "skipped", SKIP_EARLIER_FAILED)
        for part_id in wanted
    )


def _listed(stage_id: str, result: StageResult | None) -> StageRecord:
    if result is None:
        record = StageRecord(stage_id, "skipped", SKIP_EARLIER_FAILED)
        return replace(record, parts=_parts(record))
    record = result.record
    return StageRecord(
        record.id, record.state, record.reason, _holdable(record.detail), _parts(record)
    )


def _threshold_inputs(
    records: Sequence[StageRecord],
    facts: Mapping[str, Mapping[str, Any]],
    elapsed_seconds: float,
) -> ThresholdInputs:
    return ThresholdInputs(
        stages=records,
        memorised_chrf=facts.get("evaluate", {}).get("memorised_chrf", {}),
        parity=facts.get("parity") or None,
        robustness=facts.get("robustness", {}).get("values"),
        elapsed_seconds=elapsed_seconds,
    )


def _without_duration(result: ThresholdResult) -> ThresholdResult:
    return replace(result, value=None) if result.id == WALL_TIME else result


def build_report(
    profile: str,
    qe_fake: bool,
    inputs: Inputs,
    results: Sequence[StageResult],
    elapsed_seconds: float,
    config: TinyConfig,
) -> Report:
    """The report of a run: `results` are the stages that ran (in any order, each once), and every
    stage of `STAGE_IDS` that is not among them is listed `skipped` for its closed reason.

    The thresholds take their bounds and the wall-time budget from `config` and their figures from
    the facts of `results`; `elapsed_seconds` decides the verdict of `wall_time` and is not kept. A
    detail the report cannot hold (a path, a time, a key that names a duration) is left out of its
    stage, whose state and reason stay.

    Raises:
        ValueError: a stage has two results.
        ShapeError: `profile` or `inputs` is not what the schema holds, or an `ok` result leaves out
            a part of its stage.
    """
    by_stage: dict[str, StageResult] = {}
    for result in results:
        if result.record.id in by_stage:
            raise ValueError("a stage has two results")
        by_stage[result.record.id] = result
    records = tuple(_listed(stage_id, by_stage.get(stage_id)) for stage_id in STAGE_IDS)
    facts = {stage_id: found.facts for stage_id, found in by_stage.items()}
    measured = evaluate_thresholds(
        _threshold_inputs(records, facts, elapsed_seconds),
        config.thresholds,
        config.budget_seconds,
    )
    return Report(
        profile,
        qe_fake,
        inputs,
        records,
        tuple(_without_duration(result) for result in measured),
        build_blocks(facts),
    )


# ---- reading -----------------------------------------------------------------------------------------


class _Repeated(ValueError):
    """An object of the document repeats a key."""


class _NotFinite(ValueError):
    """The document writes NaN or an infinity."""


def _one_of_each_key(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    found: dict[str, Any] = {}
    for key, value in pairs:
        if key in found:
            raise _Repeated
        found[key] = value
    return found


def _refuse_constant(name: str) -> Any:
    raise _NotFinite


def _load(data: bytes) -> Any:
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as error:
        raise DataError("report: not valid UTF-8") from error
    try:
        return json.loads(
            text, object_pairs_hook=_one_of_each_key, parse_constant=_refuse_constant
        )
    except _Repeated as error:
        raise DataError("report: an object repeats a key") from error
    except _NotFinite as error:
        raise DataError("report: NaN and infinity are not allowed") from error
    except json.JSONDecodeError as error:
        raise DataError(
            f"report: not valid JSON at line {error.lineno}, column {error.colno}"
        ) from error
    except (ValueError, RecursionError) as error:
        raise DataError("report: not valid JSON") from error


def _stage_record(position: int, item: Mapping[str, Any]) -> StageRecord:
    place = f"report.stages[{position}]"
    parts = []
    for index, part in enumerate(item["parts"]):
        try:
            parts.append(PartRecord(part["id"], part["state"], part["reason"]))
        except ValueError as error:
            raise DataError(f"{place}.parts[{index}]: {error}") from error
    try:
        return StageRecord(
            item["id"], item["state"], item["reason"], item["detail"], tuple(parts)
        )
    except ValueError as error:
        raise DataError(f"{place}: {error}") from error


def parse_report(data: bytes) -> Report:
    """The report in `data`, which must be the canonical bytes of a report.

    Raises:
        DataError: the document is not valid UTF-8 or JSON (a repeated key, NaN and infinity are
            refused), is not the layout of the schema, holds a state, a reason, a verdict, a number or
            a text that the report never holds, lists the stages or the thresholds out of order, has a
            verdict that does not follow, or is not in the canonical form; the message names the place
            and quotes nothing the document held.
    """
    document = _load(data)
    try:
        DOCUMENT.check(document, "report")
        stages = tuple(
            _stage_record(position, item)
            for position, item in enumerate(document["stages"])
        )
        report = Report(
            document["profile"],
            document["qe_fake"],
            Inputs(**document["inputs"]),
            stages,
            tuple(ThresholdResult(**item) for item in document["thresholds"]),
            document["blocks"],
        )
    except ShapeError as error:
        raise DataError(str(error)) from error
    if document["verdict"] != report.verdict:
        raise DataError(
            "report.verdict: does not follow from the stages and the thresholds"
        )
    if to_bytes(report) != data:
        raise DataError(
            "report: not in the canonical form: sorted keys, two-space indent, one final line feed"
        )
    return report
