"""The run record of D13 (spec T11): `run.json`, schema `rinchen.eval.run/1`, the table of contents
of a run. It names the suite (its name and SHA-256) and the purpose of the run (`selection` for a
`dev` suite, `report` for the others: the report refuses the first), the system record, the
settings of the scoring, one entry for EVERY direction the suite declares (its state, the reason
when it did not run, its row count, and the path of its result file when it ran) and each file
the run wrote with its SHA-256. A report reads a run through it and checks every file against it.

`RunRecord` is the unit that is validated, whether it is built by a command or read back
(`run_from_bytes`): exactly the keys of the format, every state one of the five, a reason exactly
when a direction did not run, a result path exactly when it did, no path that leaves the run's
directory, digests of 64 lowercase hexadecimal characters, the hypotheses file and every result
file listed, `not_selected` exactly for what the selection leaves out. A refusal names the field
(`directions.2.state`) and never quotes a value; the file itself carries no timestamp and no
absolute path.

`rinchen.eval.result` re-exports the names of this module, which is where `result.json` and
`run.json` are described together.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Final, get_args

from rinchen.errors import DataError
from rinchen.eval.bootstrap import MIN_RESAMPLES
from rinchen.eval.field_rules import (
    document_bytes,
    load_document,
    problem,
    require_choice,
    require_digest,
    require_direction,
    require_fields,
    require_id,
    require_integer,
    require_text,
    system_from,
)
from rinchen.eval.robustness_input import checked_source
from rinchen.eval.suite_rules import PURPOSE_OF_SPLIT
from rinchen.eval.systems import DirectionState, SystemRecord
from rinchen.metrics.registry import SEGMENT_SCORE_METRICS

__all__ = [
    "OUTPUTS_FILE",
    "RESULT_FILE",
    "RUN_FILE",
    "RUN_SCHEMA",
    "RunDirection",
    "RunRecord",
    "RunSettings",
    "RunSuite",
    "result_path",
    "run_from_bytes",
]

RUN_SCHEMA: Final = "rinchen.eval.run/1"
RUN_FILE: Final = "run.json"
OUTPUTS_FILE: Final = "hypotheses.jsonl"
RESULT_FILE: Final = "result.json"
_PURPOSES: Final = tuple(dict.fromkeys(PURPOSE_OF_SPLIT.values()))
_STATES: Final = get_args(DirectionState)
_FIELDS: Final = (
    "schema",
    "suite",
    "purpose",
    "system",
    "settings",
    "directions",
    "files",
    "robustness_source",
    "segment_scores",
)
_SETTINGS_FIELDS: Final = ("resamples", "seed", "alpha", "selection")
_DIRECTION_FIELDS: Final = ("direction", "state", "reason", "rows", "result")
_MAX_PATH_CHARS: Final = 200
_MAX_SEED: Final = 2**64 - 1


@dataclass(frozen=True, slots=True)
class RunSuite:
    """The suite a run scored: its `name` and the SHA-256 of its file."""

    name: str
    sha256: str


@dataclass(frozen=True, slots=True)
class RunSettings:
    """How the run scored: the bootstrap's `resamples`, `seed` and `alpha`, and the
    `selection` of directions the operator asked for (None: all of them)."""

    resamples: int
    seed: int
    alpha: float
    selection: tuple[str, ...] | None


@dataclass(frozen=True, slots=True)
class RunDirection:
    """What became of one direction of the suite: its `state`, the `reason` when it did not
    run (None when it did), the `rows` the suite has for it and the relative path of its
    `result` file (None when it did not run)."""

    direction: str
    state: DirectionState
    reason: str | None
    rows: int
    result: str | None


@dataclass(frozen=True, slots=True)
class RunRecord:
    """A run: its `suite` and `purpose`, the `system` record, the `settings`, a `RunDirection`
    for every direction of the suite, the SHA-256 of every `files` entry (relative paths) the
    run wrote, the `robustness_source` of the block its results carry (None without one) and
    the SHA-256 of the scores file of each `segment_scores` metric.

    Raises:
        ValueError: a field breaks a rule of the format (the message names it, never its
            value).
    """

    suite: RunSuite
    purpose: str
    system: SystemRecord
    settings: RunSettings
    directions: Sequence[RunDirection]
    files: Mapping[str, str]
    robustness_source: Mapping[str, str] | None
    segment_scores: Mapping[str, str]

    def __post_init__(self) -> None:
        require_id(self.suite.name, "suite.name")
        require_digest(self.suite.sha256, "suite.sha256")
        require_choice(self.purpose, "purpose", _PURPOSES)
        selection = _checked_settings(self.settings)
        directions = tuple(self.directions)
        _check_directions(directions, selection)
        files = _checked_files(self.files, directions)
        source = (
            None
            if self.robustness_source is None
            else checked_source(self.robustness_source, "robustness_source")
        )
        object.__setattr__(self, "directions", directions)
        object.__setattr__(self, "files", files)
        object.__setattr__(self, "robustness_source", source)
        object.__setattr__(self, "segment_scores", _checked_scores(self.segment_scores))
        object.__setattr__(
            self,
            "settings",
            RunSettings(
                self.settings.resamples,
                self.settings.seed,
                self.settings.alpha,
                selection,
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        """The document as plain dicts, lists, numbers, strings and None."""
        settings = self.settings
        return {
            "schema": RUN_SCHEMA,
            "suite": {"name": self.suite.name, "sha256": self.suite.sha256},
            "purpose": self.purpose,
            "system": self.system.to_dict(),
            "settings": {
                "resamples": settings.resamples,
                "seed": settings.seed,
                "alpha": settings.alpha,
                "selection": None
                if settings.selection is None
                else list(settings.selection),
            },
            "directions": [
                {
                    "direction": d.direction,
                    "state": d.state,
                    "reason": d.reason,
                    "rows": d.rows,
                    "result": d.result,
                }
                for d in self.directions
            ],
            "files": dict(self.files),
            "robustness_source": None
            if self.robustness_source is None
            else dict(self.robustness_source),
            "segment_scores": dict(self.segment_scores),
        }

    def to_bytes(self) -> bytes:
        """The UTF-8 bytes of `run.json`: sorted keys, two-space indent, one final LF."""
        return document_bytes(self.to_dict())


def run_from_bytes(data: bytes) -> RunRecord:
    """The run in `data`, a `run.json`.

    Raises:
        DataError: `data` is not UTF-8, JSON or a run of this format; the message names the
            field and never a value.
    """
    document = load_document(data, "an evaluation run")
    try:
        return _run_of(document)
    except ValueError as error:
        raise DataError(f"not an evaluation run: {error}") from error


def _run_of(document: Mapping[str, Any]) -> RunRecord:
    if document.get("schema") != RUN_SCHEMA:
        raise problem("schema", "is not the first version of the run format")
    fields = require_fields(document, None, _FIELDS)
    suite = require_fields(fields["suite"], "suite", ("name", "sha256"))
    settings = require_fields(fields["settings"], "settings", _SETTINGS_FIELDS)
    selection = settings["selection"]
    entries = fields["directions"]
    if not isinstance(entries, list) or not entries:
        raise problem("directions", "must be a list of at least one direction")
    return RunRecord(
        suite=RunSuite(suite["name"], suite["sha256"]),
        purpose=fields["purpose"],
        system=system_from(fields["system"], "system"),
        settings=RunSettings(
            settings["resamples"],
            settings["seed"],
            settings["alpha"],
            tuple(selection) if isinstance(selection, list) else selection,
        ),
        directions=[_direction_of(index, entry) for index, entry in enumerate(entries)],
        files=fields["files"],
        robustness_source=fields["robustness_source"],
        segment_scores=fields["segment_scores"],
    )


def _direction_of(index: int, entry: object) -> RunDirection:
    fields = require_fields(entry, f"directions.{index}", _DIRECTION_FIELDS)
    return RunDirection(
        fields["direction"],
        fields["state"],
        fields["reason"],
        fields["rows"],
        fields["result"],
    )


def _checked_settings(settings: RunSettings) -> tuple[str, ...] | None:
    require_integer(settings.resamples, "settings.resamples", minimum=MIN_RESAMPLES)
    require_integer(settings.seed, "settings.seed", maximum=_MAX_SEED)
    if not isinstance(settings.alpha, float) or not 0 < settings.alpha < 0.5:
        raise problem("settings.alpha", "must be a number above 0 and below 0.5")
    selection = settings.selection
    if selection is None:
        return None
    if not isinstance(selection, tuple | list) or not selection:
        raise problem("settings.selection", "must be a list of at least one direction")
    for entry in selection:
        require_direction(entry, "settings.selection")
    if len(set(selection)) != len(selection):
        raise problem("settings.selection", "names a direction more than once")
    return tuple(selection)


def _check_directions(
    directions: tuple[RunDirection, ...], selection: tuple[str, ...] | None
) -> None:
    if not directions:
        raise problem("directions", "must list at least one direction")
    names: set[str] = set()
    for index, entry in enumerate(directions):
        where = f"directions.{index}"
        require_direction(entry.direction, f"{where}.direction")
        if entry.direction in names:
            raise problem(f"{where}.direction", "appears more than once")
        names.add(entry.direction)
        require_choice(entry.state, f"{where}.state", _STATES)
        if entry.state == "run":
            if entry.reason is not None:
                raise problem(
                    f"{where}.reason", "must be null for a direction that ran"
                )
            if entry.result != result_path(entry.direction):
                raise problem(
                    f"{where}.result",
                    f"must be the path <direction>/{RESULT_FILE} of its own direction",
                )
        else:
            require_text(entry.reason, f"{where}.reason")
            if entry.result is not None:
                raise problem(
                    f"{where}.result", "must be null for a direction that did not run"
                )
        require_integer(entry.rows, f"{where}.rows", minimum=1)
    if selection is not None:
        for chosen in selection:
            if chosen not in names:
                raise problem(
                    "settings.selection", "names a direction the run does not list"
                )
    for index, entry in enumerate(directions):
        left_out = selection is not None and entry.direction not in selection
        if (entry.state == "not_selected") != left_out:
            raise problem(
                f"directions.{index}.state",
                "must be not_selected exactly for a direction the selection leaves out",
            )


def result_path(direction: str) -> str:
    """The path, relative to the run's directory, of the result file of `direction`."""
    return f"{direction}/{RESULT_FILE}"


def _checked_files(
    files: Mapping[str, str], directions: tuple[RunDirection, ...]
) -> Mapping[str, str]:
    if not isinstance(files, Mapping):
        raise problem("files", "must be an object")
    for path, digest in files.items():
        if not _is_run_path(path):
            raise problem(
                "files",
                "a file name must be a relative path of at most "
                f"{_MAX_PATH_CHARS} characters: no '..', no leading '/', no backslash "
                "and no control character",
            )
        require_digest(digest, "files")
    if RUN_FILE in files:
        raise problem(
            "files", f"must not list {RUN_FILE}: a file cannot hold its own digest"
        )
    wanted = [OUTPUTS_FILE] + [d.result for d in directions if d.result is not None]
    for path in wanted:
        if path not in files:
            raise problem("files", f"lacks {path}")
    return MappingProxyType(dict(files))


def _is_run_path(path: object) -> bool:
    return (
        isinstance(path, str)
        and 0 < len(path) <= _MAX_PATH_CHARS
        and path.isprintable()
        and "\\" not in path
        and all(part not in ("", ".", "..") for part in path.split("/"))
    )


def _checked_scores(scores: Mapping[str, str]) -> Mapping[str, str]:
    if not isinstance(scores, Mapping):
        raise problem("segment_scores", "must be an object")
    for metric, digest in scores.items():
        if metric not in SEGMENT_SCORE_METRICS:
            raise problem(
                "segment_scores",
                f"is only for {', '.join(SEGMENT_SCORE_METRICS)}",
            )
        require_digest(digest, f"segment_scores.{metric}")
    return MappingProxyType(dict(scores))
