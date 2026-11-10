"""The robustness result of D12 (spec T3 to T10, T13): a suite scored with the hypotheses a
system wrote, section by section, and the document `robustness.json` that says so.

`compute_robustness` checks that the hypotheses are exactly the requests of the suite (a missing
one or a stray one is a `DataError` that names the section, the id and the arm), scores each
section (`robust_sections`) and returns a `RobustnessResult`. Its `to_bytes` is the document,
schema `rinchen.robustness/1`: sorted keys, two-space indent, one final LF, no timestamp and no
text of the suite or of a hypothesis, so the same inputs give the same bytes. Layout:

    system {name, model_sha256, manifest_sha256, options}  suite {name, sha256, rows}
    robustness {the six values}   sections {section: {unit, rows, scored, excluded, ok, value,
    details, by_category, by_direction, statuses}}   rows {section: [one outcome per row]}

`robustness_block` is the six values the reports embed: `numbers_acc`, `placeholder_acc`,
`term_acc`, `script_purity`, `entity_acc` and `mong_source_retention`, each a float or None (the
section is absent, or nothing in it was scored: never 0 or 1). `robustness_coverage` is what each
rests on: how many units were scored and how many were left out (rows; mentions for the entities
and the terms), because a value over ten scored rows of a thousand looks like one over all of
them. `mong_source_retention` is a ratio of two corpus chrF scores: it can exceed 1 (the direct
arm beat the control) and it says nothing when the Cyrillic chrF is near zero (two poor scores
give a ratio near 1), so its coverage entry carries both chrF scores. Nothing here measures a
model; the floors of the gate are the gate's, not the result's."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Final

from rinchen.errors import DataError
from rinchen.eval.robust_hypotheses import Hypothesis, request_place
from rinchen.eval.robust_outcomes import SectionResult
from rinchen.eval.robust_sections import score_section
from rinchen.eval.robust_suite import RequestKey, Suite, suite_requests
from rinchen.eval.robust_suite_rules import is_id

__all__ = [
    "RESULT_SCHEMA",
    "RobustnessResult",
    "RobustnessSystem",
    "compute_robustness",
    "robustness_block",
    "robustness_coverage",
]

RESULT_SCHEMA: Final = "rinchen.robustness/1"
# The key of the block and the section whose headline it is, in the order of the block.
_BLOCK: Final = (
    ("numbers_acc", "numbers"),
    ("placeholder_acc", "placeholders"),
    ("term_acc", "terms"),
    ("script_purity", "script"),
    ("entity_acc", "entities"),
    ("mong_source_retention", "mong"),
)
_SHA256: Final = re.compile(r"[0-9a-f]{64}")


@dataclass(frozen=True)
class RobustnessSystem:
    """The system that wrote the hypotheses: a `name` (the characters of an id), the SHA-256
    of its `model.bin` and of the manifest that vouches for the export (None for a hypotheses
    file) and the decode `options` it ran with (None for a hypotheses file).

    Raises:
        ValueError: the name is not an id, or a digest is not a SHA-256 hex digest or None.
        TypeError: an option is not a string, an integer or a boolean under a string name.
    """

    name: str
    model_sha256: str | None = None
    manifest_sha256: str | None = None
    options: Mapping[str, str | int | bool] | None = None

    def __post_init__(self) -> None:
        if not is_id(self.name):
            raise ValueError(
                "name must be 1 to 64 characters: letters, digits, '.', '_' and '-'"
            )
        for field in ("model_sha256", "manifest_sha256"):
            value = getattr(self, field)
            if value is not None and not _is_sha256(value):
                raise ValueError(f"{field} must be a SHA-256 hex digest or None")
        if self.options is not None:
            for name, value in self.options.items():
                if not isinstance(name, str) or not isinstance(value, str | int):
                    raise TypeError(
                        "options must map names to strings, integers or booleans"
                    )
            object.__setattr__(self, "options", MappingProxyType(dict(self.options)))

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "model_sha256": self.model_sha256,
            "manifest_sha256": self.manifest_sha256,
            "options": None if self.options is None else dict(self.options),
        }


@dataclass(frozen=True)
class RobustnessResult:
    """A suite scored with the hypotheses of one system: `suite_name`, `suite_sha256` and
    `suite_rows` (rows per section) say what was scored, `sections` hold the figures and the
    per-row outcomes of each section the suite has."""

    system: RobustnessSystem
    suite_name: str
    suite_sha256: str
    suite_rows: Mapping[str, int]
    sections: Mapping[str, SectionResult]

    def __post_init__(self) -> None:
        object.__setattr__(self, "suite_rows", MappingProxyType(dict(self.suite_rows)))
        object.__setattr__(self, "sections", MappingProxyType(dict(self.sections)))

    def to_dict(self) -> dict[str, Any]:
        """The document as plain dicts, lists, numbers, strings, booleans and None."""
        return {
            "schema": RESULT_SCHEMA,
            "system": self.system.to_dict(),
            "suite": {
                "name": self.suite_name,
                "sha256": self.suite_sha256,
                "rows": dict(self.suite_rows),
            },
            "robustness": robustness_block(self),
            "sections": {name: s.to_dict() for name, s in self.sections.items()},
            "rows": {
                name: [outcome.to_dict() for outcome in s.outcomes]
                for name, s in self.sections.items()
            },
        }

    def to_bytes(self) -> bytes:
        """The UTF-8 bytes of `robustness.json`.

        Raises:
            ValueError: a value is not a finite number (JSON has no NaN).
        """
        text = json.dumps(
            self.to_dict(),
            sort_keys=True,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )
        return f"{text}\n".encode()


def robustness_block(result: RobustnessResult) -> dict[str, float | None]:
    """The six values of the result, in the order of the contract; None for a section the
    suite does not have and for one in which nothing was scored."""
    return {
        key: result.sections[section].value if section in result.sections else None
        for key, section in _BLOCK
    }


def robustness_coverage(
    result: RobustnessResult,
) -> dict[str, dict[str, float | int | None]]:
    """What each of the six values rests on, in the order of the block: the units `scored`,
    those `excluded` (under any reason) and their `total`, in the unit of the section (rows;
    mentions for the entities and the terms). All three are 0 for a section the suite does not
    have. The entry of `mong_source_retention` also holds the corpus chrF of both arms the
    ratio is made of (`chrf_cyrl`, `chrf_mong`; None without rows)."""
    coverage: dict[str, dict[str, float | int | None]] = {}
    for key, name in _BLOCK:
        section = result.sections.get(name)
        coverage[key] = {
            "scored": 0 if section is None else section.scored,
            "excluded": 0 if section is None else section.excluded_total,
            "total": 0 if section is None else section.total,
        }
    mong = result.sections.get("mong")
    coverage["mong_source_retention"] |= {
        "chrf_cyrl": None if mong is None else mong.details["chrf_cyrl"],
        "chrf_mong": None if mong is None else mong.details["chrf_mong"],
    }
    return coverage


def compute_robustness(
    suite: Suite,
    hypotheses: Mapping[RequestKey, Hypothesis],
    *,
    suite_sha256: str,
    system: RobustnessSystem,
) -> RobustnessResult:
    """Score `suite` with `hypotheses`, which must hold one for every request of the suite
    (the `converted` arm of the `mong` section for every row of it or for none) and no other.

    Raises:
        DataError: a request has no hypothesis, or a hypothesis answers no request; the
            message names the section, the id and the arm.
        ValueError: `suite_sha256` is not a SHA-256 hex digest, or a row of the suite lacks
            what its section needs.
    """
    if not _is_sha256(suite_sha256):
        raise ValueError("suite_sha256 must be a SHA-256 hex digest")
    requests = suite_requests(suite, _converted_arm(suite, hypotheses))
    _require_exactly([request.key for request in requests], hypotheses)
    sections = {
        section: score_section(section, rows, suite.entities, hypotheses)
        for section, rows in suite.sections.items()
    }
    return RobustnessResult(
        system=system,
        suite_name=suite.name,
        suite_sha256=suite_sha256,
        suite_rows=suite.row_counts(),
        sections=sections,
    )


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and _SHA256.fullmatch(value) is not None


def _converted_arm(
    suite: Suite, hypotheses: Mapping[RequestKey, Hypothesis]
) -> dict[str, str] | None:
    """The `converted` arm is read off the hypotheses: when any row of the `mong` section has
    one, every row is expected to (the texts do not matter here, only the keys)."""
    ids = [row.id for row in suite.sections.get("mong", ())]
    if any(RequestKey("mong", row_id, "converted") in hypotheses for row_id in ids):
        return dict.fromkeys(ids, "")
    return None


def _require_exactly(
    wanted: list[RequestKey], hypotheses: Mapping[RequestKey, Hypothesis]
) -> None:
    missing = [key for key in wanted if key not in hypotheses]
    if missing:
        raise DataError(
            f"the hypotheses lack {len(missing)} request(s), the first is "
            f"{request_place(*missing[0])}"
        )
    asked = set(wanted)
    stray = [key for key in hypotheses if key not in asked]
    if stray:
        raise DataError(
            f"the hypotheses answer {len(stray)} request(s) that are not in the "
            f"suite, the first is {request_place(*stray[0])}"
        )
