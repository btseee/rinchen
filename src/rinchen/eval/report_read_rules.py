"""What the strict reader of the report of D13 shares between its parts (`rinchen.eval.report_read`):
the names of the suite format it keeps its own copy of (a card never needs the module that reads a
suite; a test pins them equal to `rinchen.eval.suite_rules`'), the systems and suites a report lists
(which every entry elsewhere must name) and the small checks of a list, a text and a number. A refusal
names a dotted field path the reader built (`runs.0.system`) and never quotes a value or a key."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Final, get_args

from rinchen.config.languages import known_language_codes
from rinchen.eval.field_rules import (
    problem,
    require_direction,
    require_number,
    require_text,
)
from rinchen.eval.report_format import ABSENT_STATE
from rinchen.eval.report_view import SuiteView
from rinchen.eval.systems import DirectionState

__all__ = [
    "COVERAGE_STATES",
    "PURPOSES",
    "SPLITS",
    "SUITE_KINDS",
    "SYSTEM_KINDS",
    "Names",
    "items",
    "language",
    "number_or_null",
    "text_or_null",
]

# The names of the suite format, as `rinchen.eval.suite_rules` has them (a test pins them equal).
SUITE_KINDS: Final = ("benchmark", "human", "fixture")
SPLITS: Final = ("dev", "devtest", "test")
PURPOSES: Final = ("selection", "report")
SYSTEM_KINDS: Final = ("rinchen", "baseline", "hypotheses")
COVERAGE_STATES: Final = (*get_args(DirectionState), ABSENT_STATE)


class Names:
    """The systems and suites a report lists, which every entry elsewhere must name."""

    def __init__(
        self, systems: tuple[str, ...], suites: Mapping[str, SuiteView]
    ) -> None:
        self.systems = systems
        self.suites = suites

    def system(self, value: object, field: str) -> str:
        if not isinstance(value, str) or value not in self.systems:
            raise problem(field, "names a system the report does not list")
        return value

    def suite(self, value: object, field: str) -> SuiteView:
        if not isinstance(value, str) or value not in self.suites:
            raise problem(field, "names a suite the report does not list")
        return self.suites[value]

    def declared(self, suite: SuiteView, value: object, field: str) -> str:
        """A direction the suite declares (a row count in its list of expected directions)."""
        direction = require_direction(value, field)
        if not any(
            entry.direction == direction and entry.rows is not None
            for entry in suite.directions
        ):
            raise problem(field, "names a direction the suite does not declare")
        return direction


def items(raw: object, field: str) -> list[Any]:
    if not isinstance(raw, list):
        raise problem(field, "must be a list")
    return raw


def text_or_null(raw: object, field: str) -> str | None:
    return None if raw is None else require_text(raw, field)


def number_or_null(raw: object, field: str) -> float | int | None:
    """A finite number or null, kept as the file has it (a whole number stays whole)."""
    require_number(raw, field, nullable=True)
    return raw  # type: ignore[return-value]


def language(code: object, field: str) -> str:
    if not isinstance(code, str) or code not in known_language_codes():
        raise problem(field, "must be language codes of the registry")
    return code
