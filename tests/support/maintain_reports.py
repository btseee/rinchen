"""Reports built by hand for the tests of `rinchen.maintain.report` and `report_markdown` (D14,
Task 4): six steps in the documented shapes, with no run behind them.

`report_of` makes the six steps (`ok` unless a state is named) and takes the detail of any step by
its id; `check_row`, `drift_entry`, `rerun_item` and the `*_detail` functions write the shapes of
the document that spec M11 lists, so a test of the Markdown can set exactly the cells it looks at.
Nothing here imports the check: a report is data.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Literal, cast

from rinchen.maintain.report import GROUP_STEP_IDS, STEP_IDS, Report, Step
from rinchen.maintain.states import (
    SKIP_NO_DRIFT,
    State,
    fail_uv_exit,
)

DIGESTS = {"config": "a" * 64, "lock": "b" * 64, "pyproject": "c" * 64}
LOCK_FIGURES = {"packages": 6, "names": 5, "format": {"version": 1, "revision": 5}}


def default_reason(step_id: str, state: str) -> str | None:
    """The reason a step of this state must have when a test names none: a `skipped` step has
    one, a `fail` step has one unless it is a group, and the others have none."""
    if state == "skipped":
        return SKIP_NO_DRIFT
    if state == "fail" and step_id not in GROUP_STEP_IDS:
        return fail_uv_exit(1)
    return None


def report_of(
    states: Mapping[str, str] | None = None,
    reasons: Mapping[str, str] | None = None,
    *,
    cadence: str = "nightly",
    **details: Any,
) -> Report:
    """A report whose six steps have these states (`ok` unless named), reasons (the closed one
    that `default_reason` gives unless named) and details (empty unless named)."""
    wanted = dict(states or {})
    why = dict(reasons or {})
    steps = tuple(
        Step(
            step_id,
            cast(State, wanted.get(step_id, "ok")),
            why.get(step_id, default_reason(step_id, wanted.get(step_id, "ok"))),
            details.get(step_id, {}),
        )
        for step_id in STEP_IDS
    )
    return Report(cadence=cadence, inputs=DIGESTS, lock=LOCK_FIGURES, steps=steps)


def check_row(
    check_id: str, state: str = "ok", reason: str | None = None
) -> dict[str, Any]:
    return {"id": check_id, "state": state, "reason": reason}


def doctor_detail(*rows: Mapping[str, Any]) -> dict[str, Any]:
    return {"checks": list(rows)}


def drift_entry(
    package: str,
    change: Literal["update", "add", "remove"] = "update",
    locked: str | None = "1.0",
    available: str | None = "1.1",
    klass: Literal["drift", "blocked"] = "drift",
    rule: str | None = None,
    requires: Sequence[str] = (),
) -> dict[str, Any]:
    return {
        "package": package,
        "change": change,
        "locked": locked,
        "available": available,
        "class": klass,
        "rule": rule,
        "requires": list(requires),
    }


def drift_detail(
    entries: Sequence[Mapping[str, Any]], warnings: int = 0
) -> dict[str, Any]:
    changes = [entry["change"] for entry in entries]
    classes = [entry["class"] for entry in entries]
    return {
        "entries": list(entries),
        "counts": {
            "update": changes.count("update"),
            "add": changes.count("add"),
            "remove": changes.count("remove"),
            "drift": classes.count("drift"),
            "blocked": classes.count("blocked"),
        },
        "warnings": warnings,
    }


def rerun_item(
    rerun_id: str,
    state: str = "ok",
    reason: str | None = None,
    *,
    tests: tuple[int, int, int, int] | None = None,
    figures: Mapping[str, float | int] | None = None,
    kind: str = "pytest",
    cadence: str = "nightly",
) -> dict[str, Any]:
    """`tests` is `(passed, failed, errors, skipped)`."""
    counted = (
        None
        if tests is None
        else dict(zip(("passed", "failed", "errors", "skipped"), tests, strict=True))
    )
    return {
        "id": rerun_id,
        "kind": kind,
        "cadence": cadence,
        "state": state,
        "reason": reason,
        "tests": counted,
        "figures": None if figures is None else dict(figures),
    }


def reruns_detail(*items: Mapping[str, Any]) -> dict[str, Any]:
    return {"items": list(items)}
