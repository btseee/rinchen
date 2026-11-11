"""The report of a maintenance check (D14, spec M2, M3 and M11): the steps with their states, the
rule that turns them into one state and one exit code, and the bytes of `maintain_report.json`.

A `Report` holds the six steps of the check, once each and in the order of `STEP_IDS`: a step is
never dropped, an unmeasured thing is a `skipped` step with its reason or a `fail` with its reason.
The document is built from observations only (the digests of the inputs, the figures of the lock
and each step's state, reason and detail): no timestamp, no path, no environment value, no line
of any program's output. The same observations give the same bytes. The shapes of a step's
`detail` are those of spec M11 and are written by `rinchen.maintain.check`:

- `doctor`: `{checks: [{id, state, reason}]}`;
- `environment`: `{installed, mismatched: [{name, installed, locked}], mismatched_more, extra,
  extra_more, not_installed}`;
- `drift`: `{entries: [{package, change, locked, available, class, rule, requires}], counts:
  {update, add, remove, drift, blocked}, warnings}`, and `{}` when the step failed or was skipped;
- `reruns`: `{items: [{id, kind, cadence, state, reason, tests, figures}]}`;
- `lock_file` and `lock_consistent`: `{}`.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Final, Literal

from rinchen.maintain.states import STATES, State, is_closed_reason

SCHEMA: Final = "rinchen.maintain/1"
STEP_IDS: Final = (
    "doctor",
    "lock_file",
    "lock_consistent",
    "environment",
    "drift",
    "reruns",
)
# The steps that hold several items: they list the reason of each item in their detail and have
# none of their own, unless they were skipped as a whole.
GROUP_STEP_IDS: Final = ("doctor", "reruns")
EXIT_SUCCESS: Final = 0
EXIT_FAILED: Final = 4

_DRIFT_STEP: Final = "drift"

GroupState = Literal["ok", "fail", "skipped"]
OverallState = Literal["ok", "drift", "fail"]


@dataclass(frozen=True, slots=True)
class Step:
    """One step: its state, the reason and the detail of its shape.

    Only the drift step can be in the `drift` state. An `ok` or `drift` step has no reason; a
    `skipped` step has one; a `fail` step has one unless it is a group step (`GROUP_STEP_IDS`),
    whose items carry their own; and a reason is always a closed one of
    `rinchen.maintain.states`, never free text.
    """

    id: str
    state: State
    reason: str | None
    detail: Mapping[str, Any]

    def __post_init__(self) -> None:
        if self.state not in STATES:
            raise ValueError(f"a step's state is none of {', '.join(STATES)}")
        if self.state == "drift" and self.id != _DRIFT_STEP:
            raise ValueError("only the drift step can be in the drift state")
        if self.state in ("ok", "drift"):
            if self.reason is not None:
                raise ValueError("an ok or drift step has no reason")
        elif self.reason is None:
            if self.state == "skipped" or self.id not in GROUP_STEP_IDS:
                raise ValueError(
                    "a skipped step, and a failed step that is no group, has a reason"
                )
        elif not is_closed_reason(self.reason):
            raise ValueError("a step's reason is none of the closed reasons")


@dataclass(frozen=True, slots=True)
class Report:
    """The whole check: its cadence, the SHA-256 of each input, the figures of the lock and the
    steps, which must be the six of `STEP_IDS` in order."""

    cadence: str
    inputs: Mapping[str, str]
    lock: Mapping[str, Any]
    steps: tuple[Step, ...]

    def __post_init__(self) -> None:
        if tuple(step.id for step in self.steps) != STEP_IDS:
            raise ValueError(
                "steps must be the six steps of the check, once each, in order"
            )


def group_state(states: Sequence[State]) -> GroupState:
    """The state of a step that holds several items: `fail` if any item failed, else `ok` if
    any is `ok`, else `skipped` (also for no items).

    Raises:
        ValueError: an item is `drift`; only the drift step is, and it is not a group.
    """
    if "drift" in states:
        raise ValueError("a group has no drift state: only the drift step can be drift")
    if "fail" in states:
        return "fail"
    if "ok" in states:
        return "ok"
    return "skipped"


def overall_state(report: Report) -> OverallState:
    """`fail` if any step failed, else `drift` if the drift step is `drift`, else `ok`."""
    states = {step.id: step.state for step in report.steps}
    if "fail" in states.values():
        return "fail"
    return "drift" if states[_DRIFT_STEP] == "drift" else "ok"


def exit_code_of(report: Report, *, fail_on_drift: bool) -> int:
    """0 for `ok` and `drift`; 4 for `fail`, and for `drift` when `fail_on_drift` is set."""
    state = overall_state(report)
    if state == "fail" or (state == "drift" and fail_on_drift):
        return EXIT_FAILED
    return EXIT_SUCCESS


def _plain(value: Any) -> Any:
    """The value with every mapping a `dict` and every sequence a `list`, as JSON has them."""
    if isinstance(value, Mapping):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_plain(item) for item in value]
    return value


def report_document(report: Report) -> dict[str, Any]:
    """The document of `maintain_report.json`: the schema, the cadence, the overall state, the
    digests, the lock's figures and the steps in order."""
    return {
        "schema": SCHEMA,
        "cadence": report.cadence,
        "state": overall_state(report),
        "inputs": _plain(report.inputs),
        "lock": _plain(report.lock),
        "steps": [
            {
                "id": step.id,
                "state": step.state,
                "reason": step.reason,
                "detail": _plain(step.detail),
            }
            for step in report.steps
        ],
    }


def report_to_bytes(report: Report) -> bytes:
    """The UTF-8 bytes of the document: sorted keys, two-space indent, no ASCII escapes, one
    final line feed.

    Raises:
        ValueError: a number in a detail is NaN or infinite.
    """
    text = json.dumps(
        report_document(report),
        sort_keys=True,
        indent=2,
        ensure_ascii=False,
        allow_nan=False,
    )
    return f"{text}\n".encode()
