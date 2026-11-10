"""The runner of D13 (spec T6): ask a system which directions of a suite it can run, translate the
runnable ones with ONE call and say, for EVERY direction the suite declares, what became of it.

The state of a direction is `run`, `not_trained` (a model was not trained for it: the engine would
answer in the wrong language, so it is never translated or scored), `unsupported_language` (a
baseline lacks a language), `not_in_file` (a file holds no output for it) or `not_selected` (the
operator selected other directions). A state other than `run` is a reason, never a missing line:
every declared direction appears in `SuiteRun.directions`, in the order of the suite, with the
number of rows it has.

The system gets the requests of the runnable directions in the order of the suite's directions and
rows (the draws of the bootstrap are bound to that order) and must answer one output per request;
a different number is an `UpstreamError`. A status that is not `ok` is not an error: it is the
system's own word about the text, scored and counted later.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Final

from rinchen.errors import ConfigError, UpstreamError
from rinchen.eval.outputs import Output, OutputKey, direction_place
from rinchen.eval.suites import Suite
from rinchen.eval.systems import (
    DirectionState,
    SuiteRequest,
    SuiteSystem,
    SystemRecord,
    Verdict,
)

__all__ = ["NOT_SELECTED_REASON", "DirectionRun", "SuiteRun", "run_suite"]

NOT_SELECTED_REASON: Final = "not in the selection"
_DIGEST: Final = re.compile(r"[0-9a-f]{64}")


@dataclass(frozen=True, slots=True)
class DirectionRun:
    """What became of one declared direction: its `state`, the `reason` when it did not run
    and the number of `rows` the suite has for it (whatever the state: the denominator of
    everything said about it)."""

    direction: str
    state: DirectionState
    reason: str | None
    rows: int


@dataclass(frozen=True, slots=True)
class SuiteRun:
    """A system over a suite: the suite's name and digest, the `system` record, one
    `DirectionRun` for every direction the suite declares (in its order) and the `outputs` of
    the directions that ran, keyed in the order of the suite's directions and rows."""

    suite_name: str
    suite_sha256: str
    system: SystemRecord
    directions: tuple[DirectionRun, ...]
    outputs: Mapping[OutputKey, Output]

    def __post_init__(self) -> None:
        object.__setattr__(self, "directions", tuple(self.directions))
        object.__setattr__(self, "outputs", MappingProxyType(dict(self.outputs)))


def run_suite(
    suite: Suite,
    suite_sha256: str,
    system: SuiteSystem,
    selection: Sequence[str] | None,
) -> SuiteRun:
    """Run `system` over `suite`: the directions in `selection` (all of them when it is None)
    that the system can run, with one call.

    Raises:
        ValueError: `suite_sha256` is not 64 lowercase hexadecimal characters.
        ConfigError: `selection` names a direction the suite does not declare (named only when
            it has the shape of a direction).
        UpstreamError: the system answered a different number of outputs than requests.
    """
    if not _DIGEST.fullmatch(suite_sha256):
        raise ValueError("suite_sha256 must be 64 lowercase hexadecimal characters")
    selected = _selected(suite, selection)
    runs: list[DirectionRun] = []
    requests: list[SuiteRequest] = []
    for direction in suite.directions:
        view = suite.view(direction)
        if selected is not None and direction not in selected:
            verdict = Verdict("not_selected", NOT_SELECTED_REASON)
        else:
            verdict = system.verdict(direction)
        runs.append(
            DirectionRun(direction, verdict.state, verdict.reason, len(view.ids))
        )
        if verdict.state == "run":
            requests.extend(
                SuiteRequest(direction, row_id, view.src_lang, view.tgt_lang, text)
                for row_id, text in zip(view.ids, view.sources, strict=True)
            )
    outputs = _translated(system, requests)
    return SuiteRun(suite.name, suite_sha256, system.record, tuple(runs), outputs)


def _selected(suite: Suite, selection: Sequence[str] | None) -> frozenset[str] | None:
    if selection is None:
        return None
    for entry in selection:
        if entry not in suite.directions:
            raise ConfigError(
                "the selection names a direction the suite does not declare: "
                f"{direction_place(entry)}"
            )
    return frozenset(selection)


def _translated(
    system: SuiteSystem, requests: list[SuiteRequest]
) -> dict[OutputKey, Output]:
    """The outputs of `requests`, keyed in their order; no call when there is none."""
    if not requests:
        return {}
    answers = system.translate(requests)
    if len(answers) != len(requests):
        raise UpstreamError(
            f"the system answered {len(answers)} outputs for {len(requests)} requests"
        )
    return {
        OutputKey(request.direction, request.id): answer
        for request, answer in zip(requests, answers, strict=True)
    }
