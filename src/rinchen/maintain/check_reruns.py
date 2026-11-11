"""The reruns step of a maintenance check (D14, spec M2, M8 and M9): which of the configured
parity reruns this run makes, and the item for each.

Every configured rerun is listed, run or not. The selection is `--no-reruns` (none), else the
reruns named with `--rerun` (exactly those, whatever the cadence), else the reruns due at the
cadence. A rerun that is not selected is `skipped`: as not selected with `--rerun` when
`--rerun` named others (it may well be due), else as not due at the cadence. A selected one is
`skipped` too, and not run, when the doctor or the environment check failed (its numbers would be
about another stack), and otherwise run by `rinchen.maintain.reruns`. The step's state is that of
the group; a step in which nothing ran says why with the reason of the first selected rerun.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from rinchen.config.maintain import MaintainConfig, RerunConfig, reruns_due
from rinchen.maintain import states
from rinchen.maintain.check_types import CheckOptions
from rinchen.maintain.commands import CommandRunner
from rinchen.maintain.report import Step, group_state
from rinchen.maintain.reruns import RerunResult, run_parity_rerun, run_pytest_rerun


def _selected_ids(config: MaintainConfig, options: CheckOptions) -> frozenset[str]:
    """The reruns this run makes when it makes any: exactly the ones named with `--rerun` when
    there are any, else those due at the cadence."""
    if options.selected:
        return frozenset(options.selected)
    return frozenset(rerun.id for rerun in reruns_due(config, options.cadence))


def _skipped(rerun: RerunConfig, reason: str) -> RerunResult:
    return RerunResult(rerun.id, rerun.kind, rerun.cadence, "skipped", reason)


def _left_out_reason(options: CheckOptions) -> str:
    """Why a rerun that is not selected was not made: `--rerun` named others, or it is not due."""
    if options.selected:
        return states.SKIP_NOT_SELECTED
    return states.skip_not_due(options.cadence)


def _rerun_item(result: RerunResult) -> dict[str, Any]:
    tests = result.tests
    return {
        "id": result.id,
        "kind": result.kind,
        "cadence": result.cadence,
        "state": result.state,
        "reason": result.reason,
        "tests": None
        if tests is None
        else {
            "passed": tests.passed,
            "failed": tests.failed,
            "errors": tests.errors,
            "skipped": tests.skipped,
        },
        "figures": None if result.figures is None else dict(result.figures),
    }


def _skipped_reason(
    options: CheckOptions, results: Sequence[RerunResult], selected: frozenset[str]
) -> str:
    """Why no rerun ran: `--no-reruns`, else the reason of the first rerun that was selected,
    else that none was due."""
    if not options.reruns:
        return states.SKIP_NO_RERUNS
    for result in results:
        if result.id in selected and result.reason is not None:
            return result.reason
    return states.skip_not_due(options.cadence)


def reruns_step(
    config: MaintainConfig,
    options: CheckOptions,
    runner: CommandRunner,
    python: str,
    parent_env: Mapping[str, str],
    project_dir: Path,
    out: Path,
    *,
    environment_failed: bool,
) -> Step:
    """Make the selected reruns (see the module docstring) and list every configured one."""
    selected = _selected_ids(config, options)
    results: list[RerunResult] = []
    for rerun in config.reruns:
        if not options.reruns:
            result = _skipped(rerun, states.SKIP_NO_RERUNS)
        elif rerun.id not in selected:
            result = _skipped(rerun, _left_out_reason(options))
        elif environment_failed:
            result = _skipped(rerun, states.SKIP_ENVIRONMENT)
        elif rerun.kind == "pytest":
            result = run_pytest_rerun(
                rerun,
                runner=runner,
                python=python,
                project_dir=project_dir,
                out_dir=out,
                parent_env=parent_env,
            )
        else:
            result = run_parity_rerun(
                rerun,
                runner=runner,
                python=python,
                project_dir=project_dir,
                out_dir=out,
                parent_env=parent_env,
                inputs=options.export,
            )
        results.append(result)
    state = group_state([result.state for result in results])
    reason = _skipped_reason(options, results, selected) if state == "skipped" else None
    return Step("reruns", state, reason, {"items": [_rerun_item(r) for r in results]})
