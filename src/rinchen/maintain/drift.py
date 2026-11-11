"""The drift step's decision and its table (D14, spec M6 and M7): what one run of
`uv lock --upgrade --dry-run` says, set against the lock file and the pin rules.

`drift_outcome` decides in a fixed order, and the first thing that applies ends it with a reason
that is one of the strings of `rinchen.maintain.states` and holds nothing of the output:

1. the run timed out;
2. uv did not run to an exit (it was not found);
3. its exit code is not 0;
4. its text is longer than the limit of the log (`text` is None): the text is never parsed, since
   the lines after a cut could hold the one that says the table is not complete;
5. its text, read by `parse_dry_run`: an unrecognised line, the positive line next to change lines
   and no result at all are each a failure;
6. the cross-check: an update and a remove must name a version the lock lists for that name, an
   add must name a package the lock does not hold, and no `(package, change)` is listed twice.
   Each change line that breaks one of these counts once. A mis-parse, or a format that changed,
   cannot get past this: it needs the lock's own table;
7. nothing but the positive line is `ok`; changes are `drift`, one entry each, sorted by package
   and then by change. `locked` is the lock's string, never the output's.

`blocked` marks an entry some pin rule names. It says which reruns to make before taking the
package and never changes the state: a table of blocked entries is still `drift`, information and
not a failure.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from rinchen.config.maintain import PinRule, rule_for
from rinchen.maintain import states
from rinchen.maintain.commands import LOG_LIMIT
from rinchen.maintain.lock import LockFile
from rinchen.maintain.uv_output import Change, ChangeKind, parse_dry_run

EntryClass = Literal["drift", "blocked"]
DriftState = Literal["ok", "drift", "fail"]


@dataclass(frozen=True, slots=True)
class DriftEntry:
    """One row of the table. `locked` is the lock's version of the package (several, joined by
    `, `, when the lock holds more than one) and is None for an add; `available` is the version
    uv would take and is None for a remove. `rule` and `requires` are those of the pin rule that
    names the package, when `klass` is `blocked`."""

    package: str
    change: ChangeKind
    locked: str | None
    available: str | None
    klass: EntryClass
    rule: str | None
    requires: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class DriftOutcome:
    """`reason` is None unless the state is `fail`; `entries` is empty unless it is `drift`;
    `warnings` is the number of warning lines of the output, 0 when it was never read."""

    state: DriftState
    reason: str | None
    entries: tuple[DriftEntry, ...]
    warnings: int


def _failure(reason: str, warnings: int = 0) -> DriftOutcome:
    return DriftOutcome("fail", reason, (), warnings)


def _disagrees(change: Change, lock: LockFile) -> bool:
    locked = lock.versions.get(change.name)
    if change.kind == "add":
        return locked is not None
    return locked is None or change.old not in locked


def _failing_changes(changes: Sequence[Change], lock: LockFile) -> int:
    """How many change lines break the cross-check or repeat an earlier `(package, change)`."""
    seen: set[tuple[str, ChangeKind]] = set()
    failing = 0
    for change in changes:
        key = (change.name, change.kind)
        if key in seen or _disagrees(change, lock):
            failing += 1
        seen.add(key)
    return failing


def _entry(change: Change, lock: LockFile, rules: Sequence[PinRule]) -> DriftEntry:
    locked = lock.versions.get(change.name)
    rule = rule_for(rules, change.name)
    return DriftEntry(
        package=change.name,
        change=change.kind,
        locked=", ".join(locked) if locked else None,
        available=change.new,
        klass="drift" if rule is None else "blocked",
        rule=None if rule is None else rule.id,
        requires=() if rule is None else tuple(rule.requires),
    )


def drift_outcome(
    *,
    exit_code: int | None,
    timed_out: bool,
    text: str | None,
    lock: LockFile,
    rules: Sequence[PinRule],
    timeout_seconds: int,
    log_limit: int = LOG_LIMIT,
) -> DriftOutcome:
    """The state, the reason and the table of one run of uv; `exit_code` is None when the
    process did not run to an exit, and `text` is None when its output was longer than
    `log_limit` bytes. See the module docstring for the order of the decision."""
    if timed_out:
        return _failure(states.fail_timeout(timeout_seconds))
    if exit_code is None:
        return _failure(states.FAIL_UV_MISSING)
    if exit_code != 0:
        return _failure(states.fail_uv_exit(exit_code))
    if text is None:
        return _failure(states.fail_log_too_long(log_limit))

    run = parse_dry_run(text)
    if run.unrecognised:
        return _failure(states.fail_unrecognised(run.unrecognised), run.warnings)
    if run.no_changes and run.changes:
        return _failure(states.FAIL_CONTRADICTION, run.warnings)
    if not run.no_changes and not run.changes:
        return _failure(states.FAIL_NO_RESULT, run.warnings)
    if failing := _failing_changes(run.changes, lock):
        return _failure(states.fail_cross_check(failing), run.warnings)
    if not run.changes:
        return DriftOutcome("ok", None, (), run.warnings)
    entries = sorted(
        (_entry(change, lock, rules) for change in run.changes),
        key=lambda entry: (entry.package, entry.change),
    )
    return DriftOutcome("drift", None, tuple(entries), run.warnings)
