"""The installed distributions against the lock (D14, spec M5): is this machine the stack the
lock names?

An installed package whose name the lock does not hold is `extra` (an exact sync removes it) and
an installed version the lock does not list for that name is a `mismatch`: both make the result
`fail`, because every later number of a maintenance run would be about another stack. A locked
package that is not installed is only COUNTED: a lock holds the packages of other platforms,
each behind a marker, so "locked and absent" is normal. A lock may also list several versions of
one name (a vendored wheel per platform), and an environment that has any one of them is equal to
it. Versions are strings compared for equality, never numbers, and both sides are put in PEP 503
form before they are compared.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from importlib import metadata
from typing import Literal

from rinchen.maintain.lock import LockFile, normalize_name


@dataclass(frozen=True, slots=True)
class Mismatch:
    """An installed name some of whose versions the lock does not list for it: all the installed
    versions and all the locked ones, each in sorted order."""

    name: str
    installed: tuple[str, ...]
    locked: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class EnvironmentResult:
    """`fail` when `mismatched` or `extra` is not empty; `installed` is the number of installed
    names and `not_installed` the number of locked names with no installed version."""

    state: Literal["ok", "fail"]
    installed: int
    mismatched: tuple[Mismatch, ...]
    extra: tuple[str, ...]
    not_installed: int


def installed_versions() -> dict[str, tuple[str, ...]]:
    """The distinct versions of each installed distribution, by normalised name.

    A distribution without a name or without a version is skipped: there is nothing to compare.
    """
    found: dict[str, set[str]] = {}
    for distribution in metadata.distributions():
        fields = distribution.metadata
        name = fields.get("Name")
        version = fields.get("Version")
        if name is None or version is None:
            continue
        found.setdefault(normalize_name(name), set()).add(version)
    return {name: tuple(sorted(found[name])) for name in sorted(found)}


def _merged(table: Mapping[str, Sequence[str]]) -> dict[str, set[str]]:
    """The table with its names normalised (two spellings of one name are one name) and their
    versions merged."""
    merged: dict[str, set[str]] = {}
    for name, versions in table.items():
        merged.setdefault(normalize_name(name), set()).update(versions)
    return merged


def check_environment(
    lock: LockFile, installed: Mapping[str, Sequence[str]]
) -> EnvironmentResult:
    """Compare the installed table with the lock; a name with no installed version is not
    installed."""
    present = {name: found for name, found in _merged(installed).items() if found}
    locked = _merged(lock.versions)
    extra = []
    mismatched = []
    for name in sorted(present):
        if name not in locked:
            extra.append(name)
        elif not present[name] <= locked[name]:
            mismatched.append(
                Mismatch(
                    name, tuple(sorted(present[name])), tuple(sorted(locked[name]))
                )
            )
    return EnvironmentResult(
        state="fail" if extra or mismatched else "ok",
        installed=len(present),
        mismatched=tuple(mismatched),
        extra=tuple(extra),
        not_installed=sum(1 for name in locked if name not in present),
    )
