"""The parser of the text `uv lock --upgrade --dry-run` prints (D14, spec M6).

The lines uv prints for an update, an add and a remove are UNMEASURED: the one dry run that could
print them queries the package index, and no agent runs it. A parser that skipped what it did not
know would turn a changed format into "nothing to upgrade", the worst failure a drift check can
have. So recognition is positive and total. After the escape sequences (`ESC [ digits ; letter`)
and one trailing carriage return are removed and the spaces and tabs at both ends are trimmed,
every non-blank line of the text, which is the text between newline characters, is exactly one of:

- boilerplate: `Using CPython <token> interpreter at: <rest>` or `Resolved <n> package(s) in
  <token>`, skipped and not counted;
- a warning: `warning:` alone or followed by a space and anything, counted;
- the positive line `No lockfile changes detected`, which sets `no_changes`;
- a change line: the verb `Update` or `Updated` (an update: `<name> <old> -> <new>`) or `Add`,
  `Added`, `Remove`, `Removed` (`<name> <version>`), each version with an optional `v`. A name is
  letters, digits and `._-`, starting and ending with a letter or a digit, and is put in PEP 503
  form; a version starts with a digit and holds letters, digits and `.+!_-` (a local label, a
  pre-release and an epoch are versions);
- UNRECOGNISED, counted: everything else, the `error:`, `cause:` and `hint:` lines of a failed
  run included. Tokens are separated by exactly one space, and a line is never split inside.

The parser is pure and never raises. It judges nothing about the changes (a name listed twice, a
version the lock does not hold): `rinchen.maintain.drift` does, with the lock file in hand.
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Final, Literal

from rinchen.maintain.lock import normalize_name

ChangeKind = Literal["update", "add", "remove"]

_ESCAPE: Final = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
_USING: Final = re.compile(r"Using CPython [^ \t]+ interpreter at: .+")
_RESOLVED: Final = re.compile(r"Resolved [0-9]+ packages? in [^ \t]+")
_WARNING: Final = re.compile(r"warning:(?: .*)?")
_POSITIVE: Final = "No lockfile changes detected"
_NAME: Final = r"[A-Za-z0-9](?:[A-Za-z0-9._-]*[A-Za-z0-9])?"
_VERSION: Final = r"v?([0-9][0-9A-Za-z.+!_-]*)"
_CHANGE: Final = re.compile(
    rf"(Update|Updated|Add|Added|Remove|Removed) ({_NAME}) {_VERSION}(?: -> {_VERSION})?"
)
_KINDS: Final[dict[str, ChangeKind]] = {
    "Update": "update",
    "Updated": "update",
    "Add": "add",
    "Added": "add",
    "Remove": "remove",
    "Removed": "remove",
}


@dataclass(frozen=True, slots=True)
class Change:
    """One change line. An update has `old` and `new`, an add only `new`, a remove only `old`;
    `name` is in PEP 503 form and a version has no `v` before it."""

    kind: ChangeKind
    name: str
    old: str | None
    new: str | None


@dataclass(frozen=True, slots=True)
class DryRun:
    """What a text says: its changes in the order of the text, whether the positive line is in
    it, and how many warning lines and how many unrecognised lines it has."""

    changes: tuple[Change, ...]
    no_changes: bool
    warnings: int
    unrecognised: int


def _lines(text: str) -> Iterator[str]:
    for raw in _ESCAPE.sub("", text).split("\n"):
        line = raw.removesuffix("\r").strip(" \t")
        if line:
            yield line


def _change_of(line: str) -> Change | None:
    """The change a line says, or None when it is not a change line (a verb and the arrow must
    agree: an update has a second version, an add and a remove have none)."""
    found = _CHANGE.fullmatch(line)
    if found is None:
        return None
    verb, name, first, second = found.groups()
    kind = _KINDS[verb]
    if (kind == "update") != (second is not None):
        return None
    name = normalize_name(name)
    if kind == "update":
        return Change("update", name, first, second)
    if kind == "add":
        return Change("add", name, None, first)
    return Change("remove", name, first, None)


def parse_dry_run(text: str) -> DryRun:
    """Sort every non-blank line of `text` into its class; see the module docstring."""
    changes: list[Change] = []
    no_changes = False
    warnings = 0
    unrecognised = 0
    for line in _lines(text):
        if _USING.fullmatch(line) or _RESOLVED.fullmatch(line):
            continue
        if _WARNING.fullmatch(line):
            warnings += 1
        elif line == _POSITIVE:
            no_changes = True
        elif (change := _change_of(line)) is not None:
            changes.append(change)
        else:
            unrecognised += 1
    return DryRun(tuple(changes), no_changes, warnings, unrecognised)
