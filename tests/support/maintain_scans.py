"""The content rules of a maintenance report (D14, spec M11), as scans the tests share.

A report and its Markdown are built from observations only: no absolute path, no timestamp, no
host name, no line of another program's output and no value of the environment. `violations`
returns what a text holds of those, so that a test asserts the list is empty and a failure names
the rule. `reasons_of` lists every `reason` of a report document and `closed_reason` says whether
a string is an instance of one of the templates of `rinchen.maintain.states.ALL_REASONS` (each
`<name>` stands for a number or, for the cadence, one of the two words): a reason that is not is
free text, which the report never holds.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Iterator, Mapping
from typing import Any, Final

from rinchen.maintain.states import ALL_REASONS

PATH_MARKERS: Final = (
    "/home",
    "/tmp",
    "/var",
    "/root",
    "/Users",
    "/mnt",
    "/opt",
    "C:\\",
)
TIMESTAMP: Final = re.compile(r"\d{4}-\d{2}-\d{2}")
OUTPUT_LINES: Final = (
    "Using CPython",
    "Resolved",
    "No lockfile changes",
    "error:",
    "hint:",
)
HOST_MARKERS: Final = ("hostname",)

_PLACEHOLDER = re.compile(r"<([a-z]+)>")


def violations(text: str, secrets: Iterable[str] = ()) -> list[str]:
    """The content rules `text` breaks, each as a short name."""
    found = [f"path {marker}" for marker in PATH_MARKERS if marker in text]
    if TIMESTAMP.search(text):
        found.append("timestamp")
    found += [f"host {marker}" for marker in HOST_MARKERS if marker in text]
    found += [f"output {line}" for line in OUTPUT_LINES if line in text]
    found += [
        f"secret {n}" for n, value in enumerate(secrets) if value and value in text
    ]
    return found


def _template_pattern(template: str) -> re.Pattern[str]:
    pieces = []
    last = 0
    for found in _PLACEHOLDER.finditer(template):
        pieces.append(re.escape(template[last : found.start()]))
        pieces.append(
            "(?:nightly|weekly)" if found.group(1) == "cadence" else r"-?[0-9]+"
        )
        last = found.end()
    pieces.append(re.escape(template[last:]))
    return re.compile("".join(pieces))


_TEMPLATES: Final = tuple(_template_pattern(template) for template in ALL_REASONS)


def closed_reason(reason: str) -> bool:
    """Whether `reason` is one of the closed strings or an instance of one of its templates."""
    return any(pattern.fullmatch(reason) for pattern in _TEMPLATES)


def reasons_of(document: Mapping[str, Any]) -> Iterator[str]:
    """Every non-null reason of a report document: the steps', the doctor's checks' and the
    reruns'."""
    for step in document["steps"]:
        if step["reason"] is not None:
            yield step["reason"]
        detail = step["detail"]
        for row in [*detail.get("checks", []), *detail.get("items", [])]:
            if row["reason"] is not None:
                yield row["reason"]
