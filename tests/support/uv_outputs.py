"""The recorded outputs of `uv lock ... --dry-run` that the D14 tests parse, each named and
documented as MEASURED or HAND-WRITTEN, and the lock file those outputs speak about.

What was measured (2026-11-11, uv 0.13.7, a copy of the project, offline): the output of a dry run
that finds nothing to change, and the output of an offline dry run that cannot resolve. What
was NOT measured: the lines `uv lock --upgrade --dry-run` prints for an update, an add and a
remove, because that run queries the package index and no agent runs it. `CHANGES_HAND_WRITTEN` is
therefore written by hand from recollection, and nothing here is evidence of uv's real format; the
parser refuses what it does not recognise for exactly that reason, and the first real run is where
the format is learned. The interpreter path in each text is a placeholder: the measured one is the
absolute path of the machine's interpreter.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from types import MappingProxyType

from rinchen.maintain.lock import LockFile

# MEASURED: `uv lock --offline --dry-run` on an unchanged project exited 0 printing these three
# lines. The output of `--upgrade --dry-run` when nothing can be upgraded is assumed to be the
# same (UNMEASURED).
NO_CHANGE_MEASURED = (
    "Using CPython 3.14.10 interpreter at: /opt/python/bin/python3\n"
    "Resolved 187 packages in 1ms\n"
    "No lockfile changes detected\n"
)

# MEASURED: `uv lock --offline --dry-run` after a change of the requirements exited 1 printing
# these lines, with blank lines between the groups: the cache of that machine held one platform's
# resolution only. A non-zero exit decides before any line is read.
OFFLINE_FAILURE_MEASURED = (
    "Using CPython 3.14.10 interpreter at: /opt/python/bin/python3\n"
    "error: No solution found when resolving dependencies for split (markers: "
    "python_full_version == '3.14.*' and sys_platform == 'win32')\n"
    "  cause: Because ctranslate2 was not found in the cache and your project depends on "
    "ctranslate2==4.8.4, we can conclude that your project's requirements are unsatisfiable.\n"
    "\n"
    "hint: The resolution failed for an environment that is not the current one, consider "
    "limiting the environments with `tool.uv.environments`.\n"
    "\n"
    "hint: Packages were unavailable because the network was disabled. When the network is "
    "disabled, registry packages may only be read from the cache.\n"
)

# HAND-WRITTEN (UNMEASURED): what a `uv lock --upgrade --dry-run` that finds changes might print:
# five updates, one add and one remove. The verbs, the `v` before a version and the arrow are
# recalled, never observed.
CHANGES_HAND_WRITTEN = (
    "Using CPython 3.14.10 interpreter at: /opt/python/bin/python3\n"
    "Resolved 189 packages in 2.31s\n"
    "Update numpy v2.5.4 -> v2.5.5\n"
    "Update rich v15.0.0 -> v15.0.1\n"
    "Update huggingface-hub v2.3.0 -> v2.3.1\n"
    "Update safetensors v0.8.0 -> v0.8.1\n"
    "Update nvidia-cudnn-cu13 v9.24.0.43 -> v9.25.0.10\n"
    "Add tomli-w v1.2.0\n"
    "Remove sortedcontainers v2.4.0\n"
)

# The versions of the lock the samples above speak about: `tomli-w` is not in it (it is an add).
SAMPLE_LOCKED = {
    "numpy": ("2.5.4",),
    "rich": ("15.0.0",),
    "huggingface-hub": ("2.3.0",),
    "safetensors": ("0.8.0",),
    "nvidia-cudnn-cu13": ("9.24.0.43",),
    "sortedcontainers": ("2.4.0",),
}


def lock_of(versions: Mapping[str, Sequence[str]]) -> LockFile:
    """A lock file holding these names (already normalised) and versions, built by hand."""
    table = {name: tuple(sorted(found)) for name, found in sorted(versions.items())}
    return LockFile(
        format_version=1,
        revision=5,
        entries=sum(len(found) for found in table.values()),
        versions=MappingProxyType(table),
    )


def lock_for_samples() -> LockFile:
    """The lock of `SAMPLE_LOCKED`."""
    return lock_of(SAMPLE_LOCKED)
