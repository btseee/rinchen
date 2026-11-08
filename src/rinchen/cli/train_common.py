"""What the `rinchen train` commands share: the checks and the exits that come before and
after a stage writes, and the manifest of a stage.

`ablate` (`rinchen.cli.train`), `launch` and `average` (`rinchen.cli.train_runs`) each
hash their inputs before they read them, refuse an output directory that is an input's,
create the output directory early, exit with a clear message when it cannot be written,
and write a manifest last. The pieces live here once; what a command does with them, and
the words of its own messages, stay with the command.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import NoReturn

from rinchen.cli.errors import FAILURE_EXIT_CODE, exit_with_error
from rinchen.errors import ConfigError, DataError
from rinchen.manifest import store
from rinchen.manifest.model import Manifest, ManifestFile
from rinchen.manifest.stage import build_inputs, code_version, tool_versions


def require_usable_timeout(timeout: float | None) -> None:
    """Raise ConfigError unless `timeout` is None or a positive, finite number."""
    if timeout is not None and not 0 < timeout < math.inf:
        raise ConfigError(
            f"--timeout must be a positive, finite number of seconds, got {timeout}"
        )


def require_file(
    path: Path, *, what: str, missing: type[ConfigError | DataError]
) -> None:
    """Raise `missing`, saying `what` was not found, unless `path` is a file."""
    if not path.is_file():
        raise missing(f"{what} not found: {path}")


def hash_input(
    path: Path, *, what: str, missing: type[ConfigError | DataError]
) -> ManifestFile:
    """The manifest entry of the input file `path`, taken before anything reads it.
    A path that is not a file is `missing`, with the message of the loader that would
    have refused it; a file that cannot be read is a DataError."""
    require_file(path, what=what, missing=missing)
    try:
        return build_inputs([path])[0]
    except OSError as exc:
        raise DataError(f"cannot read {what} {path}: {exc}") from exc


def require_out_apart_from_inputs(
    out_dir: Path, sources: Sequence[Path], *, option: str
) -> None:
    """The run writes into `out_dir` (given with `option`), so an input file that lives
    there could be replaced by an output. Compares resolved paths, so a symlink or a
    `..` does not hide the clash, for each file as named and for the file it links to."""
    resolved = out_dir.resolve()
    for source in sources:
        if resolved in (source.resolve().parent, source.parent.resolve()):
            raise ConfigError(
                f"{option} {out_dir} is the directory of the input file {source}; "
                f"pass a different {option}"
            )


def create_out_dir(out_dir: Path, *, what: str) -> None:
    """Create `out_dir` (with its parents), or exit with code 1 saying `what` cannot be
    written there."""
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        exit_cannot_write(out_dir, exc, what=what)


def exit_cannot_write(out_dir: Path, exc: OSError, *, what: str) -> NoReturn:
    exit_with_error(
        f"cannot write the {what} under {out_dir}: {exc}", FAILURE_EXIT_CODE
    )


def write_manifest(
    out_dir: Path,
    *,
    stage: str,
    config_sha256: str,
    seed: int,
    inputs: list[ManifestFile],
    outputs: list[ManifestFile],
    metrics: Mapping[str, float],
    packages: Sequence[str],
    status: Mapping[str, str],
) -> Path:
    """Write the manifest of `stage` into `out_dir`, whole or not at all, and return its
    path. It is the commit marker of the stage: call it after every output it lists is
    written. `packages` are the installed tools whose versions it records."""
    manifest = Manifest(
        stage=stage,
        created=datetime.now(UTC).isoformat(),
        code_version=code_version(),
        config_sha256=config_sha256,
        seed=seed,
        inputs=inputs,
        outputs=outputs,
        metrics=dict(metrics),
        tool_versions=tool_versions(packages),
        status=dict(status),
    )
    return store.write(out_dir, manifest)
