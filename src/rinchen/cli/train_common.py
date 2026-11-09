"""What the `rinchen train` commands share: the checks and the exits that come before and
after a stage writes, and the manifest of a stage.

`ablate` (`rinchen.cli.train`), `launch` and `average` (`rinchen.cli.train_runs`) and
`docify` (`rinchen.cli.train_docify`) each hash their inputs before they read them, refuse
an output directory that is an input's, create the output directory early, exit with a
clear message when it cannot be written, and write a manifest last. The pieces live here
once; what a command does with them, and the words of its own messages, stay with the
command.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final, NoReturn

from rinchen.cli.errors import FAILURE_EXIT_CODE, exit_with_error
from rinchen.errors import ConfigError, DataError
from rinchen.manifest import store
from rinchen.manifest.model import Manifest, ManifestFile
from rinchen.manifest.stage import (
    build_inputs,
    code_version,
    output_file,
    tool_versions,
)
from rinchen.train.corpus_files import CorpusFiles, corpus_sidecar_path
from rinchen.train.distill_saved_config import PlacementCheck

# What `distill export` and `distill evaluate` print about how the stated tag placement was
# settled, so that a placement nothing could check is never silent.
_PLACEMENT_CHECK_LINES: Final[dict[PlacementCheck, str]] = {
    "checked": "tag placement: checked against the saved config",
    "stated": "tag placement: stated, not recorded in the checkpoint",
}


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


def reject_repeated_corpora(paths: Sequence[Path]) -> None:
    """A corpus file given twice (whatever the spelling or link) would be read twice
    and double every row of it."""
    given: dict[Path, Path] = {}
    for path in paths:
        resolved = path.resolve()
        if resolved in given:
            raise ConfigError(
                f"--corpus {path} names the same file as --corpus {given[resolved]}: "
                "it was given twice and would count every row of it twice"
            )
        given[resolved] = path


def numbers_under(prefix: str, document: Mapping[str, Any]) -> dict[str, float]:
    """The numbers of a statistics document under dotted keys, `prefix.key` and
    `prefix.key.subkey`, for the metrics of a manifest."""
    numbers: dict[str, float] = {}
    for key, value in document.items():
        name = f"{prefix}.{key}"
        if isinstance(value, Mapping):
            numbers.update(numbers_under(name, value))
        else:
            numbers[name] = float(value)
    return numbers


def corpus_manifest_files(corpus: CorpusFiles) -> list[ManifestFile]:
    """The manifest entries of a corpus on disk: its two text files, with the digests and
    the row count the build recorded or checked against the files (not read again), then
    its sidecar record."""
    texts = (
        (corpus.src_path, corpus.src_sha256),
        (corpus.tgt_path, corpus.tgt_sha256),
    )
    return [
        *(
            ManifestFile(
                path=str(path),
                sha256=digest,
                bytes=path.stat().st_size,
                rows=corpus.rows,
            )
            for path, digest in texts
        ),
        output_file(corpus_sidecar_path(corpus), None),
    ]


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


def tag_placement_line(check: PlacementCheck) -> str:
    """The line a command prints about how its tag placement was settled."""
    return _PLACEMENT_CHECK_LINES[check]


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
