"""The files `rinchen eval report` stands on (D13): the suites, the metrics configuration and the run
directories, every file hashed one by one BEFORE any is read, and what is parsed is what was hashed
(`verified_content`).

A run is read through its `manifest.json` and its `run.json`: the manifest must be that of an
`eval_run` stage that finished (`eval: ok`) and must list `run.json` with the digest it has now; every
file `run.json` names (a relative path inside the run directory, so a directory that was moved
still works) is then hashed, compared with the digest `run.json` lists and only then parsed. A
suite is matched to the runs by the SHA-256 of its FILE, never by its name or its path. Nothing is
read from the environment and no text of a suite or of an output is ever in a message.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Final, NamedTuple

from rinchen.cli.eval_run_output import STAGE as RUN_STAGE
from rinchen.cli.train_common import hash_input
from rinchen.config.eval_metrics import EvalMetricsConfig, load_eval_metrics_config
from rinchen.errors import ConfigError, DataError
from rinchen.eval.outputs import outputs_from_bytes
from rinchen.eval.report_runs import RunInput
from rinchen.eval.result import (
    OUTPUTS_FILE,
    RUN_FILE,
    RunRecord,
    result_from_bytes,
    run_from_bytes,
)
from rinchen.eval.suites import Suite
from rinchen.export.verified import verified_content
from rinchen.manifest import store
from rinchen.manifest.model import Manifest, ManifestFile

__all__ = [
    "HashedFiles",
    "HashedRun",
    "LoadedRun",
    "hash_inputs",
    "read_config",
    "read_run_record",
    "read_suites",
    "run_input",
]

_SUITE: Final = "suite file"
_METRICS: Final = "eval metrics config"
_MANIFEST: Final = "run manifest"
_RECORD: Final = "run record"
_FILE: Final = "run file"
_MANIFEST_FILE: Final = store.manifest_path(Path()).name


class HashedRun(NamedTuple):
    """A run directory: where it is, its manifest and its `run.json`, each with the digest taken
    before it was read."""

    directory: Path
    manifest: ManifestFile
    run: ManifestFile


class HashedFiles(NamedTuple):
    """Every file of the command with the digest taken before it was read: the suites (in the
    order given), the metrics configuration and the run directories (in the order given)."""

    suites: tuple[ManifestFile, ...]
    metrics: ManifestFile
    runs: tuple[HashedRun, ...]


class LoadedRun(NamedTuple):
    """A run whose `run.json` was read: the directory, the parsed record, the SHA-256 of its
    `run.json`, its manifest and `run.json` as inputs and the files `run.json` names, hashed and
    checked against what it lists (relative path to entry), in the order `run.json` lists them."""

    directory: Path
    record: RunRecord
    run_sha256: str
    inputs: tuple[ManifestFile, ...]
    files: Mapping[str, ManifestFile]


def hash_inputs(
    suites: Sequence[Path], metrics: Path, runs: Sequence[Path]
) -> HashedFiles:
    """Hash every suite, the metrics configuration and the manifest and `run.json` of every run
    directory, one by one, reading none.

    Raises:
        DataError: a suite file, a manifest or a `run.json` is not a file or cannot be read.
        ConfigError: the metrics configuration is not a file.
    """
    hashed_suites = tuple(
        hash_input(path, what=_SUITE, missing=DataError) for path in suites
    )
    config = hash_input(metrics, what=_METRICS, missing=ConfigError)
    hashed_runs = tuple(
        HashedRun(
            directory,
            hash_input(directory / _MANIFEST_FILE, what=_MANIFEST, missing=DataError),
            hash_input(directory / RUN_FILE, what=_RECORD, missing=DataError),
        )
        for directory in runs
    )
    return HashedFiles(hashed_suites, config, hashed_runs)


# ---- the configuration and the suites -----------------------------------------------------------------


def read_config(entry: ManifestFile) -> EvalMetricsConfig:
    """The metrics configuration from the bytes that were hashed.

    Raises:
        ConfigError: the file is not a configuration, or changed after it was hashed.
    """
    return load_eval_metrics_config(verified_content(Path(entry.path), entry.sha256))


def read_suites(entries: Sequence[ManifestFile]) -> dict[str, Suite]:
    """The suites by the SHA-256 of their files (two files with the same bytes are one suite).

    Raises:
        DataError: a file is not a suite.
        ConfigError: a file changed after it was hashed.
    """
    return {
        entry.sha256: Suite.from_bytes(verified_content(Path(entry.path), entry.sha256))
        for entry in entries
    }


# ---- a run ---------------------------------------------------------------------------------------------


def read_run_record(hashed: HashedRun) -> LoadedRun:
    """The record of the run directory: its manifest (of a finished `eval_run` stage that lists
    `run.json` as it is now), its `run.json`, and every file `run.json` names hashed and checked
    against the digest `run.json` lists (none parsed yet).

    Raises:
        DataError: the manifest is not a manifest of a finished `eval_run` stage that vouches for
            `run.json`, `run.json` is not a run, or a file it names is missing or is not the file it
            lists.
        ConfigError: a file changed after it was hashed.
    """
    directory = hashed.directory
    manifest = _manifest_of(
        verified_content(Path(hashed.manifest.path), hashed.manifest.sha256), directory
    )
    _require_a_finished_run(manifest, directory)
    listed = {Path(item.path).name: item.sha256 for item in manifest.outputs}
    if listed.get(RUN_FILE) != hashed.run.sha256:
        raise DataError(
            f"the manifest of the run directory {directory} does not vouch for {RUN_FILE} "
            "as it is now: it is not listed, or it changed after the run that wrote it"
        )
    record = run_from_bytes(verified_content(Path(hashed.run.path), hashed.run.sha256))
    files: dict[str, ManifestFile] = {}
    for name in record.files:
        files[name] = hash_input(directory / name, what=_FILE, missing=DataError)
    for name, entry in files.items():
        if entry.sha256 != record.files[name]:
            raise DataError(
                f"{name} of the run directory {directory} is not the file that {RUN_FILE} "
                "lists: its SHA-256 differs"
            )
    return LoadedRun(
        directory,
        record,
        hashed.run.sha256,
        (hashed.manifest, hashed.run),
        files,
    )


def _manifest_of(data: bytes, directory: Path) -> Manifest:
    try:
        return Manifest.model_validate_json(data)
    except ValueError as error:
        raise DataError(
            f"the manifest of the run directory {directory} is not a manifest of a stage"
        ) from error


def _require_a_finished_run(manifest: Manifest, directory: Path) -> None:
    if manifest.stage != RUN_STAGE:
        raise DataError(
            f"the manifest of the run directory {directory} is not the manifest of an "
            f"{RUN_STAGE} stage"
        )
    if manifest.status.get("eval") != "ok":
        raise DataError(
            f"the manifest of the run directory {directory} does not say that the run "
            "finished (eval: ok)"
        )


def run_input(loaded: LoadedRun, suite: Suite) -> RunInput:
    """The run as the report reads it: its results and its outputs parsed from the bytes that
    were hashed (`outputs_from_bytes` against the suite the run scored).

    Raises:
        DataError: a result or the outputs are not what their formats say.
        ConfigError: a file changed after it was hashed.
    """
    record = loaded.record
    outputs = outputs_from_bytes(_content(loaded.files[OUTPUTS_FILE]), suite)
    results = tuple(
        result_from_bytes(_content(loaded.files[entry.result]))
        for entry in record.directions
        if entry.result is not None
    )
    return RunInput(
        record=record,
        results=results,
        outputs=outputs,
        suite_sha256=record.suite.sha256,
        run_sha256=loaded.run_sha256,
    )


def _content(entry: ManifestFile) -> bytes:
    return verified_content(Path(entry.path), entry.sha256)
