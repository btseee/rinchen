"""What `rinchen eval run` writes and prints (D13): the documents of a run built from what was
measured, the files and the manifest of the commit, and the lines on standard output. The order of
the steps before it, the hashing and the exit codes are the command's (`rinchen.cli.eval_run`); the
scoring, `result.json` and `run.json` are `rinchen.eval`'s.

`documents_of` is pure: it scores every direction that ran, builds every result (a block of D12's
robustness is bound to the system here, so a refusal is a `ConfigError` before anything is
written), the run record and the bytes of every file, and nothing touches the disk. `commit`
then writes them under `<out>/<suite>/<system>/`: the files an earlier manifest of this stage listed
go first, each taken by the last parts of its listed path INSIDE the directory (so a directory that
was moved is cleaned and a path in a manifest cannot reach outside it) and no other file is touched;
the old manifest goes before any of them, so that a failure part-way leaves none that vouches for a
mix of old and new files; `hypotheses.jsonl`, the results, `run.json` and, LAST, `manifest.json`
follow.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Final, NamedTuple

from rinchen.cli.eval_suite_output import counted, options_digest
from rinchen.cli.train_common import create_out_dir, exit_cannot_write, write_manifest
from rinchen.errors import DataError
from rinchen.eval.bootstrap import BootstrapSettings
from rinchen.eval.outputs import MAX_OUTPUTS_BYTES, OutputKey
from rinchen.eval.result import (
    OUTPUTS_FILE,
    RESULT_FILE,
    RUN_FILE,
    DirectionResult,
    RunDirection,
    RunRecord,
    RunSettings,
    RunSuite,
    build_result,
)
from rinchen.eval.robustness_input import RobustnessInput, bind, source_record
from rinchen.eval.run_record import result_path
from rinchen.eval.runner import SuiteRun
from rinchen.eval.scoring import score_direction
from rinchen.eval.suites import Suite
from rinchen.manifest import store
from rinchen.manifest.model import ManifestFile
from rinchen.manifest.stage import begin_commit, output_file
from rinchen.utils.part_file import replace_file

__all__ = [
    "STAGE",
    "Documents",
    "Prepared",
    "PreparedFile",
    "begin_run_outputs",
    "commit",
    "documents_of",
    "options_of",
    "prepared",
    "result_lines",
]

STAGE: Final = "eval_run"
_OUTPUTS: Final = "evaluation run"
_PACKAGES: Final = ("ctranslate2", "sentencepiece", "sacrebleu", "numpy", "httpx")
_HEADLINE: Final = "chrf_pp"


class PreparedFile(NamedTuple):
    """A file of the run: its path relative to the run's directory, its bytes and, for a table,
    its row count."""

    name: str
    data: bytes
    rows: int | None


class Documents(NamedTuple):
    """What a run came to: the results of the directions that ran (in the suite's order), the
    unit of their bootstrap (`document`, `segment`, `mixed` or `none`) and the files to write
    in the order they are written (`hypotheses.jsonl`, the results, `run.json`)."""

    results: list[DirectionResult]
    unit: str
    files: list[PreparedFile]


class Prepared(NamedTuple):
    """A run ready to be written: its directory, its `documents`, the inputs in order with the
    digests taken before they were read, the digest of the options and the seed the manifest
    stands on, and the lines that are printed."""

    run_dir: Path
    documents: Documents
    inputs: list[ManifestFile]
    config_sha256: str
    seed: int
    status: dict[str, str]
    metrics: dict[str, float]
    lines: list[str]


def documents_of(
    *,
    suite: Suite,
    suite_sha256: str,
    run: SuiteRun,
    outputs: bytes,
    settings: BootstrapSettings,
    selection: tuple[str, ...] | None,
    robustness: RobustnessInput | None,
    scores: Mapping[str, Mapping[OutputKey, float]],
    score_files: Mapping[str, str],
) -> Documents:
    """The results, the run record and the bytes of every file of a run that produced `outputs`
    (the bytes of `hypotheses.jsonl`, which `run.outputs` serialise to). `scores` are the
    per-segment scores of the neural seam by metric and `score_files` the digest of each file.

    Raises:
        DataError: the outputs file would be larger than a reader accepts.
        ConfigError: `robustness` cannot be bound to the system of the run.
        ValueError: a field of a document breaks its format (a defect of the caller).
    """
    if len(outputs) > MAX_OUTPUTS_BYTES:
        raise DataError(
            f"the outputs of the run are larger than {MAX_OUTPUTS_BYTES // 2**20} MiB, "
            "which the reader of the outputs file does not accept"
        )
    outputs_sha256 = hashlib.sha256(outputs).hexdigest()
    source = None
    if robustness is not None:
        # Bound even when no direction ran: a block that describes another system never passes.
        source = source_record(robustness, bind(robustness, run.system))
    results: list[DirectionResult] = []
    for direction in run.directions:
        if direction.state != "run":
            continue
        view = suite.view(direction.direction)
        keys = [OutputKey(direction.direction, row_id) for row_id in view.ids]
        given = {
            metric: [values[key] for key in keys] for metric, values in scores.items()
        }
        results.append(
            build_result(
                scores=score_direction(
                    view,
                    [run.outputs[key] for key in keys],
                    settings,
                    segment_scores=given or None,
                ),
                system=run.system,
                suite_name=suite.name,
                suite_sha256=suite_sha256,
                direction=direction.direction,
                outputs_sha256=outputs_sha256,
                trained=True,
                robustness=robustness,
            )
        )
    result_bytes = {result.direction: result.to_bytes() for result in results}
    record = RunRecord(
        suite=RunSuite(suite.name, suite_sha256),
        purpose=suite.purpose,
        system=run.system,
        settings=RunSettings(
            settings.resamples, settings.seed, settings.alpha, selection
        ),
        directions=[
            RunDirection(
                d.direction,
                d.state,
                d.reason,
                d.rows,
                result_path(d.direction) if d.state == "run" else None,
            )
            for d in run.directions
        ],
        files={
            OUTPUTS_FILE: outputs_sha256,
            **{
                result_path(direction): hashlib.sha256(data).hexdigest()
                for direction, data in result_bytes.items()
            },
        },
        robustness_source=source,
        segment_scores=score_files,
    )
    files = [
        PreparedFile(OUTPUTS_FILE, outputs, len(run.outputs)),
        *(
            PreparedFile(result_path(direction), data, None)
            for direction, data in result_bytes.items()
        ),
        PreparedFile(RUN_FILE, record.to_bytes(), None),
    ]
    return Documents(results, _unit_of(results), files)


def _unit_of(results: Sequence[DirectionResult]) -> str:
    units = {result.bootstrap.unit for result in results}
    if not units:
        return "none"
    return units.pop() if len(units) == 1 else "mixed"


def options_of(
    *,
    run: SuiteRun,
    settings: BootstrapSettings,
    selection: tuple[str, ...] | None,
    metrics_sha256: str,
) -> str:
    """The digest the manifest's `config_sha256` stands on: this command has no config file of
    its own, so the options that decide the numbers stand in for one. They are the system's name
    and kind and its effective options, the bootstrap, the selection of directions and the SHA-256
    of the metrics configuration."""
    options = {
        "system": run.system.name,
        "kind": run.system.kind,
        "resamples": str(settings.resamples),
        "seed": str(settings.seed),
        "alpha": repr(settings.alpha),
        "selection": "all" if selection is None else ",".join(selection),
        "metrics_config": metrics_sha256,
        **{f"option.{key}": str(value) for key, value in run.system.options.items()},
    }
    return options_digest(options)


def prepared(
    *,
    run_dir: Path,
    suite: Suite,
    run: SuiteRun,
    documents: Documents,
    inputs: list[ManifestFile],
    config_sha256: str,
    settings: BootstrapSettings,
) -> Prepared:
    """A run ready to be written: the manifest's status and metrics and the lines that are
    printed, from the results."""
    return Prepared(
        run_dir=run_dir,
        documents=documents,
        inputs=inputs,
        config_sha256=config_sha256,
        seed=settings.seed,
        status={
            "eval": "ok",
            "system_kind": run.system.kind,
            "purpose": suite.purpose,
            "unit": documents.unit,
        },
        metrics=_metrics_of(documents.results),
        lines=result_lines(suite, run, documents.results),
    )


def _metrics_of(results: Sequence[DirectionResult]) -> dict[str, float]:
    """`n_segments.<direction>` and every metric that has a value, `<metric>.<direction>`."""
    metrics: dict[str, float] = {}
    for result in results:
        metrics[f"n_segments.{result.direction}"] = float(result.n_segments)
        for name, value in result.metrics.items():
            if value is not None:
                metrics[f"{name}.{result.direction}"] = value
    return metrics


# ---- the lines -------------------------------------------------------------------------------------------


def result_lines(
    suite: Suite, run: SuiteRun, results: Sequence[DirectionResult]
) -> list[str]:
    """`suite:`, `system:` and `purpose:`, then one line for every direction the suite
    declares: the chrF++ with its interval and the segments, or `n/a (<state>)` for a direction
    that did not run (its reason is in `run.json`)."""
    by_direction = {result.direction: result for result in results}
    lines = [
        f"suite: {suite.name}",
        f"system: {run.system.name}",
        f"purpose: {suite.purpose}",
    ]
    for direction in run.directions:
        result = by_direction.get(direction.direction)
        if result is None:
            lines.append(f"{direction.direction}: n/a ({direction.state})")
        else:
            lines.append(f"{direction.direction}: {_headline(result)}")
    return lines


def _headline(result: DirectionResult) -> str:
    value = result.metrics[_HEADLINE]
    interval = result.ci95[_HEADLINE]
    assert value is not None, "chrF++ is a count metric: it always has a value"
    spread = "n/a" if interval is None else f"[{interval[0]:.2f}, {interval[1]:.2f}]"
    return (
        f"chrF++ {value:.2f} {spread} "
        f"({counted(result.n_segments, 'segment')}, {_statuses(result)})"
    )


def _statuses(result: DirectionResult) -> str:
    """The statuses of the outputs, `ok` first and the others in name order."""
    names = sorted(result.statuses, key=lambda status: (status != "ok", status))
    return ", ".join(f"{result.statuses[name]} {name}" for name in names)


# ---- the commit ------------------------------------------------------------------------------------------


def commit(ready: Prepared) -> Path:
    """Write the run into its directory (made when missing) and, last, the manifest; return the
    manifest's path. The files of the old manifest of this stage go first. Exits with code 1
    when anything cannot be written.

    Raises:
        DataError: the directory holds the manifest of another stage.
    """
    target = ready.run_dir
    create_out_dir(target, what=_OUTPUTS)
    try:
        begin_run_outputs(target)
        outputs: list[ManifestFile] = []
        for item in ready.documents.files:
            path = target / item.name
            path.parent.mkdir(parents=True, exist_ok=True)
            replace_file(path, item.data)
            outputs.append(output_file(path, item.rows))
        return write_manifest(
            target,
            stage=STAGE,
            config_sha256=ready.config_sha256,
            seed=ready.seed,
            inputs=ready.inputs,
            outputs=outputs,
            metrics=ready.metrics,
            packages=_PACKAGES,
            status=ready.status,
        )
    except OSError as exc:
        exit_cannot_write(target, exc, what=_OUTPUTS)


def begin_run_outputs(target: Path) -> None:
    """Start writing the outputs of a run into `target`: the files the old manifest of this
    stage lists are removed (and the directory of a result that is left empty), then the old
    manifest itself. The manifest of another stage is refused first and nothing is removed.

    Raises:
        DataError: `target` holds the manifest of another stage.
        OSError: a file cannot be removed.
    """
    old = _listed_by_old_manifest(target)
    begin_commit(target, STAGE)
    for relative in old:
        (target / relative).unlink(missing_ok=True)
    for directory in sorted({(target / relative).parent for relative in old}):
        if directory != target and directory.is_dir() and not any(directory.iterdir()):
            directory.rmdir()


def _listed_by_old_manifest(target: Path) -> list[Path]:
    """What the old manifest of this stage lists, as paths inside `target`: `<name>` for a file
    of the run and `<direction>/result.json` for a result, taken from the end of the listed
    path. A manifest that cannot be read or is of another stage lists nothing."""
    try:
        manifest = store.read(target)
    except DataError:
        return []
    if manifest is None or manifest.stage != STAGE:
        return []
    listed: list[Path] = []
    for item in manifest.outputs:
        path = Path(item.path)
        depth = 2 if path.name == RESULT_FILE else 1
        names = path.parts[-depth:]
        if len(names) == depth and all(_is_plain_name(name) for name in names):
            listed.append(Path(*names))
    return listed


def _is_plain_name(name: str) -> bool:
    return Path(name).name == name and name not in (".", "..")
