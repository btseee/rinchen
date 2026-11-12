"""What `rinchen tiny run` writes and prints (D15, spec P11 and P13): the names an output directory may hold,
the old run's files that go before a new run starts, the two documents, the manifest that ends the commit
and the lines on standard output. The order of the steps and the exit codes are the command's
(`rinchen.cli.tiny`); the report and its Markdown are `rinchen.tiny.report`'s and `report_markdown`'s.

An existing output directory may hold only the names this command writes (`require_only_own_entries`):
the command removes and replaces files in it. The commit has two halves. `begin` runs before the stages:
the old manifest is removed FIRST, so that a run that is interrupted or fails part-way leaves none that
vouches for a mix of old and new files, and then the stage directories, the logs and the two documents a
run owns (a run never resumes: it starts from nothing). `write_documents` and `write_manifest` end it, the
manifest last.

The manifest (stage `tiny_run`) records every file the run hashed, the two documents and the checkpoint
files that exist when it is written (the SentencePiece model, the converter's `model.bin`, the averaged
weights and the `model.bin` of each export: a run that failed lists fewer), the digest of the profile file,
the seed, every package version of the lock (a name locked at several versions joined by `, ` in sorted
order) and `python`, the seconds of each stage (`seconds.<stage>`, 0.0 for a stage that did not run), the
two figures a run gives (`rows.kept` and `train.steps`, left out when the stage did not give them) and the
state of each stage. It holds seconds and a creation time, so it is not byte-stable across runs: the report
is.
"""

from __future__ import annotations

import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Final

from rinchen.config.tiny import STAGE_IDS, TinyConfig
from rinchen.errors import ConfigError, DataError, RinchenError
from rinchen.maintain.lock import LockFile
from rinchen.manifest import store
from rinchen.manifest.model import Manifest, ManifestFile
from rinchen.manifest.stage import begin_commit, code_version, output_file
from rinchen.tiny.context import STAGE_DIRECTORIES, stage_dir
from rinchen.tiny.model_stages import CONVERTER_MODEL_DIR, SPM_MODEL_FILE
from rinchen.tiny.model_stages_rest import ABLATE_DIR, WEIGHTS_FILE
from rinchen.tiny.pipeline import PipelineRun
from rinchen.tiny.report import WALL_TIME, Report, parse_report, to_bytes
from rinchen.tiny.report_markdown import render
from rinchen.tiny.states import StageRecord
from rinchen.tiny.thresholds import ThresholdResult
from rinchen.utils.part_file import replace_file

__all__ = [
    "OWN_ENTRIES",
    "REPORT_FILE",
    "STAGE",
    "SUMMARY_FILE",
    "ReportDefect",
    "begin",
    "build_manifest",
    "checkpoint_files",
    "clear_run",
    "observe_code_version",
    "report_documents",
    "require_only_own_entries",
    "result_lines",
    "write_documents",
    "write_manifest",
]

STAGE: Final = "tiny_run"
REPORT_FILE: Final = "report.json"
SUMMARY_FILE: Final = "report.md"
LOGS: Final = "logs"
# The directories of the stages, once each, in the order of the stages.
_STAGE_DIRECTORIES: Final = tuple(dict.fromkeys(STAGE_DIRECTORIES.values()))
# What a run removes before it starts: everything it writes but the manifest, which goes first.
_OWNED: Final = (REPORT_FILE, SUMMARY_FILE, LOGS, *_STAGE_DIRECTORIES)
OWN_ENTRIES: Final = (store.manifest_path(Path()).name, *_OWNED)
# The name of the file CTranslate2 reads in a model directory (the converter's and each export's).
_CT2_MODEL_FILE: Final = "model.bin"
# A run that is killed while it writes a file leaves `<name>.<hex>.part` beside it.
_PART_HINT: Final = "; an entry ending in .part is left by a run that was killed while it wrote: delete it"
_DEFECT: Final = (
    "the report of the run is not a valid report: this is a defect of the program; "
    "the stages that ran wrote their directories, which remain, but "
    "report.json, report.md and manifest.json were not written, and "
    "the files of an earlier run were removed when this run began"
)


class ReportDefect(RinchenError):
    """The report of a run is not a report that parses back to itself: a defect of the program and not
    of the run. It is raised after the stages ran, so the old manifest, documents and stage directories
    are already gone (`begin` removed them) and the stage directories of this run remain; only the two
    documents and the manifest of this run are absent. The message says so and names no value."""

    def __init__(self) -> None:
        super().__init__(_DEFECT)


# ---- the code version ------------------------------------------------------------------------------------


def observe_code_version() -> str:
    """The code version of the checkout, to be taken BEFORE the command writes anything: the output
    directory may be inside the checkout, and what the run writes would make the tree `-dirty` (an
    untracked file counts) in every manifest that asked afterwards."""
    return code_version()


# ---- what an output directory may hold ------------------------------------------------------------------


def require_only_own_entries(out_dir: Path) -> None:
    """Raise ConfigError unless an existing `out_dir` holds only the names this command writes at its top
    level (`OWN_ENTRIES`); a directory that does not exist, and an empty one, pass.

    The command removes and replaces files in the directory it is given. The message counts the other
    entries and names none.
    """
    if not out_dir.is_dir():
        return
    others = [
        entry.name for entry in out_dir.iterdir() if entry.name not in OWN_ENTRIES
    ]
    if others:
        hint = _PART_HINT if any(name.endswith(".part") for name in others) else ""
        raise ConfigError(
            f"--out-dir {out_dir} holds {len(others)} "
            f"{'entry' if len(others) == 1 else 'entries'} that this command does not write "
            f"(it writes only {', '.join(OWN_ENTRIES)} there); pass an empty directory, one that "
            f"does not exist yet or the one an earlier run wrote{hint}"
        )


# ---- the first half: the old run goes -----------------------------------------------------------------------


def _remove(path: Path) -> None:
    """Remove a file, a link (never what it points at) or a directory tree; nothing when there is none."""
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.is_dir():
        shutil.rmtree(path)


def clear_run(out_dir: Path) -> None:
    """Remove from `out_dir` the stage directories, the logs and the two documents a run owns, and nothing
    else (the manifest is `begin`'s). Nothing is made when the directory does not exist.

    Raises:
        OSError: something cannot be removed.
    """
    for name in _OWNED:
        _remove(out_dir / name)


def begin(out_dir: Path) -> None:
    """Start writing into `out_dir`: the old manifest is removed first and then the files of the old run.

    Raises:
        DataError: the directory holds the manifest of another stage; nothing is removed.
        OSError: a file cannot be removed.
    """
    begin_commit(out_dir, STAGE)
    clear_run(out_dir)


# ---- the second half: the documents and the manifest ------------------------------------------------------


def report_documents(report: Report) -> tuple[bytes, str]:
    """The bytes of `report.json` and the text of `report.md`, after the bytes were read back as a report
    and that report rendered the same text.

    Raises:
        ReportDefect: the bytes do not parse back, or the Markdown of what they parse to differs.
    """
    data = to_bytes(report)
    markdown = render(report)
    try:
        again = render(parse_report(data))
    except (DataError, ValueError) as error:
        raise ReportDefect from error
    if again != markdown:
        raise ReportDefect
    return data, markdown


def write_documents(
    out_dir: Path, report_bytes: bytes, markdown: str
) -> list[ManifestFile]:
    """Write `report.json` and `report.md` into `out_dir`, each whole, and list them with their digests.

    Raises:
        OSError: a file cannot be written.
    """
    entries = []
    for name, data in (
        (REPORT_FILE, report_bytes),
        (SUMMARY_FILE, markdown.encode("utf-8")),
    ):
        path = out_dir / name
        replace_file(path, data)
        entries.append(output_file(path, None))
    return entries


def checkpoint_files(out_dir: Path, config: TinyConfig) -> list[Path]:
    """The checkpoint files of the run that exist, in the order of the stages: the SentencePiece model,
    the converter's `model.bin`, the averaged weights of the variant and the `model.bin` of each export."""

    def at(stage_id: str, *parts: str) -> Path:
        directory = stage_dir(out_dir, stage_id)
        assert directory is not None, (
            "the stages that write checkpoints have a directory"
        )
        return directory.joinpath(*parts)

    wanted = [
        at("spm", SPM_MODEL_FILE),
        at("converter", CONVERTER_MODEL_DIR, _CT2_MODEL_FILE),
        *sorted(
            at("average", ABLATE_DIR, "variants").glob(f"*/run/averaged/{WEIGHTS_FILE}")
        ),
        *(
            at("export", quantization, _CT2_MODEL_FILE)
            for quantization in config.export.quantizations
        ),
    ]
    return [path for path in wanted if path.is_file()]


def _fact(run: PipelineRun, stage_id: str, key: str) -> int | None:
    """A count a stage gave in its facts, or None when it gave none (or something that is not a count)."""
    for result in run.results:
        if result.record.id == stage_id:
            value = result.facts.get(key)
            return value if type(value) is int and value >= 0 else None
    return None


def _metrics(run: PipelineRun) -> dict[str, float]:
    metrics = {
        f"seconds.{stage_id}": float(run.seconds[stage_id]) for stage_id in STAGE_IDS
    }
    for name, stage_id, key in (
        ("rows.kept", "filter", "kept"),
        ("train.steps", "train", "steps"),
    ):
        count = _fact(run, stage_id, key)
        if count is not None:
            metrics[name] = float(count)
    return metrics


def _tool_versions(lock: LockFile, python_version: str) -> dict[str, str]:
    versions = {name: ", ".join(found) for name, found in lock.versions.items()}
    versions["python (tool)" if "python" in versions else "python"] = python_version
    return versions


def build_manifest(
    *,
    out_dir: Path,
    config: TinyConfig,
    config_sha256: str,
    seed: int,
    inputs: list[ManifestFile],
    documents: list[ManifestFile],
    lock: LockFile,
    run: PipelineRun,
    code_version: str,
    python_version: str,
) -> Manifest:
    """The manifest of a run: `inputs` are the files hashed before they were read, `documents` the two
    written, and the checkpoint files that exist under `out_dir` are listed after them. `code_version` is
    the one `observe_code_version` gave before anything was written."""
    return Manifest(
        stage=STAGE,
        created=datetime.now(UTC).isoformat(),
        code_version=code_version,
        config_sha256=config_sha256,
        seed=seed,
        inputs=inputs,
        outputs=[
            *documents,
            *(output_file(path, None) for path in checkpoint_files(out_dir, config)),
        ],
        metrics=_metrics(run),
        tool_versions=_tool_versions(lock, python_version),
        status={result.record.id: result.record.state for result in run.results},
    )


def write_manifest(out_dir: Path, manifest: Manifest) -> Path:
    """Write the manifest into `out_dir`, whole or not at all, and return its path: it is the commit
    marker, so it is written after every output it lists.

    Raises:
        OSError: it cannot be written.
    """
    return store.write(out_dir, manifest)


# ---- the printed lines ----------------------------------------------------------------------------------------


def _stage_line(record: StageRecord) -> str:
    reason = f" ({record.reason})" if record.reason else ""
    return f"{record.id}: {record.state}{reason}"


def _threshold_line(result: ThresholdResult, elapsed_seconds: float) -> str:
    # The report keeps the verdict of the wall time and not its seconds; the screen shows them.
    value = elapsed_seconds if result.id == WALL_TIME else result.value
    shown = "-" if value is None else format(value, ".6g")
    return (
        f"threshold {result.id}: {result.verdict} (value {shown}, bound {result.bound})"
    )


def result_lines(
    report: Report, elapsed_seconds: float, out_dir: Path, manifest_path: Path
) -> list[str]:
    """The lines the command prints: a line for each stage with its state and closed reason, one for each
    threshold with its value (and, for the wall time, the seconds the run took), the verdict and the
    paths of the two documents and the manifest. Only the last three hold a path."""
    return [
        *(_stage_line(record) for record in report.stages),
        *(_threshold_line(result, elapsed_seconds) for result in report.thresholds),
        f"verdict: {report.verdict}",
        f"report: {out_dir / REPORT_FILE}",
        f"summary: {out_dir / SUMMARY_FILE}",
        f"manifest: {manifest_path}",
    ]
