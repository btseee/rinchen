"""What `rinchen maintain check` writes and prints (D14, spec M12): the two documents, the manifest
that ends the commit and the lines on standard output. The order of the steps before it, the
hashing and the exit codes are the command's (`rinchen.cli.maintain`); the report, the Markdown and
the figures are `rinchen.maintain`'s.

An existing output directory may hold only the names this command writes
(`require_only_own_entries`): the command removes and replaces files in it.

The commit has two halves, `begin` before the check runs and `commit` after it. `begin` removes the
old manifest first, so that a run that is interrupted or fails part-way leaves none that vouches for
a mix of old and new files, and then the files it listed: each is taken by its place INSIDE the
directory (a path in a manifest never reaches outside it, and only the shapes this command writes
are ever removed: the two documents and one file in `logs`, `junit` or `parity_export`), so a file
that the manifest did not list is never touched. `commit` writes the two documents whole
(`replace_file`), lists every file the run wrote with its digest (the logs of the commands that ran,
the JUnit report of each rerun that ran and the figures of the parity rerun) and writes the manifest
LAST.

The manifest records the lock file's resolved versions, not the installed ones (they are equal when
the environment step is `ok`): every name of the lock with its versions joined by `, ` in sorted
order, plus `python` and `uv`. A locked package that is itself named `python` or `uv` keeps its own
entry, and the tool goes under `<name> (tool)`. The status holds each step's state and no reason.
Only the last three printed lines hold a path.
"""

from __future__ import annotations

import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final

from rinchen.errors import ConfigError, DataError
from rinchen.maintain.check_types import CheckResult
from rinchen.maintain.lock import LockFile
from rinchen.maintain.report import Report, Step, overall_state
from rinchen.maintain.reruns import JUNIT_DIR, LOGS_DIR, PARITY_DIR, PARITY_FILE
from rinchen.manifest import store
from rinchen.manifest.model import Manifest, ManifestFile
from rinchen.manifest.stage import begin_commit, code_version, output_file
from rinchen.utils.part_file import replace_file

__all__ = [
    "REPORT_FILE",
    "STAGE",
    "SUMMARY_FILE",
    "UNKNOWN_VERSION",
    "begin",
    "commit",
    "observe_code_version",
    "parse_uv_version",
    "require_only_own_entries",
    "result_lines",
]

STAGE: Final = "maintain_check"
REPORT_FILE: Final = "maintain_report.json"
SUMMARY_FILE: Final = "maintain_report.md"
UNKNOWN_VERSION: Final = "unknown"

# The check has no seed: nothing random happens in it.
_NO_SEED: Final = -1
_DOCUMENTS: Final = (REPORT_FILE, SUMMARY_FILE)
_SUBDIRECTORIES: Final = (LOGS_DIR, JUNIT_DIR, PARITY_DIR)
# The top-level names of an output directory that this command writes.
_OWN_ENTRIES: Final = (*_DOCUMENTS, store.manifest_path(Path()).name, *_SUBDIRECTORIES)
_UV_VERSION: Final = re.compile(r"uv ([0-9][0-9A-Za-z.+-]{0,31})(?: .*)?")


# ---- the code version ------------------------------------------------------------------------------------


def observe_code_version() -> str:
    """The code version of the checkout, to be taken BEFORE the command writes anything: the
    nightly's output directory is inside the checkout, and the logs and documents of the run would
    make the tree `-dirty` (an untracked file counts) in every manifest that asked afterwards."""
    return code_version()


# ---- what an output directory may hold --------------------------------------------------------------------


def require_only_own_entries(out_dir: Path) -> None:
    """Raise ConfigError unless an existing `out_dir` holds only the names this command writes at
    its top level (the two documents, the manifest and the directories `logs`, `junit` and
    `parity_export`); a directory that does not exist, and an empty one, pass.

    The command removes and replaces files in the directory it is given: an output directory that
    was any project subdirectory would take them. The message counts the other entries and names
    none of them.
    """
    if not out_dir.is_dir():
        return
    others = sum(1 for entry in out_dir.iterdir() if entry.name not in _OWN_ENTRIES)
    if others:
        raise ConfigError(
            f"--out-dir {out_dir} holds {others} {'entry' if others == 1 else 'entries'} "
            f"that this command does not write (it writes only {', '.join(_OWN_ENTRIES)} "
            "there); pass an empty directory, one that does not exist yet or the one an "
            "earlier run wrote"
        )


# ---- the first half: the old files go ---------------------------------------------------------------------


def begin(out_dir: Path) -> None:
    """Start writing into `out_dir`: the old manifest is removed first and then the files it listed.

    Nothing is made when the directory does not exist.

    Raises:
        DataError: the directory holds the manifest of another stage; nothing is removed.
        OSError: a file cannot be removed.
    """
    stale = _listed_by_old_manifest(out_dir)
    begin_commit(out_dir, STAGE)
    for path in stale:
        path.unlink(missing_ok=True)


def _listed_by_old_manifest(out_dir: Path) -> list[Path]:
    """The files inside `out_dir` that the old manifest of this stage lists and that exist.

    A manifest that cannot be read lists nothing. The manifest of another stage is refused by
    `begin_commit` before anything that this lists is removed.
    """
    try:
        manifest = store.read(out_dir)
    except DataError:
        return []
    if manifest is None:
        return []
    root = out_dir.resolve()
    found = []
    for item in manifest.outputs:
        relative = _place_inside(out_dir, item.path)
        if relative is None:
            continue
        path = out_dir / relative
        if path.is_file() and path.parent.resolve().is_relative_to(root):
            found.append(path)
    return found


def _place_inside(out_dir: Path, listed: str) -> Path | None:
    """Where a file that a manifest lists is inside `out_dir`, or None when it is not one of the
    files this command writes: relative to the directory when the path is under it, else (the
    directory was moved) by the last parts of the path."""
    path = Path(listed)
    try:
        relative = path.relative_to(out_dir)
    except ValueError:
        parent = path.parent.name
        relative = (
            Path(parent, path.name) if parent in _SUBDIRECTORIES else Path(path.name)
        )
    parts = relative.parts
    if len(parts) == 1 and parts[0] in _DOCUMENTS:
        return relative
    if len(parts) == 2 and parts[0] in _SUBDIRECTORIES:
        return relative
    return None


# ---- the second half: the documents and the manifest --------------------------------------------------------


def commit(
    out_dir: Path,
    *,
    report_bytes: bytes,
    markdown: str,
    result: CheckResult,
    inputs: list[ManifestFile],
    lock: LockFile,
    config_sha256: str,
    code_version: str,
    python_version: str,
    uv_version: str,
) -> Path:
    """Write the two documents into `out_dir`, then the manifest, and return the manifest's path.

    `code_version` is the one `observe_code_version` gave before anything was written. `inputs`
    are the files that were hashed before they were read; the outputs the manifest lists
    are the two documents, the logs the run wrote, the JUnit report of each pytest rerun that ran
    and the parity figures of the parity rerun when it wrote them, whichever of them exist.

    Raises:
        OSError: a file cannot be written; the manifest is then not written.
    """
    outputs: list[ManifestFile] = []
    for name, data in zip(
        _DOCUMENTS, (report_bytes, markdown.encode("utf-8")), strict=True
    ):
        path = out_dir / name
        replace_file(path, data)
        outputs.append(output_file(path, None))
    outputs += [output_file(path, None) for path in _written_files(out_dir, result)]
    manifest = Manifest(
        stage=STAGE,
        created=datetime.now(UTC).isoformat(),
        code_version=code_version,
        config_sha256=config_sha256,
        seed=_NO_SEED,
        inputs=inputs,
        outputs=outputs,
        metrics=_metrics(lock, result),
        tool_versions=_tool_versions(lock, python_version, uv_version),
        status=_status(result.report),
    )
    return store.write(out_dir, manifest)


def _step(report: Report, step_id: str) -> Step:
    return next(step for step in report.steps if step.id == step_id)


def _written_files(out_dir: Path, result: CheckResult) -> list[Path]:
    """The files of this run that exist: its logs, then the JUnit report of each pytest rerun that
    ran, then the parity figures when the parity rerun read them."""
    files = [out_dir / name for name in result.logs]
    for item in _step(result.report, "reruns").detail.get("items", []):
        if item["kind"] == "pytest" and item["state"] != "skipped":
            files.append(out_dir / JUNIT_DIR / f"{item['id']}.xml")
        if item["kind"] == "export_parity" and item["figures"] is not None:
            files.append(out_dir / PARITY_DIR / PARITY_FILE)
    return [path for path in files if path.is_file()]


def _tool_versions(
    lock: LockFile, python_version: str, uv_version: str
) -> dict[str, str]:
    versions = {name: ", ".join(found) for name, found in lock.versions.items()}
    for tool, version in (("python", python_version), ("uv", uv_version)):
        versions[f"{tool} (tool)" if tool in versions else tool] = version
    return versions


def _metrics(lock: LockFile, result: CheckResult) -> dict[str, float]:
    entries = _step(result.report, "drift").detail.get("entries", [])
    items = _step(result.report, "reruns").detail.get("items", [])
    return {
        "packages": float(lock.entries),
        "entries": float(len(entries)),
        "blocked": float(sum(1 for entry in entries if entry["class"] == "blocked")),
        "reruns_run": float(sum(1 for item in items if item["state"] != "skipped")),
        **{
            f"seconds.{name}": float(seconds)
            for name, seconds in result.seconds.items()
        },
    }


def _status(report: Report) -> dict[str, str]:
    return {
        **{step.id: step.state for step in report.steps},
        "state": overall_state(report),
    }


def parse_uv_version(text: str) -> str:
    """The version in the first line of `uv --version` (`uv 0.13.7 (...)`), or `unknown` when that
    line is anything else: only a version token ever reaches the manifest."""
    first = text.split("\n", 1)[0].rstrip("\r")
    found = _UV_VERSION.fullmatch(first)
    return found.group(1) if found else UNKNOWN_VERSION


# ---- the printed lines ---------------------------------------------------------------------------------------


def result_lines(report: Report, out_dir: Path, manifest_path: Path) -> list[str]:
    """The lines the command prints: the state and the cadence, one line for the doctor, the lock,
    the environment and the drift, one for each rerun, then the paths of the two documents and the
    manifest. Every line before the paths holds states, counts and closed reasons only."""
    steps = {step.id: step for step in report.steps}
    return [
        f"maintain: {overall_state(report)} ({report.cadence})",
        _doctor_line(steps["doctor"]),
        _lock_line(report, steps["lock_consistent"]),
        _environment_line(steps["environment"]),
        _drift_line(steps["drift"]),
        *_rerun_lines(steps["reruns"]),
        f"report: {out_dir / REPORT_FILE}",
        f"summary: {out_dir / SUMMARY_FILE}",
        f"manifest: {manifest_path}",
    ]


def _line(label: str, state: str, text: str) -> str:
    return f"{label}: {state} ({text})" if text else f"{label}: {state}"


def _doctor_line(step: Step) -> str:
    states = [check["state"] for check in step.detail["checks"]]
    counts = ", ".join(
        f"{states.count(name)} {name}" for name in ("ok", "fail", "skipped")
    )
    return _line("doctor", step.state, counts)


def _lock_line(report: Report, step: Step) -> str:
    figures = f"{report.lock['packages']} entries, {report.lock['names']} names"
    return _line("lock", step.state, step.reason or figures)


def _environment_line(step: Step) -> str:
    detail: dict[str, Any] = dict(step.detail)
    differ = len(detail["mismatched"]) + detail["mismatched_more"]
    extra = len(detail["extra"]) + detail["extra_more"]
    figures = (
        f"{detail['installed']} installed, {differ} differ, {extra} extra, "
        f"{detail['not_installed']} not installed"
    )
    return _line("environment", step.state, step.reason or figures)


def _drift_line(step: Step) -> str:
    if step.reason is not None:
        return _line("drift", step.state, step.reason)
    if step.state == "ok":
        return _line("drift", step.state, "no changes")
    counts = step.detail["counts"]
    figures = (
        f"{len(step.detail['entries'])} entries: {counts['update']} update, "
        f"{counts['add']} add, {counts['remove']} remove; {counts['blocked']} blocked"
    )
    return _line("drift", step.state, figures)


def _rerun_lines(step: Step) -> list[str]:
    return [
        _line(
            f"rerun {item['id']}", item["state"], item["reason"] or _rerun_figures(item)
        )
        for item in step.detail["items"]
    ]


def _rerun_figures(item: dict[str, Any]) -> str:
    if item["tests"] is not None:
        return f"{item['tests']['passed']} passed"
    if item["figures"] is not None:
        figures = item["figures"]
        return (
            f"{figures['exact']}/{figures['sentences']} exact, "
            f"mean |delta| {figures['mean_abs_delta']:.2e}"
        )
    return ""
