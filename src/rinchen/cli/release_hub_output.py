"""What `rinchen release hub` writes and prints (D13): `hub_plan.json`, after an upload `hub_commit.json`,
and the manifest that ends the commit, and the lines on standard output. The order of the steps, the
hashing, the token and the exit codes are the command's (`rinchen.cli.release_hub`); the plan, the commit
and their documents are `rinchen.export.hub`'s.

`commit` writes nothing before everything is known. The files the old manifest of this stage listed go
first (each taken by its file name INSIDE the directory, and no other file is touched), the old manifest
before any of them, so that a failure part-way leaves none that vouches for a mix of old and new files;
then the files, each read back and compared with the bytes it was meant to hold, and, LAST, the manifest.
A dry run is refused into the directory of an upload: it would remove the only local record of the
commit. Nothing here holds the token: it is not a parameter of any function of this module.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from pathlib import Path
from typing import Final, Literal, NamedTuple

from rinchen.cli.eval_suite_output import begin_outputs, options_digest
from rinchen.cli.train_common import create_out_dir, exit_cannot_write, write_manifest
from rinchen.errors import DataError
from rinchen.export.hub import UploadPlan
from rinchen.manifest import store
from rinchen.manifest.model import ManifestFile
from rinchen.manifest.stage import output_file
from rinchen.utils.part_file import replace_file

__all__ = [
    "COMMIT_FILE",
    "PLAN_FILE",
    "STAGE",
    "Mode",
    "Prepared",
    "commit",
    "plan_lines",
    "refuse_to_erase_an_upload",
    "result_lines",
]

STAGE: Final = "release_hub"
PLAN_FILE: Final = "hub_plan.json"
COMMIT_FILE: Final = "hub_commit.json"
Mode = Literal["dry-run", "upload"]
_OUTPUTS: Final = "Hub plan"
# Nothing random happens in a plan or an upload.
_NO_SEED: Final = -1
_DIGEST_SHOWN: Final = 12
_ONE_PART: Final = 1


class Prepared(NamedTuple):
    """A plan ready to be recorded: the plan, every input in order with the digest it was read
    with, the mode, and whether an upload would make the repository public."""

    plan: UploadPlan
    inputs: list[ManifestFile]
    mode: Mode
    public: bool

    @property
    def visibility(self) -> str:
        return "public" if self.public else "private"


def plan_lines(plan: UploadPlan, mode: Mode) -> list[str]:
    """The mode, the repository, a line `<name> <bytes> <first 12 hex digits of the SHA-256>` for
    each file in the order of the upload and the total."""
    return [
        f"mode: {mode}",
        f"repo: {plan.repo_id}",
        *(
            f"{file.name} {file.bytes} {file.sha256[:_DIGEST_SHOWN]}"
            for file in plan.files
        ),
        f"total: {plan.total_bytes}",
    ]


def refuse_to_erase_an_upload(target: Path, mode: Mode) -> None:
    """A dry run must not replace the record of an upload.

    Raises:
        DataError: `target` holds the manifest of an upload and `mode` is `dry-run`.
    """
    if mode != "dry-run":
        return
    try:
        old = store.read(target)
    except DataError:
        return
    if old is not None and old.stage == STAGE and old.status.get("mode") == "upload":
        raise DataError(
            f"{target} holds the record of an upload and a dry run would erase it; "
            "use another --out-dir"
        )


def commit(
    target: Path,
    ready: Prepared,
    *,
    commit_record: bytes | None,
    commit_oid: str | None,
    visibility: str | None = None,
) -> Path:
    """Write the files into `target` (made when missing) and, last, the manifest; return the
    manifest's path. `commit_record`, `commit_oid` and `visibility` (what the Hub REPORTED for the
    repository, which the manifest records in place of the one that was asked for) are given after
    an upload and only then. Exits with code 1 when anything cannot be written or a file read back
    differs.

    Raises:
        DataError: the directory holds the manifest of another stage.
    """
    contents: dict[str, bytes] = {PLAN_FILE: ready.plan.to_bytes()}
    if commit_record is not None:
        contents[COMMIT_FILE] = commit_record
    create_out_dir(target, what=_OUTPUTS)
    try:
        begin_outputs(target, STAGE, parts=_ONE_PART)
        outputs: list[ManifestFile] = []
        for name, data in contents.items():
            path = target / name
            replace_file(path, data)
            entry = output_file(path, None)
            if entry.sha256 != hashlib.sha256(data).hexdigest():
                raise OSError(f"the copy of {name} differs from its source")
            outputs.append(entry)
        return write_manifest(
            target,
            stage=STAGE,
            config_sha256=options_digest(
                {
                    "repo_id": ready.plan.repo_id,
                    "mode": ready.mode,
                    "visibility": ready.visibility,
                }
            ),
            seed=_NO_SEED,
            inputs=ready.inputs,
            outputs=outputs,
            metrics={
                "files": float(len(ready.plan.files)),
                "bytes": float(ready.plan.total_bytes),
            },
            packages=("huggingface_hub",) if ready.mode == "upload" else (),
            status=_status(ready, commit_oid, visibility),
        )
    except OSError as exc:
        exit_cannot_write(target, exc, what=_OUTPUTS)


def _status(
    ready: Prepared, commit_oid: str | None, visibility: str | None
) -> Mapping[str, str]:
    status = {
        "mode": ready.mode,
        "repo_id": ready.plan.repo_id,
        "visibility": ready.visibility if visibility is None else visibility,
    }
    if commit_oid is not None:
        status["commit_oid"] = commit_oid
    return status


def result_lines(target: Path, manifest_path: Path, *, committed: bool) -> list[str]:
    """`plan:` with the plan file, after an upload `commit record:`, and `manifest:`."""
    return [
        f"plan: {target / PLAN_FILE}",
        *([f"commit record: {target / COMMIT_FILE}"] if committed else []),
        f"manifest: {manifest_path}",
    ]
