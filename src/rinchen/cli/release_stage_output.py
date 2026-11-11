"""What `rinchen release stage` writes and prints (D13): the nine files of the release and the manifest
that ends the commit, and the lines on standard output. The order of the steps before it, the hashing
and the exit codes are the command's (`rinchen.cli.release_stage`); the card and the release record
are `rinchen.export`'s.

`commit` writes nothing before everything is built. The files the old manifest of this stage listed go
first (each taken by its file name INSIDE the directory, and no other file is touched), the old manifest
before any of them, so that a failure part-way leaves none that vouches for a mix of old and new files;
then the files in `WRITE_ORDER`, each read back and compared with the bytes it was meant to hold (a copy
that differs from its source is a failed write), and, LAST, `manifest.json`. The manifest is not one of
the nine files: it holds local paths and is never uploaded.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from pathlib import Path
from typing import Final, NamedTuple

from rinchen.cli.eval_suite_output import begin_outputs, options_digest
from rinchen.cli.train_common import create_out_dir, exit_cannot_write, write_manifest
from rinchen.export.record import EXPORT_FILES
from rinchen.export.release import RELEASE_FILE, UPLOAD_NAMES
from rinchen.manifest.model import ManifestFile
from rinchen.manifest.stage import output_file
from rinchen.utils.part_file import replace_file

__all__ = [
    "STAGE",
    "WRITE_ORDER",
    "Prepared",
    "commit",
    "options_of",
    "prepared",
    "result_lines",
]

STAGE: Final = "release_stage"
CARD_FILE: Final = "README.md"
# The order the files are written in, which is the order of the manifest's outputs: the export's
# five files, the two reports, the card and the release record (which lists the digest of the card).
WRITE_ORDER: Final = (
    *EXPORT_FILES,
    "report.json",
    "report.md",
    CARD_FILE,
    RELEASE_FILE,
)
_OUTPUTS: Final = "release stage"
# Nothing random happens in a stage.
_NO_SEED: Final = -1
_DIGEST_SHOWN: Final = 12
_ONE_PART: Final = 1


class Prepared(NamedTuple):
    """A release ready to be written: the bytes of each file by name, the inputs in order with the
    digests taken before they were read, the digest of the options and the manifest's status."""

    files: Mapping[str, bytes]
    inputs: list[ManifestFile]
    config_sha256: str
    status: dict[str, str]


def options_of(*, repo_id: str, system: str) -> str:
    """The digest the manifest's `config_sha256` stands on: this command has no config file of its
    own, so the options that decide the release stand in for one."""
    return options_digest({"repo_id": repo_id, "system": system})


def prepared(
    *,
    files: Mapping[str, bytes],
    inputs: list[ManifestFile],
    repo_id: str,
    system: str,
    evidence: str,
) -> Prepared:
    """A release ready to be written; `files` holds every name of `WRITE_ORDER`."""
    assert set(files) == set(WRITE_ORDER)
    return Prepared(
        files=files,
        inputs=inputs,
        config_sha256=options_of(repo_id=repo_id, system=system),
        status={"release": "ok", "repo_id": repo_id, "evidence": evidence},
    )


def commit(target: Path, ready: Prepared) -> Path:
    """Write the release into `target` (made when missing) and, last, the manifest; return the
    manifest's path. Exits with code 1 when anything cannot be written or a copy differs from its
    source.

    Raises:
        DataError: the directory holds the manifest of another stage.
    """
    create_out_dir(target, what=_OUTPUTS)
    try:
        begin_outputs(target, STAGE, parts=_ONE_PART)
        outputs: list[ManifestFile] = []
        for name in WRITE_ORDER:
            path = target / name
            replace_file(path, ready.files[name])
            entry = output_file(path, None)
            if entry.sha256 != hashlib.sha256(ready.files[name]).hexdigest():
                raise OSError(f"the copy of {name} differs from its source")
            outputs.append(entry)
        return write_manifest(
            target,
            stage=STAGE,
            config_sha256=ready.config_sha256,
            seed=_NO_SEED,
            inputs=ready.inputs,
            outputs=outputs,
            metrics={
                "files": float(len(outputs)),
                "bytes": float(sum(item.bytes or 0 for item in outputs)),
            },
            packages=(),
            status=ready.status,
        )
    except OSError as exc:
        exit_cannot_write(target, exc, what=_OUTPUTS)


def result_lines(ready: Prepared, target: Path, manifest_path: Path) -> list[str]:
    """`stage:` with the directory, a line `<name> <bytes> <first 12 hex digits of the SHA-256>` for
    each uploaded file in the order of the upload and `manifest:` with its path."""
    return [
        f"stage: {target}",
        *(
            f"{name} {len(ready.files[name])} "
            f"{hashlib.sha256(ready.files[name]).hexdigest()[:_DIGEST_SHOWN]}"
            for name in UPLOAD_NAMES
        ),
        f"manifest: {manifest_path}",
    ]
