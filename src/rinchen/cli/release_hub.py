"""`rinchen release hub`: plan the upload of a staged release to the Hub, and with `--upload` make it
(D13). Parsing, the order of the steps, the hashing, the token and the exit codes only; the plan, the
upload and the client are `rinchen.export.hub`'s and the files written, the lines and the manifest are
`rinchen.cli.release_hub_output`'s.

    rinchen release hub --stage DIR --repo-id ORG/NAME --out-dir DIR [--upload] [--public]

The default is a DRY RUN: the stage's `manifest.json` (stage `release_stage`) is hashed and must list the
nine files of the upload, the plan is built (every file is read once and the text files are scanned for
a token and for an absolute local path), the plan is printed (`mode: dry-run`, the repository, a line
`<name> <bytes> <first 12 digits of the SHA-256>` per file and the total) and `hub_plan.json` and the
manifest (stage `release_hub`) are written into `--out-dir`. A dry run builds no operation, creates no
client and reads no environment variable.

With `--upload` the variable `HF_TOKEN` must be set and not blank: it is read here, once, before
anything else is hashed. The plan is printed, the repository is created PRIVATE (`--public` is explicit
and never the default) and the Hub is asked what it is: an existing repository that is public (creating
it again changes nothing about it) is refused unless `--public` is given, before any commit. ONE commit
holds the nine files, `commit: <oid>` and `visibility: <what the Hub reported>` are printed, and
`hub_commit.json`, `hub_plan.json` and, last, the manifest are written; `hub_commit.json` and the
manifest record the visibility the Hub reported. The token is passed to the client and to nothing
else: it is never printed, never written, never in a message; an error of the Hub is named by its
class. `rinchen` never logs in and never reads a token file. Nothing is uploaded unless `--upload` is
given, and no test or workflow of D13 gives it.

Exit codes: 0; 2 for a bad option, a repository other than the one the stage was made for, an
`--out-dir` that is the stage, a missing or blank `HF_TOKEN` for `--upload`, a stage whose file
changed after the plan and a public repository without `--public`; 3 for a stage that is not what its
manifest vouches for or not a release that can be uploaded (`rinchen.export.hub`: each file at fault is
named, no text is), another stage's manifest in `--out-dir`, and a dry run into the directory of an
upload; 4 when the Hub (or its client) fails or does not say what the repository is; 1 when the
output cannot be written; 130 when interrupted (nothing is written, an earlier output stays as it
was). Nothing here touches a GPU.
"""

from __future__ import annotations

import os
from collections import Counter
from pathlib import Path
from typing import Annotated, Final

import typer

from rinchen.cli.errors import rinchen_errors_as_exit_codes
from rinchen.cli.release_hub_output import (
    STAGE,
    Mode,
    Prepared,
    commit,
    plan_lines,
    refuse_to_erase_an_upload,
    require_writable,
    result_lines,
)
from rinchen.cli.release_stage_output import STAGE as RELEASE_STAGE
from rinchen.cli.train_common import hash_input, require_out_apart_from_inputs
from rinchen.errors import ConfigError, DataError
from rinchen.export.hub import (
    HfHubClient,
    HubClient,
    UploadPlan,
    commit_to_bytes,
    plan_upload,
    upload,
)
from rinchen.export.release import UPLOAD_NAMES, check_repo_id
from rinchen.export.verified import verified_content
from rinchen.manifest import store
from rinchen.manifest.model import Manifest, ManifestFile
from rinchen.manifest.stage import refuse_foreign_manifest

_MANIFEST_FILE: Final = store.manifest_path(Path()).name
_STATUS_KEY: Final = "release"


# The docstring below is the help text: typer keeps a line break inside a paragraph, so each
# paragraph is one source line and the help wraps it by the width of the terminal.
def hub_command(
    stage: Annotated[
        Path,
        typer.Option(
            "--stage",
            help="The directory `rinchen release stage` wrote: nine files and manifest.json.",
        ),
    ],
    repo_id: Annotated[
        str,
        typer.Option("--repo-id", help="The repository the release is for, ORG/NAME."),
    ],
    out_dir: Annotated[
        Path,
        typer.Option(
            "--out-dir",
            help="Output directory: hub_plan.json, after an upload hub_commit.json, and manifest.json.",
        ),
    ],
    upload_it: Annotated[
        bool,
        typer.Option(
            "--upload",
            help="Upload for real. Needs the environment variable HF_TOKEN. Without it nothing is sent: the command is a dry run.",
        ),
    ] = False,
    public: Annotated[
        bool,
        typer.Option(
            "--public",
            help="Make the repository public when it is created. The default is private.",
        ),
    ] = False,
) -> None:
    """Plan the upload of a staged release to the Hub: a dry run unless --upload is given.

    The plan lists the nine files with their sizes and digests and refuses a stage that holds a stray file, a token or an absolute local path. With --upload the repository is created private and one commit holds the nine files; the token is read from HF_TOKEN and is never printed or written."""
    mode: Mode = "upload" if upload_it else "dry-run"
    with rinchen_errors_as_exit_codes():
        repo = check_repo_id(repo_id)
        token = token_from_environment() if upload_it else None
        stage_dir = Path(os.path.abspath(stage))
        target = Path(os.path.abspath(out_dir))
        ready = _prepare(stage_dir, repo, target, mode, public)
        for line in plan_lines(ready.plan, mode):
            typer.echo(line)
        if token is None:
            manifest = commit(target, ready, commit_record=None, commit_oid=None)
            for line in result_lines(target, manifest, committed=False):
                typer.echo(line)
            return
        result = upload(ready.plan, stage_dir, open_client(token), token, public=public)
        typer.echo(f"commit: {result.commit_oid}")
        typer.echo(f"visibility: {result.visibility}")
        manifest = commit(
            target,
            ready,
            commit_record=commit_to_bytes(ready.plan, result),
            commit_oid=result.commit_oid,
            visibility=result.visibility,
        )
        for line in result_lines(target, manifest, committed=True):
            typer.echo(line)


def token_from_environment() -> str:
    """The token of the `--upload` path, from the environment variable `HF_TOKEN`: the one place
    that reads the environment, called only for an upload.

    Raises:
        ConfigError: the variable is not set or is blank.
    """
    token = os.environ.get("HF_TOKEN", "").strip()
    if not token:
        raise ConfigError(
            "--upload needs the environment variable HF_TOKEN, set and not blank"
        )
    return token


def open_client(token: str) -> HubClient:
    """The client for the real Hub. A seam: the tests replace it, and no test builds this one over
    the real library."""
    return HfHubClient(token)


# ---- the steps -------------------------------------------------------------------------------------


def _prepare(
    stage: Path, repo_id: str, target: Path, mode: Mode, public: bool
) -> Prepared:
    """Everything up to the output: hash and read the manifest of the stage, plan the upload, bind
    the plan to the manifest and check the output directory (for an upload also that it can be
    written, which is found out before the commit and not after it). Nothing is left written."""
    entry = hash_input(stage / _MANIFEST_FILE, what="stage manifest", missing=DataError)
    manifest = _stage_manifest(verified_content(Path(entry.path), entry.sha256), stage)
    _require_the_nine_listed(manifest, stage)
    plan = plan_upload(stage, repo_id)
    _require_the_plan_vouched_for(manifest, plan, stage)
    inputs = [entry, *_plan_inputs(plan, stage)]
    require_out_apart_from_inputs(
        target, [Path(item.path) for item in inputs], option="--out-dir"
    )
    refuse_foreign_manifest(target, STAGE)
    refuse_to_erase_an_upload(target, mode)
    if mode == "upload":
        require_writable(target)
    return Prepared(plan=plan, inputs=inputs, mode=mode, public=public)


def _stage_manifest(data: bytes, stage: Path) -> Manifest:
    try:
        manifest = Manifest.model_validate_json(data)
    except ValueError as error:
        raise DataError(
            f"the manifest of the stage {stage} is not a manifest of a stage"
        ) from error
    if manifest.stage != RELEASE_STAGE:
        raise DataError(
            f"the manifest of the stage {stage} is not the manifest of a {RELEASE_STAGE} stage"
        )
    if manifest.status.get(_STATUS_KEY) != "ok":
        raise DataError(
            f"the manifest of the stage {stage} does not say that the release was staged "
            f"({_STATUS_KEY}: ok)"
        )
    return manifest


def _listed_digests(manifest: Manifest) -> dict[str, str]:
    """The digest the manifest lists for each file name; a name listed twice is not listed (it
    does not say which file it means)."""
    names = Counter(Path(item.path).name for item in manifest.outputs)
    return {
        Path(item.path).name: item.sha256
        for item in manifest.outputs
        if names[Path(item.path).name] == 1
    }


def _require_the_nine_listed(manifest: Manifest, stage: Path) -> None:
    listed = _listed_digests(manifest)
    missing = [name for name in UPLOAD_NAMES if name not in listed]
    if missing:
        raise DataError(
            f"the manifest of the stage {stage} does not vouch for: "
            + "; ".join(
                f"{name} (not listed by the manifest of the stage)" for name in missing
            )
        )


def _require_the_plan_vouched_for(
    manifest: Manifest, plan: UploadPlan, stage: Path
) -> None:
    listed = _listed_digests(manifest)
    changed = [file.name for file in plan.files if listed[file.name] != file.sha256]
    if changed:
        raise DataError(
            f"the manifest of the stage {stage} does not vouch for: "
            + "; ".join(
                f"{name} (differs from what the manifest of the stage lists)"
                for name in changed
            )
        )


def _plan_inputs(plan: UploadPlan, stage: Path) -> list[ManifestFile]:
    return [
        ManifestFile(path=str(stage / file.name), sha256=file.sha256, bytes=file.bytes)
        for file in plan.files
    ]
