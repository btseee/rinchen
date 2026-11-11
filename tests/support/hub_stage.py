"""A staged release to plan and upload, and the helpers that damage it in exactly one way (D13, Task 10).

`staged_release(tmp_path)` is a stage that the real `rinchen release stage` wrote over a fabricated
export and the golden report (`tests.support.release_fixture`): the nine files of `UPLOAD_NAMES` and
`manifest.json`. `rewrite(stage, name, data)` replaces one file and keeps every document that vouches
for it in step (the `files` table of `release.json`, then `release.json`'s own entry in the manifest
and the manifest), so that a refusal a test provokes is the one it means and not a digest that
differs on the way; `stale=True` leaves the vouching documents as they were, which is how a changed
file is made. `bytes_under` reads every file under a directory, for the searches of the tests that
the token is in none of them.

Nothing here loads a model, opens a socket or reads the environment.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any, Final

from click.testing import Result

from rinchen.cli.main import app
from rinchen.export.release import RELEASE_FILE, release_to_bytes
from rinchen.manifest import store
from tests.support.eval_cli import runner
from tests.support.release_fixture import Release, fabricated_release

MANIFEST: Final = "manifest.json"
PLAN_FILE: Final = "hub_plan.json"
COMMIT_FILE: Final = "hub_commit.json"


def staged_release(tmp_path: Path, **options: Any) -> Release:
    """A fabricated release whose stage was written by `rinchen release stage`."""
    return fabricated_release(tmp_path, staged=True, **options)


def hub_args(
    release: Release,
    out_dir: Path,
    *extra: str,
    stage: Path | None = None,
    repo_id: str | None = None,
) -> list[str]:
    """`release hub --stage S --repo-id I --out-dir O` and `extra`, each option the release's own
    unless it is given."""
    return [
        "release",
        "hub",
        "--stage",
        str(release.stage if stage is None else stage),
        "--repo-id",
        release.repo_id if repo_id is None else repo_id,
        "--out-dir",
        str(out_dir),
        *extra,
    ]


def invoke_hub(
    release: Release,
    out_dir: Path,
    *extra: str,
    env: dict[str, str | None] | None = None,
    **options: Any,
) -> Result:
    """Run `rinchen release hub` over `release` in this process; `env` sets (or, for None, unsets)
    variables of the environment for the call."""
    return runner.invoke(app, hub_args(release, out_dir, *extra, **options), env=env)


def sha256_of(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def rewrite(
    stage: Path,
    name: str,
    data: bytes,
    *,
    stale: bool = False,
    in_manifest: bool = True,
) -> None:
    """Replace `stage/name` by `data`. Unless `stale`, `release.json` lists the new size and digest
    of a file it lists and the stage's manifest lists the new digest of the file (unless
    `in_manifest` is false) and, for a change of another file, of `release.json`, so that nothing
    but the content differs."""
    (stage / name).write_bytes(data)
    if stale:
        return
    if name != RELEASE_FILE:
        _relist_in_release(stage, name, data)
    if in_manifest:
        _relist_in_manifest(stage, name)
    if name != RELEASE_FILE:
        _relist_in_manifest(stage, RELEASE_FILE)


def _relist_in_release(stage: Path, name: str, data: bytes) -> None:
    document = json.loads((stage / RELEASE_FILE).read_text(encoding="utf-8"))
    if name in document["files"]:
        document["files"][name] = {"bytes": len(data), "sha256": sha256_of(data)}
        (stage / RELEASE_FILE).write_bytes(release_to_bytes(document))


def _relist_in_manifest(stage: Path, name: str) -> None:
    manifest = store.read(stage)
    assert manifest is not None
    data = (stage / name).read_bytes()
    for entry in manifest.outputs:
        if Path(entry.path).name == name:
            entry.sha256 = sha256_of(data)
            entry.bytes = len(data)
    store.write(stage, manifest)


def release_document(stage: Path) -> dict[str, Any]:
    """The parsed `release.json` of the stage."""
    document = json.loads((stage / RELEASE_FILE).read_text(encoding="utf-8"))
    assert isinstance(document, dict)
    return document


def bytes_under(directory: Path) -> list[bytes]:
    """The bytes of every file under `directory`, the files' names (as bytes) included."""
    found: list[bytes] = []
    for path in sorted(directory.rglob("*")):
        found.append(str(path.relative_to(directory)).encode())
        if path.is_file() and not path.is_symlink():
            found.append(path.read_bytes())
    return found


def holds(needle: str, haystacks: Iterable[bytes | str]) -> bool:
    """Whether `needle` is in any of `haystacks` (text and bytes alike)."""
    wanted = needle.encode()
    return any(
        wanted in (item if isinstance(item, bytes) else item.encode("utf-8", "replace"))
        for item in haystacks
    )
