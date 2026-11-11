"""What `rinchen eval suite` writes and prints (D13): the suite and the manifest of a build, the
manifest of a fetch, the removal of what an earlier run of the same stage wrote, and the lines
`inspect` prints. The order of the steps before it, the hashing and the exit codes are the
commands' (`rinchen.cli.eval_suite`).

The files an earlier manifest of the same stage lists are removed before the new ones are written,
each by the last parts of its listed path taken INSIDE the output directory (so a directory that was
moved is still cleaned, and a path in a manifest cannot reach outside it), and no other file is
touched; the old manifest goes first, so that a failure part-way leaves none that vouches for a mix
of old and new files. The manifest is written last.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Final, NamedTuple

from rinchen.cli.eval_suite_files import FETCH_STAGE, UNPINNED
from rinchen.cli.eval_suite_plan import BuildPlan
from rinchen.cli.train_common import create_out_dir, exit_cannot_write, write_manifest
from rinchen.errors import DataError
from rinchen.eval.suite_rules import SuiteSource
from rinchen.eval.suites import Suite, write_suite
from rinchen.manifest import store
from rinchen.manifest.model import ManifestFile
from rinchen.manifest.stage import begin_commit, output_file

BUILD_STAGE: Final = "eval_suite"
# Nothing random happens in a fetch or a build.
_NO_SEED: Final = -1
_BUILD_PACKAGES: Final = ("polars",)
_FETCH_PACKAGES: Final = ("datasets", "huggingface_hub", "polars")
# A shard of a fetch is listed as `<split>/<language>/<file>`, a suite as `<name>.json`.
SHARD_PATH_PARTS: Final = 3
SUITE_PATH_PARTS: Final = 1


class Built(NamedTuple):
    """What a build came to and `commit_build` writes: the suite, the inputs in order with their
    digests, the manifest's status and the digest of the options it stands on, and `provenance`,
    the sentence the command prints (and the status records) when no fetch vouches for the shards."""

    suite: Suite
    inputs: list[ManifestFile]
    status: dict[str, str]
    options: str
    provenance: str | None = None


def status_of(plan: BuildPlan, source: SuiteSource | None) -> dict[str, str]:
    """The status of a build's manifest: what the suite is and, when a fetch vouches for it, the
    revision it was pinned to (`null` otherwise)."""
    revision = None if source is None else source.revision
    return {
        "benchmark": plan.benchmark,
        "split": plan.split,
        "kind": plan.kind,
        "revision": "null" if revision is None else revision,
        "pinned": "true" if revision is not None else "false",
    }


def options_digest(options: Mapping[str, str]) -> str:
    """The commands have no config file of their own, so their options stand in for one: the
    SHA-256 of their canonical JSON (sorted keys, no spaces)."""
    canonical = json.dumps(dict(options), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def commit_build(target: Path, built: Built) -> tuple[Path, Path, str]:
    """Write the suite into `target` (made when missing) and, last, the manifest; return the
    paths of the two and the SHA-256 of the suite. The files the old manifest of this stage lists
    go first. Exits with code 1 when anything cannot be written.

    Raises:
        DataError: another stage's manifest is in `target`, or the suite is larger than the
            reader accepts.
    """
    create_out_dir(target, what="suite")
    suite_path = target / f"{built.suite.name}.json"
    try:
        begin_outputs(target, BUILD_STAGE, parts=SUITE_PATH_PARTS)
        digest = write_suite(built.suite, suite_path)
        manifest_path = write_manifest(
            target,
            stage=BUILD_STAGE,
            config_sha256=built.options,
            seed=_NO_SEED,
            inputs=built.inputs,
            outputs=[output_file(suite_path, len(built.suite.rows))],
            metrics={
                **{
                    f"rows.{language}": float(count)
                    for language, count in built.suite.languages().items()
                },
                "directions": float(len(built.suite.directions)),
            },
            packages=_BUILD_PACKAGES,
            status=built.status,
        )
    except OSError as exc:
        exit_cannot_write(target, exc, what="suite")
    return suite_path, manifest_path, digest


def write_fetch_manifest(
    target: Path,
    config_entry: ManifestFile,
    outputs: list[ManifestFile],
    counts: Mapping[str, int],
    *,
    benchmark: str,
    split: str,
    dataset: str,
    revision: str,
) -> Path:
    """Write the manifest of a fetch into `target` and return its path; `revision` is the commit
    hash or `unpinned`. Exits with code 1 when it cannot be written."""
    try:
        return write_manifest(
            target,
            stage=FETCH_STAGE,
            config_sha256=options_digest(
                {
                    "benchmark": benchmark,
                    "split": split,
                    "revision": revision,
                    "config": config_entry.sha256,
                }
            ),
            seed=_NO_SEED,
            inputs=[config_entry],
            outputs=outputs,
            metrics={f"rows.{language}": float(n) for language, n in counts.items()},
            packages=_FETCH_PACKAGES,
            status={
                "revision": revision,
                "benchmark": benchmark,
                "split": split,
                "dataset": dataset,
            },
        )
    except OSError as exc:
        exit_cannot_write(target, exc, what="fetch manifest")


def begin_outputs(target: Path, stage: str, *, parts: int) -> None:
    """Start writing the outputs of `stage` into `target`: the files the old manifest of this
    stage lists are removed, each by the last `parts` parts of its listed path taken inside
    `target`, then the old manifest itself. The manifest of another stage is refused first and
    nothing is removed.

    Raises:
        DataError: `target` holds the manifest of another stage.
        OSError: a file cannot be removed.
    """
    old = _listed_by_old_manifest(target, stage, parts)
    begin_commit(target, stage)
    for relative in old:
        (target / relative).unlink(missing_ok=True)


def _listed_by_old_manifest(target: Path, stage: str, parts: int) -> list[Path]:
    try:
        manifest = store.read(target)
    except DataError:
        return []
    if manifest is None or manifest.stage != stage:
        return []
    listed: list[Path] = []
    for item in manifest.outputs:
        names = Path(item.path).parts[-parts:]
        if names and all(
            Path(name).name == name and name not in (".", "..") for name in names
        ):
            listed.append(Path(*names))
    return listed


def description(suite: Suite, digest: str) -> list[str]:
    """The lines `inspect` prints: what the suite is, its sentences and directions and, when the
    source says, its revision (`unpinned` for a fetch that had none) and the languages it
    lacks."""
    lines = [
        f"name: {suite.name}",
        f"kind: {suite.kind}",
        f"split: {suite.split}",
        f"purpose: {suite.purpose}",
        f"sha256: {digest}",
        "rows: " + ", ".join(f"{code} {n}" for code, n in suite.languages().items()),
        f"directions: {len(suite.directions)}",
        *(f"  {direction}" for direction in suite.directions),
    ]
    if suite.source.pinned:
        lines.append(f"revision: {suite.source.revision}")
    elif suite.source.dataset is not None:
        lines.append(UNPINNED)
    if suite.source.missing_languages:
        lines.append("missing: " + ", ".join(suite.source.missing_languages))
    return lines


def counted(count: int, noun: str) -> str:
    """`1 row`, `3 rows`."""
    return f"{count} {noun}" if count == 1 else f"{count} {noun}s"
