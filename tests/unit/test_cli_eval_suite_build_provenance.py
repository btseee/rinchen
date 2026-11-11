"""`rinchen eval suite build` and the provenance of a fetch (D13 fix wave 1, M3).

A fetch writes `<out>/manifest.json` and shards under `<out>/<split>/<language>/`. Built from `<out>`
the suite is bound to the fetch (dataset, revision, a cross-check of every shard). Built from the
split directory `<out>/<split>` it used to lose all of it without a word. Now the manifest of the
parent directory is used when it is the manifest of a fetch of this benchmark and split (the
cross-check applies), and when no fetch manifest is found at all the command says so in a line
`provenance: ...` and records the same in the status of the build manifest. A source that is not a
directory of shards (text files, a human export) has no fetch to look for and prints nothing.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import pytest

from rinchen.eval.suites import Suite
from tests.support.distill_cli import write_foreign_manifest
from tests.support.eval_suite_cli import (
    MANIFEST_FILE,
    REVISION,
    SHARD,
    FakeLoadDataset,
    SuiteWorkspace,
    fixture_rows,
    install_fake_load_dataset,
    invoke,
    make_workspace,
    read_manifest,
    sha256_of,
    write_config,
)
from tests.support.no_network import no_network  # noqa: F401 - the fixture of the mark

pytestmark = pytest.mark.usefixtures("no_network")

NOTE = "no fetch manifest found, dataset and revision are not recorded"


@pytest.fixture
def workspace(tmp_path: Path) -> SuiteWorkspace:
    return make_workspace(tmp_path)


@pytest.fixture
def fetched(
    workspace: SuiteWorkspace, monkeypatch: pytest.MonkeyPatch
) -> tuple[Path, Path]:
    config = write_config(workspace.root / "suites.yaml")
    install_fake_load_dataset(monkeypatch, FakeLoadDataset(fixture_rows()))
    target = workspace.root / "fetched"
    result = invoke(
        [
            "eval",
            "suite",
            "fetch",
            "--benchmark",
            "flores_plus",
            "--split",
            "devtest",
            "--out-dir",
            str(target),
            "--revision",
            REVISION,
            "--config",
            str(config),
        ]
    )
    assert result.exit_code == 0, result.output
    return target, config


def build_from(
    workspace: SuiteWorkspace, config: Path, source: Path, *extra: str
) -> Any:
    return invoke(workspace.build_args("--config", str(config), *extra, source=source))


# ---- no fetch manifest: the line ------------------------------------------------------------------


def test_loose_shards_build_and_say_that_nothing_vouches_for_their_provenance(
    workspace: SuiteWorkspace,
) -> None:
    result = invoke(workspace.build_args())

    assert result.exit_code == 0, result.output
    assert result.stdout.splitlines()[-1] == f"provenance: {NOTE}"
    assert result.stdout.splitlines()[:3] == [
        f"suite: {workspace.suite}",
        f"sha256: {sha256_of(workspace.suite)}",
        f"manifest: {workspace.manifest}",
    ]


def test_the_build_manifest_records_that_no_fetch_manifest_was_found(
    workspace: SuiteWorkspace,
) -> None:
    invoke(workspace.build_args())

    assert read_manifest(workspace.out_dir)["status"]["provenance"] == NOTE


def test_the_exit_code_for_hand_made_shards_is_unchanged(
    workspace: SuiteWorkspace,
) -> None:
    result = invoke(workspace.build_args())

    assert result.exit_code == 0
    suite = Suite.from_bytes(workspace.suite.read_bytes())
    assert (suite.source.dataset, suite.source.revision, suite.source.pinned) == (
        None,
        None,
        False,
    )


# ---- the whole fetch directory -------------------------------------------------------------------


def test_a_build_from_the_fetch_directory_is_bound_to_the_fetch_and_prints_no_note(
    workspace: SuiteWorkspace, fetched: tuple[Path, Path]
) -> None:
    target, config = fetched

    result = build_from(workspace, config, target)

    assert result.exit_code == 0, result.output
    assert "provenance:" not in result.stdout
    assert "provenance" not in read_manifest(workspace.out_dir)["status"]
    suite = Suite.from_bytes(workspace.suite.read_bytes())
    assert (suite.source.revision, suite.source.pinned) == (REVISION, True)


# ---- the split directory of a fetch -----------------------------------------------------------------


def test_a_build_from_the_split_directory_uses_the_manifest_of_the_fetch_above_it(
    workspace: SuiteWorkspace, fetched: tuple[Path, Path]
) -> None:
    target, config = fetched

    result = build_from(workspace, config, target / "devtest")

    assert result.exit_code == 0, result.output
    assert "provenance:" not in result.stdout
    suite = Suite.from_bytes(workspace.suite.read_bytes())
    assert suite.source.dataset == "openlanguagedata/flores_plus"
    assert (suite.source.revision, suite.source.pinned) == (REVISION, True)
    manifest = read_manifest(workspace.out_dir)
    assert (manifest["status"]["revision"], manifest["status"]["pinned"]) == (
        REVISION,
        "true",
    )
    assert "provenance" not in manifest["status"]
    assert str(target / MANIFEST_FILE) in {item["path"] for item in manifest["inputs"]}


def test_the_suite_from_the_split_directory_is_the_suite_from_the_fetch_directory(
    workspace: SuiteWorkspace, fetched: tuple[Path, Path], tmp_path: Path
) -> None:
    target, config = fetched
    build_from(workspace, config, target)
    whole = workspace.suite.read_bytes()

    again = tmp_path / "again"
    result = invoke(
        workspace.build_args(
            "--config", str(config), source=target / "devtest", out_dir=again
        )
    )

    assert result.exit_code == 0, result.output
    assert (again / workspace.suite.name).read_bytes() == whole


def test_a_shard_changed_after_the_fetch_is_refused_when_the_split_directory_is_given(
    workspace: SuiteWorkspace, fetched: tuple[Path, Path]
) -> None:
    target, config = fetched
    shard = target / "devtest" / "eng_Latn" / SHARD
    shard.write_bytes(shard.read_bytes() + b"x")

    result = build_from(workspace, config, target / "devtest")

    assert result.exit_code == 3
    assert "changed after the fetch" in result.stderr
    assert not workspace.out_dir.exists()


def test_a_shard_the_fetch_listed_and_the_split_directory_lacks_is_refused(
    workspace: SuiteWorkspace, fetched: tuple[Path, Path]
) -> None:
    target, config = fetched
    shutil.rmtree(target / "devtest" / "rus_Cyrl")

    result = build_from(workspace, config, target / "devtest")

    assert result.exit_code == 3
    assert "is not in the source" in result.stderr


def test_a_manifest_above_the_split_directory_that_is_not_a_fetch_is_not_used(
    workspace: SuiteWorkspace, fetched: tuple[Path, Path]
) -> None:
    target, config = fetched
    (target / MANIFEST_FILE).unlink()
    write_foreign_manifest(target)

    result = build_from(workspace, config, target / "devtest")

    assert result.exit_code == 0, result.output
    assert result.stdout.splitlines()[-1] == f"provenance: {NOTE}"
    assert read_manifest(workspace.out_dir)["status"]["provenance"] == NOTE


def test_a_split_directory_with_no_manifest_above_it_says_so(
    workspace: SuiteWorkspace, fetched: tuple[Path, Path]
) -> None:
    target, config = fetched
    (target / MANIFEST_FILE).unlink()

    result = build_from(workspace, config, target / "devtest")

    assert result.exit_code == 0, result.output
    assert result.stdout.splitlines()[-1] == f"provenance: {NOTE}"


def test_a_directory_not_named_for_the_split_does_not_look_above_itself(
    workspace: SuiteWorkspace, fetched: tuple[Path, Path], tmp_path: Path
) -> None:
    # The shards moved to a directory of another name under the fetch: nothing says they are the
    # fetch's, so the manifest above is not applied to them.
    target, config = fetched
    moved = target / "moved"
    shutil.copytree(target / "devtest", moved)

    result = build_from(workspace, config, moved)

    assert result.exit_code == 0, result.output
    assert result.stdout.splitlines()[-1] == f"provenance: {NOTE}"


# ---- the sources that have no fetch -------------------------------------------------------------------


def test_a_directory_of_text_files_prints_no_provenance_line(tmp_path: Path) -> None:
    source = tmp_path / "lines"
    source.mkdir()
    for code, text in (("eng_Latn", "one"), ("khk_Cyrl", "нэг"), ("rus_Cyrl", "один")):
        (source / f"{code}.txt").write_text(f"{text}\n{text}\n", encoding="utf-8")
    out = tmp_path / "out"

    result = invoke(
        [
            "eval",
            "suite",
            "build",
            "--benchmark",
            "ntrex",
            "--split",
            "devtest",
            "--from",
            str(source),
            "--out-dir",
            str(out),
        ]
    )

    assert result.exit_code == 0, result.output
    assert "provenance:" not in result.stdout
