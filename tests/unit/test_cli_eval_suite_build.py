"""`rinchen eval suite build` over the shards of a multi-way parallel benchmark (D13, Task 4): the
suite file, the manifest that vouches for it and the order in which the command hashes, reads and
writes. The shards are written by the real `rinchen.acquire.hf.write_shards`, as the fetch writes
them; the fetch itself is run for the tests of the cross-check against `datasets.load_dataset`
replaced by a recording fake (`tests.support.eval_suite_cli`). What the command refuses is in
`test_cli_eval_suite_build_refusals`, the other readers (a directory of text files, an export of a
human set) in `test_cli_eval_suite_build_readers`."""

from __future__ import annotations

import hashlib
import importlib.metadata
import shutil
from pathlib import Path
from typing import Any

import pytest

from rinchen.cli import eval_suite, eval_suite_files
from rinchen.config.eval_suites import DEFAULT_EVAL_SUITES_PATH
from rinchen.eval.suites import Suite
from tests.support.distill_cli import write_foreign_manifest
from tests.support.eval_suite_cli import (
    DOCUMENT_URLS,
    LANGUAGES,
    MANIFEST_FILE,
    REVISION,
    SHARD,
    SUITE_FILE,
    FakeLoadDataset,
    SuiteWorkspace,
    fixture_rows,
    install_fake_load_dataset,
    invoke,
    make_workspace,
    read_manifest,
    sha256_of,
    shard_rows,
    write_config,
    write_language_shards,
)
from tests.support.no_network import no_network  # noqa: F401 - the fixture of the mark
from tests.support.robust_cli import record_replaces

pytestmark = pytest.mark.usefixtures("no_network")

SIX_OTHERS = ["deu_Latn", "jpn_Jpan", "kaz_Cyrl", "kor_Hang", "tur_Latn", "zho_Hans"]


@pytest.fixture
def workspace(tmp_path: Path) -> SuiteWorkspace:
    return make_workspace(tmp_path)


def built(workspace: SuiteWorkspace, *extra: str, **options: Any) -> Suite:
    result = invoke(workspace.build_args(*extra, **options))
    assert result.exit_code == 0, result.output
    return Suite.from_bytes(workspace.suite.read_bytes())


# ---- the suite ----------------------------------------------------------------------------


def test_a_build_writes_the_suite_the_shards_make(workspace: SuiteWorkspace) -> None:
    suite = built(workspace)

    assert (suite.name, suite.kind, suite.benchmark, suite.split) == (
        "flores_plus-devtest",
        "benchmark",
        "flores_plus",
        "devtest",
    )
    assert suite.directions == (
        "khk_Cyrl-eng_Latn",
        "eng_Latn-khk_Cyrl",
        "khk_Cyrl-rus_Cyrl",
        "rus_Cyrl-khk_Cyrl",
    )
    assert [row.id for row in suite.rows] == ["s1", "s2", "s3", "s4"]
    assert dict(suite.rows[0].text) == {
        "eng_Latn": "the cat sat on the mat",
        "khk_Cyrl": "муур дэвсгэр дээр сууж байв",
        "rus_Cyrl": "кошка сидела на коврике",
    }
    assert suite.source.missing_languages == tuple(SIX_OTHERS)


def test_the_addresses_of_the_url_column_become_document_ids_that_tell_the_documents_apart(
    workspace: SuiteWorkspace,
) -> None:
    suite = built(workspace)

    docs = [row.doc for row in suite.rows]
    assert docs[0] == docs[1]
    assert docs[2] == docs[3]
    assert docs[0] != docs[2]
    assert all(doc is not None and "/" not in doc for doc in docs)
    assert not any(url in str(docs) for url in DOCUMENT_URLS)


def test_the_documents_are_the_same_in_two_builds(tmp_path: Path) -> None:
    first = make_workspace(tmp_path / "one")
    second = make_workspace(tmp_path / "two")

    assert built(first).to_bytes() == built(second).to_bytes()


def test_a_document_id_that_is_already_an_id_is_kept(tmp_path: Path) -> None:
    root = tmp_path
    for language in LANGUAGES:
        rows = shard_rows(language)
        for row, doc in zip(rows, ("doc-1", "doc-1", "doc-2", "doc-2"), strict=True):
            row["url"] = doc
        write_language_shards(root / "shards", language, rows)
    workspace = SuiteWorkspace(root, root / "shards", root / "out")

    assert [row.doc for row in built(workspace).rows] == [
        "doc-1",
        "doc-1",
        "doc-2",
        "doc-2",
    ]


def test_an_integer_id_column_gives_the_ids_as_text(tmp_path: Path) -> None:
    for language in LANGUAGES:
        write_language_shards(
            tmp_path / "shards", language, shard_rows(language, ids=(1, 2, 3, 4))
        )
    workspace = SuiteWorkspace(tmp_path, tmp_path / "shards", tmp_path / "out")

    assert [row.id for row in built(workspace).rows] == ["1", "2", "3", "4"]


def test_the_shards_of_a_language_are_read_in_the_order_of_their_names(
    tmp_path: Path,
) -> None:
    for language in LANGUAGES:
        write_language_shards(
            tmp_path / "shards", language, shard_rows(language), shard_size=3
        )
    workspace = SuiteWorkspace(tmp_path, tmp_path / "shards", tmp_path / "out")

    suite = built(workspace)

    assert [row.id for row in suite.rows] == ["s1", "s2", "s3", "s4"]
    assert sorted(suite.source.files) == [
        f"{language}/shard-0000{index}.parquet"
        for language in LANGUAGES
        for index in (0, 1)
    ]


def test_the_source_of_a_build_from_loose_shards_is_unpinned_and_names_every_shard_with_its_digest(
    workspace: SuiteWorkspace,
) -> None:
    suite = built(workspace)

    assert (suite.source.dataset, suite.source.revision, suite.source.pinned) == (
        None,
        None,
        False,
    )
    assert dict(suite.source.files) == {
        f"{language}/{SHARD}": sha256_of(workspace.shard(language))
        for language in LANGUAGES
    }


def test_the_kind_of_a_fixture_and_a_name_are_taken_from_the_options(
    workspace: SuiteWorkspace,
) -> None:
    result = invoke(
        workspace.build_args("--kind", "fixture", "--name", "my-suite", split="dev")
    )

    assert result.exit_code == 0, result.output
    suite = Suite.from_bytes((workspace.out_dir / "my-suite.json").read_bytes())
    assert (suite.name, suite.kind, suite.split, suite.purpose) == (
        "my-suite",
        "fixture",
        "dev",
        "selection",
    )


def test_two_builds_of_the_same_shards_give_the_same_suite_bytes(
    tmp_path: Path,
) -> None:
    one = make_workspace(tmp_path / "a")
    two = make_workspace(tmp_path / "b")
    for workspace in (one, two):
        assert invoke(workspace.build_args()).exit_code == 0

    assert one.suite.read_bytes() == two.suite.read_bytes()


def test_the_command_prints_the_suite_its_digest_and_the_manifest(
    workspace: SuiteWorkspace,
) -> None:
    result = invoke(workspace.build_args())

    assert result.exit_code == 0, result.output
    assert result.stdout.splitlines() == [
        f"suite: {workspace.suite}",
        f"sha256: {sha256_of(workspace.suite)}",
        f"manifest: {workspace.manifest}",
        "provenance: no fetch manifest found, dataset and revision are not recorded",
    ]
    assert result.stderr == ""


# ---- the manifest -------------------------------------------------------------------------


def test_the_manifest_vouches_for_the_inputs_and_the_suite(
    workspace: SuiteWorkspace,
) -> None:
    built(workspace)
    manifest = read_manifest(workspace.out_dir)

    assert manifest["stage"] == "eval_suite"
    assert manifest["seed"] == -1
    assert {item["path"]: item["sha256"] for item in manifest["inputs"]} == {
        str(DEFAULT_EVAL_SUITES_PATH): sha256_of(DEFAULT_EVAL_SUITES_PATH),
        **{
            str(workspace.shard(language)): sha256_of(workspace.shard(language))
            for language in LANGUAGES
        },
    }
    assert [
        (item["path"], item["sha256"], item["rows"]) for item in manifest["outputs"]
    ] == [(str(workspace.suite), sha256_of(workspace.suite), 4)]
    assert manifest["outputs"][0]["bytes"] == workspace.suite.stat().st_size


def test_the_manifest_records_the_rows_of_each_language_and_the_directions(
    workspace: SuiteWorkspace,
) -> None:
    built(workspace)

    assert read_manifest(workspace.out_dir)["metrics"] == {
        "rows.eng_Latn": 4.0,
        "rows.khk_Cyrl": 4.0,
        "rows.rus_Cyrl": 4.0,
        "directions": 4.0,
    }


def test_the_manifest_records_what_the_suite_is(workspace: SuiteWorkspace) -> None:
    built(workspace)
    manifest = read_manifest(workspace.out_dir)

    assert manifest["status"] == {
        "benchmark": "flores_plus",
        "split": "devtest",
        "kind": "benchmark",
        "revision": "null",
        "pinned": "false",
        "provenance": "no fetch manifest found, dataset and revision are not recorded",
    }
    assert manifest["tool_versions"] == {"polars": importlib.metadata.version("polars")}
    assert len(manifest["config_sha256"]) == 64


def test_the_digest_of_the_manifest_config_follows_the_options_and_the_configuration_file(
    workspace: SuiteWorkspace, tmp_path: Path
) -> None:
    built(workspace)
    first = read_manifest(workspace.out_dir)["config_sha256"]
    built(workspace)
    assert read_manifest(workspace.out_dir)["config_sha256"] == first

    built(workspace, "--kind", "fixture")
    assert read_manifest(workspace.out_dir)["config_sha256"] != first

    config = write_config(tmp_path / "other.yaml")
    built(workspace, "--config", str(config))
    assert read_manifest(workspace.out_dir)["config_sha256"] != first


def test_the_manifest_is_the_last_file_written(
    workspace: SuiteWorkspace, monkeypatch: pytest.MonkeyPatch
) -> None:
    names = record_replaces(monkeypatch)

    assert invoke(workspace.build_args()).exit_code == 0

    assert names == [SUITE_FILE, MANIFEST_FILE]


def test_a_configuration_given_by_the_option_is_the_one_that_is_read(
    tmp_path: Path,
) -> None:
    workspace = make_workspace(tmp_path, ("eng_Latn", "khk_Cyrl"))
    config = write_config(tmp_path / "mine.yaml", languages=("khk_Cyrl", "eng_Latn"))

    suite = built(workspace, "--config", str(config))

    assert suite.source.missing_languages == ()
    assert suite.directions == ("khk_Cyrl-eng_Latn", "eng_Latn-khk_Cyrl")
    manifest = read_manifest(workspace.out_dir)
    assert str(config) in {item["path"] for item in manifest["inputs"]}


# ---- hash before read ---------------------------------------------------------------------


def test_every_shard_is_hashed_before_any_shard_is_parsed(
    workspace: SuiteWorkspace, monkeypatch: pytest.MonkeyPatch
) -> None:
    events: list[str] = []
    real_hash = eval_suite.hash_input
    real_read = eval_suite_files.pl.read_parquet

    def hashing(path: Path, **options: Any) -> Any:
        events.append(f"hash {path.name}")
        return real_hash(path, **options)

    def reading(*args: Any, **options: Any) -> Any:
        events.append("parse")
        return real_read(*args, **options)

    monkeypatch.setattr(eval_suite, "hash_input", hashing)
    monkeypatch.setattr(eval_suite_files.pl, "read_parquet", reading)

    assert invoke(workspace.build_args()).exit_code == 0

    assert events.count("parse") == 3
    assert events.index("parse") > max(
        index for index, event in enumerate(events) if event == f"hash {SHARD}"
    )


def test_a_shard_that_changed_after_it_was_hashed_is_refused_and_nothing_is_written(
    workspace: SuiteWorkspace, monkeypatch: pytest.MonkeyPatch
) -> None:
    real_hash = eval_suite.hash_input

    def hashing(path: Path, **options: Any) -> Any:
        entry = real_hash(path, **options)
        if path == workspace.shard("eng_Latn"):
            write_language_shards(
                workspace.shards, "eng_Latn", shard_rows("eng_Latn", texts=("x",) * 4)
            )
        return entry

    monkeypatch.setattr(eval_suite, "hash_input", hashing)

    result = invoke(workspace.build_args())

    assert result.exit_code == 2, result.output
    assert "changed after it was hashed" in result.stderr
    assert not workspace.out_dir.exists()


# ---- the cross-check with the manifest of a fetch -----------------------------------------


@pytest.fixture
def fetched(
    workspace: SuiteWorkspace, monkeypatch: pytest.MonkeyPatch
) -> tuple[Path, Path]:
    """A directory a real `fetch` wrote (over the fake `load_dataset`) for the three languages
    and the configuration it used."""
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


def build_from_fetch(
    workspace: SuiteWorkspace, fetched: tuple[Path, Path], *extra: str
) -> Any:
    target, config = fetched
    return invoke(
        workspace.build_args(
            "--config", str(config), *extra, source=target, split="devtest"
        )
    )


def test_a_build_from_a_fetch_records_its_revision_and_dataset_and_is_pinned(
    workspace: SuiteWorkspace, fetched: tuple[Path, Path]
) -> None:
    result = build_from_fetch(workspace, fetched)

    assert result.exit_code == 0, result.output
    suite = Suite.from_bytes(workspace.suite.read_bytes())
    assert suite.source.revision == REVISION
    assert suite.source.pinned is True
    assert suite.source.dataset == "openlanguagedata/flores_plus"
    status = read_manifest(workspace.out_dir)["status"]
    assert (status["revision"], status["pinned"]) == (REVISION, "true")


def test_the_manifest_of_the_fetch_is_an_input_of_the_build(
    workspace: SuiteWorkspace, fetched: tuple[Path, Path]
) -> None:
    target, _ = fetched
    build_from_fetch(workspace, fetched)

    inputs = {
        item["path"]: item["sha256"]
        for item in read_manifest(workspace.out_dir)["inputs"]
    }

    assert inputs[str(target / MANIFEST_FILE)] == sha256_of(target / MANIFEST_FILE)
    assert str(target / "devtest" / "eng_Latn" / SHARD) in inputs


def test_a_fetch_without_a_pin_gives_a_suite_that_says_so(
    workspace: SuiteWorkspace, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = write_config(workspace.root / "suites.yaml")
    install_fake_load_dataset(monkeypatch, FakeLoadDataset(fixture_rows()))
    target = workspace.root / "fetched"
    fetch = invoke(
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
            "--unpinned",
            "--config",
            str(config),
        ]
    )
    assert fetch.exit_code == 0, fetch.output

    result = build_from_fetch(workspace, (target, config))

    assert result.exit_code == 0, result.output
    suite = Suite.from_bytes(workspace.suite.read_bytes())
    assert (suite.source.revision, suite.source.pinned) == (None, False)
    assert suite.source.dataset == "openlanguagedata/flores_plus"


def test_a_shard_changed_after_the_fetch_is_refused_and_nothing_is_written(
    workspace: SuiteWorkspace, fetched: tuple[Path, Path]
) -> None:
    target, _ = fetched
    write_language_shards(
        target / "devtest", "eng_Latn", shard_rows("eng_Latn", texts=("x",) * 4)
    )

    result = build_from_fetch(workspace, fetched)

    assert result.exit_code == 3, result.output
    assert "eng_Latn" in result.stderr
    assert "changed after the fetch" in result.stderr
    assert not workspace.out_dir.exists()


def test_a_shard_the_fetch_manifest_does_not_list_is_refused(
    workspace: SuiteWorkspace, fetched: tuple[Path, Path]
) -> None:
    target, _ = fetched
    shard = target / "devtest" / "eng_Latn" / SHARD
    shutil.copyfile(shard, shard.with_name("shard-00001.parquet"))

    result = build_from_fetch(workspace, fetched)

    assert result.exit_code == 3, result.output
    assert "does not list" in result.stderr
    assert "eng_Latn/shard-00001.parquet" in result.stderr
    assert not workspace.out_dir.exists()


def test_a_shard_the_fetch_manifest_lists_and_is_missing_is_refused(
    workspace: SuiteWorkspace, fetched: tuple[Path, Path]
) -> None:
    target, _ = fetched
    shutil.rmtree(target / "devtest" / "rus_Cyrl")

    result = build_from_fetch(workspace, fetched)

    assert result.exit_code == 3, result.output
    assert "rus_Cyrl" in result.stderr
    assert "lists is not in the source" in result.stderr
    assert not workspace.out_dir.exists()


def test_the_manifest_of_another_stage_in_the_source_directory_is_refused(
    workspace: SuiteWorkspace,
) -> None:
    write_foreign_manifest(workspace.shards, "export")

    result = invoke(workspace.build_args())

    assert result.exit_code == 3, result.output
    assert "not the one of a suite fetch" in result.stderr
    assert not workspace.out_dir.exists()


def test_a_manifest_that_says_what_a_fetch_says_but_is_of_another_stage_is_refused(
    workspace: SuiteWorkspace, fetched: tuple[Path, Path]
) -> None:
    target, _ = fetched
    manifest = target / MANIFEST_FILE
    manifest.write_text(
        manifest.read_text(encoding="utf-8").replace(
            '"stage": "eval_suite_fetch"', '"stage": "eval_suite"'
        ),
        encoding="utf-8",
    )

    result = build_from_fetch(workspace, fetched)

    assert result.exit_code == 3, result.output
    assert "not the one of a suite fetch" in result.stderr


def test_a_manifest_that_is_not_one_is_refused(
    workspace: SuiteWorkspace, fetched: tuple[Path, Path]
) -> None:
    target, _ = fetched
    (target / MANIFEST_FILE).write_text("not a manifest", encoding="utf-8")

    result = build_from_fetch(workspace, fetched)

    assert result.exit_code == 3, result.output
    assert "cannot be read" in result.stderr


def test_a_fetch_of_another_split_is_refused(
    workspace: SuiteWorkspace, fetched: tuple[Path, Path]
) -> None:
    target, config = fetched

    result = invoke(
        workspace.build_args("--config", str(config), source=target, split="dev")
    )

    assert result.exit_code == 3, result.output
    assert "split" in result.stderr
    assert not workspace.out_dir.exists()


def test_a_fetch_of_another_benchmark_is_refused(
    workspace: SuiteWorkspace, fetched: tuple[Path, Path]
) -> None:
    target, config = fetched

    result = invoke(
        workspace.build_args(
            "--config", str(config), source=target, benchmark="flores_copy"
        )
    )

    assert result.exit_code == 3, result.output
    assert "benchmark" in result.stderr
    assert not workspace.out_dir.exists()


def test_the_digest_of_a_shard_is_compared_for_the_files_under_the_directory_it_is_in_now(
    workspace: SuiteWorkspace, fetched: tuple[Path, Path], tmp_path: Path
) -> None:
    # A fetch directory that was moved still checks out: the manifest's own paths are not read.
    target, config = fetched
    moved = tmp_path / "moved"
    target.rename(moved)

    result = invoke(
        workspace.build_args("--config", str(config), source=moved, split="devtest")
    )

    assert result.exit_code == 0, result.output
    assert Suite.from_bytes(workspace.suite.read_bytes()).source.revision == REVISION


def test_the_sha256_of_the_suite_is_the_one_the_manifest_lists(
    workspace: SuiteWorkspace,
) -> None:
    built(workspace)

    assert (
        read_manifest(workspace.out_dir)["outputs"][0]["sha256"]
        == hashlib.sha256(workspace.suite.read_bytes()).hexdigest()
    )
