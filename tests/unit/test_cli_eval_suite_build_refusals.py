"""`rinchen eval suite build`: what it refuses and with which exit code (D13, Task 4). 2 for a bad
option or a configuration that is not usable, an output directory that is an input's and a file that
changed after it was hashed; 3 for an input that is missing, unreadable or not what the
configuration says (a language that is not one, the ids of the languages that do not agree, a
column that is not there) and another stage's manifest in the output directory; 1 when the output
cannot be written. A refusal writes nothing and leaves what an earlier build wrote as it was."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import polars as pl
import pytest

from rinchen.eval.suites import Suite
from tests.support.distill_cli import write_foreign_manifest
from tests.support.eval_suite_cli import (
    MANIFEST_FILE,
    SHARD,
    SUITE_FILE,
    SuiteWorkspace,
    invoke,
    make_workspace,
    shard_rows,
    write_language_shards,
)
from tests.support.launch_cli import assert_one_error, snapshot
from tests.support.no_network import no_network  # noqa: F401 - the fixture of the mark
from tests.support.robust_cli import fail_replace

pytestmark = pytest.mark.usefixtures("no_network")


@pytest.fixture
def workspace(tmp_path: Path) -> SuiteWorkspace:
    return make_workspace(tmp_path)


def assert_refused(
    workspace: SuiteWorkspace, result: Any, code: int, start: str
) -> None:
    """Refused with `code` in one line that starts with `start`, with standard output empty and
    nothing written."""
    assert_one_error(result, code, start)
    assert result.stdout == ""
    assert not workspace.out_dir.exists()


# ---- the options --------------------------------------------------------------------------


def test_an_unknown_benchmark_is_refused(workspace: SuiteWorkspace) -> None:
    result = invoke(workspace.build_args(benchmark="nothing"))

    assert_refused(workspace, result, 2, "the benchmark")


@pytest.mark.parametrize(
    ("benchmark", "split"),
    [("flores_plus", "test"), ("ntrex", "dev"), ("flores_plus", "train")],
)
def test_a_split_the_benchmark_does_not_have_is_refused(
    workspace: SuiteWorkspace, benchmark: str, split: str
) -> None:
    result = invoke(workspace.build_args(benchmark=benchmark, split=split))

    assert_refused(workspace, result, 2, "the split")


def test_a_build_without_a_split_is_refused(workspace: SuiteWorkspace) -> None:
    args = workspace.build_args()
    index = args.index("--split")
    del args[index : index + 2]

    assert_refused(workspace, invoke(args), 2, "--split")


def test_a_human_kind_is_for_the_human_benchmark_only(
    workspace: SuiteWorkspace,
) -> None:
    assert_refused(
        workspace, invoke(workspace.build_args("--kind", "human")), 2, "--kind"
    )


def test_a_kind_that_does_not_exist_is_a_usage_error(workspace: SuiteWorkspace) -> None:
    result = invoke(workspace.build_args("--kind", "gold"))

    assert result.exit_code == 2
    assert not workspace.out_dir.exists()


@pytest.mark.parametrize("name", ["a b", "", "x" * 65, "a/b"])
def test_a_name_that_is_not_an_id_is_refused(
    workspace: SuiteWorkspace, name: str
) -> None:
    assert_refused(workspace, invoke(workspace.build_args("--name", name)), 2, "--name")


def test_a_configuration_that_is_missing_is_refused(
    workspace: SuiteWorkspace, tmp_path: Path
) -> None:
    result = invoke(workspace.build_args("--config", str(tmp_path / "none.yaml")))

    assert_refused(workspace, result, 2, "eval suites config not found")


def test_a_configuration_that_breaks_the_schema_is_refused(
    workspace: SuiteWorkspace, tmp_path: Path
) -> None:
    config = tmp_path / "bad.yaml"
    config.write_text("benchmarks: {}\n", encoding="utf-8")

    result = invoke(workspace.build_args("--config", str(config)))

    assert_refused(workspace, result, 2, "invalid eval suites config")


# ---- the inputs ---------------------------------------------------------------------------


def test_a_source_directory_that_does_not_exist_is_refused(
    workspace: SuiteWorkspace, tmp_path: Path
) -> None:
    result = invoke(workspace.build_args(source=tmp_path / "nothing"))

    assert_refused(workspace, result, 3, "")


def test_a_source_that_is_a_file_and_not_a_directory_is_refused(
    workspace: SuiteWorkspace,
) -> None:
    result = invoke(workspace.build_args(source=workspace.shard("eng_Latn")))

    assert_refused(workspace, result, 3, "")


def test_inner_mongolian_as_a_language_directory_is_refused_by_name(
    workspace: SuiteWorkspace,
) -> None:
    write_language_shards(workspace.shards, "khk_Mong", shard_rows("khk_Cyrl"))

    result = invoke(workspace.build_args())

    assert_refused(workspace, result, 3, "")
    assert "khk_Mong is refused by name" in result.stderr


def test_a_directory_named_like_a_code_outside_the_registry_is_refused(
    workspace: SuiteWorkspace,
) -> None:
    write_language_shards(workspace.shards, "xx_Latn", shard_rows("khk_Cyrl"))

    result = invoke(workspace.build_args())

    assert_refused(workspace, result, 3, "")
    assert "xx_Latn" in result.stderr
    assert "registry" in result.stderr


def test_a_registry_language_the_benchmark_does_not_look_for_is_refused(
    workspace: SuiteWorkspace,
) -> None:
    write_language_shards(workspace.shards, "mn_Mong", shard_rows("khk_Cyrl"))

    result = invoke(workspace.build_args())

    assert_refused(workspace, result, 3, "")
    assert "mn_Mong" in result.stderr


def test_a_directory_that_is_not_a_language_at_all_is_refused_with_its_path(
    workspace: SuiteWorkspace,
) -> None:
    (workspace.shards / "notes").mkdir()

    result = invoke(workspace.build_args())

    assert_refused(workspace, result, 3, "")
    assert str(workspace.shards / "notes") in result.stderr


def test_a_missing_hub_language_is_refused(tmp_path: Path) -> None:
    workspace = make_workspace(tmp_path, ("eng_Latn", "rus_Cyrl"))

    result = invoke(workspace.build_args())

    assert_refused(workspace, result, 3, "")
    assert "khk_Cyrl" in result.stderr


def test_a_language_directory_without_a_shard_is_refused(
    workspace: SuiteWorkspace,
) -> None:
    (workspace.shards / "rus_Cyrl" / SHARD).unlink()

    result = invoke(workspace.build_args())

    assert_refused(workspace, result, 3, "")
    assert "rus_Cyrl" in result.stderr


def test_a_shard_that_is_not_a_parquet_file_is_refused(
    workspace: SuiteWorkspace,
) -> None:
    workspace.shard("eng_Latn").write_bytes(b"not parquet at all")

    result = invoke(workspace.build_args())

    assert_refused(workspace, result, 3, "")
    assert "eng_Latn" in result.stderr


def test_a_shard_that_is_not_a_file_is_refused(workspace: SuiteWorkspace) -> None:
    (workspace.shards / "eng_Latn" / "shard-00001.parquet").mkdir()

    result = invoke(workspace.build_args())

    assert_refused(workspace, result, 3, "")


@pytest.mark.parametrize("column", ["id", "text"])
def test_a_column_the_configuration_names_and_the_shard_lacks_is_named(
    workspace: SuiteWorkspace, column: str
) -> None:
    frame = pl.read_parquet(workspace.shard("eng_Latn")).drop(column)
    frame.write_parquet(workspace.shard("eng_Latn"))

    result = invoke(workspace.build_args())

    assert_refused(workspace, result, 3, "")
    assert f"'{column}'" in result.stderr
    assert "eng_Latn" in result.stderr


def test_a_document_column_that_is_not_there_is_named_too(
    workspace: SuiteWorkspace,
) -> None:
    pl.read_parquet(workspace.shard("rus_Cyrl")).drop("url").write_parquet(
        workspace.shard("rus_Cyrl")
    )

    result = invoke(workspace.build_args())

    assert_refused(workspace, result, 3, "")
    assert "'url'" in result.stderr


@pytest.mark.parametrize("column", ["id", "text"])
def test_a_null_in_a_required_column_is_refused_naming_the_column_and_the_row(
    workspace: SuiteWorkspace, column: str
) -> None:
    rows = shard_rows("eng_Latn")
    rows[2][column] = None
    write_language_shards(workspace.shards, "eng_Latn", rows)

    result = invoke(workspace.build_args())

    assert_refused(workspace, result, 3, "")
    assert f"'{column}'" in result.stderr
    assert "row 2" in result.stderr


def test_an_id_column_that_is_neither_text_nor_a_whole_number_is_refused(
    workspace: SuiteWorkspace,
) -> None:
    write_language_shards(
        workspace.shards, "eng_Latn", shard_rows("eng_Latn", ids=(1.5, 2.5, 3.5, 4.5))
    )

    result = invoke(workspace.build_args())

    assert_refused(workspace, result, 3, "")
    assert "'id'" in result.stderr


def test_an_id_column_of_booleans_is_refused(workspace: SuiteWorkspace) -> None:
    write_language_shards(
        workspace.shards,
        "eng_Latn",
        shard_rows("eng_Latn", ids=(True, False, True, False)),
    )

    result = invoke(workspace.build_args())

    assert_refused(workspace, result, 3, "")
    assert "'id'" in result.stderr


def test_a_text_column_that_is_not_text_is_refused(workspace: SuiteWorkspace) -> None:
    rows = shard_rows("eng_Latn")
    for row in rows:
        row["text"] = 7
    write_language_shards(workspace.shards, "eng_Latn", rows)

    result = invoke(workspace.build_args())

    assert_refused(workspace, result, 3, "")
    assert "'text'" in result.stderr


def test_a_language_with_fewer_ids_is_refused_with_the_counts_and_the_suite_is_not_made(
    workspace: SuiteWorkspace,
) -> None:
    write_language_shards(
        workspace.shards,
        "eng_Latn",
        shard_rows("eng_Latn")[:3],
    )

    result = invoke(workspace.build_args())

    assert_refused(workspace, result, 3, "")
    assert "eng_Latn" in result.stderr
    assert "3" in result.stderr and "4" in result.stderr


def test_a_sentence_the_suite_refuses_is_named_by_row_and_field(
    workspace: SuiteWorkspace,
) -> None:
    rows = shard_rows("eng_Latn")
    rows[1]["text"] = "   "
    write_language_shards(workspace.shards, "eng_Latn", rows)

    result = invoke(workspace.build_args())

    assert_refused(workspace, result, 3, "not an evaluation suite")
    assert "row index 1" in result.stderr


# ---- the output ---------------------------------------------------------------------------


def test_an_output_directory_that_is_the_directory_of_an_input_is_refused(
    workspace: SuiteWorkspace,
) -> None:
    before = snapshot(workspace.shards)

    result = invoke(workspace.build_args(out_dir=workspace.shards / "eng_Latn"))

    assert_one_error(result, 2, "--out-dir")
    assert snapshot(workspace.shards) == before


def test_the_manifest_of_another_stage_in_the_output_directory_is_refused_and_left_alone(
    workspace: SuiteWorkspace,
) -> None:
    foreign = write_foreign_manifest(workspace.out_dir)
    before = snapshot(workspace.out_dir)

    result = invoke(workspace.build_args())

    assert_one_error(result, 3, "")
    assert "refusing to erase" in result.stderr
    assert snapshot(workspace.out_dir) == before
    assert (workspace.out_dir / MANIFEST_FILE).read_bytes() == foreign


def test_a_suite_that_cannot_be_written_is_exit_1_and_leaves_no_manifest(
    workspace: SuiteWorkspace, monkeypatch: pytest.MonkeyPatch
) -> None:
    fail_replace(monkeypatch, SUITE_FILE)

    result = invoke(workspace.build_args())

    assert_one_error(result, 1, "cannot write the")
    assert not (workspace.out_dir / MANIFEST_FILE).exists()


def test_a_manifest_that_cannot_be_written_is_exit_1_and_the_suite_stays_unvouched(
    workspace: SuiteWorkspace, monkeypatch: pytest.MonkeyPatch
) -> None:
    fail_replace(monkeypatch, MANIFEST_FILE)

    result = invoke(workspace.build_args())

    assert_one_error(result, 1, "cannot write the")
    assert not (workspace.out_dir / MANIFEST_FILE).exists()


def test_an_output_directory_that_cannot_be_made_is_exit_1(
    workspace: SuiteWorkspace, tmp_path: Path
) -> None:
    blocker = tmp_path / "blocker"
    blocker.write_text("a file where a directory should be", encoding="utf-8")

    result = invoke(workspace.build_args(out_dir=blocker / "out"))

    assert_one_error(result, 1, "cannot write the")


# ---- what an earlier build left -----------------------------------------------------------


def test_a_refused_build_leaves_what_an_earlier_build_wrote_as_it_was(
    workspace: SuiteWorkspace,
) -> None:
    assert invoke(workspace.build_args()).exit_code == 0
    before = snapshot(workspace.out_dir)
    write_language_shards(workspace.shards, "khk_Mong", shard_rows("khk_Cyrl"))

    result = invoke(workspace.build_args())

    assert result.exit_code == 3
    assert snapshot(workspace.out_dir) == before


def test_a_new_build_removes_the_suite_the_old_manifest_lists_and_nothing_else(
    workspace: SuiteWorkspace,
) -> None:
    assert invoke(workspace.build_args("--name", "first")).exit_code == 0
    stray = workspace.out_dir / "stray.json"
    stray.write_text("{}", encoding="utf-8")
    unlisted = workspace.out_dir / "second.txt"
    unlisted.write_text("kept", encoding="utf-8")

    assert invoke(workspace.build_args("--name", "second")).exit_code == 0

    assert not (workspace.out_dir / "first.json").exists()
    assert (
        Suite.from_bytes((workspace.out_dir / "second.json").read_bytes()).name
        == "second"
    )
    assert stray.read_text(encoding="utf-8") == "{}"
    assert unlisted.read_text(encoding="utf-8") == "kept"


def test_the_names_of_an_old_manifest_are_taken_inside_the_output_directory(
    workspace: SuiteWorkspace, tmp_path: Path
) -> None:
    assert invoke(workspace.build_args("--name", "first")).exit_code == 0
    outside = tmp_path / "outside.txt"
    outside.write_text("precious", encoding="utf-8")
    manifest_file = workspace.out_dir / MANIFEST_FILE
    text = manifest_file.read_text(encoding="utf-8")
    manifest_file.write_text(
        text.replace(str(workspace.out_dir / "first.json"), str(outside)),
        encoding="utf-8",
    )

    assert invoke(workspace.build_args("--name", "second")).exit_code == 0

    assert outside.read_text(encoding="utf-8") == "precious"


def test_a_build_into_a_copy_of_the_output_directory_works_after_it_was_moved(
    workspace: SuiteWorkspace, tmp_path: Path
) -> None:
    assert invoke(workspace.build_args("--name", "first")).exit_code == 0
    moved = tmp_path / "moved"
    shutil.copytree(workspace.out_dir, moved)

    result = invoke(workspace.build_args("--name", "second", out_dir=moved))

    assert result.exit_code == 0, result.output
    assert not (moved / "first.json").exists()
    assert (moved / "second.json").is_file()
