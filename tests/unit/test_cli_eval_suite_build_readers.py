"""`rinchen eval suite build` over the two other kinds of source (D13, Task 4): a directory of text
files with one sentence per line and a file per language (NTREX, or any multi-way parallel text),
and the export of a human set of D2's annotation app in JSON Lines, TSV or Parquet. What a build
over shards does is in `test_cli_eval_suite_build`."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from rinchen.config.eval_suites import DEFAULT_EVAL_SUITES_PATH
from rinchen.eval.suites import Suite
from tests.support.eval_suite_cli import (
    LANGUAGES,
    MANIFEST_FILE,
    SENTENCES,
    export_row,
    golden_export,
    invoke,
    read_manifest,
    sha256_of,
    write_export,
)
from tests.support.launch_cli import assert_one_error
from tests.support.no_network import no_network  # noqa: F401 - the fixture of the mark

pytestmark = pytest.mark.usefixtures("no_network")


def write_lines(root: Path, languages: tuple[str, ...] = LANGUAGES) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    for language in languages:
        (root / f"{language}.txt").write_text(
            "".join(f"{text}\n" for text in SENTENCES[language]), encoding="utf-8"
        )
    return root


def lines_args(
    source: Path, out_dir: Path, *extra: str, split: str = "devtest"
) -> list[str]:
    return [
        "eval",
        "suite",
        "build",
        "--benchmark",
        "ntrex",
        "--split",
        split,
        "--from",
        str(source),
        "--out-dir",
        str(out_dir),
        *extra,
    ]


def human_args(source: Path, out_dir: Path, *extra: str) -> list[str]:
    return [
        "eval",
        "suite",
        "build",
        "--benchmark",
        "human",
        "--from",
        str(source),
        "--out-dir",
        str(out_dir),
        *extra,
    ]


# ---- a directory of text files ------------------------------------------------------------


def test_a_directory_of_text_files_gives_a_suite_with_the_line_numbers_as_ids(
    tmp_path: Path,
) -> None:
    source = write_lines(tmp_path / "ntrex")
    out_dir = tmp_path / "out"

    result = invoke(lines_args(source, out_dir))

    assert result.exit_code == 0, result.output
    suite = Suite.from_bytes((out_dir / "ntrex-devtest.json").read_bytes())
    assert (suite.name, suite.kind, suite.benchmark, suite.split) == (
        "ntrex-devtest",
        "benchmark",
        "ntrex",
        "devtest",
    )
    assert [row.id for row in suite.rows] == [
        "l000001",
        "l000002",
        "l000003",
        "l000004",
    ]
    assert [row.doc for row in suite.rows] == [None] * 4
    assert suite.directions == (
        "khk_Cyrl-eng_Latn",
        "eng_Latn-khk_Cyrl",
        "khk_Cyrl-rus_Cyrl",
        "rus_Cyrl-khk_Cyrl",
    )
    assert suite.source.missing_languages == (
        "deu_Latn",
        "jpn_Jpan",
        "kaz_Cyrl",
        "kor_Hang",
        "tur_Latn",
        "zho_Hans",
    )
    assert dict(suite.source.files) == {
        f"{language}.txt": sha256_of(source / f"{language}.txt")
        for language in LANGUAGES
    }


def test_the_manifest_of_a_build_from_text_files_lists_each_file_once(
    tmp_path: Path,
) -> None:
    source = write_lines(tmp_path / "ntrex")
    out_dir = tmp_path / "out"
    assert invoke(lines_args(source, out_dir)).exit_code == 0

    manifest = read_manifest(out_dir)

    paths = [item["path"] for item in manifest["inputs"]]
    assert len(paths) == 4
    assert sorted(
        path for path in paths if path != str(DEFAULT_EVAL_SUITES_PATH)
    ) == sorted(str(source / f"{language}.txt") for language in LANGUAGES)
    assert manifest["status"]["benchmark"] == "ntrex"
    assert manifest["status"]["pinned"] == "false"


def test_files_that_are_not_text_files_of_a_language_are_left_alone(
    tmp_path: Path,
) -> None:
    source = write_lines(tmp_path / "ntrex")
    (source / "README.md").write_text("notes", encoding="utf-8")
    (source / MANIFEST_FILE).write_text("not looked at", encoding="utf-8")

    assert invoke(lines_args(source, tmp_path / "out")).exit_code == 0


@pytest.mark.parametrize("stem", ["khk_Mong", "xx_Latn", "mn_Mong", "notes"])
def test_a_text_file_of_a_language_outside_the_registry_or_the_benchmark_is_refused(
    tmp_path: Path, stem: str
) -> None:
    source = write_lines(tmp_path / "ntrex")
    (source / f"{stem}.txt").write_text("a\nb\nc\nd\n", encoding="utf-8")

    result = invoke(lines_args(source, tmp_path / "out"))

    assert_one_error(result, 3, "")
    if stem == "khk_Mong":
        assert "khk_Mong is refused by name" in result.stderr
    assert not (tmp_path / "out").exists()


def test_a_blank_line_is_refused_with_the_language_and_the_line(tmp_path: Path) -> None:
    source = write_lines(tmp_path / "ntrex")
    (source / "eng_Latn.txt").write_text("one\n\nthree\nfour\n", encoding="utf-8")

    result = invoke(lines_args(source, tmp_path / "out"))

    assert_one_error(result, 3, "")
    assert "eng_Latn" in result.stderr
    assert "line 2" in result.stderr
    assert "one" not in result.stderr
    assert not (tmp_path / "out").exists()


def test_a_file_with_windows_line_ends_is_refused(tmp_path: Path) -> None:
    source = write_lines(tmp_path / "ntrex")
    (source / "rus_Cyrl.txt").write_bytes(b"a\r\nb\r\nc\r\nd\r\n")

    result = invoke(lines_args(source, tmp_path / "out"))

    assert_one_error(result, 3, "")
    assert "line 1" in result.stderr


def test_files_of_different_lengths_are_refused_with_the_counts(tmp_path: Path) -> None:
    source = write_lines(tmp_path / "ntrex")
    (source / "eng_Latn.txt").write_text("one\ntwo\nthree\n", encoding="utf-8")

    result = invoke(lines_args(source, tmp_path / "out"))

    assert_one_error(result, 3, "")
    assert "eng_Latn" in result.stderr
    assert "3" in result.stderr and "4" in result.stderr


def test_a_directory_without_the_hub_language_is_refused(tmp_path: Path) -> None:
    source = write_lines(tmp_path / "ntrex", ("eng_Latn", "rus_Cyrl"))

    result = invoke(lines_args(source, tmp_path / "out"))

    assert_one_error(result, 3, "")
    assert "khk_Cyrl" in result.stderr


def test_a_directory_with_no_text_file_is_refused(tmp_path: Path) -> None:
    source = tmp_path / "empty"
    source.mkdir()

    assert_one_error(invoke(lines_args(source, tmp_path / "out")), 3, "")


def test_the_split_dev_is_not_a_split_of_the_text_benchmark(tmp_path: Path) -> None:
    source = write_lines(tmp_path / "ntrex")

    result = invoke(lines_args(source, tmp_path / "out", split="dev"))

    assert_one_error(result, 2, "the split")


# ---- the export of a human set ------------------------------------------------------------


@pytest.mark.parametrize("suffix", [".jsonl", ".tsv", ".parquet"])
def test_an_export_in_any_of_the_three_formats_gives_the_suite_of_a_human_set(
    tmp_path: Path, suffix: str
) -> None:
    export = write_export(tmp_path / f"export{suffix}", golden_export())
    out_dir = tmp_path / "out"

    result = invoke(human_args(export, out_dir))

    assert result.exit_code == 0, result.output
    suite = Suite.from_bytes((out_dir / "human-test.json").read_bytes())
    assert (suite.name, suite.kind, suite.benchmark, suite.split, suite.purpose) == (
        "human-test",
        "human",
        "human",
        "test",
        "report",
    )
    assert [row.id for row in suite.rows] == [
        "d1-00001",
        "d1-00002",
        "d1-00003",
        "d2-00001",
        "d2-00002",
        "d2-00003",
    ]
    assert suite.directions == ("eng_Latn-khk_Cyrl",)
    assert dict(suite.source.files) == {export.name: sha256_of(export)}
    assert (suite.source.dataset, suite.source.revision, suite.source.pinned) == (
        None,
        None,
        False,
    )


def test_the_three_formats_give_the_same_rows(tmp_path: Path) -> None:
    suites = []
    for suffix in (".jsonl", ".tsv", ".parquet"):
        export = write_export(tmp_path / f"export{suffix}", golden_export())
        out_dir = tmp_path / suffix.lstrip(".")
        assert invoke(human_args(export, out_dir)).exit_code == 0
        suites.append(Suite.from_bytes((out_dir / "human-test.json").read_bytes()))

    rows = [
        [(row.id, row.doc, dict(row.text)) for row in suite.rows] for suite in suites
    ]
    assert rows[0] == rows[1] == rows[2]
    assert suites[0].directions == suites[1].directions == suites[2].directions


def test_a_row_without_a_document_or_a_segment_number_is_numbered_in_every_format(
    tmp_path: Path,
) -> None:
    rows = [
        export_row(None, None, "Beta.", "B."),
        export_row(None, None, "Alpha.", "A."),
    ]
    found = []
    for suffix in (".jsonl", ".tsv", ".parquet"):
        export = write_export(tmp_path / f"export{suffix}", rows)
        out_dir = tmp_path / suffix.lstrip(".")
        assert invoke(human_args(export, out_dir)).exit_code == 0
        suite = Suite.from_bytes((out_dir / "human-test.json").read_bytes())
        found.append([(row.id, row.doc) for row in suite.rows])

    assert found == [[("s000001", None), ("s000002", None)]] * 3


def test_the_name_option_names_the_suite_file(tmp_path: Path) -> None:
    export = write_export(tmp_path / "export.jsonl", golden_export())
    out_dir = tmp_path / "out"

    result = invoke(human_args(export, out_dir, "--name", "mongolia-news"))

    assert result.exit_code == 0, result.output
    assert (
        Suite.from_bytes((out_dir / "mongolia-news.json").read_bytes()).name
        == "mongolia-news"
    )


def test_the_manifest_of_a_human_build_vouches_for_the_export_and_the_suite(
    tmp_path: Path,
) -> None:
    export = write_export(tmp_path / "export.jsonl", golden_export())
    out_dir = tmp_path / "out"
    assert invoke(human_args(export, out_dir)).exit_code == 0

    manifest = read_manifest(out_dir)

    assert manifest["stage"] == "eval_suite"
    assert [(item["path"], item["sha256"]) for item in manifest["inputs"]] == [
        (str(export), sha256_of(export))
    ]
    assert manifest["outputs"][0]["rows"] == 6
    assert manifest["metrics"] == {
        "rows.eng_Latn": 6.0,
        "rows.khk_Cyrl": 6.0,
        "directions": 1.0,
    }
    assert manifest["status"] == {
        "benchmark": "human",
        "split": "test",
        "kind": "human",
        "revision": "null",
        "pinned": "false",
    }


def test_the_split_of_a_human_set_is_test_and_may_be_left_out_or_given(
    tmp_path: Path,
) -> None:
    export = write_export(tmp_path / "export.jsonl", golden_export())

    given = invoke(human_args(export, tmp_path / "a", "--split", "test"))
    wrong = invoke(human_args(export, tmp_path / "b", "--split", "dev"))

    assert given.exit_code == 0, given.output
    assert_one_error(wrong, 2, "--split")
    assert not (tmp_path / "b").exists()


def test_a_kind_other_than_human_is_refused_for_the_human_benchmark(
    tmp_path: Path,
) -> None:
    export = write_export(tmp_path / "export.jsonl", golden_export())

    result = invoke(human_args(export, tmp_path / "out", "--kind", "benchmark"))

    assert_one_error(result, 2, "--kind")


def test_an_export_with_another_suffix_is_refused(tmp_path: Path) -> None:
    export = tmp_path / "export.csv"
    export.write_text("src,tgt\n", encoding="utf-8")

    result = invoke(human_args(export, tmp_path / "out"))

    assert_one_error(result, 2, "--from")


def test_an_export_that_is_missing_or_is_a_directory_is_refused(tmp_path: Path) -> None:
    (tmp_path / "dir.jsonl").mkdir()

    for source in (tmp_path / "nothing.jsonl", tmp_path / "dir.jsonl"):
        assert_one_error(invoke(human_args(source, tmp_path / "out")), 3, "")
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize(("field", "value"), [("origin", "web"), ("split", "train")])
def test_a_row_that_is_not_a_human_test_row_is_refused_naming_its_index(
    tmp_path: Path, field: str, value: str
) -> None:
    rows = golden_export()
    rows[3][field] = value
    export = write_export(tmp_path / "export.jsonl", rows)

    result = invoke(human_args(export, tmp_path / "out"))

    assert_one_error(result, 3, "")
    assert "row index 3" in result.stderr
    assert field in result.stderr
    assert not (tmp_path / "out").exists()


def test_two_sets_in_one_export_are_refused(tmp_path: Path) -> None:
    rows = golden_export()
    rows[5]["set"] = "other-set"
    export = write_export(tmp_path / "export.tsv", rows)

    result = invoke(human_args(export, tmp_path / "out"))

    assert_one_error(result, 3, "")
    assert "other-set" not in result.stderr


def test_inner_mongolian_in_an_export_is_refused_by_name(tmp_path: Path) -> None:
    rows = golden_export()
    rows[0]["tgt_lang"] = "khk_Mong"
    export = write_export(tmp_path / "export.jsonl", rows)

    result = invoke(human_args(export, tmp_path / "out"))

    assert_one_error(result, 3, "")
    assert "khk_Mong is refused by name" in result.stderr


@pytest.mark.parametrize(
    ("lines", "needle"),
    [
        ('{"src": "a"}\n\n{"src": "b"}\n', "line 2"),
        ("not json\n", "line 1"),
        ("[1, 2]\n", "line 1"),
        ('{"src": "a", "src": "b"}\n', "line 1"),
    ],
    ids=["blank-line", "not-json", "not-an-object", "repeated-key"],
)
def test_a_json_lines_export_that_is_not_one_is_refused_with_its_line(
    tmp_path: Path, lines: str, needle: str
) -> None:
    export = tmp_path / "export.jsonl"
    export.write_text(lines, encoding="utf-8")

    result = invoke(human_args(export, tmp_path / "out"))

    assert_one_error(result, 3, "")
    assert needle in result.stderr


def test_an_export_that_is_not_utf_8_is_refused(tmp_path: Path) -> None:
    export = tmp_path / "export.jsonl"
    export.write_bytes(b'{"src": "\xff"}\n')

    result = invoke(human_args(export, tmp_path / "out"))

    assert_one_error(result, 3, "")
    assert "UTF-8" in result.stderr


def test_a_tsv_export_with_another_header_is_refused(tmp_path: Path) -> None:
    export = tmp_path / "export.tsv"
    export.write_text("src\ttgt\nhello\tсайн\n", encoding="utf-8")

    result = invoke(human_args(export, tmp_path / "out"))

    assert_one_error(result, 3, "")
    assert "header" in result.stderr


def test_a_tsv_export_with_a_segment_number_that_is_not_a_number_is_refused_with_its_row(
    tmp_path: Path,
) -> None:
    rows = golden_export()
    export = write_export(tmp_path / "export.tsv", rows)
    export.write_text(
        export.read_text(encoding="utf-8").replace("\t2\tnews", "\tx\tnews", 1),
        encoding="utf-8",
    )

    result = invoke(human_args(export, tmp_path / "out"))

    assert_one_error(result, 3, "")
    assert "seg_idx" in result.stderr


def test_a_tsv_export_with_a_short_row_is_refused(tmp_path: Path) -> None:
    export = write_export(tmp_path / "export.tsv", golden_export())
    text = export.read_text(encoding="utf-8")
    export.write_text(text + "only\ttwo\n", encoding="utf-8")

    result = invoke(human_args(export, tmp_path / "out"))

    assert_one_error(result, 3, "")


def test_a_parquet_export_with_other_columns_is_refused(tmp_path: Path) -> None:
    import polars as pl

    export = tmp_path / "export.parquet"
    pl.DataFrame({"src": ["a"], "tgt": ["b"]}).write_parquet(export)

    result = invoke(human_args(export, tmp_path / "out"))

    assert_one_error(result, 3, "")


def test_a_parquet_file_that_is_not_parquet_is_refused(tmp_path: Path) -> None:
    export = tmp_path / "export.parquet"
    export.write_bytes(b"not parquet")

    assert_one_error(invoke(human_args(export, tmp_path / "out")), 3, "")


def test_a_json_lines_export_keeps_unicode_text_and_the_ids_it_gives(
    tmp_path: Path,
) -> None:
    rows: list[dict[str, Any]] = [export_row("d1", 1, "Hello.", "Сайн байна уу.")]
    export = tmp_path / "export.jsonl"
    export.write_text(json.dumps(rows[0], ensure_ascii=True) + "\n", encoding="utf-8")

    assert invoke(human_args(export, tmp_path / "out")).exit_code == 0
    suite = Suite.from_bytes((tmp_path / "out" / "human-test.json").read_bytes())

    assert suite.rows[0].text["khk_Cyrl"] == "Сайн байна уу."
