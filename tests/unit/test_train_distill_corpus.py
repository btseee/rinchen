"""Tests for `rinchen.train.distill_corpus`: reading the KD files a generation wrote back,
refusing a file its manifest does not vouch for, and counting every row once, in the kept
rows or under the first reason (in the order of the module's table) that left it out.

The counts are worked out by hand from the rows each test plants.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import polars as pl
import pytest

import rinchen.train.distill_corpus as distill_corpus_module
from rinchen.corpus.schema import corpus_frame
from rinchen.errors import ConfigError, DataError
from rinchen.manifest import store
from rinchen.manifest.model import Manifest
from rinchen.manifest.stage import output_file
from rinchen.train.distill_corpus import (
    KD_EXCLUSION_REASONS,
    KdRows,
    KdStats,
    load_kd_rows,
)
from rinchen.train.mixture import LINE_BREAK_CHARACTERS
from tests.support.corpus_rows import EN_MN, MN_EN, corpus_row
from tests.support.planted_kd import KD_FILE, write_kd_files

DIRECTIONS = (EN_MN, MN_EN)
NO_EXCLUSIONS = {reason: 0 for reason in KD_EXCLUSION_REASONS}


def kd_row(
    src: str = "Hello",
    tgt: str = "Сайн байна уу",
    *,
    label: str = "ft:toy",
    origin: str = "ft",
    **options: str,
) -> dict[str, str]:
    """A row of a KD corpus: origin `ft`, split `train`, the corpus label `ft:<name>`."""
    return {**corpus_row(src, tgt, origin=origin, **options), "corpus": label}


def kd_file(directory: Path, contents: list[dict[str, str]], **options: Any) -> Path:
    return write_kd_files(directory, corpus_frame(contents), **options)


def _load(*paths: Path, directions: Any = DIRECTIONS) -> KdRows:
    return load_kd_rows(list(paths), directions=directions)


# --- a file with its manifest --------------------------------------------------------


def test_a_kd_file_with_its_manifest_loads_its_rows_sorted_by_id_source_target(
    tmp_path: Path,
) -> None:
    rows = [
        kd_row("b", "2"),
        kd_row("a", "1"),
        kd_row("a", "0"),
        kd_row("c", "3", src_lang="khk_Cyrl", tgt_lang="eng_Latn"),
    ]
    path = kd_file(tmp_path, rows)

    loaded = _load(path)

    frame = loaded.frame
    assert frame.columns == ["id", "src_lang", "tgt_lang", "src", "tgt", "direction"]
    assert frame.schema == {
        "id": pl.String,
        "src_lang": pl.String,
        "tgt_lang": pl.String,
        "src": pl.String,
        "tgt": pl.String,
        "direction": pl.String,
    }
    ordered = sorted(
        corpus_frame(rows).select("id", "src", "tgt").iter_rows(), key=lambda r: r
    )
    assert list(frame.select("id", "src", "tgt").iter_rows()) == ordered
    assert sorted(frame["direction"].unique()) == [EN_MN, MN_EN]
    assert loaded.stats == KdStats(
        rows_read=4,
        rows_kept=4,
        excluded=NO_EXCLUSIONS,
        duplicate_ids=0,
        rows_by_direction={EN_MN: 3, MN_EN: 1},
        rows_by_label={"ft:toy": 4},
    )


def test_the_statistics_are_json_ready_with_every_reason_present(
    tmp_path: Path,
) -> None:
    path = kd_file(tmp_path, [kd_row("a", "1"), kd_row("b", "2", split="dev")])

    document = _load(path).stats.to_dict()

    assert json.loads(json.dumps(document)) == {
        "rows_read": 2,
        "rows_kept": 1,
        "excluded": {**NO_EXCLUSIONS, "not_train": 1},
        "duplicate_ids": 0,
        "rows_by_direction": {EN_MN: 1},
        "rows_by_label": {"ft:toy": 1},
    }
    assert list(document["excluded"]) == list(KD_EXCLUSION_REASONS)


def test_the_reasons_are_the_six_of_the_ruling_in_their_order() -> None:
    assert KD_EXCLUSION_REASONS == (
        "not_train",
        "filtered",
        "wrong_origin",
        "empty",
        "line_break",
        "unknown_direction",
    )


def test_a_file_in_a_directory_with_glob_characters_is_read_as_named(
    tmp_path: Path,
) -> None:
    path = kd_file(tmp_path / "run[1]*", [kd_row("a", "1")])

    loaded = _load(path)

    assert loaded.stats.rows_kept == 1


def test_a_file_that_the_manifest_lists_under_another_directory_is_matched_by_name(
    tmp_path: Path,
) -> None:
    # The corpus was generated in one place and moved with its manifest: the manifest
    # names the file by the path it had, and the name is what ties the two.
    elsewhere = tmp_path / "generated"
    path = kd_file(elsewhere, [kd_row("a", "1")])
    moved = tmp_path / "moved"
    moved.mkdir()
    (moved / KD_FILE).write_bytes(path.read_bytes())
    (moved / "manifest.json").write_bytes((elsewhere / "manifest.json").read_bytes())

    assert _load(moved / KD_FILE).stats.rows_kept == 1


# --- the manifest vouches, or the file is refused --------------------------------------


def test_a_kd_file_without_a_manifest_is_a_config_error_naming_the_file(
    tmp_path: Path,
) -> None:
    path = kd_file(tmp_path, [kd_row()])
    (tmp_path / "manifest.json").unlink()

    with pytest.raises(ConfigError, match=rf"{KD_FILE}.*manifest"):
        _load(path)


def test_a_manifest_of_another_stage_is_a_config_error(tmp_path: Path) -> None:
    path = kd_file(tmp_path, [kd_row()], stage="synth_run")

    with pytest.raises(ConfigError, match=rf"{KD_FILE}.*synth_run.*distill_generate"):
        _load(path)


@pytest.mark.parametrize(
    "status",
    [
        {"generate": "partial: stopped"},
        {"generate": "skipped: nothing"},
        {"qe": "ok"},
        {},
        {"generate": "OK"},
    ],
)
def test_a_manifest_whose_generate_status_is_not_ok_is_a_config_error(
    tmp_path: Path, status: dict[str, str]
) -> None:
    path = kd_file(tmp_path, [kd_row()], status=status)

    with pytest.raises(ConfigError, match=rf"{KD_FILE}.*generate"):
        _load(path)


def test_a_manifest_that_does_not_list_the_file_is_a_config_error(
    tmp_path: Path,
) -> None:
    path = kd_file(tmp_path, [kd_row()], listed=False)

    with pytest.raises(ConfigError, match=rf"{KD_FILE}.*does not list"):
        _load(path)


def test_a_manifest_that_lists_the_file_twice_is_a_config_error(tmp_path: Path) -> None:
    path = kd_file(tmp_path, [kd_row()])
    manifest = store.read(tmp_path)
    assert manifest is not None
    twice = manifest.model_copy(
        update={"outputs": [*manifest.outputs, manifest.outputs[0]]}
    )
    store.write(tmp_path, twice)

    with pytest.raises(ConfigError, match=rf"{KD_FILE}.*twice"):
        _load(path)


def test_a_file_that_changed_after_the_manifest_is_a_config_error(
    tmp_path: Path,
) -> None:
    path = kd_file(tmp_path, [kd_row()])
    path.write_bytes(path.read_bytes() + b"\0")

    with pytest.raises(ConfigError, match=rf"{KD_FILE}.*SHA-256"):
        _load(path)


def test_a_flipped_byte_is_refused_by_its_hash_before_anything_parses_the_file(
    tmp_path: Path,
) -> None:
    # The flip makes the Parquet unreadable: a reader would say DataError. The hash is
    # taken first, so the refusal is the manifest's.
    path = kd_file(tmp_path, [kd_row(str(index), "t") for index in range(50)])
    data = bytearray(path.read_bytes())
    data[len(data) // 2] ^= 0xFF
    path.write_bytes(bytes(data))

    with pytest.raises(ConfigError, match="SHA-256"):
        _load(path)


def test_a_recorded_row_count_that_is_not_the_files_is_a_config_error(
    tmp_path: Path,
) -> None:
    path = kd_file(tmp_path, [kd_row("a", "1"), kd_row("b", "2")], rows=3)

    with pytest.raises(ConfigError, match=rf"{KD_FILE}.*3 rows.*2"):
        _load(path)


def test_a_manifest_without_a_row_count_for_the_file_is_a_config_error(
    tmp_path: Path,
) -> None:
    path = kd_file(tmp_path, [kd_row()])
    manifest = store.read(tmp_path)
    assert manifest is not None
    entry = manifest.outputs[0].model_copy(update={"rows": None})
    store.write(tmp_path, manifest.model_copy(update={"outputs": [entry]}))

    with pytest.raises(ConfigError, match=rf"{KD_FILE}.*row count"):
        _load(path)


def test_an_unreadable_manifest_is_a_config_error_naming_the_file(
    tmp_path: Path,
) -> None:
    path = kd_file(tmp_path, [kd_row()])
    (tmp_path / "manifest.json").write_text("{not json", encoding="utf-8")

    with pytest.raises(ConfigError, match=rf"{KD_FILE}.*manifest"):
        _load(path)


def test_a_missing_kd_file_is_a_data_error(tmp_path: Path) -> None:
    kd_file(tmp_path, [kd_row()])

    with pytest.raises(DataError, match=r"missing\.parquet"):
        _load(tmp_path / "missing.parquet")


def test_a_directory_instead_of_a_kd_file_is_a_data_error(tmp_path: Path) -> None:
    with pytest.raises(DataError, match="not found"):
        _load(tmp_path)


def test_the_file_is_hashed_once_and_before_it_is_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = kd_file(tmp_path, [kd_row("a", "1")])
    events: list[str] = []
    original_scan = pl.scan_parquet

    def scanning(source: Any, *args: Any, **kwargs: Any) -> Any:
        events.append("scan")
        return original_scan(source, *args, **kwargs)

    original_hash = distill_corpus_module.sha256_of

    def hashing(target: Path) -> str:
        events.append(f"hash {target.name}")
        return original_hash(target)

    monkeypatch.setattr(pl, "scan_parquet", scanning)
    monkeypatch.setattr(distill_corpus_module, "sha256_of", hashing)

    _load(path)

    assert events[0] == f"hash {KD_FILE}"
    assert events.count(f"hash {KD_FILE}") == 1
    assert "scan" in events[1:]


def test_one_bad_file_among_several_refuses_all_and_names_it(tmp_path: Path) -> None:
    good = kd_file(tmp_path / "good", [kd_row("a", "1")])
    bad = kd_file(tmp_path / "bad", [kd_row("b", "2")])
    (tmp_path / "bad" / "manifest.json").unlink()

    with pytest.raises(ConfigError, match=r"bad"):
        _load(good, bad)


# --- the reasons -------------------------------------------------------------------------


def _frame_with(rows: list[dict[str, str]], **columns: list[Any]) -> pl.DataFrame:
    frame = corpus_frame(rows)
    for name, values in columns.items():
        frame = frame.with_columns(pl.Series(name, values, dtype=frame.schema[name]))
    return frame


def test_every_row_is_counted_once_under_its_reason_or_kept(tmp_path: Path) -> None:
    rows = [
        kd_row("kept 1", "t1"),
        kd_row("kept 2", "t2", src_lang="khk_Cyrl", tgt_lang="eng_Latn"),
        # not_train: dev, test, holdout
        kd_row("dev", "t", split="dev"),
        kd_row("test", "t", split="test"),
        kd_row("holdout", "t", split="holdout"),
        # filtered (below)
        kd_row("filtered", "t"),
        # wrong_origin
        kd_row("human", "t", origin="human"),
        kd_row("bt", "t", origin="bt"),
        kd_row("synth", "t", origin="synth_nmt"),
        # empty
        kd_row("", "t"),
        kd_row("blank target", "   "),
        kd_row("separator", chr(0x1F)),
        kd_row(chr(0x1C), "t"),
        # line_break (below: one per character, in the loop)
        # unknown_direction
        kd_row("unknown", "t", src_lang="rus_Cyrl", tgt_lang="khk_Cyrl"),
    ]
    rows += [kd_row(f"a{c}b", "t") for c in LINE_BREAK_CHARACTERS[:5]]
    rows += [kd_row("a", f"b{c}c") for c in LINE_BREAK_CHARACTERS[5:]]
    filtered_index = 5
    reasons = [None] * len(rows)
    reasons[filtered_index] = "length"
    frame = _frame_with(rows, filter_reason=reasons)
    path = write_kd_files(tmp_path, frame)

    stats = _load(path).stats

    assert stats == KdStats(
        rows_read=len(rows),
        rows_kept=2,
        excluded={
            "not_train": 3,
            "filtered": 1,
            "wrong_origin": 3,
            "empty": 4,
            "line_break": 10,
            "unknown_direction": 1,
        },
        duplicate_ids=0,
        rows_by_direction={EN_MN: 1, MN_EN: 1},
        rows_by_label={"ft:toy": 2},
    )
    assert stats.rows_read == stats.rows_kept + sum(stats.excluded.values())


def test_there_are_ten_line_break_characters_and_each_one_is_a_line_break_row(
    tmp_path: Path,
) -> None:
    assert len(LINE_BREAK_CHARACTERS) == 10
    for position, character in enumerate(LINE_BREAK_CHARACTERS):
        directory = tmp_path / str(position)
        path = kd_file(
            directory, [kd_row("fine", "fine"), kd_row("a", f"x{character}y")]
        )

        stats = _load(path).stats

        assert stats.excluded["line_break"] == 1, hex(ord(character))
        assert stats.rows_kept == 1


@pytest.mark.parametrize(
    "blank", ["", " ", "\t", "  \t ", chr(0xA0), chr(0x2003), chr(0x1C), chr(0x1F)]
)
@pytest.mark.parametrize("side", ["src", "tgt"])
def test_a_blank_text_on_either_side_is_empty(
    tmp_path: Path, blank: str, side: str
) -> None:
    row = kd_row(blank, "t") if side == "src" else kd_row("s", blank)
    path = kd_file(tmp_path, [row, kd_row("fine", "fine")])

    stats = _load(path).stats

    assert stats.excluded["empty"] == 1
    assert stats.rows_kept == 1


def test_a_null_text_is_empty(tmp_path: Path) -> None:
    frame = _frame_with(
        [kd_row("a", "1"), kd_row("b", "2"), kd_row("c", "3")],
        src=["a", None, "c"],
        tgt=["1", "2", None],
    )
    path = write_kd_files(tmp_path, frame)

    stats = _load(path).stats

    assert stats.excluded["empty"] == 2
    assert stats.rows_kept == 1


def test_the_first_reason_in_the_order_decides(tmp_path: Path) -> None:
    rows = [
        # not_train beats everything below it
        kd_row("", "t", split="test", origin="human"),
        # filtered beats wrong_origin, empty, line_break and unknown_direction
        kd_row("", "t", origin="human", src_lang="rus_Cyrl"),
        # wrong_origin beats empty
        kd_row("", "t", origin="web"),
        # empty beats unknown_direction
        kd_row("", "t", src_lang="rus_Cyrl"),
        # line_break beats unknown_direction
        kd_row("a\nb", "t", src_lang="rus_Cyrl"),
        # a blank text is `empty`, not a line break, whatever whitespace it is
        kd_row("\n", "t"),
        kd_row("fine", "fine"),
    ]
    frame = _frame_with(
        rows, filter_reason=[None, "length", None, None, None, None, None]
    )
    path = write_kd_files(tmp_path, frame)

    excluded = _load(path).stats.excluded

    assert excluded == {
        "not_train": 1,
        "filtered": 1,
        "wrong_origin": 1,
        "empty": 2,
        "line_break": 1,
        "unknown_direction": 0,
    }


def test_a_direction_the_originals_lack_is_unknown_and_a_missing_language_too(
    tmp_path: Path,
) -> None:
    frame = _frame_with(
        [kd_row("a", "1"), kd_row("b", "2"), kd_row("c", "3")],
        src_lang=["eng_Latn", "rus_Cyrl", "eng_Latn"],
        tgt_lang=["khk_Cyrl", "khk_Cyrl", None],
    )
    path = write_kd_files(tmp_path, frame)

    loaded = _load(path, directions=[EN_MN])

    assert loaded.stats.rows_kept == 1
    assert loaded.stats.excluded["unknown_direction"] == 2
    assert loaded.frame["direction"].to_list() == [EN_MN]


def test_a_null_origin_is_a_wrong_origin_and_nothing_else(tmp_path: Path) -> None:
    frame = _frame_with([kd_row("a", "1"), kd_row("b", "2")], origin=[None, "ft"])
    path = write_kd_files(tmp_path, frame)

    stats = _load(path).stats

    assert stats.excluded["wrong_origin"] == 1
    assert stats.rows_kept == 1


def test_the_directions_are_the_ones_given_whatever_the_collection(
    tmp_path: Path,
) -> None:
    path = kd_file(tmp_path, [kd_row("a", "1")])

    for directions in ([EN_MN], (EN_MN,), {EN_MN}, frozenset({EN_MN})):
        assert _load(path, directions=directions).stats.rows_kept == 1
    assert _load(path, directions=[MN_EN, EN_MN]).stats.rows_kept == 1


def test_no_direction_given_leaves_every_row_unknown_and_is_a_data_error(
    tmp_path: Path,
) -> None:
    path = kd_file(tmp_path, [kd_row("a", "1")])

    with pytest.raises(DataError, match=r"unknown_direction.: 1"):
        _load(path, directions=[])


def test_the_split_filter_is_lazy_so_the_text_of_another_split_is_not_looked_at(
    tmp_path: Path,
) -> None:
    # A test row whose source is null is counted `not_train`: had its text been looked at,
    # it would be `empty`.
    frame = _frame_with(
        [
            kd_row("a", "1"),
            kd_row("t", "2", split="test"),
            kd_row("d", "3", split="dev"),
        ],
        src=["a", None, None],
    )
    path = write_kd_files(tmp_path, frame)

    stats = _load(path).stats

    assert stats.excluded["not_train"] == 2
    assert stats.excluded["empty"] == 0
    assert stats.rows_kept == 1


def test_only_the_train_rows_of_a_mixed_file_are_held_in_memory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rows = [
        *(kd_row(f"train {i}", "t") for i in range(3)),
        *(kd_row(f"dev {i}", "t", split="dev") for i in range(400)),
        *(kd_row(f"test {i}", "t", split="test") for i in range(400)),
    ]
    path = kd_file(tmp_path, rows)
    heights: list[int] = []
    original = pl.LazyFrame.collect

    def recording(self: pl.LazyFrame, *args: Any, **kwargs: Any) -> Any:
        collected = original(self, *args, **kwargs)
        heights.append(getattr(collected, "height", 0))
        return collected

    monkeypatch.setattr(pl.LazyFrame, "collect", recording)

    stats = _load(path).stats

    assert stats.rows_read == 803
    assert stats.excluded["not_train"] == 800
    assert max(heights) <= 3


# --- duplicates, several files, nothing kept -----------------------------------------


def test_duplicate_ids_are_kept_and_counted(tmp_path: Path) -> None:
    rows = [kd_row("a", "1"), kd_row("a", "1"), kd_row("a", "1"), kd_row("b", "2")]
    path = kd_file(tmp_path, rows)

    loaded = _load(path)

    assert loaded.stats.rows_kept == 4
    assert loaded.stats.duplicate_ids == 2
    assert loaded.frame.height == 4


def test_two_kd_files_merge_and_the_labels_are_counted(tmp_path: Path) -> None:
    first = kd_file(
        tmp_path / "one",
        [kd_row("a", "1", label="ft:one"), kd_row("b", "2", label="ft:one")],
    )
    second = kd_file(
        tmp_path / "two",
        [
            kd_row("c", "3", src_lang="khk_Cyrl", tgt_lang="eng_Latn", label="ft:two"),
            kd_row("a", "1", label="ft:two"),
            kd_row("x", "", label="ft:two"),
        ],
    )

    loaded = _load(first, second)

    assert loaded.stats == KdStats(
        rows_read=5,
        rows_kept=4,
        excluded={**NO_EXCLUSIONS, "empty": 1},
        duplicate_ids=1,
        rows_by_direction={EN_MN: 3, MN_EN: 1},
        rows_by_label={"ft:one": 2, "ft:two": 2},
    )
    assert loaded.frame.height == 4
    ids = loaded.frame["id"].to_list()
    assert ids == sorted(ids)


def test_the_merged_frame_does_not_depend_on_the_order_of_the_files(
    tmp_path: Path,
) -> None:
    first = kd_file(
        tmp_path / "one",
        [kd_row("a", "1", label="ft:one"), kd_row("b", "2", label="ft:one")],
    )
    second = kd_file(
        tmp_path / "two",
        [kd_row("c", "3", label="ft:two"), kd_row("a", "0", label="ft:two")],
    )

    forward = _load(first, second)
    backward = _load(second, first)

    assert forward.frame.equals(backward.frame)
    assert forward.stats == backward.stats


def test_no_kept_row_is_a_data_error_naming_the_counts(tmp_path: Path) -> None:
    path = kd_file(
        tmp_path,
        [
            kd_row("dev", "t", split="dev"),
            kd_row("human", "t", origin="human"),
            kd_row("blank", " "),
        ],
    )

    with pytest.raises(DataError) as caught:
        _load(path)

    message = str(caught.value)
    assert "3 rows read" in message
    assert "0 kept" in message
    assert "'not_train': 1" in message
    assert "'wrong_origin': 1" in message
    assert "'empty': 1" in message


def test_no_file_at_all_is_a_data_error() -> None:
    with pytest.raises(DataError, match="no KD file"):
        _load()


def test_a_kept_row_without_an_id_is_a_data_error(tmp_path: Path) -> None:
    frame = _frame_with([kd_row("a", "1"), kd_row("b", "2")], id=["x", None])
    path = write_kd_files(tmp_path, frame)

    with pytest.raises(DataError, match="null id"):
        _load(path)


def test_a_kept_row_without_a_corpus_label_is_a_data_error(tmp_path: Path) -> None:
    frame = _frame_with([kd_row("a", "1"), kd_row("b", "2")], corpus=["ft:toy", None])
    path = write_kd_files(tmp_path, frame)

    with pytest.raises(DataError, match="null corpus"):
        _load(path)


@pytest.mark.parametrize("where", ["scan", "collect"])
def test_a_reader_panic_is_a_data_error_naming_the_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, where: str
) -> None:
    # A damaged file can make the Rust reader panic, which is not an Exception.
    path = kd_file(tmp_path, [kd_row("a", "1")])

    def panic(*args: Any, **kwargs: Any) -> None:
        raise pl.exceptions.PanicException("assertion failed in the parquet reader")

    if where == "scan":
        monkeypatch.setattr(pl, "scan_parquet", panic)
    else:
        monkeypatch.setattr(pl.LazyFrame, "collect", panic)

    with pytest.raises(DataError, match=rf"cannot read KD file.*{KD_FILE}.*assertion"):
        _load(path)


def test_a_row_without_an_id_that_is_left_out_is_only_counted(tmp_path: Path) -> None:
    frame = _frame_with(
        [kd_row("a", "1"), kd_row("b", "2", split="dev")], id=["x", None]
    )
    path = write_kd_files(tmp_path, frame)

    assert _load(path).stats.excluded["not_train"] == 1


def test_a_file_that_is_not_a_corpus_table_is_a_data_error(tmp_path: Path) -> None:
    path = tmp_path / KD_FILE
    pl.DataFrame({"x": [1]}).write_parquet(path)
    # A manifest that vouches for it, so that only its content is wrong.
    store.write(
        tmp_path,
        Manifest(
            stage="distill_generate",
            created="2026-01-01T00:00:00+00:00",
            code_version="t",
            config_sha256="0" * 64,
            seed=1,
            outputs=[output_file(path, 1)],
            status={"generate": "ok"},
        ),
    )

    with pytest.raises(DataError, match="not a corpus table"):
        _load(path)
