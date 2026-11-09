"""Tests for `rinchen.train.mixture`: which corpus rows train and which are counted
out, the development sets and the statistics.

Counts below are worked out by hand from the rules in the module docstring and the D7
spec (R2, R7, R9), not from the code. The properties over random rows are in
`tests/property/test_train_mixture_property.py`.
"""

from __future__ import annotations

import json
import random
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import polars as pl
import pytest

from rinchen.corpus.schema import CORPUS_SCHEMA, corpus_frame
from rinchen.errors import ConfigError, DataError
from rinchen.train.mixture import (
    LINE_BREAK_CHARACTERS,
    TrainingRows,
    blank_text,
    has_line_break,
    load_dev_rows,
    load_training_rows,
    source_collision_rate,
)
from tests.support.corpus_rows import (
    EN_MN,
    RU_MN,
    corpus_row,
    numbered_rows,
    training_from_rows,
    write_frame,
    write_rows,
)

_LINE_BREAKS = {
    "LF": 0x0A,
    "CR": 0x0D,
    "VT": 0x0B,
    "FF": 0x0C,
    "FS": 0x1C,
    "GS": 0x1D,
    "RS": 0x1E,
    "NEL": 0x85,
    "LS": 0x2028,
    "PS": 0x2029,
}


# --- load_training_rows: which rows train ------------------------------------------


def test_only_unfiltered_authentic_training_rows_are_kept_and_the_rest_counted_by_reason(
    tmp_path: Path,
) -> None:
    rows = [
        corpus_row("kept 1", "t1"),
        corpus_row("kept 2", "t2", origin="web"),
        corpus_row("dev", "t", split="dev"),
        corpus_row("test", "t", split="test"),
        corpus_row("holdout", "t", split="holdout"),
        corpus_row("filtered", "t"),
        corpus_row("ft", "t", origin="ft"),
        corpus_row("llm", "t", origin="synth_llm"),
        corpus_row("nmt", "t", origin="synth_nmt"),
        corpus_row("", "t"),
        corpus_row("blank target", "   "),
        corpus_row("two\nlines", "t"),
    ]
    reasons: list[str | None] = [None] * len(rows)
    reasons[5] = "lid_src"

    training = training_from_rows(tmp_path, rows, filter_reasons=reasons)

    stats = training.stats
    assert training.frame["src"].sort().to_list() == ["kept 1", "kept 2"]
    assert stats.rows_read == 12
    assert stats.rows_kept == 2
    assert stats.excluded == {
        "not_train": 3,
        "filtered": 1,
        "excluded_origin": 3,
        "empty": 2,
        "line_break": 1,
    }
    assert stats.excluded_by_origin == {"ft": 1, "synth_llm": 1, "synth_nmt": 1}


def test_every_row_is_counted_exactly_once_so_the_counts_add_up(
    tmp_path: Path,
) -> None:
    rows = [
        corpus_row("a", "b"),
        corpus_row("c", "d", split="dev"),
        corpus_row("e", "f", origin="ft"),
        corpus_row("", ""),
        corpus_row("g\r", "h"),
        corpus_row("i", "j", origin="bt"),
        corpus_row("k", "l"),
    ]
    reasons: list[str | None] = [None, None, None, None, None, None, "numbers"]

    stats = training_from_rows(tmp_path, rows, filter_reasons=reasons).stats

    assert stats.rows_read == 7
    assert stats.rows_read == stats.rows_kept + sum(stats.excluded.values())
    assert sum(stats.excluded_by_origin.values()) == stats.excluded["excluded_origin"]
    assert sum(stats.rows_by_corpus.values()) == stats.rows_kept


@pytest.mark.parametrize(
    ("first_match", "other"),
    [
        # A dev row that is also filtered, of an excluded origin and empty is not_train.
        (
            {"split": "dev", "origin": "ft", "src": ""},
            "not_train",
        ),
        ({"origin": "ft", "src": ""}, "excluded_origin"),
        ({"src": "\n"}, "empty"),
    ],
)
def test_a_row_is_counted_under_the_first_reason_that_matches(
    tmp_path: Path, first_match: dict[str, str], other: str
) -> None:
    row = corpus_row("text", "t") | first_match
    reasons = ["lid_src"] if other == "not_train" else [None]

    stats = training_from_rows(
        tmp_path, [row, corpus_row("kept", "t")], filter_reasons=[*reasons, None]
    ).stats

    assert stats.excluded[other] == 1
    assert sum(stats.excluded.values()) == 1


def test_a_filtered_row_of_an_excluded_origin_is_counted_as_filtered_only(
    tmp_path: Path,
) -> None:
    stats = training_from_rows(
        tmp_path,
        [corpus_row("a", "b", origin="ft"), corpus_row("kept", "t")],
        filter_reasons=["lid_src", None],
    ).stats

    assert stats.excluded["filtered"] == 1
    assert stats.excluded["excluded_origin"] == 0
    assert stats.excluded_by_origin == {"ft": 0, "synth_llm": 0, "synth_nmt": 0}


def test_an_excluded_origin_is_counted_by_origin_only_among_training_rows(
    tmp_path: Path,
) -> None:
    rows = [
        corpus_row("a", "b", origin="ft"),
        corpus_row("c", "d", origin="ft", split="dev"),
        corpus_row("e", "f", origin="synth_llm", split="test"),
        corpus_row("kept", "t"),
    ]

    stats = training_from_rows(tmp_path, rows).stats

    assert stats.excluded_by_origin == {"ft": 1, "synth_llm": 0, "synth_nmt": 0}
    assert stats.excluded["not_train"] == 2


@pytest.mark.parametrize("name", list(_LINE_BREAKS))
@pytest.mark.parametrize("side", ["src", "tgt"])
def test_a_line_break_character_in_either_text_excludes_the_row(
    tmp_path: Path, name: str, side: str
) -> None:
    text = f"left{chr(_LINE_BREAKS[name])}right"
    row = corpus_row(text, "t") if side == "src" else corpus_row("s", text)

    stats = training_from_rows(tmp_path, [row, corpus_row("kept", "t")]).stats

    assert stats.excluded["line_break"] == 1
    assert stats.rows_kept == 1


def test_a_carriage_return_and_line_feed_pair_is_one_excluded_row(
    tmp_path: Path,
) -> None:
    stats = training_from_rows(
        tmp_path, [corpus_row("a\r\nb", "t"), corpus_row("kept", "t")]
    ).stats

    assert stats.excluded["line_break"] == 1


def test_a_tab_a_nul_and_a_zero_width_space_are_kept_in_a_text(tmp_path: Path) -> None:
    rows = [
        corpus_row("a\tb", "t1"),
        corpus_row("a" + chr(0) + "b", "t2"),
        corpus_row("a" + chr(0x200B) + "b", "t3"),
        corpus_row("a" + chr(0x1F) + "b", "t4"),
    ]

    stats = training_from_rows(tmp_path, rows).stats

    assert stats.rows_kept == 4
    assert sum(stats.excluded.values()) == 0


def test_the_line_break_characters_are_the_ones_splitlines_splits_at() -> None:
    expected = {
        chr(code)
        for code in range(0x110000)
        if not 0xD800 <= code <= 0xDFFF
        and ("a" + chr(code) + "b").splitlines() != ["a" + chr(code) + "b"]
    }

    assert set(LINE_BREAK_CHARACTERS) == expected
    assert len(LINE_BREAK_CHARACTERS) == 10


def test_a_text_is_empty_when_python_would_strip_it_to_nothing(tmp_path: Path) -> None:
    space = [chr(code) for code in range(0x110000) if chr(code).isspace()]
    not_space = [
        chr(code)
        for code in range(0x20000)
        if not chr(code).isspace() and not 0xD800 <= code <= 0xDFFF
    ][::11]
    rows = [corpus_row(char, "t") for char in space] + [
        corpus_row("s", char, origin="web") for char in not_space
    ]

    stats = training_from_rows(tmp_path, rows).stats

    assert stats.excluded["empty"] == len(space)
    assert stats.rows_kept == len(not_space)
    assert stats.excluded["line_break"] == 0


def test_a_null_text_is_an_empty_one(tmp_path: Path) -> None:
    frame = corpus_frame([corpus_row("a", "b"), corpus_row("kept", "t")]).with_columns(
        pl.Series("src", [None, "kept"], dtype=pl.String)
    )

    stats = load_training_rows([write_frame(tmp_path, "nulls", frame)]).stats

    assert stats.excluded["empty"] == 1
    assert stats.rows_kept == 1


@pytest.mark.parametrize(
    ("origin", "group"),
    [
        ("human", "auth"),
        ("web", "auth"),
        ("mined", "auth"),
        ("converted", "auth"),
        ("bt", "bt"),
    ],
)
def test_the_origin_decides_the_group_and_the_direction_is_its_two_languages(
    tmp_path: Path, origin: str, group: str
) -> None:
    rows = [
        corpus_row("kept", "t", origin=origin),
        corpus_row("auth", "t", origin="human"),
    ]

    training = training_from_rows(tmp_path, rows)

    kept = training.frame.filter(pl.col("src") == "kept")
    assert kept["group"].to_list() == [group]
    assert kept["direction"].to_list() == [EN_MN]
    assert kept["origin"].to_list() == [origin]


def test_the_training_rows_are_counted_by_corpus_name(tmp_path: Path) -> None:
    rows = [
        *numbered_rows(3),
        *numbered_rows(2, origin="bt"),
        *[corpus_row(f"r{i}", f"t{i}", src_lang="rus_Cyrl") for i in range(4)],
    ]

    training = training_from_rows(tmp_path, rows)

    assert training.stats.rows_by_corpus == {
        "eng_Latn-khk_Cyrl.auth": 3,
        "eng_Latn-khk_Cyrl.bt": 2,
        "rus_Cyrl-khk_Cyrl.auth": 4,
    }
    assert training.directions == (EN_MN, RU_MN)


def test_a_direction_that_has_only_bt_rows_is_not_a_training_direction(
    tmp_path: Path,
) -> None:
    rows = [*numbered_rows(2), *numbered_rows(2, src_lang="rus_Cyrl", origin="bt")]

    training = training_from_rows(tmp_path, rows)

    assert training.directions == (EN_MN,)
    assert training.stats.rows_by_corpus[f"{RU_MN}.bt"] == 2


def test_rows_of_several_files_are_read_and_counted_together(tmp_path: Path) -> None:
    first = write_rows(
        tmp_path, "first", [*numbered_rows(3), corpus_row("dev", "t", split="dev")]
    )
    second = write_rows(tmp_path, "second", [corpus_row("x", "y", src_lang="rus_Cyrl")])

    stats = load_training_rows([first, second]).stats

    assert stats.rows_read == 5
    assert stats.rows_kept == 4
    assert stats.excluded["not_train"] == 1


def test_a_language_outside_the_registry_is_a_config_error_that_names_it(
    tmp_path: Path,
) -> None:
    rows = [*numbered_rows(2), corpus_row("a", "b", tgt_lang="xxx_Latn")]

    with pytest.raises(ConfigError, match="xxx_Latn"):
        training_from_rows(tmp_path, rows)


def test_a_language_outside_the_registry_in_an_excluded_row_is_not_checked(
    tmp_path: Path,
) -> None:
    rows = [*numbered_rows(2), corpus_row("a", "b", src_lang="xxx_Latn", split="dev")]

    assert training_from_rows(tmp_path, rows).stats.rows_kept == 2


def test_the_same_id_in_two_files_keeps_both_rows_and_counts_the_repeat(
    tmp_path: Path,
) -> None:
    first = write_rows(
        tmp_path, "first", [corpus_row("same", "text"), corpus_row("other", "text")]
    )
    second = write_rows(tmp_path, "second", [corpus_row("same", "text")])

    training = load_training_rows([first, second])

    assert training.stats.rows_kept == 3
    assert training.stats.duplicate_ids == 1
    assert training.frame.height == 3


def test_a_triple_id_counts_as_two_repeats(tmp_path: Path) -> None:
    rows = [
        corpus_row("a", "b"),
        corpus_row("a", "b"),
        corpus_row("a", "b"),
        corpus_row("c", "d"),
    ]

    assert training_from_rows(tmp_path, rows).stats.duplicate_ids == 2


def test_an_id_repeated_only_among_excluded_rows_is_not_counted(tmp_path: Path) -> None:
    rows = [
        corpus_row("a", "b", split="dev"),
        corpus_row("a", "b", split="dev"),
        corpus_row("c", "d"),
    ]

    assert training_from_rows(tmp_path, rows).stats.duplicate_ids == 0


@pytest.mark.parametrize("name", ["run[1]", "run*", "run?", "[a-z]", "a b"])
def test_a_path_with_glob_characters_names_that_one_file(
    tmp_path: Path, name: str
) -> None:
    directory = tmp_path / name
    directory.mkdir()
    path = write_rows(directory, "rows", numbered_rows(3))

    assert load_training_rows([path]).stats.rows_read == 3


def test_a_file_named_with_a_star_does_not_read_its_neighbours(tmp_path: Path) -> None:
    write_rows(tmp_path, "ab", [corpus_row("neighbour a", "t")])
    write_rows(tmp_path, "a1", [corpus_row("neighbour b", "t")])
    path = write_rows(tmp_path, "a*", numbered_rows(2))

    training = load_training_rows([path])

    assert sorted(training.frame["src"].to_list()) == ["s0", "s1"]


def test_a_missing_file_is_a_data_error(tmp_path: Path) -> None:
    with pytest.raises(DataError, match="not found"):
        load_training_rows([tmp_path / "missing.parquet"])


@pytest.mark.parametrize(
    "content",
    [b"", b"id,src\n1,a\n", b"PAR1" + bytes(40) + b"PAR1"],
    ids=["zero bytes", "csv text", "parquet magic only"],
)
def test_a_file_that_is_not_parquet_is_a_data_error(
    tmp_path: Path, content: bytes
) -> None:
    path = tmp_path / "rows.parquet"
    path.write_bytes(content)

    with pytest.raises(DataError, match=r"cannot read.*rows\.parquet"):
        load_training_rows([path])


def test_a_file_cut_short_is_a_data_error(tmp_path: Path) -> None:
    path = write_rows(tmp_path, "rows", numbered_rows(50))
    path.write_bytes(path.read_bytes()[:-40])

    with pytest.raises(DataError, match=r"cannot read.*rows\.parquet"):
        load_training_rows([path])


@pytest.mark.parametrize("where", ["scan", "collect"])
def test_a_reader_panic_is_a_data_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, where: str
) -> None:
    # A damaged file can make the Rust reader panic, which is not an Exception.
    path = write_rows(tmp_path, "rows", numbered_rows(2))

    def panic(*args: Any, **kwargs: Any) -> None:
        raise pl.exceptions.PanicException("assertion failed in the parquet reader")

    if where == "scan":
        monkeypatch.setattr(pl, "scan_parquet", panic)
    else:
        monkeypatch.setattr(pl.LazyFrame, "collect", panic)

    with pytest.raises(DataError, match=r"cannot read.*rows\.parquet.*assertion"):
        load_training_rows([path])


@pytest.mark.parametrize("column", ["id", "src", "origin", "split", "filter_reason"])
def test_a_file_without_a_required_column_is_a_data_error(
    tmp_path: Path, column: str
) -> None:
    path = tmp_path / "rows.parquet"
    # Not through `write_corpus`, which refuses a table without the column.
    corpus_frame(numbered_rows(2)).drop(column).write_parquet(path)

    with pytest.raises(DataError, match=rf"missing columns: {column}"):
        load_training_rows([path])


def test_no_file_and_no_rows_are_a_data_error(tmp_path: Path) -> None:
    empty = write_frame(tmp_path, "empty", corpus_frame([]))

    with pytest.raises(DataError, match="no training rows"):
        load_training_rows([])
    with pytest.raises(DataError, match="no training rows"):
        load_training_rows([empty])


def test_rows_that_are_all_excluded_are_a_data_error_that_gives_the_counts(
    tmp_path: Path,
) -> None:
    rows = [corpus_row("a", "b", split="dev"), corpus_row("c", "d", origin="ft")]

    with pytest.raises(DataError, match=r"no training rows.*2 rows read"):
        training_from_rows(tmp_path, rows)


def test_training_rows_without_any_authentic_row_are_a_data_error(
    tmp_path: Path,
) -> None:
    with pytest.raises(DataError, match="no authentic training rows.*2.*bt"):
        training_from_rows(tmp_path, numbered_rows(2, origin="bt"))


@pytest.mark.parametrize(
    ("column", "value"),
    [
        ("id", "x"),
        ("src_lang", "eng_Latn"),
        ("tgt_lang", "khk_Cyrl"),
        ("origin", "human"),
    ],
)
def test_a_null_identity_in_a_row_that_would_be_kept_is_a_data_error(
    tmp_path: Path, column: str, value: str
) -> None:
    frame = corpus_frame(numbered_rows(2)).with_columns(
        pl.Series(column, [None, value], dtype=CORPUS_SCHEMA[column])
    )
    path = write_frame(tmp_path, "rows", frame)

    with pytest.raises(DataError, match=rf"null {column}"):
        load_training_rows([path])


def test_a_null_split_is_a_row_that_is_not_for_training(tmp_path: Path) -> None:
    frame = corpus_frame(numbered_rows(2)).with_columns(
        pl.Series("split", [None, "train"], dtype=CORPUS_SCHEMA["split"])
    )
    path = write_frame(tmp_path, "rows", frame)

    stats = load_training_rows([path]).stats

    assert stats.excluded["not_train"] == 1
    assert stats.rows_kept == 1


def test_the_statistics_are_json_ready(tmp_path: Path) -> None:
    stats = training_from_rows(
        tmp_path, [*numbered_rows(2), corpus_row("a", "b", origin="ft")]
    ).stats

    document = json.loads(json.dumps(stats.to_dict()))

    assert document["rows_read"] == 3
    assert document["excluded_by_origin"]["ft"] == 1
    assert document["rows_by_corpus"] == {"eng_Latn-khk_Cyrl.auth": 2}


# --- source collision rate (R9) ----------------------------------------------------


def test_the_source_collision_rate_counts_rows_whose_source_has_another_target_language() -> (
    None
):
    # A has three target languages: all three rows collide. B twice into one language,
    # and C once, never collide: 3 of 6.
    frame = pl.DataFrame(
        {
            "src": ["A", "A", "A", "B", "B", "C"],
            "tgt_lang": [
                "khk_Cyrl",
                "rus_Cyrl",
                "eng_Latn",
                "khk_Cyrl",
                "khk_Cyrl",
                "khk_Cyrl",
            ],
        }
    )

    assert source_collision_rate(frame) == 0.5


def test_the_source_collision_rate_of_no_rows_is_zero() -> None:
    frame = pl.DataFrame(
        {"src": [], "tgt_lang": []}, schema={"src": pl.String, "tgt_lang": pl.String}
    )

    assert source_collision_rate(frame) == 0.0


def test_the_statistics_carry_the_collision_rate_of_the_kept_rows_before_any_subsample(
    tmp_path: Path,
) -> None:
    # The same source into two languages (one of them a bt row), a repeat into one,
    # and an excluded forward-translated row that would collide with the unique one.
    rows = [
        corpus_row("shared", "a", tgt_lang="khk_Cyrl"),
        corpus_row("shared", "b", tgt_lang="rus_Cyrl", origin="bt"),
        corpus_row("twice", "c"),
        corpus_row("twice", "d"),
        corpus_row("alone", "e"),
        corpus_row("alone", "f", tgt_lang="rus_Cyrl", origin="ft"),
    ]

    stats = training_from_rows(tmp_path, rows).stats

    assert stats.rows_kept == 5
    assert stats.source_collision_rate == pytest.approx(2 / 5)


# --- load_dev_rows -----------------------------------------------------------------


def _dev(path: Path, **kwargs: Any) -> Any:
    kwargs.setdefault("max_lines", 100)
    kwargs.setdefault("directions", [EN_MN])
    return load_dev_rows([path], **kwargs)


def test_the_dev_rows_of_a_direction_are_those_with_the_smallest_ids(
    tmp_path: Path,
) -> None:
    rows = [corpus_row(f"dev {i}", f"t{i}", split="dev") for i in range(5)]
    path = write_rows(tmp_path, "dev", rows, ids=["d4", "d2", "d0", "d3", "d1"])

    dev = _dev(path, max_lines=3)

    assert dev.sets[EN_MN].ids == ("d0", "d1", "d2")
    assert dev.sets[EN_MN].src == ("dev 2", "dev 4", "dev 1")
    assert dev.sets[EN_MN].tgt == ("t2", "t4", "t1")
    assert dev.stats.lines_by_direction == {EN_MN: 3}
    assert dev.stats.available_by_direction == {EN_MN: 5}


def test_equal_ids_are_ordered_by_source_then_target(tmp_path: Path) -> None:
    rows = [
        corpus_row("b", "1", split="dev"),
        corpus_row("a", "2", split="dev"),
        corpus_row("a", "1", split="dev"),
    ]
    path = write_rows(tmp_path, "dev", rows, ids=["same", "same", "same"])

    dev = _dev(path)

    assert list(zip(dev.sets[EN_MN].src, dev.sets[EN_MN].tgt, strict=True)) == [
        ("a", "1"),
        ("a", "2"),
        ("b", "1"),
    ]


def test_a_dev_set_is_independent_of_the_order_of_the_rows(tmp_path: Path) -> None:
    rows = [corpus_row(f"dev {i}", f"t{i}", split="dev") for i in range(8)]
    shuffled = list(rows)
    random.Random(3).shuffle(shuffled)
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    first = write_rows(tmp_path / "a", "dev", rows)
    second = write_rows(tmp_path / "b", "dev", shuffled)

    assert _dev(first, max_lines=5).sets == _dev(second, max_lines=5).sets


def test_the_language_codes_and_the_direction_of_a_dev_set_are_recorded(
    tmp_path: Path,
) -> None:
    path = write_rows(tmp_path, "dev", [corpus_row("a", "b", split="dev")])

    dev_set = _dev(path).sets[EN_MN]

    assert (dev_set.direction, dev_set.src_lang, dev_set.tgt_lang) == (
        EN_MN,
        "eng_Latn",
        "khk_Cyrl",
    )


def test_dev_rows_that_cannot_be_used_are_counted_by_reason(tmp_path: Path) -> None:
    rows = [
        corpus_row("kept", "t", split="dev"),
        corpus_row("train row", "t"),
        corpus_row("test row", "t", split="test"),
        corpus_row("filtered", "t", split="dev"),
        corpus_row("", "t", split="dev"),
        corpus_row("a", "  ", split="dev"),
        corpus_row("line" + chr(0x2028) + "break", "t", split="dev"),
        corpus_row("s", "line" + chr(0x2029) + "break", split="dev"),
        # Blank and a line break: the first rule that matches, empty, decides.
        corpus_row("\n", "t", split="dev"),
        corpus_row("tab\tkept", "t", split="dev"),
    ]
    reasons: list[str | None] = [None] * len(rows)
    reasons[3] = "lid_tgt"
    path = write_rows(tmp_path, "dev", rows, filter_reasons=reasons)

    stats = _dev(path).stats

    assert stats.rows_read == 10
    assert stats.excluded == {"not_dev": 2, "filtered": 1, "empty": 3, "line_break": 2}
    assert stats.rows_usable == 2
    assert stats.rows_read == stats.rows_usable + sum(stats.excluded.values())
    assert stats.available_by_direction == {EN_MN: 2}


def test_dev_rows_of_a_direction_without_training_rows_are_ignored_and_counted(
    tmp_path: Path,
) -> None:
    rows = [
        corpus_row("a", "b", split="dev"),
        corpus_row("c", "d", src_lang="rus_Cyrl", split="dev"),
        corpus_row("e", "f", src_lang="rus_Cyrl", split="dev"),
        corpus_row("g", "h", src_lang="xxx_Latn", split="dev"),
    ]
    path = write_rows(tmp_path, "dev", rows)

    dev = _dev(path)

    assert list(dev.sets) == [EN_MN]
    assert dev.stats.ignored_by_direction == {
        RU_MN: 2,
        "xxx_Latn-khk_Cyrl": 1,
    }
    assert dev.stats.rows_usable == 4


def test_a_direction_with_training_rows_and_no_dev_rows_is_a_data_error(
    tmp_path: Path,
) -> None:
    path = write_rows(tmp_path, "dev", [corpus_row("a", "b", split="dev")])

    with pytest.raises(DataError, match=rf"no usable dev rows.*{RU_MN}"):
        _dev(path, directions=[EN_MN, RU_MN])


def test_a_dev_set_whose_rows_are_all_excluded_is_empty_and_a_data_error(
    tmp_path: Path,
) -> None:
    path = write_rows(
        tmp_path,
        "dev",
        [corpus_row("", "b", split="dev"), corpus_row("c\n", "d", split="dev")],
    )

    with pytest.raises(DataError, match=rf"no usable dev rows.*{EN_MN}"):
        _dev(path)


def test_a_dev_set_never_reads_a_test_row(tmp_path: Path) -> None:
    path = write_rows(tmp_path, "dev", [corpus_row("a", "b", split="test")])

    with pytest.raises(DataError, match="no usable dev rows"):
        _dev(path)


@pytest.mark.parametrize("max_lines", [0, -1])
def test_max_lines_must_be_positive(tmp_path: Path, max_lines: int) -> None:
    path = write_rows(tmp_path, "dev", [corpus_row("a", "b", split="dev")])

    with pytest.raises(ValueError, match="max_lines"):
        _dev(path, max_lines=max_lines)


def test_dev_rows_of_several_files_with_glob_characters_in_their_paths(
    tmp_path: Path,
) -> None:
    (tmp_path / "d[1]").mkdir()
    first = write_rows(tmp_path / "d[1]", "dev", [corpus_row("a", "b", split="dev")])
    second = write_rows(tmp_path, "dev*", [corpus_row("c", "d", split="dev")])

    dev = load_dev_rows([first, second], max_lines=10, directions=[EN_MN])

    assert sorted(dev.sets[EN_MN].src) == ["a", "c"]


def test_no_dev_file_means_no_usable_dev_rows() -> None:
    with pytest.raises(DataError, match="no usable dev rows"):
        load_dev_rows([], max_lines=10, directions=[EN_MN])


def test_an_unreadable_dev_file_is_a_data_error(tmp_path: Path) -> None:
    path = tmp_path / "dev.parquet"
    path.write_bytes(b"not parquet")

    with pytest.raises(DataError, match=r"cannot read.*dev\.parquet"):
        _dev(path)


def test_a_dev_row_without_an_id_is_a_data_error(tmp_path: Path) -> None:
    frame = corpus_frame([corpus_row("a", "b", split="dev")]).with_columns(
        pl.Series("id", [None], dtype=pl.String)
    )
    path = write_frame(tmp_path, "dev", frame)

    with pytest.raises(DataError, match="null id"):
        _dev(path)


def test_a_null_split_is_a_row_that_is_not_dev(tmp_path: Path) -> None:
    frame = corpus_frame([corpus_row("a", "1", split="dev")] * 2).with_columns(
        pl.Series("split", [None, "dev"], dtype=CORPUS_SCHEMA["split"])
    )
    path = write_frame(tmp_path, "rows", frame)

    stats = _dev(path).stats

    assert stats.rows_read == 2
    assert stats.excluded["not_dev"] == 1
    assert stats.rows_usable == 1


def _collected_heights(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    """The number of rows of every lazy frame polars collects from now on."""
    heights: list[int] = []
    original = pl.LazyFrame.collect

    def recording(self: pl.LazyFrame, *args: Any, **kwargs: Any) -> Any:
        collected = original(self, *args, **kwargs)
        heights.append(getattr(collected, "height", 0))
        return collected

    monkeypatch.setattr(pl.LazyFrame, "collect", recording)
    return heights


def test_a_file_with_no_dev_rows_never_holds_its_other_rows_in_memory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rows = [*numbered_rows(500), corpus_row("test", "t", split="test")]
    path = write_rows(tmp_path, "rows", rows)
    heights = _collected_heights(monkeypatch)

    dev = load_dev_rows([path], max_lines=10, directions=[])

    assert dev.stats.rows_read == 501
    assert dev.stats.excluded["not_dev"] == 501
    assert dev.stats.rows_usable == 0
    assert max(heights) <= 1


def test_only_the_dev_rows_of_a_mixed_file_are_held_in_memory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rows = [
        *numbered_rows(500),
        corpus_row("d1", "t", split="dev"),
        corpus_row("d2", "t", split="dev"),
    ]
    path = write_rows(tmp_path, "rows", rows)
    heights = _collected_heights(monkeypatch)

    dev = _dev(path)

    assert dev.stats.rows_read == 502
    assert dev.stats.excluded["not_dev"] == 500
    assert dev.stats.rows_usable == 2
    assert max(heights) <= 2


def test_only_the_training_rows_of_a_mixed_file_are_held_in_memory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rows = [
        *numbered_rows(3),
        *(corpus_row(f"dev {i}", "t", split="dev") for i in range(400)),
        *(corpus_row(f"test {i}", "t", split="test") for i in range(400)),
    ]
    path = write_rows(tmp_path, "rows", rows)
    heights = _collected_heights(monkeypatch)

    stats = load_training_rows([path]).stats

    assert stats.rows_read == 803
    assert stats.excluded["not_train"] == 800
    assert stats.rows_kept == 3
    assert max(heights) <= 3


# --- the dev rows that are also training rows --------------------------------------


def _write_train_and_dev(
    tmp_path: Path, train: Sequence[dict[str, str]], dev: Sequence[dict[str, str]]
) -> tuple[TrainingRows, Path]:
    training = training_from_rows(tmp_path / "train", train)
    return training, write_rows(tmp_path, "dev", dev)


def test_a_dev_line_whose_id_is_a_kept_training_row_is_counted(tmp_path: Path) -> None:
    # The id is derived from the languages and both texts, so the same sentence pair
    # in a training row and a dev row has one id.
    training, dev_path = _write_train_and_dev(
        tmp_path,
        [corpus_row("s0", "t0"), corpus_row("s1", "t1"), corpus_row("s2", "t2")],
        [
            corpus_row("s1", "t1", split="dev"),
            corpus_row("s2", "t2", split="dev"),
            corpus_row("only dev", "t", split="dev"),
        ],
    )

    dev = _dev(dev_path, training=training)

    assert dev.stats.dev_in_train == 2
    assert dev.stats.lines_by_direction == {EN_MN: 3}


def test_no_dev_line_in_the_training_rows_is_a_count_of_zero(tmp_path: Path) -> None:
    training, dev_path = _write_train_and_dev(
        tmp_path, [corpus_row("s0", "t0")], [corpus_row("d0", "t", split="dev")]
    )

    assert _dev(dev_path, training=training).stats.dev_in_train == 0


def test_a_dev_row_that_is_a_training_row_of_the_same_file_is_counted(
    tmp_path: Path,
) -> None:
    rows = [corpus_row("s0", "t0"), corpus_row("s0", "t0", split="dev")]
    path = write_rows(tmp_path, "rows", rows)

    dev = load_dev_rows(
        [path],
        max_lines=10,
        directions=[EN_MN],
        training=load_training_rows([path]),
    )

    assert dev.stats.dev_in_train == 1


def test_a_back_translated_training_row_counts_like_any_other_kept_row(
    tmp_path: Path,
) -> None:
    training, dev_path = _write_train_and_dev(
        tmp_path,
        [corpus_row("auth", "t"), corpus_row("bt0", "t0", origin="bt")],
        [corpus_row("bt0", "t0", split="dev")],
    )

    assert _dev(dev_path, training=training).stats.dev_in_train == 1


def test_a_training_row_that_was_left_out_does_not_count(tmp_path: Path) -> None:
    training, dev_path = _write_train_and_dev(
        tmp_path,
        [
            corpus_row("auth", "t"),
            corpus_row("ft row", "t1", origin="ft"),
            corpus_row("", "t2"),
        ],
        [
            corpus_row("ft row", "t1", split="dev"),
            corpus_row("", "t2", split="dev"),
        ],
    )

    dev = _dev(dev_path, training=training)

    # The first dev line shares its id with the forward-translated row, which does not
    # train; the second is blank and not a dev line at all.
    assert dev.stats.lines_by_direction == {EN_MN: 1}
    assert dev.stats.dev_in_train == 0


def test_only_the_dev_lines_that_are_returned_are_counted(tmp_path: Path) -> None:
    dev_rows = [corpus_row(f"d{i}", "t", split="dev") for i in range(3)]
    dev_path = write_rows(tmp_path, "dev", dev_rows, ids=["a", "b", "c"])
    training = training_from_rows(
        tmp_path / "train",
        [corpus_row("s0", "t0"), corpus_row("s1", "t1")],
        ids=["a", "c"],
    )

    dev = _dev(dev_path, max_lines=2, training=training)

    # The first two by id are "a" and "b"; "c" is a training id but is not returned.
    assert dev.sets[EN_MN].ids == ("a", "b")
    assert dev.stats.dev_in_train == 1


def test_a_dev_id_repeated_in_the_dev_lines_counts_each_line(tmp_path: Path) -> None:
    training = training_from_rows(tmp_path / "train", [corpus_row("s0", "t0")])
    dev_path = write_rows(
        tmp_path,
        "dev",
        [corpus_row("s0", "t0", split="dev"), corpus_row("s0", "t0", split="dev")],
    )

    assert _dev(dev_path, training=training).stats.dev_in_train == 2


def test_without_training_rows_to_compare_the_count_is_not_known(
    tmp_path: Path,
) -> None:
    path = write_rows(tmp_path, "dev", [corpus_row("a", "b", split="dev")])

    assert _dev(path).stats.dev_in_train is None


def test_the_dev_in_train_count_is_in_the_json_document(tmp_path: Path) -> None:
    training, dev_path = _write_train_and_dev(
        tmp_path, [corpus_row("s0", "t0")], [corpus_row("s0", "t0", split="dev")]
    )

    document = json.loads(json.dumps(_dev(dev_path, training=training).stats.to_dict()))

    assert document["dev_in_train"] == 1
    assert (
        json.loads(json.dumps(_dev(dev_path).stats.to_dict()))["dev_in_train"] is None
    )


def test_the_dev_statistics_are_json_ready(tmp_path: Path) -> None:
    path = write_rows(tmp_path, "dev", [corpus_row("a", "b", split="dev")])

    document = json.loads(json.dumps(_dev(path).stats.to_dict()))

    assert document["lines_by_direction"] == {EN_MN: 1}


# --- the text rules the document windows share -------------------------------------


@pytest.mark.parametrize(
    "text, blank",
    [
        (None, True),
        ("", True),
        (" ", True),
        (" \t\n ", True),
        (chr(0x3000), True),
        (chr(0x1C), True),
        ("a", False),
        (" a ", False),
        ("0", False),
    ],
)
def test_blank_text_is_null_empty_or_only_whitespace(
    text: str | None, blank: bool
) -> None:
    frame = pl.DataFrame({"text": [text]}, schema={"text": pl.String})

    assert frame.select(blank_text("text")).item() is blank


@pytest.mark.parametrize("separator", LINE_BREAK_CHARACTERS)
def test_has_line_break_sees_every_character_that_splitlines_splits_at(
    separator: str,
) -> None:
    frame = pl.DataFrame({"text": [f"a{separator}b", "a b", "a" + chr(0x2003)]})

    assert frame.select(has_line_break("text"))["text"].to_list() == [
        True,
        False,
        False,
    ]
