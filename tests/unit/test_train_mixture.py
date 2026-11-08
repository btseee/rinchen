"""Tests for `rinchen.train.mixture`: which rows train and which are counted out,
the development sets, the corpus text files with the bt subsample, the Eole `data`
section and the pre-tokenised validation corpus.

Counts and file contents below are worked out by hand from the rules in the module
docstring and the D7 spec (R2, R4, R6, R7, R9), not from the code. The properties
over random rows are in `tests/property/test_train_mixture_property.py`.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

import polars as pl
import pytest
import sentencepiece

from rinchen.config.loader import REPO_ROOT
from rinchen.corpus.schema import CORPUS_SCHEMA, corpus_frame
from rinchen.corpus.writers import write_corpus
from rinchen.errors import ConfigError, DataError
from rinchen.train.eole_cfg import merge, read_config, validate_train_config
from rinchen.train.mixture import (
    LINE_BREAK_CHARACTERS,
    BuiltCorpora,
    CorpusFiles,
    DevSet,
    TrainingRows,
    ValidPaths,
    build_corpora,
    corpus_sizes,
    data_section,
    dev_line_ranges,
    load_dev_rows,
    load_training_rows,
    source_collision_rate,
    write_validation,
)
from rinchen.train.sampling import CorpusSize, integer_weights, mixing_probabilities
from tests.support.tiny_parallel import (
    build_toy,
    train_toy_spm,
    write_corpus_parquet,
)

EN_MN = "eng_Latn-khk_Cyrl"
MN_EN = "khk_Cyrl-eng_Latn"
RU_MN = "rus_Cyrl-khk_Cyrl"
MN_RU = "khk_Cyrl-rus_Cyrl"
TINY_BASE = REPO_ROOT / "configs" / "eole" / "multilingual_tiny.yaml"

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


@pytest.fixture(scope="module")
def toy_spm(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return train_toy_spm(tmp_path_factory.mktemp("mixture_spm"))


def _row(
    src: str = "Hello",
    tgt: str = "Сайн",
    *,
    src_lang: str = "eng_Latn",
    tgt_lang: str = "khk_Cyrl",
    origin: str = "human",
    split: str = "train",
) -> dict[str, str]:
    return {
        "src_lang": src_lang,
        "tgt_lang": tgt_lang,
        "src": src,
        "tgt": tgt,
        "origin": origin,
        "corpus": "test",
        "split": split,
    }


def _numbered(count: int, **kwargs: str) -> list[dict[str, str]]:
    """`count` rows with the sources `s0`, `s1`, ... (`bt0`, `web0` ... when an
    origin other than human is given, so no two origins share a row id)."""
    origin = kwargs.get("origin", "human")
    stem = "s" if origin == "human" else origin
    return [_row(f"{stem}{index}", f"t{index}", **kwargs) for index in range(count)]


def _write(
    directory: Path,
    name: str,
    rows: Sequence[Mapping[str, Any]],
    *,
    ids: Sequence[str] | None = None,
    filter_reasons: Sequence[str | None] | None = None,
) -> Path:
    frame = corpus_frame(rows)
    if ids is not None:
        frame = frame.with_columns(pl.Series("id", list(ids), dtype=pl.String))
    if filter_reasons is not None:
        frame = frame.with_columns(
            pl.Series("filter_reason", list(filter_reasons), dtype=pl.String)
        )
    return _write_frame(directory, name, frame)


def _write_frame(directory: Path, name: str, frame: pl.DataFrame) -> Path:
    path = directory / f"{name}.parquet"
    write_corpus(frame, path)
    return path


def _training(
    directory: Path, rows: Sequence[Mapping[str, Any]], **kwargs: Any
) -> TrainingRows:
    directory.mkdir(parents=True, exist_ok=True)
    return load_training_rows([_write(directory, "rows", rows, **kwargs)])


def _corpus(built: BuiltCorpora, name: str) -> CorpusFiles:
    (found,) = [corpus for corpus in built.corpora if corpus.name == name]
    return found


def _lines(path: Path) -> list[str]:
    text = path.read_bytes().decode("utf-8")
    assert text.endswith("\n")
    return text.split("\n")[:-1]


# --- load_training_rows: which rows train ------------------------------------------


def test_only_unfiltered_authentic_training_rows_are_kept_and_the_rest_counted_by_reason(
    tmp_path: Path,
) -> None:
    rows = [
        _row("kept 1", "t1"),
        _row("kept 2", "t2", origin="web"),
        _row("dev", "t", split="dev"),
        _row("test", "t", split="test"),
        _row("holdout", "t", split="holdout"),
        _row("filtered", "t"),
        _row("ft", "t", origin="ft"),
        _row("llm", "t", origin="synth_llm"),
        _row("nmt", "t", origin="synth_nmt"),
        _row("", "t"),
        _row("blank target", "   "),
        _row("two\nlines", "t"),
    ]
    reasons: list[str | None] = [None] * len(rows)
    reasons[5] = "lid_src"

    training = _training(tmp_path, rows, filter_reasons=reasons)

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
        _row("a", "b"),
        _row("c", "d", split="dev"),
        _row("e", "f", origin="ft"),
        _row("", ""),
        _row("g\r", "h"),
        _row("i", "j", origin="bt"),
        _row("k", "l"),
    ]
    reasons: list[str | None] = [None, None, None, None, None, None, "numbers"]

    stats = _training(tmp_path, rows, filter_reasons=reasons).stats

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
    row = _row("text", "t") | first_match
    reasons = ["lid_src"] if other == "not_train" else [None]

    stats = _training(
        tmp_path, [row, _row("kept", "t")], filter_reasons=[*reasons, None]
    ).stats

    assert stats.excluded[other] == 1
    assert sum(stats.excluded.values()) == 1


def test_a_filtered_row_of_an_excluded_origin_is_counted_as_filtered_only(
    tmp_path: Path,
) -> None:
    stats = _training(
        tmp_path,
        [_row("a", "b", origin="ft"), _row("kept", "t")],
        filter_reasons=["lid_src", None],
    ).stats

    assert stats.excluded["filtered"] == 1
    assert stats.excluded["excluded_origin"] == 0
    assert stats.excluded_by_origin == {"ft": 0, "synth_llm": 0, "synth_nmt": 0}


def test_an_excluded_origin_is_counted_by_origin_only_among_training_rows(
    tmp_path: Path,
) -> None:
    rows = [
        _row("a", "b", origin="ft"),
        _row("c", "d", origin="ft", split="dev"),
        _row("e", "f", origin="synth_llm", split="test"),
        _row("kept", "t"),
    ]

    stats = _training(tmp_path, rows).stats

    assert stats.excluded_by_origin == {"ft": 1, "synth_llm": 0, "synth_nmt": 0}
    assert stats.excluded["not_train"] == 2


@pytest.mark.parametrize("name", list(_LINE_BREAKS))
@pytest.mark.parametrize("side", ["src", "tgt"])
def test_a_line_break_character_in_either_text_excludes_the_row(
    tmp_path: Path, name: str, side: str
) -> None:
    text = f"left{chr(_LINE_BREAKS[name])}right"
    row = _row(text, "t") if side == "src" else _row("s", text)

    stats = _training(tmp_path, [row, _row("kept", "t")]).stats

    assert stats.excluded["line_break"] == 1
    assert stats.rows_kept == 1


def test_a_carriage_return_and_line_feed_pair_is_one_excluded_row(
    tmp_path: Path,
) -> None:
    stats = _training(tmp_path, [_row("a\r\nb", "t"), _row("kept", "t")]).stats

    assert stats.excluded["line_break"] == 1


def test_a_tab_a_nul_and_a_zero_width_space_are_kept_in_a_text(tmp_path: Path) -> None:
    rows = [
        _row("a\tb", "t1"),
        _row("a" + chr(0) + "b", "t2"),
        _row("a" + chr(0x200B) + "b", "t3"),
        _row("a" + chr(0x1F) + "b", "t4"),
    ]

    stats = _training(tmp_path, rows).stats

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
    rows = [_row(char, "t") for char in space] + [
        _row("s", char, origin="web") for char in not_space
    ]

    stats = _training(tmp_path, rows).stats

    assert stats.excluded["empty"] == len(space)
    assert stats.rows_kept == len(not_space)
    assert stats.excluded["line_break"] == 0


def test_a_null_text_is_an_empty_one(tmp_path: Path) -> None:
    frame = corpus_frame([_row("a", "b"), _row("kept", "t")]).with_columns(
        pl.Series("src", [None, "kept"], dtype=pl.String)
    )

    stats = load_training_rows([_write_frame(tmp_path, "nulls", frame)]).stats

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
    rows = [_row("kept", "t", origin=origin), _row("auth", "t", origin="human")]

    training = _training(tmp_path, rows)

    kept = training.frame.filter(pl.col("src") == "kept")
    assert kept["group"].to_list() == [group]
    assert kept["direction"].to_list() == [EN_MN]
    assert kept["origin"].to_list() == [origin]


def test_the_training_rows_are_counted_by_corpus_name(tmp_path: Path) -> None:
    rows = [
        *_numbered(3),
        *_numbered(2, origin="bt"),
        *[_row(f"r{i}", f"t{i}", src_lang="rus_Cyrl") for i in range(4)],
    ]

    training = _training(tmp_path, rows)

    assert training.stats.rows_by_corpus == {
        "eng_Latn-khk_Cyrl.auth": 3,
        "eng_Latn-khk_Cyrl.bt": 2,
        "rus_Cyrl-khk_Cyrl.auth": 4,
    }
    assert training.directions == (EN_MN, RU_MN)


def test_a_direction_that_has_only_bt_rows_is_not_a_training_direction(
    tmp_path: Path,
) -> None:
    rows = [*_numbered(2), *_numbered(2, src_lang="rus_Cyrl", origin="bt")]

    training = _training(tmp_path, rows)

    assert training.directions == (EN_MN,)
    assert training.stats.rows_by_corpus[f"{RU_MN}.bt"] == 2


def test_rows_of_several_files_are_read_and_counted_together(tmp_path: Path) -> None:
    first = _write(tmp_path, "first", [*_numbered(3), _row("dev", "t", split="dev")])
    second = _write(tmp_path, "second", [_row("x", "y", src_lang="rus_Cyrl")])

    stats = load_training_rows([first, second]).stats

    assert stats.rows_read == 5
    assert stats.rows_kept == 4
    assert stats.excluded["not_train"] == 1


def test_a_language_outside_the_registry_is_a_config_error_that_names_it(
    tmp_path: Path,
) -> None:
    rows = [*_numbered(2), _row("a", "b", tgt_lang="xxx_Latn")]

    with pytest.raises(ConfigError, match="xxx_Latn"):
        _training(tmp_path, rows)


def test_a_language_outside_the_registry_in_an_excluded_row_is_not_checked(
    tmp_path: Path,
) -> None:
    rows = [*_numbered(2), _row("a", "b", src_lang="xxx_Latn", split="dev")]

    assert _training(tmp_path, rows).stats.rows_kept == 2


def test_the_same_id_in_two_files_keeps_both_rows_and_counts_the_repeat(
    tmp_path: Path,
) -> None:
    first = _write(tmp_path, "first", [_row("same", "text"), _row("other", "text")])
    second = _write(tmp_path, "second", [_row("same", "text")])

    training = load_training_rows([first, second])

    assert training.stats.rows_kept == 3
    assert training.stats.duplicate_ids == 1
    assert training.frame.height == 3


def test_a_triple_id_counts_as_two_repeats(tmp_path: Path) -> None:
    rows = [_row("a", "b"), _row("a", "b"), _row("a", "b"), _row("c", "d")]

    assert _training(tmp_path, rows).stats.duplicate_ids == 2


def test_an_id_repeated_only_among_excluded_rows_is_not_counted(tmp_path: Path) -> None:
    rows = [_row("a", "b", split="dev"), _row("a", "b", split="dev"), _row("c", "d")]

    assert _training(tmp_path, rows).stats.duplicate_ids == 0


@pytest.mark.parametrize("name", ["run[1]", "run*", "run?", "[a-z]", "a b"])
def test_a_path_with_glob_characters_names_that_one_file(
    tmp_path: Path, name: str
) -> None:
    directory = tmp_path / name
    directory.mkdir()
    path = _write(directory, "rows", _numbered(3))

    assert load_training_rows([path]).stats.rows_read == 3


def test_a_file_named_with_a_star_does_not_read_its_neighbours(tmp_path: Path) -> None:
    _write(tmp_path, "ab", [_row("neighbour a", "t")])
    _write(tmp_path, "a1", [_row("neighbour b", "t")])
    path = _write(tmp_path, "a*", _numbered(2))

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
    path = _write(tmp_path, "rows", _numbered(50))
    path.write_bytes(path.read_bytes()[:-40])

    with pytest.raises(DataError, match=r"cannot read.*rows\.parquet"):
        load_training_rows([path])


@pytest.mark.parametrize("where", ["scan", "collect"])
def test_a_reader_panic_is_a_data_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, where: str
) -> None:
    # A damaged file can make the Rust reader panic, which is not an Exception.
    path = _write(tmp_path, "rows", _numbered(2))

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
    corpus_frame(_numbered(2)).drop(column).write_parquet(path)

    with pytest.raises(DataError, match=rf"missing columns: {column}"):
        load_training_rows([path])


def test_no_file_and_no_rows_are_a_data_error(tmp_path: Path) -> None:
    empty = _write_frame(tmp_path, "empty", corpus_frame([]))

    with pytest.raises(DataError, match="no training rows"):
        load_training_rows([])
    with pytest.raises(DataError, match="no training rows"):
        load_training_rows([empty])


def test_rows_that_are_all_excluded_are_a_data_error_that_gives_the_counts(
    tmp_path: Path,
) -> None:
    rows = [_row("a", "b", split="dev"), _row("c", "d", origin="ft")]

    with pytest.raises(DataError, match=r"no training rows.*2 rows read"):
        _training(tmp_path, rows)


def test_training_rows_without_any_authentic_row_are_a_data_error(
    tmp_path: Path,
) -> None:
    with pytest.raises(DataError, match="no authentic training rows.*2.*bt"):
        _training(tmp_path, _numbered(2, origin="bt"))


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
    frame = corpus_frame(_numbered(2)).with_columns(
        pl.Series(column, [None, value], dtype=CORPUS_SCHEMA[column])
    )
    path = _write_frame(tmp_path, "rows", frame)

    with pytest.raises(DataError, match=rf"null {column}"):
        load_training_rows([path])


def test_a_null_split_is_a_row_that_is_not_for_training(tmp_path: Path) -> None:
    frame = corpus_frame(_numbered(2)).with_columns(
        pl.Series("split", [None, "train"], dtype=CORPUS_SCHEMA["split"])
    )
    path = _write_frame(tmp_path, "rows", frame)

    stats = load_training_rows([path]).stats

    assert stats.excluded["not_train"] == 1
    assert stats.rows_kept == 1


def test_the_statistics_are_json_ready(tmp_path: Path) -> None:
    stats = _training(tmp_path, [*_numbered(2), _row("a", "b", origin="ft")]).stats

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
        _row("shared", "a", tgt_lang="khk_Cyrl"),
        _row("shared", "b", tgt_lang="rus_Cyrl", origin="bt"),
        _row("twice", "c"),
        _row("twice", "d"),
        _row("alone", "e"),
        _row("alone", "f", tgt_lang="rus_Cyrl", origin="ft"),
    ]

    stats = _training(tmp_path, rows).stats

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
    rows = [_row(f"dev {i}", f"t{i}", split="dev") for i in range(5)]
    path = _write(tmp_path, "dev", rows, ids=["d4", "d2", "d0", "d3", "d1"])

    dev = _dev(path, max_lines=3)

    assert dev.sets[EN_MN].ids == ("d0", "d1", "d2")
    assert dev.sets[EN_MN].src == ("dev 2", "dev 4", "dev 1")
    assert dev.sets[EN_MN].tgt == ("t2", "t4", "t1")
    assert dev.stats.lines_by_direction == {EN_MN: 3}
    assert dev.stats.available_by_direction == {EN_MN: 5}


def test_equal_ids_are_ordered_by_source_then_target(tmp_path: Path) -> None:
    rows = [
        _row("b", "1", split="dev"),
        _row("a", "2", split="dev"),
        _row("a", "1", split="dev"),
    ]
    path = _write(tmp_path, "dev", rows, ids=["same", "same", "same"])

    dev = _dev(path)

    assert list(zip(dev.sets[EN_MN].src, dev.sets[EN_MN].tgt, strict=True)) == [
        ("a", "1"),
        ("a", "2"),
        ("b", "1"),
    ]


def test_a_dev_set_is_independent_of_the_order_of_the_rows(tmp_path: Path) -> None:
    rows = [_row(f"dev {i}", f"t{i}", split="dev") for i in range(8)]
    shuffled = list(rows)
    random.Random(3).shuffle(shuffled)
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    first = _write(tmp_path / "a", "dev", rows)
    second = _write(tmp_path / "b", "dev", shuffled)

    assert _dev(first, max_lines=5).sets == _dev(second, max_lines=5).sets


def test_the_language_codes_and_the_direction_of_a_dev_set_are_recorded(
    tmp_path: Path,
) -> None:
    path = _write(tmp_path, "dev", [_row("a", "b", split="dev")])

    dev_set = _dev(path).sets[EN_MN]

    assert (dev_set.direction, dev_set.src_lang, dev_set.tgt_lang) == (
        EN_MN,
        "eng_Latn",
        "khk_Cyrl",
    )


def test_dev_rows_that_cannot_be_used_are_counted_by_reason(tmp_path: Path) -> None:
    rows = [
        _row("kept", "t", split="dev"),
        _row("train row", "t"),
        _row("test row", "t", split="test"),
        _row("filtered", "t", split="dev"),
        _row("", "t", split="dev"),
        _row("a", "  ", split="dev"),
        _row("line" + chr(0x2028) + "break", "t", split="dev"),
        _row("s", "line" + chr(0x2029) + "break", split="dev"),
        # Blank and a line break: the first rule that matches, empty, decides.
        _row("\n", "t", split="dev"),
        _row("tab\tkept", "t", split="dev"),
    ]
    reasons: list[str | None] = [None] * len(rows)
    reasons[3] = "lid_tgt"
    path = _write(tmp_path, "dev", rows, filter_reasons=reasons)

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
        _row("a", "b", split="dev"),
        _row("c", "d", src_lang="rus_Cyrl", split="dev"),
        _row("e", "f", src_lang="rus_Cyrl", split="dev"),
        _row("g", "h", src_lang="xxx_Latn", split="dev"),
    ]
    path = _write(tmp_path, "dev", rows)

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
    path = _write(tmp_path, "dev", [_row("a", "b", split="dev")])

    with pytest.raises(DataError, match=rf"no usable dev rows.*{RU_MN}"):
        _dev(path, directions=[EN_MN, RU_MN])


def test_a_dev_set_whose_rows_are_all_excluded_is_empty_and_a_data_error(
    tmp_path: Path,
) -> None:
    path = _write(
        tmp_path, "dev", [_row("", "b", split="dev"), _row("c\n", "d", split="dev")]
    )

    with pytest.raises(DataError, match=rf"no usable dev rows.*{EN_MN}"):
        _dev(path)


def test_a_dev_set_never_reads_a_test_row(tmp_path: Path) -> None:
    path = _write(tmp_path, "dev", [_row("a", "b", split="test")])

    with pytest.raises(DataError, match="no usable dev rows"):
        _dev(path)


@pytest.mark.parametrize("max_lines", [0, -1])
def test_max_lines_must_be_positive(tmp_path: Path, max_lines: int) -> None:
    path = _write(tmp_path, "dev", [_row("a", "b", split="dev")])

    with pytest.raises(ValueError, match="max_lines"):
        _dev(path, max_lines=max_lines)


def test_dev_rows_of_several_files_with_glob_characters_in_their_paths(
    tmp_path: Path,
) -> None:
    (tmp_path / "d[1]").mkdir()
    first = _write(tmp_path / "d[1]", "dev", [_row("a", "b", split="dev")])
    second = _write(tmp_path, "dev*", [_row("c", "d", split="dev")])

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
    frame = corpus_frame([_row("a", "b", split="dev")]).with_columns(
        pl.Series("id", [None], dtype=pl.String)
    )
    path = _write_frame(tmp_path, "dev", frame)

    with pytest.raises(DataError, match="null id"):
        _dev(path)


def test_the_dev_statistics_are_json_ready(tmp_path: Path) -> None:
    path = _write(tmp_path, "dev", [_row("a", "b", split="dev")])

    document = json.loads(json.dumps(_dev(path).stats.to_dict()))

    assert document["lines_by_direction"] == {EN_MN: 1}


# --- build_corpora: the text files -------------------------------------------------


def test_each_corpus_is_a_pair_of_text_files_named_after_its_direction_and_group(
    tmp_path: Path,
) -> None:
    rows = [
        *_numbered(2),
        *_numbered(2, src_lang="khk_Cyrl", tgt_lang="rus_Cyrl", origin="web"),
    ]
    training = _training(tmp_path / "in", rows)

    built = build_corpora(training, bt_ratio=0.0, seed=1, data_dir=tmp_path / "data")

    assert [corpus.name for corpus in built.corpora] == [
        "eng_Latn-khk_Cyrl.auth",
        "khk_Cyrl-rus_Cyrl.auth",
    ]
    first = built.corpora[0]
    assert first.src_path == tmp_path / "data" / "auth" / "eng_Latn-khk_Cyrl.auth.src"
    assert first.tgt_path == tmp_path / "data" / "auth" / "eng_Latn-khk_Cyrl.auth.tgt"
    assert (first.direction, first.group) == (EN_MN, "auth")
    assert (first.src_lang, first.tgt_lang) == ("eng_Latn", "khk_Cyrl")
    assert first.rows == 2


def test_the_files_hold_one_sentence_per_line_utf8_with_lf_sorted_by_id_source_target(
    tmp_path: Path,
) -> None:
    rows = [
        _row("Хоёр", "Two"),
        _row("Нэг", "One"),
        _row("Гурав", "Three"),
        _row("Гурав", "Another three"),
    ]
    training = _training(tmp_path / "in", rows, ids=["b", "a", "c", "c"])

    built = build_corpora(training, bt_ratio=0.0, seed=1, data_dir=tmp_path / "data")

    corpus = built.corpora[0]
    assert corpus.src_path.read_bytes() == "Нэг\nХоёр\nГурав\nГурав\n".encode()
    assert corpus.tgt_path.read_bytes() == b"One\nTwo\nAnother three\nThree\n"


def test_a_tab_stays_in_its_line_and_the_two_files_keep_the_same_line_count(
    tmp_path: Path,
) -> None:
    rows = [_row("a\tb", "c\td"), _row("e" + chr(0x2028) + "f", "g"), _row("h", "i\r")]
    training = _training(tmp_path / "in", rows, ids=["1", "2", "3"])

    built = build_corpora(training, bt_ratio=0.0, seed=1, data_dir=tmp_path / "data")

    corpus = built.corpora[0]
    assert corpus.rows == 1
    assert corpus.src_path.read_bytes() == b"a\tb\n"
    assert corpus.tgt_path.read_bytes() == b"c\td\n"
    assert training.stats.excluded["line_break"] == 2


def test_no_file_of_the_toy_data_holds_a_character_that_splits_lines_but_the_lf(
    tmp_path: Path,
) -> None:
    toy = build_toy()
    path = write_corpus_parquet([*toy.train, *toy.bt], tmp_path / "toy.parquet")

    built = build_corpora(
        load_training_rows([path]), bt_ratio=0.5, seed=1, data_dir=tmp_path / "data"
    )

    for corpus in built.corpora:
        for file in (corpus.src_path, corpus.tgt_path):
            text = file.read_text(encoding="utf-8")
            assert text.endswith("\n")
            assert len(text.splitlines()) == text.count("\n") == corpus.rows


def test_the_corpora_carry_the_sizes_the_sampler_takes(tmp_path: Path) -> None:
    rows = [*_numbered(3), *_numbered(2, origin="bt")]
    training = _training(tmp_path / "in", rows)

    built = build_corpora(training, bt_ratio=1.0, seed=1, data_dir=tmp_path / "data")

    assert corpus_sizes(built.corpora) == [
        CorpusSize("eng_Latn-khk_Cyrl.auth", EN_MN, 3),
        CorpusSize("eng_Latn-khk_Cyrl.bt", EN_MN, 2),
    ]


def test_the_corpus_hashes_are_the_sha256_of_the_files(tmp_path: Path) -> None:
    training = _training(tmp_path / "in", _numbered(4))

    corpus = build_corpora(
        training, bt_ratio=0.0, seed=1, data_dir=tmp_path / "data"
    ).corpora[0]

    assert corpus.src_sha256 == hashlib.sha256(corpus.src_path.read_bytes()).hexdigest()
    assert corpus.tgt_sha256 == hashlib.sha256(corpus.tgt_path.read_bytes()).hexdigest()


def test_duplicate_rows_are_written_as_often_as_they_occur(tmp_path: Path) -> None:
    training = _training(
        tmp_path / "in", [_row("a", "b"), _row("a", "b"), _row("c", "d")]
    )

    corpus = build_corpora(
        training, bt_ratio=0.0, seed=1, data_dir=tmp_path / "data"
    ).corpora[0]

    assert corpus.rows == 3
    assert sorted(_lines(corpus.src_path)) == ["a", "a", "c"]


def test_a_data_directory_with_glob_characters_is_written_as_named(
    tmp_path: Path,
) -> None:
    training = _training(tmp_path / "in", [*_numbered(3), *_numbered(2, origin="bt")])
    data_dir = tmp_path / "data[1]*?"

    built = build_corpora(training, bt_ratio=1.0, seed=1, data_dir=data_dir)

    assert [corpus.rows for corpus in built.corpora] == [3, 2]
    assert all(corpus.src_path.is_file() for corpus in built.corpora)


# --- build_corpora: the bt subsample (R4) ------------------------------------------


def _toy_training(tmp_path: Path) -> TrainingRows:
    toy = build_toy()
    path = write_corpus_parquet(
        [*toy.train, *toy.bt, *toy.dev], tmp_path / "toy.parquet"
    )
    return load_training_rows([path])


def test_a_bt_ratio_of_zero_writes_no_bt_file(tmp_path: Path) -> None:
    training = _toy_training(tmp_path)

    built = build_corpora(training, bt_ratio=0.0, seed=1, data_dir=tmp_path / "data")

    assert {corpus.group for corpus in built.corpora} == {"auth"}
    assert not (tmp_path / "data" / "bt").exists()
    assert {
        d: (u.bt_available, u.bt_requested, u.bt_used)
        for d, u in built.bt_by_direction.items()
    } == {
        EN_MN: (300, 0, 0),
        MN_EN: (300, 0, 0),
        RU_MN: (300, 0, 0),
        MN_RU: (300, 0, 0),
    }


@pytest.mark.parametrize(
    ("ratio", "expected"),
    [
        # round(r * n_auth) with n_auth 1200 / 1200 / 600 / 600, at most 300 available.
        (0.1, {EN_MN: 120, MN_EN: 120, RU_MN: 60, MN_RU: 60}),
        (0.2, {EN_MN: 240, MN_EN: 240, RU_MN: 120, MN_RU: 120}),
        (0.5, {EN_MN: 300, MN_EN: 300, RU_MN: 300, MN_RU: 300}),
        (1.0, {EN_MN: 300, MN_EN: 300, RU_MN: 300, MN_RU: 300}),
    ],
)
def test_a_direction_gets_the_smaller_of_the_available_and_ratio_times_its_authentic_rows(
    tmp_path: Path, ratio: float, expected: dict[str, int]
) -> None:
    training = _toy_training(tmp_path)

    built = build_corpora(training, bt_ratio=ratio, seed=1, data_dir=tmp_path / "data")

    written = {c.direction: c.rows for c in built.corpora if c.group == "bt"}
    assert written == expected
    assert {d: u.bt_used for d, u in built.bt_by_direction.items()} == expected
    assert built.bt_by_direction[EN_MN].bt_requested == round(ratio * 1200)
    assert built.bt_by_direction[RU_MN].bt_requested == round(ratio * 600)
    for corpus in built.corpora:
        assert (
            len(_lines(corpus.src_path)) == len(_lines(corpus.tgt_path)) == corpus.rows
        )


def test_a_shortfall_is_recorded_not_an_error(tmp_path: Path) -> None:
    training = _toy_training(tmp_path)

    built = build_corpora(training, bt_ratio=1.0, seed=1, data_dir=tmp_path / "data")

    usage = built.bt_by_direction[EN_MN]
    assert (usage.bt_available, usage.bt_requested, usage.bt_used) == (300, 1200, 300)


@pytest.mark.parametrize(
    ("auth_rows", "expected"),
    [
        # Python's round: ties go to the even integer.
        (1, 0),  # 0.5
        (3, 2),  # 1.5
        (5, 2),  # 2.5
        (7, 4),  # 3.5
    ],
)
def test_a_half_rounds_to_the_even_integer(
    tmp_path: Path, auth_rows: int, expected: int
) -> None:
    rows = [
        *_numbered(auth_rows),
        *[_row(f"b{i}", "t", origin="bt") for i in range(10)],
    ]
    training = _training(tmp_path / "in", rows)

    built = build_corpora(training, bt_ratio=0.5, seed=1, data_dir=tmp_path / "data")

    assert built.bt_by_direction[EN_MN].bt_requested == expected
    assert built.bt_by_direction[EN_MN].bt_used == expected
    assert [c.name for c in built.corpora if c.group == "bt"] == (
        [f"{EN_MN}.bt"] if expected else []
    )


def test_a_direction_with_bt_rows_and_no_authentic_rows_gets_no_bt_corpus(
    tmp_path: Path,
) -> None:
    rows = [*_numbered(4), *_numbered(3, src_lang="rus_Cyrl", origin="bt")]
    training = _training(tmp_path / "in", rows)

    built = build_corpora(training, bt_ratio=1.0, seed=1, data_dir=tmp_path / "data")

    assert [c.name for c in built.corpora] == [f"{EN_MN}.auth"]
    assert built.bt_by_direction[RU_MN].bt_available == 3
    assert built.bt_by_direction[RU_MN].bt_requested == 0
    assert built.bt_by_direction[RU_MN].bt_used == 0
    assert built.bt_by_direction[EN_MN].bt_available == 0


def test_the_bt_subsample_is_the_rows_with_the_smallest_seeded_hash(
    tmp_path: Path,
) -> None:
    # sha256("1234\t<id>") ranks id08 < id03 < id07 < id00 ... (computed from the rule);
    # six authentic rows and a ratio of 0.5 ask for three.
    auth = [_row(f"auth {i}", "t") for i in range(6)]
    bt = [_row(f"bt {i}", "t", origin="bt") for i in range(10)]
    ids = [f"a{i}" for i in range(6)] + [f"id{i:02d}" for i in range(10)]
    training = _training(tmp_path / "in", [*auth, *bt], ids=ids)

    built = build_corpora(training, bt_ratio=0.5, seed=1234, data_dir=tmp_path / "data")

    assert _lines(_corpus(built, f"{EN_MN}.bt").src_path) == ["bt 3", "bt 7", "bt 8"]


def test_another_seed_picks_other_rows_and_a_directory_of_its_own(
    tmp_path: Path,
) -> None:
    # Seed 99 ranks id06 < id04 < id07 < ...
    auth = [_row(f"auth {i}", "t") for i in range(6)]
    bt = [_row(f"bt {i}", "t", origin="bt") for i in range(10)]
    ids = [f"a{i}" for i in range(6)] + [f"id{i:02d}" for i in range(10)]
    training = _training(tmp_path / "in", [*auth, *bt], ids=ids)

    first = build_corpora(training, bt_ratio=0.5, seed=1234, data_dir=tmp_path / "data")
    second = build_corpora(training, bt_ratio=0.5, seed=99, data_dir=tmp_path / "data")

    assert _lines(_corpus(second, f"{EN_MN}.bt").src_path) == ["bt 4", "bt 6", "bt 7"]
    assert (
        _corpus(first, f"{EN_MN}.bt").src_path.parent
        == tmp_path / "data" / "bt" / "r0.5_s1234"
    )
    assert (
        _corpus(second, f"{EN_MN}.bt").src_path.parent
        == tmp_path / "data" / "bt" / "r0.5_s99"
    )
    assert _lines(_corpus(first, f"{EN_MN}.bt").src_path) == ["bt 3", "bt 7", "bt 8"]


def test_bt_rows_with_one_id_are_chosen_by_source_whatever_the_input_order(
    tmp_path: Path,
) -> None:
    auth = [_row("auth 0", "t"), _row("auth 1", "t")]
    twins = [_row(f"twin {i}", "t", origin="bt") for i in range(3)]
    ids = ["a0", "a1", "same", "same", "same"]
    chosen = []
    for name, bt in (("forward", twins), ("backward", twins[::-1])):
        training = _training(tmp_path / name, [*auth, *bt], ids=ids)
        built = build_corpora(
            training, bt_ratio=0.5, seed=1, data_dir=tmp_path / f"data_{name}"
        )
        chosen.append(_lines(_corpus(built, f"{EN_MN}.bt").src_path))

    # One row of the three is asked for: the equal hashes and ids leave the source.
    assert chosen == [["twin 0"], ["twin 0"]]


def test_the_ratio_names_the_bt_directory_the_way_variant_ids_print_it(
    tmp_path: Path,
) -> None:
    training = _training(tmp_path / "in", [*_numbered(4), *_numbered(4, origin="bt")])

    for ratio, text in ((1.0, "r1_s5"), (0.25, "r0.25_s5"), (2, "r2_s5")):
        built = build_corpora(
            training, bt_ratio=ratio, seed=5, data_dir=tmp_path / "data"
        )
        assert _corpus(built, f"{EN_MN}.bt").src_path.parent.name == text


def test_the_subsample_does_not_depend_on_the_order_of_the_rows_or_the_files(
    tmp_path: Path,
) -> None:
    toy = build_toy()
    rows = [*toy.train, *toy.bt]
    shuffled = list(rows)
    random.Random(5).shuffle(shuffled)
    one = write_corpus_parquet(rows, tmp_path / "one.parquet")
    half = len(shuffled) // 2
    two_a = write_corpus_parquet(shuffled[:half], tmp_path / "two_a.parquet")
    two_b = write_corpus_parquet(shuffled[half:], tmp_path / "two_b.parquet")

    first = build_corpora(
        load_training_rows([one]), bt_ratio=0.1, seed=7, data_dir=tmp_path / "first"
    )
    second = build_corpora(
        load_training_rows([two_b, two_a]),
        bt_ratio=0.1,
        seed=7,
        data_dir=tmp_path / "second",
    )

    assert [(c.name, c.src_sha256, c.tgt_sha256, c.rows) for c in first.corpora] == [
        (c.name, c.src_sha256, c.tgt_sha256, c.rows) for c in second.corpora
    ]
    assert first.bt_by_direction == second.bt_by_direction


def test_the_same_seed_gives_the_same_rows_and_another_seed_other_rows(
    tmp_path: Path,
) -> None:
    training = _toy_training(tmp_path)

    def bt_hashes(seed: int, name: str) -> list[str]:
        built = build_corpora(
            training, bt_ratio=0.1, seed=seed, data_dir=tmp_path / name
        )
        return [c.src_sha256 for c in built.corpora if c.group == "bt"]

    assert bt_hashes(7, "a") == bt_hashes(7, "b")
    assert bt_hashes(7, "a") != bt_hashes(8, "c")


def test_a_subsample_is_part_of_the_available_bt_rows(tmp_path: Path) -> None:
    toy = build_toy()
    training = _toy_training(tmp_path)

    built = build_corpora(training, bt_ratio=0.1, seed=3, data_dir=tmp_path / "data")

    for corpus in built.corpora:
        if corpus.group != "bt":
            continue
        available = {
            (row["src"], row["tgt"])
            for row in toy.bt
            if f"{row['src_lang']}-{row['tgt_lang']}" == corpus.direction
        }
        chosen = set(zip(_lines(corpus.src_path), _lines(corpus.tgt_path), strict=True))
        assert chosen <= available
        assert len(chosen) == corpus.rows


@pytest.mark.parametrize("ratio", [-0.5, float("nan"), float("inf")])
def test_a_ratio_that_is_not_a_finite_non_negative_number_is_refused(
    tmp_path: Path, ratio: float
) -> None:
    training = _training(tmp_path / "in", _numbered(2))

    with pytest.raises(ValueError, match="bt_ratio"):
        build_corpora(training, bt_ratio=ratio, seed=1, data_dir=tmp_path / "data")


def test_the_bt_usage_is_json_ready(tmp_path: Path) -> None:
    training = _toy_training(tmp_path)
    built = build_corpora(training, bt_ratio=0.1, seed=1, data_dir=tmp_path / "data")

    document = json.loads(json.dumps(built.bt_to_dict()))

    assert document[EN_MN] == {"bt_available": 300, "bt_requested": 120, "bt_used": 120}


# --- build_corpora: reuse of the written files -------------------------------------


def _inodes(built: BuiltCorpora) -> list[int]:
    return [os.stat(c.src_path).st_ino for c in built.corpora]


def test_a_second_build_of_the_same_rows_reuses_every_file(tmp_path: Path) -> None:
    training = _toy_training(tmp_path)
    first = build_corpora(training, bt_ratio=0.1, seed=1, data_dir=tmp_path / "data")
    inodes = _inodes(first)

    second = build_corpora(training, bt_ratio=0.1, seed=1, data_dir=tmp_path / "data")

    assert not any(corpus.reused for corpus in first.corpora)
    assert all(corpus.reused for corpus in second.corpora)
    assert _inodes(second) == inodes
    assert [c.src_sha256 for c in second.corpora] == [
        c.src_sha256 for c in first.corpora
    ]


def test_one_changed_source_row_rewrites_only_its_corpus(tmp_path: Path) -> None:
    rows = [
        *_numbered(3),
        *_numbered(3, src_lang="rus_Cyrl"),
    ]
    first = build_corpora(
        _training(tmp_path / "in1", rows),
        bt_ratio=0.0,
        seed=1,
        data_dir=tmp_path / "data",
    )
    changed = [*rows[:2], _row("s2 changed", "t2"), *rows[3:]]

    second = build_corpora(
        _training(tmp_path / "in2", changed),
        bt_ratio=0.0,
        seed=1,
        data_dir=tmp_path / "data",
    )

    assert [c.reused for c in second.corpora] == [False, True]
    assert sorted(_lines(_corpus(second, f"{EN_MN}.auth").src_path)) == [
        "s0",
        "s1",
        "s2 changed",
    ]
    assert (
        _corpus(second, f"{EN_MN}.auth").src_sha256
        != _corpus(first, f"{EN_MN}.auth").src_sha256
    )


def test_a_truncated_file_is_rewritten(tmp_path: Path) -> None:
    training = _training(tmp_path / "in", _numbered(20))
    first = build_corpora(training, bt_ratio=0.0, seed=1, data_dir=tmp_path / "data")
    corpus = first.corpora[0]
    complete = corpus.tgt_path.read_bytes()
    corpus.tgt_path.write_bytes(complete[:-3])

    second = build_corpora(training, bt_ratio=0.0, seed=1, data_dir=tmp_path / "data")

    assert second.corpora[0].reused is False
    assert corpus.tgt_path.read_bytes() == complete


def test_a_file_that_changed_in_place_with_the_same_size_is_rewritten(
    tmp_path: Path,
) -> None:
    training = _training(tmp_path / "in", _numbered(20))
    first = build_corpora(training, bt_ratio=0.0, seed=1, data_dir=tmp_path / "data")
    corpus = first.corpora[0]
    complete = corpus.src_path.read_bytes()
    corpus.src_path.write_bytes(complete.replace(b"s1", b"x1", 1))

    second = build_corpora(training, bt_ratio=0.0, seed=1, data_dir=tmp_path / "data")

    assert second.corpora[0].reused is False
    assert corpus.src_path.read_bytes() == complete


def test_a_missing_file_is_written_again(tmp_path: Path) -> None:
    training = _training(tmp_path / "in", _numbered(5))
    first = build_corpora(training, bt_ratio=0.0, seed=1, data_dir=tmp_path / "data")
    first.corpora[0].src_path.unlink()

    second = build_corpora(training, bt_ratio=0.0, seed=1, data_dir=tmp_path / "data")

    assert second.corpora[0].reused is False
    assert len(_lines(second.corpora[0].src_path)) == 5


def _sidecar(corpus: CorpusFiles) -> Path:
    (found,) = [
        path
        for path in corpus.src_path.parent.iterdir()
        if path.name.startswith(corpus.name) and path.suffix == ".json"
    ]
    return found


@pytest.mark.parametrize(
    "damage",
    ["missing", "empty", "not json", "list", "wrong rows", "wrong digest", "no digest"],
)
def test_a_missing_or_unusable_sidecar_means_the_files_are_rewritten(
    tmp_path: Path, damage: str
) -> None:
    training = _training(tmp_path / "in", _numbered(5))
    first = build_corpora(training, bt_ratio=0.0, seed=1, data_dir=tmp_path / "data")
    sidecar = _sidecar(first.corpora[0])
    document = json.loads(sidecar.read_text(encoding="utf-8"))
    intact = dict(document)
    if damage == "missing":
        sidecar.unlink()
    elif damage == "empty":
        sidecar.write_bytes(b"")
    elif damage == "not json":
        sidecar.write_bytes(b"{ not json")
    elif damage == "list":
        sidecar.write_text("[]", encoding="utf-8")
    elif damage == "wrong rows":
        sidecar.write_text(json.dumps({**document, "rows": 4}), encoding="utf-8")
    elif damage == "wrong digest":
        sidecar.write_text(
            json.dumps({**document, "src_sha256": "0" * 64}), encoding="utf-8"
        )
    else:
        del document["tgt_sha256"]
        sidecar.write_text(json.dumps(document), encoding="utf-8")

    second = build_corpora(training, bt_ratio=0.0, seed=1, data_dir=tmp_path / "data")

    assert second.corpora[0].reused is False
    assert json.loads(_sidecar(second.corpora[0]).read_text(encoding="utf-8")) == intact


def test_the_sidecar_records_the_row_count_and_the_digests_of_both_files(
    tmp_path: Path,
) -> None:
    training = _training(tmp_path / "in", _numbered(5))

    corpus = build_corpora(
        training, bt_ratio=0.0, seed=1, data_dir=tmp_path / "data"
    ).corpora[0]

    assert json.loads(_sidecar(corpus).read_text(encoding="utf-8")) == {
        "rows": 5,
        "src_sha256": corpus.src_sha256,
        "tgt_sha256": corpus.tgt_sha256,
    }


def test_a_bt_subsample_is_reused_too_and_a_new_ratio_leaves_the_old_one_alone(
    tmp_path: Path,
) -> None:
    training = _toy_training(tmp_path)
    first = build_corpora(training, bt_ratio=0.1, seed=1, data_dir=tmp_path / "data")
    other = build_corpora(training, bt_ratio=0.2, seed=1, data_dir=tmp_path / "data")

    again = build_corpora(training, bt_ratio=0.1, seed=1, data_dir=tmp_path / "data")

    assert all(c.reused for c in again.corpora)
    assert [c.reused for c in other.corpora if c.group == "auth"] == [True] * 4
    assert [c.reused for c in other.corpora if c.group == "bt"] == [False] * 4
    assert [c.src_sha256 for c in again.corpora] == [
        c.src_sha256 for c in first.corpora
    ]


def test_a_failed_write_leaves_the_earlier_file_and_no_temporary_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    first = build_corpora(
        _training(tmp_path / "in1", _numbered(3)),
        bt_ratio=0.0,
        seed=1,
        data_dir=tmp_path / "data",
    )
    before = first.corpora[0].src_path.read_bytes()
    original_replace = Path.replace

    def failing_replace(self: Path, target: Any) -> Path:
        if str(target).endswith(".src"):
            raise OSError("disk full")
        return original_replace(self, target)

    monkeypatch.setattr(Path, "replace", failing_replace)

    with pytest.raises(OSError, match="disk full"):
        build_corpora(
            _training(tmp_path / "in2", _numbered(4)),
            bt_ratio=0.0,
            seed=1,
            data_dir=tmp_path / "data",
        )

    assert first.corpora[0].src_path.read_bytes() == before
    assert not list((tmp_path / "data").rglob("*.part"))


def test_a_build_leaves_no_temporary_file_behind(tmp_path: Path) -> None:
    build_corpora(
        _toy_training(tmp_path), bt_ratio=0.1, seed=1, data_dir=tmp_path / "data"
    )

    assert not list((tmp_path / "data").rglob("*.part"))


# --- data_section ------------------------------------------------------------------


def _files(direction: str, group: str, *, directory: str = "data/auth") -> CorpusFiles:
    source, target = direction.split("-")
    name = f"{direction}.{group}"
    return CorpusFiles(
        name=name,
        direction=direction,
        group=group,
        src_lang=source,
        tgt_lang=target,
        src_path=Path(f"{directory}/{name}.src"),
        tgt_path=Path(f"{directory}/{name}.tgt"),
        rows=10,
        src_sha256="0" * 64,
        tgt_sha256="1" * 64,
        reused=False,
    )


_VALID = ValidPaths(src=Path("variants/v/valid.src"), tgt=Path("variants/v/valid.tgt"))


def test_the_source_placement_puts_the_target_tag_in_the_source_prefix() -> None:
    section = data_section(
        [_files(EN_MN, "auth")],
        {f"{EN_MN}.auth": 7},
        "src",
        valid_paths=_VALID,
        filtertoolong=False,
    )

    assert section == {
        f"{EN_MN}.auth": {
            "path_src": f"data/auth/{EN_MN}.auth.src",
            "path_tgt": f"data/auth/{EN_MN}.auth.tgt",
            "transforms": ["sentencepiece", "prefix"],
            "weight": 7,
            "src_prefix": "<2khk_Cyrl>",
            "tgt_prefix": "",
        },
        "valid": {
            "path_src": "variants/v/valid.src",
            "path_tgt": "variants/v/valid.tgt",
        },
    }


def test_the_source_placement_puts_the_bt_tag_after_the_direction_tag() -> None:
    section = data_section(
        [_files(EN_MN, "auth"), _files(EN_MN, "bt", directory="data/bt/r1_s1")],
        {f"{EN_MN}.auth": 3, f"{EN_MN}.bt": 1},
        "src",
        valid_paths=_VALID,
        filtertoolong=False,
    )

    assert section[f"{EN_MN}.bt"] == {
        "path_src": f"data/bt/r1_s1/{EN_MN}.bt.src",
        "path_tgt": f"data/bt/r1_s1/{EN_MN}.bt.tgt",
        "transforms": ["sentencepiece", "prefix"],
        "weight": 1,
        "src_prefix": "<2khk_Cyrl> <bt>",
        "tgt_prefix": "",
    }
    assert list(section) == [f"{EN_MN}.auth", f"{EN_MN}.bt", "valid"]


def test_the_target_placement_puts_the_tag_in_the_target_prefix_and_bt_alone_in_the_source() -> (
    None
):
    section = data_section(
        [_files(EN_MN, "auth"), _files(EN_MN, "bt", directory="data/bt/r1_s1")],
        {f"{EN_MN}.auth": 3, f"{EN_MN}.bt": 1},
        "tgt",
        valid_paths=_VALID,
        filtertoolong=False,
    )

    assert section[f"{EN_MN}.auth"]["src_prefix"] == ""
    assert section[f"{EN_MN}.auth"]["tgt_prefix"] == "<2khk_Cyrl>"
    assert section[f"{EN_MN}.bt"]["src_prefix"] == "<bt>"
    assert section[f"{EN_MN}.bt"]["tgt_prefix"] == "<2khk_Cyrl>"


@pytest.mark.parametrize(
    ("direction", "tag"),
    [(EN_MN, "<2khk_Cyrl>"), (MN_EN, "<2eng_Latn>"), (MN_RU, "<2rus_Cyrl>")],
)
def test_the_tag_is_that_of_the_target_language_of_the_corpus(
    direction: str, tag: str
) -> None:
    section = data_section(
        [_files(direction, "auth")],
        {f"{direction}.auth": 1},
        "src",
        valid_paths=_VALID,
        filtertoolong=False,
    )

    assert section[f"{direction}.auth"]["src_prefix"] == tag


def test_filtertoolong_is_added_after_prefix_only_when_asked() -> None:
    corpora = [_files(EN_MN, "auth")]
    weights = {f"{EN_MN}.auth": 1}

    with_filter = data_section(
        corpora, weights, "src", valid_paths=_VALID, filtertoolong=True
    )
    without = data_section(
        corpora, weights, "src", valid_paths=_VALID, filtertoolong=False
    )

    assert with_filter[f"{EN_MN}.auth"]["transforms"] == [
        "sentencepiece",
        "prefix",
        "filtertoolong",
    ]
    assert without[f"{EN_MN}.auth"]["transforms"] == ["sentencepiece", "prefix"]


def test_the_valid_entry_carries_the_two_paths_and_nothing_else() -> None:
    section = data_section(
        [_files(EN_MN, "auth")],
        {f"{EN_MN}.auth": 1},
        "tgt",
        valid_paths=_VALID,
        filtertoolong=True,
    )

    assert set(section["valid"]) == {"path_src", "path_tgt"}


def test_the_weights_are_the_integers_given_whatever_the_corpus_sizes() -> None:
    corpora = [_files(EN_MN, "auth"), _files(RU_MN, "auth")]

    section = data_section(
        corpora,
        {f"{EN_MN}.auth": 61, f"{RU_MN}.auth": 39},
        "src",
        valid_paths=_VALID,
        filtertoolong=False,
    )

    assert [section[name]["weight"] for name in (f"{EN_MN}.auth", f"{RU_MN}.auth")] == [
        61,
        39,
    ]
    assert all(
        type(section[n]["weight"]) is int for n in (f"{EN_MN}.auth", f"{RU_MN}.auth")
    )


@pytest.mark.parametrize("weight", [0, -1, 1.0, True, "3", None])
def test_a_weight_that_is_not_a_positive_integer_is_a_value_error(weight: Any) -> None:
    with pytest.raises(ValueError, match=rf"corpus.*{EN_MN}\.auth.*positive integer"):
        data_section(
            [_files(EN_MN, "auth")],
            {f"{EN_MN}.auth": weight},
            "src",
            valid_paths=_VALID,
            filtertoolong=False,
        )


def test_a_corpus_without_a_weight_is_a_value_error() -> None:
    with pytest.raises(ValueError, match=rf"no weight for corpus.*{RU_MN}\.auth"):
        data_section(
            [_files(EN_MN, "auth"), _files(RU_MN, "auth")],
            {f"{EN_MN}.auth": 1},
            "src",
            valid_paths=_VALID,
            filtertoolong=False,
        )


def test_a_weight_for_a_corpus_that_is_not_there_is_a_value_error() -> None:
    with pytest.raises(ValueError, match=r"not in the mixture.*typo\.auth"):
        data_section(
            [_files(EN_MN, "auth")],
            {f"{EN_MN}.auth": 1, "typo.auth": 1},
            "src",
            valid_paths=_VALID,
            filtertoolong=False,
        )


@pytest.mark.parametrize("placement", ["suffix", "", "SRC", None])
def test_a_placement_other_than_src_or_tgt_is_a_value_error(placement: Any) -> None:
    with pytest.raises(ValueError, match="placement"):
        data_section(
            [_files(EN_MN, "auth")],
            {f"{EN_MN}.auth": 1},
            placement,
            valid_paths=_VALID,
            filtertoolong=False,
        )


def test_no_corpus_is_a_value_error() -> None:
    with pytest.raises(ValueError, match="corpus"):
        data_section([], {}, "src", valid_paths=_VALID, filtertoolong=False)


@pytest.mark.parametrize(
    "where", ["corpus src", "corpus tgt", "valid src", "valid tgt"]
)
def test_a_dollar_sign_in_any_path_is_a_config_error(where: str) -> None:
    corpus = _files(EN_MN, "auth")
    if where == "corpus src":
        corpus = replace(corpus, src_path=Path("data/$HOME/a.src"))
    elif where == "corpus tgt":
        corpus = replace(corpus, tgt_path=Path("data/${HOME}/a.tgt"))
    valid = {
        "valid src": ValidPaths(Path("$V/valid.src"), Path("v/valid.tgt")),
        "valid tgt": ValidPaths(Path("v/valid.src"), Path("$V/valid.tgt")),
    }.get(where, _VALID)

    with pytest.raises(ConfigError, match=r"'\$'"):
        data_section(
            [corpus],
            {corpus.name: 1},
            "src",
            valid_paths=valid,
            filtertoolong=False,
        )


def test_two_corpora_of_one_name_are_a_value_error() -> None:
    corpus = _files(EN_MN, "auth")

    with pytest.raises(ValueError, match=rf"unique.*{EN_MN}\.auth"):
        data_section(
            [corpus, corpus],
            {corpus.name: 1},
            "src",
            valid_paths=_VALID,
            filtertoolong=False,
        )


def test_paths_are_written_as_posix_strings_as_given() -> None:
    corpus = _files(EN_MN, "auth", directory="/abs/data[1]/auth")

    section = data_section(
        [corpus], {corpus.name: 1}, "src", valid_paths=_VALID, filtertoolong=False
    )

    assert section[corpus.name]["path_src"] == f"/abs/data[1]/auth/{corpus.name}.src"


def _validated(section: dict[str, Any]) -> dict[str, Any]:
    base = read_config(TINY_BASE, "tiny base")
    config = merge(
        base,
        {
            "data": section,
            "transforms_configs": {
                "sentencepiece": {
                    "src_subword_model": "spm/toy.model",
                    "tgt_subword_model": "spm/toy.model",
                }
            },
            "src_vocab": "run/vocab.src",
            "training": {"model_path": "run/ckpt"},
        },
    )
    validate_train_config(config)
    return config


@pytest.mark.parametrize("placement", ["src", "tgt"])
@pytest.mark.parametrize("with_bt", [False, True])
def test_the_section_merged_into_the_tiny_base_passes_the_config_validator(
    placement: str, with_bt: bool
) -> None:
    corpora = [_files(EN_MN, "auth"), _files(MN_RU, "auth")]
    if with_bt:
        corpora += [_files(EN_MN, "bt", directory="data/bt/r1_s1")]
    weights = {corpus.name: index + 1 for index, corpus in enumerate(corpora)}

    section = data_section(
        corpora, weights, placement, valid_paths=_VALID, filtertoolong=True
    )

    config = _validated(section)
    assert config["data"]["valid"] == {
        "path_src": "variants/v/valid.src",
        "path_tgt": "variants/v/valid.tgt",
    }


def test_the_toy_corpora_weighted_by_temperature_make_a_valid_config(
    tmp_path: Path,
) -> None:
    training = _toy_training(tmp_path)
    built = build_corpora(training, bt_ratio=0.2, seed=1, data_dir=tmp_path / "data")
    probabilities = mixing_probabilities(corpus_sizes(built.corpora), 0.5)
    weights = integer_weights(probabilities, total=100)

    section = data_section(
        built.corpora, weights, "tgt", valid_paths=_VALID, filtertoolong=True
    )

    config = _validated(section)
    assert (
        sum(
            entry["weight"] for name, entry in config["data"].items() if name != "valid"
        )
        == 100
    )
    assert len(config["data"]) == 9


# --- write_validation --------------------------------------------------------------


def _dev_set(direction: str, pairs: Sequence[tuple[str, str]]) -> DevSet:
    source, target = direction.split("-")
    return DevSet(
        direction=direction,
        src_lang=source,
        tgt_lang=target,
        ids=tuple(f"id{index}" for index in range(len(pairs))),
        src=tuple(pair[0] for pair in pairs),
        tgt=tuple(pair[1] for pair in pairs),
    )


def _toy_dev() -> dict[str, DevSet]:
    toy = build_toy()
    by_direction: dict[str, list[tuple[str, str]]] = {}
    for row in toy.dev:
        by_direction.setdefault(f"{row['src_lang']}-{row['tgt_lang']}", []).append(
            (row["src"], row["tgt"])
        )
    # Two lines of each direction, entered in an order that is not the sorted one.
    return {
        direction: _dev_set(direction, by_direction[direction][:2])
        for direction in (MN_RU, EN_MN, RU_MN, MN_EN)
    }


def _pieces(spm_model: Path, text: str) -> list[str]:
    processor = sentencepiece.SentencePieceProcessor(model_file=str(spm_model))
    return list(processor.encode(text, out_type=str))


@pytest.mark.parametrize("placement", ["src", "tgt"])
def test_the_validation_lines_are_pre_tokenised_with_the_target_tag_on_one_side(
    tmp_path: Path, toy_spm: Path, placement: str
) -> None:
    dev = _toy_dev()

    paths = write_validation(dev, toy_spm, placement, tmp_path / "out")

    assert paths == ValidPaths(
        tmp_path / "out" / "valid.src", tmp_path / "out" / "valid.tgt"
    )
    expected_src: list[str] = []
    expected_tgt: list[str] = []
    for direction in sorted(dev):
        tag = {
            "khk_Cyrl": "<2khk_Cyrl>",
            "eng_Latn": "<2eng_Latn>",
            "rus_Cyrl": "<2rus_Cyrl>",
        }[dev[direction].tgt_lang]
        for source, target in zip(dev[direction].src, dev[direction].tgt, strict=True):
            source_pieces, target_pieces = (
                _pieces(toy_spm, source),
                _pieces(toy_spm, target),
            )
            if placement == "src":
                expected_src.append(" ".join([tag, *source_pieces]))
                expected_tgt.append(" ".join(target_pieces))
            else:
                expected_src.append(" ".join(source_pieces))
                expected_tgt.append(" ".join([tag, *target_pieces]))
    assert _lines(paths.src) == expected_src
    assert _lines(paths.tgt) == expected_tgt
    assert len(expected_src) == 8


def test_the_tag_is_the_first_piece_of_the_source_line_for_the_source_placement(
    tmp_path: Path, toy_spm: Path
) -> None:
    paths = write_validation(
        {EN_MN: _dev_set(EN_MN, [("This is the book.", "Энэ бол ном.")])},
        toy_spm,
        "src",
        tmp_path,
    )

    (source,) = _lines(paths.src)
    (target,) = _lines(paths.tgt)
    assert source.split(" ")[0] == "<2khk_Cyrl>"
    assert "<2khk_Cyrl>" not in target.split(" ")
    assert source.split(" ")[1:] == _pieces(toy_spm, "This is the book.")


def test_the_tag_is_the_first_piece_of_the_target_line_for_the_target_placement(
    tmp_path: Path, toy_spm: Path
) -> None:
    paths = write_validation(
        {MN_EN: _dev_set(MN_EN, [("Энэ бол ном.", "This is the book.")])},
        toy_spm,
        "tgt",
        tmp_path,
    )

    (source,) = _lines(paths.src)
    (target,) = _lines(paths.tgt)
    assert target.split(" ")[0] == "<2eng_Latn>"
    assert target.split(" ")[1:] == _pieces(toy_spm, "This is the book.")
    assert "<2eng_Latn>" not in source.split(" ")


def test_the_directions_are_concatenated_in_sorted_order_with_their_line_ranges(
    tmp_path: Path, toy_spm: Path
) -> None:
    dev = {
        MN_RU: _dev_set(MN_RU, [("Энэ бол ном.", "Это книга.")] * 3),
        EN_MN: _dev_set(EN_MN, [("This is the book.", "Энэ бол ном.")] * 2),
        RU_MN: _dev_set(RU_MN, [("Это книга.", "Энэ бол ном.")]),
    }

    paths = write_validation(dev, toy_spm, "src", tmp_path)

    assert dev_line_ranges(dev) == {
        EN_MN: range(2),
        MN_RU: range(2, 5),
        RU_MN: range(5, 6),
    }
    sources = _lines(paths.src)
    assert len(sources) == len(_lines(paths.tgt)) == 6
    assert [line.split(" ")[0] for line in sources] == [
        "<2khk_Cyrl>",
        "<2khk_Cyrl>",
        "<2rus_Cyrl>",
        "<2rus_Cyrl>",
        "<2rus_Cyrl>",
        "<2khk_Cyrl>",
    ]


def test_the_validation_files_are_lf_only_utf8_with_one_line_per_dev_row(
    tmp_path: Path, toy_spm: Path
) -> None:
    dev = _toy_dev()

    paths = write_validation(dev, toy_spm, "src", tmp_path)

    for path in (paths.src, paths.tgt):
        data = path.read_bytes()
        assert data.endswith(b"\n")
        assert b"\r" not in data
        assert data.count(b"\n") == sum(len(d.src) for d in dev.values())
        data.decode("utf-8")


def test_a_line_is_pieces_joined_by_single_spaces_only(
    tmp_path: Path, toy_spm: Path
) -> None:
    paths = write_validation(_toy_dev(), toy_spm, "tgt", tmp_path)

    for line in (*_lines(paths.src), *_lines(paths.tgt)):
        assert "  " not in line
        assert line == line.strip(" ")
        assert all(piece for piece in line.split(" "))


def test_the_output_directory_is_created_and_may_hold_glob_characters(
    tmp_path: Path, toy_spm: Path
) -> None:
    paths = write_validation(
        _toy_dev(), toy_spm, "src", tmp_path / "out[1]*?" / "valid"
    )

    assert paths.src.parent == tmp_path / "out[1]*?" / "valid"
    assert paths.src.is_file()


def test_a_direction_tag_that_is_not_a_piece_of_the_model_is_a_config_error(
    tmp_path: Path,
) -> None:
    text = tmp_path / "plain.txt"
    text.write_text(
        "".join(f"line {i} of some text\n" for i in range(200)), encoding="utf-8"
    )
    sentencepiece.SentencePieceTrainer.train(
        input=str(text),
        model_prefix=str(tmp_path / "plain"),
        vocab_size=40,
        minloglevel=2,
    )

    with pytest.raises(ConfigError, match=r"<2khk_Cyrl>"):
        write_validation(
            {EN_MN: _dev_set(EN_MN, [("a", "b")])},
            tmp_path / "plain.model",
            "src",
            tmp_path / "out",
        )


@pytest.mark.parametrize("what", ["missing", "directory", "not a model"])
def test_a_model_that_cannot_be_loaded_is_a_config_error(
    tmp_path: Path, what: str
) -> None:
    model = tmp_path / "model"
    if what == "directory":
        model.mkdir()
    elif what == "not a model":
        model.write_bytes(b"not a model")

    with pytest.raises(ConfigError, match="SentencePiece model"):
        write_validation(
            {EN_MN: _dev_set(EN_MN, [("a", "b")])}, model, "src", tmp_path / "out"
        )


def test_an_unknown_placement_or_no_dev_set_is_a_value_error(
    tmp_path: Path, toy_spm: Path
) -> None:
    placement: Any = "suffix"
    with pytest.raises(ValueError, match="placement"):
        write_validation(_toy_dev(), toy_spm, placement, tmp_path)
    with pytest.raises(ValueError, match="dev"):
        write_validation({}, toy_spm, "src", tmp_path)


def test_the_line_ranges_of_no_dev_sets_are_empty() -> None:
    assert dev_line_ranges({}) == {}
