"""Tests for `rinchen.train.corpus_files`: the corpus text files, the bt subsample
and the reuse of files already written.

File contents are worked out by hand from the module docstring and the D7 spec (R4,
R6), not from the code. The properties over random rows are in
`tests/property/test_train_mixture_property.py`.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import sys
from pathlib import Path
from typing import Any

import polars as pl
import pytest

from rinchen.train.corpus_files import (
    BuiltCorpora,
    CorpusFiles,
    PlannedFiles,
    bt_usage_by_direction,
    build_corpora,
    corpus_name,
    planned_corpus_files,
    planned_corpus_sizes,
    write_corpus_files,
)
from rinchen.train.mixture import load_training_rows
from rinchen.train.sampling import CorpusSize
from tests.support.corpus_rows import (
    EN_MN,
    MN_EN,
    MN_RU,
    RU_MN,
    corpus_row,
    file_lines,
    numbered_rows,
    sizes_of,
    toy_training_rows,
    training_from_rows,
)
from tests.support.tiny_parallel import build_toy, write_corpus_parquet


def _corpus(built: BuiltCorpora, name: str) -> CorpusFiles:
    (found,) = [corpus for corpus in built.corpora if corpus.name == name]
    return found


# --- build_corpora: the text files -------------------------------------------------


def test_each_corpus_is_a_pair_of_text_files_named_after_its_direction_and_group(
    tmp_path: Path,
) -> None:
    rows = [
        *numbered_rows(2),
        *numbered_rows(2, src_lang="khk_Cyrl", tgt_lang="rus_Cyrl", origin="web"),
    ]
    training = training_from_rows(tmp_path / "in", rows)

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
        corpus_row("Хоёр", "Two"),
        corpus_row("Нэг", "One"),
        corpus_row("Гурав", "Three"),
        corpus_row("Гурав", "Another three"),
    ]
    training = training_from_rows(tmp_path / "in", rows, ids=["b", "a", "c", "c"])

    built = build_corpora(training, bt_ratio=0.0, seed=1, data_dir=tmp_path / "data")

    corpus = built.corpora[0]
    assert corpus.src_path.read_bytes() == "Нэг\nХоёр\nГурав\nГурав\n".encode()
    assert corpus.tgt_path.read_bytes() == b"One\nTwo\nAnother three\nThree\n"


def test_a_tab_stays_in_its_line_and_the_two_files_keep_the_same_line_count(
    tmp_path: Path,
) -> None:
    rows = [
        corpus_row("a\tb", "c\td"),
        corpus_row("e" + chr(0x2028) + "f", "g"),
        corpus_row("h", "i\r"),
    ]
    training = training_from_rows(tmp_path / "in", rows, ids=["1", "2", "3"])

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
    rows = [*numbered_rows(3), *numbered_rows(2, origin="bt")]
    training = training_from_rows(tmp_path / "in", rows)

    built = build_corpora(training, bt_ratio=1.0, seed=1, data_dir=tmp_path / "data")

    assert sizes_of(built.corpora) == [
        CorpusSize("eng_Latn-khk_Cyrl.auth", EN_MN, 3),
        CorpusSize("eng_Latn-khk_Cyrl.bt", EN_MN, 2),
    ]


def test_the_corpus_hashes_are_the_sha256_of_the_files(tmp_path: Path) -> None:
    training = training_from_rows(tmp_path / "in", numbered_rows(4))

    corpus = build_corpora(
        training, bt_ratio=0.0, seed=1, data_dir=tmp_path / "data"
    ).corpora[0]

    assert corpus.src_sha256 == hashlib.sha256(corpus.src_path.read_bytes()).hexdigest()
    assert corpus.tgt_sha256 == hashlib.sha256(corpus.tgt_path.read_bytes()).hexdigest()


def test_duplicate_rows_are_written_as_often_as_they_occur(tmp_path: Path) -> None:
    training = training_from_rows(
        tmp_path / "in",
        [corpus_row("a", "b"), corpus_row("a", "b"), corpus_row("c", "d")],
    )

    corpus = build_corpora(
        training, bt_ratio=0.0, seed=1, data_dir=tmp_path / "data"
    ).corpora[0]

    assert corpus.rows == 3
    assert sorted(file_lines(corpus.src_path)) == ["a", "a", "c"]


def test_a_data_directory_with_glob_characters_is_written_as_named(
    tmp_path: Path,
) -> None:
    training = training_from_rows(
        tmp_path / "in", [*numbered_rows(3), *numbered_rows(2, origin="bt")]
    )
    data_dir = tmp_path / "data[1]*?"

    built = build_corpora(training, bt_ratio=1.0, seed=1, data_dir=data_dir)

    assert [corpus.rows for corpus in built.corpora] == [3, 2]
    assert all(corpus.src_path.is_file() for corpus in built.corpora)


# --- build_corpora: the bt subsample (R4) ------------------------------------------


def test_a_bt_ratio_of_zero_writes_no_bt_file(tmp_path: Path) -> None:
    training = toy_training_rows(tmp_path)

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
    training = toy_training_rows(tmp_path)

    built = build_corpora(training, bt_ratio=ratio, seed=1, data_dir=tmp_path / "data")

    written = {c.direction: c.rows for c in built.corpora if c.group == "bt"}
    assert written == expected
    assert {d: u.bt_used for d, u in built.bt_by_direction.items()} == expected
    assert built.bt_by_direction[EN_MN].bt_requested == round(ratio * 1200)
    assert built.bt_by_direction[RU_MN].bt_requested == round(ratio * 600)
    for corpus in built.corpora:
        assert (
            len(file_lines(corpus.src_path))
            == len(file_lines(corpus.tgt_path))
            == corpus.rows
        )


def test_a_shortfall_is_recorded_not_an_error(tmp_path: Path) -> None:
    training = toy_training_rows(tmp_path)

    built = build_corpora(training, bt_ratio=1.0, seed=1, data_dir=tmp_path / "data")

    usage = built.bt_by_direction[EN_MN]
    assert (usage.bt_available, usage.bt_requested, usage.bt_used) == (300, 1200, 300)


@pytest.mark.parametrize("ratio", [1e308, sys.float_info.max])
def test_a_ratio_whose_product_overflows_asks_for_more_than_there_is_and_gets_all(
    tmp_path: Path, ratio: float
) -> None:
    # 1200 authentic rows times 1e308 is beyond what a float holds.
    training = toy_training_rows(tmp_path)

    built = build_corpora(training, bt_ratio=ratio, seed=1, data_dir=tmp_path / "data")

    usage = built.bt_by_direction[EN_MN]
    assert (usage.bt_available, usage.bt_used) == (300, 300)
    assert usage.bt_requested == int(ratio) * 1200
    assert _corpus(built, f"{EN_MN}.bt").rows == 300


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
        *numbered_rows(auth_rows),
        *[corpus_row(f"b{i}", "t", origin="bt") for i in range(10)],
    ]
    training = training_from_rows(tmp_path / "in", rows)

    built = build_corpora(training, bt_ratio=0.5, seed=1, data_dir=tmp_path / "data")

    assert built.bt_by_direction[EN_MN].bt_requested == expected
    assert built.bt_by_direction[EN_MN].bt_used == expected
    assert [c.name for c in built.corpora if c.group == "bt"] == (
        [f"{EN_MN}.bt"] if expected else []
    )


def test_a_direction_with_bt_rows_and_no_authentic_rows_gets_no_bt_corpus(
    tmp_path: Path,
) -> None:
    rows = [*numbered_rows(4), *numbered_rows(3, src_lang="rus_Cyrl", origin="bt")]
    training = training_from_rows(tmp_path / "in", rows)

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
    auth = [corpus_row(f"auth {i}", "t") for i in range(6)]
    bt = [corpus_row(f"bt {i}", "t", origin="bt") for i in range(10)]
    ids = [f"a{i}" for i in range(6)] + [f"id{i:02d}" for i in range(10)]
    training = training_from_rows(tmp_path / "in", [*auth, *bt], ids=ids)

    built = build_corpora(training, bt_ratio=0.5, seed=1234, data_dir=tmp_path / "data")

    assert file_lines(_corpus(built, f"{EN_MN}.bt").src_path) == [
        "bt 3",
        "bt 7",
        "bt 8",
    ]


def test_another_seed_picks_other_rows_and_a_directory_of_its_own(
    tmp_path: Path,
) -> None:
    # Seed 99 ranks id06 < id04 < id07 < ...
    auth = [corpus_row(f"auth {i}", "t") for i in range(6)]
    bt = [corpus_row(f"bt {i}", "t", origin="bt") for i in range(10)]
    ids = [f"a{i}" for i in range(6)] + [f"id{i:02d}" for i in range(10)]
    training = training_from_rows(tmp_path / "in", [*auth, *bt], ids=ids)

    first = build_corpora(training, bt_ratio=0.5, seed=1234, data_dir=tmp_path / "data")
    second = build_corpora(training, bt_ratio=0.5, seed=99, data_dir=tmp_path / "data")

    assert file_lines(_corpus(second, f"{EN_MN}.bt").src_path) == [
        "bt 4",
        "bt 6",
        "bt 7",
    ]
    assert (
        _corpus(first, f"{EN_MN}.bt").src_path.parent
        == tmp_path / "data" / "bt" / "r0.5_s1234"
    )
    assert (
        _corpus(second, f"{EN_MN}.bt").src_path.parent
        == tmp_path / "data" / "bt" / "r0.5_s99"
    )
    assert file_lines(_corpus(first, f"{EN_MN}.bt").src_path) == [
        "bt 3",
        "bt 7",
        "bt 8",
    ]


def test_bt_rows_with_one_id_are_chosen_by_source_whatever_the_input_order(
    tmp_path: Path,
) -> None:
    auth = [corpus_row("auth 0", "t"), corpus_row("auth 1", "t")]
    twins = [corpus_row(f"twin {i}", "t", origin="bt") for i in range(3)]
    ids = ["a0", "a1", "same", "same", "same"]
    chosen = []
    for name, bt in (("forward", twins), ("backward", twins[::-1])):
        training = training_from_rows(tmp_path / name, [*auth, *bt], ids=ids)
        built = build_corpora(
            training, bt_ratio=0.5, seed=1, data_dir=tmp_path / f"data_{name}"
        )
        chosen.append(file_lines(_corpus(built, f"{EN_MN}.bt").src_path))

    # One row of the three is asked for: the equal hashes and ids leave the source.
    assert chosen == [["twin 0"], ["twin 0"]]


def test_the_ratio_names_the_bt_directory_the_way_variant_ids_print_it(
    tmp_path: Path,
) -> None:
    training = training_from_rows(
        tmp_path / "in", [*numbered_rows(4), *numbered_rows(4, origin="bt")]
    )

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
    training = toy_training_rows(tmp_path)

    def bt_hashes(seed: int, name: str) -> list[str]:
        built = build_corpora(
            training, bt_ratio=0.1, seed=seed, data_dir=tmp_path / name
        )
        return [c.src_sha256 for c in built.corpora if c.group == "bt"]

    assert bt_hashes(7, "a") == bt_hashes(7, "b")
    assert bt_hashes(7, "a") != bt_hashes(8, "c")


def test_a_subsample_is_part_of_the_available_bt_rows(tmp_path: Path) -> None:
    toy = build_toy()
    training = toy_training_rows(tmp_path)

    built = build_corpora(training, bt_ratio=0.1, seed=3, data_dir=tmp_path / "data")

    for corpus in built.corpora:
        if corpus.group != "bt":
            continue
        available = {
            (row["src"], row["tgt"])
            for row in toy.bt
            if f"{row['src_lang']}-{row['tgt_lang']}" == corpus.direction
        }
        chosen = set(
            zip(file_lines(corpus.src_path), file_lines(corpus.tgt_path), strict=True)
        )
        assert chosen <= available
        assert len(chosen) == corpus.rows


@pytest.mark.parametrize("ratio", [-0.5, float("nan"), float("inf")])
def test_a_ratio_that_is_not_a_finite_non_negative_number_is_refused(
    tmp_path: Path, ratio: float
) -> None:
    training = training_from_rows(tmp_path / "in", numbered_rows(2))

    with pytest.raises(ValueError, match="bt_ratio"):
        build_corpora(training, bt_ratio=ratio, seed=1, data_dir=tmp_path / "data")


# --- build_corpora: reuse of the written files -------------------------------------


def _inodes(built: BuiltCorpora) -> list[int]:
    return [os.stat(c.src_path).st_ino for c in built.corpora]


def test_a_second_build_of_the_same_rows_reuses_every_file(tmp_path: Path) -> None:
    training = toy_training_rows(tmp_path)
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
        *numbered_rows(3),
        *numbered_rows(3, src_lang="rus_Cyrl"),
    ]
    first = build_corpora(
        training_from_rows(tmp_path / "in1", rows),
        bt_ratio=0.0,
        seed=1,
        data_dir=tmp_path / "data",
    )
    changed = [*rows[:2], corpus_row("s2 changed", "t2"), *rows[3:]]

    second = build_corpora(
        training_from_rows(tmp_path / "in2", changed),
        bt_ratio=0.0,
        seed=1,
        data_dir=tmp_path / "data",
    )

    assert [c.reused for c in second.corpora] == [False, True]
    assert sorted(file_lines(_corpus(second, f"{EN_MN}.auth").src_path)) == [
        "s0",
        "s1",
        "s2 changed",
    ]
    assert (
        _corpus(second, f"{EN_MN}.auth").src_sha256
        != _corpus(first, f"{EN_MN}.auth").src_sha256
    )


def test_a_truncated_file_is_rewritten(tmp_path: Path) -> None:
    training = training_from_rows(tmp_path / "in", numbered_rows(20))
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
    training = training_from_rows(tmp_path / "in", numbered_rows(20))
    first = build_corpora(training, bt_ratio=0.0, seed=1, data_dir=tmp_path / "data")
    corpus = first.corpora[0]
    complete = corpus.src_path.read_bytes()
    corpus.src_path.write_bytes(complete.replace(b"s1", b"x1", 1))

    second = build_corpora(training, bt_ratio=0.0, seed=1, data_dir=tmp_path / "data")

    assert second.corpora[0].reused is False
    assert corpus.src_path.read_bytes() == complete


def test_a_missing_file_is_written_again(tmp_path: Path) -> None:
    training = training_from_rows(tmp_path / "in", numbered_rows(5))
    first = build_corpora(training, bt_ratio=0.0, seed=1, data_dir=tmp_path / "data")
    first.corpora[0].src_path.unlink()

    second = build_corpora(training, bt_ratio=0.0, seed=1, data_dir=tmp_path / "data")

    assert second.corpora[0].reused is False
    assert len(file_lines(second.corpora[0].src_path)) == 5


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
    training = training_from_rows(tmp_path / "in", numbered_rows(5))
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
    training = training_from_rows(tmp_path / "in", numbered_rows(5))

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
    training = toy_training_rows(tmp_path)
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
        training_from_rows(tmp_path / "in1", numbered_rows(3)),
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
            training_from_rows(tmp_path / "in2", numbered_rows(4)),
            bt_ratio=0.0,
            seed=1,
            data_dir=tmp_path / "data",
        )

    assert first.corpora[0].src_path.read_bytes() == before
    assert not list((tmp_path / "data").rglob("*.part"))


def test_a_build_leaves_no_temporary_file_behind(tmp_path: Path) -> None:
    build_corpora(
        toy_training_rows(tmp_path), bt_ratio=0.1, seed=1, data_dir=tmp_path / "data"
    )

    assert not list((tmp_path / "data").rglob("*.part"))


# --- planned_corpus_sizes: what a build would write, without writing ---------------

_RATIOS = [0.0, 0.1, 0.25, 0.5, 1.0, 2.5, 1e308]


@pytest.mark.parametrize("ratio", _RATIOS)
def test_the_planned_sizes_are_the_sizes_of_the_corpora_a_build_writes(
    tmp_path: Path, ratio: float
) -> None:
    training = toy_training_rows(tmp_path)

    planned = planned_corpus_sizes(training, ratio)
    built = build_corpora(training, bt_ratio=ratio, seed=1, data_dir=tmp_path / "data")

    assert planned == sizes_of(built.corpora)
    assert bt_usage_by_direction(training, ratio) == built.bt_by_direction


@pytest.mark.parametrize("bt_rows", [0, 3, 10])
@pytest.mark.parametrize("ratio", [0.0, 0.5, 1.5, 3.0])
def test_the_plan_equals_the_build_for_a_direction_with_few_or_no_bt_rows(
    tmp_path: Path, ratio: float, bt_rows: int
) -> None:
    rows = [
        *numbered_rows(5),
        *numbered_rows(bt_rows, origin="bt"),
        *numbered_rows(2, src_lang="rus_Cyrl", origin="web"),
        *numbered_rows(4, src_lang="khk_Cyrl", tgt_lang="rus_Cyrl", origin="bt"),
    ]
    training = training_from_rows(tmp_path / "in", rows)

    built = build_corpora(training, bt_ratio=ratio, seed=7, data_dir=tmp_path / "data")

    assert planned_corpus_sizes(training, ratio) == sizes_of(built.corpora)
    assert bt_usage_by_direction(training, ratio) == built.bt_by_direction


def test_the_planned_sizes_of_the_toy_rows_are_the_hand_counted_ones(
    tmp_path: Path,
) -> None:
    training = toy_training_rows(tmp_path)

    assert planned_corpus_sizes(training, 0.0) == [
        CorpusSize(f"{EN_MN}.auth", EN_MN, 1200),
        CorpusSize(f"{MN_EN}.auth", MN_EN, 1200),
        CorpusSize(f"{MN_RU}.auth", MN_RU, 600),
        CorpusSize(f"{RU_MN}.auth", RU_MN, 600),
    ]
    # 0.2 * 1200 = 240 and 0.2 * 600 = 120 of the 300 bt rows each direction has.
    assert planned_corpus_sizes(training, 0.2) == [
        CorpusSize(f"{EN_MN}.auth", EN_MN, 1200),
        CorpusSize(f"{EN_MN}.bt", EN_MN, 240),
        CorpusSize(f"{MN_EN}.auth", MN_EN, 1200),
        CorpusSize(f"{MN_EN}.bt", MN_EN, 240),
        CorpusSize(f"{MN_RU}.auth", MN_RU, 600),
        CorpusSize(f"{MN_RU}.bt", MN_RU, 120),
        CorpusSize(f"{RU_MN}.auth", RU_MN, 600),
        CorpusSize(f"{RU_MN}.bt", RU_MN, 120),
    ]


# --- planned_corpus_files: where a build would write them ---------------------------


def _as_built(corpora: tuple[CorpusFiles, ...]) -> list[tuple[Any, ...]]:
    return [
        (c.name, c.direction, c.group, c.tgt_lang, c.src_path, c.tgt_path, c.rows)
        for c in corpora
    ]


def _as_planned(files: list[PlannedFiles]) -> list[tuple[Any, ...]]:
    return [
        (f.name, f.direction, f.group, f.tgt_lang, f.src_path, f.tgt_path, f.rows)
        for f in files
    ]


@pytest.mark.parametrize("ratio", _RATIOS)
def test_the_planned_files_are_the_files_a_build_writes(
    tmp_path: Path, ratio: float
) -> None:
    training = toy_training_rows(tmp_path)
    data_dir = tmp_path / "data"

    planned = planned_corpus_files(training, ratio, 7, data_dir)
    built = build_corpora(training, bt_ratio=ratio, seed=7, data_dir=data_dir)

    assert _as_planned(planned) == _as_built(built.corpora)


def test_the_planned_files_of_the_toy_rows_are_the_hand_laid_out_ones(
    tmp_path: Path,
) -> None:
    training = toy_training_rows(tmp_path)
    data = Path("/runs/ab/data")

    planned = planned_corpus_files(training, 0.5, 7, data)

    assert _as_planned(planned)[:2] == [
        (
            f"{EN_MN}.auth",
            EN_MN,
            "auth",
            "khk_Cyrl",
            data / "auth" / f"{EN_MN}.auth.src",
            data / "auth" / f"{EN_MN}.auth.tgt",
            1200,
        ),
        (
            f"{EN_MN}.bt",
            EN_MN,
            "bt",
            "khk_Cyrl",
            data / "bt" / "r0.5_s7" / f"{EN_MN}.bt.src",
            data / "bt" / "r0.5_s7" / f"{EN_MN}.bt.tgt",
            300,
        ),
    ]
    assert [(f.name, f.tgt_lang) for f in planned][4:6] == [
        (f"{MN_RU}.auth", "rus_Cyrl"),
        (f"{MN_RU}.bt", "rus_Cyrl"),
    ]


def test_planning_the_files_writes_nothing(tmp_path: Path) -> None:
    training = toy_training_rows(tmp_path)
    before = sorted(tmp_path.rglob("*"))

    planned_corpus_files(training, 0.5, 7, tmp_path / "data")

    assert sorted(tmp_path.rglob("*")) == before


def test_the_planned_files_follow_the_planned_sizes(tmp_path: Path) -> None:
    training = toy_training_rows(tmp_path)

    files = planned_corpus_files(training, 0.2, 1, tmp_path / "data")

    assert [(f.name, f.direction, f.rows) for f in files] == [
        (size.name, size.direction, size.rows)
        for size in planned_corpus_sizes(training, 0.2)
    ]


@pytest.mark.parametrize("ratio", [-1.0, float("nan"), float("inf")])
def test_files_cannot_be_planned_for_an_unusable_ratio(
    tmp_path: Path, ratio: float
) -> None:
    with pytest.raises(ValueError, match="bt_ratio"):
        planned_corpus_files(toy_training_rows(tmp_path), ratio, 1, tmp_path / "data")


def test_a_direction_with_bt_rows_only_has_no_planned_corpus_but_a_usage_entry(
    tmp_path: Path,
) -> None:
    rows = [*numbered_rows(4), *numbered_rows(3, src_lang="rus_Cyrl", origin="bt")]
    training = training_from_rows(tmp_path / "in", rows)

    assert planned_corpus_sizes(training, 1.0) == [
        CorpusSize(f"{EN_MN}.auth", EN_MN, 4)
    ]
    usage = bt_usage_by_direction(training, 1.0)
    assert list(usage) == [EN_MN, RU_MN]
    assert (usage[RU_MN].bt_available, usage[RU_MN].bt_used) == (3, 0)


@pytest.mark.parametrize("ratio", [-0.5, float("nan"), float("inf")])
def test_the_planned_sizes_refuse_the_ratios_a_build_refuses(
    tmp_path: Path, ratio: float
) -> None:
    training = training_from_rows(tmp_path / "in", numbered_rows(2))

    with pytest.raises(ValueError, match="bt_ratio"):
        planned_corpus_sizes(training, ratio)
    with pytest.raises(ValueError, match="bt_ratio"):
        bt_usage_by_direction(training, ratio)


def test_planning_writes_nothing_and_leaves_the_rows_as_they_were(
    tmp_path: Path,
) -> None:
    training = toy_training_rows(tmp_path)
    before = sorted(path.name for path in tmp_path.rglob("*"))
    frame = training.frame.clone()

    planned_corpus_sizes(training, 0.5)

    assert sorted(path.name for path in tmp_path.rglob("*")) == before
    assert training.frame.equals(frame)


def test_a_direction_with_one_authentic_row_is_planned_and_built(
    tmp_path: Path,
) -> None:
    rows = [*numbered_rows(1), *numbered_rows(1, origin="bt")]
    training = training_from_rows(tmp_path / "in", rows)

    built = build_corpora(training, bt_ratio=1.0, seed=1, data_dir=tmp_path / "data")

    assert planned_corpus_sizes(training, 1.0) == [
        CorpusSize(f"{EN_MN}.auth", EN_MN, 1),
        CorpusSize(f"{EN_MN}.bt", EN_MN, 1),
    ]
    assert sizes_of(built.corpora) == planned_corpus_sizes(training, 1.0)


# --- write_corpus_files: the corpus of any frame -----------------------------------


def test_a_corpus_name_is_the_direction_and_the_group() -> None:
    assert corpus_name(EN_MN, "auth") == "eng_Latn-khk_Cyrl.auth"
    assert corpus_name(MN_EN, "doc") == "khk_Cyrl-eng_Latn.doc"


def test_a_frame_is_written_as_the_corpus_of_a_direction_and_a_group(
    tmp_path: Path,
) -> None:
    frame = pl.DataFrame(
        {
            "id": ["b", "a"],
            "src": ["Two", "One"],
            "tgt": ["Хоёр", "Нэг"],
            "src_lang": ["eng_Latn", "eng_Latn"],
            "tgt_lang": ["khk_Cyrl", "khk_Cyrl"],
        }
    )

    first = write_corpus_files(frame, tmp_path, EN_MN, "doc")
    again = write_corpus_files(frame, tmp_path, EN_MN, "doc")

    assert first.name == "eng_Latn-khk_Cyrl.doc"
    assert (first.direction, first.group, first.rows) == (EN_MN, "doc", 2)
    assert (first.src_lang, first.tgt_lang) == ("eng_Latn", "khk_Cyrl")
    assert first.src_path == tmp_path / "eng_Latn-khk_Cyrl.doc.src"
    assert first.src_path.read_bytes() == b"One\nTwo\n"
    assert first.tgt_path.read_bytes() == "Нэг\nХоёр\n".encode()
    assert (first.reused, again.reused) == (False, True)
    assert (tmp_path / "eng_Latn-khk_Cyrl.doc.json").is_file()
