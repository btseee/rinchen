"""`rinchen.tiny.corpus`: the pair corpus of the tiny pipeline and its expansion to both directions
(D15, spec P6, brief G3.1).

`pair_rows` makes one row per PAIR, in the direction the pair was acquired, from the parsed fixtures
and a fixed stamp alone, so the corpus is a function of the fixtures. `expand_directions` is applied
AFTER the filter: every kept row of the splits `train` and `dev` of no noise corpus, plus its
reverse except for the traditional-script rows. The expansion is checked against a brute-force swap
on 300 random frames; the corpus is checked against the fixtures read without the library under
test, and in two processes under two hash seeds.
"""

from __future__ import annotations

import dataclasses
import hashlib
import random
from collections import Counter
from datetime import UTC, datetime
from typing import Any, Final

import polars as pl
import pytest

from rinchen.config.loader import REPO_ROOT
from rinchen.corpus.schema import (
    CORPUS_SCHEMA,
    ORIGINS,
    SPLITS,
    corpus_frame,
    validate_corpus_schema,
)
from rinchen.errors import DataError
from rinchen.filter.reasons import with_scores
from rinchen.tiny.corpus import (
    MONG_CORPUS,
    NOISE_CORPUS,
    build_pair_corpus,
    expand_directions,
    pair_rows,
)
from rinchen.tiny.fixtures import Fixtures, fixture_digests, read_fixtures
from tests.support.no_network import no_network  # noqa: F401 - the fixture of the mark
from tests.support.tiny_children import run_python

pytestmark = pytest.mark.usefixtures("no_network")

FIXTURES_DIR: Final = REPO_ROOT / "tests" / "fixtures" / "parallel"
CREATED: Final = datetime(2026, 11, 11, tzinfo=UTC)
OTHER_STAMP: Final = datetime(2020, 1, 2, 3, 4, 5, tzinfo=UTC)
ENG_LABEL: Final = "tiny-eng_Latn-khk_Cyrl"
RUS_LABEL: Final = "tiny-rus_Cyrl-khk_Cyrl"
COLUMNS: Final = (
    "src_lang",
    "tgt_lang",
    "src",
    "tgt",
    "origin",
    "corpus",
    "split",
    "doc_id",
    "seg_idx",
)
CHILD_PROGRAM: Final = """
import hashlib
import sys
from datetime import UTC, datetime
from pathlib import Path

from rinchen.tiny.corpus import expand_directions, pair_rows
from rinchen.tiny.fixtures import fixture_digests, read_fixtures

directory = Path(sys.argv[1])
fixtures = read_fixtures(directory, fixture_digests(directory))
created = datetime(2026, 11, 11, tzinfo=UTC)
frame = pair_rows(fixtures, created)
digest = hashlib.sha256()
for part in (frame, expand_directions(frame, created)):
    digest.update(repr(part.rows()).encode("utf-8"))
print(digest.hexdigest())
"""


@pytest.fixture(scope="module")
def fixtures() -> Fixtures:
    return read_fixtures(FIXTURES_DIR, fixture_digests(FIXTURES_DIR))


@pytest.fixture(scope="module")
def corpus(fixtures: Fixtures) -> pl.DataFrame:
    return pair_rows(fixtures, CREATED)


def counts(frame: pl.DataFrame, *columns: str) -> dict[Any, int]:
    return dict(Counter(frame.select(columns).iter_rows()))


# ---- G3.1: the pair corpus -------------------------------------------------------------------


def test_the_pair_corpus_has_one_row_per_pair_of_the_four_files(
    corpus: pl.DataFrame,
) -> None:
    assert corpus.height == 345 + 345 + 70 + 10 == 770
    assert counts(corpus, "corpus") == {
        (ENG_LABEL,): 345,
        (RUS_LABEL,): 345,
        (MONG_CORPUS,): 70,
        (NOISE_CORPUS,): 10,
    }


def test_the_corpus_labels_are_tiny_and_the_stem_of_the_file(
    corpus: pl.DataFrame,
) -> None:
    assert (MONG_CORPUS, NOISE_CORPUS) == ("tiny-mn_Mong", "tiny-noise")
    assert set(corpus["corpus"]) == {ENG_LABEL, RUS_LABEL, MONG_CORPUS, NOISE_CORPUS}


def test_the_splits_are_those_of_the_fixtures_and_the_noise_is_held_out(
    corpus: pl.DataFrame,
) -> None:
    assert counts(corpus, "split") == {
        ("train",): 650,
        ("dev",): 40,
        ("test",): 70,
        ("holdout",): 10,
    }
    noise = corpus.filter(pl.col("corpus") == NOISE_CORPUS)
    assert set(noise["split"]) == {"holdout"}
    assert corpus.filter(pl.col("split") == "holdout").height == noise.height


def test_the_origins_say_where_each_row_came_from(corpus: pl.DataFrame) -> None:
    assert counts(corpus, "corpus", "origin") == {
        (ENG_LABEL, "human"): 345,
        (RUS_LABEL, "human"): 345,
        (MONG_CORPUS, "converted"): 70,
        (NOISE_CORPUS, "web"): 10,
    }


def test_each_row_keeps_the_direction_it_was_acquired_in(corpus: pl.DataFrame) -> None:
    assert counts(corpus, "corpus", "src_lang", "tgt_lang") == {
        (ENG_LABEL, "eng_Latn", "khk_Cyrl"): 345,
        (RUS_LABEL, "rus_Cyrl", "khk_Cyrl"): 345,
        (MONG_CORPUS, "mn_Mong", "eng_Latn"): 70,
        (NOISE_CORPUS, "eng_Latn", "khk_Cyrl"): 10,
    }


def test_the_corpora_come_in_the_order_of_the_files(corpus: pl.DataFrame) -> None:
    assert list(dict.fromkeys(corpus["corpus"])) == [
        ENG_LABEL,
        RUS_LABEL,
        MONG_CORPUS,
        NOISE_CORPUS,
    ]


def test_the_texts_are_those_of_the_fixtures_in_the_order_of_the_files(
    fixtures: Fixtures, corpus: pl.DataFrame
) -> None:
    def pairs_of(label: str) -> list[tuple[str, str]]:
        part = corpus.filter(pl.col("corpus") == label)
        return list(zip(part["src"], part["tgt"], strict=True))

    assert pairs_of(ENG_LABEL) == [
        (r.src, r.tgt) for r in fixtures.pairs["eng_Latn-khk_Cyrl"]
    ]
    assert pairs_of(RUS_LABEL) == [
        (r.src, r.tgt) for r in fixtures.pairs["rus_Cyrl-khk_Cyrl"]
    ]
    assert pairs_of(MONG_CORPUS) == [(r.mong, r.eng) for r in fixtures.mong]
    assert pairs_of(NOISE_CORPUS) == [(r.src, r.tgt) for r in fixtures.noise]
    split_of = corpus.filter(pl.col("corpus") == ENG_LABEL)["split"].to_list()
    assert split_of == [r.split for r in fixtures.pairs["eng_Latn-khk_Cyrl"]]


def test_the_noise_rows_keep_their_own_directions(fixtures: Fixtures) -> None:
    mixed = dataclasses.replace(
        fixtures,
        noise=(dataclasses.replace(fixtures.noise[0], src_lang="rus_Cyrl"),)
        + fixtures.noise[1:],
    )
    frame = pair_rows(mixed, CREATED)
    noise = frame.filter(pl.col("corpus") == NOISE_CORPUS)
    assert noise["src_lang"].to_list()[0] == "rus_Cyrl"
    assert noise["src_lang"].to_list()[1:] == ["eng_Latn"] * 9


def test_one_created_stamp_is_on_every_row_and_the_clock_is_not_read(
    fixtures: Fixtures, corpus: pl.DataFrame
) -> None:
    assert corpus["created"].unique().to_list() == [CREATED]
    assert pair_rows(fixtures, OTHER_STAMP)["created"].unique().to_list() == [
        OTHER_STAMP
    ]
    assert pair_rows(fixtures, CREATED).equals(corpus)


def test_the_frame_is_a_valid_corpus_with_the_ids_of_the_corpus_contract(
    corpus: pl.DataFrame,
) -> None:
    validate_corpus_schema(corpus.schema, source="test")
    assert corpus.schema == CORPUS_SCHEMA
    for row in corpus.iter_rows(named=True):
        joined = f"{row['src_lang']}\t{row['tgt_lang']}\t{row['src']}\t{row['tgt']}"
        assert row["id"] == hashlib.sha256(joined.encode()).hexdigest()[:16]
        assert row["filter_reason"] is None
        assert row["scores"] is None


def test_nothing_is_dropped_from_the_committed_fixtures(fixtures: Fixtures) -> None:
    built = build_pair_corpus(fixtures, CREATED)
    assert dict(built.dropped) == {
        ENG_LABEL: 0,
        RUS_LABEL: 0,
        MONG_CORPUS: 0,
        NOISE_CORPUS: 0,
    }
    assert built.frame.equals(pair_rows(fixtures, CREATED))


def test_a_text_that_is_not_one_sentence_is_counted_and_left_out(
    fixtures: Fixtures,
) -> None:
    key = "eng_Latn-khk_Cyrl"
    rows = list(fixtures.pairs[key])
    two = "One sentence here. Another sentence here."
    rows[0] = dataclasses.replace(rows[0], src=two)
    rows[1] = dataclasses.replace(rows[1], tgt=f"{rows[3].tgt} {rows[4].tgt}")
    rows[2] = dataclasses.replace(rows[2], src="   ")
    changed = dataclasses.replace(fixtures, pairs={**fixtures.pairs, key: tuple(rows)})

    built = build_pair_corpus(changed, CREATED)

    assert built.dropped[ENG_LABEL] == 3
    assert built.dropped[RUS_LABEL] == 0
    assert counts(built.frame, "corpus")[(ENG_LABEL,)] == 342
    assert built.frame.height == 770 - 3
    remaining = built.frame.filter(pl.col("corpus") == ENG_LABEL)["src"].to_list()
    assert two not in remaining
    assert remaining == [r.src for r in rows[3:]]


def test_a_text_is_normalised_before_it_is_split(fixtures: Fixtures) -> None:
    key = "rus_Cyrl-khk_Cyrl"
    rows = list(fixtures.pairs[key])
    rows[0] = dataclasses.replace(
        rows[0], tgt=rows[0].tgt.replace(" ", "  ", 1) + "\u200b"
    )
    changed = dataclasses.replace(fixtures, pairs={**fixtures.pairs, key: tuple(rows)})

    frame = pair_rows(changed, CREATED)

    first = frame.filter(pl.col("corpus") == RUS_LABEL)["tgt"].to_list()[0]
    assert first == fixtures.pairs[key][0].tgt


def test_the_corpus_is_equal_in_two_processes_under_two_hash_seeds(
    corpus: pl.DataFrame,
) -> None:
    digest = hashlib.sha256()
    for part in (corpus, expand_directions(corpus, CREATED)):
        digest.update(repr(part.rows()).encode("utf-8"))
    mine = digest.hexdigest()

    for seed in ("0", "3"):
        found = run_python(CHILD_PROGRAM, str(FIXTURES_DIR), hash_seed=seed).strip()
        assert found == mine, seed


# ---- G3.1: the expansion ---------------------------------------------------------------------


def test_an_unfiltered_corpus_expands_to_1320_rows(corpus: pl.DataFrame) -> None:
    expanded = expand_directions(corpus, CREATED)

    assert expanded.height == 2 * (600 + 30) + 50 + 10 == 1320
    assert expanded["id"].n_unique() == 1320
    assert set(expanded["split"]) == {"train", "dev"}
    assert NOISE_CORPUS not in set(expanded["corpus"])
    validate_corpus_schema(expanded.schema, source="test")


def test_both_directions_of_each_pair_are_there_and_one_of_each_traditional_row(
    corpus: pl.DataFrame,
) -> None:
    expanded = expand_directions(corpus, CREATED)

    assert counts(expanded, "corpus", "src_lang", "tgt_lang") == {
        (ENG_LABEL, "eng_Latn", "khk_Cyrl"): 315,
        (ENG_LABEL, "khk_Cyrl", "eng_Latn"): 315,
        (RUS_LABEL, "rus_Cyrl", "khk_Cyrl"): 315,
        (RUS_LABEL, "khk_Cyrl", "rus_Cyrl"): 315,
        (MONG_CORPUS, "mn_Mong", "eng_Latn"): 60,
    }
    assert counts(expanded, "split") == {("train",): 1250, ("dev",): 70}


def test_the_reverse_of_a_row_swaps_the_sides_and_keeps_the_rest(
    corpus: pl.DataFrame,
) -> None:
    expanded = expand_directions(corpus, CREATED)
    forward = {
        (r["src_lang"], r["tgt_lang"], r["src"], r["tgt"], r["corpus"], r["split"])
        for r in expanded.iter_rows(named=True)
    }
    train_dev = corpus.filter(
        pl.col("split").is_in(["train", "dev"]) & (pl.col("corpus") != NOISE_CORPUS)
    )
    for row in train_dev.iter_rows(named=True):
        key = (row["src_lang"], row["tgt_lang"], row["src"], row["tgt"]) + (
            row["corpus"],
            row["split"],
        )
        assert key in forward
        if row["corpus"] != MONG_CORPUS:
            swapped = (row["tgt_lang"], row["src_lang"], row["tgt"], row["src"]) + (
                row["corpus"],
                row["split"],
            )
            assert swapped in forward


def test_the_forward_rows_come_first_and_then_the_reversed_ones_in_the_same_order(
    corpus: pl.DataFrame,
) -> None:
    expanded = expand_directions(corpus, CREATED)
    kept = corpus.filter(
        pl.col("split").is_in(["train", "dev"]) & (pl.col("corpus") != NOISE_CORPUS)
    )

    assert expanded.head(kept.height)["src"].to_list() == kept["src"].to_list()
    reversible = kept.filter(pl.col("corpus") != MONG_CORPUS)
    assert (
        expanded.tail(reversible.height)["tgt"].to_list() == reversible["src"].to_list()
    )


def test_the_expansion_takes_its_stamp_from_the_argument_and_drops_the_scores(
    corpus: pl.DataFrame,
) -> None:
    scored = with_scores(corpus.lazy(), labse=pl.lit(0.5)).collect()
    assert scored["scores"].null_count() == 0
    expanded = expand_directions(scored, OTHER_STAMP)

    assert expanded["created"].unique().to_list() == [OTHER_STAMP]
    assert expanded["scores"].null_count() == expanded.height
    assert expanded["filter_reason"].null_count() == expanded.height


def test_a_corpus_with_nothing_to_expand_gives_an_empty_corpus(
    corpus: pl.DataFrame,
) -> None:
    only_held_out = corpus.filter(pl.col("split").is_in(["test", "holdout"]))

    expanded = expand_directions(only_held_out, CREATED)

    assert expanded.height == 0
    assert expanded.schema == CORPUS_SCHEMA


def test_the_expansion_refuses_rows_the_filter_rejected(corpus: pl.DataFrame) -> None:
    scored = corpus.with_columns(filter_reason=pl.lit("duplicate"))

    with pytest.raises(DataError, match="770"):
        expand_directions(scored, CREATED)


def test_the_expansion_refuses_a_table_that_is_not_a_corpus() -> None:
    with pytest.raises(DataError):
        expand_directions(pl.DataFrame({"src": ["a"]}), CREATED)


def test_the_expansion_leaves_its_input_alone(corpus: pl.DataFrame) -> None:
    before = corpus.clone()

    expand_directions(corpus, CREATED)

    assert corpus.equals(before)


# ---- the differential check: a brute-force swap on 300 random frames ---------------------------

LANGS: Final = ("eng_Latn", "khk_Cyrl", "rus_Cyrl", "mn_Mong")
CORPORA: Final = (ENG_LABEL, RUS_LABEL, MONG_CORPUS, NOISE_CORPUS, "other")
ALPHABET: Final = "abcxyz 12.,"


def random_frame(seed: int) -> pl.DataFrame:
    rnd = random.Random(seed)
    rows: list[dict[str, Any]] = []
    for _ in range(rnd.randint(0, 40)):
        src_lang, tgt_lang = rnd.sample(LANGS, 2)
        rows.append(
            {
                "src_lang": src_lang,
                "tgt_lang": tgt_lang,
                "src": "".join(rnd.choices(ALPHABET, k=rnd.randint(1, 8))),
                "tgt": "".join(rnd.choices(ALPHABET, k=rnd.randint(1, 8))),
                "origin": rnd.choice(ORIGINS),
                "corpus": rnd.choice(CORPORA),
                "split": rnd.choice(SPLITS),
                "doc_id": rnd.choice([None, "doc-1", "doc-2"]),
                "seg_idx": rnd.choice([None, 0, 7]),
            }
        )
    return corpus_frame(rows, created=OTHER_STAMP)


def brute_force_expansion(frame: pl.DataFrame) -> list[tuple[Any, ...]]:
    rows = frame.to_dicts()
    wanted = [
        r
        for r in rows
        if r["split"] in ("train", "dev") and r["corpus"] != "tiny-noise"
    ]
    forward = [tuple(r[c] for c in COLUMNS) for r in wanted]
    swapped = []
    for r in wanted:
        if r["corpus"] == "tiny-mn_Mong":
            continue
        flipped = dict(r, src_lang=r["tgt_lang"], tgt_lang=r["src_lang"])
        flipped["src"], flipped["tgt"] = r["tgt"], r["src"]
        swapped.append(tuple(flipped[c] for c in COLUMNS))
    return forward + swapped


def test_the_expansion_equals_a_brute_force_swap_on_300_random_frames() -> None:
    nonempty = 0
    for seed in range(300):
        frame = random_frame(seed)
        expected = brute_force_expansion(frame)

        got = expand_directions(frame, CREATED)

        assert got.select(COLUMNS).rows() == expected, seed
        ids = [
            hashlib.sha256(f"{r[0]}\t{r[1]}\t{r[2]}\t{r[3]}".encode()).hexdigest()[:16]
            for r in expected
        ]
        assert got["id"].to_list() == ids, seed
        assert got["created"].unique().to_list() in ([CREATED], []), seed
        nonempty += bool(expected)
    assert nonempty > 150
