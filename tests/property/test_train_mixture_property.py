"""Property tests for `rinchen.train.mixture` over random rows: every row of every
file is counted exactly once and the counts equal an independent Python account of
the rules; the bt subsample does not depend on the order of the rows or the files,
is the same on every run, and is the rows with the smallest seeded hash among the
available ones; the source collision rate agrees with a dictionary count; the line
ranges of the validation corpus tile it.

The tests that write files cap their examples: every example writes Parquet.
"""

from __future__ import annotations

import hashlib
import tempfile
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import polars as pl
from hypothesis import given, settings
from hypothesis import strategies as st

from rinchen.corpus.schema import ORIGINS, SPLITS, corpus_frame, make_row_id
from rinchen.corpus.writers import write_corpus
from rinchen.train.mixture import (
    DevSet,
    build_corpora,
    dev_line_ranges,
    load_training_rows,
    source_collision_rate,
)

_LANGUAGES = ("eng_Latn", "khk_Cyrl", "rus_Cyrl")
_EXCLUDED_ORIGINS = ("ft", "synth_llm", "synth_nmt")
# Texts that are fine, empty, blank, or hold a character that splits lines.
_TEXTS = st.sampled_from(
    [
        "a",
        "b",
        "ab ba",
        "Зүгээр",
        "tab\there",
        "",
        " ",
        chr(0x3000),
        chr(0x1C),
        chr(0x1F),
        "\n",
        "\r\n",
        "x\ny",
        "x\r",
        "x" + chr(0x2028),
        "x" + chr(0x85),
        "x" + chr(0x1C) + "y",
    ]
)
_ROWS = st.fixed_dictionaries(
    {
        "src_lang": st.sampled_from(_LANGUAGES),
        "tgt_lang": st.sampled_from(_LANGUAGES),
        "src": _TEXTS,
        "tgt": _TEXTS,
        # Weighted towards the rows that can train, so that most examples keep some.
        "origin": st.sampled_from(["human", "human", "bt", *ORIGINS]),
        "corpus": st.just("prop"),
        "split": st.sampled_from(["train", "train", "train", *SPLITS]),
    }
)
# One row that always trains, so that a file set always has an authentic row.
_ANCHOR = {
    "src_lang": "eng_Latn",
    "tgt_lang": "khk_Cyrl",
    "src": "anchor",
    "tgt": "anchor",
    "origin": "human",
    "corpus": "prop",
    "split": "train",
}
_FILTER_REASONS = st.sampled_from([None, None, None, "lid_src", "numbers"])
_FILE_COUNT = st.integers(min_value=1, max_value=3)


def _write(
    directory: Path,
    name: str,
    rows: Sequence[dict[str, str]],
    reasons: Sequence[str | None] | None = None,
) -> Path:
    frame = corpus_frame(rows)
    if reasons is not None:
        frame = frame.with_columns(
            pl.Series("filter_reason", list(reasons), dtype=pl.String)
        )
    path = directory / f"{name}.parquet"
    write_corpus(frame, path)
    return path


def _python_reason(row: dict[str, str], filter_reason: str | None) -> str | None:
    """The reason the rules give a row, written as Python: `str.strip` for empty and
    `str.splitlines` for a line break."""
    if row["split"] != "train":
        return "not_train"
    if filter_reason is not None:
        return "filtered"
    if row["origin"] in _EXCLUDED_ORIGINS:
        return "excluded_origin"
    if not row["src"].strip() or not row["tgt"].strip():
        return "empty"
    if any(text.splitlines() != [text] for text in (row["src"], row["tgt"])):
        return "line_break"
    return None


@settings(max_examples=60)
@given(
    extra=st.lists(st.tuples(_ROWS, _FILTER_REASONS), max_size=25),
    files=_FILE_COUNT,
)
def test_every_row_is_counted_once_and_the_counts_match_a_python_account(
    extra: list[tuple[dict[str, str], str | None]], files: int
) -> None:
    rows = [(_ANCHOR, None), *extra]
    expected = {
        "not_train": 0,
        "filtered": 0,
        "excluded_origin": 0,
        "empty": 0,
        "line_break": 0,
    }
    by_origin = dict.fromkeys(_EXCLUDED_ORIGINS, 0)
    kept: list[dict[str, str]] = []
    for row, reason in rows:
        outcome = _python_reason(row, reason)
        if outcome is None:
            kept.append(row)
            continue
        expected[outcome] += 1
        if outcome == "excluded_origin":
            by_origin[row["origin"]] += 1
    with tempfile.TemporaryDirectory() as name:
        directory = Path(name)
        paths = [
            _write(
                directory,
                f"part{index}",
                [row for row, _ in rows[index::files]],
                [reason for _, reason in rows[index::files]],
            )
            for index in range(files)
        ]
        stats = load_training_rows(paths).stats

    assert stats.rows_read == len(rows)
    assert stats.rows_kept == len(kept)
    assert stats.excluded == expected
    assert stats.excluded_by_origin == by_origin
    assert stats.rows_read == stats.rows_kept + sum(stats.excluded.values())
    assert sum(stats.rows_by_corpus.values()) == stats.rows_kept


_WORDS = st.text(alphabet="abcdef", min_size=1, max_size=6)


def _bt_input(auth: Sequence[str], bt: Sequence[str]) -> list[dict[str, str]]:
    base = {
        "src_lang": "eng_Latn",
        "tgt_lang": "khk_Cyrl",
        "corpus": "prop",
        "split": "train",
    }
    return [
        *(
            {**base, "src": f"a-{word}", "tgt": f"A-{word}", "origin": "human"}
            for word in auth
        ),
        *(
            {**base, "src": f"b-{word}", "tgt": f"B-{word}", "origin": "bt"}
            for word in bt
        ),
    ]


_AUTH_WORDS = st.lists(_WORDS, min_size=1, max_size=12, unique=True)
_BT_WORDS = st.lists(_WORDS, max_size=20, unique=True)
_RATIOS = st.sampled_from([0.25, 0.5, 1.0, 1.5, 3.0])
_SEEDS = st.integers(min_value=0, max_value=2**31)


def _build(
    directory: Path,
    rows: Sequence[dict[str, str]],
    *,
    ratio: float,
    seed: int,
    parts: int = 1,
) -> Any:
    (directory / "in").mkdir(parents=True)
    paths = [
        _write(directory / "in", f"part{index}", rows[index::parts])
        for index in range(parts)
    ]
    return build_corpora(
        load_training_rows(paths),
        bt_ratio=ratio,
        seed=seed,
        data_dir=directory / "data",
    )


def _fingerprint(built: Any) -> list[tuple[str, str, str, int]]:
    return [(c.name, c.src_sha256, c.tgt_sha256, c.rows) for c in built.corpora]


@settings(max_examples=30)
@given(
    auth=_AUTH_WORDS,
    bt=_BT_WORDS,
    ratio=_RATIOS,
    seed=_SEEDS,
    data=st.data(),
)
def test_the_files_do_not_depend_on_the_order_of_the_rows_or_the_files(
    auth: list[str], bt: list[str], ratio: float, seed: int, data: st.DataObject
) -> None:
    rows = _bt_input(auth, bt)
    shuffled = data.draw(st.permutations(rows))
    parts = data.draw(st.integers(min_value=1, max_value=3))
    with tempfile.TemporaryDirectory() as name:
        directory = Path(name)
        first = _build(directory / "first", rows, ratio=ratio, seed=seed)
        second = _build(
            directory / "second", shuffled, ratio=ratio, seed=seed, parts=parts
        )

        assert _fingerprint(first) == _fingerprint(second)
        assert first.bt_by_direction == second.bt_by_direction


@settings(max_examples=30)
@given(auth=_AUTH_WORDS, bt=_BT_WORDS, ratio=_RATIOS, seed=_SEEDS)
def test_a_second_run_gives_the_same_files(
    auth: list[str], bt: list[str], ratio: float, seed: int
) -> None:
    rows = _bt_input(auth, bt)
    with tempfile.TemporaryDirectory() as name:
        directory = Path(name)

        first = _build(directory / "first", rows, ratio=ratio, seed=seed)
        second = _build(directory / "second", rows, ratio=ratio, seed=seed)

        assert _fingerprint(first) == _fingerprint(second)


@settings(max_examples=30)
@given(auth=_AUTH_WORDS, bt=_BT_WORDS, ratio=_RATIOS, seed=_SEEDS)
def test_the_subsample_is_the_smallest_seeded_hashes_of_the_available_bt_rows(
    auth: list[str], bt: list[str], ratio: float, seed: int
) -> None:
    rows = _bt_input(auth, bt)
    ids = {
        row["src"]: make_row_id(
            row["src_lang"], row["tgt_lang"], row["src"], row["tgt"]
        )
        for row in rows
    }
    expected_size = min(len(bt), round(ratio * len(auth)))
    with tempfile.TemporaryDirectory() as name:
        built = _build(Path(name), rows, ratio=ratio, seed=seed)

        chosen: set[str] = set()
        for corpus in (c for c in built.corpora if c.group == "bt"):
            chosen = set(corpus.src_path.read_text(encoding="utf-8").splitlines())
    available = {f"b-{word}" for word in bt}

    assert len(chosen) == expected_size
    assert chosen <= available
    digest = {
        src: hashlib.sha256(f"{seed}\t{ids[src]}".encode()).hexdigest()
        for src in available
    }
    assert all(
        digest[kept] <= digest[other] for kept in chosen for other in available - chosen
    )
    usage = built.bt_by_direction["eng_Latn-khk_Cyrl"]
    assert (usage.bt_available, usage.bt_requested, usage.bt_used) == (
        len(bt),
        round(ratio * len(auth)),
        expected_size,
    )


@given(
    pairs=st.lists(
        st.tuples(st.sampled_from("ABCD"), st.sampled_from(_LANGUAGES)), max_size=30
    )
)
def test_the_collision_rate_agrees_with_a_dictionary_count(
    pairs: list[tuple[str, str]],
) -> None:
    frame = pl.DataFrame(
        {"src": [src for src, _ in pairs], "tgt_lang": [lang for _, lang in pairs]},
        schema={"src": pl.String, "tgt_lang": pl.String},
    )
    languages: dict[str, set[str]] = defaultdict(set)
    for src, lang in pairs:
        languages[src].add(lang)
    colliding = sum(1 for src, _ in pairs if len(languages[src]) > 1)

    rate = source_collision_rate(frame)

    assert 0.0 <= rate <= 1.0
    assert rate == (colliding / len(pairs) if pairs else 0.0)


@given(
    sizes=st.dictionaries(
        st.sampled_from(["a-b", "c-d", "e-f", "g-h"]), st.integers(1, 6)
    )
)
def test_the_line_ranges_tile_the_concatenated_corpus_in_sorted_order(
    sizes: dict[str, int],
) -> None:
    dev = {
        direction: DevSet(
            direction=direction,
            src_lang="eng_Latn",
            tgt_lang="khk_Cyrl",
            ids=tuple(str(i) for i in range(count)),
            src=("s",) * count,
            tgt=("t",) * count,
        )
        for direction, count in sizes.items()
    }

    ranges = dev_line_ranges(dev)

    assert list(ranges) == sorted(sizes)
    position = 0
    for direction in sorted(sizes):
        assert ranges[direction] == range(position, position + sizes[direction])
        position += sizes[direction]
