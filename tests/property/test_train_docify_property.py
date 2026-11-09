"""Property tests for `rinchen.train.docify` over random documents: whatever the rows
(gaps and repeats in `seg_idx`, empty and `<sep>` sentences, every origin and split, null
document ids), every row is counted once, the windows stay within their bounds, hold only
consecutive segments of one document and one direction, and do not depend on the order of
the rows."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from hypothesis import given
from hypothesis import strategies as st

from rinchen.train.docify import docify
from tests.support.doc_rows import doc_row, rows_frame
from tests.support.tiny_parallel import train_toy_spm

# Mostly clean rows with a few of each fault, so that documents of consecutive sentences
# are common and the faults still turn up in every example set.
_ORIGINS = ["human"] * 12 + ["web", "mined", "bt", "ft", "synth_nmt"]
_SPLITS = ["train"] * 10 + ["dev", "test"]
_LANGUAGES = [("eng_Latn", "khk_Cyrl"), ("khk_Cyrl", "eng_Latn")]
_DOCUMENTS = ["d1", "d2", "d3", "d1", "d2", "d3", "", "  ", None]
_REASONS = [None] * 12 + ["lid"]
_TEXTS = [None] * 14 + ["", "  \t", "a\nb", "x <sep> y"]


@pytest.fixture(scope="module")
def toy_spm(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return train_toy_spm(tmp_path_factory.mktemp("docify_property_spm"))


@st.composite
def corpus_rows(draw: st.DrawFn) -> tuple[list[dict[str, Any]], list[str | None]]:
    """Rows of up to four documents, each in one direction, with the faults a corpus has:
    gaps and repeats in `seg_idx`, a missing one, blank and `<sep>` texts, other splits
    and origins, filtered rows and a document id that is missing or blank."""
    rows: list[dict[str, Any]] = []
    reasons: list[str | None] = []
    for _ in range(draw(st.integers(0, 4))):
        doc_id = draw(st.sampled_from(_DOCUMENTS))
        languages = draw(st.sampled_from(_LANGUAGES))
        for seg in draw(st.lists(st.integers(0, 11), max_size=14)):
            row = doc_row(
                doc_id,
                None if draw(st.integers(0, 40)) == 0 else seg,
                draw(st.integers(0, 7)),
                languages=languages,
                origin=draw(st.sampled_from(_ORIGINS)),
                split=draw(st.sampled_from(_SPLITS)),
            )
            text = draw(st.sampled_from(_TEXTS))
            if text is not None:
                row[draw(st.sampled_from(["src", "tgt"]))] = text
            rows.append(row)
            reasons.append(draw(st.sampled_from(_REASONS)))
    return rows, reasons


_SIZES = st.integers(1, 4).flatmap(
    lambda low: st.tuples(st.just(low), st.integers(low, 5))
)


@given(
    corpus_rows(),
    _SIZES,
    st.integers(8, 60),
    st.integers(0, 100),
    st.randoms(use_true_random=False),
)
def test_windows_count_every_row_once_and_stay_within_their_bounds(
    toy_spm: Path,
    drawn: tuple[list[dict[str, Any]], list[str | None]],
    sizes: tuple[int, int],
    max_pieces: int,
    seed: int,
    shuffler: Any,
) -> None:
    rows, reasons = drawn
    low, high = sizes
    frame = rows_frame(rows, filter_reasons=reasons) if rows else None
    if frame is None:
        return

    windows = docify(
        frame,
        spm_model=toy_spm,
        min_size=low,
        max_size=high,
        max_pieces=max_pieces,
        seed=seed,
    )

    stats = windows.stats
    # The first reasons are plain facts about a row.
    in_split = [
        (row, why)
        for row, why in zip(rows, reasons, strict=True)
        if row["split"] == "train"
    ]
    assert stats.excluded["not_train"] == len(rows) - len(in_split)
    assert stats.excluded["filtered"] == sum(why is not None for _, why in in_split)
    assert stats.excluded["excluded_origin"] == sum(
        why is None and row["origin"] in ("bt", "ft", "synth_nmt")
        for row, why in in_split
    )
    # Conservation: every row is in a window or under one reason.
    assert stats.rows_read == len(rows)
    assert stats.rows_in_windows + sum(stats.excluded.values()) == len(rows)
    assert stats.rows_in_windows == sum(windows.frame["n_sentences"])
    assert stats.windows == windows.frame.height == sum(stats.windows_by_size.values())

    spans: dict[tuple[str, str, str], list[tuple[int, int]]] = {}
    for window in windows.frame.iter_rows(named=True):
        size = window["n_sentences"]
        # Bounds: only a budget makes a window smaller than the smallest size.
        assert 1 <= size <= high
        # Equal numbers of separators on both sides.
        assert window["src"].count("<sep>") == window["tgt"].count("<sep>") == size - 1
        # The budget.
        assert window["src_pieces"] <= max_pieces
        assert window["tgt_pieces"] <= max_pieces
        key = (window["src_lang"], window["tgt_lang"], window["doc_id"])
        spans.setdefault(key, []).append((window["first_seg"], size))

    # Consecutive segments of one document: the positions a window covers exist in the
    # input as train rows of that document and direction, and no two windows overlap.
    present = {
        (row["src_lang"], row["tgt_lang"], row["doc_id"], row["seg_idx"])
        for row in rows
        if row["split"] == "train" and "doc_id" in row and "seg_idx" in row
    }
    for (src_lang, tgt_lang, doc_id), windows_of_document in spans.items():
        covered: set[int] = set()
        for first, size in windows_of_document:
            positions = set(range(first, first + size))
            assert not positions & covered
            covered |= positions
            assert all((src_lang, tgt_lang, doc_id, p) in present for p in positions)
        assert doc_id.strip()

    # The order of the rows does not matter.
    shuffled = list(range(len(rows)))
    shuffler.shuffle(shuffled)
    again = docify(
        rows_frame(
            [rows[i] for i in shuffled], filter_reasons=[reasons[i] for i in shuffled]
        ),
        spm_model=toy_spm,
        min_size=low,
        max_size=high,
        max_pieces=max_pieces,
        seed=seed,
    )
    assert again.frame.equals(windows.frame)
    assert again.stats == stats


@given(st.integers(0, 40), st.integers(0, 10_000))
def test_a_clean_document_of_n_sentences_is_all_in_windows_but_a_tail(
    toy_spm: Path, sentences: int, seed: int
) -> None:
    rows = [doc_row("d1", position, position) for position in range(sentences)]
    if not rows:
        return

    windows = docify(rows_frame(rows), spm_model=toy_spm, max_pieces=10_000, seed=seed)

    stats = windows.stats
    left_out = stats.excluded["tail"] + stats.excluded["gap_single"]
    assert stats.rows_in_windows + left_out == sentences
    assert left_out <= 1
    assert all(2 <= size <= 4 for size in windows.frame["n_sentences"])
