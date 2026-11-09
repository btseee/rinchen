"""Hand-built document rows for the tests of `rinchen.train.docify` and
`rinchen.train.doc_corpus`.

`SENTENCES` are eight toy sentence pairs (English and Mongolian) whose piece counts with
the toy SentencePiece model (`tiny_parallel.train_toy_spm`) were counted once and are
written beside them, so a test can say how long a window is without calling the code that
cuts it. The pieces of a window of k sentences are the sum of the pieces of the sentences
and two for each `<sep>` between them (the word-boundary piece U+2581 and `<sep>`).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import polars as pl

from rinchen.corpus.schema import corpus_frame

# (English, Mongolian), then the pieces of each alone: (English, Mongolian).
SENTENCES: tuple[tuple[str, str], ...] = (
    ("This is the book.", "Энэ бол ном."),  # 7, 7
    ("The water is here.", "Ус энд байна."),  # 8, 6
    ("Where is the dog?", "Нохой хаана байна вэ?"),  # 8, 12
    ("The tree is there.", "Тэнд мод байна."),  # 9, 6
    ("The book and the water.", "Ном ба ус."),  # 8, 4
    ("This is the home.", "Энэ бол гэр."),  # 7, 7
    ("The sun is here.", "Нар энд байна."),  # 8, 6
    ("Where is the moon?", "Сар хаана байна вэ?"),  # 8, 12
)
SOURCE_PIECES = (7, 8, 8, 9, 8, 7, 8, 8)
TARGET_PIECES = (7, 6, 12, 6, 4, 7, 6, 12)
JOIN = " <sep> "

EN_MN = ("eng_Latn", "khk_Cyrl")
MN_EN = ("khk_Cyrl", "eng_Latn")


def doc_row(
    doc_id: str | None,
    seg_idx: int | None,
    index: int = 0,
    *,
    languages: tuple[str, str] = EN_MN,
    origin: str = "human",
    split: str = "train",
    src: str | None = None,
    tgt: str | None = None,
) -> dict[str, Any]:
    """One corpus row of sentence `index` of `SENTENCES` (in the direction `languages`)
    in the document `doc_id` at `seg_idx`; `src` and `tgt` replace the text. The key
    of a `None` id or index is left out, so the column is null."""
    english, mongolian = SENTENCES[index % len(SENTENCES)]
    forward = languages == EN_MN
    row: dict[str, Any] = {
        "src_lang": languages[0],
        "tgt_lang": languages[1],
        "src": (english if forward else mongolian) if src is None else src,
        "tgt": (mongolian if forward else english) if tgt is None else tgt,
        "origin": origin,
        "corpus": "doc-test",
        "split": split,
    }
    if doc_id is not None:
        row["doc_id"] = doc_id
    if seg_idx is not None:
        row["seg_idx"] = seg_idx
    return row


def document(
    doc_id: str, sentences: Sequence[int] | range, **kwargs: Any
) -> list[dict[str, Any]]:
    """The rows of a document: sentence `sentences[i]` of `SENTENCES` at `seg_idx` i."""
    return [
        doc_row(doc_id, position, index, **kwargs)
        for position, index in enumerate(sentences)
    ]


def rows_frame(
    rows: Sequence[Mapping[str, Any]],
    *,
    filter_reasons: Sequence[str | None] | None = None,
) -> pl.DataFrame:
    """`rows` as the frame `docify` takes (what `read_corpus_rows` returns): the corpus
    columns with `origin` and `split` as text. `filter_reasons` gives each row its
    `filter_reason`."""
    frame = corpus_frame(rows)
    if filter_reasons is not None:
        frame = frame.with_columns(
            pl.Series("filter_reason", list(filter_reasons), dtype=pl.String)
        )
    return frame.with_columns(
        pl.col("origin").cast(pl.String), pl.col("split").cast(pl.String)
    )
