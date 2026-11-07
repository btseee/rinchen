"""Document matching for mining: pair the documents of two languages that are
translations of each other, before their sentences are aligned.

Two passes. First documents whose URLs are the same once the language has been
stripped (`normalise_url`) are paired, when the key is unique on both sides.
Then the documents left over are paired by mutual nearest neighbours of their
embeddings: a source and a target document match when each is the other's most
similar candidate and their cosine is at least `min_cosine`. When both dates
are known, a candidate more than `max_date_gap_days` away does not count.

The embedding pass holds the cosine matrix of the leftovers in memory, as
float32, so it suits a few thousand to about 10 000 documents per side (a
10 000 x 10 000 matrix is 400 MB). Measured with tracemalloc at 4000 documents
per side, the peak is 74 MB without dates and 83 MB with dates, 1.15 and 1.3
times the 64 MB matrix; before the matrix was kept as float32 and the date mask
built in blocks it was 385 MB, three times a float64 matrix. Scores are
cosines only if the embedder returns unit-length rows, as the `Embedder`
contract says; otherwise they are plain dot products.

A document is left out of the embedding pass when the part of its text that is
embedded (the first 2000 characters) is blank: all blank texts embed alike, so
two of them would match each other.
"""

from __future__ import annotations

import datetime
import math
import re
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from rinchen.mine.embed import Embedder

# Path segments and host labels that only name the language of a page.
_LANGUAGE_MARKERS = frozenset(
    {
        "mn",
        "mon",
        "khk",
        "en",
        "eng",
        "ru",
        "rus",
        "zh",
        "zho",
        "cn",
        "ja",
        "jpn",
        "jp",
        "ko",
        "kor",
        "kr",
        "de",
        "deu",
        "ger",
        "kk",
        "kaz",
        "tr",
        "tur",
    }
)
_LANGUAGE_QUERY_PARAMETERS = frozenset({"lang", "language", "hl", "locale"})
_INDEX_FILES = frozenset({"index.html", "index.php"})
# `https://`, or a bare `//` of a scheme-relative URL.
_SCHEME = re.compile(r"^(?:[a-z][a-z0-9+.\-]*:)?//")
_WWW_PREFIX = "www."
# The registered domain, e.g. `example.mn`; labels in front of it can be a
# language, and these never are.
_REGISTERED_DOMAIN_LABELS = 2
_EMBEDDED_CHARACTERS = 2000
# Rows (or columns) of the similarity matrix handled at a time where numpy
# would otherwise copy the whole matrix: the date mask and the best source per
# target.
_MATRIX_BLOCK_SIZE = 512

MatchMethod = Literal["url", "embedding"]


@dataclass(frozen=True, slots=True)
class Document:
    doc_id: str
    text: str
    url: str | None = None
    date: datetime.date | None = None


@dataclass(frozen=True, slots=True)
class DocMatch:
    src_id: str
    tgt_id: str
    score: float
    method: MatchMethod


@dataclass(frozen=True, slots=True)
class _Pair:
    """A match by position in the two input sequences."""

    src: int
    tgt: int
    score: float
    method: MatchMethod


def normalise_url(url: str) -> str:
    """The URL without what only names the language or the presentation, so
    that the pages of one article in two languages share a key.

    Lower-cased; without the scheme, a leading `www.`, the fragment, the query
    parameters `lang`, `language`, `hl` and `locale`, trailing and repeated
    slashes, a final `index.html` / `index.php`, and every path segment or
    host label that is a language marker (`mn`, `en`, `ru`, ...), except the
    last two host labels. The rest of the query is kept, sorted."""
    without_scheme = _SCHEME.sub("", url.strip().lower())
    without_fragment = without_scheme.split("#", 1)[0]
    location, _, query = without_fragment.partition("?")
    host, _, path = location.partition("/")
    parts = [_normalised_host(host), *_normalised_path(path)]
    key = "/".join(part for part in parts if part)
    remaining_query = _remaining_query(query)
    return f"{key}?{remaining_query}" if remaining_query else key


def _normalised_host(host: str) -> str:
    labels = host.removeprefix(_WWW_PREFIX).split(".")
    registered = labels[-_REGISTERED_DOMAIN_LABELS:]
    leading = labels[: len(labels) - len(registered)]
    return ".".join([*(l for l in leading if l not in _LANGUAGE_MARKERS), *registered])


def _normalised_path(path: str) -> list[str]:
    segments = [segment for segment in path.split("/") if segment]
    if segments and segments[-1] in _INDEX_FILES:
        segments.pop()
    return [segment for segment in segments if segment not in _LANGUAGE_MARKERS]


def _remaining_query(query: str) -> str:
    parameters = [
        parameter
        for parameter in query.split("&")
        if parameter and parameter.partition("=")[0] not in _LANGUAGE_QUERY_PARAMETERS
    ]
    return "&".join(sorted(parameters))


def match_documents(
    src_docs: Sequence[Document],
    tgt_docs: Sequence[Document],
    embed: Embedder,
    *,
    max_date_gap_days: int = 2,
    min_cosine: float = 0.7,
) -> list[DocMatch]:
    """Pair source with target documents, each document at most once; the
    result is sorted by source id. Raises `ValueError` for a negative
    `max_date_gap_days`, a non-finite `min_cosine`, or an embedder that returns
    the wrong number of rows."""
    if max_date_gap_days < 0:
        raise ValueError(
            f"max_date_gap_days must not be negative, got {max_date_gap_days}"
        )
    if not math.isfinite(min_cosine):
        raise ValueError(f"min_cosine must be finite, got {min_cosine}")

    by_url = _match_by_url(src_docs, tgt_docs)
    by_embedding = _match_by_embedding(
        src_docs,
        tgt_docs,
        _positions_left(len(src_docs), {pair.src for pair in by_url}),
        _positions_left(len(tgt_docs), {pair.tgt for pair in by_url}),
        embed,
        max_date_gap_days=max_date_gap_days,
        min_cosine=min_cosine,
    )
    return sorted(
        (
            DocMatch(src_docs[p.src].doc_id, tgt_docs[p.tgt].doc_id, p.score, p.method)
            for p in [*by_url, *by_embedding]
        ),
        key=lambda match: match.src_id,
    )


def _positions_left(count: int, matched: set[int]) -> list[int]:
    return [position for position in range(count) if position not in matched]


def _match_by_url(
    src_docs: Sequence[Document], tgt_docs: Sequence[Document]
) -> list[_Pair]:
    src_by_key = _unique_positions_by_key(src_docs)
    tgt_by_key = _unique_positions_by_key(tgt_docs)
    return [
        _Pair(src, tgt_by_key[key], 1.0, "url")
        for key, src in src_by_key.items()
        if key in tgt_by_key
    ]


def _unique_positions_by_key(docs: Sequence[Document]) -> dict[str, int]:
    """The position of the one document that holds each URL key; a key held by
    several documents is ambiguous and left out, as are documents without a
    URL."""
    keys = [normalise_url(doc.url) if doc.url else "" for doc in docs]
    counts = Counter(keys)
    return {
        key: position for position, key in enumerate(keys) if key and counts[key] == 1
    }


def _match_by_embedding(
    src_docs: Sequence[Document],
    tgt_docs: Sequence[Document],
    src_left: list[int],
    tgt_left: list[int],
    embed: Embedder,
    *,
    max_date_gap_days: int,
    min_cosine: float,
) -> list[_Pair]:
    src_candidates = [i for i in src_left if _has_embeddable_text(src_docs[i])]
    tgt_candidates = [j for j in tgt_left if _has_embeddable_text(tgt_docs[j])]
    if not src_candidates or not tgt_candidates:
        return []
    src_side = [src_docs[i] for i in src_candidates]
    tgt_side = [tgt_docs[j] for j in tgt_candidates]
    similarity = _cosine_matrix(src_side, tgt_side, embed)
    _exclude_by_date(similarity, src_side, tgt_side, max_date_gap_days)
    # argmax takes the lowest index among equals, which settles ties.
    best_tgt = similarity.argmax(axis=1)
    best_src = _best_row_per_column(similarity)
    pairs = []
    for row, column in enumerate(best_tgt.tolist()):
        raw = float(similarity[row, column])
        # Rounding can leave a float32 dot product a hair outside [-1, 1]. A
        # non-finite value is a pair the date window blocked (-inf): clipping
        # must not turn it into a match for a low `min_cosine`.
        score = min(1.0, max(-1.0, raw))
        if best_src[column] == row and np.isfinite(raw) and score >= min_cosine:
            pairs.append(
                _Pair(src_candidates[row], tgt_candidates[column], score, "embedding")
            )
    return pairs


def _best_row_per_column(similarity: NDArray[np.float32]) -> NDArray[np.intp]:
    """`similarity.argmax(axis=0)`, a block of columns at a time: numpy copies
    the matrix to find the maximum of a column, which would double the memory
    of the pass."""
    return np.concatenate(
        [
            similarity[:, start : start + _MATRIX_BLOCK_SIZE].argmax(axis=0)
            for start in range(0, similarity.shape[1], _MATRIX_BLOCK_SIZE)
        ]
    )


def _embedded_text(doc: Document) -> str:
    return doc.text[:_EMBEDDED_CHARACTERS]


def _has_embeddable_text(doc: Document) -> bool:
    """False when the part of the text that is embedded is blank: the
    embedding then says nothing about the document."""
    return bool(_embedded_text(doc).strip())


def _cosine_matrix(
    src_docs: Sequence[Document], tgt_docs: Sequence[Document], embed: Embedder
) -> NDArray[np.float32]:
    """Rows are source documents, columns target documents. The dot product is
    the cosine only if the embedder's rows are unit length, as its contract
    says."""
    src_vectors = _embed_documents(src_docs, embed)
    tgt_vectors = _embed_documents(tgt_docs, embed)
    return src_vectors @ tgt_vectors.T


def _embed_documents(docs: Sequence[Document], embed: Embedder) -> NDArray[np.float32]:
    texts = [_embedded_text(doc) for doc in docs]
    vectors = np.asarray(embed.embed(texts), dtype=np.float32)
    if vectors.ndim != 2 or vectors.shape[0] != len(texts):
        raise ValueError(
            f"embedder must return one row per text: expected {len(texts)} rows "
            f"in a 2-D array, got shape {vectors.shape}"
        )
    return vectors


def _exclude_by_date(
    similarity: NDArray[np.float32],
    src_docs: Sequence[Document],
    tgt_docs: Sequence[Document],
    max_gap_days: int,
) -> None:
    """Set to -inf, in place, every pair whose dates are both known and further
    apart than `max_gap_days`. Nothing is computed when one side has no date,
    and otherwise the mask is built `_MATRIX_BLOCK_SIZE` rows at a time."""
    src_days, src_known = _day_numbers(src_docs)
    tgt_days, tgt_known = _day_numbers(tgt_docs)
    if not (src_known.any() and tgt_known.any()):
        return
    for start in range(0, len(src_docs), _MATRIX_BLOCK_SIZE):
        rows = slice(start, start + _MATRIX_BLOCK_SIZE)
        gap = src_days[rows, np.newaxis] - tgt_days[np.newaxis, :]
        np.abs(gap, out=gap)
        blocked = src_known[rows, np.newaxis] & tgt_known[np.newaxis, :]
        blocked &= gap > max_gap_days
        similarity[rows][blocked] = -np.inf


def _day_numbers(
    docs: Sequence[Document],
) -> tuple[NDArray[np.int32], NDArray[np.bool_]]:
    """The proleptic Gregorian ordinal of each date (0 when unknown) and
    whether it is known."""
    days = np.array(
        [doc.date.toordinal() if doc.date else 0 for doc in docs], dtype=np.int32
    )
    return days, np.array([doc.date is not None for doc in docs], dtype=np.bool_)
