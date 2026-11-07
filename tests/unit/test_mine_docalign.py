"""Document matching: URL normalisation, the URL pass, and the embedding pass
(mutual nearest neighbours with a date window), over the hash embedder, whose
cosine between two texts is about the share of words they have in common."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date

import numpy as np
import pytest
from numpy.typing import NDArray

from rinchen.mine.docalign import (
    DocMatch,
    Document,
    match_documents,
    normalise_url,
)
from tests.support.embedders import HashEmbedder

# Six-word texts: two texts that share five words have cosine about 0.83, three
# words about 0.58, none about 0.
BASE = "kor tam vel nis ruk dap"
FIVE_OF_SIX = "kor tam vel nis ruk gim"
THREE_OF_SIX = "kor tam vel lur gim sol"
OTHER = "zik tul qua bor fen wen"
THIRD = "hat mux jol pav yed cin"


class RecordingEmbedder:
    """The hash embedder, remembering what it was asked to embed."""

    def __init__(self) -> None:
        self.calls: list[list[str]] = []
        self._inner = HashEmbedder()

    def embed(self, texts: Sequence[str]) -> NDArray[np.float32]:
        self.calls.append(list(texts))
        return self._inner.embed(texts)


class WrongShapeEmbedder:
    def embed(self, texts: Sequence[str]) -> NDArray[np.float32]:
        return np.zeros((len(texts) + 1, 4), dtype=np.float32)


def _doc(
    doc_id: str,
    text: str = "",
    *,
    url: str | None = None,
    date: date | None = None,
) -> Document:
    return Document(doc_id, text, url, date)


def _pairs(matches: Sequence[DocMatch]) -> list[tuple[str, str, str]]:
    return [(m.src_id, m.tgt_id, m.method) for m in matches]


# --- normalise_url ----------------------------------------------------------


def test_urls_that_differ_in_scheme_www_language_and_fragment_are_equal() -> None:
    first = normalise_url("https://www.Example.mn/mn/news/42/?lang=mn#top")
    second = normalise_url("http://example.mn/en/news/42")

    assert first == second == "example.mn/news/42"


def test_a_language_label_in_the_host_is_dropped() -> None:
    assert normalise_url("https://mn.example.mn/a") == "example.mn/a"
    assert normalise_url("https://en.example.mn/a") == "example.mn/a"


def test_the_query_is_kept_so_that_different_ids_differ() -> None:
    assert normalise_url("http://example.mn/n?id=3") == "example.mn/n?id=3"
    assert normalise_url("http://example.mn/n?id=3") != normalise_url(
        "http://example.mn/n?id=4"
    )


def test_the_language_parameters_leave_the_query_and_the_rest_is_sorted() -> None:
    url = "http://example.mn/n?lang=en&b=2&hl=mn&a=1&language=eng&locale=en_US"

    assert normalise_url(url) == "example.mn/n?a=1&b=2"


def test_a_query_of_language_parameters_only_leaves_no_question_mark() -> None:
    assert normalise_url("http://example.mn/n?lang=mn") == "example.mn/n"


@pytest.mark.parametrize("index", ["index.html", "index.php"])
def test_a_final_index_file_is_dropped(index: str) -> None:
    assert normalise_url(f"http://example.mn/news/{index}") == "example.mn/news"


def test_an_index_file_before_the_final_segment_stays() -> None:
    assert normalise_url("http://example.mn/index.html/a") == "example.mn/index.html/a"


def test_trailing_and_repeated_slashes_do_not_matter() -> None:
    assert normalise_url("http://example.mn/a//b/") == "example.mn/a/b"
    assert normalise_url("http://example.mn/") == "example.mn"
    assert normalise_url("http://example.mn") == "example.mn"


def test_case_is_ignored_in_the_whole_url() -> None:
    assert normalise_url("HTTP://Example.MN/News/ID?Q=A") == "example.mn/news/id?q=a"


LANGUAGE_MARKERS = (
    "mn mon khk en eng ru rus zh zho cn ja jpn jp ko kor kr de deu ger kk kaz tr tur"
)


@pytest.mark.parametrize("marker", LANGUAGE_MARKERS.split())
def test_every_language_marker_is_dropped_as_a_path_segment_and_a_host_label(
    marker: str,
) -> None:
    assert normalise_url(f"http://example.mn/{marker}/a") == "example.mn/a"
    assert normalise_url(f"http://{marker}.example.mn/a") == "example.mn/a"
    assert normalise_url(f"http://www.{marker}.example.mn/a") == "example.mn/a"


def test_the_last_two_host_labels_are_never_dropped() -> None:
    assert normalise_url("http://en.mn/a") == "en.mn/a"
    assert normalise_url("http://mn.mn/a") == "mn.mn/a"
    assert normalise_url("http://mn.news.com.mn/a") == "news.com.mn/a"


def test_a_segment_must_equal_a_marker_to_be_dropped() -> None:
    url = "http://example.mn/english/mn-news/entr/en.html"

    assert normalise_url(url) == "example.mn/english/mn-news/entr/en.html"


def test_a_url_without_a_scheme_is_normalised_like_one_with() -> None:
    assert normalise_url("www.example.mn/mn/a") == "example.mn/a"
    assert normalise_url("//example.mn/mn/a") == "example.mn/a"


@pytest.mark.parametrize(
    "url",
    [
        "https://www.Example.mn/mn/news/42/?lang=mn#top",
        "http://mn.example.mn/a/index.php?b=2&a=1",
        "example.mn",
        "",
    ],
)
def test_normalising_twice_changes_nothing(url: str) -> None:
    once = normalise_url(url)

    assert normalise_url(once) == once


# --- the URL pass -----------------------------------------------------------


def test_documents_with_the_same_normalised_url_are_paired_with_score_one() -> None:
    src = [_doc("s1", url="https://example.mn/mn/news/42")]
    tgt = [_doc("t1", url="http://www.example.mn/en/news/42/")]
    embedder = RecordingEmbedder()

    matches = match_documents(src, tgt, embedder)

    assert matches == [DocMatch("s1", "t1", 1.0, "url")]
    assert embedder.calls == []


def test_a_key_that_two_sources_share_is_ambiguous_and_matches_nothing_by_url() -> None:
    src = [
        _doc("s1", OTHER, url="http://example.mn/mn/a"),
        _doc("s2", THIRD, url="http://example.mn/a"),
    ]
    tgt = [_doc("t1", BASE, url="http://example.mn/en/a")]

    assert match_documents(src, tgt, HashEmbedder()) == []


def test_a_key_that_two_targets_share_is_ambiguous_too() -> None:
    src = [_doc("s1", OTHER, url="http://example.mn/a")]
    tgt = [
        _doc("t1", BASE, url="http://example.mn/en/a"),
        _doc("t2", THIRD, url="http://example.mn/ru/a"),
    ]

    assert match_documents(src, tgt, HashEmbedder()) == []


def test_documents_of_an_ambiguous_key_are_left_to_the_embedding_pass() -> None:
    src = [
        _doc("s1", BASE, url="http://example.mn/mn/a"),
        _doc("s2", OTHER, url="http://example.mn/a"),
    ]
    tgt = [_doc("t1", BASE, url="http://example.mn/en/a")]

    matches = match_documents(src, tgt, HashEmbedder())

    assert _pairs(matches) == [("s1", "t1", "embedding")]


def test_a_document_without_a_url_is_never_matched_by_url() -> None:
    src = [_doc("s1", BASE), _doc("s2", OTHER, url="")]
    tgt = [_doc("t1", THIRD), _doc("t2", THIRD, url="")]

    assert match_documents(src, tgt, HashEmbedder()) == []


def test_documents_matched_by_url_are_not_matched_again() -> None:
    src = [_doc("s1", BASE, url="http://example.mn/mn/a")]
    tgt = [
        _doc("t1", OTHER, url="http://example.mn/en/a"),
        _doc("t2", BASE),
    ]
    embedder = RecordingEmbedder()

    matches = match_documents(src, tgt, embedder)

    assert _pairs(matches) == [("s1", "t1", "url")]
    assert embedder.calls == []


def test_the_embedding_pass_sees_only_the_documents_left_over() -> None:
    src = [
        _doc("s1", BASE, url="http://example.mn/mn/a"),
        _doc("s2", THIRD),
    ]
    tgt = [
        _doc("t1", OTHER, url="http://example.mn/en/a"),
        _doc("t2", THIRD),
    ]
    embedder = RecordingEmbedder()

    matches = match_documents(src, tgt, embedder)

    assert _pairs(matches) == [("s1", "t1", "url"), ("s2", "t2", "embedding")]
    assert embedder.calls == [[THIRD], [THIRD]]


# --- the embedding pass -----------------------------------------------------


def test_mutual_nearest_neighbours_above_the_threshold_are_paired() -> None:
    src = [_doc("s1", BASE), _doc("s2", OTHER), _doc("s3", THIRD)]
    tgt = [_doc("t3", THIRD), _doc("t1", FIVE_OF_SIX), _doc("t2", OTHER)]
    embedder = HashEmbedder()

    matches = match_documents(src, tgt, embedder)

    assert _pairs(matches) == [
        ("s1", "t1", "embedding"),
        ("s2", "t2", "embedding"),
        ("s3", "t3", "embedding"),
    ]
    vectors = embedder.embed([BASE, FIVE_OF_SIX])
    assert matches[0].score == pytest.approx(float(vectors[0] @ vectors[1]), abs=1e-6)
    assert matches[1].score == pytest.approx(1.0, abs=1e-6)


def test_a_pair_below_min_cosine_is_rejected_and_above_it_accepted() -> None:
    src = [_doc("s1", BASE)]
    tgt = [_doc("t1", THREE_OF_SIX)]

    assert match_documents(src, tgt, HashEmbedder()) == []
    assert _pairs(match_documents(src, tgt, HashEmbedder(), min_cosine=0.5)) == [
        ("s1", "t1", "embedding")
    ]


def test_min_cosine_itself_is_accepted() -> None:
    src = [_doc("s1", BASE)]
    tgt = [_doc("t1", FIVE_OF_SIX)]
    [found] = match_documents(src, tgt, HashEmbedder(), min_cosine=0.5)

    at_the_score = match_documents(src, tgt, HashEmbedder(), min_cosine=found.score)

    assert at_the_score == [found]


def test_a_candidate_that_prefers_another_document_is_not_paired() -> None:
    # t1 is closest to s1 (identical); s2 is above the threshold for t1 but is
    # not t1's nearest, so it stays unmatched.
    src = [_doc("s1", BASE), _doc("s2", FIVE_OF_SIX)]
    tgt = [_doc("t1", BASE)]

    matches = match_documents(src, tgt, HashEmbedder())

    assert _pairs(matches) == [("s1", "t1", "embedding")]


def test_each_side_picks_its_own_nearest_not_the_first_above_the_threshold() -> None:
    # s1 likes t2 best, t2 likes s2 best: no mutual pair for s1 or t2, and
    # s2 pairs with t2 only because each is the other's best.
    src = [_doc("s1", FIVE_OF_SIX), _doc("s2", BASE)]
    tgt = [_doc("t1", OTHER), _doc("t2", BASE)]

    matches = match_documents(src, tgt, HashEmbedder())

    assert _pairs(matches) == [("s2", "t2", "embedding")]


def test_ties_go_to_the_lowest_index() -> None:
    src = [_doc("s1", BASE)]
    tgt = [_doc("t1", BASE), _doc("t2", BASE)]

    assert _pairs(match_documents(src, tgt, HashEmbedder())) == [
        ("s1", "t1", "embedding")
    ]


def test_a_date_gap_beyond_the_window_blocks_a_match() -> None:
    src = [_doc("s1", BASE, date=date(2025, 3, 1))]
    tgt = [_doc("t1", BASE, date=date(2025, 3, 10))]

    assert match_documents(src, tgt, HashEmbedder()) == []
    assert match_documents(src, tgt, HashEmbedder(), max_date_gap_days=9)


def test_a_gap_equal_to_the_window_is_allowed_in_either_direction() -> None:
    early, late = date(2025, 3, 1), date(2025, 3, 3)

    forward = match_documents(
        [_doc("s1", BASE, date=early)], [_doc("t1", BASE, date=late)], HashEmbedder()
    )
    backward = match_documents(
        [_doc("s1", BASE, date=late)], [_doc("t1", BASE, date=early)], HashEmbedder()
    )

    assert len(forward) == len(backward) == 1


def test_a_missing_date_on_either_side_does_not_block() -> None:
    dated = date(2025, 3, 1)
    src = [_doc("s1", BASE, date=dated), _doc("s2", OTHER)]
    tgt = [_doc("t1", BASE), _doc("t2", OTHER, date=dated)]

    assert _pairs(match_documents(src, tgt, HashEmbedder())) == [
        ("s1", "t1", "embedding"),
        ("s2", "t2", "embedding"),
    ]


def test_a_blocked_candidate_is_skipped_for_the_next_best_one() -> None:
    src = [_doc("s1", BASE, date=date(2025, 3, 1))]
    tgt = [
        _doc("t1", BASE, date=date(2025, 4, 1)),
        _doc("t2", FIVE_OF_SIX, date=date(2025, 3, 2)),
    ]

    assert _pairs(match_documents(src, tgt, HashEmbedder())) == [
        ("s1", "t2", "embedding")
    ]


def test_texts_are_embedded_by_their_first_two_thousand_characters() -> None:
    long_text = "a" * 1999 + "bc" + "d" * 1000
    src = [_doc("s1", long_text)]
    tgt = [_doc("t1", "x")]
    embedder = RecordingEmbedder()

    match_documents(src, tgt, embedder)

    assert embedder.calls == [[long_text[:2000]], ["x"]]
    assert len(embedder.calls[0][0]) == 2000


def test_each_side_is_embedded_in_one_call() -> None:
    src = [_doc("s1", BASE), _doc("s2", OTHER), _doc("s3", THIRD)]
    tgt = [_doc("t1", BASE), _doc("t2", OTHER)]
    embedder = RecordingEmbedder()

    match_documents(src, tgt, embedder)

    assert embedder.calls == [[BASE, OTHER, THIRD], [BASE, OTHER]]


def test_documents_with_blank_text_never_match_by_embedding() -> None:
    # The hash embedder gives every blank text the same vector (cosine 1.0).
    src = [_doc("s1", ""), _doc("s2", "   \n")]
    tgt = [_doc("t1", ""), _doc("t2", "\t")]
    embedder = RecordingEmbedder()

    assert match_documents(src, tgt, embedder) == []
    assert embedder.calls == []


def test_a_blank_document_still_matches_by_url_and_leaves_the_others_alone() -> None:
    src = [_doc("s1", "", url="http://example.mn/mn/a"), _doc("s2", BASE)]
    tgt = [
        _doc("t1", "", url="http://example.mn/en/a"),
        _doc("t2", BASE),
        _doc("t3", ""),
    ]

    matches = match_documents(src, tgt, HashEmbedder())

    assert _pairs(matches) == [("s1", "t1", "url"), ("s2", "t2", "embedding")]


def test_the_result_is_sorted_by_source_id_whatever_the_input_order() -> None:
    src = [_doc("s3", THIRD), _doc("s1", BASE), _doc("s2", OTHER)]
    tgt = [_doc("t2", OTHER), _doc("t3", THIRD), _doc("t1", BASE)]

    matches = match_documents(src, tgt, HashEmbedder())

    assert [m.src_id for m in matches] == ["s1", "s2", "s3"]
    assert [m.tgt_id for m in matches] == ["t1", "t2", "t3"]


def test_url_and_embedding_matches_are_merged_in_source_id_order() -> None:
    src = [_doc("s2", THIRD), _doc("s1", BASE, url="http://example.mn/mn/a")]
    tgt = [_doc("t1", OTHER, url="http://example.mn/en/a"), _doc("t2", THIRD)]

    assert _pairs(match_documents(src, tgt, HashEmbedder())) == [
        ("s1", "t1", "url"),
        ("s2", "t2", "embedding"),
    ]


# --- edges ------------------------------------------------------------------


@pytest.mark.parametrize(
    ("src", "tgt"),
    [
        ([], []),
        ([_doc("s1", BASE)], []),
        ([], [_doc("t1", BASE)]),
    ],
)
def test_an_empty_side_gives_no_matches_and_no_embedding_work(
    src: list[Document], tgt: list[Document]
) -> None:
    embedder = RecordingEmbedder()

    assert match_documents(src, tgt, embedder) == []
    assert embedder.calls == []


def test_an_embedder_that_returns_the_wrong_number_of_rows_is_an_error() -> None:
    with pytest.raises(ValueError, match="one row per"):
        match_documents([_doc("s1", BASE)], [_doc("t1", BASE)], WrongShapeEmbedder())


def test_a_negative_date_window_is_an_error() -> None:
    with pytest.raises(ValueError, match="max_date_gap_days"):
        match_documents([], [], HashEmbedder(), max_date_gap_days=-1)


def test_a_nan_threshold_is_an_error() -> None:
    with pytest.raises(ValueError, match="min_cosine"):
        match_documents([], [], HashEmbedder(), min_cosine=float("nan"))


def test_documents_and_matches_are_frozen_values() -> None:
    document = Document("d", "text")
    match = DocMatch("s", "t", 1.0, "url")

    assert (document.url, document.date) == (None, None)
    with pytest.raises(AttributeError):
        document.text = "other"  # type: ignore[misc]
    with pytest.raises(AttributeError):
        match.score = 0.0  # type: ignore[misc]
