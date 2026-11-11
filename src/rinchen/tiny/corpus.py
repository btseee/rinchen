"""The corpus of the tiny pipeline and its expansion to both directions (D15, spec P6).

`pair_rows` makes ONE row per pair, in the direction the pair was acquired: the filter runs once per
pair, because a row and its reverse share almost every shingle and every row would be a near-duplicate
of its reverse. Both sides of a pair are normalised and split into sentences, and a pair with a side
that is not exactly one sentence is left out and counted. The corpus is a function of the fixtures
and of a fixed stamp alone: nothing here reads the clock.

`expand_directions` runs AFTER the filter, on the kept rows: every kept row of the splits `train` and
`dev` and of no noise corpus, then the same rows reversed, except the traditional-script rows, whose
one trained direction is into English. The noise rows are held out (split `holdout`), so a planted row
that the filter lets through is still never trained on.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from types import MappingProxyType
from typing import Any, Final

import polars as pl

from rinchen.corpus.schema import corpus_frame, validate_corpus_schema
from rinchen.errors import DataError
from rinchen.split.rules import split as split_sentences
from rinchen.text.normalize import normalize
from rinchen.tiny.fixtures import MONGOLIAN_FILE, NOISE_FILE, Fixtures

_LABEL_PREFIX: Final = "tiny-"
NOISE_CORPUS: Final = _LABEL_PREFIX + NOISE_FILE.removesuffix(".tsv")
MONG_CORPUS: Final = _LABEL_PREFIX + MONGOLIAN_FILE.removesuffix(".tsv")
EXPANDED_SPLITS: Final = ("train", "dev")
HELD_OUT_SPLIT: Final = "holdout"
_MONGOLIAN_LANGUAGE: Final = "mn_Mong"
_ENGLISH: Final = "eng_Latn"
_CREATED_FORMAT: Final = "%Y-%m-%dT%H:%M:%SZ"
_ROW_COLUMNS: Final = (
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


@dataclass(frozen=True, slots=True, eq=False)
class PairCorpus:
    """The pair corpus and, for each corpus label, how many pairs were left out because a side was
    not exactly one sentence."""

    frame: pl.DataFrame
    dropped: Mapping[str, int]


def created_stamp(text: str) -> datetime:
    """The UTC time of a profile's `created` (`YYYY-MM-DDTHH:MM:SSZ`, which its loader checked)."""
    return datetime.strptime(text, _CREATED_FORMAT).replace(tzinfo=UTC)


def _one_sentence(text: str, language: str) -> str | None:
    sentences = split_sentences(normalize(text, language), language)
    return sentences[0] if len(sentences) == 1 else None


def clean_pair(
    src_lang: str, src: str, tgt_lang: str, tgt: str
) -> tuple[str, str] | None:
    """The two sides of a pair, normalised and each exactly one sentence, or None when a side is
    not."""
    source = _one_sentence(src, src_lang)
    target = _one_sentence(tgt, tgt_lang)
    if source is None or target is None:
        return None
    return source, target


def build_pair_corpus(fixtures: Fixtures, created: datetime) -> PairCorpus:
    """One row per pair of the four files, in the order of the files (the two pair files by name,
    the traditional-script rows, the noise), and the count of pairs left out per corpus label."""
    rows: list[dict[str, Any]] = []
    dropped: dict[str, int] = {}

    def take(
        label: str,
        origin: str,
        languages: tuple[str, str],
        texts: tuple[str, str],
        split: str,
    ) -> None:
        pair = clean_pair(languages[0], texts[0], languages[1], texts[1])
        if pair is None:
            dropped[label] += 1
            return
        rows.append(
            {
                "src_lang": languages[0],
                "tgt_lang": languages[1],
                "src": pair[0],
                "tgt": pair[1],
                "origin": origin,
                "corpus": label,
                "split": split,
            }
        )

    for key in sorted(fixtures.pairs):
        label = _LABEL_PREFIX + key
        dropped[label] = 0
        src_lang, tgt_lang = key.split("-")
        for row in fixtures.pairs[key]:
            take(label, "human", (src_lang, tgt_lang), (row.src, row.tgt), row.split)
    dropped[MONG_CORPUS] = 0
    for mong in fixtures.mong:
        languages = (_MONGOLIAN_LANGUAGE, _ENGLISH)
        take(MONG_CORPUS, "converted", languages, (mong.mong, mong.eng), mong.split)
    dropped[NOISE_CORPUS] = 0
    for bad in fixtures.noise:
        languages = (bad.src_lang, bad.tgt_lang)
        take(NOISE_CORPUS, "web", languages, (bad.src, bad.tgt), HELD_OUT_SPLIT)
    return PairCorpus(
        frame=corpus_frame(rows, created=created),
        dropped=MappingProxyType(dropped),
    )


def pair_rows(fixtures: Fixtures, created: datetime) -> pl.DataFrame:
    """The pair corpus: one row per pair, stamped `created` (a timezone-aware time)."""
    return build_pair_corpus(fixtures, created).frame


def _reversed(row: dict[str, Any]) -> dict[str, Any]:
    return {
        **row,
        "src_lang": row["tgt_lang"],
        "tgt_lang": row["src_lang"],
        "src": row["tgt"],
        "tgt": row["src"],
    }


def expand_directions(kept: pl.DataFrame, created: datetime) -> pl.DataFrame:
    """The training corpus of the kept rows: the rows of the splits `train` and `dev` of no noise
    corpus, then the reverse of each of them except the traditional-script ones; scores and reasons
    are not carried, the ids are new (the rows are new pairs) and the stamp is `created`.

    Raises:
        DataError: `kept` is not a corpus table, or holds rows that a stage rejected (the filter's
            scored table, not its kept rows, was handed in); the message counts them.
    """
    validate_corpus_schema(kept.schema, source="the rows to expand")
    rejected = kept.height - kept["filter_reason"].null_count()
    if rejected:
        raise DataError(
            f"the rows to expand hold {rejected} rows that a filter stage rejected"
        )
    wanted = kept.filter(
        pl.col("split").is_in(EXPANDED_SPLITS) & (pl.col("corpus") != NOISE_CORPUS)
    )
    forward = wanted.select(_ROW_COLUMNS).to_dicts()
    reverse = [_reversed(row) for row in forward if row["corpus"] != MONG_CORPUS]
    return corpus_frame(forward + reverse, created=created)
