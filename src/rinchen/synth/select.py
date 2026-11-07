"""Selection of synthetic candidates and the corpus rows built from the winners.

A generator returns several candidate translations of each real monolingual
text. Per candidate, in this order and counting the first rule that fails:
empty, script purity (the share of its letters in the scripts of the synthetic
language), a reference-free QE floor (only when a scorer is given) and copy (the
candidate is the real text). Of the survivors of a source, minimum Bayes risk
over chrF picks the one the others agree with most; a lone survivor wins
unscored. Every dropped candidate and every source without a winner is counted
under a reason code, so nothing disappears silently.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from types import MappingProxyType
from typing import Literal, get_args

import polars as pl

from rinchen.config.filters import QeConfig
from rinchen.config.languages import load_languages
from rinchen.config.synth import SelectionConfig
from rinchen.corpus.schema import SCORES_DTYPE, corpus_frame
from rinchen.errors import ConfigError, UpstreamError
from rinchen.filter.qe import QEScorer
from rinchen.lang.scripts import Script, script_of
from rinchen.mbr import mbr
from rinchen.mbr.utility import ChrfUtility
from rinchen.synth.tags import origin_for

Kind = Literal["bt", "ft"]

SOURCES = "sources"
CANDIDATES = "candidates"
DROPPED_EMPTY = "dropped_empty"
DROPPED_PURITY = "dropped_purity"
DROPPED_QE = "dropped_qe"
DROPPED_COPY = "dropped_copy"
NO_CANDIDATES = "no_candidates"
NO_SURVIVOR = "no_survivor"
SELECTED = "selected"
COUNT_KEYS = (
    SOURCES,
    CANDIDATES,
    DROPPED_EMPTY,
    DROPPED_PURITY,
    DROPPED_QE,
    DROPPED_COPY,
    NO_CANDIDATES,
    NO_SURVIVOR,
    SELECTED,
)

QE_OK = "ok"
QE_SKIPPED_NO_SCORER = "skipped: no scorer"

_QE_COLUMNS = ("metricx_qe", "cometkiwi")
_SCORE_FIELDS = tuple(SCORES_DTYPE.to_schema())


@dataclass(frozen=True, slots=True)
class SourceRows:
    """The real monolingual side of a job: one id and one text per source."""

    ids: tuple[str, ...]
    texts: tuple[str, ...]

    def __post_init__(self) -> None:
        if len(self.ids) != len(self.texts):
            raise ValueError(f"{len(self.ids)} ids but {len(self.texts)} texts")


@dataclass(frozen=True, slots=True)
class Selection:
    """The outcome of `select_candidates`, one entry per source in source
    order. `chosen` is the index of the winning candidate in that source's
    original candidate list, or None when no candidate survived; the QE score
    and the purity are the winner's (None without a winner, and the QE score is
    None too when no scorer ran). `counts` has every key of `COUNT_KEYS`."""

    chosen: tuple[int | None, ...]
    qe_scores: tuple[float | None, ...]
    purity: tuple[float | None, ...]
    counts: Mapping[str, int]
    qe_status: str


@dataclass(frozen=True, slots=True)
class _Candidate:
    source: int
    index: int
    text: str
    purity: float | None = None
    qe: float | None = None


def select_candidates(
    sources: SourceRows,
    candidates: Sequence[Sequence[str]],
    *,
    kind: Kind,
    synthetic_lang: str,
    cfg: SelectionConfig,
    scorer: QEScorer | None,
) -> Selection:
    """Pick the best surviving candidate of every source.

    `candidates[i]` are the generator's candidates for `sources.texts[i]`, all
    written in `synthetic_lang`; an empty list is a refusal and counts as
    `no_candidates`. With a scorer, the candidates that passed the empty and
    purity rules are scored `cfg.qe.batch_size` pairs per call, as (source,
    translation) in the orientation of the final corpus pair: for `bt` the
    candidate then the real text, for `ft` the real text then the candidate.
    Without a scorer the QE rule is skipped and `qe_status` says so.

    Raises `ValueError` for a `kind` other than bt or ft and for a candidate
    list count that differs from the source count, `TypeError` for a candidate
    list that is a string, `ConfigError` for a `synthetic_lang` outside the
    language registry and `UpstreamError` when the scorer returns the wrong
    number of scores. An exception raised by the scorer propagates unchanged.
    """
    _require_kind(kind)
    _require_one_list_per_source(sources, candidates)
    scripts = _registry_scripts(synthetic_lang)

    counts = dict.fromkeys(COUNT_KEYS, 0)
    counts[SOURCES] = len(sources.ids)
    counts[CANDIDATES] = sum(len(texts) for texts in candidates)

    remaining = [
        _Candidate(source, index, text)
        for source, texts in enumerate(candidates)
        for index, text in enumerate(texts)
    ]
    remaining = _without_empty(remaining, counts)
    remaining = _without_impure(remaining, scripts, cfg.min_script_share, counts)
    if scorer is None:
        qe_status = QE_SKIPPED_NO_SCORER
    else:
        qe_status = QE_OK
        remaining = _above_qe_floor(remaining, sources, kind, cfg.qe, scorer, counts)
    remaining = _without_copies(remaining, sources, counts)

    winners = _pick_winners(remaining)
    counts[NO_CANDIDATES] = sum(1 for texts in candidates if not texts)
    counts[NO_SURVIVOR] = sum(
        1 for source, texts in enumerate(candidates) if texts and source not in winners
    )
    counts[SELECTED] = len(winners)
    return _selection(winners, len(sources.ids), counts, qe_status)


def build_rows(
    sources: SourceRows,
    candidates: Sequence[Sequence[str]],
    selection: Selection,
    *,
    kind: Kind,
    generator_kind: str,
    generator_name: str,
    mono_lang: str,
    other_lang: str,
    qe_scorer: str | None,
    created: datetime | None = None,
) -> pl.DataFrame:
    """One corpus row per source that has a chosen candidate, in source order.

    The generator always writes into `other_lang` and the real text is in
    `mono_lang`. A `bt` row has the candidate as `src` and the real text as
    `tgt`; an `ft` row the other way round. The purity of the winner goes to
    `scores.script_src` (`bt`) or `scores.script_tgt` (`ft`); its QE score goes
    to the `scores` field named by `qe_scorer` (`metricx_qe` or `cometkiwi`,
    None when no scorer ran). Every other score is null. `created` defaults to
    now, in UTC.

    Raises `ValueError` for a `kind` other than bt or ft, a `qe_scorer` that is
    not a QE score field, QE scores in `selection` without a `qe_scorer`, and a
    `selection` or `candidates` that does not belong to `sources`.
    """
    _require_kind(kind)
    _require_one_list_per_source(sources, candidates)
    _require_selection_of(sources, candidates, selection)
    _require_qe_column(qe_scorer, selection)

    origin = origin_for(kind, generator_kind)
    corpus = f"{kind}:{generator_name}"
    rows = []
    scores = []
    for source, (pick, qe, purity) in enumerate(
        zip(selection.chosen, selection.qe_scores, selection.purity, strict=True)
    ):
        if pick is None:
            continue
        synthetic, real = candidates[source][pick], sources.texts[source]
        pair = _pair(kind, synthetic, real, mono_lang, other_lang)
        rows.append({**pair, "origin": origin, "corpus": corpus, "split": "train"})
        scores.append(_row_scores(kind, purity, qe, qe_scorer))
    frame = corpus_frame(rows, created=created)
    return frame.with_columns(pl.Series("scores", scores, dtype=SCORES_DTYPE))


def _is_empty(text: str) -> bool:
    return not any(char.isalnum() for char in text)


def _purity(text: str, scripts: Sequence[Script]) -> float:
    """The share of the letters of `text` that are in `scripts`; 0.0 for a text
    without letters."""
    shares = script_of(text)
    return sum(shares.get(script, 0.0) for script in scripts)


def _passes_qe(score: float, cfg: QeConfig) -> bool:
    """A score exactly at the threshold passes. NaN compares false either way,
    so it fails."""
    if cfg.scorer == "metricx_qe":
        return score <= cfg.max_metricx_qe  # lower is better
    return score >= cfg.min_cometkiwi


def _without_empty(
    remaining: list[_Candidate], counts: dict[str, int]
) -> list[_Candidate]:
    kept = [item for item in remaining if not _is_empty(item.text)]
    counts[DROPPED_EMPTY] += len(remaining) - len(kept)
    return kept


def _without_impure(
    remaining: list[_Candidate],
    scripts: Sequence[Script],
    min_script_share: float,
    counts: dict[str, int],
) -> list[_Candidate]:
    kept = []
    for item in remaining:
        purity = _purity(item.text, scripts)
        if purity >= min_script_share:
            kept.append(dataclasses.replace(item, purity=purity))
    counts[DROPPED_PURITY] += len(remaining) - len(kept)
    return kept


def _above_qe_floor(
    remaining: list[_Candidate],
    sources: SourceRows,
    kind: Kind,
    cfg: QeConfig,
    scorer: QEScorer,
    counts: dict[str, int],
) -> list[_Candidate]:
    real_texts = [sources.texts[item.source] for item in remaining]
    synthetic_texts = [item.text for item in remaining]
    pair = (
        (synthetic_texts, real_texts) if kind == "bt" else (real_texts, synthetic_texts)
    )
    scores = _score_in_chunks(scorer, *pair, batch_size=cfg.batch_size)
    kept = [
        dataclasses.replace(item, qe=score)
        for item, score in zip(remaining, scores, strict=True)
        if _passes_qe(score, cfg)
    ]
    counts[DROPPED_QE] += len(remaining) - len(kept)
    return kept


def _score_in_chunks(
    scorer: QEScorer,
    source_texts: Sequence[str],
    translations: Sequence[str],
    *,
    batch_size: int,
) -> list[float]:
    scores: list[float] = []
    for start in range(0, len(source_texts), batch_size):
        chunk_sources = source_texts[start : start + batch_size]
        chunk_translations = translations[start : start + batch_size]
        chunk_scores = scorer.score(chunk_sources, chunk_translations)
        if len(chunk_scores) != len(chunk_sources):
            raise UpstreamError(
                f"QE scorer returned {len(chunk_scores)} scores "
                f"for {len(chunk_sources)} pairs"
            )
        scores.extend(float(score) for score in chunk_scores)
    return scores


def _without_copies(
    remaining: list[_Candidate], sources: SourceRows, counts: dict[str, int]
) -> list[_Candidate]:
    kept = [item for item in remaining if item.text != sources.texts[item.source]]
    counts[DROPPED_COPY] += len(remaining) - len(kept)
    return kept


def _pick_winners(survivors: list[_Candidate]) -> dict[int, _Candidate]:
    """The winning survivor of every source that has one. A lone survivor wins
    unscored; the sources with several go through one MBR call, which scores
    each survivor against the other survivors of its source."""
    by_source: dict[int, list[_Candidate]] = {}
    for item in survivors:
        by_source.setdefault(item.source, []).append(item)
    winners = {
        source: items[0] for source, items in by_source.items() if len(items) == 1
    }
    contested = {source: items for source, items in by_source.items() if len(items) > 1}
    if contested:
        picks = mbr.select(
            [[item.text for item in items] for items in contested.values()],
            ChrfUtility(),
        )
        for (source, items), pick in zip(contested.items(), picks, strict=True):
            winners[source] = items[pick]
    return winners


def _selection(
    winners: Mapping[int, _Candidate],
    source_count: int,
    counts: dict[str, int],
    qe_status: str,
) -> Selection:
    per_source = [winners.get(source) for source in range(source_count)]
    return Selection(
        chosen=tuple(None if won is None else won.index for won in per_source),
        qe_scores=tuple(None if won is None else won.qe for won in per_source),
        purity=tuple(None if won is None else won.purity for won in per_source),
        counts=MappingProxyType(counts),
        qe_status=qe_status,
    )


def _pair(
    kind: Kind, synthetic: str, real: str, mono_lang: str, other_lang: str
) -> dict[str, str]:
    """The languages and sides of the pair a synthetic text makes with its real
    text: the synthetic text is the source of a `bt` pair and the target of an
    `ft` pair."""
    if kind == "bt":
        return {
            "src_lang": other_lang,
            "tgt_lang": mono_lang,
            "src": synthetic,
            "tgt": real,
        }
    return {
        "src_lang": mono_lang,
        "tgt_lang": other_lang,
        "src": real,
        "tgt": synthetic,
    }


def _row_scores(
    kind: Kind, purity: float | None, qe: float | None, qe_scorer: str | None
) -> dict[str, float | None]:
    scores: dict[str, float | None] = dict.fromkeys(_SCORE_FIELDS)
    scores["script_src" if kind == "bt" else "script_tgt"] = purity
    if qe_scorer is not None:
        scores[qe_scorer] = qe
    return scores


def _registry_scripts(language: str) -> tuple[Script, ...]:
    for entry in load_languages():
        if entry.code == language:
            # Registry order, so that a float sum over the scripts is the same
            # on every run.
            return tuple(dict.fromkeys(entry.scripts))
    raise ConfigError(
        f"language code {language!r} is not in the language registry "
        "(configs/languages.yaml)"
    )


def _require_kind(kind: str) -> None:
    if kind not in get_args(Kind):
        raise ValueError(f"kind must be one of 'bt', 'ft', got {kind!r}")


def _require_one_list_per_source(
    sources: SourceRows, candidates: Sequence[Sequence[str]]
) -> None:
    if len(candidates) != len(sources.ids):
        raise ValueError(
            f"{len(candidates)} candidate lists for {len(sources.ids)} sources"
        )
    for source, texts in enumerate(candidates):
        if isinstance(texts, str):
            raise TypeError(
                f"source {source}: candidates must be a sequence of strings, "
                "not a string"
            )


def _require_selection_of(
    sources: SourceRows,
    candidates: Sequence[Sequence[str]],
    selection: Selection,
) -> None:
    if len(selection.chosen) != len(sources.ids):
        raise ValueError(
            f"{len(selection.chosen)} chosen candidates for {len(sources.ids)} sources"
        )
    for source, pick in enumerate(selection.chosen):
        if pick is not None and not 0 <= pick < len(candidates[source]):
            raise ValueError(
                f"source {source}: chosen candidate {pick} is outside the "
                f"candidate list of {len(candidates[source])}"
            )


def _require_qe_column(qe_scorer: str | None, selection: Selection) -> None:
    if qe_scorer is not None and qe_scorer not in _QE_COLUMNS:
        raise ValueError(
            f"qe_scorer must be one of {', '.join(map(repr, _QE_COLUMNS))} or "
            f"None, got {qe_scorer!r}"
        )
    if qe_scorer is None and any(score is not None for score in selection.qe_scores):
        raise ValueError("the selection has QE scores: name their scorer in qe_scorer")
