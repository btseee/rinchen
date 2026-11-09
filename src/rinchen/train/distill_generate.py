"""The generation of a knowledge-distillation (KD) corpus (D10): the teacher translates the
sources, D6's selection keeps one hypothesis per source and the winners become corpus rows.

This module is the loop that joins D6's pieces for a teacher that is a checkpoint of ours:
D6's shards (`plan_shards`, `ShardStore`: one Parquet file of candidates per shard, written
before the next shard is asked for, so a crash resumes), D6's selection (`select_candidates`
and `build_rows` with the kind `ft` and the generator kind `teacher`, which gives the origin
`ft`) and D6's QE hook. What is new is what D6's job does not have: the fingerprint that
names the shard directory covers the teacher's files (D6's covers a model's name, not its
weights), the sources of several directions share one directory, and the counts are kept
per direction.

The KD target of a source is the teacher's best hypothesis (`TeacherOptions.n` is 1 by
default; with `n > 1` the selection picks among the survivors by minimum Bayes risk, which
lowered the agreement with the references on the toy). A source without a letter or a digit
(only whitespace, punctuation or invisible characters such as a zero-width space) is not
sent to the teacher, which would make up a sentence for it: it has no candidates, is counted
under `no_candidates` and in `KdResult.letterless_sources`.

`KdResult.teacher_counters` are the counters of the translator of THIS process: a shard read
back from disk adds nothing to them. No GPU is used or claimed by this module; the teacher
decides the device.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from itertools import pairwise
from pathlib import Path
from types import MappingProxyType
from typing import Final, Protocol

import polars as pl

from rinchen.config.synth import SelectionConfig
from rinchen.corpus.schema import CORPUS_SCHEMA
from rinchen.errors import ConfigError
from rinchen.filter.qe import QEScorer
from rinchen.synth._candidates import has_alphanumeric
from rinchen.synth.select import (
    COUNT_KEYS,
    QE_OK,
    QE_SKIPPED_NO_SCORER,
    QeStatus,
    Selection,
    SourceRows,
    build_rows,
    select_candidates,
)
from rinchen.synth.shards import Shard, ShardStore, plan_shards, shard_directory
from rinchen.synth.translator import Translator
from rinchen.train.distill_export import CT2_FILES, SPM_FILE, VerifiedExport
from rinchen.train.distill_shards import REUSED, Sourced, shard_candidates
from rinchen.train.distill_sources import KdSources
from rinchen.train.distill_teacher import TeacherOptions

TOTAL: Final = "total"

_NAME: Final = re.compile(r"[A-Za-z0-9._-]{1,64}")
_DEFAULT_NAME_PREFIX: Final = "rinchen-"
_DIGEST_SHOWN: Final = 12
_TEACHER_FILES: Final = (*CT2_FILES, SPM_FILE)
# The options that decide a hypothesis; the batch size and the thread count do not.
_DECODING_OPTIONS: Final = (
    "beam_size",
    "n",
    "max_decoding_length",
    "max_source_pieces",
    "device",
    "compute_type",
)


class KdTranslator(Translator, Protocol):
    """D6's `Translator` that also reports what it did: `TeacherTranslator` is one."""

    def counters(self) -> Mapping[str, int]: ...


@dataclass(frozen=True, slots=True)
class ShardProgress:
    """The candidates of shard `index` of `total` of `direction` are obtained (reused from
    disk, or generated and written); its selection follows."""

    direction: str
    index: int
    total: int
    reused: bool


@dataclass(frozen=True, slots=True, eq=False)
class KdResult:
    """What `generate_kd` made.

    `corpus` holds the rows in the corpus schema, direction by direction, each in the order
    of the source ids. `counts` has one entry per direction and `total`, each with every key
    of `synth.select.COUNT_KEYS` (the invariants of `Selection` hold for each);
    `matches_reference` counts the rows whose target is an authentic target of their source
    (a diagnostic); `letterless_sources` counts the sources that were not sent for lack of a
    letter or a digit (a part of `no_candidates`). `shard_stats` has `planned`, `reused`,
    `generated` and, of those, `regenerated_invalid`. `qe_status` says whether a scorer ran.
    """

    corpus: pl.DataFrame
    counts: Mapping[str, Mapping[str, int]]
    teacher_counters: Mapping[str, int]
    matches_reference: Mapping[str, int]
    letterless_sources: Mapping[str, int]
    qe_status: QeStatus
    shard_stats: Mapping[str, int]
    fingerprint: str
    shard_directory: Path


def kd_name(export: VerifiedExport, name: str | None) -> str:
    """The name of a KD corpus: `name` when it is 1 to 64 characters of `A-Z a-z 0-9 . _ -`,
    else (when None) `rinchen-` and the first 12 hex characters of the SHA-256 of the
    teacher's `model.bin`.

    Raises:
        ConfigError: `name` is not such a name.
    """
    if name is None:
        return f"{_DEFAULT_NAME_PREFIX}{export.sha256['model.bin'][:_DIGEST_SHOWN]}"
    if not _NAME.fullmatch(name):
        raise ConfigError(
            "the name of a KD corpus must be 1 to 64 characters of A-Z, a-z, 0-9, "
            f"'.', '_' and '-', got {name[:80]!r}"
        )
    return name


def kd_fingerprint(
    export: VerifiedExport,
    options: TeacherOptions,
    sources: Sequence[KdSources],
    *,
    shard_size: int,
    seed: int,
) -> str:
    """A SHA-256 hex digest over everything that decides what a shard holds: the teacher (the
    digests of `model.bin`, `shared_vocabulary.json`, `config.json` and `spm.model`, and the
    tag placement), the decoding options that decide a hypothesis (not the batch size or the
    thread count, which change the speed only), `shard_size`, `seed` (it only names the
    directory: decoding is deterministic) and, per direction in direction order, the
    `(id, text)` rows in id order. The selection thresholds are not in it: a rerun with
    others reuses the shards, as D6's jobs do. The rows are hashed one at a time.

    Raises:
        ValueError: `shard_size` is below 1.
        ConfigError: two frames are of one direction.
    """
    if shard_size < 1:
        raise ValueError(f"shard_size must be at least 1, got {shard_size}")
    header = {
        "teacher": {
            **{name: export.sha256[name] for name in _TEACHER_FILES},
            "tag_placement": export.record.tag_placement,
        },
        "decoding": {name: getattr(options, name) for name in _DECODING_OPTIONS},
        "shard_size": shard_size,
        "seed": seed,
    }
    digest = hashlib.sha256()
    digest.update(_canonical_json(header))
    for source in _by_direction(sources):
        rows = source.frame.select("id", "text").sort("id", maintain_order=True)
        # The direction names both languages (KdSources checks it); an object between the
        # arrays of two directions' rows tells where one ends.
        digest.update(_canonical_json({"direction": source.direction}))
        for row in rows.iter_rows():
            digest.update(_canonical_json(row))
    return digest.hexdigest()


def _canonical_json(value: object) -> bytes:
    """`value` as canonical JSON (sorted keys, no spaces, ASCII only); every value hashed
    here is an object or an array, so one cannot run into the next."""
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("ascii")


def _by_direction(sources: Sequence[KdSources]) -> list[KdSources]:
    ordered = sorted(sources, key=lambda source: source.direction)
    for before, after in pairwise(ordered):
        if before.direction == after.direction:
            raise ConfigError(
                f"two source frames are of the direction {before.direction}"
            )
    return ordered


def generate_kd(
    sources: Sequence[KdSources],
    translator: KdTranslator,
    export: VerifiedExport,
    options: TeacherOptions,
    selection: SelectionConfig,
    scorer: QEScorer | None,
    *,
    out_dir: Path,
    shard_size: int,
    seed: int,
    name: str,
    force: bool = False,
    progress: Callable[[ShardProgress], None] | None = None,
) -> KdResult:
    """Translate `sources` with `translator` (the teacher of `export`, decoding as `options`
    says) and select the KD rows, direction by direction in direction order.

    The shards of every direction live in `<out_dir>/shards-<fingerprint[:12]>/<direction>/`.
    A shard with a valid file is reused (decoded once); otherwise it is translated and
    written before the next one is asked for, so a run that dies keeps the finished shards;
    a damaged file is regenerated and counted `regenerated_invalid`; with `force` every
    shard is generated again and none is counted invalid. `progress` is called once per
    shard in order, after the shard is on disk or found there and before it is selected. Rows
    carry the label `ft:<name>` and one `created` stamp. With a `scorer` the QE floor of
    `selection.qe` runs on the pairs (real source, candidate).

    Raises:
        ConfigError: no sources, two frames of one direction, a bad `name`, or an error of
            that kind of the translator (named after its direction and shard).
        ValueError: `shard_size` is below 1.
        DataError: the translator's reply is not a list with one list of strings per text,
            or holds a text UTF-8 cannot encode (nothing is written for that shard), or the
            translator raised one.
        UpstreamError: the translator raised an UpstreamError or an OSError, or the scorer
            failed (D6's rules).
    """
    name = kd_name(export, name)
    ordered = _by_direction(sources)
    if not ordered:
        raise ConfigError("no sources to translate")
    fingerprint = kd_fingerprint(
        export, options, ordered, shard_size=shard_size, seed=seed
    )
    directory = shard_directory(out_dir, fingerprint)
    run = _Run(translator, selection, scorer, name, datetime.now(UTC), force, progress)
    parts = [
        _generate_direction(
            run, source, ShardStore(directory / source.direction), shard_size
        )
        for source in ordered
    ]
    frames = [frame for part in parts for frame in part.rows]
    return KdResult(
        corpus=pl.concat(frames) if frames else pl.DataFrame(schema=CORPUS_SCHEMA),
        counts=_with_total({part.direction: part.counts for part in parts}),
        teacher_counters=dict(translator.counters()),
        matches_reference=_with_total_of(parts, lambda part: part.matches),
        letterless_sources=_with_total_of(parts, lambda part: part.letterless),
        qe_status=QE_OK if scorer is not None else QE_SKIPPED_NO_SCORER,
        shard_stats=_shard_stats([how for part in parts for how in part.sourced]),
        fingerprint=fingerprint,
        shard_directory=directory,
    )


@dataclass(frozen=True, slots=True)
class _Run:
    """What every shard of a generation shares."""

    translator: KdTranslator
    selection: SelectionConfig
    scorer: QEScorer | None
    name: str
    created: datetime
    force: bool
    progress: Callable[[ShardProgress], None] | None


@dataclass(slots=True)
class _Direction:
    """What one direction made: its row frames (one per shard), the summed counts, and how
    its shards were obtained."""

    direction: str
    rows: list[pl.DataFrame]
    counts: dict[str, int]
    matches: int
    letterless: int
    sourced: list[Sourced]


def _generate_direction(
    run: _Run, source: KdSources, store: ShardStore, shard_size: int
) -> _Direction:
    shards = plan_shards(source.frame, shard_size)
    made = _Direction(source.direction, [], dict.fromkeys(COUNT_KEYS, 0), 0, 0, [])
    for shard in shards:
        candidates, how = shard_candidates(
            run.translator, source, store, shard, len(shards), force=run.force
        )
        made.sourced.append(how)
        if run.progress is not None:
            run.progress(
                ShardProgress(source.direction, shard.index, len(shards), how == REUSED)
            )
        _select_shard(run, source, shard, candidates, made)
    return made


def _select_shard(
    run: _Run,
    source: KdSources,
    shard: Shard,
    candidates: list[list[str]],
    made: _Direction,
) -> None:
    """Select the shard's candidates and add its rows, counts and tallies to `made`."""
    rows = SourceRows(ids=shard.ids, texts=shard.texts)
    chosen = select_candidates(
        rows,
        candidates,
        kind="ft",
        synthetic_lang=source.tgt_lang,
        cfg=run.selection,
        scorer=run.scorer,
    )
    made.rows.append(
        build_rows(
            rows,
            candidates,
            chosen,
            kind="ft",
            generator_kind="teacher",
            generator_name=run.name,
            mono_lang=source.src_lang,
            other_lang=source.tgt_lang,
            qe_scorer=run.selection.qe.scorer if run.scorer is not None else None,
            created=run.created,
        )
    )
    for key in COUNT_KEYS:
        made.counts[key] += chosen.counts[key]
    made.matches += _agreeing(source, shard, candidates, chosen)
    made.letterless += sum(not has_alphanumeric(text) for text in shard.texts)


def _agreeing(
    source: KdSources,
    shard: Shard,
    candidates: Sequence[Sequence[str]],
    chosen: Selection,
) -> int:
    """How many selected targets are authentic targets of their own source (an exact
    comparison: the teacher's text is stripped, the reference is as the corpus has it)."""
    return sum(
        1
        for row_id, texts, pick in zip(
            shard.ids, candidates, chosen.chosen, strict=True
        )
        if pick is not None
        and texts[pick] in source.references.get(row_id, frozenset())
    )


def _with_total(
    counts: Mapping[str, dict[str, int]],
) -> Mapping[str, Mapping[str, int]]:
    total = {key: sum(part[key] for part in counts.values()) for key in COUNT_KEYS}
    return MappingProxyType(
        {
            **{direction: MappingProxyType(part) for direction, part in counts.items()},
            TOTAL: MappingProxyType(total),
        }
    )


def _with_total_of(
    parts: Sequence[_Direction], count: Callable[[_Direction], int]
) -> Mapping[str, int]:
    per_direction = {part.direction: count(part) for part in parts}
    return MappingProxyType({**per_direction, TOTAL: sum(per_direction.values())})


def _shard_stats(sourced: list[Sourced]) -> Mapping[str, int]:
    tally = Counter(sourced)
    return MappingProxyType(
        {
            "planned": len(sourced),
            "reused": tally["reused"],
            "generated": tally["generated"] + tally["regenerated"],
            "regenerated_invalid": tally["regenerated"],
        }
    )
