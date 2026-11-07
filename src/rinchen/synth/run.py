"""The job runner: plan a synthetic-data job, then shard by shard get the
candidates (from disk, or generated and written there) and select the best
candidate of every source, returning the corpus rows with the counts. The CLI
owns every file the run leaves besides the shards: this module never writes the
corpus or a manifest."""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from types import MappingProxyType
from typing import Literal

import polars as pl

from rinchen.config.synth import (
    GeneratorConfig,
    JobConfig,
    SelectionConfig,
    SynthConfig,
)
from rinchen.errors import ConfigError, DataError, UpstreamError
from rinchen.filter.qe import QEScorer
from rinchen.synth import bt, ft
from rinchen.synth.plan import Plan
from rinchen.synth.select import (
    COUNT_KEYS,
    QeStatus,
    SourceRows,
    build_rows,
    select_candidates,
)
from rinchen.synth.shards import (
    Invalid,
    Loaded,
    Missing,
    Shard,
    ShardStore,
    job_fingerprint,
    plan_shards,
    shard_directory,
)
from rinchen.synth.translator import Translator

_PLANNERS: Mapping[str, Callable[[JobConfig, pl.DataFrame], Plan]] = {
    "bt": bt.plan,
    "ft": ft.plan,
}


@dataclass(frozen=True, slots=True)
class ShardProgress:
    """One shard is done: reused from disk, or generated and written."""

    index: int
    total: int
    reused: bool


@dataclass(frozen=True, slots=True)
class RunResult:
    """What a run produced: the corpus rows and the selection counts (see
    `rinchen.synth.select.Selection` for the keys), whether a QE scorer ran, the
    shard statistics (`planned`, `reused`, `generated`, `regenerated_invalid`,
    all always present), the job fingerprint and the directory of its shards."""

    corpus: pl.DataFrame
    counts: Mapping[str, int]
    qe_status: QeStatus
    shard_stats: Mapping[str, int]
    fingerprint: str
    shard_directory: Path


def run_job(
    job: JobConfig,
    generator_name: str,
    generator_cfg: GeneratorConfig,
    translator: Translator,
    frame: pl.DataFrame,
    *,
    out_dir: Path,
    cfg: SynthConfig,
    scorer: QEScorer | None,
    seed: int,
    force: bool = False,
    progress: Callable[[ShardProgress], None] | None = None,
) -> RunResult:
    """Run `job` over the monolingual `frame`, as `read_monolingual` returns it
    for `job.mono_lang`, with the generator `generator_name` (the one `job`
    names, a key of `cfg.generators`, whose config is `generator_cfg`).

    The frame is cut into shards of `job.shard_size` rows, kept in
    `<out_dir>/shards-<fingerprint[:12]>/`, and the shards are handled one at a
    time in index order. A shard with a valid file there is reused (the file is
    decoded once). Otherwise it is sent to `translator` and written before the
    next shard is started, so a run that dies keeps the finished shards and a
    rerun resumes after them. Either way the shard's candidates are then
    selected on their own, with `scorer` (or None to skip the QE floor), and
    the shard's candidates are dropped. Selection is per source, so the rows
    and the summed counts are those of one selection over the whole frame; only
    the grouping of the scorer calls differs. `progress` is called once per
    shard in order, after the shard is on disk or found there and before it is
    selected. `seed` only names the shard directory: the translators derive
    their per-request seeds themselves.

    With `force` the store is ignored: every shard is generated again and
    overwritten, whatever is on disk, and `regenerated_invalid` stays 0. The
    shard statistics count the shards `planned`, `reused` from disk,
    `generated` and, of those, the `regenerated_invalid` ones whose file was
    present but damaged or for other ids. All rows carry one `created` stamp.

    Raises ConfigError when `job.generator` is not `generator_name`, when
    `generator_name` is not in `cfg` or when `generator_cfg` is not its config,
    and whatever the job's planner raises for a frame that does not fit the job
    (DataError). A `translator` that raises UpstreamError for a shard gives an
    UpstreamError naming the shard, chained to it; a reply with other than one
    list of strings per text is a DataError naming the shard, and nothing is
    written for it. Any other exception propagates as it is, with the shards
    written so far on disk.

    Memory: the caller's frame is the largest object (planning copies every
    text into Python tuples, about 400 MB per million rows of 190 characters,
    so drop your reference to it after the call), plus the planned shards, the
    corpus rows accumulated so far and the candidates of ONE shard. Candidates
    the translators dropped (empty, duplicate, no letter or digit, not
    encodable) are counted nowhere; a source whose generator refused shows up
    as `no_candidates`."""
    _require_configured_generator(job, cfg, generator_name, generator_cfg)
    plan = _PLANNERS[job.kind](job, frame)
    shards = plan_shards(frame, job.shard_size)
    fingerprint = job_fingerprint(frame, job, generator_cfg, job.shard_size, seed)
    directory = shard_directory(out_dir, fingerprint)
    store = ShardStore(directory)
    selector = _ShardSelector(
        plan=plan,
        cfg=cfg.selection,
        scorer=scorer,
        generator_kind=generator_cfg.kind,
        generator_name=generator_name,
        created=datetime.now(UTC),
    )

    selected: list[_Selected] = []
    sourced: list[_Sourced] = []
    for shard in shards:
        candidates, how = _candidates_of(
            store, shard, len(shards), plan, translator, force
        )
        sourced.append(how)
        if progress is not None:
            progress(ShardProgress(shard.index, len(shards), how == "reused"))
        selected.append(selector.select(shard, candidates))
    return RunResult(
        corpus=pl.concat([part.rows for part in selected]),
        counts=MappingProxyType(
            {key: sum(part.counts[key] for part in selected) for key in COUNT_KEYS}
        ),
        qe_status=selected[0].qe_status,
        shard_stats=_shard_stats(sourced),
        fingerprint=fingerprint,
        shard_directory=directory,
    )


def _require_configured_generator(
    job: JobConfig,
    cfg: SynthConfig,
    generator_name: str,
    generator_cfg: GeneratorConfig,
) -> None:
    if job.generator != generator_name:
        raise ConfigError(
            f"the job names generator {job.generator!r} but {generator_name!r} "
            "was given"
        )
    configured = cfg.generators.get(generator_name)
    if configured is None:
        raise ConfigError(
            f"generator {generator_name!r} is not in the config; "
            f"configured generators: {sorted(cfg.generators)}"
        )
    if configured != generator_cfg:
        raise ConfigError(
            f"the generator config given for {generator_name!r} differs from "
            "the one in the synth config"
        )


# How a shard's candidates were obtained: read from a valid file, generated
# because there was no file, or generated to replace a file that failed
# validation.
_Sourced = Literal["reused", "generated", "regenerated"]


def _candidates_of(
    store: ShardStore,
    shard: Shard,
    total: int,
    plan: Plan,
    translator: Translator,
    force: bool,
) -> tuple[list[list[str]], _Sourced]:
    how: _Sourced = "generated"
    if not force:
        match store.load(shard.index, shard.ids):
            case Loaded(candidates=candidates):
                return candidates, "reused"
            case Invalid():
                how = "regenerated"
            case Missing():
                pass
    return _generate_shard(store, shard, total, plan, translator), how


def _generate_shard(
    store: ShardStore, shard: Shard, total: int, plan: Plan, translator: Translator
) -> list[list[str]]:
    """The translator's candidate lists for the shard, validated and written."""
    try:
        reply = translator.translate(
            shard.texts, plan.generator_src_lang, plan.generator_tgt_lang
        )
    except UpstreamError as error:
        raise UpstreamError(
            f"shard {shard.index} of {total} failed: {error}"
        ) from error
    _require_one_candidate_list_per_text(shard, reply)
    store.write(shard.index, shard.ids, reply)
    return reply


def _require_one_candidate_list_per_text(shard: Shard, reply: list[list[str]]) -> None:
    if len(reply) != len(shard.texts):
        raise DataError(
            f"shard {shard.index}: the translator returned {len(reply)} "
            f"candidate lists for {len(shard.texts)} texts"
        )
    for position, candidates in enumerate(reply):
        if not isinstance(candidates, list) or not all(
            isinstance(text, str) for text in candidates
        ):
            raise DataError(
                f"shard {shard.index}: the translator's reply for text "
                f"{position} is not a list of strings"
            )


@dataclass(frozen=True, slots=True)
class _Selected:
    """The selection of one shard: its corpus rows and its counts."""

    rows: pl.DataFrame
    counts: Mapping[str, int]
    qe_status: QeStatus


@dataclass(frozen=True, slots=True)
class _ShardSelector:
    """What every shard of a run is selected with."""

    plan: Plan
    cfg: SelectionConfig
    scorer: QEScorer | None
    generator_kind: str
    generator_name: str
    created: datetime

    def select(self, shard: Shard, candidates: list[list[str]]) -> _Selected:
        sources = SourceRows(ids=shard.ids, texts=shard.texts)
        selection = select_candidates(
            sources,
            candidates,
            kind=self.plan.kind,
            synthetic_lang=self.plan.synthetic_lang,
            cfg=self.cfg,
            scorer=self.scorer,
        )
        rows = build_rows(
            sources,
            candidates,
            selection,
            kind=self.plan.kind,
            generator_kind=self.generator_kind,
            generator_name=self.generator_name,
            mono_lang=self.plan.real_lang,
            other_lang=self.plan.synthetic_lang,
            qe_scorer=self.cfg.qe.scorer if self.scorer is not None else None,
            created=self.created,
        )
        return _Selected(rows, selection.counts, selection.qe_status)


def _shard_stats(sourced: list[_Sourced]) -> Mapping[str, int]:
    tally = Counter(sourced)
    return MappingProxyType(
        {
            "planned": len(sourced),
            "reused": tally["reused"],
            "generated": tally["generated"] + tally["regenerated"],
            "regenerated_invalid": tally["regenerated"],
        }
    )
