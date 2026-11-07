"""The job runner: plan a synthetic-data job, generate the shards of candidates
that are not on disk yet, select the best candidate of every source and return
the corpus rows with the counts. The CLI owns every file the run leaves besides
the shards: this module never writes the corpus or a manifest."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

import polars as pl

from rinchen.config.synth import GeneratorConfig, JobConfig, SynthConfig
from rinchen.errors import ConfigError, DataError, UpstreamError
from rinchen.filter.qe import QEScorer
from rinchen.synth import bt, ft
from rinchen.synth.plan import Plan
from rinchen.synth.select import (
    QeStatus,
    SourceRows,
    build_rows,
    select_candidates,
)
from rinchen.synth.shards import (
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
    for `job.mono_lang`, with the generator `generator_name` (a key of
    `cfg.generators`, whose config is `generator_cfg`).

    The frame is cut into shards of `job.shard_size` rows, kept in
    `<out_dir>/shards-<fingerprint[:12]>/`. Each shard without a valid file
    there is sent to `translator` and written before the next one is started,
    so a run that dies keeps the finished shards and a rerun resumes after them.
    With `force` every shard is generated again. Then every shard is read back
    from disk and the candidates are selected, so a reused and a fresh shard
    reach the selection the same way, and `scorer` (or None to skip the QE
    floor) rates them. `seed` only names the shard directory: the translators
    derive their per-request seeds themselves. `progress` is called once per
    shard in order, after the shard is on disk or found there.

    The shard statistics count the shards `planned`, `reused` from disk,
    `generated` and, of those, the `regenerated_invalid` ones whose file was
    present but damaged or for other ids (always 0 with `force`).

    Raises ConfigError when `generator_name` is not in `cfg` or `generator_cfg`
    is not its config, and whatever the job's planner raises for a frame that
    does not fit the job (DataError). A `translator` that raises UpstreamError
    for a shard gives an UpstreamError naming the shard, chained to it; a reply
    with other than one list of strings per text is a DataError naming the
    shard, and nothing is written for it. Any other exception propagates as it
    is, with the finished shards on disk.

    The caller's frame is the largest object in memory: planning copies every
    text into Python tuples (about 400 MB per million rows of 190 characters),
    so drop your reference to it after the call. Candidates the translators
    dropped (empty, duplicate, no letter or digit, not encodable) are counted
    nowhere; a source whose generator refused shows up as `no_candidates`."""
    _require_configured_generator(cfg, generator_name, generator_cfg)
    plan = _PLANNERS[job.kind](job, frame)
    shards = plan_shards(frame, job.shard_size)
    fingerprint = job_fingerprint(frame, job, generator_cfg, job.shard_size, seed)
    directory = shard_directory(out_dir, fingerprint)
    store = ShardStore(directory)

    done = frozenset() if force else store.completed(shards)
    invalid = frozenset() if force else store.invalid(shards)
    _generate_missing_shards(store, shards, plan, translator, done, progress)

    sources = SourceRows(
        ids=tuple(id_ for shard in shards for id_ in shard.ids),
        texts=tuple(text for shard in shards for text in shard.texts),
    )
    candidates = [
        texts for shard in shards for texts in store.read(shard.index, shard.ids)
    ]
    selection = select_candidates(
        sources,
        candidates,
        kind=plan.kind,
        synthetic_lang=plan.synthetic_lang,
        cfg=cfg.selection,
        scorer=scorer,
    )
    corpus = build_rows(
        sources,
        candidates,
        selection,
        kind=plan.kind,
        generator_kind=generator_cfg.kind,
        generator_name=generator_name,
        mono_lang=plan.real_lang,
        other_lang=plan.synthetic_lang,
        qe_scorer=cfg.selection.qe.scorer if scorer is not None else None,
    )
    return RunResult(
        corpus=corpus,
        counts=selection.counts,
        qe_status=selection.qe_status,
        shard_stats=MappingProxyType(
            {
                "planned": len(shards),
                "reused": len(done),
                "generated": len(shards) - len(done),
                "regenerated_invalid": len(invalid),
            }
        ),
        fingerprint=fingerprint,
        shard_directory=directory,
    )


def _require_configured_generator(
    cfg: SynthConfig, generator_name: str, generator_cfg: GeneratorConfig
) -> None:
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


def _generate_missing_shards(
    store: ShardStore,
    shards: tuple[Shard, ...],
    plan: Plan,
    translator: Translator,
    done: frozenset[int],
    progress: Callable[[ShardProgress], None] | None,
) -> None:
    for shard in shards:
        reused = shard.index in done
        if not reused:
            _generate_shard(store, shard, len(shards), plan, translator)
        if progress is not None:
            progress(ShardProgress(shard.index, len(shards), reused))


def _generate_shard(
    store: ShardStore, shard: Shard, total: int, plan: Plan, translator: Translator
) -> None:
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
