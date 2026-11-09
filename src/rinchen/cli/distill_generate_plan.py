"""What `rinchen distill generate` would do (D10): the plan of a run and the lines that print
it. A plan holds the verified teacher, how it decodes, the selection, the sources of each
direction, the shards they make and how many of those are already on disk, and the
fingerprint that names the shard directory. Making one reads the shard files and writes
nothing, so a dry run and a real run print the same plan.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from rinchen.config.synth import SelectionConfig
from rinchen.export.verified import VerifiedExport
from rinchen.synth.shards import ShardStore, plan_shards, shard_directory
from rinchen.train.distill_generate import kd_fingerprint
from rinchen.train.distill_sources import KdSources
from rinchen.train.distill_teacher import TeacherOptions


@dataclass(frozen=True, slots=True, eq=False)
class GeneratePlan:
    """The inputs of a run and what follows from them: `shards_by_direction` (shards each
    direction makes), `shards_reusable` (those already on disk with a valid file, 0 when
    `force` ignores them), `fingerprint` and `shard_directory`. `qe` says whether the QE
    floor was asked for."""

    export: VerifiedExport
    options: TeacherOptions
    selection: SelectionConfig
    sources: list[KdSources]
    name: str
    shard_size: int
    seed: int
    qe: bool
    force: bool
    corpus_files: int
    fingerprint: str
    shards_by_direction: dict[str, int]
    shards_reusable: int
    shard_directory: Path


def make_plan(
    export: VerifiedExport,
    options: TeacherOptions,
    selection: SelectionConfig,
    *,
    sources: list[KdSources],
    name: str,
    shard_size: int,
    seed: int,
    qe_asked: bool,
    force: bool,
    corpus_files: int,
    target: Path,
) -> GeneratePlan:
    """The plan of a run into `target`: its fingerprint, the shards each direction makes and
    how many of them a run would reuse; reads the shard files and writes nothing."""
    fingerprint = kd_fingerprint(
        export, options, sources, shard_size=shard_size, seed=seed
    )
    directory = shard_directory(target, fingerprint)
    shards = {
        source.direction: plan_shards(source.frame, shard_size) for source in sources
    }
    reusable = (
        0
        if force
        else sum(
            len(ShardStore(directory / direction).completed(planned))
            for direction, planned in shards.items()
        )
    )
    return GeneratePlan(
        export=export,
        options=options,
        selection=selection,
        sources=sources,
        name=name,
        shard_size=shard_size,
        seed=seed,
        qe=qe_asked,
        force=force,
        corpus_files=corpus_files,
        fingerprint=fingerprint,
        shards_by_direction={
            direction: len(planned) for direction, planned in shards.items()
        },
        shards_reusable=reusable,
        shard_directory=directory,
    )


def plan_lines(plan: GeneratePlan) -> list[str]:
    """The plan, one fact per line."""
    options = plan.options
    return [
        f"corpus files: {plan.corpus_files}",
        f"teacher: {plan.name} (tag placement {plan.export.record.tag_placement})",
        *(
            f"direction {source.direction}: sources {source.frame.height}, "
            f"shards {plan.shards_by_direction[source.direction]}"
            for source in plan.sources
        ),
        f"sources: {sum(source.frame.height for source in plan.sources)}",
        (
            f"shards: {sum(plan.shards_by_direction.values())} "
            f"(reusable {plan.shards_reusable})"
        ),
        (
            f"decoding: beam size {options.beam_size}, n {options.n}, max decoding "
            f"length {options.max_decoding_length}, max source pieces "
            f"{options.max_source_pieces}, batch size {options.batch_size}, device "
            f"{options.device}, compute type {options.compute_type}, threads "
            f"{options.threads}"
        ),
        f"qe: {'requested' if plan.qe else 'not requested'}",
        f"fingerprint: {plan.fingerprint}",
        f"shard directory: {plan.shard_directory}",
    ]
