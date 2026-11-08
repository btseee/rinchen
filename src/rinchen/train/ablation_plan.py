"""The plan of an ablation, before anything is built or trained: the training and the
dev rows read from the corpus files, the variants of the matrix, and for each variant
its corpora with their sizes, mixing probabilities, integer weights and upsampling.

Planning reads and decides; it writes nothing (a dry run prints it). The sizes of the
corpora come from `corpus_files.planned_corpus_sizes`, the same function the build of
a variant uses, so a plan cannot disagree with what is built.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Final

from rinchen.config.ablation import AblationMatrix
from rinchen.errors import ConfigError, DataError
from rinchen.train.corpus_files import (
    BtUsage,
    bt_usage_by_direction,
    planned_corpus_sizes,
)
from rinchen.train.eole_cfg import reject_dollar
from rinchen.train.mixture import (
    DevRows,
    TrainingRows,
    load_dev_rows,
    load_training_rows,
)
from rinchen.train.sampling import integer_weights, mixing_probabilities
from rinchen.train.variants import Variant, expand_variants

# The keys of a variant's Eole config that the run sets itself, by dotted path: a
# matrix override of one would be lost or would break the variant.
RESERVED_OVERRIDE_KEYS: Final = (
    "data",
    "src_vocab",
    "seed",
    "training.model_path",
    "transforms_configs.sentencepiece",
)


@dataclass(frozen=True, slots=True)
class PlannedCorpus:
    """A training corpus of a variant as the plan has it: its rows, the probability of
    drawing it, its integer weight and `upsampling`, `(weight / sum of weights) /
    (rows / sum of rows)`: 1.0 is the share it would get if every row of the mixture
    were drawn alike, more than 1 means its rows are seen more often than that."""

    name: str
    direction: str
    rows: int
    probability: float
    weight: int
    upsampling: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class PlannedVariant:
    """One variant before it runs: its corpora in the order of the Eole config, the bt
    usage of each direction and the source collision rate of the training rows."""

    variant: Variant
    corpora: tuple[PlannedCorpus, ...]
    bt_by_direction: dict[str, BtUsage]
    source_collision_rate: float

    @property
    def weights(self) -> dict[str, int]:
        return {corpus.name: corpus.weight for corpus in self.corpora}

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready: the variant's axes and id flat, then `corpora`, `bt` (by
        direction) and `source_collision_rate`."""
        return {
            **asdict(self.variant),
            "corpora": [corpus.to_dict() for corpus in self.corpora],
            "bt": {
                direction: asdict(usage)
                for direction, usage in self.bt_by_direction.items()
            },
            "source_collision_rate": self.source_collision_rate,
        }


@dataclass(frozen=True, slots=True, eq=False)
class AblationPlan:
    """What `plan_ablation` read and decided: the matrix, the training rows with their
    statistics, the dev sets with theirs, and every variant planned. `out_dir` and
    `spm_model` are absolute."""

    matrix: AblationMatrix
    corpus_paths: tuple[Path, ...]
    spm_model: Path
    out_dir: Path
    training: TrainingRows
    dev: DevRows
    variants: tuple[PlannedVariant, ...]

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready, for the dry run to print."""
        return {
            "name": self.matrix.name,
            "out_dir": self.out_dir.as_posix(),
            "spm_model": self.spm_model.as_posix(),
            "corpus_paths": [path.as_posix() for path in self.corpus_paths],
            "weight_total": self.matrix.weight_total,
            "statistics": self.training.stats.to_dict(),
            "dev_statistics": self.dev.stats.to_dict(),
            "variants": [planned.to_dict() for planned in self.variants],
        }


def plan_ablation(
    matrix: AblationMatrix,
    corpus_paths: Sequence[Path],
    spm_model: Path,
    out_dir: Path,
) -> AblationPlan:
    """Read the training and the dev rows of the corpus files and plan every variant
    of `matrix`: the corpus sizes, mixing probabilities, integer weights (summing to
    `matrix.weight_total`) and upsampling. Nothing is trained and nothing is written:
    `out_dir` is not even created.

    Raises:
        ConfigError: the overrides set a key the run sets itself
            (`RESERVED_OVERRIDE_KEYS`); `out_dir` or `spm_model` holds `$`, which Eole
            would expand as a variable; a language outside the registry; a variant's
            temperature is so large that the sizes raised to it overflow; more corpora
            than `weight_total`.
        DataError: a corpus file cannot be read; no training row, no authentic one, no
            corpus; no dev row for a direction that trains.
    """
    _reject_reserved_overrides(matrix.overrides)
    reject_dollar("output directory", out_dir)
    reject_dollar("SentencePiece model path", spm_model)
    training = load_training_rows(corpus_paths)
    dev = load_dev_rows(
        corpus_paths,
        max_lines=matrix.dev_max_lines,
        directions=training.directions,
        training=training,
    )
    return AblationPlan(
        matrix=matrix,
        corpus_paths=tuple(corpus_paths),
        spm_model=spm_model.resolve(),
        out_dir=out_dir.resolve(),
        training=training,
        dev=dev,
        variants=tuple(
            _plan_variant(training, variant, matrix.weight_total)
            for variant in expand_variants(matrix)
        ),
    )


def _reject_reserved_overrides(overrides: Mapping[str, Any]) -> None:
    for dotted in RESERVED_OVERRIDE_KEYS:
        if _overrides_touch(overrides, dotted.split(".")):
            raise ConfigError(
                f"invalid overrides: {dotted}: the ablation sets it for every "
                "variant, so it cannot be overridden"
            )


def _overrides_touch(overrides: Mapping[str, Any], path: Sequence[str]) -> bool:
    """Whether the overrides set, delete or replace the value at `path` (a value that
    is no mapping, `None` included, replaces everything below its key)."""
    node: Any = overrides
    for key in path:
        if not isinstance(node, Mapping):
            return True
        if key not in node:
            return False
        node = node[key]
    return True


def _plan_variant(
    training: TrainingRows, variant: Variant, weight_total: int
) -> PlannedVariant:
    sizes = planned_corpus_sizes(training, variant.bt_ratio)
    if not sizes:
        raise DataError("the training rows hold no corpus to train on")
    try:
        probabilities = mixing_probabilities(sizes, variant.temperature)
        weights = integer_weights(probabilities, total=weight_total)
    except ValueError as exc:
        raise ConfigError(f"variant {variant.id}: {exc}") from exc
    row_total = sum(size.rows for size in sizes)
    weight_sum = sum(weights.values())
    return PlannedVariant(
        variant=variant,
        corpora=tuple(
            PlannedCorpus(
                name=size.name,
                direction=size.direction,
                rows=size.rows,
                probability=probabilities[size.name],
                weight=weights[size.name],
                upsampling=(weights[size.name] / weight_sum) / (size.rows / row_total),
            )
            for size in sizes
        ),
        bt_by_direction=bt_usage_by_direction(training, variant.bt_ratio),
        source_collision_rate=training.stats.source_collision_rate,
    )
