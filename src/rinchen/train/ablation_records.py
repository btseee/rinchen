"""What a variant and a run leave on record: the `VariantResult` that `result.json`
holds, the `AblationResult` the tables and the manifest are written from, the
`VariantProgress` events, and the check that decides whether a `result.json` on disk
may be trusted.

A variant is complete when its `result.json` parses, says `ok`, holds the fingerprint
of its inputs as they are now, scores the plan's dev sets, and the checkpoint file it
names still has the recorded digest: only then is it reused (`complete_result`).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any, Final, Literal

from pydantic import BaseModel, ConfigDict, Field

from rinchen.manifest.stage import sha256_of
from rinchen.train.ablation_plan import AblationPlan, PlannedVariant
from rinchen.train.ablation_prepare import PreparedVariant
from rinchen.train.evaluate import DevResult
from rinchen.train.mixture import DevSet

_STATUS_OK: Final = "ok"


@dataclass(frozen=True, slots=True)
class VariantProgress:
    """A variant is done, `trained` just now or `reused` from an earlier run (its
    `result.json` is on disk either way). `index` counts from 1. The seconds are those
    of this call: 0.0 for a reused variant."""

    variant_id: str
    index: int
    total: int
    outcome: Literal["reused", "trained"]
    train_seconds: float
    eval_seconds: float


@dataclass(frozen=True, slots=True)
class VariantResult:
    """A complete variant: its plan, the dev scores, the seconds training and
    evaluation took (those of the run that trained it, also when `reused`), and its
    checkpoint file with the SHA-256 of that file. `fingerprint` is that of the inputs
    it was trained from."""

    planned: PlannedVariant
    reused: bool
    fingerprint: str
    dev: DevResult
    train_seconds: float
    eval_seconds: float
    checkpoint_path: Path
    checkpoint_sha256: str

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready: what `result.json` holds. `reused` is left out: it says what a
        run did, not what the variant is."""
        return {
            **self.planned.to_dict(),
            "status": _STATUS_OK,
            "fingerprint": self.fingerprint,
            "train_seconds": self.train_seconds,
            "eval_seconds": self.eval_seconds,
            "checkpoint_file": self.checkpoint_path.as_posix(),
            "checkpoint_sha256": self.checkpoint_sha256,
            "dev": self.dev.to_dict(),
        }


@dataclass(frozen=True, slots=True, eq=False)
class AblationResult:
    """A finished run: the plan, the threads it ran on and one `VariantResult` per
    variant in plan order."""

    plan: AblationPlan
    threads: int
    variants: tuple[VariantResult, ...]

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready: what `results.json` holds."""
        return {
            "name": self.plan.matrix.name,
            "threads": self.threads,
            "matrix": self.plan.matrix.model_dump(mode="json"),
            "statistics": self.plan.training.stats.to_dict(),
            "dev_statistics": self.plan.dev.stats.to_dict(),
            "variants": [variant.to_dict() for variant in self.variants],
        }


class _Recorded(BaseModel):
    """What a `result.json` must say before it counts: strictly typed, extra keys
    ignored."""

    model_config = ConfigDict(strict=True, allow_inf_nan=False, frozen=True)

    status: Literal["ok"]
    fingerprint: str
    train_seconds: Annotated[float, Field(ge=0)]
    eval_seconds: Annotated[float, Field(ge=0)]
    checkpoint_sha256: str
    dev: DevResult


def complete_result(
    plan: AblationPlan, prepared: PreparedVariant
) -> VariantResult | None:
    """The result `prepared.result_path` records if it vouches for the variant as it
    is now, else None: the file does not parse as a result, says another status, holds
    another fingerprint, scores other dev sets than the plan's, or the checkpoint file
    is missing or has another digest. Nothing in the file is trusted before that."""
    try:
        recorded = _Recorded.model_validate_json(prepared.result_path.read_bytes())
        vouches = (
            recorded.fingerprint == prepared.fingerprint
            and scores_the_dev_sets(recorded.dev, plan.dev.sets)
            and sha256_of(prepared.checkpoint_path) == recorded.checkpoint_sha256
        )
    except OSError, ValueError:
        return None
    if not vouches:
        return None
    return VariantResult(
        planned=prepared.planned,
        reused=True,
        fingerprint=recorded.fingerprint,
        dev=recorded.dev,
        train_seconds=recorded.train_seconds,
        eval_seconds=recorded.eval_seconds,
        checkpoint_path=prepared.checkpoint_path,
        checkpoint_sha256=recorded.checkpoint_sha256,
    )


def scores_the_dev_sets(dev: DevResult, sets: Mapping[str, DevSet]) -> bool:
    """Whether `dev` scores exactly the directions of `sets`, as many lines in each as
    the set has."""
    return set(dev.directions) == set(sets) and all(
        scores.lines == len(sets[direction].src)
        for direction, scores in dev.directions.items()
    )
