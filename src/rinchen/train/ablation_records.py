"""What a variant and a run leave on record: the `VariantResult` that `result.json`
holds, the `AblationResult` the tables and the manifest are written from, the
`VariantProgress` events, and the checks that decide whether a `result.json` or a
`train.json` on disk may be trusted.

A variant is complete when its `result.json` parses, says `ok`, holds the fingerprint
of its inputs as they are now, scores the plan's dev sets, and the checkpoint file it
names still has the recorded digest: only then is it reused (`complete_result`).

`train.json` is written once a training has finished, before the evaluation starts. A
variant that is not complete but whose `train.json` parses, holds the current
fingerprint and names the digest the checkpoint file has now needs only its
evaluation again (`trained_checkpoint`); else it trains again.
"""

from __future__ import annotations

import json
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
from rinchen.utils.part_file import replace_file

_STATUS_OK: Final = "ok"
_STATUS_TRAINED: Final = "trained"


Outcome = Literal["trained", "evaluated", "reused"]


@dataclass(frozen=True, slots=True)
class VariantProgress:
    """A variant is done (its `result.json` is on disk): `trained` and evaluated just
    now, `evaluated` just now from a checkpoint an earlier run trained, or `reused`
    whole from an earlier run. `index` counts from 1. The seconds are those of this
    call: no training seconds unless it trained, no evaluation seconds if it reused."""

    variant_id: str
    index: int
    total: int
    outcome: Outcome
    train_seconds: float
    eval_seconds: float


@dataclass(frozen=True, slots=True)
class VariantResult:
    """A complete variant: its plan, the dev scores, the seconds training and
    evaluation took (those of the runs that did them, also when the variant was
    `evaluated` or `reused` now), and its checkpoint file with the SHA-256 of that
    file. `fingerprint` is that of the inputs it was trained from. `outcome` says what
    this call did for it, as `VariantProgress.outcome` does."""

    planned: PlannedVariant
    outcome: Outcome
    fingerprint: str
    dev: DevResult
    train_seconds: float
    eval_seconds: float
    checkpoint_path: Path
    checkpoint_sha256: str

    @property
    def reused(self) -> bool:
        return self.outcome == "reused"

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready: what `result.json` holds. `outcome` is left out: it says what a
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
        outcome="reused",
        fingerprint=recorded.fingerprint,
        dev=recorded.dev,
        train_seconds=recorded.train_seconds,
        eval_seconds=recorded.eval_seconds,
        checkpoint_path=prepared.checkpoint_path,
        checkpoint_sha256=recorded.checkpoint_sha256,
    )


class TrainRecord(BaseModel):
    """What a `train.json` must say before it counts: a finished training of the
    variant as it is now. Strictly typed, extra keys ignored."""

    model_config = ConfigDict(strict=True, allow_inf_nan=False, frozen=True)

    status: Literal["trained"]
    fingerprint: str
    checkpoint_sha256: str
    train_seconds: Annotated[float, Field(ge=0)]


def write_train_record(
    prepared: PreparedVariant, *, checkpoint_sha256: str, train_seconds: float
) -> None:
    """Record that `prepared` has trained: its fingerprint, the SHA-256 of the
    checkpoint file and the seconds the training took, in its `train.json`, whole or
    not at all."""
    document = {
        "status": _STATUS_TRAINED,
        "fingerprint": prepared.fingerprint,
        "checkpoint_sha256": checkpoint_sha256,
        "train_seconds": train_seconds,
    }
    replace_file(
        prepared.train_record_path, (json.dumps(document, indent=2) + "\n").encode()
    )


def trained_checkpoint(prepared: PreparedVariant) -> TrainRecord | None:
    """The record in `prepared.train_record_path` if it vouches for the checkpoint as
    it is now, else None: the file does not parse as a record, says another status,
    holds another fingerprint, or the checkpoint file is missing or has another
    digest. Nothing in the file is trusted before that."""
    try:
        recorded = TrainRecord.model_validate_json(
            prepared.train_record_path.read_bytes()
        )
        vouches = (
            recorded.fingerprint == prepared.fingerprint
            and sha256_of(prepared.checkpoint_path) == recorded.checkpoint_sha256
        )
    except OSError, ValueError:
        return None
    return recorded if vouches else None


def scores_the_dev_sets(dev: DevResult, sets: Mapping[str, DevSet]) -> bool:
    """Whether `dev` scores exactly the directions of `sets`, as many lines in each as
    the set has."""
    return set(dev.directions) == set(sets) and all(
        scores.lines == len(sets[direction].src)
        for direction, scores in dev.directions.items()
    )
