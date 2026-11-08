"""The variants of an ablation matrix: one per choice of temperature, bt ratio, tag
placement and seed."""

from __future__ import annotations

import itertools
from dataclasses import dataclass

from rinchen.config.ablation import AblationMatrix, TagPlacement, id_number


@dataclass(frozen=True)
class Variant:
    """One run of the ablation. `id` names its axes (`t0.5_bt0.5_src_s1234`) and is
    unique within a matrix."""

    id: str
    temperature: float
    bt_ratio: float
    tag_placement: TagPlacement
    seed: int


def expand_variants(matrix: AblationMatrix) -> list[Variant]:
    """The product of the axes and the seeds, with the seed changing fastest, then
    the tag placement, the bt ratio and the temperature last. The matrix guarantees
    distinct values per axis, so the ids are unique, and at most `max_variants`."""
    return [
        Variant(
            id=(
                f"t{id_number(temperature)}_bt{id_number(bt_ratio)}"
                f"_{tag_placement}_s{seed}"
            ),
            temperature=temperature,
            bt_ratio=bt_ratio,
            tag_placement=tag_placement,
            seed=seed,
        )
        for temperature, bt_ratio, tag_placement, seed in itertools.product(
            matrix.axes.temperature,
            matrix.axes.bt_ratio,
            matrix.axes.tag_placement,
            matrix.seeds,
        )
    ]
