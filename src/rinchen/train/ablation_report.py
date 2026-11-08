"""The files that summarise an ablation run: `results.tsv` (one row per variant),
`results.json` (everything) and `results.md` (the main columns and a summary over the
seeds), written from an `AblationResult`.

Numbers are fixed in the tables: scores and averages to two decimals, the source
collision rate (a share, mostly below 0.1) to four, seconds to one. The axes
(temperature, bt ratio) print as they do in a variant id, so `0.125` is not cut to
`0.12`. The raw floats are in `results.json`.
"""

from __future__ import annotations

import json
import statistics
from collections.abc import Sequence
from pathlib import Path
from typing import Final, NamedTuple

from rinchen.config.ablation import id_number
from rinchen.train.ablation_records import AblationResult, VariantResult
from rinchen.utils.part_file import replace_file

RESULTS_TSV: Final = "results.tsv"
RESULTS_JSON: Final = "results.json"
RESULTS_MD: Final = "results.md"

_AXIS_COLUMNS: Final = ("variant", "temperature", "bt_ratio", "tag_placement", "seed")
_SUMMARY_COLUMNS: Final = (
    "chrf_mean",
    "chrf_min",
    "tag_leaks",
    "train_seconds",
    "source_collision_rate",
)
_MAIN_TABLE_RULE: Final = (
    "| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |"
)
_SEED_SUMMARY_COLUMNS: Final = (
    "temperature",
    "bt_ratio",
    "tag_placement",
    "seeds",
    "chrf_mean",
    "chrf_min",
)
_SEED_SUMMARY_RULE: Final = "| ---: | ---: | --- | ---: | ---: | ---: |"


class ResultFiles(NamedTuple):
    """The three files of a run's results."""

    tsv: Path
    json: Path
    md: Path


def result_files(out_dir: Path) -> ResultFiles:
    """Where the run in `out_dir` keeps its result tables."""
    return ResultFiles(
        tsv=out_dir / RESULTS_TSV,
        json=out_dir / RESULTS_JSON,
        md=out_dir / RESULTS_MD,
    )


def write_results(result: AblationResult) -> ResultFiles:
    """Write the three result files into the output directory of `result.plan`, each
    whole or not at all, and return them. `results.tsv` has a header and one row per
    variant in plan order: the variant id and its axes, chrF and BLEU of each direction
    (directions sorted, chrF before BLEU), the mean and the minimum chrF, the tag
    leaks, the training seconds and the source collision rate. `results.md` has the
    main columns and the mean of `chrf_mean` and `chrf_min` over the seeds of each
    `(temperature, bt_ratio, tag_placement)`, with the number of seeds.

    Raises:
        OSError: a file cannot be written.
    """
    files = result_files(result.plan.out_dir)
    files.tsv.parent.mkdir(parents=True, exist_ok=True)
    for path, text in (
        (files.tsv, _tsv(result)),
        (files.json, _json(result)),
        (files.md, _markdown(result)),
    ):
        # A name that holds a lone surrogate must not fail the write at the end of a run.
        replace_file(path, text.encode("utf-8", errors="backslashreplace"))
    return files


def _score(value: float) -> str:
    return f"{value:.2f}"


def _seconds(value: float) -> str:
    return f"{value:.1f}"


def _rate(value: float) -> str:
    return f"{value:.4f}"


def _axis_cells(variant: VariantResult) -> list[str]:
    axes = variant.planned.variant
    return [
        axes.id,
        id_number(axes.temperature),
        id_number(axes.bt_ratio),
        axes.tag_placement,
        str(axes.seed),
    ]


def _summary_cells(variant: VariantResult) -> list[str]:
    return [
        _score(variant.dev.chrf_mean),
        _score(variant.dev.chrf_min),
        str(variant.dev.tag_leaks),
        _seconds(variant.train_seconds),
        _rate(variant.planned.source_collision_rate),
    ]


def _tsv(result: AblationResult) -> str:
    directions = sorted(result.plan.dev.sets)
    header = [
        *_AXIS_COLUMNS,
        *(f"{metric}_{name}" for name in directions for metric in ("chrf", "bleu")),
        *_SUMMARY_COLUMNS,
    ]
    rows = [header]
    for variant in result.variants:
        scores: list[str] = []
        for name in directions:
            direction = variant.dev.directions[name]
            scores += [_score(direction.chrf), _score(direction.bleu)]
        rows.append([*_axis_cells(variant), *scores, *_summary_cells(variant)])
    return "".join("\t".join(row) + "\n" for row in rows)


def _json(result: AblationResult) -> str:
    return json.dumps(result.to_dict(), indent=2) + "\n"


def _markdown_row(cells: Sequence[str]) -> str:
    return "| " + " | ".join(cells) + " |"


def _markdown(result: AblationResult) -> str:
    title = " ".join(result.plan.matrix.name.split())
    lines = [
        f"# Ablation results: {title}",
        "",
        _markdown_row([*_AXIS_COLUMNS, *_SUMMARY_COLUMNS]),
        _MAIN_TABLE_RULE,
        *(
            _markdown_row([*_axis_cells(variant), *_summary_cells(variant)])
            for variant in result.variants
        ),
        "",
        "## Summary over the seeds",
        "",
        _markdown_row(_SEED_SUMMARY_COLUMNS),
        _SEED_SUMMARY_RULE,
        *(_markdown_row(cells) for cells in _seed_summary(result.variants)),
        "",
    ]
    return "\n".join(lines)


def _seed_summary(variants: Sequence[VariantResult]) -> list[list[str]]:
    """One row per `(temperature, bt_ratio, tag_placement)` in the order it first
    occurs: the number of its variants (one per seed) and the mean of their
    `chrf_mean` and of their `chrf_min`."""
    groups: dict[tuple[float, float, str], list[VariantResult]] = {}
    for variant in variants:
        axes = variant.planned.variant
        key = (axes.temperature, axes.bt_ratio, axes.tag_placement)
        groups.setdefault(key, []).append(variant)
    return [
        [
            id_number(temperature),
            id_number(bt_ratio),
            placement,
            str(len(members)),
            _score(statistics.fmean(m.dev.chrf_mean for m in members)),
            _score(statistics.fmean(m.dev.chrf_min for m in members)),
        ]
        for (temperature, bt_ratio, placement), members in groups.items()
    ]
