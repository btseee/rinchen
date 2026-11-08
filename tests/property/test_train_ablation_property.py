"""Property tests for `rinchen.train.ablation` over random axes and random scores:
the plan has the product of the axes with weights that add up, a run with fake seams
trains in plan order and a second run reuses everything, and the tables summarise
whatever scores they are given."""

from __future__ import annotations

import json
import math
import statistics
import tempfile
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from rinchen.config.ablation import AblationMatrix
from rinchen.train.ablation_plan import AblationPlan, plan_ablation
from rinchen.train.ablation_records import AblationResult, VariantResult
from rinchen.train.ablation_report import write_results
from rinchen.train.variants import expand_variants
from tests.support.fake_ablation import (
    DIRECTIONS,
    AblationRun,
    dev_result,
    make_matrix,
    write_toy_corpus,
)
from tests.support.tiny_parallel import train_toy_spm

_TEMPERATURES = st.lists(
    st.sampled_from([0.1, 0.3, 0.5, 1.0, 2.0]), min_size=1, max_size=3, unique=True
)
_RATIOS = st.lists(
    st.sampled_from([0.0, 0.1, 0.5, 1.0, 3.0]), min_size=1, max_size=3, unique=True
)
_PLACEMENTS = st.lists(
    st.sampled_from(["src", "tgt"]), min_size=1, max_size=2, unique=True
)
_SEEDS = st.lists(st.integers(0, 10_000), min_size=1, max_size=3, unique=True)


@pytest.fixture(scope="module")
def toy_spm(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return train_toy_spm(tmp_path_factory.mktemp("property_spm"))


@pytest.fixture(scope="module")
def corpus(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return write_toy_corpus(tmp_path_factory.mktemp("property_corpus"))


def _matrix(
    temperatures: Sequence[float],
    ratios: Sequence[float],
    placements: Sequence[str],
    seeds: Sequence[int],
    weight_total: int = 100,
) -> AblationMatrix:
    return make_matrix(
        axes={
            "temperature": list(temperatures),
            "bt_ratio": list(ratios),
            "tag_placement": list(placements),
        },
        seeds=list(seeds),
        weight_total=weight_total,
    )


@settings(max_examples=40, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(
    temperatures=_TEMPERATURES,
    ratios=_RATIOS,
    placements=_PLACEMENTS,
    seeds=_SEEDS,
    weight_total=st.integers(8, 400),
)
def test_a_plan_has_one_variant_per_choice_and_weights_that_add_up(
    corpus: Path,
    toy_spm: Path,
    temperatures: list[float],
    ratios: list[float],
    placements: list[str],
    seeds: list[int],
    weight_total: int,
) -> None:
    matrix = _matrix(temperatures, ratios, placements, seeds, weight_total)

    plan = plan_ablation(matrix, [corpus], toy_spm, Path("never-created"))

    ids = [planned.variant.id for planned in plan.variants]
    assert len(ids) == len(temperatures) * len(ratios) * len(placements) * len(seeds)
    assert len(set(ids)) == len(ids)
    assert [planned.variant for planned in plan.variants] == expand_variants(matrix)
    for planned in plan.variants:
        weights = [corpus_plan.weight for corpus_plan in planned.corpora]
        assert sum(weights) == weight_total
        assert min(weights) >= 1
        assert math.fsum(c.probability for c in planned.corpora) == pytest.approx(1.0)
        assert all(c.upsampling > 0 for c in planned.corpora)
        # Weighted by the rows, the upsampling averages 1: every draw is somewhere.
        rows = sum(c.rows for c in planned.corpora)
        assert sum(c.rows * c.upsampling for c in planned.corpora) / rows == (
            pytest.approx(1.0)
        )


@settings(max_examples=12, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(
    temperatures=st.lists(
        st.sampled_from([0.5, 1.0]), min_size=1, max_size=2, unique=True
    ),
    ratios=st.lists(st.sampled_from([0.0, 0.5]), min_size=1, max_size=2, unique=True),
    placements=_PLACEMENTS,
)
def test_a_run_follows_the_plan_and_a_second_run_reuses_every_variant(
    corpus: Path,
    toy_spm: Path,
    temperatures: list[float],
    ratios: list[float],
    placements: list[str],
) -> None:
    matrix = _matrix(temperatures, ratios, placements, [1])
    with tempfile.TemporaryDirectory() as directory:
        plan = plan_ablation(matrix, [corpus], toy_spm, Path(directory) / "out")
        expected = [planned.variant.id for planned in plan.variants]

        first = AblationRun(plan)
        second = AblationRun(plan)

        assert first.train.variant_ids == expected
        assert first.evaluate.variant_ids == expected
        assert [r.planned.variant.id for r in first.result.variants] == expected
        assert second.train.calls == [] and second.evaluate.calls == []
        assert all(r.reused for r in second.result.variants)
        for variant_id in expected:
            assert first.result_path(variant_id).is_file()


_SCORE = st.floats(min_value=0.0, max_value=100.0, allow_nan=False)
_SCORES = st.lists(
    st.tuples(
        st.lists(_SCORE, min_size=len(DIRECTIONS), max_size=len(DIRECTIONS)),
        st.lists(_SCORE, min_size=len(DIRECTIONS), max_size=len(DIRECTIONS)),
        st.integers(0, 50),
        st.floats(min_value=0.0, max_value=1e5, allow_nan=False),
    ),
    min_size=1,
    max_size=24,
)


@pytest.fixture(scope="module")
def big_plan(corpus: Path, toy_spm: Path) -> AblationPlan:
    """24 variants: two temperatures, two bt ratios, two placements, three seeds, each
    axis in descending order."""
    # Not in sorted order, so that the order of the plan is not also the sorted one.
    matrix = _matrix([1.0, 0.5], [0.5, 0.0], ["tgt", "src"], [3, 1, 2])
    return plan_ablation(matrix, [corpus], toy_spm, Path("never-created"))


def _results_for(
    plan: AblationPlan,
    scores: Sequence[tuple[list[float], list[float], int, float]],
) -> list[VariantResult]:
    results = []
    for planned, (chrfs, bleus, leaks, seconds) in zip(
        plan.variants, scores, strict=False
    ):
        results.append(
            VariantResult(
                planned=planned,
                outcome="trained",
                fingerprint="f" * 64,
                dev=dev_result(
                    dict(zip(DIRECTIONS, chrfs, strict=True)),
                    dict(zip(DIRECTIONS, bleus, strict=True)),
                    tag_leaks=leaks,
                ),
                train_seconds=seconds,
                eval_seconds=1.0,
                checkpoint_path=Path("ckpt") / planned.variant.id,
                checkpoint_sha256="a" * 64,
            )
        )
    return results


@settings(max_examples=40, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(scores=_SCORES)
def test_the_tables_summarise_any_scores_row_by_row_and_by_setting(
    big_plan: AblationPlan,
    scores: list[tuple[list[float], list[float], int, float]],
) -> None:
    results = _results_for(big_plan, scores)
    with tempfile.TemporaryDirectory() as directory:
        plan = replace(big_plan, out_dir=Path(directory))
        files = write_results(AblationResult(plan, 2, tuple(results)))
        tsv = files.tsv.read_text(encoding="utf-8").split("\n")
        md = files.md.read_text(encoding="utf-8")
        document = json.loads(files.json.read_text(encoding="utf-8"))

    # One TSV row per variant in plan order, every row as wide as the header.
    assert tsv[-1] == ""
    header, *rows = tsv[:-1]
    assert len(rows) == len(results)
    assert {len(row.split("\t")) for row in [header, *rows]} == {18}
    names = header.split("\t")
    for row, result in zip(rows, results, strict=True):
        cells = dict(zip(names, row.split("\t"), strict=True))
        assert cells["variant"] == result.planned.variant.id
        assert abs(float(cells["chrf_mean"]) - result.dev.chrf_mean) <= 0.0051
        assert abs(float(cells["chrf_min"]) - result.dev.chrf_min) <= 0.0051
        assert int(cells["tag_leaks"]) == result.dev.tag_leaks
    # The JSON holds the raw floats.
    assert [v["dev"]["chrf_mean"] for v in document["variants"]] == [
        r.dev.chrf_mean for r in results
    ]
    # The summary has one row per setting with the number of its seeds and the mean
    # over them.
    groups: dict[tuple[float, float, str], list[VariantResult]] = defaultdict(list)
    for result in results:
        variant = result.planned.variant
        groups[(variant.temperature, variant.bt_ratio, variant.tag_placement)].append(
            result
        )
    summary = md.split("## Summary over the seeds")[1].strip().split("\n")[2:]
    assert len(summary) == len(groups)
    for line, members in zip(summary, groups.values(), strict=True):
        _, _, _, seeds, chrf_mean, chrf_min = (
            cell.strip() for cell in line.strip("|").split("|")
        )
        assert int(seeds) == len(members)
        mean_of_means = statistics.fmean(m.dev.chrf_mean for m in members)
        mean_of_mins = statistics.fmean(m.dev.chrf_min for m in members)
        assert abs(float(chrf_mean) - mean_of_means) <= 0.0051
        assert abs(float(chrf_min) - mean_of_mins) <= 0.0051
    assert sum(len(members) for members in groups.values()) == len(results)
