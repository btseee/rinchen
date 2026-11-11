"""`rinchen.tiny.data_stages.run_filter`: the filter stage of the tiny pipeline (D15, spec P6, brief
G3.3).

The stage filters the pair corpus ONCE (one row per pair: filtering both directions makes each row a
near-duplicate of its reverse), then expands the kept rows of the splits `train` and `dev` to both
directions. The planted stand-ins of `--qe-fake` stand for LaBSE and the QE scorer; the ten planted
bad pairs of `noise.tsv` must each get the reason they expect from the stages that are active, and
the stand-ins may flag at most a small share of the 760 authentic rows. Everything below runs over
the committed fixtures with no model and no network; the numbers are those measured in the spike
(`p6`): the reasons `script_tgt` 2, `len_ratio_char` 3, `numbers` 2, `duplicate` 4 and `labse` 3, and
756 of the 770 rows kept.
"""

from __future__ import annotations

import dataclasses
import json
from collections import Counter
from collections.abc import Iterator
from pathlib import Path
from typing import Any, Final

import polars as pl
import pytest

from rinchen.config.tiny import FilterSection, TinyConfig
from rinchen.corpus.readers import scan_corpus
from rinchen.filter.pipeline import PipelineResult
from rinchen.tiny import data_stages
from rinchen.tiny.context import StageContext, StageResult
from rinchen.tiny.corpus import NOISE_CORPUS, expand_directions
from rinchen.tiny.data_stages import run_corpus, run_filter
from rinchen.tiny.fixtures import Fixtures
from rinchen.tiny.states import (
    FAIL_INPUT,
    SKIP_NO_LABSE,
    SKIP_NO_LID,
    SKIP_NO_QE,
    SKIP_TOKEN_RATIO,
    PartRecord,
    fail_check,
    is_skip_reason,
)
from tests.support.no_network import no_network  # noqa: F401 - the fixture of the mark
from tests.support.tiny_stage_inputs import (
    make_context,
    offline_in,
    shipped_config,
    shipped_fixtures,
)

pytestmark = pytest.mark.usefixtures("no_network")

PART_ORDER: Final = (
    "script_ratio",
    "numbers",
    "length",
    "lid",
    "minhash",
    "labse",
    "qe",
)
SPIKE_REASONS: Final = {
    "script_tgt": 2,
    "len_ratio_char": 3,
    "numbers": 2,
    "duplicate": 4,
    "labse": 3,
}
PLANTED_CHECK: Final = fail_check("planted_noise")


@pytest.fixture(autouse=True)
def work_elsewhere(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Run from a directory that holds nothing: a stage must not read the repository by itself."""
    here = tmp_path / "cwd"
    here.mkdir()
    monkeypatch.chdir(here)
    yield


def run_filter_in(
    out_dir: Path,
    *,
    qe_fake: bool = True,
    config: TinyConfig | None = None,
    fixtures: Fixtures | None = None,
) -> tuple[StageContext, StageResult]:
    ctx = make_context(out_dir, config=config, fixtures=fixtures, qe_fake=qe_fake)
    assert run_corpus(ctx).record.state == "ok"
    return ctx, run_filter(ctx)


def with_filter(**changes: Any) -> TinyConfig:
    config = shipped_config()
    section: FilterSection = config.filter.model_copy(update=changes)
    return config.model_copy(update={"filter": section})


def scored_of(out_dir: Path) -> pl.DataFrame:
    return scan_corpus(out_dir / "03_filter" / "scored.parquet").collect()


def noise_reasons(scored: pl.DataFrame) -> list[str | None]:
    return scored.filter(pl.col("corpus") == NOISE_CORPUS)["filter_reason"].to_list()


def reason_counts(frame: pl.DataFrame) -> dict[str, int]:
    found = frame["filter_reason"].drop_nulls().to_list()
    return dict(Counter(found))


@pytest.fixture(scope="module")
def with_fake(
    tmp_path_factory: pytest.TempPathFactory,
) -> tuple[Path, StageContext, StageResult]:
    out_dir = tmp_path_factory.mktemp("with-fake") / "out"
    with offline_in(out_dir.parent / "cwd"):
        ctx, result = run_filter_in(out_dir)
    return out_dir, ctx, result


@pytest.fixture(scope="module")
def without_fake(
    tmp_path_factory: pytest.TempPathFactory,
) -> tuple[Path, StageContext, StageResult]:
    out_dir = tmp_path_factory.mktemp("without-fake") / "out"
    with offline_in(out_dir.parent / "cwd"):
        ctx, result = run_filter_in(out_dir, qe_fake=False)
    return out_dir, ctx, result


# ---- with the stand-ins ------------------------------------------------------------------------


def test_the_filter_reasons_are_those_measured_in_the_spike(
    with_fake: tuple[Path, StageContext, StageResult],
) -> None:
    out_dir, _, result = with_fake
    scored = scored_of(out_dir)

    assert result.record.id == "filter"
    assert (result.record.state, result.record.reason) == ("ok", None)
    assert scored.height == 770
    assert reason_counts(scored) == SPIKE_REASONS
    assert result.facts["reasons"] == SPIKE_REASONS
    assert result.facts["rows"] == 770
    assert result.facts["kept"] == 756 == scored["filter_reason"].null_count()


def test_every_planted_row_has_the_reason_it_expects(
    with_fake: tuple[Path, StageContext, StageResult],
) -> None:
    out_dir, ctx, result = with_fake

    assert noise_reasons(scored_of(out_dir)) == [r.expect for r in ctx.fixtures.noise]
    assert result.facts["planted"] == {
        "total": 10,
        "detectable": 10,
        "matched": 10,
        "expected": {
            "script_tgt": 2,
            "len_ratio_char": 2,
            "numbers": 2,
            "duplicate": 2,
            "labse": 2,
        },
        "found": {
            "script_tgt": 2,
            "len_ratio_char": 2,
            "numbers": 2,
            "duplicate": 2,
            "labse": 2,
        },
    }


def test_four_authentic_rows_are_flagged_two_duplicates_one_pair_and_one_length(
    with_fake: tuple[Path, StageContext, StageResult],
) -> None:
    out_dir, _, result = with_fake
    authentic = scored_of(out_dir).filter(pl.col("corpus") != NOISE_CORPUS)

    assert authentic.height == 760 == result.facts["authentic_rows"]
    assert reason_counts(authentic) == {"duplicate": 2, "labse": 1, "len_ratio_char": 1}
    assert result.facts["authentic_flagged"] == 4
    assert result.facts["authentic_flagged_share"] == pytest.approx(4 / 760)


def test_the_stage_lists_its_seven_parts_the_length_check_noted_and_the_lid_skipped(
    with_fake: tuple[Path, StageContext, StageResult],
) -> None:
    _, _, result = with_fake

    assert [(p.id, p.state, p.reason) for p in result.record.parts] == [
        ("script_ratio", "ok", None),
        ("numbers", "ok", None),
        ("length", "ok", None),
        ("lid", "skipped", SKIP_NO_LID),
        ("minhash", "ok", None),
        ("labse", "ok", None),
        ("qe", "ok", None),
    ]
    assert tuple(p.id for p in result.record.parts) == PART_ORDER
    assert result.record.detail["token_ratio_check"] == SKIP_TOKEN_RATIO
    assert is_skip_reason(SKIP_TOKEN_RATIO)


def test_the_files_are_the_scored_the_kept_and_the_training_corpus_and_the_counts(
    with_fake: tuple[Path, StageContext, StageResult],
) -> None:
    out_dir, _, _ = with_fake

    assert sorted(p.name for p in out_dir.iterdir()) == ["02_corpus", "03_filter"]
    assert sorted(p.name for p in (out_dir / "03_filter").iterdir()) == [
        "counts.json",
        "kept.parquet",
        "scored.parquet",
        "train.parquet",
    ]


def test_the_kept_rows_are_the_unflagged_ones_of_the_scored_corpus(
    with_fake: tuple[Path, StageContext, StageResult],
) -> None:
    out_dir, _, _ = with_fake
    scored = scored_of(out_dir)
    kept = scan_corpus(out_dir / "03_filter" / "kept.parquet").collect()

    assert kept.height == 756
    assert kept.equals(scored.filter(pl.col("filter_reason").is_null()))


def test_the_filter_scores_every_row_with_the_stand_ins(
    with_fake: tuple[Path, StageContext, StageResult],
) -> None:
    out_dir, _, _ = with_fake
    scores = scored_of(out_dir)["scores"].struct.unnest()

    assert scores["labse"].null_count() == 0
    assert scores["metricx_qe"].null_count() == 0


def test_the_training_corpus_is_the_expansion_of_the_kept_rows(
    with_fake: tuple[Path, StageContext, StageResult],
) -> None:
    out_dir, _, result = with_fake
    kept = scan_corpus(out_dir / "03_filter" / "kept.parquet").collect()
    train = scan_corpus(out_dir / "03_filter" / "train.parquet").collect()
    created = train["created"].unique().to_list()

    assert train.equals(expand_directions(kept, created[0]))
    assert len(created) == 1
    assert created[0].isoformat() == "2026-11-11T00:00:00+00:00"
    assert set(train["split"]) == {"train", "dev"}
    assert NOISE_CORPUS not in set(train["corpus"])
    assert train.height == result.facts["train_rows"]
    assert train.height < 1320


def test_a_flagged_authentic_row_is_not_trained_on_in_either_direction(
    with_fake: tuple[Path, StageContext, StageResult],
) -> None:
    out_dir, _, _ = with_fake
    scored = scored_of(out_dir)
    train = scan_corpus(out_dir / "03_filter" / "train.parquet").collect()
    trained = {
        (r["corpus"], r["src"], r["tgt"], r["split"])
        for r in train.iter_rows(named=True)
    }
    flagged = scored.filter(
        pl.col("filter_reason").is_not_null()
        & pl.col("split").is_in(["train", "dev"])
        & (pl.col("corpus") != NOISE_CORPUS)
    )

    assert flagged.height >= 3
    for row in flagged.iter_rows(named=True):
        assert (row["corpus"], row["src"], row["tgt"], row["split"]) not in trained
        assert (row["corpus"], row["tgt"], row["src"], row["split"]) not in trained


def test_the_counts_file_is_the_facts_and_quotes_no_text(
    with_fake: tuple[Path, StageContext, StageResult],
) -> None:
    out_dir, ctx, result = with_fake
    raw = (out_dir / "03_filter" / "counts.json").read_bytes()

    assert raw.isascii()
    assert (
        raw == (json.dumps(json.loads(raw), sort_keys=True, indent=2) + "\n").encode()
    )
    assert json.loads(raw) == json.loads(json.dumps(result.facts))
    texts = {r.src for r in ctx.fixtures.noise} | {r.tgt for r in ctx.fixtures.noise}
    assert not any(text in raw.decode() or text in repr(result) for text in texts)


# ---- without the stand-ins ---------------------------------------------------------------------


def test_without_the_stand_ins_the_stage_is_ok_and_has_no_labse_reason(
    without_fake: tuple[Path, StageContext, StageResult],
) -> None:
    out_dir, _, result = without_fake
    scored = scored_of(out_dir)

    assert (result.record.state, result.record.reason) == ("ok", None)
    assert reason_counts(scored) == {
        "script_tgt": 2,
        "len_ratio_char": 3,
        "numbers": 2,
        "duplicate": 4,
    }
    assert result.facts["kept"] == 759
    assert scored["scores"].struct.field("labse").null_count() == 770


def test_without_the_stand_ins_the_two_misaligned_rows_pass_and_are_held_out(
    without_fake: tuple[Path, StageContext, StageResult],
) -> None:
    out_dir, ctx, result = without_fake
    scored = scored_of(out_dir)
    noise = scored.filter(pl.col("corpus") == NOISE_CORPUS)
    expected = [r.expect for r in ctx.fixtures.noise]

    labse_rows = [i for i, expect in enumerate(expected) if expect == "labse"]
    assert len(labse_rows) == 2
    for index in labse_rows:
        assert noise["filter_reason"][index] is None
        assert noise["split"][index] == "holdout"
    assert noise["filter_reason"].to_list() == [
        None if e == "labse" else e for e in expected
    ]
    planted = result.facts["planted"]
    assert (planted["total"], planted["detectable"], planted["matched"]) == (10, 8, 8)
    assert planted["found"]["none"] == 2
    assert result.facts["authentic_flagged"] == 3


def test_without_the_stand_ins_those_rows_are_not_trained_on(
    without_fake: tuple[Path, StageContext, StageResult],
) -> None:
    # The planted rows are made of sentences of authentic rows, so a sentence of one may be
    # trained on; the PAIR of a misaligned row, and the noise corpus, are not.
    out_dir, ctx, _ = without_fake
    train = scan_corpus(out_dir / "03_filter" / "train.parquet").collect()
    trained = set(zip(train["src"], train["tgt"], strict=True))
    misaligned = {(r.src, r.tgt) for r in ctx.fixtures.noise if r.expect == "labse"}

    assert len(misaligned) == 2
    assert not misaligned & trained
    assert NOISE_CORPUS not in set(train["corpus"])
    assert sorted(set(train["split"])) == ["dev", "train"]


def test_without_the_stand_ins_the_labse_and_qe_parts_are_skipped_with_their_reasons(
    without_fake: tuple[Path, StageContext, StageResult],
) -> None:
    _, _, result = without_fake

    assert [(p.id, p.state, p.reason) for p in result.record.parts] == [
        ("script_ratio", "ok", None),
        ("numbers", "ok", None),
        ("length", "ok", None),
        ("lid", "skipped", SKIP_NO_LID),
        ("minhash", "ok", None),
        ("labse", "skipped", SKIP_NO_LABSE),
        ("qe", "skipped", SKIP_NO_QE),
    ]


# ---- the checks of the stage -------------------------------------------------------------------


def altered_to_be_authentic(fixtures: Fixtures, index: int) -> Fixtures:
    """The fixtures with one planted row replaced by a pair that is authentic: the English line and
    the Khalkha line of a monolingual row, which belongs to no other file."""
    authentic = fixtures.mono[0]
    noise = list(fixtures.noise)
    noise[index] = dataclasses.replace(
        noise[index], src=authentic.eng, tgt=authentic.khk
    )
    return dataclasses.replace(fixtures, noise=tuple(noise))


@pytest.mark.parametrize("index", range(10))
def test_a_planted_row_that_is_not_flagged_with_its_reason_fails_the_stage(
    tmp_path: Path, index: int
) -> None:
    fixtures = altered_to_be_authentic(shipped_fixtures(), index)

    _, result = run_filter_in(tmp_path / "out", fixtures=fixtures)

    assert result.record.state == "fail"
    assert result.record.reason == PLANTED_CHECK
    planted = result.facts["planted"]
    assert planted["matched"] == 9 < planted["detectable"] == 10
    assert (tmp_path / "out" / "03_filter" / "counts.json").is_file()


def test_a_planted_row_that_got_another_reason_than_its_own_fails_the_stage(
    tmp_path: Path,
) -> None:
    fixtures = shipped_fixtures()
    noise = list(fixtures.noise)
    noise[0] = dataclasses.replace(noise[0], expect="numbers")
    changed = dataclasses.replace(fixtures, noise=tuple(noise))

    _, result = run_filter_in(tmp_path / "out", fixtures=changed)

    assert (result.record.state, result.record.reason) == ("fail", PLANTED_CHECK)


def test_without_the_stand_ins_a_missed_row_that_cannot_be_detected_is_not_asked(
    tmp_path: Path,
) -> None:
    fixtures = shipped_fixtures()
    index = next(i for i, r in enumerate(fixtures.noise) if r.expect == "labse")

    _, result = run_filter_in(
        tmp_path / "out",
        qe_fake=False,
        fixtures=altered_to_be_authentic(fixtures, index),
    )

    assert result.record.state == "ok"


def test_without_the_stand_ins_a_missed_row_the_cpu_stages_detect_still_fails(
    tmp_path: Path,
) -> None:
    fixtures = shipped_fixtures()
    index = next(i for i, r in enumerate(fixtures.noise) if r.expect == "numbers")

    _, result = run_filter_in(
        tmp_path / "out",
        qe_fake=False,
        fixtures=altered_to_be_authentic(fixtures, index),
    )

    assert (result.record.state, result.record.reason) == ("fail", PLANTED_CHECK)
    assert result.facts["planted"]["matched"] == 7


def test_a_share_of_authentic_rows_above_the_bound_fails_the_stage(
    tmp_path: Path,
) -> None:
    config = with_filter(max_authentic_flagged_share=0.001)

    _, result = run_filter_in(tmp_path / "out", config=config)

    assert (result.record.state, result.record.reason) == ("fail", PLANTED_CHECK)
    assert result.facts["planted"]["matched"] == 10
    assert result.facts["authentic_flagged"] == 4


def test_the_share_bound_is_inclusive(tmp_path: Path) -> None:
    just_above = with_filter(max_authentic_flagged_share=0.0053)
    just_below = with_filter(max_authentic_flagged_share=0.0052)

    _, passing = run_filter_in(tmp_path / "pass", config=just_above)
    _, failing = run_filter_in(tmp_path / "fail", config=just_below)

    assert 0.0052 < 4 / 760 < 0.0053
    assert passing.record.state == "ok"
    assert failing.record.state == "fail"


def test_the_bounds_of_the_profile_are_the_ones_the_filter_uses(tmp_path: Path) -> None:
    loose = with_filter(labse_min_cosine=0.5)
    strict = with_filter(qe_max_metricx=0.5)
    exact_only = with_filter(minhash_threshold=0.95)
    narrow = with_filter(char_ratio_min=0.6, char_ratio_max=1.5)

    _, relaxed = run_filter_in(tmp_path / "a", config=loose)
    _, tight = run_filter_in(tmp_path / "b", config=strict)
    _, near = run_filter_in(tmp_path / "c", config=exact_only)
    _, short = run_filter_in(tmp_path / "d", config=narrow)

    assert relaxed.facts["reasons"]["labse"] == 2
    assert tight.facts["reasons"].get("qe", 0) > 0
    assert near.facts["reasons"]["duplicate"] == 2
    assert short.facts["reasons"]["len_ratio_char"] > 3


def test_a_stage_the_pipeline_skipped_is_a_failed_part(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    real = data_stages.run_pipeline

    def skipping(*args: Any, **kwargs: Any) -> PipelineResult:
        result = real(*args, **kwargs)
        status = {**result.stage_status, "qe": "skipped: no QE scorer configured"}
        return dataclasses.replace(result, stage_status=status)

    monkeypatch.setattr(data_stages, "run_pipeline", skipping)

    _, result = run_filter_in(tmp_path / "out")

    assert result.record.state == "fail"
    assert result.record.reason == fail_check("stage_skipped")
    assert result.record.parts[-1] == PartRecord(
        "qe", "fail", fail_check("stage_skipped")
    )


def test_a_corpus_that_is_missing_fails_the_stage_and_writes_nothing(
    tmp_path: Path,
) -> None:
    out_dir = tmp_path / "out"
    ctx = make_context(out_dir)

    result = run_filter(ctx)

    assert (result.record.state, result.record.reason) == ("fail", FAIL_INPUT)
    assert not out_dir.exists()


def test_a_corpus_that_is_not_a_corpus_fails_the_stage(tmp_path: Path) -> None:
    out_dir = tmp_path / "out"
    ctx = make_context(out_dir)
    (out_dir / "02_corpus").mkdir(parents=True)
    pl.DataFrame({"src": ["a"]}).write_parquet(out_dir / "02_corpus" / "corpus.parquet")

    result = run_filter(ctx)

    assert (result.record.state, result.record.reason) == ("fail", FAIL_INPUT)
    assert not (out_dir / "03_filter").exists()


def test_the_stage_gives_the_same_files_in_another_directory(tmp_path: Path) -> None:
    run_filter_in(tmp_path / "first")
    run_filter_in(tmp_path / "second")

    for name in ("scored.parquet", "kept.parquet", "train.parquet", "counts.json"):
        assert (tmp_path / "first" / "03_filter" / name).read_bytes() == (
            tmp_path / "second" / "03_filter" / name
        ).read_bytes(), name
