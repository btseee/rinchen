"""Tests for `rinchen.train.ablation_report`: `results.tsv`, `results.json` and
`results.md` of a run, and the `result.json` of each variant that `results.json`
repeats (D7 spec section 5.8).

The run is four variants (tag placement `src` and `tgt`, seeds 1 and 2, at temperature
0.5 and bt ratio 0.5) with scripted dev scores, durations and a corpus that gives a
source collision rate of 0.04 (200 of 5,000 training rows share a source with another
target language), so every number in a table is worked out by hand:

    variant            chrF per direction (en-mn mn-en mn-ru ru-mn)  mean    min   leaks  train
    src s1             90.126  80  70  60                           75.0315  60     0    12.54
    src s2             92      84  72  64                           78       64     1    13.76
    tgt s1             80      70  60  50                           65       50     2    20.26
    tgt s2             82      74  64  52                           68       52     0    21.04

    src, over the seeds: chrf_mean (75.0315 + 78) / 2 = 76.51575, chrf_min (60 + 64) / 2 = 62
    tgt, over the seeds: chrf_mean (65 + 68) / 2 = 66.5,            chrf_min (50 + 52) / 2 = 51
"""

from __future__ import annotations

import dataclasses
import json
import math
from pathlib import Path

import pytest

from rinchen.train.ablation_plan import plan_ablation
from rinchen.train.ablation_records import AblationResult
from rinchen.train.ablation_report import result_files, write_results
from rinchen.train.eole_run import CHECKPOINT_FILE
from tests.support.corpus_rows import EN_MN, MN_EN, MN_RU, RU_MN
from tests.support.fake_ablation import (
    BLEU_SIGNATURE,
    CHRF_SIGNATURE,
    AblationRun,
    FakeClock,
    FakeEvaluate,
    FakeTrain,
    colliding_rows,
    dev_result,
    make_matrix,
    one_variant_axes,
    write_toy_corpus,
)
from tests.support.tiny_parallel import train_toy_spm

SRC_1 = "t0.5_bt0.5_src_s1"
SRC_2 = "t0.5_bt0.5_src_s2"
TGT_1 = "t0.5_bt0.5_tgt_s1"
TGT_2 = "t0.5_bt0.5_tgt_s2"

SCORES = {
    SRC_1: dev_result(
        {EN_MN: 90.126, MN_EN: 80.0, MN_RU: 70.0, RU_MN: 60.0},
        {EN_MN: 45.0, MN_EN: 40.0, MN_RU: 35.0, RU_MN: 30.0},
        tag_leaks=0,
    ),
    SRC_2: dev_result(
        {EN_MN: 92.0, MN_EN: 84.0, MN_RU: 72.0, RU_MN: 64.0},
        {EN_MN: 46.0, MN_EN: 42.0, MN_RU: 36.0, RU_MN: 32.0},
        tag_leaks=1,
    ),
    TGT_1: dev_result(
        {EN_MN: 80.0, MN_EN: 70.0, MN_RU: 60.0, RU_MN: 50.0},
        {EN_MN: 40.0, MN_EN: 35.0, MN_RU: 30.0, RU_MN: 25.0},
        tag_leaks=2,
    ),
    TGT_2: dev_result(
        {EN_MN: 82.0, MN_EN: 74.0, MN_RU: 64.0, RU_MN: 52.0},
        {EN_MN: 41.0, MN_EN: 37.0, MN_RU: 32.0, RU_MN: 26.0},
        tag_leaks=0,
    ),
}
TRAIN_SECONDS = {SRC_1: 12.54, SRC_2: 13.76, TGT_1: 20.26, TGT_2: 21.04}
EVAL_SECONDS = dict.fromkeys(TRAIN_SECONDS, 3.0)

HEADER = "\t".join(
    [
        "variant",
        "temperature",
        "bt_ratio",
        "tag_placement",
        "seed",
        f"chrf_{EN_MN}",
        f"bleu_{EN_MN}",
        f"chrf_{MN_EN}",
        f"bleu_{MN_EN}",
        f"chrf_{MN_RU}",
        f"bleu_{MN_RU}",
        f"chrf_{RU_MN}",
        f"bleu_{RU_MN}",
        "chrf_mean",
        "chrf_min",
        "tag_leaks",
        "train_seconds",
        "source_collision_rate",
    ]
)


@pytest.fixture(scope="module")
def toy_spm(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return train_toy_spm(tmp_path_factory.mktemp("results_spm"))


@pytest.fixture(scope="module")
def scripted(toy_spm: Path, tmp_path_factory: pytest.TempPathFactory) -> AblationRun:
    """The four-variant run with the scripted scores and durations."""
    root = tmp_path_factory.mktemp("results_run")
    corpus = write_toy_corpus(root / "in", extra_rows=colliding_rows(100))
    plan = plan_ablation(make_matrix(seeds=[1, 2]), [corpus], toy_spm, root / "out")
    clock = FakeClock()
    return AblationRun(
        plan,
        train=FakeTrain(clock=clock, seconds=TRAIN_SECONDS),
        evaluate=FakeEvaluate(SCORES, clock=clock, seconds=EVAL_SECONDS),
        clock=clock,
    )


def _read(run: AblationRun, name: str) -> str:
    return (run.plan.out_dir / name).read_text(encoding="utf-8")


# --- results.tsv ---------------------------------------------------------------------


def test_the_table_has_the_header_and_one_row_per_variant_in_plan_order(
    scripted: AblationRun,
) -> None:
    lines = _read(scripted, "results.tsv").split("\n")

    assert lines[0] == HEADER
    assert [line.split("\t")[0] for line in lines[1:5]] == [SRC_1, SRC_2, TGT_1, TGT_2]
    assert lines[5] == ""
    assert len(lines) == 6


def test_the_rows_hold_the_axes_the_scores_to_two_decimals_and_the_seconds_to_one(
    scripted: AblationRun,
) -> None:
    rows = _read(scripted, "results.tsv").split("\n")[1:5]

    assert rows == [
        "\t".join(
            ["t0.5_bt0.5_src_s1", "0.5", "0.5", "src", "1"]
            + ["90.13", "45.00", "80.00", "40.00", "70.00", "35.00", "60.00", "30.00"]
            + ["75.03", "60.00", "0", "12.5", "0.0400"]
        ),
        "\t".join(
            ["t0.5_bt0.5_src_s2", "0.5", "0.5", "src", "2"]
            + ["92.00", "46.00", "84.00", "42.00", "72.00", "36.00", "64.00", "32.00"]
            + ["78.00", "64.00", "1", "13.8", "0.0400"]
        ),
        "\t".join(
            ["t0.5_bt0.5_tgt_s1", "0.5", "0.5", "tgt", "1"]
            + ["80.00", "40.00", "70.00", "35.00", "60.00", "30.00", "50.00", "25.00"]
            + ["65.00", "50.00", "2", "20.3", "0.0400"]
        ),
        "\t".join(
            ["t0.5_bt0.5_tgt_s2", "0.5", "0.5", "tgt", "2"]
            + ["82.00", "41.00", "74.00", "37.00", "64.00", "32.00", "52.00", "26.00"]
            + ["68.00", "52.00", "0", "21.0", "0.0400"]
        ),
    ]


def test_the_table_uses_lf_only_and_every_row_has_the_columns_of_the_header(
    scripted: AblationRun,
) -> None:
    raw = (scripted.plan.out_dir / "results.tsv").read_bytes()

    assert b"\r" not in raw
    assert raw.endswith(b"\n") and not raw.endswith(b"\n\n")
    columns = {len(line.split("\t")) for line in raw.decode().splitlines()}
    assert columns == {18}


def test_an_axis_value_is_printed_as_in_the_variant_id_not_rounded_to_two_decimals(
    toy_spm: Path, tmp_path: Path
) -> None:
    plan = plan_ablation(
        make_matrix(axes={"temperature": [0.125, 1.0], "bt_ratio": [0.0]}),
        [write_toy_corpus(tmp_path / "in")],
        toy_spm,
        tmp_path / "out",
    )

    run = AblationRun(plan)

    rows = _read(run, "results.tsv").split("\n")[1:3]
    assert [row.split("\t")[:5] for row in rows] == [
        ["t0.125_bt0_src_s1234", "0.125", "0", "src", "1234"],
        ["t1_bt0_src_s1234", "1", "0", "src", "1234"],
    ]


def test_scores_at_the_ends_of_the_range_keep_two_decimals(
    toy_spm: Path, tmp_path: Path
) -> None:
    plan = plan_ablation(
        make_matrix(axes=one_variant_axes()),
        [write_toy_corpus(tmp_path / "in")],
        toy_spm,
        tmp_path / "out",
    )
    best = dev_result(100.0, 0.0, tag_leaks=12)
    clock = FakeClock()

    run = AblationRun(
        plan,
        train=FakeTrain(clock=clock),
        evaluate=FakeEvaluate({"t0.5_bt0.5_src_s1234": best}, clock=clock),
        clock=clock,
    )

    row = _read(run, "results.tsv").split("\n")[1].split("\t")
    assert row[5:13] == ["100.00", "0.00"] * 4
    assert row[13:] == ["100.00", "100.00", "12", "1.0", "0.0000"]


def test_the_source_collision_rate_is_printed_to_four_decimals_in_both_tables(
    toy_spm: Path, tmp_path: Path
) -> None:
    # 90 sources, each in two training rows with different target languages: 180 of
    # 4,980 training rows, 180 / 4980 = 0.0361446 (two decimals would say 0.04).
    corpus = write_toy_corpus(tmp_path / "in", extra_rows=colliding_rows(90))
    plan = plan_ablation(
        make_matrix(axes=one_variant_axes()), [corpus], toy_spm, tmp_path / "out"
    )

    run = AblationRun(plan)

    assert _read(run, "results.tsv").split("\n")[1].split("\t")[-1] == "0.0361"
    markdown_row = _read(run, "results.md").split("\n")[4]
    assert markdown_row.endswith("| 0.0361 |")


# --- results.md ----------------------------------------------------------------------


EXPECTED_MARKDOWN = """\
# Ablation results: toy

| variant | temperature | bt_ratio | tag_placement | seed | chrf_mean | chrf_min | tag_leaks | train_seconds | source_collision_rate |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| t0.5_bt0.5_src_s1 | 0.5 | 0.5 | src | 1 | 75.03 | 60.00 | 0 | 12.5 | 0.0400 |
| t0.5_bt0.5_src_s2 | 0.5 | 0.5 | src | 2 | 78.00 | 64.00 | 1 | 13.8 | 0.0400 |
| t0.5_bt0.5_tgt_s1 | 0.5 | 0.5 | tgt | 1 | 65.00 | 50.00 | 2 | 20.3 | 0.0400 |
| t0.5_bt0.5_tgt_s2 | 0.5 | 0.5 | tgt | 2 | 68.00 | 52.00 | 0 | 21.0 | 0.0400 |

## Summary over the seeds

| temperature | bt_ratio | tag_placement | seeds | chrf_mean | chrf_min |
| ---: | ---: | --- | ---: | ---: | ---: |
| 0.5 | 0.5 | src | 2 | 76.52 | 62.00 |
| 0.5 | 0.5 | tgt | 2 | 66.50 | 51.00 |
"""


def test_the_markdown_has_the_table_and_the_summary_over_the_seeds(
    scripted: AblationRun,
) -> None:
    assert _read(scripted, "results.md") == EXPECTED_MARKDOWN


def test_a_setting_with_one_seed_shows_one_seed_and_its_own_scores(
    toy_spm: Path, tmp_path: Path
) -> None:
    plan = plan_ablation(
        make_matrix(
            axes={
                "temperature": [0.5, 1.0],
                "bt_ratio": [0.0],
                "tag_placement": ["src"],
            }
        ),
        [write_toy_corpus(tmp_path / "in")],
        toy_spm,
        tmp_path / "out",
    )
    scores = {
        "t0.5_bt0_src_s1234": dev_result(70.0),
        "t1_bt0_src_s1234": dev_result(
            {EN_MN: 90.0, MN_EN: 50.0, MN_RU: 60.0, RU_MN: 40.0}
        ),
    }

    run = AblationRun(plan, evaluate=FakeEvaluate(scores))

    summary = _read(run, "results.md").split("## Summary over the seeds")[1]
    assert summary.strip().split("\n")[-2:] == [
        "| 0.5 | 0 | src | 1 | 70.00 | 70.00 |",
        "| 1 | 0 | src | 1 | 60.00 | 40.00 |",
    ]


# --- results.json --------------------------------------------------------------------


def test_the_json_file_ends_with_a_line_feed(scripted: AblationRun) -> None:
    assert _read(scripted, "results.json").endswith("}\n")


def test_the_json_holds_the_matrix_the_statistics_and_every_variant(
    scripted: AblationRun,
) -> None:
    document = json.loads(_read(scripted, "results.json"))
    plan = scripted.plan

    assert document["name"] == "toy"
    assert document["threads"] == 2
    assert document["matrix"] == {
        "name": "toy",
        "base": str(plan.matrix.base),
        "axes": {
            "temperature": [0.5],
            "bt_ratio": [0.5],
            "tag_placement": ["src", "tgt"],
        },
        "seeds": [1, 2],
        "weight_total": 100,
        "dev_max_lines": 1000,
        "max_variants": 64,
        "overrides": {},
    }
    assert document["statistics"] == plan.training.stats.to_dict()
    assert document["statistics"]["rows_kept"] == 5000
    assert document["statistics"]["source_collision_rate"] == 0.04
    assert document["dev_statistics"] == plan.dev.stats.to_dict()
    assert document["dev_statistics"]["dev_in_train"] == 0
    assert [v["id"] for v in document["variants"]] == [SRC_1, SRC_2, TGT_1, TGT_2]


def test_a_variant_of_the_json_holds_everything_the_table_summarises(
    scripted: AblationRun,
) -> None:
    variant = json.loads(_read(scripted, "results.json"))["variants"][1]
    result = scripted.result.variants[1]

    assert variant["id"] == SRC_2
    assert (
        variant["temperature"],
        variant["bt_ratio"],
        variant["tag_placement"],
        variant["seed"],
    ) == (0.5, 0.5, "src", 2)
    assert variant["status"] == "ok"
    assert variant["fingerprint"] == result.fingerprint
    assert variant["train_seconds"] == pytest.approx(13.76)
    assert variant["eval_seconds"] == pytest.approx(3.0)
    assert variant["source_collision_rate"] == 0.04
    assert variant["checkpoint_file"] == result.checkpoint_path.as_posix()
    assert variant["checkpoint_sha256"] == result.checkpoint_sha256
    assert variant["bt"]["eng_Latn-khk_Cyrl"] == {
        "bt_available": 300,
        "bt_requested": 600,
        "bt_used": 300,
    }
    first = variant["corpora"][0]
    assert first["name"] == f"{EN_MN}.auth"
    assert first["rows"] == 1200
    assert sum(c["weight"] for c in variant["corpora"]) == 100
    assert set(first) == {
        "name",
        "direction",
        "rows",
        "probability",
        "weight",
        "upsampling",
    }


def test_the_json_keeps_the_raw_scores_and_the_signatures(
    scripted: AblationRun,
) -> None:
    variant = json.loads(_read(scripted, "results.json"))["variants"][0]

    assert variant["dev"] == SCORES[SRC_1].to_dict()
    assert variant["dev"]["directions"][EN_MN] == {
        "lines": 100,
        "chrf": 90.126,
        "bleu": 45.0,
    }
    assert variant["dev"]["chrf_mean"] == pytest.approx(75.0315)
    assert variant["dev"]["chrf_signature"] == CHRF_SIGNATURE
    assert variant["dev"]["bleu_signature"] == BLEU_SIGNATURE


def test_each_variant_of_the_json_is_the_content_of_its_result_file(
    scripted: AblationRun,
) -> None:
    variants = json.loads(_read(scripted, "results.json"))["variants"]

    for variant in variants:
        on_disk = json.loads(
            scripted.result_path(variant["id"]).read_text(encoding="utf-8")
        )
        assert on_disk == variant


def test_the_json_is_the_dictionary_of_the_result(scripted: AblationRun) -> None:
    result: AblationResult = scripted.result

    assert json.loads(_read(scripted, "results.json")) == json.loads(
        json.dumps(result.to_dict())
    )


def test_the_checkpoint_of_each_variant_is_the_file_of_its_run_directory(
    scripted: AblationRun,
) -> None:
    for variant in scripted.result.variants:
        expected = (
            scripted.variant_dir(variant.planned.variant.id)
            / "run"
            / "ckpt"
            / CHECKPOINT_FILE
        )
        assert variant.checkpoint_path == expected
        assert expected.is_file()


def test_the_tables_are_written_again_with_the_same_bytes(
    scripted: AblationRun,
) -> None:
    before = {
        name: _read(scripted, name)
        for name in ("results.tsv", "results.json", "results.md")
    }

    files = write_results(scripted.result)

    assert files == result_files(scripted.plan.out_dir)
    assert {name: _read(scripted, name) for name in before} == before
    assert not list(scripted.plan.out_dir.rglob("*.part"))


def test_a_score_that_is_not_finite_writes_no_table_at_all(
    scripted: AblationRun, tmp_path: Path
) -> None:
    # `results.json` must stay JSON (no `NaN` token), and the three tables are written
    # whole or not at all: this is the guard behind the check of `run_ablation`.
    first, *others = scripted.result.variants
    broken = dataclasses.replace(
        first, dev=dataclasses.replace(first.dev, chrf_mean=math.nan)
    )
    plan = dataclasses.replace(scripted.plan, out_dir=tmp_path / "out")
    result = AblationResult(plan, scripted.result.threads, (broken, *others))

    with pytest.raises(ValueError, match="not JSON compliant"):
        write_results(result)

    assert list((tmp_path / "out").iterdir()) == []


def test_a_matrix_name_with_a_lone_surrogate_does_not_fail_the_write(
    toy_spm: Path, tmp_path: Path
) -> None:
    plan = plan_ablation(
        make_matrix(name="toy" + chr(0xD800), axes=one_variant_axes()),
        [write_toy_corpus(tmp_path / "in")],
        toy_spm,
        tmp_path / "out",
    )

    run = AblationRun(plan)

    assert _read(run, "results.md").startswith("# Ablation results: toy\\ud800\n")
    assert json.loads(_read(run, "results.json"))["name"] == "toy" + chr(0xD800)


def test_the_tables_are_written_into_an_output_directory_that_is_not_there_yet(
    scripted: AblationRun, tmp_path: Path
) -> None:
    plan = dataclasses.replace(scripted.plan, out_dir=tmp_path / "new" / "dir")
    result = AblationResult(plan, scripted.result.threads, scripted.result.variants)

    files = write_results(result)

    assert files.tsv.read_text(encoding="utf-8") == _read(scripted, "results.tsv")


def test_the_direction_columns_are_sorted_whatever_the_order_of_the_dev_sets(
    scripted: AblationRun, tmp_path: Path
) -> None:
    sets = dict(reversed(list(scripted.plan.dev.sets.items())))
    assert list(sets) != sorted(sets)
    plan = dataclasses.replace(
        scripted.plan,
        out_dir=tmp_path,
        dev=dataclasses.replace(scripted.plan.dev, sets=sets),
    )

    files = write_results(
        AblationResult(plan, scripted.result.threads, scripted.result.variants)
    )

    assert files.tsv.read_text(encoding="utf-8") == _read(scripted, "results.tsv")


def test_a_matrix_name_over_several_lines_is_one_line_in_the_markdown_title(
    toy_spm: Path, tmp_path: Path
) -> None:
    plan = plan_ablation(
        make_matrix(name="toy  run\nsecond", axes=one_variant_axes()),
        [write_toy_corpus(tmp_path / "in")],
        toy_spm,
        tmp_path / "out",
    )

    run = AblationRun(plan)

    assert _read(run, "results.md").startswith("# Ablation results: toy run second\n\n")
