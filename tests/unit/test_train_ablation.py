"""Tests for `rinchen.train.ablation`: the plan of an ablation, the preparation of one
variant, and the run that trains and evaluates the variants one after the other.

Nothing starts Eole: `train` and `evaluate` are the fakes of
`tests/support/fake_ablation.py`, which record what they are handed, write a
checkpoint file and return scripted dev scores. The toy corpus is written as Parquet in
a temporary directory. The numbers of the plan are worked out by hand from the toy
data (D7 spec R3 to R5, section 5.8):

    train rows      en-mn and mn-en 1,200 each, ru-mn and mn-ru 600 each
    bt rows         300 in each of the four directions
    bt ratio 0.5    min(300, 0.5 * 1,200 = 600) = 300 bt rows in each direction
    temperature 0.5 direction totals 1,500, 1,500, 900, 900 give shares
                    sqrt(1500) : sqrt(900) = 38.73 : 30, i.e. 0.28175 and 0.21825;
                    inside a direction the corpora split by rows (4 : 1 and 2 : 1)
    weights (100)   22 6 22 6 15 7 15 7 (floors 22 5 22 5 14 7 14 7 = 96, the four
                    largest remainders .635 .635 .550 .550 take one each)
"""

from __future__ import annotations

import dataclasses
import hashlib
import importlib.metadata
import inspect
import json
import re
import shutil
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from rinchen.errors import ConfigError, DataError, UpstreamError
from rinchen.train import ablation, ablation_plan
from rinchen.train.ablation import (
    AblationPlan,
    PreparedVariant,
    VariantProgress,
    plan_ablation,
    prepare_variant,
    run_ablation,
)
from rinchen.train.corpus_files import build_corpora
from rinchen.train.eole_cfg import dump_config, read_config, validate_train_config
from rinchen.train.eole_run import CHECKPOINT_FILE, run_training
from rinchen.train.evaluate import DevResult, evaluate_variant
from rinchen.train.mixture import TrainingRows, load_training_rows
from rinchen.train.variants import expand_variants
from tests.support.corpus_rows import EN_MN, MN_EN, MN_RU, RU_MN
from tests.support.fake_ablation import (
    DEV_LINES,
    DIRECTIONS,
    ID_SRC,
    ID_TGT,
    AblationRun,
    FakeClock,
    FakeEvaluate,
    FakeTrain,
    ProgressLog,
    dev_result,
    flip_a_byte,
    make_matrix,
    one_variant_axes,
    toy_plan,
    write_toy_corpus,
)
from tests.support.tiny_parallel import build_toy, train_toy_spm, write_corpus_parquet

# (name, rows, weight, probability, upsampling) of the default variant, in the order
# of the plan: directions sorted, `auth` before `bt`.
DEFAULT_CORPORA = [
    (f"{EN_MN}.auth", 1200, 22, 0.225403, 0.88),
    (f"{EN_MN}.bt", 300, 6, 0.056351, 0.96),
    (f"{MN_EN}.auth", 1200, 22, 0.225403, 0.88),
    (f"{MN_EN}.bt", 300, 6, 0.056351, 0.96),
    (f"{MN_RU}.auth", 600, 15, 0.145497, 1.2),
    (f"{MN_RU}.bt", 300, 7, 0.072749, 1.12),
    (f"{RU_MN}.auth", 600, 15, 0.145497, 1.2),
    (f"{RU_MN}.bt", 300, 7, 0.072749, 1.12),
]


@pytest.fixture(scope="module")
def toy_spm(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return train_toy_spm(tmp_path_factory.mktemp("ablation_spm"))


@pytest.fixture(scope="module")
def other_spm(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return train_toy_spm(tmp_path_factory.mktemp("ablation_other_spm"), vocab_size=600)


@pytest.fixture(scope="module")
def corpus(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return write_toy_corpus(tmp_path_factory.mktemp("ablation_corpus"))


@pytest.fixture(scope="module")
def default_plan(
    corpus: Path, toy_spm: Path, tmp_path_factory: pytest.TempPathFactory
) -> AblationPlan:
    """The plan of the default matrix (src and tgt at temperature 0.5, bt 0.5). Its
    output directory is never created: tests take it through `_plan_in`."""
    out_dir = tmp_path_factory.mktemp("ablation_default") / "out"
    return plan_ablation(make_matrix(), [corpus], toy_spm, out_dir)


def _plan_in(plan: AblationPlan, tmp_path: Path) -> AblationPlan:
    return dataclasses.replace(plan, out_dir=(tmp_path / "out").resolve())


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


Run = AblationRun


@pytest.fixture(scope="module")
def completed(
    default_plan: AblationPlan, tmp_path_factory: pytest.TempPathFactory
) -> Run:
    """A finished run of the default plan, for tests that only read what it wrote."""
    plan = _plan_in(default_plan, tmp_path_factory.mktemp("ablation_completed"))
    return Run(plan)


# --- plan_ablation: what is read -----------------------------------------------------


def test_the_plan_counts_the_training_rows_and_what_it_left_out(
    default_plan: AblationPlan,
) -> None:
    stats = default_plan.training.stats

    # 3,600 authentic + 1,200 bt rows train; 400 dev and 400 test rows do not.
    assert stats.rows_read == 5600
    assert stats.rows_kept == 4800
    assert stats.excluded["not_train"] == 800
    assert stats.rows_by_corpus == {
        f"{EN_MN}.auth": 1200,
        f"{EN_MN}.bt": 300,
        f"{MN_EN}.auth": 1200,
        f"{MN_EN}.bt": 300,
        f"{MN_RU}.auth": 600,
        f"{MN_RU}.bt": 300,
        f"{RU_MN}.auth": 600,
        f"{RU_MN}.bt": 300,
    }


def test_the_plan_reads_only_the_dev_rows_for_the_dev_sets_and_compares_them_to_training(
    default_plan: AblationPlan,
) -> None:
    dev = default_plan.dev.stats

    assert dev.rows_read == 5600
    assert dev.rows_usable == 400
    assert dev.excluded["not_dev"] == 5200
    assert dev.lines_by_direction == dict.fromkeys(DIRECTIONS, DEV_LINES)
    # Not None: the training rows were given, so the overlap was counted.
    assert dev.dev_in_train == 0
    assert sorted(default_plan.dev.sets) == list(DIRECTIONS)


def test_the_dev_sets_hold_at_most_the_lines_the_matrix_allows(
    corpus: Path, toy_spm: Path, tmp_path: Path
) -> None:
    plan = plan_ablation(
        make_matrix(dev_max_lines=10), [corpus], toy_spm, tmp_path / "out"
    )

    assert plan.dev.stats.lines_by_direction == dict.fromkeys(DIRECTIONS, 10)
    assert plan.dev.stats.available_by_direction == dict.fromkeys(DIRECTIONS, DEV_LINES)
    assert all(len(dev_set.src) == 10 for dev_set in plan.dev.sets.values())


def test_the_plan_keeps_the_paths_it_was_given_as_absolute_resolved_paths(
    corpus: Path, toy_spm: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    shutil.copy(toy_spm, tmp_path / "m.model")
    monkeypatch.chdir(tmp_path)

    plan = plan_ablation(make_matrix(), [corpus], Path("m.model"), Path("out"))

    assert plan.out_dir == (tmp_path / "out").resolve()
    assert plan.spm_model == (tmp_path / "m.model").resolve()
    assert plan.corpus_paths == (corpus,)


# --- plan_ablation: the variants -----------------------------------------------------


def test_the_variants_of_the_plan_are_those_of_the_matrix_in_expansion_order(
    corpus: Path, toy_spm: Path, tmp_path: Path
) -> None:
    matrix = make_matrix(
        axes={
            "temperature": [0.5, 1.0],
            "bt_ratio": [0.0, 0.5],
            "tag_placement": ["src", "tgt"],
        },
        seeds=[1, 2],
    )

    plan = plan_ablation(matrix, [corpus], toy_spm, tmp_path / "out")

    ids = [planned.variant.id for planned in plan.variants]
    assert ids[:5] == [
        "t0.5_bt0_src_s1",
        "t0.5_bt0_src_s2",
        "t0.5_bt0_tgt_s1",
        "t0.5_bt0_tgt_s2",
        "t0.5_bt0.5_src_s1",
    ]
    assert ids[-1] == "t1_bt0.5_tgt_s2"
    assert len(ids) == 16
    assert [planned.variant for planned in plan.variants] == expand_variants(matrix)


def test_the_planned_corpora_of_a_variant_are_the_hand_counted_ones(
    default_plan: AblationPlan,
) -> None:
    planned = default_plan.variants[0]

    assert planned.variant.id == ID_SRC
    assert [
        (c.name, c.rows, c.weight, c.probability, c.upsampling) for c in planned.corpora
    ] == [
        (name, rows, weight, pytest.approx(prob, abs=1e-6), pytest.approx(up))
        for name, rows, weight, prob, up in DEFAULT_CORPORA
    ]
    assert [c.direction for c in planned.corpora] == [
        EN_MN,
        EN_MN,
        MN_EN,
        MN_EN,
        MN_RU,
        MN_RU,
        RU_MN,
        RU_MN,
    ]


def test_the_two_tag_placements_plan_the_same_mixture(
    default_plan: AblationPlan,
) -> None:
    src, tgt = default_plan.variants

    assert src.corpora == tgt.corpora
    assert src.bt_by_direction == tgt.bt_by_direction


def test_the_weights_of_a_variant_without_bt_follow_the_proportional_temperature(
    corpus: Path, toy_spm: Path, tmp_path: Path
) -> None:
    # Temperature 1 is proportional: shares 1/3, 1/3, 1/6, 1/6 of 100 are 33.3 and
    # 16.7; the floors 33 33 16 16 leave 2, which go to the two largest remainders.
    plan = plan_ablation(
        make_matrix(axes={"temperature": [1.0], "bt_ratio": [0.0]}),
        [corpus],
        toy_spm,
        tmp_path / "out",
    )

    planned = plan.variants[0]
    assert planned.weights == {
        f"{EN_MN}.auth": 33,
        f"{MN_EN}.auth": 33,
        f"{MN_RU}.auth": 17,
        f"{RU_MN}.auth": 17,
    }
    assert [c.probability for c in planned.corpora] == pytest.approx(
        [1 / 3, 1 / 3, 1 / 6, 1 / 6]
    )
    # (w / 100) / (rows / 3,600)
    assert [c.upsampling for c in planned.corpora] == pytest.approx(
        [0.99, 0.99, 1.02, 1.02]
    )
    assert planned.bt_by_direction[EN_MN].bt_used == 0
    assert planned.bt_by_direction[EN_MN].bt_available == 300


@pytest.mark.parametrize("weight_total", [8, 37, 100, 1000])
def test_the_weights_of_every_variant_sum_to_the_weight_total_and_none_is_zero(
    corpus: Path, toy_spm: Path, tmp_path: Path, weight_total: int
) -> None:
    plan = plan_ablation(
        make_matrix(
            axes={"temperature": [0.3, 1.0, 2.0], "bt_ratio": [0.0, 0.1, 0.5]},
            weight_total=weight_total,
        ),
        [corpus],
        toy_spm,
        tmp_path / "out",
    )

    for planned in plan.variants:
        weights = list(planned.weights.values())
        assert sum(weights) == weight_total, planned.variant.id
        assert min(weights) >= 1, planned.variant.id


def test_the_upsampling_is_the_weight_share_over_the_row_share(
    default_plan: AblationPlan,
) -> None:
    # 4,800 rows in all and weights summing to 100: auth 1200 -> (22/100)/(1200/4800).
    corpora = {c.name: c for c in default_plan.variants[0].corpora}

    assert corpora[f"{EN_MN}.auth"].upsampling == pytest.approx(0.22 / 0.25)
    assert corpora[f"{EN_MN}.bt"].upsampling == pytest.approx(0.06 / 0.0625)
    assert corpora[f"{RU_MN}.auth"].upsampling == pytest.approx(0.15 / 0.125)
    assert corpora[f"{RU_MN}.bt"].upsampling == pytest.approx(0.07 / 0.0625)
    # A direction's share of the rows is what it would get at temperature 1.
    uniform = sum(c.rows * c.upsampling for c in corpora.values()) / 4800
    assert uniform == pytest.approx(1.0)


@pytest.mark.parametrize("ratio", [0.0, 0.1, 0.5, 1.0, 2.5])
def test_the_plan_never_disagrees_with_the_build(
    corpus: Path, toy_spm: Path, tmp_path: Path, ratio: float
) -> None:
    plan = plan_ablation(
        make_matrix(axes=one_variant_axes("src", bt_ratio=ratio)),
        [corpus],
        toy_spm,
        tmp_path / "out",
    )

    built = build_corpora(
        plan.training, bt_ratio=ratio, seed=1234, data_dir=tmp_path / "data"
    )

    planned = plan.variants[0]
    assert [(c.name, c.direction, c.rows) for c in planned.corpora] == [
        (c.name, c.direction, c.rows) for c in built.corpora
    ]
    assert planned.bt_by_direction == built.bt_by_direction


def test_planning_creates_no_file_and_no_directory(
    corpus: Path, toy_spm: Path, tmp_path: Path
) -> None:
    out_dir = tmp_path / "does" / "not" / "exist"
    before = sorted(path for path in tmp_path.rglob("*"))

    plan_ablation(make_matrix(), [corpus], toy_spm, out_dir)

    assert not out_dir.exists()
    assert not (tmp_path / "does").exists()
    assert sorted(path for path in tmp_path.rglob("*")) == before


def test_the_plan_prints_the_weight_total_of_the_matrix(
    corpus: Path, toy_spm: Path, tmp_path: Path
) -> None:
    plan = plan_ablation(
        make_matrix(weight_total=50), [corpus], toy_spm, tmp_path / "out"
    )

    assert plan.to_dict()["weight_total"] == 50


def test_the_plan_is_json_ready_and_prints_what_a_dry_run_shows(
    default_plan: AblationPlan,
) -> None:
    document = json.loads(json.dumps(default_plan.to_dict()))

    assert document["name"] == "toy"
    assert document["weight_total"] == 100
    assert document["statistics"]["rows_kept"] == 4800
    assert document["dev_statistics"]["dev_in_train"] == 0
    assert [v["id"] for v in document["variants"]] == [ID_SRC, ID_TGT]
    first = document["variants"][0]
    assert (first["temperature"], first["bt_ratio"], first["tag_placement"]) == (
        0.5,
        0.5,
        "src",
    )
    assert first["seed"] == 1234
    assert first["corpora"][0] == {
        "name": f"{EN_MN}.auth",
        "direction": EN_MN,
        "rows": 1200,
        "probability": pytest.approx(0.225403, abs=1e-6),
        "weight": 22,
        "upsampling": pytest.approx(0.88),
    }
    assert first["bt"][EN_MN] == {
        "bt_available": 300,
        "bt_requested": 600,
        "bt_used": 300,
    }


# --- plan_ablation: what is refused --------------------------------------------------


@pytest.mark.parametrize(
    ("reserved", "overrides"),
    [
        ("data", {"data": {"extra": {"path_src": "x"}}}),
        ("data", {"data": None}),
        ("src_vocab", {"src_vocab": "elsewhere/vocab"}),
        ("seed", {"seed": 5}),
        ("training.model_path", {"training": {"model_path": "elsewhere"}}),
        ("training.model_path", {"training": {"model_path": None}}),
        ("training.model_path", {"training": None}),
        ("training.model_path", {"training": "replaced"}),
        (
            "transforms_configs.sentencepiece",
            {"transforms_configs": {"sentencepiece": {"src_subword_model": "x"}}},
        ),
        (
            "transforms_configs.sentencepiece",
            {"transforms_configs": {"sentencepiece": None}},
        ),
        ("transforms_configs.sentencepiece", {"transforms_configs": None}),
    ],
)
def test_an_override_of_a_per_run_key_is_refused_naming_the_key(
    corpus: Path,
    toy_spm: Path,
    tmp_path: Path,
    reserved: str,
    overrides: dict[str, Any],
) -> None:
    matrix = make_matrix(overrides=overrides)

    with pytest.raises(ConfigError, match=re.escape(reserved)):
        plan_ablation(matrix, [corpus], toy_spm, tmp_path / "out")


def test_a_reserved_override_is_refused_before_any_corpus_is_read(
    toy_spm: Path, tmp_path: Path
) -> None:
    matrix = make_matrix(overrides={"seed": 5})

    with pytest.raises(ConfigError, match="seed"):
        plan_ablation(matrix, [tmp_path / "missing.parquet"], toy_spm, tmp_path / "o")


@pytest.mark.parametrize(
    "overrides",
    [
        {},
        {"training": {}},
        {"training": {"dropout": [0.1], "learning_rate": 0.7}},
        {"training": {"keep_checkpoint": None}},
        {"transforms_configs": {}},
        {"transforms_configs": {"filtertoolong": {"src_seq_length": 32}}},
        {"model": {"encoder": {"layers": 3}}},
    ],
)
def test_overrides_of_the_other_keys_are_accepted_at_plan_time(
    corpus: Path, toy_spm: Path, tmp_path: Path, overrides: dict[str, Any]
) -> None:
    plan = plan_ablation(
        make_matrix(overrides=overrides), [corpus], toy_spm, tmp_path / "out"
    )

    assert len(plan.variants) == 2


def test_a_temperature_whose_powers_overflow_is_a_config_error_naming_the_variant(
    corpus: Path, toy_spm: Path, tmp_path: Path
) -> None:
    matrix = make_matrix(axes={"temperature": [0.5, 1e6], "bt_ratio": [0.5]})

    with pytest.raises(ConfigError, match=r"variant t1e\+06_bt0\.5_src_s1234") as info:
        plan_ablation(matrix, [corpus], toy_spm, tmp_path / "out")

    assert "too large" in str(info.value)
    assert isinstance(info.value.__cause__, ValueError)


def test_more_corpora_than_the_weight_total_is_a_config_error_naming_the_need(
    corpus: Path, toy_spm: Path, tmp_path: Path
) -> None:
    # bt ratio 0.5 makes eight corpora and every corpus needs a weight of 1.
    matrix = make_matrix(weight_total=7)

    with pytest.raises(ConfigError, match=r"weight_total.*at least 8"):
        plan_ablation(matrix, [corpus], toy_spm, tmp_path / "out")


def test_the_smallest_weight_total_that_holds_every_corpus_is_accepted(
    corpus: Path, toy_spm: Path, tmp_path: Path
) -> None:
    plan = plan_ablation(
        make_matrix(weight_total=8), [corpus], toy_spm, tmp_path / "out"
    )

    assert sorted(plan.variants[0].weights.values()) == [1] * 8


def test_corpus_files_without_a_training_row_are_a_data_error(
    toy_spm: Path, tmp_path: Path
) -> None:
    dev_only = write_corpus_parquet(build_toy().dev, tmp_path / "dev.parquet")

    with pytest.raises(DataError, match="no training rows"):
        plan_ablation(make_matrix(), [dev_only], toy_spm, tmp_path / "out")


def test_a_mixture_without_any_corpus_is_a_data_error_before_any_weight_is_taken(
    corpus: Path, toy_spm: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    real = load_training_rows([corpus])
    emptied = dataclasses.replace(real, frame=real.frame.head(0))

    def load_nothing(paths: Any) -> TrainingRows:
        return emptied

    monkeypatch.setattr(ablation_plan, "load_training_rows", load_nothing)

    with pytest.raises(DataError, match="no corpus"):
        plan_ablation(make_matrix(), [corpus], toy_spm, tmp_path / "out")


def test_a_missing_corpus_file_is_a_data_error(toy_spm: Path, tmp_path: Path) -> None:
    with pytest.raises(DataError, match="missing.parquet"):
        plan_ablation(
            make_matrix(), [tmp_path / "missing.parquet"], toy_spm, tmp_path / "out"
        )


# --- prepare_variant -----------------------------------------------------------------


def _prepared(
    plan: AblationPlan, tmp_path: Path, index: int = 0, *, threads: int = 2
) -> PreparedVariant:
    return prepare_variant(
        _plan_in(plan, tmp_path), plan.variants[index], threads=threads
    )


def test_a_prepared_variant_lays_its_files_out_under_variants_and_its_id(
    default_plan: AblationPlan, tmp_path: Path
) -> None:
    plan = _plan_in(default_plan, tmp_path)

    prepared = prepare_variant(plan, plan.variants[0], threads=2)

    directory = plan.out_dir / "variants" / ID_SRC
    assert prepared.planned is plan.variants[0]
    assert prepared.directory == directory
    assert prepared.config_path == directory / "config.yaml"
    assert prepared.run_dir == directory / "run"
    assert prepared.eval_dir == directory / "eval"
    assert prepared.result_path == directory / "result.json"
    assert prepared.ckpt_dir == directory / "run" / "ckpt"
    assert prepared.checkpoint_path == directory / "run" / "ckpt" / CHECKPOINT_FILE
    assert prepared.valid.src == directory / "valid.src"
    assert prepared.valid.tgt == directory / "valid.tgt"
    assert prepared.config_path.is_file()
    assert prepared.valid.src.is_file() and prepared.valid.tgt.is_file()
    # Nothing of the run itself exists yet.
    assert not prepared.run_dir.exists()
    assert not prepared.eval_dir.exists()
    assert not prepared.result_path.exists()


def test_a_prepared_variant_builds_the_corpora_and_names_the_weights_of_the_plan(
    default_plan: AblationPlan, tmp_path: Path
) -> None:
    plan = _plan_in(default_plan, tmp_path)

    prepared = prepare_variant(plan, plan.variants[0], threads=2)

    assert [(c.name, c.rows) for c in prepared.corpora.corpora] == [
        (name, rows) for name, rows, *_ in DEFAULT_CORPORA
    ]
    assert prepared.weights == plan.variants[0].weights
    assert prepared.weights[f"{EN_MN}.auth"] == 22
    assert all(c.src_path.is_file() for c in prepared.corpora.corpora)
    assert (plan.out_dir / "data" / "auth" / f"{EN_MN}.auth.src").is_file()
    assert prepared.corpora.bt_by_direction == plan.variants[0].bt_by_direction


def test_the_rendered_config_passes_the_schema_and_carries_the_per_run_values(
    completed: Run,
) -> None:
    cfg = completed.config(ID_SRC)
    run_dir = completed.variant_dir(ID_SRC) / "run"
    validate_train_config(cfg)

    assert cfg["seed"] == 1234
    assert cfg["src_vocab"] == (run_dir / "vocab.src").as_posix()
    assert cfg["training"]["model_path"] == (run_dir / "ckpt").as_posix()
    spm = completed.plan.spm_model.as_posix()
    assert cfg["transforms_configs"]["sentencepiece"] == {
        "src_subword_model": spm,
        "tgt_subword_model": spm,
    }
    # The base's own values stay.
    assert cfg["training"]["batch_size"] == 2048
    assert cfg["model"]["hidden_size"] == 64
    assert cfg["transforms_configs"]["filtertoolong"] == {
        "src_seq_length": 64,
        "tgt_seq_length": 64,
    }
    assert cfg["overwrite"] is True and cfg["share_vocab"] is True


def test_the_rendered_config_has_the_weights_of_the_plan_and_the_prepared_corpus_files(
    completed: Run,
) -> None:
    cfg = completed.config(ID_SRC)
    data_dir = completed.plan.out_dir / "data"

    assert list(cfg["data"]) == [name for name, *_ in DEFAULT_CORPORA] + ["valid"]
    for name, _, weight, *_ in DEFAULT_CORPORA:
        corpus = cfg["data"][name]
        assert corpus["weight"] == weight
        group = name.rpartition(".")[2]
        folder = (
            data_dir / "auth" if group == "auth" else data_dir / "bt" / "r0.5_s1234"
        )
        assert corpus["path_src"] == (folder / f"{name}.src").as_posix()
        assert corpus["path_tgt"] == (folder / f"{name}.tgt").as_posix()
        assert corpus["transforms"] == ["sentencepiece", "prefix", "filtertoolong"]
    valid = cfg["data"]["valid"]
    directory = completed.variant_dir(ID_SRC)
    assert valid == {
        "path_src": (directory / "valid.src").as_posix(),
        "path_tgt": (directory / "valid.tgt").as_posix(),
    }


def test_the_tag_goes_to_the_source_prefix_for_src_and_to_the_target_prefix_for_tgt(
    completed: Run,
) -> None:
    src = completed.config(ID_SRC)["data"]
    tgt = completed.config(ID_TGT)["data"]

    assert (src[f"{EN_MN}.auth"]["src_prefix"], src[f"{EN_MN}.auth"]["tgt_prefix"]) == (
        "<2khk_Cyrl>",
        "",
    )
    assert src[f"{MN_EN}.bt"]["src_prefix"] == "<2eng_Latn> <bt>"
    assert src[f"{MN_EN}.bt"]["tgt_prefix"] == ""
    assert (tgt[f"{EN_MN}.auth"]["src_prefix"], tgt[f"{EN_MN}.auth"]["tgt_prefix"]) == (
        "",
        "<2khk_Cyrl>",
    )
    assert (tgt[f"{MN_RU}.bt"]["src_prefix"], tgt[f"{MN_RU}.bt"]["tgt_prefix"]) == (
        "<bt>",
        "<2rus_Cyrl>",
    )


def test_the_validation_files_hold_the_tag_where_the_placement_says(
    completed: Run,
) -> None:
    src_dir = completed.variant_dir(ID_SRC)
    tgt_dir = completed.variant_dir(ID_TGT)

    first_src = (src_dir / "valid.src").read_text(encoding="utf-8").split("\n")[0]
    first_tgt = (tgt_dir / "valid.tgt").read_text(encoding="utf-8").split("\n")[0]
    assert first_src.startswith("<2khk_Cyrl> ")
    assert first_tgt.startswith("<2khk_Cyrl> ")
    assert "<2khk_Cyrl>" not in (tgt_dir / "valid.src").read_text(encoding="utf-8")
    assert len((src_dir / "valid.src").read_text(encoding="utf-8").split("\n")) == 401


def test_every_path_in_the_config_is_absolute_posix_text(completed: Run) -> None:
    cfg = completed.config(ID_TGT)
    paths = [
        cfg["src_vocab"],
        cfg["training"]["model_path"],
        *cfg["transforms_configs"]["sentencepiece"].values(),
        *(
            corpus[key]
            for corpus in cfg["data"].values()
            for key in ("path_src", "path_tgt")
        ),
    ]

    assert all(Path(path).is_absolute() and "\\" not in path for path in paths)


def test_the_seed_of_the_variant_is_the_seed_key_of_the_config(
    corpus: Path, toy_spm: Path, tmp_path: Path
) -> None:
    plan = toy_plan(
        tmp_path, toy_spm, corpus=corpus, axes=one_variant_axes(), seeds=[7, 42]
    )

    second = prepare_variant(plan, plan.variants[1], threads=2)

    assert read_config(second.config_path, "config")["seed"] == 42
    assert second.planned.variant.id == "t0.5_bt0.5_src_s42"


def test_the_matrix_overrides_are_merged_over_the_base_and_keep_the_per_run_values(
    corpus: Path, toy_spm: Path, tmp_path: Path
) -> None:
    overrides = {
        "training": {"dropout": [0.25], "learning_rate": 0.7, "keep_checkpoint": None},
        "model": {"encoder": {"layers": 3}},
    }
    plan = toy_plan(
        tmp_path,
        toy_spm,
        corpus=corpus,
        axes=one_variant_axes(),
        overrides=overrides,
    )

    prepared = prepare_variant(plan, plan.variants[0], threads=2)

    cfg = read_config(prepared.config_path, "config")
    assert cfg["training"]["dropout"] == [0.25]
    assert cfg["training"]["learning_rate"] == 0.7
    # None deletes the base's key, though it merges into a section the run also sets.
    assert "keep_checkpoint" not in cfg["training"]
    assert cfg["training"]["model_path"] == (prepared.run_dir / "ckpt").as_posix()
    assert cfg["training"]["batch_size"] == 2048
    assert cfg["model"]["encoder"] == {"layers": 3}
    assert cfg["model"]["decoder"] == {"layers": 2}


def _base_without_filtertoolong(tmp_path: Path) -> Path:
    base = read_config(make_matrix().base, "base")
    del base["transforms_configs"]["filtertoolong"]
    path = tmp_path / "base_without.yaml"
    dump_config(base, path)
    return path


@pytest.mark.parametrize(
    ("base_has", "overrides", "expected"),
    [
        (True, {}, True),
        (False, {}, False),
        # The corpora use the transform when the finished config configures it.
        (
            False,
            {"transforms_configs": {"filtertoolong": {"src_seq_length": 32}}},
            True,
        ),
        (True, {"transforms_configs": {"filtertoolong": None}}, False),
    ],
)
def test_the_corpora_run_filtertoolong_when_the_finished_config_configures_it(
    corpus: Path,
    toy_spm: Path,
    tmp_path: Path,
    base_has: bool,
    overrides: dict[str, Any],
    expected: bool,
) -> None:
    base = make_matrix().base if base_has else _base_without_filtertoolong(tmp_path)
    plan = toy_plan(
        tmp_path,
        toy_spm,
        corpus=corpus,
        axes=one_variant_axes(),
        base=str(base),
        overrides=overrides,
    )

    prepared = prepare_variant(plan, plan.variants[0], threads=2)

    cfg = read_config(prepared.config_path, "config")
    transforms = {
        tuple(c["transforms"]) for n, c in cfg["data"].items() if n != "valid"
    }
    expected_transforms = ("sentencepiece", "prefix") + (
        ("filtertoolong",) if expected else ()
    )
    assert transforms == {expected_transforms}


def test_preparing_twice_gives_the_same_config_and_fingerprint_and_reuses_the_files(
    default_plan: AblationPlan, tmp_path: Path
) -> None:
    plan = _plan_in(default_plan, tmp_path)

    first = prepare_variant(plan, plan.variants[0], threads=2)
    second = prepare_variant(plan, plan.variants[0], threads=2)

    assert second.config_text == first.config_text
    assert second.fingerprint == first.fingerprint
    assert second.fingerprint_inputs == first.fingerprint_inputs
    assert not any(c.reused for c in first.corpora.corpora)
    assert all(c.reused for c in second.corpora.corpora)


def test_the_config_text_is_what_the_file_holds(
    default_plan: AblationPlan, tmp_path: Path
) -> None:
    prepared = _prepared(default_plan, tmp_path)

    assert prepared.config_text == prepared.config_path.read_text(encoding="utf-8")
    assert prepared.config_text.endswith("\n")


def test_the_fingerprint_inputs_name_every_input_by_its_digest(
    default_plan: AblationPlan, tmp_path: Path
) -> None:
    prepared = _prepared(default_plan, tmp_path, 1, threads=3)

    corpora = {
        c.name: [_sha256(c.src_path), _sha256(c.tgt_path)]
        for c in prepared.corpora.corpora
    }
    assert len(corpora) == 8
    assert prepared.fingerprint_inputs == {
        "config_sha256": hashlib.sha256(prepared.config_text.encode()).hexdigest(),
        "corpora": corpora,
        "validation": [_sha256(prepared.valid.src), _sha256(prepared.valid.tgt)],
        "spm_sha256": _sha256(default_plan.spm_model),
        "threads": 3,
        "eole": importlib.metadata.version("eole"),
        "placement": "tgt",
    }
    canonical = json.dumps(
        prepared.fingerprint_inputs, sort_keys=True, separators=(",", ":")
    )
    assert prepared.fingerprint == hashlib.sha256(canonical.encode()).hexdigest()


def test_the_fingerprint_changes_with_each_of_its_inputs(
    default_plan: AblationPlan,
    tmp_path: Path,
    corpus: Path,
    toy_spm: Path,
    other_spm: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base = _prepared(default_plan, tmp_path)
    seen = {base.fingerprint}

    def distinct(fingerprint: str) -> None:
        assert fingerprint not in seen
        seen.add(fingerprint)

    # The thread count.
    distinct(_prepared(default_plan, tmp_path, threads=3).fingerprint)
    # The placement.
    distinct(_prepared(default_plan, tmp_path, 1).fingerprint)
    # The installed Eole.
    with monkeypatch.context() as patch:
        patch.setattr(ablation, "tool_versions", lambda packages: {"eole": "9.9.9"})
        distinct(_prepared(default_plan, tmp_path).fingerprint)
    # The SentencePiece model, at another path and with other bytes at the same path.
    elsewhere = _plan_in(default_plan, tmp_path)
    distinct(
        prepare_variant(
            dataclasses.replace(elsewhere, spm_model=other_spm),
            elsewhere.variants[0],
            threads=2,
        ).fingerprint
    )
    in_place = tmp_path / "in_place.model"
    shutil.copy(toy_spm, in_place)
    plan = dataclasses.replace(elsewhere, spm_model=in_place)
    first = prepare_variant(plan, plan.variants[0], threads=2).fingerprint
    shutil.copy(other_spm, in_place)
    assert prepare_variant(plan, plan.variants[0], threads=2).fingerprint != first
    # An override.
    other = toy_plan(
        tmp_path / "ov",
        toy_spm,
        corpus=corpus,
        overrides={"training": {"dropout": [0.1]}},
    )
    distinct(
        prepare_variant(
            dataclasses.replace(other, out_dir=elsewhere.out_dir),
            other.variants[0],
            threads=2,
        ).fingerprint
    )


def test_the_threads_must_be_at_least_one(
    default_plan: AblationPlan, tmp_path: Path
) -> None:
    plan = _plan_in(default_plan, tmp_path)

    with pytest.raises(ValueError, match="threads"):
        prepare_variant(plan, plan.variants[0], threads=0)

    assert not plan.out_dir.exists()


def test_one_thread_is_enough(default_plan: AblationPlan, tmp_path: Path) -> None:
    prepared = _prepared(default_plan, tmp_path, threads=1)

    assert prepared.fingerprint_inputs["threads"] == 1
    plan = _plan_in(default_plan, tmp_path)
    run = Run(plan, threads=1)
    assert {call.threads for call in run.train.calls} == {1}


def test_a_missing_sentencepiece_model_is_a_config_error_at_preparation(
    default_plan: AblationPlan, tmp_path: Path
) -> None:
    plan = dataclasses.replace(
        _plan_in(default_plan, tmp_path), spm_model=tmp_path / "gone.model"
    )

    with pytest.raises(ConfigError, match="gone.model"):
        prepare_variant(plan, plan.variants[0], threads=2)


# --- plan_ablation: paths Eole would expand -------------------------------------------


@pytest.mark.parametrize("which", ["out_dir", "spm_model"])
def test_a_dollar_in_the_out_dir_or_the_model_path_is_a_config_error_at_plan_time(
    corpus: Path, toy_spm: Path, tmp_path: Path, which: str
) -> None:
    out_dir = tmp_path / ("$HOME" if which == "out_dir" else "fine") / "out"
    spm_model = tmp_path / "$HOME.model" if which == "spm_model" else toy_spm

    with pytest.raises(ConfigError, match=r"'\$'") as info:
        plan_ablation(make_matrix(), [corpus], spm_model, out_dir)

    assert ("output directory" if which == "out_dir" else "SentencePiece") in str(
        info.value
    )
    assert not (tmp_path / "$HOME").exists() and not (tmp_path / "fine").exists()


# --- run_ablation: sequencing --------------------------------------------------------


def test_the_seams_default_to_the_real_training_and_evaluation() -> None:
    parameters = inspect.signature(run_ablation).parameters

    assert parameters["train"].default is run_training
    assert parameters["evaluate"].default is evaluate_variant
    assert parameters["progress"].default is None
    for name in ("threads", "timeout", "force"):
        assert parameters[name].default is inspect.Parameter.empty


def test_the_variants_run_in_the_order_of_the_expansion(
    corpus: Path, toy_spm: Path, tmp_path: Path
) -> None:
    plan = toy_plan(
        tmp_path,
        toy_spm,
        corpus=corpus,
        axes={
            "temperature": [1.0, 0.5],
            "bt_ratio": [0.0],
            "tag_placement": ["tgt", "src"],
        },
        seeds=[2, 1],
    )

    run = Run(plan)

    expected = [v.id for v in expand_variants(plan.matrix)]
    assert expected[:3] == ["t1_bt0_tgt_s2", "t1_bt0_tgt_s1", "t1_bt0_src_s2"]
    assert run.train.variant_ids == expected
    assert run.evaluate.variant_ids == expected
    assert [r.planned.variant.id for r in run.result.variants] == expected


def test_training_is_given_the_config_the_run_directory_the_validator_and_the_limits(
    completed: Run,
) -> None:
    calls = {call.variant_id: call for call in completed.train.calls}

    for variant_id in (ID_SRC, ID_TGT):
        call = calls[variant_id]
        directory = completed.variant_dir(variant_id)
        assert call.cfg_path == directory / "config.yaml"
        assert call.run_dir == directory / "run"
        assert call.validate is validate_train_config
        assert (call.threads, call.timeout) == (2, 30.0)
        assert call.config_text == (directory / "config.yaml").read_text(
            encoding="utf-8"
        )


def test_the_evaluation_is_given_the_checkpoint_the_dev_sets_and_the_placement(
    completed: Run,
) -> None:
    calls = {call.variant_id: call for call in completed.evaluate.calls}

    for variant_id, placement in ((ID_SRC, "src"), (ID_TGT, "tgt")):
        call = calls[variant_id]
        directory = completed.variant_dir(variant_id)
        assert call.ckpt_dir == directory / "run" / "ckpt"
        assert call.dev is completed.plan.dev.sets
        assert call.spm_model == completed.plan.spm_model
        assert call.placement == placement
        assert call.out_dir == directory / "eval"
        assert (call.threads, call.timeout) == (2, 30.0)


def test_the_result_file_is_written_after_the_evaluation_and_not_before(
    completed: Run,
) -> None:
    assert [call.result_existed for call in completed.evaluate.calls] == [False, False]
    for variant_id in (ID_SRC, ID_TGT):
        assert completed.result_path(variant_id).is_file()


def test_the_result_of_a_variant_is_what_the_fakes_returned_plus_the_checkpoint_hash(
    completed: Run,
) -> None:
    src, tgt = completed.result.variants

    assert (src.planned.variant.id, tgt.planned.variant.id) == (ID_SRC, ID_TGT)
    assert src.dev == dev_result(80.0)
    assert not src.reused and not tgt.reused
    for result in (src, tgt):
        directory = completed.variant_dir(result.planned.variant.id)
        file = directory / "run" / "ckpt" / CHECKPOINT_FILE
        assert result.checkpoint_path == file
        assert result.checkpoint_sha256 == _sha256(file)
        assert (
            result.fingerprint
            == prepare_variant(completed.plan, result.planned, threads=2).fingerprint
        )


def test_the_result_file_records_the_variant_the_scores_and_the_checkpoint_hash(
    completed: Run,
) -> None:
    record = json.loads(completed.result_path(ID_TGT).read_text(encoding="utf-8"))
    result = completed.result.variants[1]

    assert record["status"] == "ok"
    assert record["id"] == ID_TGT
    assert record["tag_placement"] == "tgt"
    assert record["fingerprint"] == result.fingerprint
    assert record["checkpoint_sha256"] == result.checkpoint_sha256
    assert record["checkpoint_file"] == result.checkpoint_path.as_posix()
    assert record["dev"] == result.dev.to_dict()
    weights = {c["name"]: c["weight"] for c in record["corpora"]}
    assert weights[f"{EN_MN}.auth"] == 22
    assert sum(weights.values()) == 100
    assert completed.result_path(ID_TGT).read_text(encoding="utf-8").endswith("}\n")


def test_the_durations_are_those_of_the_clock_around_each_call(
    default_plan: AblationPlan, tmp_path: Path
) -> None:
    clock = FakeClock()
    run = Run(
        _plan_in(default_plan, tmp_path),
        train=FakeTrain(clock=clock, seconds={ID_SRC: 12.5, ID_TGT: 20.25}),
        evaluate=FakeEvaluate(clock=clock, seconds={ID_SRC: 3.0, ID_TGT: 4.75}),
        clock=clock,
    )

    assert [(r.train_seconds, r.eval_seconds) for r in run.result.variants] == [
        (12.5, 3.0),
        (20.25, 4.75),
    ]
    record = json.loads(run.result_path(ID_TGT).read_text(encoding="utf-8"))
    assert (record["train_seconds"], record["eval_seconds"]) == (20.25, 4.75)


def test_progress_is_reported_once_per_variant_after_its_result_is_written(
    default_plan: AblationPlan, tmp_path: Path
) -> None:
    plan = _plan_in(default_plan, tmp_path)
    clock = FakeClock()
    seen: list[tuple[VariantProgress, bool]] = []

    def watch(event: VariantProgress) -> None:
        written = (
            plan.out_dir / "variants" / event.variant_id / "result.json"
        ).exists()
        seen.append((event, written))

    Run(
        plan,
        train=FakeTrain(clock=clock, seconds={ID_SRC: 12.5, ID_TGT: 20.25}),
        evaluate=FakeEvaluate(clock=clock, seconds={ID_SRC: 3.0, ID_TGT: 4.75}),
        clock=clock,
        progress=watch,
    )

    assert seen == [
        (VariantProgress(ID_SRC, 1, 2, "trained", 12.5, 3.0), True),
        (VariantProgress(ID_TGT, 2, 2, "trained", 20.25, 4.75), True),
    ]


def test_a_progress_event_is_frozen_and_names_its_fields() -> None:
    event = VariantProgress("v", 1, 2, "reused", 0.0, 0.0)

    assert dataclasses.asdict(event) == {
        "variant_id": "v",
        "index": 1,
        "total": 2,
        "outcome": "reused",
        "train_seconds": 0.0,
        "eval_seconds": 0.0,
    }
    with pytest.raises(dataclasses.FrozenInstanceError):
        event.index = 3  # type: ignore[misc]


def test_the_threads_must_be_at_least_one_and_nothing_runs_without_them(
    default_plan: AblationPlan, tmp_path: Path
) -> None:
    train, evaluate = FakeTrain(), FakeEvaluate()
    plan = _plan_in(default_plan, tmp_path)

    with pytest.raises(ValueError, match="threads"):
        run_ablation(
            plan, threads=0, timeout=None, force=False, train=train, evaluate=evaluate
        )

    assert not train.calls and not evaluate.calls
    assert not plan.out_dir.exists()


# --- run_ablation: resuming ----------------------------------------------------------


def test_a_second_run_reuses_every_complete_variant_and_calls_neither_seam(
    default_plan: AblationPlan, tmp_path: Path
) -> None:
    plan = _plan_in(default_plan, tmp_path)
    first = Run(plan)
    log = ProgressLog()

    second = Run(plan, progress=log)

    assert not second.train.calls and not second.evaluate.calls
    assert log.outcomes == [(ID_SRC, "reused"), (ID_TGT, "reused")]
    assert [(e.train_seconds, e.eval_seconds) for e in log.events] == [(0.0, 0.0)] * 2
    assert [e.index for e in log.events] == [1, 2]
    assert [r.reused for r in second.result.variants] == [True, True]
    for old, new in zip(first.result.variants, second.result.variants, strict=True):
        assert dataclasses.replace(new, reused=False) == old


def test_force_reruns_every_variant(default_plan: AblationPlan, tmp_path: Path) -> None:
    plan = _plan_in(default_plan, tmp_path)
    Run(plan)
    log = ProgressLog()

    again = Run(plan, force=True, progress=log)

    assert again.train.variant_ids == [ID_SRC, ID_TGT]
    assert log.outcomes == [(ID_SRC, "trained"), (ID_TGT, "trained")]
    assert [r.reused for r in again.result.variants] == [False, False]


def test_a_failing_training_stops_the_run_and_leaves_the_earlier_variants_complete(
    default_plan: AblationPlan, tmp_path: Path
) -> None:
    plan = _plan_in(default_plan, tmp_path)
    failure = UpstreamError("eole train exited with code 1")
    log = ProgressLog()

    with pytest.raises(UpstreamError, match="eole train exited with code 1") as info:
        Run(plan, train=FakeTrain(fail={ID_TGT: failure}), progress=log)

    assert info.value is failure
    assert log.outcomes == [(ID_SRC, "trained")]
    variants = plan.out_dir / "variants"
    assert (variants / ID_SRC / "result.json").is_file()
    assert not (variants / ID_TGT / "result.json").exists()
    assert not (plan.out_dir / "results.tsv").exists()

    resumed = Run(plan, progress=(again := ProgressLog()))

    assert resumed.train.variant_ids == [ID_TGT]
    assert resumed.evaluate.variant_ids == [ID_TGT]
    assert again.outcomes == [(ID_SRC, "reused"), (ID_TGT, "trained")]


def test_a_failing_evaluation_leaves_no_result_and_the_variant_trains_again(
    default_plan: AblationPlan, tmp_path: Path
) -> None:
    plan = _plan_in(default_plan, tmp_path)
    failure = DataError("no dev lines")

    with pytest.raises(DataError) as info:
        Run(plan, evaluate=FakeEvaluate(fail={ID_SRC: failure}))

    assert info.value is failure
    assert not (plan.out_dir / "variants" / ID_SRC / "result.json").exists()
    assert not (plan.out_dir / "variants" / ID_TGT).exists()

    resumed = Run(plan)

    assert resumed.train.variant_ids == [ID_SRC, ID_TGT]


@pytest.mark.parametrize(
    "error",
    [
        UpstreamError("eole is not installed"),
        ConfigError("bad config"),
        DataError("bad data"),
        OSError("disk full"),
        RuntimeError("bug"),
    ],
)
def test_an_error_of_training_or_evaluation_reaches_the_caller_as_it_is(
    default_plan: AblationPlan, tmp_path: Path, error: Exception
) -> None:
    plan = _plan_in(default_plan, tmp_path)

    with pytest.raises(type(error)) as from_train:
        Run(plan, train=FakeTrain(fail={ID_SRC: error}))
    with pytest.raises(type(error)) as from_evaluate:
        Run(plan, evaluate=FakeEvaluate(fail={ID_SRC: error}))

    assert from_train.value is error
    assert from_evaluate.value is error


@pytest.mark.parametrize(
    "scored",
    [
        dev_result({EN_MN: 80.0}),
        dev_result(dict.fromkeys(DIRECTIONS, 80.0), lines=99),
        dev_result({**dict.fromkeys(DIRECTIONS, 80.0), "eng_Latn-rus_Cyrl": 80.0}),
    ],
    ids=["a direction missing", "another number of lines", "a direction too many"],
)
def test_an_evaluation_that_does_not_score_the_dev_sets_of_the_plan_is_an_upstream_error(
    default_plan: AblationPlan, tmp_path: Path, scored: DevResult
) -> None:
    plan = _plan_in(default_plan, tmp_path)

    with pytest.raises(UpstreamError, match=ID_SRC):
        Run(plan, evaluate=FakeEvaluate({ID_SRC: scored}))

    assert not (plan.out_dir / "variants" / ID_SRC / "result.json").exists()


def test_an_interrupt_stops_the_run_and_leaves_no_result_for_the_variant(
    default_plan: AblationPlan, tmp_path: Path
) -> None:
    plan = _plan_in(default_plan, tmp_path)
    evaluate = FakeEvaluate()

    with pytest.raises(KeyboardInterrupt):
        Run(
            plan, train=FakeTrain(fail={ID_SRC: KeyboardInterrupt()}), evaluate=evaluate
        )

    assert not evaluate.calls
    assert not (plan.out_dir / "variants" / ID_SRC / "result.json").exists()


def test_a_forced_rerun_that_fails_leaves_no_result_of_the_earlier_run(
    default_plan: AblationPlan, tmp_path: Path
) -> None:
    plan = _plan_in(default_plan, tmp_path)
    Run(plan)

    with pytest.raises(UpstreamError):
        Run(plan, force=True, train=FakeTrain(fail={ID_SRC: UpstreamError("x")}))

    variants = plan.out_dir / "variants"
    assert not (variants / ID_SRC / "result.json").exists()
    assert (variants / ID_TGT / "result.json").is_file()


def test_a_training_that_leaves_no_checkpoint_file_is_an_upstream_error(
    default_plan: AblationPlan, tmp_path: Path
) -> None:
    plan = _plan_in(default_plan, tmp_path)
    evaluate = FakeEvaluate()

    with pytest.raises(UpstreamError, match=CHECKPOINT_FILE):
        Run(plan, train=FakeTrain(write_checkpoint=False), evaluate=evaluate)

    assert not evaluate.calls
    assert not (plan.out_dir / "variants" / ID_SRC / "result.json").exists()


def test_an_output_directory_that_cannot_be_made_is_an_os_error(
    default_plan: AblationPlan, tmp_path: Path
) -> None:
    blocker = tmp_path / "blocker"
    blocker.write_text("a file, not a directory", encoding="utf-8")
    plan = dataclasses.replace(default_plan, out_dir=blocker / "out")
    train = FakeTrain()

    with pytest.raises(OSError):
        Run(plan, train=train)

    assert not train.calls


# --- run_ablation: what makes a variant complete -------------------------------------


def test_a_changed_bt_row_reruns_only_the_variants_that_use_bt_rows(
    corpus: Path, toy_spm: Path, tmp_path: Path
) -> None:
    axes = {"temperature": [0.5], "bt_ratio": [0.0, 3.0], "tag_placement": ["src"]}
    first = toy_plan(tmp_path, toy_spm, corpus=corpus, axes=axes)
    Run(first)
    changed = write_toy_corpus(tmp_path / "changed", changed_bt_row=True)
    second = plan_ablation(make_matrix(axes=axes), [changed], toy_spm, first.out_dir)

    run = Run(second)

    assert run.train.variant_ids == ["t0.5_bt3_src_s1234"]


def test_another_sentencepiece_model_reruns_every_variant(
    default_plan: AblationPlan, corpus: Path, other_spm: Path, tmp_path: Path
) -> None:
    plan = _plan_in(default_plan, tmp_path)
    Run(plan)
    changed = plan_ablation(make_matrix(), [corpus], other_spm, plan.out_dir)

    run = Run(changed)

    assert run.train.variant_ids == [ID_SRC, ID_TGT]


def test_a_changed_override_reruns_every_variant(
    corpus: Path, toy_spm: Path, tmp_path: Path
) -> None:
    first = toy_plan(tmp_path, toy_spm, corpus=corpus)
    Run(first)
    second = plan_ablation(
        make_matrix(overrides={"training": {"dropout": [0.1]}}),
        [corpus],
        toy_spm,
        first.out_dir,
    )

    run = Run(second)

    assert run.train.variant_ids == [ID_SRC, ID_TGT]
    assert "0.1" in run.train.calls[0].config_text


def test_another_thread_count_reruns_every_variant(
    default_plan: AblationPlan, tmp_path: Path
) -> None:
    plan = _plan_in(default_plan, tmp_path)
    Run(plan, threads=2)

    run = Run(plan, threads=3)

    assert run.train.variant_ids == [ID_SRC, ID_TGT]
    assert {call.threads for call in run.train.calls} == {3}
    assert {call.threads for call in run.evaluate.calls} == {3}
    assert run.result.threads == 3
    document = json.loads((plan.out_dir / "results.json").read_text(encoding="utf-8"))
    assert document["threads"] == 3
    assert Run(plan, threads=3).train.calls == []


def test_another_eole_version_reruns_every_variant(
    default_plan: AblationPlan, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan = _plan_in(default_plan, tmp_path)
    Run(plan)
    monkeypatch.setattr(ablation, "tool_versions", lambda packages: {"eole": "9.9.9"})

    run = Run(plan)

    assert run.train.variant_ids == [ID_SRC, ID_TGT]


def test_a_different_timeout_does_not_rerun_a_complete_variant(
    default_plan: AblationPlan, tmp_path: Path
) -> None:
    plan = _plan_in(default_plan, tmp_path)
    Run(plan, timeout=30.0)

    run = Run(plan, timeout=None)

    assert run.train.calls == []


def test_a_modified_checkpoint_reruns_exactly_its_variant(
    default_plan: AblationPlan, tmp_path: Path
) -> None:
    plan = _plan_in(default_plan, tmp_path)
    first = Run(plan)
    flip_a_byte(first.result.variants[0].checkpoint_path)

    run = Run(plan)

    assert run.train.variant_ids == [ID_SRC]
    assert Run(plan).train.calls == []


@pytest.mark.parametrize("how", ["file", "directory", "empty"])
def test_a_missing_checkpoint_reruns_exactly_its_variant(
    default_plan: AblationPlan, tmp_path: Path, how: str
) -> None:
    plan = _plan_in(default_plan, tmp_path)
    first = Run(plan)
    checkpoint = first.result.variants[1].checkpoint_path
    if how == "file":
        checkpoint.unlink()
    elif how == "directory":
        shutil.rmtree(checkpoint.parent)
    else:
        checkpoint.write_bytes(b"")

    run = Run(plan)

    assert run.train.variant_ids == [ID_TGT]


def test_a_missing_result_file_reruns_exactly_its_variant(
    default_plan: AblationPlan, tmp_path: Path
) -> None:
    plan = _plan_in(default_plan, tmp_path)
    first = Run(plan)
    first.result_path(ID_SRC).unlink()

    run = Run(plan)

    assert run.train.variant_ids == [ID_SRC]


def _tamper(path: Path, change: Callable[[dict[str, Any]], Any]) -> None:
    record = json.loads(path.read_text(encoding="utf-8"))
    path.write_text(json.dumps(change(record)), encoding="utf-8")


def _set(**fields: Any) -> Callable[[dict[str, Any]], dict[str, Any]]:
    return lambda record: {**record, **fields}


def _without(key: str) -> Callable[[dict[str, Any]], dict[str, Any]]:
    return lambda record: {k: v for k, v in record.items() if k != key}


def _dev_with(**fields: Any) -> Callable[[dict[str, Any]], dict[str, Any]]:
    return lambda record: {**record, "dev": {**record["dev"], **fields}}


def _direction_with(**fields: Any) -> Callable[[dict[str, Any]], dict[str, Any]]:
    def change(record: dict[str, Any]) -> dict[str, Any]:
        directions = {k: dict(v) for k, v in record["dev"]["directions"].items()}
        directions[EN_MN].update(fields)
        return {**record, "dev": {**record["dev"], "directions": directions}}

    return change


def _drop_a_direction(record: dict[str, Any]) -> dict[str, Any]:
    directions = dict(record["dev"]["directions"])
    del directions[EN_MN]
    return {**record, "dev": {**record["dev"], "directions": directions}}


def _add_a_direction(record: dict[str, Any]) -> dict[str, Any]:
    directions = dict(record["dev"]["directions"])
    directions["eng_Latn-rus_Cyrl"] = directions[EN_MN]
    return {**record, "dev": {**record["dev"], "directions": directions}}


UNTRUSTED_CHANGES: list[tuple[str, Callable[[dict[str, Any]], Any]]] = [
    ("another fingerprint", _set(fingerprint="0" * 64)),
    ("fingerprint of the wrong type", _set(fingerprint=5)),
    ("no fingerprint", _without("fingerprint")),
    ("status failed", _set(status="failed")),
    ("status true", _set(status=True)),
    ("no status", _without("status")),
    ("another checkpoint hash", _set(checkpoint_sha256="f" * 64)),
    ("checkpoint hash of the wrong type", _set(checkpoint_sha256=7)),
    ("no checkpoint hash", _without("checkpoint_sha256")),
    ("train seconds as text", _set(train_seconds="1.5")),
    ("negative train seconds", _set(train_seconds=-1.0)),
    ("train seconds a boolean", _set(train_seconds=True)),
    ("no train seconds", _without("train_seconds")),
    ("no eval seconds", _without("eval_seconds")),
    ("negative eval seconds", _set(eval_seconds=-0.5)),
    ("no dev result", _without("dev")),
    ("dev result a list", _set(dev=[])),
    ("dev result null", _set(dev=None)),
    ("leaks as text", _dev_with(tag_leaks="0")),
    ("leaks a boolean", _dev_with(tag_leaks=True)),
    ("leaks a float", _dev_with(tag_leaks=0.5)),
    ("chrf mean null", _dev_with(chrf_mean=None)),
    ("chrf min missing", _dev_with(chrf_min=None)),
    ("signature of the wrong type", _dev_with(chrf_signature=3)),
    ("chrf mean not a number", _dev_with(chrf_mean=float("nan"))),
    ("chrf in a direction infinite", _direction_with(chrf=float("inf"))),
    ("train seconds infinite", _set(train_seconds=float("inf"))),
    ("eval seconds not a number", _set(eval_seconds=float("nan"))),
    ("chrf as text", _direction_with(chrf="80.0")),
    ("bleu a boolean", _direction_with(bleu=True)),
    ("lines as a float", _direction_with(lines=100.5)),
    ("another number of lines", _direction_with(lines=99)),
    ("a direction missing", _drop_a_direction),
    ("a direction too many", _add_a_direction),
    ("a list", lambda record: [record]),
    ("null", lambda record: None),
    ("text", lambda record: "ok"),
    ("a number", lambda record: 5),
    ("an empty object", lambda record: {}),
    ("another schema", lambda record: {"status": "ok", "fingerprint": "abc"}),
]


@pytest.mark.parametrize(
    "change",
    [item[1] for item in UNTRUSTED_CHANGES],
    ids=[i[0] for i in UNTRUSTED_CHANGES],
)
def test_a_result_file_that_does_not_vouch_for_the_variant_is_never_trusted(
    default_plan: AblationPlan, tmp_path: Path, change: Callable[[dict[str, Any]], Any]
) -> None:
    plan = _plan_in(default_plan, tmp_path)
    first = Run(plan)
    _tamper(first.result_path(ID_TGT), change)

    run = Run(plan)

    assert run.train.variant_ids == [ID_TGT]
    assert (
        json.loads(run.result_path(ID_TGT).read_text(encoding="utf-8"))["status"]
        == "ok"
    )
    assert Run(plan).train.calls == []


def test_a_result_file_with_unknown_extra_keys_is_still_trusted(
    default_plan: AblationPlan, tmp_path: Path
) -> None:
    plan = _plan_in(default_plan, tmp_path)
    first = Run(plan)
    _tamper(first.result_path(ID_TGT), _set(note="added by hand"))

    assert Run(plan).train.calls == []


@pytest.mark.parametrize(
    "content",
    [
        b"",
        b"{",
        b"not json",
        b"\xff\xfe\x00\x01",
        b"[" * 100_000,
        b'{"status": "ok", "fingerprint": NaN}',
        b"\x00" * 64,
    ],
    ids=["empty", "open brace", "text", "not utf-8", "deeply nested", "nan", "nul"],
)
def test_a_result_file_that_is_not_json_is_never_trusted_and_never_a_crash(
    default_plan: AblationPlan, tmp_path: Path, content: bytes
) -> None:
    plan = _plan_in(default_plan, tmp_path)
    first = Run(plan)
    first.result_path(ID_SRC).write_bytes(content)

    run = Run(plan)

    assert run.train.variant_ids == [ID_SRC]


def test_a_truncated_result_file_is_never_trusted(
    default_plan: AblationPlan, tmp_path: Path
) -> None:
    plan = _plan_in(default_plan, tmp_path)
    first = Run(plan)
    text = first.result_path(ID_SRC).read_bytes()
    first.result_path(ID_SRC).write_bytes(text[: len(text) // 2])

    assert Run(plan).train.variant_ids == [ID_SRC]


def test_a_directory_in_place_of_the_result_file_is_an_os_error_not_a_reuse(
    default_plan: AblationPlan, tmp_path: Path
) -> None:
    plan = _plan_in(default_plan, tmp_path)
    first = Run(plan)
    path = first.result_path(ID_SRC)
    path.unlink()
    path.mkdir()

    with pytest.raises(OSError):
        Run(plan)


# --- run_ablation: the tables --------------------------------------------------------


def test_the_run_writes_the_three_result_files_under_the_out_dir(
    completed: Run,
) -> None:
    for name in ("results.tsv", "results.json", "results.md"):
        assert (completed.plan.out_dir / name).is_file()
    assert not list(completed.plan.out_dir.rglob("*.part"))


def test_the_result_files_are_written_again_when_every_variant_is_reused(
    default_plan: AblationPlan, tmp_path: Path
) -> None:
    plan = _plan_in(default_plan, tmp_path)
    Run(plan)
    names = ("results.tsv", "results.json", "results.md")
    before = {name: (plan.out_dir / name).read_bytes() for name in names}
    for name in names:
        (plan.out_dir / name).unlink()

    again = Run(plan)

    assert again.train.calls == []
    assert {name: (plan.out_dir / name).read_bytes() for name in names} == before


def test_a_failure_to_write_the_tables_is_an_os_error_and_keeps_the_variants(
    default_plan: AblationPlan, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan = _plan_in(default_plan, tmp_path)

    def fail(path: Path, content: bytes) -> None:
        raise OSError("disk full")

    with monkeypatch.context() as patch:
        patch.setattr("rinchen.train.ablation_report.replace_file", fail)
        with pytest.raises(OSError, match="disk full"):
            Run(plan)

    assert Run(plan).train.calls == []


def test_the_checkpoint_hashes_are_exposed_for_the_manifest(completed: Run) -> None:
    hashes = {
        r.planned.variant.id: r.checkpoint_sha256 for r in completed.result.variants
    }

    assert hashes[ID_SRC] == _sha256(
        completed.variant_dir(ID_SRC) / "run" / "ckpt" / CHECKPOINT_FILE
    )
    assert hashes[ID_SRC] != hashes[ID_TGT]


def test_the_result_names_the_files_it_wrote(completed: Run) -> None:
    files = completed.result.files

    assert files.tsv == completed.plan.out_dir / "results.tsv"
    assert files.json == completed.plan.out_dir / "results.json"
    assert files.md == completed.plan.out_dir / "results.md"
