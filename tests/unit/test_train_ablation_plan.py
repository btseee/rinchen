"""Tests for `rinchen.train.ablation_plan`: the plan of an ablation, read from the
corpus files before anything is built or trained, and what it refuses.

The toy corpus is written as Parquet in a temporary directory. The numbers of the
plan are worked out by hand from the toy data (D7 spec R3 to R5, section 5.8):

    train rows      en-mn and mn-en 1,200 each, ru-mn and mn-ru 600 each
    bt rows         300 in each of the four directions
    bt ratio 0.5    min(300, 0.5 * 1,200 = 600) = 300 bt rows in each direction
    temperature 0.5 direction totals 1,500, 1,500, 900, 900 give shares
                    sqrt(1500) : sqrt(900) = 38.73 : 30, i.e. 0.28175 and 0.21825;
                    inside a direction the corpora split by rows (4 : 1 and 2 : 1)
    weights (100)   22 6 22 6 15 7 15 7 (floors 22 5 22 5 14 7 14 7 = 96, the four
                    largest remainders .635 .635 .550 .550 take one each)
"""

import dataclasses
import re
import shutil
from pathlib import Path
from typing import Any

import pytest
import sentencepiece

from rinchen.errors import ConfigError, DataError
from rinchen.train import ablation_plan
from rinchen.train.ablation_plan import AblationPlan, plan_ablation
from rinchen.train.corpus_files import BtUsage, build_corpora
from rinchen.train.mixture import TrainingRows, load_training_rows
from rinchen.train.variants import expand_variants
from tests.support.corpus_rows import EN_MN, MN_EN, MN_RU, RU_MN
from tests.support.fake_ablation import (
    DEV_LINES,
    DIRECTIONS,
    ID_SRC,
    make_matrix,
    one_variant_axes,
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
    # 0.5 * 1,200 authentic rows asks for 600 bt rows, of which 300 exist.
    assert planned.bt_by_direction[EN_MN] == BtUsage(
        bt_available=300, bt_requested=600, bt_used=300
    )
    assert planned.bt_by_direction[RU_MN] == BtUsage(
        bt_available=300, bt_requested=300, bt_used=300
    )
    assert planned.source_collision_rate == 0.0


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


# --- plan_ablation: the config every variant will render ------------------------------
#
# A dry run that passes must not be followed by a refusal for the same input, and a
# real run must fail before it deletes the manifest of an earlier run: so the plan
# checks what the preparation of each variant would render.


@pytest.mark.parametrize(
    ("overrides", "named"),
    [
        ({"training": {"train_stepz": 100}}, "training.train_stepz"),
        ({"trainning": {"train_steps": 100}}, "trainning"),
        ({"model": {"encoder": {"layerz": 3}}}, "model.encoder.layerz"),
        ({"model": {"encoder": {"rotary_theta": 1}}}, "rotary_theta"),
        ({"model": {"embeddings": {"position_encoding_type": "Rotary"}}}, "position"),
        ({"overwrite": False}, "overwrite"),
        ({"training": {"dropout": "$HOME"}}, "$"),
    ],
    ids=[
        "typo in a key",
        "typo in a section",
        "typo deep down",
        "forbidden key",
        "other position encoding",
        "overwrite off",
        "a variable",
    ],
)
def test_an_override_that_makes_the_config_invalid_is_a_config_error_at_plan_time(
    corpus: Path,
    toy_spm: Path,
    tmp_path: Path,
    overrides: dict[str, Any],
    named: str,
) -> None:
    out_dir = tmp_path / "out"

    with pytest.raises(ConfigError, match="invalid Eole config") as info:
        plan_ablation(make_matrix(overrides=overrides), [corpus], toy_spm, out_dir)

    assert named in str(info.value)
    assert not out_dir.exists()


def test_the_config_of_every_variant_is_checked_with_its_own_corpora_and_weights(
    corpus: Path, toy_spm: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[dict[str, Any]] = []
    real = ablation_plan.variant_config

    def recording(matrix: Any, variant: Any, **kwargs: Any) -> dict[str, Any]:
        calls.append({"variant": variant, **kwargs})
        return real(matrix, variant, **kwargs)  # type: ignore[no-any-return]

    monkeypatch.setattr(ablation_plan, "variant_config", recording)
    axes = {"temperature": [0.5, 1.0], "bt_ratio": [0.0, 0.5], "tag_placement": ["tgt"]}
    out_dir = tmp_path / "out"

    plan = plan_ablation(make_matrix(axes=axes), [corpus], toy_spm, out_dir)

    assert [call["variant"] for call in calls] == [p.variant for p in plan.variants]
    for call, planned in zip(calls, plan.variants, strict=True):
        assert call["weights"] == planned.weights
        assert call["spm_model"] == toy_spm.resolve()
        assert call["paths"].directory == (
            out_dir.resolve() / "variants" / planned.variant.id
        )
        assert [c.name for c in call["corpora"]] == [c.name for c in planned.corpora]
        assert all(
            file.src_path.is_relative_to(out_dir.resolve() / "data")
            and file.tgt_path.is_relative_to(out_dir.resolve() / "data")
            for file in call["corpora"]
        )
    # Without bt rows the mixture has fewer corpora: the check sees that variant's.
    assert len(calls[0]["corpora"]) == 4 and len(calls[1]["corpora"]) == 8


def test_an_invalid_config_of_a_later_variant_is_found_too(
    corpus: Path, toy_spm: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    real = ablation_plan.variant_config

    def refuse_the_last(matrix: Any, variant: Any, **kwargs: Any) -> dict[str, Any]:
        if variant.seed == 2:
            raise ConfigError(f"invalid Eole config: seed {variant.seed}")
        return real(matrix, variant, **kwargs)  # type: ignore[no-any-return]

    monkeypatch.setattr(ablation_plan, "variant_config", refuse_the_last)

    with pytest.raises(ConfigError, match="seed 2"):
        plan_ablation(make_matrix(seeds=[1, 2]), [corpus], toy_spm, tmp_path / "out")


def test_the_config_is_checked_with_the_planned_paths_not_the_ones_given(
    corpus: Path, toy_spm: Path, tmp_path: Path
) -> None:
    # The checkpoint of this plan is under `out`, resolved: a relative `..` in the
    # argument must not hide a path the config would hold.
    out_dir = tmp_path / "x" / ".." / "out"

    plan = plan_ablation(make_matrix(), [corpus], toy_spm, out_dir)

    assert plan.out_dir == (tmp_path / "out").resolve()


@pytest.mark.parametrize("what", ["missing", "directory", "not a model"])
def test_a_sentencepiece_model_that_cannot_be_loaded_is_a_config_error_at_plan_time(
    corpus: Path, tmp_path: Path, what: str
) -> None:
    model = tmp_path / "bad.model"
    if what == "directory":
        model.mkdir()
    elif what == "not a model":
        model.write_bytes(b"this is not a SentencePiece model")
    out_dir = tmp_path / "out"

    with pytest.raises(ConfigError, match="SentencePiece model"):
        plan_ablation(make_matrix(), [corpus], model, out_dir)

    assert not out_dir.exists()


def test_a_sentencepiece_model_without_a_direction_tag_is_a_config_error_at_plan_time(
    corpus: Path, tmp_path: Path
) -> None:
    text = tmp_path / "plain.txt"
    text.write_text(
        "".join(f"line {i} of some text\n" for i in range(200)), encoding="utf-8"
    )
    sentencepiece.SentencePieceTrainer.train(
        input=str(text),
        model_prefix=str(tmp_path / "plain"),
        vocab_size=40,
        minloglevel=2,
    )

    with pytest.raises(ConfigError, match=r"direction tag <2[a-z]+_[A-Za-z]+>"):
        plan_ablation(make_matrix(), [corpus], tmp_path / "plain.model", tmp_path / "o")


def test_planning_the_config_creates_no_file_and_no_directory(
    corpus: Path, toy_spm: Path, tmp_path: Path
) -> None:
    before = sorted(tmp_path.rglob("*"))

    plan_ablation(
        make_matrix(overrides={"training": {"dropout": [0.1]}}),
        [corpus],
        toy_spm,
        tmp_path / "deep" / "out",
    )

    assert sorted(tmp_path.rglob("*")) == before


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


@pytest.mark.parametrize("char", ["[", "]", "*", "?"])
def test_a_glob_character_in_the_out_dir_is_a_config_error_at_plan_time(
    corpus: Path, toy_spm: Path, tmp_path: Path, char: str
) -> None:
    # Eole globs `<checkpoint dir>/model.*.safetensors` when it loads the checkpoint, so
    # this directory would train for minutes and then fail to predict (or load the
    # checkpoints of its siblings).
    out_dir = tmp_path / f"out{char}1" / "run"

    with pytest.raises(ConfigError) as info:
        plan_ablation(make_matrix(), [corpus], toy_spm, out_dir)

    assert str(info.value) == (
        f"the output directory {out_dir} holds {char!r}, which Eole reads as a glob "
        "pattern when it loads a checkpoint from it"
    )
    assert not (tmp_path / f"out{char}1").exists()


def test_a_glob_character_in_the_directory_a_relative_out_dir_resolves_to_is_refused(
    corpus: Path, toy_spm: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The Eole config names absolute paths, so the checkpoint directory carries the
    # working directory's `[` although the argument does not.
    working = tmp_path / "work[1]"
    working.mkdir()
    monkeypatch.chdir(working)

    with pytest.raises(ConfigError, match=r"holds '\['") as info:
        plan_ablation(make_matrix(), [corpus], toy_spm, Path("out"))

    assert str(working.resolve() / "out") in str(info.value)


@pytest.mark.parametrize(
    "name", ["out dir", "ünï-хураангуй"], ids=["space", "non-ASCII"]
)
def test_an_out_dir_without_glob_characters_is_planned(
    corpus: Path, toy_spm: Path, tmp_path: Path, name: str
) -> None:
    plan = plan_ablation(make_matrix(), [corpus], toy_spm, tmp_path / name / "out")

    assert plan.out_dir == (tmp_path / name / "out").resolve()
    assert not (tmp_path / name).exists()


def test_a_glob_character_in_the_model_path_is_not_a_problem_for_eole(
    corpus: Path, toy_spm: Path, tmp_path: Path
) -> None:
    # Only the checkpoint directory is globbed; the SentencePiece model is read as a
    # file (and the corpora are read by the repository with `glob=False`).
    odd = tmp_path / "models[1]"
    odd.mkdir()
    spm_model = odd / toy_spm.name
    shutil.copy(toy_spm, spm_model)

    plan = plan_ablation(make_matrix(), [corpus], spm_model, tmp_path / "out")

    assert plan.spm_model == spm_model.resolve()
