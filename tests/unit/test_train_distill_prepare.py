"""Tests for `rinchen.train.distill_prepare.prepare_student`: the files of a student run (the
original corpora of D7, the KD corpora, the validation corpus cut with the student's
SentencePiece model and the config) and the mixed weights. What it refuses, the dry run, the
rerun and the statistics are in `test_train_distill_prepare_refusals.py`; the inputs and the
numbers of the toy are in `tests.support.planted_student`.
"""

from __future__ import annotations

import os
from fractions import Fraction
from pathlib import Path

import polars as pl
import pytest
import sentencepiece
import yaml

from rinchen.corpus.schema import corpus_frame
from rinchen.errors import ConfigError
from rinchen.train.corpus_files import build_corpora
from rinchen.train.distill_mix import Mix
from rinchen.train.distill_prepare import PreparedStudent
from rinchen.train.eole_cfg import read_config, validate_train_config
from rinchen.train.mixture import load_training_rows
from tests.support.corpus_rows import EN_MN, MN_EN, MN_RU, RU_MN, file_lines
from tests.support.planted_kd import write_kd_files
from tests.support.planted_student import (
    BASE,
    BT_DIRECTORY,
    DIRECTIONS,
    KD_ROWS,
    KD_WEIGHTS,
    ORIGINAL_WEIGHTS,
    Inputs,
    W,
    build_inputs,
    config_of,
    files_under,
    prepare,
    sha256_hex,
)
from tests.support.tiny_parallel import build_toy


@pytest.fixture(scope="module")
def inputs(tmp_path_factory: pytest.TempPathFactory) -> Inputs:
    return build_inputs(tmp_path_factory.mktemp("distill_prepare_inputs"))


@pytest.fixture(scope="module")
def student_dir(inputs: Inputs, tmp_path_factory: pytest.TempPathFactory) -> Path:
    return tmp_path_factory.mktemp("student") / "student"


@pytest.fixture(scope="module")
def student(inputs: Inputs, student_dir: Path) -> PreparedStudent:
    """The standard run, `1:1`, made once; tests only read what it wrote."""
    return prepare(inputs, student_dir)


# --- what a run writes -----------------------------------------------------------------


def test_the_run_writes_the_d7_corpora_the_kd_corpora_the_validation_files_and_the_config(
    student: PreparedStudent, student_dir: Path
) -> None:
    expected = sorted(
        [
            "config.yaml",
            "valid.src",
            "valid.tgt",
            *(
                f"data/auth/{direction}.auth.{ext}"
                for direction in DIRECTIONS
                for ext in ("src", "tgt", "json")
            ),
            *(
                f"{BT_DIRECTORY}/{direction}.bt.{ext}"
                for direction in DIRECTIONS
                for ext in ("src", "tgt", "json")
            ),
            *(
                f"data/kd/{direction}.kd.{ext}"
                for direction in DIRECTIONS
                for ext in ("src", "tgt", "json")
            ),
        ]
    )

    assert files_under(student_dir) == expected
    assert student.written is True
    assert student.config == student_dir / "config.yaml"
    assert student.run_dir == student_dir / "run"


def test_the_kd_files_hold_the_kd_rows_of_each_direction_sorted_one_line_each_with_lf(
    student: PreparedStudent, student_dir: Path, inputs: Inputs
) -> None:
    kd = pl.read_parquet(inputs.kd)
    for corpus in student.corpora:
        if corpus.group != "kd":
            continue
        direction = corpus.direction
        expected = kd.filter(
            (pl.col("src_lang") + "-" + pl.col("tgt_lang")) == direction
        ).sort("id", "src", "tgt")
        src_path = student_dir / "data" / "kd" / f"{direction}.kd.src"
        tgt_path = student_dir / "data" / "kd" / f"{direction}.kd.tgt"

        assert file_lines(src_path) == expected["src"].to_list()
        assert file_lines(tgt_path) == expected["tgt"].to_list()
        assert b"\r" not in src_path.read_bytes()
        assert corpus.rows == expected.height
        assert corpus.src_path == src_path
        assert corpus.tgt_path == tgt_path


def test_the_kd_corpora_are_named_after_the_direction_with_the_kd_group(
    student: PreparedStudent,
) -> None:
    kd = [corpus for corpus in student.corpora if corpus.group == "kd"]

    assert [corpus.name for corpus in kd] == list(KD_ROWS)
    assert {corpus.name: corpus.rows for corpus in kd} == KD_ROWS
    assert [corpus.direction for corpus in kd] == list(DIRECTIONS)
    assert [corpus.tgt_lang for corpus in kd] == [
        "khk_Cyrl",
        "eng_Latn",
        "rus_Cyrl",
        "khk_Cyrl",
    ]
    assert not any(corpus.reused for corpus in student.corpora)


def test_the_original_corpora_are_the_files_d7_writes_alone(
    student: PreparedStudent, student_dir: Path, inputs: Inputs, tmp_path: Path
) -> None:
    training = load_training_rows([inputs.corpus])
    alone = build_corpora(
        training, bt_ratio=0.5, seed=1234, data_dir=tmp_path / "alone" / "data"
    )

    assert len(alone.corpora) == 8
    for corpus in alone.corpora:
        directory = "auth" if corpus.group == "auth" else "bt/r0.5_s1234"
        for suffix in ("src", "tgt", "json"):
            written = student_dir / "data" / directory / f"{corpus.name}.{suffix}"
            reference = corpus.src_path.parent / f"{corpus.name}.{suffix}"
            assert written.read_bytes() == reference.read_bytes(), written


def test_the_corpora_come_in_the_order_of_the_data_section(
    student: PreparedStudent,
) -> None:
    assert [corpus.name for corpus in student.corpora] == [
        *ORIGINAL_WEIGHTS,
        *KD_WEIGHTS,
    ]
    assert [(size.name, size.direction, size.rows) for size in student.sizes] == [
        (corpus.name, corpus.direction, corpus.rows) for corpus in student.corpora
    ]


def test_the_validation_files_hold_one_line_per_dev_row_cut_with_the_student_model(
    student: PreparedStudent, student_dir: Path
) -> None:
    assert student.valid_lines == 400
    assert len(file_lines(student_dir / "valid.src")) == 400
    assert len(file_lines(student_dir / "valid.tgt")) == 400
    assert file_lines(student_dir / "valid.src")[0] == (
        f"<2khk_Cyrl> {W}The {W}door {W}and {W}the {W}road ."
    )
    assert file_lines(student_dir / "valid.tgt")[0] == f"{W}Хаалга {W}ба {W}зам ."


# --- the weights -----------------------------------------------------------------------


def test_one_to_one_weights_are_the_d7_weights_and_the_sum_of_each_direction_for_kd(
    student: PreparedStudent,
) -> None:
    assert student.weights == {**ORIGINAL_WEIGHTS, **KD_WEIGHTS}
    assert list(student.weights) == [*ORIGINAL_WEIGHTS, *KD_WEIGHTS]
    assert sum(student.weights.values()) == 200


def test_the_repeat_factors_say_how_much_more_often_a_kd_row_is_drawn(
    student: PreparedStudent,
) -> None:
    # (28 / 1200) / (28 / 1500) and (22 / 600) / (22 / 900)
    assert student.repeat == {
        f"{EN_MN}.kd": Fraction(5, 4),
        f"{MN_EN}.kd": Fraction(5, 4),
        f"{MN_RU}.kd": Fraction(3, 2),
        f"{RU_MN}.kd": Fraction(3, 2),
    }


def test_three_to_one_triples_the_kd_weights_and_keeps_the_originals(
    inputs: Inputs, tmp_path: Path
) -> None:
    prepared = prepare(inputs, tmp_path / "out", mix=Mix(3, 1))

    assert prepared.weights == {
        **ORIGINAL_WEIGHTS,
        f"{EN_MN}.kd": 84,
        f"{MN_EN}.kd": 84,
        f"{MN_RU}.kd": 66,
        f"{RU_MN}.kd": 66,
    }
    assert sum(prepared.weights.values()) == 400
    assert prepared.repeat[f"{EN_MN}.kd"] == Fraction(15, 4)
    assert prepared.repeat[f"{MN_RU}.kd"] == Fraction(9, 2)
    config = config_of(tmp_path / "out")
    for name, weight in prepared.weights.items():
        assert config["data"][name]["weight"] == weight


def test_one_to_three_triples_the_originals_and_keeps_the_kd_weights(
    inputs: Inputs, tmp_path: Path
) -> None:
    prepared = prepare(inputs, tmp_path / "out", mix=Mix(1, 3))

    assert prepared.weights == {
        **{name: 3 * weight for name, weight in ORIGINAL_WEIGHTS.items()},
        **KD_WEIGHTS,
    }
    # The originals weigh three times more, so a KD row is drawn less often than before:
    # (28 / 1200) / (84 / 1500) and (22 / 600) / (66 / 900).
    assert prepared.repeat == {
        f"{EN_MN}.kd": Fraction(5, 12),
        f"{MN_EN}.kd": Fraction(5, 12),
        f"{MN_RU}.kd": Fraction(1, 2),
        f"{RU_MN}.kd": Fraction(1, 2),
    }


def test_a_mix_that_is_not_reduced_gives_the_weights_of_the_reduced_one(
    inputs: Inputs, tmp_path: Path
) -> None:
    prepared = prepare(inputs, tmp_path / "out", mix=Mix(2, 2))

    assert prepared.weights == {**ORIGINAL_WEIGHTS, **KD_WEIGHTS}


def test_the_weight_total_is_that_of_the_original_weights_before_the_mix(
    inputs: Inputs, tmp_path: Path
) -> None:
    prepared = prepare(inputs, tmp_path / "out", weight_total=200)

    weights = prepared.weights
    assert sum(weights.values()) == 400
    for direction in DIRECTIONS:
        originals = weights[f"{direction}.auth"] + weights[f"{direction}.bt"]
        assert weights[f"{direction}.kd"] == originals


def test_without_back_translation_the_kd_weights_follow_the_authentic_weights(
    inputs: Inputs, tmp_path: Path
) -> None:
    prepared = prepare(inputs, tmp_path / "out", bt_ratio=0.0)

    assert prepared.weights == {
        f"{EN_MN}.auth": 29,
        f"{MN_EN}.auth": 29,
        f"{MN_RU}.auth": 21,
        f"{RU_MN}.auth": 21,
        f"{EN_MN}.kd": 29,
        f"{MN_EN}.kd": 29,
        f"{MN_RU}.kd": 21,
        f"{RU_MN}.kd": 21,
    }
    assert not any(corpus.group == "bt" for corpus in prepared.corpora)
    assert not (tmp_path / "out" / "data" / "bt").exists()
    assert set(config_of(tmp_path / "out")["data"]) == {*prepared.weights, "valid"}


def test_a_direction_without_kd_rows_has_no_kd_corpus_and_keeps_its_share(
    inputs: Inputs, tmp_path: Path
) -> None:
    toy = build_toy()
    only_en_mn = [
        {**row, "origin": "ft", "corpus": "ft:toy"}
        for row in toy.train
        if f"{row['src_lang']}-{row['tgt_lang']}" == EN_MN
    ]
    kd = write_kd_files(tmp_path / "kd", corpus_frame(only_en_mn))

    prepared = prepare(inputs, tmp_path / "out", kd_files=[kd])

    # The other directions have no KD, so their originals are scaled by 1 + 1 like every
    # direction's total: their share of the draws stays the temperature's.
    assert prepared.weights == {
        f"{EN_MN}.auth": 22,
        f"{EN_MN}.bt": 6,
        f"{MN_EN}.auth": 44,
        f"{MN_EN}.bt": 12,
        f"{MN_RU}.auth": 30,
        f"{MN_RU}.bt": 14,
        f"{RU_MN}.auth": 30,
        f"{RU_MN}.bt": 14,
        f"{EN_MN}.kd": 28,
    }
    assert sum(prepared.weights.values()) == 200
    assert set(prepared.repeat) == {f"{EN_MN}.kd"}


# --- the config -----------------------------------------------------------------------


def test_the_config_is_valid_and_names_the_run_files_the_corpora_and_the_vocabulary_size(
    student: PreparedStudent, student_dir: Path, inputs: Inputs
) -> None:
    config = config_of(student_dir)

    validate_train_config(config)
    assert config["seed"] == 1234
    assert config["src_vocab_size"] == 641
    assert config["src_vocab"] == (student_dir / "run" / "vocab.src").as_posix()
    assert config["training"]["model_path"] == (student_dir / "run" / "ckpt").as_posix()
    assert config["transforms_configs"]["sentencepiece"] == {
        "src_subword_model": inputs.spm.as_posix(),
        "tgt_subword_model": inputs.spm.as_posix(),
    }
    assert config["data"]["valid"] == {
        "path_src": (student_dir / "valid.src").as_posix(),
        "path_tgt": (student_dir / "valid.tgt").as_posix(),
    }
    assert list(config["data"]) == [*ORIGINAL_WEIGHTS, *KD_WEIGHTS, "valid"]


def test_the_student_shape_and_the_recipe_of_the_base_are_kept(
    student: PreparedStudent, student_dir: Path
) -> None:
    config = config_of(student_dir)
    base = read_config(BASE, "base")

    assert config["model"] == base["model"]
    assert config["model"]["encoder"] == {"layers": 2}
    assert config["model"]["decoder"] == {"layers": 1}
    assert config["training"]["train_steps"] == 2000
    assert config["transforms_configs"]["filtertoolong"] == {
        "src_seq_length": 64,
        "tgt_seq_length": 64,
    }


def test_every_kd_corpus_has_the_ft_tag_after_the_direction_tag(
    student: PreparedStudent, student_dir: Path
) -> None:
    data = config_of(student_dir)["data"]

    assert {name: data[name]["src_prefix"] for name in KD_WEIGHTS} == {
        f"{EN_MN}.kd": "<2khk_Cyrl> <ft>",
        f"{MN_EN}.kd": "<2eng_Latn> <ft>",
        f"{MN_RU}.kd": "<2rus_Cyrl> <ft>",
        f"{RU_MN}.kd": "<2khk_Cyrl> <ft>",
    }
    assert all(data[name]["tgt_prefix"] == "" for name in KD_WEIGHTS)
    assert data[f"{EN_MN}.auth"]["src_prefix"] == "<2khk_Cyrl>"
    assert data[f"{EN_MN}.bt"]["src_prefix"] == "<2khk_Cyrl> <bt>"
    for name, entry in data.items():
        if name != "valid":
            assert entry["transforms"] == ["sentencepiece", "prefix", "filtertoolong"]


def test_a_config_holds_neither_train_from_nor_reset_optim(
    student: PreparedStudent, student_dir: Path
) -> None:
    text = (student_dir / "config.yaml").read_text(encoding="utf-8")

    assert "train_from" not in text
    assert "reset_optim" not in text
    assert "update_vocab" not in text


def test_the_base_is_left_as_it_was(inputs: Inputs, tmp_path: Path) -> None:
    before = BASE.read_bytes()

    prepare(inputs, tmp_path / "out")

    assert BASE.read_bytes() == before


def test_without_the_kd_tag_the_kd_corpora_are_prefixed_as_authentic_ones(
    inputs: Inputs, tmp_path: Path
) -> None:
    prepare(inputs, tmp_path / "out", kd_tag=False)

    data = config_of(tmp_path / "out")["data"]
    assert data[f"{EN_MN}.kd"]["src_prefix"] == "<2khk_Cyrl>"
    assert data[f"{MN_EN}.kd"]["src_prefix"] == "<2eng_Latn>"
    assert "<ft>" not in (tmp_path / "out" / "config.yaml").read_text(encoding="utf-8")


def test_the_target_placement_puts_the_ft_tag_alone_in_the_source_prefix(
    inputs: Inputs, tmp_path: Path
) -> None:
    prepare(inputs, tmp_path / "out", placement="tgt")

    data = config_of(tmp_path / "out")["data"]
    assert data[f"{EN_MN}.kd"]["src_prefix"] == "<ft>"
    assert data[f"{EN_MN}.kd"]["tgt_prefix"] == "<2khk_Cyrl>"
    assert data[f"{EN_MN}.auth"]["src_prefix"] == ""
    assert file_lines(tmp_path / "out" / "valid.tgt")[0] == (
        f"<2khk_Cyrl> {W}Хаалга {W}ба {W}зам ."
    )
    assert file_lines(tmp_path / "out" / "valid.src")[0] == (
        f"{W}The {W}door {W}and {W}the {W}road ."
    )


def test_a_base_without_filtertoolong_gives_corpora_without_that_transform(
    inputs: Inputs, tmp_path: Path
) -> None:
    base = read_config(BASE, "base")
    del base["transforms_configs"]["filtertoolong"]
    plain = tmp_path / "base.yaml"
    plain.write_text(yaml.safe_dump(base), encoding="utf-8")

    prepare(inputs, tmp_path / "out", base=plain)

    data = config_of(tmp_path / "out")["data"]
    assert data[f"{EN_MN}.kd"]["transforms"] == ["sentencepiece", "prefix"]
    assert data[f"{EN_MN}.auth"]["transforms"] == ["sentencepiece", "prefix"]


def test_the_paths_in_the_config_are_absolute_whatever_the_working_directory(
    inputs: Inputs, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    relative_model = Path("model") / "spm.model"
    relative_model.parent.mkdir()
    relative_model.write_bytes(inputs.spm.read_bytes())

    prepared = prepare(inputs, Path("rel") / "out", spm_model=relative_model)

    config = config_of(tmp_path / "rel" / "out")
    assert prepared.config == tmp_path / "rel" / "out" / "config.yaml"
    assert (
        config["src_vocab"]
        == (tmp_path / "rel" / "out" / "run" / "vocab.src").as_posix()
    )
    assert (
        config["transforms_configs"]["sentencepiece"]["src_subword_model"]
        == (tmp_path / "model" / "spm.model").as_posix()
    )
    for name in ORIGINAL_WEIGHTS:
        assert os.path.isabs(config["data"][name]["path_src"])
    assert os.path.isabs(config["data"]["valid"]["path_src"])


# --- the student's own SentencePiece model ------------------------------------------------


def test_another_student_model_changes_the_validation_files_and_the_vocabulary_size_only(
    inputs: Inputs, student: PreparedStudent, student_dir: Path, tmp_path: Path
) -> None:
    prepared = prepare(inputs, tmp_path / "out", spm_model=inputs.spm500)

    assert file_lines(tmp_path / "out" / "valid.src")[0] == (
        f"<2khk_Cyrl> {W}The {W} d o o r {W}and {W}the {W}road ."
    )
    assert file_lines(tmp_path / "out" / "valid.tgt")[0] == (
        f"{W} Х а а л г а {W}ба {W}зам ."
    )
    assert sha256_hex(tmp_path / "out" / "valid.src") != sha256_hex(
        student_dir / "valid.src"
    )
    assert sha256_hex(tmp_path / "out" / "valid.tgt") != sha256_hex(
        student_dir / "valid.tgt"
    )
    config = config_of(tmp_path / "out")
    assert config["src_vocab_size"] == 501
    assert config["transforms_configs"]["sentencepiece"]["src_subword_model"] == (
        inputs.spm500.as_posix()
    )
    assert prepared.valid_lines == student.valid_lines
    # The corpus files are text: they do not depend on the model.
    assert [(c.name, c.src_sha256, c.tgt_sha256) for c in prepared.corpora] == [
        (c.name, c.src_sha256, c.tgt_sha256) for c in student.corpora
    ]


@pytest.mark.parametrize(("stated", "kept"), [(641, 641), (700, 700), (64001, 64001)])
def test_a_stated_vocabulary_size_that_holds_every_piece_is_kept(
    inputs: Inputs, tmp_path: Path, stated: int, kept: int
) -> None:
    base = read_config(BASE, "base")
    base["src_vocab_size"] = stated
    path = tmp_path / "base.yaml"
    path.write_text(yaml.safe_dump(base), encoding="utf-8")

    prepare(inputs, tmp_path / "out", base=path)

    assert config_of(tmp_path / "out")["src_vocab_size"] == kept


@pytest.mark.parametrize("key", ["src_vocab_size", "tgt_vocab_size"])
def test_a_stated_vocabulary_size_below_the_pieces_plus_one_is_a_config_error_naming_both(
    inputs: Inputs, tmp_path: Path, key: str
) -> None:
    base = read_config(BASE, "base")
    base[key] = 600
    path = tmp_path / "base.yaml"
    path.write_text(yaml.safe_dump(base), encoding="utf-8")

    with pytest.raises(ConfigError, match=rf"{key}.*600.*641|{key}.*641.*600"):
        prepare(inputs, tmp_path / "out", base=path)

    assert not (tmp_path / "out").exists()


def test_a_direction_tag_that_is_not_one_piece_of_the_model_is_a_config_error(
    inputs: Inputs, tmp_path: Path
) -> None:
    model = tmp_path / "plain.model"
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

    with pytest.raises(ConfigError, match=r"<2\w+>.*not one piece"):
        prepare(inputs, tmp_path / "out", spm_model=model)

    assert not (tmp_path / "out").exists()


def test_the_kd_tag_must_be_one_piece_of_the_student_model(
    inputs: Inputs, tmp_path: Path
) -> None:
    with pytest.raises(ConfigError, match=r"<ft>.*not one piece"):
        prepare(inputs, tmp_path / "out", spm_model=inputs.no_ft_model)

    assert not (tmp_path / "out").exists()


def test_without_the_kd_tag_a_model_without_it_is_fine(
    inputs: Inputs, tmp_path: Path
) -> None:
    prepare(inputs, tmp_path / "out", spm_model=inputs.no_ft_model, kd_tag=False)

    config = config_of(tmp_path / "out")
    assert config["src_vocab_size"] == 201
    assert config["data"][f"{EN_MN}.kd"]["src_prefix"] == "<2khk_Cyrl>"
