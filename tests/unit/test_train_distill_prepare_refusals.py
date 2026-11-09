"""Tests for `rinchen.train.distill_prepare.prepare_student`, what it refuses before it
writes, the dry run, the rerun and the statistics. The files, the weights and the config of a
run are in `test_train_distill_prepare.py`; the inputs and the numbers of the toy are in
`tests.support.planted_student`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from rinchen.corpus.schema import corpus_frame
from rinchen.errors import ConfigError, DataError
from rinchen.train.distill_mix import Mix
from rinchen.train.distill_prepare import PreparedStudent
from tests.support.corpus_rows import EN_MN, MN_EN, MN_RU, RU_MN
from tests.support.eole_train_config import make_config
from tests.support.planted_kd import write_kd_files
from tests.support.planted_student import (
    DIRECTIONS,
    Inputs,
    base_with,
    build_inputs,
    config_of,
    files_under,
    prepare,
    sha256_hex,
)
from tests.support.tiny_parallel import build_toy, write_corpus_parquet


@pytest.fixture(scope="module")
def inputs(tmp_path_factory: pytest.TempPathFactory) -> Inputs:
    return build_inputs(tmp_path_factory.mktemp("distill_prepare_inputs"))


@pytest.fixture(scope="module")
def student(
    inputs: Inputs, tmp_path_factory: pytest.TempPathFactory
) -> PreparedStudent:
    """The standard run, `1:1`, made once; tests only read what it returned."""
    return prepare(inputs, tmp_path_factory.mktemp("student") / "student")


# --- what it refuses before it writes --------------------------------------------------


@pytest.mark.parametrize(
    ("changes", "named"),
    [
        ({"train_from": "step_100"}, "train_from"),
        ({"reset_optim": "all"}, "reset_optim"),
        ({"data": {"en-mn.auth": make_config()["data"]["en-mn.auth"]}}, "data"),
        ({"data": {"valid": make_config()["data"]["valid"]}}, "data"),
    ],
)
def test_a_base_with_corpora_or_a_checkpoint_to_start_from_is_a_config_error_and_nothing_is_written(
    inputs: Inputs, tmp_path: Path, changes: dict[str, Any], named: str
) -> None:
    base = base_with(tmp_path, **changes)

    with pytest.raises(ConfigError, match=named):
        prepare(inputs, tmp_path / "out", base=base)

    assert not (tmp_path / "out").exists()


def test_a_missing_base_is_a_config_error(inputs: Inputs, tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="base"):
        prepare(inputs, tmp_path / "out", base=tmp_path / "absent.yaml")

    assert not (tmp_path / "out").exists()


def test_a_kd_file_without_its_manifest_is_a_config_error_and_nothing_is_written(
    inputs: Inputs, tmp_path: Path
) -> None:
    kd = write_kd_files(
        tmp_path / "kd",
        corpus_frame([{**row, "origin": "ft"} for row in build_toy().train[:50]]),
    )
    (tmp_path / "kd" / "manifest.json").unlink()

    with pytest.raises(ConfigError, match="manifest"):
        prepare(inputs, tmp_path / "out", kd_files=[kd])

    assert not (tmp_path / "out").exists()


def test_a_tampered_kd_file_is_a_config_error_and_nothing_is_written(
    inputs: Inputs, tmp_path: Path
) -> None:
    kd = write_kd_files(
        tmp_path / "kd",
        corpus_frame([{**row, "origin": "ft"} for row in build_toy().train[:50]]),
    )
    kd.write_bytes(kd.read_bytes() + b"\0")

    with pytest.raises(ConfigError, match="SHA-256"):
        prepare(inputs, tmp_path / "out", kd_files=[kd])

    assert not (tmp_path / "out").exists()


def test_kd_rows_that_are_not_forward_translations_are_counted_and_not_trained_on(
    inputs: Inputs, tmp_path: Path
) -> None:
    toy = build_toy()
    rows = [{**row, "origin": "ft", "corpus": "ft:toy"} for row in toy.train[:1200]]
    rows += [
        {**row, "origin": "human", "corpus": "ft:toy"} for row in toy.train[1200:1300]
    ]
    rows += [{**row, "origin": "ft", "split": "dev"} for row in toy.dev[:10]]
    kd = write_kd_files(tmp_path / "kd", corpus_frame(rows))

    prepared = prepare(inputs, tmp_path / "out", kd_files=[kd])

    assert prepared.kd_stats.rows_read == 1310
    assert prepared.kd_stats.rows_kept == 1200
    assert prepared.kd_stats.excluded["wrong_origin"] == 100
    assert prepared.kd_stats.excluded["not_train"] == 10
    assert (
        sum(corpus.rows for corpus in prepared.corpora if corpus.group == "kd") == 1200
    )


def test_a_kd_direction_without_an_original_corpus_is_counted_and_not_trained(
    inputs: Inputs, tmp_path: Path
) -> None:
    toy = build_toy()
    in_en_mn = [
        row
        for row in (*toy.train, *toy.bt, *toy.dev)
        if f"{row['src_lang']}-{row['tgt_lang']}" == EN_MN
    ]
    corpus = write_corpus_parquet(in_en_mn, tmp_path / "en_mn.parquet")

    prepared = prepare(inputs, tmp_path / "out", corpora=[corpus])

    assert prepared.kd_stats.excluded["unknown_direction"] == 2400
    assert prepared.kd_stats.rows_kept == 1200
    assert prepared.kd_stats.rows_by_direction == {EN_MN: 1200}
    assert list(prepared.weights) == [
        f"{EN_MN}.auth",
        f"{EN_MN}.bt",
        f"{EN_MN}.kd",
    ]
    assert list(config_of(tmp_path / "out")["data"]) == [*prepared.weights, "valid"]
    assert not (tmp_path / "out" / "data" / "kd" / f"{MN_EN}.kd.src").exists()


def test_kd_rows_that_all_fall_away_are_a_data_error_and_nothing_is_written(
    inputs: Inputs, tmp_path: Path
) -> None:
    rows = [{**row, "origin": "ft", "split": "dev"} for row in build_toy().dev[:20]]
    kd = write_kd_files(tmp_path / "kd", corpus_frame(rows))

    with pytest.raises(DataError, match="no KD rows"):
        prepare(inputs, tmp_path / "out", kd_files=[kd])

    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("what", ["dollar", "glob", "dollar_model"])
def test_a_path_that_eole_would_expand_is_a_config_error_and_nothing_is_written(
    inputs: Inputs, tmp_path: Path, what: str
) -> None:
    arguments: dict[str, Any] = {}
    if what == "dollar":
        arguments["out_dir"] = tmp_path / "$HOME" / "out"
    elif what == "glob":
        arguments["out_dir"] = tmp_path / "run[1]" / "out"
    else:
        model = tmp_path / "$x.model"
        model.write_bytes(inputs.spm.read_bytes())
        arguments["spm_model"] = model
    out_dir = arguments.pop("out_dir", tmp_path / "out")

    with pytest.raises(ConfigError):
        prepare(inputs, out_dir, **arguments)

    assert (
        sorted(path.name for path in tmp_path.iterdir() if path.name != "$x.model")
        == []
    )


def test_a_weight_total_below_the_number_of_corpora_is_a_config_error(
    inputs: Inputs, tmp_path: Path
) -> None:
    with pytest.raises(ConfigError, match="weight_total"):
        prepare(inputs, tmp_path / "out", weight_total=3)

    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize(
    "arguments",
    [
        {"temperature": 0.0},
        {"temperature": float("nan")},
        {"bt_ratio": -1.0},
        {"bt_ratio": float("inf")},
        {"dev_max_lines": 0},
        {"placement": "middle"},
    ],
)
def test_an_option_out_of_range_is_a_value_error_and_nothing_is_written(
    inputs: Inputs, tmp_path: Path, arguments: dict[str, Any]
) -> None:
    with pytest.raises(ValueError):
        prepare(inputs, tmp_path / "out", **arguments)

    assert not (tmp_path / "out").exists()


# --- dry run, rerun --------------------------------------------------------------------


def test_a_dry_run_writes_nothing_not_even_the_output_directory(
    inputs: Inputs, tmp_path: Path, student: PreparedStudent
) -> None:
    parent = tmp_path / "parent"
    parent.mkdir()

    prepared = prepare(inputs, parent / "out", dry_run=True)

    assert list(parent.iterdir()) == []
    assert prepared.written is False
    assert prepared.config is None
    assert prepared.corpora == ()
    assert prepared.run_dir == parent / "out" / "run"
    assert prepared.weights == student.weights
    assert prepared.repeat == student.repeat
    assert prepared.sizes == student.sizes
    assert prepared.kd_stats == student.kd_stats
    assert prepared.training == student.training
    assert prepared.dev == student.dev
    assert prepared.valid_lines == student.valid_lines


def test_a_dry_run_still_refuses_what_a_run_refuses(
    inputs: Inputs, tmp_path: Path
) -> None:
    base = base_with(tmp_path, train_from="step_1")

    with pytest.raises(ConfigError, match="train_from"):
        prepare(inputs, tmp_path / "out", base=base, dry_run=True)
    with pytest.raises(ConfigError, match="<ft>.*not one piece"):
        prepare(inputs, tmp_path / "out", spm_model=inputs.no_ft_model, dry_run=True)
    with pytest.raises(ConfigError, match="641"):
        prepare(
            inputs,
            tmp_path / "out",
            base=base_with(tmp_path, src_vocab_size=600),
            dry_run=True,
        )


def test_a_rerun_with_unchanged_inputs_reuses_every_corpus_file(
    inputs: Inputs, tmp_path: Path
) -> None:
    first = prepare(inputs, tmp_path / "out")
    before = {
        path: sha256_hex(tmp_path / "out" / path)
        for path in files_under(tmp_path / "out")
    }

    second = prepare(inputs, tmp_path / "out")

    assert not any(corpus.reused for corpus in first.corpora)
    assert len(second.corpora) == 12
    assert all(corpus.reused for corpus in second.corpora)
    assert second.weights == first.weights
    after = {
        path: sha256_hex(tmp_path / "out" / path)
        for path in files_under(tmp_path / "out")
    }
    assert after == before


def test_a_rerun_after_a_change_of_the_mix_keeps_the_corpus_files_and_changes_the_weights(
    inputs: Inputs, tmp_path: Path
) -> None:
    prepare(inputs, tmp_path / "out")

    second = prepare(inputs, tmp_path / "out", mix=Mix(3, 1))

    assert all(corpus.reused for corpus in second.corpora)
    assert config_of(tmp_path / "out")["data"][f"{EN_MN}.kd"]["weight"] == 84


# --- the statistics --------------------------------------------------------------------


def test_the_statistics_are_those_of_the_three_readers(
    student: PreparedStudent,
) -> None:
    assert student.training.rows_kept == 4800
    assert student.training.rows_by_corpus == {
        f"{EN_MN}.auth": 1200,
        f"{EN_MN}.bt": 300,
        f"{MN_EN}.auth": 1200,
        f"{MN_EN}.bt": 300,
        f"{MN_RU}.auth": 600,
        f"{MN_RU}.bt": 300,
        f"{RU_MN}.auth": 600,
        f"{RU_MN}.bt": 300,
    }
    assert student.dev.lines_by_direction == dict.fromkeys(DIRECTIONS, 100)
    assert student.dev.dev_in_train == 0
    assert student.kd_stats.rows_read == 3600
    assert student.kd_stats.rows_kept == 3600
    assert student.kd_stats.rows_by_direction == {
        EN_MN: 1200,
        MN_EN: 1200,
        MN_RU: 600,
        RU_MN: 600,
    }
    assert student.kd_stats.rows_by_label == {"ft:toy": 3600}
    assert student.kd_stats.duplicate_ids == 0
