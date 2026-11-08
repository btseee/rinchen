"""Tests for `rinchen.train.ablation_layout`: where an ablation keeps its files."""

from __future__ import annotations

from pathlib import Path

from rinchen.train.ablation_layout import (
    CONFIG_FILE,
    DATA_DIR,
    RESULT_FILE,
    TRAIN_FILE,
    VARIANTS_DIR,
    data_dir,
    variant_paths,
)

OUT = Path("/runs/ab")


def test_the_file_names_are_those_the_runbook_and_the_manifest_name() -> None:
    assert (DATA_DIR, VARIANTS_DIR, CONFIG_FILE, RESULT_FILE, TRAIN_FILE) == (
        "data",
        "variants",
        "config.yaml",
        "result.json",
        "train.json",
    )


def test_the_corpus_files_of_every_variant_share_one_directory() -> None:
    assert data_dir(OUT) == Path("/runs/ab/data")


def test_a_variant_keeps_its_files_in_a_directory_of_its_id() -> None:
    paths = variant_paths(OUT, "t0.5_bt0.5_src_s1234")

    variant = Path("/runs/ab/variants/t0.5_bt0.5_src_s1234")
    assert paths.directory == variant
    assert paths.config == variant / "config.yaml"
    assert paths.valid.src == variant / "valid.src"
    assert paths.valid.tgt == variant / "valid.tgt"
    assert paths.run_dir == variant / "run"
    assert paths.vocab == variant / "run" / "vocab.src"
    assert paths.ckpt_dir == variant / "run" / "ckpt"
    assert paths.checkpoint == variant / "run" / "ckpt" / "model.00.safetensors"
    assert paths.eval_dir == variant / "eval"
    assert paths.train_record == variant / "train.json"
    assert paths.result == variant / "result.json"


def test_the_paths_of_two_variants_do_not_overlap() -> None:
    first = variant_paths(OUT, "a")
    second = variant_paths(OUT, "b")

    assert {first.directory, first.run_dir, first.ckpt_dir}.isdisjoint(
        {second.directory, second.run_dir, second.ckpt_dir}
    )


def test_nothing_is_created_by_asking_for_paths(tmp_path: Path) -> None:
    variant_paths(tmp_path / "out", "x")
    data_dir(tmp_path / "out")

    assert list(tmp_path.iterdir()) == []
