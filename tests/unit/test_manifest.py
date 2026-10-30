"""Tests for the manifest model, store, and hash-based stage skip mechanism."""
from pathlib import Path

from rinchen.manifest import stage, store
from rinchen.manifest.model import Manifest


def _write_manifest(stage_dir: Path, config_sha256: str, inputs: list[Path]) -> None:
    manifest = Manifest(
        stage="test_stage",
        created="2026-10-30T00:00:00Z",
        code_version="deadbeef",
        config_sha256=config_sha256,
        seed=0,
        inputs=stage.build_inputs(inputs),
    )
    store.write(stage_dir, manifest)


def test_store_round_trips(tmp_path):
    stage_dir = tmp_path / "stage"
    manifest = Manifest(
        stage="demo",
        created="2026-10-30T00:00:00Z",
        code_version="deadbeef",
        config_sha256="cfg-abc",
        seed=0,
    )

    path = store.write(stage_dir, manifest)
    assert path.is_file()

    loaded = store.read(stage_dir)
    assert loaded is not None
    assert loaded.stage == "demo"
    assert loaded.schema == "rinchen.manifest/1"


def test_store_read_returns_none_when_missing(tmp_path):
    assert store.read(tmp_path / "nonexistent") is None


def test_identical_inputs_and_config_are_skipped(tmp_path):
    stage_dir = tmp_path / "stage"
    input_file = tmp_path / "input.txt"
    input_file.write_text("hello", encoding="utf-8")

    _write_manifest(stage_dir, "cfg-abc", [input_file])

    assert stage.should_skip(stage_dir, [input_file], "cfg-abc") is True


def test_changed_config_reruns(tmp_path):
    stage_dir = tmp_path / "stage"
    input_file = tmp_path / "input.txt"
    input_file.write_text("hello", encoding="utf-8")

    _write_manifest(stage_dir, "cfg-abc", [input_file])

    assert stage.should_skip(stage_dir, [input_file], "cfg-different") is False


def test_changed_input_contents_reruns(tmp_path):
    stage_dir = tmp_path / "stage"
    input_file = tmp_path / "input.txt"
    input_file.write_text("hello", encoding="utf-8")

    _write_manifest(stage_dir, "cfg-abc", [input_file])
    input_file.write_text("changed", encoding="utf-8")

    assert stage.should_skip(stage_dir, [input_file], "cfg-abc") is False


def test_no_prior_manifest_never_skips(tmp_path):
    stage_dir = tmp_path / "stage"
    input_file = tmp_path / "input.txt"
    input_file.write_text("hello", encoding="utf-8")

    assert stage.should_skip(stage_dir, [input_file], "cfg-abc") is False


def test_force_always_reruns(tmp_path):
    stage_dir = tmp_path / "stage"
    input_file = tmp_path / "input.txt"
    input_file.write_text("hello", encoding="utf-8")

    _write_manifest(stage_dir, "cfg-abc", [input_file])

    assert stage.should_skip(stage_dir, [input_file], "cfg-abc", force=True) is False
