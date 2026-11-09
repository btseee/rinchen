"""Tests for what an export checks about itself beyond its hashes (D10).

The record states `vocabulary_entries`, the length of the checkpoint's source vocabulary. The
converter writes `shared_vocabulary.json` from the checkpoint itself, so the two must agree:
the export refuses a converter whose file does not (nothing is left but its log). The reader
that refuses a directory whose record and file disagree moved to `rinchen.export.verified`
(D11) with its tests (`test_export_verified.py`). A checkpoint whose saved config names one
SentencePiece model for the source and another for the target cannot be exported with one
`spm.model`.

The converter is `tests.support.distill_exports.fake_converter`: nothing starts a process.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from rinchen.errors import ConfigError, UpstreamError
from rinchen.export.eole_ct2 import export_log_path
from rinchen.train.distill_export import export_checkpoint
from tests.support.distill_cli import STATED_DIRECTIONS
from tests.support.distill_exports import FakeConverter, fake_converter
from tests.support.step_dirs import (
    SOURCE_VOCABULARY,
    FabricatedRun,
    eole_model_section,
    make_finished_run,
    tree_digest,
    write_spm_model,
)


@pytest.fixture
def teacher(tmp_path: Path) -> FabricatedRun:
    return make_finished_run(
        tmp_path / "teacher",
        model=eole_model_section(encoder_layers=2, decoder_layers=1),
    )


@pytest.fixture
def spm_model(tmp_path: Path) -> Path:
    return write_spm_model(tmp_path / "spm")


@pytest.fixture
def out_dir(tmp_path: Path) -> Path:
    return tmp_path / "export"


def vocabulary_of(count: int) -> bytes:
    return json.dumps([f"piece{index}" for index in range(count)]).encode("utf-8")


# ---- at export time ---------------------------------------------------------------------


@pytest.mark.parametrize("count", [0, 1, 123, 125, 1000])
def test_a_converter_whose_vocabulary_has_another_length_is_refused_with_nothing_left(
    teacher: FabricatedRun,
    spm_model: Path,
    out_dir: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    count: int,
) -> None:
    fake_converter(
        monkeypatch, replace={"shared_vocabulary.json": vocabulary_of(count)}
    )
    before = {path.name for path in tmp_path.iterdir()}
    digest = tree_digest(teacher.run_dir)

    with pytest.raises(UpstreamError) as raised:
        export_checkpoint(
            teacher.final,
            out_dir,
            spm_model=spm_model,
            tag_placement="src",
            directions=STATED_DIRECTIONS,
        )

    message = str(raised.value)
    assert f"{out_dir / 'shared_vocabulary.json'} has {count} entries" in message
    assert f"vocabulary has {len(SOURCE_VOCABULARY)}:" in message
    assert {path.name for path in tmp_path.iterdir()} - before == {"export.export.log"}
    assert export_log_path(out_dir).is_file()
    assert tree_digest(teacher.run_dir) == digest


@pytest.mark.parametrize(
    "content",
    [b"not json", b'{"a": 1}', b'["a", 2]', b""],
    ids=["not json", "object", "non-string", "empty"],
)
def test_a_converter_vocabulary_that_is_not_a_list_of_strings_is_refused(
    teacher: FabricatedRun,
    spm_model: Path,
    out_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
    content: bytes,
) -> None:
    fake_converter(monkeypatch, replace={"shared_vocabulary.json": content})

    with pytest.raises(UpstreamError) as raised:
        export_checkpoint(
            teacher.final,
            out_dir,
            spm_model=spm_model,
            tag_placement="src",
            directions=STATED_DIRECTIONS,
        )

    assert f"{out_dir / 'shared_vocabulary.json'} is not a JSON list of strings" in str(
        raised.value
    )
    assert not out_dir.exists()


def test_a_converter_vocabulary_of_the_right_length_is_exported(
    teacher: FabricatedRun,
    spm_model: Path,
    out_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_converter(monkeypatch)

    record = export_checkpoint(
        teacher.final,
        out_dir,
        spm_model=spm_model,
        tag_placement="src",
        directions=STATED_DIRECTIONS,
    ).record

    assert record.vocabulary_entries == len(SOURCE_VOCABULARY)
    assert len(json.loads((out_dir / "shared_vocabulary.json").read_text("utf-8"))) == (
        record.vocabulary_entries
    )


# ---- one SentencePiece model for both sides --------------------------------------------------


def test_a_checkpoint_with_two_sentencepiece_models_is_refused_before_the_converter(
    tmp_path: Path, spm_model: Path, out_dir: Path
) -> None:
    run = make_finished_run(tmp_path / "two", tgt_spm_name="other.model")
    converter = FakeConverter()
    digest = tree_digest(run.run_dir)

    with pytest.MonkeyPatch.context() as patched:
        patched.setattr("rinchen.export.ct2.export_ct2", converter)
        with pytest.raises(ConfigError) as raised:
            export_checkpoint(
                run.final,
                out_dir,
                spm_model=spm_model,
                tag_placement="src",
                directions=STATED_DIRECTIONS,
            )

    message = str(raised.value)
    assert str(run.final / "config.json") in message
    assert "${MODEL_PATH}/toy.model" in message
    assert "${MODEL_PATH}/other.model" in message
    assert converter.calls == []
    assert not out_dir.exists()
    assert tree_digest(run.run_dir) == digest
