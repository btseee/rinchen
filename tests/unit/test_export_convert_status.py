"""Tests for the `extra_status` of `rinchen.export.ct2.convert` (D11, Task 2): facts the
caller settled that the manifest of an export carries beside the status `convert` writes
(how the tag placement, the directions and the document level were found out).

The record does not hold them, so the manifest is where a command reports them from; they
travel into the manifest `convert` writes into its private directory, because the output
appears whole, manifest included, or not at all. The converter is
`tests.support.model_exports.fake_convert_converter`: nothing starts a process.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from rinchen.export.ct2 import convert
from rinchen.export.verified import verify_export
from rinchen.manifest import store
from tests.support.model_exports import (
    FakeConvertConverter,
    checkpoint_inputs,
    export_facts,
    fake_convert_converter,
)
from tests.support.step_dirs import SPM_BYTES

CHECKS = {
    "tag_placement_check": "checked",
    "directions_check": "derived",
    "document_level_check": "stated",
}


@pytest.fixture
def converter(monkeypatch: pytest.MonkeyPatch) -> FakeConvertConverter:
    return fake_convert_converter(monkeypatch)


@pytest.fixture
def checkpoint(tmp_path: Path) -> Path:
    path = tmp_path / "run" / "ckpt" / "step_200"
    path.mkdir(parents=True)
    return path


def run_convert(checkpoint: Path, out_dir: Path, **options: object) -> None:
    convert(
        checkpoint,
        out_dir,
        facts=export_facts(),
        spm_bytes=SPM_BYTES,
        inputs=checkpoint_inputs(checkpoint),
        **options,  # type: ignore[arg-type]
    )


def test_the_extra_status_is_in_the_manifest_beside_the_status_of_an_export(
    checkpoint: Path, converter: FakeConvertConverter, tmp_path: Path
) -> None:
    out_dir = tmp_path / "export"

    run_convert(checkpoint, out_dir, extra_status=CHECKS)

    manifest = store.read(out_dir)
    assert manifest is not None
    assert manifest.status == {
        "export": "ok",
        "tag_placement": "src",
        "quantization": "float32",
        "checkpoint_kind": "step",
        **CHECKS,
    }


def test_an_export_with_extra_status_is_one_verify_export_accepts(
    checkpoint: Path, converter: FakeConvertConverter, tmp_path: Path
) -> None:
    run_convert(checkpoint, tmp_path / "export", extra_status=CHECKS)

    assert verify_export(tmp_path / "export").record.quantization == "float32"


def test_no_extra_status_leaves_the_status_as_it_was(
    checkpoint: Path, converter: FakeConvertConverter, tmp_path: Path
) -> None:
    run_convert(checkpoint, tmp_path / "export")

    manifest = store.read(tmp_path / "export")
    assert manifest is not None
    assert manifest.status == {
        "export": "ok",
        "tag_placement": "src",
        "quantization": "float32",
        "checkpoint_kind": "step",
    }


@pytest.mark.parametrize(
    "key", ["export", "tag_placement", "quantization", "checkpoint_kind"]
)
def test_extra_status_cannot_replace_a_status_convert_writes(
    checkpoint: Path, converter: FakeConvertConverter, tmp_path: Path, key: str
) -> None:
    before = {path.name for path in tmp_path.iterdir()}

    with pytest.raises(ValueError, match=key):
        run_convert(checkpoint, tmp_path / "export", extra_status={key: "x"})

    assert converter.calls == []
    assert {path.name for path in tmp_path.iterdir()} == before
