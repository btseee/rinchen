"""Fabricated CTranslate2 exports for the tests of D10, over the format of D11.

D10's tests know a fabricated export by `write_export` and a fake converter by
`fake_converter`; D11 replaced the format (`rinchen.json`, stage `export`) and moved the
converter call into `rinchen.export.ct2`, so both are thin wrappers over
`tests.support.model_exports` that keep D10's keyword API (`tag_placement`, the options of
the record and of the manifest) and add the new fields of the record (`directions`,
`quantization`, `document_level`, `name`). The names D10's tests import from here
(`converter_files`, `MODEL_BYTES`, `sha256_of_bytes`, the digests of a fabricated checkpoint)
live in `model_exports` and are re-exported. Nothing starts a process.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from rinchen.config.ablation import TagPlacement
from tests.support.model_exports import (
    CHECKPOINT_MANIFEST_SHA256,
    CONFIG_SHA256,
    CONVERTER_CONFIG,
    MODEL_BYTES,
    VOCAB_SHA256,
    WEIGHTS_SHA256,
    ConverterCall,
    FakeConvertConverter,
    WrittenModel,
    checkpoint_record,
    converter_files,
    fake_convert_converter,
    sha256_of_bytes,
    write_model_export,
)

__all__ = [
    "CHECKPOINT_MANIFEST_SHA256",
    "CONFIG_SHA256",
    "CONVERTER_CONFIG",
    "MODEL_BYTES",
    "VOCAB_SHA256",
    "WEIGHTS_SHA256",
    "ConverterCall",
    "FakeConverter",
    "WrittenExport",
    "checkpoint_record",
    "converter_files",
    "fake_converter",
    "sha256_of_bytes",
    "write_export",
]

FakeConverter = FakeConvertConverter
WrittenExport = WrittenModel


def fake_converter(
    monkeypatch: pytest.MonkeyPatch, **options: Any
) -> FakeConvertConverter:
    """Install a fake converter (`FakeConvertConverter(**options)`) as the converter of the
    export and return it."""
    return fake_convert_converter(monkeypatch, **options)


def write_export(
    directory: Path, *, tag_placement: TagPlacement = "src", **options: Any
) -> WrittenModel:
    """A complete export in `directory` as D10's tests asked for one: `write_model_export`
    with `tag_placement` as its `placement`, every other keyword as it is."""
    return write_model_export(directory, placement=tag_placement, **options)
