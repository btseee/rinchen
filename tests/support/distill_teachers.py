"""A teacher over a fabricated export, for the tests of D10.

`scripted_teacher` is a `TeacherTranslator` over a verified fabricated export
(`tests.support.distill_exports.write_export`), a `ScriptedCt2` answering from a script
and a `FakeProcessor` that knows the direction tags: no model, no process, no GPU. The
other helpers make the pieces a test damages one at a time: an export whose converter
files are replaced, a verified export, the bytes of a converter `config.json`."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from rinchen.train.distill_export import VerifiedExport, verify_export
from rinchen.train.distill_teacher import TeacherOptions, TeacherTranslator
from tests.support import distill_exports
from tests.support.distill_exports import write_export
from tests.support.fake_ct2 import FakeProcessor, Script, ScriptedCt2

SP = chr(0x2581)
ENG, KHK, RUS = "eng_Latn", "khk_Cyrl", "rus_Cyrl"
TAG = "<2khk_Cyrl>"
END = "</s>"


def hyp(*tokens: str, score: float = -0.5) -> tuple[list[str], float]:
    """One scripted hypothesis: its tokens and a score."""
    return (list(tokens), score)


def tag_processor() -> FakeProcessor:
    """A processor in which the tags of English, Mongolian and Russian are one piece."""
    return FakeProcessor([TAG, "<2eng_Latn>", "<2rus_Cyrl>"])


def verified_export(tmp_path: Path, **export_options) -> VerifiedExport:
    """A fabricated export in `tmp_path / "export"`, verified."""
    write_export(tmp_path / "export", **export_options)
    return verify_export(tmp_path / "export")


def export_with_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, **files: bytes
) -> VerifiedExport:
    """A verified export whose converter files include `files` instead of the usual."""
    usual = distill_exports.converter_files

    def converter_files(tokens):
        return {**usual(tokens), **files}

    monkeypatch.setattr(distill_exports, "converter_files", converter_files)
    return verified_export(tmp_path)


def config_bytes(**fields) -> bytes:
    """A converter `config.json` of these fields only."""
    return json.dumps(fields).encode()


def scripted_teacher(
    tmp_path: Path,
    script: Script,
    *,
    placement="src",
    options: TeacherOptions | None = None,
    processor=None,
    **export_options,
) -> tuple[TeacherTranslator, ScriptedCt2]:
    """A teacher of the export `placement` and `export_options` describe, whose CTranslate2
    answers from `script`; the scripted library is returned beside it."""
    export = verified_export(tmp_path, tag_placement=placement, **export_options)
    ct2 = ScriptedCt2(script)
    teacher = TeacherTranslator(
        export,
        options or TeacherOptions(),
        translator=ct2,
        processor=tag_processor() if processor is None else processor,
    )
    return teacher, ct2
