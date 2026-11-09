"""Tests for the facts `export_checkpoint` records beyond D10's (D11, T1, T3, T5 and T10): the
DIRECTIONS the model was trained for, whether it decodes `<doc>` windows, the quantisation and
the name.

The directions and the document level are derived from the saved `config.json` of the
checkpoint (`distill_saved_config.saved_directions`, `saved_document_level`): its corpora are
named `<src>-<tgt>.<group>` and a window corpus has `<doc>` in its `src_prefix`. A caller may
state them for a checkpoint that says nothing (a baseline, a config without a `data` section),
and a statement that contradicts what the config says is refused before the converter starts,
as a wrong tag placement is. Each fact comes back beside the record as `derived` or `stated`
(`ExportOutcome`), because the record cannot say which.

The runs are fabricated by `tests.support.step_dirs` and the converter is
`tests.support.model_exports.fake_convert_converter`: nothing starts a process.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import pytest

from rinchen.errors import ConfigError
from rinchen.manifest import store
from rinchen.train import distill_export
from rinchen.train.distill_export import ExportOutcome, export_checkpoint
from tests.support.model_exports import FakeConvertConverter, fake_convert_converter
from tests.support.step_dirs import (
    SPM_BYTES,
    FabricatedRun,
    eole_model_section,
    launcher_data,
    make_finished_run,
    sha256,
    tree_digest,
    write_spm_model,
)

TOY_CORPORA = (
    ("eng_Latn-khk_Cyrl.auth", "<2khk_Cyrl>", None),
    ("khk_Cyrl-eng_Latn.auth", "<2eng_Latn>", None),
    ("khk_Cyrl-rus_Cyrl.auth", "<2rus_Cyrl>", None),
    ("rus_Cyrl-khk_Cyrl.auth", "<2khk_Cyrl>", None),
)
TOY_DIRECTIONS = (
    "eng_Latn-khk_Cyrl",
    "khk_Cyrl-eng_Latn",
    "khk_Cyrl-rus_Cyrl",
    "rus_Cyrl-khk_Cyrl",
)
MODEL = eole_model_section(encoder_layers=2, decoder_layers=1)


@pytest.fixture
def converter(monkeypatch: pytest.MonkeyPatch) -> FakeConvertConverter:
    return fake_convert_converter(monkeypatch)


@pytest.fixture
def spm_model(tmp_path: Path) -> Path:
    return write_spm_model(tmp_path / "spm")


@pytest.fixture
def out_dir(tmp_path: Path) -> Path:
    return tmp_path / "export"


def run_of(tmp_path: Path, data: Any = None) -> FabricatedRun:
    """A finished run whose saved config holds `data` (the toy teacher's four directions
    when it is not given: pass `{}` for none)."""
    if data is None:
        data = launcher_data("src", TOY_CORPORA)
    return make_finished_run(tmp_path / "teacher", model=MODEL, data=data)


def undecided_run(tmp_path: Path) -> FabricatedRun:
    """A finished run whose saved config has no `data` section: it says nothing."""
    return make_finished_run(tmp_path / "bare", model=MODEL)


def export(
    run: FabricatedRun, out_dir: Path, spm_model: Path, **options: Any
) -> ExportOutcome:
    return export_checkpoint(
        run.final, out_dir, spm_model=spm_model, tag_placement="src", **options
    )


def refused(
    run: FabricatedRun,
    out_dir: Path,
    spm_model: Path,
    converter: FakeConvertConverter,
    *,
    error: type[Exception] = ConfigError,
    **options: Any,
) -> str:
    """The message of the refusal; nothing was made, nothing was started and the run was
    not touched."""
    digest = tree_digest(run.run_dir)
    with pytest.raises(error) as raised:
        export(run, out_dir, spm_model, **options)
    assert not out_dir.exists()
    assert not Path(f"{out_dir}.export.log").exists()
    assert converter.calls == []
    assert tree_digest(run.run_dir) == digest
    return str(raised.value)


# ---- directions ---------------------------------------------------------------------------


def test_the_directions_of_the_saved_corpora_are_recorded_and_reported_as_derived(
    tmp_path: Path, spm_model: Path, converter: FakeConvertConverter, out_dir: Path
) -> None:
    outcome = export(run_of(tmp_path), out_dir, spm_model)

    assert outcome.record.directions == TOY_DIRECTIONS
    assert outcome.record.languages == ("eng_Latn", "khk_Cyrl", "rus_Cyrl")
    assert dict(outcome.record.tags) == {
        "eng_Latn": "<2eng_Latn>",
        "khk_Cyrl": "<2khk_Cyrl>",
        "rus_Cyrl": "<2rus_Cyrl>",
    }
    assert outcome.directions_check == "derived"


def test_back_translated_and_distilled_corpora_add_no_direction(
    tmp_path: Path, spm_model: Path, converter: FakeConvertConverter, out_dir: Path
) -> None:
    corpora = (
        *TOY_CORPORA,
        ("eng_Latn-khk_Cyrl.bt", "<2khk_Cyrl>", "<bt>"),
        ("khk_Cyrl-eng_Latn.kd", "<2eng_Latn>", "<ft>"),
    )

    outcome = export(
        run_of(tmp_path, launcher_data("src", corpora)), out_dir, spm_model
    )

    assert outcome.record.directions == TOY_DIRECTIONS


def test_the_directions_come_from_the_corpora_and_not_from_the_valid_entry(
    tmp_path: Path, spm_model: Path, converter: FakeConvertConverter, out_dir: Path
) -> None:
    data = launcher_data("src", TOY_CORPORA)
    data["valid"] = {
        "path_src": "/data/eng_Latn-rus_Cyrl.src",
        "path_tgt": "/data/eng_Latn-rus_Cyrl.tgt",
    }

    outcome = export(run_of(tmp_path, data), out_dir, spm_model)

    assert outcome.record.directions == TOY_DIRECTIONS


@pytest.mark.parametrize(
    "stated",
    [
        TOY_DIRECTIONS,
        tuple(reversed(TOY_DIRECTIONS)),
        (*TOY_DIRECTIONS, TOY_DIRECTIONS[0]),
    ],
    ids=["as derived", "in another order", "with a repeat"],
)
def test_a_stated_set_equal_to_the_derived_one_is_accepted_and_reported_as_derived(
    tmp_path: Path,
    spm_model: Path,
    converter: FakeConvertConverter,
    out_dir: Path,
    stated: tuple[str, ...],
) -> None:
    outcome = export(run_of(tmp_path), out_dir, spm_model, directions=stated)

    assert outcome.record.directions == TOY_DIRECTIONS
    assert outcome.directions_check == "derived"
    assert len(converter.calls) == 1


@pytest.mark.parametrize(
    "stated",
    [
        TOY_DIRECTIONS[:3],
        (*TOY_DIRECTIONS, "eng_Latn-rus_Cyrl"),
        ("eng_Latn-rus_Cyrl", "rus_Cyrl-eng_Latn"),
    ],
    ids=["a subset", "a superset", "another set"],
)
def test_a_stated_set_that_differs_from_the_derived_one_is_refused_with_nothing_made(
    tmp_path: Path,
    spm_model: Path,
    converter: FakeConvertConverter,
    out_dir: Path,
    stated: tuple[str, ...],
) -> None:
    run = run_of(tmp_path)

    message = refused(run, out_dir, spm_model, converter, directions=stated)

    assert "directions" in message
    assert str(run.final / "config.json") in message
    for direction in TOY_DIRECTIONS:
        assert direction in message
    assert "--direction" in message


def test_directions_nobody_can_derive_are_a_refusal_that_names_the_option(
    tmp_path: Path, spm_model: Path, converter: FakeConvertConverter, out_dir: Path
) -> None:
    run = undecided_run(tmp_path)

    message = refused(run, out_dir, spm_model, converter)

    assert "--direction" in message
    assert str(run.final / "config.json") in message


@pytest.mark.parametrize(
    "data", [{}, {"valid": {"path_src": "/v.src", "path_tgt": "/v.tgt"}}], ids=str
)
def test_a_data_section_with_no_corpus_derives_nothing_either(
    tmp_path: Path,
    spm_model: Path,
    converter: FakeConvertConverter,
    out_dir: Path,
    data: dict[str, Any],
) -> None:
    message = refused(run_of(tmp_path, data), out_dir, spm_model, converter)

    assert "--direction" in message


def test_directions_stated_for_a_config_that_names_none_are_recorded_as_stated(
    tmp_path: Path, spm_model: Path, converter: FakeConvertConverter, out_dir: Path
) -> None:
    outcome = export(
        undecided_run(tmp_path),
        out_dir,
        spm_model,
        directions=("khk_Cyrl-eng_Latn", "eng_Latn-khk_Cyrl"),
    )

    assert outcome.record.directions == ("eng_Latn-khk_Cyrl", "khk_Cyrl-eng_Latn")
    assert outcome.record.languages == ("eng_Latn", "khk_Cyrl")
    assert outcome.directions_check == "stated"


@pytest.mark.parametrize(
    "stated",
    [
        (),
        ("eng_Latn-xxx_Latn",),
        ("eng_Latn-eng_Latn",),
        ("eng_Latn",),
        ("eng_Latn-khk_Cyrl-rus_Cyrl",),
        ("khk_Cyrl_eng_Latn",),
    ],
    ids=[
        "none",
        "a code outside the registry",
        "a language to itself",
        "no target",
        "three parts",
        "no hyphen",
    ],
)
def test_stated_directions_a_record_cannot_hold_are_refused_before_anything_is_read(
    tmp_path: Path,
    spm_model: Path,
    converter: FakeConvertConverter,
    out_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
    stated: tuple[str, ...],
) -> None:
    def read_nothing(*_: object) -> None:
        raise AssertionError("the checkpoint was read")

    monkeypatch.setattr(distill_export, "verify_source", read_nothing)

    with pytest.raises(ConfigError):
        export(undecided_run(tmp_path), out_dir, spm_model, directions=stated)

    assert not out_dir.exists()
    assert converter.calls == []


# ---- the document level -------------------------------------------------------------------


def window_data(*, windows: bool) -> dict[str, Any]:
    data = launcher_data("src", TOY_CORPORA)
    if windows:
        data["eng_Latn-khk_Cyrl.auth"]["src_prefix"] = "<2khk_Cyrl> <doc>"
    return data


@pytest.mark.parametrize("windows", [True, False])
def test_the_document_level_is_derived_from_the_prefixes_of_the_saved_corpora(
    tmp_path: Path,
    spm_model: Path,
    converter: FakeConvertConverter,
    out_dir: Path,
    windows: bool,
) -> None:
    run = run_of(tmp_path, window_data(windows=windows))

    outcome = export(run, out_dir, spm_model)

    assert outcome.record.document_level is windows
    assert outcome.document_level_check == "derived"


@pytest.mark.parametrize("windows", [True, False])
def test_a_statement_equal_to_the_derived_level_is_accepted_and_reported_as_derived(
    tmp_path: Path,
    spm_model: Path,
    converter: FakeConvertConverter,
    out_dir: Path,
    windows: bool,
) -> None:
    run = run_of(tmp_path, window_data(windows=windows))

    outcome = export(run, out_dir, spm_model, document_level=windows)

    assert outcome.record.document_level is windows
    assert outcome.document_level_check == "derived"


@pytest.mark.parametrize("windows", [True, False])
def test_a_statement_that_contradicts_the_derived_level_is_refused_with_nothing_made(
    tmp_path: Path,
    spm_model: Path,
    converter: FakeConvertConverter,
    out_dir: Path,
    windows: bool,
) -> None:
    run = run_of(tmp_path, window_data(windows=windows))

    message = refused(run, out_dir, spm_model, converter, document_level=not windows)

    assert "document level" in message
    assert str(run.final / "config.json") in message
    assert "--document-level" in message


@pytest.mark.parametrize("level", [True, False])
def test_a_level_stated_for_a_config_that_says_nothing_is_recorded_as_stated(
    tmp_path: Path,
    spm_model: Path,
    converter: FakeConvertConverter,
    out_dir: Path,
    level: bool,
) -> None:
    outcome = export(
        undecided_run(tmp_path),
        out_dir,
        spm_model,
        directions=TOY_DIRECTIONS,
        document_level=level,
    )

    assert outcome.record.document_level is level
    assert outcome.document_level_check == "stated"


def test_with_nothing_derived_and_nothing_stated_the_model_is_a_sentence_model(
    tmp_path: Path, spm_model: Path, converter: FakeConvertConverter, out_dir: Path
) -> None:
    outcome = export(
        undecided_run(tmp_path), out_dir, spm_model, directions=TOY_DIRECTIONS
    )

    assert outcome.record.document_level is False
    assert outcome.document_level_check == "stated"


# ---- quantisation and name ----------------------------------------------------------------


@pytest.mark.parametrize("quantization", ["float32", "float16", "int8", "int8_float16"])
def test_the_quantisation_reaches_the_converter_and_the_record(
    tmp_path: Path,
    spm_model: Path,
    converter: FakeConvertConverter,
    out_dir: Path,
    quantization: str,
) -> None:
    outcome = export(run_of(tmp_path), out_dir, spm_model, quantization=quantization)

    assert outcome.record.quantization == quantization
    assert [call.quantization for call in converter.calls] == [quantization]


def test_the_default_quantisation_is_float32_and_is_passed_to_the_converter(
    tmp_path: Path, spm_model: Path, converter: FakeConvertConverter, out_dir: Path
) -> None:
    outcome = export(run_of(tmp_path), out_dir, spm_model)

    assert outcome.record.quantization == "float32"
    assert [call.quantization for call in converter.calls] == ["float32"]


@pytest.mark.parametrize("quantization", ["int4", "int16", "auto", "", "INT8"])
def test_a_quantisation_that_is_not_offered_is_refused_before_anything_is_read(
    tmp_path: Path,
    spm_model: Path,
    converter: FakeConvertConverter,
    out_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
    quantization: str,
) -> None:
    def read_nothing(*_: object) -> None:
        raise AssertionError("the checkpoint was read")

    monkeypatch.setattr(distill_export, "verify_source", read_nothing)

    with pytest.raises(ConfigError, match="float32, float16, int8, int8_float16"):
        export(run_of(tmp_path), out_dir, spm_model, quantization=quantization)

    assert not out_dir.exists()
    assert converter.calls == []


def test_a_name_is_recorded_and_the_default_is_made_from_the_weights(
    tmp_path: Path, spm_model: Path, converter: FakeConvertConverter, out_dir: Path
) -> None:
    run = run_of(tmp_path)

    named = export(run, out_dir, spm_model, name="toy-teacher.v1")
    plain = export(run, tmp_path / "plain", spm_model)

    assert named.record.name == "toy-teacher.v1"
    assert plain.record.name == "rinchen-07148803f29a"


@pytest.mark.parametrize("name", ["", "a b", "x" * 65, "na" + chr(0xEF) + "ve"])
def test_a_name_a_record_cannot_hold_is_refused_before_anything_is_read(
    tmp_path: Path,
    spm_model: Path,
    converter: FakeConvertConverter,
    out_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
    name: str,
) -> None:
    def read_nothing(*_: object) -> None:
        raise AssertionError("the checkpoint was read")

    monkeypatch.setattr(distill_export, "verify_source", read_nothing)

    with pytest.raises(ConfigError, match="a model name must be"):
        export(run_of(tmp_path), out_dir, spm_model, name=name)

    assert not out_dir.exists()


# ---- the manifest and what it vouches for --------------------------------------------------


def test_the_manifest_says_how_each_fact_was_settled(
    tmp_path: Path, spm_model: Path, converter: FakeConvertConverter, out_dir: Path
) -> None:
    export(
        undecided_run(tmp_path),
        out_dir,
        spm_model,
        directions=TOY_DIRECTIONS,
        quantization="int8",
    )

    manifest = store.read(out_dir)
    assert manifest is not None
    assert manifest.status == {
        "export": "ok",
        "tag_placement": "src",
        "quantization": "int8",
        "checkpoint_kind": "step",
        "tag_placement_check": "stated",
        "directions_check": "stated",
        "document_level_check": "stated",
    }


def test_the_inputs_of_the_manifest_are_the_checkpoints_files_and_the_spm_model(
    tmp_path: Path, spm_model: Path, converter: FakeConvertConverter, out_dir: Path
) -> None:
    run = run_of(tmp_path)

    export(run, out_dir, spm_model)

    manifest = store.read(out_dir)
    assert manifest is not None
    assert [(item.path, item.sha256) for item in manifest.inputs] == [
        (
            str(run.final / name),
            sha256(run.final / name),
        )
        for name in ("model.00.safetensors", "config.json", "vocab.json")
    ] + [(str(spm_model), sha256(spm_model))]
    assert sha256(spm_model) == hashlib.sha256(SPM_BYTES).hexdigest()
