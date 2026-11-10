"""`rinchen distill export`: what it prints and writes (the five files of an export and the
manifest that vouches for them), what it records of the checkpoint it came from, and each
refusal of `export_checkpoint` as the exit code the command documents. D11 made the command
the same function as `rinchen export convert` (`test_cli_export.py`) with a new record
(`rinchen.json`), stage (`export`) and printed lines; D10's expectations are kept here with
those changes.

The converter is `tests.support.distill_exports.fake_converter` and the runs are fabricated
by `tests.support.step_dirs`, so nothing starts a process. What the library checks is in
`test_train_distill_export.py`; these tests check how the command reports it.
"""

from __future__ import annotations

import importlib.metadata
import json
import os
from pathlib import Path
from typing import Any

import pytest
from click.testing import Result

from rinchen.cli import train_common
from rinchen.errors import UpstreamError
from rinchen.export import ct2
from rinchen.export.record import EXPORT_FILES, ModelRecord
from rinchen.export.verified import verify_export
from rinchen.manifest import stage, store
from rinchen.manifest.model import Manifest
from tests.support.distill_cli import (
    ExportWorkspace,
    invoke,
    make_export_workspace,
    run_ok,
)
from tests.support.distill_exports import FakeConverter, fake_converter
from tests.support.launch_cli import assert_one_error, snapshot
from tests.support.step_dirs import (
    eole_model_section,
    flip_one_byte,
    launcher_data,
    sha256,
    tree_digest,
)

CHECKPOINT_FILES = ("model.00.safetensors", "config.json", "vocab.json")
# The SHA-256 of the canonical JSON `{"quantization":"float32","tag_placement":"src"}` and of
# the one for `tgt`.
SRC_CONFIG_SHA256 = "42d06047b312429f6fe21256c25647fee455b790f1b05f2445380cf89c0af323"
TGT_CONFIG_SHA256 = "d9ca68b457cfcc0fa5d06181b1a15706d17e99c4ea86c2f111a704ce8ba46ffc"
WRITTEN = (
    "config.json",
    "manifest.json",
    "model.bin",
    "rinchen.json",
    "shared_vocabulary.json",
    "spm.model",
)


@pytest.fixture
def workspace(tmp_path: Path) -> ExportWorkspace:
    return make_export_workspace(tmp_path)


@pytest.fixture
def converter(monkeypatch: pytest.MonkeyPatch) -> FakeConverter:
    return fake_converter(monkeypatch)


def manifest_of(workspace: ExportWorkspace) -> Manifest:
    manifest = store.read(workspace.out_dir)
    assert manifest is not None
    return manifest


# --- help ----------------------------------------------------------------------------------


def test_the_distill_group_lists_export_and_generate() -> None:
    result = invoke(["distill", "--help"])

    assert result.exit_code == 0
    assert "export" in result.stdout
    assert "generate" in result.stdout


def test_help_names_every_option_and_marks_exactly_the_four_required_ones() -> None:
    result = invoke(["distill", "export", "--help"])

    assert result.exit_code == 0
    for option in (
        "--checkpoint",
        "--spm-model",
        "--tag-placement",
        "--out-dir",
        "--timeout",
    ):
        assert option in result.stdout, option
    assert "<src|tgt>" in result.stdout
    assert result.stdout.count("[required]") == 4


def test_the_tag_placement_has_no_default(
    workspace: ExportWorkspace, converter: FakeConverter
) -> None:
    result = invoke(workspace.args(tag_placement=None))

    assert result.exit_code == 2, result.output
    assert "--tag-placement" in result.stderr
    assert not workspace.out_dir.exists()
    assert converter.calls == []


def test_a_tag_placement_that_is_neither_src_nor_tgt_is_refused(
    workspace: ExportWorkspace, converter: FakeConverter
) -> None:
    result = invoke(workspace.args(tag_placement="both"))

    assert result.exit_code == 2, result.output
    assert not workspace.out_dir.exists()
    assert converter.calls == []


# --- what it prints and writes -----------------------------------------------------------------


def test_a_run_prints_where_it_exported_to_the_record_and_the_manifest(
    workspace: ExportWorkspace, converter: FakeConverter
) -> None:
    result = run_ok(workspace.args())

    assert result.stdout.splitlines() == [
        f"exported to {workspace.out_dir}",
        "tag placement: stated, not recorded in the checkpoint",
        "directions: stated (2)",
        "document level: stated (false)",
        "quantization: float32",
        f"record: {workspace.out_dir}/rinchen.json",
        f"manifest: {workspace.out_dir}/manifest.json",
    ]


def test_the_output_directory_holds_the_five_files_and_the_manifest_only(
    workspace: ExportWorkspace, converter: FakeConverter
) -> None:
    run_ok(workspace.args())

    assert sorted(workspace.written()) == sorted(WRITTEN)
    assert workspace.log.is_file()


def test_the_converter_is_called_once_with_the_checkpoint_and_the_output_directory(
    workspace: ExportWorkspace, converter: FakeConverter
) -> None:
    run_ok(workspace.args())

    (call,) = converter.calls
    assert call.checkpoint == workspace.checkpoint
    # A child of a private directory beside the output, renamed onto it; the log is beside it.
    assert call.out_dir.parent.parent == workspace.out_dir.parent
    assert call.log_path == workspace.log
    assert call.timeout is None


def test_the_record_holds_the_placement_and_the_shape_of_the_checkpoint(
    workspace: ExportWorkspace, converter: FakeConverter
) -> None:
    run_ok(workspace.args(tag_placement="tgt"))

    record = ModelRecord.from_bytes(workspace.record.read_bytes())
    assert record.tag_placement == "tgt"
    assert (record.encoder_layers, record.decoder_layers) == (2, 1)
    assert record.hidden_size == 64
    assert record.vocabulary_entries == 124
    assert record.checkpoint.kind == "step"
    assert record.checkpoint.step == 200
    assert record.checkpoint.weights_sha256 == sha256(
        workspace.checkpoint / "model.00.safetensors"
    )


def test_the_manifest_is_the_one_verify_export_accepts(
    workspace: ExportWorkspace, converter: FakeConverter
) -> None:
    run_ok(workspace.args())

    verified = verify_export(workspace.out_dir)

    assert verified.record.tag_placement == "src"
    assert dict(verified.sha256) == {
        name: sha256(workspace.out_dir / name) for name in EXPORT_FILES
    }
    assert verified.manifest_sha256 == sha256(workspace.manifest)


def test_the_manifest_lists_the_five_files_with_their_digests_and_no_row_counts(
    workspace: ExportWorkspace, converter: FakeConverter
) -> None:
    run_ok(workspace.args())

    manifest = manifest_of(workspace)
    assert manifest.stage == "export"
    assert [Path(item.path) for item in manifest.outputs] == [
        workspace.out_dir / name for name in EXPORT_FILES
    ]
    for item in manifest.outputs:
        assert item.sha256 == sha256(Path(item.path)), item.path
        assert item.bytes == Path(item.path).stat().st_size, item.path
        assert item.rows is None, item.path


def test_the_manifest_vouches_for_the_checkpoint_files_with_the_digests_of_the_record(
    workspace: ExportWorkspace, converter: FakeConverter
) -> None:
    run_ok(workspace.args())

    manifest = manifest_of(workspace)
    assert [(Path(item.path), item.sha256) for item in manifest.inputs] == [
        *(
            (workspace.checkpoint / name, sha256(workspace.checkpoint / name))
            for name in CHECKPOINT_FILES
        ),
        (workspace.spm_model, sha256(workspace.spm_model)),
    ]


def test_the_command_hashes_only_the_files_it_wrote(
    workspace: ExportWorkspace,
    converter: FakeConverter,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    hashed: list[Path] = []
    real_convert = ct2.sha256_of
    real_stage = stage.sha256_of

    def spying(path: Path) -> str:
        hashed.append(path)
        return real_convert(path)

    def spying_stage(path: Path) -> str:
        hashed.append(path)
        return real_stage(path)

    monkeypatch.setattr(ct2, "sha256_of", spying)
    monkeypatch.setattr(stage, "sha256_of", spying_stage)

    run_ok(workspace.args())

    # `convert` hashes each file once, in the private directory it renames onto the output.
    assert sorted(path.name for path in hashed) == sorted(EXPORT_FILES)


def test_the_metrics_status_and_seed_are_those_of_an_export(
    workspace: ExportWorkspace, converter: FakeConverter
) -> None:
    run_ok(workspace.args())

    manifest = manifest_of(workspace)
    assert manifest.metrics == {
        "encoder_layers": 2.0,
        "decoder_layers": 1.0,
        "hidden_size": 64.0,
        "vocabulary_entries": 124.0,
        "model_bytes": 30.0,
    }
    assert manifest.status == {
        "export": "ok",
        "tag_placement": "src",
        "quantization": "float32",
        "checkpoint_kind": "step",
        "tag_placement_check": "stated",
        "directions_check": "stated",
        "document_level_check": "stated",
    }
    assert manifest.seed == -1
    assert manifest.config_sha256 == SRC_CONFIG_SHA256


def test_the_config_digest_stands_for_the_tag_placement_and_the_quantisation(
    workspace: ExportWorkspace, converter: FakeConverter
) -> None:
    run_ok(workspace.args(tag_placement="tgt"))

    manifest = manifest_of(workspace)
    assert manifest.config_sha256 == TGT_CONFIG_SHA256
    assert manifest.status["tag_placement"] == "tgt"


def test_the_manifest_records_the_installed_tools(
    workspace: ExportWorkspace, converter: FakeConverter
) -> None:
    run_ok(workspace.args())

    assert manifest_of(workspace).tool_versions == {
        package: importlib.metadata.version(package)
        for package in ("ctranslate2", "eole", "torch")
    }


def test_an_averaged_model_is_exported_and_recorded_as_one(
    tmp_path: Path, converter: FakeConverter
) -> None:
    workspace = make_export_workspace(tmp_path, averaged=True)

    run_ok(workspace.args(tag_placement="tgt"))

    manifest = manifest_of(workspace)
    assert manifest.status == {
        "export": "ok",
        "tag_placement": "tgt",
        "quantization": "float32",
        "checkpoint_kind": "averaged",
        "tag_placement_check": "stated",
        "directions_check": "stated",
        "document_level_check": "stated",
    }
    assert [Path(item.path) for item in manifest.inputs[:3]] == [
        workspace.checkpoint / name for name in CHECKPOINT_FILES
    ]
    record = ModelRecord.from_bytes(workspace.record.read_bytes())
    assert (record.checkpoint.kind, record.checkpoint.step) == ("averaged", None)
    assert record.checkpoint.manifest_stage == "train_average"


def test_a_relative_checkpoint_and_output_directory_are_recorded_as_absolute_paths(
    workspace: ExportWorkspace,
    converter: FakeConverter,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(workspace.root)
    relative_checkpoint = workspace.checkpoint.relative_to(workspace.root)

    run_ok(
        workspace.args(
            checkpoint=relative_checkpoint,
            spm_model=workspace.spm_model.relative_to(workspace.root),
            out_dir=Path("export"),
        )
    )

    manifest = manifest_of(workspace)
    assert all(
        os.path.isabs(item.path) for item in (*manifest.inputs, *manifest.outputs)
    )
    assert Path(manifest.inputs[0].path) == workspace.checkpoint / CHECKPOINT_FILES[0]


# --- the manifest comes last -------------------------------------------------------------------


def test_the_manifest_is_written_after_every_file_it_lists(
    workspace: ExportWorkspace,
    converter: FakeConverter,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # `convert` writes it into the private directory that is renamed onto the output, so the
    # output appears whole, manifest included, or not at all.
    seen: list[tuple[list[str], bool]] = []
    real_write = train_common.store.write

    def looking(out_dir: Path, manifest: Manifest) -> Path:
        seen.append(
            (
                sorted(path.name for path in out_dir.iterdir()),
                workspace.out_dir.exists(),
            )
        )
        return real_write(out_dir, manifest)

    monkeypatch.setattr(train_common.store, "write", looking)

    run_ok(workspace.args())

    ((names, output_exists),) = seen
    assert names == sorted(name for name in WRITTEN if name != "manifest.json")
    assert not output_exists
    assert sorted(workspace.written()) == sorted(WRITTEN)


@pytest.mark.parametrize(
    ("error", "exit_code"),
    [(OSError("disk full"), 1), (KeyboardInterrupt(), 130)],
    ids=["a write error", "an interrupt"],
)
def test_a_failure_while_writing_the_manifest_leaves_no_output_directory(
    workspace: ExportWorkspace,
    converter: FakeConverter,
    monkeypatch: pytest.MonkeyPatch,
    error: BaseException,
    exit_code: int,
) -> None:
    def failing(*args: object, **kwargs: object) -> Path:
        raise error

    monkeypatch.setattr(train_common.store, "write", failing)

    result = invoke(workspace.args())

    assert result.exit_code == exit_code, result.output
    assert not workspace.out_dir.exists()
    if exit_code == 1:
        assert_one_error(
            result, 1, f"cannot write the CTranslate2 export under {workspace.out_dir}"
        )
        assert "disk full" in result.stderr


# --- --timeout and the converter ---------------------------------------------------------------


def test_the_timeout_reaches_the_converter(
    workspace: ExportWorkspace, converter: FakeConverter
) -> None:
    run_ok(workspace.args("--timeout", "12.5"))

    assert [call.timeout for call in converter.calls] == [12.5]


@pytest.mark.parametrize("timeout", ["0", "-1", "inf", "nan"])
def test_a_timeout_that_is_not_positive_and_finite_is_refused_before_any_work(
    workspace: ExportWorkspace, converter: FakeConverter, timeout: str
) -> None:
    result = invoke(workspace.args("--timeout", timeout))

    assert_one_error(
        result, 2, "--timeout must be a positive, finite number of seconds"
    )
    assert not workspace.out_dir.exists()
    assert converter.calls == []


def test_a_converter_that_fails_is_an_upstream_error_and_leaves_no_directory(
    workspace: ExportWorkspace, monkeypatch: pytest.MonkeyPatch
) -> None:
    converter = fake_converter(
        monkeypatch, error=UpstreamError("the CTranslate2 export failed: exit code 1")
    )

    result = invoke(workspace.args())

    assert_one_error(result, 4, "the CTranslate2 export failed: exit code 1")
    assert len(converter.calls) == 1
    assert not workspace.out_dir.exists()
    assert workspace.log.is_file()


def test_a_converter_that_cannot_write_is_a_write_error_and_leaves_no_directory(
    workspace: ExportWorkspace, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake_converter(monkeypatch, error=OSError("no space left on device"))

    result = invoke(workspace.args())

    assert_one_error(
        result, 1, f"cannot write the CTranslate2 export under {workspace.out_dir}"
    )
    assert "no space left on device" in result.stderr
    assert not workspace.out_dir.exists()


def test_an_interrupt_exits_130_and_leaves_no_directory(
    workspace: ExportWorkspace, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake_converter(monkeypatch, error=KeyboardInterrupt())

    result = invoke(workspace.args())

    assert result.exit_code == 130, result.output
    assert not workspace.out_dir.exists()


# --- what it refuses ---------------------------------------------------------------------------


def assert_refused(
    result: Result,
    workspace: ExportWorkspace,
    converter: FakeConverter,
    *,
    exit_code: int,
    message: str,
    digest: str,
) -> None:
    assert_one_error(result, exit_code, message)
    assert not workspace.out_dir.exists()
    assert converter.calls == []
    assert tree_digest(workspace.root / "teacher") == digest


def test_a_run_without_our_manifest_is_refused_as_a_configuration_error(
    tmp_path: Path, converter: FakeConverter
) -> None:
    workspace = make_export_workspace(tmp_path, manifest=False)
    digest = tree_digest(tmp_path / "teacher")

    result = invoke(workspace.args())

    assert_refused(
        result,
        workspace,
        converter,
        exit_code=2,
        message="cannot continue from",
        digest=digest,
    )


def test_a_flipped_byte_in_the_weights_is_refused(
    workspace: ExportWorkspace, converter: FakeConverter
) -> None:
    flip_one_byte(workspace.checkpoint / "model.00.safetensors")
    digest = tree_digest(workspace.root / "teacher")

    result = invoke(workspace.args())

    assert_refused(
        result,
        workspace,
        converter,
        exit_code=2,
        message="cannot continue from",
        digest=digest,
    )
    assert "model.00.safetensors" in result.stderr


def test_a_step_that_is_not_the_final_one_is_refused(
    workspace: ExportWorkspace, converter: FakeConverter
) -> None:
    assert workspace.run is not None
    digest = tree_digest(workspace.root / "teacher")

    result = invoke(workspace.args(checkpoint=workspace.run.step(150)))

    assert_refused(
        result,
        workspace,
        converter,
        exit_code=2,
        message="cannot continue from",
        digest=digest,
    )


def test_a_sentencepiece_model_that_is_not_the_one_of_the_checkpoint_is_refused(
    workspace: ExportWorkspace, converter: FakeConverter
) -> None:
    workspace.spm_model.write_bytes(b"another model")
    digest = tree_digest(workspace.root / "teacher")

    result = invoke(workspace.args())

    assert_refused(
        result,
        workspace,
        converter,
        exit_code=2,
        message=f"{workspace.spm_model} (SHA-256",
        digest=digest,
    )
    assert "is not the SentencePiece model the checkpoint was trained with" in (
        result.stderr
    )


def test_a_sentencepiece_model_that_is_not_a_file_is_refused(
    workspace: ExportWorkspace, converter: FakeConverter
) -> None:
    missing = workspace.root / "nowhere.model"
    digest = tree_digest(workspace.root / "teacher")

    result = invoke(workspace.args(spm_model=missing))

    assert_refused(
        result,
        workspace,
        converter,
        exit_code=2,
        message=f"the SentencePiece model {missing} is not a file",
        digest=digest,
    )


def test_a_config_without_the_layer_count_is_a_data_error(
    tmp_path: Path, converter: FakeConverter
) -> None:
    model = eole_model_section(encoder_layers=2, decoder_layers=1)
    del model["encoder"]["layers"]
    workspace = make_export_workspace(tmp_path, model=model)
    digest = tree_digest(tmp_path / "teacher")

    result = invoke(workspace.args())

    assert_refused(
        result,
        workspace,
        converter,
        exit_code=3,
        message=f"{workspace.checkpoint / 'config.json'}: model.encoder.layers",
        digest=digest,
    )


@pytest.mark.parametrize("character", ["$", "[", "]", "*", "?"])
def test_a_checkpoint_path_with_a_dollar_or_a_glob_character_is_refused(
    tmp_path: Path, converter: FakeConverter, character: str
) -> None:
    workspace = make_export_workspace(tmp_path / f"run{character}x")
    target = tmp_path / "clean-export"

    result = invoke(workspace.args(out_dir=target))

    assert result.exit_code == 2, result.output
    assert "checkpoint path" in result.stderr
    assert not target.exists()
    assert converter.calls == []


@pytest.mark.parametrize("character", ["$", "[", "]", "*", "?"])
def test_an_output_directory_with_a_dollar_or_a_glob_character_is_refused(
    workspace: ExportWorkspace, converter: FakeConverter, character: str
) -> None:
    target = workspace.root / f"export{character}x"

    result = invoke(workspace.args(out_dir=target))

    assert result.exit_code == 2, result.output
    assert "output directory" in result.stderr
    assert not target.exists()
    assert converter.calls == []


def test_an_existing_output_directory_is_refused_and_left_as_it_was(
    workspace: ExportWorkspace, converter: FakeConverter
) -> None:
    run_ok(workspace.args())
    before = snapshot(workspace.out_dir)
    converter.calls.clear()

    result = invoke(workspace.args())

    assert_one_error(result, 2, f"{workspace.out_dir} exists: remove it")
    assert snapshot(workspace.out_dir) == before
    assert converter.calls == []


def test_an_output_directory_with_another_stages_manifest_is_refused_untouched(
    workspace: ExportWorkspace, converter: FakeConverter
) -> None:
    workspace.out_dir.mkdir()
    train_common.write_manifest(
        workspace.out_dir,
        stage="train_launch",
        config_sha256="0" * 64,
        seed=1,
        inputs=[],
        outputs=[],
        metrics={},
        packages=[],
        status={"launch": "ok"},
    )
    before = snapshot(workspace.out_dir)

    result = invoke(workspace.args())

    assert_one_error(result, 2, f"{workspace.out_dir} exists: remove it")
    assert snapshot(workspace.out_dir) == before
    assert converter.calls == []


# --- the tag placement against the saved config of the checkpoint ------------------------------


def test_a_placement_the_saved_config_contradicts_is_refused_with_nothing_made(
    tmp_path: Path, converter: FakeConverter
) -> None:
    workspace = make_export_workspace(tmp_path, data=launcher_data("src"))
    digest = tree_digest(tmp_path / "teacher")

    result = invoke(workspace.args(tag_placement="tgt"))

    assert_refused(
        result,
        workspace,
        converter,
        exit_code=2,
        message="the tag placement 'tgt' is not the one the checkpoint was trained with",
        digest=digest,
    )
    assert "pass --tag-placement src" in result.stderr
    assert not workspace.log.exists()


@pytest.mark.parametrize("placement", ["src", "tgt"])
def test_the_placement_the_saved_config_records_is_exported_and_stated_in_the_manifest(
    tmp_path: Path, converter: FakeConverter, placement: str
) -> None:
    workspace = make_export_workspace(tmp_path, data=launcher_data(placement))

    run_ok(workspace.args(tag_placement=placement))

    assert manifest_of(workspace).status["tag_placement"] == placement
    assert verify_export(workspace.out_dir).record.tag_placement == placement


def test_a_saved_config_whose_corpora_disagree_is_a_data_error(
    tmp_path: Path, converter: FakeConverter
) -> None:
    data = launcher_data("src")
    data["eng_Latn-khk_Cyrl.bt"] = launcher_data("tgt")["eng_Latn-khk_Cyrl.bt"]
    workspace = make_export_workspace(tmp_path, data=data)
    digest = tree_digest(tmp_path / "teacher")

    result = invoke(workspace.args())

    assert_refused(
        result,
        workspace,
        converter,
        exit_code=3,
        message=f"{workspace.checkpoint / 'config.json'}: the corpora place the direction tag",
        digest=digest,
    )


# --- whether the tag placement was checked ------------------------------------------------------

CHECKED_LINE = "tag placement: checked against the saved config"
STATED_LINE = "tag placement: stated, not recorded in the checkpoint"


def placement_frame(result: Result, workspace: ExportWorkspace) -> list[str]:
    """The lines of a run that frame the one about the tag placement: the first, that
    line itself and the last two (the lines between them are about the directions, the
    document level and the quantisation, which `test_cli_export.py` checks in full)."""
    lines = result.stdout.splitlines()
    assert len(lines) == 7, lines
    assert lines[0] == f"exported to {workspace.out_dir}"
    assert lines[-2:] == [
        f"record: {workspace.out_dir}/rinchen.json",
        f"manifest: {workspace.out_dir}/manifest.json",
    ]
    return [lines[1]]


@pytest.mark.parametrize("placement", ["src", "tgt"])
def test_a_placement_the_saved_config_records_is_reported_as_checked(
    tmp_path: Path, converter: FakeConverter, placement: str
) -> None:
    workspace = make_export_workspace(tmp_path, data=launcher_data(placement))

    result = run_ok(workspace.args(tag_placement=placement))

    assert placement_frame(result, workspace) == [CHECKED_LINE]
    assert manifest_of(workspace).status["tag_placement_check"] == "checked"


@pytest.mark.parametrize("placement", ["src", "tgt"])
@pytest.mark.parametrize(
    "data",
    [None, {}, {"corpus": {"src_prefix": "<bt>", "tgt_prefix": ""}}],
    ids=["no data section", "empty data section", "no direction tag"],
)
def test_a_placement_the_checkpoint_does_not_record_is_reported_as_stated(
    tmp_path: Path,
    converter: FakeConverter,
    placement: str,
    data: dict[str, Any] | None,
) -> None:
    workspace = make_export_workspace(tmp_path, data=data)

    result = run_ok(workspace.args(tag_placement=placement))

    assert placement_frame(result, workspace) == [STATED_LINE]
    assert manifest_of(workspace).status["tag_placement_check"] == "stated"


def test_an_averaged_model_is_reported_as_checked_like_a_step(
    tmp_path: Path, converter: FakeConverter
) -> None:
    workspace = make_export_workspace(
        tmp_path, averaged=True, data=launcher_data("tgt")
    )

    result = run_ok(workspace.args(tag_placement="tgt"))

    assert CHECKED_LINE in result.stdout.splitlines()
    assert manifest_of(workspace).status["tag_placement_check"] == "checked"


def test_the_fact_is_in_the_manifest_only_and_the_record_and_the_files_do_not_hold_it(
    tmp_path: Path, converter: FakeConverter
) -> None:
    workspace = make_export_workspace(tmp_path, data=launcher_data("src"))

    run_ok(workspace.args())

    record = json.loads(workspace.record.read_text(encoding="utf-8"))
    assert "placement_check" not in record
    assert "tag_placement_check" not in record
    assert "checked" not in json.dumps(record)
    assert verify_export(workspace.out_dir).record.tag_placement == "src"


def test_a_refused_placement_prints_no_line_about_the_check(
    tmp_path: Path, converter: FakeConverter
) -> None:
    workspace = make_export_workspace(tmp_path, data=launcher_data("src"))

    result = invoke(workspace.args(tag_placement="tgt"))

    assert result.exit_code == 2
    assert "tag placement:" not in result.stdout
