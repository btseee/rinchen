"""`rinchen distill export`: what it prints and writes (the five files of an export and the
manifest that vouches for them), what it records of the checkpoint it came from, and each
refusal of `export_checkpoint` as the exit code the command documents.

The converter is `tests.support.distill_exports.fake_converter` and the runs are fabricated
by `tests.support.step_dirs`, so nothing starts a process. What the library checks is in
`test_train_distill_export.py`; these tests check how the command reports it.
"""

from __future__ import annotations

import importlib.metadata
import os
from pathlib import Path

import pytest
from click.testing import Result

from rinchen.cli import train_common
from rinchen.errors import UpstreamError
from rinchen.manifest import stage, store
from rinchen.manifest.model import Manifest
from rinchen.train.distill_export import (
    EXPORT_FILES,
    ExportRecord,
    verify_export,
)
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
    sha256,
    tree_digest,
)

CHECKPOINT_FILES = ("model.00.safetensors", "config.json", "vocab.json")
# The SHA-256 of the canonical JSON `{"tag_placement":"src"}` and of the one for `tgt`.
SRC_CONFIG_SHA256 = "4307033aca624ba5f9468b04cc39a3548307673a45f71a2745cb8cfde9dafe09"
TGT_CONFIG_SHA256 = "e32a8d964094bdcfbb00810e80610a21d2f6bf5fc7de217f65b9c7497fc9ac40"
WRITTEN = (
    "config.json",
    "export.json",
    "manifest.json",
    "model.bin",
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
        f"record: {workspace.out_dir}/export.json",
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
    assert call.out_dir == workspace.out_dir
    assert call.timeout is None


def test_the_record_holds_the_placement_and_the_shape_of_the_checkpoint(
    workspace: ExportWorkspace, converter: FakeConverter
) -> None:
    run_ok(workspace.args(tag_placement="tgt"))

    record = ExportRecord.from_bytes(workspace.record.read_bytes())
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
    assert manifest.stage == "distill_export"
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
    real = stage.sha256_of

    def spying(path: Path) -> str:
        hashed.append(path)
        return real(path)

    monkeypatch.setattr(stage, "sha256_of", spying)

    run_ok(workspace.args())

    assert sorted(hashed) == sorted(workspace.out_dir / name for name in EXPORT_FILES)


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
        "checkpoint_kind": "step",
    }
    assert manifest.seed == -1
    assert manifest.config_sha256 == SRC_CONFIG_SHA256


def test_the_config_digest_stands_for_the_tag_placement(
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
        "checkpoint_kind": "averaged",
    }
    assert [Path(item.path) for item in manifest.inputs[:3]] == [
        workspace.checkpoint / name for name in CHECKPOINT_FILES
    ]
    record = ExportRecord.from_bytes(workspace.record.read_bytes())
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
    seen: list[tuple[list[bool], list[str]]] = []
    real_write = train_common.store.write

    def looking(out_dir: Path, manifest: Manifest) -> Path:
        seen.append(
            (
                [Path(item.path).is_file() for item in manifest.outputs],
                sorted(path.name for path in out_dir.iterdir()),
            )
        )
        return real_write(out_dir, manifest)

    monkeypatch.setattr(train_common.store, "write", looking)

    run_ok(workspace.args())

    ((exist, names),) = seen
    assert all(exist)
    assert names == sorted(name for name in WRITTEN if name != "manifest.json")


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
