"""`rinchen export convert` and `rinchen export verify` (D11, Task 2).

`export convert` is the same command function as `rinchen distill export` (D10's name, kept):
the two print the same lines and make the same directory for the same inputs. What the command
does with a checkpoint, line by line, is here; what the library refuses is in
`test_train_distill_export_facts.py` and, for D10's refusals, `test_cli_distill_export.py`.
`export verify` reads a directory back through `verify_export`.

The converter is `tests.support.model_exports.fake_convert_converter` and the runs are
fabricated by `tests.support.step_dirs`: nothing starts a process.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import pytest
from click.testing import Result

from rinchen.errors import UpstreamError
from rinchen.export import ct2
from rinchen.export.record import EXPORT_FILES
from rinchen.manifest import store
from tests.support.distill_cli import (
    DEFAULT_NAME,
    ExportWorkspace,
    invoke,
    make_export_workspace,
    run_ok,
)
from tests.support.launch_cli import assert_one_error, snapshot
from tests.support.model_exports import (
    TOY_DIRECTIONS,
    FakeConvertConverter,
    fake_convert_converter,
    write_model_export,
)
from tests.support.step_dirs import flip_one_byte, launcher_data, sha256, tree_digest

CONVERT = ("export", "convert")
TOY_CORPORA = (
    ("eng_Latn-khk_Cyrl.auth", "<2khk_Cyrl>", None),
    ("khk_Cyrl-eng_Latn.auth", "<2eng_Latn>", None),
    ("khk_Cyrl-rus_Cyrl.auth", "<2rus_Cyrl>", None),
    ("rus_Cyrl-khk_Cyrl.auth", "<2khk_Cyrl>", None),
)


@pytest.fixture
def converter(monkeypatch: pytest.MonkeyPatch) -> FakeConvertConverter:
    return fake_convert_converter(monkeypatch)


@pytest.fixture
def workspace(tmp_path: Path) -> ExportWorkspace:
    return make_export_workspace(tmp_path)


def window_data() -> dict[str, Any]:
    data = launcher_data("src")
    data["eng_Latn-khk_Cyrl.auth"]["src_prefix"] = "<2khk_Cyrl> <doc>"
    return data


# ---- help -----------------------------------------------------------------------------------


def test_the_export_group_lists_convert_and_verify() -> None:
    result = invoke(["export", "--help"])

    assert result.exit_code == 0
    assert "convert" in result.stdout
    assert "verify" in result.stdout


@pytest.mark.parametrize("command", [("export", "convert"), ("distill", "export")])
def test_help_names_every_option_and_marks_exactly_the_four_required_ones(
    command: tuple[str, str],
) -> None:
    result = invoke([*command, "--help"])

    assert result.exit_code == 0
    for option in (
        "--checkpoint",
        "--spm-model",
        "--tag-placement",
        "--out-dir",
        "--quantization",
        "--direction",
        "--document-level",
        "--no-document-level",
        "--name",
        "--timeout",
    ):
        assert option in result.stdout, option
    assert "int8_float16" in result.stdout
    assert result.stdout.count("[required]") == 4


# ---- what a convert prints and records ---------------------------------------------------------


def test_a_run_that_derives_everything_prints_the_seven_lines(
    tmp_path: Path, converter: FakeConvertConverter
) -> None:
    workspace = make_export_workspace(tmp_path, data=launcher_data("src"))

    result = run_ok(workspace.args(command=CONVERT, directions=None))

    assert result.stdout.splitlines() == [
        f"exported to {workspace.out_dir}",
        "tag placement: checked against the saved config",
        "directions: derived from the saved config (2)",
        "document level: derived from the saved config (false)",
        "quantization: float32",
        f"record: {workspace.out_dir}/rinchen.json",
        f"manifest: {workspace.out_dir}/manifest.json",
    ]


def test_a_run_that_states_everything_prints_the_seven_lines(
    workspace: ExportWorkspace, converter: FakeConvertConverter
) -> None:
    result = run_ok(
        workspace.args(
            "--quantization",
            "int8",
            "--document-level",
            command=CONVERT,
            directions=("khk_Cyrl-rus_Cyrl", "eng_Latn-khk_Cyrl", "rus_Cyrl-khk_Cyrl"),
        )
    )

    assert result.stdout.splitlines() == [
        f"exported to {workspace.out_dir}",
        "tag placement: stated, not recorded in the checkpoint",
        "directions: stated (3)",
        "document level: stated (true)",
        "quantization: int8",
        f"record: {workspace.out_dir}/rinchen.json",
        f"manifest: {workspace.out_dir}/manifest.json",
    ]


def test_the_directions_line_counts_directions_and_not_languages(
    tmp_path: Path, converter: FakeConvertConverter
) -> None:
    # Four directions among three languages.
    workspace = make_export_workspace(tmp_path, data=launcher_data("src", TOY_CORPORA))

    result = run_ok(workspace.args(command=CONVERT, directions=None))

    assert result.stdout.splitlines()[2] == (
        "directions: derived from the saved config (4)"
    )


def test_a_window_model_prints_a_derived_true(
    tmp_path: Path, converter: FakeConvertConverter
) -> None:
    workspace = make_export_workspace(tmp_path, data=window_data())

    result = run_ok(workspace.args(command=CONVERT, directions=None))

    assert "document level: derived from the saved config (true)" in (
        result.stdout.splitlines()
    )


def test_the_manifest_says_how_each_fact_was_settled(
    tmp_path: Path, converter: FakeConvertConverter
) -> None:
    workspace = make_export_workspace(tmp_path, data=launcher_data("src"))

    run_ok(
        workspace.args("--quantization", "float16", command=CONVERT, directions=None)
    )

    manifest = store.read(workspace.out_dir)
    assert manifest is not None
    assert manifest.stage == "export"
    assert manifest.status == {
        "export": "ok",
        "tag_placement": "src",
        "quantization": "float16",
        "checkpoint_kind": "step",
        "tag_placement_check": "checked",
        "directions_check": "derived",
        "document_level_check": "derived",
    }


def test_a_statement_made_for_a_checkpoint_that_says_nothing_is_stated_in_the_manifest(
    workspace: ExportWorkspace, converter: FakeConvertConverter
) -> None:
    run_ok(workspace.args(command=CONVERT))

    manifest = store.read(workspace.out_dir)
    assert manifest is not None
    assert {
        key: value for key, value in manifest.status.items() if key.endswith("_check")
    } == {
        "tag_placement_check": "stated",
        "directions_check": "stated",
        "document_level_check": "stated",
    }


def test_the_record_holds_the_directions_the_level_the_quantisation_and_the_name(
    workspace: ExportWorkspace, converter: FakeConvertConverter
) -> None:
    run_ok(
        workspace.args(
            "--quantization",
            "int8_float16",
            "--no-document-level",
            "--name",
            "toy-teacher.v1",
            command=CONVERT,
        )
    )

    record = json.loads(workspace.record.read_text(encoding="utf-8"))
    assert record["schema"] == "rinchen.model/1"
    assert record["name"] == "toy-teacher.v1"
    assert record["directions"] == ["eng_Latn-khk_Cyrl", "khk_Cyrl-eng_Latn"]
    assert record["languages"] == ["eng_Latn", "khk_Cyrl"]
    assert record["tags"] == {"eng_Latn": "<2eng_Latn>", "khk_Cyrl": "<2khk_Cyrl>"}
    assert record["document_level"] is False
    assert record["quantization"] == "int8_float16"
    assert record["tag_placement"] == "src"


def test_the_default_name_is_made_from_the_weights(
    workspace: ExportWorkspace, converter: FakeConvertConverter
) -> None:
    run_ok(workspace.args(command=CONVERT))

    assert json.loads(workspace.record.read_text(encoding="utf-8"))["name"] == (
        DEFAULT_NAME
    )


@pytest.mark.parametrize("quantization", ["float32", "float16", "int8", "int8_float16"])
def test_the_quantisation_reaches_the_converter(
    workspace: ExportWorkspace, converter: FakeConvertConverter, quantization: str
) -> None:
    run_ok(workspace.args("--quantization", quantization, command=CONVERT))

    assert [call.quantization for call in converter.calls] == [quantization]


def test_the_default_quantisation_is_float32(
    workspace: ExportWorkspace, converter: FakeConvertConverter
) -> None:
    result = run_ok(workspace.args(command=CONVERT))

    assert [call.quantization for call in converter.calls] == ["float32"]
    assert "quantization: float32" in result.stdout.splitlines()


@pytest.mark.parametrize("quantization", ["int4", "int16", "auto", "INT8"])
def test_a_quantisation_that_is_not_offered_is_refused_with_nothing_made(
    workspace: ExportWorkspace,
    converter: FakeConvertConverter,
    quantization: str,
) -> None:
    result = invoke(workspace.args("--quantization", quantization, command=CONVERT))

    assert_one_error(result, 2, f"quantization {quantization!r} is not offered")
    assert "float32, float16, int8, int8_float16" in result.stderr
    assert not workspace.out_dir.exists()
    assert converter.calls == []


def test_directions_may_be_given_in_any_order_and_more_than_once(
    workspace: ExportWorkspace, converter: FakeConvertConverter
) -> None:
    result = run_ok(
        workspace.args(
            command=CONVERT,
            directions=("khk_Cyrl-eng_Latn", "eng_Latn-khk_Cyrl", "khk_Cyrl-eng_Latn"),
        )
    )

    assert "directions: stated (2)" in result.stdout.splitlines()
    record = json.loads(workspace.record.read_text(encoding="utf-8"))
    assert record["directions"] == ["eng_Latn-khk_Cyrl", "khk_Cyrl-eng_Latn"]


# ---- what a convert refuses ----------------------------------------------------------------


def assert_nothing_made(
    workspace: ExportWorkspace, converter: FakeConvertConverter, digest: str
) -> None:
    assert not workspace.out_dir.exists()
    assert not workspace.log.exists()
    assert converter.calls == []
    assert tree_digest(workspace.root / "teacher") == digest


def test_directions_that_differ_from_the_ones_the_saved_config_names_are_exit_2(
    tmp_path: Path, converter: FakeConvertConverter
) -> None:
    workspace = make_export_workspace(tmp_path, data=launcher_data("src"))
    digest = tree_digest(tmp_path / "teacher")

    result = invoke(
        workspace.args(
            command=CONVERT, directions=("eng_Latn-khk_Cyrl", "eng_Latn-rus_Cyrl")
        )
    )

    assert_one_error(result, 2, "the directions eng_Latn-khk_Cyrl, eng_Latn-rus_Cyrl")
    assert "eng_Latn-khk_Cyrl, khk_Cyrl-eng_Latn" in result.stderr
    assert_nothing_made(workspace, converter, digest)


def test_directions_nobody_derives_or_states_are_exit_2_naming_the_option(
    workspace: ExportWorkspace, converter: FakeConvertConverter
) -> None:
    digest = tree_digest(workspace.root / "teacher")

    result = invoke(workspace.args(command=CONVERT, directions=None))

    assert_one_error(
        result, 2, f"{workspace.checkpoint / 'config.json'} names no directions"
    )
    assert "--direction" in result.stderr
    assert_nothing_made(workspace, converter, digest)


@pytest.mark.parametrize("direction", ["eng_Latn-xxx_Latn", "eng_Latn", "a-b-c"])
def test_a_direction_a_record_cannot_hold_is_exit_2(
    workspace: ExportWorkspace, converter: FakeConvertConverter, direction: str
) -> None:
    result = invoke(workspace.args(command=CONVERT, directions=(direction,)))

    assert result.exit_code == 2, result.output
    assert not workspace.out_dir.exists()
    assert converter.calls == []


@pytest.mark.parametrize(
    ("windows", "flag"), [(True, "--no-document-level"), (False, "--document-level")]
)
def test_a_document_level_that_differs_from_the_saved_config_is_exit_2(
    tmp_path: Path, converter: FakeConvertConverter, windows: bool, flag: str
) -> None:
    data = window_data() if windows else launcher_data("src")
    workspace = make_export_workspace(tmp_path, data=data)
    digest = tree_digest(tmp_path / "teacher")

    result = invoke(workspace.args(flag, command=CONVERT))

    assert_one_error(result, 2, "the document level")
    assert_nothing_made(workspace, converter, digest)


@pytest.mark.parametrize("name", ["a b", "x" * 65, "na" + chr(0xEF) + "ve"])
def test_a_name_a_record_cannot_hold_is_exit_2(
    workspace: ExportWorkspace, converter: FakeConvertConverter, name: str
) -> None:
    result = invoke(workspace.args("--name", name, command=CONVERT))

    assert_one_error(result, 2, "a model name must be")
    assert not workspace.out_dir.exists()
    assert converter.calls == []


# ---- a failure while the output is made --------------------------------------------------------


@pytest.mark.parametrize(
    ("error", "exit_code"),
    [(OSError("disk full"), 1), (KeyboardInterrupt(), 130)],
    ids=["a write error", "an interrupt"],
)
def test_a_failure_while_the_manifest_is_written_leaves_no_directory(
    workspace: ExportWorkspace,
    converter: FakeConvertConverter,
    monkeypatch: pytest.MonkeyPatch,
    error: BaseException,
    exit_code: int,
) -> None:
    def failing(*args: object, **kwargs: object) -> Path:
        raise error

    monkeypatch.setattr(store, "write", failing)

    result = invoke(workspace.args(command=CONVERT))

    assert result.exit_code == exit_code, result.output
    assert not workspace.out_dir.exists()
    assert {path.name for path in workspace.root.iterdir()} == {
        "teacher",
        "spm",
        "export.export.log",
    }
    if exit_code == 1:
        assert_one_error(
            result, 1, f"cannot write the CTranslate2 export under {workspace.out_dir}"
        )
        assert "disk full" in result.stderr


def test_a_failure_while_the_hashes_are_taken_is_a_write_error_too(
    workspace: ExportWorkspace,
    converter: FakeConvertConverter,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unreadable(path: Path) -> str:
        raise OSError("input/output error")

    monkeypatch.setattr(ct2, "sha256_of", unreadable)

    result = invoke(workspace.args(command=CONVERT))

    assert_one_error(
        result, 1, f"cannot write the CTranslate2 export under {workspace.out_dir}"
    )
    assert "input/output error" in result.stderr
    assert not workspace.out_dir.exists()


def test_a_converter_that_fails_is_exit_4_and_leaves_no_directory(
    workspace: ExportWorkspace, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = fake_convert_converter(
        monkeypatch, error=UpstreamError("the CTranslate2 export failed: exit code 1")
    )

    result = invoke(workspace.args(command=CONVERT))

    assert_one_error(result, 4, "the CTranslate2 export failed: exit code 1")
    assert len(fake.calls) == 1
    assert not workspace.out_dir.exists()
    assert workspace.log.is_file()


def test_the_manifest_is_written_into_the_private_directory_after_the_five_files(
    workspace: ExportWorkspace,
    converter: FakeConvertConverter,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: list[tuple[Path, list[str], bool]] = []
    real = store.write

    def looking(out_dir: Path, manifest: Any) -> Path:
        seen.append(
            (
                out_dir,
                sorted(path.name for path in out_dir.iterdir()),
                workspace.out_dir.exists(),
            )
        )
        return real(out_dir, manifest)

    monkeypatch.setattr(store, "write", looking)

    run_ok(workspace.args(command=CONVERT))

    ((written_into, names, final_exists),) = seen
    assert names == sorted(EXPORT_FILES)
    assert written_into.parent.parent == workspace.out_dir.parent
    assert written_into != workspace.out_dir
    assert not final_exists
    assert (workspace.out_dir / "manifest.json").is_file()


def test_only_the_five_files_it_wrote_are_hashed_by_convert(
    workspace: ExportWorkspace,
    converter: FakeConvertConverter,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    hashed: list[Path] = []
    real = ct2.sha256_of

    def spying(path: Path) -> str:
        hashed.append(path)
        return real(path)

    monkeypatch.setattr(ct2, "sha256_of", spying)

    run_ok(workspace.args(command=CONVERT))

    assert sorted(path.name for path in hashed) == sorted(EXPORT_FILES)


# ---- `distill export` and `export convert` are one command --------------------------------------


def snapshot_without_time(directory: Path) -> dict[str, Any]:
    """The files under `directory` (their bytes), the manifest without its `created`."""
    files: dict[str, Any] = dict(snapshot(directory))
    manifest = json.loads(files["manifest.json"])
    manifest["created"] = ""
    files["manifest.json"] = manifest
    return files


@pytest.mark.parametrize(
    "extra",
    [
        (),
        ("--quantization", "int8", "--name", "same", "--document-level"),
        ("--timeout", "30"),
    ],
    ids=["defaults", "every option", "a timeout"],
)
def test_distill_export_and_export_convert_print_the_same_and_make_the_same_directory(
    workspace: ExportWorkspace, converter: FakeConvertConverter, extra: tuple[str, ...]
) -> None:
    first = run_ok(workspace.args(*extra))
    made = snapshot_without_time(workspace.out_dir)
    shutil.rmtree(workspace.out_dir)
    workspace.log.unlink()

    second = run_ok(workspace.args(*extra, command=CONVERT))

    assert second.stdout == first.stdout
    assert snapshot_without_time(workspace.out_dir) == made
    assert len(converter.calls) == 2
    assert converter.calls[0].quantization == converter.calls[1].quantization


def test_the_two_names_are_one_function() -> None:
    from rinchen.cli import distill, export

    distill_names = {c.name: c.callback for c in distill.app.registered_commands}
    export_names = {c.name: c.callback for c in export.app.registered_commands}

    assert distill_names["export"] is export_names["convert"]


# ---- export verify ------------------------------------------------------------------------------


def made_export(workspace: ExportWorkspace) -> Path:
    run_ok(
        workspace.args("--quantization", "int8", "--document-level", command=CONVERT)
    )
    return workspace.out_dir


def test_verify_prints_what_the_record_says_and_the_digests(
    workspace: ExportWorkspace, converter: FakeConvertConverter
) -> None:
    directory = made_export(workspace)

    result = run_ok(["export", "verify", str(directory)])

    assert result.stdout.splitlines() == [
        f"export: {directory}",
        f"name: {DEFAULT_NAME}",
        "quantization: int8",
        "tag placement: src",
        "directions (2): eng_Latn-khk_Cyrl, khk_Cyrl-eng_Latn",
        "document level: true",
        f"model sha256 {sha256(directory / 'model.bin')[:12]}",
        f"manifest sha256 {sha256(directory / 'manifest.json')[:12]}",
    ]
    assert result.stderr == ""


def test_verify_reads_the_directions_of_a_fabricated_export(tmp_path: Path) -> None:
    written = write_model_export(tmp_path / "export", placement="tgt", name="toy")

    result = run_ok(["export", "verify", str(written.directory)])

    assert result.stdout.splitlines()[:6] == [
        f"export: {written.directory}",
        "name: toy",
        "quantization: float32",
        "tag placement: tgt",
        f"directions ({len(TOY_DIRECTIONS)}): {', '.join(TOY_DIRECTIONS)}",
        "document level: false",
    ]
    assert (
        result.stdout.splitlines()[6]
        == f"model sha256 {written.sha256['model.bin'][:12]}"
    )
    assert (
        result.stdout.splitlines()[7]
        == f"manifest sha256 {written.manifest_sha256[:12]}"
    )


def test_verify_writes_nothing(
    workspace: ExportWorkspace, converter: FakeConvertConverter
) -> None:
    directory = made_export(workspace)
    before = snapshot(directory)

    run_ok(["export", "verify", str(directory)])

    assert snapshot(directory) == before


def test_verify_makes_a_relative_directory_absolute(
    workspace: ExportWorkspace,
    converter: FakeConvertConverter,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    made_export(workspace)
    monkeypatch.chdir(workspace.root)

    result = run_ok(["export", "verify", "export"])

    assert result.stdout.splitlines()[0] == f"export: {workspace.out_dir}"


def assert_refused(result: Result, exit_code: int, message: str) -> None:
    assert_one_error(result, exit_code, message)
    assert result.stdout == ""


def test_the_layout_of_the_distillation_is_exit_2_and_names_the_remedy(
    tmp_path: Path,
) -> None:
    written = write_model_export(tmp_path / "old")
    (written.directory / "rinchen.json").rename(written.directory / "export.json")

    result = invoke(["export", "verify", str(written.directory)])

    assert_refused(result, 2, f"cannot use the export {written.directory}")
    assert "export.json" in result.stderr
    assert "rinchen export convert" in result.stderr


@pytest.mark.parametrize("name", EXPORT_FILES)
def test_a_flipped_byte_in_any_of_the_five_files_is_exit_2_naming_the_file(
    tmp_path: Path, name: str
) -> None:
    written = write_model_export(tmp_path / "export")
    flip_one_byte(written.directory / name)

    result = invoke(["export", "verify", str(written.directory)])

    assert_refused(result, 2, f"cannot use the export {written.directory}")
    assert f"{written.directory / name} differs from the manifest" in result.stderr


def test_a_directory_with_no_manifest_is_exit_2(tmp_path: Path) -> None:
    written = write_model_export(tmp_path / "export")
    (written.directory / "manifest.json").unlink()

    result = invoke(["export", "verify", str(written.directory)])

    assert_refused(result, 2, f"cannot use the export {written.directory}")
    assert "no manifest vouches for it" in result.stderr


def test_a_directory_that_does_not_exist_is_exit_2(tmp_path: Path) -> None:
    result = invoke(["export", "verify", str(tmp_path / "nowhere")])

    assert_refused(
        result, 2, f"the export directory {tmp_path / 'nowhere'} does not exist"
    )


def test_a_manifest_that_cannot_be_read_is_exit_3(tmp_path: Path) -> None:
    written = write_model_export(tmp_path / "export")
    (written.directory / "manifest.json").write_text("not json", encoding="utf-8")

    result = invoke(["export", "verify", str(written.directory)])

    assert_refused(
        result, 3, f"{written.directory / 'manifest.json'}: is not a readable"
    )


def test_verify_needs_a_directory() -> None:
    result = invoke(["export", "verify"])

    assert result.exit_code == 2
    assert "Missing argument" in result.stderr
