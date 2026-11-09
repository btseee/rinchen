"""Opening an `Engine` (D11): what a directory must be, what is checked before a model is
loaded, how the library is called, and what the engine says about itself.

`ctranslate2` is made unimportable for every test except where one installs a fake of it, so
a test that reached the real library would fail instead of loading it. Nothing touches a GPU.
"""

from __future__ import annotations

import ast
import os
import subprocess
import sys
import types
from pathlib import Path
from typing import Any

import pytest

import rinchen.infer
from rinchen.errors import ConfigError, DataError, UpstreamError
from rinchen.export.record import EXPORT_FILES
from rinchen.infer import engine as engine_module
from rinchen.infer.engine import Engine
from rinchen.infer.engine_types import ENGINE_COUNTERS, LoadOptions
from tests.support import model_exports
from tests.support.engines import (
    ENG,
    ENGINE_PIECES,
    ENGINE_TOKENS,
    KHK,
    SP,
    TAG_ENG,
    TAG_KHK,
    LiveCt2,
    engine_environment,  # noqa: F401 - the fixture of `pytestmark`
    scripted_engine,
)
from tests.support.fake_ct2 import FakeProcessor, ScriptedCt2
from tests.support.model_exports import TOY_DIRECTIONS, write_model_export
from tests.support.tiny_parallel import train_toy_spm

pytestmark = pytest.mark.usefixtures("engine_environment")


def opened(directory: Path, **keywords: Any) -> Engine:
    keywords.setdefault("translator", LiveCt2())
    keywords.setdefault("processor", FakeProcessor(ENGINE_PIECES))
    return Engine.open(directory, **keywords)


# -- what the engine says about itself


def test_the_counters_are_the_sixteen_names_of_the_engine_and_start_at_zero(
    tmp_path: Path,
) -> None:
    assert ENGINE_COUNTERS == (
        "sources",
        "empty",
        "too_long",
        "truncated",
        "no_output",
        "reserved_symbol",
        "line_break",
        "zero_shot",
        "script_retries",
        "script_retry_improved",
        "script_impure",
        "script_conversions",
        "converter_rule_fallbacks",
        "converter_invalid_outputs",
        "windows",
        "window_fallbacks",
    )
    assert scripted_engine(tmp_path).engine.counters() == dict.fromkeys(
        ENGINE_COUNTERS, 0
    )


def test_the_counters_are_a_copy(tmp_path: Path) -> None:
    engine = scripted_engine(tmp_path).engine

    engine.counters()["sources"] = 99

    assert engine.counters()["sources"] == 0


def test_the_info_names_the_model_and_the_digests_it_was_vouched_for_by(
    tmp_path: Path,
) -> None:
    fixture = scripted_engine(
        tmp_path, quantization="int8", compute_type="int8_float32"
    )

    info = fixture.engine.info

    assert info.name == fixture.written.record.name
    assert info.quantization == "int8"
    assert info.device == "cpu"
    assert info.compute_type == "int8_float32"
    assert info.placement == "src"
    assert info.directions == TOY_DIRECTIONS
    assert info.targets == (ENG, KHK, "rus_Cyrl")
    assert info.document_level is False
    assert info.model_sha256 == fixture.written.sha256["model.bin"]
    assert info.manifest_sha256 == fixture.written.manifest_sha256


def test_the_compute_type_is_the_one_the_library_reports_and_is_logged_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    logged: list[tuple[str, dict[str, Any]]] = []

    class Recorder:
        def info(self, event: str, **fields: Any) -> None:
            logged.append((event, fields))

    names: list[str] = []

    def get(name: str) -> Recorder:
        names.append(name)
        return Recorder()

    monkeypatch.setattr("rinchen.utils.logging.get", get)

    fixture = scripted_engine(
        tmp_path, quantization="int8", compute_type="int8_float32"
    )
    fixture.engine.translate([], None)

    assert names == ["rinchen.infer.engine"]
    assert logged == [
        (
            "compute_type_effective",
            {
                "model": fixture.written.record.name,
                "quantization": "int8",
                "device": "cpu",
                "compute_type": "int8_float32",
            },
        )
    ]


# -- what is refused before a model is loaded


def test_a_directory_of_the_old_format_is_refused_with_the_remedy(
    tmp_path: Path,
) -> None:
    old = tmp_path / "old"
    old.mkdir()
    (old / "export.json").write_text("{}", encoding="utf-8")

    with pytest.raises(ConfigError, match="rinchen export convert"):
        opened(old)


def test_a_directory_with_no_manifest_is_refused(tmp_path: Path) -> None:
    written = write_model_export(tmp_path / "model", tokens=ENGINE_TOKENS)
    (written.directory / "manifest.json").unlink()

    with pytest.raises(ConfigError, match="manifest"):
        opened(written.directory)


@pytest.mark.parametrize("name", EXPORT_FILES)
def test_one_flipped_byte_in_any_of_the_five_files_is_refused(
    tmp_path: Path, name: str
) -> None:
    written = write_model_export(tmp_path / "model", tokens=ENGINE_TOKENS)
    file = written.directory / name
    content = bytearray(file.read_bytes())
    content[len(content) // 2] ^= 0x01
    file.write_bytes(bytes(content))

    with pytest.raises(ConfigError, match=name.replace(".", r"\.")):
        opened(written.directory)


def test_a_missing_directory_is_a_config_error(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="does not exist"):
        opened(tmp_path / "nowhere")


def test_a_tag_of_the_record_that_is_not_in_the_vocabulary_is_refused_at_load(
    tmp_path: Path,
) -> None:
    tokens = tuple(token for token in ENGINE_TOKENS if token != TAG_ENG)
    written = write_model_export(tmp_path / "model", tokens=tokens)

    with pytest.raises(ConfigError) as excinfo:
        opened(written.directory)

    assert TAG_ENG in str(excinfo.value)
    assert "shared_vocabulary.json" in str(excinfo.value)


def test_a_tag_that_is_not_one_piece_of_the_sentencepiece_model_is_refused_at_load(
    tmp_path: Path,
) -> None:
    written = write_model_export(tmp_path / "model", tokens=ENGINE_TOKENS)
    pieces = [piece for piece in ENGINE_PIECES if piece != TAG_ENG]

    with pytest.raises(ConfigError) as excinfo:
        opened(written.directory, processor=FakeProcessor(pieces))

    assert TAG_ENG in str(excinfo.value)
    assert "one piece" in str(excinfo.value)


def test_a_tag_that_is_a_piece_of_another_spelling_is_refused(tmp_path: Path) -> None:
    class Splitting(FakeProcessor):
        def id_to_piece(self, piece_id: int) -> str:
            piece = super().id_to_piece(piece_id)
            return SP + piece if piece == TAG_ENG else piece

    written = write_model_export(tmp_path / "model", tokens=ENGINE_TOKENS)

    with pytest.raises(ConfigError, match="one piece"):
        opened(written.directory, processor=Splitting(ENGINE_PIECES))


def test_every_tag_of_the_record_is_checked_not_only_the_first(tmp_path: Path) -> None:
    written = write_model_export(tmp_path / "model", tokens=ENGINE_TOKENS)
    pieces = [piece for piece in ENGINE_PIECES if piece != "<2rus_Cyrl>"]

    with pytest.raises(ConfigError, match="<2rus_Cyrl>"):
        opened(written.directory, processor=FakeProcessor(pieces))


def test_a_config_without_an_end_token_is_a_data_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    usual = model_exports.converter_files

    def converter_files(tokens: Any) -> dict[str, bytes]:
        return {**usual(tokens), "config.json": b'{"bos_token": "<s>"}'}

    monkeypatch.setattr(model_exports, "converter_files", converter_files)
    written = write_model_export(tmp_path / "model", tokens=ENGINE_TOKENS)

    with pytest.raises(DataError, match="eos_token"):
        opened(written.directory)


def test_a_vocabulary_that_is_not_a_list_of_strings_is_a_data_error(
    tmp_path: Path,
) -> None:
    written = write_model_export(
        tmp_path / "model", tokens=ENGINE_TOKENS, vocabulary_bytes=b'{"a": 1}'
    )

    with pytest.raises(DataError, match=r"shared_vocabulary\.json"):
        opened(written.directory)


def test_a_sentencepiece_model_that_does_not_load_is_a_config_error(
    tmp_path: Path,
) -> None:
    written = write_model_export(
        tmp_path / "model", tokens=ENGINE_TOKENS, spm_bytes=b"not a model"
    )

    with pytest.raises(ConfigError, match=r"spm\.model"):
        Engine.open(written.directory, translator=LiveCt2())


@pytest.mark.parametrize("name", ["shared_vocabulary.json", "config.json", "spm.model"])
def test_a_file_that_changes_between_the_verification_and_its_read_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, name: str
) -> None:
    written = write_model_export(tmp_path / "model", tokens=ENGINE_TOKENS)
    real = engine_module.verify_export

    def verify_then_change(directory: Path) -> Any:
        export = real(directory)
        (export.directory / name).write_bytes(b"changed")
        return export

    monkeypatch.setattr(engine_module, "verify_export", verify_then_change)

    with pytest.raises(ConfigError, match=rf"{name.replace('.', r'\.')} changed"):
        Engine.open(written.directory, translator=LiveCt2())


def test_a_wrong_file_costs_no_model_load(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    loaded: list[str] = []
    fake = types.SimpleNamespace(
        Translator=lambda *args, **kwargs: loaded.append("loaded")
    )
    monkeypatch.setitem(sys.modules, "ctranslate2", fake)
    written = write_model_export(tmp_path / "model", tokens=ENGINE_TOKENS)
    pieces = [piece for piece in ENGINE_PIECES if piece != TAG_ENG]

    with pytest.raises(ConfigError):
        Engine.open(written.directory, processor=FakeProcessor(pieces))

    assert loaded == []


@pytest.mark.parametrize(
    "options",
    [
        {"compute_type": "auto"},
        {"compute_type": "AUTO"},
        {"compute_type": " "},
        {"device": "auto"},
        {"device": "gpu"},
        {"inter_threads": 0},
        {"inter_threads": True},
        {"inter_threads": 1.5},
        {"intra_threads": 0},
        {"intra_threads": False},
        {"intra_threads": -2},
        {"max_queued_batches": True},
        {"max_queued_batches": -2},
    ],
)
def test_options_the_engine_cannot_use_are_config_errors(
    options: dict[str, Any],
) -> None:
    with pytest.raises(ConfigError):
        LoadOptions(**options)


def test_auto_is_refused_with_the_reason() -> None:
    with pytest.raises(ConfigError, match="quantises"):
        LoadOptions(compute_type="auto")
    with pytest.raises(ConfigError, match="GPU"):
        LoadOptions(device="auto")


def test_the_defaults_are_the_cpu_one_thread_each_and_what_the_file_stores() -> None:
    assert LoadOptions() == LoadOptions(
        device="cpu",
        compute_type="default",
        inter_threads=1,
        intra_threads=1,
        max_queued_batches=-1,
    )


# -- how the library is called


class FakeCtranslate2:
    """Stands in for the `ctranslate2` module: `Translator(...)` records how it was called
    and returns `stub`, or raises `error`."""

    def __init__(self, stub: Any = None, error: BaseException | None = None) -> None:
        self.stub = LiveCt2() if stub is None else stub
        self.error = error
        self.created: list[tuple[tuple[Any, ...], dict[str, Any]]] = []

    def Translator(self, *args: Any, **kwargs: Any) -> Any:
        self.created.append((args, kwargs))
        if self.error is not None:
            raise self.error
        return self.stub


def install(monkeypatch: pytest.MonkeyPatch, fake: FakeCtranslate2) -> None:
    monkeypatch.setitem(
        sys.modules, "ctranslate2", types.SimpleNamespace(Translator=fake.Translator)
    )


def written_model(tmp_path: Path) -> Path:
    return write_model_export(tmp_path / "model", tokens=ENGINE_TOKENS).directory


def test_the_library_translator_is_made_with_every_option_named(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    directory = written_model(tmp_path)
    fake = FakeCtranslate2()
    install(monkeypatch, fake)

    Engine.open(directory, processor=FakeProcessor(ENGINE_PIECES))

    assert fake.created == [
        (
            (str(directory),),
            {
                "device": "cpu",
                "compute_type": "default",
                "inter_threads": 1,
                "intra_threads": 1,
                "max_queued_batches": -1,
            },
        )
    ]


def test_the_device_the_type_and_the_threads_reach_the_library(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    directory = written_model(tmp_path)
    fake = FakeCtranslate2()
    install(monkeypatch, fake)
    load = LoadOptions(
        device="cuda",
        compute_type="int8_float16",
        inter_threads=3,
        intra_threads=2,
        max_queued_batches=8,
    )

    Engine.open(directory, load, processor=FakeProcessor(ENGINE_PIECES))

    ((_, keywords),) = fake.created
    assert keywords == {
        "device": "cuda",
        "compute_type": "int8_float16",
        "inter_threads": 3,
        "intra_threads": 2,
        "max_queued_batches": 8,
    }


def test_a_device_or_type_the_library_rejects_is_a_config_error_with_the_cause(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    directory = written_model(tmp_path)
    error = ValueError("unsupported compute type int8_float16")
    install(monkeypatch, FakeCtranslate2(error=error))

    with pytest.raises(ConfigError, match="cpu.*default") as excinfo:
        Engine.open(directory, processor=FakeProcessor(ENGINE_PIECES))

    assert "unsupported compute type" in str(excinfo.value)
    assert excinfo.value.__cause__ is error


@pytest.mark.parametrize(
    "error", [RuntimeError("Unable to open file 'model.bin'"), OSError("gone")]
)
def test_a_model_the_library_cannot_load_is_a_data_error_naming_the_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, error: Exception
) -> None:
    directory = written_model(tmp_path)
    install(monkeypatch, FakeCtranslate2(error=error))

    with pytest.raises(DataError) as excinfo:
        Engine.open(directory, processor=FakeProcessor(ENGINE_PIECES))

    assert str(directory) in str(excinfo.value)
    assert excinfo.value.__cause__ is error


def test_a_translator_given_to_open_is_used_and_the_library_is_never_imported(
    tmp_path: Path,
) -> None:
    # `ctranslate2` is None in `sys.modules` for every test of this file: an import fails.
    fixture = scripted_engine(tmp_path)

    assert fixture.engine.translate([], None) == []


def test_the_real_sentencepiece_model_is_read_from_the_verified_bytes_of_the_export(
    tmp_path_factory: pytest.TempPathFactory,
) -> None:
    model = train_toy_spm(tmp_path_factory.mktemp("engine_spm"))
    import sentencepiece

    real = sentencepiece.SentencePieceProcessor(model_file=str(model))
    tokens = (
        "<unk>",
        "<blank>",
        "<s>",
        "</s>",
        *(real.id_to_piece(index) for index in range(3, real.get_piece_size())),
    )
    written = write_model_export(
        tmp_path_factory.mktemp("engine_export") / "model",
        tokens=tokens,
        spm_bytes=model.read_bytes(),
    )

    engine = Engine.open(written.directory, translator=LiveCt2())

    assert engine.info.targets == (ENG, KHK, "rus_Cyrl")


# -- no heavy library at import time


PACKAGE = Path(rinchen.infer.__file__).parent
HEAVY = ("ctranslate2", "sentencepiece", "eole", "torch")


@pytest.mark.parametrize("name", ["engine.py", "engine_types.py", "translator.py"])
def test_the_engine_modules_import_no_heavy_library_at_the_top_level(name: str) -> None:
    imported: set[str] = set()
    for node in ast.parse((PACKAGE / name).read_text(encoding="utf-8")).body:
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            imported.add(node.module)

    assert [m for m in imported if m.split(".")[0] in HEAVY] == []


def test_importing_the_engine_loads_no_heavy_library() -> None:
    source_root = str(PACKAGE.parent.parent)
    code = (
        "import sys\n"
        "sys.path.insert(0, sys.argv[1])\n"
        "import rinchen.infer.engine, rinchen.infer.translator\n"
        "assert rinchen.__file__.startswith(sys.argv[1]), rinchen.__file__\n"
        f"print([m for m in {HEAVY!r} if m in sys.modules])\n"
    )

    result = subprocess.run(
        [sys.executable, "-I", "-c", code, source_root],
        capture_output=True,
        text=True,
        env={**os.environ, "CUDA_VISIBLE_DEVICES": ""},
        timeout=120,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "[]"


# -- closing and warming up


def test_close_unloads_the_model_once_and_a_closed_engine_translates_nothing(
    tmp_path: Path,
) -> None:
    fixture = scripted_engine(tmp_path)

    fixture.engine.close()
    fixture.engine.close()

    assert isinstance(fixture.ct2, LiveCt2) and fixture.ct2.unloaded == 1
    with pytest.raises(ConfigError, match="closed"):
        fixture.engine.translate([], None)


def test_a_model_without_an_unload_method_closes_without_one(tmp_path: Path) -> None:
    written = write_model_export(tmp_path / "model", tokens=ENGINE_TOKENS)
    engine = Engine.open(
        written.directory,
        translator=ScriptedCt2(lambda line: []),
        processor=FakeProcessor(ENGINE_PIECES),
    )

    engine.close()


def test_the_warmup_translates_one_fixed_text_into_the_first_target_and_counts_nothing(
    tmp_path: Path,
) -> None:
    fixture = scripted_engine(tmp_path)

    fixture.engine.warmup()

    ((call),) = fixture.ct2.calls
    assert [line[0] for line in call.source] == [TAG_ENG]
    assert len(call.source) == 1
    assert fixture.engine.counters() == dict.fromkeys(ENGINE_COUNTERS, 0)


def test_a_warmup_the_library_fails_is_an_upstream_error(tmp_path: Path) -> None:
    fixture = scripted_engine(tmp_path)
    fixture.ct2.error = RuntimeError("out of memory")

    with pytest.raises(UpstreamError):
        fixture.engine.warmup()


def test_the_warmup_asks_for_the_first_target_of_a_model_that_has_one(
    tmp_path: Path,
) -> None:
    fixture = scripted_engine(tmp_path, directions=("eng_Latn-khk_Cyrl",))

    fixture.engine.warmup()

    assert fixture.engine.info.targets == (KHK,)
    assert fixture.ct2.calls[0].source[0][0] == TAG_KHK
