"""The teacher's pre-flight of the direction tags and how it loads its files and its model
(D10). `ctranslate2` is made unimportable for every test except where a test installs a
fake of it, so a test that reached the real library would fail instead of loading it.
Nothing touches a GPU."""

from __future__ import annotations

import subprocess
import sys
import types

import pytest

from rinchen.errors import ConfigError, DataError
from rinchen.train import distill_teacher
from rinchen.train.distill_teacher import TeacherOptions, TeacherTranslator
from tests.support.distill_teachers import (
    END,
    ENG,
    KHK,
    RUS,
    SP,
    TAG,
    config_bytes,
    export_with_files,
    hyp,
    scripted_teacher,
    tag_processor,
    verified_export,
)
from tests.support.fake_ct2 import FakeProcessor, ScriptedCt2


@pytest.fixture(autouse=True)
def real_ctranslate2_is_unimportable(monkeypatch):
    monkeypatch.setitem(sys.modules, "ctranslate2", None)


# -- the pre-flight of the direction tags


def test_check_directions_passes_when_every_target_tag_is_known_and_one_piece(
    tmp_path,
):
    teacher, _ = scripted_teacher(tmp_path, {})

    teacher.check_directions([f"{ENG}-{KHK}", f"{RUS}-{KHK}", f"{KHK}-{ENG}"])
    teacher.check_directions([])


def test_check_directions_names_a_tag_that_is_not_in_the_vocabulary(tmp_path):
    teacher, _ = scripted_teacher(
        tmp_path, {}, tokens=("<unk>", "<blank>", "<s>", END, "a", TAG)
    )

    with pytest.raises(ConfigError) as excinfo:
        teacher.check_directions([f"{KHK}-{ENG}"])

    assert "<2eng_Latn>" in str(excinfo.value)
    assert "shared_vocabulary.json" in str(excinfo.value)


def test_check_directions_names_a_tag_the_sentencepiece_model_would_split(tmp_path):
    teacher, _ = scripted_teacher(tmp_path, {}, processor=FakeProcessor([TAG]))

    with pytest.raises(ConfigError) as excinfo:
        teacher.check_directions([f"{KHK}-{KHK}", f"{KHK}-{ENG}"])

    assert "<2eng_Latn>" in str(excinfo.value)
    assert "one piece" in str(excinfo.value)


def test_check_directions_looks_only_at_the_target_languages(tmp_path):
    teacher, _ = scripted_teacher(
        tmp_path, {}, tokens=("<unk>", "<blank>", "<s>", END, TAG)
    )

    teacher.check_directions([f"{ENG}-{KHK}", f"{RUS}-{KHK}"])


@pytest.mark.parametrize(
    ("direction", "named"),
    [
        ("engkhk", "engkhk"),
        (f"{ENG}-", f"{ENG}-"),
        (f"-{KHK}", f"-{KHK}"),
        (f"{ENG}-{KHK}-{RUS}", f"{ENG}-{KHK}-{RUS}"),
        (f"{ENG}-fra_Latn", "fra_Latn"),
        (f"fra_Latn-{KHK}", "fra_Latn"),
    ],
)
def test_check_directions_refuses_what_is_not_a_registry_direction(
    tmp_path, direction, named
):
    teacher, _ = scripted_teacher(tmp_path, {})

    with pytest.raises(ConfigError) as excinfo:
        teacher.check_directions([direction])

    assert named in str(excinfo.value)


# -- loading


def test_the_vocabulary_and_the_config_are_read_through_the_verified_digests(
    tmp_path,
):
    export = verified_export(tmp_path)
    (tmp_path / "export" / "shared_vocabulary.json").write_bytes(b'["<unk>"]')

    with pytest.raises(ConfigError, match=r"shared_vocabulary\.json.*changed"):
        TeacherTranslator(
            export,
            TeacherOptions(),
            translator=ScriptedCt2({}),
            processor=tag_processor(),
        )


def test_a_config_changed_after_the_verification_is_refused(tmp_path):
    export = verified_export(tmp_path)
    config = tmp_path / "export" / "config.json"
    config.write_bytes(config.read_bytes() + b" ")

    with pytest.raises(ConfigError, match=r"config\.json.*changed"):
        TeacherTranslator(
            export,
            TeacherOptions(),
            translator=ScriptedCt2({}),
            processor=tag_processor(),
        )


@pytest.mark.parametrize(
    "config",
    [
        config_bytes(bos_token="<s>"),
        config_bytes(eos_token=""),
        config_bytes(eos_token=5),
        config_bytes(eos_token=None),
        b'["</s>"]',
        b"not json",
    ],
    ids=["absent", "empty", "number", "null", "not-an-object", "not-json"],
)
def test_an_end_token_the_config_does_not_state_is_a_data_error(
    tmp_path, monkeypatch, config
):
    export = export_with_files(tmp_path, monkeypatch, **{"config.json": config})

    with pytest.raises(DataError, match=r"config\.json"):
        TeacherTranslator(
            export,
            TeacherOptions(),
            translator=ScriptedCt2({}),
            processor=tag_processor(),
        )


@pytest.mark.parametrize(
    "vocabulary",
    [b'{"a": 1}', b'["a", 2]', b"not json", b"[]"],
    ids=["object", "non-string", "not-json", "empty"],
)
def test_a_shared_vocabulary_that_is_not_a_list_of_strings_is_a_data_error(
    tmp_path, monkeypatch, vocabulary
):
    export = export_with_files(
        tmp_path, monkeypatch, **{"shared_vocabulary.json": vocabulary}
    )

    with pytest.raises(DataError, match=r"shared_vocabulary\.json"):
        TeacherTranslator(
            export,
            TeacherOptions(),
            translator=ScriptedCt2({}),
            processor=tag_processor(),
        )


def test_the_processor_is_loaded_from_the_exports_copy_of_the_model(
    tmp_path, monkeypatch
):
    export = verified_export(tmp_path)
    loaded = []

    def load(path):
        loaded.append(path)
        return tag_processor()

    monkeypatch.setattr(distill_teacher, "load_processor", load)

    TeacherTranslator(export, TeacherOptions(), translator=ScriptedCt2({}))

    assert loaded == [export.directory / "spm.model"]


def test_a_model_that_is_not_the_verified_copy_is_refused_before_it_is_loaded(
    tmp_path, monkeypatch
):
    export = verified_export(tmp_path)
    model = export.directory / "spm.model"
    model.write_bytes(model.read_bytes() + b"x")
    monkeypatch.setattr(
        distill_teacher, "load_processor", lambda path: pytest.fail("loaded")
    )

    with pytest.raises(ConfigError, match=r"spm\.model.*changed"):
        TeacherTranslator(export, TeacherOptions(), translator=ScriptedCt2({}))


def test_a_sentencepiece_model_that_does_not_load_is_a_config_error(tmp_path):
    export = verified_export(tmp_path, spm_bytes=b"not a model")

    with pytest.raises(ConfigError, match=r"spm\.model"):
        TeacherTranslator(export, TeacherOptions(), translator=ScriptedCt2({}))


class FakeCtranslate2:
    """Stands in for the `ctranslate2` module: `Translator(...)` records how it was
    called and returns `stub`, or raises `error`."""

    def __init__(self, stub=None, error=None):
        self.stub = ScriptedCt2({}) if stub is None else stub
        self.error = error
        self.created: list[tuple[tuple, dict]] = []

    def Translator(self, *args, **kwargs):
        self.created.append((args, kwargs))
        if self.error is not None:
            raise self.error
        return self.stub


def _install(monkeypatch, fake):
    monkeypatch.setitem(
        sys.modules, "ctranslate2", types.SimpleNamespace(Translator=fake.Translator)
    )


def test_the_library_translator_is_made_on_the_cpu_with_one_thread_by_default(
    tmp_path, monkeypatch
):
    export = verified_export(tmp_path)
    fake = FakeCtranslate2(ScriptedCt2({(TAG, SP + "a"): [hyp(SP + "x", END)]}))
    _install(monkeypatch, fake)

    teacher = TeacherTranslator(export, TeacherOptions(), processor=tag_processor())

    assert fake.created == [
        (
            (str(export.directory),),
            {"device": "cpu", "compute_type": "default", "intra_threads": 1},
        )
    ]
    assert teacher.translate(["a"], ENG, KHK) == [["x"]]


def test_the_device_the_compute_type_and_the_threads_reach_the_library(
    tmp_path, monkeypatch
):
    export = verified_export(tmp_path)
    fake = FakeCtranslate2()
    _install(monkeypatch, fake)
    options = TeacherOptions(device="cuda", compute_type="int8_float16", threads=3)

    TeacherTranslator(export, options, processor=tag_processor())

    ((_, kwargs),) = fake.created
    assert kwargs == {
        "device": "cuda",
        "compute_type": "int8_float16",
        "intra_threads": 3,
    }


def test_a_device_or_compute_type_the_library_rejects_is_a_config_error(
    tmp_path, monkeypatch
):
    export = verified_export(tmp_path)
    error = ValueError("unsupported compute type int8_float16")
    _install(monkeypatch, FakeCtranslate2(error=error))

    with pytest.raises(ConfigError, match="cpu.*default") as excinfo:
        TeacherTranslator(export, TeacherOptions(), processor=tag_processor())

    assert "unsupported compute type" in str(excinfo.value)
    assert excinfo.value.__cause__ is error


@pytest.mark.parametrize(
    "error", [RuntimeError("Unable to open file 'model.bin'"), OSError("gone")]
)
def test_a_model_the_library_cannot_load_is_a_data_error_naming_the_directory(
    tmp_path, monkeypatch, error
):
    export = verified_export(tmp_path)
    _install(monkeypatch, FakeCtranslate2(error=error))

    with pytest.raises(DataError) as excinfo:
        TeacherTranslator(export, TeacherOptions(), processor=tag_processor())

    assert str(export.directory) in str(excinfo.value)
    assert excinfo.value.__cause__ is error


def test_a_config_without_an_end_token_is_refused_before_the_model_is_loaded(
    tmp_path, monkeypatch
):
    # `ctranslate2` is unimportable here: reaching it would raise ImportError.
    export = export_with_files(
        tmp_path, monkeypatch, **{"config.json": config_bytes(bos_token="<s>")}
    )

    with pytest.raises(DataError, match="eos_token"):
        TeacherTranslator(export, TeacherOptions(), processor=tag_processor())


def test_a_sentencepiece_model_that_does_not_load_is_refused_before_the_model_is_loaded(
    tmp_path,
):
    export = verified_export(tmp_path, spm_bytes=b"not a model")

    with pytest.raises(ConfigError, match=r"spm\.model"):
        TeacherTranslator(export, TeacherOptions())


@pytest.mark.subprocess
def test_importing_the_module_and_the_package_does_not_import_ctranslate2():
    code = (
        "import sys\n"
        "import rinchen.train\n"
        "import rinchen.train.distill_teacher\n"
        "assert 'ctranslate2' not in sys.modules, 'ctranslate2 was imported'\n"
    )

    result = subprocess.run(
        [sys.executable, "-I", "-c", code],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )

    assert result.returncode == 0, result.stderr
