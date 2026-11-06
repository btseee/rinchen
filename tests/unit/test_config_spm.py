# tests/unit/test_config_spm.py
from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from rinchen.config.loader import REPO_ROOT
from rinchen.config.spm import SpmConfig, load_spm_config
from rinchen.errors import ConfigError

SPM_CONFIG_DIR = REPO_ROOT / "configs" / "spm"

# Raw YAML scalar text per field, so a case can inject one precise defect.
_VALID_FIELDS: dict[str, str] = {
    "vocab_size": "1000",
    "input_sentence_size": "100000",
    "character_coverage": "0.9999",
    "temperature": "0.3",
    "seed": "1234",
    "corpus_dir": "corpora/spm",
    "output_prefix": "out/spm_model",
}


def _spm_yaml(**fields: str | None) -> str:
    """A valid config document; a field set to a string replaces that value's
    raw YAML text, a field set to None is dropped, and an unknown name is
    appended as an extra field."""
    merged = {**_VALID_FIELDS, **fields}
    return "".join(
        f"{key}: {value}\n" for key, value in merged.items() if value is not None
    )


def _write(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "spm.yaml"
    path.write_text(text, encoding="utf-8")
    return path


@pytest.mark.parametrize(
    ("name", "vocab_size", "input_sentence_size", "corpus_dir", "output_prefix"),
    [
        pytest.param(
            "tiny_1k",
            1000,
            100_000,
            "tests/fixtures/spm/tiny_corpus",
            "data/tiny/spm/tiny_1k",
            id="tiny_1k",
        ),
        pytest.param(
            "student_64k",
            64_000,
            10_000_000,
            "data/clean/spm_corpus",
            "models/spm/student_64k",
            id="student_64k",
        ),
        pytest.param(
            "teacher_128k",
            128_000,
            20_000_000,
            "data/clean/spm_corpus",
            "models/spm/teacher_128k",
            id="teacher_128k",
        ),
    ],
)
def test_shipped_configs_load(
    name: str,
    vocab_size: int,
    input_sentence_size: int,
    corpus_dir: str,
    output_prefix: str,
):
    config = load_spm_config(SPM_CONFIG_DIR / f"{name}.yaml")

    assert config.vocab_size == vocab_size
    assert config.input_sentence_size == input_sentence_size
    assert config.character_coverage == 0.9999
    assert config.temperature == 0.3
    assert config.seed == 1234
    assert config.corpus_dir == REPO_ROOT / corpus_dir
    assert config.output_prefix == REPO_ROOT / output_prefix
    assert config.corpus_dir.is_absolute()
    assert config.output_prefix.is_absolute()


def test_relative_paths_resolve_against_repo_root_and_absolute_stay(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    absolute_prefix = tmp_path / "abs" / "model_prefix"
    path = _write(
        tmp_path,
        _spm_yaml(corpus_dir="corpora/spm", output_prefix=str(absolute_prefix)),
    )

    config = load_spm_config(path)

    assert config.corpus_dir == REPO_ROOT / "corpora" / "spm"
    assert config.output_prefix == absolute_prefix


def test_loaded_config_is_immutable(tmp_path: Path):
    config = load_spm_config(_write(tmp_path, _spm_yaml()))

    with pytest.raises(ValidationError):
        config.vocab_size = 2000  # type: ignore[misc]


def test_valid_config_round_trips_every_field(tmp_path: Path):
    config = load_spm_config(_write(tmp_path, _spm_yaml()))

    assert config == SpmConfig(
        vocab_size=1000,
        input_sentence_size=100_000,
        character_coverage=0.9999,
        temperature=0.3,
        seed=1234,
        corpus_dir=REPO_ROOT / "corpora" / "spm",
        output_prefix=REPO_ROOT / "out" / "spm_model",
    )


@pytest.mark.parametrize(
    ("yaml_text", "message_fragment"),
    [
        pytest.param(
            _spm_yaml(vocab_size=None), "vocab_size.*Field required", id="missing-field"
        ),
        pytest.param(
            _spm_yaml(unexpected_option="1"),
            "unexpected_option.*Extra inputs are not permitted",
            id="extra-field",
        ),
        pytest.param(
            _spm_yaml(vocab_size="0"), "vocab_size.*greater than 0", id="zero-vocab"
        ),
        pytest.param(
            _spm_yaml(vocab_size="-5"),
            "vocab_size.*greater than 0",
            id="negative-vocab",
        ),
        pytest.param(
            _spm_yaml(input_sentence_size="0"),
            "input_sentence_size.*greater than 0",
            id="zero-input-sentence-size",
        ),
        pytest.param(
            _spm_yaml(temperature="0"), "temperature.*greater than 0", id="zero-temp"
        ),
        pytest.param(
            _spm_yaml(temperature="-0.3"),
            "temperature.*greater than 0",
            id="negative-temp",
        ),
        pytest.param(
            _spm_yaml(temperature=".inf"),
            "temperature.*finite",
            id="infinite-temp",
        ),
        pytest.param(
            _spm_yaml(temperature=".nan"),
            "temperature.*finite",
            id="nan-temp",
        ),
        pytest.param(
            _spm_yaml(character_coverage="1.5"),
            "character_coverage.*less than or equal to 1",
            id="coverage-above-one",
        ),
        pytest.param(
            _spm_yaml(character_coverage="0"),
            "character_coverage.*greater than 0",
            id="zero-coverage",
        ),
        pytest.param(
            _spm_yaml(character_coverage=".inf"),
            "character_coverage.*finite",
            id="infinite-coverage",
        ),
        pytest.param(
            _spm_yaml(character_coverage=".nan"),
            "character_coverage.*finite",
            id="nan-coverage",
        ),
        pytest.param(
            _spm_yaml(vocab_size="1000.5"),
            "vocab_size.*valid integer",
            id="fractional-vocab",
        ),
        pytest.param(
            _spm_yaml(vocab_size="'1000'"),
            "vocab_size.*Input should be a valid integer",
            id="quoted-vocab-string",
        ),
        pytest.param(
            _spm_yaml(seed="true"),
            "seed.*Input should be a valid integer",
            id="boolean-seed",
        ),
        pytest.param(
            _spm_yaml(temperature="warm"),
            "temperature.*valid number",
            id="non-numeric-temp",
        ),
        pytest.param(
            _spm_yaml(corpus_dir="[a, b]"),
            "corpus_dir.*Input is not a valid path",
            id="corpus-dir-list",
        ),
        pytest.param(
            _spm_yaml(corpus_dir='""'),
            "Value error, corpus_dir must not be empty",
            id="empty-corpus-dir",
        ),
        pytest.param(
            _spm_yaml(corpus_dir='"   "'),
            "Value error, corpus_dir must not be empty",
            id="blank-corpus-dir",
        ),
        pytest.param(
            _spm_yaml(output_prefix='""'),
            "Value error, output_prefix must not be empty",
            id="empty-output-prefix",
        ),
        pytest.param(
            _spm_yaml(output_prefix='"   "'),
            "Value error, output_prefix must not be empty",
            id="blank-output-prefix",
        ),
        pytest.param(
            "- vocab_size: 1000\n", "top level must be a mapping", id="top-level-list"
        ),
        pytest.param("", "top level must be a mapping", id="empty-file"),
    ],
)
def test_malformed_config_raises_config_error(
    tmp_path: Path, yaml_text: str, message_fragment: str
):
    path = _write(tmp_path, yaml_text)

    # (?s): pydantic puts the field name and its message on separate lines.
    with pytest.raises(ConfigError, match=f"(?s){message_fragment}") as excinfo:
        load_spm_config(path)

    assert path.name in str(excinfo.value)


def test_upper_bound_of_character_coverage_is_inclusive(tmp_path: Path):
    config = load_spm_config(_write(tmp_path, _spm_yaml(character_coverage="1")))

    assert config.character_coverage == 1


def test_integer_valued_temperature_loads(tmp_path: Path):
    config = load_spm_config(_write(tmp_path, _spm_yaml(temperature="1")))

    assert config.temperature == 1


def test_field_violation_chains_the_validation_error(tmp_path: Path):
    path = _write(tmp_path, _spm_yaml(vocab_size="0"))

    with pytest.raises(ConfigError) as excinfo:
        load_spm_config(path)

    assert isinstance(excinfo.value.__cause__, ValidationError)


def test_yaml_syntax_error_raises_config_error_with_cause(tmp_path: Path):
    path = _write(tmp_path, "vocab_size: [1000\n")

    with pytest.raises(ConfigError, match="spm.yaml") as excinfo:
        load_spm_config(path)

    assert isinstance(excinfo.value.__cause__, yaml.YAMLError)


def test_non_utf8_config_raises_config_error_with_cause(tmp_path: Path):
    path = tmp_path / "spm.yaml"
    path.write_bytes(b"\xff\xfe\x00")

    with pytest.raises(ConfigError, match="spm.yaml") as excinfo:
        load_spm_config(path)

    assert isinstance(excinfo.value.__cause__, UnicodeDecodeError)


def test_unreadable_config_raises_config_error_with_cause(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    path = _write(tmp_path, _spm_yaml())

    def deny(self: Path, *args: object, **kwargs: object) -> None:
        raise PermissionError(13, "Permission denied", str(self))

    monkeypatch.setattr(Path, "open", deny)

    with pytest.raises(ConfigError, match="spm.yaml") as excinfo:
        load_spm_config(path)

    assert isinstance(excinfo.value.__cause__, OSError)


def test_missing_file_raises_config_error(tmp_path: Path):
    with pytest.raises(ConfigError, match="not found"):
        load_spm_config(tmp_path / "nonexistent.yaml")
