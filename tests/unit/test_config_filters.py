# tests/unit/test_config_filters.py
from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml
from pydantic import ValidationError

from rinchen.config.filters import (
    FILTERS_CONFIG_PATH,
    Bounds,
    DirectionBounds,
    FilterConfig,
    LabseConfig,
    LengthConfig,
    LidConfig,
    MinHashConfig,
    QeConfig,
    ScriptRatioConfig,
    load_filter_config,
)
from rinchen.config.loader import REPO_ROOT
from rinchen.errors import ConfigError
from rinchen.lang.scripts import Script

ALL_STAGES = ["script_ratio", "numbers", "length", "lid", "minhash", "labse", "qe"]


def _write(tmp_path: Path, document: Any) -> Path:
    """Write `document` as YAML (or verbatim when it is already text)."""
    path = tmp_path / "filters.yaml"
    text = document if isinstance(document, str) else yaml.safe_dump(document)
    path.write_text(text, encoding="utf-8")
    return path


def test_defaults_are_the_documented_priors():
    config = FilterConfig()

    assert config.stages == ALL_STAGES
    assert config.script_ratio == ScriptRatioConfig(min_ratio=0.9)
    assert config.length == LengthConfig(
        char_ratio=Bounds(min=0.5, max=2.0),
        token_ratio=Bounds(min=0.4, max=2.5),
        char_weights={
            Script.HAN: 2.5,
            Script.HANGUL: 2.0,
            Script.HIRAGANA: 1.5,
            Script.KATAKANA: 1.5,
        },
        by_direction={},
        spm_model=None,
    )
    assert config.lid == LidConfig(
        model_path=REPO_ROOT / "models" / "lid" / "glotlid_v3.bin",
        min_prob=0.5,
        top_k=3,
        accept={},
    )
    assert config.minhash == MinHashConfig(num_perm=128, ngram=5, threshold=0.8, seed=1)
    assert config.labse == LabseConfig(
        model="sentence-transformers/LaBSE",
        min_cosine=0.7,
        batch_size=128,
        device="auto",
    )
    assert config.qe == QeConfig(
        scorer="metricx_qe", max_metricx_qe=8.0, min_cometkiwi=0.55, batch_size=32
    )


def test_shipped_filters_yaml_equals_the_defaults():
    loaded = load_filter_config()

    assert FILTERS_CONFIG_PATH == REPO_ROOT / "configs" / "data" / "filters.yaml"
    assert loaded.model_dump(mode="json") == FilterConfig().model_dump(mode="json")
    assert loaded == FilterConfig()


def test_empty_sections_use_the_defaults(tmp_path: Path):
    config = load_filter_config(_write(tmp_path, {"lid": {"top_k": 5}}))

    assert config.lid.top_k == 5
    assert config.lid.min_prob == 0.5
    assert config.stages == ALL_STAGES
    assert config.qe == QeConfig()


def test_stages_may_be_a_reordered_subset(tmp_path: Path):
    config = load_filter_config(
        _write(tmp_path, {"stages": ["numbers", "script_ratio"]})
    )

    assert config.stages == ["numbers", "script_ratio"]


def test_char_weights_accept_script_names_as_keys(tmp_path: Path):
    document = {"length": {"char_weights": {"Han": 3, "Cyrillic": 1.25}}}

    config = load_filter_config(_write(tmp_path, document))

    assert config.length.char_weights == {Script.HAN: 3.0, Script.CYRILLIC: 1.25}
    assert all(isinstance(key, Script) for key in config.length.char_weights)


def test_by_direction_loads_bounds_per_direction(tmp_path: Path):
    document = {
        "length": {
            "by_direction": {
                "khk_Cyrl->eng_Latn": {"char_ratio": {"min": 0.6, "max": 1.8}},
                "eng_Latn->khk_Cyrl": {"token_ratio": {"min": 0.5, "max": 2}},
            }
        }
    }

    config = load_filter_config(_write(tmp_path, document))

    assert config.length.by_direction == {
        "khk_Cyrl->eng_Latn": DirectionBounds(char_ratio=Bounds(min=0.6, max=1.8)),
        "eng_Latn->khk_Cyrl": DirectionBounds(token_ratio=Bounds(min=0.5, max=2.0)),
    }


def test_relative_paths_resolve_against_repo_root_and_absolute_stay(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    absolute_model = tmp_path / "abs" / "lid.bin"
    document = {
        "lid": {"model_path": "models/lid/other.bin"},
        "length": {"spm_model": "models/spm/tiny_1k.model"},
    }

    relative = load_filter_config(_write(tmp_path, document))
    absolute = load_filter_config(
        _write(tmp_path, {"lid": {"model_path": str(absolute_model)}})
    )

    assert relative.lid.model_path == REPO_ROOT / "models" / "lid" / "other.bin"
    assert relative.length.spm_model == REPO_ROOT / "models" / "spm" / "tiny_1k.model"
    assert absolute.lid.model_path == absolute_model


def test_loaded_config_is_immutable():
    config = FilterConfig()

    with pytest.raises(ValidationError):
        config.stages = ["lid"]  # type: ignore[misc]
    with pytest.raises(ValidationError):
        config.script_ratio.min_ratio = 0.5  # type: ignore[misc]


@pytest.mark.parametrize(
    ("document", "message_fragment"),
    [
        pytest.param(
            {"script_ratio": {"min_ratio": 0}},
            "min_ratio.*greater than 0",
            id="zero-min-ratio",
        ),
        pytest.param(
            {"script_ratio": {"min_ratio": 1.5}},
            "min_ratio.*less than or equal to 1",
            id="min-ratio-above-one",
        ),
        pytest.param(
            {"script_ratio": {"min_ratio": "0.9"}},
            "min_ratio.*valid number",
            id="quoted-min-ratio",
        ),
        pytest.param(
            {"script_ratio": {"min_ratio": float("nan")}},
            "min_ratio.*finite",
            id="nan-min-ratio",
        ),
        pytest.param(
            {"length": {"char_ratio": {"min": 2.0, "max": 0.5}}},
            "min must be less than max",
            id="inverted-bounds",
        ),
        pytest.param(
            {"length": {"token_ratio": {"min": 1.0, "max": 1.0}}},
            "min must be less than max",
            id="equal-bounds",
        ),
        pytest.param(
            {"length": {"char_ratio": {"min": 0, "max": 2.0}}},
            "min.*greater than 0",
            id="zero-lower-bound",
        ),
        pytest.param(
            {"length": {"char_ratio": {"min": "0.5", "max": 2.0}}},
            "min.*valid number",
            id="quoted-bound",
        ),
        pytest.param(
            {"length": {"char_ratio": {"min": 0.5}}},
            "max.*Field required",
            id="missing-bound",
        ),
        pytest.param(
            {"length": {"char_weights": {"Han": 0}}},
            "char_weights.*greater than 0",
            id="zero-weight",
        ),
        pytest.param(
            {"length": {"char_weights": {"Greek": 2.0}}},
            "char_weights.*Input should be",
            id="unknown-script",
        ),
        pytest.param(
            {"stages": ["lid", "lid"]},
            "duplicate stage",
            id="duplicate-stages",
        ),
        pytest.param(
            {"stages": ["script_ratio", "bogus"]},
            "stages.*Input should be",
            id="unknown-stage",
        ),
        pytest.param(
            {"stages": []},
            "stages.*at least 1 item",
            id="empty-stages",
        ),
        pytest.param(
            {"bogus": 1},
            "bogus.*Extra inputs are not permitted",
            id="unknown-top-level-key",
        ),
        pytest.param(
            {"lid": {"bogus": 1}},
            "lid.bogus.*Extra inputs are not permitted",
            id="unknown-nested-key",
        ),
        pytest.param(
            {
                "length": {
                    "by_direction": {"khk->eng": {"char_ratio": {"min": 1, "max": 2}}}
                }
            },
            "khk->eng",
            id="malformed-direction-key",
        ),
        pytest.param(
            {"length": {"by_direction": {"khk_Cyrl->eng_Latn": {}}}},
            "at least one of char_ratio, token_ratio",
            id="direction-without-bounds",
        ),
        pytest.param(
            {
                "length": {
                    "by_direction": {
                        "khk_Cyrl->eng_Latn": {"char_ratio": None, "token_ratio": None}
                    }
                }
            },
            "at least one of char_ratio, token_ratio",
            id="direction-with-null-bounds",
        ),
        pytest.param({"lid": {"top_k": 0}}, "top_k.*greater than 0", id="zero-top-k"),
        pytest.param(
            {"lid": {"top_k": True}}, "top_k.*valid integer", id="boolean-top-k"
        ),
        pytest.param(
            {"lid": {"min_prob": "0.5"}}, "min_prob.*valid number", id="quoted-min-prob"
        ),
        pytest.param(
            {"lid": {"model_path": "  "}},
            "model_path must not be empty",
            id="blank-model-path",
        ),
        pytest.param(
            {"lid": {"accept": {"khk_Cyrl": "khk_Cyrl"}}},
            "accept.*valid list",
            id="accept-not-a-list",
        ),
        pytest.param(
            {"minhash": {"threshold": "0.8"}},
            "threshold.*valid number",
            id="quoted-threshold",
        ),
        pytest.param(
            {"minhash": {"num_perm": 0}},
            "num_perm.*greater than 0",
            id="zero-permutations",
        ),
        pytest.param(
            {"minhash": {"threshold": 1.5}},
            "threshold.*less than or equal to 1",
            id="threshold-above-one",
        ),
        pytest.param(
            {"labse": {"device": "tpu"}}, "device.*Input should be", id="unknown-device"
        ),
        pytest.param(
            {"labse": {"min_cosine": 1.5}},
            "min_cosine.*less than or equal to 1",
            id="cosine-above-one",
        ),
        pytest.param(
            {"qe": {"scorer": "bleurt"}}, "scorer.*Input should be", id="unknown-scorer"
        ),
        pytest.param(
            {"qe": {"batch_size": 0}},
            "batch_size.*greater than 0",
            id="zero-qe-batch",
        ),
        pytest.param(["lid"], "top level must be a mapping", id="top-level-list"),
        pytest.param("", "top level must be a mapping", id="empty-file"),
    ],
)
def test_malformed_config_raises_config_error(
    tmp_path: Path, document: Any, message_fragment: str
):
    path = _write(tmp_path, document)

    # (?s): pydantic puts the field name and its message on separate lines.
    with pytest.raises(ConfigError, match=f"(?s){message_fragment}") as excinfo:
        load_filter_config(path)

    assert path.name in str(excinfo.value)


def test_field_violation_chains_the_validation_error(tmp_path: Path):
    path = _write(tmp_path, {"script_ratio": {"min_ratio": 0}})

    with pytest.raises(ConfigError) as excinfo:
        load_filter_config(path)

    assert isinstance(excinfo.value.__cause__, ValidationError)


def test_yaml_syntax_error_raises_config_error_with_cause(tmp_path: Path):
    path = _write(tmp_path, "stages: [lid\n")

    with pytest.raises(ConfigError, match="filters.yaml") as excinfo:
        load_filter_config(path)

    assert isinstance(excinfo.value.__cause__, yaml.YAMLError)


def test_non_utf8_config_raises_config_error_with_cause(tmp_path: Path):
    path = tmp_path / "filters.yaml"
    path.write_bytes(b"\xff\xfe\x00")

    with pytest.raises(ConfigError, match="filters.yaml") as excinfo:
        load_filter_config(path)

    assert isinstance(excinfo.value.__cause__, UnicodeDecodeError)


def test_unreadable_config_raises_config_error_with_cause(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    path = _write(tmp_path, {})

    def deny(self: Path, *args: object, **kwargs: object) -> None:
        raise PermissionError(13, "Permission denied", str(self))

    monkeypatch.setattr(Path, "open", deny)

    with pytest.raises(ConfigError, match="filters.yaml") as excinfo:
        load_filter_config(path)

    assert isinstance(excinfo.value.__cause__, OSError)


def test_missing_file_raises_config_error(tmp_path: Path):
    with pytest.raises(ConfigError, match="not found") as excinfo:
        load_filter_config(tmp_path / "nonexistent.yaml")

    assert "nonexistent.yaml" in str(excinfo.value)
