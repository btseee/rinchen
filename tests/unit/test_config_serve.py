"""The configuration of the service (D11, T18): the defaults, every bound, the four layers
(defaults, the YAML file, the environment, the command's flags) and what the file refuses."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml
from pydantic import ValidationError

from rinchen.config.serve import (
    ScriptConverterConfig,
    ServeConfig,
    ServeEnvSettings,
    load_serve_config,
)
from rinchen.errors import ConfigError
from rinchen.infer.engine_types import MAX_SOURCE_PIECES

MIB = 1024 * 1024

DEFAULTS: dict[str, Any] = {
    "host": "127.0.0.1",
    "port": 8000,
    "device": "cpu",
    "compute_type": "default",
    "inter_threads": 2,
    "intra_threads": 1,
    "beam_size": 4,
    "max_decoding_length": 256,
    "max_source_pieces": 256,
    "script_mask": "retry",
    "max_batch_sentences": 32,
    "max_wait_ms": 5.0,
    "max_queue_sentences": 1024,
    "max_texts": 64,
    "max_text_chars": 5000,
    "max_body_bytes": MIB,
    "request_timeout_seconds": 30.0,
    "shutdown_timeout_seconds": 20.0,
    "log_level": "info",
    "script_converter": None,
}


def write_yaml(tmp_path: Path, document: Any) -> Path:
    path = tmp_path / "serve.yaml"
    path.write_text(
        document if isinstance(document, str) else yaml.safe_dump(document),
        encoding="utf-8",
    )
    return path


def fields_refused(**values: Any) -> list[str]:
    """The names of the fields a ServeConfig built from `values` complains about."""
    with pytest.raises(ValidationError) as caught:
        ServeConfig(**values)
    return sorted({str(error["loc"][0]) for error in caught.value.errors()})


def test_the_defaults_are_these_literals() -> None:
    assert ServeConfig().model_dump() == DEFAULTS


def test_the_defaults_bind_the_loopback_on_the_cpu() -> None:
    config = ServeConfig()

    assert (config.host, config.device) == ("127.0.0.1", "cpu")


@pytest.mark.parametrize(
    ("field", "refused", "accepted"),
    [
        ("port", [0, 65536, -1], [1, 65535]),
        ("inter_threads", [0, -1], [1, 64]),
        ("intra_threads", [0], [1, 8]),
        ("beam_size", [0, 9], [1, 8]),
        ("max_decoding_length", [0], [1, 1024]),
        ("max_source_pieces", [0, 1024], [1, 1023]),
        ("max_batch_sentences", [0], [1, 256]),
        ("max_queue_sentences", [0, 63], [64, 100000]),
        ("max_texts", [0, 65], [1, 64]),
        ("max_text_chars", [0, 100001], [1, 100000]),
        ("max_body_bytes", [1023, 64 * MIB + 1], [1024, 64 * MIB]),
        ("max_wait_ms", [0, -1.0, 0.0, float("inf"), float("nan")], [0.001, 50.0]),
        ("request_timeout_seconds", [0, -5.0, float("inf")], [0.5, 300.0]),
        ("shutdown_timeout_seconds", [0, -1.0, float("nan")], [0.5, 60.0]),
        ("device", ["auto", "tpu", ""], ["cpu", "cuda"]),
        ("script_mask", ["sometimes", "RETRY", ""], ["retry", "always", "off"]),
        ("log_level", ["loud", "INFO"], ["critical", "error", "warning", "debug"]),
    ],
)
def test_every_bound_is_enforced_on_both_sides(
    field: str, refused: list[Any], accepted: list[Any]
) -> None:
    for value in refused:
        assert fields_refused(**{field: value}) == [field], (field, value)
    for value in accepted:
        assert getattr(ServeConfig(**{field: value}), field) == value


def test_the_source_limit_is_the_engines() -> None:
    assert MAX_SOURCE_PIECES == 1023
    assert ServeConfig(max_source_pieces=MAX_SOURCE_PIECES).max_source_pieces == 1023


@pytest.mark.parametrize("value", ["auto", "AUTO", " auto ", "", "   "])
def test_a_compute_type_of_auto_or_nothing_is_refused(value: str) -> None:
    with pytest.raises(ValidationError, match="compute_type"):
        ServeConfig(compute_type=value)


@pytest.mark.parametrize("value", ["default", "float32", "int8", "int8_float16"])
def test_a_named_compute_type_is_accepted(value: str) -> None:
    assert ServeConfig(compute_type=value).compute_type == value


def test_the_refusal_of_auto_says_why() -> None:
    with pytest.raises(ValidationError, match="quantises"):
        ServeConfig(compute_type="auto")


def test_a_queue_that_cannot_hold_one_whole_request_is_refused() -> None:
    assert fields_refused(max_texts=64, max_queue_sentences=63) == [
        "max_queue_sentences"
    ]
    assert ServeConfig(max_texts=10, max_queue_sentences=10).max_queue_sentences == 10


@pytest.mark.parametrize("value", ["8080", 80.0, True, None])
def test_a_port_that_is_not_an_integer_is_refused_not_coerced(value: object) -> None:
    assert fields_refused(port=value) == ["port"]


def test_a_boolean_is_not_a_thread_count() -> None:
    assert fields_refused(inter_threads=True) == ["inter_threads"]


@pytest.mark.parametrize("value", ["", "   "])
def test_a_blank_host_is_refused(value: str) -> None:
    assert fields_refused(host=value) == ["host"]


def test_an_unknown_field_is_refused_by_name() -> None:
    assert fields_refused(model="models/tiny") == ["model"]


def test_a_config_cannot_be_changed_after_it_is_built() -> None:
    config = ServeConfig()

    with pytest.raises(ValidationError):
        config.port = 9000  # type: ignore[misc]


def test_the_model_is_not_a_config_key() -> None:
    assert "model" not in ServeConfig.model_fields
    assert "model" not in ServeEnvSettings.model_fields


def test_the_script_converter_names_two_paths() -> None:
    config = ServeConfig(
        script_converter={"ct2_dir": "conv/ct2", "spm_model": "conv/spm.model"}  # type: ignore[arg-type]
    )

    assert config.script_converter == ScriptConverterConfig(
        ct2_dir=Path("conv/ct2"), spm_model=Path("conv/spm.model")
    )


@pytest.mark.parametrize(
    "converter",
    [
        {"ct2_dir": "conv/ct2"},
        {"spm_model": "conv/spm.model"},
        {"ct2_dir": "", "spm_model": "conv/spm.model"},
        {"ct2_dir": "conv/ct2", "spm_model": "   "},
        {"ct2_dir": "a", "spm_model": "b", "extra": "c"},
    ],
)
def test_a_script_converter_needs_both_paths_and_nothing_else(
    converter: dict[str, str],
) -> None:
    assert fields_refused(script_converter=converter) == ["script_converter"]


def test_the_loader_gives_the_defaults_without_any_layer() -> None:
    assert load_serve_config(None, env={}).model_dump() == DEFAULTS


def test_each_layer_wins_over_the_ones_before_it(tmp_path: Path) -> None:
    path = write_yaml(tmp_path, {"port": 8001})
    env = {"RINCHEN_SERVE_PORT": "8002"}
    overrides = {"port": 8003}

    assert load_serve_config(None, env={}).port == 8000
    assert load_serve_config(path, env={}).port == 8001
    assert load_serve_config(path, env=env).port == 8002
    assert load_serve_config(path, env=env, overrides=overrides).port == 8003
    assert load_serve_config(None, env=env, overrides=overrides).port == 8003


def test_a_layer_that_does_not_name_a_key_leaves_the_lower_one(tmp_path: Path) -> None:
    path = write_yaml(tmp_path, {"port": 8001, "beam_size": 2})

    config = load_serve_config(
        path, env={"RINCHEN_SERVE_PORT": "8002"}, overrides={"max_texts": 8}
    )

    assert (config.port, config.beam_size, config.max_texts) == (8002, 2, 8)


def test_a_flag_that_was_not_given_does_not_override(tmp_path: Path) -> None:
    path = write_yaml(tmp_path, {"port": 8001})

    config = load_serve_config(path, env={}, overrides={"port": None, "beam_size": 3})

    assert (config.port, config.beam_size) == (8001, 3)


def test_a_null_in_the_file_is_a_value_not_a_missing_one(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="port"):
        load_serve_config(write_yaml(tmp_path, "port: null\n"), env={})

    path = write_yaml(tmp_path, "script_converter: null\nbeam_size: 2\n")
    config = load_serve_config(path, env={})
    assert (config.script_converter, config.beam_size) == (None, 2)


def test_the_environment_is_read_by_prefix_and_by_type() -> None:
    env = {
        "RINCHEN_SERVE_PORT": "9090",
        "RINCHEN_SERVE_MAX_WAIT_MS": "2.5",
        "RINCHEN_SERVE_DEVICE": "cuda",
        "RINCHEN_SERVE_HOST": "0.0.0.0",
        "RINCHEN_SERVE_SCRIPT_MASK": "off",
    }

    config = load_serve_config(None, env=env)

    assert (config.port, config.max_wait_ms, config.device, config.host) == (
        9090,
        2.5,
        "cuda",
        "0.0.0.0",
    )
    assert config.script_mask == "off"


def test_the_environment_names_are_not_case_sensitive() -> None:
    assert load_serve_config(None, env={"rinchen_serve_port": "9091"}).port == 9091


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("RINCHEN_SERVE_PORT", "true"),
        ("RINCHEN_SERVE_PORT", "80.5"),
        ("RINCHEN_SERVE_PORT", "eighty"),
        ("RINCHEN_SERVE_INTER_THREADS", "false"),
        ("RINCHEN_SERVE_MAX_WAIT_MS", "soon"),
        ("RINCHEN_SERVE_PORT", "0"),
        ("RINCHEN_SERVE_DEVICE", "auto"),
        ("RINCHEN_SERVE_COMPUTE_TYPE", "auto"),
    ],
)
def test_an_environment_value_that_is_not_valid_is_a_config_error(
    name: str, value: str
) -> None:
    with pytest.raises(ConfigError):
        load_serve_config(None, env={name: value})


def test_an_empty_environment_value_is_not_set() -> None:
    config = load_serve_config(None, env={"RINCHEN_SERVE_PORT": ""})

    assert config.port == 8000


def test_an_empty_variable_of_the_process_environment_is_not_set(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("RINCHEN_SERVE_PORT", "")
    monkeypatch.setenv("RINCHEN_SERVE_BEAM_SIZE", "5")

    config = load_serve_config(None)

    assert (config.port, config.beam_size) == (8000, 5)


def test_what_is_not_under_the_prefix_in_the_environment_is_ignored() -> None:
    env = {
        "RINCHEN_MODEL": "models/tiny",
        "RINCHEN_MODEL_DIR": "models/tiny",
        "RINCHEN_SERVER_PORT": "1",
        "PORT": "1",
        "RINCHEN_SEED": "7",
    }

    assert load_serve_config(None, env=env).model_dump() == DEFAULTS


def test_the_process_environment_is_read_when_none_is_given(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("RINCHEN_SERVE_PORT", "9093")
    monkeypatch.setenv("RINCHEN_SERVE_BEAM_SIZE", "6")

    config = load_serve_config(None)

    assert (config.port, config.beam_size) == (9093, 6)


def test_a_given_environment_hides_the_process_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("RINCHEN_SERVE_PORT", "9093")

    assert load_serve_config(None, env={}).port == 8000


def test_the_settings_class_reads_the_process_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("RINCHEN_SERVE_PORT", "9094")
    monkeypatch.setenv("RINCHEN_SERVE_SCRIPT_CONVERTER_CT2_DIR", "conv/ct2")

    settings = ServeEnvSettings()

    assert settings.port == 9094
    assert settings.script_converter_ct2_dir == Path("conv/ct2")
    assert settings.beam_size is None


def test_every_field_of_the_settings_class_is_optional() -> None:
    assert all(
        field.is_required() is False for field in ServeEnvSettings.model_fields.values()
    )


def test_the_script_converter_comes_from_two_environment_variables() -> None:
    env = {
        "RINCHEN_SERVE_SCRIPT_CONVERTER_CT2_DIR": "conv/ct2",
        "RINCHEN_SERVE_SCRIPT_CONVERTER_SPM_MODEL": "conv/spm.model",
    }

    config = load_serve_config(None, env=env)

    assert config.script_converter == ScriptConverterConfig(
        ct2_dir=Path("conv/ct2"), spm_model=Path("conv/spm.model")
    )


def test_one_environment_variable_completes_the_converter_of_the_file(
    tmp_path: Path,
) -> None:
    path = write_yaml(
        tmp_path,
        {"script_converter": {"ct2_dir": "file/ct2", "spm_model": "file/spm.model"}},
    )

    config = load_serve_config(
        path, env={"RINCHEN_SERVE_SCRIPT_CONVERTER_CT2_DIR": "env/ct2"}
    )

    assert config.script_converter == ScriptConverterConfig(
        ct2_dir=Path("env/ct2"), spm_model=Path("file/spm.model")
    )


def test_half_a_converter_in_the_environment_is_refused() -> None:
    with pytest.raises(ConfigError, match="script_converter"):
        load_serve_config(
            None, env={"RINCHEN_SERVE_SCRIPT_CONVERTER_CT2_DIR": "conv/ct2"}
        )


def test_the_flags_can_name_the_converter() -> None:
    config = load_serve_config(
        None,
        env={},
        overrides={"script_converter": {"ct2_dir": "a", "spm_model": "b"}},
    )

    assert config.script_converter == ScriptConverterConfig(
        ct2_dir=Path("a"), spm_model=Path("b")
    )


def test_an_unknown_key_of_the_file_is_named(tmp_path: Path) -> None:
    path = write_yaml(tmp_path, {"port": 8001, "prot": 8002})

    with pytest.raises(ConfigError, match="prot") as caught:
        load_serve_config(path, env={})

    assert str(path) in str(caught.value)


def test_the_model_cannot_be_named_in_the_file(tmp_path: Path) -> None:
    path = write_yaml(tmp_path, {"model": "models/tiny"})

    with pytest.raises(ConfigError, match="model"):
        load_serve_config(path, env={})


def test_an_unknown_override_is_named() -> None:
    with pytest.raises(ConfigError, match="terminology"):
        load_serve_config(None, env={}, overrides={"terminology": True})


@pytest.mark.parametrize(
    "document", [{"device": "auto"}, {"compute_type": "auto"}], ids=["device", "type"]
)
def test_auto_is_refused_in_the_file(tmp_path: Path, document: dict[str, str]) -> None:
    with pytest.raises(ConfigError, match="auto"):
        load_serve_config(write_yaml(tmp_path, document), env={})


def test_a_value_out_of_range_in_the_file_is_a_config_error(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="max_texts"):
        load_serve_config(write_yaml(tmp_path, {"max_texts": 65}), env={})


def test_a_quoted_number_in_the_file_is_not_coerced(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="port"):
        load_serve_config(write_yaml(tmp_path, 'port: "8001"\n'), env={})


def test_a_missing_file_is_a_config_error_naming_it(tmp_path: Path) -> None:
    missing = tmp_path / "absent.yaml"

    with pytest.raises(ConfigError, match=r"not found.*absent\.yaml"):
        load_serve_config(missing, env={})


@pytest.mark.parametrize("text", ["- 1\n- 2\n", "just text\n", "", "port: [1\n"])
def test_a_file_that_is_not_a_mapping_is_a_config_error(
    tmp_path: Path, text: str
) -> None:
    with pytest.raises(ConfigError, match="serve.yaml"):
        load_serve_config(write_yaml(tmp_path, text), env={})


def test_a_file_that_is_not_utf8_is_a_config_error(tmp_path: Path) -> None:
    path = tmp_path / "serve.yaml"
    path.write_bytes(b"port: 8001\n# \xff\xfe\n")

    with pytest.raises(ConfigError, match="serve.yaml"):
        load_serve_config(path, env={})


def test_a_directory_in_place_of_the_file_is_a_config_error(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match=r"not found.*" + str(tmp_path)):
        load_serve_config(tmp_path, env={})


# ---- the budget of a stop -----------------------------------------------------------------


@pytest.mark.parametrize(
    ("timeout", "graceful", "drain"),
    [
        (30.0, 15, 15.0),
        (20.0, 10, 10.0),
        (5.0, 3, 2.0),
        (2.5, 2, 0.5),
        (2.0, 1, 1.0),
        (1.5, 1, 0.5),
        (1.0, 1, 0.0),
        (0.5, 0, 0.5),
        (0.2, 0, 0.2),
    ],
)
def test_the_shutdown_timeout_is_one_budget_shared_by_the_server_and_the_drain(
    timeout: float, graceful: int, drain: float
) -> None:
    config = ServeConfig(shutdown_timeout_seconds=timeout)

    assert config.graceful_wait_seconds == graceful
    assert type(config.graceful_wait_seconds) is int
    assert config.drain_wait_seconds == pytest.approx(drain)


@pytest.mark.parametrize(
    "timeout", [0.001, 0.2, 0.999, 1.0, 1.7, 2.0, 3.3, 7.0, 20.0, 59.9]
)
def test_the_two_waits_of_a_stop_never_add_up_to_more_than_the_timeout(
    timeout: float,
) -> None:
    config = ServeConfig(shutdown_timeout_seconds=timeout)

    assert config.drain_wait_seconds >= 0
    assert config.graceful_wait_seconds + config.drain_wait_seconds == pytest.approx(
        timeout
    )


# ---- a name of the environment that is no setting -----------------------------------------

SECRET_VALUE = "SECRET-VALUE-12345"


@pytest.mark.parametrize(
    ("name", "message"),
    [
        (
            "RINCHEN_SERVE_PROT",
            (
                "unknown environment variable RINCHEN_SERVE_PROT (did you mean "
                "RINCHEN_SERVE_PORT?): a setting is RINCHEN_SERVE_<FIELD> for a field of "
                "ServeConfig; the model is --model or RINCHEN_MODEL, not a setting"
            ),
        ),
        (
            "RINCHEN_SERVE__PORT",
            (
                "unknown environment variable RINCHEN_SERVE__PORT (did you mean "
                "RINCHEN_SERVE_PORT?): a setting is RINCHEN_SERVE_<FIELD> for a field of "
                "ServeConfig; the model is --model or RINCHEN_MODEL, not a setting"
            ),
        ),
        (
            "RINCHEN_SERVE_MAX_QUEUE_SENTENCE",
            (
                "unknown environment variable RINCHEN_SERVE_MAX_QUEUE_SENTENCE (did you mean "
                "RINCHEN_SERVE_MAX_QUEUE_SENTENCES?): a setting is RINCHEN_SERVE_<FIELD> for a "
                "field of ServeConfig; the model is --model or RINCHEN_MODEL, not a setting"
            ),
        ),
        (
            "RINCHEN_SERVE_MODEL",
            (
                "unknown environment variable RINCHEN_SERVE_MODEL: a setting is "
                "RINCHEN_SERVE_<FIELD> for a field of ServeConfig; the model is --model or "
                "RINCHEN_MODEL, not a setting"
            ),
        ),
        (
            "rinchen_serve_prot",
            (
                "unknown environment variable rinchen_serve_prot (did you mean "
                "RINCHEN_SERVE_PORT?): a setting is RINCHEN_SERVE_<FIELD> for a field of "
                "ServeConfig; the model is --model or RINCHEN_MODEL, not a setting"
            ),
        ),
        (
            "RINCHEN_SERVE_",
            (
                "unknown environment variable RINCHEN_SERVE_: a setting is "
                "RINCHEN_SERVE_<FIELD> for a field of ServeConfig; the model is --model or "
                "RINCHEN_MODEL, not a setting"
            ),
        ),
    ],
)
def test_a_mistyped_setting_of_the_environment_is_refused_by_name_and_never_by_value(
    name: str, message: str
) -> None:
    with pytest.raises(ConfigError) as caught:
        load_serve_config(None, env={name: SECRET_VALUE})

    assert str(caught.value) == message


def test_a_mistyped_setting_is_refused_even_when_its_value_is_empty() -> None:
    with pytest.raises(ConfigError, match="RINCHEN_SERVE_PROT"):
        load_serve_config(None, env={"RINCHEN_SERVE_PROT": ""})


def test_every_unknown_name_is_reported_and_the_known_ones_are_not() -> None:
    env = {
        "RINCHEN_SERVE_PORT": "9090",
        "RINCHEN_SERVE_ZZZ": SECRET_VALUE,
        "RINCHEN_SERVE_AAA": SECRET_VALUE,
    }

    with pytest.raises(ConfigError) as caught:
        load_serve_config(None, env=env)

    message = str(caught.value)
    assert message.startswith(
        "unknown environment variables RINCHEN_SERVE_AAA, RINCHEN_SERVE_ZZZ: "
    )
    assert "RINCHEN_SERVE_PORT" not in message
    assert SECRET_VALUE not in message


def test_a_mistyped_setting_of_the_process_environment_is_refused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("RINCHEN_SERVE_PROT", SECRET_VALUE)

    with pytest.raises(ConfigError) as caught:
        load_serve_config(None)

    assert "RINCHEN_SERVE_PROT" in str(caught.value)
    assert SECRET_VALUE not in str(caught.value)


def test_a_given_environment_hides_a_mistyped_name_of_the_process_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("RINCHEN_SERVE_PROT", "1")

    assert load_serve_config(None, env={}).port == 8000


def test_every_field_of_the_settings_class_is_a_known_name() -> None:
    env = {
        f"RINCHEN_SERVE_{name.upper()}": "" for name in ServeEnvSettings.model_fields
    }

    assert load_serve_config(None, env=env).model_dump() == DEFAULTS


def test_the_two_flat_variables_of_the_converter_are_known_names() -> None:
    env = {
        "RINCHEN_SERVE_SCRIPT_CONVERTER_CT2_DIR": "conv/ct2",
        "RINCHEN_SERVE_SCRIPT_CONVERTER_SPM_MODEL": "conv/spm.model",
    }

    assert load_serve_config(None, env=env).script_converter is not None


# ---- which layer supplied a value that is refused -----------------------------------------


def test_an_environment_value_out_of_range_names_its_variable_and_not_the_file(
    tmp_path: Path,
) -> None:
    path = write_yaml(tmp_path, {"port": 8001})

    with pytest.raises(ConfigError) as caught:
        load_serve_config(path, env={"RINCHEN_SERVE_PORT": "70000"})

    assert str(caught.value) == (
        "invalid serve config: port (from RINCHEN_SERVE_PORT): "
        "Input should be less than or equal to 65535"
    )


def test_a_value_of_the_file_names_the_file(tmp_path: Path) -> None:
    path = write_yaml(tmp_path, {"max_texts": 65})

    with pytest.raises(ConfigError) as caught:
        load_serve_config(path, env={"RINCHEN_SERVE_PORT": "9090"})

    assert str(caught.value) == (
        f"invalid serve config: max_texts (from {path}): "
        "Input should be less than or equal to 64"
    )


def test_a_flag_out_of_range_names_the_flag(tmp_path: Path) -> None:
    path = write_yaml(tmp_path, {"port": 8001})

    with pytest.raises(ConfigError) as caught:
        load_serve_config(path, env={}, overrides={"max_text_chars": 0})

    assert str(caught.value) == (
        "invalid serve config: max_text_chars (from --max-text-chars): "
        "Input should be greater than or equal to 1"
    )


def test_each_refused_value_names_the_layer_that_supplied_it(tmp_path: Path) -> None:
    path = write_yaml(tmp_path, {"max_texts": 65})

    with pytest.raises(ConfigError) as caught:
        load_serve_config(
            path,
            env={"RINCHEN_SERVE_PORT": "70000"},
            overrides={"max_text_chars": 0},
        )

    message = str(caught.value)
    assert f"max_texts (from {path})" in message
    assert "port (from RINCHEN_SERVE_PORT)" in message
    assert "max_text_chars (from --max-text-chars)" in message


def test_the_last_layer_that_holds_a_key_is_the_one_blamed(tmp_path: Path) -> None:
    path = write_yaml(tmp_path, {"port": 70001})

    with pytest.raises(ConfigError) as caught:
        load_serve_config(path, env={"RINCHEN_SERVE_PORT": "70000"})

    assert "port (from RINCHEN_SERVE_PORT)" in str(caught.value)
    assert str(path) not in str(caught.value)


def test_a_converter_path_of_the_flags_is_blamed_by_its_own_flag() -> None:
    with pytest.raises(ConfigError) as caught:
        load_serve_config(
            None,
            env={},
            overrides={"script_converter": {"ct2_dir": "   ", "spm_model": "b"}},
        )

    assert str(caught.value) == (
        "invalid serve config: script_converter.ct2_dir (from --script-converter-ct2): "
        "Value error, ct2_dir must not be empty or blank"
    )


def test_a_field_no_layer_holds_names_no_layer() -> None:
    with pytest.raises(ConfigError) as caught:
        load_serve_config(
            None, env={"RINCHEN_SERVE_SCRIPT_CONVERTER_CT2_DIR": "conv/ct2"}
        )

    assert str(caught.value) == (
        "invalid serve config: script_converter.spm_model: Field required"
    )


def test_an_environment_value_of_the_wrong_type_names_its_variable() -> None:
    with pytest.raises(ConfigError) as caught:
        load_serve_config(None, env={"RINCHEN_SERVE_PORT": "eighty"})

    assert str(caught.value) == (
        "invalid RINCHEN_SERVE_* environment: RINCHEN_SERVE_PORT: Input should be a "
        "valid integer, unable to parse string as an integer"
    )
