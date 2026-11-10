"""Configuration of the service (`configs/serve/*.yaml`, D11 T18): where it listens, how the
engine is loaded, the decode defaults, the micro-batcher's limits and the request limits.

Four layers, the last one wins: the defaults below, the YAML file, the environment
(`RINCHEN_SERVE_<FIELD>`) and the command's flags. The model is not a key: it is `--model`
or `RINCHEN_MODEL`, and a file that names it is refused as any unknown key is. Every number
is an unmeasured prior: nothing here was timed on a real model, and no GPU was used.

`device` is `cpu` or `cuda` and `compute_type` anything but `auto` (it quantises a float32
export to int8 on a CPU: measured in the D11 spike). Relative paths stay relative: the
service resolves them from its working directory, as the command line does.

`shutdown_timeout_seconds` is the budget of the WHOLE stop, not of each step of it: the server
waits for the requests in flight for `graceful_wait_seconds` of it and the end of the lifespan
for `drain_wait_seconds`, the rest. A container's stop grace period must be longer than the budget
(`docker/compose.serve.yaml`).
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from pathlib import Path
from typing import Annotated, Any, Final, Literal

import yaml
from pydantic import (
    AfterValidator,
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    ValidationError,
    ValidationInfo,
    field_validator,
)
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
)

from rinchen.config.loader import reject_blank_path
from rinchen.errors import ConfigError
from rinchen.export.quantize import require_compute_type

ENV_PREFIX: Final = "RINCHEN_SERVE_"
MIB: Final = 1024 * 1024

# The request limits an operator may set, and what a request may hold when none is set
# (`rinchen.serve.schemas` narrows its schema to the configured ones).
MAX_TEXTS_LIMIT: Final = 64
MAX_BEAM_SIZE: Final = 8
MAX_TEXT_CHARS_LIMIT: Final = 100_000
DEFAULT_MAX_TEXT_CHARS: Final = 5000

# What the engine allows of a source (`rinchen.infer.engine_types.MAX_SOURCE_PIECES`; a
# test pins the two equal): CTranslate2 cuts a longer source silently.
_MAX_SOURCE_PIECES: Final = 1023

# Strict so a quoted number or a YAML boolean is a typo to report, not a value to coerce.
_Count = Annotated[int, Field(strict=True, ge=1)]
_PositiveNumber = Annotated[float, Field(strict=True, gt=0, allow_inf_nan=False)]


def _reject_blank(value: str) -> str:
    if not value.strip():
        raise ValueError("must not be blank")
    return value


def _named_compute_type(value: str) -> str:
    try:
        return require_compute_type(value)
    except ConfigError as error:
        raise ValueError(str(error)) from error


_NonBlank = Annotated[str, AfterValidator(_reject_blank)]
_ComputeType = Annotated[str, AfterValidator(_named_compute_type)]
_Path = Annotated[Path, BeforeValidator(reject_blank_path)]


class ScriptConverterConfig(BaseModel):
    """The two files of a D5 script converter: its CTranslate2 export and its SentencePiece
    model. A service that has one turns a Cyrillic result into traditional Mongolian script
    on request."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    ct2_dir: _Path
    spm_model: _Path


class ServeConfig(BaseModel):
    """Everything an operator can set about the service; see the module's text."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    host: _NonBlank = "127.0.0.1"
    port: Annotated[int, Field(strict=True, ge=1, le=65535)] = 8000
    device: Literal["cpu", "cuda"] = "cpu"
    compute_type: _ComputeType = "default"
    # Batches decoded at once, and the threads of one: start from the cores given to the
    # process divided by `intra_threads`, with `intra_threads` at 1 or 2 (spike, CPU only).
    inter_threads: _Count = 2
    intra_threads: _Count = 1
    beam_size: Annotated[int, Field(strict=True, ge=1, le=MAX_BEAM_SIZE)] = 4
    max_decoding_length: _Count = 256
    max_source_pieces: Annotated[
        int, Field(strict=True, ge=1, le=_MAX_SOURCE_PIECES)
    ] = 256
    script_mask: Literal["retry", "always", "off"] = "retry"
    max_texts: Annotated[int, Field(strict=True, ge=1, le=MAX_TEXTS_LIMIT)] = (
        MAX_TEXTS_LIMIT
    )
    max_text_chars: Annotated[
        int, Field(strict=True, ge=1, le=MAX_TEXT_CHARS_LIMIT)
    ] = DEFAULT_MAX_TEXT_CHARS
    max_body_bytes: Annotated[int, Field(strict=True, ge=1024, le=64 * MIB)] = MIB
    max_batch_sentences: _Count = 32
    max_wait_ms: _PositiveNumber = 5.0
    max_queue_sentences: _Count = 1024
    request_timeout_seconds: _PositiveNumber = 30.0
    shutdown_timeout_seconds: _PositiveNumber = 20.0
    log_level: Literal["critical", "error", "warning", "info", "debug"] = "info"
    script_converter: ScriptConverterConfig | None = None

    @property
    def graceful_wait_seconds(self) -> int:
        """The first share of the stop budget: whole seconds (`uvicorn` takes no fraction)
        that the server waits for the connections and the requests in flight, half of
        `shutdown_timeout_seconds` rounded up, and none when the timeout is under a second."""
        if self.shutdown_timeout_seconds < 1:
            return 0
        return math.ceil(self.shutdown_timeout_seconds / 2)

    @property
    def drain_wait_seconds(self) -> float:
        """The rest of the stop budget: what the end of the lifespan waits for the queue and
        the batches still running. `uvicorn` runs the lifespan only after its own wait, so the
        two shares must add up to `shutdown_timeout_seconds`."""
        return self.shutdown_timeout_seconds - self.graceful_wait_seconds

    @field_validator("max_queue_sentences")
    @classmethod
    def _queue_holds_a_whole_request(cls, value: int, info: ValidationInfo) -> int:
        max_texts = info.data.get("max_texts")
        # A `max_texts` that failed its own check is reported once, under its own name.
        if isinstance(max_texts, int) and value < max_texts:
            raise ValueError(
                f"max_queue_sentences ({value}) must be at least max_texts ({max_texts}): "
                "a request of the largest size could never be queued"
            )
        return value


class ServeEnvSettings(BaseSettings):
    """The `RINCHEN_SERVE_*` environment variables, every field optional (a variable that
    is not set, or is empty, is not a layer). Only the types are checked here; the bounds
    are `ServeConfig`'s. The script converter is two flat variables,
    `RINCHEN_SERVE_SCRIPT_CONVERTER_CT2_DIR` and `..._SPM_MODEL`."""

    model_config = SettingsConfigDict(
        env_prefix=ENV_PREFIX, extra="ignore", env_ignore_empty=True
    )

    host: str | None = None
    port: int | None = None
    device: str | None = None
    compute_type: str | None = None
    inter_threads: int | None = None
    intra_threads: int | None = None
    beam_size: int | None = None
    max_decoding_length: int | None = None
    max_source_pieces: int | None = None
    script_mask: str | None = None
    max_batch_sentences: int | None = None
    max_wait_ms: float | None = None
    max_queue_sentences: int | None = None
    max_texts: int | None = None
    max_text_chars: int | None = None
    max_body_bytes: int | None = None
    request_timeout_seconds: float | None = None
    shutdown_timeout_seconds: float | None = None
    log_level: str | None = None
    script_converter_ct2_dir: Path | None = None
    script_converter_spm_model: Path | None = None

    def layer(self) -> dict[str, Any]:
        """The fields that were set, as the keys of a `ServeConfig`."""
        values = self.model_dump(exclude_none=True)
        converter = {
            name: values.pop(f"script_converter_{name}")
            for name in ("ct2_dir", "spm_model")
            if f"script_converter_{name}" in values
        }
        if converter:
            values["script_converter"] = converter
        return values


class _GivenEnvironment(ServeEnvSettings):
    """`ServeEnvSettings` over a mapping the caller hands in: the process environment is
    not read."""

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        return (init_settings,)


def _environment_layer(env: Mapping[str, str] | None) -> dict[str, Any]:
    try:
        if env is None:
            return ServeEnvSettings().layer()
        given: dict[str, Any] = {
            name[len(ENV_PREFIX) :].lower(): value
            for name, value in env.items()
            if name.upper().startswith(ENV_PREFIX) and value != ""
        }
        return _GivenEnvironment(**given).layer()
    except ValidationError as error:
        raise ConfigError(
            f"invalid {ENV_PREFIX}* environment: {_where(error)}"
        ) from error


def _where(error: ValidationError) -> str:
    return "; ".join(
        f"{'.'.join(str(part) for part in item['loc']) or 'config'}: {item['msg']}"
        for item in error.errors(include_input=False)
    )


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ConfigError(f"serve config file not found: {path}")
    try:
        with path.open(encoding="utf-8") as handle:
            raw = yaml.safe_load(handle)
    except (yaml.YAMLError, UnicodeDecodeError, OSError) as error:
        raise ConfigError(
            f"invalid serve config {path}: cannot read or parse: {error}"
        ) from error
    if not isinstance(raw, dict):
        raise ConfigError(
            f"invalid serve config {path}: top level must be a mapping of field names "
            f"to values, got {type(raw).__name__}"
        )
    return raw


def _stack(layers: list[Mapping[str, Any]]) -> dict[str, Any]:
    """The layers laid over each other, the last one winning; the one nested table
    (`script_converter`) is merged key by key."""
    merged: dict[str, Any] = {}
    for layer in layers:
        for key, value in layer.items():
            below = merged.get(key)
            if isinstance(value, Mapping) and isinstance(below, Mapping):
                merged[key] = {**below, **value}
            else:
                merged[key] = value
    return merged


def load_serve_config(
    path: Path | None,
    *,
    env: Mapping[str, str] | None = None,
    overrides: Mapping[str, object] | None = None,
) -> ServeConfig:
    """The service's configuration: the defaults, then `path` (a YAML file, or none), then
    the environment (`env`, the process's own when `None`), then `overrides` (the command's
    flags; a value of `None` is a flag that was not given).

    Raises:
        ConfigError: the file is missing, unreadable, not a mapping, or names an unknown key
            (the key is named), a value in any layer is of the wrong type or out of range,
            or `device` or `compute_type` is `auto`.
    """
    file_layer = _read_yaml(path) if path is not None else {}
    given = {k: v for k, v in (overrides or {}).items() if v is not None}
    merged = _stack([file_layer, _environment_layer(env), given])
    try:
        return ServeConfig.model_validate(merged)
    except ValidationError as error:
        origin = f" {path}" if path is not None else ""
        raise ConfigError(f"invalid serve config{origin}: {_where(error)}") from error
