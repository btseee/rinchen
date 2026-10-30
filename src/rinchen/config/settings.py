"""Environment-variable overlay for Rinchen configuration, via pydantic-settings."""

from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class RinchenEnvSettings(BaseSettings):
    """`RINCHEN_*` environment variables, applied before CLI `--set` overrides
    (CLI overrides win over env vars, which win over profile/base YAML)."""

    model_config = SettingsConfigDict(env_prefix="RINCHEN_", extra="ignore")

    seed: int | None = None
