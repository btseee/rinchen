"""Environment-variable overlay for Rinchen configuration, via pydantic-settings."""
from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class RinchenEnvSettings(BaseSettings):
    """`RINCHEN_*` environment variables, applied as the final overlay."""

    model_config = SettingsConfigDict(env_prefix="RINCHEN_", extra="ignore")

    seed: int | None = None
