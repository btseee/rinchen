"""Pydantic model for a stage's manifest.json, per <contracts> in the master plan."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class ManifestFile(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: str
    sha256: str
    bytes: int | None = None
    rows: int | None = None


class Manifest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # "schema" shadows pydantic BaseModel's deprecated schema() classmethod;
    # the contract (master plan <contracts>) requires this exact field name.
    schema: str = "rinchen.manifest/1"  # type: ignore[assignment]
    stage: str
    created: str
    code_version: str
    config_sha256: str
    seed: int
    inputs: list[ManifestFile] = []
    outputs: list[ManifestFile] = []
    metrics: dict[str, float] = {}
    tool_versions: dict[str, str] = {}
