"""Pydantic model for a stage's manifest.json, per <contracts> in the master plan."""

from __future__ import annotations

import warnings

from pydantic import BaseModel, ConfigDict


class ManifestFile(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: str
    sha256: str
    bytes: int | None = None
    rows: int | None = None


# The contract names a field `schema`, and pydantic warns at class creation
# that it shadows BaseModel.schema(). The CLI imports this module, so the
# warning would print on every `rinchen` command.
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", message='Field name "schema" in "Manifest"')

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
        # Per-stage outcome: "ok", "partial: ..." or "skipped: ...".
        status: dict[str, str] = {}
