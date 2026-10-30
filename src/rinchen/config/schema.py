"""Pydantic schema for Rinchen's resolved configuration.

Minimal at D0: just enough fields to prove the merge mechanism. Later
milestones add real fields as they need profile-driven config.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class RinchenConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    profile: str
    seed: int = 0
    paths: dict[str, str] = {}
