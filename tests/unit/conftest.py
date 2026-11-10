"""Fixtures of the unit tests that live in a support module, so that a test names them as a
parameter without importing them (an imported fixture used as a parameter is a redefinition)."""

from __future__ import annotations

from tests.support.serve_app import recording_log  # noqa: F401 - re-exported for pytest
