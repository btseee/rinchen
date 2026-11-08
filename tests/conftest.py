"""Project-wide pytest fixtures and startup configuration."""

from __future__ import annotations

import sys
from collections.abc import Iterator

import pytest
from hypothesis import settings

from tests.support.signal_handlers import DEFAULT_HANDLERS, handlers_installed

settings.register_profile("rinchen", derandomize=True, deadline=None, max_examples=200)
settings.load_profile("rinchen")

if sys.platform == "win32":
    # Windows' default console encoding (cp1252) can't print traditional
    # Mongol-script or other non-Latin-1 Unicode content. Without this, a
    # failing assertion that includes such content crashes with an
    # unrelated UnicodeEncodeError instead of showing the real failure.
    sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
    sys.stderr.reconfigure(encoding="utf-8", errors="backslashreplace")


@pytest.fixture
def default_signal_handlers() -> Iterator[None]:
    """SIGINT raises KeyboardInterrupt and SIGTERM and SIGHUP act by default for the
    test, whatever disposition the pytest process was started with (a job in the
    background starts with SIGINT and SIGHUP ignored); the previous ones come back
    afterwards. For tests that send themselves a real signal or expect the default."""
    with handlers_installed(DEFAULT_HANDLERS):
        yield
