"""Project-wide pytest fixtures and startup configuration."""

from __future__ import annotations

import sys

if sys.platform == "win32":
    # Windows' default console encoding (cp1252) can't print traditional
    # Mongol-script or other non-Latin-1 Unicode content. Without this, a
    # failing assertion that includes such content crashes with an
    # unrelated UnicodeEncodeError instead of showing the real failure.
    sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
    sys.stderr.reconfigure(encoding="utf-8", errors="backslashreplace")
