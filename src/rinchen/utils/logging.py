"""Structured JSON logging setup via structlog."""
from __future__ import annotations

import logging
import sys

import structlog

_CONFIGURED = False


def _configure() -> None:
    global _CONFIGURED
    if _CONFIGURED:
        return
    # basicConfig is a no-op once the root logger has handlers (pytest's logging
    # plugin installs some), and root stays at WARNING, so set both explicitly.
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter("%(message)s"))
    root = logging.getLogger()
    root.addHandler(handler)
    root.setLevel(logging.INFO)
    structlog.configure(
        processors=[
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.add_log_level,
            structlog.processors.JSONRenderer(),
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )
    _CONFIGURED = True


def get(name: str) -> structlog.stdlib.BoundLogger:
    """Return a structlog logger configured for JSON output on stderr."""
    _configure()
    return structlog.get_logger(name)
