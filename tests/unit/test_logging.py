"""Tests for rinchen.utils.logging.get."""
import json

from rinchen.utils import logging as rinchen_logging


def test_get_returns_logger_that_emits_json(capsys):
    logger = rinchen_logging.get("test_logger")
    logger.info("stage_start", stage="demo", count=3)

    captured = capsys.readouterr()
    line = captured.err.strip().splitlines()[-1]
    payload = json.loads(line)

    assert payload["event"] == "stage_start"
    assert payload["stage"] == "demo"
    assert payload["count"] == 3
