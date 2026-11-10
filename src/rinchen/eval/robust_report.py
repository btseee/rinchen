"""The narrow report of D12 (spec T15): `report.json`, schema `rinchen.report.robustness/1`, the
file the STOP gate reads. It names the system and the suite, holds the six values of
`robustness_block`, the counts each value rests on (`coverage`, from `robustness_coverage`) and
points at `robustness.json` by the SHA-256 of its bytes. No timestamp.

A value is a share of the units that were scored: read it with its `coverage`, which says how
many were left out. The retention is a ratio of two chrF scores, can exceed 1 and means nothing
when the Cyrillic chrF beside it is near zero.

D13 owns the full report: its writer embeds `robustness_block` as it is and `robustness_coverage`
beside it: the coverage is not part of the block, so a report that carries only the block hides
how many units each value left out. When it exists this module and the command's `report.json`
line are removed or kept as a per-suite artefact; the block, `robustness.json` and the command
stay."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from typing import Any, Final

from rinchen.eval.robustness import (
    RobustnessResult,
    robustness_block,
    robustness_coverage,
)

__all__ = ["REPORT_SCHEMA", "ROBUSTNESS_FILE", "report_bytes", "robustness_report"]

REPORT_SCHEMA: Final = "rinchen.report.robustness/1"
ROBUSTNESS_FILE: Final = "robustness.json"
_SHA256: Final = re.compile(r"[0-9a-f]{64}")


def robustness_report(
    result: RobustnessResult, robustness_sha256: str
) -> dict[str, Any]:
    """The report of `result`, whose `robustness.json` bytes have the SHA-256
    `robustness_sha256`.

    Raises:
        ValueError: `robustness_sha256` is not a SHA-256 hex digest.
    """
    if _SHA256.fullmatch(robustness_sha256) is None:
        raise ValueError("robustness_sha256 must be a SHA-256 hex digest")
    system = result.system
    return {
        "schema": REPORT_SCHEMA,
        "system": {
            "name": system.name,
            "model_sha256": system.model_sha256,
            "manifest_sha256": system.manifest_sha256,
        },
        "suite": {"name": result.suite_name, "sha256": result.suite_sha256},
        "robustness": robustness_block(result),
        "coverage": robustness_coverage(result),
        "robustness_file": ROBUSTNESS_FILE,
        "robustness_sha256": robustness_sha256,
    }


def report_bytes(report: Mapping[str, Any]) -> bytes:
    """The UTF-8 bytes of `report.json`: sorted keys, two-space indent, one final LF.

    Raises:
        ValueError: a value is not a finite number (JSON has no NaN).
    """
    text = json.dumps(
        report, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False
    )
    return f"{text}\n".encode()
