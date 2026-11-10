"""The documents the tests of the result and the run record of D13 start from: the golden
`result.json` as a plain dictionary built from the literals of `tests.support.eval_golden` (never
from the code under test), a valid `run.json` as a dictionary, and the helpers that change them,
write them like the product does and check what a refusal says.

Each refusal test plants `MARKER` in the very value a refusal is about, so a message that echoed
the offending value would show it. The marker is not a well-formed id, digest or code."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Final

from rinchen.eval.systems import SystemRecord
from tests.support.eval_golden import (
    GOLDEN_BOOTSTRAP_DOCUMENT,
    GOLDEN_DOCUMENT_INTERVALS,
    GOLDEN_NO_SCORER_REASON,
    GOLDEN_SEGMENT_ESTIMATES,
    GOLDEN_SIGNATURES,
)

FIXTURES: Final = Path(__file__).resolve().parents[1] / "fixtures" / "eval"
GOLDEN_RESULT_FILE: Final = FIXTURES / "golden_result_alpha_khk_eng.json"
# `sha256sum` of tests/fixtures/eval/golden_suite.json and golden_outputs_alpha.jsonl.
GOLDEN_SUITE_SHA256: Final = (
    "b707518d39133bd206092e6df28bf0bcf1e42c776ad5c16900995bf783ebc761"
)
GOLDEN_ALPHA_OUTPUTS_SHA256: Final = (
    "0708a74060c85c5d579d55d907a36ff7bb0ce419c4ab558d1cb16314c5aca2da"
)
MARKER: Final = "SECRET MARKER 66"
DIGEST: Final = "ab" * 32
OTHER_DIGEST: Final = "cd" * 32
DIRECTION: Final = "khk_Cyrl-eng_Latn"
FIRST, SECOND = "khk_Cyrl-eng_Latn", "eng_Latn-khk_Cyrl"
THIRD: Final = "eng_Latn-rus_Cyrl"
OUTPUTS_FILE: Final = "hypotheses.jsonl"
NEURAL: Final = ("comet22", "metricx24", "metricx24_qe")
STATUS_NAMES: Final = ("ok", "empty", "too_long", "truncated", "no_output")


class _Delete:
    """What `edited` is given to remove a key."""


DELETE: Final = _Delete()


def golden_result_document() -> dict[str, Any]:
    """The result of the system `alpha` over the golden suite, direction `khk_Cyrl-eng_Latn`,
    the document unit (five documents), no robustness: every number a literal of the golden
    tables, the way the file is meant to read. Fresh at every call."""
    points = {
        metric: GOLDEN_SEGMENT_ESTIMATES[(DIRECTION, "alpha", metric)].point
        for metric in ("bleu", "chrf", "chrf_pp")
    }
    intervals = {
        metric: list(GOLDEN_DOCUMENT_INTERVALS[(DIRECTION, "alpha", metric)])
        for metric in ("bleu", "chrf", "chrf_pp")
    }
    return {
        "schema": "rinchen.eval/1",
        "system": "alpha",
        "suite": "golden",
        "direction": DIRECTION,
        "n_segments": 10,
        "trained": True,
        "statuses": {"ok": 10},
        "metrics": {**points, **dict.fromkeys(NEURAL)},
        "bleu_signature": GOLDEN_SIGNATURES["bleu"],
        "chrf_signature": GOLDEN_SIGNATURES["chrf"],
        "chrf_pp_signature": GOLDEN_SIGNATURES["chrf_pp"],
        "unmeasured": dict.fromkeys(NEURAL, GOLDEN_NO_SCORER_REASON),
        "ci95": {**intervals, **dict.fromkeys(NEURAL)},
        "bootstrap": dict(GOLDEN_BOOTSTRAP_DOCUMENT),
        "robustness": None,
        "robustness_coverage": None,
        "robustness_source": None,
        "manifest_sha": None,
        "provenance": {
            "suite_sha256": GOLDEN_SUITE_SHA256,
            "outputs_sha256": GOLDEN_ALPHA_OUTPUTS_SHA256,
            "system": {
                "kind": "hypotheses",
                "name": "alpha",
                "model_sha256": None,
                "manifest_sha256": None,
                "options": {},
                "outputs_sha256": GOLDEN_ALPHA_OUTPUTS_SHA256,
            },
        },
    }


def run_document() -> dict[str, Any]:
    """A valid `run.json` of the golden suite and the system `alpha`: both directions ran and
    a third (declared by a made-up suite) did not. Fresh at every call."""
    return {
        "schema": "rinchen.eval.run/1",
        "suite": {"name": "golden", "sha256": GOLDEN_SUITE_SHA256},
        "purpose": "report",
        "system": SystemRecord(
            kind="hypotheses", name="alpha", outputs_sha256=GOLDEN_ALPHA_OUTPUTS_SHA256
        ).to_dict(),
        "settings": {
            "resamples": 1000,
            "seed": 20261110,
            "alpha": 0.05,
            "selection": None,
        },
        "directions": [
            {
                "direction": FIRST,
                "state": "run",
                "reason": None,
                "rows": 10,
                "result": f"{FIRST}/result.json",
            },
            {
                "direction": SECOND,
                "state": "run",
                "reason": None,
                "rows": 10,
                "result": f"{SECOND}/result.json",
            },
            {
                "direction": THIRD,
                "state": "not_trained",
                "reason": "not one of the model's trained directions",
                "rows": 10,
                "result": None,
            },
        ],
        "files": {
            OUTPUTS_FILE: GOLDEN_ALPHA_OUTPUTS_SHA256,
            f"{FIRST}/result.json": DIGEST,
            f"{SECOND}/result.json": OTHER_DIGEST,
        },
        "robustness_source": None,
        "segment_scores": {},
    }


def serialised(document: Any) -> bytes:
    """A document as the files of the product are written: sorted keys, two-space indent,
    characters as they are, one final line feed."""
    text = json.dumps(
        document, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False
    )
    return f"{text}\n".encode()


def edited(
    document: dict[str, Any], path: str | tuple[str, ...], value: object
) -> dict[str, Any]:
    """A deep copy of `document` with `path` set to `value` (or removed when `value` is
    `DELETE`). A path is a dotted string or, for a key that holds a dot (a file name), a tuple
    of its parts. A list is indexed by a number in the path."""
    copied = copy.deepcopy(document)
    parts = path.split(".") if isinstance(path, str) else list(path)
    target: Any = copied
    for part in parts[:-1]:
        target = target[int(part)] if isinstance(target, list) else target[part]
    last = parts[-1]
    key: Any = int(last) if isinstance(target, list) else last
    if value is DELETE:
        del target[key]
    else:
        target[key] = value
    return copied
