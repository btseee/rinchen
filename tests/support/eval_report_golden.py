"""The golden report of D13 (Task 8), written out from literals: the document `rinchen eval report`
makes of the two golden runs, built here with nothing but the standard library from the tables of
`tests.support.eval_golden` (the points and intervals of Tasks 2 and 6), the paired chrF++ rows of
Task 2 and the literals of this file, so that a comparison with `tests/fixtures/eval/
golden_report.json` is a comparison with an independent rendering and not with the code under test.

The literals of this file were derived for this task with the pure-Python reference of the design
spike (a loop over the published splitmix64 index stream, sacrebleu's own score-from-statistics of
integer sums, clamped to 100.0, no code of `rinchen` imported):

- `GOLDEN_PAIRED_BLEU`: `beta` against `alpha`, BLEU, the five documents as units (the chrF++ rows
  are Task 2's, `tests.support.eval_golden.GOLDEN_PAIRED`).
- `GAMMA_PAIRED`: `gamma` (the outputs of `alpha` with rows 1, 7 and 8 of `beta`) against `alpha`,
  the ten segments as units, 1,000 resamples from seed 20261110: 25 resamples at or below zero in
  every case, so p = 2 * 26 / 1001 = 0.05194805194805195 and `better_fraction` 0.975, with an
  interval whose low end is above zero: the case in which the interval and the p-value part.

The digests of the two `run.json` files are the digests of files the test reads: they are what
`run_sha256` is defined to be, not something to derive.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any, Final

import yaml

from tests.support.eval_golden import (
    GOLDEN_DOCUMENT_INTERVALS,
    GOLDEN_NO_SCORER_REASON,
    GOLDEN_PAIRED,
    GOLDEN_SEGMENT_ESTIMATES,
    GOLDEN_SIGNATURES,
    PairedGolden,
)

ROOT: Final = Path(__file__).resolve().parents[2]
FIXTURES: Final = ROOT / "tests" / "fixtures" / "eval"
GOLDEN_REPORT_JSON: Final = FIXTURES / "golden_report.json"
GOLDEN_REPORT_MD: Final = FIXTURES / "golden_report.md"
FIRST: Final = "khk_Cyrl-eng_Latn"
SECOND: Final = "eng_Latn-khk_Cyrl"
# `sha256sum` of tests/fixtures/eval/golden_suite.json, golden_outputs_alpha.jsonl and
# golden_outputs_beta.jsonl.
SUITE_SHA256: Final = "b707518d39133bd206092e6df28bf0bcf1e42c776ad5c16900995bf783ebc761"
OUTPUTS_SHA256: Final[Mapping[str, str]] = {
    "alpha": "0708a74060c85c5d579d55d907a36ff7bb0ce419c4ab558d1cb16314c5aca2da",
    "beta": "fcf6f29435de72e35b151e6bb22f854455c1604bc237130551cf26fcdd46ceef",
}
# The eight partners of Khalkha in the registry (the non-hub, non-traditional-script languages),
# sorted; `khk_Cyrl-p` then `p-khk_Cyrl` for each.
PARTNERS: Final = (
    "deu_Latn",
    "eng_Latn",
    "jpn_Jpan",
    "kaz_Cyrl",
    "kor_Hang",
    "rus_Cyrl",
    "tur_Latn",
    "zho_Hans",
)
STANDARD_DIRECTIONS: Final = tuple(
    direction
    for partner in PARTNERS
    for direction in (f"khk_Cyrl-{partner}", f"{partner}-khk_Cyrl")
)
COUNT_METRICS: Final = ("bleu", "chrf", "chrf_pp")
NEURAL_METRICS: Final = ("comet22", "metricx24", "metricx24_qe")

GOLDEN_PAIRED_BLEU: Final[Mapping[str, PairedGolden]] = {
    FIRST: PairedGolden(
        35.0143964409301,
        22.909663629023655,
        54.784823724544815,
        0.001998001998001998,
        1.0,
    ),
    SECOND: PairedGolden(
        40.19887253369139,
        29.19958222390276,
        57.54538352286106,
        0.001998001998001998,
        1.0,
    ),
}
GAMMA_PAIRED: Final[Mapping[tuple[str, str], PairedGolden]] = {
    (FIRST, "chrf_pp"): PairedGolden(
        8.21699012558598,
        1.9494531222593139,
        16.51836251143874,
        0.05194805194805195,
        0.975,
    ),
    (FIRST, "bleu"): PairedGolden(
        18.857239182291522,
        2.4286512430794573,
        42.04308434323276,
        0.05194805194805195,
        0.975,
    ),
    (SECOND, "chrf_pp"): PairedGolden(
        10.337777913152308,
        1.4254405741535123,
        20.09892195437284,
        0.05194805194805195,
        0.975,
    ),
    (SECOND, "bleu"): PairedGolden(
        23.41700525349392,
        4.386599849368057,
        48.601602322882854,
        0.05194805194805195,
        0.975,
    ),
}


def result_document(system: str, direction: str) -> dict[str, Any]:
    """The result of `system` over `direction` of the golden suite as the report holds it (no
    robustness fields): every number a literal of the golden tables, the document unit."""
    points = {
        metric: GOLDEN_SEGMENT_ESTIMATES[(direction, system, metric)].point
        for metric in COUNT_METRICS
    }
    intervals = {
        metric: list(GOLDEN_DOCUMENT_INTERVALS[(direction, system, metric)])
        for metric in COUNT_METRICS
    }
    return {
        "schema": "rinchen.eval/1",
        "system": system,
        "suite": "golden",
        "direction": direction,
        "n_segments": 10,
        "trained": True,
        "statuses": {"ok": 10},
        "metrics": {**points, **dict.fromkeys(NEURAL_METRICS)},
        "bleu_signature": GOLDEN_SIGNATURES["bleu"],
        "chrf_signature": GOLDEN_SIGNATURES["chrf"],
        "chrf_pp_signature": GOLDEN_SIGNATURES["chrf_pp"],
        "unmeasured": dict.fromkeys(NEURAL_METRICS, GOLDEN_NO_SCORER_REASON),
        "ci95": {**intervals, **dict.fromkeys(NEURAL_METRICS)},
        "bootstrap": {
            "resamples": 1000,
            "seed": 20261110,
            "alpha": 0.05,
            "interval": "percentile",
            "rng": "splitmix64-counter/1",
            "unit": "document",
            "units": 5,
        },
        "manifest_sha": None,
        "provenance": {
            "suite_sha256": SUITE_SHA256,
            "outputs_sha256": OUTPUTS_SHA256[system],
            "system": {
                "kind": "hypotheses",
                "name": system,
                "model_sha256": None,
                "manifest_sha256": None,
                "options": {},
                "outputs_sha256": OUTPUTS_SHA256[system],
            },
        },
    }


def comparison_document(direction: str, metric: str) -> dict[str, Any]:
    """`beta` against `alpha` in `direction`, `metric` `chrf_pp` or `bleu`."""
    paired = (
        GOLDEN_PAIRED[(direction, "document")]
        if metric == "chrf_pp"
        else GOLDEN_PAIRED_BLEU[direction]
    )
    return {
        "suite": "golden",
        "direction": direction,
        "system": "beta",
        "baseline": "alpha",
        "metric": metric,
        "higher_is_better": True,
        "delta": paired.delta,
        "ci95": [paired.low, paired.high],
        "p_value": paired.p_value,
        "better_fraction": paired.better_fraction,
        "significant": True,
        "unit": "document",
        "units": 5,
    }


def _config_targets() -> list[dict[str, Any]]:
    """The four targets of the shipped configuration, read with PyYAML and nothing of ours."""
    table = yaml.safe_load(
        (ROOT / "configs" / "eval" / "metrics.yaml").read_text("utf-8")
    )
    by_id = {target["id"]: target for target in table["targets"]}
    nothing = "no robustness run of a rinchen system was given"
    return [
        {
            "id": "beats_baseline",
            "description": by_id["beats_baseline"]["description"],
            "threshold": by_id["beats_baseline"]["min_directions"],
            "value": None,
            "status": "unmeasured",
            "basis": "0 of 16 directions measured",
        },
        {
            "id": "off_target",
            "description": by_id["off_target"]["description"],
            "threshold": by_id["off_target"]["threshold"],
            "value": None,
            "status": "unmeasured",
            "basis": "no language identification is run",
        },
        {
            "id": "numbers",
            "description": by_id["numbers"]["description"],
            "threshold": by_id["numbers"]["threshold"],
            "value": None,
            "status": "unmeasured",
            "basis": nothing,
        },
        {
            "id": "placeholders",
            "description": by_id["placeholders"]["description"],
            "threshold": by_id["placeholders"]["threshold"],
            "value": None,
            "status": "unmeasured",
            "basis": nothing,
        },
    ]


def golden_report_document(run_digests: Mapping[str, str]) -> dict[str, Any]:
    """The report of the golden runs of `alpha` and `beta` (baseline `alpha`), `run_digests`
    being the SHA-256 of each system's `run.json`."""
    declared = {FIRST: 10, SECOND: 10}
    systems = ("alpha", "beta")
    return {
        "schema": "rinchen.report/1",
        "evidence": {"kinds": ["fixture"], "fixture_only": True},
        "systems": [
            {
                "name": name,
                "kind": "hypotheses",
                "model_sha256": None,
                "manifest_sha256": None,
                "options": {},
            }
            for name in systems
        ],
        "suites": [
            {
                "name": "golden",
                "kind": "fixture",
                "benchmark": "golden",
                "split": "devtest",
                "purpose": "report",
                "sha256": SUITE_SHA256,
                "rows": 10,
                "directions": [
                    {"direction": d, "rows": declared.get(d)}
                    for d in STANDARD_DIRECTIONS
                ],
                "source": {
                    "dataset": None,
                    "revision": None,
                    "pinned": False,
                    "missing_languages": [],
                },
            }
        ],
        "runs": [
            {"suite": "golden", "system": name, "run_sha256": run_digests[name]}
            for name in systems
        ],
        "coverage": [
            {
                "suite": "golden",
                "system": name,
                "direction": direction,
                "state": "run",
                "reason": None,
            }
            for name in systems
            for direction in (FIRST, SECOND)
        ],
        "results": [
            result_document(name, direction)
            for name in systems
            for direction in (FIRST, SECOND)
        ],
        "robustness": {},
        "comparisons": [
            comparison_document(direction, metric)
            for direction in (FIRST, SECOND)
            for metric in ("chrf_pp", "bleu")
        ],
        "targets": _config_targets(),
        "bootstrap": {
            "resamples": 1000,
            "seed": 20261110,
            "alpha": 0.05,
            "interval": "percentile",
            "rng": "splitmix64-counter/1",
        },
        "note_multiple_comparisons": (
            "16 intervals and 4 paired comparisons are reported with no correction "
            "for multiple comparisons"
        ),
    }
