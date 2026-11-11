"""Shared setup of the tests of the report of D13 (Task 8): the runs of the golden suite as
`RunInput`s (built through the same functions `rinchen eval run` writes with, so a run's files are
what the command writes), the shipped metrics configuration, and the helpers that make a variant
of a run or of a suite for a refusal.

`run_input` scores a system over a suite the way the command does (`run_suite`, then
`documents_of`) and reads the `run.json` it produced back, so the `RunRecord` of a `RunInput` is
the parsed file and its `run_sha256` the digest of that file's bytes. A suite is keyed by the digest
of its bytes (`suite_digest`), which is how the report matches a run to its suite.

Nothing here loads a model, opens a socket or touches a GPU.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Final

from rinchen.cli.eval_run_output import documents_of
from rinchen.config.eval_metrics import EvalMetricsConfig, load_eval_metrics_config
from rinchen.eval.bootstrap import BootstrapSettings
from rinchen.eval.outputs import Output, OutputKey, outputs_to_bytes
from rinchen.eval.report import RunInput
from rinchen.eval.result import RUN_FILE, run_from_bytes
from rinchen.eval.robustness_input import RobustnessInput
from rinchen.eval.runner import run_suite
from rinchen.eval.suites import Suite, SuiteRow
from rinchen.eval.systems import FileSystem, SuiteSystem, SystemRecord
from tests.support.eval_fakes import FakeSuiteSystem, golden_outputs, golden_suite
from tests.support.eval_golden import (
    ALPHA_ENG,
    ALPHA_KHK,
    BETA_ENG,
    BETA_KHK,
    GOLDEN_IDS,
)

ROOT: Final = Path(__file__).resolve().parents[2]
SHIPPED_CONFIG: Final = ROOT / "configs" / "eval" / "metrics.yaml"
SETTINGS: Final = BootstrapSettings(resamples=1000, seed=20_261_110, alpha=0.05)
FIRST: Final = "khk_Cyrl-eng_Latn"
SECOND: Final = "eng_Latn-khk_Cyrl"
THIRD: Final = "eng_Latn-rus_Cyrl"
MODEL_SHA256: Final = "a1" * 32
MANIFEST_SHA256: Final = "b2" * 32
# Rows 1, 7 and 8 of `beta` in the place of `alpha`'s: against `alpha`, on the ten segments of the
# golden suite taken as units, the paired chrF++ of the first direction has 25 of 1,000 resamples
# at or below zero, so its interval [1.949, 16.518] excludes zero while its p-value 52/1001 is
# above 0.05. The one place the interval and the p-value part (`GAMMA_*` below).
GAMMA_TAKES_BETA_AT: Final = (1, 7, 8)


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def shipped_config() -> EvalMetricsConfig:
    return load_eval_metrics_config(SHIPPED_CONFIG.read_bytes())


def suite_digest(suite: Suite) -> str:
    return sha256(suite.to_bytes())


def without_documents(suite: Suite, name: str | None = None) -> Suite:
    """The suite with no document on any row: its bootstrap unit is the segment."""
    return replace(
        suite,
        name=suite.name if name is None else name,
        rows=tuple(SuiteRow(row.id, None, row.text) for row in suite.rows),
    )


def file_system(
    name: str, outputs: Mapping[OutputKey, Output] | None = None
) -> FileSystem:
    """A hypotheses-file system named `name` over `outputs` (the golden ones of `alpha` or
    `beta` by default)."""
    held = golden_outputs(name) if outputs is None else outputs
    return FileSystem(held, name, sha256(outputs_to_bytes(held)))


def mixed_outputs() -> dict[OutputKey, Output]:
    """`alpha`'s outputs with the rows of `GAMMA_TAKES_BETA_AT` taken from `beta`."""
    outputs: dict[OutputKey, Output] = {}
    for direction, alpha, beta in (
        (FIRST, ALPHA_ENG, BETA_ENG),
        (SECOND, ALPHA_KHK, BETA_KHK),
    ):
        for index, row_id in enumerate(GOLDEN_IDS):
            text = beta[index] if index in GAMMA_TAKES_BETA_AT else alpha[index]
            outputs[OutputKey(direction, row_id)] = Output(text, "ok")
    return outputs


def run_input(
    suite: Suite,
    system: SuiteSystem,
    *,
    settings: BootstrapSettings = SETTINGS,
    selection: Sequence[str] | None = None,
    robustness: RobustnessInput | None = None,
    scores: Mapping[str, Mapping[OutputKey, float]] | None = None,
) -> RunInput:
    """`system` over `suite`, scored as `rinchen eval run` scores it: the parsed `run.json`, its
    results, the outputs it wrote and the digests of the suite and of `run.json`."""
    digest = suite_digest(suite)
    run = run_suite(suite, digest, system, selection)
    documents = documents_of(
        suite=suite,
        suite_sha256=digest,
        run=run,
        outputs=outputs_to_bytes(run.outputs),
        settings=settings,
        selection=tuple(selection) if selection is not None else None,
        robustness=robustness,
        scores=scores or {},
        score_files={metric: "ab" * 32 for metric in scores or {}},
    )
    run_json = next(item.data for item in documents.files if item.name == RUN_FILE)
    return RunInput(
        record=run_from_bytes(run_json),
        results=tuple(documents.results),
        outputs=run.outputs,
        suite_sha256=digest,
        run_sha256=sha256(run_json),
    )


def golden_run(name: str, **options: object) -> RunInput:
    """The golden suite scored for the golden system `alpha` or `beta`."""
    return run_input(golden_suite(), file_system(name), **options)  # type: ignore[arg-type]


def golden_runs() -> list[RunInput]:
    return [golden_run("alpha"), golden_run("beta")]


def golden_suites() -> dict[str, Suite]:
    suite = golden_suite()
    return {suite_digest(suite): suite}


def rinchen_system(name: str, outputs: Mapping[OutputKey, Output]) -> FakeSuiteSystem:
    """A system of kind `rinchen` (a model of ours) that answers `outputs` and runs every
    direction of the suite it holds."""
    return FakeSuiteSystem(
        answer=lambda requests: [
            outputs[OutputKey(request.direction, request.id)] for request in requests
        ],
        record=SystemRecord(
            kind="rinchen",
            name=name,
            model_sha256=MODEL_SHA256,
            manifest_sha256=MANIFEST_SHA256,
        ),
    )


# ---- variants of the golden suite and of a robustness run -------------------------------------


def with_third_direction(suite: Suite) -> Suite:
    """The golden suite with a Russian sentence on every row and a third direction,
    `eng_Latn-rus_Cyrl`, which a system that only holds the first two cannot run."""
    rows = tuple(
        SuiteRow(row.id, row.doc, {**row.text, "rus_Cyrl": f"русский {row.id}"})
        for row in suite.rows
    )
    return replace(suite, directions=(*suite.directions, THIRD), rows=rows)


ROBUSTNESS_BLOCK: Final = {
    "numbers_acc": 0.9292,
    "placeholder_acc": 0.9983,
    "term_acc": 1.0,
    "script_purity": 1.0,
    "entity_acc": 0.9312,
    "mong_source_retention": 0.9976,
}


def robustness_input(
    system_name: str = "alpha",
    *,
    block: Mapping[str, float | None] | None = None,
    model_sha256: str | None = None,
    manifest_sha256: str | None = None,
) -> RobustnessInput:
    """D12's block with its coverage for `system_name`, as `read_robustness` returns it."""
    coverage: dict[str, dict[str, float | int | None]] = {
        key: {"scored": 960, "excluded": 0, "total": 960} for key in ROBUSTNESS_BLOCK
    }
    coverage["placeholder_acc"] = {"scored": 599, "excluded": 1, "total": 600}
    coverage["mong_source_retention"] = {
        "scored": 900,
        "excluded": 0,
        "total": 900,
        "chrf_cyrl": 91.25,
        "chrf_mong": 91.0,
    }
    return RobustnessInput(
        block=ROBUSTNESS_BLOCK if block is None else block,
        coverage=coverage,
        system_name=system_name,
        model_sha256=model_sha256,
        manifest_sha256=manifest_sha256,
        hypotheses_sha256=None,
        suite_name="robust",
        suite_sha256="e5" * 32,
        robustness_sha256="f6" * 32,
        report_sha256="17" * 32,
    )
