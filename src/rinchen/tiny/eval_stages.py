"""The evaluation stages of the tiny pipeline: `serve`, `robustness` and `evaluate` (D15, spec P2, P8, P9, P10
and findings F8 and F11).

`run_serve` is `serve_stage.run_serve`, the service checked in process. `run_robustness` and
`run_evaluate` run the real `rinchen eval` commands through the runner of the context, in the directories
`10_robustness` and `11_evaluate`, over suites made from the fixtures and the rows the filter kept
(`suites`): the kept rows are read from `03_filter/kept.parquet`, the file the filter stage wrote, and
never from the facts of another stage. Every decode is at the beam of the profile (greedy: F8), the
suites are split as the profile says (`test` for what the model memorised, `devtest` for what it never
saw: a name for the report, nothing is selected or tuned on them) and no run is made over a `dev` suite.

`run_robustness` measures the int8 export with the converter attached and reads `report.json`.
`run_evaluate` has four parts, in this order, each listed in the stage's record:

- `memorised`: the int8 export, then the float32 one, over the three memorised suites (five directions);
  the facts are the six metrics and the statuses of every run, the corpus chrF in the shape the
  thresholds read (`memorised_chrf`), the least and the largest drop `float32 - int8` and how often the
  two exports wrote the same text;
- `heldout`: the int8 export over the two held-out suites with the robustness block of its own robustness
  run embedded (`--robustness`; the memorised runs carry none);
- `neural_seam`: with the planted stand-ins (`ctx.qe_fake`) the hypotheses of the int8 held-out English run
  are scored with `ConceptQE` into a scores file by D13's own writer, whose header names the suite file
  and the hypotheses file by their SHA-256, and `eval run --hypotheses` takes them in; the stage checks
  that the metric is in the result of every direction and counts the rows it scored. Without the flag the
  part is skipped for its closed reason;
- `report`: `rinchen eval report` over the held-out runs, whose `report.json` is checked by its schema.

A part that fails (a command that exits other than 0, a file that is missing or not what its command
writes, a check) fails the stage with the part's reason, the parts after it are skipped as `an earlier
stage failed`, and the facts of the parts before it are still handed back.
"""

from __future__ import annotations

import copy
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, Final

import polars as pl

from rinchen.corpus.readers import scan_corpus
from rinchen.errors import DataError
from rinchen.eval.outputs import OutputKey
from rinchen.eval.segment_scores import scores_to_bytes
from rinchen.tiny import commands
from rinchen.tiny.context import StageContext, StageResult
from rinchen.tiny.eval_readers import (
    Hypothesis,
    read_hypotheses,
    read_robustness_report,
    read_run_result,
)
from rinchen.tiny.evaluate_facts import agreement, chrf_table, drop_extremes, drops_of
from rinchen.tiny.fakes import ConceptEncoder, ConceptQE
from rinchen.tiny.model_stages import (
    CONVERTER_CHAR_MODEL,
    CONVERTER_MODEL_DIR,
    CONVERTER_RUN_DIR,
    KEPT_FILE,
)
from rinchen.tiny.serve_stage import run_serve
from rinchen.tiny.states import (
    FAIL_INPUT,
    PART_IDS,
    SKIP_EARLIER_FAILED,
    SKIP_NO_NEURAL,
    PartRecord,
    StageRecord,
    fail_check,
)
from rinchen.tiny.step_runner import (
    StageCommands,
    StageStopped,
    file_digest,
    open_stage,
    path_in,
    run_stage,
)
from rinchen.tiny.suites import (
    HELDOUT_NAMES,
    MEMORISED_NAMES,
    ROBUSTNESS_NAME,
    heldout_suites,
    memorised_suites,
    robustness_suite,
    write_suites,
)
from rinchen.tiny.thresholds import GATED_SYSTEM, REFERENCE_SYSTEM
from rinchen.utils.part_file import replace_file

__all__ = ["run_evaluate", "run_robustness", "run_serve"]

SUITES_DIR: Final = "suites"
RUNS_DIR: Final = "runs"
REPORT_DIR: Final = "report"
SCORES_DIR: Final = "scores"
REPORT_FILE: Final = "report.json"
HYPOTHESES_FILE: Final = "hypotheses.jsonl"
RESULT_FILE: Final = "result.json"
REPORT_SCHEMA: Final = "rinchen.report/1"
NEURAL_METRIC: Final = "metricx24_qe"
NEURAL_SUFFIX: Final = "qe"
SYSTEM_PREFIX: Final = "tiny-"
NEURAL_CHECK: Final = fail_check("neural_scores")


def system_name(quantization: str) -> str:
    """The name `--system` gives the runs of an export: `tiny-int8`, `tiny-float32`."""
    return f"{SYSTEM_PREFIX}{quantization}"


def _kept_pairs(ctx: StageContext) -> frozenset[tuple[str, str]]:
    """The `(source, target)` texts of the train rows the filter kept, read from its `kept.parquet`.

    Raises:
        DataError: the file is missing or is not a corpus.
    """
    try:
        kept = scan_corpus(path_in(ctx, "filter", KEPT_FILE)).collect()
    except pl.exceptions.PolarsError as error:
        raise DataError(f"{KEPT_FILE}: not a corpus") from error
    train = kept.filter(pl.col("split").cast(pl.String) == "train")
    return frozenset(zip(train["src"].to_list(), train["tgt"].to_list(), strict=True))


# ---- robustness ---------------------------------------------------------------------------------------


def _robustness(stage: StageCommands) -> Mapping[str, Any]:
    ctx = stage.ctx
    suite = robustness_suite(ctx.fixtures, _kept_pairs(ctx))
    rows = suite.row_counts()
    if not rows:
        raise DataError("the kept rows make no robustness suite")
    suites = write_suites(stage.directory / SUITES_DIR, robustness=suite)
    stage.run(
        "run",
        commands.eval_robustness(
            suites[ROBUSTNESS_NAME],
            path_in(ctx, "export", GATED_SYSTEM),
            stage.directory,
            ctx.config.decode.beam_size,
            path_in(ctx, "converter", CONVERTER_MODEL_DIR),
            path_in(ctx, "converter", CONVERTER_RUN_DIR, CONVERTER_CHAR_MODEL),
        ),
    )
    report = read_robustness_report(stage.directory / REPORT_FILE)
    return {**report, "rows": rows}


def run_robustness(ctx: StageContext) -> StageResult:
    """Measure the int8 export with `rinchen eval robustness` over the robustness suite of the kept
    train rows, at the beam of the profile and with the script converter attached, in
    `10_robustness`.

    The facts are D12's six `values`, the `coverage` each rests on and the `rows` of each section of
    the suite. The command measures and has no threshold: the gate reads the values.
    """
    return run_stage(ctx, "robustness", _robustness)


# ---- evaluate -----------------------------------------------------------------------------------------


class _Evaluation:
    """The parts of the evaluation stage, over one directory; each leaves its facts in `facts`."""

    def __init__(self, stage: StageCommands) -> None:
        self.stage = stage
        self.ctx = stage.ctx
        self.runs = stage.directory / RUNS_DIR
        self.directions: dict[str, tuple[str, ...]] = {}
        self.sources: dict[str, dict[str, Mapping[str, str]]] = {}
        self.paths: dict[str, Path] = {}
        self.facts: dict[str, Any] = {}

    # ---- helpers

    def _run_dir(self, suite: str, system: str) -> Path:
        return self.runs / suite / system

    def _run_model(
        self, suite: str, quantization: str, robustness: Path | None
    ) -> dict[str, dict[str, Any]]:
        """Run the export `quantization` over `suite` and read the result of each direction."""
        ctx = self.ctx
        system = system_name(quantization)
        self.stage.run(
            f"{quantization}-{suite}",
            commands.eval_run(
                self.paths[suite],
                path_in(ctx, "export", quantization),
                system,
                self.runs,
                ctx.config.decode.beam_size,
                robustness,
                resamples=ctx.config.evaluation.resamples,
            ),
        )
        return self._results(suite, system)

    def _results(self, suite: str, system: str) -> dict[str, dict[str, Any]]:
        return {
            direction: read_run_result(
                self._run_dir(suite, system) / direction / RESULT_FILE,
                suite=suite,
                system=system,
                direction=direction,
            )
            for direction in self.directions[suite]
        }

    # ---- the parts

    def memorised(self) -> None:
        ctx = self.ctx
        for quantization in (GATED_SYSTEM, REFERENCE_SYSTEM):
            if quantization not in ctx.config.export.quantizations:
                raise DataError("the evaluation needs both exports")
        self._write_suites()
        table: dict[str, dict[str, Any]] = {}
        for quantization in (GATED_SYSTEM, REFERENCE_SYSTEM):
            for suite in MEMORISED_NAMES:
                for direction, facts in self._run_model(
                    suite, quantization, None
                ).items():
                    table.setdefault(direction, {})[quantization] = facts
        chrf = chrf_table(table)
        drops = drops_of(chrf, reference=REFERENCE_SYSTEM, gated=GATED_SYSTEM)
        self.facts.update(
            memorised=table,
            memorised_chrf=chrf,
            int8_chrf_drop=drop_extremes(drops),
            agreement=self._agreement(),
        )

    def _agreement(self) -> dict[str, Any]:
        gated: list[Hypothesis] = []
        reference: list[Hypothesis] = []
        for suite in MEMORISED_NAMES:
            for system, rows in (
                (system_name(GATED_SYSTEM), gated),
                (system_name(REFERENCE_SYSTEM), reference),
            ):
                rows.extend(
                    read_hypotheses(self._run_dir(suite, system) / HYPOTHESES_FILE)
                )
        return agreement(gated, reference)

    def heldout(self) -> None:
        robustness = path_in(self.ctx, "robustness")
        if not (robustness / REPORT_FILE).is_file():
            raise DataError("the robustness stage left no report")
        table: dict[str, dict[str, Any]] = {}
        for suite in HELDOUT_NAMES:
            for direction, facts in self._run_model(
                suite, GATED_SYSTEM, robustness
            ).items():
                table.setdefault(direction, {})[GATED_SYSTEM] = facts
        self.facts["heldout"] = table

    def neural_seam(self) -> None:
        ctx = self.ctx
        suite = HELDOUT_NAMES[0]
        system = f"{system_name(GATED_SYSTEM)}-{NEURAL_SUFFIX}"
        hypotheses_path = (
            self._run_dir(suite, system_name(GATED_SYSTEM)) / HYPOTHESES_FILE
        )
        rows = read_hypotheses(hypotheses_path)
        encoder = ConceptEncoder(ctx.fixtures.nouns, ctx.fixtures.entities)
        scores = ConceptQE(encoder).score(
            [self._source_of(suite, row) for row in rows], [row.text for row in rows]
        )
        scores_dir = self.stage.directory / SCORES_DIR
        scores_dir.mkdir(parents=True, exist_ok=True)
        scores_path = scores_dir / f"{NEURAL_METRIC}.jsonl"
        replace_file(
            scores_path,
            scores_to_bytes(
                NEURAL_METRIC,
                file_digest(self.paths[suite]),
                file_digest(hypotheses_path),
                {
                    OutputKey(row.direction, row.id): score
                    for row, score in zip(rows, scores, strict=True)
                },
            ),
        )
        self.stage.run(
            "neural",
            commands.eval_run_hypotheses(
                self.paths[suite],
                hypotheses_path,
                system,
                self.runs,
                {NEURAL_METRIC: scores_path},
                resamples=ctx.config.evaluation.resamples,
            ),
        )
        self.facts["neural_seam"] = self._neural_facts(suite, system, rows)

    def _source_of(self, suite: str, row: Hypothesis) -> str:
        language = row.direction.split("-", 1)[0]
        try:
            return self.sources[suite][row.id][language]
        except KeyError as error:
            raise DataError(
                f"{HYPOTHESES_FILE}: a row is not one of the suite"
            ) from error

    def _neural_facts(
        self, suite: str, system: str, rows: Sequence[Hypothesis]
    ) -> dict[str, Any]:
        """The metric of every direction of the scored run, which must be there and must rest on every
        row of the direction that was scored."""
        scored: dict[str, float] = {}
        counted: dict[str, int] = {}
        for direction, result in self._results(suite, system).items():
            value = result["metrics"][NEURAL_METRIC]
            expected = sum(1 for row in rows if row.direction == direction)
            if value is None or result["rows"] != expected:
                raise StageStopped(StageRecord("evaluate", "fail", NEURAL_CHECK))
            scored[direction] = value
            counted[direction] = expected
        return {
            "metric": NEURAL_METRIC,
            "system": system,
            "scores": scored,
            "rows": counted,
        }

    def report(self) -> None:
        suites = [self.paths[name] for name in HELDOUT_NAMES]
        runs = [
            self._run_dir(name, system_name(GATED_SYSTEM)) for name in HELDOUT_NAMES
        ]
        out_dir = self.stage.directory / REPORT_DIR
        self.stage.run("report", commands.eval_report(runs, suites, out_dir))
        if commands.read_json(out_dir / REPORT_FILE).get("schema") != REPORT_SCHEMA:
            raise DataError(f"{REPORT_FILE}: not a report of the evaluation")
        self.facts["report"] = {"runs": len(runs), "suites": len(suites)}

    # ---- the suites

    def _write_suites(self) -> None:
        ctx = self.ctx
        section = ctx.config.evaluation
        specs = [
            *memorised_suites(
                ctx.fixtures, _kept_pairs(ctx), split=section.memorised_split
            ),
            *heldout_suites(ctx.fixtures, split=section.heldout_split),
        ]
        self.directions = {spec.name: tuple(spec.directions) for spec in specs}
        self.sources = {
            spec.name: {row.id: row.text for row in spec.rows} for spec in specs
        }
        self.paths = write_suites(self.stage.directory / SUITES_DIR, evaluation=specs)


def run_evaluate(ctx: StageContext) -> StageResult:
    """Run the four parts of the evaluation in `11_evaluate` (see the module's text)."""
    evaluation = _Evaluation(open_stage(ctx, "evaluate"))
    steps: dict[str, Callable[[], None]] = {
        "memorised": evaluation.memorised,
        "heldout": evaluation.heldout,
        "neural_seam": evaluation.neural_seam,
        "report": evaluation.report,
    }
    parts: list[PartRecord] = []
    failure: str | None = None
    for part_id in PART_IDS["evaluate"]:
        if failure is not None:
            parts.append(PartRecord(part_id, "skipped", SKIP_EARLIER_FAILED))
        elif part_id == "neural_seam" and not ctx.qe_fake:
            evaluation.facts["neural_seam"] = None
            parts.append(PartRecord(part_id, "skipped", SKIP_NO_NEURAL))
        else:
            try:
                steps[part_id]()
            except StageStopped as stopped:
                failure = stopped.record.reason or FAIL_INPUT
            except DataError:
                failure = FAIL_INPUT
            parts.append(
                PartRecord(part_id, "ok")
                if failure is None
                else PartRecord(part_id, "fail", failure)
            )
    facts = evaluation.facts
    if failure is not None:
        return StageResult(StageRecord("evaluate", "fail", failure, {}, parts), facts)
    return StageResult(
        StageRecord("evaluate", "ok", detail=copy.deepcopy(facts), parts=parts), facts
    )
