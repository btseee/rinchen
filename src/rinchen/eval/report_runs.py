"""The runs a report is built from (D13, spec T14): what each run is, and every check that makes it
fit to be reported.

A `RunInput` is a run as the command read it: the parsed `RunRecord` (`run.json`), its results, the
outputs it wrote (`hypotheses.jsonl`), the digest of the suite file it scored and the digest of its
`run.json`. `hold_runs` refuses what cannot be reported honestly and returns the runs in a fixed
order (suite name, then system name), so that the report is a function of the SET of runs:

- a run of a `dev` suite (a `ConfigError`: it is for choosing a model and never becomes a result),
  a run that says it is for selection, a suite that was not given, a suite file that is not the one
  the run names or lists other directions than the run does;
- a run scored with other bootstrap settings than the metrics configuration's, whose intervals would
  be another kind of interval than the comparisons;
- two runs of one system over one suite, two suites with one name, a system whose record (kind,
  model, manifest, options) changes from run to run, and a suite that no run scored;
- a result that disagrees with the run that lists it (another system, suite or directions, files
  other than the run's), results of one run that disagree about D12's robustness block, and a block
  that is not the one `run.json` names.

`verified_rows` then recomputes the statistics of every direction from the suite and the outputs
and refuses a result whose segment count, bootstrap record, resampling unit, point value, interval,
statuses, sacrebleu signatures or `trained` flag is not what its outputs score (the digest chain
catches a flipped byte, not a result that was edited and re-digested): a result file is never
believed to hold more than it says, and the comparisons are made from these rows and from nothing a
result file holds. The neural metrics are the exception that cannot be recomputed here (a report holds
no segment scores): their value and interval are what the result says. What is checked is their
presence: a result holds a value for a neural metric exactly when `run.json` lists the scores of that
metric, and holds the runner's reason otherwise. What is not checked is a value or an interval that
the scores would give, a free-text reason of a metric that is unmeasured for another cause, and the
robustness block's values (its file is not in the run directory). A message names suites, systems
(ids), directions and fields and never a text or a value.
"""

from __future__ import annotations

import importlib.metadata
import math
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Final

from rinchen.errors import ConfigError, DataError
from rinchen.eval.bootstrap import BootstrapSettings
from rinchen.eval.outputs import Output, OutputKey
from rinchen.eval.result import OUTPUTS_FILE, DirectionResult
from rinchen.eval.robust_suite_rules import is_id
from rinchen.eval.run_record import RunRecord
from rinchen.eval.scoring import DirectionRows, MetricRows, direction_rows
from rinchen.eval.significance import estimate
from rinchen.eval.suites import Suite
from rinchen.metrics.registry import (
    COUNT_METRICS,
    NO_SCORER_REASON,
    SEGMENT_SCORE_METRICS,
    CountMetricName,
)
from rinchen.metrics.segment_stats import score_from_statistics

__all__ = ["HeldRun", "RunInput", "hold_runs", "robustness_of", "verified_rows"]

_ROBUSTNESS_FIELDS = ("robustness", "robustness_coverage", "robustness_source")
# A point of the report may differ from the one of the result by the rounding of a C library.
_POINT_TOLERANCE = 1e-9
_SIGNATURE_FIELDS: Final[Mapping[CountMetricName, str]] = {
    "bleu": "bleu_signature",
    "chrf": "chrf_signature",
    "chrf_pp": "chrf_pp_signature",
}


@dataclass(frozen=True, slots=True)
class RunInput:
    """A run as the command read it: its `record` (the parsed `run.json`), its `results` (one for
    each direction that ran), the `outputs` it wrote, the SHA-256 of the suite file it scored
    (`suite_sha256`, which must be the one `run.json` names) and that of its `run.json`."""

    record: RunRecord
    results: Sequence[DirectionResult]
    outputs: Mapping[OutputKey, Output]
    suite_sha256: str
    run_sha256: str


@dataclass(frozen=True, slots=True)
class HeldRun:
    """A run that passed the checks: its input, its `suite`, the names of the suite and of the
    system and its `results` by direction, in the order the run lists its directions."""

    input: RunInput
    suite: Suite
    suite_name: str
    system: str
    results: Mapping[str, DirectionResult]


def _where(system: str, suite: str) -> str:
    return f"the run of the system {system!r} over the suite {suite!r}"


def hold_runs(
    runs: Sequence[RunInput],
    suites: Mapping[str, Suite],
    settings: BootstrapSettings,
) -> list[HeldRun]:
    """The runs, checked and sorted by suite name and system name.

    Raises:
        ConfigError: no run, a suite that was not given or that no run scored, a run of a dev
            suite.
        DataError: any other fault named in the module's text.
    """
    if not runs:
        raise ConfigError("a report needs at least one run")
    ordered = sorted(
        runs, key=lambda run: (run.record.suite.name, run.record.system.name)
    )
    held = [_held(run, suites, settings) for run in ordered]
    _require_one_run_for_each(held)
    _require_one_suite_for_each_name(held)
    _require_one_record_for_each_system(held)
    _require_every_suite_used(held, suites)
    return held


def _held(
    run: RunInput, suites: Mapping[str, Suite], settings: BootstrapSettings
) -> HeldRun:
    record = run.record
    system, name = record.system.name, record.suite.name
    if not is_id(system):
        raise DataError(
            "a run names a system that is not an id: 1 to 64 characters, letters, digits, "
            "'.', '_' and '-'"
        )
    where = _where(system, name)
    if run.suite_sha256 != record.suite.sha256:
        raise DataError(
            f"{where} was matched to a suite file other than the one its run.json names"
        )
    suite = suites.get(record.suite.sha256)
    if suite is None:
        raise ConfigError(f"the suite file of {where} was not given")
    if suite.name != name:
        raise DataError(f"{where} was matched to a suite file of another name")
    if suite.split == "dev":
        raise ConfigError(
            f"the suite {name!r} is a dev suite: its runs are for choosing a model and "
            "never become a result"
        )
    if record.purpose == "selection":
        raise ConfigError(
            f"{where} says it is for selection: runs for choosing a model (dev suites) "
            "never become a result"
        )
    scored = record.settings
    if (scored.resamples, scored.seed, scored.alpha) != (
        settings.resamples,
        settings.seed,
        settings.alpha,
    ):
        raise DataError(
            f"{where} was scored with other bootstrap settings than the metrics "
            "configuration"
        )
    if tuple(entry.direction for entry in record.directions) != suite.directions:
        raise DataError(f"{where} lists other directions than the suite declares")
    return HeldRun(run, suite, name, system, _results_of(run, where))


# ---- the runs together ---------------------------------------------------------------------------


def _require_one_run_for_each(held: Sequence[HeldRun]) -> None:
    seen: set[tuple[str, str]] = set()
    for one in held:
        key = (one.suite_name, one.system)
        if key in seen:
            raise DataError(
                f"two runs of the system {one.system!r} over the suite {one.suite_name!r}"
            )
        seen.add(key)


def _require_one_suite_for_each_name(held: Sequence[HeldRun]) -> None:
    digests: dict[str, str] = {}
    for one in held:
        digest = one.input.record.suite.sha256
        if digests.setdefault(one.suite_name, digest) != digest:
            raise DataError(f"two different suites are named {one.suite_name!r}")


def _require_one_record_for_each_system(held: Sequence[HeldRun]) -> None:
    """The record a report states of a system, without the digest of the file it was read from
    (another suite is another file)."""
    seen: dict[str, tuple[Any, ...]] = {}
    for one in held:
        record = one.input.record.system
        identity = (
            record.kind,
            record.model_sha256,
            record.manifest_sha256,
            dict(record.options),
        )
        if seen.setdefault(one.system, identity) != identity:
            raise DataError(
                f"the system {one.system!r} is not the same in every run: its kind, model, "
                "manifest or options differ"
            )


def _require_every_suite_used(
    held: Sequence[HeldRun], suites: Mapping[str, Suite]
) -> None:
    used = {one.input.record.suite.sha256 for one in held}
    for digest, suite in suites.items():
        if digest not in used:
            raise ConfigError(
                f"no run scored the suite {suite.name!r}: a suite that is given and not "
                "used would be dropped silently"
            )


# ---- the results of a run --------------------------------------------------------------------------


def _results_of(run: RunInput, where: str) -> dict[str, DirectionResult]:
    """The results of the run by direction, in the order of `run.json`; each must be of this run."""
    record = run.record
    found: dict[str, DirectionResult] = {}
    for result in run.results:
        if result.direction in found:
            raise DataError(
                f"{where} has two results for the direction {result.direction}"
            )
        found[result.direction] = result
    ran = [entry.direction for entry in record.directions if entry.state == "run"]
    for direction in ran:
        if direction not in found:
            raise DataError(
                f"{where} has no result for a direction that ran ({direction})"
            )
    for direction in found:
        if direction not in ran:
            raise DataError(
                f"{where} has a result for a direction that did not run ({direction})"
            )
    for direction in ran:
        _require_the_result_of_this_run(found[direction], run, where)
    _require_one_robustness_block(found.values(), record, where)
    return {direction: found[direction] for direction in ran}


def _require_the_result_of_this_run(
    result: DirectionResult, run: RunInput, where: str
) -> None:
    record = run.record
    place = f"{where}, direction {result.direction}"
    if result.system != record.system.name:
        raise DataError(f"{place}: the result is of another system")
    if result.suite != record.suite.name:
        raise DataError(f"{place}: the result is of another suite")
    provenance = result.provenance
    if provenance.suite_sha256 != record.suite.sha256:
        raise DataError(
            f"{place}: the result was scored over another suite file than the run"
        )
    if provenance.outputs_sha256 != record.files[OUTPUTS_FILE]:
        raise DataError(
            f"{place}: the result was scored over other outputs than the run wrote"
        )
    if provenance.system != record.system:
        raise DataError(f"{place}: the result names another system record than the run")


def _require_one_robustness_block(
    results: Iterable[DirectionResult], record: RunRecord, where: str
) -> None:
    trios = [
        tuple(_plain(getattr(result, name)) for name in _ROBUSTNESS_FIELDS)
        for result in results
    ]
    if any(trio != trios[0] for trio in trios[1:]):
        raise DataError(
            f"the results of {where} do not carry the same robustness block"
        )
    if trios and trios[0][2] != _plain(record.robustness_source):
        raise DataError(
            f"{where}: the robustness source of the run differs from the one its "
            "results carry"
        )


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _plain(item) for key, item in value.items()}
    return value


def robustness_of(held: HeldRun) -> dict[str, Any] | None:
    """The robustness entry the results of a run carry (`block`, `coverage`, `source`), or None
    when they carry none."""
    for result in held.results.values():
        if result.robustness is not None:
            return {
                "block": _plain(result.robustness),
                "coverage": _plain(result.robustness_coverage),
                "source": _plain(result.robustness_source),
            }
    return None


# ---- the statistics ------------------------------------------------------------------------------


def verified_rows(
    held: HeldRun, settings: BootstrapSettings
) -> dict[str, DirectionRows]:
    """The statistics of every direction that ran, recomputed from the suite and the outputs of the
    run, after checking that each result is what those outputs score (with the bootstrap
    `settings` the report is made with).

    Raises:
        DataError: the outputs lack a segment, or a result's segment count, bootstrap record,
            resampling unit, point value, interval, statuses, signatures or `trained` flag is not
            what the outputs give.
    """
    where = _where(held.system, held.suite_name)
    rows: dict[str, DirectionRows] = {}
    for direction, result in held.results.items():
        view = held.suite.view(direction)
        keys = [OutputKey(direction, row_id) for row_id in view.ids]
        missing = [key for key in keys if key not in held.input.outputs]
        if missing:
            raise DataError(
                f"{where} has no output for {len(missing)} segments of {direction}"
            )
        outputs = [held.input.outputs[key] for key in keys]
        computed = direction_rows(view, outputs)
        place = f"{where}, direction {direction}"
        if result.n_segments != len(view.ids):
            raise DataError(
                f"{place}: the result does not match its outputs: the segments are "
                f"{result.n_segments} and the suite has {len(view.ids)}"
            )
        if (result.bootstrap.unit, result.bootstrap.units) != (
            computed.unit,
            len(computed.labels),
        ):
            raise DataError(
                f"{place}: the result does not match its outputs: the bootstrap unit "
                f"is {result.bootstrap.unit} over {result.bootstrap.units} units and the "
                f"outputs give {computed.unit} over {len(computed.labels)}"
            )
        _require_the_points(result, computed, place)
        _require_the_scoring_record(result, outputs, computed, settings, place)
        _require_the_neural_presence(result, held.input.record, place)
        rows[direction] = computed
    return rows


def _require_the_points(
    result: DirectionResult, computed: DirectionRows, place: str
) -> None:
    for name in COUNT_METRICS:
        counted = computed.metrics[name]
        point = score_from_statistics(counted.metric, counted.rows.sum(axis=0).tolist())
        reported = result.metrics[name]
        if reported is None or not _same_number(reported, point):
            raise DataError(
                f"{place}: the {name} of the result does not match its outputs"
            )


def _require_the_scoring_record(
    result: DirectionResult,
    outputs: Sequence[Output],
    computed: DirectionRows,
    settings: BootstrapSettings,
    place: str,
) -> None:
    """What a result says about HOW it was scored: the bootstrap record, the statuses, the
    signatures, the intervals and `trained`; none of it is believed because a digest vouches for
    the file, since a file that was edited can be re-digested."""
    bootstrap = result.bootstrap
    if (bootstrap.resamples, bootstrap.seed, bootstrap.alpha) != (
        settings.resamples,
        settings.seed,
        settings.alpha,
    ):
        raise DataError(
            f"{place}: the bootstrap of the result is not the one the report is made with"
        )
    if dict(result.statuses) != dict(Counter(output.status for output in outputs)):
        raise DataError(f"{place}: the statuses of the result do not match its outputs")
    for name in COUNT_METRICS:
        field = _SIGNATURE_FIELDS[name]
        if getattr(result, field) != computed.metrics[name].signature:
            raise DataError(
                f"{place}: the {field} of the result does not match its outputs: it may "
                "have been made with another version of sacrebleu than the one installed "
                f"on this machine ({_installed_sacrebleu()}) or may have been edited; "
                "run `rinchen eval run` again"
            )
    for name in COUNT_METRICS:
        if not _same_interval(
            result.ci95[name], _interval_of(computed.metrics[name], settings)
        ):
            raise DataError(
                f"{place}: the ci95 of the {name} of the result does not match its outputs"
            )
    if result.trained is not True:
        raise DataError(
            f"{place}: the trained flag of the result does not match its run: a "
            "direction that ran is a trained one"
        )


def _require_the_neural_presence(
    result: DirectionResult, record: RunRecord, place: str
) -> None:
    """A neural metric has a value in the result exactly when the run lists its scores, and has the
    runner's reason when it has none (the values themselves cannot be recomputed here)."""
    for name in SEGMENT_SCORE_METRICS:
        listed = name in record.segment_scores
        has_value = result.metrics[name] is not None
        if has_value and not listed:
            raise DataError(
                f"{place}: the {name} of the result has a value and the run lists no "
                "segment scores for it"
            )
        if listed and not has_value:
            raise DataError(
                f"{place}: the {name} of the result has no value and the run lists "
                "segment scores for it"
            )
        if not listed and result.unmeasured[name] != NO_SCORER_REASON:
            raise DataError(
                f"{place}: the unmeasured reason of the {name} of the result is not the "
                "one the runner writes"
            )


def _installed_sacrebleu() -> str:
    try:
        return importlib.metadata.version("sacrebleu")
    except importlib.metadata.PackageNotFoundError:
        return "not installed"


def _interval_of(
    counted: MetricRows, settings: BootstrapSettings
) -> tuple[float, float] | None:
    def score(totals: Sequence[int]) -> float:
        return score_from_statistics(counted.metric, totals)

    return estimate(counted.rows, score, settings).interval


def _same_number(reported: float, recomputed: float) -> bool:
    return math.isclose(
        reported, recomputed, rel_tol=_POINT_TOLERANCE, abs_tol=_POINT_TOLERANCE
    )


def _same_interval(
    reported: Sequence[float] | None, recomputed: Sequence[float] | None
) -> bool:
    if reported is None or recomputed is None:
        return reported is None and recomputed is None
    return len(reported) == len(recomputed) and all(
        _same_number(given, found)
        for given, found in zip(reported, recomputed, strict=True)
    )
