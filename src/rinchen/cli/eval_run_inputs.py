"""The files a run of `rinchen eval run` stands on (D13): which of the three sources of outputs it
has, every file hashed one by one BEFORE any is read (the export is vouched for first), and the
inputs that are read after the run: D12's robustness directory (its manifest must be of an
`eval_robustness` run and list the two files with the digests they have now) and the scores of
the neural seam (which must name the suite and the `hypotheses.jsonl` of this run). What is parsed is
what was hashed. The command (`rinchen.cli.eval_run`) orders the steps; this module holds the files.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final, NamedTuple

from rinchen.cli.eval_robustness_output import REPORT_FILE as ROBUSTNESS_REPORT_FILE
from rinchen.cli.eval_robustness_output import STAGE as ROBUSTNESS_STAGE
from rinchen.cli.eval_run_plan import GeneratorPlan, HypothesesPlan, ModelPlan, RunPlan
from rinchen.cli.train_common import hash_input
from rinchen.errors import ConfigError, DataError
from rinchen.eval.outputs import OutputKey
from rinchen.eval.robust_report import ROBUSTNESS_FILE
from rinchen.eval.robustness_input import RobustnessInput, read_robustness
from rinchen.eval.runner import SuiteRun
from rinchen.eval.segment_scores import read_scores
from rinchen.export.verified import VerifiedExport, verified_content, verify_export
from rinchen.manifest import store
from rinchen.manifest.model import Manifest, ManifestFile

_SUITE: Final = "suite file"
_METRICS: Final = "eval metrics config"
_GENERATORS: Final = "generators config"
_HYPOTHESES: Final = "hypotheses file"
_SCORES: Final = "scores file"
_ROBUSTNESS_PARTS: Final = (
    (ROBUSTNESS_FILE, "robustness result"),
    (ROBUSTNESS_REPORT_FILE, "robustness report"),
    (store.manifest_path(Path()).name, "robustness manifest"),
)


# ---- the sources of the outputs -------------------------------------------------------------------


class ModelSource(NamedTuple):
    plan: ModelPlan
    verified: VerifiedExport


class GeneratorSource(NamedTuple):
    plan: GeneratorPlan
    config: ManifestFile


class FileSource(NamedTuple):
    plan: HypothesesPlan
    entry: ManifestFile


Source = ModelSource | GeneratorSource | FileSource


class Hashed(NamedTuple):
    """The files of a run, each with the digest taken before it was read: the suite, the metrics
    configuration, what the system stands on (an export's files, the generators configuration or
    the hypotheses file), D12's three files and the scores files by metric."""

    suite: ManifestFile
    metrics: ManifestFile
    source: Source
    robustness: tuple[ManifestFile, ...]
    scores: dict[str, ManifestFile]


# ---- the files ----------------------------------------------------------------------------------


def hash_inputs(plan: RunPlan, suite_path: Path) -> Hashed:
    """The export is vouched for (a model), then every file is hashed one by one, none read."""
    system = plan.system
    model = (
        ModelSource(system, verify_export(system.model))
        if isinstance(system, ModelPlan)
        else None
    )
    suite = hash_input(suite_path, what=_SUITE, missing=DataError)
    metrics = hash_input(plan.metrics_config, what=_METRICS, missing=ConfigError)
    source: Source = model if model is not None else _file_source(system)
    robustness = (
        ()
        if plan.robustness is None
        else tuple(
            hash_input(plan.robustness / name, what=what, missing=DataError)
            for name, what in _ROBUSTNESS_PARTS
        )
    )
    scores = {
        metric: hash_input(path, what=_SCORES, missing=DataError)
        for metric, path in plan.segment_scores.items()
    }
    return Hashed(suite, metrics, source, robustness, scores)


def _file_source(system: ModelPlan | GeneratorPlan | HypothesesPlan) -> Source:
    if isinstance(system, GeneratorPlan):
        return GeneratorSource(
            system, hash_input(system.config, what=_GENERATORS, missing=ConfigError)
        )
    if isinstance(system, HypothesesPlan):
        return FileSource(
            system, hash_input(system.path, what=_HYPOTHESES, missing=DataError)
        )
    raise TypeError(
        "a model has no file to hash: the export is vouched for by verify_export"
    )


def entries(hashed: Hashed) -> list[ManifestFile]:
    """Every file that was hashed, in the order they are recorded."""
    return [
        hashed.suite,
        hashed.metrics,
        *system_entries(hashed.source),
        *hashed.robustness,
        *hashed.scores.values(),
    ]


def system_entries(source: Source) -> tuple[ManifestFile, ...]:
    if isinstance(source, ModelSource):
        return _export_entries(source.verified)
    if isinstance(source, GeneratorSource):
        return (source.config,)
    return (source.entry,)


def _export_entries(verified: VerifiedExport) -> tuple[ManifestFile, ...]:
    """The five files of the export and its manifest with the digests `verify_export` took."""
    files = [
        (verified.directory / name, digest) for name, digest in verified.sha256.items()
    ]
    files.append((verified.directory / "manifest.json", verified.manifest_sha256))
    return tuple(
        ManifestFile(path=str(path), sha256=digest, bytes=path.stat().st_size)
        for path, digest in files
    )


def system_name(plan: RunPlan, source: Source) -> str:
    if plan.name is not None:
        return plan.name
    if isinstance(source, ModelSource):
        return source.verified.record.name
    raise TypeError("the name of a generator or a file is settled by the plan")


# ---- the optional inputs ---------------------------------------------------------------------------------


def read_robustness_input(files: tuple[ManifestFile, ...]) -> RobustnessInput | None:
    """D12's block, coverage and digests from the hashed files of the robustness directory,
    after its manifest vouched for them. None when the option was not given.

    Raises:
        DataError: the manifest is not of an `eval_robustness` run or does not list the two
            files with the digests they have, or either file is not what D12 writes.
    """
    if not files:
        return None
    result, report, manifest = files
    document = _manifest_of(verified_content(Path(manifest.path), manifest.sha256))
    if document.stage != ROBUSTNESS_STAGE:
        raise DataError(
            f"the robustness directory's manifest is not the manifest of a {ROBUSTNESS_STAGE} run"
        )
    listed = {Path(item.path).name: item.sha256 for item in document.outputs}
    for entry in (result, report):
        name = Path(entry.path).name
        if listed.get(name) != entry.sha256:
            raise DataError(
                f"the robustness directory's manifest does not vouch for {name} as it is "
                "now: it is not listed, or it changed after the run that wrote it"
            )
    return read_robustness(
        verified_content(Path(report.path), report.sha256),
        verified_content(Path(result.path), result.sha256),
    )


def _manifest_of(data: bytes) -> Manifest:
    try:
        return Manifest.model_validate_json(data)
    except ValueError as error:
        raise DataError(
            "the manifest of the robustness directory is not a manifest of a stage"
        ) from error


def read_segment_scores(
    files: dict[str, ManifestFile], run: SuiteRun, outputs_sha256: str
) -> dict[str, dict[OutputKey, float]]:
    """The scores of each metric given, from the hashed files: they must name the suite and the
    `hypotheses.jsonl` of this run and give every segment the run scored one score."""
    expected = list(run.outputs)
    return {
        metric: read_scores(
            verified_content(Path(entry.path), entry.sha256),
            metric=metric,
            suite_sha256=run.suite_sha256,
            hypotheses_sha256=outputs_sha256,
            expected=expected,
        )
        for metric, entry in files.items()
    }
