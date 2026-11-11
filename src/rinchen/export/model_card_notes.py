"""The sections of the model card that state facts about the limits of the figures and where they come
from (D13, spec T16): `limits_section` and `provenance_section`.

The limits are a list of facts and claim nothing: which directions are evaluated, that a dev split is
for model selection only, what the intervals resample, that nothing is corrected for multiple
comparisons, and what was not measured. The provenance names the package version, the checkpoint the
export came from (its kind, step and digests, which the export's record states and the card cannot
prove) and the digests of the export's manifest and of the report, each in full: a card holds no path.
"""

from __future__ import annotations

from collections.abc import Sequence

from rinchen.eval.report_markdown_cells import LABELS
from rinchen.eval.report_read import ReportView
from rinchen.eval.result import DirectionResult
from rinchen.export.model_card_evaluation import unscored_neural_metrics
from rinchen.export.record import ModelRecord

__all__ = ["limits_section", "provenance_section"]


def limits_section(report: ReportView, results: Sequence[DirectionResult]) -> list[str]:
    """The limits of the figures of the card."""
    return [
        "## Limits",
        "",
        (
            "- The model is trained for the directions listed above and no others. Only the "
            "directions in the tables above are measured; a pair the model was not trained for "
            "(zero-shot) is never evaluated, and nothing here says that its output is in the "
            "language asked for."
        ),
        (
            "- A dev split is used for model selection only: the report refuses a dev run, so "
            "every figure here is from a devtest or test split."
        ),
        (
            "- Each interval resamples the unit its table names: a document when every row of "
            "a direction has a document id, a segment otherwise."
        ),
        f"- {report.note_multiple_comparisons}. Read a difference with its p-value.",
        f"- Not measured: {_not_measured(results)}.",
        "",
    ]


def _not_measured(results: Sequence[DirectionResult]) -> str:
    unscored = [LABELS[metric] for metric in unscored_neural_metrics(results)]
    parts = [
        *([f"{_names(unscored)} (no scorer is built)"] if unscored else []),
        "the share of outputs in the wrong language (no language identification is run)",
        "any run on a GPU (none was made)",
        "real-world input beyond the suites above",
    ]
    return f"{', '.join(parts[:-1])} and {parts[-1]}"


def _names(names: Sequence[str]) -> str:
    """`A`, `A and B`, `A, B and C`."""
    if len(names) < 2:
        return "".join(names)
    return f"{', '.join(names[:-1])} and {names[-1]}"


def provenance_section(
    record: ModelRecord, *, report_sha256: str, export_manifest_sha256: str
) -> list[str]:
    """Where the model and the figures come from, with every digest in full."""
    checkpoint = record.checkpoint
    step = (
        str(checkpoint.step)
        if checkpoint.step is not None
        else "n/a (an averaged model has no step)"
    )
    return [
        "## Provenance",
        "",
        f"- rinchen_version: {record.rinchen_version}",
        f"- checkpoint kind: {checkpoint.kind}",
        f"- checkpoint step: {step}",
        f"- checkpoint manifest stage: {checkpoint.manifest_stage}",
        f"- checkpoint weights sha256: {checkpoint.weights_sha256}",
        f"- checkpoint config sha256: {checkpoint.config_sha256}",
        f"- checkpoint vocabulary sha256: {checkpoint.vocab_sha256}",
        f"- checkpoint manifest sha256: {checkpoint.manifest_sha256}",
        f"- export manifest sha256: {export_manifest_sha256}",
        f"- report.json sha256: {report_sha256}",
    ]
