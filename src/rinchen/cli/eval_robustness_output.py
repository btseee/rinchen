"""What `rinchen eval robustness` prints and writes (D12): the lines about each section, the
metrics of the manifest, the result files and the commit that ends with the manifest. The order
of the steps before it, the hashing and the exit codes are the command's
(`rinchen.cli.eval_robustness`); the scoring, `robustness.json` and `report.json` are
`rinchen.eval`'s.

A section line says the headline of the section and what it rests on (`numbers: 0.9323 (895/960),
excluded 0`); a section in which nothing was scored says `n/a (no scored row)` in place of the
figure and keeps the rest of the line, so that the exclusions stay visible.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Final, NamedTuple

from rinchen.cli.train_common import create_out_dir, exit_cannot_write, write_manifest
from rinchen.eval.robust_hypotheses import Hypothesis, write_hypotheses
from rinchen.eval.robust_outcomes import SectionResult
from rinchen.eval.robust_report import (
    ROBUSTNESS_FILE,
    report_bytes,
    robustness_report,
)
from rinchen.eval.robust_suite import RequestKey
from rinchen.eval.robustness import RobustnessResult, robustness_block
from rinchen.manifest.model import ManifestFile
from rinchen.manifest.stage import begin_commit, output_file
from rinchen.utils.part_file import replace_file

STAGE: Final = "eval_robustness"
REPORT_FILE: Final = "report.json"
HYPOTHESES_FILE: Final = "hypotheses.jsonl"
# The files a run writes, whatever the mode: an earlier run's are removed before the first.
RESULT_FILES: Final = (HYPOTHESES_FILE, ROBUSTNESS_FILE, REPORT_FILE)
_OUTPUTS: Final = "robustness result"
_TOOL_PACKAGES: Final = ("ctranslate2", "sentencepiece", "sacrebleu", "regex")
# Nothing random happens in a measurement.
_NO_SEED: Final = -1
_NOTHING_SCORED: Final = "n/a (no scored row)"


class Measured(NamedTuple):
    """What a run came to and the commit writes: the `result`, the `hypotheses` to write (a
    model run's; None for a run that read them), the `config` the manifest's digest stands on,
    its `status` and its `inputs` in order, each with the digest taken before it was read."""

    result: RobustnessResult
    hypotheses: Mapping[RequestKey, Hypothesis] | None
    config: Mapping[str, str | int | bool]
    status: Mapping[str, str]
    inputs: list[ManifestFile]


def commit(target: Path, measured: Measured) -> Path:
    """Write the results of `measured` into `target` (made when missing) and, last, the
    manifest, and return the manifest's path. The old manifest and results go first, so that
    a failure leaves no manifest. Exits with code 1 when anything cannot be written.

    Raises:
        DataError: another stage's manifest is in `target`, or a hypothesis holds a lone
            surrogate.
    """
    create_out_dir(target, what=_OUTPUTS)
    try:
        begin_commit(target, STAGE)
        outputs = write_results(target, measured.result, measured.hypotheses)
        return write_manifest(
            target,
            stage=STAGE,
            config_sha256=config_sha256(measured.config),
            seed=_NO_SEED,
            inputs=measured.inputs,
            outputs=outputs,
            metrics=metrics_of(measured.result),
            packages=_TOOL_PACKAGES,
            status=measured.status,
        )
    except OSError as exc:
        exit_cannot_write(target, exc, what=_OUTPUTS)


def config_sha256(config: Mapping[str, str | int | bool]) -> str:
    """The run has no config file, so its options stand in for one: the SHA-256 of their
    canonical JSON (sorted keys, no spaces)."""
    canonical = json.dumps(dict(config), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def result_lines(result: RobustnessResult) -> list[str]:
    """`suite: <name>`, `system: <name>` and one line per section the suite has."""
    return [
        f"suite: {result.suite_name}",
        f"system: {result.system.name}",
        *(_LINES[name](name, section) for name, section in result.sections.items()),
    ]


def metrics_of(result: RobustnessResult) -> dict[str, float]:
    """The metrics of the manifest: the values of the six keys that are not null and the
    number of rows of each section of the suite (`rows.<section>`)."""
    block = robustness_block(result)
    return {
        **{key: value for key, value in block.items() if value is not None},
        **{f"rows.{name}": float(count) for name, count in result.suite_rows.items()},
    }


def write_results(
    target: Path,
    result: RobustnessResult,
    hypotheses: Mapping[RequestKey, Hypothesis] | None,
) -> list[ManifestFile]:
    """Write `hypotheses.jsonl` (when `hypotheses` is given), `robustness.json` and
    `report.json` into `target`, in that order, and return the manifest entries of what was
    written. The files of an earlier run go first, so that none of them outlives a run that
    does not write it.

    Raises:
        OSError: a file cannot be removed or written.
        DataError: a hypothesis holds a lone surrogate.
    """
    for name in RESULT_FILES:
        (target / name).unlink(missing_ok=True)
    outputs: list[ManifestFile] = []
    if hypotheses is not None:
        path = target / HYPOTHESES_FILE
        write_hypotheses(hypotheses, path)
        outputs.append(output_file(path, len(hypotheses)))
    result_path = target / ROBUSTNESS_FILE
    result_bytes = result.to_bytes()
    replace_file(result_path, result_bytes)
    outputs.append(output_file(result_path, None))
    report_path = target / REPORT_FILE
    report = robustness_report(result, hashlib.sha256(result_bytes).hexdigest())
    replace_file(report_path, report_bytes(report))
    outputs.append(output_file(report_path, None))
    return outputs


# ---- the lines of the sections ---------------------------------------------------------------


def _headline(section: SectionResult, detail: str) -> str:
    if section.value is None:
        return _NOTHING_SCORED
    return f"{section.value:.4f} ({detail})"


def _excluded(section: SectionResult) -> str:
    return f", excluded {sum(section.excluded.values())}"


def _mentions(section: SectionResult) -> str:
    return f"{section.ok}/{section.scored} mentions"


def _accuracy_line(name: str, section: SectionResult) -> str:
    """Numbers and placeholders: accuracy over the rows scored."""
    detail = f"{section.ok}/{section.scored}"
    return f"{name}: {_headline(section, detail)}{_excluded(section)}"


def _entities_line(name: str, section: SectionResult) -> str:
    return f"{name}: {_headline(section, _mentions(section))}"


def _terms_line(name: str, section: SectionResult) -> str:
    return f"{name}: {_headline(section, _mentions(section))}{_excluded(section)}"


def _script_line(name: str, section: SectionResult) -> str:
    detail = f"mean of {section.scored} rows"
    below_floor = int(section.details["below_floor"] or 0)
    return f"{name}: {_headline(section, detail)}, below floor {below_floor}"


def _mong_line(name: str, section: SectionResult) -> str:
    detail = f"retention over {section.rows} rows"
    return f"{name}: {_headline(section, detail)}"


_LINES: Final[dict[str, Callable[[str, SectionResult], str]]] = {
    "numbers": _accuracy_line,
    "placeholders": _accuracy_line,
    "entities": _entities_line,
    "terms": _terms_line,
    "script": _script_line,
    "mong": _mong_line,
}
