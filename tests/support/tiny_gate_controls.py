"""Helpers of the controls of the D15 gate (`tests/integration/test_tiny_gate_controls.py`, R7 to R12):
where the files of a finished run are, the thresholds of the pipeline replayed over the figures of a report
(so that a control changes ONE figure and the threshold that reads it is shown to fail), the rotation of a
hypotheses file and the two suites that make the placeholder and script thresholds fail.

Nothing here starts a process: the commands the controls run go through `tests/support/tiny_gate.py`.
"""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any, Final

from rinchen.config.loader import REPO_ROOT
from rinchen.config.tiny import TinyConfig, load_tiny_config
from rinchen.eval.robust_suite import write_suite
from rinchen.tiny.context import stage_dir
from rinchen.tiny.report import parse_report
from rinchen.tiny.thresholds import (
    ThresholdInputs,
    ThresholdResult,
    evaluate_thresholds,
)
from tests.support.robust_suites import (
    build_mong_output_suite,
    build_unseen_placeholder_suite,
)
from tests.support.robust_toy import build_robust_toy

__all__ = [
    "ROTATION",
    "answer_with_the_source",
    "averaged_directory",
    "converter_options",
    "export_directory",
    "judged",
    "replayed",
    "report_inputs",
    "rotate_texts",
    "shipped_config",
    "stage_path",
    "verdicts",
    "with_changes",
    "write_control_suites",
]

SHIPPED_PROFILE: Final = REPO_ROOT / "configs" / "tiny" / "tiny.yaml"
# The rotation of a hypotheses file: each text moves this many rows, so that every row is answered with
# another row's text.
ROTATION: Final = 7
# The elapsed time a replay judges: well inside the budget, so that the wall time is not what is tested.
REPLAY_SECONDS: Final = 100.0
UNSEEN_SUITE: Final = "unseen_placeholders.json"
MONG_OUTPUT_SUITE: Final = "mong_output.json"


def shipped_config() -> TinyConfig:
    """The shipped pipeline profile, loaded the way the command loads it."""
    return load_tiny_config(SHIPPED_PROFILE.read_bytes())


# ---- where the files of a run are ---------------------------------------------------------------------


def stage_path(out_dir: Path, stage_id: str, *parts: str) -> Path:
    """`parts` under the directory of the stage `stage_id` in the output directory `out_dir`."""
    directory = stage_dir(out_dir, stage_id)
    assert directory is not None, f"the stage {stage_id} writes no directory"
    return directory.joinpath(*parts)


def export_directory(out_dir: Path, quantization: str) -> Path:
    """The CTranslate2 export of one quantization (`float32`, `int8`)."""
    return stage_path(out_dir, "export", quantization)


def converter_options(out_dir: Path) -> list[str | Path]:
    """The two options that attach the run's script converter to `rinchen eval robustness`."""
    return [
        "--script-converter-ct2",
        stage_path(out_dir, "converter", "model"),
        "--script-converter-spm",
        stage_path(out_dir, "converter", "run", "converter_char.model"),
    ]


def averaged_directory(out_dir: Path) -> Path:
    """The averaged checkpoint of the one variant the run trained."""
    variants = sorted(
        p for p in stage_path(out_dir, "train", "ablate", "variants").glob("*")
    )
    assert len(variants) == 1, f"{len(variants)} variants, not one"
    return variants[0] / "run" / "averaged"


# ---- the thresholds, replayed ----------------------------------------------------------------------------


def report_inputs(out_dir: Path, *, elapsed: float = REPLAY_SECONDS) -> ThresholdInputs:
    """What the pipeline's thresholds read, taken from the `report.json` in `out_dir`: the stage
    records, the corpus chrF of each memorised direction and system, the figures of the float32 parity
    run, D12's robustness values and `elapsed` seconds."""
    data = (out_dir / "report.json").read_bytes()
    document: dict[str, Any] = json.loads(data)
    chrf: dict[str, dict[str, float]] = {}
    for row in document["blocks"]["memorised"]:
        chrf.setdefault(row["direction"], {})[row["system"]] = row["chrf"]
    parity = document["blocks"]["parity"]
    return ThresholdInputs(
        stages=parse_report(data).stages,
        memorised_chrf=chrf,
        parity={
            "exact_share": parity["exact_share"],
            "mean_abs_delta": parity["mean_abs_delta"],
        },
        robustness=document["blocks"]["robustness"]["values"],
        elapsed_seconds=elapsed,
    )


def with_changes(inputs: ThresholdInputs, **changes: Any) -> ThresholdInputs:
    """`inputs` with the named fields replaced (a control changes one figure)."""
    return replace(inputs, **changes)


def judged(inputs: ThresholdInputs) -> tuple[ThresholdResult, ...]:
    """The pipeline's eight thresholds over `inputs`, with the bounds and the budget of the shipped
    profile."""
    config = shipped_config()
    return evaluate_thresholds(inputs, config.thresholds, config.budget_seconds)


def replayed(out_dir: Path) -> tuple[ThresholdResult, ...]:
    """The thresholds over the figures of the report in `out_dir`."""
    return judged(report_inputs(out_dir))


def verdicts(results: Iterable[ThresholdResult]) -> dict[str, str]:
    """`{threshold id: verdict}` of a set of results."""
    return {result.id: result.verdict for result in results}


# ---- the files of the controls ------------------------------------------------------------------------------


def rotate_texts(path_in: Path, path_out: Path, k: int = ROTATION) -> None:
    """Write the hypotheses file `path_in` (`rinchen eval run` writes one) to `path_out` with the texts of
    each direction moved by `k` rows: inside a direction the line `i` gets the text of the line `i + k`
    (the last lines take the first ones). The ids and the statuses stay where they are, so every row is
    still answered; only what the answer says is another row's."""
    lines: Sequence[dict[str, Any]] = [
        json.loads(line) for line in path_in.read_text("utf-8").splitlines()
    ]
    members: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for line in lines:
        members[line["direction"]].append(line)
    for group in members.values():
        texts = [line["text"] for line in group]
        for position, line in enumerate(group):
            line["text"] = texts[(position + k) % len(group)]
    path_out.write_text(
        "".join(
            json.dumps(line, sort_keys=True, ensure_ascii=False) + "\n"
            for line in lines
        ),
        encoding="utf-8",
        newline="\n",
    )


def answer_with_the_source(
    suite_path: Path, hypotheses_in: Path, hypotheses_out: Path, *, section: str
) -> int:
    """Write the robustness hypotheses file `hypotheses_in` to `hypotheses_out` with the text of every
    line of `section` replaced by the SOURCE of its row in the suite: a system that copies its input,
    which writes the wrong script for every row whose source is not in the script of its target. The
    other lines are as they were. Returns how many lines changed.

    Raises:
        KeyError: a line of the section names a row the suite does not have.
    """
    suite = json.loads(suite_path.read_text(encoding="utf-8"))
    sources = {row["id"]: row["source"] for row in suite["sections"][section]}
    lines = [json.loads(line) for line in hypotheses_in.read_text("utf-8").splitlines()]
    changed = 0
    for line in lines:
        if line["section"] == section:
            line["text"] = sources[line["id"]]
            changed += 1
    hypotheses_out.write_text(
        "".join(
            json.dumps(line, sort_keys=True, ensure_ascii=False) + "\n"
            for line in lines
        ),
        encoding="utf-8",
        newline="\n",
    )
    return changed


def write_control_suites(directory: Path) -> tuple[Path, Path]:
    """Write the two robustness suites of the controls into `directory` (made): the placeholders no
    row of a model's training holds (indices of ten and up) and the Khalkha output asked in the
    traditional script. Returns the two paths."""
    directory.mkdir(parents=True, exist_ok=True)
    unseen = directory / UNSEEN_SUITE
    mong_output = directory / MONG_OUTPUT_SUITE
    write_suite(build_unseen_placeholder_suite(), unseen)
    write_suite(build_mong_output_suite(build_robust_toy()), mong_output)
    return unseen, mong_output
