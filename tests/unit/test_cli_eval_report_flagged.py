"""A failure a system flagged stays visible in `report.md` through the real commands (D13 fix wave 1,
I3): a hypotheses file whose texts carry the statuses `empty`, `truncated` and `no_output` is run
with `rinchen eval run --hypotheses` and reported with `rinchen eval report`, and `report.md` names
the counts beside the metric of every table (`report.json` keeps them in `statuses`, unchanged).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from rinchen.eval.outputs import Output, OutputKey, outputs_to_bytes
from tests.support.eval_cli import invoke
from tests.support.eval_fakes import golden_outputs
from tests.support.eval_report_cli import (
    FIRST,
    SECOND,
    ReportWorkspace,
    invoke_report,
    make_report_workspace,
)
from tests.support.no_network import no_network  # noqa: F401 - the fixture of the mark

pytestmark = pytest.mark.usefixtures("no_network")


@pytest.fixture
def workspace(tmp_path: Path) -> ReportWorkspace:
    return make_report_workspace(tmp_path)


def flagged_outputs_file(directory: Path) -> Path:
    """`alpha`'s hypotheses with one empty text, two `no_output` and one truncated in the first
    direction."""
    outputs = golden_outputs("alpha")
    for row_id, text, status in (
        ("g01", "", "empty"),
        ("g02", "", "no_output"),
        ("g03", "", "no_output"),
        ("g04", "cut off", "truncated"),
    ):
        outputs[OutputKey(FIRST, row_id)] = Output(text, status)  # type: ignore[arg-type]
    path = directory / "alpha.jsonl"
    path.write_bytes(outputs_to_bytes(outputs))
    return path


def test_flagged_statuses_run_and_report_through_the_real_commands(
    workspace: ReportWorkspace, tmp_path: Path
) -> None:
    flagged = flagged_outputs_file(tmp_path)
    again = invoke(
        workspace.ws.hypotheses_args(hypotheses=flagged, out_dir=workspace.ws.out_dir)
    )
    assert again.exit_code == 0, again.output
    assert f"{FIRST}: chrF++ " in again.stdout
    assert "(10 segments, 6 ok, 1 empty, 2 no_output, 1 truncated)" in again.stdout

    made = invoke_report(workspace)

    assert made.exit_code == 0, made.output
    markdown = (workspace.report_out / "report.md").read_text("utf-8")
    for heading in ("chrF++", "chrF", "BLEU"):
        start = markdown.index(f"#### {heading}\n")
        table = markdown[start : markdown.index("\n\n", start + 20 + len(heading))]
        first = next(
            line for line in table.splitlines() if line.startswith(f"| {FIRST} |")
        )
        second = next(
            line for line in table.splitlines() if line.startswith(f"| {SECOND} |")
        )
        assert "| 6 of 10 (1 empty, 2 no_output, 1 truncated) |" in first
        assert first.endswith("| 10 of 10 |")
        assert second.count("10 of 10") == 2
    document = json.loads((workspace.report_out / "report.json").read_text("utf-8"))
    statuses = next(
        r["statuses"]
        for r in document["results"]
        if (r["system"], r["direction"]) == ("alpha", FIRST)
    )
    assert statuses == {"empty": 1, "no_output": 2, "ok": 6, "truncated": 1}
