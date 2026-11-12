"""The agreement of the two exports and the parts of the `evaluate` stage (D15, spec P8 and P9, brief G5.4).

The stage counts, over the three memorised suites, the rows on which the int8 export and the float32 one
wrote the same text, and lists each of its four parts in its record. The expected figures are counted
from the `hypotheses.jsonl` files the recording runner wrote, with code that is not the stage's; the two
recorded exports of the spike differ on two rows of one direction, and a test makes more differ.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from rinchen.tiny.states import PART_IDS, SKIP_NO_NEURAL
from tests.support import tiny_run, tiny_run_eval
from tests.support.no_network import no_network  # noqa: F401 - the fixture of the mark
from tests.support.tiny_run_eval import (
    FLOAT32,
    INT8,
    MEMORISED,
    hypotheses_of,
    run_evaluation,
    stage_dir,
)

pytestmark = pytest.mark.usefixtures("no_network")

# The fixtures of the support modules, named here so that pytest finds them.
library_outputs = tiny_run.library_outputs
model_outputs = tiny_run_eval.model_outputs
evaluated_inputs = tiny_run_eval.evaluated_inputs
eval_out = tiny_run_eval.eval_out

run = run_evaluation


def brute_agreement(out: Path) -> tuple[dict[str, tuple[int, int]], int, int]:
    """Rows and equal rows by direction, counted from the files the runner wrote."""
    by_direction: dict[str, list[int]] = {}
    for suite in MEMORISED:
        first = hypotheses_of(
            stage_dir(out) / "runs" / suite / INT8 / "hypotheses.jsonl"
        )
        second = {
            (row["direction"], row["id"]): row["text"]
            for row in hypotheses_of(
                stage_dir(out) / "runs" / suite / FLOAT32 / "hypotheses.jsonl"
            )
        }
        for row in first:
            count = by_direction.setdefault(row["direction"], [0, 0])
            count[0] += 1
            count[1] += second[(row["direction"], row["id"])] == row["text"]
    rows = sum(count[0] for count in by_direction.values())
    equal = sum(count[1] for count in by_direction.values())
    return {d: (c[0], c[1]) for d, c in by_direction.items()}, rows, equal


def test_the_agreement_counts_the_rows_where_the_two_exports_wrote_the_same_text(
    eval_out: Path,
) -> None:
    result, _ = run(eval_out)

    by_direction, rows, equal = brute_agreement(eval_out)
    agreement = result.facts["agreement"]
    assert {
        d: (c["rows"], c["equal"]) for d, c in agreement["by_direction"].items()
    } == by_direction
    assert (agreement["rows"], agreement["equal"]) == (rows, equal)
    assert agreement["share"] == equal / rows


def test_the_two_recorded_exports_differ_on_two_rows_of_one_direction(
    eval_out: Path,
) -> None:
    result, _ = run(eval_out)

    agreement = result.facts["agreement"]
    english = agreement["by_direction"]["khk_Cyrl-eng_Latn"]
    assert english["rows"] - english["equal"] == 2
    assert agreement["rows"] - agreement["equal"] == 2
    forward = agreement["by_direction"]["eng_Latn-khk_Cyrl"]
    assert forward["rows"] == forward["equal"]


def test_a_row_that_one_export_writes_differently_is_counted_in_any_suite(
    eval_out: Path,
) -> None:
    def edit(system: str, suite: str, direction: str, row_id: str, text: str) -> str:
        changed = system == INT8 and direction in (
            "rus_Cyrl-khk_Cyrl",
            "mn_Mong-eng_Latn",
        )
        return text + " changed" if changed and row_id.endswith(("1", "2")) else text

    result, _ = run(eval_out, edit=edit)

    _, rows, equal = brute_agreement(eval_out)
    agreement = result.facts["agreement"]
    assert (agreement["rows"], agreement["equal"]) == (rows, equal)
    assert rows - equal > 2
    russian = agreement["by_direction"]["rus_Cyrl-khk_Cyrl"]
    assert russian["equal"] < russian["rows"]
    mongolian = agreement["by_direction"]["mn_Mong-eng_Latn"]
    assert mongolian["equal"] < mongolian["rows"]


def test_every_part_is_listed_and_ok(eval_out: Path) -> None:
    result, _ = run(eval_out)

    assert [(part.id, part.state, part.reason) for part in result.record.parts] == [
        ("memorised", "ok", None),
        ("heldout", "ok", None),
        ("neural_seam", "ok", None),
        ("report", "ok", None),
    ]
    assert [part.id for part in result.record.parts] == list(PART_IDS["evaluate"])


def test_without_the_stand_ins_the_neural_part_is_skipped_for_its_closed_reason(
    eval_out: Path,
) -> None:
    result, _ = run(eval_out, qe_fake=False)

    states = {part.id: (part.state, part.reason) for part in result.record.parts}
    assert states == {
        "memorised": ("ok", None),
        "heldout": ("ok", None),
        "neural_seam": ("skipped", SKIP_NO_NEURAL),
        "report": ("ok", None),
    }
    assert result.record.state == "ok"
    assert result.facts["neural_seam"] is None
    assert result.record.detail["neural_seam"] is None
