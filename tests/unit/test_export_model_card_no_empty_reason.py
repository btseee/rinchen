"""The card never prints `n/a ()` (D13 fix wave 2, finding m6).

A suite in which the system has a run and no result (every direction `not_trained`, say) printed
`COMET-22, MetricX-24, MetricX-24-QE: n/a ().` with an empty reason, because the reasons were read
off the results and there were none. Now the reason is the real one when a result has it, `no
result in this suite` when none does, and `no reason recorded` when the text of a reason is blank.
`report.md` is held to the same rule. The unit tests name each branch; a grid over the cases that
combine them and the golden files prove that no empty reason is printed anywhere.
"""

from __future__ import annotations

import copy
import itertools
import re
from pathlib import Path
from typing import Any, Final

import pytest

from rinchen.eval.report_markdown import render_markdown
from rinchen.export.record import ModelRecord
from tests.support.model_exports import write_model_export
from tests.support.no_network import no_network  # noqa: F401 - the fixture of the mark
from tests.support.release_documents import (
    GOLDEN_EXPORT_MANIFEST_SHA256,
    GOLDEN_MODEL_SHA256,
    SYSTEM,
    bound_document,
    with_comet_scores,
    with_the_note_of,
)
from tests.support.release_fixture import SIX_DIRECTIONS, card_of

pytestmark = pytest.mark.usefixtures("no_network")

FIXTURES: Final = Path(__file__).resolve().parents[1] / "fixtures" / "eval"
EMPTY_REASON: Final = re.compile(r"n/a \(\s*\)")
NEURAL_LINE: Final = "COMET-22, MetricX-24, MetricX-24-QE: n/a ("
NO_RESULT: Final = "no result in this suite"
NO_REASON: Final = "no reason recorded"
NO_SCORER: Final = (
    "no segment scores were given: the COMET and MetricX scorers are not built"
)
SECOND_SUITE: Final = "golden-second"
FIRST: Final = "khk_Cyrl-eng_Latn"


@pytest.fixture(scope="module")
def record(tmp_path_factory: pytest.TempPathFactory) -> ModelRecord:
    return write_model_export(
        tmp_path_factory.mktemp("export") / "export", directions=SIX_DIRECTIONS
    ).record


def bound() -> dict[str, Any]:
    return bound_document(GOLDEN_MODEL_SHA256, GOLDEN_EXPORT_MANIFEST_SHA256)


def with_a_run_without_results(document: dict[str, Any]) -> dict[str, Any]:
    """A second suite in which `alpha` has a run and every direction is `not_trained`: a coverage
    row for each and no result."""
    other = copy.deepcopy(document["suites"][0])
    other.update(name=SECOND_SUITE, sha256="8" * 64)
    document["suites"].append(other)
    document["runs"].append(
        {"suite": SECOND_SUITE, "system": SYSTEM, "run_sha256": "9" * 64}
    )
    for entry in other["directions"]:
        if entry["rows"] is not None:
            document["coverage"].append(
                {
                    "suite": SECOND_SUITE,
                    "system": SYSTEM,
                    "direction": entry["direction"],
                    "state": "not_trained",
                    "reason": "the model was not trained for this direction",
                }
            )
    return with_the_note_of(document)


def with_the_reason(document: dict[str, Any], text: str) -> dict[str, Any]:
    """Every result of `alpha` with this `text` as the reason of every unmeasured neural metric."""
    for result in document["results"]:
        if result["system"] == SYSTEM:
            for metric in result["unmeasured"]:
                result["unmeasured"][metric] = text
    return document


def block(card: str, suite: str) -> list[str]:
    """The lines of the card from the heading of `suite` to the next heading."""
    lines = card.split("\n")
    start = lines.index(f"### Suite {suite}")
    end = next(
        (i for i in range(start + 1, len(lines)) if lines[i].startswith("#")),
        len(lines),
    )
    return lines[start:end]


# ---- the branches of the line for the metrics without scores -----------------------------------------


def test_a_suite_with_a_run_and_no_result_says_there_is_no_result(
    record: ModelRecord,
) -> None:
    card = card_of(record, with_a_run_without_results(bound()))

    assert f"{NEURAL_LINE}{NO_RESULT})." in block(card, SECOND_SUITE)
    assert not EMPTY_REASON.search(card)


def test_the_first_suite_of_that_report_keeps_the_real_reason(
    record: ModelRecord,
) -> None:
    card = card_of(record, with_a_run_without_results(bound()))

    assert f"{NEURAL_LINE}{NO_SCORER})." in block(card, "golden")


def test_results_that_share_a_reason_give_that_reason(record: ModelRecord) -> None:
    card = card_of(record, with_the_reason(bound(), "the scorer was not run"))

    assert f"{NEURAL_LINE}the scorer was not run)." in card


def test_results_with_two_reasons_give_both(record: ModelRecord) -> None:
    document = bound()
    results = [r for r in document["results"] if r["system"] == SYSTEM]
    results[0]["unmeasured"]["comet22"] = "reason one"
    results[1]["unmeasured"]["comet22"] = "reason two"

    card = card_of(record, document)

    assert "COMET-22: n/a (reason one; reason two)." in card


def test_a_blank_reason_is_said_to_be_blank(record: ModelRecord) -> None:
    card = card_of(record, with_the_reason(bound(), "   "))

    assert f"{NEURAL_LINE}{NO_REASON})." in card
    assert f"n/a ({NO_REASON})" in card
    assert not EMPTY_REASON.search(card)


def test_a_blank_reason_in_the_cell_of_a_metric_scored_elsewhere_is_said_to_be_blank(
    record: ModelRecord,
) -> None:
    # COMET-22 is scored in one direction only, so its column exists and the other direction's
    # cell carries the reason, which is blank.
    document = with_the_reason(with_comet_scores(bound(), direction=FIRST), "   ")

    card = card_of(record, document)

    assert f"n/a ({NO_REASON}) |" in card
    assert not EMPTY_REASON.search(card)


def test_a_blank_basis_of_a_target_is_said_to_be_blank(record: ModelRecord) -> None:
    document = bound()
    document["targets"][0]["basis"] = "   "

    card = card_of(record, document)

    assert f"n/a ({NO_REASON})" in card
    assert not EMPTY_REASON.search(card)


def test_a_blank_basis_beside_the_value_of_a_target_is_said_to_be_blank(
    record: ModelRecord,
) -> None:
    document = bound()
    document["targets"][2].update(value=0.9291, status="not_met", basis="   ")

    card = card_of(record, document)

    assert f"| 0.9291 ({NO_REASON}) |" in card
    assert not EMPTY_REASON.search(card)


def test_report_md_says_a_blank_reason_is_blank() -> None:
    markdown = render_markdown(with_the_reason(bound(), "   "))

    assert f"- comet22: {NO_REASON} (2 of 4 results)" in markdown
    assert not EMPTY_REASON.search(markdown)


# ---- nowhere does an empty reason appear -----------------------------------------------------------


@pytest.mark.parametrize("name", ["golden_card.md", "golden_report.md"])
def test_the_golden_files_hold_no_empty_reason(name: str) -> None:
    assert not EMPTY_REASON.search((FIXTURES / name).read_text(encoding="utf-8"))


GRID = list(
    itertools.product(
        (False, True),  # a second suite with a run and no result
        (
            None,
            "all",
            "first",
        ),  # COMET-22 scored in no direction, in all, in the first only
        (None, "   ", "a real reason", "two\nlines"),  # the reason text, None: as is
        (None, "   "),  # the basis of a target, None: as is
    )
)


@pytest.mark.parametrize(("second", "scored", "text", "basis"), GRID)
def test_no_card_and_no_report_md_of_a_stage_of_the_grid_prints_an_empty_reason(
    record: ModelRecord,
    second: bool,
    scored: str | None,
    text: str | None,
    basis: str | None,
) -> None:
    document = bound()
    if second:
        document = with_a_run_without_results(document)
    if scored is not None:
        document = with_comet_scores(
            document, direction=None if scored == "all" else FIRST
        )
    if text is not None:
        document = with_the_reason(document, text)
    if basis is not None:
        document["targets"][0]["basis"] = basis

    card = card_of(record, document)
    markdown = render_markdown(document)

    assert not EMPTY_REASON.search(card)
    assert not EMPTY_REASON.search(markdown)
