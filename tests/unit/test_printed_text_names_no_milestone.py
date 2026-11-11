"""What the project prints into a file that is uploaded names no internal milestone (D13 fix wave 2,
finding m7).

The card, `report.md`, `README.md`, `release.json`, `hub_plan.json` and the result files are read by
people who know nothing of the milestones the project is built in (`D12`), the numbered steps of
its plans (`T8`, `F17`) or the open items of the master plan (`B1`, `C3`), nor of the design spike.
The reason a result gives for a neural metric that has no scores named two such items, and the notes
about the robustness block named a milestone; both now say it in words.

Two checks hold it. A scan of the string constants (not the docstrings and comments) of the modules
that write into those files, and the text itself: the golden card, the golden `report.md` and
`report.json`, and the card and `report.md` generated from the golden inputs with the sections that
carry those notes.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path
from typing import Final

import pytest

import rinchen
from rinchen.eval.report_markdown import render_markdown
from rinchen.export.record import ModelRecord
from rinchen.metrics.registry import NO_SCORER_REASON
from tests.support.model_exports import write_model_export
from tests.support.no_network import no_network  # noqa: F401 - the fixture of the mark
from tests.support.release_documents import (
    GOLDEN_EXPORT_MANIFEST_SHA256,
    GOLDEN_MODEL_SHA256,
    bound_document,
    with_robustness,
)
from tests.support.release_fixture import SIX_DIRECTIONS, card_of

pytestmark = pytest.mark.usefixtures("no_network")

PACKAGE_ROOT: Final = Path(rinchen.__file__).resolve().parent
FIXTURES: Final = Path(__file__).resolve().parents[1] / "fixtures" / "eval"
# A milestone (D1 to D15), a step of a plan (T1 to T23), a finding (F1 to F18), an open item (B1 to
# B4, C2 to C4) or the design spike: none may stand in text a stranger reads.
MILESTONE: Final = re.compile(
    r"\b(?:D\d{1,2}|T\d{1,2}|F\d{1,2}|B[1-4]|C[2-4])\b|(?i:spike)"
)
# The modules whose strings reach an uploaded file or a file a person reads: the report, the card,
# the release record and the plan, the results and the reasons they carry.
SCANNED: Final = (
    "eval/*.py",
    "export/*.py",
    "metrics/registry.py",
    "cli/eval_run*.py",
    "cli/eval_systems.py",
    "cli/release*.py",
)
EXEMPT: Final = {
    "export/verified.py": "refusal messages of the export check, shown to the operator and written nowhere",
}


def milestone_names(source: str) -> list[str]:
    """The milestone names in the string constants of `source`, docstrings left out."""
    tree = ast.parse(source)
    docstrings = {
        id(node.body[0].value)
        for node in ast.walk(tree)
        if isinstance(
            node, ast.Module | ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef
        )
        and node.body
        and isinstance(node.body[0], ast.Expr)
        and isinstance(node.body[0].value, ast.Constant)
        and isinstance(node.body[0].value.value, str)
    }
    return [
        match.group(0)
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant)
        and isinstance(node.value, str)
        and id(node) not in docstrings
        for match in MILESTONE.finditer(node.value)
    ]


# ---- the scan, tried on its own inputs --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "source",
    [
        'MESSAGE = "not recorded in D12\'s block"\n',
        'REASON = "no scorer is built (B1 and B2 are open)"\n',
        'TEXT = f"see T8 for {x}"\n',
        'TEXT = "found by the spike"\n',
        'def f():\n    return "F17"\n',
        'x = ("a " "C3")\n',
    ],
)
def test_the_scan_finds_a_milestone_in_a_string(source: str) -> None:
    assert milestone_names(source)


@pytest.mark.parametrize(
    "source",
    [
        '"""The D12 block (T8, B1)."""\nx = 1\n',
        'def f():\n    """Spike of D13."""\n    return 1\n',
        "# D12 and T8 in a comment\nx = 1\n",
        'x = "float16 and int8_float16"\n',
        'x = "the digest d12 is hexadecimal and b2c3"\n',
        'x = "ABD12 and D123 and B5 and C1"\n',
    ],
)
def test_the_scan_leaves_docstrings_comments_and_other_words_alone(
    source: str,
) -> None:
    assert milestone_names(source) == []


# ---- the code that prints --------------------------------------------------------------------------------------


def scanned_files() -> list[Path]:
    found = {
        path
        for pattern in SCANNED
        for path in sorted(PACKAGE_ROOT.glob(pattern))
        if path.relative_to(PACKAGE_ROOT).as_posix() not in EXEMPT
    }
    return sorted(found)


def test_the_scan_reads_the_modules_that_print() -> None:
    names = {path.relative_to(PACKAGE_ROOT).as_posix() for path in scanned_files()}

    assert len(names) > 60
    assert {
        "eval/report_markdown_notes.py",
        "export/model_card_notes.py",
        "export/release.py",
        "export/hub.py",
        "metrics/registry.py",
        "cli/eval_run_output.py",
        "cli/release_stage.py",
    } <= names
    assert not names & set(EXEMPT)


def test_every_exempt_module_exists_and_has_a_reason() -> None:
    for relative, reason in EXEMPT.items():
        assert (PACKAGE_ROOT / relative).is_file(), relative
        assert len(reason) > 20, relative


def test_no_string_of_the_modules_that_print_names_a_milestone() -> None:
    found = {
        path.relative_to(PACKAGE_ROOT).as_posix(): names
        for path in scanned_files()
        if (names := milestone_names(path.read_text(encoding="utf-8")))
    }

    assert found == {}


def test_the_reason_of_a_neural_metric_without_scores_is_in_words() -> None:
    assert not MILESTONE.search(NO_SCORER_REASON)
    assert "COMET" in NO_SCORER_REASON and "MetricX" in NO_SCORER_REASON


# ---- the text itself ----------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name",
    [
        "golden_card.md",
        "golden_report.md",
        "golden_report.json",
        "golden_result_alpha_khk_eng.json",
    ],
)
def test_the_golden_files_name_no_milestone(name: str) -> None:
    text = (FIXTURES / name).read_text(encoding="utf-8")

    assert not MILESTONE.findall(text)


@pytest.fixture(scope="module")
def record(tmp_path_factory: pytest.TempPathFactory) -> ModelRecord:
    return write_model_export(
        tmp_path_factory.mktemp("export") / "export", directions=SIX_DIRECTIONS
    ).record


def bound() -> dict:  # type: ignore[type-arg]
    return bound_document(GOLDEN_MODEL_SHA256, GOLDEN_EXPORT_MANIFEST_SHA256)


def test_the_card_and_report_md_of_the_golden_inputs_name_no_milestone(
    record: ModelRecord,
) -> None:
    document = bound()

    assert not MILESTONE.findall(card_of(record, document))
    assert not MILESTONE.findall(render_markdown(document))


def test_the_card_and_report_md_with_a_robustness_block_name_no_milestone(
    record: ModelRecord,
) -> None:
    document = with_robustness(bound())

    card = card_of(record, document)
    markdown = render_markdown(document)

    assert (
        "- The kind of the robustness suite is not recorded in the robustness block, so "
        "the evidence above counts only the suites of the evaluation tables.\n"
    ) in card
    assert (
        "The kind of the robustness suite is not recorded in the robustness block, so "
        "the evidence banner above does not count it."
    ) in markdown
    assert not MILESTONE.findall(card)
    assert not MILESTONE.findall(markdown)
