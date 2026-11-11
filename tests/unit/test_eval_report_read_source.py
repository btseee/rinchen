"""The report reader holds `dataset` and `revision` to the patterns of the suite reader, and the
cells that print them escape (D13 fix wave 2, finding m2).

A suite file can only hold a dataset id of the Hub and a 40-digit hexadecimal revision, so a report
built from suites can hold nothing else. The reader accepted any text there, so a hand-edited and
re-digested `report.json` could carry a line break and a heading into `report.md` and into the card.
Now the reader refuses such a text (a `DataError`, exit 3 of `release stage`), the patterns it uses
are pinned equal to the suite reader's, and the cells of a table and a listing escape what they
print, so that a view that was never read still cannot break the layout.
"""

from __future__ import annotations

import copy
import json
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from rinchen.errors import DataError
from rinchen.eval import suite_rules
from rinchen.eval.report import build_report, report_to_bytes
from rinchen.eval.report_markdown import render_markdown
from rinchen.eval.report_markdown_cells import source_cells
from rinchen.eval.report_read import report_from_bytes
from rinchen.eval.report_read_rules import DATASET_PATTERN, REVISION_PATTERN
from rinchen.export.model_card import render_card
from rinchen.export.record import ModelRecord
from tests.support.eval_report import golden_runs, golden_suites, shipped_config
from tests.support.model_exports import write_model_export
from tests.support.no_network import no_network  # noqa: F401 - the fixture of the mark
from tests.support.release_documents import (
    GOLDEN_EXPORT_MANIFEST_SHA256,
    GOLDEN_MODEL_SHA256,
    GOLDEN_REPORT_SHA256,
    SYSTEM,
    bound_document,
    view_of,
)
from tests.support.release_fixture import (
    REPO_ID,
    SIX_DIRECTIONS,
    Release,
    another_report_dir,
    fabricated_release,
    golden_facts,
    invoke_stage,
)

pytestmark = pytest.mark.usefixtures("no_network")

REVISION = "0123456789abcdef0123456789abcdef01234567"
# The forgery of the re-review: a cell that ends its row and opens a heading of the card.
INJECTED = "evil | x\n\n## Injected heading\n\nBuy now"
MARKER = "SECRET MARKER 41"


def golden_document() -> dict[str, Any]:
    return build_report(
        golden_runs(), golden_suites(), baseline="alpha", config=shipped_config()
    )


def with_source(**source: Any) -> bytes:
    document = copy.deepcopy(golden_document())
    document["suites"][0]["source"].update(source)
    return report_to_bytes(document)


def refusal(data: bytes) -> str:
    with pytest.raises(DataError) as caught:
        report_from_bytes(data)
    message = str(caught.value)
    assert message.startswith("not an evaluation report: ")
    return message


# ---- the patterns are the suite reader's -----------------------------------------------------------


def test_the_patterns_of_the_report_reader_are_those_of_the_suite_reader() -> None:
    # The reader keeps its own copy so that a card never needs the module that reads a suite.
    assert DATASET_PATTERN.pattern == suite_rules._DATASET.pattern
    assert DATASET_PATTERN.flags == suite_rules._DATASET.flags
    assert REVISION_PATTERN.pattern == suite_rules._REVISION.pattern
    assert REVISION_PATTERN.flags == suite_rules._REVISION.flags


# ---- what the reader keeps ----------------------------------------------------------------------------


@pytest.mark.parametrize(
    "dataset", ["facebook/flores", "a.b_c-d/e", "x", "d" * 200], ids=lambda v: v[:12]
)
def test_a_dataset_id_of_the_hub_is_read(dataset: str) -> None:
    view = report_from_bytes(with_source(dataset=dataset))

    assert view.suites[0].source.dataset == dataset


def test_a_pinned_revision_is_read() -> None:
    view = report_from_bytes(
        with_source(dataset="org/set", revision=REVISION, pinned=True)
    )

    assert view.suites[0].source.revision == REVISION


def test_a_suite_without_a_source_record_is_read() -> None:
    source = (
        report_from_bytes(with_source(dataset=None, revision=None)).suites[0].source
    )

    assert (source.dataset, source.revision) == (None, None)


# ---- what the reader refuses --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "dataset",
    [
        INJECTED,
        "a|b",
        "a b",
        "a\nb",
        "a#b",
        "a*b",
        "",
        "d" * 201,
        5,
        ["a"],
        True,
    ],
    ids=[
        "injected",
        "pipe",
        "space",
        "newline",
        "hash",
        "star",
        "empty",
        "too-long",
        "number",
        "list",
        "boolean",
    ],
)
def test_a_dataset_that_is_not_a_dataset_id_is_refused_naming_the_field(
    dataset: Any,
) -> None:
    message = refusal(with_source(dataset=dataset))

    assert "suites.0.source.dataset" in message


@pytest.mark.parametrize(
    "revision",
    [
        INJECTED,
        REVISION.upper(),
        REVISION[:39],
        REVISION + "0",
        REVISION[:39] + "g",
        REVISION[:39] + "|",
        "",
        5,
        ["a"],
    ],
    ids=[
        "injected",
        "uppercase",
        "short",
        "long",
        "not-hex",
        "pipe",
        "empty",
        "number",
        "list",
    ],
)
def test_a_revision_that_is_not_forty_hex_digits_is_refused_naming_the_field(
    revision: Any,
) -> None:
    message = refusal(with_source(dataset="org/set", revision=revision))

    assert "suites.0.source.revision" in message


def test_a_refusal_never_quotes_the_text_it_refuses() -> None:
    for source in ({"dataset": f"{MARKER} |"}, {"revision": f"{MARKER} |"}):
        assert MARKER not in refusal(with_source(**source))


# ---- the cells escape --------------------------------------------------------------------------------


def test_the_cells_of_a_source_escape_a_separator_and_a_line_break() -> None:
    # A value like this cannot come from a suite file; the cells hold the line all the same.
    assert source_cells(INJECTED, "x|y", True) == (
        "evil \\| x ## Injected heading Buy now",
        "x\\|y",
    )


def test_an_unpinned_revision_cell_escapes_too() -> None:
    assert source_cells("a", "x|y", False) == ("a", "x\\|y (unpinned)")


def test_report_md_of_a_view_that_was_never_read_has_no_injected_heading() -> None:
    document = copy.deepcopy(golden_document())
    document["suites"][0]["source"]["dataset"] = INJECTED

    lines = render_markdown(document).split("\n")

    assert not any(line.startswith("## Injected") for line in lines)
    assert not any(line.strip() == "Buy now" for line in lines)


@pytest.fixture(scope="module")
def record(tmp_path_factory: pytest.TempPathFactory) -> ModelRecord:
    return write_model_export(
        tmp_path_factory.mktemp("export") / "export", directions=SIX_DIRECTIONS
    ).record


def test_the_card_of_a_view_that_was_never_read_has_no_injected_heading(
    record: ModelRecord,
) -> None:
    view = view_of(bound_document(GOLDEN_MODEL_SHA256, GOLDEN_EXPORT_MANIFEST_SHA256))
    suite = view.suites[0]
    forged = replace(
        view,
        suites=(
            replace(
                suite,
                source=replace(
                    suite.source, dataset=INJECTED, revision="x|y", pinned=True
                ),
            ),
        ),
    )

    card = render_card(
        record,
        forged,
        SYSTEM,
        golden_facts(),
        REPO_ID,
        report_sha256=GOLDEN_REPORT_SHA256,
        export_manifest_sha256=GOLDEN_EXPORT_MANIFEST_SHA256,
    )

    lines = card.split("\n")
    assert not any(line.startswith("## Injected") for line in lines)
    assert "- dataset: evil \\| x ## Injected heading Buy now" in lines
    assert "- revision: x\\|y" in lines


# ---- the stage refuses the forged report -------------------------------------------------------------


@pytest.fixture
def release(tmp_path: Path) -> Release:
    return fabricated_release(tmp_path)


def forged_report_dir(
    release: Release, change: Callable[[dict[str, Any]], None]
) -> Path:
    """A report directory whose manifest vouches for a hand-edited `report.json` and for the
    `report.md` that edit renders to: every digest holds, only the strict reader can tell."""
    document = json.loads(release.report_bytes)
    change(document)
    return another_report_dir(release, document, "forged-report")


@pytest.mark.parametrize("field", ["dataset", "revision"])
def test_the_stage_refuses_a_forged_source_with_3_and_makes_nothing(
    release: Release, field: str
) -> None:
    other = forged_report_dir(
        release, lambda d: d["suites"][0]["source"].update({field: INJECTED})
    )

    result = invoke_stage(release, report_dir=other)

    assert result.exit_code == 3, result.output
    assert result.stderr.startswith("error: ")
    assert f"suites.0.source.{field}" in result.stderr
    assert "Injected" not in result.output
    assert not release.stage.exists()
