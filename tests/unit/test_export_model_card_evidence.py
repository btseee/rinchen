"""The evidence of a card is the evidence of the system it describes (D13 fix wave 1, I2).

A report can hold the runs of several systems over suites of several kinds. The banner of the card,
its `toy` tag and the `evidence` of `release.json` come from the kinds of the suites in which THE
CARDED SYSTEM has a result (a direction that ran), not from the whole report: a system whose every
table is from a fixture suite gets the fixture banner and the `toy` tag even when another system of
the report was also measured on a benchmark. The report's own evidence (its banner, its
`evidence` section) stays what it is.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import yaml

from rinchen.eval.report_markdown import FIXTURE_BANNER
from rinchen.export.record import ModelRecord
from rinchen.export.release import release_record
from tests.support.model_exports import write_model_export
from tests.support.no_network import no_network  # noqa: F401 - the fixture of the mark
from tests.support.release_documents import (
    GOLDEN_EXPORT_MANIFEST_SHA256,
    GOLDEN_MODEL_SHA256,
    GOLDEN_REPORT_SHA256,
    OTHER,
    REPO_ID,
    SYSTEM,
    bound_document,
    view_of,
    with_a_run_in_another_suite,
    with_the_note_of,
)
from tests.support.release_fixture import (
    SIX_DIRECTIONS,
    another_report_dir,
    card_of,
    fabricated_release,
    golden_facts,
    invoke_stage,
)

pytestmark = pytest.mark.usefixtures("no_network")

A_ONLY_IN_THE_FIXTURE = SYSTEM
B_ALSO_IN_THE_BENCHMARK = OTHER


@pytest.fixture(scope="module")
def record(tmp_path_factory: pytest.TempPathFactory) -> ModelRecord:
    return write_model_export(
        tmp_path_factory.mktemp("export") / "export", directions=SIX_DIRECTIONS
    ).record


def two_systems_one_benchmark(system: str) -> dict[str, Any]:
    """`alpha` and `beta` over the fixture suite, and `beta` alone over a benchmark; `system` is
    the one that is a model of ours (the card is of it)."""
    document = bound_document(
        GOLDEN_MODEL_SHA256, GOLDEN_EXPORT_MANIFEST_SHA256, system=system
    )
    return with_a_run_in_another_suite(document, B_ALSO_IN_THE_BENCHMARK)


def tags_of(card: str) -> list[str]:
    head = card.split("---\n", 2)[1]
    return list(yaml.safe_load(head)["tags"])


# ---- the card ------------------------------------------------------------------------------------


def test_a_system_measured_only_on_a_fixture_suite_gets_the_fixture_banner_and_the_toy_tag(
    record: ModelRecord,
) -> None:
    document = two_systems_one_benchmark(A_ONLY_IN_THE_FIXTURE)
    assert view_of(document).evidence.kinds == ("benchmark", "fixture")

    card = card_of(record, document, system=A_ONLY_IN_THE_FIXTURE)

    assert f"> {FIXTURE_BANNER}\n" in card
    assert "Evidence: suites of the kind" not in card
    assert "toy" in tags_of(card)


def test_a_system_measured_on_a_benchmark_too_gets_the_mixed_banner_and_no_toy_tag(
    record: ModelRecord,
) -> None:
    document = two_systems_one_benchmark(B_ALSO_IN_THE_BENCHMARK)

    card = card_of(record, document, system=B_ALSO_IN_THE_BENCHMARK)

    assert (
        "> Evidence: suites of the kinds benchmark, fixture. 2 directions were measured "
        "for this system. The suites of the kind fixture are generated: they test the "
        "tooling and say nothing about translation quality.\n"
    ) in card
    assert FIXTURE_BANNER not in card
    assert "toy" not in tags_of(card)


def test_the_two_cards_of_one_report_differ_in_their_evidence(
    record: ModelRecord,
) -> None:
    only_fixture = card_of(
        record,
        two_systems_one_benchmark(A_ONLY_IN_THE_FIXTURE),
        system=A_ONLY_IN_THE_FIXTURE,
    )
    also_benchmark = card_of(
        record,
        two_systems_one_benchmark(B_ALSO_IN_THE_BENCHMARK),
        system=B_ALSO_IN_THE_BENCHMARK,
    )

    assert only_fixture.split("\n## ", 1)[0] != also_benchmark.split("\n## ", 1)[0]


def test_a_system_measured_only_on_a_benchmark_gets_the_measured_banner(
    record: ModelRecord,
) -> None:
    # `beta` has a run in the fixture suite as well as in the benchmark; take the fixture run away
    # from the result list of `beta`'s card by carding a system that only ran in the benchmark.
    document = bound_document(
        GOLDEN_MODEL_SHA256, GOLDEN_EXPORT_MANIFEST_SHA256, system=OTHER
    )
    document = with_a_run_in_another_suite(document, OTHER)
    document["results"] = [
        r
        for r in document["results"]
        if not (r["system"] == OTHER and r["suite"] == "golden")
    ]
    document["runs"] = [
        r
        for r in document["runs"]
        if not (r["system"] == OTHER and r["suite"] == "golden")
    ]
    document["comparisons"] = []
    document["coverage"] = [
        c
        for c in document["coverage"]
        if not (c["system"] == OTHER and c["suite"] == "golden")
    ]

    card = card_of(record, with_the_note_of(document), system=OTHER)

    assert (
        "> Evidence: suites of the kind benchmark. 2 directions were measured" in card
    )
    assert "toy" not in tags_of(card)


def test_a_run_in_which_no_direction_ran_does_not_count_as_evidence(
    record: ModelRecord,
) -> None:
    # `alpha` has a run in the benchmark (every direction not_trained, say) but no result there:
    # nothing of its card is from the benchmark.
    document = bound_document(
        GOLDEN_MODEL_SHA256, GOLDEN_EXPORT_MANIFEST_SHA256, system=SYSTEM
    )
    document = with_a_run_in_another_suite(document, SYSTEM)
    document["results"] = [
        r
        for r in document["results"]
        if not (r["system"] == SYSTEM and r["suite"] == "golden-bench")
    ]

    card = card_of(record, with_the_note_of(document), system=SYSTEM)

    assert f"> {FIXTURE_BANNER}\n" in card
    assert "toy" in tags_of(card)


# ---- release.json --------------------------------------------------------------------------------


def label_of(record: ModelRecord, document: dict[str, Any], system: str) -> str:
    release = release_record(
        record,
        view_of(document),
        system,
        golden_facts(),
        REPO_ID,
        model_sha256=GOLDEN_MODEL_SHA256,
        export_manifest_sha256=GOLDEN_EXPORT_MANIFEST_SHA256,
        report_sha256=GOLDEN_REPORT_SHA256,
    )
    return str(release["evidence"])


def test_release_json_says_fixture_for_the_system_that_ran_only_in_the_fixture_suite(
    record: ModelRecord,
) -> None:
    document = two_systems_one_benchmark(A_ONLY_IN_THE_FIXTURE)

    assert label_of(record, document, A_ONLY_IN_THE_FIXTURE) == "fixture"


def test_release_json_says_mixed_for_the_system_that_also_ran_on_the_benchmark(
    record: ModelRecord,
) -> None:
    document = two_systems_one_benchmark(B_ALSO_IN_THE_BENCHMARK)

    assert label_of(record, document, B_ALSO_IN_THE_BENCHMARK) == "mixed"


@pytest.mark.parametrize("system", [A_ONLY_IN_THE_FIXTURE, B_ALSO_IN_THE_BENCHMARK])
def test_the_label_of_release_json_is_the_one_the_card_shows(
    record: ModelRecord, system: str
) -> None:
    document = two_systems_one_benchmark(system)
    card = card_of(record, document, system=system)
    label = label_of(record, document, system)

    assert (f"> {FIXTURE_BANNER}\n" in card) == (label == "fixture")
    assert ("toy" in tags_of(card)) == (label == "fixture")
    assert ("kinds benchmark, fixture" in card) == (label == "mixed")


# ---- through the command ---------------------------------------------------------------------------


def test_the_stage_of_the_fixture_only_system_says_fixture_everywhere_and_keeps_the_report(
    tmp_path: Path,
) -> None:
    release = fabricated_release(tmp_path, system=A_ONLY_IN_THE_FIXTURE)
    document = with_a_run_in_another_suite(release.document, B_ALSO_IN_THE_BENCHMARK)
    report_dir = another_report_dir(release, document, "report-with-a-benchmark")

    made = invoke_stage(release, report_dir=report_dir)

    assert made.exit_code == 0, made.output
    stage = release.stage
    release_json = json.loads((stage / "release.json").read_text("utf-8"))
    assert release_json["evidence"] == "fixture"
    card = (stage / "README.md").read_text("utf-8")
    assert f"> {FIXTURE_BANNER}\n" in card
    assert "toy" in tags_of(card)
    manifest = json.loads((stage / "manifest.json").read_text("utf-8"))
    assert manifest["status"]["evidence"] == "fixture"
    # The report is the report: its own evidence still names the benchmark.
    staged_report = json.loads((stage / "report.json").read_text("utf-8"))
    assert staged_report["evidence"] == {
        "kinds": ["benchmark", "fixture"],
        "fixture_only": False,
    }


def test_the_stage_of_the_system_that_also_ran_on_the_benchmark_says_mixed(
    tmp_path: Path,
) -> None:
    release = fabricated_release(tmp_path, system=B_ALSO_IN_THE_BENCHMARK)
    document = with_a_run_in_another_suite(release.document, B_ALSO_IN_THE_BENCHMARK)
    report_dir = another_report_dir(release, document, "report-with-a-benchmark")

    made = invoke_stage(release, report_dir=report_dir)

    assert made.exit_code == 0, made.output
    release_json = json.loads((release.stage / "release.json").read_text("utf-8"))
    assert release_json["evidence"] == "mixed"
    card = (release.stage / "README.md").read_text("utf-8")
    assert "kinds benchmark, fixture" in card
    assert "toy" not in tags_of(card)
    manifest = json.loads((release.stage / "manifest.json").read_text("utf-8"))
    assert manifest["status"]["evidence"] == "mixed"
