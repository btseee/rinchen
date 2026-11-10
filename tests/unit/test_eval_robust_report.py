"""The narrow report of D12 (spec T15): `report.json`, schema `rinchen.report.robustness/1`,
the file the STOP gate reads ("robustness block populated in tiny report"). It names the system
and the suite, holds the six values with the counts each rests on (`coverage`) and points at
`robustness.json` by its digest. D13 owns the full report and embeds the same block."""

from __future__ import annotations

import hashlib
import json
import math

import pytest

from rinchen.eval.robust_hypotheses import Hypothesis
from rinchen.eval.robust_report import (
    REPORT_SCHEMA,
    ROBUSTNESS_FILE,
    report_bytes,
    robustness_report,
)
from rinchen.eval.robustness import (
    RobustnessSystem,
    compute_robustness,
    robustness_block,
)
from tests.support.robust_golden import (
    GOLDEN_COVERAGE,
    GOLDEN_SUITE_SHA256,
    golden_hypotheses,
    golden_result,
    golden_suite,
    golden_suite_sha256,
    key,
    row,
    suite_of,
)

DIGEST = hashlib.sha256(golden_result().to_bytes()).hexdigest()
SIX_VALUES = {
    "numbers_acc": 0.5,
    "placeholder_acc": 0.4,
    "term_acc": 0.75,
    "script_purity": 0.75,
    "entity_acc": 0.5714285714285714,
    "mong_source_retention": 0.5017570743156547,
}


def test_the_constants_are_the_ones_of_the_spec() -> None:
    assert REPORT_SCHEMA == "rinchen.report.robustness/1"
    assert ROBUSTNESS_FILE == "robustness.json"


def test_the_report_has_exactly_the_seven_keys_of_the_spec() -> None:
    report = robustness_report(golden_result(), DIGEST)

    assert sorted(report) == [
        "coverage",
        "robustness",
        "robustness_file",
        "robustness_sha256",
        "schema",
        "suite",
        "system",
    ]
    assert report["schema"] == "rinchen.report.robustness/1"
    assert report["robustness_file"] == "robustness.json"


def test_the_report_names_the_system_without_its_options_and_the_suite_with_its_digest() -> (
    None
):
    system = RobustnessSystem(
        name="tiny",
        model_sha256="a" * 64,
        manifest_sha256="b" * 64,
        options={"beam_size": 4},
    )
    result = compute_robustness(
        golden_suite(),
        golden_hypotheses(),
        suite_sha256=golden_suite_sha256(),
        system=system,
    )

    report = robustness_report(result, DIGEST)

    assert report["system"] == {
        "name": "tiny",
        "model_sha256": "a" * 64,
        "manifest_sha256": "b" * 64,
    }
    assert report["suite"] == {"name": "golden", "sha256": GOLDEN_SUITE_SHA256}


def test_the_report_holds_the_six_values_of_the_result() -> None:
    report = robustness_report(golden_result(), DIGEST)

    assert report["robustness"] == robustness_block(golden_result())
    assert list(report["robustness"]) == list(SIX_VALUES)
    for name, value in SIX_VALUES.items():
        assert report["robustness"][name] == pytest.approx(value, abs=1e-9)


def test_the_report_points_at_the_result_by_the_digest_it_was_given() -> None:
    report = robustness_report(golden_result(), DIGEST)

    assert report["robustness_sha256"] == DIGEST
    assert DIGEST == hashlib.sha256(golden_result().to_bytes()).hexdigest()


@pytest.mark.parametrize("digest", ["", "a" * 63, "A" * 64, "g" * 64, "a" * 65])
def test_the_digest_of_the_result_must_be_a_sha256(digest: str) -> None:
    with pytest.raises(ValueError, match="robustness_sha256"):
        robustness_report(golden_result(), digest)


def test_a_value_that_is_none_in_the_result_is_null_in_the_report() -> None:
    result = compute_robustness(
        suite_of(numbers=(row("a1"),)),
        {key("numbers", "a1"): Hypothesis("Үнэ 15.", "ok")},
        suite_sha256="0" * 64,
        system=RobustnessSystem(name="s"),
    )

    block = robustness_report(result, DIGEST)["robustness"]

    assert block["numbers_acc"] == 1.0
    assert [name for name, value in block.items() if value is None] == [
        "placeholder_acc",
        "term_acc",
        "script_purity",
        "entity_acc",
        "mong_source_retention",
    ]


def test_the_bytes_are_sorted_keys_two_space_indent_and_one_line_feed() -> None:
    report = robustness_report(golden_result(), DIGEST)

    data = report_bytes(report)

    text = data.decode("utf-8")
    assert data.endswith(b"\n")
    assert not data.endswith(b"\n\n")
    assert (
        text == json.dumps(report, sort_keys=True, indent=2, ensure_ascii=False) + "\n"
    )
    assert text.startswith(
        '{\n  "coverage": {\n    "entity_acc": {\n      "excluded": 0,'
    )


def test_the_same_report_gives_the_same_bytes_and_parses_back() -> None:
    report = robustness_report(golden_result(), DIGEST)

    assert report_bytes(report) == report_bytes(
        robustness_report(golden_result(), DIGEST)
    )
    assert json.loads(report_bytes(report)) == report


def test_the_report_holds_no_timestamp() -> None:
    names = set(robustness_report(golden_result(), DIGEST))

    assert not {"created", "timestamp", "time", "date"} & names


def test_a_value_that_is_not_a_number_cannot_be_written() -> None:
    report = robustness_report(golden_result(), DIGEST)
    report["robustness"]["numbers_acc"] = math.nan

    with pytest.raises(ValueError, match="JSON"):
        report_bytes(report)


def test_the_report_says_how_many_units_each_value_rests_on() -> None:
    report = robustness_report(golden_result(), DIGEST)

    assert report["coverage"] == GOLDEN_COVERAGE
    assert list(report["coverage"]) == list(SIX_VALUES)


def test_a_value_over_a_few_scored_rows_shows_the_rows_it_left_out() -> None:
    rows = (
        row("a1"),
        *(
            row(f"x{i}", source="It costs a lot.", reference="Үнэ их.")
            for i in range(9)
        ),
    )
    result = compute_robustness(
        suite_of(numbers=rows),
        {
            key("numbers", "a1"): Hypothesis("Үнэ 15.", "ok"),
            **{key("numbers", f"x{i}"): Hypothesis("Үнэ.", "ok") for i in range(9)},
        },
        suite_sha256="0" * 64,
        system=RobustnessSystem(name="s"),
    )

    report = robustness_report(result, DIGEST)

    assert report["robustness"]["numbers_acc"] == 1.0
    assert report["coverage"]["numbers_acc"] == {
        "scored": 1,
        "excluded": 9,
        "total": 10,
    }


def test_a_section_the_suite_does_not_have_rests_on_nothing() -> None:
    result = compute_robustness(
        suite_of(numbers=(row("a1"),)),
        {key("numbers", "a1"): Hypothesis("Үнэ 15.", "ok")},
        suite_sha256="0" * 64,
        system=RobustnessSystem(name="s"),
    )

    coverage = robustness_report(result, DIGEST)["coverage"]

    assert coverage["numbers_acc"] == {"scored": 1, "excluded": 0, "total": 1}
    nothing = {"scored": 0, "excluded": 0, "total": 0}
    for name in ("placeholder_acc", "term_acc", "script_purity", "entity_acc"):
        assert coverage[name] == nothing
    assert coverage["mong_source_retention"] == {
        **nothing,
        "chrf_cyrl": None,
        "chrf_mong": None,
    }


def test_the_counts_of_every_section_add_up_to_its_total() -> None:
    coverage = robustness_report(golden_result(), DIGEST)["coverage"]

    for counts in coverage.values():
        assert counts["scored"] + counts["excluded"] == counts["total"]
