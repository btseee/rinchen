"""What the model card shows beside a figure (D13 fix wave 1: I3, I4, M6, M7), as `report.md` does.

- I3: the table of a suite has a column `outputs ok` with `n of n` when every output was `ok` and
  `k of n (1 empty, 3 truncated)` otherwise (the statuses of the report's results, by name).
- I4: the listing of a suite names the dataset and the revision (`unpinned`, `<revision>
  (unpinned)` or `n/a (no source record)`).
- M6: the comparisons have a `verdict`, `better`, `worse` or `no significant difference` (`n/a
  (<reason>)` without a p-value), and the sentence above the table explains it.
- M7: the robustness measurement is printed with `suite kind not recorded` beside its suite, and the
  limits say so (only when a robustness measurement is printed).
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from rinchen.export.model_card_evaluation import comparisons_section
from rinchen.export.record import ModelRecord
from tests.support.model_exports import write_model_export
from tests.support.no_network import no_network  # noqa: F401 - the fixture of the mark
from tests.support.release_documents import (
    GOLDEN_EXPORT_MANIFEST_SHA256,
    GOLDEN_MODEL_SHA256,
    OTHER,
    SYSTEM,
    a_candidate_in_two_comparisons,
    bound_document,
    view_of,
    with_a_comparison_of,
    with_one_unit,
    with_robustness,
    with_source,
    with_statuses,
)
from tests.support.release_fixture import SIX_DIRECTIONS, card_of

pytestmark = pytest.mark.usefixtures("no_network")

FIRST = "khk_Cyrl-eng_Latn"
SECOND = "eng_Latn-khk_Cyrl"
REVISION = "0123456789abcdef0123456789abcdef01234567"


@pytest.fixture(scope="module")
def record(tmp_path_factory: pytest.TempPathFactory) -> ModelRecord:
    return write_model_export(
        tmp_path_factory.mktemp("export") / "export", directions=SIX_DIRECTIONS
    ).record


def bound(system: str = SYSTEM) -> dict:  # type: ignore[type-arg]
    return bound_document(
        GOLDEN_MODEL_SHA256, GOLDEN_EXPORT_MANIFEST_SHA256, system=system
    )


def lines_of(card: str) -> list[str]:
    return card.split("\n")


# ---- I3: the statuses beside the metric ----------------------------------------------------------------


def test_the_evaluation_table_has_an_outputs_ok_column_after_the_segments(
    record: ModelRecord,
) -> None:
    lines = lines_of(card_of(record, bound()))

    assert (
        "| direction | segments | outputs ok | resampling units | chrF++ | chrF | BLEU |"
        in lines
    )


def test_an_all_ok_direction_says_n_of_n(record: ModelRecord) -> None:
    lines = lines_of(card_of(record, bound()))

    assert (
        f"| {FIRST} | 10 | 10 of 10 | 5 document units | 73.08 [54.91, 85.48] | "
        "72.44 [53.76, 85.63] | 59.40 [36.34, 72.14] |" in lines
    )


def test_flagged_outputs_are_named_beside_the_metric_with_their_counts(
    record: ModelRecord,
) -> None:
    document = with_statuses(
        bound(), SYSTEM, FIRST, {"empty": 1, "no_output": 2, "ok": 6, "truncated": 1}
    )

    lines = lines_of(card_of(record, document))

    row = next(line for line in lines if line.startswith(f"| {FIRST} |"))
    assert row.startswith(
        f"| {FIRST} | 10 | 6 of 10 (1 empty, 2 no_output, 1 truncated) |"
    )
    other = next(line for line in lines if line.startswith(f"| {SECOND} |"))
    assert other.startswith(f"| {SECOND} | 10 | 10 of 10 |")


def test_the_counts_are_those_of_the_carded_system_only(
    record: ModelRecord,
) -> None:
    # `beta` has flagged outputs in the report; the card of `alpha` does not show them.
    document = with_statuses(bound(), OTHER, FIRST, {"ok": 4, "too_long": 6})

    card = card_of(record, document)

    assert "too_long" not in card
    assert "10 of 10" in card


# ---- I4: the dataset and the revision ------------------------------------------------------------------


def listing(card: str) -> list[str]:
    start = card.index("### Suite golden\n")
    return [line for line in lines_of(card[start:])[2:8] if line.startswith("- ")]


def test_a_fixture_has_no_source_record_and_says_why(record: ModelRecord) -> None:
    lines = listing(card_of(record, bound()))

    assert lines[:5] == [
        "- kind: fixture",
        "- split: devtest",
        "- rows: 10",
        "- dataset: n/a (no source record)",
        "- revision: n/a (no source record)",
    ]


def test_a_pinned_suite_names_the_dataset_and_the_revision(record: ModelRecord) -> None:
    document = with_source(
        bound(), dataset="org/some-benchmark", revision=REVISION, pinned=True
    )

    lines = listing(card_of(record, document))

    assert lines[3:5] == ["- dataset: org/some-benchmark", f"- revision: {REVISION}"]


def test_a_dataset_without_a_revision_is_unpinned(record: ModelRecord) -> None:
    document = with_source(
        bound(), dataset="org/some-benchmark", revision=None, pinned=False
    )

    lines = listing(card_of(record, document))

    assert lines[3:5] == ["- dataset: org/some-benchmark", "- revision: unpinned"]


def test_a_revision_that_was_recorded_and_not_pinned_says_so(
    record: ModelRecord,
) -> None:
    document = with_source(
        bound(), dataset="org/some-benchmark", revision=REVISION, pinned=False
    )

    lines = listing(card_of(record, document))

    assert lines[4] == f"- revision: {REVISION} (unpinned)"


# ---- M6: the verdict ------------------------------------------------------------------------------------


HEADER = "| suite | direction | metric | baseline | delta | interval | p | verdict |"


def candidate_document() -> dict:  # type: ignore[type-arg]
    """`alpha` is the candidate against `beta` in both directions, chrF++ and BLEU: significantly
    worse (delta below zero, p 0.002) as the helper makes the swapped comparisons."""
    return a_candidate_in_two_comparisons(bound())


def test_a_candidate_that_is_significantly_worse_reads_worse_and_not_no(
    record: ModelRecord,
) -> None:
    lines = lines_of(card_of(record, candidate_document()))

    assert HEADER in lines
    worse = [
        line for line in lines if line.startswith("| golden |") and "| beta |" in line
    ]
    assert len(worse) == 4
    assert all(line.endswith("| worse |") for line in worse)
    assert not any(line.endswith("| no |") for line in lines)


def test_a_candidate_that_is_significantly_better_reads_better(
    record: ModelRecord,
) -> None:
    document = with_a_comparison_of(candidate_document(), 4, delta=5.0, p_value=0.004)

    lines = lines_of(card_of(record, document))

    assert any(
        line.startswith("| golden | khk_Cyrl-eng_Latn | chrF++ | beta | +5.00 |")
        and line.endswith("| 0.004 | better |")
        for line in lines
    )


def test_a_difference_that_is_not_significant_says_so_in_words(
    record: ModelRecord,
) -> None:
    document = with_a_comparison_of(candidate_document(), 4, delta=1.2, p_value=0.3)

    lines = lines_of(card_of(record, document))

    assert any(line.endswith("| 0.300 | no significant difference |") for line in lines)


def test_a_comparison_without_a_p_value_says_n_a_with_the_reason(
    record: ModelRecord,
) -> None:
    document = with_one_unit(candidate_document(), 4)

    lines = lines_of(card_of(record, document))

    assert any(
        line.endswith("| n/a (fewer than two units) | n/a (fewer than two units) |")
        for line in lines
    )


def test_the_sentence_above_the_table_explains_the_verdicts(
    record: ModelRecord,
) -> None:
    card = card_of(record, candidate_document())

    assert (
        "The verdict compares this system with the baseline and is the p-value's, not "
        "the interval's: `better` is a p-value below 0.05 with the difference in this "
        "system's favour, `worse` is a p-value below 0.05 with the difference against it, "
        "and `no significant difference` is anything else.\n"
    ) in card
    assert "| significant |" not in card


# ---- M7: the robustness suite has no kind ---------------------------------------------------------------


def test_the_robustness_measurement_says_that_its_suite_kind_is_not_recorded(
    record: ModelRecord,
) -> None:
    card = card_of(record, with_robustness(bound()))

    assert (
        "Measured on the robustness suite `robust-main` (sha256 555555555555, suite kind "
        "not recorded), not on the suites above;"
    ) in card


def test_the_limits_say_that_the_robustness_suite_kind_is_not_recorded(
    record: ModelRecord,
) -> None:
    card = card_of(record, with_robustness(bound()))

    limits = card[card.index("\n## Limits\n") :]
    assert (
        "- The kind of the robustness suite is not recorded in the robustness block, so the "
        "evidence above counts only the suites of the evaluation tables.\n"
    ) in limits


def test_without_a_robustness_measurement_the_limits_do_not_mention_it(
    record: ModelRecord,
) -> None:
    card = card_of(record, bound())

    assert "robustness suite is not recorded" not in card


# ---- the verdict follows the p-value and alpha ------------------------------------------------------------


@pytest.mark.parametrize(
    ("p_value", "verdict"),
    [
        (0.049, "better"),
        (0.011, "better"),
        (0.05, "no significant difference"),
        (0.052, "no significant difference"),
    ],
)
def test_the_verdict_is_the_p_value_against_alpha_nothing_else(
    record: ModelRecord, p_value: float, verdict: str
) -> None:
    document = with_a_comparison_of(candidate_document(), 4, delta=5.0, p_value=p_value)

    lines = lines_of(card_of(record, document))

    assert any(
        line.startswith("| golden | khk_Cyrl-eng_Latn | chrF++ | beta | +5.00 |")
        and line.endswith(f"| {verdict} |")
        for line in lines
    )


def test_the_verdict_uses_the_alpha_of_the_report_and_says_it(
    record: ModelRecord,
) -> None:
    document = candidate_document()
    document["bootstrap"]["alpha"] = 0.1
    document = with_a_comparison_of(document, 4, delta=5.0, p_value=0.08)

    card = card_of(record, document)

    assert (
        "`better` is a p-value below 0.1 with the difference in this system's" in card
    )
    assert any(
        line.startswith("| golden | khk_Cyrl-eng_Latn | chrF++ | beta | +5.00 |")
        and line.endswith("| 0.080 | better |")
        for line in lines_of(card)
    )


def test_a_metric_that_is_lower_is_better_reads_the_other_way(
    record: ModelRecord,
) -> None:
    # No count metric is lower-is-better (MetricX would be), and the strict reader keeps the flag
    # to the registry; the section follows the comparison's own flag all the same.
    view = view_of(candidate_document())
    comparisons = tuple(replace(c, higher_is_better=False) for c in view.comparisons)

    lines = comparisons_section(replace(view, comparisons=comparisons), SYSTEM)

    rows = [line for line in lines if line.startswith("| golden |")]
    assert rows and all(line.endswith("| better |") for line in rows)


# ---- the robustness limit is the carded system's -------------------------------------------------------------


def test_another_systems_robustness_block_does_not_add_the_limit_to_this_card(
    record: ModelRecord,
) -> None:
    # `beta` has a robustness measurement in the report; the card of `alpha` prints none, so its
    # limits do not speak of one.
    document = with_robustness(bound(), OTHER)

    card = card_of(record, document)

    assert "robustness suite is not recorded" not in card
    assert "n/a (no robustness run was attached)" in card
