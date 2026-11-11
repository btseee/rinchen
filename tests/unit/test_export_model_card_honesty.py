"""What the model card may claim (D13, Task 9, spec T16): the honesty rules, and the variants of the
report that move the card.

Every value that was not measured is the word `n/a` with its reason, never a blank, a zero or a one;
a sentence claims nothing that a table does not show (no ranking without the p-value, nothing about a
direction that was not measured); every number in the evaluation tables is a number of the report;
the robustness section prints the six values with the units they rest on or says that no run was
attached; a candidate in a comparison has its rows with the p-value and the verdict, and a system
that is the baseline of every comparison has none.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pytest

from rinchen.export.record import ModelRecord
from tests.support.model_exports import write_model_export
from tests.support.no_network import no_network  # noqa: F401 - the fixture of the mark
from tests.support.release_documents import (
    GOLDEN_EXPORT_MANIFEST_SHA256,
    GOLDEN_MODEL_SHA256,
    OTHER,
    SYSTEM,
    a_candidate_in_two_comparisons,
    as_benchmark,
    bound_document,
    with_a_direction_that_did_not_run,
    with_a_human_suite,
    with_comet_scores,
    with_one_unit,
    with_robustness,
    without_results_of,
)
from tests.support.release_fixture import NO_SCORER, SIX_DIRECTIONS, card_of

pytestmark = pytest.mark.usefixtures("no_network")


@pytest.fixture(scope="module")
def record(tmp_path_factory: pytest.TempPathFactory) -> ModelRecord:
    return write_model_export(
        tmp_path_factory.mktemp("export") / "export", directions=SIX_DIRECTIONS
    ).record


def bound(system: str = SYSTEM) -> dict[str, Any]:
    return bound_document(
        GOLDEN_MODEL_SHA256, GOLDEN_EXPORT_MANIFEST_SHA256, system=system
    )


def section(card: str, heading: str) -> str:
    """The text of the section `## heading` up to the next `## ` heading."""
    start = card.index(f"\n## {heading}\n")
    end = card.find("\n## ", start + 1)
    return card[start : len(card) if end < 0 else end]


# ---- an unmeasured value is n/a with its reason -------------------------------------------------


def test_every_n_a_in_every_variant_of_the_card_has_its_reason(
    record: ModelRecord,
) -> None:
    variants = [
        bound(),
        with_robustness(bound()),
        as_benchmark(bound()),
        with_a_human_suite(bound()),
        with_a_direction_that_did_not_run(
            bound(), "khk_Cyrl-rus_Cyrl", "not_selected", "x"
        ),
        with_comet_scores(bound()),
        with_one_unit(bound(OTHER), 0),
    ]

    for index, document in enumerate(variants):
        system = OTHER if index == len(variants) - 1 else SYSTEM
        card = card_of(record, document, system=system)
        assert re.findall(r"n/a(?! \()", card) == [], index
        assert "n/a ()" not in card, index


def test_no_cell_of_any_table_is_blank(record: ModelRecord) -> None:
    for document in (bound(), with_robustness(bound()), with_comet_scores(bound())):
        card = card_of(record, document)
        rows = [line for line in card.splitlines() if line.startswith("|")]

        assert rows
        assert all(cell.strip() for line in rows for cell in line.strip("|").split("|"))


def test_a_direction_that_was_not_measured_is_never_a_row_of_a_score_table(
    record: ModelRecord,
) -> None:
    card = card_of(record)
    evaluation = section(card, "Evaluation")
    rows = [
        line.split("|")[1].strip()
        for line in evaluation.splitlines()
        if line.startswith("| ") and "document units" in line
    ]

    assert rows == ["khk_Cyrl-eng_Latn", "eng_Latn-khk_Cyrl"]


def test_a_direction_that_did_not_run_is_listed_under_its_state_and_not_under_the_suite(
    record: ModelRecord,
) -> None:
    document = with_a_direction_that_did_not_run(
        bound(), "khk_Cyrl-rus_Cyrl", "not_selected", "not selected by --direction"
    )

    card = card_of(record, document)

    assert "\nn/a (not selected): khk_Cyrl-rus_Cyrl.\n" in card
    assert (
        "\nn/a (not in the suite): rus_Cyrl-khk_Cyrl, mn_Mong-eng_Latn, mn_Mong-rus_Cyrl.\n"
    ) in card


def test_a_declared_direction_the_model_was_not_trained_for_is_not_trained(
    record: ModelRecord,
) -> None:
    document = with_a_direction_that_did_not_run(
        bound(), "khk_Cyrl-deu_Latn", "not_trained", "the model was not trained for it"
    )

    card = card_of(record, document)

    assert "\nn/a (not trained): khk_Cyrl-deu_Latn, deu_Latn-khk_Cyrl, " in card


def test_the_state_of_a_direction_is_written_with_spaces(record: ModelRecord) -> None:
    document = with_a_direction_that_did_not_run(
        bound(), "khk_Cyrl-rus_Cyrl", "unsupported_language", "x"
    )

    assert "\nn/a (unsupported language): khk_Cyrl-rus_Cyrl.\n" in card_of(
        record, document
    )


def test_a_score_that_was_given_is_a_column_and_the_others_stay_n_a(
    record: ModelRecord,
) -> None:
    card = card_of(record, with_comet_scores(bound()))

    assert (
        "| direction | segments | resampling units | chrF++ | chrF | BLEU | COMET-22 |\n"
    ) in card
    assert (
        "| khk_Cyrl-eng_Latn | 10 | 5 document units | 73.08 [54.91, 85.48] | "
        "72.44 [53.76, 85.63] | 59.40 [36.34, 72.14] | 0.85 [0.79, 0.91] |\n"
    ) in card
    assert f"\nMetricX-24, MetricX-24-QE: n/a ({NO_SCORER}).\n" in card
    assert "- Not measured: MetricX-24 and MetricX-24-QE (no scorer is built), " in card
    assert "COMET-22," not in card


def test_a_score_given_for_one_direction_only_leaves_the_other_cell_n_a_with_its_reason(
    record: ModelRecord,
) -> None:
    document = with_comet_scores(bound(), direction="khk_Cyrl-eng_Latn")

    card = card_of(record, document)

    assert (
        "| khk_Cyrl-eng_Latn | 10 | 5 document units | 73.08 [54.91, 85.48] | "
        "72.44 [53.76, 85.63] | 59.40 [36.34, 72.14] | 0.85 [0.79, 0.91] |\n"
    ) in card
    assert (
        "| eng_Latn-khk_Cyrl | 10 | 5 document units | 68.06 [50.67, 80.98] | "
        f"67.41 [49.84, 80.59] | 50.87 [25.54, 67.88] | n/a ({NO_SCORER}) |\n"
    ) in card


def test_the_decoding_settings_are_the_ones_the_report_records_or_n_a(
    record: ModelRecord,
) -> None:
    none = bound_document(
        GOLDEN_MODEL_SHA256, GOLDEN_EXPORT_MANIFEST_SHA256, options={}
    )
    some = bound_document(
        GOLDEN_MODEL_SHA256,
        GOLDEN_EXPORT_MANIFEST_SHA256,
        options={"beam_size": 8, "converter": False, "script_mask": "off"},
    )

    assert "- Decoding settings of the run: n/a (none recorded).\n" in card_of(
        record, none
    )
    assert (
        "- Decoding settings of the run: beam_size=8, converter=false, script_mask=off.\n"
    ) in card_of(record, some)


# ---- the comparisons ----------------------------------------------------------------------------


def test_a_candidate_has_a_row_for_each_comparison_with_its_p_value_and_verdict(
    record: ModelRecord,
) -> None:
    card = card_of(record, bound(OTHER), system=OTHER)

    assert (
        "### Paired comparisons\n\n"
        "16 intervals and 4 paired comparisons are reported with no correction for "
        "multiple comparisons. A comparison is significant when its p-value is below "
        "0.05 and the difference is in favour of this system: the verdict is the "
        "p-value's, not the interval's.\n\n"
        "| suite | direction | metric | baseline | delta | interval | p | significant |\n"
        "| --- | --- | --- | --- | --- | --- | --- | --- |\n"
        "| golden | khk_Cyrl-eng_Latn | chrF++ | alpha | +20.79 | [+13.92, +33.41] | "
        "0.002 | yes |\n"
        "| golden | khk_Cyrl-eng_Latn | BLEU | alpha | +35.01 | [+22.91, +54.78] | "
        "0.002 | yes |\n"
        "| golden | eng_Latn-khk_Cyrl | chrF++ | alpha | +23.27 | [+12.65, +36.22] | "
        "0.002 | yes |\n"
        "| golden | eng_Latn-khk_Cyrl | BLEU | alpha | +40.20 | [+29.20, +57.55] | "
        "0.002 | yes |\n"
    ) in card


def test_a_difference_against_the_system_is_not_significant_whatever_the_p_value(
    record: ModelRecord,
) -> None:
    document = a_candidate_in_two_comparisons(bound())

    card = card_of(record, document)

    assert (
        "| golden | khk_Cyrl-eng_Latn | chrF++ | beta | -20.79 | [-33.41, -13.92] | "
        "0.002 | no |\n"
    ) in card
    assert " | yes |" not in card


def test_a_comparison_over_one_unit_has_no_interval_and_no_p_value_and_is_not_significant(
    record: ModelRecord,
) -> None:
    card = card_of(record, with_one_unit(bound(OTHER), 0), system=OTHER)

    assert (
        "| golden | khk_Cyrl-eng_Latn | chrF++ | alpha | +20.79 | "
        "n/a (fewer than two units) | n/a (fewer than two units) | no |\n"
    ) in card


def test_every_row_of_the_comparison_table_has_a_p_value_or_its_reason(
    record: ModelRecord,
) -> None:
    for document, system in (
        (bound(OTHER), OTHER),
        (a_candidate_in_two_comparisons(bound()), SYSTEM),
        (with_one_unit(bound(OTHER), 1), OTHER),
    ):
        evaluation = section(card_of(record, document, system=system), "Evaluation")
        table = [
            line for line in evaluation.splitlines() if line.startswith("| golden | ")
        ]
        assert table
        for line in table:
            cells = [cell.strip() for cell in line.strip("|").split("|")]
            assert len(cells) == 8
            assert re.fullmatch(r"\d\.\d{3}|n/a \(fewer than two units\)", cells[6]), (
                line
            )
            assert cells[7] in ("yes", "no"), line


def test_the_card_claims_no_rank_in_words(record: ModelRecord) -> None:
    banned = (
        "state of the art",
        "state-of-the-art",
        "outperform",
        "better than",
        "best ",
        "superior",
        "the best",
        "wins",
        "beats the",
    )
    for document, system in (
        (bound(), SYSTEM),
        (bound(OTHER), OTHER),
        (a_candidate_in_two_comparisons(bound()), SYSTEM),
    ):
        lowered = card_of(record, document, system=system).lower()
        for phrase in banned:
            assert phrase not in lowered, phrase


# ---- the numbers --------------------------------------------------------------------------------


def shown_decimals(card: str) -> list[str]:
    """Every decimal number of the tables of the evaluation and robustness sections."""
    found: list[str] = []
    for heading in ("Evaluation", "Robustness"):
        for line in section(card, heading).splitlines():
            if line.startswith("| "):
                found += re.findall(r"[+-]?\d+\.\d+", line)
    return found


def numbers_of(document: dict[str, Any]) -> set[str]:
    """The numbers of the report formatted as the card formats them: scores and interval ends with
    two decimals, deltas and their ends with a sign, p-values with three, robustness values with
    four."""
    numbers: set[str] = set()
    for result in document["results"]:
        for metric, value in result["metrics"].items():
            if value is not None:
                numbers.add(f"{value:.2f}")
                low, high = result["ci95"][metric]
                numbers.update((f"{low:.2f}", f"{high:.2f}"))
    for comparison in document["comparisons"]:
        numbers.add(f"{comparison['delta']:+.2f}")
        if comparison["ci95"] is not None:
            numbers.update(f"{end:+.2f}" for end in comparison["ci95"])
        if comparison["p_value"] is not None:
            numbers.add(f"{comparison['p_value']:.3f}")
    for entry in document["robustness"].values():
        numbers.update(
            f"{value:.4f}" for value in entry["block"].values() if value is not None
        )
    return numbers


def test_every_decimal_of_the_evaluation_tables_is_a_number_of_the_report(
    record: ModelRecord,
) -> None:
    documents = [
        (bound(), SYSTEM),
        (bound(OTHER), OTHER),
        (with_robustness(bound()), SYSTEM),
        (with_comet_scores(bound()), SYSTEM),
        (a_candidate_in_two_comparisons(bound()), SYSTEM),
    ]

    for document, system in documents:
        shown = shown_decimals(card_of(record, document, system=system))

        assert shown
        assert not set(shown) - numbers_of(document), sorted(
            set(shown) - numbers_of(document)
        )


def test_the_check_of_the_numbers_would_notice_a_number_that_is_not_the_reports(
    record: ModelRecord,
) -> None:
    card = card_of(record)
    altered = card.replace("73.08", "73.09")

    assert set(shown_decimals(altered)) - numbers_of(bound())


def test_the_card_of_a_system_holds_no_score_of_another_system(
    record: ModelRecord,
) -> None:
    card = card_of(record)

    # `beta`'s chrF++ in khk_Cyrl-eng_Latn is 93.87.
    assert "93.87" not in card


# ---- the robustness section ---------------------------------------------------------------------


def test_the_robustness_section_prints_the_six_values_with_scored_of_total(
    record: ModelRecord,
) -> None:
    card = card_of(record, with_robustness(bound()))

    assert (
        "## Robustness\n\n"
        "Measured on the robustness suite `robust-main` (sha256 555555555555), not on the "
        "suites above; bound to the system by its model digest. A value is a share of the "
        "units that were scored: read it with its counts.\n\n"
        "| value | result | scored of total |\n"
        "| --- | --- | --- |\n"
        "| numbers_acc | 0.9292 | 960 of 960 |\n"
        "| placeholder_acc | 0.9983 | 600 of 600 |\n"
        "| term_acc | 1.0000 | 1044 of 1056 |\n"
        "| script_purity | 1.0000 | 900 of 900 |\n"
        "| entity_acc | 0.9312 | 480 of 480 |\n"
        "| mong_source_retention | 0.9976 | 100 of 104 |\n"
    ) in card
    assert "no robustness run was attached" not in card


def test_a_run_bound_by_name_says_that_it_is_the_weaker_binding(
    record: ModelRecord,
) -> None:
    card = card_of(record, with_robustness(bound(), binding="name"))

    assert "bound to the system by its name only" in card


def test_a_value_without_a_scored_unit_is_n_a_with_its_reason(
    record: ModelRecord,
) -> None:
    block = {
        "numbers_acc": None,
        "placeholder_acc": 1.0,
        "term_acc": 1.0,
        "script_purity": 1.0,
        "entity_acc": 1.0,
        "mong_source_retention": 1.0,
    }

    card = card_of(record, with_robustness(bound(), block=block))

    assert "| numbers_acc | n/a (no scored unit) | 960 of 960 |\n" in card


def test_the_robustness_of_another_system_is_not_the_cards(record: ModelRecord) -> None:
    document = with_robustness(bound(), OTHER)

    card = card_of(record, document)

    assert "## Robustness\n\nn/a (no robustness run was attached)\n" in card


# ---- the suites and the evidence ----------------------------------------------------------------


def test_a_suite_in_which_the_system_has_no_run_is_named_and_not_dropped(
    record: ModelRecord,
) -> None:
    card = card_of(record, with_a_human_suite(bound()))

    assert (
        "\nSuites of the report in which this system has no run: golden-human.\n"
        in card
    )
    assert "### Suite golden-human" not in card


def test_a_report_in_which_the_system_has_no_result_is_refused(
    record: ModelRecord,
) -> None:
    with pytest.raises(ValueError, match="alpha"):
        card_of(record, without_results_of(bound(), SYSTEM))


def test_the_card_of_a_measured_report_has_no_toy_in_it(record: ModelRecord) -> None:
    card = card_of(record, as_benchmark(bound()))

    assert "toy" not in card
    assert "Evidence: suites of the kind benchmark" in card


def test_the_directions_section_of_a_model_with_one_direction_is_singular(
    tmp_path: Path,
) -> None:
    single = write_model_export(
        tmp_path / "export", directions=("eng_Latn-khk_Cyrl",)
    ).record

    card = card_of(single)

    assert "Trained for 1 direction: eng_Latn-khk_Cyrl.\n" in card
    assert "Targets and their tags: khk_Cyrl `<2khk_Cyrl>`. The default script" in card


def test_a_model_that_writes_the_traditional_script_says_nothing_about_a_converter(
    tmp_path: Path,
) -> None:
    both = write_model_export(
        tmp_path / "export", directions=("eng_Latn-khk_Cyrl", "eng_Latn-mn_Mong")
    ).record

    card = card_of(both)

    assert "needs the script converter" not in card


def test_a_target_placement_and_a_document_model_are_described(tmp_path: Path) -> None:
    other = write_model_export(
        tmp_path / "export",
        directions=SIX_DIRECTIONS,
        placement="tgt",
        document_level=True,
        quantization="int8",
    ).record

    card = card_of(other)

    assert "exported for CTranslate2 (int8)" in card
    assert (
        "The tag of the target language is forced as the first piece of the target (`tgt`)."
    ) in card
    assert "Document level: yes." in card
