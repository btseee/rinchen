"""The thresholds of `rinchen.tiny.thresholds` (D15, spec P8): eight results, in the order of the
table, each `pass` or `fail`, from the stage records and the figures the stages measured.

A bound compares with `>=` or `<=` exactly as the table says, so a value equal to its bound passes;
the tests put every bound at, just above and just below its value. A value that is missing (a `None`,
a key that is not there, a stage that never ran) or that cannot be a measurement (not a number, not
finite, outside its range) is a `fail` that names it and never a pass. `train_chrf` is the least of
the int8 export's figures over the five memorised directions and nothing else: the float32 export's
figures and a direction the table does not name never feed it.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from typing import Any

import pytest

from rinchen.config.tiny import ThresholdsSection
from rinchen.tiny.thresholds import (
    GATED_SYSTEM,
    MEMORISED_DIRECTIONS,
    REFERENCE_SYSTEM,
    THRESHOLD_IDS,
    ThresholdInputs,
    ThresholdResult,
    evaluate_thresholds,
)
from tests.support.no_network import no_network  # noqa: F401 - the fixture of the mark
from tests.support.tiny_threshold_inputs import (
    BUDGET,
    FIRST,
    LAST,
    LIMITS,
    chrf_table,
    good_inputs,
    good_stages,
    one,
    parity_with,
    results_of,
    robustness_with,
    with_changes,
)

pytestmark = pytest.mark.usefixtures("no_network")


# ---- the table ---------------------------------------------------------------------------------


def test_the_threshold_ids_are_the_eight_of_the_table_in_order() -> None:
    assert THRESHOLD_IDS == (
        "stages",
        "train_chrf",
        "parity_exact",
        "parity_delta",
        "int8_chrf_drop",
        "script_purity",
        "placeholder_acc",
        "wall_time",
    )


def test_the_memorised_directions_are_the_five_of_the_fixtures() -> None:
    assert MEMORISED_DIRECTIONS == (
        "eng_Latn-khk_Cyrl",
        "khk_Cyrl-eng_Latn",
        "rus_Cyrl-khk_Cyrl",
        "khk_Cyrl-rus_Cyrl",
        "mn_Mong-eng_Latn",
    )
    assert (GATED_SYSTEM, REFERENCE_SYSTEM) == ("int8", "float32")


def test_the_good_inputs_pass_every_threshold_in_the_order_of_the_table() -> None:
    results = evaluate_thresholds(good_inputs(), LIMITS, BUDGET)
    assert isinstance(results, tuple)
    assert [result.id for result in results] == list(THRESHOLD_IDS)
    assert [result.verdict for result in results] == ["pass"] * 8
    assert all(isinstance(result, ThresholdResult) for result in results)


def test_a_result_is_frozen_with_the_fields_of_the_interface() -> None:
    result = one("train_chrf", good_inputs())
    assert [field.name for field in dataclasses.fields(result)] == [
        "id",
        "bound",
        "value",
        "verdict",
        "detail",
    ]
    with pytest.raises(dataclasses.FrozenInstanceError):
        result.verdict = "fail"  # type: ignore[misc]


def test_the_inputs_are_frozen_with_the_fields_of_the_interface() -> None:
    inputs = good_inputs()
    assert [field.name for field in dataclasses.fields(inputs)] == [
        "stages",
        "memorised_chrf",
        "parity",
        "robustness",
        "elapsed_seconds",
    ]
    with pytest.raises(dataclasses.FrozenInstanceError):
        inputs.elapsed_seconds = 1.0  # type: ignore[misc]


def test_the_bounds_are_printed_as_a_comparison_and_the_bound_of_the_profile() -> None:
    bounds = {
        result.id: result.bound
        for result in evaluate_thresholds(good_inputs(), LIMITS, BUDGET)
    }
    assert bounds == {
        "stages": "<= 0",
        "train_chrf": ">= 50.0",
        "parity_exact": ">= 0.95",
        "parity_delta": "<= 0.001",
        "int8_chrf_drop": "<= 1.0",
        "script_purity": ">= 1.0",
        "placeholder_acc": ">= 1.0",
        "wall_time": "<= 1200",
    }


def test_the_values_are_the_figures_that_were_measured() -> None:
    inputs = ThresholdInputs(
        stages=good_stages(),
        memorised_chrf=chrf_table(**{FIRST: (88.25, 90.0), LAST: (97.5, 96.0)}),
        parity={"exact_share": 0.97, "mean_abs_delta": 0.0005},
        robustness={"script_purity": 0.999, "placeholder_acc": 0.9958},
        elapsed_seconds=321.5,
    )
    values = {r.id: r.value for r in evaluate_thresholds(inputs, LIMITS, BUDGET)}
    assert values == {
        "stages": 0,
        "train_chrf": 88.25,
        "parity_exact": 0.97,
        "parity_delta": 0.0005,
        "int8_chrf_drop": 1.75,
        "script_purity": 0.999,
        "placeholder_acc": 0.9958,
        "wall_time": 321.5,
    }


def test_evaluating_changes_none_of_its_inputs() -> None:
    inputs = good_inputs()
    before = (
        dict(inputs.memorised_chrf),
        dict(inputs.parity or {}),
        dict(inputs.robustness or {}),
        list(inputs.stages),
    )
    evaluate_thresholds(inputs, LIMITS, BUDGET)
    after = (
        dict(inputs.memorised_chrf),
        dict(inputs.parity or {}),
        dict(inputs.robustness or {}),
        list(inputs.stages),
    )
    assert before == after


# ---- each bound at, just above and just below its value --------------------------------------


def chrf_edit(direction: str, int8: float, float32: float = 99.5) -> dict[str, Any]:
    return {"memorised_chrf": chrf_table(**{direction: (int8, float32)})}


BOUNDARY_CASES: list[tuple[str, dict[str, Any], str]] = [
    ("train_chrf", chrf_edit(FIRST, 50.0, 50.0), "pass"),
    ("train_chrf", chrf_edit(FIRST, 50.01, 50.01), "pass"),
    ("train_chrf", chrf_edit(FIRST, 49.99, 49.99), "fail"),
    ("train_chrf", chrf_edit(LAST, 50.0, 50.0), "pass"),
    ("train_chrf", chrf_edit(LAST, 49.99, 49.99), "fail"),
    ("parity_exact", {"parity": parity_with(exact_share=0.95)}, "pass"),
    ("parity_exact", {"parity": parity_with(exact_share=0.951)}, "pass"),
    ("parity_exact", {"parity": parity_with(exact_share=0.949)}, "fail"),
    ("parity_delta", {"parity": parity_with(mean_abs_delta=1e-3)}, "pass"),
    ("parity_delta", {"parity": parity_with(mean_abs_delta=0.999e-3)}, "pass"),
    ("parity_delta", {"parity": parity_with(mean_abs_delta=1.001e-3)}, "fail"),
    ("parity_delta", {"parity": parity_with(mean_abs_delta=0.0)}, "pass"),
    ("int8_chrf_drop", chrf_edit(FIRST, 98.5, 99.5), "pass"),
    ("int8_chrf_drop", chrf_edit(FIRST, 98.5, 99.51), "fail"),
    ("int8_chrf_drop", chrf_edit(FIRST, 98.5, 99.49), "pass"),
    ("int8_chrf_drop", chrf_edit(LAST, 98.5, 99.5), "pass"),
    ("int8_chrf_drop", chrf_edit(LAST, 98.5, 99.51), "fail"),
    ("int8_chrf_drop", chrf_edit(FIRST, 99.75, 99.5), "pass"),
    ("script_purity", {"robustness": robustness_with(script_purity=1.0)}, "pass"),
    ("script_purity", {"robustness": robustness_with(script_purity=0.999)}, "fail"),
    ("placeholder_acc", {"robustness": robustness_with(placeholder_acc=1.0)}, "pass"),
    (
        "placeholder_acc",
        {"robustness": robustness_with(placeholder_acc=0.9958)},
        "fail",
    ),
    ("wall_time", {"elapsed_seconds": 1200.0}, "pass"),
    ("wall_time", {"elapsed_seconds": 1199.9}, "pass"),
    ("wall_time", {"elapsed_seconds": 1200.1}, "fail"),
    ("wall_time", {"elapsed_seconds": 0.0}, "pass"),
]


@pytest.mark.parametrize(("threshold_id", "changes", "verdict"), BOUNDARY_CASES)
def test_each_bound_passes_at_its_value_and_fails_just_beyond_it(
    threshold_id: str, changes: dict[str, Any], verdict: str
) -> None:
    results = results_of(with_changes(**changes))
    assert results[threshold_id].verdict == verdict
    # Only the threshold of the figure that changed can move, except the two that share the
    # memorised table.
    shared = (
        {"train_chrf", "int8_chrf_drop"}
        if "memorised_chrf" in changes
        else {threshold_id}
    )
    for other, result in results.items():
        if other not in shared:
            assert result.verdict == "pass", other


def test_the_bounds_are_those_of_the_section_and_the_budget_that_were_given() -> None:
    strict = ThresholdsSection(
        train_chrf_min=99.6,
        parity_exact_min=1.0,
        parity_delta_max=1e-7,
        int8_chrf_drop_max=0.0,
        script_purity_min=1.0,
        placeholder_acc_min=1.0,
    )
    inputs = with_changes(
        memorised_chrf=chrf_table(**{FIRST: (99.5, 99.75)}),
        parity=parity_with(exact_share=0.99),
    )
    results = {r.id: r for r in evaluate_thresholds(inputs, strict, 100)}
    assert results["train_chrf"].verdict == "fail"
    assert results["train_chrf"].bound == ">= 99.6"
    assert results["parity_exact"].verdict == "fail"
    assert results["parity_delta"].verdict == "fail"
    assert results["int8_chrf_drop"].verdict == "fail"
    assert results["wall_time"].verdict == "fail"
    assert results["wall_time"].bound == "<= 100"
    assert results["script_purity"].verdict == "pass"


def test_a_looser_section_passes_what_the_default_fails() -> None:
    loose = ThresholdsSection(
        train_chrf_min=10.0,
        parity_exact_min=0.5,
        parity_delta_max=0.5,
        int8_chrf_drop_max=5.0,
        script_purity_min=0.5,
        placeholder_acc_min=0.5,
    )
    inputs = ThresholdInputs(
        stages=good_stages(),
        memorised_chrf=chrf_table(**{FIRST: (20.0, 24.0)}),
        parity={"exact_share": 0.6, "mean_abs_delta": 0.01},
        robustness={"script_purity": 0.6, "placeholder_acc": 0.6},
        elapsed_seconds=5000.0,
    )
    assert [r.verdict for r in evaluate_thresholds(inputs, loose, 6000)] == ["pass"] * 8
    verdicts = [r.verdict for r in evaluate_thresholds(inputs, LIMITS, BUDGET)]
    assert verdicts == ["pass"] + ["fail"] * 7


# ---- train_chrf: the least over the named directions of the named system ---------------------


def test_train_chrf_is_the_least_of_the_five_directions_and_names_it() -> None:
    inputs = with_changes(
        memorised_chrf=chrf_table(**{"khk_Cyrl-rus_Cyrl": (61.25, 99.5)})
    )
    result = one("train_chrf", inputs)
    assert result.value == 61.25
    assert result.verdict == "pass"
    assert "khk_Cyrl-rus_Cyrl" in result.detail


def test_a_low_float32_figure_never_feeds_train_chrf() -> None:
    inputs = with_changes(memorised_chrf=chrf_table(**{FIRST: (99.5, 10.0)}))
    result = one("train_chrf", inputs)
    assert result.verdict == "pass"
    assert result.value == 99.5


def test_a_direction_the_table_does_not_name_never_feeds_train_chrf() -> None:
    table = chrf_table()
    table["deu_Latn-khk_Cyrl"] = {GATED_SYSTEM: 1.0, REFERENCE_SYSTEM: 1.0}
    results = results_of(with_changes(memorised_chrf=table))
    assert results["train_chrf"].verdict == "pass"
    assert results["train_chrf"].value == 99.5
    assert results["int8_chrf_drop"].verdict == "pass"
    assert results["int8_chrf_drop"].value == 0.0


def test_a_system_the_table_does_not_name_never_feeds_train_chrf() -> None:
    table = chrf_table()
    table[FIRST]["float16"] = 1.0
    table[FIRST]["hypotheses"] = 1.0
    result = one("train_chrf", with_changes(memorised_chrf=table))
    assert (result.verdict, result.value) == ("pass", 99.5)


def test_the_least_is_taken_not_the_greatest_or_the_first_or_the_mean() -> None:
    inputs = with_changes(
        memorised_chrf=chrf_table(
            **{
                "eng_Latn-khk_Cyrl": (99.0, 99.0),
                "khk_Cyrl-eng_Latn": (51.0, 99.0),
                "rus_Cyrl-khk_Cyrl": (75.0, 99.0),
                "khk_Cyrl-rus_Cyrl": (49.5, 99.0),
                "mn_Mong-eng_Latn": (80.0, 99.0),
            }
        )
    )
    result = one("train_chrf", inputs)
    assert (result.verdict, result.value) == ("fail", 49.5)


def test_the_drop_is_the_largest_over_the_directions_not_the_mean_or_the_first() -> (
    None
):
    inputs = with_changes(
        memorised_chrf=chrf_table(
            **{
                "eng_Latn-khk_Cyrl": (99.0, 99.25),
                "khk_Cyrl-eng_Latn": (98.0, 99.75),
                "rus_Cyrl-khk_Cyrl": (99.5, 99.0),
                "khk_Cyrl-rus_Cyrl": (97.0, 99.0),
                "mn_Mong-eng_Latn": (99.0, 99.0),
            }
        )
    )
    result = one("int8_chrf_drop", inputs)
    assert result.value == 2.0
    assert result.verdict == "fail"
    assert "khk_Cyrl-rus_Cyrl" in result.detail


def test_a_drop_that_is_negative_everywhere_is_the_largest_of_the_negative_ones() -> (
    None
):
    inputs = with_changes(
        memorised_chrf=chrf_table(**{FIRST: (99.75, 99.5), LAST: (99.5, 99.0)})
    )
    result = one("int8_chrf_drop", inputs)
    assert result.verdict == "pass"
    assert result.value == 0.0


def test_a_tuple_of_records_is_read_like_a_list() -> None:
    inputs = with_changes(stages=tuple(good_stages()))
    assert results_of(inputs)["stages"].verdict == "pass"


def test_the_detail_of_a_pass_is_not_empty() -> None:
    for result in evaluate_thresholds(good_inputs(), LIMITS, BUDGET):
        assert result.detail
        assert isinstance(result.detail, str)


def test_a_mapping_that_is_not_a_dict_is_read_like_one() -> None:
    class Table(Mapping[str, Any]):
        def __init__(self, data: Mapping[str, Any]) -> None:
            self._data = data

        def __getitem__(self, key: str) -> Any:
            return self._data[key]

        def __iter__(self) -> Any:
            return iter(self._data)

        def __len__(self) -> int:
            return len(self._data)

    inputs = with_changes(
        memorised_chrf=Table({d: Table(v) for d, v in chrf_table().items()}),
        parity=Table(parity_with()),
        robustness=Table(robustness_with()),
    )
    assert [r.verdict for r in evaluate_thresholds(inputs, LIMITS, BUDGET)] == [
        "pass"
    ] * 8
