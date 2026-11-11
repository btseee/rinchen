"""The release targets of D13 (Task 8, spec T13): `evaluate_targets` turns the comparisons, the
results and D12's robustness blocks of a report into a status per target, `met`, `not_met` or
`unmeasured`, and a sentence saying what the status rests on.

A target is a measurement: nothing here fails because a target is not met, and `unmeasured` is the
answer whenever the evidence does not decide (never a guess in either direction). `beats_baseline`
counts the directions of the benchmark `flores_plus`, split `devtest`, in which a model of ours
(system kind `rinchen`) is significantly better than the baseline on chrF++; `met` needs
`min_directions` of them, `not_met` is for when the directions still unmeasured cannot make up the
difference, and anything between is `unmeasured`. The robustness targets read the block of the
model's robustness run and `met` needs every unit scored. The cases are hand-built dictionaries of
the shape the report holds, with the counts the brief gives.
"""

from __future__ import annotations

import ast
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pytest

import rinchen
from rinchen.eval.targets import TARGET_STATUSES, TargetOutcome, evaluate_targets
from tests.support.eval_report import shipped_config
from tests.support.no_network import no_network  # noqa: F401 - the fixture of the mark

pytestmark = pytest.mark.usefixtures("no_network")

STANDARD = (
    "khk_Cyrl-deu_Latn",
    "deu_Latn-khk_Cyrl",
    "khk_Cyrl-eng_Latn",
    "eng_Latn-khk_Cyrl",
    "khk_Cyrl-jpn_Jpan",
    "jpn_Jpan-khk_Cyrl",
    "khk_Cyrl-kaz_Cyrl",
    "kaz_Cyrl-khk_Cyrl",
    "khk_Cyrl-kor_Hang",
    "kor_Hang-khk_Cyrl",
    "khk_Cyrl-rus_Cyrl",
    "rus_Cyrl-khk_Cyrl",
    "khk_Cyrl-tur_Latn",
    "tur_Latn-khk_Cyrl",
    "khk_Cyrl-zho_Hans",
    "zho_Hans-khk_Cyrl",
)
FLORES = {"name": "flores", "benchmark": "flores_plus", "split": "devtest"}
SUITES = {"flores": FLORES}
ALPHA = 0.05
NUMBERS_BLOCK = {
    "numbers_acc": 0.9292,
    "placeholder_acc": 0.9983,
    "term_acc": 1.0,
    "script_purity": 1.0,
    "entity_acc": 0.9312,
    "mong_source_retention": 0.9976,
}
ENTRY = {"scored": 960, "excluded": 0, "total": 960}


def comparison(
    direction: str,
    *,
    system: str = "toy",
    suite: str = "flores",
    metric: str = "chrf_pp",
    delta: float = 5.0,
    p_value: float | None = 0.002,
) -> dict[str, Any]:
    """A comparison of the shape the report holds: `significant` follows from the p-value and
    the delta as the report decides it (below alpha and better than zero)."""
    significant = p_value is not None and p_value < ALPHA and delta > 0
    return {
        "suite": suite,
        "direction": direction,
        "system": system,
        "baseline": "nllb",
        "metric": metric,
        "higher_is_better": True,
        "delta": delta,
        "ci95": None if p_value is None else [delta - 1.0, delta + 1.0],
        "p_value": p_value,
        "better_fraction": None if p_value is None else 1.0,
        "significant": significant,
        "unit": "segment",
        "units": 1 if p_value is None else 400,
    }


def wins(count: int, *, nonwins: int = 0, **options: Any) -> list[dict[str, Any]]:
    """`count` directions with a significant win followed by `nonwins` with a p-value above
    alpha, taken from the standard directions in order."""
    chosen = STANDARD[: count + nonwins]
    return [
        comparison(
            direction,
            **({"p_value": 0.4} if index >= count else {}),
            **options,
        )
        for index, direction in enumerate(chosen)
    ]


def result_of(system: str, kind: str = "rinchen") -> dict[str, Any]:
    return {"system": system, "provenance": {"system": {"kind": kind}}}


def beats(
    comparisons: Sequence[Mapping[str, Any]],
    *,
    systems: Sequence[tuple[str, str]] = (("toy", "rinchen"),),
    suites: Mapping[str, Mapping[str, Any]] = SUITES,
) -> TargetOutcome:
    config = shipped_config()
    results = [result_of(name, kind) for name, kind in systems]
    outcomes = evaluate_targets(config, comparisons, results, {}, suites=suites)
    return next(outcome for outcome in outcomes if outcome.id == "beats_baseline")


def robustness_of(
    system: str = "toy", block: Mapping[str, float | None] | None = None, **entries: Any
) -> dict[str, Any]:
    coverage = {key: dict(ENTRY) for key in NUMBERS_BLOCK}
    coverage.update(entries)
    return {
        system: {
            "block": dict(NUMBERS_BLOCK if block is None else block),
            "coverage": coverage,
            "source": {"binding": "model_digest"},
        }
    }


def outcome_of(
    target: str,
    robustness: Mapping[str, Any],
    *,
    systems: Sequence[tuple[str, str]] = (("toy", "rinchen"),),
) -> TargetOutcome:
    results = [result_of(name, kind) for name, kind in systems]
    outcomes = evaluate_targets(
        shipped_config(), [], results, robustness, suites=SUITES
    )
    return next(outcome for outcome in outcomes if outcome.id == target)


# ---- beats_baseline ---------------------------------------------------------------------------


def test_the_golden_comparisons_leave_beats_baseline_unmeasured_with_nothing_measured() -> (
    None
):
    golden = {"name": "golden", "benchmark": "golden", "split": "devtest"}
    comparisons = [
        comparison("khk_Cyrl-eng_Latn", suite="golden"),
        comparison("eng_Latn-khk_Cyrl", suite="golden"),
    ]

    outcome = beats(comparisons, suites={"golden": golden})

    assert outcome.status == "unmeasured"
    assert outcome.basis == "0 of 16 directions measured"
    assert outcome.value is None
    assert outcome.threshold == 12


def test_twelve_significant_wins_in_sixteen_directions_meet_the_target() -> None:
    outcome = beats(wins(12, nonwins=4))

    assert outcome.status == "met"
    assert outcome.value == 12
    assert outcome.threshold == 12
    assert outcome.basis == "12 of 16 directions significantly better (12 are needed)"


def test_sixteen_wins_meet_it_too() -> None:
    assert beats(wins(16)).status == "met"


def test_eleven_wins_with_four_directions_missing_is_unmeasured() -> None:
    outcome = beats(wins(11, nonwins=1))

    assert outcome.status == "unmeasured"
    assert outcome.value == 11
    assert outcome.basis == "12 of 16 directions measured, 11 significantly better"


def test_eleven_wins_in_sixteen_measured_directions_cannot_meet_it() -> None:
    outcome = beats(wins(11, nonwins=5))

    assert outcome.status == "not_met"
    assert outcome.value == 11
    assert outcome.basis == (
        "11 of 16 directions significantly better and none left to measure "
        "(12 are needed)"
    )


def test_twelve_wins_of_which_one_has_p_0_06_count_as_eleven() -> None:
    comparisons = wins(12, nonwins=4)
    comparisons[0] = comparison(STANDARD[0], p_value=0.06)

    outcome = beats(comparisons)

    assert comparisons[0]["delta"] > 0 and comparisons[0]["significant"] is False
    assert outcome.status == "not_met"
    assert outcome.value == 11


@pytest.mark.parametrize(
    ("won", "lost", "status"),
    [
        (4, 4, "unmeasured"),  # 4 wins + 8 unmeasured = 12: still reachable
        (3, 5, "not_met"),  # 3 wins + 8 unmeasured = 11: out of reach
        (12, 0, "met"),
        (11, 0, "unmeasured"),  # five directions unmeasured can make up one
        (0, 16, "not_met"),
        (0, 0, "unmeasured"),
    ],
)
def test_the_status_turns_at_the_counts_the_spec_names(
    won: int, lost: int, status: str
) -> None:
    assert beats(wins(won, nonwins=lost)).status == status


def test_a_comparison_of_a_suite_that_is_not_flores_plus_devtest_is_not_counted() -> (
    None
):
    suites = {
        "flores": FLORES,
        "other": {"name": "other", "benchmark": "ntrex", "split": "devtest"},
        "later": {"name": "later", "benchmark": "flores_plus", "split": "test"},
    }
    comparisons = wins(12, nonwins=4, suite="other") + wins(
        12, nonwins=4, suite="later"
    )

    outcome = beats(comparisons, suites=suites)

    assert outcome.status == "unmeasured"
    assert outcome.basis == "0 of 16 directions measured"


def test_a_comparison_of_a_suite_the_report_does_not_describe_is_not_counted() -> None:
    assert beats(wins(12, nonwins=4, suite="unknown")).status == "unmeasured"


def test_only_the_primary_metric_counts() -> None:
    comparisons = wins(12, nonwins=4, metric="bleu")

    assert beats(comparisons).status == "unmeasured"


def test_a_comparison_without_a_p_value_is_unmeasured_never_a_win() -> None:
    comparisons = wins(12, nonwins=3)
    comparisons.append(comparison(STANDARD[15], p_value=None))

    outcome = beats(comparisons)

    assert outcome.status == "met"  # twelve wins stand
    assert beats(
        wins(11, nonwins=4) + [comparison(STANDARD[15], p_value=None)]
    ).status == ("unmeasured")
    assert outcome.basis.startswith("12 of 16 directions significantly better")


def test_a_comparison_whose_unit_count_leaves_no_p_value_is_not_counted_as_measured() -> (
    None
):
    outcome = beats([comparison(STANDARD[0], p_value=None)])

    assert outcome.status == "unmeasured"
    assert outcome.basis == "0 of 16 directions measured"


def test_a_system_that_is_not_one_of_ours_is_not_a_candidate() -> None:
    comparisons = wins(12, nonwins=4, system="a-file")

    outcome = beats(comparisons, systems=(("a-file", "hypotheses"),))

    assert outcome.status == "unmeasured"
    assert outcome.basis == "0 of 16 directions measured"


def test_a_system_with_no_result_is_not_a_candidate() -> None:
    assert beats(wins(12, nonwins=4), systems=()).status == "unmeasured"


def test_a_direction_in_two_suites_is_a_win_only_when_every_comparison_of_it_is() -> (
    None
):
    suites = {
        "flores": FLORES,
        "again": {"name": "again", "benchmark": "flores_plus", "split": "devtest"},
    }
    comparisons = wins(12, nonwins=4)
    comparisons.append(comparison(STANDARD[0], suite="again", p_value=0.3))
    comparisons.extend(
        comparison(direction, suite="again") for direction in STANDARD[1:12]
    )

    outcome = beats(comparisons, suites=suites)

    assert outcome.status == "not_met"
    assert outcome.value == 11


def test_two_candidates_are_judged_one_by_one_and_the_worse_decides() -> None:
    comparisons = wins(12, nonwins=4) + wins(11, nonwins=5, system="int8")

    outcome = beats(comparisons, systems=(("toy", "rinchen"), ("int8", "rinchen")))

    assert outcome.status == "not_met"
    assert outcome.value == 11
    assert outcome.basis == (
        "int8: 11 of 16 directions significantly better and none left to measure "
        "(12 are needed); toy: 12 of 16 directions significantly better (12 are needed)"
    )


def test_two_candidates_that_both_win_meet_it() -> None:
    comparisons = wins(12, nonwins=4) + wins(14, nonwins=2, system="int8")

    outcome = beats(comparisons, systems=(("toy", "rinchen"), ("int8", "rinchen")))

    assert outcome.status == "met"
    assert outcome.value == 12


# ---- off_target -------------------------------------------------------------------------------


@pytest.mark.parametrize("comparisons", [[], wins(16)])
def test_off_target_is_always_unmeasured(comparisons: list[dict[str, Any]]) -> None:
    outcomes = evaluate_targets(
        shipped_config(),
        comparisons,
        [result_of("toy")],
        robustness_of(),
        suites=SUITES,
    )

    off = next(outcome for outcome in outcomes if outcome.id == "off_target")

    assert off.status == "unmeasured"
    assert off.basis == "no language identification is run"
    assert off.value is None
    assert off.threshold == 0.01


# ---- the robustness targets -------------------------------------------------------------------


@pytest.mark.parametrize(
    ("value", "status"),
    [(0.9292, "not_met"), (0.995, "met"), (0.99, "met"), (0.9899, "not_met")],
)
def test_the_numbers_target_follows_the_block_value(value: float, status: str) -> None:
    block = {**NUMBERS_BLOCK, "numbers_acc": value}

    outcome = outcome_of("numbers", robustness_of(block=block))

    assert outcome.status == status
    assert outcome.value == value
    assert outcome.threshold == 0.99


def test_the_basis_of_a_robustness_target_names_the_value_and_the_threshold() -> None:
    low = outcome_of("numbers", robustness_of())
    high = outcome_of(
        "numbers", robustness_of(block={**NUMBERS_BLOCK, "numbers_acc": 0.995})
    )

    assert low.basis == "0.9292 is below 0.99"
    assert high.basis == "0.9950 is at least 0.99"


def test_a_null_value_is_unmeasured() -> None:
    block = {**NUMBERS_BLOCK, "numbers_acc": None}

    outcome = outcome_of("numbers", robustness_of(block=block))

    assert outcome.status == "unmeasured"
    assert outcome.value is None
    assert outcome.basis == "the value is null: no unit was scored"


def test_no_robustness_run_is_unmeasured() -> None:
    outcome = outcome_of("numbers", {})

    assert outcome.status == "unmeasured"
    assert outcome.value is None
    assert outcome.basis == "no robustness run of a rinchen system was given"


def test_a_block_of_a_system_that_is_not_one_of_ours_does_not_count() -> None:
    outcome = outcome_of(
        "numbers",
        robustness_of("a-file", block={**NUMBERS_BLOCK, "numbers_acc": 0.995}),
        systems=(("a-file", "hypotheses"),),
    )

    assert outcome.status == "unmeasured"
    assert outcome.basis == "no robustness run of a rinchen system was given"


@pytest.mark.parametrize(
    ("value", "status"), [(0.9983, "not_met"), (1.0, "met"), (0.9999, "not_met")]
)
def test_the_placeholders_target_needs_exactly_one(value: float, status: str) -> None:
    block = {**NUMBERS_BLOCK, "placeholder_acc": value}

    outcome = outcome_of("placeholders", robustness_of(block=block))

    assert outcome.status == status
    assert outcome.threshold == 1.0


def test_a_one_over_units_that_were_left_out_is_unmeasured_with_the_reason() -> None:
    block = {**NUMBERS_BLOCK, "placeholder_acc": 1.0}
    entry = {"scored": 599, "excluded": 1, "total": 600}

    outcome = outcome_of(
        "placeholders", robustness_of(block=block, placeholder_acc=entry)
    )

    assert outcome.status == "unmeasured"
    assert outcome.value == 1.0
    assert (
        outcome.basis
        == "1.0000 is at least 1 on the scored units only: 1 of 600 were left out"
    )


def test_a_value_below_the_threshold_is_not_met_whatever_was_left_out() -> None:
    entry = {"scored": 590, "excluded": 10, "total": 600}

    outcome = outcome_of("placeholders", robustness_of(placeholder_acc=entry))

    assert outcome.status == "not_met"


def test_every_system_of_ours_that_has_a_block_is_judged_and_the_worse_decides() -> (
    None
):
    good = {**NUMBERS_BLOCK, "numbers_acc": 0.995}
    robustness = {**robustness_of("toy", block=good), **robustness_of("int8")}

    outcome = outcome_of(
        "numbers", robustness, systems=(("toy", "rinchen"), ("int8", "rinchen"))
    )

    assert outcome.status == "not_met"
    assert outcome.value == 0.9292
    assert outcome.basis == "int8: 0.9292 is below 0.99; toy: 0.9950 is at least 0.99"


# ---- the table --------------------------------------------------------------------------------


def test_the_outcomes_follow_the_configuration_in_order() -> None:
    config = shipped_config()

    outcomes = evaluate_targets(config, [], [], {}, suites=SUITES)

    assert [outcome.id for outcome in outcomes] == [t.id for t in config.targets]
    assert [outcome.id for outcome in outcomes] == [
        "beats_baseline",
        "off_target",
        "numbers",
        "placeholders",
    ]
    assert [outcome.description for outcome in outcomes] == [
        t.description for t in config.targets
    ]


def test_an_outcome_is_a_plain_dictionary_with_the_six_keys() -> None:
    outcome = evaluate_targets(shipped_config(), [], [], {}, suites=SUITES)[1]

    assert list(outcome.to_dict()) == [
        "id",
        "description",
        "threshold",
        "value",
        "status",
        "basis",
    ]


def test_a_status_outside_the_three_is_refused() -> None:
    with pytest.raises(ValueError, match="status"):
        TargetOutcome("x", "a target", None, None, "passed", "because")
    assert TARGET_STATUSES == ("met", "not_met", "unmeasured")


# ---- the module ------------------------------------------------------------------------------


def test_the_modules_of_the_report_import_no_reader_of_a_suite() -> None:
    package = Path(rinchen.__file__).parent / "eval"
    guarded = (
        "rinchen.eval.suites",
        "rinchen.eval.suite_rules",
        "rinchen.eval.suite_sources",
    )
    names = (
        "targets.py",
        "report_format.py",
        "report_markdown.py",
        "report_markdown_cells.py",
        "report_markdown_notes.py",
        "report_markdown_scores.py",
        "report_read.py",
        "report_read_rules.py",
        "report_read_sections.py",
        "report_view.py",
    )
    for name in names:
        tree = ast.parse((package / name).read_text(encoding="utf-8"))
        imported: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module is not None:
                imported.add(node.module)
                imported.update(f"{node.module}.{alias.name}" for alias in node.names)
        assert not [m for m in imported if m in guarded], name
