"""The `mn_Mong` source metric of D12 (spec T9): the corpus chrF of the direct arm over the
corpus chrF of the Cyrillic control arm (`retention`, never clipped, `None` when the control
scores 0), the share of rows whose two arms wrote the same text (`agreement`) and the same
two figures for the optional converted arm. The golden rows m1 to m3 and every expectation
are literals; the arms are the texts a system wrote into the target language."""

from __future__ import annotations

import dataclasses

import pytest

from rinchen.eval.robust_mong import MongScores, score_mong

MARKER = "SECRET-MARKER-91"

REFERENCES = ["This is a book.", "The water is here.", "Where is the tree?"]
GOLDEN_MONG = ["This is a book.", "The star is here.", ""]
GOLDEN_CHRF_MONG = 50.17570743156548
GOLDEN_RETENTION = 0.5017570743156547
NOT_CLIPPED_RETENTION = 100.0 / GOLDEN_CHRF_MONG
EMPTY = ["", "", ""]


def test_arms_equal_to_the_references_keep_everything() -> None:
    scores = score_mong(REFERENCES, REFERENCES, REFERENCES)

    assert scores == MongScores(
        rows=3,
        chrf_cyrl=100.0,
        chrf_mong=100.0,
        retention=1.0,
        agreement=1.0,
        chrf_converted=None,
        retention_converted=None,
    )


def test_the_golden_rows() -> None:
    scores = score_mong(REFERENCES, REFERENCES, GOLDEN_MONG)

    assert scores.rows == 3
    assert scores.chrf_cyrl == 100.0
    assert scores.chrf_mong == pytest.approx(GOLDEN_CHRF_MONG, abs=1e-9)
    assert scores.retention == pytest.approx(GOLDEN_RETENTION, abs=1e-9)
    assert scores.agreement == pytest.approx(1 / 3)
    assert scores.chrf_converted is None
    assert scores.retention_converted is None


def test_a_direct_arm_that_wrote_nothing_retains_nothing() -> None:
    scores = score_mong(REFERENCES, REFERENCES, EMPTY)

    assert scores.chrf_cyrl == 100.0
    assert scores.chrf_mong == 0.0
    assert scores.retention == 0.0
    assert scores.agreement == 0.0


def test_a_control_arm_that_wrote_nothing_has_no_retention() -> None:
    scores = score_mong(REFERENCES, EMPTY, REFERENCES)

    assert scores.chrf_cyrl == 0.0
    assert scores.chrf_mong == 100.0
    assert scores.retention is None
    assert scores.agreement == 0.0


def test_references_that_are_all_empty_have_no_retention() -> None:
    scores = score_mong(EMPTY, REFERENCES, GOLDEN_MONG)

    assert scores.chrf_cyrl == 0.0
    assert scores.chrf_mong == 0.0
    assert scores.retention is None
    assert scores.agreement == pytest.approx(1 / 3)


def test_a_retention_above_one_is_not_clipped() -> None:
    scores = score_mong(REFERENCES, GOLDEN_MONG, REFERENCES)

    assert scores.chrf_cyrl == pytest.approx(GOLDEN_CHRF_MONG, abs=1e-9)
    assert scores.chrf_mong == 100.0
    assert scores.retention is not None
    assert scores.retention > 1.0
    assert scores.retention == pytest.approx(NOT_CLIPPED_RETENTION, abs=1e-9)


def test_the_arms_are_not_swapped() -> None:
    forward = score_mong(REFERENCES, REFERENCES, GOLDEN_MONG)
    backward = score_mong(REFERENCES, GOLDEN_MONG, REFERENCES)

    assert forward.chrf_cyrl == pytest.approx(backward.chrf_mong, abs=1e-9)
    assert forward.chrf_mong == pytest.approx(backward.chrf_cyrl, abs=1e-9)
    assert forward.retention is not None
    assert backward.retention is not None
    assert forward.retention < 1.0 < backward.retention


def test_the_converted_arm_is_scored_against_the_control() -> None:
    scores = score_mong(REFERENCES, REFERENCES, REFERENCES, GOLDEN_MONG)

    assert scores.retention == 1.0
    assert scores.chrf_converted == pytest.approx(GOLDEN_CHRF_MONG, abs=1e-9)
    assert scores.retention_converted == pytest.approx(GOLDEN_RETENTION, abs=1e-9)


def test_the_converted_arm_and_the_direct_arm_do_not_mix() -> None:
    scores = score_mong(REFERENCES, REFERENCES, GOLDEN_MONG, REFERENCES)

    assert scores.chrf_mong == pytest.approx(GOLDEN_CHRF_MONG, abs=1e-9)
    assert scores.chrf_converted == 100.0
    assert scores.retention_converted == 1.0
    assert scores.retention == pytest.approx(GOLDEN_RETENTION, abs=1e-9)


def test_a_control_that_scores_little_but_not_zero_still_has_a_retention() -> None:
    nearly_empty = ["T" + "z" * 12, "", ""]

    scores = score_mong(REFERENCES, nearly_empty, REFERENCES, REFERENCES)

    assert scores.chrf_cyrl == pytest.approx(0.46040515653775327, abs=1e-9)
    assert scores.retention == pytest.approx(100.0 / 0.46040515653775327)
    assert scores.retention_converted == pytest.approx(100.0 / 0.46040515653775327)


def test_a_converted_arm_has_no_retention_when_the_control_scores_zero() -> None:
    scores = score_mong(REFERENCES, EMPTY, REFERENCES, REFERENCES)

    assert scores.chrf_cyrl == 0.0
    assert scores.chrf_converted == 100.0
    assert scores.retention_converted is None


def test_a_converted_arm_that_wrote_nothing_retains_nothing() -> None:
    scores = score_mong(REFERENCES, REFERENCES, REFERENCES, EMPTY)

    assert scores.chrf_converted == 0.0
    assert scores.retention_converted == 0.0


def test_without_a_converted_arm_its_two_figures_are_none() -> None:
    scores = score_mong(REFERENCES, REFERENCES, GOLDEN_MONG, None)

    assert scores.chrf_converted is None
    assert scores.retention_converted is None


def test_the_converted_arm_does_not_change_the_other_figures() -> None:
    without = score_mong(REFERENCES, REFERENCES, GOLDEN_MONG)
    with_arm = score_mong(REFERENCES, REFERENCES, GOLDEN_MONG, EMPTY)

    assert (
        dataclasses.replace(with_arm, chrf_converted=None, retention_converted=None)
        == without
    )


# --- agreement ----------------------------------------------------------------------------


def test_agreement_counts_the_rows_whose_two_arms_are_the_same_text() -> None:
    cyrl = ["a b c", "d e f", "g h i", "j k l"]
    mong = ["a b c", "d e f", "G h i", "j k l "]

    scores = score_mong(cyrl, cyrl, mong)

    assert scores.agreement == 0.5


def test_agreement_does_not_ask_whether_the_arms_are_right() -> None:
    wrong = ["This is a star.", "The star is here.", "Where is the star?"]

    scores = score_mong(REFERENCES, wrong, wrong)

    assert scores.agreement == 1.0
    assert scores.retention == 1.0
    assert 0.0 < scores.chrf_mong < 100.0


# --- shape and refusals -------------------------------------------------------------------


def test_the_rows_are_the_number_of_references() -> None:
    assert score_mong(["a b", "c d"], ["a b", "c d"], ["a b", "c d"]).rows == 2
    assert score_mong(["a b"], ["a b"], ["a b"]).rows == 1


def test_no_rows_is_a_value_error() -> None:
    with pytest.raises(ValueError, match="at least one row"):
        score_mong([], [], [])


@pytest.mark.parametrize(
    ("cyrl", "mong", "converted", "arm"),
    [
        (REFERENCES[:2], REFERENCES, None, "cyrl"),
        (REFERENCES, REFERENCES[:2], None, "mong"),
        (REFERENCES, REFERENCES, REFERENCES[:2], "converted"),
        (REFERENCES, REFERENCES, REFERENCES + ["more"], "converted"),
        (REFERENCES + ["more"], REFERENCES, None, "cyrl"),
        (REFERENCES, [], None, "mong"),
    ],
    ids=[
        "short_cyrl",
        "short_mong",
        "short_converted",
        "long_converted",
        "long_cyrl",
        "empty_mong",
    ],
)
def test_an_arm_that_does_not_match_the_references_is_a_value_error_naming_it(
    cyrl: list[str], mong: list[str], converted: list[str] | None, arm: str
) -> None:
    with pytest.raises(ValueError, match=arm) as raised:
        score_mong(REFERENCES, cyrl, mong, converted)

    assert f"{len(REFERENCES)} references" in str(raised.value)


def test_a_refusal_never_quotes_a_text() -> None:
    texts = [f"{MARKER} one", f"{MARKER} two"]

    with pytest.raises(ValueError) as raised:
        score_mong(texts + [f"{MARKER} three"], texts, texts)

    assert MARKER not in str(raised.value)


def test_the_scores_are_a_frozen_value_without_a_dict() -> None:
    scores = score_mong(REFERENCES, REFERENCES, REFERENCES)

    with pytest.raises(dataclasses.FrozenInstanceError):
        scores.retention = 0.0  # type: ignore[misc]
    assert not hasattr(scores, "__dict__")
    assert [field.name for field in dataclasses.fields(MongScores)] == [
        "rows",
        "chrf_cyrl",
        "chrf_mong",
        "retention",
        "agreement",
        "chrf_converted",
        "retention_converted",
    ]


def test_the_arguments_are_not_changed() -> None:
    references = list(REFERENCES)
    cyrl = list(REFERENCES)
    mong = list(GOLDEN_MONG)
    converted = list(EMPTY)

    score_mong(references, cyrl, mong, converted)

    assert (references, cyrl, mong, converted) == (
        REFERENCES,
        REFERENCES,
        GOLDEN_MONG,
        EMPTY,
    )


def test_tuples_and_lists_score_alike() -> None:
    assert score_mong(
        tuple(REFERENCES), tuple(REFERENCES), tuple(GOLDEN_MONG)
    ) == score_mong(REFERENCES, REFERENCES, GOLDEN_MONG)
