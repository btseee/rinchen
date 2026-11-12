"""The check of the back-translated rows that the gate makes on a report (`tests/support/tiny_gate.py`,
`synth_problems`, used by R1 of `tests/integration/test_tiny_gate.py`).

The `synth` stage once passed with nothing selected and no row made (the loopback server answered an empty text for
every source), the training then ran with no back-translated row and no threshold noticed. The gate reads the
`synth` block of the report it ran for (`counts.selected` equal to `counts.sources`, `rows` above zero and equal to
`counts.selected`) and the `train` block (`bt_rows` above zero). The helper is tested here on the golden report, which
is what a healthy run writes, and on copies of it with one figure spoiled at a time, so that an assertion over the
real run cannot be vacuous: it names the key it did not find as well as the figure that is wrong.
"""

from __future__ import annotations

import copy
from typing import Any

import pytest

from tests.support.tiny_gate import synth_problems
from tests.support.tiny_report_walk import golden_document


def _spoiled(path: tuple[str, ...], value: object) -> dict[str, Any]:
    document = copy.deepcopy(golden_document())
    holder: Any = document
    for key in path[:-1]:
        holder = holder[key]
    holder[path[-1]] = value
    return document


def _without(path: tuple[str, ...]) -> dict[str, Any]:
    document = copy.deepcopy(golden_document())
    holder: Any = document
    for key in path[:-1]:
        holder = holder[key]
    del holder[path[-1]]
    return document


def test_the_golden_report_has_what_the_gate_looks_for() -> None:
    document = golden_document()
    synth = document["blocks"]["synth"]

    assert synth["counts"]["selected"] == synth["counts"]["sources"] == 64
    assert synth["rows"] == 64
    assert document["blocks"]["train"]["bt_rows"] > 0


def test_a_healthy_report_has_no_problem() -> None:
    assert synth_problems(golden_document()) == []


@pytest.mark.parametrize(
    ("path", "value", "words"),
    [
        (("blocks", "synth", "counts", "selected"), 0, "selected 0 of 64"),
        (("blocks", "synth", "counts", "selected"), 63, "selected 63 of 64"),
        (("blocks", "synth", "counts", "sources"), 65, "selected 64 of 65"),
        (("blocks", "synth", "rows"), 0, "no row"),
        (("blocks", "synth", "rows"), 63, "is not the selected count"),
        (("blocks", "synth", "rows"), 65, "is not the selected count"),
        (("blocks", "train", "bt_rows"), 0, "no back-translated row"),
        (("blocks", "train", "bt_rows"), -1, "no back-translated row"),
    ],
    ids=[
        "nothing selected",
        "one source not selected",
        "a source more than selected",
        "no row",
        "a row fewer",
        "a row more",
        "no back-translated row in the training",
        "a negative count",
    ],
)
def test_a_report_with_one_figure_spoiled_has_the_problem_that_figure_makes(
    path: tuple[str, ...], value: object, words: str
) -> None:
    problems = synth_problems(_spoiled(path, value))

    assert any(words in problem for problem in problems), problems


def test_the_report_of_a_server_that_answered_nothing_has_every_problem() -> None:
    document = golden_document()
    document["blocks"]["synth"]["counts"]["selected"] = 0
    document["blocks"]["synth"]["counts"]["no_candidates"] = 64
    document["blocks"]["synth"]["rows"] = 0
    document["blocks"]["train"]["bt_rows"] = 0

    problems = synth_problems(document)

    assert len(problems) >= 3
    assert all(isinstance(problem, str) and problem for problem in problems)


@pytest.mark.parametrize(
    "path",
    [
        ("blocks", "synth", "counts", "selected"),
        ("blocks", "synth", "counts", "sources"),
        ("blocks", "synth", "counts"),
        ("blocks", "synth", "rows"),
        ("blocks", "synth"),
        ("blocks", "train", "bt_rows"),
        ("blocks", "train"),
        ("blocks",),
    ],
)
def test_a_key_that_is_missing_is_a_problem_and_not_a_pass_or_a_crash(
    path: tuple[str, ...],
) -> None:
    problems = synth_problems(_without(path))

    assert problems
    assert all(isinstance(problem, str) for problem in problems)


@pytest.mark.parametrize(
    "path", [("blocks", "synth"), ("blocks", "train"), ("blocks", "synth", "counts")]
)
def test_a_block_that_is_null_is_a_problem(path: tuple[str, ...]) -> None:
    assert synth_problems(_spoiled(path, None))


@pytest.mark.parametrize(
    "path",
    [
        ("blocks", "synth", "counts", "selected"),
        ("blocks", "synth", "counts", "sources"),
        ("blocks", "synth", "rows"),
        ("blocks", "train", "bt_rows"),
    ],
)
@pytest.mark.parametrize("value", [True, "64", 64.0, None])
def test_a_figure_that_is_not_a_whole_number_is_a_problem(
    path: tuple[str, ...], value: object
) -> None:
    problems = synth_problems(_spoiled(path, value))

    assert any("is missing or not a whole number" in problem for problem in problems)


def test_no_text_of_the_report_is_quoted_in_a_problem() -> None:
    document = _spoiled(("blocks", "synth", "generator"), "a-text-that-must-not-leak")
    document["blocks"]["synth"]["counts"]["selected"] = 0

    assert all("a-text-that-must-not-leak" not in p for p in synth_problems(document))
