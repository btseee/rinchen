"""The blocks of `report.json` and the facts they are built from (D15 Task 6, spec P10):
`rinchen.tiny.report_blocks.build_blocks`.

The blocks are built from the facts the stages observed and from nothing else, each one from the facts
of its own stage: a block whose figures are not all there is null, and the other blocks are not touched.
The facts of a record that failed are still facts (a parity that missed its gate has its figures, an
evaluation that stopped after its first part has the figures of that part). A fact the schema does not
list is left out (the synthetic job's fingerprint names a directory and the port of a listener), and a
fact that is not of the kind the schema holds makes its block null and never an error.
"""

from __future__ import annotations

import copy
import json
import random
from collections.abc import Mapping
from types import MappingProxyType
from typing import Any

import pytest

from rinchen.tiny.report_blocks import build_blocks
from rinchen.tiny.report_schema import BLOCK_IDS, BLOCKS
from tests.support.tiny_report_inputs import golden_results
from tests.support.tiny_report_walk import golden_document


def golden_facts() -> dict[str, dict[str, Any]]:
    return {result.record.id: dict(result.facts) for result in golden_results()}


def without(
    facts: dict[str, dict[str, Any]], *stages: str
) -> dict[str, dict[str, Any]]:
    return {stage: value for stage, value in facts.items() if stage not in stages}


def nulls(blocks: Mapping[str, Any]) -> set[str]:
    return {name for name, block in blocks.items() if block is None}


def test_the_golden_facts_give_the_golden_blocks() -> None:
    assert build_blocks(golden_facts()) == golden_document()["blocks"]


def test_the_blocks_are_the_eleven_in_a_fixed_order_and_fit_the_schema() -> None:
    blocks = build_blocks(golden_facts())
    assert list(blocks) == list(BLOCK_IDS)
    BLOCKS.check(blocks, "blocks")


def test_no_facts_give_eleven_null_blocks() -> None:
    blocks = build_blocks({})
    assert list(blocks) == list(BLOCK_IDS)
    assert nulls(blocks) == set(BLOCK_IDS)


@pytest.mark.parametrize(
    ("stages", "blocks"),
    [
        (("filter",), {"filter"}),
        (("synth",), {"synth"}),
        (("converter",), {"converter"}),
        (("train",), {"train"}),
        (("parity",), {"parity"}),
        (("serve",), {"serve"}),
        (("robustness",), {"robustness"}),
        (("evaluate",), {"memorised", "heldout", "neural_seam", "int8"}),
        (("export",), {"int8"}),
        (("environment", "fetch", "corpus", "spm", "prepare"), set()),
    ],
)
def test_a_block_is_built_from_the_facts_of_its_own_stage_and_no_other(
    stages: tuple[str, ...], blocks: set[str]
) -> None:
    built = build_blocks(without(golden_facts(), *stages))
    assert nulls(built) == blocks
    untouched = golden_document()["blocks"]
    for name in set(BLOCK_IDS) - blocks:
        assert built[name] == untouched[name], name


def test_the_train_block_keeps_its_figures_when_the_averaging_never_ran() -> None:
    built = build_blocks(without(golden_facts(), "average"))
    assert built["train"]["weights_sha256"] is None
    assert built["train"]["steps"] == 1500
    assert build_blocks(golden_facts())["train"]["weights_sha256"] is not None


def test_the_facts_of_a_parity_that_missed_its_gate_still_make_the_block() -> None:
    facts = golden_facts()
    facts["parity"].update(exact=60, exact_share=60 / 70, passes=False)
    block = build_blocks(facts)["parity"]
    assert block["passes"] is False
    assert block["exact"] == 60
    assert block["exact_share"] == 60 / 70


def test_a_parity_that_wrote_no_file_has_no_block() -> None:
    facts = golden_facts()
    facts["parity"] = {}
    assert build_blocks(facts)["parity"] is None


def test_the_parity_digests_may_be_missing_from_the_facts() -> None:
    facts = golden_facts()
    del facts["parity"]["model_sha256"]
    del facts["parity"]["checkpoint_weights_sha256"]
    block = build_blocks(facts)["parity"]
    assert block["model_sha256"] is None
    assert block["checkpoint_weights_sha256"] is None


def test_an_evaluation_that_stopped_after_its_first_part_has_the_figures_of_that_part() -> (
    None
):
    facts = golden_facts()
    for key in ("heldout", "neural_seam", "report"):
        del facts["evaluate"][key]
    built = build_blocks(facts)
    assert nulls(built) == {"heldout", "neural_seam"}
    assert built["memorised"] == golden_document()["blocks"]["memorised"]
    assert built["int8"] == golden_document()["blocks"]["int8"]


def test_a_skipped_neural_part_leaves_a_null_block_and_nothing_else() -> None:
    facts = golden_facts()
    facts["evaluate"]["neural_seam"] = None
    built = build_blocks(facts)
    assert nulls(built) == {"neural_seam"}


def test_the_int8_block_needs_the_digest_of_the_int8_export() -> None:
    facts = golden_facts()
    del facts["export"]["model_sha256"]["int8"]
    assert nulls(build_blocks(facts)) == {"int8"}


def test_the_runs_are_listed_by_direction_and_then_by_system_whatever_the_order_of_the_facts() -> (
    None
):
    facts = golden_facts()
    keys = list(facts["evaluate"]["memorised"])
    random.Random(5).shuffle(keys)
    shuffled = {
        direction: dict(
            reversed(list(facts["evaluate"]["memorised"][direction].items()))
        )
        for direction in keys
    }
    facts["evaluate"]["memorised"] = shuffled
    entries = build_blocks(facts)["memorised"]
    order = [(entry["direction"], entry["system"]) for entry in entries]
    assert order == sorted(order)
    assert len(order) == 10
    assert entries == golden_document()["blocks"]["memorised"]


def test_a_run_lists_its_rows_its_three_surface_scores_and_its_statuses_and_no_neural_score() -> (
    None
):
    entry = build_blocks(golden_facts())["heldout"][0]
    assert sorted(entry) == [
        "bleu",
        "chrf",
        "chrf_pp",
        "direction",
        "rows",
        "statuses",
        "system",
    ]
    assert entry["statuses"] == {"ok": 30}


def test_the_coverage_keeps_the_counts_and_not_the_chrf_of_the_retention() -> None:
    block = build_blocks(golden_facts())["robustness"]
    for key, counts in block["coverage"].items():
        assert sorted(counts) == ["excluded", "scored", "total"], key
    assert "chrf_cyrl" not in json.dumps(block)


def test_a_value_the_robustness_run_could_not_measure_is_null() -> None:
    facts = golden_facts()
    facts["robustness"]["values"]["mong_source_retention"] = None
    block = build_blocks(facts)["robustness"]
    assert block["values"]["mong_source_retention"] is None
    assert block["values"]["numbers_acc"] == 1.0


@pytest.mark.parametrize(
    "leaked",
    [
        "fingerprint",
        "export_name",
        "chrf_cyrl",
        "max_lines",
        "manifest_sha256",
        "hypotheses_sha256",
    ],
)
def test_a_fact_the_schema_does_not_list_never_reaches_a_block(leaked: str) -> None:
    facts = golden_facts()
    facts["synth"]["fingerprint"] = "f" * 64
    facts["parity"]["manifest_sha256"] = "e" * 64
    facts["evaluate"]["hypotheses_sha256"] = "d" * 64
    assert leaked not in json.dumps(build_blocks(facts))


def test_the_synthetic_block_never_holds_the_fingerprint_of_the_job() -> None:
    facts = golden_facts()
    facts["synth"]["fingerprint"] = "f" * 64
    block = build_blocks(facts)["synth"]
    assert "fingerprint" not in block
    assert "f" * 64 not in json.dumps(block)


BROKEN = [
    ("filter", lambda f: f["filter"].update(rows="770")),
    ("filter", lambda f: f["filter"].update(rows=True)),
    ("filter", lambda f: f["filter"].update(authentic_flagged_share=1.5)),
    ("filter", lambda f: f["filter"]["planted"].pop("found")),
    ("filter", lambda f: f["filter"]["reasons"].update({"not a name": 1})),
    ("filter", lambda f: f["filter"].update(planted=None)),
    ("synth", lambda f: f["synth"].update(counts={"candidates": 1})),
    ("synth", lambda f: f["synth"].update(qe_status="skipped: " + "x" * 600)),
    ("synth", lambda f: f["synth"].update(qe_status="skipped: /tmp/run")),
    ("synth", lambda f: f.update(synth=None)),
    ("converter", lambda f: f["converter"].update(cer=float("nan"))),
    ("converter", lambda f: f["converter"].update(n=-1)),
    ("train", lambda f: f["train"].update(rows_by_direction=[1, 2])),
    ("train", lambda f: f["average"].update(weights_sha256="abc")),
    ("parity", lambda f: f["parity"].update(exact_share=1.5)),
    ("parity", lambda f: f["parity"].update(passes="yes")),
    ("parity", lambda f: f["parity"].update(model_sha256="abc")),
    ("serve", lambda f: f["serve"].update(status_codes=[200])),
    ("serve", lambda f: f["serve"].update(status_codes=None)),
    ("robustness", lambda f: f["robustness"]["values"].pop("term_acc")),
    ("robustness", lambda f: f["robustness"]["coverage"].pop("term_acc")),
    ("robustness", lambda f: f["robustness"]["values"].update(numbers_acc=2.0)),
    (
        "memorised",
        lambda f: f["evaluate"]["memorised"]["eng_Latn-khk_Cyrl"]["int8"][
            "metrics"
        ].update(chrf=None),
    ),
    (
        "memorised",
        lambda f: f["evaluate"]["memorised"]["eng_Latn-khk_Cyrl"]["int8"].update(
            statuses={}
        ),
    ),
    ("memorised", lambda f: f["evaluate"].update(memorised={})),
    (
        "heldout",
        lambda f: f["evaluate"]["heldout"]["eng_Latn-khk_Cyrl"]["int8"].pop("rows"),
    ),
    ("heldout", lambda f: f["evaluate"].update(heldout=[])),
    ("neural_seam", lambda f: f["evaluate"]["neural_seam"].pop("scores")),
    ("neural_seam", lambda f: f["evaluate"]["neural_seam"].update(rows={})),
    ("int8", lambda f: f["evaluate"]["agreement"].update(share=2.0)),
    ("int8", lambda f: f["evaluate"].pop("int8_chrf_drop")),
    ("int8", lambda f: f["export"].update(model_sha256="not a mapping")),
]


@pytest.mark.parametrize(
    ("block", "damage"),
    BROKEN,
    ids=lambda item: item if isinstance(item, str) else "damage",
)
def test_a_fact_that_is_not_what_the_schema_holds_makes_its_block_null_and_raises_nothing(
    block: str, damage: Any
) -> None:
    facts = golden_facts()
    damage(facts)
    built = build_blocks(facts)
    assert built[block] is None
    BLOCKS.check(built, "blocks")


def test_a_broken_block_does_not_touch_the_others() -> None:
    facts = golden_facts()
    facts["filter"]["rows"] = "770"
    built = build_blocks(facts)
    assert nulls(built) == {"filter"}


def test_the_blocks_share_nothing_with_the_facts() -> None:
    facts = golden_facts()
    before = copy.deepcopy(facts)
    blocks = build_blocks(facts)
    blocks["filter"]["reasons"]["duplicate"] = 99
    blocks["memorised"][0]["statuses"]["ok"] = 99
    blocks["robustness"]["values"]["numbers_acc"] = 0.0
    assert facts == before
    again = build_blocks(facts)
    assert again["filter"]["reasons"]["duplicate"] == 4


def test_the_facts_may_be_any_mapping_and_the_blocks_are_plain_dicts_and_lists() -> (
    None
):
    facts = {stage: MappingProxyType(value) for stage, value in golden_facts().items()}
    blocks = build_blocks(facts)
    assert blocks == golden_document()["blocks"]
    assert type(blocks["filter"]) is dict
    assert type(blocks["memorised"]) is list
    assert json.loads(json.dumps(blocks)) == blocks
