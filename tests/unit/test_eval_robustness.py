"""The robustness result of D12 (spec T3 to T10, T13): the metrics of Tasks 1 to 3 applied to
the rows of a suite and the hypotheses a system wrote, section by section, with one outcome per
row for the paired bootstrap of D13.

THE GOLDEN RUN is the centre: `golden_suite.json` scored with `golden_hypotheses.jsonl`. Every
number below was derived by hand in the design spike with independent prototypes
(`make_golden.txt` there), not by this code. `golden_robustness.json` is the serialisation of
that result: the literals pin the meaning, the file pins the layout. The controls, the rows
that are not the golden ones and the document are in `test_eval_robustness_rows` and
`test_eval_robustness_document`."""

from __future__ import annotations

import hashlib
import json
from typing import Any

import pytest
import sacrebleu

from rinchen.eval.robustness import RESULT_SCHEMA, robustness_block
from tests.support.robust_golden import (
    EKH,
    GOLDEN_HYPOTHESES_SHA256,
    GOLDEN_RESULT,
    GOLDEN_RESULT_BYTES,
    GOLDEN_RESULT_SHA256,
    GOLDEN_SUITE_SHA256,
    KHK,
    RUS,
    golden_result,
)

SIX_KEYS = (
    "numbers_acc",
    "placeholder_acc",
    "term_acc",
    "script_purity",
    "entity_acc",
    "mong_source_retention",
)
EN_KH, EN_RU, KH_EN = f"{EKH}-{KHK}", f"{EKH}-{RUS}", f"{KHK}-{EKH}"
# The signature sacrebleu gives its default chrF (the settings and its version), which says how
# every chrF of the section was computed.
MONG_CHRF_SIGNATURE = (
    "nrefs:1|case:mixed|eff:yes|nc:6|nw:0|space:no|version:" + sacrebleu.__version__
)


def sections() -> dict[str, Any]:
    return golden_result().to_dict()["sections"]  # type: ignore[no-any-return]


# ---- the golden run: every number of the spike ------------------------------------------


def test_the_golden_block_is_the_six_values_of_the_spike() -> None:
    block = robustness_block(golden_result())

    assert list(block) == list(SIX_KEYS)
    assert block["numbers_acc"] == 0.5
    assert block["placeholder_acc"] == 0.4
    assert block["term_acc"] == 0.75
    assert block["script_purity"] == 0.75
    assert block["entity_acc"] == 0.5714285714285714
    assert block["mong_source_retention"] == pytest.approx(0.5017570743156547, abs=1e-9)


def test_the_golden_suite_description_is_in_the_result() -> None:
    document = golden_result().to_dict()

    assert document["schema"] == RESULT_SCHEMA == "rinchen.robustness/1"
    assert document["suite"] == {
        "name": "golden",
        "sha256": GOLDEN_SUITE_SHA256,
        "rows": {
            "numbers": 8,
            "placeholders": 6,
            "entities": 4,
            "terms": 3,
            "script": 6,
            "mong": 3,
        },
    }
    assert document["system"] == {
        "name": "golden",
        "model_sha256": None,
        "manifest_sha256": None,
        "hypotheses_sha256": GOLDEN_HYPOTHESES_SHA256,
        "options": None,
    }
    assert document["robustness"] == robustness_block(golden_result())
    assert sorted(document) == [
        "robustness",
        "rows",
        "schema",
        "sections",
        "suite",
        "system",
    ]


def test_the_golden_numbers_section() -> None:
    section = sections()["numbers"]

    assert section["unit"] == "row"
    assert (section["rows"], section["scored"], section["ok"]) == (8, 6, 3)
    assert section["excluded"] == {"no_numbers": 1, "source_reference_disagree": 1}
    assert section["value"] == 0.5
    assert section["details"] == {
        "recall": 9 / 11,
        "extra_tokens": 1,
        "surface_exact_rows": 4,
        "unit_failures": 1,
        "ambiguous_rows": 0,
        "capped_rows": 0,
    }
    assert section["by_category"] == {
        "currency": {"scored": 1, "ok": 0},
        "date": {"scored": 1, "ok": 1},
        "grouped": {"scored": 1, "ok": 1},
        "int": {"scored": 2, "ok": 1},
        "percent": {"scored": 1, "ok": 0},
    }
    assert section["by_direction"] == {
        f"{EKH}-{KHK}": {"scored": 5, "ok": 2},
        f"{EKH}-{RUS}": {"scored": 1, "ok": 1},
    }
    assert section["statuses"] == {"empty": 1, "ok": 7}


def test_the_golden_placeholders_section() -> None:
    section = sections()["placeholders"]

    assert section["unit"] == "row"
    assert (section["rows"], section["scored"], section["ok"]) == (6, 5, 2)
    assert section["excluded"] == {"no_placeholders": 1}
    assert section["value"] == 0.4
    assert section["details"] == {
        "missing": 2,
        "extra": 1,
        "damaged": 2,
        "exact_rows": 2,
        "order_ok_rows": 1,
    }
    assert section["by_category"] == {
        "noun": {"scored": 4, "ok": 1},
        "reorder": {"scored": 1, "ok": 1},
    }
    assert section["by_direction"] == {EN_KH: {"scored": 5, "ok": 2}}
    assert section["statuses"] == {"ok": 6}


def test_the_golden_entities_section_counts_mentions() -> None:
    section = sections()["entities"]

    assert section["unit"] == "mention"
    assert (section["rows"], section["scored"], section["ok"]) == (4, 7, 4)
    assert section["excluded"] == {}
    assert section["value"] == 0.5714285714285714
    assert section["details"] == {"rows_all_found": 2}
    assert section["by_category"] == {
        "city": {"scored": 1, "ok": 0},
        "person_city": {"scored": 6, "ok": 4},
    }
    assert section["by_direction"] == {
        EN_KH: {"scored": 5, "ok": 2},
        KH_EN: {"scored": 2, "ok": 2},
    }
    assert section["statuses"] == {"ok": 4}


def test_the_golden_terms_section_counts_mentions_and_the_excluded_term() -> None:
    section = sections()["terms"]

    assert section["unit"] == "mention"
    assert (section["rows"], section["scored"], section["ok"]) == (3, 4, 3)
    assert section["excluded"] == {"absent_in_source": 1}
    assert section["value"] == 0.75
    assert section["details"] == {"rows_all_found": 2}
    assert section["by_category"] == {"glossary": {"scored": 4, "ok": 3}}
    assert section["by_direction"] == {EN_KH: {"scored": 4, "ok": 3}}
    assert section["statuses"] == {"ok": 3}


def test_the_golden_script_section() -> None:
    section = sections()["script"]

    assert section["unit"] == "row"
    assert (section["rows"], section["scored"], section["ok"]) == (6, 5, 3)
    assert section["excluded"] == {"no_letters": 1}
    assert section["value"] == 0.75
    assert section["details"] == {
        "mean": 0.75,
        "min": 0.25,
        "pure_share": 0.6,
        "below_floor": 2,
        "no_letters": 1,
        "floor": 0.98,
    }
    assert section["by_category"] == {
        "mong_output": {"scored": 2, "ok": 1},
        "trained": {"scored": 2, "ok": 1},
        "untrained": {"scored": 1, "ok": 1},
    }
    assert section["by_direction"] == {
        EN_KH: {"scored": 4, "ok": 2},
        f"{EKH}-{EKH}": {"scored": 1, "ok": 1},
    }
    assert section["statuses"] == {"empty": 1, "ok": 5}


def test_the_golden_mong_section() -> None:
    section = sections()["mong"]

    assert section["unit"] == "row"
    assert (section["rows"], section["scored"], section["ok"]) == (3, 3, 1)
    assert section["excluded"] == {}
    assert section["value"] == pytest.approx(0.5017570743156547, abs=1e-9)
    details = section["details"]
    assert sorted(details) == [
        "agreement",
        "chrf_converted",
        "chrf_cyrl",
        "chrf_mong",
        "chrf_signature",
        "retention_converted",
        "rows",
    ]
    assert details["chrf_signature"] == MONG_CHRF_SIGNATURE
    assert details["rows"] == 3
    assert details["chrf_cyrl"] == 100.0
    assert details["chrf_mong"] == pytest.approx(50.17570743156548, abs=1e-9)
    assert details["agreement"] == 0.3333333333333333
    assert details["chrf_converted"] is None
    assert details["retention_converted"] is None
    assert section["by_category"] == {"base": {"scored": 3, "ok": 1}}
    assert section["by_direction"] == {KH_EN: {"scored": 3, "ok": 1}}
    assert section["statuses"] == {"empty": 1, "ok": 5}


# (id, category, direction, scored, reason, ok, expected, found, value)
GOLDEN_ROWS: dict[str, list[tuple[Any, ...]]] = {
    "numbers": [
        ("n1", "grouped", EN_KH, True, None, True, 1, 1, None),
        ("n2", "date", EN_RU, True, None, True, 5, 5, None),
        ("n3", "currency", EN_KH, True, None, False, 1, 1, None),
        ("n4", "int", EN_KH, True, None, False, 1, 0, None),
        ("n5", "none", EN_KH, False, "no_numbers", None, None, None, None),
        (
            "n6",
            "words",
            EN_KH,
            False,
            "source_reference_disagree",
            None,
            None,
            None,
            None,
        ),
        ("n7", "int", EN_KH, True, None, True, 2, 2, None),
        ("n8", "percent", EN_KH, True, None, False, 1, 0, None),
    ],
    "placeholders": [
        ("p1", "noun", EN_KH, True, None, True, 1, 1, None),
        ("p2", "reorder", EN_KH, True, None, True, 2, 2, None),
        ("p3", "noun", EN_KH, True, None, False, 1, 0, None),
        ("p4", "noun", EN_KH, True, None, False, 1, 1, None),
        ("p5", "noun", EN_KH, True, None, False, 1, 0, None),
        ("p6", "none", EN_KH, False, "no_placeholders", None, None, None, None),
    ],
    "entities": [
        ("e1", "person_city", EN_KH, True, None, True, 2, 2, None),
        ("e2", "person_city", EN_KH, True, None, False, 2, 0, None),
        ("e3", "person_city", KH_EN, True, None, True, 2, 2, None),
        ("e4", "city", EN_KH, True, None, False, 1, 0, None),
    ],
    "terms": [
        ("t1", "glossary", EN_KH, True, None, True, 1, 1, None),
        ("t2", "glossary", EN_KH, True, None, True, 2, 2, None),
        ("t3", "glossary", EN_KH, True, None, False, 1, 0, None),
    ],
    "script": [
        ("s1", "trained", EN_KH, True, None, True, None, None, 1.0),
        ("s2", "trained", EN_KH, True, None, False, None, None, 0.25),
        (
            "s3",
            "untrained",
            "und-khk_Cyrl",
            False,
            "no_letters",
            None,
            None,
            None,
            None,
        ),
        ("s4", "mong_output", EN_KH, True, None, True, None, None, 1.0),
        ("s5", "mong_output", EN_KH, True, None, False, None, None, 0.5),
        ("s6", "untrained", f"{EKH}-{EKH}", True, None, True, None, None, 1.0),
    ],
    "mong": [
        ("m1", "base", KH_EN, True, None, True, None, None, 100.0),
        ("m2", "base", KH_EN, True, None, False, None, None, 52.9619),
        ("m3", "base", KH_EN, True, None, False, None, None, 0.0),
    ],
}
ROW_FIELDS = (
    "id",
    "category",
    "direction",
    "scored",
    "reason",
    "ok",
    "expected",
    "found",
    "value",
    "value_cyrl",
)


@pytest.mark.parametrize("section", list(GOLDEN_ROWS))
def test_every_row_of_the_golden_run_has_its_outcome_in_file_order(
    section: str,
) -> None:
    outcomes = golden_result().to_dict()["rows"][section]

    assert all(sorted(outcome) == sorted(ROW_FIELDS) for outcome in outcomes)
    found = [tuple(outcome[field] for field in ROW_FIELDS) for outcome in outcomes]
    expected = GOLDEN_ROWS[section]
    assert [f[:8] for f in found] == [e[:8] for e in expected]
    for got, want in zip(found, expected, strict=True):
        if want[8] is None:
            assert got[8] is None
        else:
            assert got[8] == pytest.approx(want[8], abs=1e-3)


def test_a_mong_row_stores_the_chrf_of_both_arms_and_no_other_row_stores_one() -> None:
    outcomes = golden_result().to_dict()["rows"]

    # Independent of this code: sacrebleu's sentence chrF of each arm against the reference
    # (the Cyrillic arm of the golden hypotheses is the reference itself).
    assert [o["value"] for o in outcomes["mong"]] == pytest.approx(
        [100.0, 52.96190894473853, 0.0], abs=1e-9
    )
    assert [o["value_cyrl"] for o in outcomes["mong"]] == [100.0, 100.0, 100.0]
    for section, rows in outcomes.items():
        if section != "mong":
            assert [o["value_cyrl"] for o in rows] == [None] * len(rows), section


def test_the_golden_file_is_the_serialisation_of_the_golden_run() -> None:
    assert golden_result().to_bytes() == GOLDEN_RESULT.read_bytes()


def test_the_golden_file_is_the_one_whose_size_and_digest_are_pinned() -> None:
    data = GOLDEN_RESULT.read_bytes()

    assert len(data) == GOLDEN_RESULT_BYTES
    assert hashlib.sha256(data).hexdigest() == GOLDEN_RESULT_SHA256


def test_the_golden_file_is_the_result_as_json() -> None:
    result = golden_result()

    assert json.loads(GOLDEN_RESULT.read_bytes()) == result.to_dict()
    assert json.loads(result.to_bytes()) == result.to_dict()
