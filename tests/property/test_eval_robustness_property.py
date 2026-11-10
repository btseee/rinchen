"""Property tests for the robustness result and the hypotheses file of D12: whatever a system
wrote, scoring never raises, never loses a row or a mention (`scored` plus what was excluded is
everything the suite holds), keeps `ok <= scored`, gives only finite values or `None`, and
serialises to a document that parses back to the same dict; a hypotheses file survives its
bytes in both directions."""

from __future__ import annotations

import json
import math
from typing import Any

from hypothesis import given, settings
from hypothesis import strategies as st

from rinchen.eval.robust_hypotheses import (
    Hypothesis,
    hypotheses_from_bytes,
    hypotheses_to_bytes,
)
from rinchen.eval.robust_suite import RequestKey
from rinchen.eval.robustness import compute_robustness, robustness_block
from tests.support.robust_golden import (
    GOLDEN_SYSTEM,
    golden_requests,
    golden_suite,
    golden_suite_sha256,
)

KEYS = [request.key for request in golden_requests()]
STATUSES = st.sampled_from(("ok", "empty", "too_long", "truncated", "no_output"))
# Characters that make the metrics work: digits and their separators, currency signs, the
# bracket glyphs, letters of three scripts, spaces.
INTERESTING = "0123456789 .,:/-+%$()abабвгдеж" + "".join(
    chr(code) for code in (0x20AE, 0x27E6, 0x27E7, 0xA0, 0x202F, 0x2009, 0x1821, 0x1828)
)
TEXTS = st.one_of(
    st.text(alphabet=INTERESTING, max_size=40),
    st.text(max_size=40),
)
ANSWERS = st.lists(st.tuples(TEXTS, STATUSES), min_size=len(KEYS), max_size=len(KEYS))
SUITE = golden_suite()
MENTIONS = {
    "entities": sum(len(r.entities) for r in SUITE.sections["entities"]),
    "terms": sum(len(r.terms) for r in SUITE.sections["terms"]),
}


def hypotheses_of(answers: list[tuple[str, str]]) -> dict[RequestKey, Hypothesis]:
    return {
        key: Hypothesis(text, status)  # type: ignore[arg-type]
        for key, (text, status) in zip(KEYS, answers, strict=True)
    }


def finite_or_none(value: object) -> bool:
    return value is None or (isinstance(value, int | float) and math.isfinite(value))


@settings(max_examples=60)
@given(ANSWERS)
def test_scoring_any_hypotheses_never_raises_and_loses_nothing(
    answers: list[tuple[str, str]],
) -> None:
    result = compute_robustness(
        SUITE,
        hypotheses_of(answers),
        suite_sha256=golden_suite_sha256(),
        system=GOLDEN_SYSTEM,
    )

    document: dict[str, Any] = result.to_dict()

    assert list(robustness_block(result)) == [
        "numbers_acc",
        "placeholder_acc",
        "term_acc",
        "script_purity",
        "entity_acc",
        "mong_source_retention",
    ]
    for name, section in document["sections"].items():
        units = MENTIONS.get(name, section["rows"])
        assert 0 <= section["ok"] <= section["scored"]
        assert section["scored"] + sum(section["excluded"].values()) == units
        assert finite_or_none(section["value"])
        assert all(
            (v is None or isinstance(v, str))
            if k == "chrf_signature"
            else finite_or_none(v)
            for k, v in section["details"].items()
        )
        assert len(document["rows"][name]) == section["rows"]
        assert sum(section["statuses"].values()) == len(
            [k for k in KEYS if k.section == name]
        )
        for outcome in document["rows"][name]:
            assert outcome["scored"] == (
                outcome["reason"] is None and outcome["ok"] is not None
            )
            assert finite_or_none(outcome["value"])


@settings(max_examples=60)
@given(ANSWERS)
def test_the_document_parses_back_to_the_same_dict_and_is_deterministic(
    answers: list[tuple[str, str]],
) -> None:
    def score() -> bytes:
        return compute_robustness(
            SUITE,
            hypotheses_of(answers),
            suite_sha256=golden_suite_sha256(),
            system=GOLDEN_SYSTEM,
        ).to_bytes()

    data = score()

    assert data == score()
    assert (
        json.loads(data)
        == compute_robustness(
            SUITE,
            hypotheses_of(answers),
            suite_sha256=golden_suite_sha256(),
            system=GOLDEN_SYSTEM,
        ).to_dict()
    )


@given(ANSWERS)
def test_a_hypotheses_file_survives_its_bytes_in_both_directions(
    answers: list[tuple[str, str]],
) -> None:
    hypotheses = hypotheses_of(answers)

    data = hypotheses_to_bytes(hypotheses)
    back = hypotheses_from_bytes(data, golden_requests())

    assert back == hypotheses
    assert list(back) == KEYS
    assert hypotheses_to_bytes(back) == data
