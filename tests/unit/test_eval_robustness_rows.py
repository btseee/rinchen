"""The robustness result of D12 on rows that are not the golden ones (spec T3 to T9, T13):
controls on the golden rows (a perfect system, a rotated one, one that writes the wrong script),
what a section with nothing scored says, how a row is read (the language of the reference and
of the source, units, order, the `und` direction), the arms of the `mong` section and the rule
that the hypotheses are exactly the requests of the suite. The golden run is in
`test_eval_robustness`."""

from __future__ import annotations

from collections.abc import Mapping

import pytest
import sacrebleu

from rinchen.errors import DataError
from rinchen.eval.robust_hypotheses import Hypothesis
from rinchen.eval.robust_suite import RequestKey, Suite, TermPair
from rinchen.eval.robustness import compute_robustness, robustness_block
from rinchen.spm.symbols import user_defined_symbols
from tests.support.robust_golden import (
    EKH,
    GOLDEN_SYSTEM,
    KHK,
    MONGOLIAN,
    RUS,
    golden_hypotheses,
    golden_result,
    golden_suite,
    golden_suite_sha256,
    key,
    row,
    run,
    suite_of,
    texts,
)

MARKER = "SECRET-MARKER-63"


# ---- controls on the golden rows -----------------------------------------------------------


def reference_hypotheses() -> dict[RequestKey, Hypothesis]:
    """What a perfect system writes: the reference of every row (the golden hypotheses
    stand in for the rows that have none, `script`)."""
    suite = golden_suite()
    found = golden_hypotheses()
    for name, rows in suite.sections.items():
        for item in rows:
            if item.reference is None:
                continue
            found[key(name, item.id, "mong" if name == "mong" else "main")] = (
                Hypothesis(item.reference, "ok")
            )
            if name == "mong":
                found[key(name, item.id, "cyrl")] = Hypothesis(item.reference, "ok")
    return found


def blocks(hypotheses: Mapping[RequestKey, Hypothesis]) -> dict[str, float | None]:
    return robustness_block(
        compute_robustness(
            golden_suite(),
            hypotheses,
            suite_sha256=golden_suite_sha256(),
            system=GOLDEN_SYSTEM,
        )
    )


def test_hypotheses_equal_to_the_references_give_one_on_every_reference_metric() -> (
    None
):
    block = blocks(reference_hypotheses())

    assert block["numbers_acc"] == 1.0
    assert block["placeholder_acc"] == 1.0
    assert block["entity_acc"] == 1.0
    assert block["term_acc"] == 1.0
    assert block["mong_source_retention"] == 1.0


def rotated(
    hypotheses: Mapping[RequestKey, Hypothesis],
) -> dict[RequestKey, Hypothesis]:
    """The texts of every section moved one place on inside it (the `mong` arm only)."""
    out = dict(hypotheses)
    by_section: dict[str, list[RequestKey]] = {}
    for found_key in hypotheses:
        if found_key.section == "mong" and found_key.arm != "mong":
            continue
        by_section.setdefault(found_key.section, []).append(found_key)
    for keys in by_section.values():
        moved = [hypotheses[k] for k in keys[1:]] + [hypotheses[keys[0]]]
        out.update(zip(keys, moved, strict=True))
    return out


def test_the_texts_of_a_section_moved_one_place_on_lower_each_metric() -> None:
    block = blocks(rotated(reference_hypotheses()))

    assert block["numbers_acc"] == 0.0
    assert block["placeholder_acc"] == 0.0
    assert block["entity_acc"] == 3 / 7
    assert block["term_acc"] == 0.25
    assert block["mong_source_retention"] is not None
    assert block["mong_source_retention"] < 0.5


def test_a_system_that_writes_the_wrong_script_scores_what_it_wrote() -> None:
    wrong = {
        found_key: Hypothesis("This is a test.", "ok")
        for found_key in golden_hypotheses()
        if found_key.section == "script"
    }

    block = blocks({**golden_hypotheses(), **wrong})

    # Five of the six rows ask for Mongolian (Cyrillic or traditional) and the sixth for English.
    assert block["script_purity"] == 1 / 6


def test_a_suite_with_one_section_has_none_for_the_other_five_values() -> None:
    suite = suite_of(numbers=(row("a1"),))

    document = run(suite, texts(a1="Үнэ 15."))

    assert document["robustness"] == {
        "numbers_acc": 1.0,
        "placeholder_acc": None,
        "term_acc": None,
        "script_purity": None,
        "entity_acc": None,
        "mong_source_retention": None,
    }
    assert list(document["sections"]) == ["numbers"]
    assert list(document["rows"]) == ["numbers"]
    assert document["suite"]["rows"] == {"numbers": 1}


def test_a_section_the_suite_holds_empty_is_present_with_nothing_scored() -> None:
    suite = suite_of(numbers=(row("a1"),), mong=())

    document = run(suite, texts(a1="Үнэ 15."))

    assert document["sections"]["mong"]["rows"] == 0
    assert document["sections"]["mong"]["scored"] == 0
    assert document["sections"]["mong"]["value"] is None
    assert document["sections"]["mong"]["details"] == {
        "rows": 0,
        "chrf_cyrl": None,
        "chrf_mong": None,
        "chrf_signature": None,
        "agreement": None,
        "chrf_converted": None,
        "retention_converted": None,
    }
    assert document["robustness"]["mong_source_retention"] is None
    assert document["rows"]["mong"] == []
    assert document["suite"]["rows"] == {"numbers": 1, "mong": 0}


# ---- nothing scored is null, never 1.0 or 0 ---------------------------------------------


def test_a_numbers_section_with_only_excluded_rows_has_no_value() -> None:
    suite = suite_of(
        numbers=(
            row("a1", source="No numbers.", reference="Тоо алга."),
            row("a2", source="About 3 million.", reference="3 000 000."),
        )
    )

    document = run(suite, texts(a1="x", a2="y"))

    section = document["sections"]["numbers"]
    assert (section["rows"], section["scored"], section["ok"]) == (2, 0, 0)
    assert section["excluded"] == {"no_numbers": 1, "source_reference_disagree": 1}
    assert section["value"] is None
    assert section["details"]["recall"] is None
    assert document["robustness"]["numbers_acc"] is None
    assert section["by_category"] == {}


def test_a_script_section_with_only_rows_without_letters_has_no_value() -> None:
    suite = suite_of(script=(row("s1", reference=None, script=None),))
    hypotheses = {key("script", "s1"): Hypothesis("123 ...", "ok")}

    document = run(suite, hypotheses)

    section = document["sections"]["script"]
    assert section["scored"] == 0
    assert section["excluded"] == {"no_letters": 1}
    assert section["value"] is None
    assert section["details"]["mean"] is None
    assert section["details"]["min"] is None
    assert section["details"]["pure_share"] is None
    assert section["details"]["below_floor"] == 0
    assert document["robustness"]["script_purity"] is None


def test_a_terms_section_whose_terms_are_all_absent_has_no_value() -> None:
    suite = suite_of(
        terms=(
            row(
                "t1",
                source="The chair is here.",
                reference=None,
                terms=(TermPair("lamp", "чийдэн"),),
            ),
        )
    )

    document = run(suite, {key("terms", "t1"): Hypothesis("Сандал", "ok")})

    section = document["sections"]["terms"]
    assert section["scored"] == 0
    assert section["excluded"] == {"absent_in_source": 1}
    assert section["value"] is None
    assert document["rows"]["terms"][0]["reason"] == "absent_in_source"
    assert document["rows"]["terms"][0]["scored"] is False
    assert document["robustness"]["term_acc"] is None


def test_a_placeholders_section_with_only_excluded_rows_has_no_value() -> None:
    suite = suite_of(placeholders=(row("p1", source="None.", reference="Алга."),))

    document = run(suite, {key("placeholders", "p1"): Hypothesis("x", "ok")})

    assert document["sections"]["placeholders"]["value"] is None
    assert document["sections"]["placeholders"]["excluded"] == {"no_placeholders": 1}
    assert document["robustness"]["placeholder_acc"] is None


def test_a_retention_is_none_when_the_control_arm_scores_nothing() -> None:
    mong_row = row("m1", src_lang=KHK, tgt_lang=EKH, source_mong="x", reference="Book.")
    suite = suite_of(mong=(mong_row,))
    hypotheses = {
        key("mong", "m1", "cyrl"): Hypothesis("", "empty"),
        key("mong", "m1", "mong"): Hypothesis("Book.", "ok"),
    }

    document = run(suite, hypotheses)

    assert document["sections"]["mong"]["details"]["chrf_cyrl"] == 0.0
    assert document["sections"]["mong"]["value"] is None
    assert document["robustness"]["mong_source_retention"] is None


# ---- how a row is read ---------------------------------------------------------------------


def test_a_number_is_judged_in_the_language_of_the_target_and_the_source_in_its_own() -> (
    None
):
    suite = suite_of(
        numbers=(
            row(
                "a1",
                tgt_lang=RUS,
                source="It is 1,250.",
                reference="Это 1.250.",
                category="grouped",
            ),
            row(
                "a2",
                tgt_lang=RUS,
                source="It is 1.250.",
                reference="Это 1,25.",
                category="decimal",
            ),
        )
    )

    document = run(suite, texts(a1="Это 1250.", a2="Это 1,25."))

    section = document["sections"]["numbers"]
    assert (section["scored"], section["ok"]) == (2, 2)
    assert section["excluded"] == {}


def test_a_unit_the_reference_states_fails_the_row_and_is_counted() -> None:
    suite = suite_of(numbers=(row("a1", source="It costs 5$.", reference="Үнэ 5$."),))

    document = run(suite, texts(a1="Үнэ 5."))

    section = document["sections"]["numbers"]
    assert section["ok"] == 0
    assert section["details"]["unit_failures"] == 1
    assert section["details"]["recall"] == 1.0


def test_the_rows_keep_the_order_of_the_file_whatever_their_ids() -> None:
    suite = suite_of(numbers=(row("n9"), row("n10"), row("a2")))

    document = run(suite, texts(n9="Үнэ 15.", n10="Үнэ 16.", a2="Үнэ 15."))

    assert [outcome["id"] for outcome in document["rows"]["numbers"]] == [
        "n9",
        "n10",
        "a2",
    ]


def test_a_row_without_a_source_language_is_in_the_und_direction() -> None:
    golden = golden_hypotheses()
    golden[key("script", "s3")] = Hypothesis("Ус", "ok")

    document = run(golden_suite(), golden)

    section = document["sections"]["script"]
    assert section["scored"] == 6
    assert section["excluded"] == {}
    assert section["by_direction"]["und-khk_Cyrl"] == {"scored": 1, "ok": 1}


def test_a_partly_excluded_term_row_is_scored_on_its_other_terms() -> None:
    document = golden_result().to_dict()

    third = document["rows"]["terms"][2]
    assert (third["scored"], third["reason"], third["expected"]) == (True, None, 1)


def test_the_status_of_every_hypothesis_of_a_section_is_counted_even_when_excluded() -> (
    None
):
    suite = suite_of(numbers=(row("a1", source="None.", reference="Алга."), row("a2")))
    hypotheses = {
        key("numbers", "a1"): Hypothesis("", "no_output"),
        key("numbers", "a2"): Hypothesis("Үнэ 15", "truncated"),
    }

    document = run(suite, hypotheses)

    assert document["sections"]["numbers"]["statuses"] == {
        "no_output": 1,
        "truncated": 1,
    }
    assert document["sections"]["numbers"]["ok"] == 1


def test_a_hypothesis_with_a_status_that_is_not_ok_is_scored_on_its_text() -> None:
    suite = suite_of(numbers=(row("a1"),))

    ok = run(suite, {key("numbers", "a1"): Hypothesis("Үнэ 15.", "truncated")})

    assert ok["sections"]["numbers"]["ok"] == 1


def test_a_reason_that_did_not_occur_is_absent_and_not_zero() -> None:
    suite = suite_of(
        terms=(
            row(
                "t1",
                source="The chair is here.",
                reference=None,
                terms=(TermPair("chair", "сандал"),),
            ),
        )
    )

    document = run(suite, {key("terms", "t1"): Hypothesis("Сандал", "ok")})

    assert document["sections"]["terms"]["excluded"] == {}
    assert document["sections"]["terms"]["scored"] == 1


def test_a_placeholder_kept_beside_a_damaged_bracket_is_not_exact() -> None:
    symbol = user_defined_symbols()[-100:][3]
    suite = suite_of(
        placeholders=(
            row(
                "p1",
                source=f"Where is {symbol}?",
                reference=f"{symbol} хаана байна.",
                category="noun",
            ),
        )
    )
    hypotheses = {
        key("placeholders", "p1"): Hypothesis(f"{symbol} хаана {symbol[-1]}", "ok")
    }

    document = run(suite, hypotheses)

    section = document["sections"]["placeholders"]
    assert (section["scored"], section["ok"]) == (1, 0)
    assert section["details"]["damaged"] == 1
    assert section["details"]["missing"] == section["details"]["extra"] == 0
    assert document["rows"]["placeholders"][0]["found"] == 1


# ---- the arms of mong ------------------------------------------------------------------------


def mong_suite() -> Suite:
    rows = tuple(
        row(
            f"m{i}",
            src_lang=KHK,
            tgt_lang=EKH,
            source="Энэ.",
            source_mong=MONGOLIAN,
            reference=reference,
            category="base",
        )
        for i, reference in enumerate(
            ("This is a book.", "The water is here."), start=1
        )
    )
    return suite_of(mong=rows)


def arms(**by_arm: list[str]) -> dict[RequestKey, Hypothesis]:
    found: dict[RequestKey, Hypothesis] = {}
    for arm, texts_of_arm in by_arm.items():
        for i, text in enumerate(texts_of_arm, start=1):
            found[key("mong", f"m{i}", arm)] = Hypothesis(text, "ok")
    return found


def test_the_cyrillic_arm_of_a_row_is_scored_beside_the_direct_one() -> None:
    suite = mong_suite()
    hypotheses = arms(
        cyrl=["This is a book.", "The star is here."],
        mong=["This is a book.", "The water is here."],
    )

    document = run(suite, hypotheses)

    # sacrebleu's sentence chrF of "The star is here." against "The water is here.".
    rows = document["rows"]["mong"]
    assert [row["value"] for row in rows] == [100.0, 100.0]
    assert [row["value_cyrl"] for row in rows] == [100.0, 52.96190894473853]
    assert [row["ok"] for row in rows] == [True, False]


def test_the_signature_of_the_chrf_that_scored_the_arms_is_recorded() -> None:
    document = run(mong_suite(), arms(cyrl=["", ""], mong=["This is a book.", ""]))

    signature = document["sections"]["mong"]["details"]["chrf_signature"]

    assert signature == (
        "nrefs:1|case:mixed|eff:yes|nc:6|nw:0|space:no|version:" + sacrebleu.__version__
    )
    assert document["sections"]["mong"]["value"] is None


def test_the_arms_of_a_row_agree_when_they_wrote_the_same_text_not_the_same_score() -> (
    None
):
    suite = mong_suite()
    same_score = arms(
        cyrl=["This is a book.", "The water is here."],
        mong=["This is  a book.", "The water is here."],
    )

    document = run(suite, same_score)

    outcomes = document["rows"]["mong"]
    assert outcomes[0]["value"] == outcomes[1]["value"] == 100.0
    assert [outcome["ok"] for outcome in outcomes] == [False, True]
    assert document["sections"]["mong"]["details"]["agreement"] == 0.5
    assert document["sections"]["mong"]["ok"] == 1


def test_the_retention_is_not_clipped_at_one() -> None:
    document = run(
        mong_suite(),
        arms(
            cyrl=["This is book.", "The water here."],
            mong=["This is a book.", "The water is here."],
        ),
    )

    section = document["sections"]["mong"]
    assert section["value"] > 1.0
    assert section["value"] == (
        section["details"]["chrf_mong"] / section["details"]["chrf_cyrl"]
    )


def test_a_converted_arm_adds_its_own_figures_and_its_statuses() -> None:
    perfect = ["This is a book.", "The water is here."]
    hypotheses = arms(cyrl=perfect, mong=perfect, converted=["This is book.", "Water."])

    document = run(mong_suite(), hypotheses)

    section = document["sections"]["mong"]
    details = section["details"]
    assert details["chrf_converted"] is not None
    assert details["chrf_converted"] < details["chrf_cyrl"] == 100.0
    assert details["retention_converted"] == (
        details["chrf_converted"] / details["chrf_cyrl"]
    )
    assert section["value"] == 1.0
    assert section["statuses"] == {"ok": 6}


def test_a_converted_arm_for_some_rows_only_is_refused_naming_the_missing_one() -> None:
    perfect = ["This is a book.", "The water is here."]
    hypotheses = arms(cyrl=perfect, mong=perfect)
    hypotheses[key("mong", "m1", "converted")] = Hypothesis("This is a book.", "ok")

    with pytest.raises(DataError) as info:
        run(mong_suite(), hypotheses)

    assert "section 'mong', id 'm2', arm 'converted'" in str(info.value)


# ---- the hypotheses must be exactly the requests -------------------------------------------


def test_a_missing_hypothesis_is_refused_by_name() -> None:
    hypotheses = golden_hypotheses()
    del hypotheses[key("placeholders", "p3")]

    with pytest.raises(DataError) as info:
        run(golden_suite(), hypotheses)

    assert "section 'placeholders', id 'p3', arm 'main'" in str(info.value)
    assert "1 request" in str(info.value)


@pytest.mark.parametrize(
    "extra",
    [
        key("numbers", "zz9"),
        key("numbers", "n1", "mong"),
        key("terms", "t1", "converted"),
        key("mong", "zz9", "converted"),
    ],
)
def test_a_hypothesis_for_a_request_the_suite_does_not_make_is_refused_by_name(
    extra: RequestKey,
) -> None:
    hypotheses = golden_hypotheses()
    hypotheses[extra] = Hypothesis("x", "ok")

    with pytest.raises(DataError) as info:
        run(golden_suite(), hypotheses)

    assert f"section {extra.section!r}, id {extra.id!r}, arm {extra.arm!r}" in str(
        info.value
    )


def test_an_odd_name_in_a_hypothesis_key_is_not_quoted() -> None:
    hypotheses = golden_hypotheses()
    hypotheses[RequestKey(MARKER, f"{MARKER}!", MARKER)] = Hypothesis("x", "ok")

    with pytest.raises(DataError) as info:
        run(golden_suite(), hypotheses)

    assert MARKER not in str(info.value)


def test_the_order_of_the_mapping_does_not_change_the_result() -> None:
    forward = golden_hypotheses()
    backward = dict(reversed(list(forward.items())))

    assert run(golden_suite(), backward) == run(golden_suite(), forward)
