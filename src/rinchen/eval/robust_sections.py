"""The scoring of each section of a robustness suite (D12, spec T3 to T9, T13): the metrics of
`robust_numbers`, `robust_placeholders`, `robust_spans`, `robust_script` and `robust_mong`
applied to the rows of one section and the hypotheses a system wrote, giving a `SectionResult`
(`robust_outcomes`).

A numbers or placeholders row is scored when its source and its reference agree (else it is
counted under the reason it was left out), an entities or terms row counts its mentions, a script
row is scored when its text has a letter, and the `mong` section scores its arms together."""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Sequence

from rinchen.eval.robust_mong import MongScores, score_mong
from rinchen.eval.robust_numbers import numbers_agree, numbers_in, score_numbers
from rinchen.eval.robust_outcomes import (
    Details,
    Hypotheses,
    SectionResult,
    Tally,
    build_section,
    excluded_outcome,
    scored_outcome,
)
from rinchen.eval.robust_placeholders import (
    placeholders_agree,
    placeholders_in,
    score_placeholders,
)
from rinchen.eval.robust_script import PURITY_FLOOR, score_script
from rinchen.eval.robust_spans import EntityTable, score_entities, score_terms
from rinchen.eval.robust_suite import RequestKey, SuiteRow
from rinchen.metrics.bleu_chrf import chrf

__all__ = ["score_section"]


def score_section(
    section: str,
    rows: Sequence[SuiteRow],
    table: EntityTable,
    hypotheses: Hypotheses,
) -> SectionResult:
    """Score the `rows` of `section` with the hypotheses of the system, which must hold one
    for every request of the section (`robustness.compute_robustness` checks that).

    Raises:
        ValueError: a row lacks what its section needs (a reference, an entity, a term).
        DataError: an entity has no spelling for the language of a row.
        ConfigError: a row asks for a script its target cannot have.
    """
    if section == "entities":
        return _entities(rows, table, hypotheses)
    scorers = {
        "numbers": _numbers,
        "placeholders": _placeholders,
        "terms": _terms,
        "script": _script,
        "mong": _mong,
    }
    return scorers[section](rows, hypotheses)


# ---- what the sections share ------------------------------------------------------------


def _text(
    hypotheses: Hypotheses, section: str, row: SuiteRow, arm: str = "main"
) -> str:
    return hypotheses[RequestKey(section, row.id, arm)].text


def _reference(row: SuiteRow, section: str) -> str:
    if row.reference is None:
        raise ValueError(f"a row of the {section} section has no reference")
    return row.reference


def _ratio(part: int, whole: int) -> float | None:
    return None if whole == 0 else part / whole


def _accuracy(tally: Tally) -> float | None:
    return _ratio(tally.ok, tally.scored)


# ---- numbers and placeholders: a row is scored when source and reference agree --------


def _numbers(rows: Sequence[SuiteRow], hypotheses: Hypotheses) -> SectionResult:
    tally, excluded, outcomes = Tally(), Counter[str](), []
    expected = found = extra = surface = unit_failures = ambiguous = capped = 0
    for row in rows:
        reference = _reference(row, "numbers")
        reason = _numbers_exclusion(row, reference)
        if reason is not None:
            excluded[reason] += 1
            outcomes.append(excluded_outcome(row, reason))
            continue
        result = score_numbers(
            reference, _text(hypotheses, "numbers", row), tgt_lang=row.tgt_lang
        )
        tally.add(row, 1, result.exact)
        expected += result.expected
        found += result.found
        extra += result.extra
        surface += result.surface_exact
        unit_failures += not result.units_ok
        ambiguous += result.ambiguous
        capped += result.capped
        outcomes.append(
            scored_outcome(
                row, ok=result.exact, expected=result.expected, found=result.found
            )
        )
    details = {
        "recall": _ratio(found, expected),
        "extra_tokens": extra,
        "surface_exact_rows": surface,
        "unit_failures": unit_failures,
        "ambiguous_rows": ambiguous,
        "capped_rows": capped,
    }
    return build_section(
        section="numbers",
        unit="row",
        rows=rows,
        tally=tally,
        excluded=excluded,
        value=_accuracy(tally),
        details=details,
        hypotheses=hypotheses,
        outcomes=outcomes,
    )


def _numbers_exclusion(row: SuiteRow, reference: str) -> str | None:
    if not numbers_in(reference, row.tgt_lang):
        return "no_numbers"
    agree = numbers_agree(
        row.source, reference, src_lang=row.src_lang, tgt_lang=row.tgt_lang
    )
    return None if agree else "source_reference_disagree"


def _placeholders(rows: Sequence[SuiteRow], hypotheses: Hypotheses) -> SectionResult:
    tally, excluded, outcomes = Tally(), Counter[str](), []
    missing = extra = damaged = exact_rows = order_ok_rows = 0
    for row in rows:
        reference = _reference(row, "placeholders")
        if not placeholders_in(row.source):
            reason: str | None = "no_placeholders"
        elif not placeholders_agree(row.source, reference):
            reason = "source_reference_disagree"
        else:
            reason = None
        if reason is not None:
            excluded[reason] += 1
            outcomes.append(excluded_outcome(row, reason))
            continue
        result = score_placeholders(
            row.source, reference, _text(hypotheses, "placeholders", row)
        )
        tally.add(row, 1, result.exact)
        missing += result.missing
        extra += result.extra
        damaged += result.damaged
        exact_rows += result.exact
        order_ok_rows += result.order_ok is True
        outcomes.append(
            scored_outcome(
                row, ok=result.exact, expected=result.expected, found=result.found
            )
        )
    details = {
        "missing": missing,
        "extra": extra,
        "damaged": damaged,
        "exact_rows": exact_rows,
        "order_ok_rows": order_ok_rows,
    }
    return build_section(
        section="placeholders",
        unit="row",
        rows=rows,
        tally=tally,
        excluded=excluded,
        value=_accuracy(tally),
        details=details,
        hypotheses=hypotheses,
        outcomes=outcomes,
    )


# ---- entities and terms: the unit is a mention ----------------------------------------


def _entities(
    rows: Sequence[SuiteRow], table: EntityTable, hypotheses: Hypotheses
) -> SectionResult:
    tally, outcomes = Tally(), []
    rows_all_found = 0
    for row in rows:
        if not row.entities:
            raise ValueError("a row of the entities section lists no entity")
        result = score_entities(
            _text(hypotheses, "entities", row), row.entities, row.tgt_lang, table
        )
        all_found = result.found == result.expected
        tally.add(row, result.expected, result.found)
        rows_all_found += all_found
        outcomes.append(
            scored_outcome(
                row, ok=all_found, expected=result.expected, found=result.found
            )
        )
    return build_section(
        section="entities",
        unit="mention",
        rows=rows,
        tally=tally,
        excluded={},
        value=_accuracy(tally),
        details={"rows_all_found": rows_all_found},
        hypotheses=hypotheses,
        outcomes=outcomes,
    )


def _terms(rows: Sequence[SuiteRow], hypotheses: Hypotheses) -> SectionResult:
    tally, excluded, outcomes = Tally(), Counter[str](), []
    rows_all_found = 0
    for row in rows:
        if not row.terms:
            raise ValueError("a row of the terms section lists no term")
        result = score_terms(row.source, _text(hypotheses, "terms", row), row.terms)
        excluded["absent_in_source"] += result.excluded
        if result.expected == 0:
            outcomes.append(excluded_outcome(row, "absent_in_source"))
            continue
        all_found = result.found == result.expected
        tally.add(row, result.expected, result.found)
        rows_all_found += all_found
        outcomes.append(
            scored_outcome(
                row, ok=all_found, expected=result.expected, found=result.found
            )
        )
    return build_section(
        section="terms",
        unit="mention",
        rows=rows,
        tally=tally,
        excluded=excluded,
        value=_accuracy(tally),
        details={"rows_all_found": rows_all_found},
        hypotheses=hypotheses,
        outcomes=outcomes,
    )


# ---- script and mong --------------------------------------------------------------------


def _script(rows: Sequence[SuiteRow], hypotheses: Hypotheses) -> SectionResult:
    tally, excluded, outcomes = Tally(), Counter[str](), []
    purities: list[float] = []
    for row in rows:
        result = score_script(
            _text(hypotheses, "script", row), row.tgt_lang, row.script
        )
        if result.purity is None:
            excluded["no_letters"] += 1
            outcomes.append(excluded_outcome(row, "no_letters"))
            continue
        pure = not result.below_floor
        purities.append(result.purity)
        tally.add(row, 1, pure)
        outcomes.append(scored_outcome(row, ok=pure, value=result.purity))
    mean = math.fsum(purities) / len(purities) if purities else None
    details = {
        "mean": mean,
        "min": min(purities) if purities else None,
        "pure_share": _ratio(tally.ok, tally.scored),
        "below_floor": tally.scored - tally.ok,
        "no_letters": excluded["no_letters"],
        "floor": PURITY_FLOOR,
    }
    return build_section(
        section="script",
        unit="row",
        rows=rows,
        tally=tally,
        excluded=excluded,
        value=mean,
        details=details,
        hypotheses=hypotheses,
        outcomes=outcomes,
    )


def _mong(rows: Sequence[SuiteRow], hypotheses: Hypotheses) -> SectionResult:
    if not rows:
        return build_section(
            section="mong",
            unit="row",
            rows=rows,
            tally=Tally(),
            excluded={},
            value=None,
            details=_mong_details(None),
            hypotheses=hypotheses,
            outcomes=[],
        )
    references = [_reference(row, "mong") for row in rows]
    cyrl = [_text(hypotheses, "mong", row, "cyrl") for row in rows]
    mong = [_text(hypotheses, "mong", row, "mong") for row in rows]
    has_converted = any(
        key.section == "mong" and key.arm == "converted" for key in hypotheses
    )
    converted = (
        [_text(hypotheses, "mong", row, "converted") for row in rows]
        if has_converted
        else None
    )
    scores = score_mong(references, cyrl, mong, converted)
    tally, outcomes = Tally(), []
    for row, reference, control, direct in zip(
        rows, references, cyrl, mong, strict=True
    ):
        agree = control == direct
        tally.add(row, 1, agree)
        outcomes.append(
            scored_outcome(
                row,
                ok=agree,
                value=chrf(direct, reference),
                value_cyrl=chrf(control, reference),
            )
        )
    return build_section(
        section="mong",
        unit="row",
        rows=rows,
        tally=tally,
        excluded={},
        value=scores.retention,
        details=_mong_details(scores),
        hypotheses=hypotheses,
        outcomes=outcomes,
    )


def _mong_details(scores: MongScores | None) -> Details:
    if scores is None:
        return {
            "rows": 0,
            "chrf_cyrl": None,
            "chrf_mong": None,
            "chrf_signature": None,
            "agreement": None,
            "chrf_converted": None,
            "retention_converted": None,
        }
    return {
        "rows": scores.rows,
        "chrf_cyrl": scores.chrf_cyrl,
        "chrf_mong": scores.chrf_mong,
        "chrf_signature": scores.chrf_signature,
        "agreement": scores.agreement,
        "chrf_converted": scores.chrf_converted,
        "retention_converted": scores.retention_converted,
    }
