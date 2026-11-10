"""The golden fixtures of D12 and the helpers that read them: a hand-written suite of 33
requests with known answers for every metric (`golden_suite.json`), the hypotheses a
made-up system wrote for them (`golden_hypotheses.jsonl`) and the result of scoring one with
the other (`golden_robustness.json`). The expected numbers are in the tests that read them,
derived by hand and by independent prototypes, not by this code."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Final

from rinchen.eval.robust_hypotheses import Hypothesis, hypotheses_from_bytes
from rinchen.eval.robust_spans import EntityTable
from rinchen.eval.robust_suite import (
    Request,
    RequestKey,
    Suite,
    SuiteRow,
    suite_requests,
)
from rinchen.eval.robustness import (
    RobustnessResult,
    RobustnessSystem,
    compute_robustness,
)

GOLDEN: Final = Path(__file__).resolve().parents[1] / "fixtures" / "robustness"
GOLDEN_SUITE: Final = GOLDEN / "golden_suite.json"
GOLDEN_HYPOTHESES: Final = GOLDEN / "golden_hypotheses.jsonl"
GOLDEN_RESULT: Final = GOLDEN / "golden_robustness.json"
GOLDEN_SUITE_SHA256: Final = (
    "2767112dc06c61b62fbc66ad7336b3dffeff81c33ce7e7e124875a44503e75b0"
)
GOLDEN_HYPOTHESES_SHA256: Final = (
    "78bb56fc23e7cc07ea27df868ead4f35d52208779f2e7c67d05efcbb755e07a8"
)


def golden_suite() -> Suite:
    return Suite.from_bytes(GOLDEN_SUITE.read_bytes())


def golden_suite_sha256() -> str:
    return hashlib.sha256(GOLDEN_SUITE.read_bytes()).hexdigest()


def golden_requests() -> list[Request]:
    return suite_requests(golden_suite())


def golden_hypotheses() -> dict[RequestKey, Hypothesis]:
    return hypotheses_from_bytes(GOLDEN_HYPOTHESES.read_bytes(), golden_requests())


GOLDEN_SYSTEM: Final = RobustnessSystem(name="golden")
EKH, KHK, RUS = "eng_Latn", "khk_Cyrl", "rus_Cyrl"
MONGOLIAN: Final = "".join(chr(code) for code in (0x1820, 0x1828, 0x1821))


def golden_result() -> RobustnessResult:
    """The golden suite scored with the golden hypotheses."""
    return compute_robustness(
        golden_suite(),
        golden_hypotheses(),
        suite_sha256=golden_suite_sha256(),
        system=GOLDEN_SYSTEM,
    )


def run(suite: Suite, hypotheses: Any) -> dict[str, Any]:
    """The document of `suite` scored with `hypotheses` (a made-up suite digest)."""
    result = compute_robustness(
        suite, hypotheses, suite_sha256="0" * 64, system=GOLDEN_SYSTEM
    )
    return result.to_dict()


def key(section: str, row_id: str, arm: str = "main") -> RequestKey:
    return RequestKey(section, row_id, arm)


def row(row_id: str, **changes: object) -> SuiteRow:
    """A `numbers` row: English into Khalkha, `It costs 15.`, with `changes` applied."""
    fields: dict[str, object] = {
        "id": row_id,
        "src_lang": EKH,
        "tgt_lang": KHK,
        "source": "It costs 15.",
        "reference": "Үнэ 15.",
        "category": "int",
    }
    fields.update(changes)
    return SuiteRow(**fields)  # type: ignore[arg-type]


def suite_of(**by_section: tuple[SuiteRow, ...]) -> Suite:
    return Suite(name="t", entities=EntityTable({}), sections=by_section)


def texts(**by_id: str) -> dict[RequestKey, Hypothesis]:
    """`ok` hypotheses for the `numbers` rows named by keyword."""
    return {
        key("numbers", row_id): Hypothesis(text, "ok") for row_id, text in by_id.items()
    }
