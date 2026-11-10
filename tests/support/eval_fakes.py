"""Fakes and golden outputs for the tests of the systems and the runner of D13 (Task 5); later
tasks grow it with the fakes of the commands.

Nothing here loads a model, starts a process, opens a socket or touches a GPU:

- `FakeSuiteEngine` is the engine as `EngineSystem` sees it: `info`, `translate(rows, options)`
  and `close()`. It answers from a function of the whole call (`answer`) or by echoing every
  text with status `ok`, records every call and counts the closings;
- `FakeCandidateTranslator` is D6's translator protocol (`translate(texts, src, tgt)`, a list
  of candidates per text): it answers from a function or echoes, and records every call;
- `FakeSuiteSystem` is a `SuiteSystem` of the test's own: verdicts by direction, answers from a
  function, every call recorded;
- `golden_suite`, `golden_outputs` and `golden_requests` are the golden suite of D13
  (`tests/fixtures/eval/golden_suite.json`) and what the systems `alpha` and `beta` of
  `tests.support.eval_golden` wrote for it; `engine_info` is a `ModelInfo` of a made-up model;
- `no_network` is re-exported: a module applies the guard with
  `pytestmark = pytest.mark.usefixtures("no_network")` after importing it from here.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final

from rinchen.eval.outputs import Output, OutputKey
from rinchen.eval.suites import Suite
from rinchen.eval.systems import SuiteRequest, SystemRecord, Verdict
from rinchen.infer.engine_types import (
    DecodeOptions,
    ModelInfo,
    Row,
    Status,
    Translation,
    selected_target,
)
from tests.support.eval_golden import (
    ALPHA_ENG,
    ALPHA_KHK,
    BETA_ENG,
    BETA_KHK,
    GOLDEN_DIRECTIONS,
    GOLDEN_IDS,
)
from tests.support.no_network import no_network  # noqa: F401 - re-exported

FIXTURES: Final = Path(__file__).resolve().parents[1] / "fixtures" / "eval"
GOLDEN_SUITE_FILE: Final = FIXTURES / "golden_suite.json"
GOLDEN_OUTPUTS_FILES: Final[Mapping[str, Path]] = {
    "alpha": FIXTURES / "golden_outputs_alpha.jsonl",
    "beta": FIXTURES / "golden_outputs_beta.jsonl",
}
# What each golden system wrote, by direction: the text of every row, in the order of the rows.
GOLDEN_TEXTS: Final[Mapping[str, Mapping[str, tuple[str, ...]]]] = {
    "alpha": {"khk_Cyrl-eng_Latn": ALPHA_ENG, "eng_Latn-khk_Cyrl": ALPHA_KHK},
    "beta": {"khk_Cyrl-eng_Latn": BETA_ENG, "eng_Latn-khk_Cyrl": BETA_KHK},
}
MODEL_SHA256: Final = "a1" * 32
MANIFEST_SHA256: Final = "b2" * 32
SUITE_SHA256: Final = "c3" * 32
OUTPUTS_SHA256: Final = "d4" * 32


def golden_suite() -> Suite:
    return Suite.from_bytes(GOLDEN_SUITE_FILE.read_bytes())


def golden_outputs(system: str) -> dict[OutputKey, Output]:
    """What `alpha` or `beta` wrote for the golden suite: both directions, ten rows each, every
    status `ok`, in the order of the suite's directions and rows."""
    texts = GOLDEN_TEXTS[system]
    return {
        OutputKey(direction, row_id): Output(text, "ok")
        for direction in GOLDEN_DIRECTIONS
        for row_id, text in zip(GOLDEN_IDS, texts[direction], strict=True)
    }


def requests_of(suite: Suite, directions: Sequence[str]) -> list[SuiteRequest]:
    """The requests of `directions` of `suite`, directions in the order given, rows in file
    order: what the runner hands a system, built here without the runner."""
    requests: list[SuiteRequest] = []
    for direction in directions:
        view = suite.view(direction)
        requests.extend(
            SuiteRequest(direction, row_id, view.src_lang, view.tgt_lang, text)
            for row_id, text in zip(view.ids, view.sources, strict=True)
        )
    return requests


def golden_requests(
    directions: Sequence[str] = GOLDEN_DIRECTIONS,
) -> list[SuiteRequest]:
    return requests_of(golden_suite(), directions)


def engine_info(
    *,
    directions: Sequence[str] = GOLDEN_DIRECTIONS,
    name: str = "toy",
    model_sha256: str = MODEL_SHA256,
    manifest_sha256: str = MANIFEST_SHA256,
) -> ModelInfo:
    """What a loaded engine says about a made-up model trained for `directions`."""
    return ModelInfo(
        name=name,
        quantization="float32",
        device="cpu",
        compute_type="float32",
        placement="src",
        directions=tuple(directions),
        targets=tuple(sorted({d.split("-")[1] for d in directions})),
        document_level=False,
        model_sha256=model_sha256,
        manifest_sha256=manifest_sha256,
    )


def translation_of(row: Row, text: str, status: Status = "ok") -> Translation:
    """The engine's answer to `row` that says `text` with `status`."""
    return Translation(
        text=text,
        status=status,
        alternatives=(),
        src=row.src or "und",
        tgt=selected_target(row.tgt, row.script),
        script_purity=None,
        retried=False,
        zero_shot=False,
    )


def answering(
    outputs: Sequence[Output],
) -> Callable[[Sequence[Row]], list[Translation]]:
    """An `answer` for `FakeSuiteEngine` that says `outputs` in order, whatever the rows are
    (a call of the wrong number of rows is a test's way to make the engine miscount)."""

    def answer(rows: Sequence[Row]) -> list[Translation]:
        return [
            translation_of(rows[min(index, len(rows) - 1)], output.text, output.status)
            for index, output in enumerate(outputs)
        ]

    return answer


@dataclass(frozen=True)
class RowsCall:
    """One call of `translate`: the rows and the options it was given."""

    rows: list[Row]
    options: DecodeOptions | None


@dataclass
class FakeSuiteEngine:
    """The engine as `EngineSystem` sees it. `answer` turns the rows of a call into the
    translations (the default echoes every text, status `ok`); `calls` records every call and
    `closed` counts the closings."""

    info: ModelInfo = field(default_factory=engine_info)
    answer: Callable[[Sequence[Row]], list[Translation]] | None = None
    calls: list[RowsCall] = field(default_factory=list)
    closed: int = 0

    def translate(
        self, rows: Sequence[Row], options: DecodeOptions | None = None
    ) -> list[Translation]:
        self.calls.append(RowsCall(list(rows), options))
        if self.answer is None:
            return [translation_of(row, row.text) for row in rows]
        return self.answer(rows)

    def close(self) -> None:
        self.closed += 1


@dataclass(frozen=True)
class TranslatorCall:
    """One call of a candidate translator: the texts, the source and the target language."""

    texts: list[str]
    src: str
    tgt: str


@dataclass
class FakeCandidateTranslator:
    """D6's translator protocol over a function: `answer(texts, src, tgt)` gives one list of
    candidates per text (the default says every text once). Every call is recorded."""

    answer: Callable[[Sequence[str], str, str], list[list[str]]] | None = None
    calls: list[TranslatorCall] = field(default_factory=list)

    def translate(
        self, texts: Sequence[str], src_lang: str, tgt_lang: str
    ) -> list[list[str]]:
        self.calls.append(TranslatorCall(list(texts), src_lang, tgt_lang))
        if self.answer is None:
            return [[text] for text in texts]
        return self.answer(texts, src_lang, tgt_lang)


@dataclass
class FakeSuiteSystem:
    """A `SuiteSystem` of the test's own. `verdicts` gives the verdict of a direction (a
    direction it does not name runs); `answer` turns the requests of a call into outputs (the
    default echoes every text, `ok`); `asked` records the directions `verdict` was asked about
    and `calls` the requests of every call of `translate`."""

    verdicts: Mapping[str, Verdict] = field(default_factory=dict)
    answer: Callable[[Sequence[SuiteRequest]], list[Output]] | None = None
    record: SystemRecord = field(
        default_factory=lambda: SystemRecord(kind="hypotheses", name="fake")
    )
    asked: list[str] = field(default_factory=list)
    calls: list[list[SuiteRequest]] = field(default_factory=list)

    def verdict(self, direction: str) -> Verdict:
        self.asked.append(direction)
        return self.verdicts.get(direction, Verdict("run"))

    def translate(self, requests: Sequence[SuiteRequest]) -> list[Output]:
        self.calls.append(list(requests))
        if self.answer is None:
            return [Output(request.text, "ok") for request in requests]
        return self.answer(requests)
