"""The systems of D13 (spec T5): what turns the requests of a suite into outputs, and which
directions each can run.

`EngineSystem` wraps anything with `translate(rows, options)` (the engine): it runs the directions
the model was trained for and makes ONE call for all its requests. `CandidateSystem` wraps D6's
translator protocol (a baseline): it runs the directions whose languages it knows, one call per
direction, the first candidate being the output and an empty list `no_output`. `FileSystem` holds
the outputs read from a file: it runs the directions the file holds. Each is exercised over fakes;
no model, generator or server is involved, and every baseline here is UNMEASURED."""

from __future__ import annotations

import json
from collections.abc import Sequence

import pytest

from rinchen.errors import DataError, UpstreamError
from rinchen.eval.outputs import Output, OutputKey
from rinchen.eval.systems import (
    CandidateSystem,
    EngineSystem,
    FileSystem,
    SuiteRequest,
    SystemRecord,
    Verdict,
)
from rinchen.infer.engine_types import DecodeOptions, Row
from tests.support.eval_fakes import (
    MANIFEST_SHA256,
    MODEL_SHA256,
    OUTPUTS_SHA256,
    FakeCandidateTranslator,
    FakeSuiteEngine,
    answering,
    engine_info,
    golden_outputs,
    golden_requests,
    golden_suite,
    no_network,  # noqa: F401 - the fixture of the mark
    requests_of,
)
from tests.support.eval_golden import GOLDEN_DIRECTIONS

pytestmark = pytest.mark.usefixtures("no_network")

FIRST, SECOND = GOLDEN_DIRECTIONS
NOT_TRAINED = "eng_Latn-rus_Cyrl"
MARKER = "SECRET-MARKER-92"
STATUSES = ("ok", "empty", "too_long", "truncated", "no_output")
TRAINED = ("eng_Latn-khk_Cyrl", "khk_Cyrl-eng_Latn")


def engine_system(
    engine: FakeSuiteEngine | None = None,
    *,
    options: DecodeOptions | None = None,
    name: str = "toy",
    options_record: dict[str, str | int | float | bool] | None = None,
) -> tuple[EngineSystem, FakeSuiteEngine]:
    engine = (
        FakeSuiteEngine(info=engine_info(directions=TRAINED))
        if engine is None
        else engine
    )
    system = EngineSystem(
        engine,
        engine.info,
        DecodeOptions() if options is None else options,
        name,
        {"beam_size": 4} if options_record is None else options_record,
    )
    return system, engine


# ---- the record ----------------------------------------------------------------------------


def test_a_record_of_a_file_system_has_the_six_keys_and_is_json_ready() -> None:
    record = SystemRecord(
        kind="hypotheses",
        name="alpha",
        outputs_sha256=OUTPUTS_SHA256,
        options={"b": 2, "a": "x", "c": True, "d": 0.5},
    )

    document = record.to_dict()

    assert list(document) == [
        "kind",
        "name",
        "model_sha256",
        "manifest_sha256",
        "options",
        "outputs_sha256",
    ]
    assert document == {
        "kind": "hypotheses",
        "name": "alpha",
        "model_sha256": None,
        "manifest_sha256": None,
        "options": {"a": "x", "b": 2, "c": True, "d": 0.5},
        "outputs_sha256": OUTPUTS_SHA256,
    }
    assert json.loads(json.dumps(document, sort_keys=True)) == document
    assert list(document["options"]) == ["a", "b", "c", "d"]  # type: ignore[call-overload]


def test_a_record_keeps_a_private_copy_of_its_options() -> None:
    options = {"a": 1}
    record = SystemRecord(kind="baseline", name="nllb", options=options)

    options["a"] = 2
    options["b"] = 3

    assert record.to_dict()["options"] == {"a": 1}
    with pytest.raises(TypeError):
        record.options["a"] = 9  # type: ignore[index]


def test_the_digests_and_the_options_default_to_none_and_nothing() -> None:
    record = SystemRecord(kind="baseline", name="nllb")

    assert record.model_sha256 is None
    assert record.manifest_sha256 is None
    assert record.outputs_sha256 is None
    assert record.to_dict()["options"] == {}


@pytest.mark.parametrize("kind", ["rinchen", "baseline", "hypotheses"])
def test_the_three_kinds_are_accepted(kind: str) -> None:
    assert SystemRecord(kind=kind, name="x").kind == kind  # type: ignore[arg-type]


@pytest.mark.parametrize("kind", ["engine", "Rinchen", "", "file"])
def test_a_kind_outside_the_three_is_a_value_error(kind: str) -> None:
    with pytest.raises(ValueError, match="kind"):
        SystemRecord(kind=kind, name="x")  # type: ignore[arg-type]


@pytest.mark.parametrize("value", [[1], (1,), {"a": 1}, None, b"x", {1}])
def test_an_option_with_a_value_that_is_not_a_scalar_is_a_type_error(
    value: object,
) -> None:
    with pytest.raises(TypeError, match="options"):
        SystemRecord(kind="baseline", name="x", options={"a": value})  # type: ignore[dict-item]


def test_an_option_named_by_a_non_string_is_a_type_error() -> None:
    with pytest.raises(TypeError, match="options"):
        SystemRecord(kind="baseline", name="x", options={1: "a"})  # type: ignore[dict-item]


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_an_option_that_is_not_a_finite_number_is_a_value_error(value: float) -> None:
    with pytest.raises(ValueError, match="finite"):
        SystemRecord(kind="baseline", name="x", options={"temperature": value})


def test_options_may_hold_a_float_because_a_baseline_records_its_temperature() -> None:
    record = SystemRecord(
        kind="baseline", name="x", options={"temperature": 0.0, "top_p": 1.0, "n": 1}
    )

    assert record.to_dict()["options"] == {"n": 1, "temperature": 0.0, "top_p": 1.0}


def test_a_name_that_is_not_a_string_is_a_type_error() -> None:
    with pytest.raises(TypeError, match="name"):
        SystemRecord(kind="baseline", name=3)  # type: ignore[arg-type]


@pytest.mark.parametrize("field", ["model_sha256", "manifest_sha256", "outputs_sha256"])
@pytest.mark.parametrize("digest", ["abc", "A" * 64, "a" * 63, "g" * 64, "", 5])
def test_a_digest_that_is_not_64_lowercase_hex_characters_is_a_value_error(
    field: str, digest: object
) -> None:
    with pytest.raises(ValueError, match=field):
        SystemRecord(kind="rinchen", name="x", **{field: digest})  # type: ignore[arg-type]


# ---- the request and the verdict -----------------------------------------------------------


def test_a_request_is_a_row_of_a_direction() -> None:
    request = SuiteRequest(FIRST, "g01", "khk_Cyrl", "eng_Latn", "муур")

    assert (request.direction, request.id, request.text) == (FIRST, "g01", "муур")


def test_a_request_whose_direction_is_not_its_two_languages_is_a_value_error() -> None:
    with pytest.raises(ValueError, match="direction"):
        SuiteRequest(FIRST, "g01", "eng_Latn", "khk_Cyrl", "x")


@pytest.mark.parametrize(
    "state",
    ["run", "not_trained", "unsupported_language", "not_in_file", "not_selected"],
)
def test_a_verdict_is_one_of_the_five_states_and_a_reason(state: str) -> None:
    verdict = Verdict(state, "why")  # type: ignore[arg-type]

    assert (verdict.state, verdict.reason) == (state, "why")
    assert Verdict(state).reason is None  # type: ignore[arg-type]


@pytest.mark.parametrize("state", ["ok", "", "skipped", "Run"])
def test_a_verdict_in_another_state_is_a_value_error(state: str) -> None:
    with pytest.raises(ValueError, match="state"):
        Verdict(state)  # type: ignore[arg-type]


# ---- the engine system ---------------------------------------------------------------------


def test_the_engine_runs_the_trained_directions_and_lists_the_others_not_trained() -> (
    None
):
    system, engine = engine_system()

    assert system.verdict("eng_Latn-khk_Cyrl") == Verdict("run")
    assert system.verdict("khk_Cyrl-eng_Latn") == Verdict("run")
    assert system.verdict(NOT_TRAINED) == Verdict(
        "not_trained", "not one of the model's trained directions"
    )
    assert engine.calls == []


def test_a_direction_is_trained_only_by_its_exact_name() -> None:
    system, _ = engine_system()

    for direction in ("eng_Latn-rus_Cyrl", "rus_Cyrl-khk_Cyrl", "khk_Cyrl-khk_Cyrl"):
        assert system.verdict(direction).state == "not_trained"


def test_one_call_of_twenty_rows_in_the_order_of_the_suite() -> None:
    system, engine = engine_system()
    requests = golden_requests()
    suite = golden_suite()

    outputs = system.translate(requests)

    assert len(engine.calls) == 1
    rows = engine.calls[0].rows
    assert len(rows) == len(outputs) == 20
    expected = [
        Row(text, view.src_lang, view.tgt_lang)
        for view in (suite.view(FIRST), suite.view(SECOND))
        for text in view.sources
    ]
    assert rows == expected
    assert all(row.script is None for row in rows)
    assert [row.src for row in rows[:10]] == ["khk_Cyrl"] * 10
    assert [row.tgt for row in rows[10:]] == ["khk_Cyrl"] * 10


def test_the_options_passed_to_the_engine_are_the_ones_given() -> None:
    options = DecodeOptions(
        beam_size=2, n=1, max_decoding_length=64, script_mask="always"
    )
    system, engine = engine_system(options=options)

    system.translate(golden_requests())

    assert engine.calls[0].options is options


def test_the_engine_texts_and_statuses_are_the_outputs_in_order() -> None:
    scripted = [
        Output(f"t{index}", STATUSES[index % 5])  # type: ignore[arg-type]
        for index in range(20)
    ]
    engine = FakeSuiteEngine(
        info=engine_info(directions=TRAINED), answer=answering(scripted)
    )
    system, _ = engine_system(engine)

    outputs = system.translate(golden_requests())

    assert outputs == scripted
    assert {output.status for output in outputs} == set(STATUSES)


def test_truncated_and_no_output_pass_through_with_their_text() -> None:
    scripted = [Output("part", "truncated"), Output("", "no_output"), Output("x", "ok")]
    engine = FakeSuiteEngine(
        info=engine_info(directions=TRAINED), answer=answering(scripted)
    )
    system, _ = engine_system(engine)

    outputs = system.translate(golden_requests()[:3])

    assert [(o.text, o.status) for o in outputs] == [
        ("part", "truncated"),
        ("", "no_output"),
        ("x", "ok"),
    ]


def test_an_engine_that_answers_nineteen_for_twenty_is_an_upstream_error() -> None:
    engine = FakeSuiteEngine(
        info=engine_info(directions=TRAINED),
        answer=answering([Output("x", "ok")] * 19),
    )
    system, _ = engine_system(engine)

    with pytest.raises(UpstreamError) as info:
        system.translate(golden_requests())

    assert "19" in str(info.value)
    assert "20" in str(info.value)


def test_an_engine_that_answers_too_many_is_an_upstream_error() -> None:
    engine = FakeSuiteEngine(
        info=engine_info(directions=TRAINED),
        answer=answering([Output("x", "ok")] * 21),
    )
    system, _ = engine_system(engine)

    with pytest.raises(UpstreamError, match="21"):
        system.translate(golden_requests())


def test_no_request_is_no_call() -> None:
    system, engine = engine_system()

    assert system.translate([]) == []
    assert engine.calls == []


def test_the_engine_record_carries_the_name_given_the_digests_and_the_options() -> None:
    system, _ = engine_system(
        name="mine", options_record={"beam_size": 4, "script_mask": "retry"}
    )

    assert system.record == SystemRecord(
        kind="rinchen",
        name="mine",
        model_sha256=MODEL_SHA256,
        manifest_sha256=MANIFEST_SHA256,
        options={"beam_size": 4, "script_mask": "retry"},
        outputs_sha256=None,
    )


def test_a_record_that_cannot_be_written_is_refused_when_the_system_is_made() -> None:
    engine = FakeSuiteEngine(info=engine_info(model_sha256="short"))

    with pytest.raises(ValueError, match="model_sha256"):
        engine_system(engine)

    with pytest.raises(TypeError, match="options"):
        engine_system(options_record={"a": [1]})  # type: ignore[dict-item]


# ---- the candidate system ------------------------------------------------------------------


def candidate_system(
    translator: FakeCandidateTranslator | None = None,
    languages: Sequence[str] = ("khk_Cyrl", "eng_Latn"),
) -> tuple[CandidateSystem, FakeCandidateTranslator]:
    translator = FakeCandidateTranslator() if translator is None else translator
    return (
        CandidateSystem(
            translator, "nllb", frozenset(languages), {"n": 1, "temperature": 0.0}
        ),
        translator,
    )


def test_a_baseline_runs_the_directions_whose_two_languages_it_knows() -> None:
    system, translator = candidate_system()

    assert system.verdict(FIRST) == Verdict("run")
    assert system.verdict(SECOND) == Verdict("run")
    assert system.verdict("khk_Cyrl-deu_Latn").state == "unsupported_language"
    assert system.verdict("deu_Latn-khk_Cyrl").state == "unsupported_language"
    assert system.verdict("deu_Latn-fra_Latn").state == "unsupported_language"
    assert translator.calls == []


def test_the_reason_of_an_unsupported_direction_says_why() -> None:
    system, _ = candidate_system()

    verdict = system.verdict("khk_Cyrl-deu_Latn")

    assert verdict.reason == "a language of the direction is not one the baseline knows"


def test_a_baseline_makes_one_call_per_direction_in_order_with_the_texts_of_the_direction() -> (
    None
):
    system, translator = candidate_system()
    suite = golden_suite()

    outputs = system.translate(golden_requests())

    assert len(outputs) == 20
    assert [(call.src, call.tgt) for call in translator.calls] == [
        ("khk_Cyrl", "eng_Latn"),
        ("eng_Latn", "khk_Cyrl"),
    ]
    assert translator.calls[0].texts == list(suite.view(FIRST).sources)
    assert translator.calls[1].texts == list(suite.view(SECOND).sources)


def test_the_first_candidate_is_the_output_and_an_empty_list_is_no_output() -> None:
    translator = FakeCandidateTranslator(
        answer=lambda texts, src, tgt: [["a", "b"], [], ["c"]]
    )
    system, _ = candidate_system(translator)

    outputs = system.translate(golden_requests(["khk_Cyrl-eng_Latn"])[:3])

    assert outputs == [Output("a", "ok"), Output("", "no_output"), Output("c", "ok")]


def test_the_outputs_come_back_in_the_order_of_the_requests_whatever_the_grouping() -> (
    None
):
    first = golden_requests([FIRST])
    second = golden_requests([SECOND])
    interleaved = [first[0], second[0], first[1], second[1]]
    translator = FakeCandidateTranslator(
        answer=lambda texts, src, tgt: [[f"{src}>{tgt}:{text}"] for text in texts]
    )
    system, _ = candidate_system(translator)

    outputs = system.translate(interleaved)

    assert [output.text for output in outputs] == [
        f"{request.src_lang}>{request.tgt_lang}:{request.text}"
        for request in interleaved
    ]
    assert [(call.src, call.tgt) for call in translator.calls] == [
        ("khk_Cyrl", "eng_Latn"),
        ("eng_Latn", "khk_Cyrl"),
    ]
    assert [len(call.texts) for call in translator.calls] == [2, 2]


def test_a_list_of_the_wrong_length_is_an_upstream_error_naming_direction_and_counts() -> (
    None
):
    translator = FakeCandidateTranslator(answer=lambda texts, src, tgt: [["a"]] * 9)
    system, _ = candidate_system(translator)

    with pytest.raises(UpstreamError) as info:
        system.translate(golden_requests([FIRST]))

    message = str(info.value)
    assert f"direction '{FIRST}'" in message
    assert "9" in message
    assert "10" in message


def test_a_translator_that_answers_too_many_lists_is_an_upstream_error() -> None:
    translator = FakeCandidateTranslator(answer=lambda texts, src, tgt: [["a"]] * 11)
    system, _ = candidate_system(translator)

    with pytest.raises(UpstreamError, match="11"):
        system.translate(golden_requests([FIRST]))


def test_no_request_is_no_call_of_the_translator() -> None:
    system, translator = candidate_system()

    assert system.translate([]) == []
    assert translator.calls == []


def test_the_candidate_record_is_a_baseline_with_the_options_given() -> None:
    system, _ = candidate_system()

    assert system.record == SystemRecord(
        kind="baseline", name="nllb", options={"n": 1, "temperature": 0.0}
    )


def test_a_candidate_that_is_not_a_string_is_refused_at_the_output() -> None:
    translator = FakeCandidateTranslator(
        answer=lambda texts, src, tgt: [[None]] * len(texts)
    )
    system, _ = candidate_system(translator)

    with pytest.raises(TypeError, match="text"):
        system.translate(golden_requests([FIRST]))


# ---- the file system -----------------------------------------------------------------------


def file_system(directions: Sequence[str] | None = None) -> FileSystem:
    outputs = golden_outputs("alpha")
    if directions is not None:
        outputs = {k: v for k, v in outputs.items() if k.direction in directions}
    return FileSystem(outputs, "alpha", OUTPUTS_SHA256)


def test_a_file_runs_the_directions_it_holds_and_lists_the_others_not_in_file() -> None:
    system = file_system([FIRST])

    assert system.verdict(FIRST) == Verdict("run")
    assert system.verdict(SECOND).state == "not_in_file"
    assert system.verdict(NOT_TRAINED).state == "not_in_file"
    assert system.verdict(SECOND).reason == "the file holds no output for the direction"


def test_a_file_system_looks_the_requests_up_in_the_order_asked() -> None:
    system = file_system()
    requests = golden_requests([SECOND, FIRST])

    outputs = system.translate(requests)

    expected = golden_outputs("alpha")
    assert outputs == [expected[OutputKey(r.direction, r.id)] for r in requests]
    assert outputs[0] == expected[OutputKey(SECOND, "g01")]


def test_a_request_the_file_lacks_is_a_data_error_naming_it_and_not_a_text() -> None:
    system = file_system([FIRST])
    request = SuiteRequest(SECOND, "g01", "eng_Latn", "khk_Cyrl", MARKER)

    with pytest.raises(DataError) as info:
        system.translate([request])

    message = str(info.value)
    assert f"direction '{SECOND}'" in message
    assert "id 'g01'" in message
    assert MARKER not in message


def test_a_request_with_an_id_the_file_lacks_is_a_data_error() -> None:
    system = file_system()
    request = SuiteRequest(FIRST, "g99", "khk_Cyrl", "eng_Latn", "x")

    with pytest.raises(DataError, match="g99"):
        system.translate([request])


def test_no_request_gives_no_outputs() -> None:
    assert file_system().translate([]) == []


def test_the_file_record_is_a_hypotheses_system_with_the_digest_of_the_file() -> None:
    assert file_system().record == SystemRecord(
        kind="hypotheses", name="alpha", outputs_sha256=OUTPUTS_SHA256
    )


def test_a_file_system_refuses_a_digest_that_is_not_one() -> None:
    with pytest.raises(ValueError, match="outputs_sha256"):
        FileSystem(golden_outputs("alpha"), "alpha", "nope")


def test_the_requests_of_the_helper_are_those_of_the_suite() -> None:
    requests = requests_of(golden_suite(), [FIRST])

    assert [r.id for r in requests] == [f"g{n:02d}" for n in range(1, 11)]
    assert {(r.src_lang, r.tgt_lang) for r in requests} == {("khk_Cyrl", "eng_Latn")}
