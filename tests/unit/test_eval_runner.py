"""The runner of D13 (spec T6): `run_suite` asks a system which directions it can run, makes ONE
call for the requests of the runnable ones (directions in the suite's order, rows in file order)
and returns the outputs keyed in that order, with EVERY declared direction listed in a state:
`run`, `not_trained`, `unsupported_language`, `not_in_file` or `not_selected`.

The systems here are the fakes of `tests.support.eval_fakes`; nothing loads a model. Every
baseline is UNMEASURED."""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from rinchen.errors import ConfigError, UpstreamError
from rinchen.eval.outputs import Output, OutputKey
from rinchen.eval.runner import (
    NOT_SELECTED_REASON,
    DirectionRun,
    SuiteRun,
    run_suite,
)
from rinchen.eval.suites import Suite, SuiteRow, SuiteSource
from rinchen.eval.systems import (
    CandidateSystem,
    EngineSystem,
    FileSystem,
    SuiteRequest,
    SystemRecord,
    Verdict,
)
from rinchen.infer.engine_types import DecodeOptions
from tests.support.eval_fakes import (
    OUTPUTS_SHA256,
    SUITE_SHA256,
    FakeCandidateTranslator,
    FakeSuiteEngine,
    FakeSuiteSystem,
    answering,
    engine_info,
    golden_outputs,
    golden_suite,
    no_network,  # noqa: F401 - the fixture of the mark
)
from tests.support.eval_golden import GOLDEN_DIRECTIONS

pytestmark = pytest.mark.usefixtures("no_network")

FIRST, SECOND = GOLDEN_DIRECTIONS
MARKER = "SECRET-MARKER-93"
TRAINED = (SECOND, FIRST)
FOUR = (
    "khk_Cyrl-eng_Latn",
    "eng_Latn-khk_Cyrl",
    "eng_Latn-rus_Cyrl",
    "khk_Cyrl-deu_Latn",
)


def four_direction_suite() -> Suite:
    """Three rows in four languages and four directions, two of which a model trained on the
    first two cannot run (`eng_Latn-rus_Cyrl`) or a baseline that lacks German cannot."""
    rows = tuple(
        SuiteRow(
            id=f"r{number}",
            doc=None,
            text={
                "eng_Latn": f"english {number}",
                "khk_Cyrl": f"khalkha {number}",
                "rus_Cyrl": f"russian {number}",
                "deu_Latn": f"german {number}",
            },
        )
        for number in range(1, 4)
    )
    return Suite(
        name="four",
        kind="fixture",
        benchmark="four",
        split="devtest",
        source=SuiteSource(None, None, False, {}, ()),
        directions=FOUR,
        rows=rows,
    )


def engine_system(
    answer_with: Sequence[Output] | None = None,
    directions: Sequence[str] = TRAINED,
) -> tuple[EngineSystem, FakeSuiteEngine]:
    engine = FakeSuiteEngine(
        info=engine_info(directions=directions),
        answer=None if answer_with is None else answering(answer_with),
    )
    return EngineSystem(
        engine, engine.info, DecodeOptions(), "toy", {"beam_size": 4}
    ), engine


def states(run: SuiteRun) -> list[tuple[str, str]]:
    return [(d.direction, d.state) for d in run.directions]


# ---- everything runs -----------------------------------------------------------------------


def test_every_direction_runs_with_one_call_and_the_outputs_are_the_scripted_ones() -> (
    None
):
    scripted = golden_outputs("alpha")
    system, engine = engine_system(list(scripted.values()))

    run = run_suite(golden_suite(), SUITE_SHA256, system, None)

    assert states(run) == [(FIRST, "run"), (SECOND, "run")]
    assert [d.rows for d in run.directions] == [10, 10]
    assert len(engine.calls) == 1
    assert len(engine.calls[0].rows) == 20
    assert run.outputs == scripted
    assert list(run.outputs) == list(scripted)


def test_the_run_names_the_suite_its_digest_and_the_system() -> None:
    system, _ = engine_system()

    run = run_suite(golden_suite(), SUITE_SHA256, system, None)

    assert run.suite_name == "golden"
    assert run.suite_sha256 == SUITE_SHA256
    assert run.system == system.record
    assert run.system.kind == "rinchen"


def test_the_outputs_follow_the_directions_of_the_suite_not_the_alphabet() -> None:
    system, _ = engine_system()

    run = run_suite(golden_suite(), SUITE_SHA256, system, None)

    assert [key.direction for key in run.outputs] == [FIRST] * 10 + [SECOND] * 10
    assert [key.id for key in list(run.outputs)[:10]] == [
        f"g{n:02d}" for n in range(1, 11)
    ]


def test_the_system_is_asked_for_the_requests_in_the_order_of_the_suite() -> None:
    system = FakeSuiteSystem()

    run_suite(golden_suite(), SUITE_SHA256, system, None)

    assert len(system.calls) == 1
    requests = system.calls[0]
    suite = golden_suite()
    assert requests == [
        SuiteRequest(direction, row_id, view.src_lang, view.tgt_lang, text)
        for direction in (FIRST, SECOND)
        for view in (suite.view(direction),)
        for row_id, text in zip(view.ids, view.sources, strict=True)
    ]


def test_the_outputs_are_keyed_by_direction_and_id_from_the_requests() -> None:
    system = FakeSuiteSystem(
        answer=lambda requests: [Output(r.id, "ok") for r in requests]
    )

    run = run_suite(golden_suite(), SUITE_SHA256, system, None)

    assert all(key.id == output.text for key, output in run.outputs.items())
    assert OutputKey(SECOND, "g05") in run.outputs


# ---- states --------------------------------------------------------------------------------


def test_a_model_runs_what_it_was_trained_for_and_lists_the_rest_not_trained() -> None:
    system, engine = engine_system()

    run = run_suite(four_direction_suite(), SUITE_SHA256, system, None)

    assert states(run) == [
        ("khk_Cyrl-eng_Latn", "run"),
        ("eng_Latn-khk_Cyrl", "run"),
        ("eng_Latn-rus_Cyrl", "not_trained"),
        ("khk_Cyrl-deu_Latn", "not_trained"),
    ]
    assert len(engine.calls) == 1
    assert len(engine.calls[0].rows) == 6
    assert {key.direction for key in run.outputs} == {FIRST, SECOND}
    assert run.directions[2].reason == "not one of the model's trained directions"
    assert run.directions[0].reason is None


def test_a_direction_that_is_not_run_is_never_scored_or_translated() -> None:
    system, engine = engine_system()

    run = run_suite(four_direction_suite(), SUITE_SHA256, system, None)

    asked = {row.tgt for call in engine.calls for row in call.rows}
    assert asked == {"eng_Latn", "khk_Cyrl"}  # never rus_Cyrl or deu_Latn
    assert not any(key.direction == "eng_Latn-rus_Cyrl" for key in run.outputs)


def test_a_baseline_is_never_called_for_a_language_it_does_not_know() -> None:
    translator = FakeCandidateTranslator()
    system = CandidateSystem(
        translator, "nllb", frozenset({"eng_Latn", "khk_Cyrl", "rus_Cyrl"}), {}
    )

    run = run_suite(four_direction_suite(), SUITE_SHA256, system, None)

    assert states(run) == [
        ("khk_Cyrl-eng_Latn", "run"),
        ("eng_Latn-khk_Cyrl", "run"),
        ("eng_Latn-rus_Cyrl", "run"),
        ("khk_Cyrl-deu_Latn", "unsupported_language"),
    ]
    assert [(c.src, c.tgt) for c in translator.calls] == [
        ("khk_Cyrl", "eng_Latn"),
        ("eng_Latn", "khk_Cyrl"),
        ("eng_Latn", "rus_Cyrl"),
    ]
    assert all("german" not in text for c in translator.calls for text in c.texts)
    assert len(run.outputs) == 9


def test_a_file_runs_the_directions_it_holds_and_lists_the_rest_not_in_file() -> None:
    held = {k: v for k, v in golden_outputs("alpha").items() if k.direction == FIRST}
    system = FileSystem(held, "alpha", OUTPUTS_SHA256)

    run = run_suite(golden_suite(), SUITE_SHA256, system, None)

    assert states(run) == [(FIRST, "run"), (SECOND, "not_in_file")]
    assert run.outputs == held
    assert [d.rows for d in run.directions] == [10, 10]


def test_the_runner_keeps_the_reason_the_system_gave() -> None:
    system = FakeSuiteSystem(
        verdicts={SECOND: Verdict("not_trained", "because of the test")}
    )

    run = run_suite(golden_suite(), SUITE_SHA256, system, None)

    assert run.directions[1] == DirectionRun(
        SECOND, "not_trained", "because of the test", 10
    )


def test_every_declared_direction_appears_once_in_the_order_of_the_suite() -> None:
    system, _ = engine_system(directions=())

    run = run_suite(four_direction_suite(), SUITE_SHA256, system, None)

    assert [d.direction for d in run.directions] == list(FOUR)
    assert {d.state for d in run.directions} == {"not_trained"}
    assert [d.rows for d in run.directions] == [3, 3, 3, 3]


def test_the_rows_of_a_direction_are_counted_in_every_state() -> None:
    system = FakeSuiteSystem(
        verdicts={
            FOUR[1]: Verdict("not_trained", "x"),
            FOUR[2]: Verdict("unsupported_language", "y"),
        }
    )

    run = run_suite(
        four_direction_suite(), SUITE_SHA256, system, [FOUR[0], FOUR[1], FOUR[2]]
    )

    assert [(d.state, d.rows) for d in run.directions] == [
        ("run", 3),
        ("not_trained", 3),
        ("unsupported_language", 3),
        ("not_selected", 3),
    ]


def test_one_call_holds_the_runnable_directions_with_the_others_skipped_in_between() -> (
    None
):
    system = FakeSuiteSystem(
        verdicts={
            FOUR[1]: Verdict("not_trained"),
            FOUR[3]: Verdict("unsupported_language"),
        }
    )

    run = run_suite(four_direction_suite(), SUITE_SHA256, system, None)

    assert len(system.calls) == 1
    assert [(r.direction, r.id) for r in system.calls[0]] == [
        (FOUR[0], "r1"),
        (FOUR[0], "r2"),
        (FOUR[0], "r3"),
        (FOUR[2], "r1"),
        (FOUR[2], "r2"),
        (FOUR[2], "r3"),
    ]
    assert [(d.direction, d.state) for d in run.directions] == [
        (FOUR[0], "run"),
        (FOUR[1], "not_trained"),
        (FOUR[2], "run"),
        (FOUR[3], "unsupported_language"),
    ]


def test_a_system_that_runs_nothing_gets_no_call_and_the_outputs_are_empty() -> None:
    system, engine = engine_system(directions=())

    run = run_suite(golden_suite(), SUITE_SHA256, system, None)

    assert engine.calls == []
    assert run.outputs == {}
    assert states(run) == [(FIRST, "not_trained"), (SECOND, "not_trained")]


# ---- the selection -------------------------------------------------------------------------


def test_a_selection_of_one_direction_lists_the_other_not_selected() -> None:
    system, engine = engine_system()

    run = run_suite(golden_suite(), SUITE_SHA256, system, [SECOND])

    assert states(run) == [(FIRST, "not_selected"), (SECOND, "run")]
    assert run.directions[0].reason == NOT_SELECTED_REASON == "not in the selection"
    assert len(engine.calls) == 1
    assert len(engine.calls[0].rows) == 10
    assert {key.direction for key in run.outputs} == {SECOND}


def test_a_direction_outside_the_selection_is_not_even_asked_about() -> None:
    system = FakeSuiteSystem()

    run_suite(golden_suite(), SUITE_SHA256, system, [FIRST])

    assert system.asked == [FIRST]


def test_a_selection_may_name_a_direction_twice() -> None:
    system = FakeSuiteSystem()

    run = run_suite(golden_suite(), SUITE_SHA256, system, [FIRST, FIRST])

    assert states(run) == [(FIRST, "run"), (SECOND, "not_selected")]


def test_the_order_of_a_selection_does_not_change_the_order_of_the_run() -> None:
    system = FakeSuiteSystem()

    run = run_suite(golden_suite(), SUITE_SHA256, system, [SECOND, FIRST])

    assert [d.direction for d in run.directions] == [FIRST, SECOND]
    assert [r.direction for r in system.calls[0]] == [FIRST] * 10 + [SECOND] * 10


def test_an_empty_selection_selects_nothing_and_makes_no_call() -> None:
    system = FakeSuiteSystem()

    run = run_suite(golden_suite(), SUITE_SHA256, system, [])

    assert {d.state for d in run.directions} == {"not_selected"}
    assert system.calls == []
    assert run.outputs == {}


def test_a_selection_naming_a_direction_the_suite_lacks_is_a_config_error() -> None:
    system = FakeSuiteSystem()

    with pytest.raises(ConfigError) as info:
        run_suite(golden_suite(), SUITE_SHA256, system, [FIRST, "eng_Latn-rus_Cyrl"])

    assert "direction 'eng_Latn-rus_Cyrl'" in str(info.value)
    assert system.asked == []
    assert system.calls == []


def test_a_selection_entry_that_is_not_a_direction_is_not_quoted() -> None:
    system = FakeSuiteSystem()

    with pytest.raises(ConfigError) as info:
        run_suite(golden_suite(), SUITE_SHA256, system, [f"{MARKER}!"])

    assert MARKER not in str(info.value)
    assert "selection" in str(info.value)
    assert system.calls == []


def test_khk_mong_in_a_selection_is_no_direction_of_the_suite() -> None:
    with pytest.raises(ConfigError, match="selection"):
        run_suite(
            golden_suite(), SUITE_SHA256, FakeSuiteSystem(), ["khk_Mong-eng_Latn"]
        )


# ---- what a system may get wrong -----------------------------------------------------------


def test_a_system_that_answers_fewer_outputs_is_an_upstream_error() -> None:
    system = FakeSuiteSystem(
        answer=lambda requests: [Output("x", "ok") for _ in requests[:-1]]
    )

    with pytest.raises(UpstreamError) as info:
        run_suite(golden_suite(), SUITE_SHA256, system, None)

    assert "19" in str(info.value)
    assert "20" in str(info.value)


def test_a_system_that_answers_more_outputs_is_an_upstream_error() -> None:
    system = FakeSuiteSystem(
        answer=lambda requests: [Output("x", "ok") for _ in [*requests, 0]]
    )

    with pytest.raises(UpstreamError, match="21"):
        run_suite(golden_suite(), SUITE_SHA256, system, None)


def test_a_system_that_reorders_inside_still_gets_its_answers_keyed_by_position() -> (
    None
):
    def reversed_inside(requests: Sequence[SuiteRequest]) -> list[Output]:
        computed = {
            index: Output(request.text, "ok")
            for index, request in reversed(list(enumerate(requests)))
        }
        return [computed[index] for index in range(len(requests))]

    system = FakeSuiteSystem(answer=reversed_inside)
    suite = golden_suite()

    run = run_suite(suite, SUITE_SHA256, system, None)

    view = suite.view(SECOND)
    assert [run.outputs[OutputKey(SECOND, i)].text for i in view.ids] == list(
        view.sources
    )
    assert list(run.outputs) == [
        OutputKey(direction, row_id)
        for direction in (FIRST, SECOND)
        for row_id in suite.view(direction).ids
    ]


def test_a_status_other_than_ok_is_kept_in_the_outputs() -> None:
    scripted = [Output(f"t{n}", "truncated" if n % 3 == 0 else "ok") for n in range(20)]
    system, _ = engine_system(scripted)

    run = run_suite(golden_suite(), SUITE_SHA256, system, None)

    assert [o.status for o in run.outputs.values()] == [o.status for o in scripted]


def test_the_error_of_the_system_propagates_untouched() -> None:
    class Failing(FakeSuiteSystem):
        def translate(self, requests: Sequence[SuiteRequest]) -> list[Output]:
            raise ConfigError("the model does not translate into 'x'")

    with pytest.raises(ConfigError, match="does not translate"):
        run_suite(golden_suite(), SUITE_SHA256, Failing(), None)


@pytest.mark.parametrize("digest", ["", "abc", "A" * 64, "z" * 64, "a" * 65])
def test_a_suite_digest_that_is_not_a_digest_is_a_value_error(digest: str) -> None:
    system = FakeSuiteSystem()

    with pytest.raises(ValueError, match="suite_sha256"):
        run_suite(golden_suite(), digest, system, None)

    assert system.calls == []


def test_the_outputs_of_a_run_are_read_only_and_a_copy() -> None:
    scripted = golden_outputs("alpha")
    system = FakeSuiteSystem(
        answer=lambda requests: [Output("x", "ok") for _ in requests]
    )

    run = run_suite(golden_suite(), SUITE_SHA256, system, None)

    with pytest.raises(TypeError):
        run.outputs[OutputKey(FIRST, "g01")] = Output("y", "ok")  # type: ignore[index]
    assert run.outputs[OutputKey(FIRST, "g01")] == Output("x", "ok")
    assert run.outputs != scripted
    assert dict(run.outputs) == {key: Output("x", "ok") for key in scripted}


def test_a_run_built_by_hand_keeps_its_own_copies() -> None:
    outputs = {OutputKey(FIRST, "g01"): Output("x", "ok")}
    directions = [DirectionRun(FIRST, "run", None, 1)]

    run = SuiteRun(
        "n",
        SUITE_SHA256,
        SystemRecord(kind="hypotheses", name="n"),
        directions,
        outputs,
    )  # type: ignore[arg-type]
    outputs.clear()
    directions.clear()

    assert len(run.outputs) == 1
    assert isinstance(run.directions, tuple)
    assert len(run.directions) == 1


def test_the_run_is_frozen() -> None:
    run = run_suite(golden_suite(), SUITE_SHA256, FakeSuiteSystem(), None)

    with pytest.raises(dataclasses.FrozenInstanceError):
        run.suite_name = "x"  # type: ignore[misc]
    with pytest.raises(dataclasses.FrozenInstanceError):
        run.directions[0].state = "run"  # type: ignore[misc]


# ---- property: the states are a function of the verdicts and the selection -----------------

SYSTEM_STATES = ("run", "not_trained", "unsupported_language", "not_in_file")


@settings(
    max_examples=120,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
@given(
    drawn=st.lists(st.sampled_from(SYSTEM_STATES), min_size=4, max_size=4),
    selection=st.one_of(
        st.none(), st.lists(st.sampled_from(FOUR), unique=True, max_size=4)
    ),
)
def test_the_states_are_a_function_of_the_verdicts_and_the_selection(
    drawn: list[str], selection: list[str] | None
) -> None:
    suite = four_direction_suite()
    verdicts = {
        direction: Verdict(state)  # type: ignore[arg-type]
        for direction, state in zip(FOUR, drawn, strict=True)
    }
    system = FakeSuiteSystem(verdicts=verdicts)

    run = run_suite(suite, SUITE_SHA256, system, selection)

    expected = [
        "not_selected"
        if selection is not None and direction not in selection
        else verdicts[direction].state
        for direction in FOUR
    ]
    assert [d.direction for d in run.directions] == list(FOUR)
    assert [d.state for d in run.directions] == expected
    assert [d.rows for d in run.directions] == [3, 3, 3, 3]
    running = [d for d, state in zip(FOUR, expected, strict=True) if state == "run"]
    assert len(system.calls) == (1 if running else 0)
    assert list(run.outputs) == [
        OutputKey(direction, f"r{n}") for direction in running for n in (1, 2, 3)
    ]
    assert system.asked == [d for d in FOUR if selection is None or d in selection]
