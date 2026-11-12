"""The `serve` stage of the tiny pipeline: the service checked in process (D15, spec P2 and F10, brief G5.3).

The stage starts the real service app over a fixture-sized export with `fastapi.testclient.TestClient`,
which calls the app through an ASGI transport inside the process: no socket is opened, so these tests
run under the `no_network` guard. Here the engine is a fake (`tests/support/serve_app.py`) passed as
`engine=`; the gate runs the stage over a real export. The stage asks five questions and makes its
verdict from the answers: the two probes, a translation of two memorised sentences, a target the model
does not write, and the counters.

The service logs JSON lines to the standard error stream through `rinchen.utils.logging`, which
configures the root logger on its first use for the rest of the process (and `test_logging` expects to
be that first use), and httpx logs each request at INFO. The stage must show NOTHING on either stream and
leave the logging state exactly as it found it; `fresh_logging` makes a pristine state for a test and puts
back whatever the session had.
"""

from __future__ import annotations

import copy
import json
import logging
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import pytest
import structlog

from rinchen.config.serve import ServeConfig
from rinchen.errors import ConfigError, DataError, UpstreamError
from rinchen.tiny import serve_stage
from rinchen.tiny.eval_stages import run_serve
from rinchen.tiny.serve_stage import exercise_service
from rinchen.tiny.states import FAIL_INPUT, fail_check, is_closed_reason
from rinchen.utils import logging as rinchen_logging
from tests.support.no_network import no_network  # noqa: F401 - the fixture of the mark
from tests.support.serve_app import FakeEngine, translation_of
from tests.support.tiny_run import changed_config
from tests.support.tiny_stage_inputs import make_context, shipped_fixtures

pytestmark = pytest.mark.usefixtures("no_network")

KHK: Final = "khk_Cyrl"
ENG: Final = "eng_Latn"
CODES: Final = [200, 200, 200, 400, 200]
# A target the fake model does not write.
REFUSED: Final = "deu_Latn"
METRICS: Final = (
    "# HELP rinchen_requests_total Requests.\n"
    "# TYPE rinchen_requests_total counter\n"
    'rinchen_requests_total{route="translate",status="200"} 1.0\n'
    'rinchen_requests_total{route="translate",status="400"} 1.0\n'
    'rinchen_requests_total{route="healthz",status="200"} 1.0\n'
)


@pytest.fixture
def fresh_logging(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """A process whose logging was never configured: no handler on the root logger, structlog's
    defaults and the flag of `rinchen.utils.logging` down. Everything is put back afterwards."""
    root, client = logging.getLogger(), logging.getLogger("httpx")
    handlers, level, client_level = list(root.handlers), root.level, client.level
    configuration = structlog.get_config()
    disabled = logging.root.manager.disable
    monkeypatch.setattr(rinchen_logging, "_CONFIGURED", False)
    root.handlers = []
    root.setLevel(logging.WARNING)
    structlog.reset_defaults()
    logging.disable(logging.NOTSET)
    try:
        yield
    finally:
        root.handlers = handlers
        root.setLevel(level)
        client.setLevel(client_level)
        structlog.configure(**configuration)
        logging.disable(disabled)


def logging_state() -> dict[str, Any]:
    root = logging.getLogger()
    return {
        "handlers": list(root.handlers),
        "level": root.level,
        "disabled": logging.root.manager.disable,
        "configured": rinchen_logging._CONFIGURED,
        "structlog": copy.copy(structlog.get_config()),
        "httpx": (
            logging.getLogger("httpx").level,
            list(logging.getLogger("httpx").handlers),
        ),
    }


def ctx_for(tmp_path: Path, **sections: Any) -> Any:
    config = changed_config(**sections) if sections else None
    return make_context(tmp_path / "out", config=config)


def stage(tmp_path: Path, engine: FakeEngine | None = None, **sections: Any) -> Any:
    return run_serve(ctx_for(tmp_path, **sections), engine=engine or FakeEngine())


# ---- the verdict -----------------------------------------------------------------------------------


def test_the_service_over_a_fake_engine_is_ok_with_the_codes_and_the_counter(
    tmp_path: Path,
) -> None:
    result = stage(tmp_path)

    assert result.record.id == "serve"
    assert result.record.state == "ok"
    assert result.record.reason is None
    assert result.facts["status_codes"] == CODES
    assert result.facts["translate_200_total"] == 1.0
    assert result.facts["statuses"] == ["ok", "ok"]
    assert result.facts["script_purity"] == [1.0, 1.0]
    assert result.facts["error_code"] == "bad_request"


def test_the_detail_of_an_ok_record_is_a_copy_of_the_facts(tmp_path: Path) -> None:
    result = stage(tmp_path)

    assert result.record.detail == result.facts
    result.facts["status_codes"].append(0)
    assert result.record.detail["status_codes"] == CODES
    json.dumps(result.record.detail)


def test_the_stage_translates_two_memorised_sentences_at_the_beam_of_the_profile(
    tmp_path: Path,
) -> None:
    engine = FakeEngine()
    config = ctx_for(tmp_path).config

    stage(tmp_path, engine)

    fixtures = shipped_fixtures()
    trained = [
        row.src
        for row in fixtures.pairs["eng_Latn-khk_Cyrl"]
        if row.split == "train" and row.kind == "base"
    ]
    shortest = sorted(trained, key=len)[:2]
    assert [call.kind for call in engine.calls] == ["rows"]
    assert engine.calls[0].texts == tuple(shortest)
    assert set(engine.calls[0].texts) <= set(trained)
    assert len(set(engine.calls[0].texts)) == 2
    assert max(map(len, engine.calls[0].texts)) <= min(
        len(text) for text in set(trained) - set(shortest)
    )
    options = engine.calls[0].options
    assert options is not None and options.beam_size == config.decode.beam_size
    assert {(row.src, row.tgt) for row in engine.calls[0].rows} == {(ENG, KHK)}


def test_the_refused_target_never_reaches_the_engine(tmp_path: Path) -> None:
    engine = FakeEngine()

    stage(tmp_path, engine)

    assert len(engine.calls) == 1
    assert [row.tgt for call in engine.calls for row in call.rows] == [KHK, KHK]
    assert REFUSED not in {row.tgt for call in engine.calls for row in call.rows}


def test_a_status_of_empty_makes_the_stage_fail(tmp_path: Path) -> None:
    engine = FakeEngine(answer=lambda row: translation_of(row, "", "empty"))

    result = stage(tmp_path, engine)

    assert result.record.state == "fail"
    assert result.record.reason == fail_check("service_translate")
    assert result.record.detail == {}
    assert result.facts["status_codes"] == CODES
    assert result.facts["statuses"] == ["empty", "empty"]


@pytest.mark.parametrize("status", ["too_long", "truncated", "no_output"])
def test_every_status_but_ok_makes_the_stage_fail(tmp_path: Path, status: str) -> None:
    engine = FakeEngine(answer=lambda row: translation_of(row, "x", status, purity=1.0))  # type: ignore[arg-type]

    assert stage(tmp_path, engine).record.reason == fail_check("service_translate")


@pytest.mark.parametrize("purity", [0.5, None, 0.999])
def test_a_script_purity_below_one_makes_the_stage_fail(
    tmp_path: Path, purity: float | None
) -> None:
    engine = FakeEngine(
        answer=lambda row: translation_of(row, "x", "ok", purity=purity)
    )

    result = stage(tmp_path, engine)

    assert result.record.state == "fail"
    assert result.record.reason == fail_check("service_translate")


def test_one_bad_answer_of_two_makes_the_stage_fail(tmp_path: Path) -> None:
    def answer(row: Any) -> Any:
        status = "ok" if row.text == texts[0] else "empty"
        return translation_of(row, "x", status, purity=1.0)

    texts = sorted(
        (
            row.src
            for row in shipped_fixtures().pairs["eng_Latn-khk_Cyrl"]
            if row.split == "train" and row.kind == "base"
        ),
        key=len,
    )[:2]

    result = stage(tmp_path, FakeEngine(answer=answer))

    assert result.record.reason == fail_check("service_translate")
    assert result.facts["statuses"] == ["ok", "empty"]


def test_a_model_that_writes_the_refused_target_makes_the_stage_fail(
    tmp_path: Path,
) -> None:
    engine = FakeEngine(targets=(KHK, ENG, REFUSED))

    result = stage(tmp_path, engine)

    assert result.record.state == "fail"
    assert result.record.reason == fail_check("service_refusal")
    assert result.facts["status_codes"][3] == 200


def test_every_reason_of_the_stage_is_of_the_closed_list(tmp_path: Path) -> None:
    for engine in (
        FakeEngine(answer=lambda row: translation_of(row, "", "empty")),
        FakeEngine(targets=(KHK, ENG, REFUSED)),
    ):
        reason = stage(tmp_path, engine).record.reason
        assert reason is not None and is_closed_reason(reason)


# ---- the questions, over a scripted client -----------------------------------------------------------


@dataclass
class Reply:
    status_code: int
    body: Any = None
    text: str = ""

    def json(self) -> Any:
        if self.body is None:
            raise ValueError("not JSON")
        return self.body


GOOD_TRANSLATION: Final = {
    "translations": ["a", "b"],
    "statuses": ["ok", "ok"],
    "script_purity": [1.0, 1.0],
}
GOOD_ERROR: Final = {"error": {"code": "bad_request", "message": "no"}}


class Scripted:
    """A client that answers each of the five questions with a given reply and records them."""

    def __init__(self, **replies: Reply) -> None:
        self.replies = {
            "healthz": Reply(200),
            "readyz": Reply(200),
            "translate": Reply(200, GOOD_TRANSLATION),
            "refused": Reply(400, GOOD_ERROR),
            "metrics": Reply(200, text=METRICS),
            **replies,
        }
        self.asked: list[tuple[str, str, Any]] = []

    def get(self, url: str) -> Reply:
        self.asked.append(("GET", url, None))
        return self.replies[url.strip("/")]

    def post(self, url: str, *, json: Any) -> Reply:
        self.asked.append(("POST", url, json))
        return self.replies["translate" if json["tgt"] == KHK else "refused"]


def exercise(
    client: Scripted, texts: tuple[str, str] = ("a", "b")
) -> tuple[dict[str, Any], str | None]:
    return exercise_service(client, texts, beam_size=1)


def test_the_five_questions_are_asked_in_order_with_the_bodies_of_the_stage() -> None:
    client = Scripted()

    facts, failure = exercise(client)

    assert failure is None
    assert facts["status_codes"] == CODES
    assert client.asked == [
        ("GET", "/healthz", None),
        ("GET", "/readyz", None),
        (
            "POST",
            "/v1/translate",
            {"texts": ["a", "b"], "src": ENG, "tgt": KHK, "beam_size": 1},
        ),
        (
            "POST",
            "/v1/translate",
            {"texts": ["a", "b"], "src": ENG, "tgt": REFUSED, "beam_size": 1},
        ),
        ("GET", "/metrics", None),
    ]


@pytest.mark.parametrize("probe", ["healthz", "readyz"])
@pytest.mark.parametrize("code", [503, 500, 404, 400, 302, 204, 201])
def test_a_probe_that_is_not_a_200_fails_the_probe_check(probe: str, code: int) -> None:
    facts, failure = exercise(Scripted(**{probe: Reply(code)}))

    assert failure == fail_check("service_probe")
    assert facts["status_codes"][["healthz", "readyz"].index(probe)] == code


@pytest.mark.parametrize(
    "reply",
    [
        Reply(500),
        Reply(400, GOOD_ERROR),
        Reply(400, GOOD_TRANSLATION),
        Reply(201, GOOD_TRANSLATION),
        Reply(200, {"statuses": ["ok", "ok"]}),
        Reply(200, {**GOOD_TRANSLATION, "statuses": ["ok"]}),
        Reply(200, {**GOOD_TRANSLATION, "statuses": ["ok", "empty"]}),
        Reply(200, {**GOOD_TRANSLATION, "script_purity": [1.0, 0.9]}),
        Reply(200, {**GOOD_TRANSLATION, "script_purity": [1.0, None]}),
        Reply(200, {**GOOD_TRANSLATION, "script_purity": [1.0, True]}),
        Reply(200, {**GOOD_TRANSLATION, "script_purity": [1.0]}),
        Reply(200, {**GOOD_TRANSLATION, "statuses": "ok"}),
        Reply(200, ["not", "an", "object"]),
        Reply(200, None),
    ],
    ids=[
        "500",
        "400",
        "400-with-good-statuses",
        "201-with-good-statuses",
        "no-purity",
        "one-status",
        "an-empty",
        "low-purity",
        "no-judgement",
        "a-boolean",
        "one-purity",
        "statuses-not-a-list",
        "not-an-object",
        "not-json",
    ],
)
def test_a_translation_that_is_not_two_ok_sentences_of_pure_script_fails(
    reply: Reply,
) -> None:
    _, failure = exercise(Scripted(translate=reply))

    assert failure == fail_check("service_translate")


@pytest.mark.parametrize(
    "reply",
    [
        Reply(200, GOOD_TRANSLATION),
        Reply(200, GOOD_ERROR),
        Reply(500, GOOD_ERROR),
        Reply(404, GOOD_ERROR),
        Reply(400, {"error": {"message": "no"}}),
        Reply(400, {"error": {"code": "", "message": "no"}}),
        Reply(400, {"error": {"code": "Not A Code!", "message": "no"}}),
        Reply(400, {"error": {"code": 5, "message": "no"}}),
        Reply(400, {"error": "bad_request"}),
        Reply(400, {}),
        Reply(400, None),
    ],
    ids=[
        "accepted",
        "accepted-with-an-error-body",
        "500",
        "404",
        "no-code",
        "empty-code",
        "free-text-code",
        "number-code",
        "error-not-an-object",
        "no-error",
        "not-json",
    ],
)
def test_a_refusal_that_is_not_a_400_with_a_code_fails(reply: Reply) -> None:
    _, failure = exercise(Scripted(refused=reply))

    assert failure == fail_check("service_refusal")


@pytest.mark.parametrize(
    "reply",
    [
        Reply(500, text=METRICS),
        Reply(400, text=METRICS),
        Reply(204, text=METRICS),
        Reply(200, text=""),
        Reply(200, text=METRICS.replace('status="200"} 1.0', 'status="200"} 2.0', 1)),
        Reply(200, text=METRICS.replace('status="200"} 1.0', 'status="200"} 0.0', 1)),
        Reply(200, text=METRICS.replace("translate", "other")),
        Reply(200, text='rinchen_requests_total{route="translate",status="200"} nan\n'),
        Reply(
            200,
            text='rinchen_requests_total{route="translate",status="200",x="y"} 1.0\n',
        ),
    ],
    ids=[
        "500",
        "400-with-the-counter",
        "204-with-the-counter",
        "empty",
        "two",
        "zero",
        "other-route",
        "not-a-number",
        "an-extra-label",
    ],
)
def test_a_counter_that_is_not_one_translation_fails_the_metrics_check(
    reply: Reply,
) -> None:
    _, failure = exercise(Scripted(metrics=reply))

    assert failure == fail_check("service_metrics")


def test_the_labels_of_the_counter_may_come_in_either_order() -> None:
    text = 'rinchen_requests_total{status="200",route="translate"} 1.0\n'

    assert exercise(Scripted(metrics=Reply(200, text=text)))[1] is None


def test_the_first_check_that_fails_is_the_one_reported_and_every_code_is_kept() -> (
    None
):
    client = Scripted(
        readyz=Reply(503), translate=Reply(500), refused=Reply(200, GOOD_TRANSLATION)
    )

    facts, failure = exercise(client)

    assert failure == fail_check("service_probe")
    assert facts["status_codes"] == [200, 503, 500, 200, 200]


def test_the_facts_hold_no_text_of_the_answers() -> None:
    reply = Reply(
        200,
        {
            "translations": ["SECRET-ONE", "SECRET-TWO"],
            "statuses": ["ok", "ok"],
            "script_purity": [1.0, 1.0],
        },
    )

    facts, _ = exercise(Scripted(translate=reply))

    assert "SECRET" not in json.dumps(facts)


def test_a_code_that_is_free_text_is_not_kept_as_a_fact() -> None:
    reply = Reply(400, {"error": {"code": "SECRET text", "message": "m"}})

    facts, failure = exercise(Scripted(refused=reply))

    assert failure == fail_check("service_refusal")
    assert "SECRET" not in json.dumps(facts)


# ---- what the stage uses ------------------------------------------------------------------------------


def test_the_service_runs_over_the_serve_config_of_the_profile_with_one_thread_each(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: list[tuple[ServeConfig, dict[str, Any]]] = []
    real = serve_stage.create_app

    def recording(config: ServeConfig, **kwargs: Any) -> Any:
        seen.append((config, kwargs))
        return real(config, engine=FakeEngine())

    monkeypatch.setattr(serve_stage, "create_app", recording)
    ctx = ctx_for(tmp_path)

    result = run_serve(ctx)

    assert result.record.state == "ok"
    config, kwargs = seen[0]
    assert (config.intra_threads, config.inter_threads) == (1, 1)
    assert config.device == "cpu"
    assert config.beam_size == 4
    assert kwargs == {
        "engine": None,
        "model": ctx.out_dir.resolve() / "08_export" / "int8",
    }


def test_an_engine_that_is_given_is_the_one_the_app_runs_over(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: list[dict[str, Any]] = []
    real = serve_stage.create_app

    def recording(config: ServeConfig, **kwargs: Any) -> Any:
        seen.append(kwargs)
        return real(config, **kwargs)

    monkeypatch.setattr(serve_stage, "create_app", recording)
    engine = FakeEngine()

    run_serve(ctx_for(tmp_path), engine=engine)

    assert seen == [{"engine": engine, "model": None}]


def test_the_environment_does_not_reach_the_service_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("RINCHEN_SERVE_PORT", "not a port")
    monkeypatch.setenv("RINCHEN_SERVE_BEAM_SIZE", "0")

    assert stage(tmp_path).record.state == "ok"


def test_the_stage_writes_nothing_into_the_output_directory(tmp_path: Path) -> None:
    ctx = ctx_for(tmp_path)

    run_serve(ctx, engine=FakeEngine())

    assert not ctx.out_dir.exists() or list(ctx.out_dir.iterdir()) == []


def test_a_serve_config_that_is_not_there_is_a_failed_input(tmp_path: Path) -> None:
    base = changed_config().serve
    missing = base.model_copy(update={"config": tmp_path / "no.yaml"})

    result = stage(tmp_path, serve=missing)

    assert result.record.state == "fail"
    assert result.record.reason == FAIL_INPUT
    assert result.facts == {}


@pytest.mark.parametrize("error", [DataError("x"), ConfigError("x")])
def test_an_engine_that_cannot_start_for_its_input_is_a_failed_input(
    tmp_path: Path, error: Exception
) -> None:
    result = stage(tmp_path, FakeEngine(warmup_failure=error))

    assert result.record.state == "fail"
    assert result.record.reason == FAIL_INPUT
    assert result.record.detail == {}


def test_an_engine_that_fails_upstream_is_a_service_that_did_not_come_up(
    tmp_path: Path,
) -> None:
    result = stage(tmp_path, FakeEngine(warmup_failure=UpstreamError("x")))

    assert result.record.state == "fail"
    assert result.record.reason == fail_check("service_probe")
    assert result.record.detail == {}


def test_an_engine_that_crashes_is_not_hidden(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="boom"):
        stage(tmp_path, FakeEngine(warmup_failure=RuntimeError("boom")))


# ---- nothing is shown and the logging is left as it was ------------------------------------------------


def test_nothing_reaches_either_stream_while_the_logging_was_never_configured(
    tmp_path: Path, capfd: pytest.CaptureFixture[str], fresh_logging: None
) -> None:
    result = stage(tmp_path)

    captured = capfd.readouterr()
    assert result.record.state == "ok"
    assert (captured.out, captured.err) == ("", "")


def test_nothing_reaches_either_stream_while_the_logging_is_configured_to_show_everything(
    tmp_path: Path, capfd: pytest.CaptureFixture[str], fresh_logging: None
) -> None:
    rinchen_logging.get("before").info("configured")
    logging.getLogger("httpx").setLevel(logging.INFO)
    capfd.readouterr()

    result = stage(tmp_path)

    captured = capfd.readouterr()
    assert result.record.state == "ok"
    assert (captured.out, captured.err) == ("", "")


def test_the_logging_state_is_the_same_before_and_after_when_it_was_never_configured(
    tmp_path: Path, fresh_logging: None
) -> None:
    before = logging_state()

    stage(tmp_path)

    assert logging_state() == before
    assert before["configured"] is False


def test_the_logging_state_is_the_same_before_and_after_when_it_was_configured(
    tmp_path: Path, fresh_logging: None
) -> None:
    rinchen_logging.get("before").info("configured")
    before = logging_state()
    assert before["configured"] is True

    stage(tmp_path)

    assert logging_state() == before


def test_the_state_is_restored_when_the_stage_fails_and_when_it_raises(
    tmp_path: Path, fresh_logging: None
) -> None:
    before = logging_state()

    stage(tmp_path, FakeEngine(answer=lambda row: translation_of(row, "", "empty")))
    assert logging_state() == before
    with pytest.raises(RuntimeError):
        stage(tmp_path, FakeEngine(warmup_failure=RuntimeError("boom")))
    assert logging_state() == before


def test_the_first_use_of_the_logging_after_the_stage_still_configures_it(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], fresh_logging: None
) -> None:
    stage(tmp_path)

    # What `tests/unit/test_logging.py` does: the first call to `get` writes JSON to standard error.
    rinchen_logging.get("test_logger").info("stage_start", stage="demo", count=3)

    line = capsys.readouterr().err.strip().splitlines()[-1]
    assert json.loads(line)["event"] == "stage_start"


def test_the_service_logs_nothing_even_through_the_loggers_it_made_itself(
    tmp_path: Path, caplog: pytest.LogCaptureFixture, fresh_logging: None
) -> None:
    caplog.set_level(logging.DEBUG)

    stage(tmp_path)

    assert caplog.records == []


def test_the_logging_of_the_process_works_again_right_after_the_stage(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    stage(tmp_path)
    caplog.set_level(logging.INFO)

    logging.getLogger("after").info("visible")

    assert [record.getMessage() for record in caplog.records] == ["visible"]
