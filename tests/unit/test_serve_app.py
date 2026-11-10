"""The service (D11, Task 9): the probes, what a request is turned into and what comes back,
the batches, the metrics and the start of the service.

Every test runs the app in process over a `FakeEngine` (`tests.support.serve_app`): an
`httpx.AsyncClient` over `ASGITransport` inside `app.router.lifespan_context(app)`, in a fresh
event loop. The refusals are in `test_serve_app_errors.py`, the stop in
`test_serve_app_shutdown.py`.
"""

from __future__ import annotations

import asyncio
import os
import subprocess
import sys
import threading
from pathlib import Path
from typing import Any

import pytest

from rinchen.config.serve import ServeConfig
from rinchen.errors import ConfigError, UpstreamError
from rinchen.infer.doc_chunker import ChunkStats
from rinchen.infer.engine_types import (
    DecodeOptions,
    DocumentResult,
    Row,
    Translation,
)
from rinchen.serve._translate import (
    DocumentKey,
    SentenceKey,
    build_response,
    document_status,
)
from rinchen.serve.app import create_app
from rinchen.serve.batcher import MicroBatcher
from rinchen.serve.schemas import TranslateRequest
from tests.support.serve_app import (
    ENG,
    KHK,
    MON,
    MONGOLIAN,
    NO_STATS,
    FakeEngine,
    RecordingLog,
    engine_started,
    error_of,
    make_app,
    make_client,
    post,
    run,
    serve_config,
    translation_of,
    until,
)

READY_BODY = {
    "status": "ready",
    "model": "toy-model",
    "quantization": "float32",
    "device": "cpu",
    "compute_type": "float32",
}
# ---- the probes ---------------------------------------------------------------------------


def test_healthz_answers_200_before_the_service_started_and_after(
    recording_log: RecordingLog,
) -> None:
    async def scenario() -> tuple[Any, Any]:
        async with make_app(recording_log) as svc:
            before = await svc.client.get("/healthz")
            async with svc.app.router.lifespan_context(svc.app):
                during = await svc.client.get("/healthz")
        return before, during

    before, during = run(scenario())

    assert (before.status_code, before.json()) == (200, {"status": "ok"})
    assert (during.status_code, during.json()) == (200, {"status": "ok"})


def test_readyz_is_503_not_ready_before_the_lifespan_ran(
    recording_log: RecordingLog,
) -> None:
    async def scenario() -> Any:
        async with make_app(recording_log) as svc:
            return await svc.client.get("/readyz")

    response = run(scenario())

    assert response.status_code == 503
    assert error_of(response)["code"] == "not_ready"


def test_readyz_is_200_with_the_five_keys_once_the_service_is_ready(
    recording_log: RecordingLog,
) -> None:
    async def scenario() -> Any:
        async with make_client(recording_log) as svc:
            return await svc.client.get("/readyz")

    response = run(scenario())

    assert response.status_code == 200
    assert response.json() == READY_BODY


def test_the_service_is_not_ready_until_the_warm_up_ran(
    recording_log: RecordingLog,
) -> None:
    engine = FakeEngine()
    engine.hold_warmup()

    async def scenario() -> tuple[Any, Any, Any, Any]:
        async with make_app(recording_log, engine) as svc:
            lifespan = svc.app.router.lifespan_context(svc.app)
            starting = asyncio.create_task(lifespan.__aenter__())
            assert await asyncio.to_thread(engine.wait_for_warmup)
            while_warming = await svc.client.get("/readyz")
            health = await svc.client.get("/healthz")
            translate = await post(svc.client)
            engine.release_warmup()
            await starting
            after = await svc.client.get("/readyz")
            await lifespan.__aexit__(None, None, None)
        return while_warming, health, translate, after

    while_warming, health, translate, after = run(scenario())

    assert (while_warming.status_code, error_of(while_warming)["code"]) == (
        503,
        "not_ready",
    )
    assert health.status_code == 200
    assert (translate.status_code, error_of(translate)["code"]) == (503, "not_ready")
    assert translate.headers["retry-after"] == "1"
    assert after.status_code == 200
    assert engine.warmups == 1


# ---- a request and its answer -------------------------------------------------------------


def test_a_request_for_one_text_gets_this_answer(recording_log: RecordingLog) -> None:
    async def scenario() -> Any:
        async with make_client(recording_log) as svc:
            return await svc.client.post(
                "/v1/translate",
                json={"texts": ["hello"], "src": ENG, "tgt": KHK},
            )

    response = run(scenario())

    body = response.json()
    latency = body.pop("latency_ms")
    assert response.status_code == 200
    assert response.headers["content-type"] == "application/json"
    assert isinstance(latency, float)
    assert latency >= 0
    assert body == {
        "translations": ["<hello>"],
        "detected_src": [ENG],
        "model": "toy-model",
        "statuses": ["ok"],
        "script_purity": [1.0],
    }


def test_the_latency_in_the_answer_includes_the_time_the_engine_took(
    recording_log: RecordingLog,
) -> None:
    async def scenario() -> Any:
        async with make_client(recording_log, FakeEngine(delay_seconds=0.1)) as svc:
            return await post(svc.client)

    assert run(scenario()).json()["latency_ms"] >= 90  # the engine took 100 ms


def test_a_request_for_three_texts_keeps_their_order_and_says_each_status(
    recording_log: RecordingLog,
) -> None:
    async def scenario() -> Any:
        async with make_client(recording_log) as svc:
            return await post(svc.client, texts=["one", "", "two"])

    response = run(scenario())

    body = response.json()
    body.pop("latency_ms")
    assert body == {
        "translations": ["<one>", "", "<two>"],
        "detected_src": ["und", "und", "und"],
        "model": "toy-model",
        "statuses": ["ok", "empty", "ok"],
        "script_purity": [1.0, None, 1.0],
    }


def test_the_model_in_the_answer_is_the_name_of_the_engine(
    recording_log: RecordingLog,
) -> None:
    async def scenario() -> Any:
        async with make_client(recording_log, FakeEngine(name="rinchen-abc")) as svc:
            return await post(svc.client)

    assert run(scenario()).json()["model"] == "rinchen-abc"


def test_auto_reports_the_script_of_each_text_and_a_given_source_is_reported_as_given(
    recording_log: RecordingLog,
) -> None:
    async def scenario() -> tuple[Any, Any, FakeEngine]:
        engine = FakeEngine(targets=(KHK, ENG, MON))
        async with make_client(recording_log, engine) as svc:
            auto = await post(svc.client, texts=[MONGOLIAN, "hello"], tgt=ENG)
            given = await post(
                svc.client, texts=[MONGOLIAN, "hello"], src="rus_Cyrl", tgt=ENG
            )
        return auto, given, engine

    auto, given, engine = run(scenario())

    assert auto.json()["detected_src"] == ["mn_Mong", "und"]
    assert given.json()["detected_src"] == ["rus_Cyrl", "rus_Cyrl"]
    assert [row.src for row in engine.calls[0].rows] == [None, None]
    assert [row.src for row in engine.calls[1].rows] == ["rus_Cyrl", "rus_Cyrl"]


def test_the_rows_say_the_target_and_the_script_that_were_asked_for(
    recording_log: RecordingLog,
) -> None:
    async def scenario() -> FakeEngine:
        engine = FakeEngine()
        async with make_client(recording_log, engine) as svc:
            response = await post(
                svc.client, texts=["a1", "b2"], tgt=KHK, script="Cyrl"
            )
            assert response.status_code == 200
        return engine

    engine = run(scenario())

    assert engine.calls[0].rows == (
        Row("a1", None, KHK, "Cyrl"),
        Row("b2", None, KHK, "Cyrl"),
    )


def test_alternatives_are_in_the_answer_only_for_a_request_that_asked_for_more_than_one(
    recording_log: RecordingLog,
) -> None:
    async def scenario() -> tuple[Any, Any]:
        async with make_client(recording_log) as svc:
            one = await post(svc.client, texts=["hello", ""])
            more = await post(svc.client, texts=["hello", ""], n_best=2)
        return one, more

    one, more = run(scenario())

    assert "alternatives" not in one.json()
    assert more.json()["alternatives"] == [["<hello>#2", "<hello>#3"], []]


def test_a_text_that_is_not_translated_is_still_a_200_with_its_status(
    recording_log: RecordingLog,
) -> None:
    async def scenario() -> Any:
        async with make_client(recording_log) as svc:
            return await post(svc.client, texts=["x" * 31, "cut...", "silent", "hello"])

    response = run(scenario())

    body = response.json()
    assert response.status_code == 200
    assert body["statuses"] == ["too_long", "truncated", "no_output", "ok"]
    assert body["translations"] == ["", "<cut...>", "", "<hello>"]
    assert body["script_purity"] == [None, 1.0, None, 1.0]


def test_a_request_is_decoded_with_the_options_of_the_config_and_its_own(
    recording_log: RecordingLog,
) -> None:
    config = serve_config(
        beam_size=2,
        max_decoding_length=64,
        max_source_pieces=100,
        script_mask="always",
        max_batch_sentences=8,
    )

    async def scenario() -> FakeEngine:
        engine = FakeEngine()
        async with make_client(recording_log, engine, config=config) as svc:
            await post(svc.client)
            await post(svc.client, beam_size=6, n_best=3)
        return engine

    engine = run(scenario())

    assert [call.options for call in engine.calls] == [
        DecodeOptions(
            beam_size=2,
            n=1,
            max_decoding_length=64,
            max_source_pieces=100,
            max_batch_size=8,
            script_mask="always",
        ),
        DecodeOptions(
            beam_size=6,
            n=3,
            max_decoding_length=64,
            max_source_pieces=100,
            max_batch_size=8,
            script_mask="always",
        ),
    ]


def test_texts_of_a_document_request_go_to_the_document_call_with_the_languages(
    recording_log: RecordingLog,
) -> None:
    async def scenario() -> tuple[Any, FakeEngine]:
        engine = FakeEngine(document_level=True)
        async with make_client(recording_log, engine) as svc:
            response = await post(
                svc.client,
                texts=["One. Two.", "Three.", "", "silent"],
                src=ENG,
                script="Cyrl",
                doc_mode=True,
            )
        return response, engine

    response, engine = run(scenario())

    body = response.json()
    body.pop("latency_ms")
    assert response.status_code == 200
    assert body == {
        "translations": ["<One. Two.>", "<Three.>", "", ""],
        "detected_src": [ENG] * 4,
        "model": "toy-model",
        "statuses": ["ok", "ok", "empty", "no_output"],
        "script_purity": [1.0, 1.0, None, None],
    }
    (call,) = engine.calls
    assert (call.kind, call.texts, call.src, call.tgt, call.script) == (
        "documents",
        ("One. Two.", "Three.", "", "silent"),
        ENG,
        KHK,
        "Cyrl",
    )


def test_a_document_request_with_n_best_has_no_alternatives_to_offer(
    recording_log: RecordingLog,
) -> None:
    async def scenario() -> Any:
        async with make_client(recording_log, FakeEngine(document_level=True)) as svc:
            return await post(svc.client, src=ENG, doc_mode=True, n_best=2)

    assert run(scenario()).json()["alternatives"] == [[]]


def test_document_and_sentence_requests_are_never_decoded_in_one_call(
    recording_log: RecordingLog,
) -> None:
    async def scenario() -> FakeEngine:
        engine = FakeEngine(document_level=True)
        async with make_client(recording_log, engine, inter_threads=1) as svc:
            engine.hold()
            first = asyncio.create_task(post(svc.client, texts=["held"]))
            await engine_started(engine, 1)
            sentence = asyncio.create_task(post(svc.client, texts=["one"]))
            document = asyncio.create_task(
                post(svc.client, texts=["two"], src=ENG, doc_mode=True)
            )
            await until(lambda: svc.state.batcher.depth == 2)
            engine.release()
            responses = await asyncio.gather(first, sentence, document)
            assert [r.status_code for r in responses] == [200, 200, 200]
        return engine

    engine = run(scenario())

    assert sorted((call.kind, call.texts) for call in engine.calls) == [
        ("documents", ("two",)),
        ("rows", ("held",)),
        ("rows", ("one",)),
    ]


def test_requests_with_different_options_are_never_decoded_in_one_call(
    recording_log: RecordingLog,
) -> None:
    async def scenario() -> FakeEngine:
        engine = FakeEngine()
        engine.hold()
        async with make_client(recording_log, engine) as svc:
            wide = asyncio.create_task(post(svc.client, texts=["a1"], beam_size=4))
            narrow = asyncio.create_task(post(svc.client, texts=["b2"], beam_size=2))
            await engine_started(engine, 2)
            engine.release()
            assert [r.status_code for r in await asyncio.gather(wide, narrow)] == [
                200,
                200,
            ]
        return engine

    engine = run(scenario())

    assert sorted(
        (c.options.beam_size, c.texts) for c in engine.calls if c.options
    ) == [
        (2, ("b2",)),
        (4, ("a1",)),
    ]


def test_requests_that_arrive_together_are_decoded_in_one_batch(
    recording_log: RecordingLog,
) -> None:
    async def scenario() -> FakeEngine:
        engine = FakeEngine()
        async with make_client(recording_log, engine, inter_threads=1) as svc:
            engine.hold()
            first = asyncio.create_task(post(svc.client, texts=["held"]))
            await engine_started(engine, 1)
            more = [
                asyncio.create_task(post(svc.client, texts=[f"t{index}"]))
                for index in range(5)
            ]
            await until(lambda: svc.state.batcher.depth == 5)
            engine.release()
            responses = await asyncio.gather(first, *more)
            assert [r.json()["translations"] for r in responses] == [
                ["<held>"],
                *[[f"<t{index}>"] for index in range(5)],
            ]
        return engine

    engine = run(scenario())

    assert [call.texts for call in engine.calls] == [
        ("held",),
        ("t0", "t1", "t2", "t3", "t4"),
    ]


def test_64_concurrent_requests_are_all_answered_each_with_its_own_text(
    recording_log: RecordingLog,
) -> None:
    async def scenario() -> list[Any]:
        async with make_client(recording_log) as svc:
            return list(
                await asyncio.gather(
                    *(post(svc.client, texts=[f"text{i}"]) for i in range(64))
                )
            )

    responses = run(scenario())

    assert [r.status_code for r in responses] == [200] * 64
    assert [r.json()["translations"] for r in responses] == [
        [f"<text{i}>"] for i in range(64)
    ]


# ---- a request that shares a batch with a refused one -------------------------------------


def test_a_refused_request_does_not_take_the_others_of_its_batch_with_it(
    recording_log: RecordingLog,
) -> None:
    async def scenario() -> tuple[Any, Any, Any, FakeEngine]:
        engine = FakeEngine()
        async with make_client(recording_log, engine, inter_threads=1) as svc:
            engine.hold()
            first = asyncio.create_task(post(svc.client, texts=["held"]))
            await engine_started(engine, 1)
            good = asyncio.create_task(post(svc.client, texts=["fine", "also"]))
            bad = asyncio.create_task(post(svc.client, texts=["lost"], tgt="deu_Latn"))
            other = asyncio.create_task(post(svc.client, texts=["last"]))
            await until(lambda: svc.state.batcher.depth == 4)
            engine.release()
            await first
            return await good, await bad, await other, engine

    good, bad, other, engine = run(scenario())

    assert good.status_code == 200
    assert good.json()["translations"] == ["<fine>", "<also>"]
    assert (bad.status_code, error_of(bad)["code"]) == (400, "bad_request")
    assert other.status_code == 200
    assert other.json()["translations"] == ["<last>"]
    assert ("fine", "also") in [call.texts for call in engine.calls]


# ---- the start of the service -------------------------------------------------------------


def test_a_warm_up_that_fails_ends_the_start_with_its_error_and_closes_the_engine(
    recording_log: RecordingLog,
) -> None:
    engine = FakeEngine(warmup_failure=UpstreamError("the library failed"))

    async def scenario() -> list[Any]:
        seen = []
        async with make_app(recording_log, engine) as svc:
            with pytest.raises(UpstreamError, match="the library failed"):
                async with svc.app.router.lifespan_context(svc.app):
                    seen.append("started")
            seen.append((await svc.client.get("/readyz")).status_code)
        return seen

    assert run(scenario()) == [503]
    assert engine.closed == 1


def test_the_engine_is_opened_by_the_start_of_the_service_and_closed_by_its_stop(
    recording_log: RecordingLog, tmp_path: Path
) -> None:
    engine = FakeEngine()
    opened: list[tuple[Path, ServeConfig]] = []

    def open_engine(model: Path, config: ServeConfig) -> FakeEngine:
        opened.append((model, config))
        return engine

    config = serve_config()

    async def scenario() -> list[int]:
        app = create_app(config, model=tmp_path, open_engine=open_engine)
        assert opened == []
        async with app.router.lifespan_context(app):
            assert opened == [(tmp_path, config)]
            assert engine.warmups == 1
            assert engine.closed == 0
        return [engine.closed]

    assert run(scenario()) == [1]


def test_the_engine_is_opened_in_a_worker_thread_so_that_the_loop_stays_free(
    recording_log: RecordingLog, tmp_path: Path
) -> None:
    threads: dict[str, int] = {}

    def open_engine(model: Path, config: ServeConfig) -> FakeEngine:
        threads["open"] = threading.get_ident()
        return FakeEngine()

    async def scenario() -> None:
        threads["loop"] = threading.get_ident()
        app = create_app(serve_config(), model=tmp_path, open_engine=open_engine)
        async with app.router.lifespan_context(app):
            pass

    run(scenario())

    assert threads["open"] != threads["loop"]


def test_an_engine_that_cannot_be_opened_ends_the_start_with_the_error(
    recording_log: RecordingLog, tmp_path: Path
) -> None:
    def open_engine(model: Path, config: ServeConfig) -> FakeEngine:
        raise ConfigError("not an export our manifest vouches for")

    async def scenario() -> None:
        app = create_app(serve_config(), model=tmp_path, open_engine=open_engine)
        with pytest.raises(ConfigError, match="not an export"):
            async with app.router.lifespan_context(app):
                raise AssertionError("the service must not start")

    run(scenario())


def test_create_app_needs_exactly_one_of_an_engine_and_a_model(
    tmp_path: Path,
) -> None:
    with pytest.raises(ConfigError, match="engine or a model"):
        create_app(serve_config())
    with pytest.raises(ConfigError, match="engine or a model"):
        create_app(serve_config(), engine=FakeEngine(), model=tmp_path)  # type: ignore[arg-type]


def test_a_script_converter_in_the_config_needs_an_opener_that_loads_it(
    tmp_path: Path,
) -> None:
    config = serve_config(
        script_converter={
            "ct2_dir": tmp_path / "ct2",
            "spm_model": tmp_path / "m.model",
        }
    )

    with pytest.raises(ConfigError, match="script converter"):
        create_app(config, model=tmp_path)


def test_create_app_with_a_given_engine_does_not_open_anything(
    recording_log: RecordingLog,
) -> None:
    app = create_app(serve_config(), engine=FakeEngine())  # type: ignore[arg-type]

    assert app.title == "rinchen"


def test_the_service_translates_with_a_real_engine_over_its_stand_ins(
    recording_log: RecordingLog, tmp_path: Path
) -> None:
    from tests.support.engines import scripted_engine

    async def scenario() -> Any:
        engine = scripted_engine(tmp_path).engine
        async with make_client(recording_log, engine) as svc:
            return await post(svc.client, texts=["Энэ ном", ""], src="rus_Cyrl")

    response = run(scenario())

    body = response.json()
    body.pop("latency_ms")
    assert response.status_code == 200
    assert body["translations"] == ["Энэ ном", ""]
    assert body["statuses"] == ["ok", "empty"]
    assert body["detected_src"] == ["rus_Cyrl", "rus_Cyrl"]
    assert body["script_purity"] == [1.0, None]
    assert body["model"].startswith("rinchen-")


def test_a_document_the_engine_refuses_does_not_take_the_other_documents_of_its_batch_with_it(
    recording_log: RecordingLog,
) -> None:
    async def scenario() -> list[Any]:
        engine = FakeEngine(document_level=True)
        async with make_client(recording_log, engine, inter_threads=1) as svc:
            engine.hold()
            first = asyncio.create_task(
                post(svc.client, texts=["held"], src=ENG, doc_mode=True)
            )
            await engine_started(engine, 1)
            good = asyncio.create_task(
                post(svc.client, texts=["fine"], src=ENG, doc_mode=True)
            )
            bad = asyncio.create_task(
                post(svc.client, texts=["a <sep> b"], src=ENG, doc_mode=True)
            )
            await until(lambda: svc.state.batcher.depth == 2)
            engine.release()
            return list(await asyncio.gather(first, good, bad))

    first, good, bad = run(scenario())

    assert (first.status_code, good.status_code) == (200, 200)
    assert good.json()["translations"] == ["<fine>"]
    assert (bad.status_code, error_of(bad)["code"]) == (400, "bad_request")
    assert "document 0" in error_of(bad)["message"]


# ---- what the answer is made of -----------------------------------------------------------


def translation(
    status: str, *, purity: float | None = 1.0, text: str = "t"
) -> Translation:
    row = Row("x", ENG, KHK)
    return translation_of(row, text, status, purity=purity)  # type: ignore[arg-type]


def document(*sentences: Translation, text: str = "joined") -> DocumentResult:
    stats: ChunkStats = NO_STATS
    return DocumentResult(text=text, translations=tuple(sentences), stats=stats)


@pytest.mark.parametrize(
    ("statuses", "expected"),
    [
        (("ok", "ok"), "ok"),
        (("ok", "empty"), "ok"),
        (("empty", "empty"), "empty"),
        ((), "empty"),
        (("ok", "truncated", "ok"), "truncated"),
        (("ok", "truncated", "too_long"), "too_long"),
        (("ok", "too_long", "no_output", "truncated"), "no_output"),
        (("empty", "truncated"), "truncated"),
    ],
)
def test_the_status_of_a_document_is_the_worst_status_of_its_sentences(
    statuses: tuple[str, ...], expected: str
) -> None:
    result = document(*(translation(status) for status in statuses))

    assert document_status(result) == expected


def request_for(**fields: Any) -> TranslateRequest:
    return TranslateRequest.model_validate({"texts": ["x"], "tgt": KHK, **fields})


def test_the_answer_for_a_document_has_the_lowest_purity_of_its_sentences_and_no_alternatives() -> (
    None
):
    answers = [
        document(
            translation("ok", purity=0.9), translation("ok", purity=0.4), text="first"
        ),
        document(translation("empty", purity=None), text=""),
    ]

    response = build_response(
        request_for(texts=["a", "b"], src=ENG, doc_mode=True, n_best=2),
        answers,
        model="m",
        latency_ms=1.5,
    )

    assert response.model_dump(mode="json") == {
        "translations": ["first", ""],
        "detected_src": [ENG, ENG],
        "model": "m",
        "latency_ms": 1.5,
        "statuses": ["ok", "empty"],
        "script_purity": [0.4, None],
        "alternatives": [[], []],
    }


def test_the_keys_of_a_request_say_what_a_batch_may_share() -> None:
    options = DecodeOptions()

    assert SentenceKey(options) == SentenceKey(DecodeOptions())
    assert hash(SentenceKey(options)) == hash(SentenceKey(DecodeOptions()))
    assert SentenceKey(options) != SentenceKey(DecodeOptions(beam_size=2))
    assert DocumentKey(ENG, KHK, None, options) != DocumentKey(
        ENG, KHK, "Cyrl", options
    )
    assert DocumentKey(ENG, KHK, None, options) != DocumentKey(None, KHK, None, options)
    assert SentenceKey(options) != DocumentKey(ENG, KHK, None, options)


def test_the_batcher_is_built_with_the_limits_of_the_config(
    recording_log: RecordingLog, monkeypatch: pytest.MonkeyPatch
) -> None:
    built: list[dict[str, Any]] = []

    def recording_batcher(run: Any, **limits: Any) -> MicroBatcher[Any, Any, Any]:
        built.append(limits)
        return MicroBatcher(run, **limits)

    monkeypatch.setattr("rinchen.serve.app.MicroBatcher", recording_batcher)
    config = serve_config(
        max_batch_sentences=7, max_wait_ms=2.5, max_queue_sentences=70, inter_threads=3
    )

    async def scenario() -> None:
        async with make_client(recording_log, config=config):
            pass

    run(scenario())

    assert built == [
        {
            "max_batch_sentences": 7,
            "max_wait_seconds": 0.0025,
            "max_queue_sentences": 70,
            "workers": 3,
        }
    ]


def test_importing_the_app_loads_no_library_that_only_the_engine_needs() -> None:
    # The subprocess imports the very tree this test runs from.
    source_root = str(Path(__file__).resolve().parent.parent.parent / "src")
    code = (
        "import sys\n"
        "sys.path.insert(0, sys.argv[1])\n"
        "import rinchen.serve.app\n"
        "assert rinchen.__file__.startswith(sys.argv[1]), rinchen.__file__\n"
        "loaded = [m for m in ('ctranslate2', 'sentencepiece', 'eole', 'torch') "
        "if m in sys.modules]\n"
        "print(loaded)\n"
    )

    result = subprocess.run(
        [sys.executable, "-I", "-c", code, source_root],
        capture_output=True,
        text=True,
        env={**os.environ, "CUDA_VISIBLE_DEVICES": ""},
        timeout=120,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "[]"
