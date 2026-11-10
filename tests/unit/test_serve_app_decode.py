"""The service (D11, fix wave): how the beam and the hypotheses of a request meet the beam the
service was configured with.

A request that names no `beam_size` is decoded with the beam of the config, so `n_best` is judged
against THAT beam, not against the schema's default of 4: a service with a beam of 8 takes
`{"n_best": 6}`, one with a beam of 2 refuses `{"n_best": 3}`, and the refusal speaks the terms
of the API (`n_best`, `beam_size`), never those of the engine's options. A request that does name
a `beam_size` is judged against it by the schema (a 422 that names `n_best`).
"""

from __future__ import annotations

from typing import Any

import pytest

from rinchen.infer.engine_types import DecodeOptions
from tests.support.serve_app import (
    SECRET,
    FakeEngine,
    RecordingLog,
    error_of,
    make_client,
    post,
    run,
)


def served(
    log: RecordingLog, configured_beam: int, **fields: Any
) -> tuple[Any, FakeEngine]:
    """The answer of a service configured with `configured_beam` to a request with `fields`,
    and its engine."""

    async def scenario() -> tuple[Any, FakeEngine]:
        engine = FakeEngine()
        async with make_client(log, engine, beam_size=configured_beam) as svc:
            return await post(svc.client, **fields), engine

    return run(scenario())


@pytest.mark.parametrize(
    ("beam_size", "n_best"), [(8, 6), (8, 8), (4, 4), (2, 2), (2, 1)]
)
def test_hypotheses_up_to_the_configured_beam_are_served(
    recording_log: RecordingLog, beam_size: int, n_best: int
) -> None:
    response, engine = served(recording_log, beam_size, n_best=n_best)

    assert response.status_code == 200
    (call,) = engine.calls
    assert call.options is not None
    assert (call.options.beam_size, call.options.n) == (beam_size, n_best)


@pytest.mark.parametrize(
    ("beam_size", "n_best", "message"),
    [
        (
            8,
            9,
            (
                "n_best (9) must not exceed beam_size (8): the beam holds no more "
                "hypotheses, and this service decodes with a beam_size of 8 when a "
                "request names none"
            ),
        ),
        (
            4,
            5,
            (
                "n_best (5) must not exceed beam_size (4): the beam holds no more "
                "hypotheses, and this service decodes with a beam_size of 4 when a "
                "request names none"
            ),
        ),
        (
            2,
            3,
            (
                "n_best (3) must not exceed beam_size (2): the beam holds no more "
                "hypotheses, and this service decodes with a beam_size of 2 when a "
                "request names none"
            ),
        ),
    ],
)
def test_more_hypotheses_than_the_configured_beam_are_a_400_in_the_terms_of_the_api(
    recording_log: RecordingLog, beam_size: int, n_best: int, message: str
) -> None:
    response, engine = served(recording_log, beam_size, texts=[SECRET], n_best=n_best)

    assert response.status_code == 400
    assert error_of(response) == {"code": "bad_request", "message": message}
    assert engine.calls == []
    assert SECRET not in response.text


def test_the_refusal_names_no_option_of_the_engine(
    recording_log: RecordingLog,
) -> None:
    response, _ = served(recording_log, 2, n_best=3)

    message = error_of(response)["message"]
    for internal in ("DecodeOptions", "CTranslate2", "TeacherOptions"):
        assert internal not in message


@pytest.mark.parametrize(
    ("beam_size", "n_best", "sent"), [(8, 6, 4), (2, 3, 4), (4, 7, 6), (2, 5, 4)]
)
def test_a_beam_the_request_names_is_the_one_its_hypotheses_are_judged_against(
    recording_log: RecordingLog, beam_size: int, n_best: int, sent: int
) -> None:
    # The schema's 422 for a request that is wrong by itself, whatever the config's beam; a
    # request that fits the beam it names is served with it even if the config's is smaller.
    response, engine = served(recording_log, beam_size, n_best=n_best, beam_size=sent)

    if n_best > sent:
        assert response.status_code == 422
        assert error_of(response) == {
            "code": "validation_error",
            "message": f"n_best: Value error, n_best ({n_best}) must not exceed "
            f"beam_size ({sent}): the beam holds no more hypotheses",
        }
        assert engine.calls == []
    else:
        assert response.status_code == 200
        (call,) = engine.calls
        assert call.options == DecodeOptions(
            beam_size=sent,
            n=n_best,
            max_batch_size=32,
            max_decoding_length=256,
            max_source_pieces=256,
            script_mask="retry",
        )


def test_a_narrower_beam_than_the_config_is_served_when_the_hypotheses_fit_it(
    recording_log: RecordingLog,
) -> None:
    response, engine = served(recording_log, 8, n_best=2, beam_size=3)

    assert response.status_code == 200
    (call,) = engine.calls
    assert call.options is not None
    assert (call.options.beam_size, call.options.n) == (3, 2)
