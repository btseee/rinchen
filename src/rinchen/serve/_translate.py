"""What the service asks of the engine, and what it makes of the answers (D11, T15, T16).

A request becomes a `BatchKey` and one `Job` per text. The key is what a batch may share (the
decode options the library takes once per call; for documents also the languages), so the
micro-batcher never puts two kinds of request in one call; a `Job` carries its text as an
engine `Row` and the number of the request it came from. `run_jobs` is the call the batcher's
worker makes: one engine call per batch, except when the engine refuses it for a reason that
lies in one request (a language it does not write, a document with a reserved marker): the
engine's errors are all-or-nothing, so a batch that holds that request and others is run again
request by request and only the guilty request's jobs answer with a `Refused` (the engine's error
as a value: the batcher would raise an exception that was returned, and a refused job must
not fail its neighbours). An error of the library (`UpstreamError`) is not isolated: it says
nothing about which request caused it.

`ServingEngine` (`rinchen.serve.engine`) is what the service reads of an engine. The status of
a document is `rinchen.infer.engine_types.document_status`, the one `rinchen translate` reports.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from itertools import groupby

from rinchen.config.serve import ServeConfig
from rinchen.errors import ConfigError, DataError
from rinchen.infer.engine_types import (
    DecodeOptions,
    DocumentResult,
    Row,
    ScriptName,
    Status,
    Translation,
    document_status,
)
from rinchen.serve.engine import ServingEngine
from rinchen.serve.schemas import TranslateRequest, TranslateResponse


@dataclass(frozen=True, slots=True)
class SentenceKey:
    """Texts translated one by one: what they share is how they are decoded."""

    options: DecodeOptions


@dataclass(frozen=True, slots=True)
class DocumentKey:
    """Texts translated as documents: they also share the languages the engine takes once."""

    src: str | None
    tgt: str
    script: ScriptName | None
    options: DecodeOptions


BatchKey = SentenceKey | DocumentKey
Answer = Translation | DocumentResult


@dataclass(frozen=True, slots=True)
class Refused:
    """The answer of a job whose request the engine refused: what it said."""

    error: ConfigError | DataError


@dataclass(frozen=True, slots=True)
class Job:
    """One text of one request, as the batcher carries it."""

    request: int
    row: Row


def decode_options(body: TranslateRequest, config: ServeConfig) -> DecodeOptions:
    """How a request is decoded: its own beam and hypotheses, the config's other limits; a
    request that names no beam gets the config's.

    Raises:
        ConfigError: more hypotheses were asked for than the beam holds, said in the terms of
            the API (the schema has refused a request that names a beam too small itself).
    """
    named = "beam_size" in body.model_fields_set
    beam = body.beam_size if named else config.beam_size
    if body.n_best > beam:
        raise ConfigError(
            f"n_best ({body.n_best}) must not exceed beam_size ({beam}): the beam holds "
            f"no more hypotheses, and this service decodes with a beam_size of {beam} "
            "when a request names none"
        )
    return DecodeOptions(
        beam_size=beam,
        n=body.n_best,
        max_decoding_length=config.max_decoding_length,
        max_source_pieces=config.max_source_pieces,
        max_batch_size=config.max_batch_sentences,
        script_mask=config.script_mask,
    )


def plan(
    body: TranslateRequest, config: ServeConfig, request: int
) -> tuple[BatchKey, list[Job]]:
    """The key and the jobs of a request. `auto` is a source the engine reads off the script
    of each text (and reports as the translation's `src`), so it goes in as none.

    Raises:
        ConfigError: as `decode_options`.
    """
    options = decode_options(body, config)
    src = None if body.src == "auto" else body.src
    jobs = [Job(request, Row(text, src, body.tgt, body.script)) for text in body.texts]
    if body.doc_mode:
        return DocumentKey(src, body.tgt, body.script, options), jobs
    return SentenceKey(options), jobs


def run_jobs(
    engine: ServingEngine, key: BatchKey, jobs: Sequence[Job]
) -> list[Answer | Refused]:
    """One answer per job, in order: what the engine says for the batch, or, when the batch
    holds several requests and the engine refuses it, the answers of each request alone,
    with a `Refused` for each job of a request it refused.

    Raises:
        ConfigError, DataError: the engine refused a batch of one request.
        UpstreamError: the library failed; the whole batch is lost.
    """
    try:
        return list(_call(engine, key, [job.row for job in jobs]))
    except ConfigError, DataError:
        requests = [list(group) for _, group in groupby(jobs, key=lambda j: j.request)]
        if len(requests) < 2:
            raise
    answers: list[Answer | Refused] = []
    for request_jobs in requests:
        try:
            answers.extend(_call(engine, key, [job.row for job in request_jobs]))
        except (ConfigError, DataError) as error:
            answers.extend([Refused(error)] * len(request_jobs))
    return answers


def _call(engine: ServingEngine, key: BatchKey, rows: list[Row]) -> Sequence[Answer]:
    if isinstance(key, DocumentKey):
        return engine.translate_documents(
            [row.text for row in rows], key.src, key.tgt, key.options, script=key.script
        )
    return engine.translate(rows, key.options)


def statuses_of(answers: Sequence[Answer]) -> list[Status]:
    """The status of each answer: a translation's own, a document's worst."""
    return [
        document_status(answer) if isinstance(answer, DocumentResult) else answer.status
        for answer in answers
    ]


def build_response(
    body: TranslateRequest,
    answers: Sequence[Answer],
    *,
    model: str,
    latency_ms: float,
) -> TranslateResponse:
    """The response to `body`: one entry per text. A document says the language it was
    given, its worst status and the lowest purity of its sentences, and has no alternatives."""
    texts: list[str] = []
    sources: list[str] = []
    purities: list[float | None] = []
    alternatives: list[list[str]] = []
    for answer in answers:
        if isinstance(answer, DocumentResult):
            shares = [
                t.script_purity
                for t in answer.translations
                if t.script_purity is not None
            ]
            texts.append(answer.text)
            sources.append(body.src)
            purities.append(min(shares, default=None))
            alternatives.append([])
        else:
            texts.append(answer.text)
            sources.append(answer.src)
            purities.append(answer.script_purity)
            alternatives.append(list(answer.alternatives))
    return TranslateResponse(
        translations=texts,
        detected_src=sources,
        model=model,
        latency_ms=latency_ms,
        statuses=list(statuses_of(answers)),
        script_purity=purities,
        alternatives=alternatives if body.n_best > 1 else None,
    )
