"""`rinchen serve`: run the translation service over an export of ours (D11, T19).

    rinchen serve --model DIR [--config FILE] [--host H] [--port P] [--device cpu|cuda]
        [--compute-type default] [--inter-threads 2] [--intra-threads 1] [--beam-size 4]
        [--script-mask retry|always|off] [--max-batch-sentences 32] [--max-wait-ms 5]
        [--max-queue-sentences 1024] [--max-texts 64] [--max-text-chars 5000]
        [--max-body-bytes 1048576] [--request-timeout-seconds 30]
        [--shutdown-timeout-seconds 20] [--log-level info]
        [--script-converter-ct2 DIR --script-converter-spm FILE]

`--model` (or `RINCHEN_MODEL`) is the export directory `rinchen export convert` made. The
settings are layered, the last one winning: the defaults of `rinchen.config.serve.ServeConfig`,
the YAML file of `--config` (`configs/serve/*.yaml`), the `RINCHEN_SERVE_<FIELD>` environment
variables and these flags. The service listens on 127.0.0.1 unless told otherwise and has no
authentication: put a gateway in front of it.

The command builds the settings and has the manifest vouch for the export BEFORE it binds a
port, so a refused model or setting is an exit code and nothing listens. The engine itself is
opened in the lifespan of the app, after `uvicorn` started it; a model that then fails to load
ends the server with a non-zero status. One process serves: more processes would load the model
more times, so scale with `--inter-threads` and with replicas. On SIGTERM the server stops
accepting, waits for the requests in flight, drains the queue and exits with the status of the
signal (143 in a shell), not 0.

`--script-converter-ct2` and `--script-converter-spm` (both or neither) attach D5's script
converter, as `rinchen translate` does. `--device cuda` is an operator's choice that was never
run here (unmeasured); `auto` is refused.

Exit codes: 2 for a setting or a model that is refused (a directory of the old format, a file
the manifest does not vouch for), 3 for files that cannot be read, 1 for any other failure of
the command.
"""

from __future__ import annotations

import logging
import math
import os
from pathlib import Path
from typing import Annotated, Literal

import typer
import uvicorn
from fastapi import FastAPI

from rinchen.cli.errors import rinchen_errors_as_exit_codes
from rinchen.cli.translate import open_engine as open_engine_with_converter
from rinchen.config.serve import ServeConfig, load_serve_config
from rinchen.export.verified import verify_export
from rinchen.serve._translate import ServingEngine
from rinchen.serve.app import create_app, load_options
from rinchen.utils import logging as rinchen_logging


def run_server(app: FastAPI, config: ServeConfig) -> None:
    """Serve `app` with `uvicorn`, in this process, until it is stopped. The one place the
    command binds a port (the tests replace it). `uvicorn` logs nothing of its own (`log_config`
    is None: the service logs through `rinchen.utils.logging`, and the access log is off:
    a request is logged without its text by the service itself) and waits the shutdown timeout
    of the config for the requests in flight."""
    # Configures the JSON handler that uvicorn's own few lines go through as well.
    rinchen_logging.get("rinchen.serve")
    logging.getLogger("rinchen").setLevel(config.log_level.upper())
    uvicorn.run(
        app,
        host=config.host,
        port=config.port,
        log_config=None,
        log_level=config.log_level,
        access_log=False,
        timeout_graceful_shutdown=math.ceil(config.shutdown_timeout_seconds),
        server_header=False,
    )


def _opener(model: Path, config: ServeConfig) -> ServingEngine:
    """Opens the engine of the service, with D5's script converter when the config names one."""
    converter = config.script_converter
    paths = (
        None
        if converter is None
        else (Path(converter.ct2_dir), Path(converter.spm_model))
    )
    engine: ServingEngine = open_engine_with_converter(
        model, load_options(config), paths
    )
    return engine


def serve_command(
    model: Annotated[
        Path,
        typer.Option(
            "--model",
            envvar="RINCHEN_MODEL",
            help="The export directory `rinchen export convert` made.",
        ),
    ],
    config_file: Annotated[
        Path | None,
        typer.Option(
            "--config", help="A YAML file of settings (configs/serve/*.yaml)."
        ),
    ] = None,
    host: Annotated[
        str | None,
        typer.Option("--host", help="Address to listen on (default 127.0.0.1)."),
    ] = None,
    port: Annotated[
        int | None, typer.Option("--port", help="Port (default 8000).")
    ] = None,
    device: Annotated[
        Literal["cpu", "cuda"] | None,
        typer.Option(
            "--device",
            help="Where CTranslate2 runs; never auto, which would pick the GPU.",
        ),
    ] = None,
    compute_type: Annotated[
        str | None,
        typer.Option(
            "--compute-type",
            help="CTranslate2's compute type; default loads what the export stores.",
        ),
    ] = None,
    inter_threads: Annotated[
        int | None, typer.Option("--inter-threads", help="Batches decoded at once.")
    ] = None,
    intra_threads: Annotated[
        int | None, typer.Option("--intra-threads", help="Threads of one batch.")
    ] = None,
    beam_size: Annotated[
        int | None,
        typer.Option("--beam-size", help="Beam of a request that names none."),
    ] = None,
    script_mask: Annotated[
        Literal["retry", "always", "off"] | None,
        typer.Option("--script-mask", help="How a wrong-script output is handled."),
    ] = None,
    max_batch_sentences: Annotated[
        int | None,
        typer.Option("--max-batch-sentences", help="Sentences of one engine call."),
    ] = None,
    max_wait_ms: Annotated[
        float | None,
        typer.Option(
            "--max-wait-ms", help="How long a sentence waits for a fuller batch."
        ),
    ] = None,
    max_queue_sentences: Annotated[
        int | None,
        typer.Option(
            "--max-queue-sentences", help="Sentences that may wait; more is a 503."
        ),
    ] = None,
    max_texts: Annotated[
        int | None, typer.Option("--max-texts", help="Texts of one request.")
    ] = None,
    max_text_chars: Annotated[
        int | None, typer.Option("--max-text-chars", help="Characters of one text.")
    ] = None,
    max_body_bytes: Annotated[
        int | None, typer.Option("--max-body-bytes", help="Bytes of one request body.")
    ] = None,
    request_timeout_seconds: Annotated[
        float | None,
        typer.Option(
            "--request-timeout-seconds",
            help="How long a request may take; more is a 504.",
        ),
    ] = None,
    shutdown_timeout_seconds: Annotated[
        float | None,
        typer.Option(
            "--shutdown-timeout-seconds",
            help="How long a stop waits for the requests in flight.",
        ),
    ] = None,
    log_level: Annotated[
        Literal["critical", "error", "warning", "info", "debug"] | None,
        typer.Option("--log-level", help="Log level of the service."),
    ] = None,
    script_converter_ct2: Annotated[
        Path | None,
        typer.Option(
            "--script-converter-ct2",
            help="D5's script converter, its CTranslate2 directory (with the next option).",
        ),
    ] = None,
    script_converter_spm: Annotated[
        Path | None,
        typer.Option(
            "--script-converter-spm",
            help="D5's script converter, its SentencePiece model (with the previous option).",
        ),
    ] = None,
) -> None:
    """Serve translations over HTTP from an export of ours.

    The settings and the export are checked before a port is bound; the model is loaded when
    the server starts."""
    with rinchen_errors_as_exit_codes():
        converter = {
            name: Path(os.path.abspath(path))
            for name, path in (
                ("ct2_dir", script_converter_ct2),
                ("spm_model", script_converter_spm),
            )
            if path is not None
        }
        config = load_serve_config(
            config_file,
            overrides={
                "host": host,
                "port": port,
                "device": device,
                "compute_type": compute_type,
                "inter_threads": inter_threads,
                "intra_threads": intra_threads,
                "beam_size": beam_size,
                "script_mask": script_mask,
                "max_batch_sentences": max_batch_sentences,
                "max_wait_ms": max_wait_ms,
                "max_queue_sentences": max_queue_sentences,
                "max_texts": max_texts,
                "max_text_chars": max_text_chars,
                "max_body_bytes": max_body_bytes,
                "request_timeout_seconds": request_timeout_seconds,
                "shutdown_timeout_seconds": shutdown_timeout_seconds,
                "log_level": log_level,
                "script_converter": converter or None,
            },
        )
        model_dir = Path(os.path.abspath(model))
        verify_export(model_dir)
        run_server(create_app(config, model=model_dir, open_engine=_opener), config)
