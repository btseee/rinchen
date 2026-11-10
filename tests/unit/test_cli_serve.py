"""`rinchen serve` (D11, Task 9): what the command does before it binds anything, how the
configuration is layered and the one call it makes to `uvicorn`.

The command touches the network in ONE place, `rinchen.cli.serve.run_server`, which the tests
replace by a recorder (or, for the call itself, `uvicorn.run`); a model is never loaded here:
the export is a fabricated directory the manifest vouches for, and the engine is opened only
when the lifespan runs, which a recorded `run_server` never starts.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest
import uvicorn
from fastapi import FastAPI
from typer.testing import CliRunner

from rinchen.cli.main import app
from rinchen.cli.serve import run_server
from rinchen.config.serve import ServeConfig
from rinchen.export import verified
from rinchen.infer.engine_types import LoadOptions
from tests.support.model_exports import write_model_export
from tests.support.serve_app import RecordingLog, serve_config

runner = CliRunner()

OPTIONS = [
    "--model",
    "--config",
    "--host",
    "--port",
    "--device",
    "--compute-type",
    "--inter-threads",
    "--intra-threads",
    "--beam-size",
    "--script-mask",
    "--max-batch-sentences",
    "--max-wait-ms",
    "--max-queue-sentences",
    "--max-texts",
    "--max-text-chars",
    "--max-body-bytes",
    "--request-timeout-seconds",
    "--shutdown-timeout-seconds",
    "--log-level",
    "--script-converter-ct2",
    "--script-converter-spm",
]


@dataclass
class Started:
    """What the command handed to `run_server`, and the order of what it did before."""

    calls: list[tuple[FastAPI, ServeConfig]] = field(default_factory=list)
    events: list[str] = field(default_factory=list)


@pytest.fixture
def started(monkeypatch: pytest.MonkeyPatch) -> Started:
    seen = Started()
    real_verify = verified.verify_export

    def verify_export(directory: Path) -> Any:
        seen.events.append("verify")
        return real_verify(directory)

    def run_server(served: FastAPI, config: ServeConfig) -> None:
        seen.events.append("run_server")
        seen.calls.append((served, config))

    monkeypatch.setattr("rinchen.cli.serve.verify_export", verify_export)
    monkeypatch.setattr("rinchen.cli.serve.run_server", run_server)
    return seen


@pytest.fixture
def export(tmp_path: Path) -> Path:
    return write_model_export(tmp_path / "model").directory


@pytest.fixture(autouse=True)
def no_model_is_loaded(
    monkeypatch: pytest.MonkeyPatch, recording_log: RecordingLog
) -> None:
    """A model is never loaded, and the service's logger is the recording one: the first call
    of the real `get` would configure the root logger on the stream of this test."""

    def refuse(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("the command must not load a model")

    monkeypatch.setattr("rinchen.infer.engine.Engine.open", refuse)


def serve(*args: str, env: dict[str, str] | None = None) -> Any:
    return runner.invoke(app, ["serve", *args], env=env)


# ---- the options --------------------------------------------------------------------------


def test_serve_help_lists_every_option_of_the_command() -> None:
    result = serve("--help", env={"COLUMNS": "250"})

    assert result.exit_code == 0
    for option in OPTIONS:
        assert option in result.stdout, option


def test_the_model_may_come_from_the_environment(
    started: Started, export: Path
) -> None:
    result = serve(env={"RINCHEN_MODEL": str(export)})

    assert result.exit_code == 0, result.stderr
    assert len(started.calls) == 1


def test_a_missing_model_is_a_usage_error_and_nothing_runs(started: Started) -> None:
    result = serve()

    assert result.exit_code == 2
    assert started.events == []


def test_the_device_auto_is_refused_before_anything_runs(
    started: Started, export: Path
) -> None:
    result = serve("--model", str(export), "--device", "auto")

    assert result.exit_code == 2
    assert started.events == []


# ---- before anything is bound -------------------------------------------------------------


def test_a_good_export_is_verified_and_then_served_with_the_config_of_the_flags(
    started: Started, export: Path
) -> None:
    result = serve(
        "--model",
        str(export),
        "--host",
        "0.0.0.0",
        "--port",
        "9001",
        "--inter-threads",
        "3",
        "--intra-threads",
        "2",
        "--beam-size",
        "2",
        "--script-mask",
        "off",
        "--max-batch-sentences",
        "16",
        "--max-wait-ms",
        "2.5",
        "--max-queue-sentences",
        "500",
        "--max-texts",
        "8",
        "--max-text-chars",
        "400",
        "--max-body-bytes",
        "4096",
        "--request-timeout-seconds",
        "9.5",
        "--shutdown-timeout-seconds",
        "4",
        "--log-level",
        "warning",
    )

    assert result.exit_code == 0, result.stderr
    assert result.stdout == ""
    assert started.events == ["verify", "run_server"]
    ((served, config),) = started.calls
    assert isinstance(served, FastAPI)
    assert config == ServeConfig(
        host="0.0.0.0",
        port=9001,
        inter_threads=3,
        intra_threads=2,
        beam_size=2,
        script_mask="off",
        max_batch_sentences=16,
        max_wait_ms=2.5,
        max_queue_sentences=500,
        max_texts=8,
        max_text_chars=400,
        max_body_bytes=4096,
        request_timeout_seconds=9.5,
        shutdown_timeout_seconds=4.0,
        log_level="warning",
    )


def test_a_relative_model_path_is_made_absolute_before_the_service_is_built(
    started: Started, export: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(export.parent)

    result = serve("--model", export.name)

    assert result.exit_code == 0, result.stderr
    ((served, _),) = started.calls
    assert served.state.serving.model == export


def test_the_defaults_are_those_of_the_config_and_the_host_is_the_loopback(
    started: Started, export: Path
) -> None:
    result = serve("--model", str(export))

    assert result.exit_code == 0, result.stderr
    ((_, config),) = started.calls
    assert config == ServeConfig()
    assert config.host == "127.0.0.1"


def test_a_directory_of_the_old_format_is_refused_with_the_remedy_and_nothing_is_run(
    started: Started, export: Path
) -> None:
    (export / "rinchen.json").rename(export / "export.json")

    result = serve("--model", str(export))

    assert result.exit_code == 2
    assert result.stdout == ""
    assert "rinchen export convert" in result.stderr
    assert started.events == ["verify"]


def test_a_missing_directory_and_a_changed_file_are_refused_with_exit_2(
    started: Started, export: Path, tmp_path: Path
) -> None:
    missing = serve("--model", str(tmp_path / "nowhere"))
    (export / "model.bin").write_bytes(b"changed")
    changed = serve("--model", str(export))

    assert (missing.exit_code, changed.exit_code) == (2, 2)
    assert "does not exist" in missing.stderr
    assert "differs from the manifest" in changed.stderr
    assert "run_server" not in started.events


def test_a_manifest_that_cannot_be_read_is_exit_3(
    started: Started, export: Path
) -> None:
    (export / "manifest.json").write_text("{not json", encoding="utf-8")

    result = serve("--model", str(export))

    assert result.exit_code == 3
    assert "not a readable Rinchen manifest" in result.stderr
    assert "run_server" not in started.events


# ---- the configuration --------------------------------------------------------------------


def test_the_flags_win_over_the_environment_and_the_environment_over_the_file(
    started: Started, export: Path, tmp_path: Path
) -> None:
    config_file = tmp_path / "serve.yaml"
    config_file.write_text(
        "port: 7001\ninter_threads: 5\nbeam_size: 3\nhost: 10.0.0.1\n", encoding="utf-8"
    )

    result = serve(
        "--model",
        str(export),
        "--config",
        str(config_file),
        "--port",
        "7003",
        env={
            "RINCHEN_SERVE_PORT": "7002",
            "RINCHEN_SERVE_INTER_THREADS": "6",
            "RINCHEN_SERVE_MAX_TEXTS": "12",
        },
    )

    assert result.exit_code == 0, result.stderr
    ((_, config),) = started.calls
    assert (config.port, config.inter_threads, config.beam_size, config.host) == (
        7003,
        6,
        3,
        "10.0.0.1",
    )
    assert config.max_texts == 12


@pytest.mark.parametrize(
    ("args", "message"),
    [
        (["--compute-type", "auto"], "auto"),
        (["--port", "0"], "port"),
        (["--inter-threads", "0"], "inter_threads"),
        (["--max-queue-sentences", "3", "--max-texts", "10"], "max_queue_sentences"),
        (["--script-converter-ct2", "ct2dir"], "spm_model"),
        (["--config", "no-such-file.yaml"], "not found"),
    ],
)
def test_a_config_that_is_refused_is_exit_2_and_no_model_is_touched(
    args: list[str], message: str, started: Started, export: Path
) -> None:
    result = serve("--model", str(export), *args)

    assert result.exit_code == 2
    assert message in result.stderr
    assert started.events == []


def test_a_bad_environment_and_an_unknown_key_of_the_file_are_exit_2(
    started: Started, export: Path, tmp_path: Path
) -> None:
    config_file = tmp_path / "serve.yaml"
    config_file.write_text("model: somewhere\n", encoding="utf-8")

    bad_environment = serve(
        "--model", str(export), env={"RINCHEN_SERVE_PORT": "eighty"}
    )
    unknown_key = serve("--model", str(export), "--config", str(config_file))

    assert (bad_environment.exit_code, unknown_key.exit_code) == (2, 2)
    assert "RINCHEN_SERVE_" in bad_environment.stderr
    assert "model" in unknown_key.stderr
    assert started.events == []


# ---- the script converter -----------------------------------------------------------------


def test_the_app_opens_its_engine_with_the_converter_the_flags_name(
    started: Started, export: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    opened: list[tuple[Path, LoadOptions, tuple[Path, Path] | None]] = []

    def open_engine(
        model: Path, load: LoadOptions, converter_paths: tuple[Path, Path] | None
    ) -> str:
        opened.append((model, load, converter_paths))
        return "an engine"

    monkeypatch.setattr("rinchen.cli.serve.open_engine_with_converter", open_engine)
    monkeypatch.chdir(tmp_path)
    ct2_dir, spm_model = tmp_path / "ct2", tmp_path / "char.model"

    result = serve(
        "--model",
        str(export),
        "--script-converter-ct2",
        "ct2",
        "--script-converter-spm",
        "char.model",
        "--inter-threads",
        "3",
        "--compute-type",
        "int8",
    )
    ((served, config),) = started.calls
    engine = served.state.serving.open_engine(export, config)

    assert result.exit_code == 0, result.stderr
    assert engine == "an engine"
    assert opened == [
        (
            export,
            LoadOptions(
                device="cpu", compute_type="int8", inter_threads=3, intra_threads=1
            ),
            (ct2_dir, spm_model),
        )
    ]


def test_without_a_converter_the_engine_is_opened_without_one(
    started: Started, export: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    opened: list[tuple[Path, LoadOptions, tuple[Path, Path] | None]] = []

    def open_engine(
        model: Path, load: LoadOptions, converter_paths: tuple[Path, Path] | None
    ) -> str:
        opened.append((model, load, converter_paths))
        return "an engine"

    monkeypatch.setattr("rinchen.cli.serve.open_engine_with_converter", open_engine)

    result = serve("--model", str(export))
    ((served, config),) = started.calls
    served.state.serving.open_engine(export, config)

    assert result.exit_code == 0, result.stderr
    assert opened == [(export, LoadOptions(inter_threads=2), None)]


# ---- the one call to uvicorn --------------------------------------------------------------


@pytest.fixture
def uvicorn_calls(monkeypatch: pytest.MonkeyPatch) -> list[tuple[Any, dict[str, Any]]]:
    calls: list[tuple[Any, dict[str, Any]]] = []

    def fake_run(served: Any, **kwargs: Any) -> None:
        calls.append((served, kwargs))

    monkeypatch.setattr(uvicorn, "run", fake_run)
    return calls


@pytest.fixture
def rinchen_logger_level() -> Iterator[None]:
    logger = logging.getLogger("rinchen")
    before = logger.level
    yield
    logger.setLevel(before)


def test_run_server_gives_uvicorn_its_own_logging_off_and_the_first_half_of_the_stop_budget(
    uvicorn_calls: list[tuple[Any, dict[str, Any]]],
    recording_log: RecordingLog,
    rinchen_logger_level: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    served = FastAPI()
    configured: list[str] = []
    monkeypatch.setattr(
        "rinchen.utils.logging.get",
        lambda name: configured.append(name) or recording_log,
    )

    run_server(served, serve_config(shutdown_timeout_seconds=20.0))

    assert configured == ["rinchen.serve"]

    assert uvicorn_calls == [
        (
            served,
            {
                "host": "127.0.0.1",
                "port": 8000,
                "log_config": None,
                "log_level": "info",
                "access_log": False,
                "timeout_graceful_shutdown": 10,
                "server_header": False,
            },
        )
    ]


def test_run_server_takes_the_host_the_port_and_the_log_level_from_the_config(
    uvicorn_calls: list[tuple[Any, dict[str, Any]]],
    recording_log: RecordingLog,
    rinchen_logger_level: None,
) -> None:
    run_server(
        FastAPI(),
        serve_config(
            host="0.0.0.0", port=9100, log_level="warning", shutdown_timeout_seconds=2.5
        ),
    )

    ((_, kwargs),) = uvicorn_calls
    assert (kwargs["host"], kwargs["port"], kwargs["log_level"]) == (
        "0.0.0.0",
        9100,
        "warning",
    )
    assert kwargs["timeout_graceful_shutdown"] == 2
    assert logging.getLogger("rinchen").level == logging.WARNING


def test_the_command_ends_with_the_run_of_the_server(
    export: Path,
    uvicorn_calls: list[tuple[Any, dict[str, Any]]],
    recording_log: RecordingLog,
    rinchen_logger_level: None,
) -> None:
    result = serve("--model", str(export), "--port", "9200")

    assert result.exit_code == 0, result.stderr
    ((served, kwargs),) = uvicorn_calls
    assert isinstance(served, FastAPI)
    assert kwargs["port"] == 9200
