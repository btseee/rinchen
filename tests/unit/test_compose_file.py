"""The compose file of the CPU container, read statically (D11, Task 10, ruling T20).

`docker/compose.serve.yaml` is parsed as YAML and checked against the Dockerfile, the serving
command and the shipped config; `docker compose config -q` (read-only: it validates the file,
contacts no daemon, builds, pulls and starts nothing) is run when Docker is installed and the
test is SKIPPED with its reason when it is not. The image, `docker compose up`, the stop
behaviour of PID 1 and a health check inside a container were never run: UNMEASURED, and the
file says so.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest
import yaml
from typer import Context
from typer.core import TyperCommand, TyperGroup
from typer.main import get_command
from typer.testing import CliRunner

from rinchen.cli.main import app
from rinchen.config.loader import REPO_ROOT
from rinchen.config.serve import ENV_PREFIX, ServeConfig, load_serve_config
from tests.support import docker_files as d

COMPOSE = REPO_ROOT / "docker" / "compose.serve.yaml"
DOCKERFILE = REPO_ROOT / "docker" / "Dockerfile.cpu"
TINY_CONFIG = REPO_ROOT / "configs" / "serve" / "tiny.yaml"

COMMAND = [
    "serve",
    "--model",
    "/models/tiny",
    "--config",
    "/app/configs/serve/tiny.yaml",
    "--host",
    "0.0.0.0",
    "--port",
    "8000",
]
GPU_KEYS = frozenset({"devices", "device_requests", "gpus", "reservations", "runtime"})
GPU_WORDS = ("nvidia", "gpu")
UNMEASURED_NOTES = (
    "never built",
    "docker compose up",
    "pid 1",
    "health check",
    "no gpu",
)
COMPOSE_TIMEOUT_SECONDS = 60
# What a stop may take beyond `shutdown_timeout_seconds`: `uvicorn` polls every 0.1 s and the
# process has to exit (0.1 to 0.2 s measured on a fake engine; the rest is room for a loaded
# host). The compose file states the formula.
STOP_MARGIN_SECONDS = 5
DURATION = re.compile(r"(\d+(?:\.\d+)?)(ms|us|h|m|s)")
SIZE = re.compile(r"(\d+)([bkmg])b?", re.IGNORECASE)
DURATION_SECONDS = {"h": 3600.0, "m": 60.0, "s": 1.0, "ms": 0.001, "us": 0.000001}
SIZE_BYTES = {"b": 1, "k": 1024, "m": 1024**2, "g": 1024**3}


def load(path: Path) -> dict[str, Any]:
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(document, dict)
    return document


@pytest.fixture(scope="module")
def compose() -> dict[str, Any]:
    return load(COMPOSE)


@pytest.fixture(scope="module")
def service(compose: dict[str, Any]) -> dict[str, Any]:
    (only,) = compose["services"].values()
    assert isinstance(only, dict)
    return only


def seconds(duration: str) -> float:
    found = DURATION.findall(duration)
    assert "".join(number + unit for number, unit in found) == duration, duration
    return sum(float(number) * DURATION_SECONDS[unit] for number, unit in found)


def size_in_bytes(size: str) -> int:
    match = SIZE.fullmatch(size)
    assert match, size
    return int(match.group(1)) * SIZE_BYTES[match.group(2).lower()]


def walk(
    node: object, path: tuple[str, ...] = ()
) -> list[tuple[tuple[str, ...], object]]:
    """Every (path, value) of a parsed document, the mapping keys on the path."""
    found: list[tuple[tuple[str, ...], object]] = [(path, node)]
    if isinstance(node, dict):
        for key, value in node.items():
            found.extend(walk(value, (*path, str(key))))
    elif isinstance(node, list):
        for number, value in enumerate(node):
            found.extend(walk(value, (*path, str(number))))
    return found


def dockerfile_instruction(keyword: str) -> d.Instruction:
    """The one instruction of that keyword in the last stage of the Dockerfile, the image's."""
    instructions = d.parse(DOCKERFILE.read_text(encoding="utf-8"))
    last = max(i.stage for i in instructions)
    (found,) = (i for i in instructions if i.keyword == keyword and i.stage == last)
    return found


def serve_command() -> TyperCommand:
    group = get_command(app)
    assert isinstance(group, TyperGroup)
    command = group.commands["serve"]
    assert isinstance(command, TyperCommand)
    return command


def parse_serve_arguments(command: list[str]) -> Context:
    """The arguments of `rinchen serve` in a command that starts with the subcommand, parsed
    by the real command (its callback does not run; an unknown option or a bad value raises)."""
    assert command[0] == "serve"
    return serve_command().make_context("serve", command[1:])


# --- the file as a whole --------------------------------------------------------------------


def test_there_is_one_service(compose: dict[str, Any]) -> None:
    assert len(compose["services"]) == 1


def test_the_service_builds_the_dockerfile_of_this_repository_from_its_root(
    service: dict[str, Any],
) -> None:
    build = service["build"]
    context = (COMPOSE.parent / build["context"]).resolve()

    assert context == REPO_ROOT
    assert build["dockerfile"] == "docker/Dockerfile.cpu"
    assert (context / build["dockerfile"]).is_file()


def test_the_image_it_names_is_tagged_and_is_not_latest(
    service: dict[str, Any],
) -> None:
    name, _, tag = service["image"].rpartition(":")

    assert name
    assert tag not in ("", "latest")


# --- no GPU ---------------------------------------------------------------------------------


def test_no_gpu_device_runtime_or_reservation_is_asked_for_anywhere(
    compose: dict[str, Any],
) -> None:
    for path, value in walk(compose):
        assert not GPU_KEYS & set(path), f"{'.'.join(path)}: a GPU key"
        words = [*path, value] if isinstance(value, str) else list(path)
        for word in words:
            assert not any(gpu in word.lower() for gpu in GPU_WORDS), (path, value)


def test_the_resources_are_limits_only(service: dict[str, Any]) -> None:
    resources = service["deploy"]["resources"]

    assert set(resources) == {"limits"}


def test_the_container_hides_the_gpu(service: dict[str, Any]) -> None:
    environment = service["environment"]

    assert "CUDA_VISIBLE_DEVICES" in environment
    assert environment["CUDA_VISIBLE_DEVICES"] == ""


# --- exposure and isolation -----------------------------------------------------------------


def test_the_port_is_published_on_the_loopback_only(service: dict[str, Any]) -> None:
    assert service["ports"] == ["127.0.0.1:8000:8000"]


def test_the_model_is_mounted_read_only_from_the_models_directory_of_the_repository(
    service: dict[str, Any],
) -> None:
    (volume,) = service["volumes"]
    default = re.fullmatch(r"\$\{RINCHEN_MODEL_DIR:-([^}]+)\}", volume["source"])

    assert volume["type"] == "bind"
    assert volume["read_only"] is True
    assert volume["target"] == "/models/tiny"
    assert default, volume["source"]
    assert (
        COMPOSE.parent / default.group(1)
    ).resolve() == REPO_ROOT / "models" / "tiny"


def test_the_root_file_system_is_read_only_with_a_tmpfs_for_tmp(
    service: dict[str, Any],
) -> None:
    assert service["read_only"] is True
    assert "/tmp" in [entry.split(":")[0] for entry in service["tmpfs"]]


def test_the_container_has_an_init_process(service: dict[str, Any]) -> None:
    assert service["init"] is True


def test_the_stop_grace_period_covers_the_whole_stop_of_the_tiny_config(
    service: dict[str, Any],
) -> None:
    # The whole stop takes `shutdown_timeout_seconds` at most (the server and the end of the
    # lifespan share it), so Docker's SIGKILL must come later by a margin.
    timeout = load_serve_config(TINY_CONFIG, env={}).shutdown_timeout_seconds

    assert seconds(str(service["stop_grace_period"])) >= timeout + STOP_MARGIN_SECONDS


def test_cpu_and_memory_are_limited(service: dict[str, Any]) -> None:
    limits = service["deploy"]["resources"]["limits"]

    assert float(limits["cpus"]) > 0
    assert size_in_bytes(str(limits["memory"])) > 0


# --- the health check -----------------------------------------------------------------------


def test_the_health_check_asks_the_readiness_route_of_the_published_port(
    service: dict[str, Any],
) -> None:
    check = service["healthcheck"]
    *_, port = service["ports"][0].split(":")

    assert check["test"][0] == "CMD"
    assert check["test"][1] == "python"
    assert f"127.0.0.1:{port}/readyz" in " ".join(check["test"])
    assert seconds(str(check["interval"])) > 0
    assert seconds(str(check["timeout"])) > 0
    assert int(check["retries"]) >= 1


def test_the_health_check_is_the_one_the_image_carries(service: dict[str, Any]) -> None:
    image_command = d.healthcheck_command(dockerfile_instruction("HEALTHCHECK"))

    assert service["healthcheck"]["test"][1:] == image_command


# --- the command ----------------------------------------------------------------------------


def test_the_command_is_the_command_of_the_image(service: dict[str, Any]) -> None:
    image_command = d.exec_form(dockerfile_instruction("CMD").arguments)

    assert service["command"] == COMMAND
    assert service["command"] == image_command


def test_the_command_parses_with_the_options_of_rinchen_serve(
    service: dict[str, Any],
) -> None:
    context = parse_serve_arguments(service["command"])
    given = {name: value for name, value in context.params.items() if value is not None}

    assert given == {
        "model": "/models/tiny",
        "config_file": "/app/configs/serve/tiny.yaml",
        "host": "0.0.0.0",
        "port": 8000,
    }


def test_every_flag_of_the_command_is_listed_by_the_help_of_rinchen_serve(
    service: dict[str, Any],
) -> None:
    flags = [word for word in service["command"] if word.startswith("--")]
    columns = {"COLUMNS": "240", "TERMINAL_WIDTH": "240", "NO_COLOR": "1"}

    result = CliRunner().invoke(app, ["serve", "--help"], env=columns)

    assert result.exit_code == 0
    assert flags
    for flag in flags:
        assert flag in result.output, flag


def test_the_published_container_port_is_the_port_the_command_listens_on(
    service: dict[str, Any],
) -> None:
    context = parse_serve_arguments(service["command"])
    *_, container_port = service["ports"][0].split(":")

    assert context.params["port"] == int(container_port)


def test_the_environment_variables_are_ones_the_service_reads(
    service: dict[str, Any], tmp_path: Path
) -> None:
    given = {
        name: str(value)
        for name, value in service["environment"].items()
        if name.startswith(ENV_PREFIX)
    }
    other = load(TINY_CONFIG) | {"inter_threads": 5, "intra_threads": 3}
    other_config = tmp_path / "other.yaml"
    other_config.write_text(yaml.safe_dump(other), encoding="utf-8")

    config = load_serve_config(other_config, env=given)

    assert given == {
        "RINCHEN_SERVE_INTER_THREADS": "2",
        "RINCHEN_SERVE_INTRA_THREADS": "1",
    }
    assert (config.inter_threads, config.intra_threads) == (2, 1)
    for name in given:
        assert name.removeprefix(ENV_PREFIX).lower() in ServeConfig.model_fields


# --- the shipped config ---------------------------------------------------------------------


def test_the_tiny_config_validates_and_listens_on_the_loopback() -> None:
    config = load_serve_config(TINY_CONFIG, env={})

    assert config.host == "127.0.0.1"
    assert config.port == 8000


def test_only_the_command_line_opens_the_service_to_the_container_network(
    service: dict[str, Any],
) -> None:
    text = TINY_CONFIG.read_text(encoding="utf-8")

    assert "0.0.0.0" not in text
    assert service["command"][service["command"].index("--host") + 1] == "0.0.0.0"


# --- docker compose config -q ---------------------------------------------------------------


def compose_config_quietly(path: Path) -> subprocess.CompletedProcess[str]:
    """`docker compose -f PATH config -q`: it validates the file and prints nothing. It builds,
    pulls and starts nothing and needs no daemon: the child's DOCKER_HOST names a socket that
    does not exist, so that a daemon could not be reached by accident."""
    if shutil.which("docker") is None:
        pytest.skip("docker is not on the PATH: `docker compose config -q` was not run")
    environment = {
        name: value
        for name, value in os.environ.items()
        if not name.startswith(("DOCKER_", "COMPOSE_", "RINCHEN_"))
    }
    environment["DOCKER_HOST"] = "unix:///nonexistent/rinchen-no-daemon.sock"
    completed = subprocess.run(
        ["docker", "compose", "-f", str(path), "config", "-q"],
        cwd=REPO_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        timeout=COMPOSE_TIMEOUT_SECONDS,
        check=False,
    )
    if "is not a docker command" in completed.stderr:
        pytest.skip("the docker compose plugin is not installed: config -q was not run")
    return completed


@pytest.mark.subprocess
def test_docker_compose_accepts_the_file() -> None:
    completed = compose_config_quietly(COMPOSE)

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout == ""


@pytest.mark.subprocess
def test_docker_compose_refuses_a_file_that_is_not_one(tmp_path: Path) -> None:
    broken = tmp_path / "broken.yaml"
    broken.write_text("services:\n  web:\n    imagee: x\n", encoding="utf-8")

    completed = compose_config_quietly(broken)

    assert completed.returncode != 0


# --- what nobody measured -------------------------------------------------------------------


def test_the_compose_file_says_what_was_never_measured() -> None:
    comments = " ".join(
        line.lstrip().lstrip("#").strip().lower()
        for line in COMPOSE.read_text(encoding="utf-8").splitlines()
        if line.lstrip().startswith("#")
    )

    assert "unmeasured" in comments
    for note in UNMEASURED_NOTES:
        assert note in comments, f"the comments do not mention: {note}"
