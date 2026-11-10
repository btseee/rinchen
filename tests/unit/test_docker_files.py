"""The CPU container recipe, read statically (D11, Task 10, ruling T20).

`docker/Dockerfile.cpu` and `.dockerignore` are checked as text: no image is built, pulled or run
here (the machine has no network and no authorisation for it), so every claim about the image
itself (that it builds, its size, how long it takes to start, how PID 1 stops, a health check
inside a container) is UNMEASURED and the files say so. What a static reading can prove is
proven: the base images and their tags, that the dependencies install from the lock without the
torch and CUDA wheels (the skip list equals the lock's own list of them), that the vendored
wheels the lock names are copied before the sync, that the process does not run as root and
hides the GPU, the health check, the exec form of the entrypoint and that every file the image
copies exists.
"""

from __future__ import annotations

import posixpath
import re
from collections.abc import Sequence
from pathlib import Path

import pytest

from rinchen.config.loader import REPO_ROOT
from tests.support import docker_files as d

DOCKERFILE = REPO_ROOT / "docker" / "Dockerfile.cpu"
DOCKERIGNORE = REPO_ROOT / ".dockerignore"
LOCK = REPO_ROOT / "uv.lock"

BASE_IMAGE = "python:3.14-slim"
UV_IMAGE_PREFIX = "ghcr.io/astral-sh/uv:"
IMAGE_REFERENCE = re.compile(r"^[a-z0-9][\w./-]*:[\w][\w.-]*$")
UV_VERSION_TAG = re.compile(r"\d+\.\d+\.\d+")
GPU_IMAGE_WORDS = ("cuda", "nvidia", "pytorch")
# The command of the image, as a literal: the compose file's `command` is held to the same one.
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
EXCLUDED_FROM_THE_CONTEXT = (
    ".venv",
    ".git",
    "data",
    "models",
    "tests",
    "docs",
    ".superpowers",
)
# What a reader of the file must be told was never measured (the phrases the comments use).
UNMEASURED_NOTES = (
    "base image tag",
    "uv image tag",
    "never built",
    "image size",
    "start time",
    "pid 1",
    "no gpu image",
)


@pytest.fixture(scope="module")
def instructions() -> list[d.Instruction]:
    return d.parse(DOCKERFILE.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def stages(instructions: list[d.Instruction]) -> list[d.Stage]:
    return d.stages(instructions)


@pytest.fixture(scope="module")
def final_stage(stages: list[d.Stage]) -> d.Stage:
    return stages[-1]


@pytest.fixture(scope="module")
def syncs(instructions: list[d.Instruction]) -> list[list[str]]:
    return d.uv_sync_commands(instructions)


def index_of_first_sync(instructions: Sequence[d.Instruction]) -> int:
    return next(
        number
        for number, instruction in enumerate(instructions)
        if instruction.keyword == "RUN" and d.uv_sync_commands([instruction])
    )


def environment_of(stage: d.Stage) -> dict[str, str]:
    values: dict[str, str] = {}
    for instruction in stage.instructions:
        if instruction.keyword == "ENV":
            values.update(d.environment(instruction))
    return values


def comments_of(path: Path) -> str:
    return " ".join(
        line.lstrip().lstrip("#").strip().lower()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.lstrip().startswith("#")
    )


def normal(source: str) -> str:
    """A path of the build context without `./` and a trailing slash."""
    return posixpath.normpath(source)


# --- the images -----------------------------------------------------------------------------


def test_the_first_stage_starts_from_the_slim_python_image(
    instructions: list[d.Instruction],
) -> None:
    first_from = next(i for i in instructions if i.keyword == "FROM")

    assert first_from.arguments.split()[0] == BASE_IMAGE


def test_every_stage_starts_from_that_same_python_image(stages: list[d.Stage]) -> None:
    assert len(stages) >= 2, "a build stage and a runtime stage"
    assert [stage.image for stage in stages] == [BASE_IMAGE] * len(stages)


def test_no_image_is_untagged_unresolved_or_latest(
    instructions: list[d.Instruction],
) -> None:
    images = d.images(instructions)

    assert BASE_IMAGE in images
    for image in images:
        assert IMAGE_REFERENCE.match(image), f"{image}: no explicit tag"
        assert image.rsplit(":", 1)[1] != "latest", image


def test_uv_comes_from_one_image_pinned_to_a_full_version(
    instructions: list[d.Instruction],
) -> None:
    uv_images = [i for i in d.images(instructions) if i.startswith(UV_IMAGE_PREFIX)]

    assert len(uv_images) == 1
    assert UV_VERSION_TAG.fullmatch(uv_images[0].removeprefix(UV_IMAGE_PREFIX))


def test_no_image_is_a_gpu_image(instructions: list[d.Instruction]) -> None:
    for image in d.images(instructions):
        assert not any(word in image.lower() for word in GPU_IMAGE_WORDS), image


# --- the dependencies -----------------------------------------------------------------------


def test_every_uv_sync_is_frozen_and_leaves_out_the_dev_group(
    syncs: list[list[str]],
) -> None:
    assert syncs, "the image installs nothing"
    for sync in syncs:
        assert "--frozen" in sync, sync
        assert "--no-dev" in sync, sync


def test_every_uv_sync_skips_exactly_the_torch_and_cuda_packages_of_the_lock(
    syncs: list[list[str]],
) -> None:
    family = d.gpu_family_packages(LOCK.read_text(encoding="utf-8"))

    assert "torch" in family, "the lock holds no torch: the check would be vacuous"
    for sync in syncs:
        assert sorted(d.skipped_packages(sync)) == family


def test_the_last_sync_installs_the_project_into_the_environment_that_is_copied(
    syncs: list[list[str]],
) -> None:
    *before, last = syncs

    for sync in before:
        assert "--no-install-project" in sync
    assert "--no-install-project" not in last
    assert "--no-editable" in last


def test_the_build_stage_never_downloads_a_python_of_its_own(
    stages: list[d.Stage],
) -> None:
    assert environment_of(stages[0]).get("UV_PYTHON_DOWNLOADS") == "never"


def test_the_manifest_the_lock_and_the_vendored_wheels_are_copied_before_the_first_sync(
    instructions: list[d.Instruction],
) -> None:
    before = d.repository_copy_sources(
        instructions[: index_of_first_sync(instructions)]
    )

    assert "pyproject.toml" in before
    assert "uv.lock" in before
    assert any(normal(source).split("/")[0] == "vendor" for source in before), before


def test_every_local_package_the_lock_names_is_a_file_under_a_copied_directory(
    instructions: list[d.Instruction],
) -> None:
    before = [
        normal(source)
        for source in d.repository_copy_sources(
            instructions[: index_of_first_sync(instructions)]
        )
    ]
    local = d.local_sources(LOCK.read_text(encoding="utf-8"))

    assert local, "the lock names no vendored wheel: the check would be vacuous"
    for path in local:
        assert (REPO_ROOT / path).is_file(), path
        assert any(path == src or path.startswith(f"{src}/") for src in before), path


def test_nothing_is_installed_with_pip(instructions: list[d.Instruction]) -> None:
    for command in d.run_commands(instructions):
        assert not (("pip" in command or "pip3" in command) and "install" in command), (
            command
        )


def test_no_download_tool_is_installed_for_the_health_check(
    instructions: list[d.Instruction],
) -> None:
    for command in d.run_commands(instructions):
        assert "curl" not in command and "wget" not in command, command
    for instruction in instructions:
        if instruction.keyword == "HEALTHCHECK":
            assert "curl" not in instruction.arguments
            assert "wget" not in instruction.arguments


# --- the process ----------------------------------------------------------------------------


def test_the_process_runs_as_a_user_that_is_not_root(final_stage: d.Stage) -> None:
    before_entrypoint: list[d.Instruction] = []
    for instruction in final_stage.instructions:
        if instruction.keyword == "ENTRYPOINT":
            break
        before_entrypoint.append(instruction)
    users = [i.arguments for i in before_entrypoint if i.keyword == "USER"]

    assert users, "no USER before the ENTRYPOINT: the process would run as root"
    user = users[-1].split(":")[0].strip()
    assert user not in ("root", ""), users[-1]
    assert not (user.isdigit() and int(user) == 0), users[-1]


def test_the_image_hides_the_gpu(final_stage: d.Stage) -> None:
    assert environment_of(final_stage).get("CUDA_VISIBLE_DEVICES") == ""


def test_the_health_check_asks_the_readiness_route_with_the_standard_library(
    final_stage: d.Stage,
) -> None:
    (healthcheck,) = (i for i in final_stage.instructions if i.keyword == "HEALTHCHECK")

    command = d.healthcheck_command(healthcheck)

    assert command is not None, "the health check is not in exec form"
    assert command[0] == "python"
    assert "/readyz" in " ".join(command)
    assert "urllib" in " ".join(command)


def test_the_entrypoint_and_the_command_are_in_exec_form(final_stage: d.Stage) -> None:
    by_keyword = {i.keyword: i for i in final_stage.instructions}

    assert d.exec_form(by_keyword["ENTRYPOINT"].arguments) == ["rinchen"]
    assert d.exec_form(by_keyword["CMD"].arguments) == COMMAND


def test_the_runtime_stage_takes_the_environment_of_the_build_stage(
    stages: list[d.Stage],
) -> None:
    build, runtime = stages[0], stages[-1]
    environment = environment_of(build)["UV_PROJECT_ENVIRONMENT"]
    from_build = [
        i.arguments
        for i in runtime.instructions
        if i.keyword == "COPY" and f"--from={build.name}" in i.arguments
    ]

    assert any(arguments.split()[-2:] == [environment] * 2 for arguments in from_build)
    assert environment_of(runtime)["PATH"].startswith(f"{environment}/bin:")


# --- the files of the image -----------------------------------------------------------------


def test_every_copy_source_is_a_path_of_the_repository(
    instructions: list[d.Instruction],
) -> None:
    sources = d.repository_copy_sources(instructions)

    assert sources
    for source in sources:
        path = normal(source)
        assert not path.startswith(("/", "..")), source
        assert path not in (".", "*"), f"{source!r} copies the whole context"
        assert list(REPO_ROOT.glob(path)), f"{source}: no such path in the repository"
        assert path.split("/")[0] not in ("data", "models"), source


def test_the_dockerignore_leaves_every_copied_path_in_the_context(
    instructions: list[d.Instruction],
) -> None:
    patterns = d.dockerignore_patterns(DOCKERIGNORE.read_text(encoding="utf-8"))

    for source in d.repository_copy_sources(instructions):
        assert not d.is_excluded(normal(source), patterns), source
    assert not d.is_excluded(
        "vendor/wheels/fasttext_wheel-0.9.2-py3-none-any.whl", patterns
    )


def test_the_dockerignore_keeps_the_repository_s_bulk_out_of_the_context() -> None:
    patterns = d.dockerignore_patterns(DOCKERIGNORE.read_text(encoding="utf-8"))

    for name in EXCLUDED_FROM_THE_CONTEXT:
        assert name in patterns, f"{name} is not excluded"
        assert d.is_excluded(f"{name}/anything", patterns)


# --- what nobody measured -------------------------------------------------------------------


def test_the_dockerfile_says_what_was_never_measured() -> None:
    comments = comments_of(DOCKERFILE)

    assert "unmeasured" in comments
    for note in UNMEASURED_NOTES:
        assert note in comments, f"the comments do not mention: {note}"


def test_the_dockerfile_discloses_what_the_build_fetches_and_for_which_platform() -> (
    None
):
    comments = comments_of(DOCKERFILE)

    for note in (
        "the build needs the network",
        "hatchling",
        "build isolation",
        "kiwipiepy-model",
        "source distribution",
        "linux/amd64",
        "pyonmttok",
        "x86_64",
    ):
        assert note.lower() in comments, f"the comments do not mention: {note}"


def test_the_number_of_skipped_packages_the_comments_state_is_the_lock_s() -> None:
    family = d.gpu_family_packages(LOCK.read_text(encoding="utf-8"))
    stated = {
        int(number)
        for number in re.findall(
            r"(\d+) packages in this lock", DOCKERFILE.read_text(encoding="utf-8")
        )
    }

    assert len(family) == 22
    assert stated == {len(family)}
