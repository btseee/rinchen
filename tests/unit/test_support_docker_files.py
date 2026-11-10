"""The reader of Dockerfiles that the static checks of the container recipe stand on (D11,
Task 10): what it makes of continuation lines, comments, stages, copies and `uv sync` commands
is what the checks of `docker/Dockerfile.cpu` see, so it is tested on a Dockerfile of its own."""

from __future__ import annotations

import pytest

from tests.support import docker_files as d

SAMPLE = """\
# a comment, then a blank line

ARG PYTHON_TAG=3.14-slim
FROM python:${PYTHON_TAG} AS build
COPY --from=ghcr.io/astral-sh/uv:0.13.7 /uv /bin/uv
ENV A=1 \\
    # a comment inside a continuation
    CUDA_VISIBLE_DEVICES=""
COPY pyproject.toml uv.lock ./
COPY ["vendor", "./vendor"]
RUN uv sync --frozen --no-dev \\
    --no-install-package torch \\
    --no-install-package=triton && echo done
FROM python:3.14-slim AS runtime
COPY --from=build --chown=1:1 /opt/venv /opt/venv
COPY configs /app/configs
USER 10001:10001
HEALTHCHECK --interval=10s --timeout=3s \\
    CMD ["python", "-c", "print(1)"]
ENTRYPOINT ["rinchen"]
CMD ["serve", "--port", "8000"]
"""


@pytest.fixture(scope="module")
def instructions() -> list[d.Instruction]:
    return d.parse(SAMPLE)


def keywords(instructions: list[d.Instruction]) -> list[str]:
    return [instruction.keyword for instruction in instructions]


def test_comments_blank_lines_and_continuations_are_read_as_docker_reads_them(
    instructions: list[d.Instruction],
) -> None:
    assert keywords(instructions) == [
        "ARG",
        "FROM",
        "COPY",
        "ENV",
        "COPY",
        "COPY",
        "RUN",
        "FROM",
        "COPY",
        "COPY",
        "USER",
        "HEALTHCHECK",
        "ENTRYPOINT",
        "CMD",
    ]
    env = instructions[3]
    assert env.arguments == 'A=1 CUDA_VISIBLE_DEVICES=""'
    assert env.line == 6


def test_an_instruction_knows_its_stage() -> None:
    instructions = d.parse(SAMPLE)

    assert [i.stage for i in instructions if i.keyword == "FROM"] == [0, 1]
    assert instructions[0].stage == -1
    assert [i.stage for i in instructions if i.keyword == "USER"] == [1]


def test_the_stages_carry_name_and_image(instructions: list[d.Instruction]) -> None:
    stages = d.stages(instructions)

    assert [(s.name, s.image) for s in stages] == [
        ("build", "python:3.14-slim"),
        ("runtime", "python:3.14-slim"),
    ]
    assert keywords(list(stages[1].instructions)) == [
        "FROM",
        "COPY",
        "COPY",
        "USER",
        "HEALTHCHECK",
        "ENTRYPOINT",
        "CMD",
    ]


def test_every_image_is_found_and_a_stage_is_not_one(
    instructions: list[d.Instruction],
) -> None:
    assert d.images(instructions) == [
        "python:3.14-slim",
        "ghcr.io/astral-sh/uv:0.13.7",
        "python:3.14-slim",
    ]


def test_a_stage_that_starts_from_an_earlier_stage_or_from_scratch_pulls_no_image() -> (
    None
):
    parsed = d.parse(
        "FROM python:3.14-slim AS build\n"
        "FROM build AS again\n"
        "FROM scratch AS empty\n"
        "COPY --from=0 /a /b\n"
        "COPY --from=again /a /b\n"
    )

    assert d.images(parsed) == ["python:3.14-slim"]


def test_the_options_of_a_from_are_not_part_of_the_image() -> None:
    parsed = d.parse("FROM --platform=linux/amd64 python:3.14-slim AS build\n")

    assert d.images(parsed) == ["python:3.14-slim"]
    assert [(s.name, s.image) for s in d.stages(parsed)] == [
        ("build", "python:3.14-slim")
    ]


def test_an_unset_build_argument_stays_visible_in_an_image_name() -> None:
    parsed = d.parse("ARG TAG\nFROM python:${TAG}\n")

    assert d.images(parsed) == ["python:${TAG}"]


def test_only_a_copy_from_the_repository_names_repository_paths(
    instructions: list[d.Instruction],
) -> None:
    assert d.repository_copy_sources(instructions) == [
        "pyproject.toml",
        "uv.lock",
        "vendor",
        "configs",
    ]


def test_the_exec_form_is_a_list_and_the_shell_form_is_not(
    instructions: list[d.Instruction],
) -> None:
    by_keyword = {i.keyword: i for i in instructions}

    assert d.exec_form(by_keyword["ENTRYPOINT"].arguments) == ["rinchen"]
    assert d.exec_form(by_keyword["CMD"].arguments) == ["serve", "--port", "8000"]
    assert d.exec_form("rinchen serve") is None
    assert d.exec_form("[not json") is None
    assert d.exec_form('["a", 1]') is None


def test_the_environment_of_an_env_instruction(
    instructions: list[d.Instruction],
) -> None:
    env = next(i for i in instructions if i.keyword == "ENV")

    assert d.environment(env) == {"A": "1", "CUDA_VISIBLE_DEVICES": ""}


def test_the_legacy_env_form_with_a_space_is_read_too() -> None:
    (env,) = d.parse("ENV KEY some value\n")

    assert d.environment(env) == {"KEY": "some value"}


def test_the_uv_sync_commands_of_run_instructions(
    instructions: list[d.Instruction],
) -> None:
    (sync,) = d.uv_sync_commands(instructions)

    assert sync[:3] == ["uv", "sync", "--frozen"]
    assert "echo" not in sync
    assert d.skipped_packages(sync) == ["torch", "triton"]


def test_a_sync_chained_after_other_commands_is_found() -> None:
    parsed = d.parse("RUN cd /app; uv sync --frozen && uv cache clean\n")

    assert d.uv_sync_commands(parsed) == [["uv", "sync", "--frozen"]]


def test_the_healthcheck_command_is_what_follows_the_options(
    instructions: list[d.Instruction],
) -> None:
    healthcheck = next(i for i in instructions if i.keyword == "HEALTHCHECK")

    assert d.healthcheck_command(healthcheck) == ["python", "-c", "print(1)"]


def test_a_shell_form_healthcheck_has_no_exec_command() -> None:
    (healthcheck,) = d.parse("HEALTHCHECK CMD curl http://localhost/readyz\n")

    assert d.healthcheck_command(healthcheck) is None


def test_the_run_commands_split_on_every_shell_separator() -> None:
    parsed = d.parse("RUN a b && c d ; e | f || g\n")

    assert d.run_commands(parsed) == [["a", "b"], ["c", "d"], ["e"], ["f"], ["g"]]


def test_a_heredoc_is_refused_rather_than_misread() -> None:
    with pytest.raises(ValueError, match="heredoc"):
        d.parse("RUN <<EOF\nuv sync\nEOF\n")


def test_the_gpu_family_of_a_lock_is_read_by_package_name() -> None:
    lock = """\
version = 1

[[package]]
name = "torch"
version = "2.14.1"
dependencies = [
    { name = "nvidia-cublas" },
]

[[package]]
name = "torch-optimi"
version = "0.5"

[[package]]
name = "torchcodec"
version = "1"

[[package]]
name = "triton"
version = "3"

[[package]]
name = "cuda-toolkit"
version = "13"

[[package]]
name = "nvidia-cublas"
version = "13"

[[package]]
name = "torchvision-not-a-family-member"
version = "1"

[[package]]
name = "cudaq"
version = "1"

[[package]]
name = "numpy"
version = "2"
"""

    assert d.gpu_family_packages(lock) == [
        "cuda-toolkit",
        "nvidia-cublas",
        "torch",
        "torch-optimi",
        "torchcodec",
        "triton",
    ]


def test_the_vendored_paths_of_a_lock_are_listed() -> None:
    lock = """\
[[package]]
name = "a"
version = "1"
source = { path = "vendor/wheels/a-1.whl" }

[[package]]
name = "b"
version = "1"
source = { registry = "https://pypi.org/simple" }

[[package]]
name = "rinchen"
version = "0"
source = { editable = "." }
"""

    assert d.local_sources(lock) == ["vendor/wheels/a-1.whl"]


@pytest.mark.parametrize(
    ("path", "excluded"),
    [
        ("vendor", False),
        ("vendor/wheels/a.whl", False),
        ("src", False),
        (".venv", True),
        (".git", True),
        ("data", True),
        ("data/raw/x.parquet", True),
        ("src/rinchen/__pycache__", True),
        ("src/rinchen/mod.pyc", True),
        ("tests", True),
        ("docs/runbooks/00.md", True),
    ],
)
def test_the_dockerignore_matcher(path: str, excluded: bool) -> None:
    patterns = d.dockerignore_patterns(
        "# a comment\n\n.venv\n/.git\ndata/\n**/__pycache__\n**/*.pyc\ntests\ndocs\n"
    )

    assert d.is_excluded(path, patterns) is excluded


def test_a_dockerignore_exception_is_refused_rather_than_ignored() -> None:
    with pytest.raises(ValueError, match="exception"):
        d.dockerignore_patterns("data\n!data/keep\n")
