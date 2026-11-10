"""A small reader of Dockerfiles, `.dockerignore` files and `uv.lock` for the static checks of
the container recipe (D11, Task 10).

It is not a Dockerfile implementation. It joins continuation lines, drops comments, splits an
instruction into its keyword and arguments, and answers the few questions the checks ask: the
images a file uses, the repository paths a `COPY` names, the commands of a `RUN`, the exec form
of an instruction, which `.dockerignore` pattern excludes a path. What it cannot read (a
heredoc, an exception line of `.dockerignore`) it refuses, so that a check never passes on a
misreading.
"""

from __future__ import annotations

import fnmatch
import json
import re
import shlex
import tomllib
from collections.abc import Sequence
from dataclasses import dataclass

# The packages of a lock that a serving image must not carry (T20): the torch family and the
# CUDA wheels it pulls in.
GPU_FAMILY = re.compile(r"torch(-.+)?|torchcodec|triton|cuda-.+|nvidia-.+")
SHELL_SEPARATORS = frozenset({"&&", "||", ";", "|", "&"})
STAGE_FREE = -1


@dataclass(frozen=True)
class Instruction:
    keyword: str
    arguments: str
    line: int
    stage: int


@dataclass(frozen=True)
class Stage:
    name: str | None
    image: str
    instructions: tuple[Instruction, ...]


def parse(text: str) -> list[Instruction]:
    """The instructions of a Dockerfile: comment and blank lines dropped (inside a continuation
    too), a trailing backslash joining the next line."""
    instructions: list[Instruction] = []
    stage = STAGE_FREE
    pending: list[str] = []
    first_line = 0
    for number, raw in enumerate(text.splitlines(), start=1):
        stripped = raw.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if not pending:
            first_line = number
        if stripped.endswith("\\"):
            pending.append(stripped[:-1].strip())
            continue
        pending.append(stripped)
        joined = " ".join(part for part in pending if part)
        pending = []
        keyword, _, arguments = joined.partition(" ")
        if "<<" in arguments:
            raise ValueError(f"line {first_line}: a heredoc is not read by this reader")
        if keyword.upper() == "FROM":
            stage += 1
        instructions.append(
            Instruction(keyword.upper(), arguments.strip(), first_line, stage)
        )
    return instructions


def stages(instructions: Sequence[Instruction]) -> list[Stage]:
    """The build stages, each with the image it starts from (build arguments expanded)."""
    variables = _variables(instructions)
    found: list[Stage] = []
    for index in sorted({i.stage for i in instructions if i.stage != STAGE_FREE}):
        members = tuple(i for i in instructions if i.stage == index)
        words = _words(members[0].arguments)
        words = [word for word in words if not word.startswith("--")]
        name = words[2] if len(words) >= 3 and words[1].upper() == "AS" else None
        found.append(Stage(name, expand(words[0], variables), members))
    return found


def images(instructions: Sequence[Instruction]) -> list[str]:
    """Every image the file pulls: each `FROM` that is not `scratch` or an earlier stage, and
    each `COPY --from` that names an image rather than a stage."""
    variables = _variables(instructions)
    known_stages: set[str] = set()
    found: list[str] = []
    for stage in stages(instructions):
        if stage.image != "scratch" and stage.image not in known_stages:
            found.append(stage.image)
        if stage.name is not None:
            known_stages.add(stage.name)
        for instruction in stage.instructions:
            if instruction.keyword != "COPY":
                continue
            source = _copy_flags(instruction)[0].get("from")
            if source is None:
                continue
            source = expand(source, variables)
            if source not in known_stages and not source.isdigit():
                found.append(source)
    return found


def repository_copy_sources(instructions: Sequence[Instruction]) -> list[str]:
    """The paths of the build context that `COPY` and `ADD` name (a copy `--from` another
    stage or image names paths of that one)."""
    sources: list[str] = []
    for instruction in instructions:
        if instruction.keyword not in ("COPY", "ADD"):
            continue
        flags, paths = _copy_flags(instruction)
        if "from" not in flags:
            sources.extend(paths[:-1])
    return sources


def exec_form(arguments: str) -> list[str] | None:
    """The JSON array an instruction was written as, or None for the shell form."""
    if not arguments.startswith("["):
        return None
    try:
        parsed = json.loads(arguments)
    except json.JSONDecodeError:
        return None
    if isinstance(parsed, list) and all(isinstance(item, str) for item in parsed):
        return parsed
    return None


def environment(instruction: Instruction) -> dict[str, str]:
    """The variables an `ENV` (or an `ARG` with a default) sets."""
    words = _words(instruction.arguments)
    if instruction.keyword == "ENV" and words and "=" not in words[0]:
        return {words[0]: instruction.arguments.partition(" ")[2].strip()}
    pairs = (word.partition("=") for word in words if "=" in word)
    return {name: value for name, _, value in pairs}


def expand(text: str, variables: dict[str, str]) -> str:
    """`$NAME` and `${NAME}` replaced by what `ARG` and `ENV` gave them; a variable without a
    value stays as written, so an image named by it is visibly unresolved."""

    def value(match: re.Match[str]) -> str:
        name = match.group(1) or match.group(2)
        return variables.get(name, match.group(0))

    return re.sub(r"\$\{(\w+)\}|\$(\w+)", value, text)


def run_commands(instructions: Sequence[Instruction]) -> list[list[str]]:
    """Every command of every `RUN`, split at `&&`, `||`, `;`, `|` and `&`."""
    commands: list[list[str]] = []
    for instruction in instructions:
        if instruction.keyword != "RUN":
            continue
        lexer = shlex.shlex(instruction.arguments, posix=True, punctuation_chars=True)
        lexer.whitespace_split = True
        current: list[str] = []
        for token in lexer:
            if token in SHELL_SEPARATORS:
                commands.append(current)
                current = []
            else:
                current.append(token)
        commands.append(current)
    return [command for command in commands if command]


def uv_sync_commands(instructions: Sequence[Instruction]) -> list[list[str]]:
    """Each `uv sync` of a `RUN`, from the word `uv` on."""
    found: list[list[str]] = []
    for command in run_commands(instructions):
        for index in range(len(command) - 1):
            if command[index] == "uv" and command[index + 1] == "sync":
                found.append(command[index:])
                break
    return found


def skipped_packages(sync: Sequence[str]) -> list[str]:
    """The names given to `--no-install-package`, as written (both `--x NAME` and `--x=NAME`)."""
    names: list[str] = []
    for index, word in enumerate(sync):
        if word == "--no-install-package" and index + 1 < len(sync):
            names.append(sync[index + 1])
        elif word.startswith("--no-install-package="):
            names.append(word.partition("=")[2])
    return names


def healthcheck_command(instruction: Instruction) -> list[str] | None:
    """The exec-form command of a `HEALTHCHECK` (what follows its options and `CMD`), or None
    for the shell form and for `NONE`."""
    rest = instruction.arguments
    while rest.startswith("--"):
        _, _, rest = rest.partition(" ")
        rest = rest.lstrip()
    keyword, _, command = rest.partition(" ")
    if keyword.upper() != "CMD":
        return None
    return exec_form(command.strip())


def gpu_family_packages(lock_text: str) -> list[str]:
    """The sorted names of the packages of a `uv.lock` that belong to the torch and CUDA
    family (the same names, built from the lock, the design spike measured: 22 of them)."""
    names = {package["name"] for package in tomllib.loads(lock_text)["package"]}
    return sorted(name for name in names if GPU_FAMILY.fullmatch(name))


def local_sources(lock_text: str) -> list[str]:
    """The paths of the packages a `uv.lock` takes from a local file (the vendored wheels)."""
    return sorted(
        package["source"]["path"]
        for package in tomllib.loads(lock_text)["package"]
        if "path" in package.get("source", {})
    )


def dockerignore_patterns(text: str) -> list[str]:
    """The patterns of a `.dockerignore`, comments dropped, a leading and a trailing slash
    removed. An exception line (`!name`) is refused: the matcher below does not know it."""
    patterns: list[str] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("!"):
            raise ValueError(f"an exception line is not read by this matcher: {line}")
        patterns.append(line.strip("/"))
    return patterns


def is_excluded(path: str, patterns: Sequence[str]) -> bool:
    """Whether a path of the build context is dropped by the patterns: a pattern that matches
    the path or one of its parent directories excludes it; `**/` lets it match at any depth."""
    parts = path.strip("/").split("/")
    for pattern in patterns:
        deep = pattern.startswith("**/")
        glob = pattern.removeprefix("**/")
        for end in range(1, len(parts) + 1):
            if fnmatch.fnmatchcase("/".join(parts[:end]), glob):
                return True
            if deep and fnmatch.fnmatchcase(parts[end - 1], glob):
                return True
    return False


def _words(arguments: str) -> list[str]:
    return shlex.split(arguments, posix=True)


def _variables(instructions: Sequence[Instruction]) -> dict[str, str]:
    variables: dict[str, str] = {}
    for instruction in instructions:
        if instruction.keyword in ("ARG", "ENV"):
            variables.update(environment(instruction))
    return variables


def _copy_flags(instruction: Instruction) -> tuple[dict[str, str], list[str]]:
    """The `--flag=value` options and the paths of a `COPY` or `ADD` (shell or exec form)."""
    listed = exec_form(instruction.arguments)
    words = listed if listed is not None else _words(instruction.arguments)
    flags: dict[str, str] = {}
    index = 0
    while index < len(words) and words[index].startswith("--"):
        name, _, value = words[index][2:].partition("=")
        flags[name] = value
        index += 1
    return flags, words[index:]
