"""The option tables of the real commands, read from their `--help` (D15, brief G4.2).

`help_table(*words)` runs `rinchen <words> --help` through `CliRunner`, reads the options table of the
output (an option's names, whether it is required, whether it takes a value and whether the value is a
path) and the positional arguments of the usage line, and returns what an argument list of the command
may hold. `click_table(*words)` reads the same facts from the command objects instead of the text, plus
whether an option may be repeated; a test keeps the two equal for every command the pipeline runs, so
the text reader is itself checked.

`check_argument_list(argv, words)` is the check a builder's list must pass: it starts with the
interpreter, `-m rinchen` and the command words, holds the command's positionals first, names only
options of the command, gives each its value (and a flag none), repeats only an option that may be
repeated, holds every required option and gives each path an absolute spelling.
"""

from __future__ import annotations

import os
import re
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Final

import typer.main
from typer.testing import CliRunner

from rinchen.cli.main import app

# The vertical border of the options box.
BORDER: Final = chr(0x2502)
# One option line of the box: the border, a marker column (`*` when required; the column is left out
# when no option of the command is required), then the names.
_OPTION_LINE: Final = re.compile(
    rf"^{BORDER} (?:([* ]) {{2}})?(--[a-z0-9-]+)(?: +(--[a-z0-9-]+))?(?= |{BORDER}|$)(.*)$"
)
_USAGE_LINE: Final = re.compile(r"^ Usage: rinchen (.*)$")
_TYPE_START: Final = re.compile(r"^(<|\d+\.\.\d+)")
HELP: Final = "--help"


@dataclass(frozen=True, slots=True)
class Option:
    required: bool
    takes_value: bool
    path_valued: bool


@dataclass(frozen=True, slots=True)
class CommandTable:
    options: Mapping[str, Option]
    positionals: int


def help_text(words: Sequence[str]) -> str:
    """The `--help` of `rinchen <words>`; the exit code must be 0."""
    result = CliRunner().invoke(app, [*words, HELP], env={"NO_COLOR": "1"})
    assert result.exit_code == 0, (
        f"--help of {' '.join(words)} exited {result.exit_code}"
    )
    return result.output


def help_table(*words: str) -> CommandTable:
    """The options and positionals of `rinchen <words>`, read from its `--help` text."""
    options: dict[str, Option] = {}
    positionals = -1
    for line in help_text(words).splitlines():
        usage = _USAGE_LINE.match(line)
        if usage:
            tail = usage.group(1).split("[OPTIONS]", 1)[1].split()
            positionals = len(tail)
            continue
        found = _OPTION_LINE.match(line)
        if not found:
            continue
        marker, first, second, rest = found.groups()
        rest = rest.strip().strip(BORDER).strip()
        option = Option(
            required=marker == "*",
            takes_value=_TYPE_START.match(rest) is not None,
            path_valued=rest.startswith("<path>"),
        )
        for name in (first, second):
            if name is not None and name != HELP:
                options[name] = option
    assert positionals >= 0, "no usage line in the help"
    return CommandTable(options, positionals)


def _command(words: Sequence[str]) -> Any:
    """The command object of `rinchen <words>`: Typer builds it from its own copy of click, whose
    classes are not the installed package's, so it is read by the attributes every click object has."""
    node: Any = typer.main.get_command(app)
    for word in words:
        node = node.commands[word]
    return node


def click_table(*words: str) -> tuple[CommandTable, frozenset[str]]:
    """The same table from the command objects, and the names of the options that may be repeated."""
    options: dict[str, Option] = {}
    repeatable: set[str] = set()
    positionals = 0
    for param in _command(words).params:
        if param.param_type_name == "argument":
            positionals += 1
            continue
        assert param.param_type_name == "option"
        option = Option(
            required=param.required,
            takes_value=not param.is_flag,
            path_valued=param.type.name == "path",
        )
        for name in [*param.opts, *param.secondary_opts]:
            if name != HELP:
                options[name] = option
                if param.multiple:
                    repeatable.add(name)
    return CommandTable(options, positionals), frozenset(repeatable)


def check_argument_list(argv: Sequence[str], words: Sequence[str]) -> list[str]:
    """What is wrong with `argv` as a call of `rinchen <words>` (empty when nothing is)."""
    table = help_table(*words)
    _, repeatable = click_table(*words)
    head = [sys.executable, "-m", "rinchen", *words]
    if list(argv[: len(head)]) != head:
        return [
            "the list does not start with the interpreter, -m rinchen and the command"
        ]
    tail = list(argv[len(head) :])
    problems: list[str] = []
    positionals = 0
    while positionals < len(tail) and not tail[positionals].startswith("--"):
        positionals += 1
    if positionals != table.positionals:
        problems.append(
            f"{positionals} positionals, the command has {table.positionals}"
        )
    seen: list[str] = []
    index = positionals
    while index < len(tail):
        name = tail[index]
        option = table.options.get(name)
        if option is None:
            problems.append(f"{name} is not an option of the command")
            index += 1
            continue
        if name in seen and name not in repeatable:
            problems.append(f"{name} is given twice")
        seen.append(name)
        index += 1
        if option.takes_value:
            if index >= len(tail) or tail[index].startswith("--"):
                problems.append(f"{name} has no value")
                continue
            if option.path_valued and not os.path.isabs(tail[index]):
                problems.append(f"the value of {name} is not an absolute path")
            index += 1
        elif index < len(tail) and not tail[index].startswith("--"):
            problems.append(f"{name} takes no value")
            index += 1
    problems.extend(
        f"the required option {name} is missing"
        for name, option in table.options.items()
        if option.required and name not in seen
    )
    return problems
