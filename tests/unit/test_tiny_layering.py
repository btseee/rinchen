"""The layering of `rinchen.tiny` and `rinchen.config.tiny` (D15, spec P1, P14 and P19), checked by
reading the source.

The library of the tiny pipeline sequences the stages of D1 to D14 and holds no model code: its CLI
stages are subprocesses, so it imports no module of the command layer, of the training, the synthetic
data, the export, the inference or the annotation code and no library of a model; it opens no socket
except the loopback listener of `rinchen.tiny.fakes`; it asks no device a question and names no device
value; and it has no branch on the name of a profile (scaling up is a change of the profile file, not
of the code). The scans run over every module of the package by a glob, so a module a later task adds
is covered without an edit here, and each scan is itself tested on strings that violate it.

The words of a device value are written in pieces: the scans of the gate read the tests as well.
"""

from __future__ import annotations

import ast
import re
from collections.abc import Iterable
from pathlib import Path
from typing import Final

import pytest

from rinchen.config.loader import REPO_ROOT
from tests.support.module_imports import loaded_by_importing
from tests.support.no_network import no_network  # noqa: F401 - the fixture of the mark

pytestmark = pytest.mark.usefixtures("no_network")

SOURCE: Final = REPO_ROOT / "src" / "rinchen"
PACKAGE: Final = SOURCE / "tiny"
CONFIG_MODULE: Final = SOURCE / "config" / "tiny.py"
COMMAND_MODULES: Final = sorted((SOURCE / "cli").glob("tiny*.py"))
MODULES: Final = [*sorted(PACKAGE.glob("*.py")), CONFIG_MODULE]
LINE_LIMIT: Final = 450

# The layers the library never imports: the commands are what runs it, and the stages that have a
# command run as that command, in a process of their own.
FORBIDDEN_LAYERS: Final = (
    "rinchen.cli",
    "rinchen.train",
    "rinchen.synth",
    "rinchen.export",
    "rinchen.infer",
    "rinchen.annotate",
)
# The libraries of a model: a stage that needs one runs a command that imports it.
FORBIDDEN_LIBRARIES: Final = ("torch", "ctranslate2", "eole", "sentencepiece")
NETWORK_NAMES: Final = (
    "socket",
    "socketserver",
    "http",
    "urllib",
    "requests",
    "httpx",
    "ssl",
)
# What the one module that serves on the loopback may import of the network.
SERVER_MODULE: Final = PACKAGE / "fakes.py"
SERVER_ALLOWED: Final = ("socket", "socketserver", "http.server")
PROFILE_NAMES: Final = ("tiny", "full")
DEVICE_WORDS: Final = ("cu" + "da", "au" + "to", "nvidia" + "-smi")
DEVICE_NAMES: Final = ("get_cu" + "da_device_count", "check_torch_cu" + "da")
_DEVICE_PATTERN: Final = re.compile(
    r"(?<![A-Za-z0-9_])(?:"
    + "|".join(re.escape(w) for w in DEVICE_WORDS)
    + r")(?![A-Za-z0-9_])",
    re.IGNORECASE,
)
# The modules that hold no stage: importing one loads nothing of the layers above or of a model.
LIGHT_MODULES: Final = (
    "rinchen.config.tiny",
    "rinchen.tiny.fixtures",
    "rinchen.tiny.states",
    "rinchen.tiny.context",
    "rinchen.tiny.thresholds",
)
WATCHED_WHEN_IMPORTED: Final = (
    "rinchen.cli",
    "rinchen.train",
    "rinchen.synth",
    "rinchen.export",
    "rinchen.infer",
    "rinchen.serve",
    "rinchen.annotate",
    "rinchen.eval",
    "rinchen.filter",
    "ctranslate2",
    "eole",
    "torch",
    "sentencepiece",
    "polars",
    "huggingface_hub",
    "httpx",
    "requests",
)


def imported_names(source: str) -> set[str]:
    """Every module a source imports, and for `from m import n` also `m.n`, wherever the import is
    (a function body included). A relative import is listed with its dots."""
    imported: set[str] = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            module = "." * node.level + (node.module or "")
            imported.add(module)
            imported.update(f"{module}.{alias.name}" for alias in node.names)
    return imported


def is_inside(name: str, package: str) -> bool:
    return name == package or name.startswith(f"{package}.")


def layer_violations(source: str) -> list[str]:
    """The imports of a source that the layering does not allow: a layer above the library, the
    library of a model, or a relative import (which hides what it imports)."""
    return sorted(
        name
        for name in imported_names(source)
        if name.startswith(".")
        or any(is_inside(name, layer) for layer in FORBIDDEN_LAYERS)
        or any(is_inside(name, library) for library in FORBIDDEN_LIBRARIES)
    )


def is_network(name: str, allowed: Iterable[str] = ()) -> bool:
    if any(is_inside(name, item) for item in allowed):
        return False
    return name.split(".")[0] in NETWORK_NAMES


def network_references(source: str, allowed: Iterable[str] = ()) -> list[str]:
    """The names of a network library that a source imports, uses as a name, or hands to an import
    as a string; `allowed` lists the modules (and their submodules) it may use."""
    permitted = tuple(allowed)
    found = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import | ast.ImportFrom):
            found.extend(
                name
                for name in imported_names(ast.unparse(node))
                if is_network(name, permitted)
            )
        elif isinstance(node, ast.Name) and is_network(node.id, permitted):
            found.append(node.id)
        elif (
            isinstance(node, ast.Constant)
            and isinstance(node.value, str)
            and is_network(node.value, permitted)
        ):
            found.append(node.value)
    return sorted(found)


def docstring_ids(tree: ast.AST) -> set[int]:
    """The ids of the string nodes that are docstrings (text about code, not code)."""
    found: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(
            node, ast.Module | ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef
        ):
            first = node.body[0] if node.body else None
            if (
                isinstance(first, ast.Expr)
                and isinstance(first.value, ast.Constant)
                and isinstance(first.value.value, str)
            ):
                found.add(id(first.value))
    return found


def device_references(source: str) -> list[str]:
    """What a source says of a device: a string that holds one of the device words (a docstring is
    text about code and is not read), or a name or attribute of a device question."""
    tree = ast.parse(source)
    skipped = docstring_ids(tree)
    found = []
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Constant)
            and isinstance(node.value, str)
            and id(node) not in skipped
        ):
            found.extend(match.lower() for match in _DEVICE_PATTERN.findall(node.value))
        elif isinstance(node, ast.Name) and node.id in DEVICE_NAMES:
            found.append(node.id)
        elif isinstance(node, ast.Attribute) and (
            node.attr in DEVICE_NAMES or node.attr == DEVICE_WORDS[0]
        ):
            found.append(node.attr)
    return sorted(found)


def _is_profile_name(node: ast.AST) -> bool:
    return isinstance(node, ast.Constant) and node.value in PROFILE_NAMES


def _holds_profile_name(node: ast.AST) -> bool:
    if isinstance(node, ast.Tuple | ast.List | ast.Set):
        return any(_is_profile_name(item) for item in node.elts)
    return _is_profile_name(node)


def _pattern_names_a_profile(pattern: ast.AST) -> bool:
    if isinstance(pattern, ast.MatchValue):
        return _is_profile_name(pattern.value)
    if isinstance(pattern, ast.MatchOr):
        return any(_pattern_names_a_profile(item) for item in pattern.patterns)
    if isinstance(pattern, ast.MatchAs) and pattern.pattern is not None:
        return _pattern_names_a_profile(pattern.pattern)
    return False


def profile_branches(source: str) -> list[int]:
    """The lines of a source that compare something with the name of a profile (`==`, `!=`, `in`,
    `is`, a `match` case) or key a mapping by it: the code has no branch on the profile."""
    lines = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Compare):
            operands = [node.left, *node.comparators]
            if any(_holds_profile_name(operand) for operand in operands):
                lines.append(node.lineno)
        elif isinstance(node, ast.match_case):
            if _pattern_names_a_profile(node.pattern):
                lines.append(node.pattern.lineno)
        elif isinstance(node, ast.Dict) and any(
            key is not None and _is_profile_name(key) for key in node.keys
        ):
            lines.append(node.lineno)
    return sorted(lines)


# --- the scans, on strings ---------------------------------------------------------------------


@pytest.mark.parametrize(
    "source",
    [
        "import rinchen.cli",
        "from rinchen import cli",
        "from rinchen.cli.maintain import app",
        "import rinchen.train.launch",
        "from rinchen.train import eole_run",
        "from rinchen import train",
        "from rinchen.synth.run import run",
        "from rinchen.export.ct2 import convert",
        "from rinchen.infer.translator import Translator",
        "from rinchen.annotate import app",
        "import torch",
        "import torch.nn as nn",
        "from torch import nn",
        "import ctranslate2",
        "from ctranslate2 import Translator",
        "import eole",
        "from eole.bin import main",
        "import sentencepiece as spm",
        "from sentencepiece import SentencePieceProcessor",
        "def f():\n    import sentencepiece",
        "if True:\n    from rinchen.train import launch",
        "from . import states",
        "from .states import StageRecord",
        "from .. import config",
        "from rinchen.errors import DataError\nimport torch",
    ],
)
def test_the_import_scan_refuses_what_the_layering_forbids(source: str) -> None:
    assert layer_violations(source) != []


@pytest.mark.parametrize(
    "source",
    [
        "from __future__ import annotations",
        "import hashlib\nimport json\nimport re",
        "from collections.abc import Mapping",
        "import yaml",
        "from pydantic import BaseModel",
        "import polars as pl",
        "import numpy as np",
        "from rinchen.errors import ConfigError, DataError",
        "from rinchen.config.tiny import TinyConfig",
        "from rinchen.config import loader",
        "from rinchen.tiny.states import StageRecord",
        "from rinchen.maintain.commands import CommandRunner",
        "from rinchen.maintain.lock import read_lock",
        "from rinchen.acquire.registry import fetch_source",
        "from rinchen.filter.pipeline import run_pipeline",
        "from rinchen.mine.embed import Embedder",
        "from rinchen.eval.suites import Suite",
        "from rinchen.serve.app import create_app",
        "from rinchen.manifest.stage import begin_commit",
        "from rinchen.trainer import x",
        "from rinchen.exporter import x",
        "# import torch\nx = 'import torch'",
    ],
)
def test_the_import_scan_allows_the_libraries_the_spec_names(source: str) -> None:
    assert layer_violations(source) == []


@pytest.mark.parametrize(
    "source",
    [
        "import socket",
        "import socket as s",
        "from socket import create_connection",
        "import socketserver",
        "import urllib.request",
        "from urllib import request",
        "import http.client",
        "import http.server",
        "from http import client",
        "from http.server import ThreadingHTTPServer",
        "import requests",
        "import httpx as h",
        "import ssl",
        "def f():\n    import socket",
        "x = socket.socket()",
        "import importlib\nimportlib.import_module('socket')",
        "m = __import__('urllib.request')",
    ],
)
def test_the_network_scan_sees_every_way_to_reach_a_network_library(
    source: str,
) -> None:
    assert network_references(source) != []


@pytest.mark.parametrize(
    "source",
    [
        "import json",
        '"""Never opens a socket, speaks http or reaches requests."""',
        "# socket urllib http requests httpx\nx = 1",
        "name = 'sockets are not used'",
        "import threading\nimport hashlib",
        "url = 'http://127.0.0.1:8000'",
    ],
)
def test_the_network_scan_ignores_words_in_comments_and_longer_names(
    source: str,
) -> None:
    assert network_references(source) == []


@pytest.mark.parametrize(
    "source",
    [
        "import socket",
        "from socket import AF_INET",
        "import socketserver",
        "import http.server",
        "from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer",
        "x = socket.socket()",
    ],
)
def test_the_server_module_may_use_the_loopback_listener_and_nothing_more(
    source: str,
) -> None:
    assert network_references(source, SERVER_ALLOWED) == []


@pytest.mark.parametrize(
    "source",
    [
        "import http",
        "import http.client",
        "from http import client",
        "import urllib.request",
        "import requests",
        "import httpx",
        "import ssl",
        "import socket\nimport requests",
    ],
)
def test_the_server_module_still_may_not_be_a_client(source: str) -> None:
    assert network_references(source, SERVER_ALLOWED) != []


CUDA, AUTO, SMI = DEVICE_WORDS
DEVICE_VIOLATIONS: Final = [
    f"x = '{CUDA}'",
    f"x = '{CUDA.upper()}'",
    f"x = '{CUDA}:0'",
    f"x = '--device={CUDA}'",
    f"device = '{AUTO}'",
    f"x = '{AUTO.title()}'",
    f"f(['--device', '{AUTO}'])",
    f"x = '{SMI}'",
    f"subprocess.run(['{SMI}', '-L'])",
    f"x = f'{{y}}/{CUDA}'",
    f"x = 'use {AUTO} here'",
    f"x = get_{CUDA}_device_count()",
    f"x = lib.get_{CUDA}_device_count",
    f"x = doctor.check_torch_{CUDA}",
    f"x = torch.{CUDA}.is_available()",
    f"x = obj.{CUDA}",
]
DEVICE_ALLOWED: Final = [
    f'def f():\n    """The device is {CUDA} or {AUTO}, says the docstring."""\n    return 1',
    f'"""Module text about {AUTO} and {SMI}."""\nx = 1',
    f"class A:\n    '''{AUTO}'''\n    x = 1",
    f"# {AUTO} {CUDA} {SMI}\nx = 1",
    f"x = '{AUTO}matic'",
    f"x = '{AUTO}r'",
    f"x = '{CUDA}graph'",
    f"x = 'xx{CUDA}'",
    f"x = 'pre{AUTO}'",
    f"x = 'a{SMI}'",
    f"x = 'b{AUTO}.'",
    f"x = 'my_{AUTO}_value'",
    f"x = '{AUTO}encoder'",
    "x = 'cpu'",
    "x = '--device'",
    f"x = '{CUDA.upper()}_VISIBLE_DEVICES'",
    f"x = {CUDA}_visible",
    f"x = {AUTO}save",
]


@pytest.mark.parametrize("source", DEVICE_VIOLATIONS)
def test_the_device_scan_sees_a_device_value_or_question(source: str) -> None:
    assert device_references(source) != []


@pytest.mark.parametrize("source", DEVICE_ALLOWED)
def test_the_device_scan_ignores_docstrings_comments_and_longer_words(
    source: str,
) -> None:
    assert device_references(source) == []


@pytest.mark.parametrize(
    "source",
    [
        "if profile == 'tiny':\n    pass",
        "if profile != 'full':\n    pass",
        "if 'tiny' == profile:\n    pass",
        "if profile in ('tiny', 'full'):\n    pass",
        "if profile in ['tiny']:\n    pass",
        "if profile in {'full', 'x'}:\n    pass",
        "if profile not in ('tiny',):\n    pass",
        "x = 1 if self.name == 'full' else 2",
        "x = [a for a in names if a == 'tiny']",
        "assert profile == 'tiny'",
        "match profile:\n    case 'tiny':\n        pass",
        "match profile:\n    case 'x' | 'full':\n        pass",
        "match profile:\n    case 'full' as name:\n        pass",
        "x = {'tiny': 1, 'full': 2}[profile]",
        "x = {'a': 1, 'full': 2}",
        "if profile is 'tiny':\n    pass",
        "if a < b == 'full':\n    pass",
    ],
)
def test_the_profile_scan_sees_every_branch_on_the_name_of_a_profile(
    source: str,
) -> None:
    assert profile_branches(source) != []


@pytest.mark.parametrize(
    "source",
    [
        "x = 'tiny'",
        "name = 'full'",
        "if profile == 'tiny-int8':\n    pass",
        "if profile == 'Tiny':\n    pass",
        "if profile == '':\n    pass",
        "if mode == 'fast':\n    pass",
        "if profile in ('tinyx', 'fully'):\n    pass",
        "match profile:\n    case 'other':\n        pass",
        "x = {'a': 1, 'b': 2}",
        "x = {**other}",
        "x = tiny == full",
        "f('tiny', 'full')",
        "# if profile == 'tiny':\nx = 1",
        "'''if profile == 'tiny': '''",
    ],
)
def test_the_profile_scan_ignores_a_name_that_is_not_compared(source: str) -> None:
    assert profile_branches(source) == []


# --- the modules ----------------------------------------------------------------------------------


def test_the_modules_that_hold_no_stage_are_the_five_of_this_task() -> None:
    assert LIGHT_MODULES == (
        "rinchen.config.tiny",
        "rinchen.tiny.fixtures",
        "rinchen.tiny.states",
        "rinchen.tiny.context",
        "rinchen.tiny.thresholds",
    )
    assert {"polars", "ctranslate2", "rinchen.cli"} <= set(WATCHED_WHEN_IMPORTED)


def test_the_line_limit_and_the_profile_names_are_those_of_the_spec() -> None:
    assert LINE_LIMIT == 450
    assert PROFILE_NAMES == ("tiny", "full")


def test_the_command_modules_are_scanned_for_a_branch_on_the_profile() -> None:
    assert "tiny.py" in {path.name for path in COMMAND_MODULES}
    for path in COMMAND_MODULES:
        assert profile_branches(path.read_text(encoding="utf-8")) == [], path.name


def test_the_modules_of_this_task_are_in_the_scan() -> None:
    names = {path.name for path in MODULES}
    assert {
        "__init__.py",
        "fixtures.py",
        "states.py",
        "context.py",
        "thresholds.py",
    } <= names
    assert "tiny.py" in names
    assert CONFIG_MODULE.is_file()
    assert CONFIG_MODULE in MODULES


@pytest.mark.parametrize(
    "path", MODULES, ids=lambda path: f"{path.parent.name}/{path.name}"
)
def test_a_module_imports_nothing_the_layering_forbids(path: Path) -> None:
    assert layer_violations(path.read_text(encoding="utf-8")) == []


@pytest.mark.parametrize(
    "path", MODULES, ids=lambda path: f"{path.parent.name}/{path.name}"
)
def test_a_module_reaches_no_network_library_but_the_server_the_loopback(
    path: Path,
) -> None:
    allowed = SERVER_ALLOWED if path == SERVER_MODULE else ()
    assert network_references(path.read_text(encoding="utf-8"), allowed) == []


@pytest.mark.parametrize(
    "path", MODULES, ids=lambda path: f"{path.parent.name}/{path.name}"
)
def test_a_module_asks_no_device_a_question_and_names_no_device_value(
    path: Path,
) -> None:
    assert device_references(path.read_text(encoding="utf-8")) == []


@pytest.mark.parametrize(
    "path", MODULES, ids=lambda path: f"{path.parent.name}/{path.name}"
)
def test_a_module_has_no_branch_on_the_name_of_a_profile(path: Path) -> None:
    assert profile_branches(path.read_text(encoding="utf-8")) == []


@pytest.mark.parametrize(
    "path", MODULES, ids=lambda path: f"{path.parent.name}/{path.name}"
)
def test_a_module_is_ascii_and_under_the_line_limit(path: Path) -> None:
    data = path.read_bytes()
    assert data.isascii(), path.name
    assert len(data.splitlines()) < LINE_LIMIT, path.name


def imports_the_library(source: str) -> bool:
    """Whether a source imports `rinchen.tiny` or `rinchen.config.tiny` (or a module of them)."""
    return any(
        is_inside(name, "rinchen.tiny") or is_inside(name, "rinchen.config.tiny")
        for name in imported_names(source)
    )


@pytest.mark.parametrize(
    "source",
    [
        "from rinchen.tiny import states",
        "import rinchen.tiny.states",
        "import rinchen.tiny",
        "from rinchen import tiny",
        "from rinchen.config.tiny import TinyConfig",
        "from rinchen.config import tiny",
        "import rinchen.config.tiny as cfg",
        "def f():\n    from rinchen.tiny.fixtures import read_fixtures",
    ],
)
def test_the_importer_scan_sees_every_way_to_import_the_library(source: str) -> None:
    assert imports_the_library(source)


@pytest.mark.parametrize(
    "source",
    [
        "from rinchen.cli import tiny",
        "from rinchen.config import loader",
        "from rinchen.config.tinyx import y",
        "from rinchen.tinyx import y",
        "import tiny",
        "import rinchen.config",
        "x = 'rinchen.tiny'",
    ],
)
def test_the_importer_scan_ignores_what_only_looks_like_the_library(
    source: str,
) -> None:
    assert not imports_the_library(source)


def test_nothing_but_the_commands_and_the_library_itself_imports_the_library() -> None:
    importers = []
    for path in sorted(SOURCE.rglob("*.py")):
        own = (
            PACKAGE in path.parents
            or path == CONFIG_MODULE
            or path.parent == SOURCE / "cli"
            and path.name.startswith("tiny")
        )
        if not own and imports_the_library(path.read_text(encoding="utf-8")):
            importers.append(path.relative_to(SOURCE).as_posix())
    assert importers == []


def test_importing_a_module_without_a_stage_loads_no_layer_above_it_and_no_library_of_a_model() -> (
    None
):
    assert loaded_by_importing(list(LIGHT_MODULES), WATCHED_WHEN_IMPORTED) == []
