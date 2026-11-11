"""The layering of `rinchen.maintain` and `rinchen.config.maintain` (D14, spec M1 and M18), checked
by reading the source.

The library that answers "is this the stack the lock names?" imports nothing but the standard
library, pydantic, PyYAML, `rinchen.errors` and itself: no model, no engine, no command layer.
It never opens a socket, speaks HTTP or reaches a client library (the one step of a night that
uses the network is `uv`'s, behind the runner seam), and importing it loads none of the heavy
libraries or the layers above it. The scan runs over every module of the package, so a module a
later task adds is covered without an edit here; it is itself tested on strings that violate it.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

import pytest

from tests.support.module_imports import loaded_by_importing
from tests.support.no_network import no_network  # noqa: F401 - the fixture of the mark

pytestmark = pytest.mark.usefixtures("no_network")

SOURCE = Path(__file__).resolve().parents[2] / "src" / "rinchen"
PACKAGE = SOURCE / "maintain"
CONFIG_MODULE = SOURCE / "config" / "maintain.py"
MODULES = [*sorted(PACKAGE.glob("*.py")), CONFIG_MODULE]
THIRD_PARTY = ("pydantic", "yaml")
# What a module of the library may import of the project, and the packages that hold those.
PROJECT_ALLOWED = ("rinchen.errors", "rinchen.maintain", "rinchen.config.maintain")
PROJECT_ANCESTORS = ("rinchen", "rinchen.config")
NETWORK_NAMES = ("socket", "urllib", "http", "requests", "httpx")
LINE_LIMIT = 450
WATCHED_WHEN_IMPORTED = (
    "rinchen.cli",
    "rinchen.train",
    "rinchen.eval",
    "rinchen.infer",
    "rinchen.serve",
    "rinchen.synth",
    "rinchen.annotate",
    "rinchen.filter",
    "rinchen.export",
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
    """Every module a source imports, and for `from m import n` also `m.n`, wherever the import
    is (a function body included). A relative import is listed with its dots."""
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


def violations(source: str) -> list[str]:
    """The imports of a source that the layering does not allow."""
    refused = []
    for name in sorted(imported_names(source)):
        top = name.split(".")[0]
        if name.startswith("."):
            refused.append(name)
        elif top == "rinchen":
            allowed = name in PROJECT_ANCESTORS or any(
                is_inside(name, package) for package in PROJECT_ALLOWED
            )
            if not allowed:
                refused.append(name)
        elif top not in sys.stdlib_module_names and top not in THIRD_PARTY:
            refused.append(name)
    return refused


def network_references(source: str) -> list[str]:
    """The names of a network library that a source imports, names or hands to an import."""
    found = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import | ast.ImportFrom):
            found.extend(
                name for name in imported_names(ast.unparse(node)) if is_network(name)
            )
        elif isinstance(node, ast.Name) and node.id in NETWORK_NAMES:
            found.append(node.id)
        elif (
            isinstance(node, ast.Constant)
            and isinstance(node.value, str)
            and is_network(node.value)
        ):
            found.append(node.value)
    return sorted(found)


def is_network(name: str) -> bool:
    return name.split(".")[0] in NETWORK_NAMES


# --- the scans, on strings -------------------------------------------------------------------------


@pytest.mark.parametrize(
    "source",
    [
        "import torch",
        "import numpy as np",
        "import requests",
        "from rinchen.eval import bootstrap",
        "from rinchen.config.loader import REPO_ROOT",
        "from rinchen.config import loader",
        "import rinchen.cli.doctor",
        "from rinchen import cli",
        "from rinchen.manifest import stage",
        "from . import lock",
        "from .lock import LockFile",
        "def f():\n    import polars",
        "if True:\n    from rinchen.train import launch",
        "from rinchen.errors import ConfigError\nimport sacrebleu",
    ],
)
def test_the_import_scan_refuses_what_the_layering_forbids(source: str) -> None:
    assert violations(source) != []


@pytest.mark.parametrize(
    "source",
    [
        "from __future__ import annotations",
        "import re\nimport tomllib\nimport hashlib",
        "import importlib.metadata",
        "from collections.abc import Mapping",
        "import yaml",
        "from pydantic import BaseModel, Field",
        "from rinchen.errors import ConfigError, DataError",
        "from rinchen import errors",
        "import rinchen.errors",
        "from rinchen.maintain.lock import LockFile",
        "from rinchen.maintain import lock, states",
        "from rinchen.config.maintain import MaintainConfig",
        "from rinchen.config import maintain",
        "def f():\n    from rinchen.maintain.states import STATES",
    ],
)
def test_the_import_scan_allows_the_standard_library_the_two_packages_and_the_errors(
    source: str,
) -> None:
    assert violations(source) == []


@pytest.mark.parametrize(
    "source",
    [
        "import socket",
        "import socket as s",
        "from socket import create_connection",
        "import urllib.request",
        "from urllib import request",
        "import http.client",
        "from http import client",
        "import requests",
        "import httpx as h",
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
    ],
)
def test_the_network_scan_ignores_words_in_comments_and_longer_names(
    source: str,
) -> None:
    assert network_references(source) == []


# --- the modules ---------------------------------------------------------------------------------------


def test_the_modules_of_this_task_are_in_the_scan() -> None:
    names = {path.name for path in MODULES}
    assert {"__init__.py", "states.py", "lock.py", "environment.py"} <= names
    assert "maintain.py" in names
    assert CONFIG_MODULE.is_file()


@pytest.mark.parametrize("path", MODULES, ids=lambda path: path.name)
def test_a_module_imports_nothing_the_layering_forbids(path: Path) -> None:
    assert violations(path.read_text(encoding="utf-8")) == []


@pytest.mark.parametrize("path", MODULES, ids=lambda path: path.name)
def test_a_module_refers_to_no_network_library(path: Path) -> None:
    assert network_references(path.read_text(encoding="utf-8")) == []


@pytest.mark.parametrize("path", MODULES, ids=lambda path: path.name)
def test_a_module_is_ascii_and_under_the_line_limit(path: Path) -> None:
    data = path.read_bytes()
    assert data.isascii(), path.name
    assert len(data.splitlines()) < LINE_LIMIT, path.name


def test_importing_the_library_loads_none_of_the_layers_above_it_or_a_heavy_library() -> (
    None
):
    modules = [
        "rinchen.config.maintain",
        *(f"rinchen.maintain.{path.stem}" for path in PACKAGE.glob("*.py")),
    ]
    modules = [name for name in modules if not name.endswith(".__init__")]
    assert loaded_by_importing(modules, WATCHED_WHEN_IMPORTED) == []
