"""The layering of the Hub upload (D13, Task 10, spec T18 and T22), checked by reading the source.

`rinchen.export.hub` (and `hub_scan`) plan and make an upload through a `HubClient` and import the
standard library, `rinchen.errors` and `rinchen.export.release`, `record` and `verified` only. The
ONE place in the whole package that imports `huggingface_hub` is `HfHubClient.__init__`: importing
the module loads no Hub client, no model library, no dataset library. The command
`rinchen release hub` (`cli/release_hub*.py`) reads the environment in one function, the one that
asks for `HF_TOKEN` on the `--upload` path, never logs in, never looks for a token file and never
asks about the GPU. `rinchen.eval` never imports the Hub plan (`test_export_release_layering.py`).
"""

from __future__ import annotations

import ast
import os
import subprocess
import sys
from pathlib import Path

import pytest

import rinchen
import rinchen.export
from tests.unit.test_eval_layering import (
    FORBIDDEN_LIBRARIES,
    gpu_queries,
    imported_names,
    is_inside,
)

PACKAGE = Path(rinchen.__file__).parent
EXPORT = Path(rinchen.export.__file__).parent
CLI = PACKAGE / "cli"
HUB_MODULES = sorted(EXPORT.glob("hub*.py"))
HUB_COMMANDS = sorted(CLI.glob("release_hub*.py"))
LIBRARY = "huggingface_hub"
# What the plan may use of the project: nothing else of `rinchen`.
ALLOWED_PROJECT_MODULES = (
    "rinchen.errors",
    "rinchen.export.release",
    "rinchen.export.record",
    "rinchen.export.verified",
    "rinchen.export.hub_scan",
    "rinchen.export.hub_files",
    "rinchen.export.hub_record",
    "rinchen.export.hub",
)
# Never imported by the plan or the command: the layers above and the heavy libraries.
NEVER = (
    "rinchen.cli",
    "rinchen.eval.suites",
    "rinchen.eval.suite_rules",
    "rinchen.eval.suite_sources",
    "rinchen.train",
    "rinchen.synth",
    "rinchen.serve",
    "rinchen.annotate",
    "rinchen.script_conv",
    "rinchen.filter",
    "rinchen.infer",
    "datasets",
    "httpx",
    "requests",
    "polars",
    *FORBIDDEN_LIBRARIES,
)
COMMAND_NEVER = tuple(layer for layer in NEVER if layer != "rinchen.cli") + (LIBRARY,)
# What a command or a test of the token handling must not name.
TOKEN_HANDLING = (
    "login",
    "hffolder",
    "get_token",
    "hf_home",
    "whoami",
    "notebook_login",
)


class _Scopes(ast.NodeVisitor):
    """Collect, with the name of the enclosing function or class, what a predicate finds."""

    def __init__(self) -> None:
        self.scope: list[str] = []
        self.found: list[tuple[str, str]] = []

    def _enter(self, node: ast.AST, name: str) -> None:
        self.scope.append(name)
        self.generic_visit(node)
        self.scope.pop()

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self._enter(node, node.name)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._enter(node, node.name)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._enter(node, node.name)

    def here(self) -> str:
        return ".".join(self.scope) or "<module>"


class _LibraryImports(_Scopes):
    def visit_Import(self, node: ast.Import) -> None:
        if any(is_inside(alias.name, LIBRARY) for alias in node.names):
            self.found.append((self.here(), LIBRARY))

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if node.module is not None and is_inside(node.module, LIBRARY):
            self.found.append((self.here(), LIBRARY))


class _EnvironmentReads(_Scopes):
    def visit_Attribute(self, node: ast.Attribute) -> None:
        if node.attr in ("environ", "getenv", "environb"):
            self.found.append((self.here(), node.attr))
        self.generic_visit(node)

    def visit_Name(self, node: ast.Name) -> None:
        if node.id in ("environ", "getenv", "environb"):
            self.found.append((self.here(), node.id))


def library_imports(source: str) -> list[str]:
    """Where a source imports `huggingface_hub`: the `Class.function` that holds the import, or
    `<module>` for an import at the top level."""
    scanner = _LibraryImports()
    scanner.visit(ast.parse(source))
    return sorted(place for place, _ in scanner.found)


def environment_readers(source: str) -> list[str]:
    """The functions of a source that touch `environ` or `getenv`."""
    scanner = _EnvironmentReads()
    scanner.visit(ast.parse(source))
    return sorted({place for place, _ in scanner.found})


def reached(source: str, forbidden: tuple[str, ...]) -> list[str]:
    return sorted(
        module
        for module in imported_names(source)
        if any(is_inside(module, layer) for layer in forbidden)
    )


# ---- the scans find what they are told to ---------------------------------------------------------------------------


def test_the_scan_of_the_library_finds_every_place_that_imports_it() -> None:
    source = (
        "import huggingface_hub\n"
        "from huggingface_hub import HfApi\n"
        "import huggingface_hub.utils as u\n"
        "class A:\n"
        "    def __init__(self):\n"
        "        from huggingface_hub import HfApi\n"
        "    def other(self):\n"
        "        import huggingface_hub\n"
        "def f():\n"
        "    from huggingface_hub.errors import X\n"
        "import huggingface_hubris\n"
    )

    assert library_imports(source) == [
        "<module>",
        "<module>",
        "<module>",
        "A.__init__",
        "A.other",
        "f",
    ]


def test_the_scan_of_the_environment_names_the_function_that_reads_it() -> None:
    source = (
        "import os\n"
        "from os import environ\n"
        "def a():\n"
        "    return os.environ.get('X')\n"
        "def b():\n"
        "    return os.getenv('X')\n"
        "def c():\n"
        "    return 1\n"
        "class D:\n"
        "    def e(self):\n"
        "        return environ['X']\n"
    )

    assert environment_readers(source) == ["D.e", "a", "b"]


# ---- the Hub library ---------------------------------------------------------------------------------------------------------


def test_the_hub_library_is_imported_in_exactly_one_place_of_the_whole_package() -> (
    None
):
    found = {
        path.relative_to(PACKAGE).as_posix(): library_imports(
            path.read_text(encoding="utf-8")
        )
        for path in sorted(PACKAGE.rglob("*.py"))
    }

    assert {name: places for name, places in found.items() if places} == {
        "export/hub.py": ["HfHubClient.__init__"]
    }


# ---- the plan --------------------------------------------------------------------------------------------------------------------


def test_the_scan_covers_the_plan_and_its_scan_of_text() -> None:
    assert {path.name for path in HUB_MODULES} >= {
        "hub.py",
        "hub_files.py",
        "hub_record.py",
        "hub_scan.py",
    }


@pytest.mark.parametrize("path", HUB_MODULES, ids=lambda path: path.name)
def test_a_module_of_the_plan_imports_the_standard_library_and_four_modules_of_the_project(
    path: Path,
) -> None:
    imported = imported_names(path.read_text(encoding="utf-8"))
    outside = sorted(
        module
        for module in imported
        if module.split(".")[0] not in sys.stdlib_module_names
        and module.split(".")[0] != "__future__"
        and not any(is_inside(module, allowed) for allowed in ALLOWED_PROJECT_MODULES)
        and module != LIBRARY
        and not module.startswith(f"{LIBRARY}.")
    )

    assert outside == []


@pytest.mark.parametrize("path", HUB_MODULES, ids=lambda path: path.name)
def test_a_module_of_the_plan_imports_no_layer_above_it_and_no_heavy_library(
    path: Path,
) -> None:
    assert reached(path.read_text(encoding="utf-8"), NEVER) == []


@pytest.mark.parametrize("path", HUB_MODULES, ids=lambda path: path.name)
def test_a_module_of_the_plan_reads_no_environment_variable(path: Path) -> None:
    assert environment_readers(path.read_text(encoding="utf-8")) == []


@pytest.mark.parametrize("path", HUB_MODULES, ids=lambda path: path.name)
def test_a_module_of_the_plan_never_logs_in_and_never_looks_for_a_token_file(
    path: Path,
) -> None:
    names = _names(path)

    assert [
        name
        for name in sorted(names)
        if any(word in name.lower() for word in TOKEN_HANDLING)
    ] == []


def test_importing_the_plan_loads_no_hub_client_no_model_library_and_no_dataset_library(
    tmp_path: Path,
) -> None:
    code = (
        "import sys\n"
        "sys.path.insert(0, sys.argv[1])\n"
        "import rinchen.export.hub\n"
        "assert rinchen.__file__.startswith(sys.argv[1]), rinchen.__file__\n"
        "watched = ('huggingface_hub', 'ctranslate2', 'sentencepiece', 'eole', 'torch',\n"
        "           'polars', 'datasets', 'httpx', 'requests', 'rinchen.cli', 'rinchen.train')\n"
        "print(sorted(m for m in sys.modules\n"
        "             if any(m == w or m.startswith(w + '.') for w in watched)))\n"
    )
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "HF_HOME": str(tmp_path)}

    completed = subprocess.run(
        [sys.executable, "-I", "-c", code, str(PACKAGE.parent)],
        capture_output=True,
        text=True,
        env=env,
        timeout=120,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "[]"


# ---- the command -------------------------------------------------------------------------------------------------------------------


def test_the_scan_covers_the_command_and_its_output_module() -> None:
    assert {path.name for path in HUB_COMMANDS} >= {"release_hub.py"}


@pytest.mark.parametrize("path", HUB_COMMANDS, ids=lambda path: path.name)
def test_the_command_imports_no_layer_it_has_no_need_of_and_not_the_hub_library(
    path: Path,
) -> None:
    assert reached(path.read_text(encoding="utf-8"), COMMAND_NEVER) == []


def test_the_command_reaches_the_hub_only_through_the_plan_module() -> None:
    imported = imported_names((CLI / "release_hub.py").read_text(encoding="utf-8"))

    assert {
        "rinchen.export.hub.plan_upload",
        "rinchen.export.hub.upload",
        "rinchen.export.hub.HfHubClient",
    } <= imported


def test_the_environment_is_read_in_one_function_of_the_command() -> None:
    found = {
        path.name: environment_readers(path.read_text(encoding="utf-8"))
        for path in HUB_COMMANDS
    }

    assert {name: places for name, places in found.items() if places} == {
        "release_hub.py": ["token_from_environment"]
    }


# ---- no GPU ------------------------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "path", [*HUB_MODULES, *HUB_COMMANDS], ids=lambda path: path.name
)
def test_no_module_of_the_hub_upload_asks_about_the_gpu(path: Path) -> None:
    assert gpu_queries(path.read_text(encoding="utf-8")) == []


def _names(path: Path) -> set[str]:
    """Every identifier, attribute, imported name and string of the code of a module that is not
    a docstring (the same reading as the scan of the other D13 commands)."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    docstrings = {
        id(node.body[0].value)
        for node in ast.walk(tree)
        if isinstance(
            node, ast.Module | ast.FunctionDef | ast.ClassDef | ast.AsyncFunctionDef
        )
        and node.body
        and isinstance(node.body[0], ast.Expr)
        and isinstance(node.body[0].value, ast.Constant)
    }
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            found.add(node.id)
        elif isinstance(node, ast.Attribute):
            found.add(node.attr)
        elif isinstance(node, ast.alias):
            found.update(node.name.split("."))
        elif (
            isinstance(node, ast.Constant)
            and isinstance(node.value, str)
            and id(node) not in docstrings
        ):
            found.add(node.value)
    return found
