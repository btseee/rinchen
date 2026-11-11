"""The hygiene scans of the D14 gate (R10), made by reading source (an AST scan, so a word in a comment or
a docstring never counts and an alias cannot hide a name): no question to a device, no write to the two
files the command only reads, no argument list that could change the stack, no upgrade flag anywhere in
the code base but the one the shipped configuration holds next to `--dry-run`, no subprocess without a
timeout, and no word of the workflow scan in the workflow.

Each scan takes a source (or a text) and returns what it found, so that a test asserts the list is empty
on the files and non-empty on a string that breaks the rule. The words of a device question are written
here in pieces: the scans cover the files of the gate, this one included.
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Final

import yaml

from rinchen.config.loader import REPO_ROOT
from tests.support.nightly_workflow import (
    FORBIDDEN_IN_RUN,
    FORBIDDEN_WORDS,
    load,
    runs_of,
)

SOURCE: Final = REPO_ROOT / "src" / "rinchen"
TESTS: Final = REPO_ROOT / "tests"
SHIPPED_CONFIG: Final = REPO_ROOT / "configs" / "maintain" / "maintain.yaml"
# The one module that spells the upgrade flag in Python: the loader's closed list of the flags a
# configuration may give uv.
LOADER: Final = SOURCE / "config" / "maintain.py"

# What a device question is made of, in pieces so that this file does not hold the words itself.
_TORCH_CUDA: Final = "torch" + ".cuda"
_DEVICE_COUNT: Final = "get_cuda" + "_device_count"
_SMI: Final = "nvidia" + "-smi"
_GPU_CHECK: Final = "check_torch" + "_cuda"
READ_ONLY_FILES: Final = ("uv.lock", "pyproject.toml")
WRITING_CALLS: Final = frozenset(
    {
        "write_text",
        "write_bytes",
        "unlink",
        "replace",
        "rename",
        "open",
        "replace_file",
        "touch",
        "chmod",
        "rmtree",
        "copyfile",
        "copy",
        "move",
        "symlink_to",
    }
)
CHANGING_ARGUMENTS: Final = frozenset({"sync", "pip", "install", "--upgrade-package"})
UPGRADE_FLAG: Final = "--upgrade"
UPGRADE_PACKAGE_FLAG: Final = "--upgrade-package"
_BLOCKING_CALLS: Final = frozenset({"run", "call", "check_call", "check_output"})
_WAITING_CALLS: Final = frozenset({"communicate", "wait"})


def runtime_files() -> list[Path]:
    """The modules of the command and its library: what runs when `rinchen maintain check` does."""
    return [
        *sorted((SOURCE / "maintain").glob("*.py")),
        *sorted((SOURCE / "cli").glob("maintain*.py")),
        LOADER,
    ]


def helper_files() -> list[Path]:
    """The files of the tests that BUILD or START a command, a workflow run or a shim (the scans of a
    device question and of a subprocess without a timeout cover them and the runtime files)."""
    support = TESTS / "support"
    return [
        support / "fake_uv.py",
        support / "maintain_gate.py",
        support / "maintain_gate_hygiene.py",
        support / "maintain_project.py",
        support / "nightly_workflow.py",
        TESTS / "integration" / "test_maintain_gate.py",
        TESTS / "integration" / "test_maintain_gate_controls.py",
    ]


def python_files_of_the_code_base() -> list[Path]:
    return sorted(SOURCE.rglob("*.py"))


def _docstrings(tree: ast.AST) -> set[int]:
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


def _strings(tree: ast.AST) -> list[str]:
    skipped = _docstrings(tree)
    return [
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant)
        and isinstance(node.value, str)
        and id(node) not in skipped
    ]


# ---- a question to a device -----------------------------------------------------------------------


def _imports_torch(node: ast.AST) -> bool:
    if isinstance(node, ast.Import):
        return any(alias.name.split(".")[0] == "torch" for alias in node.names)
    return (
        isinstance(node, ast.ImportFrom)
        and (node.module or "").split(".")[0] == "torch"
    )


def _names_the_device_count(node: ast.AST) -> bool:
    return (isinstance(node, ast.Name) and node.id == _DEVICE_COUNT) or (
        isinstance(node, ast.Attribute) and node.attr == _DEVICE_COUNT
    )


def _names_the_gpu_check(node: ast.AST) -> bool:
    """A reference to the doctor's GPU check by any route: a name, an attribute, an import."""
    if isinstance(node, ast.Name):
        return node.id == _GPU_CHECK
    if isinstance(node, ast.Attribute):
        return node.attr == _GPU_CHECK
    return isinstance(node, ast.ImportFrom) and any(
        alias.name == _GPU_CHECK for alias in node.names
    )


def device_questions(source: str) -> list[str]:
    """The questions to a device a source asks: an import of `torch`, an attribute of its CUDA module, the
    count of devices, a string that names either or the vendor's query program, a reference to the
    doctor's GPU check."""
    tree = ast.parse(source)
    found: set[str] = set()
    for node in ast.walk(tree):
        if _imports_torch(node):
            found.add("import torch")
        if isinstance(node, ast.Attribute) and ast.unparse(node).startswith(
            _TORCH_CUDA
        ):
            found.add(_TORCH_CUDA)
        if _names_the_device_count(node):
            found.add(_DEVICE_COUNT)
        if _names_the_gpu_check(node):
            found.add(_GPU_CHECK)
    for text in _strings(tree):
        found.update(name for name in (_SMI, _TORCH_CUDA) if name in text)
    return sorted(found)


# ---- the two files the command only reads ---------------------------------------------------------


def writes_a_read_only_file(source: str) -> list[str]:
    """The functions (and `<module>` for the code outside them) that name `uv.lock` or `pyproject.toml` and
    also call something that writes, deletes, renames or opens."""
    tree = ast.parse(source)
    scopes: list[tuple[str, list[ast.AST]]] = [
        (node.name, [node])
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef)
    ]
    loose = [
        node
        for node in tree.body
        if not isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef)
    ]
    scopes.append(("<module>", loose))
    refused = []
    for name, parts in scopes:
        names_a_file = any(
            any(
                text.split("/")[-1] in READ_ONLY_FILES
                for text in _strings(ast.Module(body=[part], type_ignores=[]))
            )
            for part in parts
            if isinstance(part, ast.stmt)
        )
        calls = {
            ast.unparse(node.func).split(".")[-1]
            for part in parts
            for node in ast.walk(part)
            if isinstance(node, ast.Call)
        }
        if names_a_file and calls & WRITING_CALLS:
            refused.append(name)
    return refused


# ---- an argument list that could change the stack -------------------------------------------------


def changing_argument_lists(source: str) -> list[str]:
    """The change words held by a list, tuple or set of strings: `sync`, `pip`, `install`,
    `--upgrade-package`, and `add` unless the list also holds the other change words of a dry run (those
    three make a list of their own)."""
    found: list[str] = []
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, ast.List | ast.Tuple | ast.Set):
            continue
        words = {
            item.value
            for item in node.elts
            if isinstance(item, ast.Constant) and isinstance(item.value, str)
        }
        found.extend(sorted(words & CHANGING_ARGUMENTS))
        if "add" in words and not words & {"update", "remove"}:
            found.append("add")
    return found


def upgrade_flags_in_python(source: str) -> list[str]:
    """The upgrade flags a source spells as a string: `--upgrade` and `--upgrade-package`."""
    return [
        text
        for text in _strings(ast.parse(source))
        if text in (UPGRADE_FLAG, UPGRADE_PACKAGE_FLAG)
    ]


def upgrade_flags_in_configurations() -> list[tuple[str, list[str]]]:
    """Every `--upgrade` of a YAML document under `configs/`, as the file and the argument list it is
    in (the list holds `--dry-run` or the finding says it does not), and every `--upgrade-package`."""
    found: list[tuple[str, list[str]]] = []
    for path in sorted((REPO_ROOT / "configs").rglob("*.yaml")):
        found += [
            (path.relative_to(REPO_ROOT).as_posix(), words)
            for words in _lists_with_an_upgrade_flag(
                yaml.safe_load(path.read_text("utf-8"))
            )
        ]
    return found


def _lists_with_an_upgrade_flag(node: object) -> list[list[str]]:
    if isinstance(node, dict):
        return [
            found
            for value in node.values()
            for found in _lists_with_an_upgrade_flag(value)
        ]
    if isinstance(node, list):
        words = [item for item in node if isinstance(item, str)]
        own = [words] if UPGRADE_FLAG in words or UPGRADE_PACKAGE_FLAG in words else []
        return own + [
            found for item in node for found in _lists_with_an_upgrade_flag(item)
        ]
    return []


# ---- a subprocess without a timeout ---------------------------------------------------------------


def untimed_subprocesses(source: str) -> list[str]:
    """The processes a source starts without a bound: a `subprocess.run`, `call`, `check_call` or
    `check_output` without `timeout=`, a `Popen` in a source that never waits with `timeout=`
    (`communicate` or `wait`), and a wait without one for a process that was started."""
    tree = ast.parse(source)
    found: list[str] = []
    starts_a_process = False
    waits_with_a_timeout = False
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        callee = ast.unparse(node.func)
        timed = any(keyword.arg == "timeout" for keyword in node.keywords)
        if callee.startswith("subprocess."):
            name = callee.split(".")[-1]
            if name == "Popen":
                starts_a_process = True
            elif name in _BLOCKING_CALLS and not timed:
                found.append(f"line {node.lineno}: {callee} without a timeout")
        elif callee.split(".")[-1] in _WAITING_CALLS and timed:
            waits_with_a_timeout = True
    if starts_a_process and not waits_with_a_timeout:
        found.append("a Popen that is never waited for with a timeout")
    return found


# ---- the workflow ---------------------------------------------------------------------------------


def workflow_word_findings(text: str) -> list[str]:
    """The words of Task 6's scan that a workflow holds: the words it never mentions (anywhere, comments
    included), what a `run:` never says and a secret."""
    lowered = text.lower()
    found = [f"it mentions {word}" for word in FORBIDDEN_WORDS if word in lowered]
    found += [
        f"a run holds {forbidden!r}"
        for run in runs_of(load(text))
        for forbidden in FORBIDDEN_IN_RUN
        if forbidden in run
    ]
    if "secrets." in text:
        found.append("a step names a secret")
    return found
