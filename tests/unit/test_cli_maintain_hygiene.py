"""The hygiene of the modules of `rinchen maintain check` (D14, spec M1, M4 and M18), checked by
reading their source (an AST scan, so a word in a comment never counts and an alias cannot hide a
name): the layering of the command modules, no question to a device, no write to the two files the
command only reads, no argument list that could change the stack, and no line of another program's
output in anything the command prints or writes outside the logs.

The scans run over every module of the command (`cli/maintain*.py`) and every module of the
library, so a module that a later task adds is covered without an edit here, and they are
themselves tested on source strings that break each rule.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

import pytest

from rinchen.maintain.commands import CommandResult
from tests.support.maintain_check import golden_doctor
from tests.support.maintain_cli import (
    Workspace,
    install,
    invoke,
    make_workspace,
    version_call,
)
from tests.support.maintain_fakes import FakeRunner
from tests.support.maintain_scans import violations as content_violations
from tests.support.no_network import no_network  # noqa: F401 - the fixture of the mark
from tests.support.uv_outputs import NO_CHANGE_MEASURED

SOURCE = Path(__file__).resolve().parents[2] / "src" / "rinchen"
COMMAND_MODULES = sorted((SOURCE / "cli").glob("maintain*.py"))
LIBRARY_MODULES = [
    *sorted((SOURCE / "maintain").glob("*.py")),
    SOURCE / "config" / "maintain.py",
]
ALL_MODULES = [*COMMAND_MODULES, *LIBRARY_MODULES]
LINE_LIMIT = 450

# What a command module may import of the project: the doctor, the shared command pieces, the
# configuration, the verified reader, the manifest, the part-file writer, the library and itself.
ALLOWED_PROJECT = (
    "rinchen.cli.doctor",
    "rinchen.cli.errors",
    "rinchen.cli.train_common",
    "rinchen.cli.maintain",
    "rinchen.cli.maintain_doctor",
    "rinchen.cli.maintain_inputs",
    "rinchen.cli.maintain_output",
    "rinchen.cli.maintain_signals",
    "rinchen.config.loader",
    "rinchen.config.maintain",
    "rinchen.errors",
    "rinchen.export.verified",
    "rinchen.maintain",
    "rinchen.manifest",
    "rinchen.utils.part_file",
)
ANCESTORS = (
    "rinchen",
    "rinchen.cli",
    "rinchen.config",
    "rinchen.export",
    "rinchen.utils",
)
THIRD_PARTY = ("typer",)
FORBIDDEN_ROOTS = (
    "rinchen.train",
    "rinchen.serve",
    "rinchen.eval",
    "rinchen.synth",
    "rinchen.infer",
    "rinchen.annotate",
    "rinchen.filter",
    "ctranslate2",
    "eole",
    "torch",
    "sentencepiece",
    "polars",
    "huggingface_hub",
)
DEVICE_WORDS = ("torch", "cuda", "nvidia-smi")
READ_ONLY_FILES = ("uv.lock", "pyproject.toml")
WRITING_CALLS = {
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
CHANGING_ARGUMENTS = {"sync", "pip", "--upgrade-package", "install"}


def imported_names(source: str) -> set[str]:
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


def import_violations(source: str) -> list[str]:
    """The imports of a command module that the layering of M18 does not allow."""
    refused = []
    for name in sorted(imported_names(source)):
        top = name.split(".")[0]
        if name.startswith(".") or any(is_inside(name, p) for p in FORBIDDEN_ROOTS):
            refused.append(name)
        elif top == "rinchen":
            allowed = name in ANCESTORS or any(
                is_inside(name, package) for package in ALLOWED_PROJECT
            )
            if not allowed:
                refused.append(name)
        elif top not in sys.stdlib_module_names and top not in THIRD_PARTY:
            refused.append(name)
    return refused


def strings_and_names(source: str) -> list[str]:
    """Every identifier, attribute name and string constant of a source (docstrings included),
    in lower case."""
    found = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Name):
            found.append(node.id)
        elif isinstance(node, ast.Attribute):
            found.append(node.attr)
        elif isinstance(
            node, ast.alias | ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef
        ):
            found.append(node.name)
        elif isinstance(node, ast.arg):
            found.append(node.arg)
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            found.append(node.value)
    return [item.lower() for item in found]


def device_violations(source: str) -> list[str]:
    """The words of a device question that a command module holds."""
    items = strings_and_names(source)
    return sorted({word for word in DEVICE_WORDS for item in items if word in item})


def library_device_violations(source: str) -> list[str]:
    """The device questions of D11 and D12's scans: `torch.cuda`, `get_cuda_device_count`,
    `nvidia-smi` and a call to the doctor's GPU check."""
    found = []
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and ast.unparse(node).startswith(
            "torch.cuda"
        ):
            found.append("torch.cuda")
        elif (
            isinstance(node, ast.Name)
            and node.id == "get_cuda_device_count"
            or isinstance(node, ast.Attribute)
            and node.attr == "get_cuda_device_count"
        ):
            found.append("get_cuda_device_count")
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            if "nvidia-smi" in node.value:
                found.append("nvidia-smi")
        elif isinstance(node, ast.Call):
            callee = ast.unparse(node.func)
            if callee.split(".")[-1] == "check_torch_cuda":
                found.append("check_torch_cuda()")
    return sorted(set(found))


def writes_a_read_only_file(source: str) -> list[str]:
    """The functions that name `uv.lock` or `pyproject.toml` and also call something that writes,
    deletes, renames or opens."""
    refused = []
    tree = ast.parse(source)
    for function in ast.walk(tree):
        if not isinstance(function, ast.FunctionDef | ast.AsyncFunctionDef):
            continue
        names_a_file = any(
            isinstance(node, ast.Constant)
            and isinstance(node.value, str)
            and node.value.split("/")[-1] in READ_ONLY_FILES
            for node in ast.walk(function)
        )
        calls = {
            ast.unparse(node.func).split(".")[-1]
            for node in ast.walk(function)
            if isinstance(node, ast.Call)
        }
        if names_a_file and calls & WRITING_CALLS:
            refused.append(function.name)
    return refused


def changing_argument_lists(source: str) -> list[str]:
    """The argument lists that could change the stack: a list or tuple that holds `sync`, `pip`,
    `install` or `--upgrade-package`, or an `add` with no other change word (the three change
    words of a dry run make a list of their own)."""
    found = []
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


# ---- the scans, on strings ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "source",
    [
        "import torch",
        "from rinchen.train import launch",
        "from rinchen.serve import app",
        "from rinchen.eval import suites",
        "import ctranslate2",
        "import polars as pl",
        "from huggingface_hub import HfApi",
        "from rinchen.cli import release",
        "from rinchen.cli.train import app",
        "from . import maintain",
        "import numpy",
        "import eole",
        "from rinchen.cli import main",
    ],
)
def test_the_import_scan_refuses_what_the_layering_forbids(source: str) -> None:
    assert import_violations(source) != []


@pytest.mark.parametrize(
    "source",
    [
        "import os\nimport shutil\nimport sys",
        "import typer",
        "from rinchen.cli.doctor import DoctorResult",
        "from rinchen.cli import doctor",
        "from rinchen.cli.errors import exit_with_error",
        "from rinchen.cli.train_common import hash_input",
        "from rinchen.export.verified import verified_content",
        "from rinchen.manifest import store",
        "from rinchen.manifest.stage import begin_commit",
        "from rinchen.utils.part_file import replace_file",
        "from rinchen.maintain.check import run_check",
        "from rinchen.config.maintain import load_maintain_config",
        "from rinchen.config.loader import REPO_ROOT",
        "from rinchen.errors import ConfigError",
    ],
)
def test_the_import_scan_allows_the_pieces_the_command_is_built_from(
    source: str,
) -> None:
    assert import_violations(source) == []


@pytest.mark.parametrize(
    "source",
    [
        "x = doctor.check_torch_cuda",
        "import torch",
        "x = 'nvidia-smi'",
        "def cuda_available(): pass",
        "from rinchen.cli.doctor import check_torch_cuda",
        "'''D-TORCH-CUDA'''",
    ],
)
def test_the_device_scan_sees_every_way_to_name_a_device_question(source: str) -> None:
    assert device_violations(source) != []


@pytest.mark.parametrize(
    "source",
    [
        "x = torch.cuda.is_available()",
        "x = pynvml.nvmlDeviceGetCount()\nn = get_cuda_device_count()",
        "x = lib.get_cuda_device_count",
        "subprocess.run(['nvidia-smi'])",
        "doctor.check_torch_cuda()",
        "check_torch_cuda()",
    ],
)
def test_the_library_device_scan_sees_the_questions_a_command_could_ask(
    source: str,
) -> None:
    assert library_device_violations(source) != []


@pytest.mark.parametrize(
    "source",
    [
        "def f():\n    open('uv.lock', 'w')",
        "def f(p):\n    (p / 'pyproject.toml').write_text('x')",
        "def f(p):\n    (p / 'uv.lock').unlink()",
        "def f(p):\n    p.replace(p / 'uv.lock')",
        "def f(p):\n    name = 'pyproject.toml'\n    p.rename(name)",
        "def f():\n    replace_file(project / 'uv.lock', b'')",
    ],
)
def test_the_write_scan_sees_a_function_that_names_a_read_only_file_and_writes(
    source: str,
) -> None:
    assert writes_a_read_only_file(source) != []


@pytest.mark.parametrize(
    "source",
    [
        "def f(p):\n    return (p / 'uv.lock').read_bytes()",
        "def f(p):\n    name = 'uv.lock'\n    return hash_input(p / name)",
        "def g(p):\n    p.write_text('x')",
    ],
)
def test_the_write_scan_lets_a_function_read_the_files_or_write_others(
    source: str,
) -> None:
    assert writes_a_read_only_file(source) == []


@pytest.mark.parametrize(
    "source",
    [
        "argv = [uv, 'sync', '--frozen']",
        "argv = [uv, 'pip', 'install', 'x']",
        "argv = [uv, 'add', 'numpy']",
        "argv = (uv, 'lock', '--upgrade-package', 'numpy')",
        "ARGS = {'install'}",
    ],
)
def test_the_argument_scan_sees_a_list_that_could_change_the_stack(source: str) -> None:
    assert changing_argument_lists(source) != []


@pytest.mark.parametrize(
    "source",
    [
        "argv = [uv, 'lock', '--check']",
        "argv = [uv, 'lock', '--upgrade', '--dry-run']",
        "KINDS = ('update', 'add', 'remove')",
        "x = {'add': 1}",
    ],
)
def test_the_argument_scan_lets_the_dry_run_and_the_change_words_pass(
    source: str,
) -> None:
    assert changing_argument_lists(source) == []


# ---- the modules ---------------------------------------------------------------------------------------


def test_the_command_is_made_of_the_modules_the_task_names() -> None:
    assert {path.name for path in COMMAND_MODULES} >= {
        "maintain.py",
        "maintain_output.py",
        "maintain_doctor.py",
        "maintain_inputs.py",
        "maintain_signals.py",
    }


@pytest.mark.parametrize("path", COMMAND_MODULES, ids=lambda path: path.name)
def test_a_command_module_imports_nothing_the_layering_forbids(path: Path) -> None:
    assert import_violations(path.read_text(encoding="utf-8")) == []


@pytest.mark.parametrize("path", COMMAND_MODULES, ids=lambda path: path.name)
def test_a_command_module_names_no_device_question_at_all(path: Path) -> None:
    assert device_violations(path.read_text(encoding="utf-8")) == []


@pytest.mark.parametrize("path", ALL_MODULES, ids=lambda path: path.name)
def test_no_module_of_the_check_asks_a_device_question(path: Path) -> None:
    assert library_device_violations(path.read_text(encoding="utf-8")) == []


@pytest.mark.parametrize("path", ALL_MODULES, ids=lambda path: path.name)
def test_no_module_of_the_check_writes_the_lock_or_the_project_file(path: Path) -> None:
    assert writes_a_read_only_file(path.read_text(encoding="utf-8")) == []


@pytest.mark.parametrize("path", ALL_MODULES, ids=lambda path: path.name)
def test_no_module_of_the_check_builds_an_argument_list_that_changes_the_stack(
    path: Path,
) -> None:
    assert changing_argument_lists(path.read_text(encoding="utf-8")) == []


@pytest.mark.parametrize("path", COMMAND_MODULES, ids=lambda path: path.name)
def test_a_command_module_is_ascii_and_under_the_line_limit(path: Path) -> None:
    data = path.read_bytes()
    assert data.isascii(), path.name
    assert len(data.splitlines()) < LINE_LIMIT, path.name


def test_only_the_registry_of_commands_imports_the_maintain_command() -> None:
    importers = []
    for path in sorted(SOURCE.rglob("*.py")):
        if path in COMMAND_MODULES:
            continue
        names = imported_names(path.read_text(encoding="utf-8"))
        if any(is_inside(name, "rinchen.cli.maintain") for name in names):
            importers.append(path.relative_to(SOURCE).as_posix())
    assert importers == ["cli/main.py"]


# ---- the output of another program --------------------------------------------------------------------

UV_MARKER = "SECRET-UV-LINE-77"


@pytest.fixture
def ws(tmp_path: Path) -> Workspace:
    return make_workspace(tmp_path)


def failing_uv_runner() -> FakeRunner:
    """A runner whose uv fails, for the lock check and for the dry run, with a line of output
    that holds a marker and a path (and the dry run with the lines of a quiet one after them)."""
    text = f"error: {UV_MARKER} /home/someone/project\nhint: try again\n"
    return (
        FakeRunner()
        .script(
            CommandResult(2, False),
            log_text=text,
            when=lambda argv: list(argv[1:]) == ["lock", "--check"],
        )
        .script(
            CommandResult(2, False),
            log_text=text + NO_CHANGE_MEASURED,
            when=lambda argv: "--upgrade" in argv,
        )
        .script(CommandResult(0, False), log_text="uv 0.13.7\n", when=version_call)
    )


def test_no_line_of_the_output_of_uv_reaches_the_screen_or_a_document(
    ws: Workspace, monkeypatch: pytest.MonkeyPatch
) -> None:
    install(monkeypatch, runner=failing_uv_runner(), doctor=golden_doctor())

    result = invoke(ws.args("--no-reruns"))

    assert result.exit_code == 4, result.output
    assert UV_MARKER not in result.output
    assert "/home/someone" not in result.output
    written = ws.written()
    assert UV_MARKER.encode() in written["logs/drift.log"]
    for name, data in written.items():
        if name.startswith("logs/"):
            continue
        text = data.decode()
        assert UV_MARKER not in text and "/home/someone" not in text, name
    for name in ("maintain_report.json", "maintain_report.md"):
        assert content_violations(written[name].decode()) == [], name
