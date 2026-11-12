"""The hygiene scans of the D15 gate (R14), made by reading source (an AST scan, so a word in a comment or a
docstring never counts and an alias cannot hide a name): every process the gate starts is started by ONE
function that refuses to start without the guards, in a session of its own, with the environment it was
given; nothing starts a process any other way; every wait is bounded; no argument names a device other
than the CPU; and no file D15 adds holds a word the project's content rules exclude.

Each scan takes a source (or a text) and returns what it found, so that a test asserts the list is empty
on the files and non-empty on a string that breaks the rule. The words of a device are written in pieces:
the scans cover the files of the gate, this one included.
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Final

from rinchen.config.loader import REPO_ROOT
from tests.unit.test_d13_content_rules import lines_with_a_needle

# The guards the starter's module must hold as a literal, with the value each must have.
REQUIRED_GUARDS: Final = {
    "CUDA_VISIBLE_DEVICES": "",
    "HF_HUB_OFFLINE": "1",
    "HF_DATASETS_OFFLINE": "1",
    "UV_OFFLINE": "1",
}
GUARDS_NAME: Final = "ENVIRONMENT_GUARDS"
STARTER: Final = "Popen"
REFUSAL: Final = "require_guards"
_SUBPROCESS_STARTS: Final = frozenset(
    {
        "Popen",
        "run",
        "call",
        "check_call",
        "check_output",
        "getoutput",
        "getstatusoutput",
    }
)
_OS_START_PREFIXES: Final = ("spawn", "exec")
_OS_STARTS: Final = frozenset({"system", "popen", "fork", "forkpty", "posix_spawn"})
_WAITS: Final = frozenset({"communicate", "wait"})
# The two words of a device that is not the CPU, in pieces: this file is scanned too.
_NOT_THE_CPU: Final = ("au" + "to", "cu" + "da")

# Everything D15 adds that a human reads or a run reads: code, configurations, documents, fixtures and tests.
CONTENT_GLOBS: Final = (
    "src/rinchen/tiny/*.py",
    "src/rinchen/config/tiny.py",
    "src/rinchen/cli/tiny*.py",
    "src/rinchen/eval/fixture_suites.py",
    "configs/tiny/*",
    "configs/eole/tiny.yaml",
    "configs/data/sources.tiny_parallel.yaml",
    "configs/profiles/tiny.yaml",
    "docs/superpowers/specs/2026-11-11-d15-*",
    "docs/superpowers/plans/2026-11-11-d15-*",
    "docs/runbooks/15_*.md",
    "scripts/build_parallel_fixtures.py",
    "justfile",
    "tests/fixtures/parallel/*",
    "tests/fixtures/tiny/*",
    "tests/fixtures/tiny_run/*",
    "tests/support/tiny_*.py",
    "tests/support/ci_workflow*.py",
    "tests/unit/test_tiny_*.py",
    "tests/unit/test_cli_tiny*.py",
    "tests/unit/test_config_tiny*.py",
    "tests/unit/test_support_tiny_*.py",
    "tests/unit/test_ci_workflow.py",
    "tests/unit/test_parallel_fixtures.py",
    "tests/unit/test_eval_fixture_suites.py",
    "tests/property/test_tiny_*.py",
    "tests/integration/test_tiny_*.py",
)


# ---- reading calls -----------------------------------------------------------------------------------


def _aliases(tree: ast.AST, module: str) -> tuple[set[str], dict[str, str]]:
    """The names a source gives the module `module` (`import m`, `import m as a`) and the names it
    imports from it (`from m import n as b` gives `{"b": "n"}`)."""
    modules: set[str] = set()
    names: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(
                alias.asname or alias.name
                for alias in node.names
                if alias.name == module
            )
        elif isinstance(node, ast.ImportFrom) and node.module == module:
            names.update(
                {alias.asname or alias.name: alias.name for alias in node.names}
            )
    return modules, names


def _is_os_start(name: str) -> bool:
    return name in _OS_STARTS or name.startswith(_OS_START_PREFIXES)


def _name_of_start(
    func: ast.expr,
    subprocess_modules: set[str],
    subprocess_names: dict[str, str],
    os_modules: set[str],
    os_names: dict[str, str],
) -> str | None:
    """The name of the process-starting function that a call's `func` refers to, or None."""
    if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name):
        owner, attribute = func.value.id, func.attr
        if owner in subprocess_modules and attribute in _SUBPROCESS_STARTS:
            return attribute
        if owner in os_modules and _is_os_start(attribute):
            return attribute
    elif isinstance(func, ast.Name):
        imported = subprocess_names.get(func.id)
        if imported in _SUBPROCESS_STARTS:
            return imported
        if _is_os_start(os_names.get(func.id, "")):
            return os_names[func.id]
    return None


def _starts(tree: ast.AST) -> list[tuple[int, str, ast.Call]]:
    """Every call that starts a process, as (line, the name started, the call), in source order."""
    aliases = (*_aliases(tree, "subprocess"), *_aliases(tree, "os"))
    found: list[tuple[int, str, ast.Call]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            name = _name_of_start(node.func, *aliases)
            if name is not None:
                found.append((node.lineno, name, node))
    return sorted(found, key=lambda item: item[0])


def process_starts(source: str) -> list[str]:
    """The names of the calls that start a process in a source (`Popen`, `run`, `system`, `execv`, ...),
    in source order: a call through an alias or a `from` import counts."""
    return [name for _, name, _ in _starts(ast.parse(source))]


# ---- the starter's module ----------------------------------------------------------------------------


def _guards_literal(tree: ast.Module) -> ast.expr | None:
    for node in tree.body:
        if (
            isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and node.target.id == GUARDS_NAME
        ):
            return node.value
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id == GUARDS_NAME
        ):
            return node.value
    return None


def _guard_literal_findings(tree: ast.Module) -> list[str]:
    value = _guards_literal(tree)
    if not isinstance(value, ast.Dict):
        return [f"{GUARDS_NAME} is not a dict literal"]
    held: dict[object, object] = {}
    for key, item in zip(value.keys, value.values, strict=True):
        if isinstance(key, ast.Constant) and isinstance(item, ast.Constant):
            held[key.value] = item.value
    findings = []
    for name, expected in REQUIRED_GUARDS.items():
        if name not in held:
            findings.append(f"{GUARDS_NAME} lacks {name}")
        elif held[name] != expected:
            findings.append(f"{GUARDS_NAME} has the wrong value for {name}")
    return findings


def _enclosing(tree: ast.AST, target: ast.AST) -> ast.FunctionDef | None:
    """The innermost function that holds `target`, or None for code outside any function."""
    holders = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef)
        and any(child is target for child in ast.walk(node))
    ]
    # A function nested in another starts later in the file than the one it is in.
    return max(holders, key=lambda node: node.lineno, default=None)


def _calls_the_refusal_before(function: ast.FunctionDef | None, line: int) -> bool:
    if function is None:
        return False
    return any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == REFUSAL
        and node.lineno < line
        for node in ast.walk(function)
    )


def _start_findings(tree: ast.Module) -> list[str]:
    findings = []
    for line, name, call in _starts(tree):
        if name != STARTER:
            findings.append(
                f"line {line}: {name} starts a process outside the guarded starter"
            )
            continue
        keywords = {keyword.arg: keyword.value for keyword in call.keywords}
        session = keywords.get("start_new_session")
        if not (isinstance(session, ast.Constant) and session.value is True):
            findings.append(f"line {line}: Popen without start_new_session=True")
        if "env" not in keywords:
            findings.append(f"line {line}: Popen without env=")
        if not _calls_the_refusal_before(_enclosing(tree, call), line):
            findings.append(
                f"line {line}: Popen not preceded by {REFUSAL} in its function"
            )
    return findings


def untimed_waits(source: str) -> list[str]:
    """The waits of a source (`communicate`, `wait`) that carry no `timeout=`."""
    return [
        f"line {node.lineno}: {node.func.attr} without a timeout"
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr in _WAITS
        and not any(keyword.arg == "timeout" for keyword in node.keywords)
    ]


def guard_findings(source: str) -> list[str]:
    """What is wrong in a module that starts the gate's processes: the guards are not the literal the
    scan reads, a `Popen` lacks the session, the environment or the refusal before it, a process starts
    any other way, or a wait is not bounded. A module that does it right has none."""
    tree = ast.parse(source)
    return [
        *_guard_literal_findings(tree),
        *_start_findings(tree),
        *untimed_waits(source),
    ]


# ---- a device that is not the CPU ----------------------------------------------------------------------


def _names_another_device(node: ast.expr) -> bool:
    return (
        isinstance(node, ast.Constant)
        and isinstance(node.value, str)
        and node.value.lower().startswith(_NOT_THE_CPU)
    )


def device_value_arguments(source: str) -> list[str]:
    """The arguments of a source that name a device other than the CPU: a keyword `device=` or a default
    of a parameter called `device` whose value is a string that starts with either word."""
    findings = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Call):
            findings.extend(
                f"line {node.lineno}: a call with device={keyword.value.value!r}"
                for keyword in node.keywords
                if keyword.arg == "device"
                and _names_another_device(keyword.value)
                and isinstance(keyword.value, ast.Constant)
            )
        elif isinstance(node, ast.arguments):
            positional = [*node.posonlyargs, *node.args]
            pairs = list(
                zip(reversed(positional), reversed(node.defaults), strict=False)
            ) + [
                (argument, default)
                for argument, default in zip(
                    node.kwonlyargs, node.kw_defaults, strict=True
                )
                if default is not None
            ]
            findings.extend(
                f"line {default.lineno}: a parameter device defaults to a device that is not the CPU"
                for argument, default in pairs
                if argument.arg == "device" and _names_another_device(default)
            )
    return findings


# ---- the content rules ----------------------------------------------------------------------------------


def needle_lines(text: str) -> list[tuple[int, int]]:
    """`(line number, needle number)` of every line of `text` that holds a word the project's content
    rules exclude; the word is never returned."""
    return lines_with_a_needle(text)


def content_files() -> list[Path]:
    """Every file D15 adds that the content rules cover, relative to the repository root, sorted."""
    paths = {
        path.relative_to(REPO_ROOT)
        for pattern in CONTENT_GLOBS
        for path in REPO_ROOT.glob(pattern)
        if path.is_file() and "__pycache__" not in path.parts
    }
    return sorted(paths)
