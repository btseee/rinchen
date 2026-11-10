"""What a module of the project imports, read from its source, and what importing it loads, read
from a fresh interpreter: the two halves of a layering test for a module that sits outside the
packages `tests/unit/test_eval_layering.py` already scans."""

from __future__ import annotations

import ast
import os
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

SOURCE_ROOT = Path(__file__).resolve().parents[2] / "src"


def source_of(module: str) -> Path:
    """The file of a module of `rinchen`, without importing it."""
    return SOURCE_ROOT.joinpath(*module.split(".")).with_suffix(".py")


def imported_modules(path: Path) -> set[str]:
    """Every module the file imports, wherever the import is (a function body included): the
    `m` of `import m` and of `from m import n`. `from rinchen import n` is listed as `rinchen`,
    which no allowlist holds, so it is looked at. A relative import hides what is imported and
    is refused."""
    imported: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            assert node.level == 0, "a relative import hides what is imported"
            assert node.module is not None
            imported.add(node.module)
    return imported


def project_imports(path: Path) -> set[str]:
    """The modules of `rinchen` the file imports."""
    return {
        module
        for module in imported_modules(path)
        if module == "rinchen" or module.startswith("rinchen.")
    }


def loaded_by_importing(modules: Sequence[str], watched: Sequence[str]) -> list[str]:
    """The names among `watched` (a module or a package) that are in `sys.modules` after a fresh
    interpreter has imported `modules` from this tree."""
    code = (
        "import importlib, sys\n"
        "sys.path.insert(0, sys.argv[1])\n"
        "for name in sys.argv[2].split(','):\n"
        "    importlib.import_module(name)\n"
        "assert sys.modules['rinchen'].__file__.startswith(sys.argv[1])\n"
        "watched = sys.argv[3].split(',')\n"
        "print(sorted(m for m in sys.modules\n"
        "             if any(m == w or m.startswith(w + '.') for w in watched)))\n"
    )
    result = subprocess.run(
        [
            sys.executable,
            "-I",
            "-c",
            code,
            str(SOURCE_ROOT),
            ",".join(modules),
            ",".join(watched),
        ],
        capture_output=True,
        text=True,
        env={**os.environ, "CUDA_VISIBLE_DEVICES": ""},
        timeout=120,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    loaded: list[str] = ast.literal_eval(result.stdout.strip())
    return loaded
