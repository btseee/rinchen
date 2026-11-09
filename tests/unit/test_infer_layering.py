"""The weight of the shared rules of `rinchen.infer` (D11): the modules that carry them
import neither CTranslate2, SentencePiece, Eole nor torch at the top level (a processor is
passed in; the engine loads its library inside the function that needs it), and importing
them loads none of those."""

from __future__ import annotations

import ast
import os
import subprocess
import sys
from pathlib import Path

import pytest

import rinchen.infer

PACKAGE = Path(rinchen.infer.__file__).parent
HEAVY_LIBRARIES = ("ctranslate2", "sentencepiece", "eole", "torch")
SHARED_RULES = ("source_tokens.py", "script_mask.py")


def top_level_imports(source: str) -> set[str]:
    """The modules a source imports outside any function or class body."""
    imported: set[str] = set()
    for node in ast.parse(source).body:
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            imported.add(node.module)
    return imported


def heavy_among(imported: set[str]) -> list[str]:
    return sorted(
        module
        for module in imported
        if any(
            module == name or module.startswith(f"{name}.") for name in HEAVY_LIBRARIES
        )
    )


def test_the_scan_sees_a_top_level_import_and_not_a_late_one() -> None:
    source = (
        "import torch\n"
        "from ctranslate2 import Translator\n"
        "def load():\n"
        "    import sentencepiece\n"
    )

    assert heavy_among(top_level_imports(source)) == ["ctranslate2", "torch"]


@pytest.mark.parametrize("name", SHARED_RULES)
def test_the_shared_rules_import_no_heavy_library_at_the_top_level(name: str) -> None:
    source = (PACKAGE / name).read_text(encoding="utf-8")

    assert heavy_among(top_level_imports(source)) == []


def test_importing_the_shared_rules_loads_no_heavy_library() -> None:
    # The subprocess imports the very tree this test runs from.
    source_root = str(PACKAGE.parent.parent)
    code = (
        "import sys\n"
        "sys.path.insert(0, sys.argv[1])\n"
        "import rinchen.infer.source_tokens, rinchen.infer.script_mask\n"
        "assert rinchen.__file__.startswith(sys.argv[1]), rinchen.__file__\n"
        f"loaded = [m for m in {HEAVY_LIBRARIES!r} if m in sys.modules]\n"
        "print(loaded)\n"
    )
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": ""}

    result = subprocess.run(
        [sys.executable, "-I", "-c", code, source_root],
        capture_output=True,
        text=True,
        env=env,
        timeout=120,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "[]"
