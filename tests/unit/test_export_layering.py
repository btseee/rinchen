"""The layering of `rinchen.export` (D11), checked by reading the source.

`export` is below the layers that use it: it imports nothing of `train`, `synth`, `cli`,
`script_conv`, `infer` or `serve`, and not CTranslate2, which only the engine loads (the
converter runs as a process of its own and an export is read without it). Importing the
package loads neither CTranslate2, SentencePiece, Eole nor torch.
"""

from __future__ import annotations

import ast
import os
import subprocess
import sys
from pathlib import Path

import pytest

import rinchen.export

PACKAGE = Path(rinchen.export.__file__).parent
FORBIDDEN_LAYERS = (
    "rinchen.train",
    "rinchen.synth",
    "rinchen.cli",
    "rinchen.script_conv",
    "rinchen.infer",
    "rinchen.serve",
)
FORBIDDEN_LIBRARIES = ("ctranslate2",)
MODULES = sorted(PACKAGE.glob("*.py"))


def imported_names(source: str) -> set[str]:
    """Every module a source imports, and for `from m import n` also `m.n`, wherever the
    import is (a function body included)."""
    imported: set[str] = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            assert node.level == 0, "a relative import hides what is imported"
            assert node.module is not None
            imported.add(node.module)
            imported.update(f"{node.module}.{alias.name}" for alias in node.names)
    return imported


def is_inside(module: str, package: str) -> bool:
    return module == package or module.startswith(f"{package}.")


def forbidden_among(imported: set[str]) -> list[str]:
    return sorted(
        module
        for module in imported
        if any(is_inside(module, layer) for layer in FORBIDDEN_LAYERS)
        or any(is_inside(module, library) for library in FORBIDDEN_LIBRARIES)
    )


def test_the_scan_finds_what_it_is_told_to_forbid() -> None:
    source = (
        "import ctranslate2\n"
        "import rinchen.cli.main\n"
        "from rinchen import train\n"
        "from rinchen.infer.engine import Engine\n"
        "def late():\n"
        "    from rinchen.serve import app\n"
        "    import ctranslate2.converters\n"
    )

    assert forbidden_among(imported_names(source)) == [
        "ctranslate2",
        "ctranslate2.converters",
        "rinchen.cli.main",
        "rinchen.infer.engine",
        "rinchen.infer.engine.Engine",
        "rinchen.serve",
        "rinchen.serve.app",
        "rinchen.train",
    ]


def test_the_scan_does_not_mistake_a_neighbour_for_a_layer() -> None:
    source = "from rinchen.trainer import x\nimport rinchen.servers\n"

    assert forbidden_among(imported_names(source)) == []


def test_the_scan_refuses_a_relative_import_it_cannot_follow() -> None:
    with pytest.raises(AssertionError, match="relative import"):
        imported_names("from ..train import x\n")


def test_the_scan_covers_every_module_of_the_package() -> None:
    assert [path.name for path in MODULES] == [
        "__init__.py",
        "ct2.py",
        "eole_ct2.py",
        "model_card.py",
        "model_card_evaluation.py",
        "model_card_notes.py",
        "quantize.py",
        "record.py",
        "release.py",
        "verified.py",
    ]


@pytest.mark.parametrize("module", MODULES, ids=lambda path: path.name)
def test_a_module_of_the_export_package_imports_nothing_of_the_layers_above_it(
    module: Path,
) -> None:
    imported = imported_names(module.read_text(encoding="utf-8"))

    assert forbidden_among(imported) == []


def test_importing_the_export_package_loads_no_library_that_only_the_engine_needs() -> (
    None
):
    # The subprocess imports the very tree this test runs from.
    source_root = str(PACKAGE.parent.parent)
    code = (
        "import sys\n"
        "sys.path.insert(0, sys.argv[1])\n"
        "import rinchen.export.ct2, rinchen.export.record, rinchen.export.verified\n"
        "import rinchen.export.quantize, rinchen.export.eole_ct2\n"
        "assert rinchen.__file__.startswith(sys.argv[1]), rinchen.__file__\n"
        "loaded = [m for m in ('ctranslate2', 'sentencepiece', 'eole', 'torch') "
        "if m in sys.modules]\n"
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
