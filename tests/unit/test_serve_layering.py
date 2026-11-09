"""The layering of `rinchen.serve` (D11), checked by reading the source.

`serve` may import `infer`, `export` and `config`; it imports nothing of `train`, `cli`,
`synth`, `script_conv` or `annotate`. The batcher is generic: it imports nothing of the
engine or of the export. Importing the serving modules loads neither CTranslate2,
SentencePiece, Eole nor torch (the engine loads its libraries when it opens a model).
"""

from __future__ import annotations

import ast
import os
import subprocess
import sys
from pathlib import Path

import pytest

import rinchen.serve

PACKAGE = Path(rinchen.serve.__file__).parent
FORBIDDEN_LAYERS = (
    "rinchen.train",
    "rinchen.cli",
    "rinchen.synth",
    "rinchen.script_conv",
    "rinchen.annotate",
)
HEAVY_LIBRARIES = ("ctranslate2", "sentencepiece", "eole", "torch")
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


def among(imported: set[str], packages: tuple[str, ...]) -> list[str]:
    return sorted(
        module
        for module in imported
        if any(is_inside(module, package) for package in packages)
    )


def test_the_scan_finds_what_it_is_told_to_forbid() -> None:
    source = (
        "import rinchen.cli.main\n"
        "from rinchen import train\n"
        "from rinchen.annotate.app import create_app\n"
        "def late():\n"
        "    from rinchen.script_conv import runtime\n"
        "    import rinchen.synth.run\n"
    )

    assert among(imported_names(source), FORBIDDEN_LAYERS) == [
        "rinchen.annotate.app",
        "rinchen.annotate.app.create_app",
        "rinchen.cli.main",
        "rinchen.script_conv",
        "rinchen.script_conv.runtime",
        "rinchen.synth.run",
        "rinchen.train",
    ]


def test_the_scan_does_not_mistake_a_neighbour_for_a_layer() -> None:
    source = "from rinchen.trainer import x\nimport rinchen.clients\n"

    assert among(imported_names(source), FORBIDDEN_LAYERS) == []


def test_the_scan_refuses_a_relative_import_it_cannot_follow() -> None:
    with pytest.raises(AssertionError, match="relative import"):
        imported_names("from ..train import x\n")


def test_the_scan_covers_the_modules_of_the_serving_core() -> None:
    names = {path.name for path in MODULES}

    assert {"__init__.py", "batcher.py", "metrics.py", "schemas.py"} <= names


@pytest.mark.parametrize("module", MODULES, ids=lambda path: path.name)
def test_a_module_of_the_serve_package_imports_nothing_of_the_layers_above_it(
    module: Path,
) -> None:
    imported = imported_names(module.read_text(encoding="utf-8"))

    assert among(imported, FORBIDDEN_LAYERS) == []


@pytest.mark.parametrize("module", MODULES, ids=lambda path: path.name)
def test_a_module_of_the_serve_package_loads_no_heavy_library_at_the_top_level(
    module: Path,
) -> None:
    top_level: set[str] = set()
    for node in ast.parse(module.read_text(encoding="utf-8")).body:
        if isinstance(node, ast.Import):
            top_level.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            top_level.add(node.module)

    assert among(top_level, HEAVY_LIBRARIES) == []


def test_the_batcher_is_generic_and_imports_neither_the_engine_nor_the_export() -> None:
    imported = imported_names((PACKAGE / "batcher.py").read_text(encoding="utf-8"))

    assert among(imported, ("rinchen.infer", "rinchen.export", *HEAVY_LIBRARIES)) == []


def test_the_configuration_of_the_service_does_not_import_the_service() -> None:
    config = PACKAGE.parent / "config" / "serve.py"

    imported = imported_names(config.read_text(encoding="utf-8"))

    assert among(imported, ("rinchen.serve", *FORBIDDEN_LAYERS)) == []


def test_importing_the_serving_modules_loads_no_library_that_only_the_engine_needs() -> (
    None
):
    # The subprocess imports the very tree this test runs from.
    source_root = str(PACKAGE.parent.parent)
    code = (
        "import sys\n"
        "sys.path.insert(0, sys.argv[1])\n"
        "import rinchen.config.serve, rinchen.serve.schemas\n"
        "import rinchen.serve.metrics, rinchen.serve.batcher\n"
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
