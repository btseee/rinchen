"""The layering of `rinchen.eval` (D12), checked by reading the source.

`eval` measures what a system wrote and never builds one: it imports nothing of `train`,
`synth`, `serve`, `cli`, `annotate`, `script_conv` or `filter`, and neither CTranslate2,
SentencePiece, Eole nor torch (the engine and the converter load them; the command reaches
them through those). Importing any module of the package loads none of them and not polars.
No source of `eval` and no test of the robustness suites asks about the GPU: the machine
has one that nothing here is authorised to touch.
"""

from __future__ import annotations

import ast
import os
import pkgutil
import subprocess
import sys
from pathlib import Path

import pytest

import rinchen.eval

PACKAGE = Path(rinchen.eval.__file__).parent
TESTS = Path(__file__).resolve().parents[1]
FORBIDDEN_LAYERS = (
    "rinchen.train",
    "rinchen.synth",
    "rinchen.serve",
    "rinchen.cli",
    "rinchen.annotate",
    "rinchen.script_conv",
    "rinchen.filter",
)
FORBIDDEN_LIBRARIES = ("ctranslate2", "sentencepiece", "eole", "torch")
NOT_LOADED_BY_IMPORTING = (*FORBIDDEN_LAYERS, *FORBIDDEN_LIBRARIES, "polars")
GPU_COUNT_NAME = "get_cuda_device_count"
GPU_DRIVER_TOOL = "nvidia-smi"
MODULES = sorted(PACKAGE.glob("*.py"))
ROBUSTNESS_TESTS = sorted(
    path for path in TESTS.rglob("*robust*") if path.is_file() and path.suffix == ".py"
)


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


def _imports_torch_cuda(node: ast.AST) -> bool:
    if isinstance(node, ast.Import):
        return any(is_inside(alias.name, "torch.cuda") for alias in node.names)
    if isinstance(node, ast.ImportFrom) and node.module is not None:
        return is_inside(node.module, "torch.cuda") or (
            node.module == "torch" and any(a.name == "cuda" for a in node.names)
        )
    return False


def gpu_queries(source: str) -> list[str]:
    """What a source asks of the GPU: `torch.cuda`, the CUDA device count or the driver
    tool, as a name, an attribute, an import or a string."""
    found: list[str] = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Attribute):
            if node.attr == GPU_COUNT_NAME:
                found.append(GPU_COUNT_NAME)
            is_torch = isinstance(node.value, ast.Name) and node.value.id == "torch"
            if node.attr == "cuda" and is_torch:
                found.append("torch.cuda")
        elif isinstance(node, ast.Name) and node.id == GPU_COUNT_NAME:
            found.append(GPU_COUNT_NAME)
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            if GPU_DRIVER_TOOL in node.value:
                found.append(GPU_DRIVER_TOOL)
        elif _imports_torch_cuda(node):
            found.append("torch.cuda")
    return found


def test_the_scan_finds_what_it_is_told_to_forbid() -> None:
    source = (
        "import ctranslate2\n"
        "import rinchen.cli.main\n"
        "from rinchen import train\n"
        "from rinchen.filter.numbers import digit_multiset\n"
        "from rinchen.script_conv import runtime\n"
        "def late():\n"
        "    from rinchen.serve import app\n"
        "    import sentencepiece, eole.bin\n"
        "    from torch import nn\n"
        "import rinchen.annotate, rinchen.synth.run\n"
    )

    assert forbidden_among(imported_names(source)) == [
        "ctranslate2",
        "eole.bin",
        "rinchen.annotate",
        "rinchen.cli.main",
        "rinchen.filter.numbers",
        "rinchen.filter.numbers.digit_multiset",
        "rinchen.script_conv",
        "rinchen.script_conv.runtime",
        "rinchen.serve",
        "rinchen.serve.app",
        "rinchen.synth.run",
        "rinchen.train",
        "sentencepiece",
        "torch",
        "torch.nn",
    ]


def test_the_scan_lets_the_allowed_layers_through() -> None:
    source = (
        "from rinchen.text.numerals import fold_numerals\n"
        "from rinchen.infer.script_mask import script_purity\n"
        "from rinchen.metrics.bleu_chrf import corpus_scores\n"
        "from rinchen.spm.symbols import user_defined_symbols\n"
        "import regex\n"
        "from rinchen import trainer_not_a_layer_name\n"
    )

    assert forbidden_among(imported_names(source)) == []


def test_the_gpu_scan_finds_every_way_to_ask_about_the_gpu() -> None:
    source = (
        "import torch\n"
        "import torch.cuda\n"
        "from torch.cuda import device_count\n"
        "from torch import cuda\n"
        "count = torch.cuda.device_count()\n"
        f"n = ctranslate2.{GPU_COUNT_NAME}()\n"
        f"m = {GPU_COUNT_NAME}()\n"
        f"run(['{GPU_DRIVER_TOOL}', '-L'])\n"
    )

    assert sorted(gpu_queries(source)) == sorted(
        [
            "torch.cuda",
            "torch.cuda",
            "torch.cuda",
            "torch.cuda",
            GPU_COUNT_NAME,
            GPU_COUNT_NAME,
            GPU_DRIVER_TOOL,
        ]
    )


def test_the_gpu_scan_lets_a_source_that_never_asks_through() -> None:
    source = (
        "import os\n"
        "env = {'CUDA_VISIBLE_DEVICES': ''}\n"
        "device = 'cpu'\n"
        "text = 'a number such as 12'\n"
    )

    assert gpu_queries(source) == []


def test_the_scan_covers_the_package_and_the_robustness_tests() -> None:
    assert PACKAGE / "robust_numbers.py" in MODULES
    assert TESTS / "unit" / "test_eval_robust_numbers.py" in ROBUSTNESS_TESTS
    assert TESTS / "property" / "test_eval_robust_numbers_property.py" in (
        ROBUSTNESS_TESTS
    )


@pytest.mark.parametrize("path", MODULES, ids=lambda path: path.name)
def test_a_module_of_eval_imports_no_forbidden_layer_or_heavy_library(
    path: Path,
) -> None:
    imported = imported_names(path.read_text(encoding="utf-8"))

    assert forbidden_among(imported) == []


@pytest.mark.parametrize(
    "path",
    [*MODULES, *ROBUSTNESS_TESTS],
    ids=lambda path: path.name,
)
def test_no_source_of_eval_or_of_its_tests_asks_about_the_gpu(path: Path) -> None:
    assert gpu_queries(path.read_text(encoding="utf-8")) == []


def test_importing_every_module_of_eval_loads_no_forbidden_library_and_not_polars() -> (
    None
):
    names = sorted(
        f"rinchen.eval.{module.name}" for module in pkgutil.iter_modules([str(PACKAGE)])
    )
    # The subprocess imports the very tree this test runs from.
    source_root = str(PACKAGE.parent.parent)
    code = (
        "import importlib, sys\n"
        "sys.path.insert(0, sys.argv[1])\n"
        "import rinchen.eval\n"
        "for name in sys.argv[2:]:\n"
        "    importlib.import_module(name)\n"
        "assert sys.modules['rinchen'].__file__.startswith(sys.argv[1])\n"
        f"watched = {NOT_LOADED_BY_IMPORTING!r}\n"
        "loaded = sorted(m for m in sys.modules\n"
        "                if any(m == w or m.startswith(w + '.') for w in watched))\n"
        "print(loaded)\n"
    )
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": ""}

    result = subprocess.run(
        [sys.executable, "-I", "-c", code, source_root, *names],
        capture_output=True,
        text=True,
        env=env,
        timeout=120,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "[]"
