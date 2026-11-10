"""The layering of `rinchen.eval` (D12), checked by reading the source.

`eval` measures what a system wrote and never builds one: it imports nothing of `train`,
`synth`, `serve`, `cli`, `annotate`, `script_conv` or `filter`, and neither CTranslate2,
SentencePiece, Eole nor torch (the engine and the converter load them; the command reaches
them through those). Importing any module of the package loads none of them and not polars.
No source of `eval` and no test of the robustness suites asks about the GPU: the machine
has one that nothing here is authorised to touch.

The command `rinchen eval robustness` (`cli/eval_robustness.py`, the module that reads its
options and the one that writes its files) sits above `eval`: it may import the engine
(`infer`), the export reader, the converter (`script_conv.runtime`), the manifest and the
helpers of the other commands, but never the training, synthetic-data, serving, annotation or
filter layers and never a heavy library directly; it asks about the GPU no more than `eval` does.
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
CLI = PACKAGE.parent / "cli"
COMMAND_MODULES = [
    CLI / "eval_robustness.py",
    CLI / "eval_robustness_output.py",
    CLI / "eval_robustness_plan.py",
    CLI / "eval_suite.py",
    CLI / "eval_suite_files.py",
    CLI / "eval_suite_output.py",
    CLI / "eval_suite_plan.py",
    CLI / "eval_run.py",
    CLI / "eval_run_inputs.py",
    CLI / "eval_run_output.py",
    CLI / "eval_run_plan.py",
]
COMMAND_FORBIDDEN_LAYERS = (
    "rinchen.train",
    "rinchen.synth",
    "rinchen.serve",
    "rinchen.annotate",
    "rinchen.filter",
)
# The one command module that reaches D6's generators (`rinchen eval run` builds a baseline
# through them): it may import `synth`, and only the two modules that hold the language tables and
# the directions the prompts cover; every other layer stays forbidden to it.
SYNTH_COMMAND_MODULES = [CLI / "eval_systems.py"]
SYNTH_MODULES_ALLOWED = ("rinchen.synth.ct2_baselines", "rinchen.synth.prompts")
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
    for module in (
        "robust_numbers",
        "robust_placeholders",
        "robust_spans",
        "robust_script",
        "robust_mong",
        "robust_suite",
        "robust_suite_rules",
        "robust_hypotheses",
        "robust_outcomes",
        "robust_sections",
        "robustness",
        "robust_report",
    ):
        assert PACKAGE / f"{module}.py" in MODULES
    for unit_test in (
        "test_eval_robust_numbers",
        "test_eval_robust_placeholders",
        "test_eval_robust_spans",
        "test_eval_robust_script",
        "test_eval_robust_mong",
        "test_eval_robust_suite",
        "test_eval_robust_suite_refusals",
        "test_eval_robust_suite_field_refusals",
        "test_eval_robust_hypotheses",
        "test_eval_robustness",
        "test_eval_robustness_rows",
        "test_eval_robustness_document",
        "test_eval_robust_report",
        "test_cli_eval_robustness",
        "test_cli_eval_robustness_refusals",
        "test_cli_eval_robustness_failures",
        "test_cli_eval_robustness_messages",
        "test_cli_eval_robustness_manifest",
    ):
        assert TESTS / "unit" / f"{unit_test}.py" in ROBUSTNESS_TESTS
    for property_test in (
        "test_eval_robust_numbers_property",
        "test_eval_robust_spans_property",
        "test_eval_robust_suite_property",
        "test_eval_robustness_property",
    ):
        assert TESTS / "property" / f"{property_test}.py" in ROBUSTNESS_TESTS
    assert TESTS / "support" / "robust_suite_documents.py" in ROBUSTNESS_TESTS
    assert TESTS / "support" / "robust_golden.py" in ROBUSTNESS_TESTS
    for support in ("robust_toy", "robust_toy_text", "robust_suites", "robust_cli"):
        assert TESTS / "support" / f"{support}.py" in ROBUSTNESS_TESTS
    for toy_test in ("test_support_robust_toy", "test_support_robust_suites"):
        assert TESTS / "unit" / f"{toy_test}.py" in ROBUSTNESS_TESTS


def test_the_scan_covers_the_bootstrap_modules() -> None:
    for module in ("bootstrap", "significance"):
        assert PACKAGE / f"{module}.py" in MODULES


def test_the_scan_covers_the_modules_of_the_evaluation_suite() -> None:
    for module in ("suite_rules", "suites"):
        assert PACKAGE / f"{module}.py" in MODULES


def test_the_scan_covers_the_outputs_the_systems_and_the_runner() -> None:
    for module in ("outputs", "systems", "runner"):
        assert PACKAGE / f"{module}.py" in MODULES


def test_the_scan_covers_the_scoring_the_result_documents_and_the_robustness_input() -> (
    None
):
    for module in (
        "field_rules",
        "scoring",
        "segment_scores",
        "result",
        "run_record",
        "robustness_input",
    ):
        assert PACKAGE / f"{module}.py" in MODULES


def test_the_scan_covers_the_suite_sources_and_the_suite_commands() -> None:
    assert PACKAGE / "suite_sources.py" in MODULES
    for name in (
        "eval_suite",
        "eval_suite_files",
        "eval_suite_output",
        "eval_suite_plan",
    ):
        assert CLI / f"{name}.py" in COMMAND_MODULES


def test_the_suite_commands_may_read_tables_with_polars_and_fetch_through_acquire() -> (
    None
):
    # The only two commands that name polars (the Parquet, TSV and JSON Lines reading) and
    # `rinchen.acquire.hf` (the fetch): neither is a layer or a library the rule forbids.
    imported = imported_names((CLI / "eval_suite_files.py").read_text(encoding="utf-8"))
    fetching = imported_names((CLI / "eval_suite.py").read_text(encoding="utf-8"))

    assert "polars" in imported
    assert "rinchen.acquire.hf" in fetching
    for module in sorted(CLI.glob("eval_suite*.py")):
        names = imported_names(module.read_text(encoding="utf-8"))
        assert not any(is_inside(name, "datasets") for name in names)
        assert not any(is_inside(name, "huggingface_hub") for name in names)
        assert not any(is_inside(name, "httpx") for name in names)


def test_the_suite_sources_import_none_of_the_libraries_a_fetch_or_a_table_needs() -> (
    None
):
    imported = imported_names(
        (PACKAGE / "suite_sources.py").read_text(encoding="utf-8")
    )

    for library in ("polars", "datasets", "huggingface_hub", "httpx", "csv", "yaml"):
        assert not any(is_inside(name, library) for name in imported), library
    assert not any(is_inside(name, "rinchen.cli") for name in imported)
    assert not any(is_inside(name, "rinchen.export") for name in imported)
    assert not any(is_inside(name, "rinchen.acquire") for name in imported)


@pytest.mark.parametrize("path", MODULES, ids=lambda path: path.name)
def test_a_module_of_eval_imports_no_forbidden_layer_or_heavy_library(
    path: Path,
) -> None:
    imported = imported_names(path.read_text(encoding="utf-8"))

    assert forbidden_among(imported) == []


@pytest.mark.parametrize("path", COMMAND_MODULES, ids=lambda path: path.name)
def test_the_command_imports_no_layer_it_has_no_need_of_and_no_heavy_library(
    path: Path,
) -> None:
    imported = imported_names(path.read_text(encoding="utf-8"))

    assert (
        sorted(
            module
            for module in imported
            if any(is_inside(module, layer) for layer in COMMAND_FORBIDDEN_LAYERS)
            or any(is_inside(module, library) for library in FORBIDDEN_LIBRARIES)
        )
        == []
    )


def test_the_command_reaches_the_engine_and_the_converter_through_the_modules_that_load_them() -> (
    None
):
    imported = imported_names(COMMAND_MODULES[0].read_text(encoding="utf-8"))

    assert {
        "rinchen.infer.engine.Engine",
        "rinchen.script_conv.runtime.ScriptConverter",
        "rinchen.export.verified.verify_export",
        "rinchen.export.verified.verified_content",
    } <= imported


def test_the_scan_covers_the_run_command_and_its_systems() -> None:
    for name in ("eval_run", "eval_run_inputs", "eval_run_output", "eval_run_plan"):
        assert CLI / f"{name}.py" in COMMAND_MODULES
    assert CLI / "eval_systems.py" in SYNTH_COMMAND_MODULES
    assert CLI / "eval_systems.py" not in COMMAND_MODULES


@pytest.mark.parametrize("path", SYNTH_COMMAND_MODULES, ids=lambda path: path.name)
def test_the_systems_module_imports_synth_only_for_its_tables_and_no_other_forbidden_layer(
    path: Path,
) -> None:
    imported = imported_names(path.read_text(encoding="utf-8"))

    other_layers = tuple(
        layer for layer in COMMAND_FORBIDDEN_LAYERS if layer != "rinchen.synth"
    )
    assert (
        sorted(
            module
            for module in imported
            if any(is_inside(module, layer) for layer in other_layers)
            or any(is_inside(module, library) for library in FORBIDDEN_LIBRARIES)
        )
        == []
    )
    synth = {module for module in imported if is_inside(module, "rinchen.synth")}
    assert synth
    assert all(
        any(is_inside(module, allowed) for allowed in SYNTH_MODULES_ALLOWED)
        for module in synth
    ), sorted(synth)
    assert "rinchen.cli.synth.build_translator" in imported
    assert "rinchen.infer.engine.Engine" in imported


def test_no_command_module_but_the_systems_one_reaches_the_generators() -> None:
    for module in sorted(CLI.glob("eval_run*.py")):
        imported = imported_names(module.read_text(encoding="utf-8"))
        assert not any(is_inside(name, "rinchen.synth") for name in imported), (
            module.name
        )
        assert not any(is_inside(name, "rinchen.infer.engine") for name in imported), (
            module.name
        )


def test_the_run_commands_import_no_dataset_hub_or_http_library() -> None:
    for module in (*sorted(CLI.glob("eval_run*.py")), CLI / "eval_systems.py"):
        names = imported_names(module.read_text(encoding="utf-8"))
        for library in ("datasets", "huggingface_hub", "httpx", "polars"):
            assert not any(is_inside(name, library) for name in names), (
                module.name,
                library,
            )


@pytest.mark.parametrize(
    "path",
    [*MODULES, *COMMAND_MODULES, *SYNTH_COMMAND_MODULES, *ROBUSTNESS_TESTS],
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
