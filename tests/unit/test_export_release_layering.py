"""The layering of the model card and the release stage (D13, Task 9, spec T22), checked by reading the
source.

`rinchen.export.model_card` (and the modules that render its sections) and `rinchen.export.release`
read a report through `rinchen.eval.report_read` and an export through `rinchen.export.record` and
`rinchen.export.verified`. They never import a command (`rinchen.cli`), the suite readers
(`rinchen.eval.suites`, `suite_rules`, `suite_sources`: a card never needs a benchmark's text, and
only `rinchen.eval` and the `eval*` commands may read one), the training, synthetic-data, serving,
annotation, filter or script-conversion layers, the engine (`rinchen.infer`) or a library that loads
a model, a dataset or the network. In the other direction `rinchen.eval` never imports the card, the
release module or the Hub plan: the packages stay acyclic.

The command `rinchen release stage` (`cli/release*.py`) sits above both: it may use the export
reader, the manifest and the helpers of the other commands, never the layers the other commands may
not use, and it never asks about the GPU.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

import rinchen.eval
import rinchen.export
from tests.unit.test_eval_layering import (
    FORBIDDEN_LIBRARIES,
    gpu_queries,
    imported_names,
    is_inside,
)

EVAL = Path(rinchen.eval.__file__).parent
EXPORT = Path(rinchen.export.__file__).parent
CLI = EXPORT.parent / "cli"
CARD_MODULES = [*sorted(EXPORT.glob("model_card*.py")), EXPORT / "release.py"]
STAGE_COMMANDS = sorted(CLI.glob("release*.py"))
# The layers and libraries the card and the stage never reach.
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
    "huggingface_hub",
    "httpx",
    "polars",
    *FORBIDDEN_LIBRARIES,
)
# What the command may not import (the layers the `eval` commands may not use either).
COMMAND_NEVER = (
    "rinchen.train",
    "rinchen.synth",
    "rinchen.serve",
    "rinchen.annotate",
    "rinchen.filter",
    "rinchen.eval.suites",
    "rinchen.eval.suite_rules",
    "rinchen.eval.suite_sources",
    "rinchen.infer",
    "datasets",
    "huggingface_hub",
    "httpx",
    "polars",
    *FORBIDDEN_LIBRARIES,
)
ALLOWED_EVAL_MODULES = (
    "rinchen.eval.report_read",
    "rinchen.eval.report_view",
    "rinchen.eval.report_format",
    "rinchen.eval.report_markdown",
    "rinchen.eval.report_markdown_cells",
    "rinchen.eval.result",
    "rinchen.eval.robustness_input",
    "rinchen.eval.robust_suite_rules",
)
BACK_EDGES = ("model_card", "release", "hub")


def reached(source: str, forbidden: tuple[str, ...]) -> list[str]:
    return sorted(
        module
        for module in imported_names(source)
        if any(is_inside(module, layer) for layer in forbidden)
    )


def eval_importing_the_release_side(source: str) -> list[str]:
    """What a source imports of `rinchen.export.model_card`, `release` or `hub` (and of the modules
    that render the card's sections)."""
    return sorted(
        module
        for module in imported_names(source)
        if any(
            is_inside(module, f"rinchen.export.{name}")
            or module.startswith(f"rinchen.export.{name}_")
            for name in BACK_EDGES
        )
    )


# ---- the scan -------------------------------------------------------------------------------------


def test_the_scan_finds_what_it_is_told_to_forbid() -> None:
    source = (
        "import rinchen.cli.main\n"
        "from rinchen.eval.suites import Suite\n"
        "from rinchen.eval import suite_rules\n"
        "def late():\n"
        "    from rinchen.train import eole_cfg\n"
        "    import huggingface_hub, sentencepiece\n"
        "from rinchen.eval.report_read import report_from_bytes\n"
    )

    assert reached(source, NEVER) == [
        "huggingface_hub",
        "rinchen.cli.main",
        "rinchen.eval.suite_rules",
        "rinchen.eval.suites",
        "rinchen.eval.suites.Suite",
        "rinchen.train",
        "rinchen.train.eole_cfg",
        "sentencepiece",
    ]


def test_the_scan_of_the_back_edges_finds_the_card_the_release_and_the_hub() -> None:
    source = (
        "import rinchen.export.model_card\n"
        "from rinchen.export import release\n"
        "from rinchen.export.hub import plan_upload\n"
        "from rinchen.export.model_card_evaluation import tables\n"
        "from rinchen.export.record import ModelRecord\n"
    )

    assert eval_importing_the_release_side(source) == [
        "rinchen.export.hub",
        "rinchen.export.hub.plan_upload",
        "rinchen.export.model_card",
        "rinchen.export.model_card_evaluation",
        "rinchen.export.model_card_evaluation.tables",
        "rinchen.export.release",
    ]


def test_the_scan_covers_the_card_the_release_module_and_the_command() -> None:
    assert EXPORT / "model_card.py" in CARD_MODULES
    assert EXPORT / "release.py" in CARD_MODULES
    assert {path.name for path in STAGE_COMMANDS} >= {
        "release.py",
        "release_stage.py",
        "release_stage_inputs.py",
        "release_stage_output.py",
    }


# ---- the card and the release module ----------------------------------------------------------------


@pytest.mark.parametrize("path", CARD_MODULES, ids=lambda path: path.name)
def test_a_module_of_the_card_or_the_release_imports_none_of_the_layers_it_must_not_reach(
    path: Path,
) -> None:
    assert reached(path.read_text(encoding="utf-8"), NEVER) == []


@pytest.mark.parametrize("path", CARD_MODULES, ids=lambda path: path.name)
def test_a_module_of_the_card_or_the_release_reaches_eval_only_through_the_report_modules(
    path: Path,
) -> None:
    imported = imported_names(path.read_text(encoding="utf-8"))
    eval_modules = sorted(
        module
        for module in imported
        if is_inside(module, "rinchen.eval")
        and not any(is_inside(module, allowed) for allowed in ALLOWED_EVAL_MODULES)
        and module != "rinchen.eval"
    )

    assert eval_modules == []


def test_the_card_and_the_release_module_read_a_report_through_the_strict_reader() -> (
    None
):
    for name in ("model_card.py", "release.py"):
        imported = imported_names((EXPORT / name).read_text(encoding="utf-8"))
        assert "rinchen.eval.report_read" in imported, name
    imported = imported_names((EXPORT / "release.py").read_text(encoding="utf-8"))
    assert "rinchen.export.verified" in imported
    imported = imported_names((EXPORT / "model_card.py").read_text(encoding="utf-8"))
    assert "rinchen.export.record" in imported


def test_the_release_module_holds_no_rule_of_the_card() -> None:
    imported = imported_names((EXPORT / "release.py").read_text(encoding="utf-8"))

    assert not any(
        is_inside(name, "rinchen.export.model_card_evaluation") for name in imported
    )


# ---- the back edge ---------------------------------------------------------------------------------


def test_no_module_of_eval_imports_the_card_the_release_module_or_the_hub_plan() -> (
    None
):
    found = {
        path.name: eval_importing_the_release_side(path.read_text(encoding="utf-8"))
        for path in sorted(EVAL.glob("*.py"))
    }

    assert {name: hits for name, hits in found.items() if hits} == {}


def test_no_module_of_the_export_imports_a_command() -> None:
    for path in sorted(EXPORT.glob("*.py")):
        imported = imported_names(path.read_text(encoding="utf-8"))
        assert not any(is_inside(name, "rinchen.cli") for name in imported), path.name


# ---- the command ------------------------------------------------------------------------------------


@pytest.mark.parametrize("path", STAGE_COMMANDS, ids=lambda path: path.name)
def test_the_release_commands_import_no_layer_and_no_library_they_have_no_need_of(
    path: Path,
) -> None:
    assert reached(path.read_text(encoding="utf-8"), COMMAND_NEVER) == []


# ---- no GPU --------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "path",
    [*CARD_MODULES, *STAGE_COMMANDS],
    ids=lambda path: path.name,
)
def test_no_module_of_the_card_the_release_or_the_stage_asks_about_the_gpu(
    path: Path,
) -> None:
    assert gpu_queries(path.read_text(encoding="utf-8")) == []


# ---- importing them is cheap ----------------------------------------------------------------------------


def test_importing_the_card_and_the_release_module_loads_no_model_library_dataset_or_hub_client() -> (
    None
):
    source_root = str(EXPORT.parent.parent)
    code = (
        "import sys\n"
        "sys.path.insert(0, sys.argv[1])\n"
        "import rinchen.export.model_card, rinchen.export.release\n"
        "assert rinchen.__file__.startswith(sys.argv[1]), rinchen.__file__\n"
        "watched = ('ctranslate2', 'sentencepiece', 'eole', 'torch', 'polars', 'datasets',\n"
        "           'huggingface_hub', 'httpx', 'rinchen.cli')\n"
        "loaded = sorted(m for m in sys.modules\n"
        "                if any(m == w or m.startswith(w + '.') for w in watched))\n"
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
