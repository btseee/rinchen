"""The refusals and the hygiene of the D12 robustness gate (R9 and R10 of
`tests/integration/test_robustness_gate.py`, whose docstring holds the gates and the figures
measured). Everything that runs goes through the real `rinchen` binary on the CPU with the devices
hidden (`CUDA_VISIBLE_DEVICES` is the empty string for the session); the model is the toy
teacher's export of the session fixtures.

    R9  refusals through the real binary: an export with a flipped byte exits 2, a suite with an
        unknown key exits 3, `--device auto` exits 2, each with nothing written and nothing on
        standard output
    R10 no GPU: an AST scan of the code and tests of D12 for the device questions (the `torch`
        CUDA namespace, the library call that counts CUDA devices, the vendor's query tool), no
        process of the gate's own, every command bounded by 900 seconds, and a command started in
        a process that does not hide the devices is refused before it starts
"""

from __future__ import annotations

import ast
import os
from pathlib import Path
from typing import Final

import pytest

from rinchen.config.loader import REPO_ROOT
from tests.support.d11_gate import forbidden_gpu_findings
from tests.support.distill_gate import SUBPROCESS_TIMEOUT_SECONDS
from tests.support.robust_gate import (
    Suites,
    copy_export_with_a_flipped_byte,
    run_robustness,
    suite_with_an_unknown_key,
)

pytestmark = [
    pytest.mark.subprocess,
    pytest.mark.usefixtures("d11_gpu_hidden"),
]


@pytest.fixture(scope="module")
def runs(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """The directory the runs of this module write into (one subdirectory each)."""
    return tmp_path_factory.mktemp("d12_refusals")


# =================================================================================
# R9: refusals through the real binary.


def test_an_export_with_a_flipped_byte_is_refused_and_nothing_is_written(
    d12_robust_export: Path, d12_suites: Suites, runs: Path
) -> None:
    damaged = runs / "damaged_export"
    copy_export_with_a_flipped_byte(d12_robust_export, damaged)
    out_dir = runs / "damaged_export_out"
    invocation = run_robustness(
        "--suite", d12_suites.main, "--model", damaged, "--out-dir", out_dir
    )

    assert invocation.returncode == 2, invocation.stderr
    assert invocation.stdout == []
    assert not out_dir.exists()


def test_a_suite_with_an_unknown_key_is_refused_and_nothing_is_written(
    d12_robust_export: Path, d12_suites: Suites, runs: Path
) -> None:
    suite = runs / "unknown_key_suite.json"
    suite_with_an_unknown_key(d12_suites.main, suite)
    out_dir = runs / "unknown_key_out"
    invocation = run_robustness(
        "--suite", suite, "--model", d12_robust_export, "--out-dir", out_dir
    )

    assert invocation.returncode == 3, invocation.stderr
    assert invocation.stdout == []
    # The message names the place and never the key or the text.
    assert "not part of the format" in invocation.stderr
    assert "unknown_key" not in invocation.stderr
    assert not out_dir.exists()


def test_the_device_auto_is_refused_and_nothing_is_written(
    d12_robust_export: Path, d12_suites: Suites, runs: Path
) -> None:
    out_dir = runs / "auto_out"
    invocation = run_robustness(
        "--suite",
        d12_suites.main,
        "--model",
        d12_robust_export,
        "--out-dir",
        out_dir,
        "--device",
        "auto",
    )

    assert invocation.returncode == 2, invocation.stderr
    assert invocation.stdout == []
    assert not out_dir.exists()


# =================================================================================
# R10: no GPU, bounded subprocesses.

_D12_PYTHON_FILES: Final = (
    *sorted((REPO_ROOT / "src" / "rinchen" / "eval").glob("robust*.py")),
    *sorted((REPO_ROOT / "src" / "rinchen" / "cli").glob("eval_robustness*.py")),
    *sorted((REPO_ROOT / "tests" / "support").glob("robust_*.py")),
    *sorted((REPO_ROOT / "tests" / "unit").glob("test_*robust*.py")),
    REPO_ROOT / "tests" / "integration" / "test_robustness_gate.py",
    REPO_ROOT / "tests" / "integration" / "test_robustness_gate_refusals.py",
)
_NEW_FILES: Final = (
    REPO_ROOT / "tests" / "support" / "robust_gate.py",
    REPO_ROOT / "tests" / "integration" / "test_robustness_gate.py",
    REPO_ROOT / "tests" / "integration" / "test_robustness_gate_refusals.py",
)


def test_no_d12_file_asks_a_device_question() -> None:
    assert len(_D12_PYTHON_FILES) >= 40
    for path in _D12_PYTHON_FILES:
        assert forbidden_gpu_findings(path) == [], path


def test_the_gate_starts_no_process_of_its_own_and_the_ones_it_runs_are_bounded() -> (
    None
):
    assert SUBPROCESS_TIMEOUT_SECONDS <= 900
    for path in _NEW_FILES:
        tree = ast.parse(path.read_text("utf-8"))
        imported = {
            alias.name.split(".")[0]
            for node in ast.walk(tree)
            if isinstance(node, ast.Import)
            for alias in node.names
        } | {
            (node.module or "").split(".")[0]
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom)
        }
        assert "subprocess" not in imported, path
        assert "ctranslate2" not in imported, path
    # The one place a command is started bounds it.
    runner = ast.parse(
        (REPO_ROOT / "tests" / "support" / "distill_gate.py").read_text("utf-8")
    )
    runs = [
        node
        for node in ast.walk(runner)
        if isinstance(node, ast.Call) and ast.unparse(node.func) == "subprocess.run"
    ]
    assert len(runs) == 1
    assert [
        ast.unparse(keyword.value)
        for keyword in runs[0].keywords
        if keyword.arg == "timeout"
    ] == ["SUBPROCESS_TIMEOUT_SECONDS"]


def test_every_command_runs_in_a_process_that_hides_the_devices(
    d12_robust_export: Path, d12_suites: Suites, runs: Path
) -> None:
    assert os.environ["CUDA_VISIBLE_DEVICES"] == ""
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("CUDA_VISIBLE_DEVICES", "0")
        with pytest.raises(AssertionError, match="CUDA_VISIBLE_DEVICES"):
            run_robustness(
                "--suite",
                d12_suites.main,
                "--model",
                d12_robust_export,
                "--out-dir",
                runs / "must_not_start",
            )
    assert not (runs / "must_not_start").exists()
