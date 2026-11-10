"""What the bootstrap must never depend on (D13, T9): the project's other layers, a hidden seed,
numpy's generator, sacrebleu's own significance test, the hash seed and the thread count.

The modules import only numpy and each other and the errors; no source of them reads the
environment or touches a random generator (an AST scan, itself tested on sources that do); a draw
does not move when an environment variable of a seed is set; and fresh interpreters with
different `PYTHONHASHSEED` and thread settings give byte-identical results.
"""

from __future__ import annotations

import ast
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from rinchen.eval.bootstrap import BootstrapSettings, resample_totals
from tests.support.eval_golden import (
    MEAN_BETTER_FRACTION,
    MEAN_INTERVAL_A,
    MEAN_MICRO_A,
    MEAN_P_VALUE,
    MEAN_POINT_A,
)
from tests.support.module_imports import (
    SOURCE_ROOT,
    loaded_by_importing,
    project_imports,
)
from tests.support.no_network import no_network  # noqa: F401 - the fixture of the mark

pytestmark = pytest.mark.usefixtures("no_network")

SOURCES = (
    SOURCE_ROOT / "rinchen" / "eval" / "bootstrap.py",
    SOURCE_ROOT / "rinchen" / "eval" / "significance.py",
)


def mean_rows(micro: tuple[int, ...]) -> np.ndarray:
    return np.array([[score, 1] for score in micro], dtype=np.int64)


# ---- what the modules import --------------------------------------------------------------


def test_the_bootstrap_imports_nothing_of_the_project_but_the_errors() -> None:
    assert project_imports(SOURCES[0]) == {"rinchen.errors"}


def test_the_comparison_imports_the_bootstrap_and_no_metric() -> None:
    assert project_imports(SOURCES[1]) == {"rinchen.eval.bootstrap"}


def test_importing_the_two_modules_loads_no_metric_library_and_no_heavy_one() -> None:
    watched = (
        "sacrebleu",
        "pydantic",
        "yaml",
        "polars",
        "datasets",
        "huggingface_hub",
        "ctranslate2",
        "sentencepiece",
        "torch",
        "rinchen.metrics",
        "rinchen.config",
        "rinchen.cli",
        "rinchen.train",
    )

    assert (
        loaded_by_importing(
            ["rinchen.eval.bootstrap", "rinchen.eval.significance"], watched
        )
        == []
    )


# ---- nothing hidden, nothing numpy's -------------------------------------------------------

FORBIDDEN_MODULES = ("random", "secrets", "os", "numpy.random", "sacrebleu")
FORBIDDEN_NAMES = (
    "environ",
    "getenv",
    "random",
    "default_rng",
    "RandomState",
    "Generator",
    "SeedSequence",
)


def is_forbidden_module(name: str) -> bool:
    return any(name == m or name.startswith(f"{m}.") for m in FORBIDDEN_MODULES)


def forbidden_in(source: str) -> list[str]:
    """What a source imports or touches that could hide a seed, draw from numpy's generator or
    call sacrebleu's own significance test."""
    found: set[str] = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            found.update(a.name for a in node.names if is_forbidden_module(a.name))
        elif isinstance(node, ast.ImportFrom):
            names = [
                node.module or "",
                *(f"{node.module}.{a.name}" for a in node.names),
            ]
            found.update(name for name in names if is_forbidden_module(name))
        elif isinstance(node, ast.Attribute) and node.attr in FORBIDDEN_NAMES:
            found.add(node.attr)
        elif isinstance(node, ast.Name) and node.id in FORBIDDEN_NAMES:
            found.add(node.id)
    return sorted(found)


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("import random\n", ["random"]),
        ("import secrets\n", ["secrets"]),
        ("import numpy.random\n", ["numpy.random"]),
        ("from numpy import random\n", ["numpy.random"]),
        (
            "from numpy.random import default_rng\n",
            ["numpy.random", "numpy.random.default_rng"],
        ),
        (
            "import numpy as np\nrng = np.random.default_rng(1)\n",
            ["default_rng", "random"],
        ),
        ("import os\nseed = os.environ['SEED']\n", ["environ", "os"]),
        ("from os import environ\n", ["os", "os.environ"]),
        ("import os\nseed = os.getenv('SEED')\n", ["getenv", "os"]),
        (
            "from sacrebleu.significance import PairedTest\n",
            ["sacrebleu.significance", "sacrebleu.significance.PairedTest"],
        ),
        ("import sacrebleu.significance\n", ["sacrebleu.significance"]),
        ("import numpy as np\nx = np.arange(3)\n", []),
    ],
)
def test_the_scan_for_a_hidden_seed_finds_every_way_in(
    source: str, expected: list[str]
) -> None:
    assert forbidden_in(source) == expected


@pytest.mark.parametrize("path", SOURCES, ids=lambda path: path.name)
def test_the_bootstrap_reads_no_environment_and_uses_neither_numpy_s_generator_nor_sacrebleu(
    path: Path,
) -> None:
    assert forbidden_in(path.read_text(encoding="utf-8")) == []


@pytest.mark.parametrize(
    "name", ["SACREBLEU_SEED", "RINCHEN_SEED", "BOOTSTRAP_SEED", "SEED", "NUMPY_SEED"]
)
def test_no_environment_variable_moves_a_draw(
    monkeypatch: pytest.MonkeyPatch, name: str
) -> None:
    before = resample_totals(mean_rows(MEAN_MICRO_A), BootstrapSettings()).tolist()
    monkeypatch.setenv(name, "7")

    after = resample_totals(mean_rows(MEAN_MICRO_A), BootstrapSettings()).tolist()

    assert after == before


# ---- determinism: the hash seed and the thread count ----------------------------------------

CHILD = """
import hashlib, json, sys
sys.path.insert(0, sys.argv[1])
import numpy as np
from rinchen.eval.bootstrap import BootstrapSettings, group_rows, resample_totals
from rinchen.eval.significance import compare, estimate

micro_a = [912000, 850000, 770000, 990000, 640000, 880000, 930000, 720000, 810000, 960000, 700000, 870000]
micro_b = [900000, 860000, 700000, 980000, 600000, 870000, 940000, 650000, 800000, 950000, 720000, 840000]
settings = BootstrapSettings()
rows_a = np.array([[s, 1] for s in micro_a], dtype=np.int64)
rows_b = np.array([[s, 1] for s in micro_b], dtype=np.int64)
totals = resample_totals(rows_a, settings)
mean = lambda t: t[0] / (t[1] * 1_000_000)
grouped, labels = group_rows([[1, 2], [3, 4], [5, 6], [7, 8]], ("z", "a", "z", "m"))
est = estimate(rows_a, mean, settings)
cmp_ = compare(rows_a, rows_b, mean, settings)
big = (np.arange(3000 * 6, dtype=np.int64).reshape(3000, 6) * 31) % 211
print(json.dumps({
    "sha": hashlib.sha256(totals.tobytes()).hexdigest(),
    "big": hashlib.sha256(resample_totals(big, settings).tobytes()).hexdigest(),
    "labels": labels,
    "grouped": grouped.tolist(),
    "estimate": [est.value, est.interval, est.units],
    "compare": [cmp_.delta, cmp_.interval, cmp_.p_value, cmp_.better_fraction, cmp_.units],
}))
"""

THREAD_VARIABLES = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)


def run_child(tmp_path: Path, hash_seed: str, threads: str) -> str:
    environment = {
        **os.environ,
        "PYTHONHASHSEED": hash_seed,
        "CUDA_VISIBLE_DEVICES": "",
        **dict.fromkeys(THREAD_VARIABLES, threads),
    }
    result = subprocess.run(
        [sys.executable, "-c", CHILD, str(SOURCE_ROOT)],
        capture_output=True,
        text=True,
        env=environment,
        cwd=tmp_path,
        timeout=300,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    return result.stdout


def test_the_results_are_byte_identical_across_hash_seeds_and_thread_counts(
    tmp_path: Path,
) -> None:
    runs = {
        (hash_seed, threads): run_child(tmp_path, hash_seed, threads)
        for hash_seed, threads in (("0", "1"), ("1", "4"), ("random", "16"))
    }

    outputs = set(runs.values())
    assert len(outputs) == 1
    document = json.loads(outputs.pop())
    totals = resample_totals(mean_rows(MEAN_MICRO_A), BootstrapSettings())
    assert document["sha"] == hashlib.sha256(totals.tobytes()).hexdigest()
    assert document["labels"] == ["z", "a", "m"]
    assert document["grouped"] == [[6, 8], [3, 4], [7, 8]]
    assert document["estimate"] == [MEAN_POINT_A, list(MEAN_INTERVAL_A), 12]
    assert document["compare"][2:] == [MEAN_P_VALUE, MEAN_BETTER_FRACTION, 12]
