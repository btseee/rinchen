"""The scan the D11 gate uses to show that no code asks a device question
(`tests/support/d11_gate.forbidden_gpu_findings`) finds what it looks for, so that a gate that
reports nothing means the files are clean and not that the scan is blind."""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.support.d11_gate import forbidden_gpu_findings

# Spelled in parts so that this file does not hold what the scan looks for.
_COUNT = "get_cuda" + "_device_count"
_TOOL = "nvidia" + "-smi"


def _scanned(tmp_path: Path, source: str) -> list[str]:
    path = tmp_path / "probe.py"
    path.write_text(source, encoding="utf-8")
    return forbidden_gpu_findings(path)


@pytest.mark.parametrize(
    "source",
    [
        "import torch\nready = torch.cuda.is_available()\n",
        f"import ctranslate2\ncount = ctranslate2.{_COUNT}()\n",
        f"from ctranslate2 import {_COUNT}\n",
        f"count = {_COUNT}()\n",
        f'command = ["{_TOOL}", "-L"]\n',
        f'message = "run {_TOOL} first"\n',
    ],
    ids=[
        "torch-cuda-namespace",
        "count-as-attribute",
        "count-as-import",
        "count-as-name",
        "tool-as-string",
        "tool-inside-a-string",
    ],
)
def test_the_scan_finds_a_device_question(tmp_path: Path, source: str) -> None:
    assert len(_scanned(tmp_path, source)) >= 1


@pytest.mark.parametrize(
    "source",
    [
        "import torch\nmodel = torch.nn.Linear(1, 1)\n",
        'devices = {"CUDA_VISIBLE_DEVICES": ""}\nname = "cuda"\n',
        "import ctranslate2\nct2 = ctranslate2.Translator('m', device='cpu')\n",
    ],
    ids=["torch-without-cuda", "the-word-cuda", "a-translator-on-cpu"],
)
def test_the_scan_leaves_alone_what_asks_nothing(tmp_path: Path, source: str) -> None:
    assert _scanned(tmp_path, source) == []


def test_the_scan_names_the_line(tmp_path: Path) -> None:
    findings = _scanned(tmp_path, f"x = 1\ncount = {_COUNT}()\n")

    assert findings == [f"line 2: name {_COUNT}"]
