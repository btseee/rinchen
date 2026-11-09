"""Helpers of the D10 STOP gate (`tests/integration/test_distill_tiny_student.py`): running
the real `rinchen` binary, reading what it wrote, damaging a teacher export so that exactly
one check can refuse it, and the in-process CTranslate2 smoke translation.

Nothing here starts a GPU: the module that uses it hides every device with
`CUDA_VISIBLE_DEVICES=""` and the CTranslate2 translator is built on `device="cpu"`.
"""

from __future__ import annotations

import ast
import hashlib
import json
import subprocess
import sys
import time
import unicodedata
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

from rinchen.manifest import store

# A hung Eole or converter must fail the test: the longest command, the 2,000-step student,
# takes about two minutes on a quiet machine.
SUBPROCESS_TIMEOUT_SECONDS: Final = 900
# Handed to `rinchen train launch --timeout`: it bounds each Eole command inside.
ENGINE_TIMEOUT_SECONDS: Final = 600
RINCHEN: Final = str(Path(sys.executable).with_name("rinchen"))
THREADS: Final = "2"
# Where the CTranslate2 smoke translation stops: the length Eole's prediction is held to.
GENERATION_LIMIT: Final = 128
# Eole's own cap on the length of an output (twice the longest source of a batch plus 5) is
# off, so that no output is cut before the model ends it.
NO_RATIO_CAP: Final = 0


@dataclass(frozen=True)
class Invocation:
    """One `rinchen` command: what it printed and how long it took."""

    returncode: int
    stdout: list[str]
    stderr: str
    seconds: float

    @property
    def facts(self) -> dict[str, str]:
        """The `key: value` lines a command prints (the last one wins for a repeated key)."""
        return dict(line.split(": ", 1) for line in self.stdout if ": " in line)


def run_rinchen(*args: str | Path) -> Invocation:
    """Run `rinchen <args>` to its end (at most `SUBPROCESS_TIMEOUT_SECONDS`)."""
    started = time.perf_counter()
    completed = subprocess.run(
        [RINCHEN, *map(str, args)],
        capture_output=True,
        timeout=SUBPROCESS_TIMEOUT_SECONDS,
        check=False,
    )
    return Invocation(
        returncode=completed.returncode,
        stdout=completed.stdout.decode("utf-8").splitlines(),
        stderr=completed.stderr.decode("utf-8", "replace"),
        seconds=time.perf_counter() - started,
    )


def succeeded(invocation: Invocation) -> Invocation:
    assert invocation.returncode == 0, invocation.stderr
    return invocation


def sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    document = json.loads(path.read_text("utf-8"))
    assert isinstance(document, dict), path
    return document


def replace_vocabulary_entry(export: Path, token: str, replacement: str) -> None:
    """Rename `token` to `replacement` in `shared_vocabulary.json` of the export at
    `export`, and make the export's manifest list the new bytes, so that every hash agrees
    and only a check of the vocabulary itself can refuse the export. The entry count stays
    what the record states."""
    vocabulary_file = export / "shared_vocabulary.json"
    entries = json.loads(vocabulary_file.read_text("utf-8"))
    assert isinstance(entries, list) and token in entries, token
    assert replacement not in entries, replacement
    entries[entries.index(token)] = replacement
    vocabulary_file.write_text(json.dumps(entries), encoding="utf-8")
    manifest = store.read(export)
    assert manifest is not None
    listed = [
        item
        for item in manifest.outputs
        if Path(item.path).name == vocabulary_file.name
    ]
    assert len(listed) == 1
    listed[0].sha256 = sha256_of(vocabulary_file)
    listed[0].bytes = vocabulary_file.stat().st_size
    store.write(export, manifest)


def ct2_greedy(export: Path, sources: Sequence[Sequence[str]]) -> list[list[str]]:
    """The pieces CTranslate2 writes for the token lines `sources` greedily, from the export
    at `export`, on the CPU with one thread. Uses nothing of the project but the files: the
    translator is built the way any user of the export would."""
    import ctranslate2  # imported here: only the tests that translate pay for it

    translator = ctranslate2.Translator(str(export), device="cpu", intra_threads=1)
    results = translator.translate_batch(
        [list(source) for source in sources],
        beam_size=1,
        max_decoding_length=GENERATION_LIMIT,
    )
    return [list(result.hypotheses[0]) for result in results]


def is_cyrillic_text(text: str) -> bool:
    """Whether `text` has letters and every one of them is a Cyrillic letter."""
    letters = [char for char in text if char.isalpha()]
    return bool(letters) and all(
        unicodedata.name(char, "").startswith("CYRILLIC") for char in letters
    )


@dataclass(frozen=True)
class SourceFindings:
    """What a Python file spells out: every string constant, every `name.attribute` pair
    and, for each call of `ctranslate2.Translator`, its `device` argument as source text
    (None when it has none)."""

    strings: list[str]
    dotted_names: list[str]
    translator_devices: list[str | None]


def source_findings(path: Path) -> SourceFindings:
    """Read the structure of the Python file `path` without importing it."""
    tree = ast.parse(path.read_text("utf-8"))
    devices: list[str | None] = []
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and ast.unparse(node.func) == "ctranslate2.Translator"
        ):
            devices.append(
                next(
                    (ast.unparse(k.value) for k in node.keywords if k.arg == "device"),
                    None,
                )
            )
    return SourceFindings(
        strings=[
            node.value
            for node in ast.walk(tree)
            if isinstance(node, ast.Constant) and isinstance(node.value, str)
        ],
        dotted_names=[
            f"{node.value.id}.{node.attr}"
            for node in ast.walk(tree)
            if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name)
        ],
        translator_devices=devices,
    )
