"""The `rinchen convert-script` CLI over the tiny converter: the STOP gate of D5.

The real `rinchen` command exports the checkpoint of the session's tiny run (the
fixtures of `conftest.py`, so nothing is trained again), then converts a line of
fixture words to the traditional script with `apply` and scores the model with `eval`.
The reference is `ScriptConverter` over the session's own export of the same
checkpoint. Each command is a subprocess with its output kept as bytes, so the
UTF-8 that reaches a pipe is what the test reads.
"""

from __future__ import annotations

import hashlib
import re
import subprocess
import sys
from pathlib import Path
from typing import Literal

import pytest

from rinchen.manifest import store
from rinchen.script_conv.alphabet import NNBSP
from rinchen.script_conv.eval import score_conversions
from rinchen.script_conv.runtime import ScriptConverter
from tests.support.tiny_converter import COMMAND_TIMEOUT_SECONDS, TinyRun

MAX_CER = 0.15
EVAL_PAIRS = 120
RINCHEN = str(Path(sys.executable).with_name("rinchen"))
SUMMARY = re.compile(r"rule_fallbacks=(\d+) invalid_outputs=(\d+)")


def _rinchen(
    *args: str, input: bytes | None = None
) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(
        [RINCHEN, "convert-script", *args],
        input=input,
        capture_output=True,
        timeout=COMMAND_TIMEOUT_SECONDS,
        check=False,
    )


def _ok(
    completed: subprocess.CompletedProcess[bytes],
) -> subprocess.CompletedProcess[bytes]:
    assert completed.returncode == 0, completed.stderr.decode("utf-8", "replace")
    return completed


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture(scope="module")
def model_dir(tiny_run: TinyRun, tmp_path_factory: pytest.TempPathFactory) -> Path:
    """The model directory the CLI's own `export` made from the tiny checkpoint."""
    out_dir = tmp_path_factory.mktemp("cli_export") / "model"
    _ok(
        _rinchen(
            "export",
            "--run-dir",
            str(tiny_run.run_dir),
            "--config",
            str(tiny_run.config_path),
            "--out-dir",
            str(out_dir),
            "--timeout",
            str(COMMAND_TIMEOUT_SECONDS),
        )
    )
    return out_dir


@pytest.fixture
def reference(tiny_run: TinyRun, tiny_ct2: Path) -> ScriptConverter:
    return ScriptConverter(tiny_ct2, tiny_run.spm_model)


@pytest.mark.subprocess
def test_export_leaves_a_self_contained_model_directory(
    tiny_run: TinyRun, model_dir: Path
) -> None:
    assert (model_dir / "model.bin").is_file()
    assert (model_dir / "converter_char.model").read_bytes() == (
        tiny_run.spm_model.read_bytes()
    )
    assert list(model_dir.rglob("*.part")) == []


@pytest.mark.subprocess
def test_the_export_manifest_hashes_match_the_files(
    tiny_run: TinyRun, model_dir: Path
) -> None:
    manifest = store.read(model_dir)
    assert manifest is not None

    assert manifest.stage == "script_conv_export"
    checkpoint = tiny_run.ckpt_dir / "model.00.safetensors"
    assert [(item.path, item.sha256) for item in manifest.inputs] == [
        (str(path), _sha256(path)) for path in (checkpoint, tiny_run.spm_model)
    ]
    listed = {Path(item.path): item.sha256 for item in manifest.outputs}
    on_disk = {
        path
        for path in model_dir.rglob("*")
        if path.is_file() and path.name != "manifest.json"
    }
    assert set(listed) == on_disk
    assert {path: _sha256(path) for path in on_disk} == listed
    assert listed[model_dir / "converter_char.model"] == _sha256(tiny_run.spm_model)
    assert manifest.config_sha256 == _sha256(tiny_run.config_path)


@pytest.mark.subprocess
def test_apply_converts_a_line_of_fixture_words_to_the_traditional_script(
    tiny_run: TinyRun, model_dir: Path, reference: ScriptConverter
) -> None:
    sources = tiny_run.lines("cyrl2mong.train.src")
    targets = tiny_run.lines("cyrl2mong.train.tgt")
    with_gap = [
        source
        for source, target in zip(sources, targets, strict=True)
        if NNBSP in target
    ]
    assert with_gap, "the fixture has no word whose script spelling holds an NNBSP"
    line = " ".join([*sources[:5], *with_gap[:2]]) + "\n"
    expected = reference.convert(line, "Mong")
    assert expected.text != line

    completed = _ok(
        _rinchen(
            "apply",
            "--model-dir",
            str(model_dir),
            "--target",
            "Mong",
            input=line.encode(),
        )
    )

    assert completed.stdout.decode("utf-8") == expected.text
    summary = SUMMARY.fullmatch(completed.stderr.decode("utf-8").splitlines()[-1])
    assert summary is not None
    assert (int(summary[1]), int(summary[2])) == (
        expected.rule_fallbacks,
        expected.invalid_outputs,
    )


@pytest.mark.subprocess
def test_apply_reads_a_file_and_converts_to_cyrillic(
    tiny_run: TinyRun, model_dir: Path, reference: ScriptConverter, tmp_path: Path
) -> None:
    script_words = tiny_run.lines("mong2cyrl.train.src")[:6]
    text = " ".join(script_words) + "\n"
    source = tmp_path / "script.txt"
    source.write_bytes(text.encode("utf-8"))
    expected = reference.convert(text, "Cyrl")
    assert expected.text != text

    completed = _ok(
        _rinchen(
            "apply", "--model-dir", str(model_dir), "--target", "Cyrl", str(source)
        )
    )

    assert completed.stdout.decode("utf-8") == expected.text


@pytest.mark.subprocess
@pytest.mark.parametrize("target", ["Mong", "Cyrl"])
def test_eval_scores_the_tiny_model_under_the_character_error_gate(
    tiny_run: TinyRun,
    model_dir: Path,
    reference: ScriptConverter,
    tmp_path: Path,
    target: Literal["Mong", "Cyrl"],
) -> None:
    cyrillic = tiny_run.lines("cyrl2mong.train.src")[:EVAL_PAIRS]
    script = tiny_run.lines("cyrl2mong.train.tgt")[:EVAL_PAIRS]
    pairs = tmp_path / "pairs.tsv"
    pairs.write_text(
        "cyrillic\tscript\n"
        + "".join(f"{c}\t{s}\n" for c, s in zip(cyrillic, script, strict=True)),
        encoding="utf-8",
    )
    sources, references = (cyrillic, script) if target == "Mong" else (script, cyrillic)
    expected = score_conversions(
        references, [reference.convert(word, target).text for word in sources]
    )

    completed = _ok(
        _rinchen(
            "eval",
            "--model-dir",
            str(model_dir),
            "--pairs",
            str(pairs),
            "--target",
            target,
        )
    )

    lines = completed.stdout.decode("utf-8").splitlines()
    assert lines == [
        f"cer {expected.cer:.4f}",
        f"word_accuracy {expected.word_accuracy:.4f}",
        f"n {EVAL_PAIRS}",
    ]
    assert expected.cer <= MAX_CER
