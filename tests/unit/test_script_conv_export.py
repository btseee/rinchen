"""Tests for `rinchen.script_conv.export`: the CTranslate2 export and the parity check.

The export tests replace `subprocess.run` by a recorder that writes what the real
converter would. The parity tests train a real character model on a few Latin words
(the model only has to hold the tags and some letters) and replace the CTranslate2
translator by a stub, so no model is loaded.
"""

from __future__ import annotations

import os
import subprocess
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import pytest

from rinchen.errors import UpstreamError
from rinchen.script_conv import export
from rinchen.script_conv.char_model import train_char_model
from rinchen.script_conv.export import (
    ParityReport,
    export_ct2,
    export_environment,
    parity,
)

TO_SCRIPT_TAG = "<2mn_Mong>"
DUMMY_PREFIX = chr(0x2581)


class FakeConverter:
    """Stands in for `subprocess.run` running the CTranslate2 converter: writes
    `log_lines` to the log handle, then `model_bytes` as `model.bin` in the
    `--output_dir` (nothing when `model_bytes` is None), and exits with `exit_code`."""

    def __init__(
        self,
        *,
        exit_code: int = 0,
        model_bytes: bytes | None = b"weights",
        log_lines: Sequence[str] = ("converting",),
    ) -> None:
        self.exit_code = exit_code
        self.model_bytes = model_bytes
        self.log_lines = log_lines
        self.calls: list[tuple[list[str], dict[str, Any]]] = []

    def __call__(
        self, argv: Sequence[str], **kwargs: Any
    ) -> subprocess.CompletedProcess[bytes]:
        self.calls.append((list(argv), kwargs))
        for line in self.log_lines:
            kwargs["stdout"].write(f"{line}\n".encode())
        output_dir = Path(argv[argv.index("--output_dir") + 1])
        output_dir.mkdir(parents=True, exist_ok=True)
        if self.model_bytes is not None:
            (output_dir / "model.bin").write_bytes(self.model_bytes)
        return subprocess.CompletedProcess(list(argv), self.exit_code)


def _install(monkeypatch: pytest.MonkeyPatch, fake: FakeConverter) -> FakeConverter:
    monkeypatch.setattr(subprocess, "run", fake)
    return fake


# ---- export_environment ---------------------------------------------------------


def test_export_environment_hides_the_gpu_even_when_the_parent_shows_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # CTranslate2's Eole converter segfaults and leaves an empty model.bin when a
    # CUDA device is visible.
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")

    assert export_environment()["CUDA_VISIBLE_DEVICES"] == ""


def test_export_environment_keeps_the_rest_of_the_environment_and_leaves_the_parent_alone(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")

    env = export_environment()

    assert env["PATH"] == os.environ["PATH"]
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "0"


# ---- export_ct2 -----------------------------------------------------------------


def test_export_ct2_runs_the_converter_with_the_isolated_interpreter(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    fake = _install(monkeypatch, FakeConverter())
    ckpt_dir = tmp_path / "run" / "ckpt"
    out_dir = tmp_path / "ct2"

    assert export_ct2(ckpt_dir, out_dir) == out_dir

    ((argv, _),) = fake.calls
    assert argv == [
        sys.executable,
        "-I",
        "-W",
        "ignore",
        "-m",
        "ctranslate2.converters.eole_ct2",
        "--model_path",
        str(ckpt_dir),
        "--output_dir",
        str(out_dir),
    ]


def test_export_ct2_hides_the_gpu_from_the_converter(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    fake = _install(monkeypatch, FakeConverter())

    export_ct2(tmp_path / "ckpt", tmp_path / "ct2")

    assert fake.calls[0][1]["env"]["CUDA_VISIBLE_DEVICES"] == ""


def test_export_ct2_writes_the_combined_log_beside_the_output_directory(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # Inside the output directory it would make the converter refuse to run.
    fake = _install(monkeypatch, FakeConverter(log_lines=("one", "two")))

    export_ct2(tmp_path / "ckpt", tmp_path / "ct2")

    log = tmp_path / "ct2.export.log"
    assert log.read_text(encoding="utf-8") == "one\ntwo\n"
    assert fake.calls[0][1]["stderr"] == subprocess.STDOUT


def test_export_ct2_creates_the_parent_of_the_output_directory(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _install(monkeypatch, FakeConverter())

    export_ct2(tmp_path / "ckpt", tmp_path / "deep" / "er" / "ct2")

    assert (tmp_path / "deep" / "er" / "ct2" / "model.bin").is_file()


@pytest.mark.parametrize(
    ("model_bytes", "problem"), [(None, "missing"), (b"", "empty")]
)
def test_export_ct2_fails_without_a_model_file_even_when_the_converter_exits_zero(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    model_bytes: bytes | None,
    problem: str,
) -> None:
    _install(
        monkeypatch,
        FakeConverter(model_bytes=model_bytes, log_lines=["last words"]),
    )

    with pytest.raises(UpstreamError) as raised:
        export_ct2(tmp_path / "ckpt", tmp_path / "ct2")

    message = str(raised.value)
    assert "model.bin" in message
    assert "exit code 0" in message
    assert message.endswith("last words")


def test_export_ct2_reports_a_segfault_with_the_exit_code_and_the_log_tail(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _install(
        monkeypatch,
        FakeConverter(
            exit_code=-11,
            model_bytes=b"",
            log_lines=[f"line {number}" for number in range(1, 31)],
        ),
    )

    with pytest.raises(UpstreamError) as raised:
        export_ct2(tmp_path / "ckpt", tmp_path / "ct2")

    message = str(raised.value)
    assert "exit code -11" in message
    assert message.endswith("\n".join(f"line {number}" for number in range(11, 31)))
    assert "line 10\n" not in message


# ---- ParityReport ---------------------------------------------------------------


def test_parity_report_agreement_is_the_share_of_agreeing_lines() -> None:
    assert ParityReport(agree=99, total=100, mismatches=()).agreement == 0.99


def test_parity_report_agreement_is_undefined_for_no_lines() -> None:
    with pytest.raises(ValueError, match="no lines"):
        _ = ParityReport(agree=0, total=0, mismatches=()).agreement


# ---- parity ---------------------------------------------------------------------


class StubTranslation:
    def __init__(self, pieces: list[str]) -> None:
        self.hypotheses = [pieces]


class StubTranslator:
    """Stands in for `ctranslate2.Translator` (the class and the instance): records
    how it was built and called, and answers each source line with `rewrite` applied
    to the word between the tag and the end, as pieces after a dummy prefix."""

    def __init__(self, rewrite: Callable[[str], str] = str.upper) -> None:
        self.rewrite = rewrite
        self.built: list[tuple[str, dict[str, Any]]] = []
        self.batches: list[tuple[list[list[str]], dict[str, Any]]] = []

    def __call__(self, model_path: str, **kwargs: Any) -> StubTranslator:
        self.built.append((model_path, kwargs))
        return self

    def translate_batch(
        self, source: list[list[str]], **kwargs: Any
    ) -> list[StubTranslation]:
        self.batches.append((source, kwargs))
        return [
            StubTranslation([DUMMY_PREFIX, *self.rewrite("".join(tokens[2:]))])
            for tokens in source
        ]


def _stub(monkeypatch: pytest.MonkeyPatch, stub: StubTranslator) -> StubTranslator:
    monkeypatch.setattr(export.ctranslate2, "Translator", stub)
    return stub


@pytest.fixture
def stub_translator(monkeypatch: pytest.MonkeyPatch) -> StubTranslator:
    return _stub(monkeypatch, StubTranslator())


@pytest.fixture(scope="module")
def spm_model(tmp_path_factory: pytest.TempPathFactory) -> Path:
    prefix = tmp_path_factory.mktemp("parity_model") / "chars"
    return train_char_model(
        ["abc", "cab", "bca", "ABC", "CAB", "BCA"], prefix, vocab_size=64
    )


WORDS = ["abc", "cab", "bca"]
UPPER = ["ABC", "CAB", "BCA"]


def test_parity_counts_every_line_as_agreeing_when_the_outputs_match(
    stub_translator: StubTranslator, spm_model: Path, tmp_path: Path
) -> None:
    report = parity(tmp_path, spm_model, WORDS, UPPER, tag=TO_SCRIPT_TAG)

    assert report == ParityReport(agree=3, total=3, mismatches=())
    assert report.agreement == 1.0


def test_parity_returns_the_source_and_both_outputs_of_each_mismatch(
    stub_translator: StubTranslator, spm_model: Path, tmp_path: Path
) -> None:
    eole_lines = ["ABC", "cab", "BCX"]

    report = parity(tmp_path, spm_model, WORDS, eole_lines, tag=TO_SCRIPT_TAG)

    assert report.agree == 1
    assert report.total == 3
    assert report.mismatches == (("cab", "CAB", "cab"), ("bca", "BCA", "BCX"))
    assert report.agreement == pytest.approx(1 / 3)


def test_parity_keeps_only_the_first_ten_mismatches_but_counts_them_all(
    stub_translator: StubTranslator, spm_model: Path, tmp_path: Path
) -> None:
    sources = ["abc"] * 12 + ["cab"]
    eole_lines = ["xyz"] * 12 + ["CAB"]

    report = parity(tmp_path, spm_model, sources, eole_lines, tag=TO_SCRIPT_TAG)

    assert report.total == 13
    assert report.agree == 1
    assert report.mismatches == (("abc", "ABC", "xyz"),) * 10


def test_parity_feeds_the_translator_the_tag_and_the_pieces_of_the_word(
    stub_translator: StubTranslator, spm_model: Path, tmp_path: Path
) -> None:
    parity(tmp_path, spm_model, ["abc", "cab"], ["ABC", "CAB"], tag=TO_SCRIPT_TAG)

    ((source, _),) = stub_translator.batches
    assert source == [
        [TO_SCRIPT_TAG, DUMMY_PREFIX, "a", "b", "c"],
        [TO_SCRIPT_TAG, DUMMY_PREFIX, "c", "a", "b"],
    ]


def test_parity_decodes_the_hypothesis_pieces_with_the_character_model(
    monkeypatch: pytest.MonkeyPatch, spm_model: Path, tmp_path: Path
) -> None:
    _stub(monkeypatch, StubTranslator(rewrite=lambda word: word + word))

    report = parity(tmp_path, spm_model, ["ab"], ["abab"], tag=TO_SCRIPT_TAG)

    assert report.agree == 1


def test_parity_builds_a_single_threaded_cpu_translator_and_decodes_greedily(
    stub_translator: StubTranslator, spm_model: Path, tmp_path: Path
) -> None:
    ct2_dir = tmp_path / "ct2"

    parity(ct2_dir, spm_model, WORDS, UPPER, tag=TO_SCRIPT_TAG)

    assert stub_translator.built == [
        (str(ct2_dir), {"device": "cpu", "intra_threads": 1})
    ]
    ((_, options),) = stub_translator.batches
    assert options == {"beam_size": 1, "max_decoding_length": 128}


def test_parity_passes_the_decoding_limit_to_the_translator(
    stub_translator: StubTranslator, spm_model: Path, tmp_path: Path
) -> None:
    parity(tmp_path, spm_model, WORDS, UPPER, tag=TO_SCRIPT_TAG, max_decoding_length=7)

    ((_, options),) = stub_translator.batches
    assert options["max_decoding_length"] == 7


def test_parity_rejects_line_lists_of_different_lengths(
    stub_translator: StubTranslator, spm_model: Path, tmp_path: Path
) -> None:
    with pytest.raises(ValueError, match="3 source lines but 2 Eole lines"):
        parity(tmp_path, spm_model, WORDS, UPPER[:2], tag=TO_SCRIPT_TAG)

    assert stub_translator.built == []


def test_parity_rejects_empty_input(
    stub_translator: StubTranslator, spm_model: Path, tmp_path: Path
) -> None:
    with pytest.raises(ValueError, match="no lines"):
        parity(tmp_path, spm_model, [], [], tag=TO_SCRIPT_TAG)

    assert stub_translator.built == []
