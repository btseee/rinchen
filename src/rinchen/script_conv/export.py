"""CTranslate2 export of the converter, and the check that CTranslate2 decodes like
Eole.

`export_ct2` converts an Eole checkpoint with CTranslate2's own converter in a
subprocess; `parity` compares CTranslate2's greedy output with Eole's on the same
words. The two decoders can stop at different lengths on a rare degenerate
repetition, so the gate on `ParityReport.agreement` is below 1.0 (0.99), not equal to it.
"""

from __future__ import annotations

import os
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import ctranslate2
import sentencepiece

from rinchen.script_conv._subprocess import failure_with_log_tail, run_logged

_MAX_REPORTED_MISMATCHES = 10
_MODEL_FILE = "model.bin"


@dataclass(frozen=True)
class ParityReport:
    """How many of `total` lines CTranslate2 and Eole converted to the same text.

    `mismatches` holds (source, CTranslate2 output, Eole output) for the first
    mismatching lines only, at most ten; `agree` counts all of them.
    """

    agree: int
    total: int
    mismatches: tuple[tuple[str, str, str], ...]

    @property
    def agreement(self) -> float:
        """`agree` over `total`; raises ValueError for no lines."""
        if self.total == 0:
            raise ValueError("the agreement of no lines is undefined")
        return self.agree / self.total


def export_environment() -> dict[str, str]:
    """A copy of the process environment for the CTranslate2 converter, with no CUDA
    device visible. With one visible the converter reads a device pointer as host
    memory, segfaults (exit code -11) and leaves an empty `model.bin`."""
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = ""
    return env


def export_ct2(ckpt_dir: Path, out_dir: Path) -> Path:
    """Convert the Eole checkpoint in `ckpt_dir` to a CTranslate2 model in `out_dir`
    (which must not exist yet) and return `out_dir`.

    The converter runs as `python -I -W ignore -m ctranslate2.converters.eole_ct2`
    with `export_environment()`; its combined output goes to `<out_dir>.export.log`,
    beside `out_dir` because the converter refuses a directory that exists.

    Raises:
        UpstreamError: the converter exits non-zero, or exits zero but leaves no
            `model.bin` or an empty one; the message ends with the last lines of the log.
    """
    log_path = Path(f"{out_dir}.export.log")
    run_logged(
        [
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
        ],
        env=export_environment(),
        log_path=log_path,
        what="the CTranslate2 export",
    )
    model_file = out_dir / _MODEL_FILE
    if not model_file.is_file() or model_file.stat().st_size == 0:
        raise failure_with_log_tail(
            f"the CTranslate2 export exited with exit code 0 but left no {_MODEL_FILE} "
            f"in {out_dir}, or an empty one",
            log_path,
        )
    return out_dir


def parity(
    ct2_dir: Path,
    spm_model: Path,
    src_lines: Sequence[str],
    eole_lines: Sequence[str],
    *,
    tag: str,
    max_decoding_length: int = 128,
) -> ParityReport:
    """Convert `src_lines` with the CTranslate2 model in `ct2_dir`, greedily on one
    CPU thread, and compare each result with the same position of `eole_lines`, the
    output Eole gave for the same lines.

    A source line is split by the character model at `spm_model` and starts with
    `tag`, as in training; the hypothesis pieces are decoded by the same model.

    Raises:
        ValueError: the two line sequences differ in length, or are empty.
    """
    if len(src_lines) != len(eole_lines):
        raise ValueError(
            f"{len(src_lines)} source lines but {len(eole_lines)} Eole lines"
        )
    if not src_lines:
        raise ValueError("cannot compare no lines")

    processor = sentencepiece.SentencePieceProcessor(model_file=str(spm_model))
    translator = ctranslate2.Translator(str(ct2_dir), device="cpu", intra_threads=1)
    results = translator.translate_batch(
        [[tag, *processor.encode(line, out_type=str)] for line in src_lines],
        beam_size=1,
        max_decoding_length=max_decoding_length,
    )
    mismatches = [
        (source, ct2_line, eole_line)
        for source, result, eole_line in zip(
            src_lines, results, eole_lines, strict=True
        )
        if (ct2_line := processor.decode(result.hypotheses[0])) != eole_line
    ]
    return ParityReport(
        agree=len(src_lines) - len(mismatches),
        total=len(src_lines),
        mismatches=tuple(mismatches[:_MAX_REPORTED_MISMATCHES]),
    )
