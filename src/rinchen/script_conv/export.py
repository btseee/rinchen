"""CTranslate2 export of the converter, and the check that CTranslate2 decodes like
Eole.

`export_ct2` converts an Eole checkpoint with CTranslate2's own converter in a
subprocess; it lives in `rinchen.export.eole_ct2` since D10 (the distillation exports
teachers through it) and is imported here with `export_environment` and
`export_log_path`. `parity` compares CTranslate2's greedy output with Eole's on the same
words. The two decoders can stop at different lengths on a rare degenerate
repetition, so the gate on `ParityReport.agreement` is below 1.0 (0.99), not equal to it.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import ctranslate2

from rinchen.export.eole_ct2 import export_ct2, export_environment, export_log_path
from rinchen.script_conv.char_model import load_char_processor

__all__ = [
    "ParityReport",
    "export_ct2",
    "export_environment",
    "export_log_path",
    "parity",
]

_MAX_REPORTED_MISMATCHES = 10


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
        DataError: the character model at `spm_model` cannot be loaded (checked
            before the CTranslate2 model is).
    """
    if len(src_lines) != len(eole_lines):
        raise ValueError(
            f"{len(src_lines)} source lines but {len(eole_lines)} Eole lines"
        )
    if not src_lines:
        raise ValueError("cannot compare no lines")

    processor = load_char_processor(spm_model)
    translator = ctranslate2.Translator(str(ct2_dir), device="cpu", intra_threads=1)
    results = translator.translate_batch(
        [[tag, *processor.encode(line, return_type=str)] for line in src_lines],
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
