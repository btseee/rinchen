"""The tiny converter end to end: a real Eole training run on CPU, its predictions,
the CTranslate2 export and the parity of the two decoders.

Every test uses the session fixtures of `conftest.py`, so the model is trained once.
The gates are the spec's: a missed gate means more training steps or a fix, never a
lower threshold.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from rinchen.errors import UpstreamError
from rinchen.script_conv import export
from rinchen.script_conv.alphabet import FVS, MVS, NNBSP
from rinchen.script_conv.eval import cer
from rinchen.script_conv.export import export_ct2, parity
from tests.support.tiny_converter import TO_CYRILLIC_TAG, TO_SCRIPT_TAG, TinyRun

MAX_CER = 0.15
MAX_TRAIN_SECONDS = 150
MIN_PRESERVED_SHARE = 0.90
MIN_PARITY = 0.99
# The characters whose count in a predicted word must equal the reference's.
GAP_AND_SELECTOR_MARKS = (NNBSP, MVS, *sorted(FVS))


@pytest.mark.subprocess
@pytest.mark.parametrize("direction", ["cyrl2mong", "mong2cyrl"])
def test_tiny_model_reproduces_its_training_words_in_both_directions(
    tiny_run: TinyRun, direction: str
) -> None:
    references = tiny_run.lines(f"{direction}.train.tgt")
    predictions = tiny_run.predictions[direction]

    assert len(predictions) == len(references)
    error_rate = cer(references, predictions)
    assert error_rate <= MAX_CER, f"{direction}: CER {error_rate:.4f} > {MAX_CER}"


@pytest.mark.subprocess
def test_tiny_training_finishes_inside_the_time_budget(tiny_run: TinyRun) -> None:
    assert tiny_run.train_seconds < MAX_TRAIN_SECONDS


@pytest.mark.subprocess
def test_tiny_model_keeps_suffix_gaps_and_selectors_in_the_script_it_writes(
    tiny_run: TinyRun,
) -> None:
    references = tiny_run.lines("cyrl2mong.train.tgt")
    predictions = tiny_run.predictions["cyrl2mong"]
    marked = [
        (reference, prediction)
        for reference, prediction in zip(references, predictions, strict=True)
        if any(mark in reference for mark in GAP_AND_SELECTOR_MARKS)
    ]

    assert marked, "the fixture has no word with a gap or a selector"
    preserved = sum(
        all(
            reference.count(mark) == prediction.count(mark)
            for mark in GAP_AND_SELECTOR_MARKS
        )
        for reference, prediction in marked
    )
    share = preserved / len(marked)
    assert share >= MIN_PRESERVED_SHARE, f"{preserved} of {len(marked)} words"


@pytest.mark.subprocess
def test_the_vocabulary_eole_builds_holds_both_direction_tags(
    tiny_run: TinyRun,
) -> None:
    text = (tiny_run.run_dir / "vocab.src").read_text(encoding="utf-8")
    pieces = {line.partition("\t")[0] for line in text.split("\n")}

    assert {TO_SCRIPT_TAG, TO_CYRILLIC_TAG} <= pieces


@pytest.mark.subprocess
def test_ctranslate2_decodes_the_training_words_like_eole(
    tiny_run: TinyRun, tiny_ct2: Path
) -> None:
    report = parity(
        tiny_ct2,
        tiny_run.spm_model,
        tiny_run.lines("cyrl2mong.train.src"),
        tiny_run.predictions["cyrl2mong"],
        tag=TO_SCRIPT_TAG,
    )

    assert report.agreement >= MIN_PARITY, (
        f"{report.agree} of {report.total} agree; first mismatches "
        f"(source, CT2, Eole): {report.mismatches!a}"
    )


@pytest.mark.subprocess
@pytest.mark.skipif(
    os.environ.get("RINCHEN_CT2_CUDA_PROBE") != "1",
    reason="set RINCHEN_CT2_CUDA_PROBE=1 on a machine with a visible CUDA device",
)
def test_the_converter_fails_when_a_cuda_device_is_visible(
    tiny_run: TinyRun, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Documents why `export_environment` hides the GPU: with a device visible the
    # converter segfaults on a CUDA pointer and leaves an empty model.bin.
    monkeypatch.setattr(export, "export_environment", lambda: dict(os.environ))

    with pytest.raises(UpstreamError, match="exit code -11"):
        export_ct2(tiny_run.ckpt_dir, tmp_path / "unhidden")
