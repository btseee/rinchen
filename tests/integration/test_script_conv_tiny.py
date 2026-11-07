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
from rinchen.script_conv.train import predict
from tests.support.tiny_converter import (
    COMMAND_TIMEOUT_SECONDS,
    DIRECTION_TAGS,
    TO_CYRILLIC_TAG,
    TO_SCRIPT_TAG,
    TinyRun,
    read_lines,
)

MAX_CER = 0.15
MAX_TRAIN_SECONDS = 150
MIN_PRESERVED_SHARE = 0.90
MIN_PARITY = 0.99
MIN_TAG_EFFECT = 0.10
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
@pytest.mark.parametrize("direction", ["cyrl2mong", "mong2cyrl"])
def test_ctranslate2_decodes_the_training_words_like_eole_in_both_directions(
    tiny_run: TinyRun, tiny_ct2: Path, direction: str
) -> None:
    report = parity(
        tiny_ct2,
        tiny_run.spm_model,
        tiny_run.lines(f"{direction}.train.src"),
        tiny_run.predictions[direction],
        tag=DIRECTION_TAGS[direction],
    )

    assert report.agreement >= MIN_PARITY, (
        f"{direction}: {report.agree} of {report.total} agree; first mismatches "
        f"(source, CT2, Eole): {report.mismatches!a}"
    )


@pytest.mark.subprocess
def test_predict_converts_under_the_tag_it_is_given_not_the_one_the_checkpoint_remembers(
    tiny_run: TinyRun, tmp_path: Path
) -> None:
    # Eole copies the validation corpus's prefix (the script tag) into the checkpoint
    # and lets it override the prediction config, so a `prefix` transform converted
    # every file under the script tag, script-to-Cyrillic files too. Asking for the
    # script tag on script sources, as that defect did, must change the output. (On
    # Cyrillic sources the other tag changes only 6% of the words: the source script
    # already tells the model its direction. The reverse direction is the sensitive
    # one.)
    right_tag_output = tiny_run.predictions["mong2cyrl"]
    remembered_tag_output = read_lines(
        predict(
            tiny_run.ckpt_dir,
            tiny_run.corpus.files["mong2cyrl.train.src"],
            tag=TO_SCRIPT_TAG,
            spm_model=tiny_run.spm_model,
            out_file=tmp_path / "remembered_tag.pred",
            timeout=COMMAND_TIMEOUT_SECONDS,
        )
    )

    differing = sum(
        remembered != right
        for remembered, right in zip(
            remembered_tag_output, right_tag_output, strict=True
        )
    )
    assert differing / len(right_tag_output) >= MIN_TAG_EFFECT, (
        f"only {differing} of {len(right_tag_output)} words changed with the tag"
    )
    references = tiny_run.lines("mong2cyrl.train.tgt")
    reverse_cer = cer(references, right_tag_output)
    assert reverse_cer <= MAX_CER
    assert reverse_cer < cer(references, remembered_tag_output)


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
