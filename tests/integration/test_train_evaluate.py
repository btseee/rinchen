"""The development-set evaluation against a real Eole run: a tiny toy model is trained
once per tag placement, and `evaluate_variant` translates the dev lines of all four
directions with it in one `eole predict` call.

A run this short learns little, so no score threshold is asserted here; the plumbing
is what is checked: one output line per dev line, the forced tag stripped, the output
decoded and written, scores in range. The quality gate is the ablation's.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from pathlib import Path

import pytest
import sentencepiece

from rinchen.config.ablation import TagPlacement
from rinchen.train.evaluate import DevResult, evaluate_variant
from tests.support.tiny_converter import COMMAND_TIMEOUT_SECONDS
from tests.support.tiny_parallel import TOY_DIRECTIONS
from tests.support.tiny_variant import (
    DEV_LINES_PER_DIRECTION,
    THREADS,
    TinyVariant,
    ToyInputs,
    build_toy_inputs,
    train_tiny_variant,
)

DIRECTION_TAGS = {
    "eng_Latn-khk_Cyrl": "<2khk_Cyrl>",
    "khk_Cyrl-eng_Latn": "<2eng_Latn>",
    "rus_Cyrl-khk_Cyrl": "<2khk_Cyrl>",
    "khk_Cyrl-rus_Cyrl": "<2rus_Cyrl>",
}
DEV_LINES = DEV_LINES_PER_DIRECTION * len(TOY_DIRECTIONS)


@dataclass(frozen=True)
class Evaluated:
    spm_model: Path
    placement: TagPlacement
    variant: TinyVariant
    result: DevResult
    out_dir: Path
    eval_seconds: float


def _train_and_evaluate(
    placement: TagPlacement,
    toy_inputs: ToyInputs,
    tmp_path_factory: pytest.TempPathFactory,
) -> Evaluated:
    root = tmp_path_factory.mktemp(f"evaluate_{placement}")
    variant = train_tiny_variant(toy_inputs, placement, root)
    out_dir = root / "eval"
    started = time.perf_counter()
    result = evaluate_variant(
        variant.ckpt_dir,
        toy_inputs.dev,
        spm_model=toy_inputs.spm_model,
        placement=placement,
        out_dir=out_dir,
        threads=THREADS,
        timeout=COMMAND_TIMEOUT_SECONDS,
    )
    return Evaluated(
        spm_model=toy_inputs.spm_model,
        placement=placement,
        variant=variant,
        result=result,
        out_dir=out_dir,
        eval_seconds=time.perf_counter() - started,
    )


@pytest.fixture(scope="module")
def toy_inputs(tmp_path_factory: pytest.TempPathFactory) -> ToyInputs:
    return build_toy_inputs(tmp_path_factory.mktemp("evaluate_toy"))


# One training and one evaluation per placement for the whole module.
@pytest.fixture(scope="module")
def evaluated_src(
    toy_inputs: ToyInputs, tmp_path_factory: pytest.TempPathFactory
) -> Evaluated:
    return _train_and_evaluate("src", toy_inputs, tmp_path_factory)


@pytest.fixture(scope="module")
def evaluated_tgt(
    toy_inputs: ToyInputs, tmp_path_factory: pytest.TempPathFactory
) -> Evaluated:
    return _train_and_evaluate("tgt", toy_inputs, tmp_path_factory)


@pytest.fixture(params=["src", "tgt"])
def evaluated(request: pytest.FixtureRequest) -> Evaluated:
    value: Evaluated = request.getfixturevalue(f"evaluated_{request.param}")
    return value


def _lines(path: Path) -> list[str]:
    return path.read_bytes().decode("utf-8").removesuffix("\n").split("\n")


@pytest.mark.subprocess
def test_every_direction_is_scored_on_all_its_dev_lines(evaluated: Evaluated) -> None:
    result = evaluated.result

    assert list(result.directions) == sorted(TOY_DIRECTIONS)
    for direction, scores in result.directions.items():
        assert scores.lines == DEV_LINES_PER_DIRECTION, direction


@pytest.mark.subprocess
def test_the_hypothesis_files_hold_one_line_per_dev_line(evaluated: Evaluated) -> None:
    for direction in TOY_DIRECTIONS:
        path = evaluated.out_dir / f"hyp.{direction}.txt"

        assert len(_lines(path)) == DEV_LINES_PER_DIRECTION, direction


@pytest.mark.subprocess
def test_the_scores_are_finite_numbers_between_0_and_100(evaluated: Evaluated) -> None:
    result = evaluated.result
    scores: list[float] = [result.chrf_mean, result.chrf_min]
    for direction_scores in result.directions.values():
        scores.extend((direction_scores.chrf, direction_scores.bleu))

    for score in scores:
        assert math.isfinite(score)
        assert 0.0 <= score <= 100.0
    assert result.chrf_min <= result.chrf_mean
    assert result.chrf_signature
    assert result.bleu_signature


@pytest.mark.subprocess
def test_tag_leaks_is_a_count_of_hypotheses(evaluated: Evaluated) -> None:
    leaks = evaluated.result.tag_leaks

    assert isinstance(leaks, int)
    assert 0 <= leaks <= DEV_LINES


@pytest.mark.subprocess
def test_the_hypotheses_are_eoles_output_decoded_by_direction(
    evaluated: Evaluated,
) -> None:
    # Decoded here with SentencePiece's own call and the plain piece split, the
    # independent route to what `evaluate_variant` wrote.
    processor = sentencepiece.SentencePieceProcessor(
        model_file=str(evaluated.spm_model)
    )
    produced = _lines(evaluated.out_dir / "dev.tokens.out")
    assert len(produced) == DEV_LINES
    written: list[str] = []
    for direction in sorted(TOY_DIRECTIONS):
        written.extend(_lines(evaluated.out_dir / f"hyp.{direction}.txt"))
    expected = [
        processor.decode(
            line.split(" ")[1:] if evaluated.placement == "tgt" else line.split(" ")
        )
        for line in produced
    ]

    assert written == expected
    # The short run writes text: a check on the decoding, not on the quality.
    assert any(written)


@pytest.mark.subprocess
def test_the_forced_tag_is_stripped_from_the_hypotheses_of_the_target_placement(
    evaluated_tgt: Evaluated,
) -> None:
    forced = _lines(evaluated_tgt.out_dir / "dev.tokens.tgt")
    produced = _lines(evaluated_tgt.out_dir / "dev.tokens.out")

    # Eole's own output starts with the forced tag on every line; the evaluation
    # removed it (and would have raised had a line not started with it).
    assert len(forced) == len(produced) == DEV_LINES
    for tag, line in zip(forced, produced, strict=True):
        assert line.split(" ")[0] == tag
    for direction, tag in DIRECTION_TAGS.items():
        for line in _lines(evaluated_tgt.out_dir / f"hyp.{direction}.txt"):
            assert not line.startswith(tag), direction


@pytest.mark.subprocess
def test_one_prediction_covers_all_directions_in_either_placement(
    evaluated: Evaluated,
) -> None:
    sources = _lines(evaluated.out_dir / "dev.tokens.src")
    config = (evaluated.out_dir / "dev.predict.yaml").read_text(encoding="utf-8")

    assert len(sources) == DEV_LINES
    first_piece = sources[0].split(" ")[0]
    if evaluated.placement == "src":
        assert first_piece.startswith("<2")
        assert "tgt_file_prefix" not in config
    else:
        assert not first_piece.startswith("<2")
        assert "tgt_file_prefix: true" in config
