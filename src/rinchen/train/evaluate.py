"""The development-set evaluation of a trained variant: its checkpoint translates the
dev lines of every direction in ONE `eole predict` call (the model loads once), and
each direction is scored on its own slice of the output.

Eole reads and writes token lines (SentencePiece pieces joined by U+0020): the lines
come from `validation_corpus.dev_token_lines` and the output is decoded here. Only
the dev rows it is given are read, never a test split.
"""

from __future__ import annotations

import statistics
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Final, Protocol

import sentencepiece

from rinchen.config.ablation import TagPlacement, require_placement
from rinchen.config.languages import load_languages
from rinchen.errors import DataError, UpstreamError
from rinchen.metrics.bleu_chrf import CorpusScores, corpus_scores
from rinchen.train.eole_data import bt_tag
from rinchen.train.eole_run import predict_tokens
from rinchen.train.mixture import DevSet
from rinchen.train.validation_corpus import (
    DevTokens,
    dev_token_lines,
    load_processor,
)
from rinchen.utils.part_file import lines_as_bytes, replace_file

# The files of the prediction (`dev.log`, `dev.predict.yaml`, `dev.tokens.*`) sit
# beside the hypothesis files in the evaluation directory.
PREDICT_STEM: Final = "dev"


class TokenPredictor(Protocol):
    """What `evaluate_variant` needs of a prediction: the signature of
    `eole_run.predict_tokens`, the seam a unit test replaces."""

    def __call__(
        self,
        ckpt_dir: Path,
        source_tokens: Sequence[str],
        *,
        out_dir: Path,
        forced_prefix_tokens: Sequence[str] | None = None,
        threads: int = 2,
        timeout: float | None = None,
        stem: str = "eole",
        stale_files: Sequence[Path] = (),
    ) -> list[str]: ...


@dataclass(frozen=True, slots=True)
class DirectionScores:
    """The corpus scores, in [0, 100], of one direction over its `lines` dev lines."""

    lines: int
    chrf: float
    bleu: float


@dataclass(frozen=True, slots=True)
class DevResult:
    """What `evaluate_variant` measured. `directions` are in sorted order. `chrf_mean`
    is the macro mean over directions (each counts once, whatever its size) and
    `chrf_min` the lowest chrF of a direction. `tag_leaks` is the number of hypotheses,
    over all directions, that hold a direction tag or the bt tag as text. The
    signatures say how chrF and BLEU were computed; they are the same for every
    direction.

    `to_dict` is JSON-ready; the scores are the raw floats, rounded nowhere."""

    directions: dict[str, DirectionScores]
    chrf_mean: float
    chrf_min: float
    tag_leaks: int
    chrf_signature: str
    bleu_signature: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def hypothesis_path(out_dir: Path, direction: str) -> Path:
    """Where `evaluate_variant` writes the hypotheses of `direction`."""
    return out_dir / f"hyp.{direction}.txt"


def evaluate_variant(
    ckpt_dir: Path,
    dev: Mapping[str, DevSet],
    *,
    spm_model: Path,
    placement: TagPlacement,
    out_dir: Path,
    threads: int = 2,
    timeout: float | None = None,
    predict: TokenPredictor = predict_tokens,
) -> DevResult:
    """Translate the dev lines of every direction of `dev` with the checkpoint in
    `ckpt_dir` and score each direction against its dev targets.

    With placement `src` the source line starts with the tag of the target language;
    with `tgt` it is untagged and the tag is forced as the start of the output, which
    then starts with it: the first token of each output line is checked and removed.
    The output is decoded with `spm_model` (the pieces split at U+0020 only: a piece
    may hold U+202F or other characters `str.split()` would cut at), an empty line
    giving an empty hypothesis. A line feed in a hypothesis (a byte-fallback piece can
    decode to one) becomes a space: both metrics treat the two alike, and a hypothesis
    file keeps one line per dev line.

    The hypotheses go to `hypothesis_path(out_dir, direction)` (LF-terminated UTF-8,
    after everything else has succeeded, replacing a file of an earlier run), and
    Eole's own files to `out_dir` as `dev.*`. `predict` is the seam for a test; it
    defaults to `eole_run.predict_tokens`. `threads` and `timeout` (seconds) go to it.

    Raises:
        ValueError: a placement other than `src` or `tgt`; a direction name that is
            not a file name; `threads` is less than 1.
        DataError: no dev set, or a dev set without lines.
        ConfigError: `spm_model` cannot be loaded, or a tag is not one of its pieces.
        UpstreamError: `predict` fails, writes another number of lines than there are
            dev lines, or (placement `tgt`) an output line does not start with the
            forced tag; the message names the direction and the line, counted from 1
            inside the direction, and never the text.
        OSError: `out_dir` cannot be created or a file cannot be written.
    """
    require_placement(placement)
    _require_usable_dev(dev)
    tokens = dev_token_lines(dev, spm_model, placement)
    # Loaded a second time, for decoding: milliseconds, and the lines' own function
    # keeps the model out of its interface.
    processor = load_processor(spm_model)
    markers = _leak_markers()
    # Before the prediction, which can take minutes: a directory that cannot be made
    # should not cost them.
    out_dir.mkdir(parents=True, exist_ok=True)
    files = {
        direction: hypothesis_path(out_dir, direction) for direction in tokens.ranges
    }
    output = predict(
        ckpt_dir,
        tokens.source,
        out_dir=out_dir,
        forced_prefix_tokens=tokens.forced_prefix,
        threads=threads,
        timeout=timeout,
        stem=PREDICT_STEM,
        stale_files=list(files.values()),
    )
    if len(output) != len(tokens.source):
        raise UpstreamError(
            f"the prediction wrote {len(output)} lines for {len(tokens.source)} source "
            "lines"
        )

    hypotheses = {
        direction: _decode(processor, output, tokens, direction)
        for direction in tokens.ranges
    }
    scores = {
        direction: corpus_scores(hypotheses[direction], list(dev[direction].tgt))
        for direction in tokens.ranges
    }
    for direction, lines in hypotheses.items():
        replace_file(files[direction], lines_as_bytes(lines))
    return _result(scores, hypotheses, markers)


def _require_usable_dev(dev: Mapping[str, DevSet]) -> None:
    if not dev:
        raise DataError("no dev set to evaluate")
    for direction, dev_set in dev.items():
        if not direction or any(char in direction for char in "/\\\0"):
            raise ValueError(f"direction {direction!r} cannot be a file name")
        if not dev_set.src:
            raise DataError(f"the dev set of direction {direction} has no lines")


def _leak_markers() -> tuple[str, ...]:
    """The direction tag of every language of the registry, and the bt tag."""
    return (*(entry.direction_tag for entry in load_languages()), bt_tag())


def _decode(
    processor: sentencepiece.SentencePieceProcessor,
    output: Sequence[str],
    tokens: DevTokens,
    direction: str,
) -> list[str]:
    """The hypotheses of `direction`: its lines of `output`, the forced tag removed
    (placement `tgt`), decoded."""
    hypotheses: list[str] = []
    for number, index in enumerate(tokens.ranges[direction], start=1):
        # U+0020 only: a piece may hold U+202F and other whitespace `str.split()` cuts.
        pieces = output[index].split(" ")
        if tokens.forced_prefix is not None:
            pieces = _without_forced_tag(
                pieces, tokens.forced_prefix[index], direction, number
            )
        hypotheses.append(processor.decode(pieces).replace("\n", " "))
    return hypotheses


def _without_forced_tag(
    pieces: list[str], tag: str, direction: str, number: int
) -> list[str]:
    if pieces[0] != tag:
        raise UpstreamError(
            f"eole predict did not start line {number} of direction {direction} with "
            f"the forced tag {tag}"
        )
    return pieces[1:]


def _result(
    scores: Mapping[str, CorpusScores],
    hypotheses: Mapping[str, Sequence[str]],
    markers: Sequence[str],
) -> DevResult:
    chrfs = [score.chrf for score in scores.values()]
    # The signatures do not depend on the direction: take the first.
    first = next(iter(scores.values()))
    return DevResult(
        directions={
            direction: DirectionScores(
                lines=len(hypotheses[direction]), chrf=score.chrf, bleu=score.bleu
            )
            for direction, score in scores.items()
        },
        chrf_mean=statistics.fmean(chrfs),
        chrf_min=min(chrfs),
        tag_leaks=sum(
            any(marker in hypothesis for marker in markers)
            for lines in hypotheses.values()
            for hypothesis in lines
        ),
        chrf_signature=first.chrf_signature,
        bleu_signature=first.bleu_signature,
    )
