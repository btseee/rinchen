"""The pre-tokenised validation corpus Eole reads, written from the development sets
with SentencePiece."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import sentencepiece

from rinchen.config.ablation import TagPlacement, require_placement
from rinchen.config.languages import direction_tag
from rinchen.errors import ConfigError
from rinchen.train.mixture import DevSet
from rinchen.utils.part_file import replace_file


@dataclass(frozen=True, slots=True)
class ValidPaths:
    src: Path
    tgt: Path


def dev_line_ranges(dev: Mapping[str, DevSet]) -> dict[str, range]:
    """The lines each direction occupies in the validation corpus `write_validation`
    writes (and in any file that concatenates the dev sets in the same order):
    directions in sorted order, each a run of one line per dev row."""
    ranges: dict[str, range] = {}
    start = 0
    for direction in sorted(dev):
        stop = start + len(dev[direction].src)
        ranges[direction] = range(start, stop)
        start = stop
    return ranges


def _load_processor(spm_model: Path) -> sentencepiece.SentencePieceProcessor:
    # SentencePiece aborts the interpreter on a directory, so only files pass.
    if not spm_model.is_file():
        raise ConfigError(f"SentencePiece model is not a file: {spm_model}")
    try:
        return sentencepiece.SentencePieceProcessor(model_file=str(spm_model))
    except (OSError, RuntimeError) as exc:
        raise ConfigError(
            f"SentencePiece model {spm_model} cannot be loaded: {exc}"
        ) from exc


def _require_one_piece(
    processor: sentencepiece.SentencePieceProcessor, tag: str, spm_model: Path
) -> None:
    piece_id = processor.piece_to_id(tag)
    if piece_id == processor.unk_id() or processor.id_to_piece(piece_id) != tag:
        raise ConfigError(
            f"the direction tag {tag} is not one piece of the SentencePiece model "
            f"{spm_model}"
        )


def write_validation(
    dev: Mapping[str, DevSet],
    spm_model: Path,
    placement: TagPlacement,
    out_dir: Path,
) -> ValidPaths:
    """Write `valid.src` and `valid.tgt` to `out_dir`: the dev rows of all directions,
    sorted by direction, one line each, pre-tokenised with the SentencePiece model
    (pieces joined by one U+0020) and with the tag of the target language, one piece,
    first on the source line for placement `src` and first on the target line for `tgt`.

    Eole then reads the lines as they are (the corpus has no transforms and no
    prefix), so nothing of the validation leaks into the checkpoint.

    Raises:
        ValueError: a placement other than `src` or `tgt`; no dev set.
        ConfigError: the model cannot be loaded, or a tag is not one of its pieces.
    """
    require_placement(placement)
    if not dev:
        raise ValueError("at least one dev set is needed")
    processor = _load_processor(spm_model)
    src_lines: list[str] = []
    tgt_lines: list[str] = []
    for direction in sorted(dev):
        dev_set = dev[direction]
        tag = direction_tag(dev_set.tgt_lang)
        _require_one_piece(processor, tag, spm_model)
        sources = processor.encode(list(dev_set.src), out_type=str)
        targets = processor.encode(list(dev_set.tgt), out_type=str)
        for source, target in zip(sources, targets, strict=True):
            if placement == "src":
                source = [tag, *source]
            else:
                target = [tag, *target]
            src_lines.append(" ".join(source))
            tgt_lines.append(" ".join(target))
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = ValidPaths(src=out_dir / "valid.src", tgt=out_dir / "valid.tgt")
    replace_file(paths.src, ("\n".join(src_lines) + "\n").encode("utf-8"))
    replace_file(paths.tgt, ("\n".join(tgt_lines) + "\n").encode("utf-8"))
    return paths
