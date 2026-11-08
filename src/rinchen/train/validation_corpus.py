"""The development sets as SentencePiece token lines: the validation corpus Eole reads
during training, and the lines the evaluation of a trained variant translates."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import sentencepiece

from rinchen.config.ablation import TagPlacement, require_placement
from rinchen.config.languages import direction_tag
from rinchen.errors import ConfigError
from rinchen.train.mixture import DevSet
from rinchen.utils.part_file import lines_as_bytes, replace_file


@dataclass(frozen=True, slots=True)
class ValidPaths:
    src: Path
    tgt: Path


@dataclass(frozen=True, slots=True)
class DevTokens:
    """The dev rows of all directions as token lines (pieces joined by one U+0020),
    directions in sorted order, one line per dev row.

    `ranges` are the lines of each direction (`dev_line_ranges`). `forced_prefix` is
    None for the placement `src`; for `tgt` it holds one line per dev row, the tag of
    the target language, which a prediction must force as the start of the output."""

    source: tuple[str, ...]
    target: tuple[str, ...]
    ranges: dict[str, range]
    forced_prefix: tuple[str, ...] | None


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


def load_processor(spm_model: Path) -> sentencepiece.SentencePieceProcessor:
    """The SentencePiece model in `spm_model`.

    Raises:
        ConfigError: it is not a file or cannot be loaded.
    """
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


def require_direction_tags(spm_model: Path, dev: Mapping[str, DevSet]) -> None:
    """Check that `spm_model` loads and has the tag of every target language of the dev
    sets as one piece, which is what `dev_token_lines` needs of it.

    Raises:
        ConfigError: the model cannot be loaded, or a tag is not one of its pieces.
    """
    processor = load_processor(spm_model)
    for direction in sorted(dev):
        _require_one_piece(processor, direction_tag(dev[direction].tgt_lang), spm_model)


def dev_token_lines(
    dev: Mapping[str, DevSet], spm_model: Path, placement: TagPlacement
) -> DevTokens:
    """The dev rows of all directions, sorted by direction, pre-tokenised with the
    SentencePiece model and with the tag of the target language, one piece: first on
    the source line for placement `src`, first on the target line for `tgt` (the
    source line is then untagged and `forced_prefix` holds each line's tag).

    Raises:
        ValueError: a placement other than `src` or `tgt`; no dev set.
        ConfigError: the model cannot be loaded, or a tag is not one of its pieces.
    """
    require_placement(placement)
    if not dev:
        raise ValueError("at least one dev set is needed")
    processor = load_processor(spm_model)
    src_lines: list[str] = []
    tgt_lines: list[str] = []
    tags: list[str] = []
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
            tags.append(tag)
    return DevTokens(
        source=tuple(src_lines),
        target=tuple(tgt_lines),
        ranges=dev_line_ranges(dev),
        forced_prefix=tuple(tags) if placement == "tgt" else None,
    )


def validation_paths(out_dir: Path) -> ValidPaths:
    """The files `write_validation` writes into `out_dir` (nothing is created)."""
    return ValidPaths(src=out_dir / "valid.src", tgt=out_dir / "valid.tgt")


def write_validation(
    dev: Mapping[str, DevSet],
    spm_model: Path,
    placement: TagPlacement,
    out_dir: Path,
) -> ValidPaths:
    """Write `valid.src` and `valid.tgt` to `out_dir`: the lines of `dev_token_lines`,
    one line per dev row, source and target.

    Eole then reads the lines as they are (the corpus has no transforms and no
    prefix), so nothing of the validation leaks into the checkpoint.

    Raises:
        ValueError: a placement other than `src` or `tgt`; no dev set.
        ConfigError: the model cannot be loaded, or a tag is not one of its pieces.
    """
    tokens = dev_token_lines(dev, spm_model, placement)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = validation_paths(out_dir)
    replace_file(paths.src, lines_as_bytes(tokens.source))
    replace_file(paths.tgt, lines_as_bytes(tokens.target))
    return paths
