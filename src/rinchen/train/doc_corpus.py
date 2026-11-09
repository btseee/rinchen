"""What a document-level continuation trains on and with (D9): the corpus files of the
windows `docify` cut, the Eole `data` section that names them, the validation windows,
and the config of the continuation run.

The corpus files are plain text as D7's (`corpus_files`): one `<src>-<tgt>.doc.src` and
`.tgt` per direction under `<data_dir>/doc`, one window per line, the sentences of a
window joined with ` <sep> `, and a `.json` sidecar that lets an unchanged corpus be
reused. They carry no tag: the direction tag and `<doc>` are Eole `prefix`
configuration (`src_prefix` is `<2xx> <doc>`, `tgt_prefix` is empty), as D7's tags are,
so the text files stay the same for any placement.

The validation windows are the dev windows, pre-tokenised as D7's validation corpus is
(Eole reads the lines as they are): the source is the tag, `<doc>` and the pieces, the
target the pieces.

The config of the continuation is the base config (the model and the optimiser options
the run states for itself) with the per-run values laid over it. It never holds
`train_from` or `reset_optim`: they are not keys a config file may carry (`eole_cfg`);
the launcher adds them for the continuation it checks (Task 3).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final

import polars as pl

from rinchen.config.languages import direction_tag
from rinchen.errors import ConfigError, DataError
from rinchen.train.corpus_files import CorpusFiles, write_corpus_files
from rinchen.train.docify import DOC_MARKER, DocWindows
from rinchen.train.eole_cfg import read_config, render
from rinchen.train.eole_data import data_section
from rinchen.train.sampling import CorpusSize, integer_weights, mixing_probabilities
from rinchen.train.validation_corpus import (
    ValidPaths,
    load_processor,
    require_one_piece,
    validation_paths,
)
from rinchen.utils.integers import is_strict_int
from rinchen.utils.part_file import lines_as_bytes, replace_file

DOC_GROUP: Final = "doc"
_VALID: Final = "valid"


def build_doc_corpora(windows: DocWindows, data_dir: Path) -> list[CorpusFiles]:
    """Write the windows as corpus files, one corpus `<src>-<tgt>.doc` per direction,
    sorted by direction, under `data_dir/doc` (rows sorted by id, then source, then
    target, as D7's corpora are), and describe them. An unchanged corpus is reused: its
    files are not written again (`CorpusFiles.reused`).

    Raises:
        DataError: there is no window.
        OSError: a directory or file cannot be created or written.
    """
    frame = windows.frame
    if frame.is_empty():
        raise DataError("no document windows to write")
    directory = data_dir / DOC_GROUP
    by_direction = frame.with_columns(
        (pl.col("src_lang") + "-" + pl.col("tgt_lang")).alias("direction")
    ).partition_by("direction", as_dict=True)
    return [
        write_corpus_files(part, directory, direction, DOC_GROUP)
        for (direction,), part in sorted(by_direction.items())
    ]


def doc_weights(
    corpora: Sequence[CorpusFiles], *, temperature: float, total: int = 100
) -> dict[str, int]:
    """The integer weights of the corpora as D7 mixes them (`sampling`): the
    probability of a direction is its window count to the power of `temperature`, and
    the weights are whole numbers of at least 1 that add up to `total`.

    Raises:
        ValueError: the temperature is not a positive finite number or is so large that
            the counts raised to it overflow; no corpus.
        ConfigError: `total` is below the number of corpora.
    """
    sizes = [
        CorpusSize(corpus.name, corpus.direction, corpus.rows) for corpus in corpora
    ]
    return integer_weights(mixing_probabilities(sizes, temperature), total=total)


def doc_data_section(
    corpora: Sequence[CorpusFiles],
    weights: Mapping[str, int],
    *,
    valid_paths: ValidPaths,
    filtertoolong: bool = True,
) -> dict[str, Any]:
    """The Eole `data` mapping of the doc corpora: as D7's (`eole_data.data_section`,
    tag on the source side) with `<doc>` after the tag in each `src_prefix`
    (`<2khk_Cyrl> <doc>`), an empty `tgt_prefix`, the transforms `sentencepiece`,
    `prefix` and (when `filtertoolong`) `filtertoolong`, the `weights` and the `valid`
    entry.

    Raises:
        ValueError, ConfigError: as `eole_data.data_section`.
    """
    section = data_section(
        corpora, weights, "src", valid_paths=valid_paths, filtertoolong=filtertoolong
    )
    for name, entry in section.items():
        if name != _VALID:
            entry["src_prefix"] = f"{entry['src_prefix']} {DOC_MARKER}"
    return section


def write_doc_validation(
    dev: DocWindows,
    spm_model: Path,
    out_dir: Path,
    *,
    max_windows: int | None = None,
) -> ValidPaths:
    """Write `valid.src` and `valid.tgt` to `out_dir`: the dev windows as token lines,
    one line per window in the order of the frame, the source as the tag of the target
    language, `<doc>` and the pieces of the joined text, the target as its pieces
    (SentencePiece cuts the joined text, so a `<sep>` is one piece with a word-boundary
    piece before it, as the training data has). With `max_windows`, each direction keeps
    its windows with the smallest ids (hashes, so spread over the documents).

    Raises:
        ValueError: `max_windows` is not a positive integer.
        DataError: there is no window.
        ConfigError: the model cannot be loaded or lacks `<doc>` or a tag as one piece.
    """
    if max_windows is not None and (not is_strict_int(max_windows) or max_windows < 1):
        raise ValueError(f"max_windows must be a positive integer, got {max_windows!r}")
    frame = dev.frame
    if frame.is_empty():
        raise DataError("no dev windows to write")
    if max_windows is not None:
        frame = frame.filter(
            pl.col("id").rank("ordinal").over("src_lang", "tgt_lang") <= max_windows
        )
    processor = load_processor(spm_model)
    require_one_piece(processor, DOC_MARKER, spm_model, what="document symbol")
    tags = {lang: direction_tag(lang) for lang in frame["tgt_lang"].unique()}
    for tag in tags.values():
        require_one_piece(processor, tag, spm_model)
    sources = processor.encode(frame["src"].to_list(), out_type=str)
    targets = processor.encode(frame["tgt"].to_list(), out_type=str)
    src_lines = [
        " ".join([tags[lang], DOC_MARKER, *pieces])
        for lang, pieces in zip(frame["tgt_lang"], sources, strict=True)
    ]
    tgt_lines = [" ".join(pieces) for pieces in targets]
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = validation_paths(out_dir)
    replace_file(paths.src, lines_as_bytes(src_lines))
    replace_file(paths.tgt, lines_as_bytes(tgt_lines))
    return paths


def render_continuation_config(
    base: Path,
    *,
    data: Mapping[str, Any],
    spm_model: Path,
    src_vocab: Path,
    model_path: Path,
    out: Path,
) -> Path:
    """Write to `out` the base config of a continuation with the per-run values laid
    over it, validated, and return `out`: `data` (the section of `doc_data_section`),
    `src_vocab` (a file Eole wants to exist, whose content a continuation ignores: it
    keeps the vocabulary of the checkpoint), `transforms_configs.sentencepiece` (the
    model, both sides) and `training.model_path`. The `model` section and every option
    of the base stay as they are. `train_from` and `reset_optim` are not added: the
    launcher adds them, and a base that holds one fails the validation of the config.

    Raises:
        ConfigError: the base holds corpora of its own (the continuation trains on the
            windows alone), or the finished config is not valid (what `eole_cfg.render`
            raises: a path that holds `$`, which Eole would expand as a variable, `out`
            is the base or a directory, ...).
    """
    corpora = _corpora_of(read_config(base, "base Eole config"))
    if corpora:
        raise ConfigError(
            f"invalid base Eole config {base}: data: holds the corpora {sorted(corpora)}, "
            "and a continuation trains on the document windows alone: leave `data` "
            "empty in the base"
        )
    spm = spm_model.as_posix()
    overrides: dict[str, Any] = {
        "src_vocab": src_vocab.as_posix(),
        "data": dict(data),
        "transforms_configs": {
            "sentencepiece": {"src_subword_model": spm, "tgt_subword_model": spm}
        },
        "training": {"model_path": model_path.as_posix()},
    }
    return render(base, overrides, out)


def _corpora_of(base: object) -> Mapping[Any, Any]:
    data = base.get("data") if isinstance(base, Mapping) else None
    return data if isinstance(data, Mapping) else {}
