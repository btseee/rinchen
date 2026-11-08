"""Eole runs of the converter: vocabulary and training from a rendered config, and
prediction from a trained checkpoint.

The runs themselves are `rinchen.train.eole_run`'s, shared with every other Rinchen
model; this module adds the converter's own parts: its config validator, and the
character-model tag, tokenisation and decoding of `predict`.
"""

from __future__ import annotations

from functools import partial
from pathlib import Path
from typing import NamedTuple

from rinchen.errors import ConfigError, DataError
from rinchen.script_conv.char_model import CONVERTER_TAGS, load_char_processor
from rinchen.script_conv.config import Profile, validate_converter_config
from rinchen.train import eole_run
from rinchen.train.eole_cfg import reject_dollar
from rinchen.train.eole_run import (
    CHECKPOINT_FILE,
    eole_commands,
    predict_files,
    split_lines,
    training_environment,
    write_lines,
)

# CHECKPOINT_FILE, eole_commands and training_environment moved to `eole_run`; they are
# still named here for the callers of this module.
__all__ = [
    "CHECKPOINT_FILE",
    "PredictSidePaths",
    "eole_commands",
    "predict",
    "predict_side_paths",
    "run_training",
    "training_environment",
]


def run_training(
    cfg_path: Path,
    run_dir: Path,
    *,
    threads: int = 2,
    timeout: float | None = None,
    profile: Profile | None = None,
) -> Path:
    """Build the vocabulary, then train, as `eole build_vocab` and `eole train` on the
    config at `cfg_path`; return the checkpoint directory, the config's
    `training.model_path`.

    `cfg_path` is a file `render_converter_config(..., run_dir=run_dir)` rendered.
    `run_dir` is created and takes `build_vocab.log` and `train.log` (standard output
    and error of each command). The GPU is hidden from both commands when the config
    trains on none (`training.gpu_ranks` empty). `timeout` (seconds) bounds each
    command; a command that runs past it is killed with everything it started.

    `profile` is the profile the config was rendered for, handed to
    `validate_converter_config`: with `"tiny"` a config that asks for a GPU or for
    another compute type than fp32 is refused. Without it, the default, those
    CPU-only checks are skipped and the config trains as written. The `train`
    command passes the profile it rendered.

    Raises:
        ConfigError: the config file is missing, unparseable or fails
            `validate_converter_config` (checked before anything runs), or `profile`
            is unknown; its
            `training.model_path` is missing or not inside `run_dir`; or its path,
            its text or `run_dir` holds `$`, which Eole would expand as a variable.
        ValueError: `threads` is less than 1.
        UpstreamError: a command exits non-zero or runs past `timeout` (the message
            ends with the last lines of its log), the `eole` command is missing, or
            training exits zero without writing a checkpoint after it was launched.
    """
    return eole_run.run_training(
        cfg_path,
        run_dir,
        validate=partial(validate_converter_config, profile=profile),
        threads=threads,
        timeout=timeout,
        what="converter config",
    )


class PredictSidePaths(NamedTuple):
    """The files `predict` writes next to its output file."""

    log: Path  # standard output and error of `eole predict`
    config: Path  # the prediction config
    token_source: Path  # the source lines as tokens, one line each, tag first
    token_output: Path  # Eole's hypotheses as token lines


def predict_side_paths(out_file: Path) -> PredictSidePaths:
    """The files `predict` writes next to `out_file`: `<out_file>.log`,
    `.predict.yaml`, `.tokens.src` and `.tokens.out`."""
    files = predict_files(out_file.parent, out_file.name)
    return PredictSidePaths(
        log=files.log,
        config=files.config,
        token_source=files.token_source,
        token_output=files.token_output,
    )


def predict(
    ckpt_dir: Path,
    src_file: Path,
    *,
    tag: str,
    spm_model: Path,
    out_file: Path,
    threads: int = 2,
    timeout: float | None = None,
) -> Path:
    """Convert the lines of `src_file` under `tag` with the checkpoint in `ckpt_dir`,
    greedily and on the CPU, as `eole predict`; return `out_file`, which has one line
    per line of `src_file` (an empty hypothesis is an empty line).

    Eole does not tokenise or tag here: this function does, and Eole reads and writes
    token lines (pieces joined by U+0020, the tag first). Eole's own `sentencepiece`
    and `prefix` transforms cannot be used, because Eole 0.6.4 copies the `valid`
    corpus's `src_prefix` into `transforms_configs.prefix` when it saves the
    checkpoint (`train_single.ValidationConfigurer`), and then lets the checkpoint's
    `transforms_configs` override the prediction config, per transform name
    (`config/run.py`, `_update_with_model_config`). Every prediction would run under
    the tag of the validation direction, whatever `tag` says.

    The side files go next to the output (`predict_side_paths`); a file already at
    `out_file` or at the raw output is removed first. `timeout` (seconds) bounds the
    command; one that runs past it is killed with everything it started.

    Raises:
        ConfigError: `tag` is not one of `CONVERTER_TAGS` or is not a piece of the
            model; or `ckpt_dir` or `out_file` holds `$`, which Eole would expand as
            a variable in the prediction config (they, and the token files beside
            `out_file`, are the paths the config names; `src_file` and `spm_model`
            are only read here, so `$` is fine in them).
        DataError: the character model at `spm_model` cannot be loaded, or `src_file`
            cannot be read or has no lines.
        ValueError: `threads` is less than 1.
        UpstreamError: `eole predict` exits non-zero or runs past `timeout` (the
            message ends with the last lines of its log), the `eole` command is
            missing, or it exits zero without writing its output or with another
            number of lines than the source has.
    """
    if tag not in CONVERTER_TAGS:
        raise ConfigError(
            f"the tag {tag!r} is not a converter tag: expected one of {CONVERTER_TAGS}"
        )
    # `out_file` is enough for the files Eole is given: they only add a suffix to it.
    for name, value in (("checkpoint directory", ckpt_dir), ("output file", out_file)):
        reject_dollar(name, value)
    processor = load_char_processor(spm_model)
    if processor.piece_to_id(tag) == processor.unk_id():
        raise ConfigError(f"the tag {tag!r} is not a piece of the model {spm_model}")
    source_lines = _read_source_lines(src_file)

    out_file.unlink(missing_ok=True)
    raw_lines = eole_run.predict_tokens(
        ckpt_dir,
        [
            " ".join([tag, *processor.encode(line, out_type=str)])
            for line in source_lines
        ],
        out_dir=out_file.parent,
        stem=out_file.name,
        threads=threads,
        timeout=timeout,
    )
    # Pieces hold no U+0020, so splitting at it restores them; `str.split()` would
    # also cut at the suffix gap U+202F, which is a piece of its own.
    write_lines(out_file, [processor.decode(line.split(" ")) for line in raw_lines])
    return out_file


def _read_source_lines(src_file: Path) -> list[str]:
    try:
        text = src_file.read_text(encoding="utf-8", newline="\n")
    except (OSError, UnicodeDecodeError) as exc:
        raise DataError(f"cannot read source file {src_file}: {exc}") from exc
    lines = split_lines(text)
    if not lines:
        raise DataError(f"source file {src_file} has no lines")
    return lines
