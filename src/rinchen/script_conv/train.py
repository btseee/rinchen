"""Eole runs of the converter: vocabulary and training from a rendered config, and
prediction from a trained checkpoint.

Each command runs as a subprocess with its output in a log file. Training stays on
the CPU when the config asks for no GPU, even on a machine that has one, and the
thread count is set explicitly: both make the tiny run repeatable.
"""

from __future__ import annotations

import os
import shutil
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, NamedTuple

import yaml

from rinchen.errors import ConfigError, DataError, UpstreamError
from rinchen.script_conv._subprocess import run_logged
from rinchen.script_conv.char_model import CONVERTER_TAGS, load_char_processor
from rinchen.script_conv.config import Profile, dump_config, validate_converter_config

_PREDICT_BATCH_SIZE = 64
# Eole keeps its newest checkpoint under this name in `model_path` (a link to the file
# in a step directory), so a run that trained leaves it there with a fresh
# modification time.
CHECKPOINT_FILE = "model.00.safetensors"
# File systems with coarse timestamps can round a modification time down.
_MTIME_SLACK_SECONDS = 2.0


def training_environment(threads: int, *, hide_gpu: bool) -> dict[str, str]:
    """A copy of the process environment for an Eole subprocess: OpenMP and MKL run
    `threads` threads, and no CUDA device is visible when `hide_gpu`. Raises ValueError
    for fewer than one thread."""
    if threads < 1:
        raise ValueError(f"threads must be at least 1, got {threads}")
    env = dict(os.environ)
    env["OMP_NUM_THREADS"] = str(threads)
    env["MKL_NUM_THREADS"] = str(threads)
    if hide_gpu:
        env["CUDA_VISIBLE_DEVICES"] = ""
    return env


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
    _reject_dollar("config path", cfg_path)
    _reject_dollar("run directory", run_dir)
    cfg = _load_config(cfg_path)
    validate_converter_config(cfg, profile=profile)
    model_path = _model_path_inside(cfg, run_dir)
    build_vocab_command, train_command = eole_commands(cfg_path)
    training = cfg["training"]
    trains_on_gpu = isinstance(training, Mapping) and bool(training.get("gpu_ranks"))
    env = training_environment(threads, hide_gpu=not trains_on_gpu)

    run_dir.mkdir(parents=True, exist_ok=True)

    def run_eole(command: tuple[str, ...]) -> None:
        subcommand = command[1]
        run_logged(
            list(command),
            env=env,
            log_path=run_dir / f"{subcommand}.log",
            what=f"eole {subcommand}",
            timeout=timeout,
        )

    run_eole(build_vocab_command)
    launched = time.time()
    run_eole(train_command)
    if not _written_since(
        model_path / CHECKPOINT_FILE, launched - _MTIME_SLACK_SECONDS
    ):
        raise UpstreamError(
            f"eole train exited with code 0 but wrote no new checkpoint under "
            f"{model_path}: expected a fresh {CHECKPOINT_FILE}; see "
            f"{run_dir / 'train.log'}"
        )
    return model_path


def eole_commands(cfg_path: Path) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """The commands `run_training` runs on the config at `cfg_path`, as argument
    vectors: `eole build_vocab` (counting the whole corpus), then `eole train`.

    Raises:
        UpstreamError: the `eole` command is not installed next to the interpreter.
    """
    eole = _eole_command()
    config = ("-config", str(cfg_path))
    # -n_sample -1: count the whole corpus.
    return (
        (eole, "build_vocab", *config, "-n_sample", "-1"),
        (eole, "train", *config),
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
    return PredictSidePaths(
        log=Path(f"{out_file}.log"),
        config=Path(f"{out_file}.predict.yaml"),
        token_source=Path(f"{out_file}.tokens.src"),
        token_output=Path(f"{out_file}.tokens.out"),
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
    side = predict_side_paths(out_file)
    cfg = {
        "model_path": ckpt_dir.as_posix(),
        "src": side.token_source.as_posix(),
        "output": side.token_output.as_posix(),
        "beam_size": 1,
        "batch_size": _PREDICT_BATCH_SIZE,
        "batch_type": "sents",
        "gpu_ranks": [],
        "world_size": 1,
        "transforms": [],
    }
    # `out_file` is enough for the token files in `cfg`: they only add a suffix to it.
    for name, value in (("checkpoint directory", ckpt_dir), ("output file", out_file)):
        _reject_dollar(name, value)
    processor = load_char_processor(spm_model)
    if processor.piece_to_id(tag) == processor.unk_id():
        raise ConfigError(f"the tag {tag!r} is not a piece of the model {spm_model}")
    source_lines = _read_source_lines(src_file)
    eole = _eole_command()
    env = training_environment(threads, hide_gpu=True)

    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.unlink(missing_ok=True)
    side.token_output.unlink(missing_ok=True)
    _write_lines(
        side.token_source,
        [
            " ".join([tag, *processor.encode(line, out_type=str)])
            for line in source_lines
        ],
    )
    dump_config(cfg, side.config)
    run_logged(
        [eole, "predict", "-config", str(side.config)],
        env=env,
        log_path=side.log,
        what="eole predict",
        timeout=timeout,
    )
    if not side.token_output.is_file():
        raise UpstreamError(
            f"eole predict exited with code 0 but wrote no output file "
            f"{side.token_output}"
        )
    raw_lines = _split_lines(
        side.token_output.read_text(encoding="utf-8", newline="\n")
    )
    if len(raw_lines) != len(source_lines):
        raise UpstreamError(
            f"eole predict wrote {len(raw_lines)} lines for {len(source_lines)} "
            f"source lines: {side.token_output}"
        )
    # Pieces hold no U+0020, so splitting at it restores them; `str.split()` would
    # also cut at the suffix gap U+202F, which is a piece of its own.
    _write_lines(out_file, [processor.decode(line.split(" ")) for line in raw_lines])
    return out_file


def _read_source_lines(src_file: Path) -> list[str]:
    try:
        text = src_file.read_text(encoding="utf-8", newline="\n")
    except (OSError, UnicodeDecodeError) as exc:
        raise DataError(f"cannot read source file {src_file}: {exc}") from exc
    lines = _split_lines(text)
    if not lines:
        raise DataError(f"source file {src_file} has no lines")
    return lines


def _split_lines(text: str) -> list[str]:
    """The lines of `text`, split at LF only (other line separators belong to the
    line), each ended by an LF; no text is no lines."""
    if not text:
        return []
    return text.removesuffix("\n").split("\n")


def _write_lines(path: Path, lines: Sequence[str]) -> None:
    path.write_text(
        "".join(f"{line}\n" for line in lines), encoding="utf-8", newline="\n"
    )


def _reject_dollar(what: str, value: Path | str) -> None:
    # Eole runs `os.path.expandvars` over the text of a config file, so a `$NAME`
    # in a path would be silently replaced by an environment variable's value.
    if "$" in str(value):
        raise ConfigError(
            f"the {what} holds '$', which Eole would expand as a variable"
        )


def _load_config(cfg_path: Path) -> Any:
    """What the YAML file holds, which `validate_converter_config` then checks is a
    mapping of the right shape."""
    if not cfg_path.is_file():
        raise ConfigError(f"Eole config file not found: {cfg_path}")
    try:
        text = cfg_path.read_text(encoding="utf-8")
        _reject_dollar(f"text of the config {cfg_path}", text)
        return yaml.safe_load(text)
    except (yaml.YAMLError, UnicodeDecodeError, OSError) as exc:
        raise ConfigError(
            f"invalid Eole config {cfg_path}: cannot read or parse: {exc}"
        ) from exc


def _model_path_inside(cfg: Mapping[str, Any], run_dir: Path) -> Path:
    """The config's `training.model_path`, which must lie inside `run_dir` (so the
    run keeps everything it writes in one directory)."""
    training = cfg["training"]
    model_path = training.get("model_path") if isinstance(training, Mapping) else None
    if not isinstance(model_path, str) or not model_path:
        raise ConfigError(
            "invalid converter config: training.model_path: must be a non-empty path"
        )
    resolved = Path(model_path).resolve()
    root = run_dir.resolve()
    if resolved == root or not resolved.is_relative_to(root):
        raise ConfigError(
            f"invalid converter config: training.model_path: {model_path!r} must be "
            f"inside the run directory {run_dir}"
        )
    return Path(model_path)


def _written_since(path: Path, since: float) -> bool:
    try:
        return path.stat().st_mtime >= since
    except OSError:  # missing, or a link that points nowhere
        return False


def _eole_command() -> str:
    # Eole has no `__main__`, so `python -m eole` does not run; its console script
    # sits next to the interpreter of the environment this code runs in.
    bin_dir = Path(sys.executable).parent
    command = shutil.which("eole", path=str(bin_dir))
    if command is None:
        raise UpstreamError(f"the `eole` command is not installed in {bin_dir}")
    return command
