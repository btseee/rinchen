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
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

from rinchen.errors import ConfigError, UpstreamError
from rinchen.script_conv._subprocess import run_logged
from rinchen.script_conv.config import dump_config, validate_converter_config

_PREDICT_BATCH_SIZE = 64
# Eole keeps its newest checkpoint under this name in `model_path` (a link to the file
# in a step directory), so a run that trained leaves it there with a fresh
# modification time.
_CHECKPOINT_FILE = "model.00.safetensors"
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
    cfg_path: Path, run_dir: Path, *, threads: int = 2, timeout: float | None = None
) -> Path:
    """Build the vocabulary, then train, as `eole build_vocab` and `eole train` on the
    config at `cfg_path`; return the checkpoint directory, the config's
    `training.model_path`.

    `cfg_path` is a file `render_converter_config(..., run_dir=run_dir)` rendered.
    `run_dir` is created and takes `build_vocab.log` and `train.log` (standard output
    and error of each command). The GPU is hidden from both commands when the config
    trains on none (`training.gpu_ranks` empty). `timeout` (seconds) bounds each
    command; a command that runs past it is killed with everything it started.

    Raises:
        ConfigError: the config file is missing, unparseable or fails
            `validate_converter_config` (checked before anything runs); its
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
    validate_converter_config(cfg)
    model_path = _model_path_inside(cfg, run_dir)
    eole = _eole_command()
    training = cfg["training"]
    trains_on_gpu = isinstance(training, Mapping) and bool(training.get("gpu_ranks"))
    env = training_environment(threads, hide_gpu=not trains_on_gpu)

    run_dir.mkdir(parents=True, exist_ok=True)

    def run_eole(subcommand: str, *options: str) -> None:
        run_logged(
            [eole, subcommand, "-config", str(cfg_path), *options],
            env=env,
            log_path=run_dir / f"{subcommand}.log",
            what=f"eole {subcommand}",
            timeout=timeout,
        )

    run_eole("build_vocab", "-n_sample", "-1")  # -1: count the whole corpus
    launched = time.time()
    run_eole("train")
    if not _written_since(
        model_path / _CHECKPOINT_FILE, launched - _MTIME_SLACK_SECONDS
    ):
        raise UpstreamError(
            f"eole train exited with code 0 but wrote no new checkpoint under "
            f"{model_path}: expected a fresh {_CHECKPOINT_FILE}; see "
            f"{run_dir / 'train.log'}"
        )
    return model_path


def predict_side_paths(out_file: Path) -> tuple[Path, Path]:
    """The files `predict` writes next to `out_file`: its log and its prediction
    config (`<out_file>.log` and `<out_file>.predict.yaml`)."""
    return Path(f"{out_file}.log"), Path(f"{out_file}.predict.yaml")


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
    """Convert the lines of `src_file` with the checkpoint in `ckpt_dir`, greedily and
    on the CPU, as `eole predict`; return `out_file`.

    The log and the prediction config go next to the output (`predict_side_paths`).
    Eole puts `tag` in front of each source line after the SentencePiece model at
    `spm_model` has split it, as in training. A file already at `out_file` is removed
    first. `timeout` (seconds) bounds the command; one that runs past it is killed
    with everything it started.

    Raises:
        ConfigError: the prediction config (every path and the tag) holds `$`, which
            Eole would expand as a variable.
        ValueError: `threads` is less than 1.
        UpstreamError: `eole predict` exits non-zero or runs past `timeout` (the
            message ends with the last lines of its log), the `eole` command is
            missing, or it exits zero without writing `out_file`.
    """
    cfg = {
        "model_path": ckpt_dir.as_posix(),
        "src": src_file.as_posix(),
        "output": out_file.as_posix(),
        "beam_size": 1,
        "batch_size": _PREDICT_BATCH_SIZE,
        "batch_type": "sents",
        "gpu_ranks": [],
        "world_size": 1,
        "transforms": ["sentencepiece", "prefix"],
        "transforms_configs": {
            "sentencepiece": {
                "src_subword_model": spm_model.as_posix(),
                "tgt_subword_model": spm_model.as_posix(),
            },
            "prefix": {"src_prefix": tag, "tgt_prefix": ""},
        },
    }
    _reject_dollar("prediction config", yaml.safe_dump(cfg))
    eole = _eole_command()
    env = training_environment(threads, hide_gpu=True)

    log_path, cfg_path = predict_side_paths(out_file)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.unlink(missing_ok=True)
    dump_config(cfg, cfg_path)
    run_logged(
        [eole, "predict", "-config", str(cfg_path)],
        env=env,
        log_path=log_path,
        what="eole predict",
        timeout=timeout,
    )
    if not out_file.is_file():
        raise UpstreamError(
            f"eole predict exited with code 0 but wrote no output file {out_file}"
        )
    return out_file


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
