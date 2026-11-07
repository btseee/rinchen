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
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

from rinchen.errors import ConfigError, UpstreamError
from rinchen.script_conv._subprocess import run_logged
from rinchen.script_conv.config import dump_config, validate_converter_config

# Each Eole subcommand with its extra options; `-n_sample -1` counts the whole corpus.
_TRAINING_COMMANDS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("build_vocab", ("-n_sample", "-1")),
    ("train", ()),
)
_PREDICT_BATCH_SIZE = 64
_CKPT_NAME = "ckpt"  # the `model_path` leaf `render_converter_config` writes


def training_environment(threads: int, *, hide_gpu: bool) -> dict[str, str]:
    """A copy of the process environment for an Eole subprocess: OpenMP and MKL run
    `threads` threads, and no CUDA device is visible when `hide_gpu`."""
    env = dict(os.environ)
    env["OMP_NUM_THREADS"] = str(threads)
    env["MKL_NUM_THREADS"] = str(threads)
    if hide_gpu:
        env["CUDA_VISIBLE_DEVICES"] = ""
    return env


def run_training(cfg_path: Path, run_dir: Path, *, threads: int = 2) -> Path:
    """Build the vocabulary, then train, as `eole build_vocab` and `eole train` on the
    config at `cfg_path`; return the checkpoint directory, `<run_dir>/ckpt`.

    `cfg_path` is a file `render_converter_config(..., run_dir=run_dir)` rendered:
    the checkpoint lands where that config's `model_path` says, which must be
    `<run_dir>/ckpt`. `run_dir` is created and takes `build_vocab.log` and
    `train.log` (standard output and error of each command). The GPU is hidden from
    both commands when the config trains on none (`training.gpu_ranks` empty).

    Raises:
        ConfigError: the config file is missing, unparseable or fails
            `validate_converter_config` (checked before anything runs), or its path,
            its text or `run_dir` holds `$`, which Eole would expand as a variable.
        UpstreamError: a command exits non-zero (the message ends with the last
            lines of its log), the `eole` command is missing, or training exits zero
            without a checkpoint directory.
    """
    _reject_dollar("config path", cfg_path)
    _reject_dollar("run directory", run_dir)
    cfg = _load_config(cfg_path)
    validate_converter_config(cfg)
    eole = _eole_command()

    run_dir.mkdir(parents=True, exist_ok=True)
    training = cfg["training"]
    trains_on_gpu = isinstance(training, Mapping) and bool(training.get("gpu_ranks"))
    env = training_environment(threads, hide_gpu=not trains_on_gpu)
    for subcommand, options in _TRAINING_COMMANDS:
        run_logged(
            [eole, subcommand, "-config", str(cfg_path), *options],
            env=env,
            log_path=run_dir / f"{subcommand}.log",
            what=f"eole {subcommand}",
        )

    ckpt_dir = run_dir / _CKPT_NAME
    if not ckpt_dir.is_dir():
        raise UpstreamError(
            f"eole train exited with code 0 but wrote no checkpoint directory "
            f"{ckpt_dir}: the config's training.model_path must be that path"
        )
    return ckpt_dir


def predict(
    ckpt_dir: Path,
    src_file: Path,
    *,
    tag: str,
    spm_model: Path,
    out_file: Path,
    threads: int = 2,
) -> Path:
    """Convert the lines of `src_file` with the checkpoint in `ckpt_dir`, greedily and
    on the CPU, as `eole predict`; return `out_file`.

    The prediction config goes next to the output as `<out_file>.predict.yaml` and
    the log as `<out_file>.log`. Eole puts `tag` in front of each source line after
    the SentencePiece model at `spm_model` has split it, as in training. A file
    already at `out_file` is removed first.

    Raises:
        ConfigError: a path holds `$`, which Eole would expand as a variable.
        UpstreamError: `eole predict` exits non-zero (the message ends with the last
            lines of its log), the `eole` command is missing, or it exits zero
            without writing `out_file`.
    """
    for name, path in (
        ("checkpoint directory", ckpt_dir),
        ("source file", src_file),
        ("SentencePiece model", spm_model),
        ("output file", out_file),
    ):
        _reject_dollar(name, path)
    eole = _eole_command()

    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.unlink(missing_ok=True)
    cfg_path = Path(f"{out_file}.predict.yaml")
    dump_config(
        {
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
        },
        cfg_path,
    )
    run_logged(
        [eole, "predict", "-config", str(cfg_path)],
        env=training_environment(threads, hide_gpu=True),
        log_path=Path(f"{out_file}.log"),
        what="eole predict",
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


def _eole_command() -> str:
    # Eole has no `__main__`, so `python -m eole` does not run; its console script
    # sits next to the interpreter of the environment this code runs in.
    bin_dir = Path(sys.executable).parent
    command = shutil.which("eole", path=str(bin_dir))
    if command is None:
        raise UpstreamError(f"the `eole` command is not installed in {bin_dir}")
    return command
