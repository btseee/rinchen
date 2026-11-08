"""Eole runs shared by every Rinchen model: vocabulary and training from a rendered
config, and prediction on token lines from a trained checkpoint.

Each command runs as a subprocess with its output in a log file. Training stays on
the CPU when the config asks for no GPU, even on a machine that has one, and the
thread count is set explicitly: both make a small run repeatable.
"""

from __future__ import annotations

import os
import shutil
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, NamedTuple

from rinchen.errors import ConfigError, UpstreamError
from rinchen.train.eole_cfg import (
    dump_config,
    read_config,
    reject_dollar,
    reject_glob_characters,
)
from rinchen.utils.logged_process import run_logged

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
    validate: Callable[[Any], None],
    threads: int = 2,
    timeout: float | None = None,
    what: str = "Eole config",
) -> Path:
    """Build the vocabulary, then train, as `eole build_vocab` and `eole train` on the
    config at `cfg_path`; return the checkpoint directory, the config's
    `training.model_path`.

    `run_dir` is created and takes `build_vocab.log` and `train.log` (standard output
    and error of each command). The GPU is hidden from both commands when the config
    trains on none (`training.gpu_ranks` empty or missing). `timeout` (seconds) bounds
    each command; a command that runs past it is killed with everything it started.

    `validate` is handed whatever the YAML file holds, before anything runs, and
    raises ConfigError when it is not a config its caller accepts. `what` names the
    config in the message about `training.model_path`.

    Raises:
        ConfigError: the config file is missing or unparseable, or `validate` refuses
            it; its `training.model_path` is missing or not inside `run_dir`; or its
            path, its text or `run_dir` holds `$`, which Eole would expand as a
            variable; or its `training.model_path` holds `[`, `]`, `*` or `?`, which
            Eole reads as a glob pattern when it loads a checkpoint from it.
        ValueError: `threads` is less than 1.
        UpstreamError: a command exits non-zero or runs past `timeout` (the message
            ends with the last lines of its log), the `eole` command is missing, or
            training exits zero without writing a checkpoint after it was launched.
    """
    reject_dollar("config path", cfg_path)
    reject_dollar("run directory", run_dir)
    # The file is an Eole config whoever's it is; `what` names it as its validator does.
    cfg = read_config(cfg_path, "Eole config")
    validate(cfg)
    model_path = _model_path_inside(cfg, run_dir, what)
    reject_glob_characters("checkpoint directory", model_path)
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
    if not written_since(model_path / CHECKPOINT_FILE, launched - _MTIME_SLACK_SECONDS):
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
    eole = eole_command()
    config = ("-config", str(cfg_path))
    # -n_sample -1: count the whole corpus.
    return (
        (eole, "build_vocab", *config, "-n_sample", "-1"),
        (eole, "train", *config),
    )


class PredictFiles(NamedTuple):
    """The files `predict_tokens` writes."""

    log: Path  # standard output and error of `eole predict`
    config: Path  # the prediction config
    token_source: Path  # the source lines as tokens, one line each
    token_target: Path  # the forced target prefixes, one line each (Eole's `tgt`)
    token_output: Path  # Eole's hypotheses as token lines


def predict_files(out_dir: Path, stem: str) -> PredictFiles:
    """The files `predict_tokens` writes in `out_dir` for `stem`: `<stem>.log`,
    `.predict.yaml`, `.tokens.src`, `.tokens.tgt` and `.tokens.out`. The target file
    exists only when prefixes were forced."""
    return PredictFiles(
        log=out_dir / f"{stem}.log",
        config=out_dir / f"{stem}.predict.yaml",
        token_source=out_dir / f"{stem}.tokens.src",
        token_target=out_dir / f"{stem}.tokens.tgt",
        token_output=out_dir / f"{stem}.tokens.out",
    )


def predict_tokens(
    ckpt_dir: Path,
    source_tokens: Sequence[str],
    *,
    out_dir: Path,
    forced_prefix_tokens: Sequence[str] | None = None,
    threads: int = 2,
    timeout: float | None = None,
    batch_size: int = 64,
    stem: str = "eole",
    stale_files: Sequence[Path] = (),
) -> list[str]:
    """Translate the token lines `source_tokens` (pieces joined by U+0020, a direction
    tag among them if the model wants one) with the checkpoint in `ckpt_dir`, greedily
    and on the CPU, as `eole predict`; return Eole's output token lines, exactly one
    per source line (an empty hypothesis is an empty line).

    Eole does not tokenise or tag here: the caller does, and Eole reads and writes
    token lines. Its own `sentencepiece` and `prefix` transforms stay off
    (`transforms: []`): Eole 0.6.4 copies the `valid` corpus's prefix into the
    checkpoint, which then overrides the prediction config's transforms, so every
    prediction would run under the validation corpus's tag.

    With `forced_prefix_tokens` (one line per source line, the pieces the output must
    start with) Eole is given them as its `tgt` file with `tgt_file_prefix: true`, and
    each returned line starts with its forced prefix.

    The files go in `out_dir` (`predict_files(out_dir, stem)`); an output of an
    earlier run is removed first. `stale_files` are results of an earlier run that
    this one replaces, in the caller's own places: each is removed if it exists, once
    every check has passed and `out_dir` exists and before Eole runs, so a call
    refused up to there leaves them alone and a run that fails leaves none.
    `timeout` (seconds) bounds the command; one that runs past it is killed with
    everything it started.

    Raises:
        ConfigError: `ckpt_dir`, `out_dir` or `stem` holds `$`, which Eole would
            expand as a variable in the prediction config; `ckpt_dir` holds `[`, `]`,
            `*` or `?`, which Eole reads as a glob pattern when it loads the
            checkpoint (an output directory with them is fine).
        ValueError: `threads` is less than 1; `source_tokens` is empty;
            `forced_prefix_tokens` has another number of lines than `source_tokens`;
            or a forced prefix, or a source line when prefixes are forced, is empty or
            only whitespace (Eole drops such an example without a word; the message
            names the first such line, counted from 1, forced prefix before source).
            An empty source line is fine when no prefix is forced.
        UpstreamError: `eole predict` exits non-zero or runs past `timeout` (the
            message ends with the last lines of its log), the `eole` command is
            missing, or it exits zero without writing its output or with another
            number of lines than the source has.
    """
    if forced_prefix_tokens is not None and len(forced_prefix_tokens) != len(
        source_tokens
    ):
        raise ValueError(
            f"{len(forced_prefix_tokens)} forced prefixes for "
            f"{len(source_tokens)} source lines"
        )
    if not source_tokens:
        raise ValueError("no source lines to translate")
    if forced_prefix_tokens is not None:
        _require_lines_eole_keeps(source_tokens, forced_prefix_tokens)
    for name, value in (
        ("checkpoint directory", ckpt_dir),
        ("output directory", out_dir),
        ("file stem", stem),
    ):
        reject_dollar(name, value)
    reject_glob_characters("checkpoint directory", ckpt_dir)
    eole = eole_command()
    env = training_environment(threads, hide_gpu=True)
    files = predict_files(out_dir, stem)
    cfg: dict[str, Any] = {
        "model_path": ckpt_dir.as_posix(),
        "src": files.token_source.as_posix(),
        "output": files.token_output.as_posix(),
        "beam_size": 1,
        "batch_size": batch_size,
        "batch_type": "sents",
        "gpu_ranks": [],
        "world_size": 1,
        "transforms": [],
    }
    if forced_prefix_tokens is not None:
        cfg["tgt"] = files.token_target.as_posix()
        cfg["tgt_file_prefix"] = True

    out_dir.mkdir(parents=True, exist_ok=True)
    for stale in stale_files:
        stale.unlink(missing_ok=True)
    files.token_output.unlink(missing_ok=True)
    write_lines(files.token_source, source_tokens)
    if forced_prefix_tokens is not None:
        write_lines(files.token_target, forced_prefix_tokens)
    dump_config(cfg, files.config)
    run_logged(
        [eole, "predict", "-config", str(files.config)],
        env=env,
        log_path=files.log,
        what="eole predict",
        timeout=timeout,
    )
    if not files.token_output.is_file():
        raise UpstreamError(
            f"eole predict exited with code 0 but wrote no output file "
            f"{files.token_output}"
        )
    output_lines = split_lines(
        files.token_output.read_text(encoding="utf-8", newline="\n")
    )
    if len(output_lines) != len(source_tokens):
        raise UpstreamError(
            f"eole predict wrote {len(output_lines)} lines for {len(source_tokens)} "
            f"source lines: {files.token_output}"
        )
    return output_lines


def _require_lines_eole_keeps(
    source_tokens: Sequence[str], forced_prefix_tokens: Sequence[str]
) -> None:
    # Eole drops, without a word, an example whose source or forced prefix has no
    # token, and the line count check after the run would then blame the output.
    pairs = zip(source_tokens, forced_prefix_tokens, strict=True)
    for number, (source, prefix) in enumerate(pairs, start=1):
        if not prefix.strip():
            raise ValueError(
                f"forced prefix for source line {number} is empty or only whitespace"
            )
        if not source.strip():
            raise ValueError(
                f"source line {number} is empty or only whitespace, which Eole drops "
                "when prefixes are forced"
            )


def split_lines(text: str) -> list[str]:
    """The lines of `text`, split at LF only (other line separators belong to the
    line), each ended by an LF; no text is no lines."""
    if not text:
        return []
    return text.removesuffix("\n").split("\n")


def write_lines(path: Path, lines: Sequence[str]) -> None:
    """Write `lines` to `path`, each ended by an LF, as UTF-8."""
    path.write_text(
        "".join(f"{line}\n" for line in lines), encoding="utf-8", newline="\n"
    )


def _model_path_inside(cfg: Any, run_dir: Path, what: str) -> Path:
    """The config's `training.model_path`, which must lie inside `run_dir` (so the
    run keeps everything it writes in one directory)."""
    training = cfg.get("training") if isinstance(cfg, Mapping) else None
    model_path = training.get("model_path") if isinstance(training, Mapping) else None
    if not isinstance(model_path, str) or not model_path:
        raise ConfigError(
            f"invalid {what}: training.model_path: must be a non-empty path"
        )
    resolved = Path(model_path).resolve()
    root = run_dir.resolve()
    if resolved == root or not resolved.is_relative_to(root):
        raise ConfigError(
            f"invalid {what}: training.model_path: {model_path!r} must be inside "
            f"the run directory {run_dir}"
        )
    return Path(model_path)


def written_since(path: Path, since: float) -> bool:
    """Whether `path` exists (a link to a file that exists) and was modified at or
    after `since`, in seconds since the epoch."""
    try:
        return path.stat().st_mtime >= since
    except OSError:  # missing, or a link that points nowhere
        return False


def eole_command() -> str:
    """The `eole` console script next to the interpreter of the environment this code
    runs in; raises UpstreamError when it is not installed there."""
    # Eole has no `__main__`, so `python -m eole` does not run.
    bin_dir = Path(sys.executable).parent
    command = shutil.which("eole", path=str(bin_dir))
    if command is None:
        raise UpstreamError(f"the `eole` command is not installed in {bin_dir}")
    return command
