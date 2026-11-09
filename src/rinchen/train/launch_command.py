"""The `command.sh` of a launch: the commands an operator runs by hand.

`rinchen train launch` writes it for the attempt it is about to run, or, with
`--write-only`, without running anything (for example the 8-GPU command to start under
`nohup` or in a terminal multiplexer). The text is a pure function of its arguments, so
it is golden-tested; every path in it is absolute, so it runs from anywhere (the
`cd` line is for the relative paths a config may hold).

The script fixes the state of the run at the moment it is written: which checkpoint it
resumes from is in the config it names. `launch.json`, the record of the run that lets
`rinchen train launch` take over a run started by hand with this script, is written by
the launcher beside it, not by the script. The script appends the output of `eole train`
to the attempt log (`launch/attempt-<k>.log`, through `tee -a`, with `pipefail` so that the
exit status stays Eole's), so the `rinchen train launch` that later resumes or finishes
the run reads a NaN or an early stop in a hand-run attempt exactly as it reads its own.
"""

from __future__ import annotations

import json
import shlex
from pathlib import Path
from typing import Final

_SHEBANG: Final = "#!/usr/bin/env bash"


def render_command_sh(
    *,
    run_dir: Path,
    working_directory: Path,
    eole: str,
    config: Path,
    src_vocab: Path,
    attempt_log: Path,
    build_vocab_log: Path,
    threads: int,
    cuda_visible_devices: str | None,
    attempt: int,
    resumes_from: int | None,
    continues: Path | None = None,
) -> str:
    """The text of `command.sh` for attempt `attempt` of the run in `run_dir`.

    `config` is the rendered config of that attempt (it holds `train_from` when
    `resumes_from`, the step it resumes from, is set, or `continues`, the checkpoint of
    ours the first attempt of a continuation starts from: an attempt does one or the
    other), `eole` the `eole` command and
    `src_vocab` the vocabulary file: when it is missing the script builds it first.
    The output of `eole train` is appended to `attempt_log` and that of `eole
    build_vocab` to `build_vocab_log` (`mkdir -p` makes their directory first); the
    output still reaches the terminal.
    The script exports `threads` as the OpenMP and MKL thread count, runs from
    `working_directory`, and exports `CUDA_VISIBLE_DEVICES` as `cuda_visible_devices`
    (empty hides every GPU) unless that is None, which leaves the variable to the
    environment the script runs in.

    Raises:
        ValueError: a path (`eole` and `continues` included) is not absolute, `threads`
            is below 1, or both `resumes_from` and `continues` are given.
    """
    if threads < 1:
        raise ValueError(f"threads must be at least 1, got {threads}")
    if resumes_from is not None and continues is not None:
        raise ValueError("an attempt either resumes or continues, not both")
    for name, path in (
        ("run_dir", run_dir),
        ("working_directory", working_directory),
        ("eole", Path(eole)),
        ("config", config),
        ("src_vocab", src_vocab),
        ("attempt_log", attempt_log),
        ("build_vocab_log", build_vocab_log),
        *(() if continues is None else (("continues", continues),)),
    ):
        if not path.is_absolute():
            raise ValueError(f"{name} must be an absolute path, got {str(path)!r}")
    command = shlex.quote(eole)
    config_argument = shlex.quote(str(config))
    # json.dumps, as for the run directory: a line break stays inside the comment.
    log_in_run = json.dumps(f"{attempt_log.parent.name}/{attempt_log.name}")
    lines = [
        _SHEBANG,
        # json.dumps: a line break in the path stays inside the comment.
        (
            f"# Written by `rinchen train launch` for attempt {attempt} of the run in "
            f"{json.dumps(str(run_dir))}."
        ),
        _origin(resumes_from, continues),
        "# What it resumes from is fixed at that moment: write it again with",
        "# `rinchen train launch --write-only` after the run changed.",
        (
            "# launch.json beside it records the run, so `rinchen train launch` can "
            "resume a run started"
        ),
        (
            f"# with this script. The output of `eole train` is appended to {log_in_run}, "
            "which"
        ),
        (
            "# `rinchen train launch` reads for a NaN or an early stop when it takes the "
            "run over."
        ),
        "set -euo pipefail",
        f"export OMP_NUM_THREADS={threads}",
        f"export MKL_NUM_THREADS={threads}",
    ]
    if cuda_visible_devices is not None:
        lines.append(f"export CUDA_VISIBLE_DEVICES={shlex.quote(cuda_visible_devices)}")
    log_directories = {
        shlex.quote(str(log.parent)) for log in (build_vocab_log, attempt_log)
    }
    lines += [
        f"cd {shlex.quote(str(working_directory))}",
        "mkdir -p " + " ".join(sorted(log_directories)),
        (
            f"[ -f {shlex.quote(str(src_vocab))} ] || {command} build_vocab "
            f"-config {config_argument} -n_sample -1 2>&1 | "
            f"tee -a {shlex.quote(str(build_vocab_log))}"
        ),
        (
            f"{command} train -config {config_argument} 2>&1 | "
            f"tee -a {shlex.quote(str(attempt_log))}"
        ),
    ]
    return "\n".join(lines) + "\n"


def _origin(resumes_from: int | None, continues: Path | None) -> str:
    if continues is not None:
        # json.dumps, as for the run directory: a line break stays inside the comment.
        return (
            f"# It continues {json.dumps(str(continues))} (training.train_from of the "
            "config it names, with reset_optim: all: its weights, a fresh optimizer)."
        )
    if resumes_from is None:
        return (
            "# It starts the run from scratch: no checkpoint of it was complete when "
            "this was written."
        )
    return (
        f"# It resumes from step {resumes_from} (training.train_from of the config it "
        "names)."
    )
