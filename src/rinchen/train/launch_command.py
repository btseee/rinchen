"""The `command.sh` of a launch: the commands an operator runs by hand.

`rinchen train launch` writes it for the attempt it is about to run, or, with
`--write-only`, without running anything (for example the 8-GPU command to start under
`nohup` or in a terminal multiplexer). The text is a pure function of its arguments, so
it is golden-tested; every path in it is absolute, so it runs from anywhere (the
`cd` line is for the relative paths a config may hold).

The script fixes the state of the run at the moment it is written: which checkpoint it
resumes from is in the config it names. It writes no `launch.json` and looks at no log,
so a NaN in the loss is for the `rinchen train launch` that resumes or finishes the run
to find.
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
    threads: int,
    cuda_visible_devices: str | None,
    attempt: int,
    resumes_from: int | None,
) -> str:
    """The text of `command.sh` for attempt `attempt` of the run in `run_dir`.

    `config` is the rendered config of that attempt (it holds `train_from` when
    `resumes_from`, the step it resumes from, is set), `eole` the `eole` command and
    `src_vocab` the vocabulary file: when it is missing the script builds it first.
    The script exports `threads` as the OpenMP and MKL thread count, runs from
    `working_directory`, and exports `CUDA_VISIBLE_DEVICES` as `cuda_visible_devices`
    (empty hides every GPU) unless that is None, which leaves the variable to the
    environment the script runs in.

    Raises:
        ValueError: a path (`eole` included) is not absolute, or `threads` is below 1.
    """
    if threads < 1:
        raise ValueError(f"threads must be at least 1, got {threads}")
    for name, path in (
        ("run_dir", run_dir),
        ("working_directory", working_directory),
        ("eole", Path(eole)),
        ("config", config),
        ("src_vocab", src_vocab),
    ):
        if not path.is_absolute():
            raise ValueError(f"{name} must be an absolute path, got {str(path)!r}")
    command = shlex.quote(eole)
    config_argument = shlex.quote(str(config))
    lines = [
        _SHEBANG,
        # json.dumps: a line break in the path stays inside the comment.
        (
            f"# Written by `rinchen train launch` for attempt {attempt} of the run in "
            f"{json.dumps(str(run_dir))}."
        ),
        _origin(resumes_from),
        "# What it resumes from is fixed at that moment: write it again with",
        "# `rinchen train launch --write-only` after the run changed.",
        (
            "# Unlike `rinchen train launch` it writes no launch.json and does not "
            "look for NaN."
        ),
        "set -euo pipefail",
        f"export OMP_NUM_THREADS={threads}",
        f"export MKL_NUM_THREADS={threads}",
    ]
    if cuda_visible_devices is not None:
        lines.append(f"export CUDA_VISIBLE_DEVICES={shlex.quote(cuda_visible_devices)}")
    lines += [
        f"cd {shlex.quote(str(working_directory))}",
        (
            f"[ -f {shlex.quote(str(src_vocab))} ] || {command} build_vocab "
            f"-config {config_argument} -n_sample -1"
        ),
        f"{command} train -config {config_argument}",
    ]
    return "\n".join(lines) + "\n"


def _origin(resumes_from: int | None) -> str:
    if resumes_from is None:
        return (
            "# It starts the run from scratch: no checkpoint of it was complete when "
            "this was written."
        )
    return (
        f"# It resumes from step {resumes_from} (training.train_from of the config it "
        "names)."
    )
