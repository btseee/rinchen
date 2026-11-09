"""The CTranslate2 export of an Eole checkpoint.

`export_ct2` converts a checkpoint with CTranslate2's own converter in a subprocess. It
started in D5, for the script converter, and moved here in D10 so that `rinchen.train` can
export a teacher without importing `script_conv`; `rinchen.script_conv.export` imports the
names back. D11 added the quantisation. This module imports nothing of `train`, `synth`,
`cli`, `script_conv`, `infer` or `serve`, and not CTranslate2 itself: the converter runs as a
process of its own.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from rinchen.utils.logged_process import failure_with_log_tail, run_logged

_MODEL_FILE = "model.bin"


def export_environment() -> dict[str, str]:
    """A copy of the process environment for the CTranslate2 converter, with no CUDA
    device visible. With one visible the converter reads a device pointer as host
    memory, segfaults (exit code -11) and leaves an empty `model.bin`."""
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = ""
    return env


def export_log_path(out_dir: Path) -> Path:
    """The log of the export to `out_dir`: `<out_dir>.export.log`, beside the
    directory because the converter refuses a directory that exists."""
    return Path(f"{out_dir}.export.log")


def export_ct2(
    ckpt_dir: Path,
    out_dir: Path,
    *,
    timeout: float | None = None,
    log_path: Path | None = None,
    reported_dir: Path | None = None,
    quantization: str | None = None,
) -> Path:
    """Convert the Eole checkpoint in `ckpt_dir` to a CTranslate2 model in `out_dir`
    (which must not exist yet) and return `out_dir`.

    The converter runs as `python -I -W ignore -m ctranslate2.converters.eole_ct2`
    with `export_environment()`; its combined output goes to `log_path`, by default
    `export_log_path(out_dir)`. `timeout` (seconds) bounds it; a converter that runs past
    it is killed with everything it started. A failed export can leave a partial
    `out_dir`, which must be removed before retrying: the converter refuses an existing
    directory. A caller that writes to a private directory it will rename names the final
    one as `reported_dir`, which is the path the message about a missing `model.bin`
    gives (default `out_dir`): the private one would be gone when the message is read.
    `quantization` is passed to the converter as `--quantization <value>` (the converter
    accepts `int8`, `int8_float32`, `int8_float16`, `int8_bfloat16`, `int16`, `float16`,
    `bfloat16` and `float32`; the value is its to accept or refuse); None appends nothing
    and the converter keeps the type the checkpoint stores.

    Raises:
        UpstreamError: the converter exits non-zero or runs past `timeout`, or exits
            zero but leaves no `model.bin` or an empty one; the message ends with the
            last lines of the log.
    """
    if log_path is None:
        log_path = export_log_path(out_dir)
    command = [
        sys.executable,
        "-I",
        "-W",
        "ignore",
        "-m",
        "ctranslate2.converters.eole_ct2",
        "--model_path",
        str(ckpt_dir),
        "--output_dir",
        str(out_dir),
    ]
    if quantization is not None:
        command += ["--quantization", quantization]
    run_logged(
        command,
        env=export_environment(),
        log_path=log_path,
        what="the CTranslate2 export",
        timeout=timeout,
    )
    model_file = out_dir / _MODEL_FILE
    if not model_file.is_file() or model_file.stat().st_size == 0:
        raise failure_with_log_tail(
            f"the CTranslate2 export exited with exit code 0 but left no {_MODEL_FILE} "
            f"in {out_dir if reported_dir is None else reported_dir}, or an empty one",
            log_path,
        )
    return out_dir
