"""A child Python process for the tests of the tiny pipeline that compare a result across processes
and hash seeds (D15).

`run_python` runs a program text with the interpreter of the tests, the offline switches and the
hidden device in the environment of the call, no secret, and the hash seed it is given, and returns
what the program printed. The program imports `rinchen` the way the test does (the parent's
`PYTHONPATH` is kept), reads no network and writes nothing.
"""

from __future__ import annotations

import os
import subprocess
import sys
from typing import Final

SECRET_SUFFIXES: Final = ("_TOKEN", "_SECRET", "_PASSWORD", "_API_KEY")
TIMEOUT_SECONDS: Final = 120


def child_environment(hash_seed: str) -> dict[str, str]:
    """The parent's environment without its secrets, with `PYTHONHASHSEED` set, the device hidden
    and the package index and the Hub offline."""
    return {
        **{k: v for k, v in os.environ.items() if not k.endswith(SECRET_SUFFIXES)},
        "PYTHONHASHSEED": hash_seed,
        "CUDA_VISIBLE_DEVICES": "",
        "UV_OFFLINE": "1",
        "HF_HUB_OFFLINE": "1",
        "HF_DATASETS_OFFLINE": "1",
    }


def run_python(code: str, *args: str, hash_seed: str = "0") -> str:
    """The standard output of `python -c code args...` under `hash_seed`; a non-zero exit fails the
    calling test with the child's standard error."""
    result = subprocess.run(
        [sys.executable, "-c", code, *args],
        env=child_environment(hash_seed),
        capture_output=True,
        text=True,
        timeout=TIMEOUT_SECONDS,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    return result.stdout
