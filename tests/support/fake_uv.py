"""The `uv` of the maintenance gate (D14, spec M19): an executable Python script named `uv` that the
gate puts first on `PATH`, so that the command of the nightly workflow runs for real while the one
step of a night that needs an index is answered by a hand-written sample.

What the shim does, and nothing else (every call is appended to a JSON-lines log beside it, with the
arguments it got, whether it was forwarded, the final argument list of a forwarded call and whether
it was refused):

1. `lock --upgrade --dry-run` (the two flags in any order, nothing else) prints the sample to
   standard error, exits with the code it was written with and is NEVER forwarded: the shim holds no
   path to an index, and this is the only call of a night that would reach one.
2. `lock --check` (with or without `--offline`) is forwarded as `<real uv> lock --check --offline`.
3. `--version` is forwarded unchanged.
4. `run <rest>` is forwarded as `<real uv> run --project <repository> --no-sync --frozen --offline
   <rest>`, after the leading `--frozen` and `--no-sync` of the caller are taken out of `<rest>`
   (uv refuses a flag given twice). `--project` keeps the working directory of the caller and
   finds the project's environment, so the gate can run the command in a temporary directory. Any
   other option before the command name is refused.
5. Every other argument list (`lock` without `--check` among them), and every list that holds
   `--upgrade`, `--upgrade-package`, `-U`, `-P`, `sync`, `add` or `pip` anywhere, exits 97 printing
   `refused`.

A forwarded call replaces the shim (`exec`), so a signal or a timeout reaches the real uv and its
children directly, and it runs with `UV_OFFLINE=1`, so that no uv of the call, a child of `run` included, can
reach an index. The shim imports nothing but the standard library and runs isolated (`-I`).
`write_uv_shim` takes the program to forward to as an argument: the gate passes the real uv, and the
tests of the shim itself pass a recorder.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Final

CALL_LOG: Final = "uv_calls.jsonl"
REFUSED_EXIT: Final = 97

# The program of the shim. `{constants}` is replaced by one JSON-written assignment per setting, so
# that nothing of the gate's environment is read when it runs.
_SHIM: Final = '''#!{python} -I
import json
import os
import sys

{constants}

REFUSED_WORDS = ("--upgrade", "-U", "-P", "sync", "add", "pip")
REFUSED_PREFIXES = ("--upgrade-package", "--upgrade=")
LEADING_RUN_OPTIONS = ("--frozen", "--no-sync")


def log(argv, forwarded, final, refused):
    entry = {{"argv": argv, "forwarded": forwarded, "final": final, "refused": refused}}
    with open(LOG, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry) + "\\n")


def refuse(argv):
    log(argv, False, None, True)
    sys.stderr.write("refused\\n")
    sys.exit({refused_exit})


def holds_refused_word(argv):
    return any(
        token in REFUSED_WORDS or token.startswith(REFUSED_PREFIXES) for token in argv
    )


def run_arguments(argv):
    """The command and its arguments after the caller's leading `--frozen` and `--no-sync`, or
    None when another option comes before the command name."""
    rest = argv[1:]
    seen = []
    while rest and rest[0].startswith("-"):
        if rest[0] not in LEADING_RUN_OPTIONS or rest[0] in seen:
            return None
        seen.append(rest[0])
        rest = rest[1:]
    return rest or None


def forward(argv, final):
    log(argv, True, final, False)
    os.environ["UV_OFFLINE"] = "1"
    os.execv(final[0], final)


def main(argv):
    flags = sorted(argv[1:])
    if argv[:1] == ["lock"] and flags == ["--dry-run", "--upgrade"]:
        log(argv, False, None, False)
        sys.stderr.write(DRY_RUN_OUTPUT)
        sys.exit(DRY_RUN_EXIT)
    if holds_refused_word(argv):
        refuse(argv)
    if argv[:1] == ["lock"] and flags in (["--check"], ["--check", "--offline"]):
        forward(argv, [REAL_UV, "lock", "--check", "--offline"])
    if argv == ["--version"]:
        forward(argv, [REAL_UV, "--version"])
    if argv[:1] == ["run"]:
        command = run_arguments(argv)
        if command is None or command[0].startswith("-"):
            refuse(argv)
        forward(
            argv,
            [REAL_UV, "run", "--project", PROJECT, "--no-sync", "--frozen", "--offline", *command],
        )
    refuse(argv)


main(sys.argv[1:])
'''


def write_uv_shim(
    directory: Path,
    *,
    real_uv: str,
    project: Path,
    dry_run_output: str,
    dry_run_exit: int = 0,
) -> Path:
    """Write the shim as `<directory>/uv` (executable) and an empty call log beside it, and return
    the path of the shim.

    `real_uv` is the program the shim forwards to and `project` the repository root that `run`
    names with `--project`; `dry_run_output` and `dry_run_exit` are what `lock --upgrade
    --dry-run` prints and exits with.
    """
    directory.mkdir(parents=True, exist_ok=True)
    settings: dict[str, Any] = {
        "REAL_UV": real_uv,
        "PROJECT": str(project),
        "DRY_RUN_OUTPUT": dry_run_output,
        "DRY_RUN_EXIT": dry_run_exit,
        "LOG": str(directory / CALL_LOG),
    }
    constants = "\n".join(
        f"{name} = {json.dumps(value)}" for name, value in settings.items()
    )
    path = directory / "uv"
    path.write_text(
        _SHIM.format(
            python=sys.executable, constants=constants, refused_exit=REFUSED_EXIT
        ),
        encoding="utf-8",
        newline="\n",
    )
    os.chmod(path, 0o755)
    (directory / CALL_LOG).write_text("", encoding="utf-8")
    return path


def read_calls(directory: Path) -> list[dict[str, Any]]:
    """The calls the shim in `directory` has logged, oldest first."""
    text = (directory / CALL_LOG).read_text(encoding="utf-8")
    return [json.loads(line) for line in text.splitlines() if line.strip()]
