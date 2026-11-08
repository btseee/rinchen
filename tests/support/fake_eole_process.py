"""A stand-in for the `eole` command that is a real process, for the tests that run
`command.sh` with bash.

`FakeEole` replaces `run_logged` inside the test process; a script run by bash needs an
executable. `executable(path, ...)` writes a small program that calls `main`: for `eole
build_vocab` it writes the vocabulary file, for `eole train` it prints the log a run
leaves (`fake_launch.simulate_training`) on standard output and saves the step
directories, as the real command does. How the run goes is set by the environment variable
`FAKE_EOLE_MODE` (`ok`, `nan:<step>`, `early:<step>` or `interrupted:<step>`, which stops
without a final save) and its exit status by
`FAKE_EOLE_EXIT`; every call is appended to `$FAKE_EOLE_CALLS`.
"""

from __future__ import annotations

import os
import sys
from collections.abc import Sequence
from pathlib import Path

import yaml

from tests.support import step_dirs
from tests.support.fake_launch import Script, simulate_training

REPO_ROOT = Path(__file__).resolve().parents[2]


def executable(path: Path) -> Path:
    """Write the fake `eole` program to `path` (run by the interpreter of the tests) and
    return `path`."""
    path.write_text(
        f"#!{sys.executable}\n"
        "import sys\n"
        f"sys.path.insert(0, {str(REPO_ROOT)!r})\n"
        "from tests.support.fake_eole_process import main\n"
        "sys.exit(main(sys.argv[1:]))\n",
        encoding="utf-8",
    )
    path.chmod(0o755)
    return path


def main(arguments: Sequence[str]) -> int:
    """Run `eole <arguments>`; the exit status."""
    subcommand = arguments[0]
    cfg = yaml.safe_load(
        Path(arguments[arguments.index("-config") + 1]).read_text(encoding="utf-8")
    )
    calls = os.environ.get("FAKE_EOLE_CALLS")
    if calls is not None:
        with open(calls, "a", encoding="utf-8") as handle:
            handle.write(f"{subcommand}\n")
    if subcommand == "build_vocab":
        vocabulary = Path(cfg["src_vocab"])
        vocabulary.parent.mkdir(parents=True, exist_ok=True)
        vocabulary.write_text("a\nb\n", encoding="utf-8")
        return 0
    outcome = simulate_training(cfg, _script(os.environ.get("FAKE_EOLE_MODE", "ok")))
    print("\n".join(outcome.log_lines), flush=True)
    model_path = Path(cfg["training"]["model_path"])
    for step in outcome.saves:
        step_dirs.make_step_dir(model_path, step)
    return int(os.environ.get("FAKE_EOLE_EXIT", "0"))


def _script(mode: str) -> Script:
    kind, _, step = mode.partition(":")
    if kind == "nan":
        return Script(nan_from=int(step))
    if kind == "early":
        return Script(early_stop_at=int(step))
    if kind == "interrupted":
        return Script(to_step=int(step), skip_final_save=True)
    assert kind == "ok", mode
    return Script()
