"""The runner of the commands of `rinchen tiny run` (D15, spec P3): D14's `SubprocessRunner` with one difference,
found by the first real run of the pipeline.

The runner of the maintenance check kills the process group of a command that is interrupted with SIGKILL, at
once. That is right for a command whose helpers share its group, and wrong for `rinchen train launch`: Eole
leads a session of its own (`rinchen.utils.logged_process`), so the group of the launcher does not hold it, and
a launcher that dies of SIGKILL never gets to stop it (`rinchen.train.launch` documents the limit). A SIGHUP,
SIGTERM or SIGINT sent to `rinchen tiny run` left `eole build_vocab` or `eole train` running, with the launch
lock held, after the command had exited.

`GracefulRunner` treats an interrupt as the runner already treats a timeout: the process group is asked to end
with SIGTERM, the leader has a few seconds to end, and only then is the group killed. Every command that starts
a process of its own (`train launch`, `convert-script train`) handles SIGTERM by stopping what it started. The
rest is the base runner's: the same start, the same wait, the same result for an exit, a timeout and a program
that cannot be started.
"""

from __future__ import annotations

import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path

from rinchen.maintain.commands import (
    CommandResult,
    SubprocessRunner,
    _stop_group,
    _wait_for_exit,
)

__all__ = ["GracefulRunner"]


class GracefulRunner(SubprocessRunner):
    """`SubprocessRunner` whose interrupt asks the command to end before it kills it."""

    def run(
        self,
        argv: Sequence[str],
        *,
        cwd: Path,
        env: Mapping[str, str],
        log: Path,
        timeout: int,
    ) -> CommandResult:
        """Run `argv` in `cwd` with exactly `env`, its output replacing `log` (created, with its directory),
        for at most `timeout` seconds.

        A program that cannot be started gives `CommandResult(None, False, False)` and an empty log. A
        timeout, and a `KeyboardInterrupt` or a `SystemExit` raised while waiting, ask the process group to
        end with SIGTERM and kill it after the grace; the interrupt then propagates, the timeout gives
        `CommandResult(None, True)`.
        """
        log.parent.mkdir(parents=True, exist_ok=True)
        with log.open("wb") as sink:
            try:
                process = subprocess.Popen(
                    list(argv),
                    shell=False,
                    cwd=cwd,
                    env=dict(env),
                    stdin=subprocess.DEVNULL,
                    stdout=sink,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
            except OSError:
                return CommandResult(exit_code=None, timed_out=False, started=False)
            try:
                exit_code = _wait_for_exit(process, timeout)
            except BaseException:
                _stop_group(process)
                raise
            if exit_code is None:
                _stop_group(process)
                return CommandResult(exit_code=None, timed_out=True)
            return CommandResult(exit_code=exit_code, timed_out=False)
