"""The launcher of a long Eole training: it starts a run, resumes it after any
interruption, recognises how it ended, and records what it did (D8).

`launch` is a separate runner from `eole_run.run_training`, which treats any fresh
checkpoint file as success and cannot resume (D7's short ablation runs keep it). A
launch works in `run_dir` and writes only there, besides the paths the config itself
names:

    launch.json         what the run was started with (`checkpoints.identity_of`),
                        written before Eole starts, and by `write_only` too
    command.sh          the commands of the latest attempt, for an operator to run by hand;
                        it appends the output of `eole train` to launch/attempt-<k>.log
                        and that of `eole build_vocab` to launch/build_vocab.log, so a
                        hand-run attempt leaves the log a later launch reads
    launch/lock         the lock of a running launch (see below); empty, never removed
    launch/build_vocab.log
    launch/train-<k>.yaml   the effective config of attempt k: the config file with the
                        `world_size` and `gpu_ranks` overrides laid over it, and, when
                        the attempt resumes, `training.train_from`
    launch/attempt-<k>.log  standard output and error of `eole train` in attempt k

The config file given is never changed. Each attempt has its own config and log, so what
ran is on disk, and an earlier attempt's log is never read as a later one's.

The order of the work: read and validate the config, apply the GPU rules (`gpu_rules`),
take the identity of the run, take the lock of the run, look at the checkpoints
(`checkpoints.latest_complete`), write the attempt's files and `launch.json`, then
(unless `write_only`) build the vocabulary when its file is missing, run `eole train` and
decide from its log and the step directories how it ended. A resume names the newest
COMPLETE step directory of the run, never `model_path` (Eole's saver is not atomic), and
only the run's own: the config files never hold `train_from`.

`write_only` writes `command.sh`, the attempt's config and `launch.json` and runs
nothing. The record is what lets a run that an operator started by hand from `command.sh`
be resumed by `launch` afterwards: step directories without a record are refused, a
record without step directories is just a fresh start. The same refusals hold for it as
for a real launch (another config than the run's, the NaN guard).

One launch at a time (`run_lock`). A real launch holds an exclusive, non-blocking `flock`
on `launch/lock` from before it reads the state of the run until it returns, however it
ends. The lock belongs to the open file, so a crash releases it and no stale lock is left.
Both Eole commands inherit the descriptor (`run_logged`'s `inherit_fds`): Eole leads a
session of its own and outlives a SIGKILL of the launcher, and the lock then stays held by
that Eole until it exits, so the next launch cannot start a second training into the same
checkpoint directory. A second launch into the same run directory is a `ConfigError`
naming the lock file. A launch refused for what it finds in the run directory (another
config, a NaN) has taken the lock by then and leaves the empty lock file, nothing else;
one refused for its config or options writes nothing.
`write_only` takes no lock: it only looks at one that exists and is refused while another
launch holds it, since it would replace the config that launch runs. The look is not a
lock; a launch that starts after it is not stopped.

How a run ends (an `UpstreamError` for each failure, with the log named):

- `completed`: exit code 0 and the step `training.train_steps` is saved, complete.
- `early_stopped`: the log says `Early Stop!` and the step saved right after it is the
  newest complete one, whatever the exit code (with several processes Eole exits with
  code 1 after the collective timeout, once the final checkpoint is written).
- exit code 0 without either is an interrupted run (Ctrl-C makes Eole exit 0 without a
  final checkpoint); any other exit code or a timeout is a failure.
- a NaN or infinity anywhere in the log fails any of these: Eole goes on training with
  a NaN loss and exits 0, so the log is the only place that says so. It is found when the
  attempt ends: a live abort is not done.

A launch of a run that is already finished (the step `train_steps` is complete, or the
last attempt early-stopped at the newest complete step) runs nothing. Neither does a
launch of a run whose last attempt logged a NaN at or before its newest complete step:
that checkpoint holds the damage, so it is refused instead of resumed or called finished.

Not done: a live abort on NaN. The vocabulary is built on the attempt's own config, a
resume config included (only needed when the vocabulary file is missing at a resume; that
`eole build_vocab` accepts `training.train_from` in it is not measured).

Eole is reached only through `run_logged`, which runs it in a process group of its own
and kills the group on Ctrl-C, SIGTERM, SIGHUP and a timeout; the run stays resumable
after any of them.
"""

from __future__ import annotations

import os
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final, Literal

from rinchen.errors import ConfigError, UpstreamError
from rinchen.train.checkpoints import (
    LAUNCH_FILE,
    ResumePoint,
    RunIdentity,
    StepDirectory,
    identity_of,
    latest_complete,
    read_launch_record,
    write_launch_record,
)
from rinchen.train.curves import TrainCurve, parse_train_log
from rinchen.train.eole_cfg import (
    dump_config,
    merge,
    read_config,
    reject_dollar,
    reject_glob_characters,
    validate_train_config,
)
from rinchen.train.eole_run import (
    eole_command,
    model_path_inside,
    training_environment,
)
from rinchen.train.gpu_rules import GpuSettings, check_gpu_rules
from rinchen.train.launch_command import render_command_sh
from rinchen.train.run_lock import holding_the_run, refuse_a_held_run
from rinchen.utils.logged_process import run_logged
from rinchen.utils.part_file import replace_file

LAUNCH_DIR: Final = "launch"
COMMAND_FILE: Final = "command.sh"
LOCK_FILE: Final = "lock"
LaunchStatus = Literal["completed", "early_stopped", "already_complete", "written"]

_BUILD_VOCAB_LOG: Final = "build_vocab.log"
_WHAT: Final = "Eole config"
_ATTEMPT_LOG: Final = re.compile(r"attempt-([0-9]+)\.log")
_DEVICE_LIST: Final = re.compile(r"[0-9]+(,[0-9]+)*")
_EOLE_DEFAULT_TRAIN_STEPS: Final = 100000


@dataclass(frozen=True, slots=True)
class LaunchResult:
    """What a launch did.

    status: `completed`, `early_stopped`, `already_complete` (nothing was run) or
        `written` (`write_only`: the files were written, nothing was run).
    steps: the newest complete step of the run when the launch ended; for `written`,
        the one it would resume from; None when the run has none.
    resumed_from: the step the attempt resumed (or, for `written`, would resume) from;
        None for a start from scratch and for `already_complete`.
    checkpoint: the directory of `steps` for a launch that ran or found the run
        finished; None for `written`.
    command_sh: `run_dir/command.sh`; None when nothing was written (`already_complete`
        and an early-stopped run found finished).
    log: this attempt's `eole train` log; None when none was run.
    attempt: the number of the attempt (`launch/train-<attempt>.yaml`, `attempt-<attempt>.log`);
        0 when no attempt was prepared.
    incomplete_skipped: the steps of the incomplete step directories above the resume
        point, which Eole overwrites if the run reaches them again.
    """

    status: LaunchStatus
    steps: int | None
    resumed_from: int | None
    checkpoint: Path | None
    command_sh: Path | None
    log: Path | None
    attempt: int
    incomplete_skipped: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class _Plan:
    """A config, validated and made effective."""

    cfg: dict[str, Any]
    settings: GpuSettings
    model_path: Path
    src_vocab: Path
    train_steps: int
    identity: str


def launch(
    cfg_path: Path,
    run_dir: Path,
    *,
    world_size: int | None = None,
    gpu_ranks: Sequence[int] | None = None,
    threads: int = 2,
    timeout: float | None = None,
    write_only: bool = False,
    eole: str | None = None,
    cuda_visible_devices: str | None = None,
    extra_env: Mapping[str, str] | None = None,
    before_writing: Callable[[], None] | None = None,
) -> LaunchResult:
    """Start, resume or finish the training of the config at `cfg_path` in `run_dir`.

    `world_size` and `gpu_ranks` replace the config's `training.world_size` and
    `training.gpu_ranks` when given (an empty `gpu_ranks` means no GPU); the result
    must pass `gpu_rules.check_gpu_rules`. `threads` pins OpenMP and MKL; the GPU is
    hidden from the process exactly when the run uses none, and `cuda_visible_devices`
    (digits separated by commas, at least as many as ranks) sets which GPUs a GPU run
    sees. `timeout` (seconds) bounds each Eole command. `write_only` writes
    `command.sh`, the attempt's config and `launch.json` and runs nothing.
    `eole` is the command to run (default: the `eole` next to the interpreter, or the
    caller's choice in tests); `extra_env` is laid over the process environment of the
    commands, a hook for a test shim that is never read from the environment here.
    `before_writing` is called once, right before the first file of the attempt is
    written and after every refusal and the finished-run check: not for a launch that is
    refused and not for a run found finished, which write nothing. A caller that keeps a
    record of the run's final state uses it to drop the old record at the last moment
    (`manifest.stage.begin_commit`). What it raises propagates, with nothing written.

    Raises:
        ConfigError: the config is unusable (see `eole_cfg.validate_train_config`), it
            sets no positive `training.train_steps` or a `training.model_path` outside
            `run_dir`, the GPU rules or the device list are broken, the path or the
            text holds `$` (or the checkpoint directory a glob character), the run's
            step directories have no `launch.json` or the config is not the run's
            (only `training.train_steps` may change), `train_steps` is below the
            newest complete step, or another launch holds the run (its lock file is
            named; also for `write_only`).
        DataError: `launch.json` is not a launch record, or the checkpoint directory
            cannot be listed.
        ValueError: `threads` is less than 1.
        UpstreamError: `eole` is not installed, a command exits non-zero (unless the run
            early-stopped) or runs past `timeout`, `eole train` exits 0 without
            finishing the run, or the loss in its log is NaN or infinite.
        OSError: the run directory or the lock file cannot be created or opened.
    """
    reject_dollar("config path", cfg_path)
    reject_dollar("run directory", run_dir)
    plan = _plan(cfg_path, run_dir, world_size, gpu_ranks)
    devices = _devices(cuda_visible_devices, plan.settings)
    command = eole if eole is not None else eole_command()
    env = _environment(threads, plan.settings, devices, extra_env)
    work = _Work(
        plan=plan,
        run_dir=run_dir,
        command=command,
        env=env,
        threads=threads,
        timeout=timeout,
        devices=devices,
        write_only=write_only,
        before_writing=before_writing,
    )
    lock = run_dir / LAUNCH_DIR / LOCK_FILE
    if write_only:
        refuse_a_held_run(lock)
        return _launch_guarded(work, ())
    with holding_the_run(lock) as descriptor:
        return _launch_guarded(work, (descriptor,))


@dataclass(frozen=True, slots=True)
class _Work:
    """Everything `_launch_guarded` needs, decided before the run was looked at."""

    plan: _Plan
    run_dir: Path
    command: str
    env: dict[str, str]
    threads: int
    timeout: float | None
    devices: str | None
    write_only: bool
    before_writing: Callable[[], None] | None


def _launch_guarded(work: _Work, lock_fds: tuple[int, ...]) -> LaunchResult:
    """The work of `launch` once the run is guarded: the lock is held (`lock_fds` is its
    descriptor, which both Eole commands inherit), or `write_only` has found nobody
    holding it (no descriptor)."""
    plan, run_dir, command, env = work.plan, work.run_dir, work.command, work.env
    timeout, write_only = work.timeout, work.write_only
    record = read_launch_record(run_dir)
    point = latest_complete(plan.model_path)
    _require_the_same_run(plan, run_dir, record, point)
    skipped = tuple(directory.step for directory in point.incomplete_newer)
    newest = point.directory
    if newest is not None:
        previous = _previous_attempt(run_dir)
        _refuse_checkpoints_that_hold_a_nan(previous, newest)
        if newest.step > plan.train_steps:
            raise ConfigError(
                f"training.train_steps is {plan.train_steps}, below step {newest.step}, "
                "the newest complete checkpoint of this run: a run is not shortened; "
                f"set train_steps to at least {newest.step} or use a new run directory"
            )
        if newest.step == plan.train_steps or _early_stopped_at(previous, newest):
            return _finished(newest, skipped, plan)

    resume_step = None if newest is None else newest.step
    if work.before_writing is not None:
        work.before_writing()
    attempt = _prepare_attempt(
        plan,
        run_dir,
        command=command,
        threads=work.threads,
        exported_devices=_exported_devices(plan.settings, work.devices),
        resume_step=resume_step,
    )
    _record_the_run(run_dir, plan, record)
    if write_only:
        return LaunchResult(
            status="written",
            steps=resume_step,
            resumed_from=resume_step,
            checkpoint=None,
            command_sh=run_dir / COMMAND_FILE,
            log=None,
            attempt=attempt.number,
            incomplete_skipped=skipped,
        )

    if not plan.src_vocab.exists():
        run_logged(
            [command, "build_vocab", "-config", str(attempt.config), "-n_sample", "-1"],
            env=env,
            log_path=run_dir / LAUNCH_DIR / _BUILD_VOCAB_LOG,
            what="eole build_vocab",
            timeout=timeout,
            inherit_fds=lock_fds,
        )
    failure: UpstreamError | None = None
    try:
        run_logged(
            [command, "train", "-config", str(attempt.config)],
            env=env,
            log_path=attempt.log,
            what="eole train",
            timeout=timeout,
            inherit_fds=lock_fds,
        )
    except UpstreamError as error:
        failure = error
    return _ended(plan, run_dir, attempt, failure, resume_step, skipped)


@dataclass(frozen=True, slots=True)
class _Attempt:
    """The files of one attempt."""

    number: int
    config: Path  # absolute: the command line does not depend on the working directory
    log: Path


def _prepare_attempt(
    plan: _Plan,
    run_dir: Path,
    *,
    command: str,
    threads: int,
    exported_devices: str | None,
    resume_step: int | None,
) -> _Attempt:
    """Write this attempt's config and `command.sh`; the run directory is created."""
    launch_dir = run_dir / LAUNCH_DIR
    launch_dir.mkdir(parents=True, exist_ok=True)
    number = _last_attempt(launch_dir) + 1
    config = Path(os.path.abspath(launch_dir / f"train-{number}.yaml"))
    dump_config(_attempt_config(plan, resume_step), config)
    script = render_command_sh(
        run_dir=Path(os.path.abspath(run_dir)),
        working_directory=Path.cwd(),
        eole=command,
        config=config,
        src_vocab=Path(os.path.abspath(plan.src_vocab)),
        attempt_log=Path(os.path.abspath(_attempt_log(launch_dir, number))),
        build_vocab_log=Path(os.path.abspath(launch_dir / _BUILD_VOCAB_LOG)),
        threads=threads,
        cuda_visible_devices=exported_devices,
        attempt=number,
        resumes_from=resume_step,
    )
    replace_file(run_dir / COMMAND_FILE, script.encode("utf-8"))
    return _Attempt(number, config, _attempt_log(launch_dir, number))


def _plan(
    cfg_path: Path,
    run_dir: Path,
    world_size: int | None,
    gpu_ranks: Sequence[int] | None,
) -> _Plan:
    cfg = read_config(cfg_path, _WHAT)
    validate_train_config(cfg)
    model_path = model_path_inside(cfg, run_dir, _WHAT)
    reject_glob_characters("checkpoint directory", model_path)
    if ".." in model_path.parts:
        # The resume config would refuse the train_from built on it, hours into the run.
        raise ConfigError(
            f"invalid {_WHAT}: training.model_path: holds a '..' component: {str(model_path)!r}"
        )
    train_steps = _explicit_train_steps(cfg)
    src_vocab = _vocabulary_file(cfg)
    overrides: dict[str, Any] = {}
    if world_size is not None:
        overrides["world_size"] = world_size
    if gpu_ranks is not None:
        overrides["gpu_ranks"] = list(gpu_ranks)
    effective = merge(cfg, {"training": overrides})
    settings = check_gpu_rules(effective)
    # Spelled out, so that leaving a default out and stating it are the same run.
    effective = merge(
        effective,
        {
            "training": {
                "world_size": settings.world_size,
                "gpu_ranks": list(settings.gpu_ranks),
            }
        },
    )
    return _Plan(
        effective,
        settings,
        model_path,
        src_vocab,
        train_steps,
        identity_of(effective),
    )


def _explicit_train_steps(cfg: Mapping[str, Any]) -> int:
    steps = cfg["training"].get("train_steps")
    if isinstance(steps, bool) or not isinstance(steps, int) or steps < 1:
        got = "but it is missing" if steps is None else f"got {steps!r}"
        raise ConfigError(
            f"invalid {_WHAT}: training.train_steps: must be set to a positive integer, "
            f"{got}: Eole's default of {_EOLE_DEFAULT_TRAIN_STEPS} would apply "
            "without a word"
        )
    return steps


def _vocabulary_file(cfg: Mapping[str, Any]) -> Path:
    vocabulary = cfg.get("src_vocab")
    if not isinstance(vocabulary, str) or not vocabulary:
        raise ConfigError(
            f"invalid {_WHAT}: src_vocab: must name the vocabulary file that "
            "`eole build_vocab` writes and `eole train` reads"
        )
    return Path(vocabulary)


def _devices(requested: str | None, settings: GpuSettings) -> str | None:
    if requested is None:
        return None
    if _DEVICE_LIST.fullmatch(requested) is None:
        raise ConfigError(
            "cuda_visible_devices must be device numbers separated by commas, such as "
            f"'0,1,2,3', got {requested!r}"
        )
    if not settings.uses_gpu:
        raise ConfigError(
            f"cuda_visible_devices {requested!r} is given for a run that uses no GPU "
            "(training.gpu_ranks is empty)"
        )
    devices = requested.count(",") + 1
    if devices < len(settings.gpu_ranks):
        raise ConfigError(
            f"cuda_visible_devices {requested!r} names {devices} device(s) for "
            f"{len(settings.gpu_ranks)} ranks: each rank needs its own"
        )
    return requested


def _exported_devices(settings: GpuSettings, requested: str | None) -> str | None:
    # A run without a GPU hides them all; a GPU run uses what was asked for, or else
    # whatever the environment of the script says.
    return requested if settings.uses_gpu else ""


def _environment(
    threads: int,
    settings: GpuSettings,
    devices: str | None,
    extra_env: Mapping[str, str] | None,
) -> dict[str, str]:
    env = training_environment(threads, hide_gpu=not settings.uses_gpu)
    if devices is not None:
        env["CUDA_VISIBLE_DEVICES"] = devices
    if extra_env:
        env.update(extra_env)
    return env


def _require_the_same_run(
    plan: _Plan, run_dir: Path, record: RunIdentity | None, point: ResumePoint
) -> None:
    if point.directory is None and not point.incomplete_newer:
        return  # no step directory: nothing of an earlier run to protect
    record_file = run_dir / LAUNCH_FILE
    if record is None:
        steps = [point.directory, *point.incomplete_newer]
        listed = sorted(d.step for d in steps if d is not None)
        raise ConfigError(
            f"{plan.model_path} holds step directories {listed} but {record_file} does "
            "not exist, so it is not known what run wrote them (a run started by hand "
            "with command.sh has no record): use a new run directory"
        )
    if record.identity != plan.identity:
        raise ConfigError(
            f"the config is not the one this run was started with ({record_file}): "
            f"recorded: world_size {record.world_size}, gpu_ranks "
            f"{list(record.gpu_ranks)}; this launch: world_size "
            f"{plan.settings.world_size}, gpu_ranks {list(plan.settings.gpu_ranks)}. "
            "Only training.train_steps may change when a run is resumed: Eole keeps "
            "the optimiser options of the checkpoint and ignores new ones, and a "
            "different GPU count is a different batch. Restore the original config or "
            "use a new run directory"
        )


def _previous_attempt(run_dir: Path) -> tuple[Path, TrainCurve] | None:
    """The log of the last attempt of the run and what it says, None when there is
    none."""
    launch_dir = run_dir / LAUNCH_DIR
    log = _attempt_log(launch_dir, _last_attempt(launch_dir))
    if not log.is_file():
        return None
    return log, parse_train_log(log.read_text(encoding="utf-8", errors="replace"))


def _refuse_checkpoints_that_hold_a_nan(
    previous: tuple[Path, TrainCurve] | None, newest: StepDirectory
) -> None:
    """Raise when the last attempt's loss went NaN or infinite at or before the step
    of `newest`: that checkpoint holds the damage, and neither resuming from it nor
    calling the run finished is right."""
    if previous is None:
        return
    log, curve = previous
    if curve.non_finite_step is not None and newest.step >= curve.non_finite_step:
        raise UpstreamError(
            f"the loss in the last attempt's log was not finite: "
            f"{curve.non_finite_field} at step {curve.non_finite_step} ({log}), and the "
            f"newest complete checkpoint, step {newest.step}, is not from before it, so "
            "its weights are bad; delete the step directories from that step on, or use "
            "a new run directory"
        )


def _early_stopped_at(
    previous: tuple[Path, TrainCurve] | None, newest: StepDirectory
) -> bool:
    """Whether the last attempt ended with an early stop at the step `newest`."""
    if previous is None:
        return False
    curve = previous[1]
    return curve.early_stopped and curve.early_stop_step == newest.step


def _finished(
    newest: StepDirectory, skipped: tuple[int, ...], plan: _Plan
) -> LaunchResult:
    status: LaunchStatus = (
        "already_complete" if newest.step == plan.train_steps else "early_stopped"
    )
    return LaunchResult(
        status=status,
        steps=newest.step,
        resumed_from=None,
        checkpoint=newest.path,
        command_sh=None,
        log=None,
        attempt=0,
        incomplete_skipped=skipped,
    )


def latest_attempt_log(run_dir: Path) -> tuple[int, Path] | None:
    """The number and the log (`launch/attempt-<k>.log`) of the latest attempt of the run
    in `run_dir`, None when there is none."""
    launch_dir = run_dir / LAUNCH_DIR
    number = _last_attempt(launch_dir)
    return (number, _attempt_log(launch_dir, number)) if number else None


def _last_attempt(launch_dir: Path) -> int:
    if not launch_dir.is_dir():
        return 0
    numbers = [
        int(match.group(1))
        for entry in launch_dir.iterdir()
        if (match := _ATTEMPT_LOG.fullmatch(entry.name))
    ]
    return max(numbers, default=0)


def _attempt_log(launch_dir: Path, attempt: int) -> Path:
    return launch_dir / f"attempt-{attempt}.log"


def _attempt_config(plan: _Plan, resume_step: int | None) -> dict[str, Any]:
    if resume_step is None:
        return plan.cfg
    resume = merge(
        plan.cfg,
        {"training": {"train_from": str(plan.model_path / f"step_{resume_step}")}},
    )
    validate_train_config(resume, resumes=True)
    return resume


def _record_the_run(run_dir: Path, plan: _Plan, record: RunIdentity | None) -> None:
    """Write `launch.json` unless it already records this run."""
    if record is not None and record.identity == plan.identity:
        return
    write_launch_record(
        run_dir,
        RunIdentity(
            identity=plan.identity,
            world_size=plan.settings.world_size,
            gpu_ranks=plan.settings.gpu_ranks,
            created=datetime.now(UTC).isoformat(timespec="seconds"),
        ),
    )


def _ended(
    plan: _Plan,
    run_dir: Path,
    attempt: _Attempt,
    failure: UpstreamError | None,
    resumed_from: int | None,
    skipped: tuple[int, ...],
) -> LaunchResult:
    """How the attempt whose `eole train` ended with `failure` (None: exit code 0)
    and left its log ended."""
    log = attempt.log
    curve = parse_train_log(log.read_text(encoding="utf-8", errors="replace"))
    newest = latest_complete(plan.model_path).directory
    stopped_early = (
        curve.early_stopped
        and newest is not None
        and curve.early_stop_step == newest.step
    )
    if failure is not None and not stopped_early:
        raise failure
    if curve.non_finite_step is not None:
        raise UpstreamError(
            f"the training loss is not finite: {curve.non_finite_field} at step "
            f"{curve.non_finite_step} in {log}; Eole does not stop on NaN, so the "
            "checkpoints from that step on are bad and the run is not accepted"
        )
    reached_the_end = newest is not None and newest.step == plan.train_steps
    if newest is None or not (stopped_early or reached_the_end):
        where = "no complete checkpoint" if newest is None else f"step {newest.step}"
        early = (
            " (the log says Early Stop! but the checkpoint saved after it is not "
            "complete)"
            if curve.early_stopped
            else ""
        )
        raise UpstreamError(
            f"eole train exited with code 0 but the run was interrupted: its newest "
            f"complete checkpoint is {where}, not the final step {plan.train_steps}"
            f"{early}; the run can be resumed; see {log}"
        )
    return LaunchResult(
        status="early_stopped" if stopped_early else "completed",
        steps=newest.step,
        resumed_from=resumed_from,
        checkpoint=newest.path,
        command_sh=run_dir / COMMAND_FILE,
        log=log,
        attempt=attempt.number,
        incomplete_skipped=skipped,
    )
