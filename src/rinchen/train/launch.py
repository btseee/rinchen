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
                        `world_size` and `gpu_ranks` overrides laid over it, the default
                        `tensorboard_log_dir` (`<run_dir>/tensorboard`, when `tensorboard`
                        is true and the config names none: Eole's own default is the
                        relative `runs/eole`, outside the run) and, when the attempt
                        resumes, `training.train_from`
    launch/attempt-<k>.log  standard output and error of `eole train` in attempt k

The config file given is never changed. Each attempt has its own config and log, so what
ran is on disk, and an earlier attempt's log is never read as a later one's.

The order of the work: read and validate the config, apply the GPU rules (`gpu_rules`),
take the identity of the run, take the lock of the run, (for a continuation, verify the
source and check that the config may continue it) look at the checkpoints
(`checkpoints.latest_complete`), write the attempt's files and `launch.json`, then
(unless `write_only`) build the vocabulary when its file is missing, add the reserved symbols
it lacks (`eole_run.add_reserved_symbols`: the file `eole build_vocab` writes holds only the
tokens the data showed, and a symbol it lacks would be `<unk>` to the model, see there; an
existing file is topped up the same way, and a complete one is left alone), run `eole train`
and decide from its log and the step directories how it ended. A run found finished, a
`write_only` and a refused launch leave the vocabulary file as it is. A resume names the newest
COMPLETE step directory of the run, never `model_path` (Eole's saver is not atomic), and
only the run's own: the config files never hold `train_from`. Eole ignores the config's
`model` section when it resumes and takes the checkpoint's own, so a complete step
directory whose saved model differs from the config's in a key both name (`hidden_size`,
the layers, the heads, ...) is not resumed from either: it is skipped like an incomplete
one, and when no directory is left the launch is refused, naming the key and both values.
That is a guard against a checkpoint copied in from another run; it cannot prove where a
checkpoint of the same architecture came from.

A CONTINUATION (`continue_from`, D9) is a new run that starts from one of our own
checkpoints: the final step of a finished run or an averaged model whose manifest vouches
for its files (`rinchen.train.continuation`, which also decides whether the config may
continue it). Its first attempt's config is the effective config plus `training.train_from:
<source>` and `training.reset_optim: all` (the weights only, a fresh optimizer that counts
steps from 1), and nothing else about it differs from a fresh start. Every later attempt
resumes the run's OWN newest complete step like any run, with `reset_optim: none` stated:
the checkpoints of the run record `reset_optim: all` and Eole fills a key the resume config
omits from that record, so leaving it out would restart the count at step 1 and repeat the
warm-up (measured with the real Eole). The source is not read by Eole again and its step
numbers mean nothing to the run. `launch.json`
(schema version 2) records the source and the run's identity covers it, so the same run
directory cannot be pointed at another source or continued without `continue_from`. A run
that began from random init is never a continuation, and a continuation is never a plain
run. The source is verified (hashed) before anything is written and again before any
resume, extension or `write_only` of a run that has step directories; a run found finished
is judged by its record and needs no source (it may be gone). The launcher never writes into
the source, and refuses a source that holds or lies in the new run.

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

Not done: a live abort on NaN. The vocabulary is built on the attempt's own config, a resume
config included (only needed when the vocabulary file is missing at a resume; `eole
build_vocab` accepts `training.train_from` in it: runbook 08, the D9 spike). `command.sh`
builds the vocabulary but does not add the reserved symbols: a run started by hand from it
trains on the vocabulary as Eole wrote it, and the next `launch` of that run adds them to the
file (a checkpoint keeps the vocabulary it was trained with, so that is too late for it).

Eole is reached only through `run_logged`, which runs it in a process group of its own
and kills the group on Ctrl-C, SIGTERM, SIGHUP, a timeout, and when Eole's leader exits
non-zero by itself (the per-rank processes of a multi-GPU run would otherwise train on
without the lock after a crash of the parent); the run stays resumable after any of them.
"""

from __future__ import annotations

import dataclasses
import os
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final, Literal

from rinchen.errors import ConfigError, UpstreamError
from rinchen.spm.symbols import user_defined_symbols
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
from rinchen.train.continuation import (
    SourceProvenance,
    check_continuation,
    require_source_apart,
    source_path,
    verify_source,
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
from rinchen.train.eole_cfg_keys import (
    CONTINUATION_RESET_OPTIM,
    CONTINUATION_RESUME_RESET_OPTIM,
)
from rinchen.train.eole_run import (
    add_reserved_symbols,
    eole_command,
    model_path_inside,
    training_environment,
)
from rinchen.train.gpu_rules import GpuSettings, check_gpu_rules
from rinchen.train.launch_command import render_command_sh
from rinchen.train.run_lock import holding_the_run, refuse_a_held_run
from rinchen.train.vocabulary_size import require_sizes_to_fit
from rinchen.utils.integers import is_strict_int
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
    incomplete_skipped: the steps of the step directories above the resume point that
        cannot be resumed from (incomplete, or of another model), which Eole overwrites
        if the run reaches them again.
    continued_from: the checkpoint of ours the run was started from, None for a run
        that continues none. Verified now when the launch ran or wrote something; for a
        run found finished it is what the run recorded (its files are not read).
    """

    status: LaunchStatus
    steps: int | None
    resumed_from: int | None
    checkpoint: Path | None
    command_sh: Path | None
    log: Path | None
    attempt: int
    incomplete_skipped: tuple[int, ...]
    continued_from: SourceProvenance | None = None


@dataclass(frozen=True, slots=True)
class _Plan:
    """A config, validated and made effective."""

    cfg: dict[str, Any]
    settings: GpuSettings
    model_path: Path
    src_vocab: Path
    train_steps: int
    identity: str
    continued: SourceProvenance | None = None

    def continuing(self, source: SourceProvenance) -> _Plan:
        """This plan as the continuation of `source`: its identity covers the source."""
        return dataclasses.replace(
            self,
            continued=source,
            identity=identity_of(self.cfg, continued_from=source.as_record()),
        )


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
    continue_from: Path | None = None,
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
    `continue_from` makes the run a continuation of the checkpoint of ours it names (see
    the module's text): a step directory or an averaged model, never a link. A run that
    has step directories must have begun as a continuation of the same source files.

    Raises:
        ConfigError: the config is unusable (see `eole_cfg.validate_train_config`), it has a
            vocabulary size, stated or Eole's default, below its SentencePiece model's
            (`vocabulary_size`), sets no positive `training.train_steps` or a
            `training.model_path` outside `run_dir`, the GPU rules or the device list are
            broken, the path or the text holds `$` (or the checkpoint directory a glob
            character), the run's step directories have no `launch.json` or the config is not
            the run's (only `training.train_steps` may change), `train_steps` is below the
            newest complete step, no step directory is of the config's model while some are of
            another one (see above), or another launch holds the run (its lock file is named;
            also for `write_only`); for a continuation also: the source holds `$` or a glob
            character, holds or lies in the new run (nothing is written then), is not vouched
            for by its manifest, or the config may not continue it
            (`continuation.check_continuation`), the run has step directories but did not
            begin as a continuation of this source (the lock file may have been left), or the
            source of a run that is not finished no longer exists.
        DataError: `launch.json` is not a launch record, or the checkpoint directory
            cannot be listed, or the vocabulary file cannot be read (`eole build_vocab`
            left none), or the manifest or run record of the source cannot be read.
        ValueError: `threads` is less than 1.
        UpstreamError: `eole` is not installed, a command exits non-zero (unless the run
            early-stopped) or runs past `timeout`, `eole train` exits 0 without
            finishing the run, or the loss in its log is NaN or infinite.
        OSError: the run directory or the lock file cannot be created or opened.
    """
    reject_dollar("config path", cfg_path)
    reject_dollar("run directory", run_dir)
    source = None if continue_from is None else source_path(continue_from)
    plan = _plan(cfg_path, run_dir, world_size, gpu_ranks)
    if source is not None:
        require_source_apart(source, run_dir=run_dir, vocabulary=plan.src_vocab)
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
        continue_from=source,
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
    continue_from: Path | None  # absolute, free of `$` and glob characters


def _launch_guarded(work: _Work, lock_fds: tuple[int, ...]) -> LaunchResult:
    """The work of `launch` once the run is guarded: the lock is held (`lock_fds` is its
    descriptor, which both Eole commands inherit), or `write_only` has found nobody
    holding it (no descriptor)."""
    plan, run_dir, command, env = work.plan, work.run_dir, work.command, work.env
    timeout, write_only = work.timeout, work.write_only
    record = read_launch_record(run_dir)
    point = _resume_point(plan)
    begun = point.directory is not None or bool(point.incomplete_newer)
    source = work.continue_from
    if source is not None:
        plan = _continuation_plan(plan, source, run_dir, record, begun)
    _require_the_same_run(plan, run_dir, record, point)
    _refuse_checkpoints_of_another_model(plan, point)
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

    if source is not None and begun:
        # About to extend or resume a run that exists: the source it continues must still
        # be the one it started from. Another source is another run's identity.
        plan = plan.continuing(verify_source(source))
        _require_the_same_run(plan, run_dir, record, point)
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
            continued_from=plan.continued,
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
    add_reserved_symbols(plan.src_vocab, user_defined_symbols())
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
        continues=_continued_path(plan, resume_step),
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
    require_sizes_to_fit(cfg)
    model_path = model_path_inside(cfg, run_dir, _WHAT)
    reject_glob_characters("checkpoint directory", model_path)
    if ".." in model_path.parts:
        # The resume config would refuse the train_from built on it, hours into the run.
        raise ConfigError(
            f"invalid {_WHAT}: training.model_path: holds a '..' component: {str(model_path)!r}"
        )
    train_steps = _explicit_train_steps(cfg)
    src_vocab = _vocabulary_file(cfg)
    cfg = _with_tensorboard_directory(cfg, run_dir)
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


def _with_tensorboard_directory(cfg: dict[str, Any], run_dir: Path) -> dict[str, Any]:
    """`cfg` with `tensorboard_log_dir` set to `<run_dir>/tensorboard` (absolute) when
    `tensorboard` is true and the config names no directory; else `cfg` itself.

    Eole's own default is the relative `runs/eole`: the event files would land in the
    working directory of whoever launches, one directory per attempt, outside the run.
    The default is an effective value like the others: it is in every attempt config
    and in the identity of the run, so a run directory that moves is another run (its
    `training.model_path` has to be edited then too). A directory the config names was
    validated with the rest of it; the default is plain text of a run directory that has
    passed the `$` check, so it needs no second look."""
    if cfg.get("tensorboard") is not True or "tensorboard_log_dir" in cfg:
        return cfg
    directory = Path(os.path.abspath(run_dir / "tensorboard")).as_posix()
    return merge(cfg, {"tensorboard_log_dir": directory})


def _explicit_train_steps(cfg: Mapping[str, Any]) -> int:
    steps = cfg["training"].get("train_steps")
    if not is_strict_int(steps) or steps < 1:
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


def _continuation_plan(
    plan: _Plan,
    source: Path,
    run_dir: Path,
    record: RunIdentity | None,
    begun: bool,
) -> _Plan:
    """`plan` as the continuation of `source`.

    A run with no step directory starts now: the source is verified and the config
    checked against it (T6), and the plan's identity covers what was verified. A run that
    has some began earlier, and only a continuation of a source may be taken up again:
    its identity is taken from the source its record names (its files are read before an
    attempt is prepared, not here, so a finished run needs none). A run with step
    directories and no record is left to `_require_the_same_run` to refuse."""
    if not begun:
        provenance = verify_source(source)
        check_continuation(plan.cfg, provenance)
        return plan.continuing(provenance)
    if record is None:
        return plan
    if record.continued_from is None:
        raise ConfigError(
            f"the run in {run_dir} was not started as a continuation ({run_dir / LAUNCH_FILE} "
            f"records no source) and a run that began from random init is not pointed "
            f"at a source afterwards: use a new run directory for {source}"
        )
    return plan.continuing(SourceProvenance.from_record(record.continued_from))


def _source_note(plan: _Plan, record: RunIdentity) -> str:
    """The sentence that adds the sources of both sides to a refusal, empty when neither
    continues one."""
    recorded, now = record.continued_from, plan.continued
    if recorded is None and now is None:
        return ""
    was = "continues nothing" if recorded is None else f"continues {recorded.path}"
    is_now = "continues nothing" if now is None else f"continues {now.path}"
    return f" The run {was}; this launch {is_now}."


def _continued_path(plan: _Plan, resume_step: int | None) -> Path | None:
    """The source the attempt starts from: that of a continuation that has no step of its
    own to resume from, else None."""
    if resume_step is not None or plan.continued is None:
        return None
    return plan.continued.path


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
            f"{plan.settings.world_size}, gpu_ranks {list(plan.settings.gpu_ranks)}."
            f"{_source_note(plan, record)} "
            "Only training.train_steps may change when a run is resumed: Eole keeps "
            "the optimiser options of the checkpoint and ignores new ones, and a "
            "different GPU count is a different batch. Restore the original config or "
            "use a new run directory"
        )


def _resume_point(plan: _Plan) -> ResumePoint:
    # Eole resumes into the model of the checkpoint and ignores the config's, so a
    # directory that saved another model is not one to resume from.
    return latest_complete(plan.model_path, model=plan.cfg["model"])


def _refuse_checkpoints_of_another_model(plan: _Plan, point: ResumePoint) -> None:
    """Raise when no step directory can be resumed from and the reason is that the
    complete ones are of another model (a checkpoint copied in from another run): they
    must be neither ignored, which would start the run again beside them, nor
    overwritten. With a resume point, the newer ones of another model are skipped like
    incomplete ones; Eole overwrites them when the run reaches their step."""
    if point.directory is not None:
        return
    foreign = [d for d in point.incomplete_newer if d.of_another_model]
    if foreign:
        newest = foreign[-1]
        raise ConfigError(
            f"no step directory in {plan.model_path} is of the model this config "
            f"trains, so none can be resumed from, and a checkpoint of another model "
            f"is not overwritten: {newest.path.name}: {newest.problem}. Remove it, or "
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
        continued_from=plan.continued,
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
    if resume_step is not None:
        return _resume_config(plan, resume_step)
    source = _continued_path(plan, resume_step)
    if source is None:
        return plan.cfg
    # The run has no step of its own yet: it starts from the source with a fresh
    # optimizer. Once it has one, the attempts resume it (`_resume_config`).
    first = merge(
        plan.cfg,
        {
            "training": {
                "train_from": str(source),
                "reset_optim": CONTINUATION_RESET_OPTIM,
            }
        },
    )
    validate_train_config(first, continues=source)
    return first


def _resume_config(plan: _Plan, resume_step: int) -> dict[str, Any]:
    training = {"train_from": str(plan.model_path / f"step_{resume_step}")}
    if plan.continued is None:
        resume = merge(plan.cfg, {"training": training})
        validate_train_config(resume, resumes=True)
        return resume
    # The checkpoints of a continuation record `reset_optim: all`, and Eole fills a key the
    # config omits from the checkpoint's record: stated, not left out.
    training["reset_optim"] = CONTINUATION_RESUME_RESET_OPTIM
    resume = merge(plan.cfg, {"training": training})
    validate_train_config(resume, resumes=True, resumes_continuation=True)
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
            continued_from=None
            if plan.continued is None
            else plan.continued.as_record(),
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
    newest = _resume_point(plan).directory
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
        continued_from=plan.continued,
    )
