"""`rinchen train launch` and `rinchen train average`: run or resume a long Eole training in
a run directory, and average the last checkpoints of a run. Parsing, file I/O and the
manifest only; the work happens in `rinchen.train` (`launch`, `average`). The commands
are registered on the `train` group by `rinchen.cli.train`.

launch
------
`rinchen train launch --config C --run-dir R [--world-size N] [--gpu-ranks 0,1,2,3]
[--threads 2] [--timeout S] [--cuda-visible-devices 0,1,2,3] [--write-only]
[--continue-from SOURCE]`

Without `--write-only` it starts or resumes the run (see `rinchen.train.launch`) and, when
it succeeded, prints one fact per line, always in this order:

    continued from: <step | averaged> <SOURCE>    (only for a continuation, see below)
    status: completed | early_stopped | already_complete
    steps: <the newest complete step>
    resumed from: <step> | none
    attempt: <k> (0 when no attempt was run)
    checkpoint: <the step directory>
    log: <the attempt's eole log> | none
    command.sh: <R/command.sh> | none (none: this launch wrote none)
    incomplete step directories skipped: [<steps>]    (only when there are some)
    manifest: <R/manifest.json>

With `--write-only` it prints the path of `command.sh` and nothing else, runs nothing and
writes no manifest (an existing one is left as it is). A run that is already finished has
nothing to write: the path is not printed, a note goes to standard error, the exit code is 0.

`--continue-from SOURCE` starts a NEW run from one of our own checkpoints (D9): SOURCE is the
final `step_<N>` directory of a run that `rinchen train launch` finished, or a directory that
`rinchen train average` wrote, in either case with the manifest the command wrote, which must
vouch for the files as they are now. Anything else is refused (exit 2; exit 3 for a manifest
or run record that cannot be read), and nothing but the lock file of the run is written. See
`rinchen.train.continuation` for the rules and `rinchen.train.launch` for the attempts. The
same option must be given again to resume, extend or write the files of that run; a run found
finished needs none of the source's files.

The manifest (stage `train_launch`) describes the state of the run directory when the
command ends, and comes last. Inputs: the config, hashed before it is read; for a
continuation also the files of the source with the SHA-256 the verification took (the
weights, `optimizer.pt` for a step, `config.json`, `vocab.json`) and its manifest. Outputs: the
four files of the final step directory (`model.00.safetensors`, `optimizer.pt`,
`config.json`, `vocab.json`; never the links Eole puts in the checkpoint directory),
`launch.json` and `command.sh`. Metrics: `steps`, `attempt`, `incomplete_skipped`,
`early_stopped` (0 or 1), `resumed_from` (when the attempt resumed), for a continuation
`continued_from.kind` (0 for a step, 1 for an averaged model) and `continued_from.step` (a
step source only), and from the log of
the last attempt `xent_first`, `xent_last`, `valid_ppl_first`, `valid_ppl_last` (when it
has them; they start at the attempt's own first report, so a resumed attempt's curve starts
at the resume). Its seed is the config's `seed`, or -1 (Eole's own default: none).

The old manifest is dropped by the launcher's `before_writing` hook, right before the first
file of an attempt is written, so a launch that is refused, turned away by the lock of
another launch or finds the run finished leaves it alone, and one that fails after it
started leaves none. A run found finished is a launch that changes nothing: its manifest
is kept as it is when it still vouches for the same files, and written (from the files and
the last attempt's log) when it is missing or no longer matches.

average
-------
`rinchen train average --checkpoint-dir D --last N [--out O]`

Averages the newest N (at least 2) COMPLETE `step_<k>` directories of the checkpoint
directory D (the run's `training.model_path`, not the run directory: its name is the
config's choice and the config is not read here) into the new directory O, by default
`averaged` beside D. An existing O is refused, and so is a D with no `launch.json` that parses
within three parent directories of its step directories (the path as given, lexically: a link
is not followed). An average is accepted by `launch --continue-from`, so step directories laid
out by hand would otherwise become a model it accepts in a single command. The file is all
that is checked: a hand-made `step_<k>` placed inside a real run directory passes. It guards
against a mistake, not against someone who writes the record or the steps too. Prints

    averaged: <n> checkpoints (steps <a>, <b>, ...)
    output: <O>
    weights sha256: <hex>
    incomplete step directories skipped: [<steps>]    (only when there are some)
    manifest: <O>/manifest.json

The manifest (stage `train_average`) is the last file written, inside O: the directory is
whole before it, and it is removed again if the manifest cannot be written, so a failure
leaves nothing. Inputs: the weights files of the averaged steps with the SHA-256 taken
before they were read. Outputs: the averaged weights, `averaged_from.json` and the files
copied from the last step (`config.json`, `vocab.json`, the SentencePiece artifact).
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Annotated, Any

import typer

from rinchen.cli.errors import rinchen_errors_as_exit_codes
from rinchen.cli.train_common import (
    exit_cannot_write,
    hash_input,
    require_out_apart_from_inputs,
    require_usable_timeout,
    write_manifest,
)
from rinchen.errors import ConfigError, DataError
from rinchen.manifest import store
from rinchen.manifest.model import Manifest, ManifestFile
from rinchen.manifest.stage import begin_commit, output_file, refuse_foreign_manifest
from rinchen.train.average import AVERAGED_FROM_FILE, AveragedModel
from rinchen.train.average import average_checkpoints as average_run
from rinchen.train.checkpoints import (
    CONFIG_FILE,
    LAUNCH_FILE,
    OPTIMIZER_FILE,
    RUN_DIRECTORY_LEVELS,
    VOCAB_FILE,
    WEIGHTS_FILE,
    StepDirectory,
    read_launch_record,
    run_directory_above,
    step_directories,
    step_of,
)
from rinchen.train.continuation import (
    AVERAGE_STAGE,
    LAUNCH_STAGE,
    SourceProvenance,
)
from rinchen.train.curves import parse_train_log
from rinchen.train.eole_cfg import read_config
from rinchen.train.launch import (
    COMMAND_FILE,
    LaunchResult,
    latest_attempt_log,
)
from rinchen.train.launch import launch as launch_run
from rinchen.utils.integers import is_strict_int

_DEFAULT_THREADS = 2
_LAUNCH_PACKAGES = ("eole", "torch", "safetensors", "sentencepiece")
_AVERAGE_PACKAGES = ("safetensors", "numpy")
_LAUNCH_OUTPUTS = "launch outputs"
_AVERAGE_OUTPUTS = "averaged model"
_CONFIG = "Eole config"
_STEP_FILES = (WEIGHTS_FILE, OPTIMIZER_FILE, CONFIG_FILE, VOCAB_FILE)
_UNSEEDED = -1  # Eole's own default: no seed
# A metric is a number: the kind of source a continuation started from.
_SOURCE_KIND_METRIC = {"step": 0.0, "averaged": 1.0}
_MIN_AVERAGED = 2
# A GPU number has a handful of digits; this stops a pasted blob from reaching `int`.
_MAX_RANK_DIGITS = 9
_DEFAULT_AVERAGE_DIR = "averaged"


def launch_command(
    config: Annotated[
        Path,
        typer.Option("--config", help="Eole training config (YAML) to launch."),
    ],
    run_dir: Annotated[
        Path,
        typer.Option(
            "--run-dir",
            help=(
                "Run directory: the launcher's files, command.sh, launch.json and the "
                "manifest go here, and the config's model_path must lie in it."
            ),
        ),
    ],
    world_size: Annotated[
        int | None,
        typer.Option(
            "--world-size",
            min=1,
            help="Replaces training.world_size (the total number of GPU processes).",
        ),
    ] = None,
    gpu_ranks: Annotated[
        str | None,
        typer.Option(
            "--gpu-ranks",
            help=(
                "Replaces training.gpu_ranks: distinct GPU numbers separated by "
                "commas, such as 0,1,2,3 (spaces are fine). Independent of "
                "--world-size."
            ),
        ),
    ] = None,
    threads: Annotated[
        int,
        typer.Option(
            "--threads", min=1, help="OpenMP and MKL threads of each Eole run."
        ),
    ] = _DEFAULT_THREADS,
    timeout: Annotated[
        float | None,
        typer.Option(
            "--timeout",
            help="Seconds after which each Eole command is killed; no limit by default.",
        ),
    ] = None,
    cuda_visible_devices: Annotated[
        str | None,
        typer.Option(
            "--cuda-visible-devices",
            help=(
                "GPU numbers a GPU run sees (exported as CUDA_VISIBLE_DEVICES), at "
                "least one per rank, such as 0,1,2,3. A run without a GPU hides them all."
            ),
        ),
    ] = None,
    write_only: Annotated[
        bool,
        typer.Option(
            "--write-only",
            help=(
                "Write command.sh, the attempt's config and launch.json, print the "
                "path of command.sh, and run nothing."
            ),
        ),
    ] = False,
    continue_from: Annotated[
        Path | None,
        typer.Option(
            "--continue-from",
            help=(
                "Start the run from a checkpoint of ours: the final step_<N> directory "
                "of a finished run, or a directory `train average` wrote, each with the "
                "manifest its command wrote (checked against the files). The weights "
                "only; the optimizer starts fresh and the config must state its options."
            ),
        ),
    ] = None,
) -> None:
    """Start or resume the Eole training of a config in a run directory.

    A run resumes after any interruption from its newest complete checkpoint; the
    config may change only its train_steps. A finished run is not run again. Only one
    launch at a time works in a run directory. Eole runs on the CPU unless the config
    (or --world-size and --gpu-ranks) says GPUs. With --continue-from the run is a new
    one that continues a checkpoint of ours (give the option again to resume it)."""
    with rinchen_errors_as_exit_codes():
        require_usable_timeout(timeout)
        ranks = parse_gpu_ranks(gpu_ranks)
        # The config is hashed before the launcher reads it, so that the manifest
        # vouches for the bytes the run started from.
        config_file = hash_input(config, what=_CONFIG, missing=ConfigError)
        refuse_foreign_manifest(run_dir, LAUNCH_STAGE)
        require_out_apart_from_inputs(run_dir, [config], option="--run-dir")
        seed = _config_seed(config)
        try:
            result = launch_run(
                config,
                run_dir,
                world_size=world_size,
                gpu_ranks=ranks,
                threads=threads,
                timeout=timeout,
                write_only=write_only,
                cuda_visible_devices=cuda_visible_devices,
                before_writing=None if write_only else _drop_the_manifest(run_dir),
                continue_from=continue_from,
            )
            if write_only:
                _say_what_was_written(result)
                return
            manifest_path = _record_the_run(run_dir, result, config_file, seed)
        except OSError as exc:
            exit_cannot_write(run_dir, exc, what=_LAUNCH_OUTPUTS)
        for line in _launch_facts(result, manifest_path):
            typer.echo(line)


def average_command(
    checkpoint_dir: Annotated[
        Path,
        typer.Option(
            "--checkpoint-dir",
            help=(
                "The run's checkpoint directory (training.model_path), which holds the "
                "step_<N> directories."
            ),
        ),
    ],
    last: Annotated[
        int,
        typer.Option(
            "--last",
            help="How many of the newest complete checkpoints to average (2 or more).",
        ),
    ],
    out: Annotated[
        Path | None,
        typer.Option(
            "--out",
            help=(
                "Directory to create for the averaged model; default: 'averaged' "
                "beside the checkpoint directory. An existing one is refused."
            ),
        ),
    ] = None,
) -> None:
    """Average the weights of the newest complete checkpoints of a run into one model.

    A plain mean of each floating-point tensor, written with the config and vocabulary
    of the newest checkpoint and a record of the sources (averaged_from.json). The
    result is for translating, not a checkpoint to resume from. A checkpoint directory
    with no launch.json that parses within three parent directories of its steps is
    refused (a guard against a mistake: that file is all that is checked)."""
    with rinchen_errors_as_exit_codes():
        if last < _MIN_AVERAGED:
            raise ConfigError(
                f"--last must be at least {_MIN_AVERAGED}, got {last}: the mean of one "
                "checkpoint is that checkpoint"
            )
        sources, skipped = _newest_complete(checkpoint_dir, last)
        _require_a_launched_run(checkpoint_dir, sources[-1])
        target = out if out is not None else _default_output(checkpoint_dir)
        try:
            averaged = average_run([directory.path for directory in sources], target)
            try:
                manifest_path = _write_average_manifest(averaged, last=last)
            except BaseException:
                # The manifest is the last file, and a directory without one is not
                # accepted by a rerun (which refuses an existing output).
                shutil.rmtree(averaged.directory, ignore_errors=True)
                raise
        except OSError as exc:
            exit_cannot_write(target, exc, what=_AVERAGE_OUTPUTS)
        for line in _average_facts(averaged, skipped, manifest_path):
            typer.echo(line)


def parse_gpu_ranks(text: str | None) -> list[int] | None:
    """The GPU numbers of `--gpu-ranks`: None when the option was not given, else the
    comma-separated non-negative decimal numbers in `text`, in order, with spaces around
    them dropped.

    Raises:
        ConfigError: an item is empty, negative, not a number (a sign, a decimal point,
            an exponent, other scripts' digits) or too long to be a GPU number, or a
            number is repeated.
    """
    if text is None:
        return None
    ranks: list[int] = []
    for item in text.split(","):
        piece = item.strip(" ")
        if not piece:
            raise _bad_ranks(text, "an item is empty")
        if piece.startswith("-") and _is_decimal(piece[1:]):
            raise _bad_ranks(text, f"{piece} is negative")
        if not _is_decimal(piece):
            raise _bad_ranks(text, f"{piece!r} is not a whole number")
        if len(piece) > _MAX_RANK_DIGITS:
            raise _bad_ranks(text, "a number has too many digits to be a GPU number")
        ranks.append(int(piece))
    repeated = sorted({rank for rank in ranks if ranks.count(rank) > 1})
    if repeated:
        raise _bad_ranks(text, f"{repeated} is repeated")
    return ranks


def _is_decimal(text: str) -> bool:
    # isdecimal alone accepts the digits of other scripts, which int() would read.
    return text.isascii() and text.isdecimal()


def _bad_ranks(text: str, why: str) -> ConfigError:
    return ConfigError(
        f"--gpu-ranks must be distinct GPU numbers separated by commas, such as "
        f"0,1,2,3, got {text!r}: {why}"
    )


def _config_seed(config: Path) -> int:
    """The config's `seed`, or what an unseeded Eole run records."""
    cfg = read_config(config, _CONFIG)
    seed = cfg.get("seed") if isinstance(cfg, Mapping) else None
    return seed if is_strict_int(seed) else _UNSEEDED


def _drop_the_manifest(run_dir: Path) -> Callable[[], None]:
    def begin() -> None:
        begin_commit(run_dir, LAUNCH_STAGE)

    return begin


def _say_what_was_written(result: LaunchResult) -> None:
    if result.command_sh is not None:
        typer.echo(str(result.command_sh))
        return
    typer.echo(
        f"note: nothing was written: the run is already finished "
        f"({result.status.replace('_', ' ')}, step {result.steps})",
        err=True,
    )


def _record_the_run(
    run_dir: Path, result: LaunchResult, config_file: ManifestFile, seed: int
) -> Path:
    """The manifest of the run after the launch: written, or kept when nothing ran and it
    still vouches for the same files."""
    outputs = _launch_outputs(run_dir, result)
    inputs = [config_file, *_source_inputs(result.continued_from)]
    ran = result.log is not None
    if not ran:
        existing = _readable_manifest(run_dir)
        if existing is not None and _vouches_for(
            existing, config_file, inputs, outputs
        ):
            return store.manifest_path(run_dir)
    return write_manifest(
        run_dir,
        stage=LAUNCH_STAGE,
        config_sha256=config_file.sha256,
        seed=seed,
        inputs=inputs,
        outputs=outputs,
        metrics=_launch_metrics(run_dir, result),
        packages=_LAUNCH_PACKAGES,
        status={"launch": "ok"},
    )


def _source_inputs(source: SourceProvenance | None) -> list[ManifestFile]:
    """The files a continuation started from, with the digests the verification took
    (they are not hashed again, and no size is recorded: the source may be gone when a
    finished run's manifest is written again)."""
    if source is None:
        return []
    return [
        ManifestFile(path=str(path), sha256=digest) for path, digest in source.files()
    ]


def _launch_outputs(run_dir: Path, result: LaunchResult) -> list[ManifestFile]:
    """The files of the final step directory (not the links to them in the checkpoint
    directory), then the record and the script of the run."""
    checkpoint = result.checkpoint
    assert checkpoint is not None, "a launch that ran or found a finished run has one"
    paths = [checkpoint / name for name in _STEP_FILES]
    paths.append(run_dir / LAUNCH_FILE)
    script = run_dir / COMMAND_FILE
    if script.is_file():
        paths.append(script)
    return [output_file(path, None) for path in paths]


def _readable_manifest(run_dir: Path) -> Manifest | None:
    try:
        return store.read(run_dir)
    except DataError:
        return None  # an unreadable manifest vouches for nothing: write a new one


def _vouches_for(
    manifest: Manifest,
    config_file: ManifestFile,
    inputs: list[ManifestFile],
    outputs: list[ManifestFile],
) -> bool:
    return (
        manifest.stage == LAUNCH_STAGE
        and manifest.config_sha256 == config_file.sha256
        and manifest.inputs == inputs
        and manifest.outputs == outputs
    )


def _launch_metrics(run_dir: Path, result: LaunchResult) -> dict[str, float]:
    assert result.steps is not None, (
        "a launch that ran or found a finished run has steps"
    )
    latest = latest_attempt_log(run_dir)
    # A run found finished ran no attempt now: the metrics are those of its last one.
    attempt = result.attempt or (latest[0] if latest else 0)
    log = result.log or (latest[1] if latest else None)
    metrics = {
        "steps": float(result.steps),
        "attempt": float(attempt),
        "incomplete_skipped": float(len(result.incomplete_skipped)),
        "early_stopped": 1.0 if result.status == "early_stopped" else 0.0,
    }
    if result.resumed_from is not None:
        metrics["resumed_from"] = float(result.resumed_from)
    source = result.continued_from
    if source is not None:
        metrics["continued_from.kind"] = _SOURCE_KIND_METRIC[source.kind]
        if source.step is not None:
            metrics["continued_from.step"] = float(source.step)
    if log is not None:
        curve = parse_train_log(log.read_text(encoding="utf-8", errors="replace"))
        if curve.reports:
            metrics["xent_first"] = curve.reports[0].xent
            metrics["xent_last"] = curve.reports[-1].xent
        if curve.validations:
            metrics["valid_ppl_first"] = curve.validations[0]
            metrics["valid_ppl_last"] = curve.validations[-1]
    return metrics


def _launch_facts(result: LaunchResult, manifest_path: Path) -> list[str]:
    lines = []
    if result.continued_from is not None:
        lines.append(
            f"continued from: {result.continued_from.kind} {result.continued_from.path}"
        )
    lines += [
        f"status: {result.status}",
        f"steps: {result.steps}",
        f"resumed from: {_or_none(result.resumed_from)}",
        f"attempt: {result.attempt}",
        f"checkpoint: {result.checkpoint}",
        f"log: {_or_none(result.log)}",
        f"command.sh: {_or_none(result.command_sh)}",
    ]
    if result.incomplete_skipped:
        lines.append(
            f"incomplete step directories skipped: {list(result.incomplete_skipped)}"
        )
    lines.append(f"manifest: {manifest_path}")
    return lines


def _or_none(value: object) -> str:
    return "none" if value is None else str(value)


def _newest_complete(
    checkpoint_dir: Path, count: int
) -> tuple[list[StepDirectory], list[int]]:
    """The newest `count` complete step directories of `checkpoint_dir`, oldest first,
    and the steps of the incomplete ones above the oldest of them (left out)."""
    if not checkpoint_dir.is_dir():
        raise DataError(f"checkpoint directory not found: {checkpoint_dir}")
    listed = step_directories(checkpoint_dir)
    complete = [directory for directory in listed if directory.complete]
    if len(complete) < count:
        raise DataError(
            f"{checkpoint_dir} holds {len(complete)} complete step directories "
            f"({len(listed) - len(complete)} incomplete), --last {count} needs {count}"
        )
    chosen = complete[-count:]
    skipped = [
        directory.step
        for directory in listed
        if not directory.complete and directory.step > chosen[0].step
    ]
    return chosen, skipped


def _require_a_launched_run(checkpoint_dir: Path, step: StepDirectory) -> None:
    """Refuse a checkpoint directory with no `launch.json` that parses within three parent
    directories of its step directories (`rinchen train launch` writes one in its run
    directory). `launch --continue-from` accepts any averaged model a `train_average`
    manifest vouches for, and step directories laid out by hand would otherwise become one
    after a single command. Only that file is checked, so a hand-made step directory placed
    inside a real run directory passes: a guard against that mistake, not against someone
    who also writes the launch record or the steps.

    Raises:
        ConfigError: no run directory holds a `launch.json` near enough above the steps.
        DataError: the record cannot be read.
    """
    run = run_directory_above(Path(os.path.abspath(step.path)))
    if run is None:
        raise ConfigError(
            f"cannot average {checkpoint_dir}: no {LAUNCH_FILE} within "
            f"{RUN_DIRECTORY_LEVELS} levels above its step directories; only a "
            "checkpoint directory inside a run directory that has one (`rinchen train "
            "launch` writes it) is averaged, because an average is accepted as the "
            "model to continue from"
        )
    try:
        read_launch_record(run)
    except DataError as error:
        raise DataError(f"cannot average {checkpoint_dir}: {error}") from error


def _default_output(checkpoint_dir: Path) -> Path:
    # Beside the directory as it is named: a link to a disk elsewhere stays where it is.
    return Path(os.path.abspath(checkpoint_dir)).parent / _DEFAULT_AVERAGE_DIR


def _write_average_manifest(averaged: AveragedModel, *, last: int) -> Path:
    steps = [_step_of(directory) for directory, _ in averaged.sources]
    inputs = [
        ManifestFile(
            path=str(directory / WEIGHTS_FILE),
            sha256=digest,
            bytes=(directory / WEIGHTS_FILE).stat().st_size,
        )
        for directory, digest in averaged.sources
    ]
    weights = averaged.directory / WEIGHTS_FILE
    outputs = [
        ManifestFile(
            path=str(weights),
            sha256=averaged.weights_sha256,
            bytes=weights.stat().st_size,
        ),
        *(
            output_file(averaged.directory / name, None)
            for name in (AVERAGED_FROM_FILE, *averaged.copied_files)
        ),
    ]
    return write_manifest(
        averaged.directory,
        stage=AVERAGE_STAGE,
        config_sha256=_parameters_sha256({"last": last, "steps": steps}),
        seed=0,
        inputs=inputs,
        outputs=outputs,
        metrics={
            "checkpoints": float(len(steps)),
            "first_step": float(steps[0]),
            "last_step": float(steps[-1]),
            "tensors": float(averaged.tensor_count),
            "tensors_averaged": float(averaged.averaged_count),
        },
        packages=_AVERAGE_PACKAGES,
        status={"average": "ok"},
    )


def _step_of(directory: Path) -> int:
    step = step_of(directory)
    assert step is not None, "average_checkpoints accepts only step_<N> directories"
    return step


def _parameters_sha256(parameters: Mapping[str, Any]) -> str:
    """The run has no config file, so its parameters stand in for one: the SHA-256 of
    their canonical JSON (sorted keys, no spaces)."""
    canonical = json.dumps(parameters, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _average_facts(
    averaged: AveragedModel, skipped: list[int], manifest_path: Path
) -> list[str]:
    steps = ", ".join(str(_step_of(directory)) for directory, _ in averaged.sources)
    lines = [
        f"averaged: {len(averaged.sources)} checkpoints (steps {steps})",
        f"output: {averaged.directory}",
        f"weights sha256: {averaged.weights_sha256}",
    ]
    if skipped:
        lines.append(f"incomplete step directories skipped: {skipped}")
    lines.append(f"manifest: {manifest_path}")
    return lines
