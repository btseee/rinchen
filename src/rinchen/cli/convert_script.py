"""`rinchen convert-script`: build the word-pair corpus of the Cyrillic and
traditional-script converter, train and export the model, and convert text or
score a model with the exported one. Parsing, file I/O and manifests only; the
work happens in `rinchen.script_conv`.

`train` and `export` leave a model directory that holds everything `apply` and
`eval` need: the CTranslate2 model and, beside it, the character model.
"""

from __future__ import annotations

import hashlib
import json
import shlex
import shutil
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Any, Literal, NamedTuple, TypeIs, get_args

import typer

from rinchen.cli.errors import (
    FAILURE_EXIT_CODE,
    exit_with_error,
    rinchen_errors_as_exit_codes,
)
from rinchen.config.loader import REPO_ROOT
from rinchen.errors import ConfigError, DataError, UpstreamError
from rinchen.manifest import store
from rinchen.manifest.model import Manifest, ManifestFile
from rinchen.manifest.stage import (
    begin_commit,
    build_inputs,
    code_version,
    output_file,
    refuse_foreign_manifest,
    sha256_of,
    tool_versions,
)
from rinchen.script_conv.char_model import (
    CharModelConfig,
    check_char_model,
    load_char_model_config,
    train_char_model,
)
from rinchen.script_conv.config import (
    ConverterPaths,
    Profile,
    dump_config,
    read_converter_paths,
    render_converter_config,
    validate_converter_config,
)
from rinchen.script_conv.data import (
    CORPUS_FILE_NAMES,
    CorpusFiles,
    WordPair,
    build_corpus,
    check_valid_fraction,
    read_pairs_tsv,
    rule_pairs,
)
from rinchen.script_conv.eval import score_conversions
from rinchen.script_conv.export import export_ct2, export_log_path
from rinchen.script_conv.runtime import ScriptConverter, Target
from rinchen.script_conv.train import CHECKPOINT_FILE, eole_commands, run_training
from rinchen.utils.part_file import unique_part_path

app = typer.Typer(no_args_is_help=True)

DEFAULT_CORPUS_DIR = REPO_ROOT / "data" / "script_conv" / "corpus"
DEFAULT_CHAR_MODEL_CONFIG = REPO_ROOT / "configs" / "spm" / "converter_char.yaml"

_DATA_STAGE = "script_conv_data"
_TRAIN_STAGE = "script_conv_train"
_EXPORT_STAGE = "script_conv_export"
_DEFAULT_VALID_FRACTION = 0.1
_DEFAULT_SEED = 1234
_DEFAULT_THREADS = 2
# A hung converter must not hang the command for ever.
_DEFAULT_EXPORT_TIMEOUT_SECONDS = 3600.0
_TRAIN_PACKAGES = ("eole", "torch", "sentencepiece", "ctranslate2")
_EXPORT_PACKAGES = ("ctranslate2", "eole")
# The rendered Eole config of a run, in its run directory.
_CONVERTER_CONFIG = "converter.yaml"
# The CTranslate2 model directory `export` writes inside the run directory.
_CT2_DIR = "ct2"
# What Eole saves beside its checkpoint and the CTranslate2 converter reads with it.
_CHECKPOINT_SIDE_FILES = ("config.json", "vocab.json")
# The two directions' corpus files hold the same words with the sides swapped, so the
# character model is trained on one direction's files only.
_CHAR_MODEL_CORPUS_PREFIX = "cyrl2mong."
# The character model of a run, as a path prefix inside the run directory: a model
# shared by several runs would be overwritten by the next one.
_CHAR_MODEL_PREFIX = "converter_char"
# The corpus file Eole validates on; with no row in it `eole train` divides by zero.
_VALID_SOURCE_FILE = "cyrl2mong.valid.src"
# The character model's name inside a model directory: with it there, `apply` and
# `eval` need only the directory.
_CHAR_MODEL_FILE = "converter_char.model"
_TARGET_HELP = "Script to convert to: Mong (traditional script) or Cyrl (Cyrillic)."


class _LoadedPairs(NamedTuple):
    pairs: tuple[WordPair, ...]
    words_read: int | None  # None when the pairs came from a pairs file


def load_converter(model_dir: Path) -> ScriptConverter:
    """The converter of the model directory `export` wrote: the CTranslate2 model
    and the character model beside it. Tests replace this function."""
    return ScriptConverter(model_dir, model_dir / _CHAR_MODEL_FILE)


# ---- build-data -----------------------------------------------------------------------


@app.command("build-data")
def build_data(
    pairs: Annotated[
        Path | None,
        typer.Option(
            "--pairs",
            help=(
                "Real parallel word pairs: a UTF-8 TSV headed "
                "'cyrillic<TAB>script', one pair per line."
            ),
        ),
    ] = None,
    words: Annotated[
        Path | None,
        typer.Option(
            "--words",
            help=(
                "Cyrillic words, one per line, spelled by the rule baseline: "
                "only the words its lexicon holds, unless --allow-fallback."
            ),
        ),
    ] = None,
    out_dir: Annotated[
        Path | None,
        typer.Option(
            "--out-dir",
            help=(
                "Directory for the eight corpus files and the manifest; default "
                "data/script_conv/corpus. It must not be the input's directory."
            ),
        ),
    ] = None,
    valid_fraction: Annotated[
        float,
        typer.Option(
            "--valid-fraction",
            help="Share of the pairs held out for validation, in (0, 0.5].",
        ),
    ] = _DEFAULT_VALID_FRACTION,
    seed: Annotated[
        int,
        typer.Option("--seed", help="Seed of the train and valid split."),
    ] = _DEFAULT_SEED,
    allow_fallback: Annotated[
        bool,
        typer.Option(
            "--allow-fallback",
            help=(
                "With --words: also keep the words the lexicon lacks, spelled by "
                "the phonographic guess. A guess is a weak training label."
            ),
        ),
    ] = False,
) -> None:
    """Write the train and valid corpora of both directions from Cyrillic and
    traditional-script word pairs."""
    with rinchen_errors_as_exit_codes():
        source_kind, source = _single_input(pairs, words, allow_fallback)
        target = out_dir if out_dir is not None else DEFAULT_CORPUS_DIR
        # Before anything is read: a run must not write into its own input's folder.
        _require_out_apart_from_input(target, source)
        refuse_foreign_manifest(target, _DATA_STAGE)
        # Before the old manifest is removed: a typo must not erase an intact one.
        check_valid_fraction(valid_fraction)
        run = _CorpusRun(source_kind, valid_fraction, seed, allow_fallback)
        input_files = _hash_inputs([source])
        loaded = _load_pairs(run, source)
        if loaded.words_read is not None:
            typer.echo(f"{loaded.words_read} words read, {len(loaded.pairs)} paired")
        try:
            files, manifest_path = _commit_corpus(
                run, loaded.pairs, target, input_files
            )
        except OSError as exc:
            exit_with_error(
                f"cannot write the corpus files under {target}: {_describe_os_error(exc)}",
                FAILURE_EXIT_CODE,
            )
        typer.echo(
            f"wrote {files.rows['cyrl2mong.train.src']} train and "
            f"{files.rows['cyrl2mong.valid.src']} valid pairs to {target}"
        )
        if files.dropped:
            detail = ", ".join(f"{code}={n}" for code, n in files.dropped.items())
            typer.echo(f"dropped {sum(files.dropped.values())}: {detail}")
        typer.echo(f"manifest: {manifest_path}")


@dataclass(frozen=True)
class _CorpusRun:
    """Everything besides the input file that decides what `build-data` writes."""

    source: Literal["pairs", "words"]
    valid_fraction: float
    seed: int
    allow_fallback: bool

    def config_sha256(self) -> str:
        """The run has no config file, so its parameters stand in for one: the
        SHA-256 of their canonical JSON (sorted keys, no spaces)."""
        canonical = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _single_input(
    pairs: Path | None, words: Path | None, allow_fallback: bool
) -> tuple[Literal["pairs", "words"], Path]:
    if (pairs is None) == (words is None):
        raise ConfigError("pass exactly one of --pairs and --words")
    if pairs is not None:
        if allow_fallback:
            raise ConfigError("--allow-fallback applies to --words, not to --pairs")
        return "pairs", pairs
    assert words is not None  # exactly one of the two is None
    return "words", words


def _require_out_apart_from_input(out_dir: Path, source: Path) -> None:
    """Compares resolved paths, so a symlink or a `..` does not hide the clash."""
    if out_dir.resolve() == source.resolve().parent:
        raise ConfigError(
            f"--out-dir {out_dir} is the directory of the input file {source}; "
            "pass a different --out-dir"
        )


def _load_pairs(run: _CorpusRun, source: Path) -> _LoadedPairs:
    if run.source == "pairs":
        return _LoadedPairs(read_pairs_tsv(source, origin="parallel"), None)
    words = [
        word
        for line in _decode_utf8(_read_bytes(source), str(source)).split("\n")
        if (word := line.strip())
    ]
    pairs = rule_pairs(words, allow_fallback=run.allow_fallback)
    return _LoadedPairs(pairs, len(words))


def _commit_corpus(
    run: _CorpusRun,
    pairs: Sequence[WordPair],
    target: Path,
    input_files: list[ManifestFile],
) -> tuple[CorpusFiles, Path]:
    """Write the corpus files and, last, the manifest."""
    # The old manifest goes first, so that a failure part-way leaves none.
    begin_commit(target, _DATA_STAGE)
    files = build_corpus(
        pairs, target, valid_fraction=run.valid_fraction, seed=run.seed
    )
    manifest_path = _write_manifest(
        target,
        _DATA_STAGE,
        config_sha256=run.config_sha256(),
        seed=run.seed,
        inputs=input_files,
        outputs=[
            output_file(files.files[name], files.rows[name])
            for name in CORPUS_FILE_NAMES
        ],
        metrics={
            "valid_fraction": run.valid_fraction,
            "pairs_in": float(len(pairs)),
            **{f"dropped.{code}": float(n) for code, n in files.dropped.items()},
        },
    )
    return files, manifest_path


# ---- train ----------------------------------------------------------------------------


@app.command("train")
def train_converter(
    profile: Annotated[
        str,
        typer.Option(
            "--profile",
            help="tiny: a CPU run that memorises a small fixture; full: GPU.",
        ),
    ],
    data_dir: Annotated[
        Path,
        typer.Option(
            "--data-dir", help="Directory of the eight corpus files from build-data."
        ),
    ],
    run_dir: Annotated[
        Path,
        typer.Option(
            "--run-dir",
            help=(
                "Directory for the run: converter.yaml, the vocabulary, the "
                "checkpoints, the logs and the manifest."
            ),
        ),
    ],
    spm_config: Annotated[
        Path,
        typer.Option(
            "--spm-config",
            help="Character model config (vocab_size).",
        ),
    ] = DEFAULT_CHAR_MODEL_CONFIG,
    spm_prefix: Annotated[
        Path | None,
        typer.Option(
            "--spm-prefix",
            help=(
                "Where the character model is written: <prefix>.model and "
                "<prefix>.vocab. Default <run-dir>/converter_char."
            ),
        ),
    ] = None,
    threads: Annotated[
        int,
        typer.Option(
            "--threads", min=1, help="OpenMP and MKL threads of each Eole run."
        ),
    ] = _DEFAULT_THREADS,
    dry_run: Annotated[
        bool,
        typer.Option(
            "--dry-run",
            help=(
                "Check the corpus files, the configs and the run directory, print "
                "the character model path, the config path and the two Eole "
                "commands, and write and run nothing."
            ),
        ),
    ] = False,
    confirm_full: Annotated[
        bool,
        typer.Option(
            "--confirm-full",
            help="Allow a real run of the full profile, which takes GPU hours.",
        ),
    ] = False,
    timeout: Annotated[
        float | None,
        typer.Option(
            "--timeout",
            help=(
                "Seconds after which each Eole command (the vocabulary, then the "
                "training) is killed; no limit by default."
            ),
        ),
    ] = None,
) -> None:
    """Train the character model, then the converter: the character model is
    trained and checked on the corpus words, the Eole config is rendered, and
    Eole builds the vocabulary and trains from random initialisation."""
    with rinchen_errors_as_exit_codes():
        run = _plan_training(
            profile,
            data_dir,
            run_dir,
            spm_config,
            spm_prefix,
            threads,
            timeout,
            dry_run=dry_run,
            confirm_full=confirm_full,
        )
        # A dry run that passes must not be followed by a refusal for this.
        refuse_foreign_manifest(run.run_dir, _TRAIN_STAGE)
        if dry_run:
            for line in _planned_lines(run):
                typer.echo(line)
            return
        try:
            checkpoint, manifest_path = _commit_training(run)
        except OSError as exc:
            exit_with_error(
                f"cannot write the training outputs under {run.run_dir}: "
                f"{_describe_os_error(exc)}",
                FAILURE_EXIT_CODE,
            )
        typer.echo(f"trained {run.profile}: checkpoint {checkpoint}")
        typer.echo(f"manifest: {manifest_path}")


@dataclass(frozen=True)
class _TrainingRun:
    """A training run whose inputs have been checked and whose Eole config has been
    rendered and validated, but not written."""

    profile: Profile
    spm_config: Path
    char_config: CharModelConfig
    char_prefix: Path
    char_model: Path  # what the character trainer writes for `char_prefix`
    char_vocab: Path
    corpus_files: list[Path]
    run_dir: Path
    threads: int
    timeout: float | None
    cfg: dict[str, Any]
    commands: tuple[tuple[str, ...], tuple[str, ...]]  # build_vocab, then train

    @property
    def config_path(self) -> Path:
        return self.run_dir / _CONVERTER_CONFIG


def _plan_training(
    profile: str,
    data_dir: Path,
    run_dir: Path,
    spm_config: Path,
    spm_prefix: Path | None,
    threads: int,
    timeout: float | None,
    *,
    dry_run: bool,
    confirm_full: bool,
) -> _TrainingRun:
    """Everything a run needs to be known good before it starts, none of it written:
    the same checks for a dry run and a real one, so that a real run does not fail
    on something its dry run passed, nor after the character model is trained.

    The directories, the character model config and the character model prefix (by
    default `converter_char` in the run directory) are made absolute, so that
    `converter.yaml` and the manifest mean the same from any working directory.

    Raises ConfigError for an unknown profile, an unconfirmed full run, a character
    model config or an Eole config that does not hold (a `$` in a path included),
    DataError for a missing corpus file or a valid corpus without a row, and
    UpstreamError when `eole` is not installed."""
    selected = _profile(profile)
    if selected == "full" and not (dry_run or confirm_full):
        raise ConfigError(
            "--profile full is an operator step that takes GPU hours: pass "
            "--confirm-full to run it, or --dry-run to see what it would run"
        )
    char_config = load_char_model_config(spm_config)
    data_dir, run_dir = data_dir.resolve(), run_dir.resolve()
    spm_config = spm_config.resolve()
    corpus_files = _require_corpus_files(data_dir)
    _require_valid_rows(data_dir / _VALID_SOURCE_FILE)
    prefix = (spm_prefix or run_dir / _CHAR_MODEL_PREFIX).resolve()
    char_model, char_vocab = Path(f"{prefix}.model"), Path(f"{prefix}.vocab")
    cfg = render_converter_config(
        selected, data_dir=data_dir, spm_model=char_model, run_dir=run_dir
    )
    validate_converter_config(cfg, profile=selected)
    commands = eole_commands(run_dir / _CONVERTER_CONFIG)
    return _TrainingRun(
        selected,
        spm_config,
        char_config,
        prefix,
        char_model,
        char_vocab,
        corpus_files,
        run_dir,
        threads,
        timeout,
        cfg,
        commands,
    )


def _planned_lines(run: _TrainingRun) -> list[str]:
    """What a real run would write and run: the character model path, the config
    path and the two Eole commands."""
    build_vocab_command, train_command = run.commands
    return [
        str(run.char_model),
        str(run.config_path),
        shlex.join(build_vocab_command),
        shlex.join(train_command),
    ]


def _commit_training(run: _TrainingRun) -> tuple[Path, Path]:
    """Train the character model, then the converter, and write the manifest last.
    Returns the checkpoint file and the manifest."""
    # Hashed before it is read, so that the manifest vouches for the bytes that are
    # read below.
    input_files = _hash_inputs([*run.corpus_files, run.spm_config])
    # A list, not a generator: the check reads the texts again.
    texts = _corpus_texts(run.corpus_files)
    # Just before the first write, and after everything that can fail on bad input:
    # the old manifest goes first, so that a failure part-way leaves none.
    begin_commit(run.run_dir, _TRAIN_STAGE)
    train_char_model(texts, run.char_prefix, vocab_size=run.char_config.vocab_size)
    check_char_model(run.char_model, texts)
    run.run_dir.mkdir(parents=True, exist_ok=True)
    dump_config(run.cfg, run.config_path)
    # The config Eole is given, hashed before it runs.
    config_sha256 = sha256_of(run.config_path)
    checkpoint = (
        run_training(
            run.config_path,
            run.run_dir,
            threads=run.threads,
            timeout=run.timeout,
            profile=run.profile,
        )
        / CHECKPOINT_FILE
    )
    manifest_path = _write_manifest(
        run.run_dir,
        _TRAIN_STAGE,
        config_sha256=config_sha256,
        seed=int(run.cfg["seed"]),
        inputs=input_files,
        # Both character model files: the `.vocab` hash is the identity that holds
        # in every location, the `.model` embeds its own path.
        outputs=[
            output_file(path, None)
            for path in (run.char_model, run.char_vocab, run.config_path, checkpoint)
        ],
        metrics={"threads": float(run.threads)},
        packages=_TRAIN_PACKAGES,
    )
    return checkpoint, manifest_path


def _profile(value: str) -> Profile:
    if not _is_profile(value):
        raise ConfigError(
            f"unknown profile {value!r}: expected one of {', '.join(get_args(Profile))}"
        )
    return value


def _is_profile(value: str) -> TypeIs[Profile]:
    return value in get_args(Profile)


def _require_corpus_files(data_dir: Path) -> list[Path]:
    paths = [data_dir / name for name in CORPUS_FILE_NAMES]
    for path in paths:
        if not path.is_file():
            raise DataError(
                f"corpus file not found: {path}; build it with "
                "`rinchen convert-script build-data`"
            )
    return paths


def _require_valid_rows(valid_source: Path) -> None:
    """A corpus built elsewhere may hold an empty valid file, which `eole train`
    cannot validate on; `build_corpus` never writes one."""
    text = _decode_utf8(_read_bytes(valid_source), str(valid_source))
    if not any(text.split("\n")):
        raise DataError(
            f"{valid_source} has no rows: Eole cannot validate on it; build the "
            "corpus again with more pairs or another --valid-fraction"
        )


def _corpus_texts(corpus_files: Sequence[Path]) -> list[str]:
    """Every non-blank line of the train and valid files of the Cyrillic-to-script
    direction: the words of both sides, in the order of the files."""
    return [
        line
        for path in corpus_files
        if path.name.startswith(_CHAR_MODEL_CORPUS_PREFIX)
        for line in _decode_utf8(_read_bytes(path), str(path)).split("\n")
        if line
    ]


# ---- export ---------------------------------------------------------------------------


@app.command("export")
def export_model(
    run_dir: Annotated[
        Path,
        typer.Option("--run-dir", help="Run directory of `train`."),
    ],
    config: Annotated[
        Path | None,
        typer.Option(
            "--config",
            help="Eole config to read the checkpoint and the character model from; "
            "default <run-dir>/converter.yaml.",
        ),
    ] = None,
    out_dir: Annotated[
        Path | None,
        typer.Option(
            "--out-dir",
            help=(
                "Model directory to create; it must not exist. Default "
                "<run-dir>/ct2. It also takes the manifest."
            ),
        ),
    ] = None,
    timeout: Annotated[
        float,
        typer.Option(
            "--timeout",
            help="Seconds after which the CTranslate2 converter is killed.",
        ),
    ] = _DEFAULT_EXPORT_TIMEOUT_SECONDS,
) -> None:
    """Convert the trained checkpoint to a CTranslate2 model directory and copy
    the character model into it, so that the directory is all `apply` and `eval`
    need."""
    with rinchen_errors_as_exit_codes():
        run = _plan_export(run_dir, config, out_dir, timeout)
        try:
            manifest_path = _commit_export(run)
        except OSError as exc:
            exit_with_error(
                f"cannot write the export outputs under {run.target}: "
                f"{_describe_os_error(exc)}",
                FAILURE_EXIT_CODE,
            )
        typer.echo(f"exported to {run.target}")
        typer.echo(f"manifest: {manifest_path}")


@dataclass(frozen=True)
class _ExportRun:
    run_dir: Path
    config_path: Path
    paths: ConverterPaths
    target: Path
    timeout: float

    @property
    def checkpoint(self) -> Path:
        """The newest checkpoint file itself: Eole links it to a file in a step
        directory, and the manifest names the file that holds the weights."""
        return (self.paths.checkpoint_dir / CHECKPOINT_FILE).resolve()

    @property
    def input_paths(self) -> list[Path]:
        """What the model directory is made from: the checkpoint, the character model
        and the config and vocabulary the converter reads beside the checkpoint."""
        beside = [self.paths.checkpoint_dir / name for name in _CHECKPOINT_SIDE_FILES]
        return [
            self.checkpoint,
            self.paths.char_model,
            *(path for path in beside if path.is_file()),
        ]


def _plan_export(
    run_dir: Path, config: Path | None, out_dir: Path | None, timeout: float
) -> _ExportRun:
    """Raises ConfigError for a missing or unusable config or a model directory that
    exists, and DataError for a checkpoint or character model that is not a file."""
    config_path = config if config is not None else run_dir / _CONVERTER_CONFIG
    run = _ExportRun(
        run_dir,
        config_path,
        read_converter_paths(config_path),
        out_dir if out_dir is not None else run_dir / _CT2_DIR,
        timeout,
    )
    for path in (run.checkpoint, run.paths.char_model):
        if not path.is_file():
            raise DataError(f"file not found: {path}")
    _require_no_model_directory(run.target)
    return run


def _require_no_model_directory(target: Path) -> None:
    if target.exists() or target.is_symlink():
        raise ConfigError(f"{target} exists: remove it, or pass another --out-dir")


def _commit_export(run: _ExportRun) -> Path:
    """Convert the checkpoint, put the character model beside the result and write
    the manifest last. A failure removes the model directory, which this run created:
    `_plan_export` saw that it did not exist."""
    # Hashed before the work, so that the manifest vouches for what was converted.
    # The converter refuses an existing directory, so none can hold a manifest to
    # erase: no commit start is needed.
    config_sha256 = _config_sha256(run.config_path)
    input_files = _hash_inputs(run.input_paths)
    _require_the_char_model_train_recorded(run, input_files)
    # Hashing a large checkpoint takes a while, and the plan's check is older than
    # that. Outside the `try`, because its cleanup removes `run.target`: a directory
    # that someone else created in the meantime is not this run's to remove.
    _require_no_model_directory(run.target)
    try:
        export_ct2(run.paths.checkpoint_dir, run.target, timeout=run.timeout)
        _copy_into_place(run.paths.char_model, run.target / _CHAR_MODEL_FILE)
        return _write_manifest(
            run.target,
            _EXPORT_STAGE,
            config_sha256=config_sha256,
            seed=0,
            inputs=input_files,
            outputs=[output_file(path, None) for path in _files_below(run.target)],
            packages=_EXPORT_PACKAGES,
        )
    except UpstreamError as exc:
        shutil.rmtree(run.target, ignore_errors=True)
        raise UpstreamError(
            f"{exc}\nthe unfinished model directory {run.target} was removed; "
            f"the converter's log is {export_log_path(run.target)}"
        ) from exc
    except BaseException:
        shutil.rmtree(run.target, ignore_errors=True)
        raise


def _require_the_char_model_train_recorded(
    run: _ExportRun, input_files: list[ManifestFile]
) -> None:
    """Refuse a character model that is not the one `train` recorded for this run.

    The model path is in the config, and a path can be shared by several runs, so a
    later run may have overwritten the file this checkpoint was trained with. Only a
    train manifest in the run directory that lists the model can vouch for it: a
    run directory built by hand, or a manifest of another stage, is trusted."""
    train = store.read(run.run_dir)
    if train is None or train.stage != _TRAIN_STAGE:
        return
    model = run.paths.char_model.resolve()
    recorded = next((o for o in train.outputs if Path(o.path).resolve() == model), None)
    actual = next((i for i in input_files if Path(i.path).resolve() == model), None)
    if recorded is None or actual is None or recorded.sha256 == actual.sha256:
        return
    raise DataError(
        f"the character model {run.paths.char_model} has sha256 {actual.sha256}, but "
        f"the train manifest {store.manifest_path(run.run_dir)} recorded "
        f"{recorded.sha256}: another run overwrote it after this checkpoint was "
        "trained; train again with a character model of its own (--spm-prefix)"
    )


def _copy_into_place(source: Path, destination: Path) -> None:
    """Copy `source` so that `destination` appears whole or not at all."""
    part_path = unique_part_path(destination)
    try:
        shutil.copyfile(source, part_path)
        part_path.replace(destination)
    finally:
        part_path.unlink(missing_ok=True)


def _files_below(directory: Path) -> list[Path]:
    """Every file under `directory` but its manifest, in path order."""
    manifest = store.manifest_path(directory)
    return sorted(
        path for path in directory.rglob("*") if path.is_file() and path != manifest
    )


# ---- apply ----------------------------------------------------------------------------


@app.command("apply")
def apply_script(
    model_dir: Annotated[
        Path,
        typer.Option(
            "--model-dir", help="Model directory written by `export` (or by hand)."
        ),
    ],
    target: Annotated[str, typer.Option("--target", help=_TARGET_HELP)],
    file: Annotated[
        Path | None,
        typer.Argument(
            metavar="FILE",
            help="UTF-8 text file to convert; standard input when omitted.",
        ),
    ] = None,
) -> None:
    """Convert a text to the target script and write it to standard output
    exactly, with no newline added. The whole input is one text, its line breaks
    included. A summary of the words the model could not convert goes to standard
    error. The whole input is held in memory, in several copies."""
    with rinchen_errors_as_exit_codes():
        script = _target(target)
        text = _read_input(file)
        if not text:
            return
        conversion = load_converter(model_dir).convert(text, script)
        stdout = typer.get_binary_stream("stdout")
        stdout.write(conversion.text.encode("utf-8"))
        stdout.flush()
        typer.echo(
            f"rule_fallbacks={conversion.rule_fallbacks} "
            f"invalid_outputs={conversion.invalid_outputs}",
            err=True,
        )


def _read_input(file: Path | None) -> str:
    if file is None:
        return _decode_utf8(typer.get_binary_stream("stdin").read(), "standard input")
    return _decode_utf8(_read_bytes(file), str(file))


# ---- eval -----------------------------------------------------------------------------


@app.command("eval")
def evaluate(
    model_dir: Annotated[
        Path, typer.Option("--model-dir", help="Model directory written by `export`.")
    ],
    pairs: Annotated[
        Path,
        typer.Option(
            "--pairs",
            help="UTF-8 TSV headed 'cyrillic<TAB>script' of reference word pairs.",
        ),
    ],
    target: Annotated[
        str,
        typer.Option(
            "--target",
            help=(
                f"{_TARGET_HELP} Mong converts the Cyrillic side and is scored "
                "against the script side; Cyrl the other way round."
            ),
        ),
    ] = "Mong",
) -> None:
    """Convert one side of each reference pair and print five lines: the character
    error rate (`cer`), the share of exactly right words (`word_accuracy`), the
    number of words (`n`), and two counts of words whose model output was unusable:
    those the rules spelled instead (`rule_fallbacks`) and those that stayed as they
    were (`invalid_outputs`)."""
    with rinchen_errors_as_exit_codes():
        script = _target(target)
        word_pairs = read_pairs_tsv(pairs, origin="reference")
        if not word_pairs:
            raise DataError(f"{pairs} holds no pairs")
        converter = load_converter(model_dir)
        if script == "Mong":
            sources = [pair.cyrillic for pair in word_pairs]
            references = [pair.script for pair in word_pairs]
        else:
            sources = [pair.script for pair in word_pairs]
            references = [pair.cyrillic for pair in word_pairs]
        # One call for the whole file: its line breaks are text between words, which
        # the converter leaves alone, so each word is converted as if alone.
        conversion = converter.convert("\n".join(sources), script)
        hypotheses = conversion.text.split("\n")
        if len(hypotheses) != len(sources):
            raise UpstreamError(
                f"the converter returned {len(hypotheses)} lines for "
                f"{len(sources)} words"
            )
        scores = score_conversions(references, hypotheses)
        typer.echo(f"cer {scores.cer:.4f}")
        typer.echo(f"word_accuracy {scores.word_accuracy:.4f}")
        typer.echo(f"n {scores.n}")
        typer.echo(f"rule_fallbacks {conversion.rule_fallbacks}")
        typer.echo(f"invalid_outputs {conversion.invalid_outputs}")


# ---- shared ---------------------------------------------------------------------------


def _target(value: str) -> Target:
    if not _is_target(value):
        raise ConfigError(
            f"unknown target {value!r}: expected one of {', '.join(get_args(Target))}"
        )
    return value


def _is_target(value: str) -> TypeIs[Target]:
    return value in get_args(Target)


def _hash_inputs(paths: list[Path]) -> list[ManifestFile]:
    try:
        return build_inputs(paths)
    except OSError as exc:
        raise DataError(f"cannot read the input files: {exc}") from exc


def _config_sha256(path: Path) -> str:
    try:
        return sha256_of(path)
    except OSError as exc:
        raise DataError(f"cannot read the config {path}: {exc}") from exc


def _describe_os_error(exc: OSError) -> str:
    """The failure with the path the operating system reported, when it did."""
    if exc.filename is None or not exc.strerror:
        return str(exc)
    return f"{exc.strerror}: {exc.filename}"


def _read_bytes(path: Path) -> bytes:
    try:
        return path.read_bytes()
    except OSError as exc:
        raise DataError(f"cannot read {path}: {exc}") from exc


def _decode_utf8(data: bytes, source: str) -> str:
    """`data` as UTF-8 text, with nothing changed: a byte order mark stays and a
    line end stays what it was (reading a file as text would turn CRLF into LF)."""
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise DataError(f"{source} is not valid UTF-8: {exc}") from exc


def _write_manifest(
    stage_dir: Path,
    stage: str,
    *,
    config_sha256: str,
    seed: int,
    inputs: list[ManifestFile],
    outputs: list[ManifestFile],
    metrics: dict[str, float] | None = None,
    packages: Sequence[str] = (),
) -> Path:
    """Write the manifest of `stage`. It comes last in every command, so that it
    never describes outputs that are not there."""
    manifest = Manifest(
        stage=stage,
        created=datetime.now(UTC).isoformat(),
        code_version=code_version(),
        config_sha256=config_sha256,
        seed=seed,
        inputs=inputs,
        outputs=outputs,
        metrics=metrics or {},
        tool_versions=tool_versions(packages),
    )
    return store.write(stage_dir, manifest)
