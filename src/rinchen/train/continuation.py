"""Which checkpoint may start a continuation, and whether a config may continue it (D9).

Every Rinchen model is trained from random init. The one thing that may enter a run besides
its own checkpoints is a checkpoint of OUR OWN teacher, so that it can be trained on (on
document windows, say): `rinchen train launch --continue-from SOURCE` starts a NEW run whose
first attempt loads the weights of SOURCE. This module decides what SOURCE may be, and never
accepts a checkpoint that our own launcher did not write and our own manifest does not vouch
for. `verify_source` accepts exactly two kinds of directory:

    step      the FINAL `step_<N>` directory of a run that `rinchen train launch` finished:
              the run directory (the directory that holds its `launch.json`, at most three
              levels above the step) has a `manifest.json` of stage `train_launch`, status
              ok, which lists the four files of THIS step directory with the SHA-256 they
              have now. The manifest names the final step only, so an earlier step of the
              same run is refused, and so is a step directory renamed to another number.
    averaged  a directory `rinchen train average` wrote: it holds `averaged_from.json` and
              a `manifest.json` of stage `train_average`, status ok, listing its weights,
              `config.json`, `vocab.json` and `averaged_from.json` with their SHA-256. It
              has no `optimizer.pt` (a file Eole would load that no manifest vouches for,
              so one is refused), and, like a step, no other `model.*.safetensors` than the
              one the manifest lists: Eole loads every shard of the directory. The two
              accounts of what it was made from must agree: `averaged_from.json` names at
              least two source checkpoints, the manifest lists the weights of the same
              ones (same directories, same SHA-256, same order) as inputs, and the weights
              the record names are the ones the manifest vouches for.

Nothing else is a source: no manifest, the manifest of another stage, a status that is not
ok, a manifest that does not list a file, a file whose bytes are not the ones the manifest
saw, an incomplete directory (`checkpoints.is_complete`), a symbolic link, a path with `$` or
a glob character, a directory inside the new run or one that holds it. A source that is not
usable is a `ConfigError`; a manifest, run record or vocabulary that cannot be read is a
`DataError`. No message holds what a file contains: only its name and what is wrong.

The manifest lists each file by the path it had when it was written, and a run directory
may be moved or copied since, so a file is matched to its entry by its NAME (the entries
whose last path component is `model.00.safetensors`, ...), never by the whole path. A name
that two entries share is ambiguous and refused. For a step the entries must lie in a
directory named like the source's, which is how a non-final step is told. The SHA-256 of
each file is taken once, in `verify_source`, before any check reads the file, and kept in
the `SourceProvenance`; `check_continuation` reads `config.json` and `vocab.json` (small
files) only through the digest it holds, so what it checks is what was vouched for. The
verification is not atomic with Eole's own read of the files: it guards against a mistake,
a stale directory or a foreign one, not against a writer that swaps a file after the hash.

`check_continuation` is the second gate, on the config that would continue the source (T6):
the architecture must be stated and equal to the source's, the optimizer options must be
stated (Eole inherits every omitted one from the checkpoint), the SentencePiece model the
config states must be the one the source was trained with (the copy Eole saved into its
directory, compared by SHA-256), and the vocabulary of the source must hold the symbols
the new data needs, `<doc>` and `<sep>` first among them.

The launcher (`rinchen.train.launch`) sequences these; it takes the lock of the run before
it calls `verify_source`, and writes nothing until every check has passed.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Final

from rinchen.errors import ConfigError, DataError
from rinchen.manifest import store
from rinchen.manifest.model import Manifest, ManifestFile
from rinchen.manifest.stage import sha256_of
from rinchen.train.average import AVERAGED_FROM_FILE
from rinchen.train.checkpoints import (
    CONFIG_FILE,
    LAUNCH_FILE,
    OPTIMIZER_FILE,
    RUN_DIRECTORY_LEVELS,
    VOCAB_FILE,
    WEIGHTS_FILE,
    ContinuationKind,
    ContinuedFrom,
    architecture_mismatch,
    is_complete,
    read_launch_record,
    run_directory_above,
    step_of,
)
from rinchen.train.eole_cfg import reject_dollar, reject_glob_characters
from rinchen.utils.integers import is_strict_int

# The manifest stages whose outputs a continuation accepts: the commands write them
# (the `rinchen train launch` and `average` commands), and the status key each puts its
# verdict under.
LAUNCH_STAGE: Final = "train_launch"
AVERAGE_STAGE: Final = "train_average"
_STATUS_KEY: Final = {LAUNCH_STAGE: "launch", AVERAGE_STAGE: "average"}

# The files that must be listed in the manifest and match it, by kind.
_STEP_FILES: Final = (WEIGHTS_FILE, OPTIMIZER_FILE, CONFIG_FILE, VOCAB_FILE)
_AVERAGED_FILES: Final = (WEIGHTS_FILE, CONFIG_FILE, VOCAB_FILE, AVERAGED_FROM_FILE)
_SHARDS: Final = "model.*.safetensors"
# `rinchen train average` refuses fewer than two checkpoints.
_MIN_AVERAGED_SOURCES: Final = 2
_DIGEST_SHOWN: Final = 12

# The keys of the `model` section a continuation states: it says which architecture it
# believes it continues, and `architecture_mismatch` compares them with the source's.
_MODEL_KEYS: Final = (
    ("hidden_size",),
    ("embeddings", "word_vec_size"),
    ("heads",),
    ("transformer_ff",),
    ("encoder", "layers"),
    ("decoder", "layers"),
    ("embeddings", "position_encoding_type"),
)
# What Eole takes from the checkpoint when the new config omits it.
_OPTIMIZER_OPTIONS: Final = (
    "learning_rate",
    "warmup_steps",
    "decay_method",
    "optim",
    "adam_beta2",
)
# The symbols the document windows always carry, whatever the prefixes say.
_WINDOW_SYMBOLS: Final = ("<doc>", "<sep>")
_SENTENCEPIECE_KEYS: Final = ("src_subword_model", "tgt_subword_model")
_ANGLE_TOKEN: Final = re.compile(r"<[^<>\s]+>")
_PREFIXES: Final = ("src_prefix", "tgt_prefix")
_VOCABULARY_SIDES: Final = ("src", "tgt")


@dataclass(frozen=True, slots=True)
class SourceProvenance:
    """A source that `verify_source` found vouched for.

    kind: `step` or `averaged`. path: the directory, absolute and as given (not resolved).
    The SHA-256 are those of the files as they were when verified, each taken once:
    `optimizer_sha256` is None for an averaged model. manifest_sha256 and manifest_stage:
    the manifest that vouched for them. step: N of `step_<N>`, None for an averaged model.
    source_run: the run directory of a step (it holds the manifest), None for an averaged
    model (the manifest is in `path`).
    """

    kind: ContinuationKind
    path: Path
    weights_sha256: str
    optimizer_sha256: str | None
    config_sha256: str
    vocab_sha256: str
    manifest_sha256: str
    manifest_stage: str
    step: int | None
    source_run: Path | None

    def as_record(self) -> ContinuedFrom:
        """What `launch.json` records of this source."""
        return ContinuedFrom(
            kind=self.kind,
            path=str(self.path),
            weights_sha256=self.weights_sha256,
            optimizer_sha256=self.optimizer_sha256,
            config_sha256=self.config_sha256,
            vocab_sha256=self.vocab_sha256,
            manifest_sha256=self.manifest_sha256,
            manifest_stage=self.manifest_stage,
            step=self.step,
            source_run=None if self.source_run is None else str(self.source_run),
        )

    @classmethod
    def from_record(cls, record: ContinuedFrom) -> SourceProvenance:
        """The source a launch record says the run was started from."""
        return cls(
            kind=record.kind,
            path=Path(record.path),
            weights_sha256=record.weights_sha256,
            optimizer_sha256=record.optimizer_sha256,
            config_sha256=record.config_sha256,
            vocab_sha256=record.vocab_sha256,
            manifest_sha256=record.manifest_sha256,
            manifest_stage=record.manifest_stage,
            step=record.step,
            source_run=None if record.source_run is None else Path(record.source_run),
        )

    @property
    def manifest_path(self) -> Path:
        """The manifest that vouched for the source."""
        return store.manifest_path(self.source_run or self.path)

    def files(self) -> tuple[tuple[Path, str], ...]:
        """The files the run was started from with their SHA-256, then the manifest: the
        inputs a manifest of the new run lists (nothing is read again)."""
        entries = [(self.path / WEIGHTS_FILE, self.weights_sha256)]
        if self.optimizer_sha256 is not None:
            entries.append((self.path / OPTIMIZER_FILE, self.optimizer_sha256))
        entries += [
            (self.path / CONFIG_FILE, self.config_sha256),
            (self.path / VOCAB_FILE, self.vocab_sha256),
            (self.manifest_path, self.manifest_sha256),
        ]
        return tuple(entries)


@dataclass(frozen=True, slots=True)
class _Origin:
    """What a source of one kind must have: the directory that holds its manifest, the
    stage that manifest is of, the files it must list, and for a step the run."""

    kind: ContinuationKind
    manifest_dir: Path
    stage: str
    names: tuple[str, ...]
    step: int | None
    source_run: Path | None


def source_path(source: Path) -> Path:
    """`source` made absolute (not resolved: the path Eole is given is the path the
    manifests and the record name).

    Raises:
        ConfigError: the path holds `$`, which Eole would expand as a variable, or a
            glob character, which it reads as a pattern when it loads a checkpoint.
    """
    path = Path(os.path.abspath(source))
    reject_dollar("continuation source", path)
    reject_glob_characters("continuation source", path)
    return path


def require_source_apart(source: Path, *, run_dir: Path, vocabulary: Path) -> None:
    """Refuse a source that is the new run directory or inside it, one that holds it, and a
    new vocabulary file that would be written inside it: the source is never written, and
    nothing of the new run may be mistaken for it. Links are resolved, so a shortcut does
    not hide it. Nothing is read from the source.

    Raises:
        ConfigError: one of the three.
    """
    where = Path(os.path.realpath(source))
    run = Path(os.path.realpath(run_dir))
    if where.is_relative_to(run):
        raise _refuse(source, f"it is the new run directory {run_dir} or inside it")
    if run.is_relative_to(where):
        raise _refuse(source, f"the new run directory {run_dir} is inside the source")
    if Path(os.path.realpath(vocabulary)).is_relative_to(where):
        raise _refuse(
            source,
            f"the vocabulary file {vocabulary} of the new run would be written "
            "inside the source",
        )


def verify_source(source: Path) -> SourceProvenance:
    """The provenance of `source` when it is the final step of a finished run or an
    averaged model that its manifest vouches for (see the module's text); raises when it
    is anything else. Reads the files of the source and writes nothing.

    Raises:
        ConfigError: `source` is not usable (see above); the message names the file and
            what is wrong.
        DataError: the manifest, the run record or a file cannot be read, or is not what
            it must be.
    """
    path = source_path(source)
    _require_a_real_directory(path)
    origin = (
        _averaged_origin(path)
        if (path / AVERAGED_FROM_FILE).is_file()
        else _step_origin(path)
    )
    manifest_file = store.manifest_path(origin.manifest_dir)
    manifest = _vouching_manifest(path, origin, manifest_file)
    listed = _listed_digests(path, origin, manifest, manifest_file)
    _require_complete(path, origin)
    _require_plain_files_and_no_other_shard(path, origin)
    digests = _digests_matching_the_manifest(path, origin, listed, manifest_file)
    if origin.kind == "averaged":
        _require_record_to_agree_with_manifest(path, manifest, manifest_file, digests)
    return SourceProvenance(
        kind=origin.kind,
        path=path,
        weights_sha256=digests[WEIGHTS_FILE],
        optimizer_sha256=digests.get(OPTIMIZER_FILE),
        config_sha256=digests[CONFIG_FILE],
        vocab_sha256=digests[VOCAB_FILE],
        manifest_sha256=_hashed(manifest_file, path),
        manifest_stage=manifest.stage,
        step=origin.step,
        source_run=origin.source_run,
    )


def check_continuation(cfg: Mapping[str, Any], provenance: SourceProvenance) -> None:
    """Refuse the config `cfg` as the continuation of the source `provenance` (T6), naming
    every check that fails, each with its own message, in one error:

    - the `model` section states every key of the architecture guard
      (`hidden_size`, `embeddings.word_vec_size`, `heads`, `transformer_ff`,
      `encoder.layers`, `decoder.layers`, `embeddings.position_encoding_type`), the
      `config.json` of the source records each, and `checkpoints.architecture_mismatch`
      finds no key they differ in (both values are named);
    - the optimizer options (`learning_rate`, `warmup_steps`, `decay_method`, `optim`,
      `adam_beta2`) are stated in `training`: Eole inherits an omitted one from the
      checkpoint, which a new run must not rely on;
    - the SentencePiece model the config states for both sides
      (`transforms_configs.sentencepiece.src_subword_model` and `tgt_subword_model`) has
      the bytes of the copy Eole saved into the source directory, the file its
      `config.json` names (`${MODEL_PATH}/<name>`): the vocabulary of the source is the
      pieces that model gave its data, so another model would read pieces the teacher
      lacks as `<unk>` without a word. A source with no recorded or no saved copy cannot
      be compared and is refused. The manifest does not vouch for the copy;
    - the `vocab.json` of the source holds `<doc>`, `<sep>` and every token written like a
      symbol (`<...>`) in the prefixes of the data corpora, on both sides. A teacher
      trained before the vocabulary invariant lacks the first two and would read them as
      `<unk>`: the message says so.

    `config.json` and `vocab.json` are read here, through the SHA-256 `provenance` holds.

    Raises:
        ConfigError: a check fails, or one of the two files is not the one that was
            verified.
        DataError: one of the two files is not what Eole writes (JSON, a vocabulary of
            two lists of tokens).
    """
    problems = [*_unstated_model_keys(cfg), *_unstated_optimizer_options(cfg)]
    saved = _verified_json(
        provenance.path / CONFIG_FILE, provenance.config_sha256, "config"
    )
    problems += _architecture_problems(cfg, saved)
    problems += _sentencepiece_problems(cfg, saved, provenance.path)
    vocabulary = _vocabulary_of(provenance)
    problems += _vocabulary_problems(cfg, vocabulary)
    if problems:
        raise _refuse(provenance.path, "; ".join(problems))


# ---- verify_source: where the source is and who vouches for it -------------------------


def _refuse(path: Path, problem: str) -> ConfigError:
    return ConfigError(f"cannot continue from {path}: {problem}")


def _require_a_real_directory(path: Path) -> None:
    if path.is_symlink():
        raise _refuse(
            path,
            "it is a symbolic link; name the checkpoint directory itself, so that what "
            "is hashed is what Eole loads",
        )
    if not os.path.lexists(path):
        raise _refuse(path, "it does not exist")
    if not path.is_dir():
        raise _refuse(path, "it is not a directory")


def _averaged_origin(path: Path) -> _Origin:
    return _Origin("averaged", path, AVERAGE_STAGE, _AVERAGED_FILES, None, None)


def _step_origin(path: Path) -> _Origin:
    step = step_of(path)
    if step is None:
        raise _refuse(
            path,
            f"it is neither a step_<N> directory of a run nor an averaged model (an "
            f"averaged model holds {AVERAGED_FROM_FILE}, which is missing here)",
        )
    run = _run_directory_above(path)
    try:
        read_launch_record(run)
    except DataError as error:
        raise DataError(f"cannot continue from {path}: {error}") from error
    return _Origin("step", run, LAUNCH_STAGE, _STEP_FILES, step, run)


def _run_directory_above(path: Path) -> Path:
    """The nearest directory within three levels above `path` that holds `launch.json`."""
    run = run_directory_above(path)
    if run is None:
        raise _refuse(
            path,
            f"no {LAUNCH_FILE} within {RUN_DIRECTORY_LEVELS} levels above it, so it is "
            "not a checkpoint of a run that `rinchen train launch` wrote",
        )
    return run


def _vouching_manifest(path: Path, origin: _Origin, manifest_file: Path) -> Manifest:
    try:
        manifest = store.read(origin.manifest_dir)
    except DataError as error:
        # The reader's message holds the validation errors, which quote the input.
        raise DataError(
            f"cannot continue from {path}: {manifest_file}: is not a readable Rinchen "
            "manifest"
        ) from error
    if manifest is None:
        raise _refuse(
            path,
            f"no manifest vouches for it: {manifest_file} does not exist (`rinchen "
            "train launch` and `rinchen train average` write it when they finish)",
        )
    if manifest.stage != origin.stage:
        raise _refuse(
            path,
            f"{manifest_file} is the manifest of stage {manifest.stage!r}, not of "
            f"{origin.stage!r}",
        )
    key = _STATUS_KEY[origin.stage]
    if manifest.status.get(key) != "ok":
        raise _refuse(path, f"the status in {manifest_file} does not say {key}: ok")
    return manifest


def _listed_digests(
    path: Path, origin: _Origin, manifest: Manifest, manifest_file: Path
) -> dict[str, str]:
    """The SHA-256 the manifest lists for each file the source must have, by file name."""
    by_name: dict[str, list[ManifestFile]] = {}
    for item in manifest.outputs:
        by_name.setdefault(Path(item.path).name, []).append(item)
    listed: dict[str, str] = {}
    for name in origin.names:
        entries = by_name.get(name, [])
        if not entries:
            raise _refuse(path, f"the manifest {manifest_file} does not list {name}")
        if len(entries) > 1:
            raise _refuse(
                path,
                f"the manifest {manifest_file} lists {name} more than once, so it "
                "does not say which file it means",
            )
        directory = Path(entries[0].path).parent.name
        if origin.step is not None and directory != path.name:
            raise _refuse(
                path,
                f"the manifest {manifest_file} lists the files of {directory}, not of "
                f"{path.name}: only the final step of a finished run is vouched for",
            )
        listed[name] = entries[0].sha256
    return listed


def _require_complete(path: Path, origin: _Origin) -> None:
    problem = is_complete(path)
    if origin.kind == "step":
        if problem is not None:
            raise _refuse(path, f"it is not complete: {problem}")
        return
    # An averaged model has no optimizer state, and nothing else may be wrong with it.
    if problem is None:
        raise _refuse(
            path,
            f"it holds {OPTIMIZER_FILE}, which an averaged model never has and no "
            "manifest vouches for, but Eole would load",
        )
    if problem != f"{OPTIMIZER_FILE}: is missing":
        raise _refuse(path, f"it is not complete: {problem}")


def _require_plain_files_and_no_other_shard(path: Path, origin: _Origin) -> None:
    for name in origin.names:
        file = path / name
        if file.is_symlink() or not file.is_file():
            raise _refuse(path, f"{name} is not a regular file (a link, say)")
    others = sorted(
        shard.name for shard in path.glob(_SHARDS) if shard.name != WEIGHTS_FILE
    )
    if others:
        raise _refuse(
            path,
            f"it holds {', '.join(others)} beside {WEIGHTS_FILE}: Eole loads every "
            "shard of the directory and the manifest vouches for one",
        )


def _digests_matching_the_manifest(
    path: Path, origin: _Origin, listed: dict[str, str], manifest_file: Path
) -> dict[str, str]:
    digests: dict[str, str] = {}
    for name in origin.names:
        digest = _hashed(path / name, path)
        if digest != listed[name]:
            raise _refuse(
                path,
                f"{name} differs from the manifest {manifest_file}: its SHA-256 is "
                f"{digest[:_DIGEST_SHOWN]}... and the manifest lists "
                f"{listed[name][:_DIGEST_SHOWN]}...; the file changed after the "
                "manifest vouched for it",
            )
        digests[name] = digest
    return digests


def _require_record_to_agree_with_manifest(
    path: Path, manifest: Manifest, manifest_file: Path, digests: dict[str, str]
) -> None:
    """`averaged_from.json` and the manifest are two accounts, written by one command, of
    the checkpoints an average was made from and of the weights it holds: refuse a
    directory whose two accounts differ, or that names fewer than two checkpoints (an
    average of one is not one `rinchen train average` wrote). A guard against a mistake,
    not against a manifest written by hand to match a record written by hand."""
    record_file = path / AVERAGED_FROM_FILE
    record = _verified_json(
        record_file, digests[AVERAGED_FROM_FILE], AVERAGED_FROM_FILE
    )
    recorded = _recorded_sources(record, record_file)
    weights = _recorded_weights(record, record_file)
    if len(recorded) < _MIN_AVERAGED_SOURCES:
        raise _refuse(
            path,
            f"{AVERAGED_FROM_FILE} names {len(recorded)} source checkpoints: an "
            "average of fewer than two is not one `rinchen train average` wrote",
        )
    # The manifest lists the weights file of each checkpoint, the record the directory.
    listed = [(str(Path(item.path).parent), item.sha256) for item in manifest.inputs]
    if recorded != listed:
        raise _refuse(
            path,
            f"{AVERAGED_FROM_FILE} names {len(recorded)} source checkpoints and "
            f"{manifest_file} lists {len(listed)} inputs, and they are not the same "
            "checkpoints, with the same SHA-256, in the same order",
        )
    if weights != digests[WEIGHTS_FILE]:
        raise _refuse(
            path,
            f"{AVERAGED_FROM_FILE} records other weights than the {WEIGHTS_FILE} the "
            f"manifest {manifest_file} vouches for",
        )


def _recorded_sources(record: Mapping[str, Any], file: Path) -> list[tuple[str, str]]:
    """The `(directory, SHA-256)` of each source `averaged_from.json` names."""
    sources = record.get("sources")
    if not isinstance(sources, list):
        raise _not_a_record(file, "`sources` must be a list")
    recorded: list[tuple[str, str]] = []
    for source in sources:
        if (
            not isinstance(source, dict)
            or not isinstance(source.get("path"), str)
            or not isinstance(source.get("sha256"), str)
            or not is_strict_int(source.get("step"))
        ):
            raise _not_a_record(
                file, "each source must have a `path`, a `step` and a `sha256`"
            )
        recorded.append((source["path"], source["sha256"]))
    return recorded


def _recorded_weights(record: Mapping[str, Any], file: Path) -> str:
    weights = record.get("weights")
    digest = weights.get("sha256") if isinstance(weights, dict) else None
    if not isinstance(digest, str):
        raise _not_a_record(file, "`weights` must have a `sha256`")
    return digest


def _not_a_record(file: Path, problem: str) -> DataError:
    return DataError(
        f"{file}: is not a record `rinchen train average` wrote: {problem}"
    )


def _hashed(file: Path, source: Path) -> str:
    try:
        return sha256_of(file)
    except OSError as error:
        raise DataError(
            f"cannot continue from {source}: cannot read {file}: {error.strerror}"
        ) from error


# ---- check_continuation ----------------------------------------------------------------


def _dotted(path: tuple[str, ...]) -> str:
    return ".".join(path)


def _stated(section: object, path: tuple[str, ...]) -> bool:
    node = section
    for key in path:
        if not isinstance(node, Mapping) or key not in node:
            return False
        node = node[key]
    return True


def _unstated_model_keys(cfg: Mapping[str, Any]) -> list[str]:
    model = cfg.get("model")
    return [
        f"model.{_dotted(path)} is not stated in the config: a continuation must say "
        "which architecture it believes it continues"
        for path in _MODEL_KEYS
        if not _stated(model, path)
    ]


def _unstated_optimizer_options(cfg: Mapping[str, Any]) -> list[str]:
    training = cfg.get("training")
    return [
        f"training.{key} is not stated in the config: Eole would inherit it from the "
        "checkpoint of the source, and a continuation must not rely on that"
        for key in _OPTIMIZER_OPTIONS
        if not _stated(training, (key,))
    ]


def _architecture_problems(
    cfg: Mapping[str, Any], saved: Mapping[str, Any]
) -> list[str]:
    saved_model = saved.get("model")
    unrecorded = [
        f"{CONFIG_FILE} of the source does not record model.{_dotted(path)}, so the "
        "architecture cannot be checked"
        for path in _MODEL_KEYS
        if not _stated(saved_model, path)
    ]
    model = cfg.get("model")
    if unrecorded or not isinstance(model, Mapping):
        return unrecorded
    mismatch = architecture_mismatch(saved, model)
    return [] if mismatch is None else [str(mismatch)]


def _vocabulary_of(provenance: SourceProvenance) -> dict[str, frozenset[str]]:
    file = provenance.path / VOCAB_FILE
    document = _verified_json(file, provenance.vocab_sha256, "vocabulary")
    sides: dict[str, frozenset[str]] = {}
    for side in _VOCABULARY_SIDES:
        tokens = document.get(side)
        if not isinstance(tokens, list) or not all(
            isinstance(token, str) for token in tokens
        ):
            raise DataError(
                f"{file}: is not a vocabulary as Eole writes it: {side!r} must be a "
                "list of tokens"
            )
        sides[side] = frozenset(tokens)
    return sides


def _sentencepiece_problems(
    cfg: Mapping[str, Any], saved: Mapping[str, Any], source: Path
) -> list[str]:
    """What is wrong with the SentencePiece models `cfg` states beside the one the source
    was trained with (the copy in `source` that its saved `config.json` names), per side.
    Each file is hashed once."""
    stated = _sentencepiece_section(cfg)
    recorded = _sentencepiece_section(saved)
    digests: dict[Path, str] = {}

    def digest(file: Path) -> str:
        if file not in digests:
            digests[file] = _hashed(file, source)
        return digests[file]

    problems: list[str] = []
    for key in _SENTENCEPIECE_KEYS:
        where = f"transforms_configs.sentencepiece.{key}"
        model = stated.get(key)
        if not isinstance(model, str) or not model:
            problems.append(
                f"{where} is not stated in the config: a continuation must say which "
                "SentencePiece model cuts its text, and it must be the one the source "
                "was trained with"
            )
            continue
        name = _copy_name(recorded.get(key))
        copy = None if name is None else source / name
        if name is None:
            problems.append(
                f"the config.json of the source does not record its {key}, so there is "
                f"no SentencePiece model of the source to compare {where} with"
            )
        elif copy is None or not copy.is_file():
            problems.append(
                f"the source has no copy of its SentencePiece model: its config.json "
                f"names {name}, which is not a file in {source}"
            )
        if not Path(model).is_file():
            problems.append(f"{where} names {model}, which is not a file")
        elif copy is not None and copy.is_file():
            mine, theirs = digest(Path(model)), digest(copy)
            if mine != theirs:
                problems.append(
                    f"{where} ({model}, SHA-256 {mine[:_DIGEST_SHOWN]}...) is not the "
                    f"SentencePiece model the source was trained with ({name} in the "
                    f"source, SHA-256 {theirs[:_DIGEST_SHOWN]}...): the vocabulary of "
                    "the source is the pieces that model gave its data, and a piece it "
                    "lacks would be read as <unk>"
                )
    return problems


def _sentencepiece_section(cfg: Mapping[str, Any]) -> Mapping[str, Any]:
    configs = cfg.get("transforms_configs")
    section = configs.get("sentencepiece") if isinstance(configs, Mapping) else None
    return section if isinstance(section, Mapping) else {}


def _copy_name(recorded: object) -> str | None:
    """The file name of the copy Eole saved: the last component of the path the saved
    `config.json` records (`${MODEL_PATH}/<name>`)."""
    if not isinstance(recorded, str) or not recorded:
        return None
    return PurePosixPath(recorded).name or None


def _needed_symbols(cfg: Mapping[str, Any]) -> list[str]:
    """`<doc>` and `<sep>`, then every token written like a symbol in the prefixes of the
    corpora of `data`, each once, in the order met."""
    needed = list(_WINDOW_SYMBOLS)
    data = cfg.get("data")
    corpora = data.values() if isinstance(data, Mapping) else ()
    for corpus in corpora:
        if not isinstance(corpus, Mapping):
            continue
        for key in _PREFIXES:
            prefix = corpus.get(key)
            tokens = prefix.split() if isinstance(prefix, str) else ()
            needed += [
                token
                for token in tokens
                if _ANGLE_TOKEN.fullmatch(token) and token not in needed
            ]
    return needed


def _vocabulary_problems(
    cfg: Mapping[str, Any], vocabulary: Mapping[str, frozenset[str]]
) -> list[str]:
    missing = [
        symbol
        for symbol in _needed_symbols(cfg)
        if any(symbol not in tokens for tokens in vocabulary.values())
    ]
    if not missing:
        return []
    problem = f"{VOCAB_FILE} of the source lacks {', '.join(missing)}"
    if any(symbol in _WINDOW_SYMBOLS for symbol in missing):
        return [
            (
                f"{problem}: the teacher predates the vocabulary invariant (a "
                "vocabulary file made since then holds every reserved symbol), so it "
                "would read them as <unk>"
            )
        ]
    return [
        f"{problem}: the teacher never saw these tags, so it would read them as <unk>"
    ]


def _verified_json(file: Path, digest: str, what: str) -> dict[str, Any]:
    """The JSON object in `file`, read once and only when its bytes are those with
    SHA-256 `digest`."""
    try:
        content = file.read_bytes()
    except OSError as error:
        raise DataError(f"cannot read {file}: {error.strerror}") from error
    if hashlib.sha256(content).hexdigest() != digest:
        raise ConfigError(
            f"{file} changed since it was verified against the manifest, so the "
            f"{what} that would be checked is not the one that was vouched for"
        )
    try:
        document = json.loads(content)
    except (ValueError, RecursionError) as error:
        raise DataError(f"{file}: is not valid JSON") from error
    if not isinstance(document, dict):
        raise DataError(f"{file}: is not a JSON object")
    return document
