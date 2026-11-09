"""The D9 STOP gate: a tiny document-level model is made by continuing the tiny teacher on
windows of 2 to 4 sentences joined with `<sep>`, and it decodes `<sep>`-aligned output.
Everything runs through the real `rinchen` commands with the real Eole on the CPU; nothing
here touches a GPU (the launcher hides every one from Eole and the configs train with
`gpu_ranks: []`).

The data are the toy corpora and documents of `tests/support/tiny_parallel.py`. The
document corpus handed to `rinchen train docify` holds the train and dev documents only;
the toy TEST documents are read by the chunker scenario alone, on purpose, and never for
training or for choosing anything. The base config is the checked-in
`configs/eole/teacher_doc_tiny.yaml`.

Module fixtures run the expensive commands once, in the order the scenarios need, and
take a snapshot of what the tests look at, so no test depends on the order the tests run
in:

    T  the sentence-level tiny teacher (200 steps), `average --last 2`
    D  `rinchen train docify` over the toy documents
    C  `launch --continue-from <the averaged teacher>` for the 2,500 steps of the config
    A  the dev windows (2, 3 and 4 sentences, every window of every dev document)
       translated with each checkpoint: the alignment rate per window size, the gate
    K  the chunker over the toy test documents, the doc model translating the windows and
       the averaged teacher the sentences it falls back to
    R  five continuations that must be refused (no manifest of the source: a step
       directory alone and a run without its manifest; a weights byte flipped; a
       vocabulary without `<doc>` whose file and manifest were rewritten to agree; a config
       that omits an architecture key), each exit 2 with Eole not started
    S  a continuation killed by SIGTERM after its first checkpoint, then launched again
    B  the finished continuation launched again

The toy sentences are templated: the alignment of the output is the gate, never a
translation score.
"""

from __future__ import annotations

import collections
import hashlib
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
import sentencepiece
import yaml

from rinchen.config.languages import direction_tag
from rinchen.config.loader import REPO_ROOT
from rinchen.infer.doc_chunker import (
    DOC_MARKER,
    SEGMENT_SEPARATOR,
    DocumentTranslation,
    chunk_document,
    join_window,
    one_line,
    split_output,
    translate_document,
)
from rinchen.manifest import store
from rinchen.manifest.model import Manifest
from rinchen.metrics.bleu_chrf import corpus_scores
from rinchen.spm.symbols import user_defined_symbols
from rinchen.train.ablation_layout import variant_paths
from rinchen.train.checkpoints import is_complete, step_directories
from rinchen.train.curves import TrainCurve, parse_train_log
from rinchen.train.eole_cfg import dump_config, read_config
from rinchen.train.eole_run import predict_tokens
from tests.support.step_dirs import flip_one_byte, tree_digest
from tests.support.tiny_parallel import (
    DocRow,
    ToyDocuments,
    build_toy_documents,
    write_corpus_parquet,
)
from tests.support.tiny_variant import ToyInputs, build_toy_inputs

pytestmark = pytest.mark.subprocess

# A hung Eole must fail the test: the longest command, the 2,500-step continuation, takes
# about two minutes.
SUBPROCESS_TIMEOUT_SECONDS = 900
# Handed to `rinchen train launch --timeout`: it bounds each Eole command inside.
ENGINE_TIMEOUT_SECONDS = 600
RINCHEN = str(Path(sys.executable).with_name("rinchen"))
TEACHER_BASE = REPO_ROOT / "configs" / "eole" / "teacher_tiny.yaml"
DOC_BASE = REPO_ROOT / "configs" / "eole" / "teacher_doc_tiny.yaml"
VARIANT_ID = "t0.5_bt0.5_src_s1234"
DEV_LINES = 100
THREADS = "2"

TEACHER_STEPS = 200
CONTINUATION_STEPS = 2500
SAVE_EVERY = 500
SAVED_STEPS = tuple(range(SAVE_EVERY, CONTINUATION_STEPS + 1, SAVE_EVERY))
REPORT_EVERY = 50
REPORT_STEPS = tuple(range(REPORT_EVERY, CONTINUATION_STEPS + 1, REPORT_EVERY))
WEIGHTS = "model.00.safetensors"
# The one limit of the whole gate; it is never lowered.
MIN_ALIGNMENT = 0.9
WINDOW_SIZES = (2, 3, 4)
# What `rinchen train docify` cuts windows to by default: the base config's 128 pieces,
# less the two boundary tokens Eole's length filter counts on a target.
MAX_PIECES = 126
# Eole stops an output at this many pieces: the length the doc model trained with. The
# ratio cap (twice the longest source of a batch plus 5) is off so that no output is cut
# before the model ends it, which would hide a lost `<sep>`.
GENERATION_LIMIT = 128
NO_RATIO_CAP = 0
# A launch that finds the run finished only reads it.
MAX_NOTHING_TO_DO_SECONDS = 30.0

# The interrupted continuation: 750 steps with a save every 250 steps.
INTERRUPTED_STEPS = 750
INTERRUPTED_SAVE_EVERY = 250
FIRST_CHECKPOINT_TIMEOUT_SECONDS = 300
SIGTERM_EXIT_CODE = 128 + signal.SIGTERM
POLL_SECONDS = 0.1


@dataclass(frozen=True)
class Invocation:
    """One `rinchen` command: what it printed and how long it took."""

    returncode: int
    stdout: list[str]
    stderr: str
    seconds: float

    @property
    def facts(self) -> dict[str, str]:
        """The `key: value` lines a command prints."""
        return dict(line.split(": ", 1) for line in self.stdout if ": " in line)


def _rinchen(*args: str) -> Invocation:
    started = time.perf_counter()
    completed = subprocess.run(
        [RINCHEN, *args],
        capture_output=True,
        timeout=SUBPROCESS_TIMEOUT_SECONDS,
        check=False,
    )
    return Invocation(
        returncode=completed.returncode,
        stdout=completed.stdout.decode("utf-8").splitlines(),
        stderr=completed.stderr.decode("utf-8", "replace"),
        seconds=time.perf_counter() - started,
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _ok(invocation: Invocation) -> Invocation:
    assert invocation.returncode == 0, invocation.stderr
    return invocation


def _launch_arguments(config: Path, run_dir: Path, source: Path | None) -> list[str]:
    arguments = [
        "train",
        "launch",
        "--config",
        str(config),
        "--run-dir",
        str(run_dir),
        "--threads",
        THREADS,
        "--timeout",
        str(ENGINE_TIMEOUT_SECONDS),
    ]
    if source is not None:
        arguments += ["--continue-from", str(source)]
    return arguments


def _launch(config: Path, run_dir: Path, source: Path | None = None) -> Invocation:
    return _rinchen(*_launch_arguments(config, run_dir, source))


def _config_copy(
    source: Path, destination: Path, run_dir: Path, **training: Any
) -> dict[str, Any]:
    """The rendered config at `source` for the run `run_dir`: `model_path` and `src_vocab`
    point into the run, `training` values are laid over the training section, and the
    file is written to `destination`. The corpora stay where they are; they are only
    read. Returns the mapping, which a caller may change and write again."""
    cfg = read_config(source, "Eole config")
    cfg["src_vocab"] = str(run_dir / "vocab.src")
    cfg["training"]["model_path"] = str(run_dir / "ckpt")
    cfg["training"].update(training)
    destination.parent.mkdir(parents=True, exist_ok=True)
    dump_config(cfg, destination)
    return dict(cfg)


def _attempt_logs(run_dir: Path) -> list[Path]:
    return sorted(
        (run_dir / "launch").glob("attempt-*.log"),
        key=lambda log: int(log.stem.split("-")[1]),
    )


def _training_section(run_dir: Path, attempt: int) -> dict[str, Any]:
    """`training` of the config the launcher rendered for attempt `attempt`."""
    rendered = read_config(run_dir / "launch" / f"train-{attempt}.yaml", "config")
    section = rendered["training"]
    assert isinstance(section, dict)
    return section


def _complete_steps(model_path: Path) -> list[int]:
    return [step.step for step in step_directories(model_path) if step.problem is None]


def _logs_loading_from(log: str, directory: Path) -> bool:
    """Whether Eole said it loaded the checkpoint metadata, the weights and the
    vocabulary, from `directory`."""
    return f"Loading metadata from {directory}\n" in log


# ---------------------------------------------------------------------------------
# The toy documents, windows and translation by the real Eole.


@dataclass(frozen=True)
class Window:
    """`size` consecutive sentences of a document, joined as the model reads them."""

    size: int
    tag: str
    text: str


def _documents(rows: Sequence[DocRow]) -> list[list[DocRow]]:
    """The rows of each (direction, document), in `seg_idx` order."""
    groups: dict[tuple[str, str, str], list[DocRow]] = collections.defaultdict(list)
    for row in rows:
        groups[(row["src_lang"], row["tgt_lang"], row["doc_id"])].append(row)
    return [sorted(group, key=lambda row: row["seg_idx"]) for group in groups.values()]


def _all_windows(rows: Sequence[DocRow]) -> list[Window]:
    """Every window of 2, 3 and 4 consecutive sentences of every document: the dev set of
    the alignment gate (the windows `docify` trains on do not overlap; these do, so a
    sentence is read in several positions)."""
    windows: list[Window] = []
    for document in _documents(rows):
        tag = direction_tag(document[0]["tgt_lang"])
        for size in WINDOW_SIZES:
            for start in range(len(document) - size + 1):
                sentences = [row["src"] for row in document[start : start + size]]
                windows.append(Window(size, tag, join_window(sentences)))
    return windows


def _window_tokens(
    processor: sentencepiece.SentencePieceProcessor, tag: str, text: str
) -> str:
    """The pre-tokenised source line of a window: the tag, `<doc>` and the pieces, as
    the `valid` corpus of the continuation has it."""
    return " ".join([tag, DOC_MARKER, *processor.encode(text, out_type=str)])


def _sentence_tokens(
    processor: sentencepiece.SentencePieceProcessor, tag: str, text: str
) -> str:
    return " ".join([tag, *processor.encode(text, out_type=str)])


def _translate(
    ckpt_dir: Path,
    lines: Sequence[str],
    out_dir: Path,
    processor: sentencepiece.SentencePieceProcessor,
) -> list[str]:
    """The texts `ckpt_dir` writes for the token lines `lines`, decoded."""
    tokens = predict_tokens(
        ckpt_dir,
        lines,
        out_dir=out_dir,
        threads=int(THREADS),
        timeout=SUBPROCESS_TIMEOUT_SECONDS,
        stem="gate",
        max_length=GENERATION_LIMIT,
        max_length_ratio=NO_RATIO_CAP,
    )
    return [processor.decode(line.split(" ")) for line in tokens]


# ---------------------------------------------------------------------------------
# T, D: the teacher and the doc corpora.


@dataclass(frozen=True)
class Workspace:
    root: Path
    inputs: ToyInputs
    documents: ToyDocuments
    doc_corpus: Path
    processor: sentencepiece.SentencePieceProcessor


@pytest.fixture(scope="module")
def workspace(tmp_path_factory: pytest.TempPathFactory) -> Workspace:
    root = tmp_path_factory.mktemp("train_doc")
    inputs = build_toy_inputs(root / "in")
    documents = build_toy_documents()
    # Train and dev documents only: the test documents are not in any file a training
    # command reads.
    doc_corpus = write_corpus_parquet(
        [*documents.train, *documents.dev], root / "in" / "docs.parquet"
    )
    return Workspace(
        root=root,
        inputs=inputs,
        documents=documents,
        doc_corpus=doc_corpus,
        processor=sentencepiece.SentencePieceProcessor(
            model_file=str(inputs.spm_model)
        ),
    )


@dataclass(frozen=True)
class Teacher:
    """The sentence-level tiny teacher after 200 steps, and its two-checkpoint average."""

    run_dir: Path
    step: Path
    averaged: Path
    vocab_src: Path
    launch: Invocation
    average: Invocation


@pytest.fixture(scope="module")
def teacher(workspace: Workspace) -> Teacher:
    matrix = workspace.root / "matrix.yaml"
    matrix.write_text(
        yaml.safe_dump(
            {
                "name": "teacher_tiny_doc_gate",
                "base": str(TEACHER_BASE),
                "axes": {
                    "tag_placement": ["src"],
                    "temperature": [0.5],
                    "bt_ratio": [0.5],
                },
                "seeds": [1234],
                "dev_max_lines": DEV_LINES,
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    out_dir = workspace.root / "teacher_out"
    _ok(
        _rinchen(
            "train",
            "ablate",
            "--matrix",
            str(matrix),
            "--corpus",
            str(workspace.inputs.corpus),
            "--spm-model",
            str(workspace.inputs.spm_model),
            "--out-dir",
            str(out_dir),
            "--prepare-only",
            "--threads",
            THREADS,
        )
    )
    paths = variant_paths(out_dir, VARIANT_ID)
    launch = _ok(_launch(paths.config, paths.run_dir))
    average = _ok(
        _rinchen(
            "train",
            "average",
            "--checkpoint-dir",
            str(paths.run_dir / "ckpt"),
            "--last",
            "2",
        )
    )
    return Teacher(
        run_dir=paths.run_dir,
        step=paths.run_dir / "ckpt" / f"step_{TEACHER_STEPS}",
        averaged=paths.run_dir / "averaged",
        vocab_src=paths.run_dir / "vocab.src",
        launch=launch,
        average=average,
    )


@dataclass(frozen=True)
class Docified:
    """`rinchen train docify` over the toy train and dev documents."""

    invocation: Invocation
    out_dir: Path
    config: Path
    run_dir: Path
    manifest: Manifest | None


@pytest.fixture(scope="module")
def docified(workspace: Workspace) -> Docified:
    out_dir = workspace.root / "doc"
    invocation = _ok(
        _rinchen(
            "train",
            "docify",
            "--corpus",
            str(workspace.doc_corpus),
            "--spm-model",
            str(workspace.inputs.spm_model),
            "--config",
            str(DOC_BASE),
            "--out-dir",
            str(out_dir),
        )
    )
    return Docified(
        invocation=invocation,
        out_dir=out_dir,
        config=out_dir / "config.yaml",
        run_dir=out_dir / "run",
        manifest=store.read(out_dir),
    )


# ---------------------------------------------------------------------------------
# C: the continuation.


@dataclass(frozen=True)
class Continuation:
    """The 2,500-step continuation of the averaged teacher, launched from nothing."""

    invocation: Invocation
    run_dir: Path
    log: str
    curve: TrainCurve
    launch_record: dict[str, Any]
    manifest: Manifest | None
    teacher_digest_before: str
    teacher_digest_after: str
    steps: dict[int, str | None]

    @property
    def model_path(self) -> Path:
        return self.run_dir / "ckpt"

    def step(self, step: int) -> Path:
        return self.model_path / f"step_{step}"


@pytest.fixture(scope="module")
def continuation(teacher: Teacher, docified: Docified) -> Continuation:
    before = tree_digest(teacher.run_dir)
    invocation = _ok(_launch(docified.config, docified.run_dir, teacher.averaged))
    run_dir = docified.run_dir
    log = _attempt_logs(run_dir)[-1].read_text(encoding="utf-8", errors="replace")
    return Continuation(
        invocation=invocation,
        run_dir=run_dir,
        log=log,
        curve=parse_train_log(log),
        launch_record=json.loads((run_dir / "launch.json").read_text("utf-8")),
        manifest=store.read(run_dir),
        teacher_digest_before=before,
        teacher_digest_after=tree_digest(teacher.run_dir),
        steps={
            step.step: is_complete(step.path)
            for step in step_directories(run_dir / "ckpt")
        },
    )


# ---------------------------------------------------------------------------------
# A: the alignment of the dev windows.


@dataclass(frozen=True)
class Alignment:
    """How many windows of each size came back with one segment per sentence."""

    aligned: dict[int, int]
    total: dict[int, int]
    seconds: float

    def rate(self, size: int) -> float:
        return self.aligned[size] / self.total[size]

    def rates(self) -> dict[int, float]:
        return {size: round(self.rate(size), 4) for size in WINDOW_SIZES}


def _measure_alignment(
    ckpt_dir: Path,
    windows: Sequence[Window],
    out_dir: Path,
    processor: sentencepiece.SentencePieceProcessor,
) -> Alignment:
    started = time.perf_counter()
    decoded = _translate(
        ckpt_dir,
        [_window_tokens(processor, window.tag, window.text) for window in windows],
        out_dir,
        processor,
    )
    seconds = time.perf_counter() - started
    aligned: dict[int, int] = dict.fromkeys(WINDOW_SIZES, 0)
    total: dict[int, int] = dict.fromkeys(WINDOW_SIZES, 0)
    for window, text in zip(windows, decoded, strict=True):
        total[window.size] += 1
        aligned[window.size] += _is_aligned(text, window.size)
    return Alignment(aligned, total, seconds)


def _is_aligned(output: str, size: int) -> bool:
    """The chunker's own definition: one segment per sentence, none of them empty (a
    sentence that has text and no translation is a sentence lost)."""
    segments = split_output(output, size)
    return segments is not None and all(segments)


@dataclass(frozen=True)
class AlignmentReport:
    windows: int
    by_step: dict[int, Alignment]


@pytest.fixture(scope="module")
def alignment(workspace: Workspace, continuation: Continuation) -> AlignmentReport:
    windows = _all_windows(workspace.documents.dev)
    return AlignmentReport(
        windows=len(windows),
        by_step={
            step: _measure_alignment(
                continuation.step(step),
                windows,
                workspace.root / "alignment" / f"step_{step}",
                workspace.processor,
            )
            for step in SAVED_STEPS
        },
    )


# ---------------------------------------------------------------------------------
# K: the chunker over the toy test documents.


@dataclass(frozen=True)
class Case:
    """One document in one direction: its sentences and the reference translations."""

    tag: str
    sentences: tuple[str, ...]
    references: tuple[str, ...]


@dataclass(frozen=True)
class Chunked:
    cases: list[Case]
    translations: list[DocumentTranslation]
    chunker_chrf: float
    sentence_level_chrf: float
    seconds: float


def _cases(rows: Sequence[DocRow]) -> list[Case]:
    return [
        Case(
            tag=direction_tag(document[0]["tgt_lang"]),
            sentences=tuple(row["src"] for row in document),
            references=tuple(row["tgt"] for row in document),
        )
        for document in _documents(rows)
    ]


@pytest.fixture(scope="module")
def chunked(
    workspace: Workspace, teacher: Teacher, continuation: Continuation
) -> Chunked:
    """Every test document in both directions through `translate_document`, the doc
    model translating the windows and the averaged teacher the sentences it falls back
    to, both by `predict_tokens`. A call of Eole costs seconds, so the texts the chunker
    will ask for are collected first (`chunk_document` is deterministic), translated in
    one call per model, and handed back by lookup: a text nobody asked for would be a
    KeyError, which the chunker never catches."""
    started = time.perf_counter()
    processor = workspace.processor
    cases = _cases(workspace.documents.test)

    def count_pieces(text: str) -> int:
        return len(processor.encode(text))

    windows: dict[tuple[str, str], None] = {}
    sentences: dict[tuple[str, str], None] = {}
    for case in cases:
        for chunk in chunk_document(
            case.sentences, count_pieces=count_pieces, max_pieces=MAX_PIECES
        ):
            if chunk.kind == "window":
                text = join_window([case.sentences[at] for at in chunk.positions])
                windows[(case.tag, text)] = None
        for sentence in case.sentences:
            sentences[(case.tag, one_line(sentence))] = None

    window_keys = list(windows)
    window_output = dict(
        zip(
            window_keys,
            _translate(
                continuation.step(CONTINUATION_STEPS),
                [_window_tokens(processor, tag, text) for tag, text in window_keys],
                workspace.root / "chunker" / "windows",
                processor,
            ),
            strict=True,
        )
    )
    sentence_keys = list(sentences)
    sentence_output = dict(
        zip(
            sentence_keys,
            _translate(
                teacher.averaged,
                [_sentence_tokens(processor, tag, text) for tag, text in sentence_keys],
                workspace.root / "chunker" / "sentences",
                processor,
            ),
            strict=True,
        )
    )

    def looked_up(
        outputs: Mapping[tuple[str, str], str], tag: str
    ) -> Callable[[str], str]:
        return lambda text: outputs[(tag, text)]

    translations = [
        translate_document(
            case.sentences,
            translate_window=looked_up(window_output, case.tag),
            translate_sentence=looked_up(sentence_output, case.tag),
            count_pieces=count_pieces,
            max_pieces=MAX_PIECES,
        )
        for case in cases
    ]
    references = [text for case in cases for text in case.references]
    return Chunked(
        cases=cases,
        translations=translations,
        chunker_chrf=corpus_scores(
            [text for done in translations for text in done.translations], references
        ).chrf,
        sentence_level_chrf=corpus_scores(
            [
                sentence_output[(case.tag, one_line(sentence))]
                for case in cases
                for sentence in case.sentences
            ],
            references,
        ).chrf,
        seconds=time.perf_counter() - started,
    )


# ---------------------------------------------------------------------------------
# R: continuations that must be refused.


@dataclass(frozen=True)
class Refusal:
    """A `launch --continue-from` that must be refused: what it printed, what it left in
    the run directory and whether the source changed."""

    invocation: Invocation
    run_files: list[str]
    source_before: str
    source_after: str


def _drop_a_model_key(cfg: dict[str, Any]) -> None:
    del cfg["model"]["heads"]


def _copy_of(source: Path, workspace: Workspace, name: str) -> Path:
    destination = workspace.root / "sources" / name / source.name
    shutil.copytree(source, destination)
    return destination


def _vocabulary_without_doc(averaged: Path) -> None:
    """Take `<doc>` out of the vocabulary of the averaged model at `averaged` and keep
    the manifest in step with the file, so that only the vocabulary check can refuse it
    (the weights and every other file still match what the manifest says)."""
    vocab = averaged / "vocab.json"
    document = json.loads(vocab.read_text("utf-8"))
    for side in ("src", "tgt"):
        document[side] = [token for token in document[side] if token != DOC_MARKER]
    vocab.write_text(json.dumps(document), encoding="utf-8")
    manifest = store.read(averaged)
    assert manifest is not None
    for item in manifest.outputs:
        if Path(item.path).name == "vocab.json":
            item.sha256 = _sha256(vocab)
            item.bytes = vocab.stat().st_size
    store.write(averaged, manifest)


def _refused_launch(
    workspace: Workspace,
    docified: Docified,
    name: str,
    source: Path,
    edit: Callable[[dict[str, Any]], None] | None = None,
) -> Refusal:
    """`launch --continue-from source` into a new run directory with the config of
    `docified` (after `edit` changed it, when given)."""
    root = workspace.root / "refusals" / name
    run_dir = root / "run"
    config = root / "config.yaml"
    cfg = _config_copy(docified.config, config, run_dir)
    if edit is not None:
        edit(cfg)
        dump_config(cfg, config)
    before = tree_digest(source)
    invocation = _launch(config, run_dir, source)
    return Refusal(
        invocation=invocation,
        run_files=sorted(str(path.relative_to(run_dir)) for path in run_dir.rglob("*")),
        source_before=before,
        source_after=tree_digest(source),
    )


REFUSAL_CASES = {
    "a_step_directory_alone": ("launch.json", "within 3 levels"),
    "a_run_without_its_manifest": ("no manifest vouches", "manifest.json"),
    "a_weights_byte_flipped": ("differs from the manifest", WEIGHTS),
    "a_vocabulary_without_doc": (DOC_MARKER, "vocabulary invariant"),
    "a_config_without_an_architecture_key": ("model.heads", "not stated"),
}


@pytest.fixture(scope="module")
def refusals(
    workspace: Workspace, teacher: Teacher, docified: Docified
) -> dict[str, Refusal]:
    def launch(
        name: str, source: Path, edit: Callable[[dict[str, Any]], None] | None = None
    ) -> Refusal:
        return _refused_launch(workspace, docified, name, source, edit)

    step_alone = _copy_of(teacher.step, workspace, "a_step_directory_alone")

    run_copy = workspace.root / "sources" / "a_run_without_its_manifest" / "run"
    shutil.copytree(teacher.run_dir, run_copy)
    (run_copy / "manifest.json").unlink()

    flipped = _copy_of(teacher.averaged, workspace, "a_weights_byte_flipped")
    flip_one_byte(flipped / WEIGHTS)

    no_doc = _copy_of(teacher.averaged, workspace, "a_vocabulary_without_doc")
    _vocabulary_without_doc(no_doc)

    return {
        "a_step_directory_alone": launch("a_step_directory_alone", step_alone),
        "a_run_without_its_manifest": launch(
            "a_run_without_its_manifest", run_copy / "ckpt" / f"step_{TEACHER_STEPS}"
        ),
        "a_weights_byte_flipped": launch("a_weights_byte_flipped", flipped),
        "a_vocabulary_without_doc": launch("a_vocabulary_without_doc", no_doc),
        "a_config_without_an_architecture_key": launch(
            "a_config_without_an_architecture_key",
            teacher.averaged,
            _drop_a_model_key,
        ),
    }


# ---------------------------------------------------------------------------------
# S: a continuation killed by SIGTERM and launched again.


@dataclass(frozen=True)
class Interrupted:
    run_dir: Path
    averaged: Path
    returncode: int
    complete_at_the_kill: list[int]
    resumed: Invocation
    first_log: str
    second_log: str
    second_curve: TrainCurve
    first_training: dict[str, Any]
    second_training: dict[str, Any]
    manifest: Manifest | None
    source_digest_before: str
    source_digest_after: str

    @property
    def newest(self) -> int:
        return max(self.complete_at_the_kill)


def _wait_for_the_first_checkpoint(
    process: subprocess.Popen[bytes], checkpoint: Path, output: Path
) -> None:
    """Return once `checkpoint` is complete while the launcher still runs. The wait is
    the launcher's own: `Popen.wait` with a timeout returns as soon as it ends."""
    deadline = time.monotonic() + FIRST_CHECKPOINT_TIMEOUT_SECONDS
    while not (checkpoint.is_dir() and is_complete(checkpoint) is None):
        remaining = deadline - time.monotonic()
        assert remaining > 0, f"no checkpoint in {FIRST_CHECKPOINT_TIMEOUT_SECONDS} s"
        try:
            process.wait(timeout=min(POLL_SECONDS, remaining))
        except subprocess.TimeoutExpired:
            continue
        pytest.fail(
            "the launch ended before its first checkpoint: "
            + output.read_text("utf-8", "replace")
        )


def _stop(process: subprocess.Popen[bytes]) -> None:
    """Leave nothing running: SIGTERM the launcher (it kills Eole's group) and, when it
    does not end, SIGKILL its group."""
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()


@pytest.fixture(scope="module")
def interrupted(
    workspace: Workspace, teacher: Teacher, docified: Docified
) -> Interrupted:
    root = workspace.root / "interrupted"
    run_dir = root / "run"
    config = root / "config.yaml"
    _config_copy(
        docified.config,
        config,
        run_dir,
        train_steps=INTERRUPTED_STEPS,
        valid_steps=INTERRUPTED_SAVE_EVERY,
        save_checkpoint_steps=INTERRUPTED_SAVE_EVERY,
    )
    arguments = _launch_arguments(config, run_dir, teacher.averaged)
    digest_before = tree_digest(teacher.run_dir)
    first_checkpoint = run_dir / "ckpt" / f"step_{INTERRUPTED_SAVE_EVERY}"
    root.mkdir(parents=True, exist_ok=True)
    with (root / "launch.out").open("wb") as sink:
        process = subprocess.Popen(
            [RINCHEN, *arguments],
            stdout=sink,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
        )
    try:
        _wait_for_the_first_checkpoint(process, first_checkpoint, root / "launch.out")
        process.send_signal(signal.SIGTERM)
        returncode = process.wait(timeout=60)
    except BaseException:
        _stop(process)
        raise
    complete = _complete_steps(run_dir / "ckpt")
    resumed = _ok(_rinchen(*arguments))
    logs = _attempt_logs(run_dir)
    assert len(logs) == 2, logs
    second_log = logs[1].read_text(encoding="utf-8", errors="replace")
    return Interrupted(
        run_dir=run_dir,
        averaged=teacher.averaged,
        returncode=returncode,
        complete_at_the_kill=complete,
        resumed=resumed,
        first_log=logs[0].read_text(encoding="utf-8", errors="replace"),
        second_log=second_log,
        second_curve=parse_train_log(second_log),
        first_training=_training_section(run_dir, 1),
        second_training=_training_section(run_dir, 2),
        manifest=store.read(run_dir),
        source_digest_before=digest_before,
        source_digest_after=tree_digest(teacher.run_dir),
    )


# ---------------------------------------------------------------------------------
# B: the finished continuation launched again.


@dataclass(frozen=True)
class Rerun:
    invocation: Invocation
    digest_before: str
    digest_after: str


@pytest.fixture(scope="module")
def rerun(teacher: Teacher, docified: Docified, continuation: Continuation) -> Rerun:
    before = tree_digest(continuation.run_dir)
    invocation = _ok(_launch(docified.config, docified.run_dir, teacher.averaged))
    return Rerun(invocation, before, tree_digest(continuation.run_dir))


# =================================================================================
# T: the vocabulary invariant.


def test_the_teachers_vocabulary_holds_every_reserved_symbol_including_doc_and_sep(
    teacher: Teacher,
) -> None:
    symbols = set(user_defined_symbols())
    assert {DOC_MARKER, SEGMENT_SEPARATOR} <= symbols

    listed = {
        line.split("\t")[0]
        for line in teacher.vocab_src.read_text("utf-8").splitlines()
        if line
    }
    assert symbols <= listed
    for directory in (teacher.step, teacher.averaged):
        saved = json.loads((directory / "vocab.json").read_text("utf-8"))
        assert symbols <= set(saved["src"]), directory
        assert symbols <= set(saved["tgt"]), directory


def test_the_averaged_teacher_is_vouched_for_by_its_manifest(teacher: Teacher) -> None:
    manifest = store.read(teacher.averaged)

    assert manifest is not None
    assert manifest.stage == "train_average"
    assert {Path(item.path).name for item in manifest.outputs} >= {
        WEIGHTS,
        "config.json",
        "vocab.json",
        "averaged_from.json",
    }


# D: docify.


def test_docify_cuts_windows_of_every_size_and_writes_the_continuation_files(
    workspace: Workspace, docified: Docified, record_property: Any
) -> None:
    record_property("seconds", round(docified.invocation.seconds, 1))
    facts = docified.invocation.facts
    manifest = docified.manifest
    assert manifest is not None
    record_property(
        "windows_by_size",
        {
            size: manifest.metrics[f"count.windows_by_size.{size}"]
            for size in WINDOW_SIZES
        },
    )

    assert facts["max pieces"] == str(MAX_PIECES)
    assert facts["config"] == str(docified.config)
    assert manifest.stage == "train_docify"
    for size in WINDOW_SIZES:
        assert manifest.metrics[f"count.windows_by_size.{size}"] > 0
    # No row of the test split was in the file, so none could be read into a window.
    test_sentences = {
        text for row in workspace.documents.test for text in (row["src"], row["tgt"])
    }
    windowed = {
        sentence
        for path in (docified.out_dir / "data" / "doc").iterdir()
        if path.suffix in {".src", ".tgt"}
        for line in path.read_text("utf-8").splitlines()
        for sentence in line.split(f" {SEGMENT_SEPARATOR} ")
    }
    assert windowed
    assert not windowed & test_sentences
    valid = (docified.out_dir / "valid.src").read_text("utf-8").splitlines()
    assert valid
    assert all(
        line.split(" ")[1] == DOC_MARKER and line.startswith("<2") for line in valid
    )
    cfg = read_config(docified.config, "config")
    assert "train_from" not in cfg["training"]
    assert "reset_optim" not in cfg["training"]
    assert cfg["training"]["model_path"] == str(docified.run_dir / "ckpt")


# C: the continuation.


def test_the_continuation_completes_the_steps_of_the_config_from_the_averaged_teacher(
    teacher: Teacher, continuation: Continuation, record_property: Any
) -> None:
    record_property("seconds", round(continuation.invocation.seconds, 1))
    facts = continuation.invocation.facts

    assert facts["continued from"] == f"averaged {teacher.averaged}"
    assert facts["status"] == "completed"
    assert facts["steps"] == str(CONTINUATION_STEPS)
    assert facts["resumed from"] == "none"
    assert facts["attempt"] == "1"
    assert facts["checkpoint"] == str(continuation.step(CONTINUATION_STEPS))
    assert continuation.steps == dict.fromkeys(SAVED_STEPS)


def test_the_first_attempt_starts_from_the_teachers_weights_with_a_fresh_optimizer(
    teacher: Teacher, docified: Docified, continuation: Continuation
) -> None:
    training = _training_section(continuation.run_dir, 1)

    assert training["train_from"] == str(teacher.averaged)
    assert training["reset_optim"] == "all"
    assert _logs_loading_from(continuation.log, teacher.averaged)
    # The vocabulary is the checkpoint's: Eole ignores the new vocabulary file.
    assert "Keeping metadata vocabulary" in continuation.log
    base = read_config(DOC_BASE, "base config")["training"]
    assert "train_from" not in base
    assert "reset_optim" not in base
    # Steps count from 1 again and every report is of this run.
    curve = continuation.curve
    assert tuple(report.step for report in curve.reports) == REPORT_STEPS
    assert {report.total for report in curve.reports} == {CONTINUATION_STEPS}
    assert (curve.runs, curve.unparsed_reports) == (1, 0)
    assert curve.saved_steps == SAVED_STEPS


def test_the_loss_falls_without_a_nan_while_the_model_learns_the_windows(
    continuation: Continuation, record_property: Any
) -> None:
    curve = continuation.curve
    first, last = curve.reports[0], curve.reports[-1]
    record_property("xent_first", first.xent)
    record_property("xent_last", last.xent)
    record_property("validation_perplexities", list(curve.validations))

    assert curve.non_finite_step is None
    assert last.xent < first.xent
    assert curve.validations[-1] < curve.validations[0]
    assert len(curve.validations) == len(SAVED_STEPS)
    assert not curve.early_stopped
    assert "Traceback" not in continuation.log


def test_the_teacher_is_not_written_by_the_continuation(
    continuation: Continuation,
) -> None:
    assert continuation.teacher_digest_after == continuation.teacher_digest_before


def test_the_continuation_keeps_the_vocabulary_of_the_teacher_with_doc_and_sep(
    teacher: Teacher, continuation: Continuation
) -> None:
    saved = json.loads(
        (continuation.step(CONTINUATION_STEPS) / "vocab.json").read_text("utf-8")
    )
    source = json.loads((teacher.averaged / "vocab.json").read_text("utf-8"))

    assert saved == source
    assert {DOC_MARKER, SEGMENT_SEPARATOR} <= set(saved["src"])


def test_the_launch_record_and_the_manifest_name_the_source(
    teacher: Teacher, continuation: Continuation
) -> None:
    record = continuation.launch_record
    source = record["continued_from"]
    weights = teacher.averaged / WEIGHTS

    assert record["schema_version"] == 2
    assert source["kind"] == "averaged"
    assert source["path"] == str(teacher.averaged)
    assert source["weights_sha256"] == _sha256(weights)
    assert source["manifest_stage"] == "train_average"
    manifest = continuation.manifest
    assert manifest is not None
    assert manifest.stage == "train_launch"
    assert manifest.status == {"launch": "ok"}
    assert str(weights) in {item.path for item in manifest.inputs}
    assert manifest.metrics["continued_from.kind"] == 1.0
    assert manifest.metrics["steps"] == float(CONTINUATION_STEPS)
    for item in [*manifest.inputs, *manifest.outputs]:
        assert item.sha256 == _sha256(Path(item.path)), item.path


@pytest.mark.parametrize(
    ("output", "size", "aligned"),
    [
        ("a <sep> b", 2, True),
        ("a <sep> b <sep> c", 3, True),
        ("a <sep> b <sep> c", 2, False),
        ("a", 2, False),
        ("a <sep>", 2, False),
        ("<sep> b", 2, False),
        ("a <sep> <sep> c", 3, False),
        ("", 1, False),
        ("a", 1, True),
    ],
)
def test_a_window_is_aligned_with_one_non_empty_segment_per_sentence(
    output: str, size: int, aligned: bool
) -> None:
    assert _is_aligned(output, size) is aligned


# A: the gate.


def test_the_doc_model_decodes_sep_aligned_output_for_every_window_size(
    alignment: AlignmentReport, record_property: Any
) -> None:
    final = alignment.by_step[CONTINUATION_STEPS]
    record_property("dev_windows", alignment.windows)
    record_property("windows_by_size", final.total)
    record_property("alignment_by_size", final.rates())
    record_property("predict_seconds", round(final.seconds, 1))
    record_property(
        "alignment_by_step",
        {step: report.rates() for step, report in alignment.by_step.items()},
    )

    assert set(final.total) == set(WINDOW_SIZES)
    for size in WINDOW_SIZES:
        assert final.total[size] >= 100, size
        assert final.rate(size) >= MIN_ALIGNMENT, (size, final.rates())


# K: the chunker.


def test_the_chunker_returns_one_translation_per_sentence_for_every_test_document(
    chunked: Chunked, record_property: Any
) -> None:
    stats = [done.stats for done in chunked.translations]
    windows = sum(item.windows for item in stats)
    aligned = sum(item.aligned for item in stats)
    record_property("documents", len(chunked.cases))
    record_property("sentences", sum(item.sentences for item in stats))
    record_property("windows", windows)
    record_property("aligned", aligned)
    record_property("fallbacks", windows - aligned)
    record_property("singles", sum(item.singles for item in stats))
    record_property("oversize", sum(item.oversize for item in stats))
    record_property("chunker_chrf", round(chunked.chunker_chrf, 2))
    record_property("sentence_level_chrf", round(chunked.sentence_level_chrf, 2))
    record_property("seconds", round(chunked.seconds, 1))

    assert len(chunked.cases) == len(chunked.translations) > 0
    for case, done in zip(chunked.cases, chunked.translations, strict=True):
        assert len(done.translations) == len(case.sentences)
        assert done.stats.sentences == len(case.sentences)
        assert done.stats.aligned + len(done.stats.fallbacks) == done.stats.windows
        assert done.stats.empty == 0
        assert all(text.strip() for text in done.translations)
    # The window model took part: the gate is not passed by the fallback alone.
    assert windows > 0
    assert aligned > 0
    # No score threshold, on purpose: the chunker's chrF is recorded, not gated.
    assert 0 <= chunked.chunker_chrf <= 100


# R: the refusals.


@pytest.mark.parametrize("case", list(REFUSAL_CASES))
def test_a_continuation_that_provenance_or_the_config_forbids_is_refused_before_eole(
    refusals: dict[str, Refusal], case: str
) -> None:
    refusal = refusals[case]

    assert refusal.invocation.returncode == 2, refusal.invocation.stderr
    for words in REFUSAL_CASES[case]:
        assert words in refusal.invocation.stderr, (words, refusal.invocation.stderr)
    # Eole was not started: no log, no checkpoint, no vocabulary, no launch record
    # (the lock file of the run is all a refused launch leaves).
    assert set(refusal.run_files) <= {"launch", "launch/lock"}, refusal.run_files
    # The source was only read.
    assert refusal.source_after == refusal.source_before


# S: SIGTERM, then resume.


def test_sigterm_ends_the_launcher_after_the_first_checkpoint(
    interrupted: Interrupted, record_property: Any
) -> None:
    record_property("complete_steps_at_the_kill", interrupted.complete_at_the_kill)

    assert interrupted.returncode == SIGTERM_EXIT_CODE
    assert interrupted.complete_at_the_kill
    assert interrupted.newest >= INTERRUPTED_SAVE_EVERY
    assert interrupted.newest < INTERRUPTED_STEPS


def test_the_launch_after_the_sigterm_resumes_the_continuation_from_its_own_step(
    interrupted: Interrupted,
) -> None:
    newest = interrupted.newest
    facts = interrupted.resumed.facts
    own_step = interrupted.run_dir / "ckpt" / f"step_{newest}"

    assert facts["status"] == "completed"
    assert facts["steps"] == str(INTERRUPTED_STEPS)
    assert facts["resumed from"] == str(newest)
    assert facts["attempt"] == "2"
    # The first attempt: from the teacher with a fresh optimizer. The second: from the
    # run's own checkpoint, and `reset_optim` says `none`, not nothing: Eole fills a key
    # the config omits from the checkpoint, which records `all`, and the run would start
    # again from step 1.
    assert interrupted.first_training["train_from"] == str(interrupted.averaged)
    assert interrupted.first_training["reset_optim"] == "all"
    assert interrupted.second_training["train_from"] == str(own_step)
    assert interrupted.second_training["reset_optim"] == "none"
    assert _logs_loading_from(interrupted.first_log, interrupted.averaged)
    assert _logs_loading_from(interrupted.second_log, own_step)


def test_the_resumed_attempt_continues_at_the_step_after_its_checkpoint_not_at_step_1(
    interrupted: Interrupted,
) -> None:
    newest = interrupted.newest
    curve = interrupted.second_curve

    assert tuple(report.step for report in curve.reports) == tuple(
        range(newest + REPORT_EVERY, INTERRUPTED_STEPS + 1, REPORT_EVERY)
    )
    assert {report.total for report in curve.reports} == {INTERRUPTED_STEPS}
    assert (curve.runs, curve.unparsed_reports) == (1, 0)
    assert curve.non_finite_step is None
    assert curve.saved_steps[-1] == INTERRUPTED_STEPS
    manifest = interrupted.manifest
    assert manifest is not None
    assert manifest.metrics["resumed_from"] == float(newest)
    assert manifest.metrics["attempt"] == 2.0
    assert manifest.metrics["continued_from.kind"] == 1.0
    assert interrupted.source_digest_after == interrupted.source_digest_before


# B: already complete.


def test_a_second_launch_of_the_finished_continuation_finds_it_complete_and_touches_nothing(
    teacher: Teacher, rerun: Rerun, record_property: Any
) -> None:
    record_property("seconds", round(rerun.invocation.seconds, 1))
    facts = rerun.invocation.facts

    assert facts["continued from"] == f"averaged {teacher.averaged}"
    assert facts["status"] == "already_complete"
    assert facts["steps"] == str(CONTINUATION_STEPS)
    assert facts["attempt"] == "0"
    assert facts["log"] == "none"
    assert rerun.invocation.seconds < MAX_NOTHING_TO_DO_SECONDS
    assert rerun.digest_after == rerun.digest_before
