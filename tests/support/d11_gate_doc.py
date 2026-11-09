"""The document stage of the D11 gates (`tests/integration/test_engine_surfaces.py`): a document
model made the way the D9 gate makes one (`rinchen train docify` over the toy train and dev
documents, then a continuation of the 2,500 steps of `teacher_doc_tiny.yaml` from the averaged
sentence-level teacher), exported at float32 through the real `rinchen export convert`.

The toy TEST documents are read here, on purpose, by the cases the engine translates, and by
nothing that trains or chooses: the document corpus handed to `docify` holds the train and dev
documents only. Nothing here touches a GPU (the session fixture hides every device, the
launcher hides them from Eole and the configs train with `gpu_ranks: []`).
"""

from __future__ import annotations

import collections
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import sentencepiece

from rinchen.config.languages import direction_tag
from rinchen.config.loader import REPO_ROOT
from rinchen.infer.doc_chunker import DOC_MARKER, join_window
from tests.support.d11_gate import convert_export
from tests.support.distill_gate import (
    ENGINE_TIMEOUT_SECONDS,
    THREADS,
    Invocation,
    run_rinchen,
    succeeded,
)
from tests.support.distill_gate_stages import Exported, Teacher, Workspace
from tests.support.tiny_parallel import (
    DocRow,
    ToyDocuments,
    build_toy_documents,
    write_corpus_parquet,
)

DOC_BASE: Final = REPO_ROOT / "configs" / "eole" / "teacher_doc_tiny.yaml"
# The steps of `teacher_doc_tiny.yaml`: the continuation's last save.
CONTINUATION_STEPS: Final = 2500
# The windows the chunker cuts hold 2 to 4 sentences.
WINDOW_SIZES: Final = (2, 3, 4)


@dataclass(frozen=True)
class DocumentModel:
    """The document model: the commands that made it and its float32 export."""

    docify: Invocation
    continuation: Invocation
    export: Exported
    documents: ToyDocuments
    seconds: float


def build_document_model(workspace: Workspace, teacher: Teacher) -> DocumentModel:
    """`docify` over the train and dev documents, `launch --continue-from` the averaged
    `teacher`, and the export of the last step (the placement is `src`, as the teacher's)."""
    started = time.perf_counter()
    documents = build_toy_documents()
    corpus = write_corpus_parquet(
        [*documents.train, *documents.dev], workspace.root / "doc_in" / "docs.parquet"
    )
    out_dir = workspace.root / "doc"
    docify = succeeded(
        run_rinchen(
            "train",
            "docify",
            "--corpus",
            corpus,
            "--spm-model",
            workspace.inputs.spm_model,
            "--config",
            DOC_BASE,
            "--out-dir",
            out_dir,
        )
    )
    continuation = succeeded(
        run_rinchen(
            "train",
            "launch",
            "--config",
            out_dir / "config.yaml",
            "--run-dir",
            out_dir / "run",
            "--continue-from",
            teacher.averaged,
            "--threads",
            THREADS,
            "--timeout",
            str(ENGINE_TIMEOUT_SECONDS),
        )
    )
    final_step = out_dir / "run" / "ckpt" / f"step_{CONTINUATION_STEPS}"
    exported = convert_export(
        workspace, final_step, workspace.root / "doc_export", placement="src"
    )
    return DocumentModel(
        docify=docify,
        continuation=continuation,
        export=exported,
        documents=documents,
        seconds=time.perf_counter() - started,
    )


@dataclass(frozen=True)
class Case:
    """One toy document in one direction: its sentences and the reference translations."""

    src_lang: str
    tgt_lang: str
    sentences: tuple[str, ...]
    references: tuple[str, ...]

    @property
    def text(self) -> str:
        """The document as the engine reads it: its sentences joined by a space."""
        return " ".join(self.sentences)


def cases_of(rows: Sequence[DocRow]) -> list[Case]:
    """The rows of each (direction, document), in `seg_idx` order."""
    groups: dict[tuple[str, str, str], list[DocRow]] = collections.defaultdict(list)
    for row in rows:
        groups[(row["src_lang"], row["tgt_lang"], row["doc_id"])].append(row)
    cases = []
    for group in groups.values():
        ordered = sorted(group, key=lambda row: row["seg_idx"])
        cases.append(
            Case(
                src_lang=ordered[0]["src_lang"],
                tgt_lang=ordered[0]["tgt_lang"],
                sentences=tuple(row["src"] for row in ordered),
                references=tuple(row["tgt"] for row in ordered),
            )
        )
    return cases


def windows_of_size(cases: Sequence[Case], size: int) -> list[Case]:
    """Every run of `size` consecutive sentences of every case, as a case of its own: a
    document of `size` sentences is exactly one window of that size for the chunker."""
    return [
        Case(
            case.src_lang,
            case.tgt_lang,
            case.sentences[start : start + size],
            case.references[start : start + size],
        )
        for case in cases
        for start in range(len(case.sentences) - size + 1)
    ]


def window_lengths(directory: Path, windows: Sequence[Case]) -> list[int]:
    """How many pieces, the end token included, CTranslate2 writes for each of `windows` (one
    window each) greedily from the document export at `directory`, with nothing of the engine
    but the export's files. On the CPU, one thread."""
    import ctranslate2

    processor = sentencepiece.SentencePieceProcessor(
        model_file=str(directory / "spm.model")
    )
    translator = ctranslate2.Translator(
        str(directory), device="cpu", compute_type="default", intra_threads=1
    )
    sources = [
        [
            direction_tag(window.tgt_lang),
            DOC_MARKER,
            *processor.encode(join_window(window.sentences), out_type=str),
        ]
        for window in windows
    ]
    results = translator.translate_batch(
        sources, beam_size=1, max_decoding_length=128, return_end_token=True
    )
    return [len(result.hypotheses[0]) for result in results]
