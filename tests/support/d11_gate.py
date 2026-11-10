"""The stages and helpers of the D11 export and engine gates (`tests/integration/
test_export_parity.py` and `tests/integration/test_engine_surfaces.py`): every stage runs the
real `rinchen` binary (the real Eole and CTranslate2, on the CPU) or the library the way a
user of the package would, and returns a snapshot of what the tests look at.

The session fixtures of `tests/integration/conftest.py` hide every device
(`CUDA_VISIBLE_DEVICES=""`) before they call any of them. Eole trains with `gpu_ranks: []`,
the converter runs with the device hidden and every CTranslate2 translator of the tests is
built on `device="cpu"` with the compute type the file stores (never `auto`).

The data are the toy corpora of `tests/support/tiny_parallel.py`: templated sentences, so
every number the gates look at tests machinery and ranks nothing.

The checks that count and compare take their expectations from the tests as literals; the
helpers here only run, read and count. The letters of a text are classified by the Unicode
names of its characters, not by `rinchen.lang.scripts`, so a script rule of the project cannot
hide its own mistake from the gate.
"""

from __future__ import annotations

import ast
import json
import subprocess
import time
import unicodedata
from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Final

import sentencepiece

from rinchen.config.ablation import TagPlacement
from rinchen.config.languages import direction_tag
from rinchen.export.verified import verify_export
from rinchen.infer.engine import Engine
from rinchen.infer.engine_types import DecodeOptions, LoadOptions, Row, Translation
from rinchen.infer.script_mask import suppression_list
from rinchen.metrics.bleu_chrf import corpus_scores
from rinchen.train.export_parity import ParityReport, ParitySource, strict_parity
from tests.support.distill_gate import (
    ENGINE_TIMEOUT_SECONDS,
    RINCHEN,
    SUBPROCESS_TIMEOUT_SECONDS,
    THREADS,
    Invocation,
    read_json,
    run_rinchen,
    succeeded,
)
from tests.support.distill_gate_stages import (
    DIRECTIONS,
    Exported,
    Teacher,
    Workspace,
    manifest_of,
    train_teacher,
)

SOURCE_TEACHER_STEPS: Final = 2000
TARGET_TEACHER_STEPS: Final = 1000
PARITY_LINES_PER_DIRECTION: Final = 50
# The shortest decode the gates call "greedy": the length Eole's prediction is held to.
GREEDY: Final = DecodeOptions(beam_size=1, max_decoding_length=128)
# Requests the toy never trained for (and so are not directions of any export of the gates):
# the same language, and the two directions the toy has no corpus for.
UNTRAINED_PAIRS: Final = (
    ("eng_Latn", "eng_Latn"),
    ("eng_Latn", "rus_Cyrl"),
    ("khk_Cyrl", "khk_Cyrl"),
    ("rus_Cyrl", "rus_Cyrl"),
    ("rus_Cyrl", "eng_Latn"),
)
SOURCES_PER_LANGUAGE: Final = 100
# The prefix of the Unicode name of a letter, by language of the toy.
_SCRIPT_NAME: Final = {
    "eng_Latn": "LATIN",
    "khk_Cyrl": "CYRILLIC",
    "rus_Cyrl": "CYRILLIC",
}


# ---- the teachers ----------------------------------------------------------------------


@dataclass(frozen=True)
class Teachers:
    """The tiny teachers of the gates: the direction tag in the source (2,000 steps) and as
    the forced prefix of the target (1,000 steps), each averaged over its last two saves."""

    source: Teacher
    target: Teacher


def trained_teachers(workspace: Workspace) -> Teachers:
    """Train both teachers through D10's stage (`train ablate --prepare-only`, `train launch`
    and `train average`); the target-tag one in a directory of its own."""
    target_root = workspace.root / "target_teacher"
    target_root.mkdir()
    return Teachers(
        source=train_teacher(workspace, placement="src", steps=SOURCE_TEACHER_STEPS),
        target=train_teacher(
            replace(workspace, root=target_root),
            placement="tgt",
            steps=TARGET_TEACHER_STEPS,
        ),
    )


# ---- the exports -----------------------------------------------------------------------


def convert_export(
    workspace: Workspace,
    checkpoint: Path,
    out_dir: Path,
    *,
    placement: TagPlacement,
    quantization: str = "float32",
) -> Exported:
    """`rinchen export convert` of `checkpoint` with the toy model: the placement is stated
    and the command derives and checks the rest."""
    invocation = succeeded(
        run_rinchen(
            "export",
            "convert",
            "--checkpoint",
            checkpoint,
            "--spm-model",
            workspace.inputs.spm_model,
            "--tag-placement",
            placement,
            "--quantization",
            quantization,
            "--out-dir",
            out_dir,
        )
    )
    return Exported(
        invocation=invocation,
        directory=out_dir,
        record=read_json(out_dir / "rinchen.json"),
        manifest=manifest_of(out_dir),
        converter_log=out_dir.parent / f"{out_dir.name}.export.log",
    )


@dataclass(frozen=True)
class Exports:
    """The sentence-level exports of the gates: the source-tag teacher at float32, int8 and
    int8_float16, and the target-tag teacher at float32."""

    float32: Exported
    int8: Exported
    int8_float16: Exported
    target_float32: Exported


def converted_exports(workspace: Workspace, teachers: Teachers) -> Exports:
    base = workspace.root / "exports"
    base.mkdir()
    source = teachers.source.averaged
    return Exports(
        float32=convert_export(
            workspace, source, base / "src_float32", placement="src"
        ),
        int8=convert_export(
            workspace, source, base / "src_int8", placement="src", quantization="int8"
        ),
        int8_float16=convert_export(
            workspace,
            source,
            base / "src_int8_float16",
            placement="src",
            quantization="int8_float16",
        ),
        target_float32=convert_export(
            workspace, teachers.target.averaged, base / "tgt_float32", placement="tgt"
        ),
    )


# ---- strict parity ---------------------------------------------------------------------


@dataclass(frozen=True)
class Parity:
    """One `rinchen export parity` run: the command, the result it wrote and its manifest."""

    invocation: Invocation
    out_dir: Path
    result: dict[str, Any]


def run_parity(
    workspace: Workspace, exported: Exported, checkpoint: Path, out_dir: Path
) -> Parity:
    """The 50 first dev sentences of each direction of the export through Eole and
    CTranslate2. The exit code is the caller's to read: a miss of the gate is exit 4."""
    invocation = run_rinchen(
        "export",
        "parity",
        "--checkpoint",
        checkpoint,
        "--export",
        exported.directory,
        "--corpus",
        workspace.inputs.corpus,
        "--out-dir",
        out_dir,
        "--max-lines",
        str(PARITY_LINES_PER_DIRECTION),
        "--threads",
        THREADS,
        "--timeout",
        str(ENGINE_TIMEOUT_SECONDS),
    )
    result_file = out_dir / "parity.json"
    return Parity(
        invocation=invocation,
        out_dir=out_dir,
        result=read_json(result_file) if result_file.is_file() else {},
    )


class _ScoresWithPrefix:
    """A CTranslate2 translator whose scores keep the log-probability of the forced prefix.

    The parity library drops the first log-probability of every scored line when the placement
    is `tgt`, because Eole's own score leaves the forced tag out. This wrapper hands the library
    one extra leading entry, which that drop removes, so that the tag's log-probability stays
    in the sum: the comparison a library that forgot the drop would make."""

    def __init__(self, translator: Any) -> None:
        self._translator = translator

    def translate_batch(self, source: list[list[str]], **options: Any) -> Any:
        return self._translator.translate_batch(source, **options)

    def score_batch(
        self, source: list[list[str]], target: list[list[str]], **options: Any
    ) -> list[Any]:
        class Padded:
            def __init__(self, log_probs: Sequence[float]) -> None:
                self.log_probs = [0.0, *log_probs]

        scored = self._translator.score_batch(source, target, **options)
        return [Padded(each.log_probs) for each in scored]


def parity_without_the_prefix_drop(
    workspace: Workspace, exported: Exported, checkpoint: Path, out_dir: Path
) -> ParityReport:
    """The library's strict parity of the export at `exported` (a `tgt` placement) with the
    forced tag's log-probability kept in the CTranslate2 score: the trap the library avoids."""
    import ctranslate2

    verified = verify_export(exported.directory)
    translator = ctranslate2.Translator(
        str(exported.directory),
        device="cpu",
        compute_type="default",
        intra_threads=int(THREADS),
    )
    sources = [
        ParitySource(text, dev.tgt_lang)
        for direction in DIRECTIONS
        for dev in [workspace.inputs.dev[direction]]
        for text in dev.src[:PARITY_LINES_PER_DIRECTION]
    ]
    return strict_parity(
        checkpoint,
        verified,
        sources,
        out_dir=out_dir,
        threads=int(THREADS),
        timeout=SUBPROCESS_TIMEOUT_SECONDS,
        engine=_ScoresWithPrefix(translator),
    )


# ---- the engine over a dev set ---------------------------------------------------------


def open_engine(directory: Path, *, compute_type: str = "default") -> Engine:
    """The engine over the export at `directory`, on the CPU, one thread of each kind."""
    return Engine.open(directory, LoadOptions(device="cpu", compute_type=compute_type))


def dev_rows(workspace: Workspace) -> tuple[list[Row], list[str]]:
    """Every dev row of the four directions (100 each) as a request and its reference."""
    rows: list[Row] = []
    references: list[str] = []
    for direction in DIRECTIONS:
        dev = workspace.inputs.dev[direction]
        rows += [Row(text, dev.src_lang, dev.tgt_lang) for text in dev.src]
        references += list(dev.tgt)
    return rows, references


def greedy_chrf(engine: Engine, workspace: Workspace) -> float:
    """chrF of the engine's greedy translations of the 400 dev rows."""
    rows, references = dev_rows(workspace)
    translations = engine.translate(rows, GREEDY)
    assert all(translation.status == "ok" for translation in translations)
    return corpus_scores([each.text for each in translations], references).chrf


def raw_greedy_texts(
    directory: Path, rows: Sequence[Row], *, placement: TagPlacement
) -> list[str]:
    """The texts CTranslate2 writes for `rows` greedily from the export at `directory`, with
    nothing of the engine but the export's files: the tag goes where `placement` says and, for
    `tgt`, the forced tag the hypothesis starts with is taken off. On the CPU, one thread."""
    import ctranslate2

    processor = sentencepiece.SentencePieceProcessor(
        model_file=str(directory / "spm.model")
    )
    translator = ctranslate2.Translator(
        str(directory), device="cpu", compute_type="default", intra_threads=1
    )
    tags = [[direction_tag(row.tgt)] for row in rows]
    pieces = [processor.encode(row.text, out_type=str) for row in rows]
    if placement == "src":
        results = translator.translate_batch(
            [[*tag, *each] for tag, each in zip(tags, pieces, strict=True)],
            beam_size=1,
            max_decoding_length=128,
        )
        return [processor.decode(list(each.hypotheses[0])) for each in results]
    results = translator.translate_batch(
        pieces, target_prefix=tags, beam_size=1, max_decoding_length=128
    )
    return [processor.decode(list(each.hypotheses[0][1:])) for each in results]


# ---- script masking --------------------------------------------------------------------


def untrained_requests(workspace: Workspace) -> list[Row]:
    """500 requests the toy never trained for: 100 dev sources of the language of each
    pair, `src=None` (a language given would refuse the same-language pairs)."""
    sources = {
        "eng_Latn": workspace.inputs.dev["eng_Latn-khk_Cyrl"].src,
        "khk_Cyrl": workspace.inputs.dev["khk_Cyrl-eng_Latn"].src,
        "rus_Cyrl": workspace.inputs.dev["rus_Cyrl-khk_Cyrl"].src,
    }
    return [
        Row(text, None, target)
        for source, target in UNTRAINED_PAIRS
        for text in sources[source][:SOURCES_PER_LANGUAGE]
    ]


def trained_requests(workspace: Workspace) -> list[Row]:
    """The 400 dev sources of the four directions the toy was trained for, `src=None`."""
    rows: list[Row] = []
    for direction in DIRECTIONS:
        dev = workspace.inputs.dev[direction]
        rows += [
            Row(text, None, dev.tgt_lang) for text in dev.src[:SOURCES_PER_LANGUAGE]
        ]
    return rows


def forbidden_letters(text: str, target: str) -> int:
    """How many letters of `text` are in no script of the language `target` writes (one of
    the three languages of the toy), from the Unicode names of the characters."""
    allowed = _SCRIPT_NAME[target]
    return sum(
        1
        for character in text
        if character.isalpha()
        and not unicodedata.name(character, "").startswith(allowed)
    )


def outputs_with_forbidden_letters(
    rows: Sequence[Row], translations: Sequence[Translation]
) -> tuple[int, int]:
    """(outputs with a letter of a script their target does not write, such letters)."""
    counts = [
        forbidden_letters(translation.text, row.tgt)
        for row, translation in zip(rows, translations, strict=True)
    ]
    return sum(1 for count in counts if count), sum(counts)


def delta(before: dict[str, int], after: dict[str, int]) -> dict[str, int]:
    """The counters that moved between two readings of `Engine.counters`."""
    return {
        name: after[name] - before[name]
        for name in after
        if after[name] != before[name]
    }


@dataclass(frozen=True)
class ConverterLeak:
    """The D5 tiny converter driven through CTranslate2 into a script it must not write."""

    words: int
    with_script_unmasked: int
    with_script_masked: int
    listed_entries: int


def has_mongolian_letter(text: str) -> bool:
    return any(
        character.isalpha() and unicodedata.name(character, "").startswith("MONGOLIAN")
        for character in text
    )


def converter_leak(
    ct2_dir: Path, spm_model: Path, words: Sequence[str], tag: str
) -> ConverterLeak:
    """The words through the converter that is asked, by `tag`, for the traditional script,
    without a suppression list and with the list of a Cyrillic target (which forbids every
    script but Cyrillic). The converter is a character model with tags of its own, so it
    cannot go through `Engine`: this is CTranslate2 with `suppression_list` directly."""
    import ctranslate2

    processor = sentencepiece.SentencePieceProcessor(model_file=str(spm_model))
    vocabulary = json.loads((ct2_dir / "shared_vocabulary.json").read_text("utf-8"))
    listed = suppression_list(vocabulary, processor, "khk_Cyrl", end_token="</s>")
    translator = ctranslate2.Translator(
        str(ct2_dir), device="cpu", compute_type="default", intra_threads=1
    )
    sources = [[tag, *processor.encode(word, out_type=str)] for word in words]

    def with_script(**options: Any) -> int:
        results = translator.translate_batch(
            sources, beam_size=1, max_decoding_length=64, **options
        )
        return sum(
            has_mongolian_letter(processor.decode(list(result.hypotheses[0])))
            for result in results
        )

    return ConverterLeak(
        words=len(words),
        with_script_unmasked=with_script(),
        with_script_masked=with_script(suppress_sequences=listed, disable_unk=True),
        listed_entries=len(listed),
    )


# ---- `rinchen translate` ---------------------------------------------------------------


@dataclass(frozen=True)
class Translated:
    """One `rinchen translate` run: the output split at LF, the standard error and the time."""

    returncode: int
    lines: list[str]
    stdout: bytes
    stderr: str
    seconds: float


def run_translate(model: Path, lines: Sequence[str], *args: str | Path) -> Translated:
    """`rinchen translate --model <model> <args>` with `lines` on standard input, one per
    line, run to its end (at most `SUBPROCESS_TIMEOUT_SECONDS`)."""
    started = time.perf_counter()
    completed = subprocess.run(
        [RINCHEN, "translate", "--model", str(model), *map(str, args)],
        input="".join(f"{line}\n" for line in lines).encode("utf-8"),
        capture_output=True,
        timeout=SUBPROCESS_TIMEOUT_SECONDS,
        check=False,
    )
    text = completed.stdout.decode("utf-8")
    return Translated(
        returncode=completed.returncode,
        lines=text.removesuffix("\n").split("\n") if text else [],
        stdout=completed.stdout,
        stderr=completed.stderr.decode("utf-8", "replace"),
        seconds=time.perf_counter() - started,
    )


def run_translate_into_a_closed_pipe(
    model: Path, input_file: Path, *args: str | Path
) -> Translated:
    """`rinchen translate --model <model> --input <input_file> <args>` whose standard output
    is a pipe that is closed after the first line, as `... | head -1` does: the result
    holds that one line, the exit status and what the command said on standard error."""
    started = time.perf_counter()
    process = subprocess.Popen(
        [RINCHEN, "translate", "--model", str(model), "--input", str(input_file)]
        + [str(arg) for arg in args],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert process.stdout is not None and process.stderr is not None
    try:
        first = process.stdout.readline()
        process.stdout.close()
        stderr = process.stderr.read()
        returncode = process.wait(timeout=SUBPROCESS_TIMEOUT_SECONDS)
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()
        process.stderr.close()
    return Translated(
        returncode=returncode,
        lines=[first.decode("utf-8").removesuffix("\n")],
        stdout=first,
        stderr=stderr.decode("utf-8", "replace"),
        seconds=time.perf_counter() - started,
    )


# ---- no GPU ------------------------------------------------------------------------------

# Spelled in two parts so that this file does not hold what it looks for.
_GPU_TOOL_STRING: Final = "nvidia" + "-smi"
_GPU_COUNT_NAME: Final = "get_cuda" + "_device_count"
_CUDA_NAMESPACE: Final = ("torch", "cuda")


def forbidden_gpu_findings(path: Path) -> list[str]:
    """What the Python file `path` spells out that asks a device question: the `torch.cuda`
    namespace, the library call that counts CUDA devices (as an attribute, a name or an import)
    or the name of the vendor's query tool inside a string. Reads the structure without
    importing the file."""
    found: list[str] = []
    for node in ast.walk(ast.parse(path.read_text("utf-8"))):
        if isinstance(node, ast.Attribute):
            if node.attr == _GPU_COUNT_NAME:
                found.append(f"line {node.lineno}: attribute {node.attr}")
            if (
                isinstance(node.value, ast.Name)
                and (node.value.id, node.attr) == _CUDA_NAMESPACE
            ):
                found.append(f"line {node.lineno}: {'.'.join(_CUDA_NAMESPACE)}")
        elif isinstance(node, ast.Name) and node.id == _GPU_COUNT_NAME:
            found.append(f"line {node.lineno}: name {node.id}")
        elif (
            isinstance(node, ast.alias) and node.name.split(".")[-1] == _GPU_COUNT_NAME
        ):
            found.append(f"import of {node.name}")
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            if _GPU_TOOL_STRING in node.value or _GPU_COUNT_NAME in node.value:
                found.append(f"line {node.lineno}: a string naming a device query")
    return found
