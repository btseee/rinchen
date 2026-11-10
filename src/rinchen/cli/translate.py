"""`rinchen translate`: translate text with a model of ours (an export `rinchen export convert`
wrote), from standard input or a file to standard output or a file. Parsing, the order of the
steps and the exit codes only; the work is `rinchen.infer.engine.Engine`'s, the reading, the
lines written and the summary `rinchen.cli.translate_io`'s.

    rinchen translate --model DIR --tgt CODE [--src CODE] [--script Cyrl|Mong]
        [--beam-size 4] [--n-best 1] [--max-decoding-length 256] [--max-source-pieces 256]
        [--batch-size 32] [--script-mask retry|always|off] [--device cpu|cuda]
        [--compute-type default] [--inter-threads 1] [--intra-threads 1] [--doc]
        [--format text|jsonl] [--input FILE] [--output FILE] [--strict]
        [--script-converter-ct2 DIR --script-converter-spm FILE]

`--model` (or `RINCHEN_MODEL`) is the export directory: it is opened, which has our manifest
vouch for every file of it, BEFORE any input is read, so a refused model writes nothing. The
texts are the lines of the input (UTF-8, read strictly, split at LF only; a CR before the LF is
dropped), translated in chunks of `--batch-size` lines as they are read and written, flushed,
chunk by chunk: there is exactly one output line per input line, in order, whatever the
statuses (a line that is empty, too long for the model or not translated gives an empty line;
`--format jsonl` says why: `text`, `status`, `alternatives`, `src`, `tgt`, `script_purity`).
`--tgt` is a registry code the model writes; `--src` the language of the texts when known (a
hint: the model is told only the target) and `--script` (`Cyrl` or `Mong`) picks the script of
a Mongolian target. With `--doc` the input is documents, blocks of lines separated by blank
lines, each translated as a whole (`--src` is then required): one output line per document,
a blank line between two. `--n-best` hypotheses are written by `--format jsonl` only.

Standard output carries only translations. The summary goes to standard error: `translated N
lines: ok A, empty B, too_long C, truncated D, no_output E`, the first line numbers of each
status that is a problem and the engine's counters that are not zero. The exit code is 0 unless
`--strict` and some line is not `ok` (4, after all the output is written).

`--device` is `cpu` unless an operator says `cuda`, which was never run here (unmeasured);
`auto` is refused, as is the compute type `auto`: it would choose the GPU, and it quantises a
float32 export to int8 on a CPU. `--script-converter-ct2` and `--script-converter-spm` (both or
neither) attach D5's script converter, so that a model with no `mn_Mong` target can answer a
`--script Mong` request by translating into Cyrillic and converting the text.

Exit codes: 0; 2 for a bad option, a model our manifest does not vouch for (a directory of the
old format, a changed file), a target the model does not write, an unknown language, a script
for a language that has none, `--doc` without `--src`, `--output` that is `--input`; 3 for an
input that cannot be read or is not UTF-8 (the message names the line, never quotes it) or
that the engine refuses; 4 when CTranslate2 fails or `--strict` meets a line that is not ok;
1 when the output cannot be written; 130 when interrupted. A failure of a chunk names its
lines; the chunks written before it stay written.
"""

from __future__ import annotations

import os
import re
from collections.abc import Callable, Iterator, Sequence
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from functools import partial
from itertools import batched
from pathlib import Path
from typing import IO, Annotated, Final, Literal, NoReturn

import typer

from rinchen.cli.errors import (
    FAILURE_EXIT_CODE,
    UPSTREAM_ERROR_EXIT_CODE,
    exit_with_error,
    rinchen_errors_as_exit_codes,
)
from rinchen.cli.translate_io import (
    Output,
    OutputFormat,
    Summary,
    Unit,
    document_line,
    read_blocks,
    read_lines,
    translation_line,
)
from rinchen.errors import ConfigError, DataError, UpstreamError
from rinchen.infer.engine import Engine
from rinchen.infer.engine_types import (
    DecodeOptions,
    LoadOptions,
    RequestTextError,
    Row,
    ScriptMode,
    ScriptName,
    document_status,
    selected_target,
)
from rinchen.script_conv.runtime import ScriptConverter

_SINGULAR: Final = {"lines": "line", "documents": "document"}
# How the engine names the text of a refusal: its place in the call, counted from 0.
_ENGINE_PLACE: Final = re.compile(r"(?:row|document) (\d+)")
# What a text of the input is joined with when a document has several lines.
_LINE_JOINER: Final = " "


def open_engine(
    model: Path, load: LoadOptions, converter_paths: tuple[Path, Path] | None
) -> Engine:
    """The engine over the export `model`: the one place the command touches the model (the
    tests replace it). With `converter_paths` (the CTranslate2 directory and the SentencePiece
    model of D5's script converter) the converter is loaded first and attached, on the device
    of the engine.

    Raises:
        ConfigError, DataError: as `Engine.open` and `ScriptConverter`.
    """
    converter = None
    if converter_paths is not None:
        ct2_dir, spm_model = converter_paths
        converter = ScriptConverter(ct2_dir, spm_model, device=load.device)
    return Engine.open(model, load, script_converter=converter)


def translate_command(
    model: Annotated[
        Path,
        typer.Option(
            "--model",
            envvar="RINCHEN_MODEL",
            help="The export directory `rinchen export convert` made.",
        ),
    ],
    tgt: Annotated[
        str,
        typer.Option(
            "--tgt", help="Registry code of the language to write, such as khk_Cyrl."
        ),
    ],
    src: Annotated[
        str | None,
        typer.Option(
            "--src",
            help=(
                "Registry code of the language of the texts, when known (a hint, not an "
                "input of the model); required with --doc."
            ),
        ),
    ] = None,
    script: Annotated[
        ScriptName | None,
        typer.Option(
            "--script",
            help="Script of a Mongolian target: Cyrl (khk_Cyrl) or Mong (mn_Mong).",
        ),
    ] = None,
    beam_size: Annotated[
        int, typer.Option("--beam-size", min=1, help="Beam width.")
    ] = 4,
    n_best: Annotated[
        int,
        typer.Option(
            "--n-best",
            min=1,
            help="Hypotheses kept per text, at most the beam's; --format jsonl lists them.",
        ),
    ] = 1,
    max_decoding_length: Annotated[
        int,
        typer.Option(
            "--max-decoding-length", min=1, help="Longest hypothesis, in pieces."
        ),
    ] = 256,
    max_source_pieces: Annotated[
        int,
        typer.Option(
            "--max-source-pieces",
            min=1,
            help="Longest text that is sent, in pieces; a longer one is too_long.",
        ),
    ] = 256,
    batch_size: Annotated[
        int,
        typer.Option(
            "--batch-size",
            min=1,
            help="Lines (documents with --doc) read, translated and written at a time.",
        ),
    ] = 32,
    script_mask: Annotated[
        ScriptMode,
        typer.Option(
            "--script-mask",
            help=(
                "retry decodes again, with the other scripts suppressed, an output that is "
                "not in the target's script; always suppresses them throughout; off never."
            ),
        ),
    ] = "retry",
    device: Annotated[
        Literal["cpu", "cuda"],
        typer.Option(
            "--device",
            help="Where CTranslate2 runs; never auto, which would pick the GPU.",
        ),
    ] = "cpu",
    compute_type: Annotated[
        str,
        typer.Option(
            "--compute-type",
            help="CTranslate2's compute type; default loads what the export stores.",
        ),
    ] = "default",
    inter_threads: Annotated[
        int,
        typer.Option(
            "--inter-threads", min=1, help="Batches CTranslate2 decodes at once."
        ),
    ] = 1,
    intra_threads: Annotated[
        int,
        typer.Option(
            "--intra-threads", min=1, help="Threads CTranslate2 uses per batch."
        ),
    ] = 1,
    doc: Annotated[
        bool,
        typer.Option(
            "--doc",
            help=(
                "Translate documents: blocks of lines between blank lines, one output "
                "line per document (needs --src and a document-level model)."
            ),
        ),
    ] = False,
    output_format: Annotated[
        OutputFormat,
        typer.Option(
            "--format",
            help="text: the translations; jsonl: an object per line with the status.",
        ),
    ] = "text",
    input_file: Annotated[
        Path | None,
        typer.Option("--input", help="Read the texts here instead of standard input."),
    ] = None,
    output_file: Annotated[
        Path | None,
        typer.Option(
            "--output", help="Write the output here instead of standard output."
        ),
    ] = None,
    strict: Annotated[
        bool,
        typer.Option(
            "--strict",
            help="Exit 4, after all the output is written, when a line is not ok.",
        ),
    ] = False,
    script_converter_ct2: Annotated[
        Path | None,
        typer.Option(
            "--script-converter-ct2",
            help="D5's script converter, its CTranslate2 directory (with the next option).",
        ),
    ] = None,
    script_converter_spm: Annotated[
        Path | None,
        typer.Option(
            "--script-converter-spm",
            help="D5's script converter, its SentencePiece model (with the previous option).",
        ),
    ] = None,
) -> None:
    """Translate text with a model of ours, one output line per input line.

    The model is opened before anything is read. Texts come from standard input or --input and
    go to standard output or --output; the summary goes to standard error."""
    with rinchen_errors_as_exit_codes():
        load = LoadOptions(
            device=device,
            compute_type=compute_type,
            inter_threads=inter_threads,
            intra_threads=intra_threads,
        )
        decode = DecodeOptions(
            beam_size=beam_size,
            n=n_best,
            max_decoding_length=max_decoding_length,
            max_source_pieces=max_source_pieces,
            max_batch_size=batch_size,
            script_mask=script_mask,
        )
        if doc and src is None:
            raise ConfigError(
                "--doc needs --src: a document is cut into sentences by the rules of its "
                "language"
            )
        request = _Request(
            tgt=tgt,
            # Raises for a script on a target that has none, before the model is opened.
            reported_target=selected_target(tgt, script),
            src=src,
            script=script,
            decode=decode,
            batch_size=batch_size,
            output_format=output_format,
        )
        converter_paths = _converter_paths(script_converter_ct2, script_converter_spm)
        _require_output_apart_from_input(input_file, output_file)
        engine = open_engine(Path(os.path.abspath(model)), load, converter_paths)
        try:
            # Before the output file is opened (and so emptied) and for an empty input too.
            engine.require_request(src, tgt, script, documents=doc)
            summary = _translate(
                engine, request, input_file, output_file, documents=doc
            )
        finally:
            engine.close()
        typer.echo(summary.stderr_text(), err=True)
        if strict and summary.problems:
            exit_with_error(
                f"--strict: {summary.problems} of {summary.total} {summary.unit} "
                "were not translated ok",
                UPSTREAM_ERROR_EXIT_CODE,
            )


@dataclass(frozen=True, slots=True)
class _Request:
    """What every call of the engine is asked, whatever the chunk: `tgt` as given and
    `reported_target` as the script option selects it."""

    tgt: str
    reported_target: str
    src: str | None
    script: ScriptName | None
    decode: DecodeOptions
    batch_size: int
    output_format: OutputFormat


def _converter_paths(
    ct2_dir: Path | None, spm_model: Path | None
) -> tuple[Path, Path] | None:
    if ct2_dir is None and spm_model is None:
        return None
    if ct2_dir is None or spm_model is None:
        raise ConfigError(
            "--script-converter-ct2 and --script-converter-spm go together: the converter "
            "needs its CTranslate2 directory and its SentencePiece model"
        )
    return Path(os.path.abspath(ct2_dir)), Path(os.path.abspath(spm_model))


def _require_output_apart_from_input(
    input_file: Path | None, output_file: Path | None
) -> None:
    """The output is truncated when it is opened, so an input that is the same file would be
    lost before it is read."""
    if input_file is None or output_file is None:
        return
    try:
        same = input_file.samefile(output_file)
    except OSError:
        # One of them is not there: they are not the same file.
        same = False
    if same:
        raise ConfigError(
            f"--output is the same file as --input ({input_file}): it would be emptied "
            "before it is read"
        )


# ---- the streams -------------------------------------------------------------------------


def _cannot_write(output_file: Path | None, error: OSError) -> NoReturn:
    where = "to standard output" if output_file is None else str(output_file)
    exit_with_error(f"cannot write the output {where}: {error}", FAILURE_EXIT_CODE)


def _discard_what_standard_output_still_holds(stream: IO[bytes]) -> None:
    """Standard output cannot be written (a closed pipe: `... | head -1`), and what its buffer
    still holds would fail again when the interpreter flushes it at exit, with a second
    message and the status 120 instead of ours: point its descriptor at the null device, so
    that the flush has somewhere to go."""
    try:
        descriptor = stream.fileno()
    except OSError, ValueError:
        # No descriptor behind it (a stream of a test): nothing is flushed at exit.
        return
    null = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(null, descriptor)
    finally:
        os.close(null)


def _input_stream(input_file: Path | None, stack: ExitStack) -> IO[bytes]:
    if input_file is None:
        return typer.get_binary_stream("stdin")
    try:
        return stack.enter_context(input_file.open("rb"))
    except OSError as error:
        raise DataError(f"cannot read the input {input_file}: {error}") from error


@contextmanager
def _output_stream(output_file: Path | None) -> Iterator[IO[bytes]]:
    """Standard output, or `output_file` opened for writing (and so emptied) and closed
    when the run is over: a failure to close it after a good run is a failure to write."""
    if output_file is None:
        yield typer.get_binary_stream("stdout")
        return
    try:
        stream = output_file.open("wb")
    except OSError as error:
        _cannot_write(output_file, error)
    try:
        yield stream
    except BaseException:
        try:
            stream.close()
        except OSError:
            # The run is already failing for another reason, which is the one to report.
            pass
        raise
    try:
        stream.close()
    except OSError as error:
        _cannot_write(output_file, error)


# ---- the run -----------------------------------------------------------------------------


def _translate(
    engine: Engine,
    request: _Request,
    input_file: Path | None,
    output_file: Path | None,
    *,
    documents: bool,
) -> Summary:
    """Read the input, translate it chunk by chunk and write each chunk when it is done."""
    summary = Summary("documents" if documents else "lines")
    chunks = _document_chunks if documents else _line_chunks
    with ExitStack() as stack:
        source = _input_stream(input_file, stack)
        sink = stack.enter_context(_output_stream(output_file))
        output = Output(
            sink, blank_between=documents and request.output_format == "text"
        )
        for lines in chunks(engine, request, source, summary):
            try:
                output.write(lines)
            except OSError as error:
                if output_file is None:
                    _discard_what_standard_output_still_holds(sink)
                _cannot_write(output_file, error)
    summary.counters = engine.counters()
    return summary


def _line_chunks(
    engine: Engine, request: _Request, source: IO[bytes], summary: Summary
) -> Iterator[list[str]]:
    """The output lines of each chunk of lines, as it is translated; every line is counted
    in `summary`."""
    done = 0
    for chunk in batched(read_lines(source), request.batch_size):
        rows = [Row(text, request.src, request.tgt, request.script) for text in chunk]
        translations = _answered(
            partial(engine.translate, rows, request.decode),
            "lines",
            done + 1,
            len(rows),
        )
        for number, translation in enumerate(translations, start=done + 1):
            summary.add(number, translation.status)
        done += len(rows)
        yield [
            translation_line(translation, request.output_format)
            for translation in translations
        ]


def _document_chunks(
    engine: Engine, request: _Request, source: IO[bytes], summary: Summary
) -> Iterator[list[str]]:
    """As `_line_chunks`, for documents: the lines of a block are one document."""
    assert request.src is not None
    done = 0
    for chunk in batched(read_blocks(source), request.batch_size):
        texts = [_LINE_JOINER.join(block) for block in chunk]
        ask = partial(
            engine.translate_documents,
            texts,
            request.src,
            request.tgt,
            request.decode,
            script=request.script,
        )
        results = _answered(ask, "documents", done + 1, len(texts))
        for number, result in enumerate(results, start=done + 1):
            summary.add(number, document_status(result))
        done += len(texts)
        yield [
            document_line(
                result,
                src=request.src,
                tgt=request.reported_target,
                output_format=request.output_format,
            )
            for result in results
        ]


def _span(unit: Unit, first: int, count: int) -> str:
    """Which texts a message is about: `line 4` or `lines 4 to 6`."""
    if count == 1:
        return f"{_SINGULAR[unit]} {first}"
    return f"{unit} {first} to {first + count - 1}"


def _named_by_place(
    error: DataError | UpstreamError, unit: Unit, first: int, count: int
) -> DataError | UpstreamError:
    """`error` as the command reports it. A refusal of one text names that text by its place
    in the input (`line 4`, `document 2`, counted from 1): the engine counted from 0 inside the
    call, and `first` is the number of the call's first text. Any other error says which texts
    the call was translating (`lines 3 to 4`), and then whatever the engine said."""
    span = _span(unit, first, count)
    if isinstance(error, RequestTextError):
        place = _ENGINE_PLACE.fullmatch(error.label)
        if place is None:
            return DataError(f"{span}: {error}")
        label = f"{_SINGULAR[unit]} {first + int(place[1])}"
        return RequestTextError(label, error.problem)
    return type(error)(f"{span}: {error}")


def _answered[T](
    ask: Callable[[], Sequence[T]], unit: Unit, first: int, count: int
) -> Sequence[T]:
    """What `ask` returns for `count` texts that start at the number `first`: one answer per
    text. An error of the data or of the library says which texts it was translating (the
    engine names a text by its place in the call)."""
    try:
        answers = ask()
    except (DataError, UpstreamError) as error:
        raise _named_by_place(error, unit, first, count) from error
    if len(answers) != count:
        raise UpstreamError(
            f"{_span(unit, first, count)}: the engine answered {len(answers)} times for "
            f"{count} texts"
        )
    return answers
