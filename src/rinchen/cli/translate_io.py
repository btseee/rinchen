"""What `rinchen translate` does with bytes: the lines and blocks it reads, the lines it writes
and the summary it prints on stderr. The engine and the options are `rinchen.cli.translate`'s.

Input is UTF-8 read strictly (a byte that is not UTF-8 is a `DataError` naming the line) and
split at LF only: U+2028, U+0085, a lone CR and the other characters `str.splitlines` splits at
stay inside their line, so a text never changes the number of lines. One CR before the LF (or at
the end of the last line) is dropped, so CRLF input reads as LF input. A final LF does not make a
last empty line; a last line needs no LF.

With `--doc` a block is the lines between blank lines (a line of only whitespace is blank) and
runs of blank lines count as one separator, so no block is ever empty.

Output is UTF-8 with LF line ends whatever the platform, flushed at each write. A line of output
never holds a line break (a translation's own are turned into a space, as the engine does: the
command's contract is one line out per line in), and a JSON line holds none either.
"""

from __future__ import annotations

import json
from collections.abc import Iterator, Mapping, Sequence
from typing import IO, Final, Literal

from rinchen.errors import DataError
from rinchen.infer.engine_types import LINE_BREAKS, DocumentResult, Status, Translation

OutputFormat = Literal["text", "jsonl"]
Unit = Literal["lines", "documents"]

STATUSES: Final[tuple[Status, ...]] = (
    "ok",
    "empty",
    "too_long",
    "truncated",
    "no_output",
)
# The worst status of a document first. `empty` is no problem of a sentence (a sentence of a
# document with no letter is normal): a document is `empty` only when nothing in it is not.
_SEVERITY: Final[tuple[Status, ...]] = ("no_output", "too_long", "truncated")
# Engine counters the summary names, when they are not zero: the pair was not one the model was
# trained for, and everything about the script (the retry, the purity, the converter).
SUMMARY_COUNTERS: Final = (
    "zero_shot",
    "script_retries",
    "script_retry_improved",
    "script_impure",
    "script_conversions",
    "converter_rule_fallbacks",
    "converter_invalid_outputs",
)
_SHOWN_NUMBERS: Final = 5
# JSON allows these raw; a reader that splits lines at them would cut the line.
_JSON_LINE_BREAKS: Final = {
    ord(character): f"\\u{ord(character):04x}"
    for character in (chr(0x85), chr(0x2028), chr(0x2029))
}


# ---- reading -----------------------------------------------------------------------------


def read_lines(stream: IO[bytes]) -> Iterator[str]:
    """The lines of `stream`, one at a time as they are asked for (see the module's text).

    Raises:
        DataError: a line is not valid UTF-8, or the stream cannot be read; the message names
            the line and never quotes it.
    """
    number = 0
    while True:
        number += 1
        try:
            raw = stream.readline()
        except OSError as error:
            raise DataError(
                f"cannot read the input at line {number}: {error}"
            ) from error
        if not raw:
            return
        try:
            text = raw.removesuffix(b"\n").removesuffix(b"\r").decode("utf-8")
        except UnicodeDecodeError as error:
            raise DataError(
                f"input line {number} is not valid UTF-8: invalid byte at offset "
                f"{error.start} of the line"
            ) from error
        yield text


def read_blocks(stream: IO[bytes]) -> Iterator[list[str]]:
    """The documents of `stream`: the lines between blank lines (see the module's text).

    Raises:
        DataError: as `read_lines`.
    """
    block: list[str] = []
    for line in read_lines(stream):
        if line.strip():
            block.append(line)
        elif block:
            yield block
            block = []
    if block:
        yield block


# ---- the output lines --------------------------------------------------------------------


def _one_line(text: str) -> str:
    return LINE_BREAKS.sub(" ", text)


def _json_line(
    *,
    text: str,
    status: Status,
    alternatives: Sequence[str],
    src: str,
    tgt: str,
    script_purity: float | None,
) -> str:
    document = {
        "text": text,
        "status": status,
        "alternatives": list(alternatives),
        "src": src,
        "tgt": tgt,
        "script_purity": script_purity,
    }
    line = json.dumps(document, ensure_ascii=False, separators=(",", ":"))
    return line.translate(_JSON_LINE_BREAKS)


def translation_line(translation: Translation, output_format: OutputFormat) -> str:
    """The line for one text: its translation, or a JSON object with `text`, `status`,
    `alternatives`, `src`, `tgt` and `script_purity`."""
    if output_format == "text":
        return _one_line(translation.text)
    return _json_line(
        text=translation.text,
        status=translation.status,
        alternatives=translation.alternatives,
        src=translation.src,
        tgt=translation.tgt,
        script_purity=translation.script_purity,
    )


def document_status(result: DocumentResult) -> Status:
    """The status of a document: the worst of its sentences (`no_output`, `too_long`,
    `truncated`), else `ok`, and `empty` when no sentence is anything but empty."""
    statuses = {translation.status for translation in result.translations}
    for status in _SEVERITY:
        if status in statuses:
            return status
    return "ok" if "ok" in statuses else "empty"


def document_line(
    result: DocumentResult, *, src: str, tgt: str, output_format: OutputFormat
) -> str:
    """The line for one document: its text, or a JSON object of the keys of a sentence with
    the document's status, no alternatives and the lowest purity of its sentences."""
    if output_format == "text":
        return _one_line(result.text)
    purities = [
        translation.script_purity
        for translation in result.translations
        if translation.script_purity is not None
    ]
    return _json_line(
        text=result.text,
        status=document_status(result),
        alternatives=(),
        src=src,
        tgt=tgt,
        script_purity=min(purities, default=None),
    )


class Output:
    """The lines written to `stream`, each ending with LF, flushed as soon as they are
    written; `blank_between` puts a blank line between two lines written (across writes), not
    before the first or after the last."""

    def __init__(self, stream: IO[bytes], *, blank_between: bool = False) -> None:
        self._stream = stream
        self._blank_between = blank_between
        self._written = False

    def write(self, lines: Sequence[str]) -> None:
        """Write `lines` and flush.

        Raises:
            OSError: the stream cannot be written.
        """
        if not lines:
            return
        pieces = []
        for line in lines:
            if self._written and self._blank_between:
                pieces.append("\n")
            pieces.append(f"{line}\n")
            self._written = True
        self._stream.write("".join(pieces).encode("utf-8"))
        self._stream.flush()


# ---- the summary -------------------------------------------------------------------------


class Summary:
    """What a run came to: the number of texts per status, the first numbers of each status
    that is a problem, and the engine's counters; `stderr_text` is what the command prints.

    A text is a line, or a document with `--doc` (`unit`): its number counts from 1."""

    def __init__(self, unit: Unit) -> None:
        self.unit = unit
        self.counts: dict[Status, int] = dict.fromkeys(STATUSES, 0)
        self._first: dict[Status, list[int]] = {status: [] for status in STATUSES}
        self.counters: Mapping[str, int] = {}

    def add(self, number: int, status: Status) -> None:
        """Count the text `number` with `status`.

        Raises:
            ValueError: `status` is not one of the engine's.
        """
        if status not in self.counts:
            raise ValueError(f"not a status of the engine: {status!r}")
        self.counts[status] += 1
        if status != "ok" and len(self._first[status]) < _SHOWN_NUMBERS:
            self._first[status].append(number)

    @property
    def total(self) -> int:
        return sum(self.counts.values())

    @property
    def problems(self) -> int:
        """The texts that are not `ok`: what `--strict` makes an error."""
        return self.total - self.counts["ok"]

    def stderr_text(self) -> str:
        """The summary, one line without a final line end per part: the counts, the first
        numbers of each problem and the counters that are not zero."""
        tally = ", ".join(f"{status} {self.counts[status]}" for status in STATUSES)
        lines = [f"translated {self.total} {self.unit}: {tally}"]
        lines.extend(
            f"{status}: first {self.unit} " + ", ".join(map(str, self._first[status]))
            for status in STATUSES
            if self._first[status]
        )
        named = [
            f"{name} {self.counters[name]}"
            for name in SUMMARY_COUNTERS
            if self.counters.get(name, 0)
        ]
        if named:
            lines.append("counters: " + ", ".join(named))
        return "\n".join(lines)
