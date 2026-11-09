"""The types of the engine (D11): what it is opened with, what it is asked, what it answers.

`LoadOptions` say how the model is loaded (device, compute type, threads); `DecodeOptions` how
a call decodes; a `Row` is one text with the languages it is asked for; a `Translation` is the
answer to one row, a `DocumentResult` the answer to one document. All of them are frozen and
validate what they are given when they are made, so an option the library cannot honour never
reaches a decode.

`device="auto"` and `compute_type="auto"` are refused: `auto` quantises every export to int8
on a CPU (measured in the D11 spike) and would choose the GPU when one is visible. No GPU was
used and none is claimed: `cuda` is an operator's explicit choice that was never run.

The second half of the module is what `rinchen.infer.engine` works with inside a call (a
route, a job, what the hypotheses of a source came to) and the small rules about a text, a
language and a script that it applies; nothing outside the engine needs them.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Final, Literal, Protocol

from rinchen.errors import ConfigError, DataError
from rinchen.export.quantize import require_compute_type
from rinchen.export.record import CT2_FILES, ModelRecord
from rinchen.export.verified import VerifiedExport
from rinchen.infer import script_mask
from rinchen.infer.doc_chunker import SEGMENT_SEPARATOR, ChunkStats, one_line
from rinchen.infer.script_mask import PURITY_FLOOR, allowed_scripts, script_purity
from rinchen.lang.scripts import Script
from rinchen.utils.integers import is_strict_int

Status = Literal["ok", "empty", "too_long", "truncated", "no_output"]
ScriptMode = Literal["retry", "always", "off"]
ScriptName = Literal["Cyrl", "Mong"]

SCRIPT_MODES: Final = ("retry", "always", "off")
SCRIPT_NAMES: Final = ("Cyrl", "Mong")

# What the engine counts, cumulative since it was opened (`Engine.counters`).
ENGINE_COUNTERS: Final = (
    "sources",
    "empty",
    "too_long",
    "truncated",
    "no_output",
    "reserved_symbol",
    "line_break",
    "zero_shot",
    "script_retries",
    "script_retry_improved",
    "script_impure",
    "script_conversions",
    "converter_rule_fallbacks",
    "converter_invalid_outputs",
    "windows",
    "window_fallbacks",
)

# CTranslate2's `max_input_length` is 1,024 and cuts a longer source silently; a tag in the
# source takes one place of it. The same limit D10's teacher keeps.
MAX_SOURCE_PIECES: Final = 1023

_DEVICES: Final = ("cpu", "cuda")


@dataclass(frozen=True, slots=True)
class LoadOptions:
    """How the model is loaded.

    `device` is `cpu` or `cuda`; `compute_type` is CTranslate2's (`default` loads what the
    export stores: the quantisation chosen at conversion); `inter_threads` is the number of
    batches decoded at once and `intra_threads` the threads of one; `max_queued_batches`
    bounds the library's queue (-1: unbounded).

    Raises:
        ConfigError: `device` is not `cpu` or `cuda` (`auto` is refused with the reason),
            `compute_type` is blank or `auto`, a thread count is not an integer of at least
            1 (a bool is not one) or `max_queued_batches` is not an integer of at least -1.
    """

    device: Literal["cpu", "cuda"] = "cpu"
    compute_type: str = "default"
    inter_threads: int = 1
    intra_threads: int = 1
    max_queued_batches: int = -1

    def __post_init__(self) -> None:
        if self.device == "auto":
            raise ConfigError(
                "LoadOptions.device 'auto' is refused: CTranslate2 would choose the GPU "
                "when one is visible; say 'cpu' or 'cuda'"
            )
        if self.device not in _DEVICES:
            raise ConfigError(
                f"LoadOptions.device must be 'cpu' or 'cuda', got {self.device!r}"
            )
        if not isinstance(self.compute_type, str):
            raise ConfigError(
                f"LoadOptions.compute_type must be a string, got {self.compute_type!r}"
            )
        require_compute_type(self.compute_type)
        for name in ("inter_threads", "intra_threads"):
            _require_count(f"LoadOptions.{name}", getattr(self, name), minimum=1)
        _require_count(
            "LoadOptions.max_queued_batches", self.max_queued_batches, minimum=-1
        )


@dataclass(frozen=True, slots=True)
class DecodeOptions:
    """How a call decodes.

    `beam_size` and `n` (the hypotheses kept per source: at most the beam's),
    `max_decoding_length` (pieces of a hypothesis, the end token included),
    `max_source_pieces` (the longest source that is sent: a text of more is `too_long`),
    `max_batch_size` (what the library splits a call into) and `script_mask`: `retry`
    decodes without a script list and decodes an impure output again with it, `always`
    decodes every row with it, `off` never suppresses a script.

    Raises:
        ConfigError: a number is not an integer of at least 1 (a bool is not one), `n` is
            above `beam_size`, `max_source_pieces` is above 1,023 or `script_mask` is not
            one of the three.
    """

    beam_size: int = 4
    n: int = 1
    max_decoding_length: int = 256
    max_source_pieces: int = 256
    max_batch_size: int = 32
    script_mask: ScriptMode = "retry"

    def __post_init__(self) -> None:
        for name in (
            "beam_size",
            "n",
            "max_decoding_length",
            "max_source_pieces",
            "max_batch_size",
        ):
            _require_count(f"DecodeOptions.{name}", getattr(self, name), minimum=1)
        if self.n > self.beam_size:
            raise ConfigError(
                f"DecodeOptions.n ({self.n}) must not exceed DecodeOptions.beam_size "
                f"({self.beam_size}): CTranslate2 returns at most as many hypotheses as "
                "the beam holds"
            )
        if self.max_source_pieces > MAX_SOURCE_PIECES:
            raise ConfigError(
                f"DecodeOptions.max_source_pieces ({self.max_source_pieces}) must be at "
                f"most {MAX_SOURCE_PIECES}: CTranslate2 cuts a source of more than 1,024 "
                "pieces silently, and the tag takes one"
            )
        if self.script_mask not in SCRIPT_MODES:
            raise ConfigError(
                f"DecodeOptions.script_mask must be one of {', '.join(SCRIPT_MODES)}, "
                f"got {self.script_mask!r}"
            )


def _require_count(name: str, value: object, *, minimum: int) -> None:
    if not is_strict_int(value) or value < minimum:
        raise ConfigError(
            f"{name} must be an integer of at least {minimum}, got {value!r}"
        )


@dataclass(frozen=True, slots=True)
class Row:
    """One text and what it is asked for: `src` the language of the text (a registry code,
    or None when unknown: a hint, never an input of the model), `tgt` the registry code of
    the language to write and `script` (`Cyrl` or `Mong`) the script of a Mongolian target."""

    text: str
    src: str | None
    tgt: str
    script: ScriptName | None = None


@dataclass(frozen=True, slots=True)
class Translation:
    """The answer to one `Row`.

    `status` is `ok`, `empty` (no letter or digit, or no piece: not sent), `too_long` (more
    pieces than `max_source_pieces`: not sent), `truncated` (the model was cut before the end
    token: `text` is the partial translation) or `no_output` (every hypothesis was dropped:
    `text` is empty). `alternatives` are the other hypotheses, distinct and in order. `src`
    is the language given, or the script hint (`mn_Mong` or `und`) when none was; `tgt` the
    language asked for. `script_purity` is the share of the letters of `text` in the scripts
    of `tgt` (None when there is no text), `retried` whether the output was decoded again with
    the script list and `zero_shot` whether the pair is one the model was not trained for.
    """

    text: str
    status: Status
    alternatives: tuple[str, ...]
    src: str
    tgt: str
    script_purity: float | None
    retried: bool
    zero_shot: bool


@dataclass(frozen=True, slots=True)
class DocumentResult:
    """The answer to one document: `text` is its translations joined (by a space, by nothing
    for a target written without spaces), `translations` has one entry per sentence in order
    and `stats` is how the chunker cut and translated it."""

    text: str
    translations: tuple[Translation, ...]
    stats: ChunkStats


@dataclass(frozen=True, slots=True)
class ModelInfo:
    """What a loaded engine knows about its model: the record's `name`, `quantization`,
    `placement`, `directions`, `targets` and `document_level`, the `device` and the effective
    `compute_type` it runs on and the SHA-256 of `model.bin` and of the manifest that vouched
    for the export."""

    name: str
    quantization: str
    device: str
    compute_type: str
    placement: str
    directions: tuple[str, ...]
    targets: tuple[str, ...]
    document_level: bool
    model_sha256: str
    manifest_sha256: str


class ConversionLike(Protocol):
    """What the engine reads of a script converter's answer (D5's `Conversion` has it)."""

    @property
    def text(self) -> str: ...

    @property
    def rule_fallbacks(self) -> int: ...

    @property
    def invalid_outputs(self) -> int: ...


class ScriptConverterLike(Protocol):
    """What the engine asks of a script converter (D5's `ScriptConverter` fits): the text
    converted to the script `target`, `Mong` or `Cyrl`."""

    def convert(self, text: str, target: ScriptName) -> ConversionLike: ...


# ---- what the engine works with inside a call ---------------------------------------------

CYRILLIC_MONGOLIAN: Final = "khk_Cyrl"
TRADITIONAL_MONGOLIAN: Final = "mn_Mong"
MONGOLIAN_TARGETS: Final = (CYRILLIC_MONGOLIAN, TRADITIONAL_MONGOLIAN)
# `normalize` depends on the language only for `mn_Mong`; the script hint chooses between it
# and any other language when a text comes without one.
_NO_HINT_LANGUAGE: Final = "eng_Latn"
# The characters `str.splitlines` splits at: LF, CR, VT, FF, FS, GS, RS, NEL, LS, PS.
LINE_BREAKS: Final = re.compile(
    "["
    + re.escape(
        "".join(
            chr(code)
            for code in (0x0A, 0x0D, 0x0B, 0x0C, 0x1C, 0x1D, 0x1E, 0x85, 0x2028, 0x2029)
        )
    )
    + "]+"
)
_SHOWN: Final = 60

# The key of a suppression list: the scripts it forbids (None for the control entries alone),
# whether the document separator is let through, and whether scripts are listed at all.
ListKey = tuple[frozenset[Script] | None, bool, bool]


class HypothesesLike(Protocol):
    """What the engine reads of one source's result (`ctranslate2.TranslationResult`)."""

    @property
    def hypotheses(self) -> Sequence[Sequence[str]]: ...


class Ct2Like(Protocol):
    """What the engine asks of `ctranslate2.Translator`."""

    def translate_batch(
        self, source: list[list[str]], **options: Any
    ) -> Sequence[HypothesesLike]: ...


class ProcessorLike(script_mask.ProcessorLike, Protocol):
    """What the engine asks of a SentencePiece processor (`SentencePieceProcessor` has it)."""

    def encode(self, text: str, out_type: type = str) -> list[str]: ...

    def decode(self, pieces: list[str]) -> str: ...


@dataclass(frozen=True, slots=True)
class Route:
    """Where a row is decoded: `reported_tgt` is what the caller asked for and sees,
    `decode_tgt` the language whose tag the model is given (the Cyrillic one when the
    traditional script is made by the script converter, `convert`)."""

    reported_tgt: str
    decode_tgt: str
    tag: str
    convert: bool
    zero_shot: bool


@dataclass(frozen=True, slots=True)
class Job:
    """One source for the library: `label` names it in an error (`row 3`, `window 0`)."""

    label: str
    tokens: list[str]
    prefix: list[str] | None
    route: Route
    allow_separator: bool = False


@dataclass(frozen=True, slots=True)
class Prepared:
    """A row after the checks that need no model: not sent when `status` is set."""

    route: Route
    src: str
    status: Status | None
    job: Job | None


@dataclass(frozen=True, slots=True)
class Parsed:
    """What the hypotheses of one source came to: the best text, the other texts, and whether
    the best was finished (None: no hypothesis survived)."""

    text: str
    alternatives: tuple[str, ...]
    complete: bool | None

    @property
    def status(self) -> Status:
        if self.complete is None:
            return "no_output"
        return "ok" if self.complete else "truncated"


@dataclass(frozen=True, slots=True)
class Outcome:
    """A parsed source and whether it was decoded again with the script list."""

    parsed: Parsed
    retried: bool


def new_tally() -> dict[str, int]:
    """Counters at zero: what one call adds to the engine's when it succeeds."""
    return dict.fromkeys(ENGINE_COUNTERS, 0)


def shown(value: object) -> str:
    """`value` as a message may show it: its `repr`, cut short."""
    return repr(value)[:_SHOWN]


def require_encodable(text: str, label: str) -> None:
    """A lone surrogate cannot be encoded as UTF-8, and SentencePiece raises a bare
    `TypeError` on it. The error names the text by its place and never quotes it.

    Raises:
        DataError: `text` holds a lone surrogate.
    """
    try:
        text.encode()
    except UnicodeEncodeError as error:
        raise DataError(
            f"{label} cannot be encoded as UTF-8: a lone surrogate at position "
            f"{error.start}"
        ) from error


def has_alnum(text: str) -> bool:
    """Whether `text` has a letter or a digit: a text without one is empty to the engine."""
    return any(character.isalnum() for character in text)


def source_language(text: str, src: str | None) -> tuple[str, str]:
    """What a result says about the source (the language given, else the script hint) and
    the language the text is normalised for."""
    if src is not None:
        return src, src
    hint = script_mask.script_hint(text)
    return hint, TRADITIONAL_MONGOLIAN if hint == "mn_Mong" else _NO_HINT_LANGUAGE


def selected_target(tgt: str, script: str | None) -> str:
    """The target a request names once its `script` is applied: `Mong` selects `mn_Mong`
    and `Cyrl` `khk_Cyrl`, and only a Mongolian target takes a script.

    Raises:
        ConfigError: `script` is neither `Cyrl` nor `Mong`, or `tgt` is not Mongolian.
    """
    if script is None:
        return tgt
    if script not in SCRIPT_NAMES:
        raise ConfigError(f"script must be 'Cyrl' or 'Mong', got {shown(script)}")
    if tgt not in MONGOLIAN_TARGETS:
        raise ConfigError(
            "a script applies to a Mongolian target only (khk_Cyrl or mn_Mong), not to "
            f"{shown(tgt)}"
        )
    return TRADITIONAL_MONGOLIAN if script == "Mong" else CYRILLIC_MONGOLIAN


def without_separators(text: str) -> str:
    """`text` with the window separator read as a space: its letters are no script's."""
    return text.replace(SEGMENT_SEPARATOR, " ")


def purity(parsed: Parsed, job: Job) -> float:
    """The share of the letters of the best text in the scripts of the language it was
    decoded for; -1 for no text, so that any text is purer."""
    if not parsed.text:
        return -1.0
    return script_purity(
        without_separators(parsed.text), allowed_scripts(job.route.decode_tgt)
    )


def is_impure(parsed: Parsed, job: Job) -> bool:
    """Whether there is a text and its purity is below the floor."""
    return bool(parsed.text) and purity(parsed, job) < PURITY_FLOOR


def raw_translation(item: Prepared, outcome: Outcome | None) -> Translation:
    """The translation of a prepared row before its conversion and purity: the status the
    row was given when it was not sent, else what its hypotheses came to."""
    text = ""
    alternatives: tuple[str, ...] = ()
    retried = False
    if outcome is None:
        assert item.status is not None
        status = item.status
    else:
        text, alternatives = outcome.parsed.text, outcome.parsed.alternatives
        status, retried = outcome.parsed.status, outcome.retried
    return Translation(
        text=text,
        status=status,
        alternatives=alternatives,
        src=item.src,
        tgt=item.route.reported_tgt,
        script_purity=None,
        retried=retried,
        zero_shot=item.route.zero_shot,
    )


def model_info(export: VerifiedExport, device: str, compute_type: str) -> ModelInfo:
    """What a loaded engine says about `export`, running on `device` with `compute_type`."""
    record = export.record
    return ModelInfo(
        name=record.name,
        quantization=record.quantization,
        device=device,
        compute_type=compute_type,
        placement=record.tag_placement,
        directions=record.directions,
        targets=record.targets,
        document_level=record.document_level,
        model_sha256=export.sha256[CT2_FILES[0]],
        manifest_sha256=export.manifest_sha256,
    )


def refusal(target: str, record: ModelRecord) -> str:
    """Why the model cannot write `target`: it is none of its targets, and for the traditional
    script no script converter that could make it from the Cyrillic one is attached."""
    targets = ", ".join(record.targets)
    if target == TRADITIONAL_MONGOLIAN:
        return (
            f"the model does not translate into {target} (its targets are {targets}) and "
            f"no script converter is attached that could write it from {CYRILLIC_MONGOLIAN}"
        )
    return (
        f"the model does not translate into {shown(target)}: its targets are {targets}"
    )


def require_document_model(record: ModelRecord, src: str | None) -> None:
    """Documents need a model that decodes `<doc>` windows with the tag in the source, and the
    language of the document to cut it into sentences.

    Raises:
        ConfigError: the record says otherwise, or `src` is None.
    """
    if not record.document_level:
        raise ConfigError(
            "the model does not translate documents: its record says document_level is false"
        )
    if record.tag_placement != "src":
        raise ConfigError(
            "documents need a model whose direction tag goes in the source (placement src), "
            f"this one's is {record.tag_placement}"
        )
    if src is None:
        raise ConfigError(
            "a document needs its source language (src) to be cut into sentences"
        )


def document_sentence(
    sentence: str,
    translated: str,
    src: str,
    route: Route,
    decoded: dict[str, Translation],
) -> Translation:
    """The raw translation of one sentence of a document: what the sentence translator gave
    when `translated` is its text (`decoded` maps the sentences it was given), else a segment
    of an aligned window (`ok`), else nothing (a sentence with no text)."""
    known = decoded.get(one_line(sentence))
    if known is not None and known.text == translated:
        return known
    return Translation(
        text=translated,
        status="ok" if translated else "empty",
        alternatives=(),
        src=src,
        tgt=route.reported_tgt,
        script_purity=None,
        retried=False,
        zero_shot=route.zero_shot,
    )
