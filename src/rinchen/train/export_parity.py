"""Strict parity of an export with the checkpoint it was made from (D11): the same sources go
through `eole predict` and through CTranslate2, both greedy and on the CPU, and the report says
how many translations are the same token for token and how far the two models' log-probabilities
of Eole's own translations are from each other.

The Eole run asks for `with_score` and `length_penalty: none` (the default, `avg`, divides the
score by the length), `beam_size: 1`, `transforms: []` and `max_length: 128` with no length
ratio. Its score is then the sum of the log-probabilities of the generated pieces, the end
token included. CTranslate2 answers the same sources with the same limit, and `score_batch`
scores Eole's pieces: it appends the end token itself, and with a forced target prefix it counts
the log-probability of the prefix, which Eole's score leaves out, so for the placement `tgt` the
first log-probability is dropped (measured in the D11 spike: 0.48 per sentence against 4e-6).
The delta of a sentence is `|sum(log_probs) - eole_score| / len(log_probs)` of the
log-probabilities that were compared; the report holds the mean and the maximum over the
sentences.

`exact` compares the pieces without the end token and, for `tgt`, without the forced tag, which
both start with. The thresholds are the master plan's: an exact share of 0.95 and, for a
float32 export only, a mean delta of 1e-3 per token. A quantised export reports both numbers and
needs the exact share alone: int8 measured 1.45e-3 and int8_float16 1.69e-3 in the spike, and
its quality is gated by chrF instead. A greedy hypothesis cut at `max_length` has no end token,
so the end token `score_batch` appends is no part of Eole's score for it: such a sentence adds
a spurious delta (none occurred on the toy).

The checkpoint must be the one the export record names: the SHA-256 of its weights is the
record's, else `ConfigError` before anything starts. Eole runs as a subprocess with
`CUDA_VISIBLE_DEVICES=""` and `gpu_ranks: []`, CTranslate2 on `device="cpu"` with the compute
type the file stores (`default`, never `auto`); no GPU is used or claimed.
"""

from __future__ import annotations

import math
import statistics
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Protocol

from rinchen.config.ablation import TagPlacement, require_placement
from rinchen.errors import ConfigError, DataError, UpstreamError
from rinchen.export.record import SPM_FILE, ModelRecord
from rinchen.export.verified import VerifiedExport, verified_bytes
from rinchen.infer.source_tokens import ct2_source_tokens
from rinchen.manifest.stage import sha256_of
from rinchen.train.checkpoints import WEIGHTS_FILE
from rinchen.train.eole_cfg import (
    dump_config,
    reject_dollar,
    reject_glob_characters,
)
from rinchen.train.eole_run import (
    PredictFiles,
    eole_command,
    predict_files,
    split_lines,
    training_environment,
    write_lines,
)
from rinchen.utils.logged_process import run_logged

MIN_EXACT_SHARE: Final = 0.95
MAX_MEAN_DELTA: Final = 1e-3
# The export whose log-probabilities are held to `MAX_MEAN_DELTA`.
GATED_QUANTIZATION: Final = "float32"
# How many mismatching translations a report keeps.
MISMATCHES_KEPT: Final = 10
# The files of the Eole run are `parity.log`, `parity.predict.yaml`, `parity.tokens.*`.
PREDICT_STEM: Final = "parity"
_MAX_LENGTH: Final = 128
_BATCH_SIZE: Final = 64
# `tokens<TAB>score<TAB>estimated quality`: the last field is not used.
_SCORED_FIELDS: Final = 3
_SHOWN: Final = 12


@dataclass(frozen=True, slots=True)
class ParitySource:
    """One text to translate into `tgt_lang`, a language code of the export's record."""

    text: str
    tgt_lang: str


@dataclass(frozen=True, slots=True)
class ParityReport:
    """What a strict parity run measured over `sentences` sources.

    `exact` of them are translated by the same pieces by Eole and by CTranslate2
    (`exact_share` is `exact / sentences`); `mean_abs_delta` and `max_abs_delta` are the mean
    and the largest difference of the two models' log-probability of Eole's translation, per
    token. `gated` says whether the mean delta is held to its bound (a float32 export).
    `mismatches` are the first `MISMATCHES_KEPT` translations that differ, as the pair of Eole's
    pieces and CTranslate2's, each joined by U+0020, without the forced tag."""

    sentences: int
    exact: int
    exact_share: float
    mean_abs_delta: float
    max_abs_delta: float
    quantization: str
    placement: TagPlacement
    gated: bool
    mismatches: tuple[tuple[str, str], ...]

    @property
    def passes(self) -> bool:
        """The exact share reaches `MIN_EXACT_SHARE` and, when `gated`, the mean delta is at
        most `MAX_MEAN_DELTA`."""
        if self.exact_share < MIN_EXACT_SHARE:
            return False
        return not self.gated or self.mean_abs_delta <= MAX_MEAN_DELTA

    def to_dict(self) -> dict[str, Any]:
        """The fields as JSON-ready values (the mismatches as lists of two lines)."""
        return {
            "sentences": self.sentences,
            "exact": self.exact,
            "exact_share": self.exact_share,
            "mean_abs_delta": self.mean_abs_delta,
            "max_abs_delta": self.max_abs_delta,
            "quantization": self.quantization,
            "placement": self.placement,
            "gated": self.gated,
            "mismatches": [list(pair) for pair in self.mismatches],
        }


class TokenProcessor(Protocol):
    """What the parity asks of a SentencePiece processor (`SentencePieceProcessor` has it)."""

    def encode(self, text: str, /, return_type: type[str] = str) -> list[str]: ...


class _Hypotheses(Protocol):
    @property
    def hypotheses(self) -> Sequence[Sequence[str]]: ...


class _Scoring(Protocol):
    @property
    def log_probs(self) -> Sequence[float]: ...


class ParityTranslator(Protocol):
    """What the parity asks of `ctranslate2.Translator`."""

    def translate_batch(
        self, source: list[list[str]], **options: Any
    ) -> Sequence[_Hypotheses]: ...

    def score_batch(
        self, source: list[list[str]], target: list[list[str]], **options: Any
    ) -> Sequence[_Scoring]: ...


def parse_scored_line(line: str) -> tuple[list[str], float]:
    """The pieces and the score of a line `eole predict` wrote with `with_score`:
    `tokens<TAB>score<TAB>estimated quality`. The pieces are split at U+0020 only (a piece may
    hold U+202F) and no pieces are an empty hypothesis; the last field is not used.

    Raises:
        UpstreamError: the line has another number of fields, or its score is no finite
            number; the message never quotes the line.
    """
    fields = line.split("\t")
    if len(fields) != _SCORED_FIELDS:
        raise UpstreamError(
            f"an Eole output line has {len(fields)} tab-separated fields, not "
            f"{_SCORED_FIELDS} (tokens, score and estimated quality)"
        )
    tokens, score_text, _ = fields
    try:
        score = float(score_text)
    except ValueError:
        raise UpstreamError(
            "an Eole output line has a score that is no number"
        ) from None
    if not math.isfinite(score):
        raise UpstreamError("an Eole output line has a score that is not finite")
    return (tokens.split(" ") if tokens else []), score


def parity_config(
    checkpoint: Path, files: PredictFiles, *, placement: TagPlacement
) -> dict[str, Any]:
    """The config of the `eole predict` run, as the dict `dump_config` writes: greedy, no
    transform, the end of a hypothesis at 128 pieces and no length ratio, the score of every
    line written (`with_score`, `length_penalty: none`), no GPU. With the placement `tgt` the
    token file of the tags is the forced prefix of the output (`tgt_file_prefix`).

    Raises:
        ValueError: `placement` is neither `src` nor `tgt`.
        ConfigError: a path holds `$`, which Eole would expand as a variable, or the
            checkpoint's holds `[`, `]`, `*` or `?`, which it reads as a glob pattern.
    """
    require_placement(placement)
    reject_dollar("checkpoint directory", checkpoint)
    reject_dollar("output directory", files.token_output.parent)
    reject_glob_characters("checkpoint directory", checkpoint)
    forced_tags = (
        {"tgt": files.token_target.as_posix(), "tgt_file_prefix": True}
        if placement == "tgt"
        else {}
    )
    return {
        "model_path": checkpoint.as_posix(),
        "src": files.token_source.as_posix(),
        **forced_tags,
        "output": files.token_output.as_posix(),
        "beam_size": 1,
        "batch_size": _BATCH_SIZE,
        "batch_type": "sents",
        "gpu_ranks": [],
        "world_size": 1,
        "transforms": [],
        "max_length": _MAX_LENGTH,
        "max_length_ratio": 0,
        "with_score": True,
        "length_penalty": "none",
    }


def require_checkpoint_of_export(
    record: ModelRecord,
    weights_sha256: str,
    *,
    config_sha256: str | None = None,
    vocab_sha256: str | None = None,
) -> None:
    """Raise ConfigError unless `weights_sha256` is the SHA-256 of the weights the export
    `record` says it was made from, and, when they are given, `config_sha256` that of its
    `config.json` and `vocab_sha256` that of its `vocab.json`: weights alone leave room for
    another architecture or vocabulary (a parity of two different models says nothing)."""
    checkpoint = record.checkpoint
    for subject, origin, got, expected in (
        ("weights have", "weights", weights_sha256, checkpoint.weights_sha256),
        ("config.json has", "a config.json", config_sha256, checkpoint.config_sha256),
        ("vocab.json has", "a vocab.json", vocab_sha256, checkpoint.vocab_sha256),
    ):
        if got is not None and got != expected:
            raise ConfigError(
                f"the checkpoint's {subject} SHA-256 {got[:_SHOWN]}... but the export "
                f"{record.name} was made from {origin} with {expected[:_SHOWN]}...: "
                "a parity of two different models says nothing"
            )


def strict_parity(
    checkpoint: Path,
    export: VerifiedExport,
    sources: Sequence[ParitySource],
    *,
    out_dir: Path,
    threads: int = 2,
    timeout: float | None = None,
    engine: ParityTranslator | None = None,
    processor: TokenProcessor | None = None,
    weights_sha256: str | None = None,
) -> ParityReport:
    """Translate `sources` with Eole from the checkpoint in `checkpoint` and with CTranslate2
    from `export`, and report how far they agree (see the module's text).

    Each text is cut into pieces with the export's own SentencePiece model and given the tag of
    its target language where the record's placement puts it (`ct2_source_tokens`); both
    systems get the same token lines. Eole's files go to `out_dir` (`parity.*`), which is
    created. `threads` are those of Eole's OpenMP and of CTranslate2's intra-op pool;
    `timeout` (seconds) bounds the Eole command.

    `engine` and `processor` are the seams of a test: a CTranslate2 translator and a
    SentencePiece processor; by default the real ones, the first on the CPU with the compute
    type the file stores. `weights_sha256` is the digest of the checkpoint's weights a caller
    took already; the file is hashed here when it is None.

    Raises:
        ValueError: no source or `threads` below 1.
        ConfigError: a source asks for a language the export has no tag for; the checkpoint's
            weights are not the export's (before Eole starts); a path holds `$`, or the
            checkpoint's a glob character; a file of the export changed since it was
            verified; the library cannot run the export.
        DataError: the weights or the library's model cannot be read; a text has no piece
            (it names the source by its number); the output of Eole cannot be read.
        UpstreamError: Eole or CTranslate2 fails, or answers with another number of results
            or with a line of another shape, or a hypothesis does not start with the forced
            tag; messages never quote a text.
    """
    if not sources:
        raise ValueError("no source to compare")
    if threads < 1:
        raise ValueError(f"threads must be at least 1, got {threads}")
    record = export.record
    tags = [_tag_of(record, source) for source in sources]
    files = predict_files(out_dir, PREDICT_STEM)
    cfg = parity_config(checkpoint, files, placement=record.tag_placement)
    digest = _weights_digest(checkpoint) if weights_sha256 is None else weights_sha256
    require_checkpoint_of_export(record, digest)
    pieces = _pieces_of(
        sources, _load_processor(export) if processor is None else processor
    )
    lines, prefixes = _token_lines(record.tag_placement, tags, pieces)
    translator = _load_translator(export, threads) if engine is None else engine
    eole = _eole_answers(cfg, files, lines, prefixes, threads=threads, timeout=timeout)
    eole_tokens = [tokens for tokens, _ in eole]
    return _report(
        record,
        _without_prefixes("eole predict", eole_tokens, prefixes),
        _without_prefixes(
            "CTranslate2", _hypotheses(translator, lines, prefixes), prefixes
        ),
        [score for _, score in eole],
        _log_probs(translator, lines, eole_tokens, dropped=prefixes is not None),
    )


def _tag_of(record: ModelRecord, source: ParitySource) -> str:
    tag = record.tags.get(source.tgt_lang)
    if tag is None:
        raise ConfigError(
            f"the export {record.name} has no direction tag for the target language "
            f"{source.tgt_lang}: it can write {', '.join(record.targets)}"
        )
    return tag


def _weights_digest(checkpoint: Path) -> str:
    weights = checkpoint / WEIGHTS_FILE
    if not weights.is_file():
        raise ConfigError(f"checkpoint weights not found: {weights}")
    try:
        return sha256_of(weights)
    except OSError as error:
        raise DataError(
            f"cannot read the checkpoint weights {weights}: {error.strerror}"
        ) from error


def _load_processor(export: VerifiedExport) -> TokenProcessor:
    """The SentencePiece model, parsed from the verified bytes of the export's copy."""
    content = verified_bytes(export, SPM_FILE)
    # Imported here so that importing this module never loads the library.
    import sentencepiece

    try:
        return sentencepiece.SentencePieceProcessor(model_proto=content)
    except (OSError, RuntimeError, TypeError) as error:
        raise ConfigError(
            f"the SentencePiece model {export.directory / SPM_FILE} cannot be loaded"
        ) from error


def _load_translator(export: VerifiedExport, threads: int) -> ParityTranslator:
    # Imported here so that importing this module never loads the library.
    import ctranslate2

    try:
        return ctranslate2.Translator(
            str(export.directory),
            device="cpu",
            compute_type="default",
            intra_threads=threads,
        )
    except ValueError as error:
        raise ConfigError(
            f"CTranslate2 cannot run the export {export.directory} on the cpu with the "
            f"compute type it stores: {error}"
        ) from error
    except (OSError, RuntimeError) as error:
        raise DataError(
            f"cannot load the CTranslate2 model in {export.directory}: {error}"
        ) from error


def _pieces_of(
    sources: Sequence[ParitySource], processor: TokenProcessor
) -> list[list[str]]:
    pieces: list[list[str]] = []
    for number, source in enumerate(sources, start=1):
        each = processor.encode(source.text, return_type=str)
        if not each:
            # Eole drops such a line without a word, and there would be nothing to compare.
            raise DataError(f"source {number} has no piece: its text is empty or blank")
        pieces.append(each)
    return pieces


def _token_lines(
    placement: TagPlacement, tags: Sequence[str], pieces: Sequence[Sequence[str]]
) -> tuple[list[list[str]], list[list[str]] | None]:
    """The source line of each text, and for the placement `tgt` its forced prefix."""
    lines: list[list[str]] = []
    prefixes: list[list[str]] = []
    for tag, each in zip(tags, pieces, strict=True):
        line, prefix = ct2_source_tokens(placement, tag, each)
        lines.append(line)
        if prefix is not None:
            prefixes.append(prefix)
    return lines, prefixes if placement == "tgt" else None


def _eole_answers(
    cfg: dict[str, Any],
    files: PredictFiles,
    lines: Sequence[Sequence[str]],
    prefixes: Sequence[Sequence[str]] | None,
    *,
    threads: int,
    timeout: float | None,
) -> list[tuple[list[str], float]]:
    """Run `eole predict` once over `lines` and return the pieces and the score of each."""
    command = [eole_command(), "predict", "-config", str(files.config)]
    env = training_environment(threads, hide_gpu=True)
    files.config.parent.mkdir(parents=True, exist_ok=True)
    files.token_output.unlink(missing_ok=True)
    write_lines(files.token_source, [" ".join(line) for line in lines])
    if prefixes is not None:
        write_lines(files.token_target, [" ".join(prefix) for prefix in prefixes])
    dump_config(cfg, files.config)
    run_logged(
        command, env=env, log_path=files.log, what="eole predict", timeout=timeout
    )
    if not files.token_output.is_file():
        raise UpstreamError(
            f"eole predict exited with code 0 but wrote no output file {files.token_output}"
        )
    try:
        text = files.token_output.read_text(encoding="utf-8", newline="\n")
    except (OSError, UnicodeDecodeError) as error:
        # A read failure is the data's, not a failure to write the result: say so.
        reason = error.strerror if isinstance(error, OSError) else "not valid UTF-8"
        raise DataError(
            f"cannot read the output of eole predict {files.token_output}: {reason}"
        ) from error
    written = split_lines(text)
    if len(written) != len(lines):
        raise UpstreamError(
            f"eole predict wrote {len(written)} lines for {len(lines)} source lines: "
            f"{files.token_output}"
        )
    answers: list[tuple[list[str], float]] = []
    for number, line in enumerate(written, start=1):
        try:
            answers.append(parse_scored_line(line))
        except UpstreamError as error:
            raise UpstreamError(
                f"line {number} of {files.token_output}: {error}"
            ) from error
    return answers


def _hypotheses(
    translator: ParityTranslator,
    lines: list[list[str]],
    prefixes: list[list[str]] | None,
) -> list[list[str]]:
    """CTranslate2's greedy hypothesis for each line, without the end token."""
    options: dict[str, Any] = {
        "beam_size": 1,
        "max_decoding_length": _MAX_LENGTH,
        "return_end_token": False,
        # Eole reads the whole line, so the library must as well.
        "max_input_length": 0,
    }
    if prefixes is not None:
        options["target_prefix"] = prefixes
    results = _answer_of(
        lambda: translator.translate_batch(lines, **options), "translate", len(lines)
    )
    if len(results) != len(lines):
        raise UpstreamError(
            f"CTranslate2 returned {len(results)} results for {len(lines)} sources"
        )
    return [
        list(result.hypotheses[0]) if result.hypotheses else [] for result in results
    ]


def _log_probs(
    translator: ParityTranslator,
    lines: list[list[str]],
    eole_tokens: list[list[str]],
    *,
    dropped: bool,
) -> list[Sequence[float]]:
    """The log-probability CTranslate2 gives each piece of Eole's translation, the end token
    included (the library appends it) and, with a forced prefix, the prefix's own left out."""
    scored = _answer_of(
        lambda: translator.score_batch(lines, eole_tokens, max_input_length=0),
        "score",
        len(lines),
    )
    if len(scored) != len(lines):
        raise UpstreamError(
            f"CTranslate2 scored {len(scored)} of the {len(lines)} translations"
        )
    compared = [each.log_probs[1:] if dropped else each.log_probs for each in scored]
    if not all(compared):
        raise UpstreamError(
            "CTranslate2 scored a translation without a log-probability"
        )
    return compared


def _answer_of(
    call: Callable[[], Sequence[Any]], action: str, count: int
) -> Sequence[Any]:
    try:
        return call()
    except (RuntimeError, ValueError) as error:
        # The library's message stays in the chained cause: it is not ours to vouch for.
        raise UpstreamError(
            f"CTranslate2 failed to {action} {count} sources ({type(error).__name__})"
        ) from error


def _without_prefixes(
    who: str, tokens: list[list[str]], prefixes: list[list[str]] | None
) -> list[list[str]]:
    """The pieces of each line without the forced prefix it must start with (`prefixes`)."""
    if prefixes is None:
        return tokens
    stripped: list[list[str]] = []
    for number, (line, prefix) in enumerate(
        zip(tokens, prefixes, strict=True), start=1
    ):
        if line[: len(prefix)] != prefix:
            raise UpstreamError(
                f"{who} did not start line {number} with the forced tag {' '.join(prefix)}, "
                "the forced prefix of the placement tgt"
            )
        stripped.append(line[len(prefix) :])
    return stripped


def _report(
    record: ModelRecord,
    eole_pieces: Sequence[Sequence[str]],
    ct2_pieces: Sequence[Sequence[str]],
    scores: Sequence[float],
    log_probs: Sequence[Sequence[float]],
) -> ParityReport:
    deltas = [
        abs(math.fsum(each) - score) / len(each)
        for each, score in zip(log_probs, scores, strict=True)
    ]
    different = [
        (" ".join(a), " ".join(b))
        for a, b in zip(eole_pieces, ct2_pieces, strict=True)
        if a != b
    ]
    return ParityReport(
        sentences=len(deltas),
        exact=len(deltas) - len(different),
        exact_share=(len(deltas) - len(different)) / len(deltas),
        mean_abs_delta=statistics.fmean(deltas),
        max_abs_delta=max(deltas),
        quantization=record.quantization,
        placement=record.tag_placement,
        gated=record.quantization == GATED_QUANTIZATION,
        mismatches=tuple(different[:MISMATCHES_KEPT]),
    )
