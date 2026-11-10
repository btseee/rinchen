"""The engine (D11): a verified export of ours, loaded once, translating batches of texts.

`Engine.open` has `verify_export` vouch for a directory (every file hashed once, before it is
parsed) and loads CTranslate2 on it. `translate` takes rows (a text and the languages it is
asked for) and answers each with a `Translation`, the same number in the same order;
`translate_documents` takes documents and answers each with a `DocumentResult`.

What CTranslate2 does silently, the engine does not (measured in the D10 and D11 spikes):

- the direction tag goes where the record says (in the source, or as a forced prefix of the
  target, which the hypotheses START with and lose here); the tag of every target of the
  record is checked at load (a vocabulary entry and one SentencePiece piece);
- a hypothesis counts as finished only with the end token: a cut text is `truncated` and keeps
  its partial text; a source of more than `max_source_pieces` is `too_long` and a text with no
  letter, digit or piece is `empty`, neither is sent;
- a hypothesis with a piece no translation holds is dropped and counted (`no_output` when none
  is left); a line break in the output becomes a space; the script is masked as
  `DecodeOptions.script_mask` says;
- a document model (the record's `document_level`) writes `<sep>` after a sentence decoded
  alone, because it was continued on windows of 2 to 4 sentences only: its hypothesis is cut at
  the first `<sep>` and the head is the translation, finished (the separator is the model
  saying so); one that starts with `<sep>` is dropped. A window lets `<sep>` through instead.

A request is all-or-nothing in its errors (an unknown target, a `script` for a target that is not
Mongolian, the same source and target, a lone surrogate raise before anything is sent), a
text-level problem is a status, a failed call adds nothing to the counters, the arguments are
never mutated and no message or log names a text.

Thread safety: the library's `translate_batch` is thread-safe and SentencePiece too; the only
mutable state is the counters (one lock, one update per call) and the suppression lists (each
built once under a lock); calls into the script converter, which is not thread-safe, take a
lock of their own. No GPU was used and none is claimed: `cuda` is an operator's explicit
choice, never run; `auto` is refused. `ctranslate2` and `sentencepiece` load inside `open` only.
"""

from __future__ import annotations

import os
import threading
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any, Final

from rinchen.config.languages import known_language_codes
from rinchen.errors import ConfigError, DataError, UpstreamError
from rinchen.export.record import CT2_FILES, SPM_FILE
from rinchen.export.verified import (
    VerifiedExport,
    end_token,
    verified_bytes,
    verify_export,
    vocabulary_entries,
)
from rinchen.infer.doc_chunker import (
    DOC_MARKER,
    SEGMENT_SEPARATOR,
    translate_document,
)
from rinchen.infer.engine_types import (
    CYRILLIC_MONGOLIAN,
    ENGINE_COUNTERS,
    LINE_BREAKS,
    TRADITIONAL_MONGOLIAN,
    Ct2Like,
    DecodeOptions,
    DocumentResult,
    HypothesesLike,
    Job,
    ListKey,
    LoadOptions,
    ModelInfo,
    Outcome,
    Parsed,
    Prepared,
    ProcessorLike,
    Route,
    Row,
    ScriptConverterLike,
    ScriptName,
    Translation,
    document_sentence,
    has_alnum,
    is_impure,
    model_info,
    new_tally,
    purity,
    raw_translation,
    refusal,
    require_document_model,
    require_encodable,
    reserved_symbol_error,
    selected_target,
    shown,
    source_language,
    without_separators,
)
from rinchen.infer.script_mask import (
    PURITY_FLOOR,
    allowed_scripts,
    forbidden_scripts,
    script_purity,
    suppression_list,
)
from rinchen.infer.source_tokens import (
    ct2_source_tokens,
    reserved_pieces,
    source_symbols,
)
from rinchen.split.rules import split
from rinchen.text.normalize import normalize
from rinchen.utils import logging as rinchen_logging

_VOCABULARY_FILE: Final = CT2_FILES[1]
# Languages written without a space between sentences.
_UNSPACED_TARGETS: Final = ("zho_Hans", "jpn_Jpan")
_WARMUP_TEXT: Final = "Hello"


class Engine:
    """A loaded model: `Engine.open` is how one is made (see the module's text)."""

    def __init__(
        self,
        *,
        export: VerifiedExport,
        translator: Ct2Like,
        processor: ProcessorLike,
        vocabulary: tuple[str, ...],
        end: str,
        load: LoadOptions,
        script_converter: ScriptConverterLike | None,
    ) -> None:
        self._record = export.record
        self._placement = export.record.tag_placement
        self._ct2 = translator
        self._processor = processor
        self._vocabulary = vocabulary
        self._end = end
        self._converter = script_converter
        # D5's converter caches words in the instance and is not thread-safe: one call at a time.
        self._converter_lock = threading.Lock()
        self._reserved = reserved_pieces(processor, end, vocabulary)
        self._reserved_in_windows = reserved_pieces(
            processor, end, vocabulary, allow=(SEGMENT_SEPARATOR,)
        )
        # What a request's text must not spell (see `source_symbols`), in the order of the
        # vocabulary for a text and as a set for the pieces of one.
        self._source_symbols = source_symbols()
        self._source_symbol_set = frozenset(self._source_symbols)
        self._lists: dict[ListKey, list[list[str]]] = {}
        self._lists_lock = threading.Lock()
        self._counters = dict.fromkeys(ENGINE_COUNTERS, 0)
        self._counters_lock = threading.Lock()
        self._closed = False
        self._info = model_info(
            export, load.device, str(getattr(translator, "compute_type", "unknown"))
        )

    @classmethod
    def open(
        cls,
        directory: str | os.PathLike[str],
        load: LoadOptions | None = None,
        *,
        script_converter: ScriptConverterLike | None = None,
        translator: Ct2Like | None = None,
        processor: ProcessorLike | None = None,
    ) -> Engine:
        """The engine over the export in `directory`: the export is verified and its files
        read through the verified bytes, every tag is checked, and only then is CTranslate2
        loaded with `load` (default: the CPU, one thread of each kind, what the file stores).
        `translator` and `processor` are the seams of the tests (the library is not imported
        with them); `script_converter` (D5's fits) lets a model without a `mn_Mong` target
        answer for the traditional script.

        Raises:
            ConfigError: the directory is not an export our manifest vouches for, a tag is
                not usable, or the SentencePiece model, the device or the type is refused.
            DataError: a file of the export or the library's model is unreadable.
        """
        load = LoadOptions() if load is None else load
        export = verify_export(Path(directory))
        vocabulary = vocabulary_entries(export)
        end = end_token(export)
        pieces = processor if processor is not None else _load_processor(export)
        _require_usable_tags(export, frozenset(vocabulary), pieces)
        library = translator if translator is not None else _load_ct2(export, load)
        engine = cls(
            export=export,
            translator=library,
            processor=pieces,
            vocabulary=vocabulary,
            end=end,
            load=load,
            script_converter=script_converter,
        )
        rinchen_logging.get("rinchen.infer.engine").info(
            "compute_type_effective",
            model=engine.info.name,
            quantization=engine.info.quantization,
            device=engine.info.device,
            compute_type=engine.info.compute_type,
        )
        return engine

    @property
    def info(self) -> ModelInfo:
        """What the engine knows about its model."""
        return self._info

    def counters(self) -> dict[str, int]:
        """A copy of the counters, one per name of `ENGINE_COUNTERS`, cumulative since the
        engine was opened. A call that raises adds nothing."""
        with self._counters_lock:
            return dict(self._counters)

    def close(self) -> None:
        """Unload the model. Calling it again does nothing; a closed engine translates
        nothing. The caller lets the calls in flight finish first."""
        with self._counters_lock:
            if self._closed:
                return
            self._closed = True
        unload = getattr(self._ct2, "unload_model", None)
        if callable(unload):
            unload()

    def warmup(self) -> None:
        """One fixed translation into the first target of the model, so that the first
        request pays no load. It is not counted.

        Raises:
            UpstreamError: the library fails.
        """
        self._require_open()
        options = DecodeOptions(beam_size=1, max_decoding_length=8, script_mask="off")
        row = Row(_WARMUP_TEXT, None, self._record.targets[0])
        self._translate_rows([row], options, new_tally())

    def translate(
        self, rows: Sequence[Row], options: DecodeOptions | None = None
    ) -> list[Translation]:
        """One `Translation` per row, in order (see the module's text for the statuses).

        Raises:
            ConfigError: a row asks for a target the model does not write (and no converter
                makes), a `script` for a non-Mongolian target, or the same source and target.
            DataError: a text holds a lone surrogate; the message names the row.
            UpstreamError: the library fails or answers wrongly; never quotes a text.
            TypeError: a row is not a `Row` or its text is not text.
        """
        self._require_open()
        tally = new_tally()
        translations = self._translate_rows(
            rows, DecodeOptions() if options is None else options, tally
        )
        self._commit(tally)
        return translations

    def translate_documents(
        self,
        documents: Sequence[str],
        src: str | None,
        tgt: str,
        options: DecodeOptions | None = None,
        *,
        script: ScriptName | None = None,
    ) -> list[DocumentResult]:
        """One `DocumentResult` per document, in order: cut into sentences (`src` is
        required), the sentences into windows of the D9 chunker, a window one call whose
        source is `[tag, <doc>, *pieces]`; a window that was cut or does not line up is
        translated sentence by sentence.

        Raises:
            ConfigError: the model is no document model of placement `src`, `src` is None, or
                the rules of `translate` for the languages.
            DataError: a document holds a lone surrogate or the literal `<sep>`; the message
                names the document and never quotes it.
            UpstreamError: as `translate`.
        """
        self._require_open()
        require_document_model(self._record, src)
        route = self._route(src, tgt, script)
        options = DecodeOptions() if options is None else options
        texts = self._normalised_documents(documents, src)
        tally = new_tally()
        results = [
            self._translate_document(text, src, tgt, script, route, options, tally)
            for text in texts
        ]
        self._commit(tally)
        return results

    # ---- the requests --------------------------------------------------------------------

    def _require_open(self) -> None:
        if self._closed:
            raise ConfigError("the engine is closed: open it again to translate")

    def _commit(self, tally: dict[str, int]) -> None:
        with self._counters_lock:
            for name, count in tally.items():
                self._counters[name] += count

    def _translate_rows(
        self, rows: Sequence[Row], options: DecodeOptions, tally: dict[str, int]
    ) -> list[Translation]:
        return [
            self._seal(translation, route, tally)
            for route, translation in self._raw_rows(rows, options, tally)
        ]

    def _raw_rows(
        self, rows: Sequence[Row], options: DecodeOptions, tally: dict[str, int]
    ) -> list[tuple[Route, Translation]]:
        """The decoded rows before their script conversion and purity: one per row."""
        prepared = self._prepare(rows, options)
        tally["sources"] += len(prepared)
        jobs = [item.job for item in prepared if item.job is not None]
        outcomes = iter(self._decode_masked(jobs, options, tally))
        return [
            (
                item.route,
                raw_translation(item, None if item.job is None else next(outcomes)),
            )
            for item in prepared
        ]

    def _seal(
        self, translation: Translation, route: Route, tally: dict[str, int]
    ) -> Translation:
        """The translation as the caller gets it: converted to the traditional script when
        the route says so, with its purity over the scripts of the target asked for. Counts
        what the translation is (status, zero-shot pair, impure output)."""
        text, alternatives = translation.text, translation.alternatives
        if route.convert and text:
            converted = [self._convert(each, tally) for each in (text, *alternatives)]
            text, alternatives = converted[0], tuple(converted[1:])
        share = None
        if text:
            share = script_purity(
                without_separators(text), allowed_scripts(route.reported_tgt)
            )
            if share < PURITY_FLOOR:
                tally["script_impure"] += 1
        if translation.status != "ok":
            tally[translation.status] += 1
        if translation.zero_shot:
            tally["zero_shot"] += 1
        return replace(
            translation, text=text, alternatives=alternatives, script_purity=share
        )

    def _convert(self, text: str, tally: dict[str, int]) -> str:
        assert self._converter is not None
        with self._converter_lock:
            conversion = self._converter.convert(text, "Mong")
        tally["script_conversions"] += 1
        tally["converter_rule_fallbacks"] += conversion.rule_fallbacks
        tally["converter_invalid_outputs"] += conversion.invalid_outputs
        return conversion.text

    # ---- the languages -------------------------------------------------------------------

    def _route(self, src: str | None, tgt: str, script: str | None) -> Route:
        if src is not None and src not in known_language_codes():
            raise ConfigError(
                f"the source language {shown(src)} is not in the language registry"
            )
        selected = selected_target(tgt, script)
        if src == selected:
            raise ConfigError(
                f"the source and the target language are both {selected}: nothing to "
                "translate"
            )
        tags = self._record.tags
        decode = selected
        convert = False
        if selected not in tags:
            decode = CYRILLIC_MONGOLIAN
            convert = (
                selected == TRADITIONAL_MONGOLIAN
                and self._converter is not None
                and decode in tags
            )
            if not convert:
                raise ConfigError(refusal(selected, self._record))
        zero_shot = src is not None and f"{src}-{decode}" not in self._record.directions
        return Route(selected, decode, tags[decode], convert, zero_shot)

    # ---- the rows ------------------------------------------------------------------------

    def _prepare(self, rows: Sequence[Row], options: DecodeOptions) -> list[Prepared]:
        """Check every row (raising before anything is sent), then normalise and cut the
        texts; a text that cannot be sent is given its status here."""
        routes = []
        for index, row in enumerate(rows):
            if not isinstance(row, Row):
                raise TypeError(f"row {index} must be a Row, got {type(row).__name__}")
            if not isinstance(row.text, str):
                raise TypeError(
                    f"row {index}: the text must be str, got {type(row.text).__name__}"
                )
            routes.append(self._route(row.src, row.tgt, row.script))
            require_encodable(row.text, f"row {index}")
        return [
            self._prepared(f"row {index}", row, route, options)
            for index, (row, route) in enumerate(zip(rows, routes, strict=True))
        ]

    def _prepared(
        self, label: str, row: Row, route: Route, options: DecodeOptions
    ) -> Prepared:
        src, language = source_language(row.text, row.src)
        text = normalize(row.text, language)
        pieces = self._processor.encode(text, out_type=str) if has_alnum(text) else []
        if not pieces:
            return Prepared(route, src, "empty", None)
        for piece in pieces:
            if piece in self._source_symbol_set:
                raise reserved_symbol_error(label, piece)
        if len(pieces) > options.max_source_pieces:
            return Prepared(route, src, "too_long", None)
        tokens, prefix = ct2_source_tokens(self._placement, route.tag, pieces)
        job = Job(
            label,
            tokens,
            prefix,
            route,
            cut_at_separator=self._record.document_level,
        )
        return Prepared(route, src, None, job)

    # ---- decoding ------------------------------------------------------------------------

    def _decode_masked(
        self, jobs: Sequence[Job], options: DecodeOptions, tally: dict[str, int]
    ) -> list[Outcome]:
        """Decode `jobs` as the script mode says: `always` with the script list, `off` and
        `retry` without it, and `retry` decodes an impure output again with it, replacing
        the output only when the second one is purer."""
        mode = options.script_mask
        first = self._decode_pass(jobs, options, mode == "always", tally)
        outcomes = [Outcome(parsed, False) for parsed in first]
        if mode != "retry":
            return outcomes
        impure = [
            position
            for position, (job, parsed) in enumerate(zip(jobs, first, strict=True))
            if is_impure(parsed, job)
        ]
        if not impure:
            return outcomes
        tally["script_retries"] += len(impure)
        again = self._decode_pass([jobs[p] for p in impure], options, True, tally)
        for position, parsed in zip(impure, again, strict=True):
            job = jobs[position]
            if purity(parsed, job) > purity(first[position], job):
                tally["script_retry_improved"] += 1
                outcomes[position] = Outcome(parsed, True)
            else:
                outcomes[position] = Outcome(first[position], True)
        return outcomes

    def _decode_pass(
        self,
        jobs: Sequence[Job],
        options: DecodeOptions,
        scripts: bool,
        tally: dict[str, int],
    ) -> list[Parsed]:
        """One `translate_batch` per distinct suppression list, the jobs of a list in the
        order they were given; the answers back in the order of `jobs`."""
        groups: dict[ListKey, list[int]] = {}
        for position, job in enumerate(jobs):
            groups.setdefault(self._list_key(job, scripts), []).append(position)
        parsed: dict[int, Parsed] = {}
        for key, positions in groups.items():
            batch = [jobs[position] for position in positions]
            suppressed = self._suppression(key, batch[0])
            results = self._translate_batch(batch, suppressed, options)
            for position, job, result in zip(positions, batch, results, strict=True):
                parsed[position] = self._parse(job, result, tally)
        return [parsed[position] for position in range(len(jobs))]

    def _list_key(self, job: Job, scripts: bool) -> ListKey:
        forbidden = forbidden_scripts(job.route.decode_tgt) if scripts else None
        return (forbidden, job.allow_separator, scripts)

    def _suppression(self, key: ListKey, job: Job) -> list[list[str]]:
        """The suppression list of `key`, built once however many threads ask for it."""
        with self._lists_lock:
            listed = self._lists.get(key)
            if listed is None:
                listed = suppression_list(
                    self._vocabulary,
                    self._processor,
                    job.route.decode_tgt,
                    end_token=self._end,
                    allow=(SEGMENT_SEPARATOR,) if job.allow_separator else (),
                    scripts=key[2],
                )
                self._lists[key] = listed
            return listed

    def _translate_batch(
        self,
        batch: Sequence[Job],
        suppressed: list[list[str]],
        options: DecodeOptions,
    ) -> Sequence[HypothesesLike]:
        sources = [job.tokens for job in batch]
        call: dict[str, Any] = {
            "beam_size": options.beam_size,
            "num_hypotheses": options.n,
            "max_batch_size": options.max_batch_size,
            "max_decoding_length": options.max_decoding_length,
            "return_end_token": True,
            "disable_unk": True,
            "suppress_sequences": suppressed,
        }
        if self._placement == "tgt":
            call["target_prefix"] = [job.prefix for job in batch]
        try:
            results = self._ct2.translate_batch(sources, **call)
        except (RuntimeError, ValueError) as error:
            # The library's message stays in the chained cause: it is not ours to vouch for.
            raise UpstreamError(
                f"CTranslate2 failed to translate {len(sources)} sources "
                f"({type(error).__name__})"
            ) from error
        if len(results) != len(sources):
            raise UpstreamError(
                f"CTranslate2 returned {len(results)} results for {len(sources)} sources"
            )
        return results

    def _parse(self, job: Job, result: HypothesesLike, tally: dict[str, int]) -> Parsed:
        """The texts of the hypotheses of one source (see the module's text for what is
        dropped); the best one first."""
        reserved = self._reserved_in_windows if job.allow_separator else self._reserved
        texts: dict[str, bool] = {}
        for hypothesis in result.hypotheses:
            tokens = list(hypothesis)
            if self._placement == "tgt":
                if tokens[:1] != [job.route.tag]:
                    raise UpstreamError(
                        f"CTranslate2 returned a hypothesis for {job.label} that does not "
                        f"start with the forced prefix {job.route.tag}"
                    )
                tokens = tokens[1:]
            complete = bool(tokens) and tokens[-1] == self._end
            pieces = tokens[:-1] if complete else tokens
            if job.cut_at_separator and SEGMENT_SEPARATOR in pieces:
                # The separator is the model saying the sentence is finished.
                pieces = pieces[: pieces.index(SEGMENT_SEPARATOR)]
                complete = True
            if any(piece in reserved for piece in pieces):
                tally["reserved_symbol"] += 1
                continue
            text = self._processor.decode(pieces)
            if LINE_BREAKS.search(text):
                tally["line_break"] += 1
                text = LINE_BREAKS.sub(" ", text)
            text = text.strip()
            if text and text not in texts:
                texts[text] = complete
        if not texts:
            return Parsed("", (), None)
        best, *others = texts
        return Parsed(best, tuple(others), texts[best])

    # ---- documents -----------------------------------------------------------------------

    def _normalised_documents(
        self, documents: Sequence[str], src: str | None
    ) -> list[str]:
        """Every document checked and normalised before anything is sent."""
        assert src is not None
        texts = []
        for index, document in enumerate(documents):
            if not isinstance(document, str):
                raise TypeError(
                    f"document {index} must be text, got {type(document).__name__}"
                )
            require_encodable(document, f"document {index}")
            text = normalize(document, src)
            for symbol in self._source_symbols:
                # The engine writes `<sep>` and `<doc>` into a window itself: only the text of
                # the request is checked, and a window would read its `<sep>` as the boundary
                # of two sentences.
                if symbol in text:
                    raise reserved_symbol_error(f"document {index}", symbol)
            texts.append(text)
        return texts

    def _translate_document(
        self,
        text: str,
        src: str | None,
        tgt: str,
        script: ScriptName | None,
        route: Route,
        options: DecodeOptions,
        tally: dict[str, int],
    ) -> DocumentResult:
        assert src is not None
        sentences = split(text, src)
        window_options = replace(options, n=1)
        decoded: dict[str, Translation] = {}

        def translate_window(joined: str) -> str:
            label = f"window {tally['windows']}"
            tally["windows"] += 1
            pieces = self._processor.encode(joined, out_type=str)
            tokens = [route.tag, DOC_MARKER, *pieces]
            job = Job(label, tokens, None, route, allow_separator=True)
            (outcome,) = self._decode_masked([job], window_options, tally)
            return outcome.parsed.text if outcome.parsed.complete else ""

        def translate_sentence(sentence: str) -> str:
            row = Row(sentence, src, tgt, script)
            ((_, translation),) = self._raw_rows([row], options, tally)
            decoded[sentence] = translation
            return translation.text

        chunked = translate_document(
            sentences,
            translate_window=translate_window,
            translate_sentence=translate_sentence,
            count_pieces=lambda joined: len(
                self._processor.encode(joined, out_type=str)
            ),
            max_pieces=options.max_source_pieces,
        )
        tally["window_fallbacks"] += len(chunked.stats.fallbacks)
        translations = tuple(
            self._seal(
                document_sentence(sentence, translated, src, route, decoded),
                route,
                tally,
            )
            for sentence, translated in zip(
                sentences, chunked.translations, strict=True
            )
        )
        joiner = "" if route.reported_tgt in _UNSPACED_TARGETS else " "
        return DocumentResult(
            text=joiner.join(t.text for t in translations if t.text),
            translations=translations,
            stats=chunked.stats,
        )


# ---- loading ------------------------------------------------------------------------------


def _load_processor(export: VerifiedExport) -> ProcessorLike:
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


def _require_usable_tags(
    export: VerifiedExport, vocabulary: frozenset[str], processor: ProcessorLike
) -> None:
    """The tag of every target of the record is an entry of the vocabulary (CTranslate2 turns
    any other token into `<unk>` without a word) and one piece of the SentencePiece model."""
    for code, tag in sorted(export.record.tags.items()):
        if tag not in vocabulary:
            raise ConfigError(
                f"the direction tag {tag} of {code} is not an entry of "
                f"{export.directory / _VOCABULARY_FILE}: CTranslate2 would turn it into "
                "<unk> and translate into the wrong language without a word"
            )
        piece_id = processor.piece_to_id(tag)
        if piece_id == processor.unk_id() or processor.id_to_piece(piece_id) != tag:
            raise ConfigError(
                f"the direction tag {tag} of {code} is not one piece of the SentencePiece "
                f"model {export.directory / SPM_FILE}"
            )


def _load_ct2(export: VerifiedExport, load: LoadOptions) -> Ct2Like:
    # Imported here so that importing this module, or the package, never loads the library.
    import ctranslate2

    try:
        return ctranslate2.Translator(
            str(export.directory),
            device=load.device,
            compute_type=load.compute_type,
            inter_threads=load.inter_threads,
            intra_threads=load.intra_threads,
            max_queued_batches=load.max_queued_batches,
        )
    except ValueError as error:
        raise ConfigError(
            f"CTranslate2 cannot use device {load.device!r} with compute type "
            f"{load.compute_type!r}: {error}"
        ) from error
    except (OSError, RuntimeError) as error:
        raise DataError(
            f"cannot load the CTranslate2 model in {export.directory}: {error}"
        ) from error
