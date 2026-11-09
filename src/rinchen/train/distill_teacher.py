"""The teacher of a distillation as a `Translator` (D10): a CTranslate2 export of OURS behind
the protocol of D6's generators.

`TeacherTranslator` translates the texts of a call in one `translate_batch`. It guards
against what the library does silently (measured in the D10 spike):

- the direction tag goes where the export's record says: in the source (`src`) or as a forced
  prefix of the target (`tgt`; the hypotheses then START with the tag and lose it here). A tag
  that is not an entry of `shared_vocabulary.json` becomes `<unk>`, so the tag of a target
  language must be an entry of it and one piece of the SentencePiece model before anything is
  translated (`check_directions`, and every `translate`);
- the library cuts a hypothesis at `max_decoding_length` and a source at 1,024 pieces without a
  sign: a hypothesis counts only when it ends with the end token of the converter's
  `config.json` (`return_end_token`), and a source of more than `max_source_pieces` is not sent;
- a source without a piece gets no candidates (a bare tag is translated into a made-up sentence)
  and a hypothesis that holds a line break is dropped (it would split a text file's line).

`counters()` is what THIS translator did since it was made: `sources` (texts given),
`source_too_long` and `empty_source` (texts not sent), `truncated` and `line_break` (hypotheses
dropped). A failed call adds nothing. The files read at construction are checked against the
SHA-256 `verify_export` took (`distill_export_files`); `model.bin` is loaded by CTranslate2.

No GPU was used and none is claimed: the device is `cpu` by default, `cuda` is an explicit
operator choice that was never run (`auto` is refused: it would pick the GPU), and everything
about CTranslate2 on another device or on a real teacher is unmeasured. `ctranslate2` is
imported only when a model is loaded here.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Final, Literal, Protocol

from rinchen.config.ablation import TagPlacement, require_placement
from rinchen.config.languages import direction_tag, known_language_codes
from rinchen.errors import ConfigError, DataError, UpstreamError
from rinchen.synth._candidates import clean_candidates
from rinchen.synth._texts import require_encodable
from rinchen.train.distill_export import SPM_FILE, VerifiedExport
from rinchen.train.distill_export_files import (
    VOCABULARY_FILE,
    end_token,
    shared_vocabulary,
    verified_bytes,
)
from rinchen.train.mixture import LINE_BREAK_CHARACTERS
from rinchen.train.validation_corpus import load_processor, require_one_piece
from rinchen.utils.integers import is_strict_int

COUNTER_KEYS: Final = (
    "sources",
    "source_too_long",
    "empty_source",
    "truncated",
    "line_break",
)

# CTranslate2's `max_input_length` is 1,024 and cuts a longer source silently; a tag in the
# source takes one place of it.
MAX_SOURCE_PIECES: Final = 1023

_DEVICES: Final = ("cpu", "cuda")
_NUMBERS: Final = (
    "beam_size",
    "n",
    "max_decoding_length",
    "max_source_pieces",
    "batch_size",
    "threads",
)


@dataclass(frozen=True, slots=True)
class TeacherOptions:
    """How the teacher decodes and where it runs.

    `beam_size` and `n` (the hypotheses kept per source; CTranslate2 allows at most the
    beam's), `max_decoding_length` (pieces of a hypothesis, the end token included),
    `max_source_pieces` (the longest source that is sent), `batch_size` (what the library
    splits a call into), `device` (`cpu` or `cuda`), `compute_type` (CTranslate2's, such as
    `default` or `int8`) and `threads` (intra-op threads).

    Raises:
        ConfigError: a number is not an integer of at least 1 (a bool is not one), `n` is
            above `beam_size`, `max_source_pieces` is above 1,023, `device` is not `cpu`
            or `cuda` (`auto` is refused with the reason) or `compute_type` is blank.
    """

    beam_size: int = 4
    n: int = 1
    max_decoding_length: int = 256
    max_source_pieces: int = 256
    batch_size: int = 64
    device: Literal["cpu", "cuda"] = "cpu"
    compute_type: str = "default"
    threads: int = 1

    def __post_init__(self) -> None:
        for name in _NUMBERS:
            value = getattr(self, name)
            if not is_strict_int(value) or value < 1:
                raise ConfigError(
                    f"TeacherOptions.{name} must be an integer of at least 1, "
                    f"got {value!r}"
                )
        if self.n > self.beam_size:
            raise ConfigError(
                f"TeacherOptions.n ({self.n}) must not exceed TeacherOptions.beam_size "
                f"({self.beam_size}): CTranslate2 returns at most as many hypotheses as "
                "the beam holds"
            )
        if self.max_source_pieces > MAX_SOURCE_PIECES:
            raise ConfigError(
                f"TeacherOptions.max_source_pieces ({self.max_source_pieces}) must be at "
                f"most {MAX_SOURCE_PIECES}: CTranslate2 cuts a source of more than 1,024 "
                "pieces silently, and the tag takes one"
            )
        if self.device == "auto":
            raise ConfigError(
                "TeacherOptions.device 'auto' is refused: CTranslate2 would choose the "
                "GPU when one is visible; say 'cpu' or 'cuda'"
            )
        if self.device not in _DEVICES:
            raise ConfigError(
                f"TeacherOptions.device must be 'cpu' or 'cuda', got {self.device!r}"
            )
        if not isinstance(self.compute_type, str) or not self.compute_type.strip():
            raise ConfigError(
                "TeacherOptions.compute_type must be a non-empty string, "
                f"got {self.compute_type!r}"
            )


def ct2_source_tokens(
    placement: TagPlacement, tag: str, pieces: Sequence[str]
) -> tuple[list[str], list[str] | None]:
    """The source line and the forced target prefix CTranslate2 is given for a text cut into
    `pieces` and asked for the language `tag` stands for: for placement `src` the tag in
    front of the pieces and no prefix, for `tgt` the pieces alone and the prefix `[tag]`.

    Raises:
        ValueError: `placement` is neither `src` nor `tgt`.
    """
    require_placement(placement)
    if placement == "src":
        return [tag, *pieces], None
    return list(pieces), [tag]


class _Translation(Protocol):
    @property
    def hypotheses(self) -> Sequence[Sequence[str]]: ...


class Ct2Like(Protocol):
    """What the teacher asks of `ctranslate2.Translator`."""

    def translate_batch(
        self, source: list[list[str]], **options: Any
    ) -> Sequence[_Translation]: ...


class ProcessorLike(Protocol):
    """What the teacher asks of a SentencePiece processor (`SentencePieceProcessor` has it)."""

    def encode(self, text: str, out_type: type = str) -> list[str]: ...

    def decode(self, pieces: list[str]) -> str: ...

    def piece_to_id(self, piece: str) -> int: ...

    def unk_id(self) -> int: ...

    def id_to_piece(self, piece_id: int) -> str: ...


class TeacherTranslator:
    """The verified export `export` as a `rinchen.synth.translator.Translator`.

    The model is `translator` when given (tests, or an engine that owns one) and else a
    `ctranslate2.Translator` loaded from the export's directory on `options.device`; the
    SentencePiece processor is `processor` when given and else the export's `spm.model`.
    The translator keeps no state between calls but its counters and the tags it checked.
    """

    def __init__(
        self,
        export: VerifiedExport,
        options: TeacherOptions,
        *,
        translator: Ct2Like | None = None,
        processor: ProcessorLike | None = None,
    ) -> None:
        """Raises:
        ConfigError: a file of the export changed after `verify_export` hashed it, the
            SentencePiece model cannot be loaded, or CTranslate2 rejects the device or
            the compute type.
        DataError: `shared_vocabulary.json` or `config.json` is unreadable or not what
            the converter writes (`config.json` must name `eos_token`), or CTranslate2
            cannot load the model; the library's error is chained.
        """
        self._export = export
        self._options = options
        self._placement = export.record.tag_placement
        self._vocabulary = shared_vocabulary(export)
        self._end_token = end_token(export)
        # The cheap loads first: a wrong file costs no model load.
        self._processor = (
            processor if processor is not None else _load_processor(export)
        )
        self._translator = (
            translator if translator is not None else _load_ct2(export, options)
        )
        self._checked_tags: set[str] = set()
        self._counters = dict.fromkeys(COUNTER_KEYS, 0)

    def counters(self) -> dict[str, int]:
        """A copy of the counters (see the module's text), one per key of `COUNTER_KEYS`."""
        return dict(self._counters)

    def check_directions(self, directions: Sequence[str]) -> None:
        """Check, before anything is translated, that the tag of the target language of each
        direction (`<src_lang>-<tgt_lang>`, registry codes) is an entry of the export's
        `shared_vocabulary.json` and one piece of its SentencePiece model.

        Raises:
            ConfigError: a direction is malformed or names a language outside the
                registry, or a tag fails a check; the message names it and says which.
        """
        for direction in directions:
            source, target = _languages_of(direction)
            self._require_registry_language(source)
            self._tag_of(target)

    def translate(
        self, texts: Sequence[str], src_lang: str, tgt_lang: str
    ) -> list[list[str]]:
        """One list of candidates per text, in input order, best hypothesis first (see the
        module's text for what is dropped). A text that is not sent, or all of whose
        hypotheses were dropped, has an empty list.

        Raises:
            ConfigError: a language is outside the registry, or the tag of `tgt_lang`
                fails the checks of `check_directions`; also when `texts` is empty.
            DataError: a text cannot be encoded as UTF-8 (a lone surrogate); the message
                names its index.
            UpstreamError: the library fails, returns another number of results than
                sources, or (placement `tgt`) a hypothesis does not start with the forced
                tag; the message names the source index and never quotes a text.
        """
        self._require_registry_language(src_lang)
        tag = self._tag_of(tgt_lang)
        if not texts:
            return []
        require_encodable(texts)
        pieces = [self._processor.encode(text, out_type=str) for text in texts]
        tally = dict.fromkeys(COUNTER_KEYS, 0)
        tally["sources"] = len(texts)
        sent = self._sendable(pieces, tally)
        candidates: list[list[str]] = [[] for _ in texts]
        if sent:
            results = self._translate_batch([pieces[index] for index in sent], tag)
            for index, result in zip(sent, results, strict=True):
                candidates[index] = self._candidates(index, result, tag, tally)
        for key, count in tally.items():
            self._counters[key] += count
        return candidates

    def _require_registry_language(self, language: str) -> None:
        if language not in known_language_codes():
            raise ConfigError(f"language {language!r} is not in the language registry")

    def _tag_of(self, language: str) -> str:
        """The direction tag of `language`, once it passed the checks that keep the library
        from translating into the wrong language silently."""
        self._require_registry_language(language)
        tag = direction_tag(language)
        if tag not in self._checked_tags:
            self._require_usable_tag(tag)
            self._checked_tags.add(tag)
        return tag

    def _require_usable_tag(self, tag: str) -> None:
        if tag not in self._vocabulary:
            raise ConfigError(
                f"the direction tag {tag} is not an entry of "
                f"{self._export.directory / VOCABULARY_FILE}: CTranslate2 would turn "
                "it into <unk> and translate into the wrong language without a word"
            )
        require_one_piece(self._processor, tag, self._export.directory / SPM_FILE)

    def _sendable(
        self, pieces: Sequence[Sequence[str]], tally: dict[str, int]
    ) -> list[int]:
        """The indexes of the sources that go to the library; the others are counted."""
        sent = []
        for index, line in enumerate(pieces):
            if not line:
                tally["empty_source"] += 1
            elif len(line) > self._options.max_source_pieces:
                tally["source_too_long"] += 1
            else:
                sent.append(index)
        return sent

    def _translate_batch(
        self, pieces: Sequence[Sequence[str]], tag: str
    ) -> Sequence[_Translation]:
        lines: list[list[str]] = []
        prefixes: list[list[str]] = []
        for line in pieces:
            tokens, prefix = ct2_source_tokens(self._placement, tag, line)
            lines.append(tokens)
            if prefix is not None:
                prefixes.append(prefix)
        options: dict[str, Any] = {
            "beam_size": self._options.beam_size,
            "num_hypotheses": self._options.n,
            "max_batch_size": self._options.batch_size,
            "max_decoding_length": self._options.max_decoding_length,
            "return_end_token": True,
            "return_scores": False,
        }
        if prefixes:
            options["target_prefix"] = prefixes
        try:
            results = self._translator.translate_batch(lines, **options)
        except (RuntimeError, ValueError) as error:
            raise UpstreamError(
                f"CTranslate2 failed to translate {len(lines)} sources: {error}"
            ) from error
        if len(results) != len(lines):
            raise UpstreamError(
                f"CTranslate2 returned {len(results)} results for {len(lines)} sources"
            )
        return results

    def _candidates(
        self, index: int, result: _Translation, tag: str, tally: dict[str, int]
    ) -> list[str]:
        kept = []
        for hypothesis in result.hypotheses:
            tokens = list(hypothesis)
            if self._placement == "tgt":
                if tokens[:1] != [tag]:
                    raise UpstreamError(
                        f"CTranslate2 returned a hypothesis for source {index} that does "
                        f"not start with the forced prefix {tag}"
                    )
                tokens = tokens[1:]
            if not tokens or tokens[-1] != self._end_token:
                tally["truncated"] += 1
                continue
            text = self._processor.decode(tokens[:-1])
            if any(character in text for character in LINE_BREAK_CHARACTERS):
                tally["line_break"] += 1
                continue
            kept.append(text)
        return clean_candidates(kept)


def _languages_of(direction: str) -> tuple[str, str]:
    source, _, target = direction.partition("-")
    if not source or not target or "-" in target:
        raise ConfigError(f"direction {direction!r} is not '<src_lang>-<tgt_lang>'")
    return source, target


def _load_processor(export: VerifiedExport) -> ProcessorLike:
    verified_bytes(export, SPM_FILE)
    return load_processor(export.directory / SPM_FILE)


def _load_ct2(export: VerifiedExport, options: TeacherOptions) -> Ct2Like:
    # Imported here so that importing this module, or the package, never loads the library.
    import ctranslate2

    try:
        return ctranslate2.Translator(
            str(export.directory),
            device=options.device,
            compute_type=options.compute_type,
            intra_threads=options.threads,
        )
    except ValueError as error:
        raise ConfigError(
            f"CTranslate2 cannot use device {options.device!r} with compute type "
            f"{options.compute_type!r}: {error}"
        ) from error
    except (OSError, RuntimeError) as error:
        raise DataError(
            f"cannot load the CTranslate2 model in {export.directory}: {error}"
        ) from error
