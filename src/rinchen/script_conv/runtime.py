"""Convert text between Cyrillic and traditional Mongolian script with the exported model.

`ScriptConverter` is what the serving engine calls. It splits the text into Cyrillic,
traditional-script and other runs, sends each distinct word of the source script through
the CTranslate2 model once, checks what comes back, and puts the text together again
with everything else unchanged.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol

import ctranslate2

from rinchen.errors import ConfigError, DataError, UpstreamError
from rinchen.script_conv.alphabet import LETTERS, SCRIPT_TO_CYRILLIC_PUNCTUATION
from rinchen.script_conv.char_model import (
    TO_CYRILLIC_TAG,
    TO_SCRIPT_TAG,
    load_char_processor,
)
from rinchen.script_conv.rules import transliterate, transliterate_word
from rinchen.script_conv.segments import Segment, segment
from rinchen.script_conv.validate import is_script_word
from rinchen.text.normalize import normalize

Target = Literal["Mong", "Cyrl"]
_Spelling = Literal["model", "rules", "kept"]

_CYRILLIC = "khk_Cyrl"
_SCRIPT = "mn_Mong"
# Longer words are far outside what the model was trained on.
_MAX_WORD_LENGTH = 80
_MAX_DECODING_LENGTH = 256
# CTranslate2 sorts the words of one call by length and decodes them in chunks of
# this size; its default (0) never splits, so a huge call would be one huge decode.
_MAX_BATCH_SIZE = 1024
# The cache never holds more than this many words, so a long-running process fed
# arbitrary text keeps a bounded memory.
_MAX_CACHED_WORDS = 100_000
_SCRIPT_TO_CYRILLIC_MARKS = str.maketrans(dict(SCRIPT_TO_CYRILLIC_PUNCTUATION))


class _Translation(Protocol):
    @property
    def hypotheses(self) -> Sequence[Sequence[str]]: ...


class _Translator(Protocol):
    def translate_batch(
        self,
        source: list[list[str]],
        *,
        beam_size: int,
        max_decoding_length: int,
        max_batch_size: int,
    ) -> Sequence[_Translation]: ...


class _Processor(Protocol):
    def encode(self, text: str, out_type: type = str) -> list[str]: ...

    def decode(self, tokens: list[str]) -> str: ...


@dataclass(frozen=True)
class Conversion:
    """The converted `text` and how often the model's output could not be used.

    `rule_fallbacks` counts the occurrences, in text converted to script, that the
    rule baseline spelled because the model's output for the word was empty or not a
    script word (`rinchen.script_conv.validate.is_script_word`), or because the word
    was too long for the model. `invalid_outputs` counts the occurrences that kept
    their source-script text because no usable output existed: in text converted to
    Cyrillic, an output that is not one Cyrillic word, or a word too long for the
    model; in text converted to script, a word that the rules cannot spell either.
    """

    text: str
    rule_fallbacks: int
    invalid_outputs: int


@dataclass(frozen=True)
class _Direction:
    """What the two targets do differently in the steps they share: the tag the
    model needs, which segments are words to convert, and how a word is normalized
    the way the training words were."""

    target: Target
    tag: str
    is_source_word: Callable[[Segment], bool]
    normalize_word: Callable[[str], str]


def _is_cyrillic_word(piece: Segment) -> bool:
    return piece.kind == "cyrillic"


def _normalize_cyrillic_word(word: str) -> str:
    return normalize(word.lower(), _CYRILLIC)


def _is_script_segment_with_letter(piece: Segment) -> bool:
    # A run of selectors, joiners or strokes with no letter is not a word.
    return piece.kind == "mongolian" and not LETTERS.isdisjoint(piece.text)


def _normalize_script_word(word: str) -> str:
    return normalize(word, _SCRIPT)


_TO_SCRIPT = _Direction(
    "Mong", TO_SCRIPT_TAG, _is_cyrillic_word, _normalize_cyrillic_word
)
_TO_CYRILLIC = _Direction(
    "Cyrl", TO_CYRILLIC_TAG, _is_script_segment_with_letter, _normalize_script_word
)


@dataclass(frozen=True)
class _Plan:
    """A text cut into segments, and the model's output for its words.

    `normalized` maps each word to convert, as written, to its normalized form;
    `outputs` maps a normalized word that went to the model to what it returned (a
    word too long for the model is not in it).
    """

    pieces: tuple[Segment, ...]
    normalized: Mapping[str, str]
    outputs: Mapping[str, str]


class ScriptConverter:
    """Converts text between Cyrillic and traditional Mongolian script.

    Conversion is word-level: the model sees one word at a time, so what a word
    means in its sentence cannot change its spelling. Each distinct word of a call
    is converted once, in one batch, and the model's output for it is cached for the
    life of the instance. An instance is NOT thread-safe, because of that cache:
    give each thread its own.

    Text converted to script: Cyrillic words become script, script words pass
    through, and in the text between words a full stop and a comma become the
    script's own marks (except inside numbers and Latin text, see
    `rinchen.script_conv.rules.transliterate`). Text converted to Cyrillic: script
    words become Cyrillic, Cyrillic words pass through, and the script's marks become
    "." and ",". Text with no word of the source script comes back unchanged.
    """

    def __init__(self, ct2_dir: Path, spm_model: Path, *, device: str = "cpu") -> None:
        """Load the character model `spm_model` and the CTranslate2 model in `ct2_dir`.

        The character model loads first: it is the cheap one, so a wrong path to it
        costs nothing.

        Raises:
            DataError: either model is missing or cannot be loaded; the library's
                error is chained as the cause.
            ConfigError: CTranslate2 does not support `device`; its error is chained.
        """
        processor = load_char_processor(spm_model)
        self._bind(_load_translator(ct2_dir, device), processor)

    @classmethod
    def from_parts(
        cls, translator: _Translator, processor: _Processor
    ) -> ScriptConverter:
        """A converter over an already built `translator` and `processor`.

        They are a `ctranslate2.Translator` and a `sentencepiece.SentencePieceProcessor`,
        or stand-ins with the same `translate_batch`, `encode` and `decode`.
        """
        converter = cls.__new__(cls)
        converter._bind(translator, processor)
        return converter

    def _bind(self, translator: _Translator, processor: _Processor) -> None:
        self._translator = translator
        self._processor = processor
        self._cache: dict[tuple[Target, str], str] = {}

    def convert(self, text: str, target: Target) -> Conversion:
        """Convert the words of `text` to the script `target`.

        `target` is "Mong" for traditional script and "Cyrl" for Cyrillic. Never
        raises for odd text; whatever is not a word of the source script passes
        through the "other" path.

        Raises:
            ConfigError: `target` is neither "Mong" nor "Cyrl".
            UpstreamError: the translator fails (its `RuntimeError` is chained), or
                answers with a different number of results than words, or with a
                result that has no hypothesis. Nothing of that call is cached. An
                empty hypothesis string is not an error: it is an invalid output.
        """
        if target == "Mong":
            return self._to_script(text)
        if target == "Cyrl":
            return self._to_cyrillic(text)
        raise ConfigError(f"unknown target {target!r}: expected 'Mong' or 'Cyrl'")

    def _to_script(self, text: str) -> Conversion:
        plan = self._plan(text, _TO_SCRIPT)
        if plan is None:
            return Conversion(text, 0, 0)
        converted: list[str] = []
        rule_fallbacks = 0
        invalid_outputs = 0
        for piece in plan.pieces:
            if _TO_SCRIPT.is_source_word(piece):
                word = plan.normalized[piece.text]
                spelling, how = _spell_in_script(
                    piece.text, word, plan.outputs.get(word)
                )
                converted.append(spelling)
                if how == "rules":
                    rule_fallbacks += 1
                elif how == "kept":
                    invalid_outputs += 1
            elif piece.kind == "other":
                converted.append(transliterate(piece.text))
            else:
                converted.append(piece.text)
        return Conversion("".join(converted), rule_fallbacks, invalid_outputs)

    def _to_cyrillic(self, text: str) -> Conversion:
        plan = self._plan(text, _TO_CYRILLIC)
        if plan is None:
            return Conversion(text, 0, 0)
        converted: list[str] = []
        invalid_outputs = 0
        for piece in plan.pieces:
            if _TO_CYRILLIC.is_source_word(piece):
                output = plan.outputs.get(plan.normalized[piece.text])
                if output is not None and _is_one_cyrillic_word(output):
                    converted.append(output)
                else:
                    converted.append(piece.text)
                    invalid_outputs += 1
            elif piece.kind == "other":
                converted.append(piece.text.translate(_SCRIPT_TO_CYRILLIC_MARKS))
            else:
                converted.append(piece.text)
        return Conversion("".join(converted), 0, invalid_outputs)

    def _plan(self, text: str, direction: _Direction) -> _Plan | None:
        """The segments of `text` and the model's output for its words to convert,
        or None when `text` has no such word and comes back unchanged."""
        pieces = segment(text)
        normalized = {
            piece.text: direction.normalize_word(piece.text)
            for piece in pieces
            if direction.is_source_word(piece)
        }
        if not normalized:
            return None
        to_send = [
            word for word in dict.fromkeys(normalized.values()) if _fits_the_model(word)
        ]
        outputs = self._outputs(direction, to_send)
        return _Plan(pieces, normalized, outputs)

    def _outputs(self, direction: _Direction, words: Sequence[str]) -> dict[str, str]:
        """The model's output for each of the distinct `words`: from the cache, and
        for the rest from one batch."""
        outputs = {
            word: self._cache[(direction.target, word)]
            for word in words
            if (direction.target, word) in self._cache
        }
        missing = [word for word in words if word not in outputs]
        if missing:
            fresh = self._translate(direction.tag, missing)
            self._remember(direction.target, fresh)
            outputs.update(fresh)
        return outputs

    def _translate(self, tag: str, words: Sequence[str]) -> dict[str, str]:
        source = [[tag, *self._processor.encode(word, out_type=str)] for word in words]
        longest = max(len(word) for word in words)
        try:
            results = self._translator.translate_batch(
                source,
                beam_size=1,
                max_decoding_length=min(_MAX_DECODING_LENGTH, 4 * longest + 8),
                max_batch_size=_MAX_BATCH_SIZE,
            )
        except RuntimeError as exc:
            raise UpstreamError(
                f"CTranslate2 failed to translate {len(words)} words: {exc}"
            ) from exc
        if len(results) != len(words):
            raise UpstreamError(
                f"the translator returned {len(results)} results for {len(words)} words"
            )
        outputs: dict[str, str] = {}
        for word, result in zip(words, results, strict=True):
            if not result.hypotheses:
                raise UpstreamError(
                    f"the translator returned no hypothesis for word {word!r}"
                )
            outputs[word] = self._processor.decode(list(result.hypotheses[0]))
        return outputs

    def _remember(self, target: Target, outputs: dict[str, str]) -> None:
        if len(outputs) > _MAX_CACHED_WORDS:
            # This one call has more fresh (not yet cached) words than the cache can
            # hold: keeping them would only push every other entry out.
            return
        if len(self._cache) + len(outputs) > _MAX_CACHED_WORDS:
            # Drops the whole cache, warm entries included: simpler than evicting
            # one word at a time, and later calls refill it.
            self._cache.clear()
        for word, output in outputs.items():
            self._cache[(target, word)] = output


def _load_translator(ct2_dir: Path, device: str) -> _Translator:
    try:
        return ctranslate2.Translator(str(ct2_dir), device=device, intra_threads=1)
    except ValueError as exc:
        raise ConfigError(
            f"cannot use device {device!r} for CTranslate2: {exc}"
        ) from exc
    except (OSError, RuntimeError) as exc:
        raise DataError(
            f"cannot load the CTranslate2 model in {ct2_dir}: {exc}"
        ) from exc


def _fits_the_model(word: str) -> bool:
    return 0 < len(word) <= _MAX_WORD_LENGTH


def _is_one_cyrillic_word(text: str) -> bool:
    pieces = segment(text)
    return len(pieces) == 1 and pieces[0].kind == "cyrillic"


def _spell_in_script(
    original: str, normalized: str, output: str | None
) -> tuple[str, _Spelling]:
    """The script spelling of a Cyrillic word and where it came from: the model's
    `output` when it is there and a script word, else the rules', else the `original`
    word."""
    if output is not None and is_script_word(output):
        return output, "model"
    rule_spelling = transliterate_word(normalized or original).text
    if rule_spelling:
        return rule_spelling, "rules"
    return original, "kept"
