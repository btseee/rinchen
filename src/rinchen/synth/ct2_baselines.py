"""NLLB-200 and MADLAD-400 models converted for CTranslate2, as a `Translator`.

They are external generators used for inference only: a job asks them for the
n-best translations of its texts and nothing is ever trained from them. The
token conventions follow the models' Hugging Face tokenizers (a source ends with
`</s>`; NLLB names both languages as tokens, MADLAD names only the target as a
tag in the text). They have never been run against a real converted model.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from types import MappingProxyType
from typing import Any, Protocol

from rinchen.config.synth import Ct2Config
from rinchen.errors import ConfigError, DataError, UpstreamError
from rinchen.script_conv.char_model import load_char_processor
from rinchen.synth._candidates import clean_candidates
from rinchen.synth._texts import require_encodable

_END_OF_SENTENCE = "</s>"

# Keyed by registry code. The registry's FLORES codes are NLLB's own.
NLLB_CODES: Mapping[str, str] = MappingProxyType(
    {
        "eng_Latn": "eng_Latn",
        "rus_Cyrl": "rus_Cyrl",
        "zho_Hans": "zho_Hans",
        "kor_Hang": "kor_Hang",
        "jpn_Jpan": "jpn_Jpan",
        "deu_Latn": "deu_Latn",
        "kaz_Cyrl": "kaz_Cyrl",
        "tur_Latn": "tur_Latn",
        "khk_Cyrl": "khk_Cyrl",
    }
)
MADLAD_CODES: Mapping[str, str] = MappingProxyType(
    {
        "eng_Latn": "<2en>",
        "rus_Cyrl": "<2ru>",
        "zho_Hans": "<2zh>",
        "kor_Hang": "<2ko>",
        "jpn_Jpan": "<2ja>",
        "deu_Latn": "<2de>",
        "kaz_Cyrl": "<2kk>",
        "tur_Latn": "<2tr>",
        "khk_Cyrl": "<2mn>",
    }
)


class _TranslationLike(Protocol):
    @property
    def hypotheses(self) -> Sequence[Sequence[str]]: ...


class _TranslatorLike(Protocol):
    def translate_batch(
        self, source: list[list[str]], **options: Any
    ) -> Sequence[_TranslationLike]: ...


class _ProcessorLike(Protocol):
    def encode(self, text: str, /, return_type: type[str] = str) -> list[str]: ...

    def decode(self, tokens: list[str]) -> str: ...


class _Family(ABC):
    """What NLLB and MADLAD do differently: the table of language codes, the
    source tokens, whether the target is forced as a prefix, and what the
    decoder's output starts with."""

    name: str
    codes: Mapping[str, str]

    @abstractmethod
    def source_tokens(
        self,
        processor: _ProcessorLike,
        text: str,
        source_code: str,
        target_code: str,
    ) -> list[str]: ...

    @abstractmethod
    def target_prefix(self, target_code: str) -> list[str] | None:
        """The tokens the decoder must start with, or None when it is free."""

    @abstractmethod
    def decoded_tokens(self, hypothesis: Sequence[str], target_code: str) -> list[str]:
        """The tokens of a hypothesis that make up the translation."""


class _Nllb(_Family):
    name = "nllb"
    codes = NLLB_CODES

    def source_tokens(
        self,
        processor: _ProcessorLike,
        text: str,
        source_code: str,
        target_code: str,
    ) -> list[str]:
        return [source_code, *processor.encode(text, return_type=str), _END_OF_SENTENCE]

    def target_prefix(self, target_code: str) -> list[str] | None:
        return [target_code]

    def decoded_tokens(self, hypothesis: Sequence[str], target_code: str) -> list[str]:
        tokens = list(hypothesis)
        # The forced prefix comes back as the first token of the hypothesis.
        return tokens[1:] if tokens[:1] == [target_code] else tokens


class _Madlad(_Family):
    name = "madlad"
    codes = MADLAD_CODES

    def source_tokens(
        self,
        processor: _ProcessorLike,
        text: str,
        source_code: str,
        target_code: str,
    ) -> list[str]:
        # The tag is a user-defined SentencePiece symbol, which is matched only
        # when it is encoded inside the text. The source language is not part
        # of the input.
        pieces = processor.encode(f"{target_code} {text}", return_type=str)
        return [*pieces, _END_OF_SENTENCE]

    def target_prefix(self, target_code: str) -> list[str] | None:
        return None

    def decoded_tokens(self, hypothesis: Sequence[str], target_code: str) -> list[str]:
        return list(hypothesis)


_FAMILIES: Mapping[str, _Family] = {"nllb": _Nllb(), "madlad": _Madlad()}


class CT2Translator:
    """Translates with a CTranslate2 model of the family in `cfg`: `cfg.n`
    hypotheses per text from a beam search of `cfg.beam_size`, each at most
    `cfg.max_tokens` tokens long (`max_decoding_length`; the library's default
    would cut longer output silently), all the texts of a call in one batch that
    the library splits into `cfg.batch_size`. The longest input stays at the
    library's default (`max_input_length`, 1024 tokens). Candidates are cleaned
    like the LLM clients' (stripped, never empty, without duplicates), so a text
    may get fewer than `cfg.n`.

    A model that is not passed in is loaded from `cfg.model_dir` and
    `cfg.spm_model`; passing both in is how tests avoid any real model. The
    translator keeps no state between calls; running calls concurrently is up
    to CTranslate2, whose `Translator` accepts concurrent `translate_batch`
    calls."""

    def __init__(
        self,
        cfg: Ct2Config,
        *,
        translator: _TranslatorLike | None = None,
        processor: _ProcessorLike | None = None,
    ) -> None:
        """Raises:
        DataError: the model directory or the SentencePiece file is missing or
            cannot be loaded; the library's error is chained when there is one.
        ConfigError: CTranslate2 does not support `cfg.device`.
        """
        self._cfg = cfg
        self._family = _FAMILIES[cfg.family]
        # The SentencePiece file first: it is the cheap one, so a wrong path to
        # it costs no model load.
        self._processor = (
            processor if processor is not None else load_char_processor(cfg.spm_model)
        )
        self._translator = translator if translator is not None else _load(cfg)

    def translate(
        self, texts: Sequence[str], src_lang: str, tgt_lang: str
    ) -> list[list[str]]:
        """One list of candidates per text, best first.

        Raises:
            ConfigError: a language is not in the family's table, even when
                `texts` is empty.
            DataError: a text cannot be encoded as UTF-8 (a lone surrogate);
                the message names its index.
            UpstreamError: the library fails, or returns a result count other
                than `len(texts)`.
        """
        source_code = self._code(src_lang)
        target_code = self._code(tgt_lang)
        if not texts:
            return []
        require_encodable(texts)
        source = [
            self._family.source_tokens(self._processor, text, source_code, target_code)
            for text in texts
        ]
        results = self._translate_batch(source, target_code)
        if len(results) != len(texts):
            raise UpstreamError(
                f"CTranslate2 returned {len(results)} results for {len(texts)} texts"
            )
        return [self._candidates(result, target_code) for result in results]

    def _code(self, language: str) -> str:
        code = self._family.codes.get(language)
        if code is None:
            raise ConfigError(
                f"language {language!r} is not in the {self._family.name} table; "
                f"supported: {sorted(self._family.codes)}"
            )
        return code

    def _translate_batch(
        self, source: list[list[str]], target_code: str
    ) -> Sequence[_TranslationLike]:
        options: dict[str, Any] = {
            "beam_size": self._cfg.beam_size,
            "num_hypotheses": self._cfg.n,
            "max_batch_size": self._cfg.batch_size,
            "max_decoding_length": self._cfg.max_tokens,
            "return_scores": False,
        }
        prefix = self._family.target_prefix(target_code)
        if prefix is not None:
            options["target_prefix"] = [prefix for _ in source]
        try:
            return self._translator.translate_batch(source, **options)
        except (RuntimeError, ValueError) as exc:
            raise UpstreamError(
                f"CTranslate2 failed to translate {len(source)} texts: {exc}"
            ) from exc

    def _candidates(self, result: _TranslationLike, target_code: str) -> list[str]:
        return clean_candidates(
            self._processor.decode(self._family.decoded_tokens(hypothesis, target_code))
            for hypothesis in result.hypotheses
        )


def _load(cfg: Ct2Config) -> _TranslatorLike:
    if not cfg.model_dir.is_dir():
        raise DataError(
            f"cannot load the CTranslate2 model: {cfg.model_dir} is not a directory"
        )
    # Imported here so that importing this module stays cheap, and a bad path or
    # a job that uses another generator never loads the library.
    import ctranslate2

    try:
        return ctranslate2.Translator(str(cfg.model_dir), device=cfg.device)
    except ValueError as exc:
        raise ConfigError(
            f"cannot use device {cfg.device!r} for CTranslate2: {exc}"
        ) from exc
    except (OSError, RuntimeError) as exc:
        raise DataError(
            f"cannot load the CTranslate2 model in {cfg.model_dir}: {exc}"
        ) from exc
