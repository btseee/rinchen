"""`rinchen.Translator` (D11): one export directory, loaded once, translating text.

A convenience over one `Engine`: a text gives a `Translation`, a sequence of texts a list of
them, and the keywords are the options of the engine's calls. Everything the engine does (the
guards, the statuses, the script mask, the counters) is the engine's; see `rinchen.infer.engine`.

    with rinchen.Translator("models/tiny", device="cpu") as translator:
        result = translator.translate("Hello", tgt="khk_Cyrl")

The `Translator` is thread-safe as the engine is. No GPU was used and none is claimed: `device`
is `cpu` unless an operator says `cuda`, which was never run, and `auto` is refused.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from types import TracebackType
from typing import Literal, Self, overload

from rinchen.infer.engine import Engine
from rinchen.infer.engine_types import (
    DecodeOptions,
    DocumentResult,
    LoadOptions,
    ModelInfo,
    Row,
    ScriptConverterLike,
    ScriptMode,
    ScriptName,
    Translation,
)


class Translator:
    """The model in `directory`, an export that `rinchen export convert` wrote.

    Raises (at construction):
        ConfigError: `device` or `compute_type` is refused, a thread count is not an integer
            of at least 1, or the directory is not an export our manifest vouches for.
        DataError: a file of the export or the model is unreadable.
    """

    def __init__(
        self,
        directory: str | os.PathLike[str],
        *,
        device: Literal["cpu", "cuda"] = "cpu",
        compute_type: str = "default",
        inter_threads: int = 1,
        intra_threads: int = 1,
        script_converter: ScriptConverterLike | None = None,
    ) -> None:
        load = LoadOptions(
            device=device,
            compute_type=compute_type,
            inter_threads=inter_threads,
            intra_threads=intra_threads,
        )
        self._engine = Engine.open(directory, load, script_converter=script_converter)

    @property
    def info(self) -> ModelInfo:
        """What the model is (see `ModelInfo`)."""
        return self._engine.info

    def counters(self) -> dict[str, int]:
        """The engine's counters since this translator was made."""
        return self._engine.counters()

    # A `str` is a `Sequence[str]` too: the first overload is the one a `str` gets.
    @overload
    def translate(  # type: ignore[overload-overlap]
        self,
        texts: str,
        *,
        tgt: str,
        src: str | None = None,
        script: ScriptName | None = None,
        beam_size: int = 4,
        n: int = 1,
        max_decoding_length: int = 256,
        max_source_pieces: int = 256,
        script_mask: ScriptMode = "retry",
    ) -> Translation: ...

    @overload
    def translate(
        self,
        texts: Sequence[str],
        *,
        tgt: str,
        src: str | None = None,
        script: ScriptName | None = None,
        beam_size: int = 4,
        n: int = 1,
        max_decoding_length: int = 256,
        max_source_pieces: int = 256,
        script_mask: ScriptMode = "retry",
    ) -> list[Translation]: ...

    def translate(
        self,
        texts: str | Sequence[str],
        *,
        tgt: str,
        src: str | None = None,
        script: ScriptName | None = None,
        beam_size: int = 4,
        n: int = 1,
        max_decoding_length: int = 256,
        max_source_pieces: int = 256,
        script_mask: ScriptMode = "retry",
    ) -> Translation | list[Translation]:
        """A `Translation` for a text, a list of them for a sequence of texts.

        `tgt` is the registry code of the language to write (`script`, `Cyrl` or `Mong`,
        for a Mongolian target); `src` the language of the texts when known (a hint: None
        means unknown). The other keywords are `DecodeOptions`'. Raises what
        `Engine.translate` raises.
        """
        options = DecodeOptions(
            beam_size=beam_size,
            n=n,
            max_decoding_length=max_decoding_length,
            max_source_pieces=max_source_pieces,
            script_mask=script_mask,
        )
        if isinstance(texts, str):
            return self._engine.translate([Row(texts, src, tgt, script)], options)[0]
        rows = [Row(text, src, tgt, script) for text in texts]
        return self._engine.translate(rows, options)

    @overload
    def translate_document(  # type: ignore[overload-overlap]
        self,
        document: str,
        *,
        tgt: str,
        src: str,
        script: ScriptName | None = None,
        beam_size: int = 4,
        n: int = 1,
        max_decoding_length: int = 256,
        max_source_pieces: int = 256,
        script_mask: ScriptMode = "retry",
    ) -> DocumentResult: ...

    @overload
    def translate_document(
        self,
        document: Sequence[str],
        *,
        tgt: str,
        src: str,
        script: ScriptName | None = None,
        beam_size: int = 4,
        n: int = 1,
        max_decoding_length: int = 256,
        max_source_pieces: int = 256,
        script_mask: ScriptMode = "retry",
    ) -> list[DocumentResult]: ...

    def translate_document(
        self,
        document: str | Sequence[str],
        *,
        tgt: str,
        src: str,
        script: ScriptName | None = None,
        beam_size: int = 4,
        n: int = 1,
        max_decoding_length: int = 256,
        max_source_pieces: int = 256,
        script_mask: ScriptMode = "retry",
    ) -> DocumentResult | list[DocumentResult]:
        """A `DocumentResult` for a document, a list of them for a sequence of documents.
        `src` is required: a document is cut into sentences by the rules of its language.
        Raises what `Engine.translate_documents` raises."""
        options = DecodeOptions(
            beam_size=beam_size,
            n=n,
            max_decoding_length=max_decoding_length,
            max_source_pieces=max_source_pieces,
            script_mask=script_mask,
        )
        if isinstance(document, str):
            return self._engine.translate_documents(
                [document], src, tgt, options, script=script
            )[0]
        return self._engine.translate_documents(
            document, src, tgt, options, script=script
        )

    def close(self) -> None:
        """Unload the model. Calling it again does nothing."""
        self._engine.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()
