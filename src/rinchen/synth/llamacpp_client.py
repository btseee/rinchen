"""The llama.cpp client. Support for `n` varies between llama.cpp server
builds, so it asks for the candidates one request at a time."""

from __future__ import annotations

from collections.abc import Sequence

from rinchen.config.synth import LlamaCppConfig
from rinchen.synth._chat_completions import ChatCompletionsTranslator, RequestShape


class LlamaCppTranslator(ChatCompletionsTranslator[LlamaCppConfig]):
    """`n` requests per text, each with `n` 1 and the text's seed plus the
    candidate's index, merged in that order."""

    def _request_shapes(self) -> Sequence[RequestShape]:
        return tuple(RequestShape(n=1, seed_offset=k) for k in range(self._cfg.n))
