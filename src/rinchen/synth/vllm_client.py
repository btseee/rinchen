"""The vLLM client: an OpenAI-compatible chat endpoint that honours `n`, so one
request per text asks for every candidate."""

from __future__ import annotations

from collections.abc import Sequence

from rinchen.config.synth import VllmConfig
from rinchen.synth._chat_completions import ChatCompletionsTranslator, RequestShape


class OpenAICompatTranslator(ChatCompletionsTranslator[VllmConfig]):
    """One request per text with `n` set to the configured candidate count and
    the text's seed as is."""

    def _request_shapes(self) -> Sequence[RequestShape]:
        return (RequestShape(n=self._cfg.n, seed_offset=0),)
