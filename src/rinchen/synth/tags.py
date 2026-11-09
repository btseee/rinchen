"""Which corpus origin a synthetic-data job produces, and which training-time
data tag an origin carries."""

from __future__ import annotations

from rinchen.corpus.schema import ORIGINS
from rinchen.errors import ConfigError

# Forward translation is labelled by the family of its generator: an LLM or a
# translation model. The plain `ft` origin is forward translation by our own
# teacher, the generator kind `teacher` (D10's distillation; the configurable
# generators of a synthetic-data job cannot be one).
_FORWARD_ORIGIN_BY_GENERATOR_KIND = {
    "vllm": "synth_llm",
    "llamacpp": "synth_llm",
    "ct2": "synth_nmt",
    "teacher": "ft",
}
_BACKWARD_ORIGIN = "bt"
_DATA_TAG_BY_ORIGIN = {
    "bt": "<bt>",
    "ft": "<ft>",
    "synth_llm": "<ft>",
    "synth_nmt": "<ft>",
}


def origin_for(kind: str, generator_kind: str) -> str:
    """The corpus origin of the pairs a `kind` job makes with a generator of
    `generator_kind`. Raises ConfigError for a kind it does not know."""
    if generator_kind not in _FORWARD_ORIGIN_BY_GENERATOR_KIND:
        raise ConfigError(f"unknown generator kind {generator_kind!r}")
    if kind == "bt":
        return _BACKWARD_ORIGIN
    if kind == "ft":
        return _FORWARD_ORIGIN_BY_GENERATOR_KIND[generator_kind]
    raise ConfigError(f"unknown job kind {kind!r}")


def data_origin_tag(origin: str) -> str | None:
    """The data tag a training row of this origin carries, or None when the
    origin is not marked. Raises ConfigError for an origin the corpus schema
    does not have."""
    if origin not in ORIGINS:
        raise ConfigError(f"unknown corpus origin {origin!r}")
    return _DATA_TAG_BY_ORIGIN.get(origin)
