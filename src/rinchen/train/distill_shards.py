"""The candidates of one shard of a KD generation (D10): read back from the shard file when it
is valid, otherwise asked of the teacher and written before anything else happens.

A shard file is judged by D6's `ShardStore.load` (`Loaded`, `Missing`, `Invalid`). A shard that
is asked of the translator is checked against the contract of `Translator` (a list with one
list of strings per text; a text UTF-8 cannot encode is refused by the store, before the disk
is touched) and written, so that a run that dies keeps the finished shards. Only the texts with
a letter or a digit are sent: the others have no candidates (a bare invisible character would
be translated into a made-up sentence). Every error names the direction and the shard.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Final, Literal

from rinchen.errors import ConfigError, DataError, UpstreamError
from rinchen.synth._candidates import has_alphanumeric
from rinchen.synth.shards import Invalid, Loaded, Missing, Shard, ShardStore
from rinchen.synth.translator import Translator
from rinchen.train.distill_sources import KdSources

# How the candidates of a shard were obtained: read from a valid file, generated because there
# was no file, or generated to replace a file that failed validation.
Sourced = Literal["reused", "generated", "regenerated"]
REUSED: Final[Sourced] = "reused"


def shard_candidates(
    translator: Translator,
    source: KdSources,
    store: ShardStore,
    shard: Shard,
    total: int,
    *,
    force: bool,
) -> tuple[list[list[str]], Sourced]:
    """The candidate lists of `shard` (shard `shard.index` of `total` of `source`), one per
    text in order, and how they were obtained. With `force` the store is ignored: the shard is
    generated again and overwritten.

    Raises:
        ConfigError: the translator raised one (named after the direction and the shard).
        DataError: the translator raised one, or its reply is not one list of strings per
            text, or holds a text UTF-8 cannot encode; nothing is written for the shard.
        UpstreamError: the translator raised an UpstreamError or an OSError.
    """
    how: Sourced = "generated"
    if not force:
        match store.load(shard.index, shard.ids):
            case Loaded(candidates=candidates):
                return candidates, REUSED
            case Invalid():
                how = "regenerated"
            case Missing():
                pass
    where = f"direction {source.direction}, shard {shard.index} of {total}"
    return _generate_shard(translator, source, store, shard, where), how


def _generate_shard(
    translator: Translator,
    source: KdSources,
    store: ShardStore,
    shard: Shard,
    where: str,
) -> list[list[str]]:
    sendable = [
        position for position, text in enumerate(shard.texts) if has_alphanumeric(text)
    ]
    texts = [shard.texts[position] for position in sendable]
    answers = _translated(translator, source, texts, where) if texts else []
    candidates: list[list[str]] = [[] for _ in shard.texts]
    for position, answer in zip(sendable, answers, strict=True):
        candidates[position] = answer
    try:
        store.write(shard.index, shard.ids, candidates)
    except DataError as error:
        raise DataError(f"{where}: {error}") from error
    return candidates


def _translated(
    translator: Translator, source: KdSources, texts: list[str], where: str
) -> list[list[str]]:
    try:
        reply = translator.translate(texts, source.src_lang, source.tgt_lang)
    except (DataError, ConfigError) as error:
        raise type(error)(f"{where}: {error}") from error
    except (UpstreamError, OSError) as error:
        raise UpstreamError(f"{where} failed: {error}") from error
    return _one_list_of_strings_per_text(where, texts, reply)


def _one_list_of_strings_per_text(
    where: str, texts: Sequence[str], reply: object
) -> list[list[str]]:
    if not isinstance(reply, list):
        raise DataError(
            f"{where}: the translator returned a {type(reply).__name__}, not a list"
        )
    if len(reply) != len(texts):
        raise DataError(
            f"{where}: the translator returned {len(reply)} candidate lists for "
            f"{len(texts)} texts"
        )
    for position, candidates in enumerate(reply):
        if not isinstance(candidates, list) or not all(
            isinstance(text, str) for text in candidates
        ):
            raise DataError(
                f"{where}: the translator's reply for text {position} is not a list "
                "of strings"
            )
    return reply
