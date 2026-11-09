"""What the saved `config.json` of a checkpoint of ours says about its tag placement (D10).

CTranslate2's files do not hold where the direction tag goes (`src`: in the source, `tgt`: a
forced prefix of the target), but the `config.json` Eole saves beside the weights does: its
`data` section keeps the `src_prefix` and `tgt_prefix` the launcher wrote for each corpus
(`train/eole_data.py`). The direction tag `<2xx>` is in `src_prefix` for placement `src`
and in `tgt_prefix` for placement `tgt`; an origin marker (`<bt>`, `<ft>`, `<doc>`) may sit
in `src_prefix` either way and tells nothing. `saved_tag_placement` reads the placement back
and `require_stated_placement` refuses a caller's statement that disagrees: with the tag in
the wrong place the converter still succeeds and every translation is wrong.

A saved config without a `data` section, or whose prefixes hold no direction tag, says
nothing, and the caller's statement stands. A saved config that is not what Eole writes, or
whose corpora place the tag differently, is a `DataError`: no statement can be right.

The bytes of a file are read ONCE, by `verified_content`, and only when their SHA-256 is
the one taken before (hash before read); `json_object` parses them.

`distill evaluate` accepts models of any origin, so the corpus names and the prefixes of a
saved config are untrusted text: a message quotes them escaped on one line (`repr`) and cut to
80 characters, and a corpus name that is a plain word is named as it is (`data.corpus`).
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Literal

from rinchen.config.ablation import TagPlacement
from rinchen.errors import ConfigError, DataError

# One whole word of a prefix, as `direction_tag` writes it: `<2khk_Cyrl>`.
_DIRECTION_TAG: Final = re.compile(r"<2[^\s<>]+>")
_SIDES: Final[tuple[tuple[str, TagPlacement], ...]] = (
    ("src_prefix", "src"),
    ("tgt_prefix", "tgt"),
)
# How much of a name or a prefix of the config a message shows, and what marks the cut.
_SHOWN_CHARS: Final = 80
_ELLIPSIS: Final = "..."
_PLAIN_NAME: Final = re.compile(rf"[A-Za-z0-9_.-]{{1,{_SHOWN_CHARS}}}")

# How the placement of a stated tag was settled: `checked` against the saved config, which
# records it, or `stated` only, because the saved config says nothing about it.
PlacementCheck = Literal["checked", "stated"]


@dataclass(frozen=True, slots=True)
class _Evidence:
    """Where a saved corpus entry puts the tag: the key (`data.<corpus>.src_prefix`), the
    prefix it holds and the placement that means."""

    key: str
    prefix: str
    placement: TagPlacement


def verified_content(file: Path, digest: str) -> bytes:
    """The bytes of `file`, read once, only when they are the bytes with SHA-256 `digest`:
    what is parsed is what was hashed.

    Raises:
        DataError: the file cannot be read.
        ConfigError: its SHA-256 is not `digest` (it changed after it was hashed).
    """
    try:
        content = file.read_bytes()
    except OSError as error:
        raise DataError(f"cannot read {file}: {error.strerror}") from error
    if hashlib.sha256(content).hexdigest() != digest:
        raise ConfigError(
            f"{file} changed after it was hashed, so what would be read is not what "
            "was vouched for"
        )
    return content


def json_object(content: bytes, file: Path) -> dict[str, Any]:
    """The JSON object in `content`, the bytes of `file`.

    Raises:
        DataError: the bytes are not JSON or not an object; the message never quotes them.
    """
    try:
        document = json.loads(content)
    except (ValueError, RecursionError) as error:
        raise DataError(f"{file}: is not valid JSON") from error
    if not isinstance(document, dict):
        raise DataError(f"{file}: is not a JSON object")
    return document


def saved_tag_placement(config: Mapping[str, Any], file: Path) -> TagPlacement | None:
    """The placement the corpora of the saved config `config` (the content of `file`)
    give the direction tag, None when it holds no corpus or no direction tag.

    Raises:
        DataError: `data` is not a mapping of mappings, a prefix is not text, a corpus
            has a direction tag on both sides or the corpora place it differently.
    """
    found = _evidence(config, file)
    return None if found is None else found.placement


def require_stated_placement(
    config: Mapping[str, Any], file: Path, stated: TagPlacement
) -> PlacementCheck:
    """Accept `stated` unless the saved config `config` (the content of `file`) places
    the tag elsewhere, and say whether it was `checked` (the saved config places it where
    `stated` says) or only `stated` (the saved config places it nowhere).

    Raises:
        ConfigError: the saved config places the tag in the other place than `stated`; the
            message names both values and the key that says so.
        DataError: as `saved_tag_placement`.
    """
    found = _evidence(config, file)
    if found is None:
        return "stated"
    if found.placement == stated:
        return "checked"
    raise ConfigError(
        f"the tag placement {stated!r} is not the one the checkpoint was trained with: "
        f"{file} has {found.key} = {_quoted(found.prefix)}, so the direction tag is in the "
        f"{'source' if found.placement == 'src' else 'target'} prefix, which is placement "
        f"{found.placement!r}; pass --tag-placement {found.placement}. With the tag in "
        "the wrong place the model would be given an input it was never trained on"
    )


def require_saved_placement(
    file: Path, digest: str, stated: TagPlacement
) -> PlacementCheck:
    """`require_stated_placement` for the saved config in `file`, which is read once and
    only if its SHA-256 is `digest`.

    Raises:
        ConfigError: as `verified_content` and `require_stated_placement`.
        DataError: as `verified_content`, `json_object` and `saved_tag_placement`.
    """
    config = json_object(verified_content(file, digest), file)
    return require_stated_placement(config, file, stated)


def _evidence(config: Mapping[str, Any], file: Path) -> _Evidence | None:
    data = config.get("data")
    if data is None:
        return None
    if not isinstance(data, Mapping):
        raise DataError(f"{file}: 'data' is not an object, as Eole saves it")
    first: _Evidence | None = None
    for name, corpus in data.items():
        if not isinstance(corpus, Mapping):
            raise DataError(
                f"{file}: {_entry(name)} is not an object, as Eole saves it"
            )
        found = _entry_evidence(name, corpus, file)
        if found is None:
            continue
        if first is not None and found.placement != first.placement:
            raise DataError(
                f"{file}: the corpora place the direction tag differently ({first.key} "
                f"is a {first.placement!r} prefix, {found.key} a {found.placement!r} "
                "one), so no tag placement fits the checkpoint"
            )
        first = first or found
    return first


def _entry_evidence(
    name: str, corpus: Mapping[str, Any], file: Path
) -> _Evidence | None:
    tagged: list[_Evidence] = []
    for key, placement in _SIDES:
        prefix = corpus.get(key)
        if prefix is None:
            continue
        if not isinstance(prefix, str):
            raise DataError(
                f"{file}: {_entry(name)}.{key} is not text, as Eole saves it"
            )
        if any(_DIRECTION_TAG.fullmatch(word) for word in prefix.split()):
            tagged.append(_Evidence(f"{_entry(name)}.{key}", prefix, placement))
    if len(tagged) > 1:
        raise DataError(
            f"{file}: {_entry(name)} has a direction tag in both its source and its "
            "target prefix, so no tag placement fits the checkpoint"
        )
    return tagged[0] if tagged else None


def _entry(name: str) -> str:
    """How a message names the corpus entry `name` of `data`: `data.<name>` for a plain word,
    `data[<quoted name>]` for anything else."""
    if _PLAIN_NAME.fullmatch(name):
        return f"data.{name}"
    return f"data[{_quoted(name)}]"


def _quoted(text: str) -> str:
    """`text` as a literal on one line (`repr` escapes line breaks and control characters),
    cut to `_SHOWN_CHARS` characters, with an ellipsis after the closing quote when it was."""
    cut = _ELLIPSIS if len(text) > _SHOWN_CHARS else ""
    return f"{text[:_SHOWN_CHARS]!r}{cut}"
