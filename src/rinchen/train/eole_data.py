"""The Eole `data` section of one variant: a corpus entry per corpus file with its
paths, transforms, weight and the prefixes that carry the direction tag, and the
`valid` entry."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final, Protocol

from rinchen.config.ablation import TagPlacement, require_placement
from rinchen.config.languages import direction_tag
from rinchen.errors import ConfigError
from rinchen.synth.tags import data_origin_tag
from rinchen.train.eole_cfg import reject_dollar
from rinchen.train.mixture import BT_GROUP, BT_ORIGIN, KD_GROUP
from rinchen.train.validation_corpus import ValidPaths


class CorpusEntry(Protocol):
    """What the `data` section needs of a corpus: a corpus written (`CorpusFiles`) and
    one a plan only lists (`PlannedFiles`) both have it."""

    @property
    def name(self) -> str: ...

    @property
    def group(self) -> str: ...

    @property
    def tgt_lang(self) -> str: ...

    @property
    def src_path(self) -> Path: ...

    @property
    def tgt_path(self) -> Path: ...


# The origin of the rows of a KD corpus: forward translation by our own teacher.
_KD_ORIGIN: Final = "ft"


def bt_tag() -> str:
    """The tag that marks a back-translated corpus (`<bt>`)."""
    tag = data_origin_tag(BT_ORIGIN)
    if tag is None:
        raise ConfigError(f"the origin {BT_ORIGIN!r} has no data tag")
    return tag


def ft_tag() -> str:
    """The tag that marks a corpus of knowledge distillation (`<ft>`)."""
    tag = data_origin_tag(_KD_ORIGIN)
    if tag is None:
        raise ConfigError(f"the origin {_KD_ORIGIN!r} has no data tag")
    return tag


def _origin_marker(group: str, *, kd_tag: bool) -> str | None:
    """The tag that marks the origin of the corpora of `group`, none for an authentic one
    and for a KD corpus when `kd_tag` is false."""
    if group == BT_GROUP:
        return bt_tag()
    if group == KD_GROUP and kd_tag:
        return ft_tag()
    return None


def _prefixes(corpus: CorpusEntry, placement: str, *, kd_tag: bool) -> tuple[str, str]:
    """(`src_prefix`, `tgt_prefix`) of a corpus: the tag of its target language and, for a
    back-translated corpus the bt tag, for a KD corpus the ft tag, on the side the
    placement names."""
    tag = direction_tag(corpus.tgt_lang)
    marker = _origin_marker(corpus.group, kd_tag=kd_tag)
    if placement == "src":
        return (f"{tag} {marker}" if marker else tag), ""
    return (marker or ""), tag


def _weight(name: str, weights: Mapping[str, int]) -> int:
    if name not in weights:
        raise ValueError(f"no weight for corpus {name!r}")
    weight = weights[name]
    if isinstance(weight, bool) or not isinstance(weight, int) or weight < 1:
        raise ValueError(
            f"the weight of corpus {name!r} must be a positive integer, got {weight!r}"
        )
    return weight


def data_section(
    corpora: Sequence[CorpusEntry],
    weights: Mapping[str, int],
    placement: TagPlacement,
    *,
    valid_paths: ValidPaths,
    filtertoolong: bool,
    kd_tag: bool = True,
) -> dict[str, Any]:
    """The Eole `data` mapping: one entry per corpus with its paths, `transforms`
    (`sentencepiece`, `prefix` and, when `filtertoolong`, `filtertoolong`), `weight`
    and the two prefixes that carry the direction tag, and `valid` with its two paths.

    The tag is that of the corpus's target language; with placement `src` it is the
    source prefix and a bt corpus adds `<bt>` after it, with `tgt` it is the target
    prefix and a bt corpus has `<bt>` alone as source prefix. A KD corpus (group `kd`) is
    marked the same way with `<ft>`, unless `kd_tag` is false: then it is prefixed as an
    authentic corpus (the keyword changes no other corpus). Paths are written as given, as
    POSIX text. The text files carry no tags: the placement is configuration.

    Raises:
        ValueError: a placement other than `src` or `tgt`; no corpus or two corpora of
            one name; a weight that is missing, not an integer or below 1; a weight
            for a name that is no corpus.
        ConfigError: a path holds `$`, which Eole would expand as a variable.
    """
    require_placement(placement)
    if not corpora:
        raise ValueError("at least one corpus is needed")
    names = [corpus.name for corpus in corpora]
    repeated = sorted({name for name in names if names.count(name) > 1})
    if repeated:
        raise ValueError(f"corpus names must be unique, repeated: {repeated}")
    unknown = sorted(set(weights) - set(names))
    if unknown:
        raise ValueError(f"weights name corpora that are not in the mixture: {unknown}")
    transforms = [
        "sentencepiece",
        "prefix",
        *(["filtertoolong"] if filtertoolong else []),
    ]
    section: dict[str, Any] = {}
    for corpus in corpora:
        reject_dollar(f"source path of corpus {corpus.name}", corpus.src_path)
        reject_dollar(f"target path of corpus {corpus.name}", corpus.tgt_path)
        src_prefix, tgt_prefix = _prefixes(corpus, placement, kd_tag=kd_tag)
        section[corpus.name] = {
            "path_src": corpus.src_path.as_posix(),
            "path_tgt": corpus.tgt_path.as_posix(),
            "transforms": list(transforms),
            "weight": _weight(corpus.name, weights),
            "src_prefix": src_prefix,
            "tgt_prefix": tgt_prefix,
        }
    reject_dollar("source path of the validation corpus", valid_paths.src)
    reject_dollar("target path of the validation corpus", valid_paths.tgt)
    section["valid"] = {
        "path_src": valid_paths.src.as_posix(),
        "path_tgt": valid_paths.tgt.as_posix(),
    }
    return section
