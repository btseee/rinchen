"""The Eole `data` section of one variant: a corpus entry per corpus file with its
paths, transforms, weight and the prefixes that carry the direction tag, and the
`valid` entry."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from rinchen.config.ablation import TagPlacement, require_placement
from rinchen.config.languages import direction_tag
from rinchen.errors import ConfigError
from rinchen.synth.tags import data_origin_tag
from rinchen.train.corpus_files import CorpusFiles
from rinchen.train.eole_cfg import reject_dollar
from rinchen.train.mixture import BT_GROUP, BT_ORIGIN
from rinchen.train.validation_corpus import ValidPaths


def _bt_tag() -> str:
    tag = data_origin_tag(BT_ORIGIN)
    if tag is None:
        raise ConfigError(f"the origin {BT_ORIGIN!r} has no data tag")
    return tag


def _prefixes(corpus: CorpusFiles, placement: str) -> tuple[str, str]:
    """(`src_prefix`, `tgt_prefix`) of a corpus: the tag of its target language and,
    for a back-translated corpus, the bt tag, on the side the placement names."""
    tag = direction_tag(corpus.tgt_lang)
    back_translated = corpus.group == BT_GROUP
    if placement == "src":
        return (f"{tag} {_bt_tag()}" if back_translated else tag), ""
    return (_bt_tag() if back_translated else ""), tag


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
    corpora: Sequence[CorpusFiles],
    weights: Mapping[str, int],
    placement: TagPlacement,
    *,
    valid_paths: ValidPaths,
    filtertoolong: bool,
) -> dict[str, Any]:
    """The Eole `data` mapping: one entry per corpus with its paths, `transforms`
    (`sentencepiece`, `prefix` and, when `filtertoolong`, `filtertoolong`), `weight`
    and the two prefixes that carry the direction tag, and `valid` with its two paths.

    The tag is that of the corpus's target language; with placement `src` it is the
    source prefix and a bt corpus adds `<bt>` after it, with `tgt` it is the target
    prefix and a bt corpus has `<bt>` alone as source prefix. Paths are written as
    given, as POSIX text. The text files carry no tags: the placement is configuration.

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
        src_prefix, tgt_prefix = _prefixes(corpus, placement)
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
