"""The sources of a knowledge-distillation run (D10): the texts a teacher translates.

A source is the `src` text of an AUTHENTIC training row. The rows come from D7's
`load_training_rows`, which keeps only rows of the split `train` without a `filter_reason`
and leaves out the origins `ft`, `synth_llm` and `synth_nmt`; of those, only the group
`auth` (not the back-translated rows) is used here. So no dev, test or holdout row, no
filtered row, no back-translated row and no earlier forward translation is ever a source.

There is one source per direction and text, however often the corpus repeats the text and
whatever its targets are: `KdSources.references` keeps the authentic targets of each, which
the generation only COUNTS against (a diagnostic, never a filter). The id of a source is
`make_row_id(src_lang, tgt_lang, text, "")`: it depends on the direction and the text alone.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import polars as pl

from rinchen.corpus.schema import make_row_id
from rinchen.errors import ConfigError
from rinchen.train.mixture import AUTH_GROUP, TrainingRows

_COLUMNS = ["id", "lang", "text"]


@dataclass(frozen=True, slots=True, eq=False)
class KdSources:
    """The sources of one direction (`<src_lang>-<tgt_lang>`, registry codes).

    `frame` has exactly the String columns `id`, `lang` (always `src_lang`) and `text` and is
    sorted by `id`; `references` maps the id of a source to the authentic targets of its text.

    Raises:
        ValueError: `direction` is not `<src_lang>-<tgt_lang>`, `frame` has other columns than
            `id`, `lang` and `text`, or a row of it is not in `src_lang`.
    """

    direction: str
    src_lang: str
    tgt_lang: str
    frame: pl.DataFrame
    references: Mapping[str, frozenset[str]]

    def __post_init__(self) -> None:
        if self.direction != f"{self.src_lang}-{self.tgt_lang}":
            raise ValueError(
                f"direction {self.direction!r} is not '{self.src_lang}-{self.tgt_lang}'"
            )
        if self.frame.columns != _COLUMNS:
            raise ValueError(
                f"the sources of {self.direction} must have the columns {_COLUMNS}, "
                f"got {self.frame.columns}"
            )
        if (self.frame["lang"] != self.src_lang).fill_null(True).any():
            raise ValueError(
                f"the sources of {self.direction} hold a row whose lang is not "
                f"{self.src_lang!r}"
            )


def kd_sources(
    rows: TrainingRows, directions: Sequence[str] | None = None
) -> list[KdSources]:
    """The sources of the authentic training `rows`, one `KdSources` per direction, sorted
    by direction: all the directions that have authentic rows, or those of `directions`.

    Raises:
        ConfigError: `directions` is empty, or names a direction without authentic training
            rows; the message names it and the directions that have some.
    """
    wanted = _requested(rows.directions, directions)
    authentic = rows.frame.filter(pl.col("group") == AUTH_GROUP)
    return [
        _sources_of(direction, authentic.filter(pl.col("direction") == direction))
        for direction in wanted
    ]


def _requested(available: Sequence[str], directions: Sequence[str] | None) -> list[str]:
    if directions is None:
        return list(available)
    if not directions:
        raise ConfigError("no direction was requested for the sources")
    unknown = sorted(set(directions) - set(available))
    if unknown:
        raise ConfigError(
            f"no authentic training rows for direction(s) {unknown}; the directions "
            f"that have some: {list(available)}"
        )
    return sorted(set(directions))


def _sources_of(direction: str, rows: pl.DataFrame) -> KdSources:
    src_lang = rows["src_lang"][0]
    tgt_lang = rows["tgt_lang"][0]
    by_text = rows.group_by("src").agg(pl.col("tgt").unique())
    sources = sorted(
        (make_row_id(src_lang, tgt_lang, text, ""), text, targets)
        for text, targets in by_text.iter_rows()
    )
    frame = pl.DataFrame(
        {
            "id": [row_id for row_id, _, _ in sources],
            "lang": [src_lang] * len(sources),
            "text": [text for _, text, _ in sources],
        },
        schema={"id": pl.String, "lang": pl.String, "text": pl.String},
    )
    return KdSources(
        direction=direction,
        src_lang=src_lang,
        tgt_lang=tgt_lang,
        frame=frame,
        references={row_id: frozenset(targets) for row_id, _, targets in sources},
    )
