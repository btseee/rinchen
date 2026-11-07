"""The translation direction of a synthetic-data job: which language the
generator reads and writes and which side of the pair is real. Back- and
forward-translation share it; `bt` and `ft` add the kind, which decides the
orientation of the corpus rows built from the generator's output."""

from __future__ import annotations

from dataclasses import dataclass

import polars as pl

from rinchen.config.synth import JobConfig
from rinchen.errors import ConfigError, DataError
from rinchen.synth.select import Kind


@dataclass(frozen=True, slots=True)
class Plan:
    """The generator reads `generator_src_lang` text and writes
    `generator_tgt_lang`. `real_lang` is the language of the monolingual rows
    and `synthetic_lang` the one the generator writes, the same two languages
    in the other roles."""

    kind: Kind
    generator_src_lang: str
    generator_tgt_lang: str
    real_lang: str
    synthetic_lang: str


def plan_job(kind: Kind, job: JobConfig, frame: pl.DataFrame) -> Plan:
    """The plan of `job`, which must be a `kind` job over `frame`: a
    monolingual frame (columns `id`, `lang`, `text`) with at least one row,
    all in `job.mono_lang`. In both kinds the generator translates the
    monolingual text into `job.other_lang`.

    Raises ConfigError for a job of another kind and DataError for a frame
    without rows, without a `lang` column or with rows in another language."""
    if job.kind != kind:
        raise ConfigError(f"a {job.kind!r} job cannot be planned as {kind!r}")
    if frame.is_empty():
        raise DataError("no monolingual rows to translate")
    _require_language(frame, job.mono_lang)
    return Plan(
        kind=kind,
        generator_src_lang=job.mono_lang,
        generator_tgt_lang=job.other_lang,
        real_lang=job.mono_lang,
        synthetic_lang=job.other_lang,
    )


def _require_language(frame: pl.DataFrame, language: str) -> None:
    if "lang" not in frame.columns:
        raise DataError("the monolingual frame has no 'lang' column")
    foreign = frame.filter(~pl.col("lang").eq_missing(language))["lang"]
    if len(foreign):
        raise DataError(
            f"{len(foreign)} of {frame.height} monolingual rows are not in "
            f"{language!r} (first: {foreign[0]!r})"
        )
