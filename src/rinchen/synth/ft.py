"""Forward translation: the real monolingual text is the source of each pair
and the generator writes its target. The generator translates the monolingual
language into the other one, and the translation becomes the `tgt` of the
corpus row."""

from __future__ import annotations

import polars as pl

from rinchen.config.synth import JobConfig
from rinchen.synth.plan import Plan, plan_job


def plan(job: JobConfig, frame: pl.DataFrame) -> Plan:
    """The plan of an `ft` job over the monolingual `frame`; see `plan_job`."""
    return plan_job("ft", job, frame)
