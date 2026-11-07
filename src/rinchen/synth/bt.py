"""Back-translation: the real monolingual text is the target of each pair and
the generator writes its source. The generator translates the monolingual
language into the other one, and the translation becomes the `src` of the
corpus row."""

from __future__ import annotations

import polars as pl

from rinchen.config.synth import JobConfig
from rinchen.synth.plan import Plan, plan_job


def plan(job: JobConfig, frame: pl.DataFrame) -> Plan:
    """The plan of a `bt` job over the monolingual `frame`; see `plan_job`."""
    return plan_job("bt", job, frame)
