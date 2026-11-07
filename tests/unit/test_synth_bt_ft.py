"""The two job planners: which language the generator reads and writes and
which side of the pair is real, for back-translation and forward-translation,
and the input they refuse."""

from __future__ import annotations

import dataclasses
from collections.abc import Callable

import polars as pl
import pytest

from rinchen.config.synth import JobConfig, SynthConfig
from rinchen.errors import ConfigError, DataError
from rinchen.synth import bt, ft
from rinchen.synth.plan import Plan
from rinchen.synth.select import Kind
from tests.support.planted_synth import planted_monolingual

DEFAULT_JOBS = SynthConfig().jobs

Planner = Callable[[JobConfig, pl.DataFrame], Plan]


def _job(kind: Kind, mono_lang: str, other_lang: str) -> JobConfig:
    return JobConfig(
        kind=kind,
        mono_lang=mono_lang,
        other_lang=other_lang,
        generator="vllm_default",
    )


def test_the_default_back_translation_job_reads_mongolian_and_writes_english() -> None:
    plan = bt.plan(DEFAULT_JOBS["bt_eng_khk"], planted_monolingual())

    assert plan == Plan(
        kind="bt",
        generator_src_lang="khk_Cyrl",
        generator_tgt_lang="eng_Latn",
        real_lang="khk_Cyrl",
        synthetic_lang="eng_Latn",
    )


def test_the_default_forward_translation_job_reads_mongolian_and_writes_english() -> (
    None
):
    plan = ft.plan(DEFAULT_JOBS["ft_khk_eng"], planted_monolingual())

    assert plan == Plan(
        kind="ft",
        generator_src_lang="khk_Cyrl",
        generator_tgt_lang="eng_Latn",
        real_lang="khk_Cyrl",
        synthetic_lang="eng_Latn",
    )


@pytest.mark.parametrize(("planner", "kind"), [(bt.plan, "bt"), (ft.plan, "ft")])
def test_a_job_over_english_text_reads_english_and_writes_mongolian(
    planner: Planner, kind: Kind
) -> None:
    job = _job(kind, mono_lang="eng_Latn", other_lang="khk_Cyrl")

    plan = planner(job, planted_monolingual(3, lang="eng_Latn"))

    assert plan == Plan(
        kind=kind,
        generator_src_lang="eng_Latn",
        generator_tgt_lang="khk_Cyrl",
        real_lang="eng_Latn",
        synthetic_lang="khk_Cyrl",
    )


def test_a_plan_is_frozen() -> None:
    plan = bt.plan(DEFAULT_JOBS["bt_eng_khk"], planted_monolingual(2))

    with pytest.raises(dataclasses.FrozenInstanceError):
        plan.kind = "ft"  # type: ignore[misc]


def test_the_back_translation_planner_refuses_a_forward_translation_job() -> None:
    with pytest.raises(ConfigError, match=r"'ft' job.*'bt'"):
        bt.plan(DEFAULT_JOBS["ft_khk_eng"], planted_monolingual())


def test_the_forward_translation_planner_refuses_a_back_translation_job() -> None:
    with pytest.raises(ConfigError, match=r"'bt' job.*'ft'"):
        ft.plan(DEFAULT_JOBS["bt_eng_khk"], planted_monolingual())


@pytest.mark.parametrize(
    ("planner", "job_name"), [(bt.plan, "bt_eng_khk"), (ft.plan, "ft_khk_eng")]
)
def test_a_planner_refuses_a_frame_without_rows(
    planner: Planner, job_name: str
) -> None:
    empty = planted_monolingual(1).clear()

    with pytest.raises(DataError, match="no monolingual rows"):
        planner(DEFAULT_JOBS[job_name], empty)


@pytest.mark.parametrize(
    ("planner", "job_name"), [(bt.plan, "bt_eng_khk"), (ft.plan, "ft_khk_eng")]
)
def test_a_planner_refuses_a_frame_in_another_language_naming_it(
    planner: Planner, job_name: str
) -> None:
    english = planted_monolingual(3, lang="eng_Latn")

    with pytest.raises(DataError, match=r"3 of 3 .*'khk_Cyrl'.*'eng_Latn'"):
        planner(DEFAULT_JOBS[job_name], english)


def test_a_planner_refuses_rows_in_other_languages_naming_the_first() -> None:
    frame = planted_monolingual(3).with_columns(
        lang=pl.Series(["rus_Cyrl", "khk_Cyrl", "eng_Latn"])
    )

    with pytest.raises(DataError, match=r"2 of 3 .*first: 'rus_Cyrl'"):
        bt.plan(DEFAULT_JOBS["bt_eng_khk"], frame)


def test_a_planner_refuses_a_row_without_a_language() -> None:
    frame = planted_monolingual(2).with_columns(lang=pl.Series(["khk_Cyrl", None]))

    with pytest.raises(DataError, match=r"1 of 2 .*None"):
        ft.plan(DEFAULT_JOBS["ft_khk_eng"], frame)


def test_a_planner_refuses_a_frame_without_a_language_column() -> None:
    frame = planted_monolingual(2).drop("lang")

    with pytest.raises(DataError, match="'lang' column"):
        bt.plan(DEFAULT_JOBS["bt_eng_khk"], frame)
