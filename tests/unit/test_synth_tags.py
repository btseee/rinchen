# tests/unit/test_synth_tags.py
from __future__ import annotations

import pytest

from rinchen.config.synth import SynthConfig
from rinchen.corpus.schema import ORIGINS
from rinchen.errors import ConfigError
from rinchen.spm.symbols import STRUCTURAL_SYMBOLS
from rinchen.synth.tags import data_origin_tag, origin_for


@pytest.mark.parametrize(
    ("kind", "generator_kind", "origin"),
    [
        ("bt", "vllm", "bt"),
        ("bt", "llamacpp", "bt"),
        ("bt", "ct2", "bt"),
        ("ft", "vllm", "synth_llm"),
        ("ft", "llamacpp", "synth_llm"),
        ("ft", "ct2", "synth_nmt"),
    ],
)
def test_origin_for_gives_the_role_and_generator_family(
    kind: str, generator_kind: str, origin: str
):
    assert origin_for(kind, generator_kind) == origin


@pytest.mark.parametrize(
    ("kind", "generator_kind", "unknown"),
    [
        ("rt", "vllm", "rt"),
        ("", "vllm", ""),
        ("BT", "vllm", "BT"),
        ("ft", "grpc", "grpc"),
        ("bt", "grpc", "grpc"),
        ("ft", "", ""),
    ],
)
def test_origin_for_rejects_an_unknown_kind_or_generator_kind(
    kind: str, generator_kind: str, unknown: str
):
    with pytest.raises(ConfigError, match="unknown") as excinfo:
        origin_for(kind, generator_kind)

    assert repr(unknown) in str(excinfo.value)


@pytest.mark.parametrize(
    ("origin", "tag"),
    [
        ("bt", "<bt>"),
        ("ft", "<ft>"),
        ("synth_llm", "<ft>"),
        ("synth_nmt", "<ft>"),
        ("human", None),
        ("web", None),
        ("mined", None),
        ("converted", None),
    ],
)
def test_data_origin_tag_marks_only_synthetic_origins(origin: str, tag: str | None):
    assert data_origin_tag(origin) == tag


@pytest.mark.parametrize("origin", ["synthetic", "", "BT", "<bt>"])
def test_data_origin_tag_rejects_an_unknown_origin(origin: str):
    with pytest.raises(ConfigError, match="unknown") as excinfo:
        data_origin_tag(origin)

    assert repr(origin) in str(excinfo.value)


def test_every_configurable_job_and_generator_kind_has_an_origin():
    generator_kinds = {g.kind for g in SynthConfig().generators.values()}
    job_kinds = {job.kind for job in SynthConfig().jobs.values()}

    assert generator_kinds == {"vllm", "llamacpp", "ct2"}
    assert job_kinds == {"bt", "ft"}
    for kind in job_kinds:
        for generator_kind in generator_kinds:
            assert origin_for(kind, generator_kind) in ORIGINS


def test_the_data_tags_are_structural_sentencepiece_symbols():
    tags = {data_origin_tag(origin) for origin in ORIGINS} - {None}

    assert tags == {"<bt>", "<ft>"}
    assert tags <= set(STRUCTURAL_SYMBOLS)
