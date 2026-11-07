"""Shared setup of the `rinchen synth run` tests: a job config with small shards
and the CLI's two seams replaced by fakes. The planted inputs themselves come
from `tests.support.planted_synth`."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest
import yaml

from rinchen.cli import synth as synth_cli
from rinchen.config.synth import GeneratorConfig, SelectionConfig
from rinchen.filter.qe import QEScorer
from rinchen.synth.translator import Translator

BT_JOB = "bt_eng_khk"
FT_JOB = "ft_khk_eng"
GENERATOR = "vllm_default"


def write_job_config(
    path: Path,
    *,
    shard_size: int = 8,
    generator: Mapping[str, Any] | None = None,
    generator_name: str = GENERATOR,
    unused_generators: Sequence[str] = (),
) -> Path:
    """A synth config with the generator `generator_name` and the two default
    jobs (`bt_eng_khk` and `ft_khk_eng`) cut into shards of `shard_size` rows.
    The generator is a vllm one unless `generator` says otherwise.
    `unused_generators` names more vllm generators, listed before it, that no job
    uses."""
    jobs = {
        name: {
            "kind": kind,
            "mono_lang": "khk_Cyrl",
            "other_lang": "eng_Latn",
            "generator": generator_name,
            "shard_size": shard_size,
        }
        for name, kind in ((BT_JOB, "bt"), (FT_JOB, "ft"))
    }
    generators: dict[str, Any] = {
        name: {"kind": "vllm", "base_url": f"http://{name}.invalid/v1"}
        for name in unused_generators
    }
    generators[generator_name] = {"kind": "vllm", **(generator or {})}
    config = {"generators": generators, "jobs": jobs}
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return path


@dataclass(slots=True)
class Seams:
    """What the CLI's seams were asked for, and the fakes they hand out."""

    translator: Translator
    scorer: QEScorer | None
    generators_built: list[GeneratorConfig] = field(default_factory=list)
    selections_built: list[SelectionConfig] = field(default_factory=list)


def install_seams(
    monkeypatch: pytest.MonkeyPatch,
    *,
    translator: Translator,
    scorer: QEScorer | None,
) -> Seams:
    """Make `rinchen synth run` use `translator` and `scorer`, so that no
    server, model or GPU is involved."""
    seams = Seams(translator, scorer)

    def build_translator(cfg: GeneratorConfig) -> Translator:
        seams.generators_built.append(cfg)
        return seams.translator

    def build_qe_scorer(cfg: SelectionConfig) -> QEScorer | None:
        seams.selections_built.append(cfg)
        return seams.scorer

    monkeypatch.setattr(synth_cli, "build_translator", build_translator)
    monkeypatch.setattr(synth_cli, "build_qe_scorer", build_qe_scorer)
    return seams
