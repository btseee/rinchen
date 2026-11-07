"""Shared setup of the `rinchen synth run` tests: a job config with small shards,
a workspace holding the planted input and the paths of a run, and the CLI's two
seams replaced by fakes. The planted inputs themselves come from
`tests.support.planted_synth`."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import polars as pl
import pytest
import yaml

from rinchen.cli import synth as synth_cli
from rinchen.config.synth import GeneratorConfig, SelectionConfig, load_synth_config
from rinchen.filter.qe import QEScorer
from rinchen.synth.shards import job_fingerprint
from rinchen.synth.translator import Translator
from tests.support.planted_synth import planted_monolingual

BT_JOB = "bt_eng_khk"
FT_JOB = "ft_khk_eng"
GENERATOR = "vllm_default"
SEED = 0
SHARD_SIZE = 8


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


@dataclass(frozen=True, slots=True)
class Workspace:
    """The files of one `rinchen synth run` test: the planted monolingual file,
    the config (shards of `SHARD_SIZE` rows) and the output directory, which
    does not exist yet."""

    monolingual: Path
    config: Path
    out_dir: Path
    frame: pl.DataFrame

    @property
    def corpus(self) -> Path:
        return self.out_dir / "corpus.parquet"

    @property
    def counts(self) -> Path:
        return self.out_dir / "counts.json"

    @property
    def manifest(self) -> Path:
        return self.out_dir / "manifest.json"

    def args(
        self,
        job: str = BT_JOB,
        *extra: str,
        monolingual: Path | None = None,
        out_dir: Path | None = None,
        config: Path | None = None,
    ) -> list[str]:
        return [
            *self.args_without_out_dir(job, monolingual=monolingual, config=config),
            "--out-dir",
            str(self.out_dir if out_dir is None else out_dir),
            *extra,
        ]

    def args_without_out_dir(
        self,
        job: str = BT_JOB,
        *,
        monolingual: Path | None = None,
        config: Path | None = None,
    ) -> list[str]:
        return [
            "synth",
            "run",
            job,
            "--input",
            str(self.monolingual if monolingual is None else monolingual),
            "--config",
            str(self.config if config is None else config),
        ]

    def corpus_frame(self, out_dir: Path | None = None) -> pl.DataFrame:
        return pl.read_parquet(
            (self.out_dir if out_dir is None else out_dir) / "corpus.parquet"
        )

    def shard_directories(self) -> list[Path]:
        return sorted(self.out_dir.glob("shards-*"))

    def shard_texts(self, *indexes: int) -> list[tuple[str, ...]]:
        texts = self.frame["text"].to_list()
        return [
            tuple(texts[index * SHARD_SIZE : (index + 1) * SHARD_SIZE])
            for index in indexes
        ]

    def counts_document(self) -> dict[str, Any]:
        return json.loads(self.counts.read_text(encoding="utf-8"))

    def fingerprint(self, job: str = BT_JOB, seed: int = SEED) -> str:
        cfg = load_synth_config(self.config)
        job_cfg = cfg.jobs[job]
        generator = cfg.generators[job_cfg.generator]
        return job_fingerprint(self.frame, job_cfg, generator, job_cfg.shard_size, seed)


def make_workspace(tmp_path: Path) -> Workspace:
    """The planted 40-row frame written as a Parquet file under `tmp_path`, with
    a config of `SHARD_SIZE`-row shards beside it and an output directory that
    is not created."""
    frame = planted_monolingual()
    monolingual = tmp_path / "in" / "mono.parquet"
    monolingual.parent.mkdir()
    frame.write_parquet(monolingual)
    return Workspace(
        monolingual=monolingual,
        config=write_job_config(tmp_path / "synth.yaml", shard_size=SHARD_SIZE),
        out_dir=tmp_path / "out",
        frame=frame,
    )


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
