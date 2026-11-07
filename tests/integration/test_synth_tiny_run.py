"""The tiny synthetic-data run: the D6 STOP gate. The planted 40-row Mongolian
file goes through `rinchen synth run` for the two default jobs (back- and
forward-translation with the default generator), in process through the CLI,
with a scripted generator and a planted scorer behind the seams. The counts of
what was kept and dropped for each reason are exact; the corpus validates
against the contract; a run that dies after shard 1 resumes through the CLI and
ends with the corpus of an uninterrupted run.

No server, model or GPU is involved and nothing is written outside pytest's
temporary directories.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
import polars as pl
import pytest
from click.testing import Result
from polars.testing import assert_frame_equal
from typer.testing import CliRunner

from rinchen.cli import synth as synth_cli
from rinchen.cli.main import app
from rinchen.config.synth import GeneratorConfig, VllmConfig, load_synth_config
from rinchen.corpus.schema import CORPUS_SCHEMA, validate_corpus_schema
from rinchen.errors import UpstreamError
from rinchen.manifest import store
from rinchen.synth.shards import job_fingerprint
from rinchen.synth.vllm_client import OpenAICompatTranslator
from tests.support.fake_llm_server import FakeLlmServer, table_responder
from tests.support.planted_synth import (
    ScriptedTranslator,
    planted_monolingual,
    planted_scorer,
    planted_script,
)
from tests.support.synth_cli import (
    BT_JOB,
    FT_JOB,
    GENERATOR,
    Seams,
    install_seams,
    write_job_config,
)

runner = CliRunner()

PLANTED_COUNTS = {
    "sources": 40,
    "candidates": 60,
    "dropped_empty": 5,
    "dropped_purity": 10,
    "dropped_qe": 10,
    "dropped_copy": 0,
    "no_candidates": 5,
    "no_survivor": 20,
    "selected": 15,
}
# The rows the selection keeps: those whose position modulo 8 is 0, 6 or 7, with
# the winning candidate of each (the planted outcomes are in
# `tests.support.planted_synth`).
KEPT_POSITIONS = [i for i in range(40) if i % 8 in (0, 6, 7)]
WINNERS = [
    {
        0: "The state of Mongolia {n} stands",
        6: "Mongolia is edge {n}",
        7: "Mongolia is a country {n} indeed",
    }[i % 8].format(n=f"{i:03d}")
    for i in KEPT_POSITIONS
]
MONGOLIAN = [f"Монгол улс {i:03d}" for i in KEPT_POSITIONS]
SHARD_SIZE = 8


@dataclass(frozen=True, slots=True)
class Run:
    monolingual: Path
    config: Path
    out_dir: Path
    frame: pl.DataFrame

    def args(
        self,
        job: str,
        *extra: str,
        out_dir: Path | None = None,
        config: Path | None = None,
    ) -> list[str]:
        return [
            "synth",
            "run",
            job,
            "--input",
            str(self.monolingual),
            "--config",
            str(config or self.config),
            "--out-dir",
            str(out_dir or self.out_dir),
            *extra,
        ]

    def corpus(self, out_dir: Path | None = None) -> pl.DataFrame:
        return pl.read_parquet((out_dir or self.out_dir) / "corpus.parquet")

    def counts(self) -> dict[str, Any]:
        return json.loads((self.out_dir / "counts.json").read_text(encoding="utf-8"))


@pytest.fixture
def run(tmp_path: Path) -> Run:
    frame = planted_monolingual()
    monolingual = tmp_path / "in" / "khk_mono.parquet"
    monolingual.parent.mkdir()
    frame.write_parquet(monolingual)
    return Run(
        monolingual=monolingual,
        config=write_job_config(tmp_path / "synth.yaml", shard_size=SHARD_SIZE),
        out_dir=tmp_path / "out",
        frame=frame,
    )


def _planted(monkeypatch: pytest.MonkeyPatch, run: Run, kind: str) -> Seams:
    return install_seams(
        monkeypatch,
        translator=ScriptedTranslator(planted_script(run.frame)),
        scorer=planted_scorer(run.frame, kind),
    )


def _invoke_ok(args: list[str]) -> Result:
    result = runner.invoke(app, args)
    assert result.exit_code == 0, result.output
    return result


def _without_created(frame: pl.DataFrame) -> pl.DataFrame:
    return frame.drop("created")


@pytest.mark.parametrize(
    ("job", "kind", "origin"), [(BT_JOB, "bt", "bt"), (FT_JOB, "ft", "synth_llm")]
)
def test_the_tiny_run_keeps_and_drops_exactly_the_planted_rows(
    run: Run, monkeypatch: pytest.MonkeyPatch, job: str, kind: str, origin: str
) -> None:
    _planted(monkeypatch, run, kind)

    result = _invoke_ok(run.args(job))

    assert run.counts()["counts"] == PLANTED_COUNTS
    assert result.stdout.splitlines()[-1] == (
        "rows 15 (selected 15 of 40 sources, qe ok)"
    )
    manifest = store.read(run.out_dir)
    assert manifest is not None
    assert manifest.stage == "synth"
    assert manifest.status == {"qe": "ok"}
    assert [(item.path, item.rows) for item in manifest.outputs] == [
        (str(run.out_dir / "corpus.parquet"), 15),
        (str(run.out_dir / "counts.json"), None),
    ]
    corpus = run.corpus()
    assert corpus.schema == CORPUS_SCHEMA
    validate_corpus_schema(corpus.schema, source="the tiny synthetic run")
    assert corpus["origin"].cast(pl.String).unique().to_list() == [origin]
    assert corpus["corpus"].unique().to_list() == [f"{kind}:vllm_default"]
    assert corpus["split"].cast(pl.String).unique().to_list() == ["train"]
    assert corpus["id"].n_unique() == 15


def test_a_back_translation_run_pairs_the_synthetic_english_source_with_the_real_target(
    run: Run, monkeypatch: pytest.MonkeyPatch
) -> None:
    _planted(monkeypatch, run, "bt")

    _invoke_ok(run.args(BT_JOB))

    corpus = run.corpus()
    assert corpus["src"].to_list() == WINNERS
    assert corpus["tgt"].to_list() == MONGOLIAN
    assert corpus["src_lang"].unique().to_list() == ["eng_Latn"]
    assert corpus["tgt_lang"].unique().to_list() == ["khk_Cyrl"]


def test_a_forward_translation_run_pairs_the_real_source_with_the_synthetic_english_target(
    run: Run, monkeypatch: pytest.MonkeyPatch
) -> None:
    _planted(monkeypatch, run, "ft")

    _invoke_ok(run.args(FT_JOB))

    corpus = run.corpus()
    assert corpus["src"].to_list() == MONGOLIAN
    assert corpus["tgt"].to_list() == WINNERS
    assert corpus["src_lang"].unique().to_list() == ["khk_Cyrl"]
    assert corpus["tgt_lang"].unique().to_list() == ["eng_Latn"]


@pytest.mark.parametrize(("job", "kind"), [(BT_JOB, "bt"), (FT_JOB, "ft")])
def test_a_run_that_dies_after_shard_1_resumes_to_the_uninterrupted_corpus(
    run: Run, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, job: str, kind: str
) -> None:
    crashing = ScriptedTranslator(
        planted_script(run.frame), fail_on_call={2: UpstreamError("server down")}
    )
    install_seams(
        monkeypatch, translator=crashing, scorer=planted_scorer(run.frame, kind)
    )

    first = runner.invoke(app, run.args(job, "--seed", "7"))

    assert first.exit_code == 4, first.output
    assert first.stderr.splitlines() == [
        "shard 1/5 generated",
        "shard 2/5 generated",
        "error: shard 2 of 5 failed: server down",
    ]
    assert not (run.out_dir / "corpus.parquet").exists()
    assert not (run.out_dir / "manifest.json").exists()
    (shard_directory,) = run.out_dir.glob("shards-*")
    assert sorted(path.name for path in shard_directory.iterdir()) == [
        "shard-00000.parquet",
        "shard-00001.parquet",
    ]

    resumed = _planted(monkeypatch, run, kind)
    second = _invoke_ok(run.args(job, "--seed", "7"))

    texts = run.frame["text"].to_list()
    assert isinstance(resumed.translator, ScriptedTranslator)
    assert [requested for requested, _, _ in resumed.translator.calls] == [
        tuple(texts[start : start + SHARD_SIZE]) for start in (16, 24, 32)
    ]
    assert second.stderr.splitlines() == [
        "shard 1/5 reused",
        "shard 2/5 reused",
        "shard 3/5 generated",
        "shard 4/5 generated",
        "shard 5/5 generated",
    ]
    assert run.counts()["counts"] == PLANTED_COUNTS
    assert run.counts()["shard_stats"] == {
        "planned": 5,
        "reused": 2,
        "generated": 3,
        "regenerated_invalid": 0,
    }
    manifest = store.read(run.out_dir)
    assert manifest is not None
    assert manifest.seed == 7
    cfg = load_synth_config(run.config)
    fingerprint = job_fingerprint(
        run.frame, cfg.jobs[job], cfg.generators[GENERATOR], SHARD_SIZE, 7
    )
    assert run.counts()["fingerprint"] == fingerprint
    assert shard_directory.name == f"shards-{fingerprint[:12]}"

    uninterrupted = tmp_path / "uninterrupted"
    _planted(monkeypatch, run, kind)
    _invoke_ok(run.args(job, "--seed", "7", out_dir=uninterrupted))
    assert_frame_equal(
        _without_created(run.corpus()),
        _without_created(run.corpus(uninterrupted)),
    )


def _use_the_vllm_client_over(
    monkeypatch: pytest.MonkeyPatch, run: Run, server: FakeLlmServer
) -> None:
    """The real vLLM client, talking to `server`, behind the translator seam;
    the planted scorer behind the other."""

    def build_translator(cfg: GeneratorConfig) -> OpenAICompatTranslator:
        assert isinstance(cfg, VllmConfig)
        return OpenAICompatTranslator(
            cfg, client=httpx.Client(transport=server.transport)
        )

    monkeypatch.setattr(synth_cli, "build_translator", build_translator)
    monkeypatch.setattr(
        synth_cli, "build_qe_scorer", lambda cfg: planted_scorer(run.frame, "bt")
    )


def test_the_real_vllm_client_over_a_fake_server_gives_the_same_rows(
    run: Run, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _planted(monkeypatch, run, "bt")
    _invoke_ok(run.args(BT_JOB, out_dir=tmp_path / "scripted"))
    server = FakeLlmServer(table_responder(planted_script(run.frame)))
    _use_the_vllm_client_over(monkeypatch, run, server)

    _invoke_ok(run.args(BT_JOB))

    assert len(server.requests) == 40
    assert run.counts()["counts"]["selected"] == 15
    assert_frame_equal(
        _without_created(run.corpus()),
        _without_created(run.corpus(tmp_path / "scripted")),
    )


def test_a_server_that_keeps_failing_ends_the_run_with_exit_4_and_no_output(
    run: Run, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = write_job_config(
        tmp_path / "no_retries.yaml", shard_size=SHARD_SIZE, generator={"retries": 0}
    )
    server = FakeLlmServer(
        table_responder(planted_script(run.frame)), failures=[503] * 100
    )
    _use_the_vllm_client_over(monkeypatch, run, server)

    result = runner.invoke(app, run.args(BT_JOB, config=config))

    assert result.exit_code == 4, result.output
    errors = [line for line in result.stderr.splitlines() if line.startswith("error:")]
    assert len(errors) == 1
    assert errors[0].startswith("error: shard 0 of 5 failed: ")
    # The first shard failed, so nothing was written at all.
    assert not run.out_dir.exists()
