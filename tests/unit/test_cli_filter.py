"""`rinchen filter run`: the tiny end-to-end run over the planted-failure
corpus (the D4 filter STOP gate), the manifest it writes, skipping, and the
exit code of each failure.

The LaBSE embedder and the QE scorer come from module-level factories in the
CLI; tests replace them with the concept embedder and no scorer, so no model is
downloaded. Everything lands under pytest's temporary directories.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import polars as pl
import pytest
import yaml
from typer.testing import CliRunner

from rinchen.cli import filter as filter_cli
from rinchen.cli.main import app
from rinchen.config.filters import FilterConfig, LabseConfig, QeConfig
from rinchen.corpus.readers import scan_corpus
from rinchen.corpus.schema import corpus_frame
from rinchen.corpus.writers import write_corpus
from rinchen.errors import UpstreamError
from rinchen.filter.qe_eole import EoleMetricXQEScorer
from rinchen.filter.reasons import FilterReason
from rinchen.manifest import store
from rinchen.manifest.model import Manifest
from rinchen.manifest.stage import sha256_of
from rinchen.mine.embed import SentenceTransformerEmbedder
from tests.support.embedders import ConceptEmbedder
from tests.support.planted_corpus import (
    PLANTED_REASON_COUNTS,
    PlantedCorpus,
    build_planted_corpus,
    tiny_spm_model,
)
from tests.support.tiny_lid import train_tiny_lid_model

runner = CliRunner()

CLEAN_ROWS = 28
PLANTED_ROWS = 50
NO_SCORER_STATUS = "skipped: no QE scorer configured"
OPTIONS = ["--corpus", "--config", "--out-dir", "--force", "--dry-run", "--seed"]
TOOL_PACKAGES = [
    "polars",
    "datasketch",
    "fasttext-new",
    "sentence-transformers",
    "sentencepiece",
    "torch",
    "eole",
]


@dataclass(frozen=True, slots=True)
class Workspace:
    corpus: Path
    config: Path
    out_dir: Path
    lid_model: Path
    spm_model: Path

    @property
    def scored(self) -> Path:
        return self.out_dir / "scored.parquet"

    @property
    def kept(self) -> Path:
        return self.out_dir / "kept.parquet"

    @property
    def manifest(self) -> Path:
        return self.out_dir / "manifest.json"

    def args(
        self, *extra: str, config: Path | None = None, out_dir: Path | None = None
    ) -> list[str]:
        return [
            "filter",
            "run",
            "--corpus",
            str(self.corpus),
            "--config",
            str(self.config if config is None else config),
            "--out-dir",
            str(self.out_dir if out_dir is None else out_dir),
            *extra,
        ]


class SpyQeScorer(EoleMetricXQEScorer):
    """The Eole scorer type, which the CLI releases after a run, with the model
    replaced by a constant score (or a failure) and the release counted."""

    requires_cuda = False

    def __init__(self, *, fail: bool = False) -> None:
        super().__init__()
        self.fail = fail
        self.releases = 0

    def score(self, sources: Sequence[str], translations: Sequence[str]) -> list[float]:
        if self.fail:
            raise UpstreamError("scorer unavailable")
        return [0.0] * len(sources)

    def release(self) -> None:
        self.releases += 1


@pytest.fixture(scope="module")
def planted() -> PlantedCorpus:
    return build_planted_corpus()


@pytest.fixture(scope="module")
def lid_model(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return train_tiny_lid_model(tmp_path_factory.mktemp("cli_lid") / "lid.bin")


@pytest.fixture(scope="module")
def spm_model(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return tiny_spm_model(tmp_path_factory.mktemp("cli_spm"))


@pytest.fixture
def fake_models(monkeypatch: pytest.MonkeyPatch, planted: PlantedCorpus) -> None:
    monkeypatch.setattr(
        filter_cli, "build_embedder", lambda cfg: ConceptEmbedder(planted.concepts)
    )
    monkeypatch.setattr(filter_cli, "build_qe_scorer", lambda cfg: None)


@pytest.fixture
def workspace(
    tmp_path: Path,
    planted: PlantedCorpus,
    lid_model: Path,
    spm_model: Path,
    fake_models: None,
) -> Workspace:
    corpus = tmp_path / "planted.parquet"
    write_corpus(planted.frame, corpus)
    return Workspace(
        corpus=corpus,
        config=_write_config(tmp_path / "filters.yaml", lid_model, spm_model),
        out_dir=tmp_path / "out",
        lid_model=lid_model,
        spm_model=spm_model,
    )


def _write_config(
    path: Path, lid_model: Path, spm_model: Path, **overrides: Any
) -> Path:
    """The default pipeline with the tiny LID and SentencePiece models."""
    fields = FilterConfig().model_dump(mode="json")
    fields["lid"]["model_path"] = str(lid_model)
    fields["length"]["spm_model"] = str(spm_model)
    fields.update(overrides)
    path.write_text(yaml.safe_dump(fields), encoding="utf-8")
    return path


def _manifest(workspace: Workspace) -> Manifest:
    manifest = store.read(workspace.out_dir)
    assert manifest is not None
    return manifest


def _planted_reason_metrics() -> dict[str, float]:
    return {
        f"reason.{reason.value}": float(PLANTED_REASON_COUNTS.get(reason.value, 0))
        for reason in FilterReason
    }


def test_tiny_filter_run_scores_and_filters_the_planted_corpus(
    workspace: Workspace, planted: PlantedCorpus
) -> None:
    result = runner.invoke(app, workspace.args())

    assert result.exit_code == 0, result.output
    assert result.stdout.splitlines()[-1] == f"manifest: {workspace.manifest}"
    manifest = _manifest(workspace)
    assert manifest.metrics == {
        "rows_in": float(PLANTED_ROWS),
        "rows_kept": float(CLEAN_ROWS),
        **_planted_reason_metrics(),
    }
    assert manifest.status == {
        "script_ratio": "ok",
        "numbers": "ok",
        "length": "ok",
        "lid": "ok",
        "minhash": "ok",
        "labse": "ok",
        "qe": NO_SCORER_STATUS,
    }


def test_scored_parquet_holds_every_row_with_its_planted_reason(
    workspace: Workspace, planted: PlantedCorpus
) -> None:
    runner.invoke(app, workspace.args())

    scored = scan_corpus(workspace.scored).collect()
    assert (
        dict(
            zip(scored["id"].to_list(), scored["filter_reason"].to_list(), strict=True)
        )
        == planted.expected_reason
    )


def test_kept_parquet_holds_exactly_the_clean_rows(
    workspace: Workspace, planted: PlantedCorpus
) -> None:
    runner.invoke(app, workspace.args())

    kept = scan_corpus(workspace.kept).collect()
    clean_ids = [i for i, reason in planted.expected_reason.items() if reason is None]
    assert kept["id"].to_list() == clean_ids
    assert len(clean_ids) == CLEAN_ROWS
    assert kept["filter_reason"].null_count() == CLEAN_ROWS


def test_manifest_records_inputs_outputs_config_and_environment(
    workspace: Workspace,
) -> None:
    runner.invoke(app, workspace.args())

    manifest = _manifest(workspace)
    assert manifest.stage == "filter"
    assert manifest.seed == 0
    assert manifest.config_sha256 == sha256_of(workspace.config)
    assert manifest.code_version
    created = datetime.fromisoformat(manifest.created)
    assert created.utcoffset() == timedelta(0)
    assert [(item.path, item.sha256) for item in manifest.inputs] == [
        (str(path), sha256_of(path))
        for path in (workspace.corpus, workspace.lid_model, workspace.spm_model)
    ]
    assert [
        (item.path, item.rows, item.bytes, item.sha256) for item in manifest.outputs
    ] == [
        (str(path), rows, path.stat().st_size, sha256_of(path))
        for path, rows in ((workspace.scored, PLANTED_ROWS), (workspace.kept, 28))
    ]
    assert list(manifest.tool_versions) == TOOL_PACKAGES
    assert manifest.tool_versions["polars"] == pl.__version__


def test_seed_option_is_recorded_in_the_manifest(workspace: Workspace) -> None:
    result = runner.invoke(app, workspace.args("--seed", "7"))

    assert result.exit_code == 0, result.output
    assert _manifest(workspace).seed == 7


def test_models_of_unused_stages_are_not_listed_as_inputs(
    workspace: Workspace, tmp_path: Path
) -> None:
    config = _write_config(
        tmp_path / "numbers_only.yaml",
        workspace.lid_model,
        workspace.spm_model,
        stages=["numbers"],
    )

    result = runner.invoke(app, workspace.args(config=config))

    assert result.exit_code == 0, result.output
    assert [item.path for item in _manifest(workspace).inputs] == [
        str(workspace.corpus)
    ]


def test_a_rerun_with_unchanged_inputs_skips(workspace: Workspace) -> None:
    runner.invoke(app, workspace.args())
    manifest_before = workspace.manifest.read_bytes()
    scored_before = workspace.scored.stat().st_mtime_ns

    result = runner.invoke(app, workspace.args())

    assert result.exit_code == 0, result.output
    assert result.stdout.splitlines()[0].startswith("skipped:")
    assert result.stdout.splitlines()[-1] == f"manifest: {workspace.manifest}"
    assert workspace.manifest.read_bytes() == manifest_before
    assert workspace.scored.stat().st_mtime_ns == scored_before


def test_force_reruns_even_when_nothing_changed(workspace: Workspace) -> None:
    runner.invoke(app, workspace.args())
    created_before = _manifest(workspace).created

    result = runner.invoke(app, workspace.args("--force"))

    assert result.exit_code == 0, result.output
    assert not result.stdout.startswith("skipped:")
    assert _manifest(workspace).created != created_before


def test_a_deleted_output_is_rebuilt_instead_of_skipped(workspace: Workspace) -> None:
    runner.invoke(app, workspace.args())
    workspace.kept.unlink()

    result = runner.invoke(app, workspace.args())

    assert result.exit_code == 0, result.output
    assert not result.stdout.startswith("skipped:")
    assert scan_corpus(workspace.kept).collect().height == CLEAN_ROWS


def test_a_changed_config_reruns(workspace: Workspace) -> None:
    runner.invoke(app, workspace.args())
    config = _write_config(
        workspace.config,
        workspace.lid_model,
        workspace.spm_model,
        script_ratio={"min_ratio": 0.5},
    )
    assert config == workspace.config

    result = runner.invoke(app, workspace.args())

    assert result.exit_code == 0, result.output
    assert not result.stdout.startswith("skipped:")
    assert _manifest(workspace).config_sha256 == sha256_of(workspace.config)


def test_dry_run_prints_the_stage_plan_and_writes_nothing(
    workspace: Workspace,
) -> None:
    result = runner.invoke(app, workspace.args("--dry-run"))

    assert result.exit_code == 0, result.output
    assert result.stdout.splitlines() == [
        "script_ratio",
        "numbers",
        "length",
        "lid",
        "minhash",
        "labse",
        f"qe: {NO_SCORER_STATUS}",
    ]
    assert not workspace.out_dir.exists()


def test_stages_that_are_not_configured_build_no_models(
    workspace: Workspace,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    built: list[str] = []
    monkeypatch.setattr(
        filter_cli, "build_embedder", lambda cfg: built.append("embedder")
    )
    monkeypatch.setattr(
        filter_cli, "build_qe_scorer", lambda cfg: built.append("scorer")
    )
    config = _write_config(
        workspace.config,
        workspace.lid_model,
        workspace.spm_model,
        stages=["script_ratio", "numbers"],
    )
    assert config == workspace.config

    result = runner.invoke(app, workspace.args())

    assert result.exit_code == 0, result.output
    assert built == []


def test_a_corpus_without_rows_gives_empty_outputs(
    workspace: Workspace,
) -> None:
    write_corpus(corpus_frame([]), workspace.corpus)

    result = runner.invoke(app, workspace.args())

    assert result.exit_code == 0, result.output
    manifest = _manifest(workspace)
    assert manifest.metrics == {
        "rows_in": 0.0,
        "rows_kept": 0.0,
        **{f"reason.{reason.value}": 0.0 for reason in FilterReason},
    }
    assert [item.rows for item in manifest.outputs] == [0, 0]
    assert scan_corpus(workspace.scored).collect().height == 0
    assert scan_corpus(workspace.kept).collect().height == 0


def test_a_corpus_in_which_every_row_is_rejected_keeps_nothing(
    workspace: Workspace, planted: PlantedCorpus
) -> None:
    # Duplicates are left out: they are rejected only next to their original.
    rejected = [
        row_id
        for row_id, reason in planted.expected_reason.items()
        if reason not in (None, "duplicate")
    ]
    write_corpus(planted.frame.filter(pl.col("id").is_in(rejected)), workspace.corpus)

    result = runner.invoke(app, workspace.args())

    assert result.exit_code == 0, result.output
    manifest = _manifest(workspace)
    assert manifest.metrics["rows_in"] == float(len(rejected))
    assert manifest.metrics["rows_kept"] == 0.0
    assert manifest.metrics["reason.duplicate"] == 0.0
    assert [item.rows for item in manifest.outputs] == [len(rejected), 0]
    assert scan_corpus(workspace.kept).collect().height == 0


def test_a_qe_scorer_is_used_and_released_after_the_run(
    workspace: Workspace, monkeypatch: pytest.MonkeyPatch
) -> None:
    scorer = SpyQeScorer()
    monkeypatch.setattr(filter_cli, "build_qe_scorer", lambda cfg: scorer)

    result = runner.invoke(app, workspace.args())

    assert result.exit_code == 0, result.output
    assert _manifest(workspace).status["qe"] == "ok"
    scored = scan_corpus(workspace.scored).collect()
    assert scored["scores"].struct.field("metricx_qe").null_count() == 0
    assert scorer.releases == 1


def test_a_qe_scorer_is_released_when_the_run_fails(
    workspace: Workspace, monkeypatch: pytest.MonkeyPatch
) -> None:
    scorer = SpyQeScorer(fail=True)
    monkeypatch.setattr(filter_cli, "build_qe_scorer", lambda cfg: scorer)

    result = runner.invoke(app, workspace.args())

    assert result.exit_code == 4, result.output
    assert result.stderr == "error: scorer unavailable\n"
    assert scorer.releases == 1
    assert not workspace.manifest.exists()


def test_the_qe_factory_builds_the_eole_scorer_only_for_metricx() -> None:
    assert isinstance(
        filter_cli.build_qe_scorer(QeConfig(scorer="metricx_qe")), EoleMetricXQEScorer
    )
    assert filter_cli.build_qe_scorer(QeConfig(scorer="cometkiwi")) is None


def test_the_embedder_factory_builds_a_lazy_sentence_transformer() -> None:
    embedder = filter_cli.build_embedder(LabseConfig())

    assert isinstance(embedder, SentenceTransformerEmbedder)


def test_a_missing_corpus_is_a_data_error(workspace: Workspace) -> None:
    workspace.corpus.unlink()

    result = runner.invoke(app, workspace.args())

    assert result.exit_code == 3, result.output
    assert result.stderr.startswith("error: corpus file not found: ")
    assert result.stdout == ""


@pytest.mark.parametrize("extra", [[], ["--dry-run"]])
def test_a_corpus_of_wrong_schema_is_a_data_error(
    workspace: Workspace, extra: list[str]
) -> None:
    pl.DataFrame({"text": ["only a text column"]}).write_parquet(workspace.corpus)

    result = runner.invoke(app, workspace.args(*extra))

    assert result.exit_code == 3, result.output
    assert "not a corpus table" in result.stderr
    assert not workspace.out_dir.exists()


class RecordingLogger:
    """Stands in for the structlog logger, which would bind the global
    logging setup to the stream of whichever test calls it first."""

    def __init__(self) -> None:
        self.events: list[tuple[str, dict[str, Any]]] = []

    def error(self, event: str, **fields: Any) -> None:
        self.events.append((event, fields))


def test_a_language_outside_the_registry_is_a_data_error_without_polars_noise(
    workspace: Workspace, monkeypatch: pytest.MonkeyPatch
) -> None:
    logger = RecordingLogger()
    monkeypatch.setattr(filter_cli, "get_logger", lambda name: logger)
    write_corpus(
        corpus_frame(
            [
                {
                    "src_lang": "xxx_Latn",
                    "tgt_lang": "eng_Latn",
                    "src": "abc",
                    "tgt": "def",
                    "origin": "web",
                    "corpus": "unit",
                    "split": "train",
                }
            ]
        ),
        workspace.corpus,
    )

    result = runner.invoke(app, workspace.args())

    assert result.exit_code == 3, result.output
    assert result.stderr == (
        "error: language code 'xxx_Latn' is not in the language registry "
        "(configs/languages.yaml)\n"
    )
    [(event, fields)] = logger.events
    assert event == "filter_error_detail"
    assert fields["detail"].startswith("language code 'xxx_Latn'")
    assert "This error occurred in the following expression" in fields["detail"]


@pytest.mark.parametrize(
    ("config_text", "message_start"),
    [
        pytest.param(
            "stages: [script_ratio", "error: invalid filter config", id="yaml"
        ),
        pytest.param(
            "stages: [script_ratio, nonsense]",
            "error: invalid filter config",
            id="unknown-stage",
        ),
        pytest.param("- a list", "error: invalid filter config", id="not-a-mapping"),
    ],
)
def test_an_invalid_config_is_a_config_error(
    workspace: Workspace, config_text: str, message_start: str
) -> None:
    workspace.config.write_text(config_text, encoding="utf-8")

    result = runner.invoke(app, workspace.args())

    assert result.exit_code == 2, result.output
    assert result.stderr.startswith(message_start)
    assert not workspace.out_dir.exists()


def test_a_config_error_with_several_lines_is_printed_whole(
    workspace: Workspace,
) -> None:
    workspace.config.write_text("stages: [script_ratio, nonsense]", encoding="utf-8")

    result = runner.invoke(app, workspace.args())

    assert result.exit_code == 2, result.output
    assert "validation error" in result.stderr
    assert "nonsense" in result.stderr


def test_an_output_directory_that_cannot_be_created_is_a_plain_failure(
    workspace: Workspace, tmp_path: Path
) -> None:
    blocker = tmp_path / "blocker"
    blocker.write_text("a file where a directory should go", encoding="utf-8")

    result = runner.invoke(app, workspace.args(out_dir=blocker / "out"))

    assert result.exit_code == 1, result.output
    assert result.stderr.startswith("error: cannot write the filter outputs under ")


def test_a_missing_config_is_a_config_error(workspace: Workspace) -> None:
    workspace.config.unlink()

    result = runner.invoke(app, workspace.args())

    assert result.exit_code == 2, result.output
    assert result.stderr.startswith("error: filter config file not found: ")


def test_a_missing_lid_model_is_a_config_error(
    workspace: Workspace, tmp_path: Path
) -> None:
    absent = tmp_path / "absent_lid.bin"
    _write_config(workspace.config, absent, workspace.spm_model)

    result = runner.invoke(app, workspace.args())

    assert result.exit_code == 2, result.output
    assert result.stderr.startswith("error: LID model file not found: ")
    assert str(absent) in result.stderr
    assert not workspace.manifest.exists()


def test_help_lists_the_options() -> None:
    result = runner.invoke(app, ["filter", "run", "--help"])

    assert result.exit_code == 0
    for option in OPTIONS:
        assert option in result.stdout
