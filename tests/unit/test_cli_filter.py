"""`rinchen filter run`: the tiny end-to-end run over the planted-failure
corpus (the D4 filter STOP gate), the manifest it writes, skipping, and the
exit code of each failure.

The LaBSE embedder and the QE scorer come from module-level factories in the
CLI; tests replace them with the concept embedder and no scorer, so no model is
downloaded. Everything lands under pytest's temporary directories.
"""

from __future__ import annotations

import shutil
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, NoReturn

import numpy as np
import polars as pl
import pytest
import yaml
from click.testing import Result
from numpy.typing import NDArray
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
    "fasttext-wheel",
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
            *self.args_without_out_dir(config=config),
            "--out-dir",
            str(self.out_dir if out_dir is None else out_dir),
            *extra,
        ]

    def args_without_out_dir(self, *, config: Path | None = None) -> list[str]:
        return [
            "filter",
            "run",
            "--corpus",
            str(self.corpus),
            "--config",
            str(self.config if config is None else config),
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
    # Private copies, so that a test may delete or change a model file.
    models = tmp_path / "models"
    models.mkdir()
    own_lid_model = shutil.copy(lid_model, models / "lid.bin")
    own_spm_model = shutil.copy(spm_model, models / "tiny.model")
    return Workspace(
        corpus=corpus,
        config=_write_config(
            tmp_path / "filters.yaml", Path(own_lid_model), Path(own_spm_model)
        ),
        out_dir=tmp_path / "out",
        lid_model=Path(own_lid_model),
        spm_model=Path(own_spm_model),
    )


def _run_ok(args: list[str]) -> Result:
    result = runner.invoke(app, args)
    assert result.exit_code == 0, result.output
    return result


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
    result = _run_ok(workspace.args())

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
    _run_ok(workspace.args())

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
    _run_ok(workspace.args())

    kept = scan_corpus(workspace.kept).collect()
    clean_ids = [i for i, reason in planted.expected_reason.items() if reason is None]
    assert kept["id"].to_list() == clean_ids
    assert len(clean_ids) == CLEAN_ROWS
    assert kept["filter_reason"].null_count() == CLEAN_ROWS


def test_manifest_records_inputs_outputs_config_and_environment(
    workspace: Workspace,
) -> None:
    _run_ok(workspace.args())

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
    _run_ok(workspace.args("--seed", "7"))

    assert _manifest(workspace).seed == 7


def test_a_different_seed_does_not_stop_a_rerun_from_skipping(
    workspace: Workspace,
) -> None:
    _run_ok(workspace.args("--seed", "7"))

    result = _run_ok(workspace.args("--seed", "8"))

    assert result.stdout.startswith("skipped:")
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

    _run_ok(workspace.args(config=config))

    assert [item.path for item in _manifest(workspace).inputs] == [
        str(workspace.corpus)
    ]


def test_the_default_out_dir_is_the_corpus_stem_under_the_default_root(
    workspace: Workspace, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "clean_root"
    monkeypatch.setattr(filter_cli, "DEFAULT_OUT_ROOT", root)
    expected = root / workspace.corpus.stem

    result = _run_ok(workspace.args_without_out_dir())

    assert result.stdout.splitlines()[-1] == f"manifest: {expected / 'manifest.json'}"
    assert sorted(path.name for path in expected.iterdir()) == [
        "kept.parquet",
        "manifest.json",
        "scored.parquet",
    ]
    assert not workspace.out_dir.exists()


def test_a_rerun_with_unchanged_inputs_skips(workspace: Workspace) -> None:
    _run_ok(workspace.args())
    manifest_before = workspace.manifest.read_bytes()
    scored_before = workspace.scored.stat().st_mtime_ns

    result = _run_ok(workspace.args())

    assert result.stdout.splitlines()[0].startswith("skipped:")
    assert result.stdout.splitlines()[-1] == f"manifest: {workspace.manifest}"
    assert workspace.manifest.read_bytes() == manifest_before
    assert workspace.scored.stat().st_mtime_ns == scored_before


def test_force_reruns_even_when_nothing_changed(workspace: Workspace) -> None:
    _run_ok(workspace.args())
    scored_before = workspace.scored.stat()

    result = _run_ok(workspace.args("--force"))

    scored_after = workspace.scored.stat()
    assert not result.stdout.startswith("skipped:")
    assert (scored_after.st_ino, scored_after.st_mtime_ns) != (
        scored_before.st_ino,
        scored_before.st_mtime_ns,
    )


def test_a_deleted_output_is_rebuilt_instead_of_skipped(workspace: Workspace) -> None:
    _run_ok(workspace.args())
    workspace.kept.unlink()

    result = _run_ok(workspace.args())

    assert not result.stdout.startswith("skipped:")
    assert scan_corpus(workspace.kept).collect().height == CLEAN_ROWS


def test_a_changed_config_reruns(workspace: Workspace) -> None:
    _run_ok(workspace.args())
    config = _write_config(
        workspace.config,
        workspace.lid_model,
        workspace.spm_model,
        script_ratio={"min_ratio": 0.5},
    )
    assert config == workspace.config

    result = _run_ok(workspace.args())

    assert not result.stdout.startswith("skipped:")
    assert _manifest(workspace).config_sha256 == sha256_of(workspace.config)


def test_a_changed_corpus_reruns(workspace: Workspace, planted: PlantedCorpus) -> None:
    _run_ok(workspace.args())
    write_corpus(planted.frame.head(40), workspace.corpus)

    result = _run_ok(workspace.args())

    assert not result.stdout.startswith("skipped:")
    assert _manifest(workspace).metrics["rows_in"] == 40.0


# Bytes that keep a model loadable: a trailing byte for fastText, an unknown
# protobuf field for SentencePiece.
@pytest.mark.parametrize(
    ("model", "suffix"),
    [("lid_model", b"\0"), ("spm_model", b"\xa0\x06\x01")],
)
def test_a_changed_model_file_reruns(
    workspace: Workspace, model: str, suffix: bytes
) -> None:
    _run_ok(workspace.args())
    path: Path = getattr(workspace, model)
    with path.open("ab") as handle:
        handle.write(suffix)

    result = _run_ok(workspace.args())

    assert not result.stdout.startswith("skipped:")
    recorded = {item.path: item.sha256 for item in _manifest(workspace).inputs}
    assert recorded[str(path)] == sha256_of(path)


def test_a_run_that_fails_part_way_leaves_no_manifest_for_a_later_run_to_trust(
    workspace: Workspace, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _run_ok(workspace.args())
    relaxed = _write_config(
        tmp_path / "relaxed.yaml",
        workspace.lid_model,
        workspace.spm_model,
        script_ratio={"min_ratio": 0.5},
    )
    real_write = filter_cli.write_corpus
    writes: list[Path] = []

    def fail_on_the_second_write(frame: pl.DataFrame, path: Path) -> int:
        writes.append(path)
        if len(writes) == 2:
            raise OSError("disk full")
        return real_write(frame, path)

    with monkeypatch.context() as patch:
        patch.setattr(filter_cli, "write_corpus", fail_on_the_second_write)
        failed = runner.invoke(app, workspace.args(config=relaxed))

    assert failed.exit_code == 1, failed.output
    assert failed.stderr.startswith("error: cannot write the filter outputs under ")
    assert [path.name for path in writes] == ["scored.parquet", "kept.parquet"]
    assert not workspace.manifest.exists()

    rerun = _run_ok(workspace.args())

    assert not rerun.stdout.startswith("skipped:")
    assert [(item.path, item.sha256) for item in _manifest(workspace).outputs] == [
        (str(path), sha256_of(path)) for path in (workspace.scored, workspace.kept)
    ]


def _fail_if_called(name: str) -> Callable[..., NoReturn]:
    def fail(*args: object, **kwargs: object) -> NoReturn:
        raise AssertionError(f"{name} must not be called")

    return fail


def _forbid_reading_inputs(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("load_filter_config", "scan_corpus", "build_inputs", "sha256_of"):
        monkeypatch.setattr(filter_cli, name, _fail_if_called(name))


@pytest.mark.parametrize("output", ["scored", "kept"])
@pytest.mark.parametrize("through_symlink", [False, True], ids=["path", "symlink"])
def test_a_corpus_that_is_an_output_of_the_run_is_a_config_error_that_destroys_nothing(
    workspace: Workspace,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    output: str,
    through_symlink: bool,
) -> None:
    _run_ok(workspace.args())
    first_pass = getattr(workspace, output)
    corpus = first_pass
    if through_symlink:
        corpus = tmp_path / "alias.parquet"
        corpus.symlink_to(first_pass)
    files_before = {
        path: path.read_bytes()
        for path in (workspace.scored, workspace.kept, workspace.manifest)
    }
    _forbid_reading_inputs(monkeypatch)

    result = runner.invoke(
        app,
        ["filter", "run", "--corpus", str(corpus), "--config", str(workspace.config)]
        + ["--out-dir", str(workspace.out_dir)],
    )

    assert result.exit_code == 2, result.output
    assert result.stderr.startswith("error: --corpus ")
    assert f"is the {first_pass.name} that this run writes" in result.stderr
    assert "pass a different --out-dir" in result.stderr
    assert {path: path.read_bytes() for path in files_before} == files_before


def test_a_corpus_in_the_out_dir_that_is_not_an_output_is_allowed(
    workspace: Workspace, tmp_path: Path
) -> None:
    corpus = workspace.out_dir / "input.parquet"
    workspace.out_dir.mkdir()
    shutil.copy(workspace.corpus, corpus)

    result = runner.invoke(
        app,
        ["filter", "run", "--corpus", str(corpus), "--config", str(workspace.config)]
        + ["--out-dir", str(workspace.out_dir)],
    )

    assert result.exit_code == 0, result.output
    assert corpus.is_file()


@pytest.mark.parametrize("extra", [[], ["--force"]])
def test_an_out_dir_holding_the_manifest_of_another_stage_is_a_data_error(
    workspace: Workspace, monkeypatch: pytest.MonkeyPatch, extra: list[str]
) -> None:
    foreign = Manifest(
        stage="mine_align",
        created="2026-11-07T00:00:00Z",
        code_version="deadbeef",
        config_sha256="cfg-abc",
        seed=0,
    )
    store.write(workspace.out_dir, foreign)
    before = workspace.manifest.read_bytes()
    monkeypatch.setattr(filter_cli, "build_embedder", _fail_if_called("build_embedder"))

    result = runner.invoke(app, workspace.args(*extra))

    assert result.exit_code == 3, result.output
    assert "mine_align" in result.stderr
    assert "'filter'" in result.stderr
    assert str(workspace.out_dir) in result.stderr
    assert workspace.manifest.read_bytes() == before
    assert not workspace.scored.exists()
    assert not workspace.kept.exists()


def test_a_corrupt_manifest_is_a_data_error_until_the_run_is_forced(
    workspace: Workspace,
) -> None:
    _run_ok(workspace.args())
    workspace.manifest.write_text(
        '{"schema": "rinchen.manifest/1", "st', encoding="utf-8"
    )

    result = runner.invoke(app, workspace.args())

    assert result.exit_code == 3, result.output
    assert result.stderr.startswith(f"error: unreadable manifest {workspace.manifest}")
    _run_ok(workspace.args("--force"))
    assert _manifest(workspace).stage == "filter"


def test_dry_run_prints_the_stage_plan_and_writes_nothing(
    workspace: Workspace,
) -> None:
    result = _run_ok(workspace.args("--dry-run"))

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

    _run_ok(workspace.args())

    assert built == []


def test_a_corpus_without_rows_gives_empty_outputs(
    workspace: Workspace,
) -> None:
    write_corpus(corpus_frame([]), workspace.corpus)

    _run_ok(workspace.args())

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

    _run_ok(workspace.args())

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

    _run_ok(workspace.args())

    assert _manifest(workspace).status["qe"] == "ok"
    scored = scan_corpus(workspace.scored).collect()
    assert scored["scores"].struct.field("metricx_qe").null_count() == 0
    assert scorer.releases == 1


def test_the_labse_embedder_is_released_before_the_qe_scorer_is_used(
    workspace: Workspace, planted: PlantedCorpus, monkeypatch: pytest.MonkeyPatch
) -> None:
    events: list[str] = []

    class RecordingEmbedder(ConceptEmbedder):
        def embed(self, texts: Sequence[str]) -> NDArray[np.float32]:
            events.append("embed")
            return super().embed(texts)

        def release(self) -> None:
            events.append("release")

    class RecordingScorer(SpyQeScorer):
        def score(
            self, sources: Sequence[str], translations: Sequence[str]
        ) -> list[float]:
            events.append("score")
            return super().score(sources, translations)

    scorer = RecordingScorer()
    monkeypatch.setattr(
        filter_cli, "build_embedder", lambda cfg: RecordingEmbedder(planted.concepts)
    )
    monkeypatch.setattr(filter_cli, "build_qe_scorer", lambda cfg: scorer)

    _run_ok(workspace.args())

    last_embed = len(events) - 1 - events[::-1].index("embed")
    assert events.count("release") == 1
    assert last_embed < events.index("release") < events.index("score")
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


@pytest.mark.parametrize("extra", [[], ["--dry-run"]])
@pytest.mark.parametrize(
    ("model", "key"),
    [("lid_model", "lid.model_path"), ("spm_model", "length.spm_model")],
)
def test_a_configured_model_file_that_is_missing_is_a_config_error(
    workspace: Workspace, model: str, key: str, extra: list[str]
) -> None:
    path: Path = getattr(workspace, model)
    path.unlink()

    result = runner.invoke(app, workspace.args(*extra))

    assert result.exit_code == 2, result.output
    assert result.stderr == f"error: {key} is not a file: {path}\n"
    assert not workspace.out_dir.exists()


def test_a_model_deleted_after_a_run_is_an_error_not_a_skip(
    workspace: Workspace,
) -> None:
    _run_ok(workspace.args())
    manifest_before = workspace.manifest.read_bytes()
    workspace.lid_model.unlink()

    result = runner.invoke(app, workspace.args())

    assert result.exit_code == 2, result.output
    assert result.stdout == ""
    assert workspace.manifest.read_bytes() == manifest_before


def test_a_missing_model_of_a_stage_that_is_not_configured_is_ignored(
    workspace: Workspace, tmp_path: Path
) -> None:
    config = _write_config(
        tmp_path / "numbers_only.yaml",
        tmp_path / "absent_lid.bin",
        tmp_path / "absent.model",
        stages=["numbers"],
    )

    _run_ok(workspace.args(config=config))


def test_help_lists_the_options() -> None:
    result = runner.invoke(app, ["filter", "run", "--help"])

    assert result.exit_code == 0
    for option in OPTIONS:
        assert option in result.stdout
