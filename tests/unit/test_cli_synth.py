"""`rinchen synth run`: the plan it prints, the corpus, counts file and manifest it
writes (the manifest last), the QE skip statuses and the exit code of each
failure.

The generator and the QE scorer come from module-level seams of the CLI
(`build_translator`, `build_qe_scorer`); tests replace them with the scripted
translator and the planted scorer, so no server, model or GPU is involved.
Everything lands under pytest's temporary directories.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
from collections.abc import Callable, Sequence
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, NoReturn

import polars as pl
import pytest
from click.testing import Result
from polars.testing import assert_frame_equal
from typer.testing import CliRunner

from rinchen.cli import synth as synth_cli
from rinchen.cli.main import app
from rinchen.config.filters import QeConfig
from rinchen.config.loader import REPO_ROOT
from rinchen.config.synth import (
    Ct2Config,
    LlamaCppConfig,
    SelectionConfig,
    VllmConfig,
    load_synth_config,
)
from rinchen.corpus.schema import CORPUS_SCHEMA
from rinchen.errors import UpstreamError
from rinchen.filter import qe
from rinchen.filter.qe_eole import EoleMetricXQEScorer
from rinchen.manifest import store
from rinchen.manifest.model import Manifest
from rinchen.synth.llamacpp_client import LlamaCppTranslator
from rinchen.synth.run import RunPlan, run_job
from rinchen.synth.vllm_client import OpenAICompatTranslator
from tests.support.planted_synth import (
    FIRST_RUN_STATS,
    PLANTED_COUNTS,
    FakeScorer,
    ScriptedTranslator,
    planted_monolingual,
    planted_scorer,
    planted_script,
)
from tests.support.synth_cli import (
    BT_JOB,
    FT_JOB,
    GENERATOR,
    SEED,
    SHARD_SIZE,
    Seams,
    Workspace,
    install_seams,
    make_workspace,
    write_job_config,
)

runner = CliRunner()

# What the planted run gives without a QE scorer: the ten QE drops survive.
UNSCORED_COUNTS = {
    **PLANTED_COUNTS,
    "dropped_qe": 0,
    "no_survivor": 10,
    "selected": 25,
}
TOOL_PACKAGES = [
    "httpx",
    "polars",
    "sacrebleu",
    "ctranslate2",
    "sentencepiece",
    "eole",
    "torch",
]
OPTIONS = [
    "--input",
    "--config",
    "--out-dir",
    "--seed",
    "--force",
    "--dry-run",
    "--no-qe",
]
OUTPUT_NAMES = ["corpus.parquet", "counts.json", "manifest.json"]


@pytest.fixture
def workspace(tmp_path: Path) -> Workspace:
    return make_workspace(tmp_path)


def _planted_seams(
    monkeypatch: pytest.MonkeyPatch,
    workspace: Workspace,
    kind: str = "bt",
    *,
    translator: ScriptedTranslator | None = None,
) -> Seams:
    """The planted translator (a fresh one unless given) and scorer behind the
    seams."""
    return install_seams(
        monkeypatch,
        translator=translator or ScriptedTranslator(planted_script(workspace.frame)),
        scorer=planted_scorer(workspace.frame, kind),
    )


@pytest.fixture
def seams(monkeypatch: pytest.MonkeyPatch, workspace: Workspace) -> Seams:
    return _planted_seams(monkeypatch, workspace)


def _calls(seams: Seams) -> list[tuple[str, ...]]:
    """The texts of every request the translator got, in order."""
    assert isinstance(seams.translator, ScriptedTranslator)
    return [texts for texts, _, _ in seams.translator.calls]


def _run_ok(args: list[str]) -> Result:
    result = runner.invoke(app, args)
    assert result.exit_code == 0, result.output
    return result


def _manifest(workspace: Workspace) -> Manifest:
    manifest = store.read(workspace.out_dir)
    assert manifest is not None
    return manifest


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _without_created(path: Path) -> pl.DataFrame:
    return pl.read_parquet(path).drop("created")


def _snapshot(directory: Path) -> dict[str, bytes | None]:
    """Every file (its bytes) and directory (None) below `directory`."""
    return {
        str(path.relative_to(directory)): path.read_bytes() if path.is_file() else None
        for path in sorted(directory.rglob("*"))
    }


def _part_files(directory: Path) -> list[Path]:
    return sorted(directory.rglob("*.part"))


def _assert_one_error(result: Result, exit_code: int, message_start: str) -> None:
    """The run failed with `exit_code` and said so in one line that starts with
    `message_start`, which is the last line of stderr and no traceback."""
    assert result.exit_code == exit_code, result.output
    errors = [line for line in result.stderr.splitlines() if line.startswith("error:")]
    assert len(errors) == 1, result.stderr
    assert errors[0].startswith(f"error: {message_start}"), errors[0]
    assert result.stderr.splitlines()[-1] == errors[0]
    assert "Traceback" not in result.output


def _crash_after_two_shards(
    monkeypatch: pytest.MonkeyPatch, workspace: Workspace, *extra: str
) -> Result:
    """A run whose translator fails on its third shard."""
    translator = ScriptedTranslator(
        planted_script(workspace.frame),
        fail_on_call={2: UpstreamError("server down")},
    )
    _planted_seams(monkeypatch, workspace, translator=translator)
    return runner.invoke(app, workspace.args(BT_JOB, *extra))


def _probed(seam: Callable[[Any], Any], probe: Callable[[], None]) -> Callable:
    def probed(cfg: Any) -> Any:
        probe()
        return seam(cfg)

    return probed


# --- help --------------------------------------------------------------------


def test_the_synth_group_lists_the_run_command() -> None:
    result = runner.invoke(app, ["synth", "--help"])

    assert result.exit_code == 0
    assert "Run one synthetic-data job" in result.stdout


def test_run_help_names_the_job_argument_and_every_option() -> None:
    result = runner.invoke(app, ["synth", "run", "--help"])

    assert result.exit_code == 0
    assert "synth run [OPTIONS] {JOB}" in result.stdout
    for option in OPTIONS:
        assert option in result.stdout


# --- the plan ----------------------------------------------------------------


def test_dry_run_prints_the_plan_and_creates_nothing(
    workspace: Workspace, seams: Seams
) -> None:
    result = _run_ok(workspace.args(BT_JOB, "--dry-run"))

    fingerprint = workspace.fingerprint()
    shard_directory = workspace.out_dir / f"shards-{fingerprint[:12]}"
    assert result.stdout.splitlines() == [
        "job: bt_eng_khk",
        "kind: bt",
        "generator: vllm_default (vllm)",
        "input rows: 40",
        "shards: 5 (shard_size 8)",
        f"fingerprint: {fingerprint}",
        f"shard directory: {shard_directory}",
        "valid shards present: 0",
        f"out dir: {workspace.out_dir}",
    ]
    assert not workspace.out_dir.exists()
    assert seams.generators_built == []
    assert seams.selections_built == []
    assert _calls(seams) == []


def test_dry_run_prints_what_plan_run_returns(
    workspace: Workspace, seams: Seams, monkeypatch: pytest.MonkeyPatch
) -> None:
    planned: list[RunPlan] = []
    real_plan_run = synth_cli.plan_run

    def spying_plan_run(*args: Any, **kwargs: Any) -> RunPlan:
        planned.append(real_plan_run(*args, **kwargs))
        return planned[-1]

    monkeypatch.setattr(synth_cli, "plan_run", spying_plan_run)

    lines = _run_ok(workspace.args(BT_JOB, "--dry-run")).stdout.splitlines()

    (run_plan,) = planned
    assert f"shards: {len(run_plan.shards)} (shard_size 8)" in lines
    assert f"fingerprint: {run_plan.fingerprint}" in lines
    assert f"shard directory: {run_plan.shard_directory}" in lines


def test_the_dry_run_names_the_fingerprint_and_the_directory_the_real_run_uses(
    workspace: Workspace, seams: Seams, monkeypatch: pytest.MonkeyPatch
) -> None:
    lines = _run_ok(
        workspace.args(BT_JOB, "--dry-run", "--seed", "3")
    ).stdout.splitlines()
    _planted_seams(monkeypatch, workspace)

    _run_ok(workspace.args(BT_JOB, "--seed", "3"))

    (directory,) = workspace.shard_directories()
    assert f"shard directory: {directory}" in lines
    assert f"fingerprint: {workspace.counts_document()['fingerprint']}" in lines


def test_dry_run_of_a_forward_job_names_its_kind(
    workspace: Workspace, seams: Seams
) -> None:
    lines = _run_ok(workspace.args(FT_JOB, "--dry-run")).stdout.splitlines()

    assert lines[:2] == ["job: ft_khk_eng", "kind: ft"]
    assert lines[5] == f"fingerprint: {workspace.fingerprint(FT_JOB)}"


def test_dry_run_fingerprint_follows_the_seed(
    workspace: Workspace, seams: Seams
) -> None:
    result = _run_ok(workspace.args(BT_JOB, "--dry-run", "--seed", "7"))

    assert workspace.fingerprint(seed=7) != workspace.fingerprint()
    assert f"fingerprint: {workspace.fingerprint(seed=7)}" in result.stdout.splitlines()


def test_dry_run_counts_the_valid_shards_on_disk_and_changes_nothing(
    workspace: Workspace, monkeypatch: pytest.MonkeyPatch
) -> None:
    _crash_after_two_shards(monkeypatch, workspace)
    (shard_directory,) = workspace.shard_directories()
    seams = _planted_seams(monkeypatch, workspace)
    before = _snapshot(workspace.out_dir)

    two_valid = _run_ok(workspace.args(BT_JOB, "--dry-run")).stdout.splitlines()
    unchanged = _snapshot(workspace.out_dir)
    (shard_directory / "shard-00000.parquet").write_bytes(b"not a parquet file")
    damaged = _snapshot(workspace.out_dir)
    one_valid = _run_ok(workspace.args(BT_JOB, "--dry-run")).stdout.splitlines()

    assert two_valid[6] == f"shard directory: {shard_directory}"
    assert two_valid[7] == "valid shards present: 2"
    assert one_valid[7] == "valid shards present: 1"
    assert unchanged == before
    assert _snapshot(workspace.out_dir) == damaged
    assert seams.generators_built == []


# --- a run -------------------------------------------------------------------


def test_a_run_writes_the_corpus_the_counts_and_the_manifest_and_prints_them(
    workspace: Workspace, seams: Seams
) -> None:
    result = _run_ok(workspace.args())

    assert result.stdout.splitlines() == [
        f"corpus: {workspace.corpus}",
        f"counts: {workspace.counts}",
        f"manifest: {workspace.manifest}",
        "rows 15 (selected 15 of 40 sources, qe ok)",
    ]
    assert sorted(path.name for path in workspace.out_dir.iterdir()) == [
        *OUTPUT_NAMES,
        f"shards-{workspace.fingerprint()[:12]}",
    ]
    assert _part_files(workspace.out_dir) == []


def test_a_run_reports_one_stderr_line_per_shard(
    workspace: Workspace, monkeypatch: pytest.MonkeyPatch
) -> None:
    _planted_seams(monkeypatch, workspace)
    first = _run_ok(workspace.args())
    _planted_seams(monkeypatch, workspace)
    second = _run_ok(workspace.args())

    assert first.stderr.splitlines() == [
        f"shard {number}/5 generated" for number in range(1, 6)
    ]
    assert second.stderr.splitlines() == [
        f"shard {number}/5 reused" for number in range(1, 6)
    ]


def test_the_corpus_and_the_counts_are_what_the_library_returns_for_the_same_run(
    workspace: Workspace, seams: Seams, tmp_path: Path
) -> None:
    _run_ok(workspace.args())
    cfg = load_synth_config(workspace.config)
    expected = run_job(
        cfg.jobs[BT_JOB],
        GENERATOR,
        cfg.generators[GENERATOR],
        ScriptedTranslator(planted_script(workspace.frame)),
        workspace.frame,
        out_dir=tmp_path / "direct",
        cfg=cfg,
        scorer=planted_scorer(workspace.frame, "bt"),
        seed=SEED,
    )

    corpus = pl.read_parquet(workspace.corpus)
    assert corpus.schema == CORPUS_SCHEMA
    assert_frame_equal(corpus.drop("created"), expected.corpus.drop("created"))
    document = workspace.counts_document()
    assert document["counts"] == dict(expected.counts)
    assert document["fingerprint"] == expected.fingerprint


def test_counts_json_holds_the_job_the_counts_the_shard_statistics_and_the_qe_status(
    workspace: Workspace, seams: Seams
) -> None:
    _run_ok(workspace.args())

    text = workspace.counts.read_text(encoding="utf-8")
    assert json.loads(text) == {
        "fingerprint": workspace.fingerprint(),
        "job": "bt_eng_khk",
        "generator": "vllm_default",
        "counts": PLANTED_COUNTS,
        "shard_stats": FIRST_RUN_STATS,
        "qe_status": "ok",
    }
    # Sorted keys, two-space indent, one final newline.
    assert text.startswith('{\n  "counts": {\n    "candidates": 60,\n')
    assert text.endswith("\n  }\n}\n")


def test_the_manifest_records_inputs_outputs_config_and_environment(
    workspace: Workspace, seams: Seams
) -> None:
    _run_ok(workspace.args(BT_JOB, "--seed", "7"))

    manifest = _manifest(workspace)
    assert manifest.stage == "synth"
    assert manifest.seed == 7
    assert manifest.config_sha256 == _sha256(workspace.config)
    assert manifest.code_version
    assert datetime.fromisoformat(manifest.created).utcoffset() == timedelta(0)
    assert [(i.path, i.sha256, i.bytes, i.rows) for i in manifest.inputs] == [
        (str(path), _sha256(path), path.stat().st_size, None)
        for path in (workspace.monolingual, workspace.config)
    ]
    assert [(o.path, o.sha256, o.bytes, o.rows) for o in manifest.outputs] == [
        (str(path), _sha256(path), path.stat().st_size, rows)
        for path, rows in ((workspace.corpus, 15), (workspace.counts, None))
    ]
    assert manifest.metrics == {
        "count.sources": 40.0,
        "count.candidates": 60.0,
        "count.dropped_empty": 5.0,
        "count.dropped_purity": 10.0,
        "count.dropped_qe": 10.0,
        "count.dropped_copy": 0.0,
        "count.no_candidates": 5.0,
        "count.no_survivor": 20.0,
        "count.selected": 15.0,
        "shards.planned": 5.0,
        "shards.reused": 0.0,
        "shards.generated": 5.0,
        "shards.regenerated_invalid": 0.0,
    }
    assert list(manifest.tool_versions) == TOOL_PACKAGES
    assert manifest.tool_versions["polars"] == pl.__version__
    for package in ("httpx", "eole", "torch"):
        assert manifest.tool_versions[package] == importlib.metadata.version(package)
    assert manifest.status == {"qe": "ok"}


def test_the_seed_names_the_shard_directory_and_is_recorded(
    workspace: Workspace, seams: Seams
) -> None:
    _run_ok(workspace.args(BT_JOB, "--seed", "7"))

    assert [path.name for path in workspace.shard_directories()] == [
        f"shards-{workspace.fingerprint(seed=7)[:12]}"
    ]
    assert workspace.counts_document()["fingerprint"] == workspace.fingerprint(seed=7)
    assert _manifest(workspace).seed == 7


def test_a_forward_run_gives_the_forward_corpus(
    workspace: Workspace, monkeypatch: pytest.MonkeyPatch
) -> None:
    _planted_seams(monkeypatch, workspace, "ft")

    result = _run_ok(workspace.args(FT_JOB))

    assert (
        result.stdout.splitlines()[-1] == "rows 15 (selected 15 of 40 sources, qe ok)"
    )
    assert pl.read_parquet(workspace.corpus)["corpus"].unique().to_list() == [
        "ft:vllm_default"
    ]
    assert workspace.counts_document()["job"] == FT_JOB


def test_the_job_runs_with_the_generator_it_names(
    workspace: Workspace, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = write_job_config(
        tmp_path / "other.yaml",
        shard_size=SHARD_SIZE,
        generator={"kind": "llamacpp", "n": 2},
        generator_name="llama_local",
        unused_generators=["other_server"],
    )
    seams = _planted_seams(monkeypatch, workspace)

    _run_ok(workspace.args(BT_JOB, config=config))

    (built,) = seams.generators_built
    assert isinstance(built, LlamaCppConfig)
    assert built.n == 2
    assert pl.read_parquet(workspace.corpus)["corpus"].unique().to_list() == [
        "bt:llama_local"
    ]
    assert workspace.counts_document()["generator"] == "llama_local"


def test_the_seams_get_the_generator_and_the_selection_of_the_config(
    workspace: Workspace, seams: Seams
) -> None:
    _run_ok(workspace.args())

    cfg = load_synth_config(workspace.config)
    assert seams.generators_built == [cfg.generators[GENERATOR]]
    assert seams.selections_built == [cfg.selection]


def test_the_default_out_dir_is_the_job_name_under_the_default_root(
    workspace: Workspace,
    seams: Seams,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "synth_root"
    monkeypatch.setattr(synth_cli, "DEFAULT_OUT_ROOT", root)
    expected = root / BT_JOB

    result = _run_ok(workspace.args_without_out_dir(BT_JOB))

    assert result.stdout.splitlines()[2] == f"manifest: {expected / 'manifest.json'}"
    assert (expected / "corpus.parquet").is_file()
    assert not workspace.out_dir.exists()


def test_the_default_out_root_is_data_synth_under_the_repository_root() -> None:
    assert synth_cli.DEFAULT_OUT_ROOT == REPO_ROOT / "data" / "synth"


# --- the commit protocol -----------------------------------------------------


def test_the_manifest_is_absent_while_the_seams_run_and_the_inputs_are_hashed_first(
    workspace: Workspace, monkeypatch: pytest.MonkeyPatch
) -> None:
    hashes_before_the_work = [
        _sha256(path) for path in (workspace.monolingual, workspace.config)
    ]
    manifest_present_during: list[str] = []

    def probe(where: str) -> None:
        if workspace.manifest.exists():
            manifest_present_during.append(where)

    class ProbingTranslator(ScriptedTranslator):
        def translate(
            self, texts: Sequence[str], src_lang: str, tgt_lang: str
        ) -> list[list[str]]:
            probe("translate")
            # The work changes the very files the run started from.
            workspace.monolingual.write_bytes(b"replaced while the run works")
            workspace.config.write_text("replaced: true\n", encoding="utf-8")
            return super().translate(texts, src_lang, tgt_lang)

    planted = planted_scorer(workspace.frame, "bt")

    class ProbingScorer:
        requires_cuda = False

        def score(
            self, sources: Sequence[str], translations: Sequence[str]
        ) -> list[float]:
            probe("score")
            return planted.score(sources, translations)

    install_seams(
        monkeypatch,
        translator=ProbingTranslator(planted_script(workspace.frame)),
        scorer=ProbingScorer(),
    )
    for seam in ("build_translator", "build_qe_scorer"):
        monkeypatch.setattr(
            synth_cli,
            seam,
            _probed(getattr(synth_cli, seam), lambda seam=seam: probe(seam)),
        )

    _run_ok(workspace.args())

    assert manifest_present_during == []
    manifest = _manifest(workspace)
    assert [item.sha256 for item in manifest.inputs] == hashes_before_the_work
    assert manifest.config_sha256 == hashes_before_the_work[1]


def _rewrite_config(path: Path) -> None:
    path.write_text(path.read_text(encoding="utf-8") + "# edited\n", encoding="utf-8")


def _rewrite_monolingual(path: Path) -> None:
    pl.read_parquet(path).write_parquet(path, compression="uncompressed")


@pytest.mark.parametrize(
    ("loader", "which", "rewrite"),
    [
        ("load_synth_config", "config", _rewrite_config),
        ("read_monolingual", "monolingual", _rewrite_monolingual),
    ],
)
def test_the_manifest_vouches_for_the_bytes_an_input_had_before_it_was_read(
    workspace: Workspace,
    seams: Seams,
    monkeypatch: pytest.MonkeyPatch,
    loader: str,
    which: str,
    rewrite: Callable[[Path], None],
) -> None:
    path = workspace.config if which == "config" else workspace.monolingual
    before = {
        "monolingual": _sha256(workspace.monolingual),
        "config": _sha256(workspace.config),
    }
    real_loader = getattr(synth_cli, loader)

    def loader_after_a_change(*args: Any, **kwargs: Any) -> Any:
        rewrite(path)  # the file changes just before the loader reads it
        return real_loader(*args, **kwargs)

    monkeypatch.setattr(synth_cli, loader, loader_after_a_change)

    _run_ok(workspace.args())

    assert _sha256(path) != before[which]
    manifest = _manifest(workspace)
    assert [item.sha256 for item in manifest.inputs] == [
        before["monolingual"],
        before["config"],
    ]
    assert manifest.config_sha256 == before["config"]


@pytest.mark.parametrize(
    ("which", "what"),
    [("monolingual", "monolingual file"), ("config", "synth config file")],
)
def test_an_input_that_cannot_be_read_is_a_data_error_naming_it(
    workspace: Workspace,
    seams: Seams,
    monkeypatch: pytest.MonkeyPatch,
    which: str,
    what: str,
) -> None:
    unreadable = workspace.config if which == "config" else workspace.monolingual
    real_open = Path.open

    def open_unless_unreadable(self: Path, mode: str = "r", *args: Any, **kwargs: Any):
        if self == unreadable and "b" in mode:
            raise PermissionError(13, "Permission denied", str(self))
        return real_open(self, mode, *args, **kwargs)

    monkeypatch.setattr(Path, "open", open_unless_unreadable)

    result = runner.invoke(app, workspace.args())

    _assert_one_error(result, 3, f"cannot read {what} {unreadable}: ")
    assert not workspace.out_dir.exists()
    assert seams.generators_built == []


def test_the_manifest_is_written_after_both_outputs_are_complete(
    workspace: Workspace, seams: Seams, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: list[dict[str, Any]] = []
    real_write = store.write

    def spying_write(stage_dir: Path, manifest: Manifest) -> Path:
        seen.append(
            {
                "manifest_present": workspace.manifest.exists(),
                "corpus_rows": pl.read_parquet(workspace.corpus).height,
                "counts_job": workspace.counts_document()["job"],
                "part_files": _part_files(workspace.out_dir),
            }
        )
        return real_write(stage_dir, manifest)

    monkeypatch.setattr(store, "write", spying_write)

    _run_ok(workspace.args())

    assert seen == [
        {
            "manifest_present": False,
            "corpus_rows": 15,
            "counts_job": BT_JOB,
            "part_files": [],
        }
    ]


def test_an_earlier_manifest_stands_while_the_run_generates_and_goes_before_the_outputs(
    workspace: Workspace, monkeypatch: pytest.MonkeyPatch
) -> None:
    _planted_seams(monkeypatch, workspace)
    _run_ok(workspace.args())
    old_manifest = workspace.manifest.read_bytes()
    present_while: dict[str, bool] = {}

    class ProbingTranslator(ScriptedTranslator):
        def translate(
            self, texts: Sequence[str], src_lang: str, tgt_lang: str
        ) -> list[list[str]]:
            present_while["generating"] = workspace.manifest.exists()
            return super().translate(texts, src_lang, tgt_lang)

    real_write_corpus = synth_cli.write_corpus

    def spying_write_corpus(frame: pl.DataFrame, path: Path) -> int:
        present_while["writing the corpus"] = workspace.manifest.exists()
        return real_write_corpus(frame, path)

    monkeypatch.setattr(synth_cli, "write_corpus", spying_write_corpus)
    translator = ProbingTranslator(planted_script(workspace.frame))
    _planted_seams(monkeypatch, workspace, translator=translator)

    _run_ok(workspace.args(BT_JOB, "--force"))

    assert present_while == {"generating": True, "writing the corpus": False}
    assert workspace.manifest.read_bytes() != old_manifest


def test_a_failed_generation_keeps_the_earlier_manifest_and_outputs(
    workspace: Workspace, monkeypatch: pytest.MonkeyPatch
) -> None:
    _planted_seams(monkeypatch, workspace)
    _run_ok(workspace.args())
    committed = {
        path: path.read_bytes()
        for path in (workspace.corpus, workspace.counts, workspace.manifest)
    }
    translator = ScriptedTranslator(
        planted_script(workspace.frame), fail_on_call={0: UpstreamError("down")}
    )
    _planted_seams(monkeypatch, workspace, translator=translator)

    # Another seed is another shard directory, so nothing is reused.
    result = runner.invoke(app, workspace.args(BT_JOB, "--seed", "1"))

    _assert_one_error(result, 4, "shard 0 of 5 failed: down")
    assert {path: path.read_bytes() for path in committed} == committed


def _raise_disk_full(*args: object) -> NoReturn:
    raise OSError("disk full")


@pytest.mark.parametrize("failing", ["corpus", "manifest"])
def test_a_failed_output_write_leaves_no_manifest_for_a_later_run_to_trust(
    workspace: Workspace, monkeypatch: pytest.MonkeyPatch, failing: str
) -> None:
    _planted_seams(monkeypatch, workspace)
    _run_ok(workspace.args())
    with monkeypatch.context() as patch:
        if failing == "corpus":
            patch.setattr(synth_cli, "write_corpus", _raise_disk_full)
        else:
            patch.setattr(store, "write", _raise_disk_full)
        failed = runner.invoke(app, workspace.args(BT_JOB, "--force"))

    _assert_one_error(
        failed,
        1,
        f"cannot write the synth outputs under {workspace.out_dir}: disk full",
    )
    assert not workspace.manifest.exists()
    assert _part_files(workspace.out_dir) == []

    _planted_seams(monkeypatch, workspace)
    _run_ok(workspace.args())

    assert [(o.path, o.sha256) for o in _manifest(workspace).outputs] == [
        (str(path), _sha256(path)) for path in (workspace.corpus, workspace.counts)
    ]


def test_a_counts_file_that_cannot_be_renamed_into_place_leaves_nothing_behind(
    workspace: Workspace, seams: Seams, monkeypatch: pytest.MonkeyPatch
) -> None:
    real_replace = Path.replace

    def replace_unless_counts(self: Path, target: Path) -> Path:
        if self.name.startswith("counts.json"):
            raise OSError("cannot rename")
        return real_replace(self, target)

    monkeypatch.setattr(Path, "replace", replace_unless_counts)

    result = runner.invoke(app, workspace.args())

    _assert_one_error(
        result,
        1,
        f"cannot write the synth outputs under {workspace.out_dir}: cannot rename",
    )
    assert not workspace.counts.exists()
    assert not workspace.manifest.exists()
    assert _part_files(workspace.out_dir) == []


def test_an_out_dir_that_cannot_be_created_is_a_plain_failure(
    workspace: Workspace, seams: Seams, tmp_path: Path
) -> None:
    blocker = tmp_path / "blocker"
    blocker.write_text("a file where a directory should go", encoding="utf-8")

    result = runner.invoke(app, workspace.args(out_dir=blocker / "out"))

    _assert_one_error(
        result, 1, f"cannot write the synth outputs under {blocker / 'out'}: "
    )
    # It fails before the paid work: no model is built, nothing is generated.
    assert seams.generators_built == []
    assert seams.selections_built == []
    assert _calls(seams) == []
    assert not (blocker / "out").exists()


def test_two_runs_into_one_out_dir_leave_a_manifest_that_matches_the_files(
    workspace: Workspace, monkeypatch: pytest.MonkeyPatch
) -> None:
    _planted_seams(monkeypatch, workspace)
    _run_ok(workspace.args(BT_JOB, "--seed", "1"))
    _planted_seams(monkeypatch, workspace)
    _run_ok(workspace.args(BT_JOB, "--seed", "2"))

    manifest = _manifest(workspace)
    assert manifest.seed == 2
    assert [(o.path, o.sha256) for o in manifest.outputs] == [
        (str(path), _sha256(path)) for path in (workspace.corpus, workspace.counts)
    ]
    assert len(workspace.shard_directories()) == 2
    assert _part_files(workspace.out_dir) == []


# --- resuming ----------------------------------------------------------------


def test_a_rerun_reuses_every_shard_and_gives_the_same_corpus_apart_from_created(
    workspace: Workspace, monkeypatch: pytest.MonkeyPatch
) -> None:
    first_seams = _planted_seams(monkeypatch, workspace)
    _run_ok(workspace.args())
    first = _without_created(workspace.corpus)
    rerun_seams = _planted_seams(monkeypatch, workspace)

    _run_ok(workspace.args())

    assert len(_calls(first_seams)) == 5
    assert _calls(rerun_seams) == []
    assert_frame_equal(_without_created(workspace.corpus), first)
    assert workspace.counts_document()["shard_stats"] == {
        "planned": 5,
        "reused": 5,
        "generated": 0,
        "regenerated_invalid": 0,
    }


def test_force_generates_every_shard_again(
    workspace: Workspace, monkeypatch: pytest.MonkeyPatch
) -> None:
    _planted_seams(monkeypatch, workspace)
    _run_ok(workspace.args())
    forced_seams = _planted_seams(monkeypatch, workspace)

    _run_ok(workspace.args(BT_JOB, "--force"))

    assert _calls(forced_seams) == workspace.shard_texts(0, 1, 2, 3, 4)
    assert workspace.counts_document()["shard_stats"] == FIRST_RUN_STATS


def test_a_generator_failure_exits_4_and_leaves_the_finished_shards(
    workspace: Workspace, monkeypatch: pytest.MonkeyPatch
) -> None:
    result = _crash_after_two_shards(monkeypatch, workspace)

    assert result.exit_code == 4, result.output
    assert result.stdout == ""
    assert result.stderr.splitlines() == [
        "shard 1/5 generated",
        "shard 2/5 generated",
        "error: shard 2 of 5 failed: server down",
    ]
    (shard_directory,) = workspace.shard_directories()
    assert sorted(path.name for path in shard_directory.iterdir()) == [
        "shard-00000.parquet",
        "shard-00001.parquet",
    ]
    assert [path.name for path in workspace.out_dir.iterdir()] == [shard_directory.name]
    assert _part_files(workspace.out_dir) == []


def test_the_rerun_after_a_crash_requests_only_the_missing_shards(
    workspace: Workspace, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _crash_after_two_shards(monkeypatch, workspace)
    resumed_seams = _planted_seams(monkeypatch, workspace)

    result = _run_ok(workspace.args())

    assert _calls(resumed_seams) == workspace.shard_texts(2, 3, 4)
    assert result.stderr.splitlines() == [
        "shard 1/5 reused",
        "shard 2/5 reused",
        "shard 3/5 generated",
        "shard 4/5 generated",
        "shard 5/5 generated",
    ]
    assert workspace.counts_document()["shard_stats"] == {
        "planned": 5,
        "reused": 2,
        "generated": 3,
        "regenerated_invalid": 0,
    }
    _planted_seams(monkeypatch, workspace)
    uninterrupted = tmp_path / "uninterrupted"
    _run_ok(workspace.args(BT_JOB, out_dir=uninterrupted))
    assert_frame_equal(
        _without_created(workspace.corpus),
        _without_created(uninterrupted / "corpus.parquet"),
    )


def test_a_damaged_shard_file_is_generated_again_on_the_rerun(
    workspace: Workspace, monkeypatch: pytest.MonkeyPatch
) -> None:
    _crash_after_two_shards(monkeypatch, workspace)
    (shard_directory,) = workspace.shard_directories()
    (shard_directory / "shard-00000.parquet").write_bytes(b"not a parquet file")
    resumed_seams = _planted_seams(monkeypatch, workspace)

    _run_ok(workspace.args())

    assert _calls(resumed_seams) == workspace.shard_texts(0, 2, 3, 4)
    assert workspace.counts_document()["shard_stats"] == {
        "planned": 5,
        "reused": 1,
        "generated": 4,
        "regenerated_invalid": 1,
    }


class _ReleasingScorer:
    """The planted scorer, which records its calls and its `release()`, and can
    fail on its first call."""

    requires_cuda = False

    def __init__(self, inner: FakeScorer, *, fail: bool = False) -> None:
        self._inner = inner
        self._fail = fail
        self.events: list[str] = []

    def score(self, sources: Sequence[str], translations: Sequence[str]) -> list[float]:
        self.events.append("score")
        if self._fail:
            raise RuntimeError("scorer out of memory")
        return self._inner.score(sources, translations)

    def release(self) -> None:
        self.events.append("release")


def test_the_scorer_is_released_once_after_the_run(
    workspace: Workspace, monkeypatch: pytest.MonkeyPatch
) -> None:
    scorer = _ReleasingScorer(planted_scorer(workspace.frame, "bt"))
    install_seams(
        monkeypatch,
        translator=ScriptedTranslator(planted_script(workspace.frame)),
        scorer=scorer,
    )

    _run_ok(workspace.args())

    assert scorer.events == ["score"] * 5 + ["release"]


def test_the_scorer_is_released_when_the_generator_fails(
    workspace: Workspace, monkeypatch: pytest.MonkeyPatch
) -> None:
    scorer = _ReleasingScorer(planted_scorer(workspace.frame, "bt"))
    install_seams(
        monkeypatch,
        translator=ScriptedTranslator(
            planted_script(workspace.frame), fail_on_call={2: UpstreamError("down")}
        ),
        scorer=scorer,
    )

    result = runner.invoke(app, workspace.args())

    assert result.exit_code == 4, result.output
    assert scorer.events == ["score", "score", "release"]


def test_the_scorer_is_released_when_it_fails(
    workspace: Workspace, monkeypatch: pytest.MonkeyPatch
) -> None:
    scorer = _ReleasingScorer(planted_scorer(workspace.frame, "bt"), fail=True)
    install_seams(
        monkeypatch,
        translator=ScriptedTranslator(planted_script(workspace.frame)),
        scorer=scorer,
    )

    result = runner.invoke(app, workspace.args())

    assert isinstance(result.exception, RuntimeError)
    assert scorer.events == ["score", "release"]
    assert not workspace.manifest.exists()


def test_an_interrupt_exits_130_and_leaves_the_finished_shards_and_the_old_manifest(
    workspace: Workspace, monkeypatch: pytest.MonkeyPatch
) -> None:
    _planted_seams(monkeypatch, workspace)
    _run_ok(workspace.args())
    committed = {
        path: path.read_bytes()
        for path in (workspace.corpus, workspace.counts, workspace.manifest)
    }
    translator = ScriptedTranslator(
        planted_script(workspace.frame), fail_on_call={2: KeyboardInterrupt()}
    )
    _planted_seams(monkeypatch, workspace, translator=translator)

    # Another seed is another shard directory, so nothing is reused.
    result = runner.invoke(app, workspace.args(BT_JOB, "--seed", "1"))

    assert result.exit_code == 130, result.output
    assert result.stdout == ""
    assert result.stderr.splitlines() == ["shard 1/5 generated", "shard 2/5 generated"]
    interrupted = workspace.out_dir / f"shards-{workspace.fingerprint(seed=1)[:12]}"
    assert sorted(path.name for path in interrupted.iterdir()) == [
        "shard-00000.parquet",
        "shard-00001.parquet",
    ]
    assert _part_files(workspace.out_dir) == []
    assert {path: path.read_bytes() for path in committed} == committed


# --- QE ----------------------------------------------------------------------


def _assert_unscored_run(
    workspace: Workspace, result: Result, status: str, scorer: FakeScorer | None
) -> None:
    assert result.stdout.splitlines()[-1] == (
        f"rows 25 (selected 25 of 40 sources, qe {status})"
    )
    assert _manifest(workspace).status == {"qe": status}
    document = workspace.counts_document()
    assert document["qe_status"] == status
    assert document["counts"] == UNSCORED_COUNTS
    corpus = pl.read_parquet(workspace.corpus)
    assert corpus.height == 25
    assert corpus["scores"].struct.field("metricx_qe").null_count() == 25
    if scorer is not None:
        assert scorer.calls == []


def test_no_qe_skips_the_scorer_without_building_it(
    workspace: Workspace, monkeypatch: pytest.MonkeyPatch
) -> None:
    scorer = planted_scorer(workspace.frame, "bt")
    seams = install_seams(
        monkeypatch,
        translator=ScriptedTranslator(planted_script(workspace.frame)),
        scorer=scorer,
    )

    result = _run_ok(workspace.args(BT_JOB, "--no-qe"))

    _assert_unscored_run(workspace, result, "skipped: --no-qe", scorer)
    assert seams.selections_built == []


def test_a_missing_scorer_skips_the_qe_step_and_says_so(
    workspace: Workspace, monkeypatch: pytest.MonkeyPatch
) -> None:
    seams = install_seams(
        monkeypatch,
        translator=ScriptedTranslator(planted_script(workspace.frame)),
        scorer=None,
    )

    result = _run_ok(workspace.args())

    _assert_unscored_run(workspace, result, "skipped: no scorer", None)
    assert seams.selections_built == [load_synth_config(workspace.config).selection]


def test_a_scorer_that_needs_cuda_is_skipped_when_there_is_none(
    workspace: Workspace, monkeypatch: pytest.MonkeyPatch
) -> None:
    scorer = FakeScorer({}, requires_cuda=True)
    install_seams(
        monkeypatch,
        translator=ScriptedTranslator(planted_script(workspace.frame)),
        scorer=scorer,
    )
    monkeypatch.setattr(qe, "cuda_is_available", lambda: False)

    result = _run_ok(workspace.args())

    _assert_unscored_run(workspace, result, "skipped: no CUDA device", scorer)


def test_a_scorer_that_needs_cuda_runs_when_there_is_one(
    workspace: Workspace, monkeypatch: pytest.MonkeyPatch
) -> None:
    scorer = FakeScorer({}, requires_cuda=True)
    install_seams(
        monkeypatch,
        translator=ScriptedTranslator(planted_script(workspace.frame)),
        scorer=scorer,
    )
    monkeypatch.setattr(qe, "cuda_is_available", lambda: True)

    result = _run_ok(workspace.args())

    assert scorer.calls
    assert result.stdout.splitlines()[-1].endswith("qe ok)")
    assert _manifest(workspace).status == {"qe": "ok"}


# --- refusals ----------------------------------------------------------------


@pytest.mark.parametrize("extra", [[], ["--force"], ["--dry-run"]])
def test_an_out_dir_holding_the_manifest_of_another_stage_is_a_data_error(
    workspace: Workspace, seams: Seams, extra: list[str]
) -> None:
    store.write(
        workspace.out_dir,
        Manifest(
            stage="filter",
            created="2026-11-07T00:00:00Z",
            code_version="deadbeef",
            config_sha256="cfg-abc",
            seed=0,
        ),
    )
    before = _snapshot(workspace.out_dir)

    result = runner.invoke(app, workspace.args(BT_JOB, *extra))

    _assert_one_error(result, 3, f"{workspace.out_dir} holds the manifest of stage ")
    assert "'filter'" in result.stderr
    assert "'synth'" in result.stderr
    assert _snapshot(workspace.out_dir) == before
    assert seams.generators_built == []
    assert seams.selections_built == []


def _spellings(workspace: Workspace, tmp_path: Path) -> dict[str, tuple[Path, Path]]:
    """Ways to give the input and the output directory that name one place:
    (out_dir, input file)."""
    in_dir = workspace.monolingual.parent
    (tmp_path / "x").mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(in_dir, target_is_directory=True)
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    (elsewhere / "link.parquet").symlink_to(workspace.monolingual)
    beside = tmp_path / "beside"
    beside.mkdir()
    (beside / "link.parquet").symlink_to(workspace.monolingual)
    return {
        "plain": (in_dir, workspace.monolingual),
        "dotdot": (tmp_path / "x" / ".." / "in", workspace.monolingual),
        "symlinked-out-dir": (alias, workspace.monolingual),
        "input-behind-a-link": (in_dir, elsewhere / "link.parquet"),
        "link-to-the-input-in-the-out-dir": (beside, beside / "link.parquet"),
    }


@pytest.mark.parametrize("extra", [[], ["--dry-run"]])
@pytest.mark.parametrize(
    "spelling",
    [
        "plain",
        "dotdot",
        "symlinked-out-dir",
        "input-behind-a-link",
        "link-to-the-input-in-the-out-dir",
    ],
)
def test_an_out_dir_that_is_the_directory_of_the_input_is_a_config_error(
    workspace: Workspace,
    seams: Seams,
    tmp_path: Path,
    spelling: str,
    extra: list[str],
) -> None:
    out_dir, monolingual = _spellings(workspace, tmp_path)[spelling]
    before = _snapshot(tmp_path)

    result = runner.invoke(
        app, workspace.args(BT_JOB, *extra, monolingual=monolingual, out_dir=out_dir)
    )

    _assert_one_error(result, 2, f"--out-dir {out_dir} is the directory of the input")
    assert f"file {monolingual}; pass a different --out-dir" in result.stderr
    assert _snapshot(tmp_path) == before
    assert seams.generators_built == []


def test_an_out_dir_below_the_directory_of_the_input_is_allowed(
    workspace: Workspace, seams: Seams
) -> None:
    out_dir = workspace.monolingual.parent / "out"

    _run_ok(workspace.args(BT_JOB, out_dir=out_dir))

    assert (out_dir / "corpus.parquet").is_file()


def test_an_unknown_job_is_a_config_error_that_lists_the_configured_jobs(
    workspace: Workspace, seams: Seams
) -> None:
    result = runner.invoke(app, workspace.args("bt_nope"))

    _assert_one_error(result, 2, "unknown job 'bt_nope'")
    assert "bt_eng_khk" in result.stderr
    assert "ft_khk_eng" in result.stderr
    assert not workspace.out_dir.exists()
    assert seams.generators_built == []


def test_a_config_without_jobs_says_so_when_a_job_is_asked_for(
    workspace: Workspace, seams: Seams
) -> None:
    workspace.config.write_text("generators:\n  g: {kind: vllm}\n", encoding="utf-8")

    result = runner.invoke(app, workspace.args())

    _assert_one_error(result, 2, "unknown job 'bt_eng_khk'; configured jobs: none")


def test_a_missing_input_is_a_data_error(
    workspace: Workspace, seams: Seams, tmp_path: Path
) -> None:
    missing = tmp_path / "in" / "missing.parquet"

    result = runner.invoke(app, workspace.args(monolingual=missing))

    _assert_one_error(result, 3, f"monolingual file not found: {missing}")
    assert not workspace.out_dir.exists()
    assert seams.generators_built == []


def test_an_input_that_is_not_parquet_is_a_data_error(
    workspace: Workspace, seams: Seams
) -> None:
    workspace.monolingual.write_bytes(b"not a parquet file")

    result = runner.invoke(app, workspace.args())

    _assert_one_error(result, 3, "cannot read monolingual file")
    assert not workspace.out_dir.exists()


def test_an_input_without_rows_in_the_language_of_the_job_is_a_data_error(
    workspace: Workspace, seams: Seams
) -> None:
    planted_monolingual(8, lang="eng_Latn").write_parquet(workspace.monolingual)

    result = runner.invoke(app, workspace.args())

    _assert_one_error(
        result, 3, f"{workspace.monolingual}: no rows with lang 'khk_Cyrl'"
    )
    assert not workspace.out_dir.exists()
    assert seams.generators_built == []


def test_a_missing_config_is_a_config_error(
    workspace: Workspace, seams: Seams, tmp_path: Path
) -> None:
    missing = tmp_path / "missing.yaml"

    result = runner.invoke(app, workspace.args(config=missing))

    _assert_one_error(result, 2, f"synth config file not found: {missing}")
    assert not workspace.out_dir.exists()


def test_an_invalid_config_is_a_config_error(
    workspace: Workspace, seams: Seams
) -> None:
    workspace.config.write_text("generators: [not, a, mapping]\n", encoding="utf-8")

    result = runner.invoke(app, workspace.args())

    assert result.exit_code == 2, result.output
    assert result.stderr.startswith("error: invalid synth config")
    assert not workspace.out_dir.exists()


# --- the real seams ----------------------------------------------------------


def test_build_translator_picks_the_client_of_the_generator_kind(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    built: list[Ct2Config] = []
    monkeypatch.setattr(
        synth_cli, "CT2Translator", lambda cfg: built.append(cfg) or "a ct2 model"
    )
    ct2 = Ct2Config(model_dir=Path("m"), spm_model=Path("s"), family="nllb")

    assert type(synth_cli.build_translator(VllmConfig())) is OpenAICompatTranslator
    assert type(synth_cli.build_translator(LlamaCppConfig())) is LlamaCppTranslator
    assert synth_cli.build_translator(ct2) == "a ct2 model"
    assert built == [ct2]


def test_build_qe_scorer_builds_the_eole_scorer_only_for_metricx() -> None:
    scorer = synth_cli.build_qe_scorer(SelectionConfig())

    assert isinstance(scorer, EoleMetricXQEScorer)
    assert (
        synth_cli.build_qe_scorer(SelectionConfig(qe=QeConfig(scorer="cometkiwi")))
        is None
    )
