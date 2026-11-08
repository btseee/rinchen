"""`rinchen train ablate`: the plan it prints, the progress it reports, the manifest it
writes last, the order of its checks and writes, and the exit code of each failure.

Eole never starts: the training and the evaluation come from the module-level seams of
the CLI (`run_training`, `evaluate_variant`), which tests replace with the fakes of
`tests/support/fake_ablation.py` (they write a checkpoint file and return scripted dev
scores). The toy corpus and the toy SentencePiece model are real, so the plan, the
corpora, the validation files and the rendered Eole configs are the ones a real run
makes. Everything lands under pytest's temporary directories.

The numbers of the default plan are worked out in `tests/unit/test_train_ablation_plan.py`:
5,600 rows read (3,600 training rows, 1,200 back-translated, 400 dev, 400 test), 4,800
kept, 800 left out as `not_train`, and the weights 22 6 22 6 15 7 15 7.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import re
import shutil
from collections.abc import Callable
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import polars as pl
import pytest
from click.testing import Result
from typer.testing import CliRunner

from rinchen.cli import train as train_cli
from rinchen.cli.main import app
from rinchen.errors import ConfigError, DataError, RinchenError, UpstreamError
from rinchen.manifest import store
from rinchen.manifest.model import Manifest
from rinchen.train.ablation_plan import AblationPlan
from rinchen.train.ablation_records import AblationResult
from rinchen.train.eole_cfg import read_config, validate_train_config
from tests.support.corpus_rows import corpus_row
from tests.support.fake_ablation import (
    DIRECTIONS,
    ID_SRC,
    ID_TGT,
    FakeClock,
    FakeEvaluate,
    FakeTrain,
)
from tests.support.tiny_parallel import train_toy_spm, write_corpus_parquet
from tests.support.train_cli import (
    RESULT_FILE_NAMES,
    VARIANT_IDS,
    Seams,
    Workspace,
    install_seams,
    make_workspace,
    spelling_of_one_place,
    write_matrix_file,
)

runner = CliRunner()

OPTIONS = [
    "--matrix",
    "--corpus",
    "--spm-model",
    "--out-dir",
    "--threads",
    "--timeout",
    "--force",
    "--dry-run",
]
TOOL_PACKAGES = ["eole", "torch", "sentencepiece", "sacrebleu", "polars"]
EN_MN = "eng_Latn-khk_Cyrl"
MN_EN = "khk_Cyrl-eng_Latn"
MN_RU = "khk_Cyrl-rus_Cyrl"
RU_MN = "rus_Cyrl-khk_Cyrl"
# (corpus, rows, probability, weight, upsampling) of both variants, as printed.
PLANNED_CORPORA = [
    (f"{EN_MN}.auth", 1200, "0.225403", 22, "0.880"),
    (f"{EN_MN}.bt", 300, "0.056351", 6, "0.960"),
    (f"{MN_EN}.auth", 1200, "0.225403", 22, "0.880"),
    (f"{MN_EN}.bt", 300, "0.056351", 6, "0.960"),
    (f"{MN_RU}.auth", 600, "0.145497", 15, "1.200"),
    (f"{MN_RU}.bt", 300, "0.072749", 7, "1.120"),
    (f"{RU_MN}.auth", 600, "0.145497", 15, "1.200"),
    (f"{RU_MN}.bt", 300, "0.072749", 7, "1.120"),
]
# (direction, bt rows there are, bt rows the ratio asks for, bt rows used).
BT_USAGE = [
    (EN_MN, 300, 600, 300),
    (MN_EN, 300, 600, 300),
    (MN_RU, 300, 300, 300),
    (RU_MN, 300, 300, 300),
]


@pytest.fixture
def workspace(tmp_path: Path) -> Workspace:
    return make_workspace(tmp_path)


@pytest.fixture
def seams(monkeypatch: pytest.MonkeyPatch) -> Seams:
    return install_seams(monkeypatch)


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


def _without_option(args: list[str], option: str) -> list[str]:
    """`args` without `option` and its value (every occurrence of a repeated one)."""
    kept: list[str] = []
    skip = False
    for argument in args:
        if skip:
            skip = False
        elif argument == option:
            skip = True
        else:
            kept.append(argument)
    return kept


def _dry_run_lines(workspace: Workspace) -> list[str]:
    return _run_ok(workspace.args("--dry-run")).stdout.splitlines()


def _is_progress_line(line: str, position: str, variant_id: str, outcome: str) -> bool:
    """`line` is the progress line of the variant at `position` (`1/2`), with seconds
    that a real clock decides."""
    return (
        re.fullmatch(
            rf"variant {position} {re.escape(variant_id)} {outcome} "
            r"train \d+\.\d s eval \d+\.\d s",
            line,
        )
        is not None
    )


def _no_seam_was_called(seams: Seams) -> bool:
    return seams.train.calls == [] and seams.evaluate.calls == []


# --- help and options --------------------------------------------------------------


def test_the_train_group_lists_the_ablate_command() -> None:
    result = runner.invoke(app, ["train", "--help"])

    assert result.exit_code == 0
    assert "ablate" in result.stdout


def test_ablate_help_names_every_option() -> None:
    result = runner.invoke(app, ["train", "ablate", "--help"])

    assert result.exit_code == 0
    for option in OPTIONS:
        assert option in result.stdout


@pytest.mark.parametrize("option", ["--matrix", "--corpus", "--spm-model", "--out-dir"])
def test_a_required_option_that_is_missing_is_a_usage_error(
    workspace: Workspace, seams: Seams, option: str
) -> None:
    result = runner.invoke(app, _without_option(workspace.args(), option))

    assert result.exit_code == 2, result.output
    assert f"Missing option '{option}'" in result.stderr
    assert not workspace.out_dir.exists()
    assert _no_seam_was_called(seams)


def test_the_defaults_are_two_threads_and_no_timeout(
    workspace: Workspace, seams: Seams
) -> None:
    _run_ok(workspace.args())

    assert [(call.threads, call.timeout) for call in seams.train.calls] == [
        (2, None)
    ] * 2
    assert [(call.threads, call.timeout) for call in seams.evaluate.calls] == [
        (2, None)
    ] * 2


def test_the_threads_and_the_timeout_reach_both_seams(
    workspace: Workspace, seams: Seams
) -> None:
    _run_ok(workspace.args("--threads", "3", "--timeout", "12.5"))

    assert [(call.threads, call.timeout) for call in seams.train.calls] == [
        (3, 12.5)
    ] * 2
    assert [(call.threads, call.timeout) for call in seams.evaluate.calls] == [
        (3, 12.5)
    ] * 2


def test_fewer_than_one_thread_is_a_usage_error(
    workspace: Workspace, seams: Seams
) -> None:
    result = runner.invoke(app, workspace.args("--threads", "0"))

    assert result.exit_code == 2, result.output
    assert "--threads" in result.stderr
    assert not workspace.out_dir.exists()
    assert _no_seam_was_called(seams)


@pytest.mark.parametrize("timeout", ["0", "-5", "nan", "inf"])
@pytest.mark.parametrize("extra", [[], ["--dry-run"]])
def test_a_timeout_that_is_not_a_positive_number_is_a_config_error(
    workspace: Workspace, seams: Seams, timeout: str, extra: list[str]
) -> None:
    result = runner.invoke(app, workspace.args("--timeout", timeout, *extra))

    _assert_one_error(result, 2, "--timeout must be a positive, finite number")
    assert not workspace.out_dir.exists()
    assert _no_seam_was_called(seams)


# --- the plan (dry run) ------------------------------------------------------------


def _planned_variant_lines(variant_id: str, placement: str) -> list[str]:
    lines = [
        (
            f"variant {variant_id}: temperature 0.5, bt ratio 0.5, tag placement "
            f"{placement}, seed 1234"
        )
    ]
    lines += [
        f"variant {variant_id} bt {direction}: available {available}, requested "
        f"{requested}, used {used}"
        for direction, available, requested, used in BT_USAGE
    ]
    lines += [
        f"variant {variant_id} corpus {name}: rows {rows}, probability {probability}, "
        f"weight {weight}, upsampling {upsampling}"
        for name, rows, probability, weight, upsampling in PLANNED_CORPORA
    ]
    return lines


def test_dry_run_prints_the_matrix_the_statistics_and_every_variants_weights(
    workspace: Workspace, seams: Seams
) -> None:
    result = _run_ok(workspace.args("--dry-run"))

    assert result.stdout.splitlines() == [
        "matrix: toy",
        f"out dir: {workspace.out_dir}",
        "weight total: 100",
        "corpus files: 1",
        "rows read: 5600",
        "rows kept: 4800",
        "excluded not_train: 800",
        "excluded filtered: 0",
        "excluded excluded_origin: 0",
        "excluded empty: 0",
        "excluded line_break: 0",
        "excluded by origin ft: 0",
        "excluded by origin synth_llm: 0",
        "excluded by origin synth_nmt: 0",
        "duplicate ids: 0",
        "source collision rate: 0.000000",
        f"directions: {EN_MN}, {MN_EN}, {MN_RU}, {RU_MN}",
        "dev rows read: 5600",
        "dev rows usable: 400",
        *(f"dev lines {direction}: 100" for direction in DIRECTIONS),
        "dev in train: 0",
        "variants: 2",
        *_planned_variant_lines(ID_SRC, "src"),
        *_planned_variant_lines(ID_TGT, "tgt"),
        "nothing was trained and nothing was written",
    ]
    assert result.stderr == ""


def test_dry_run_reports_the_back_translated_rows_there_are_asked_for_and_used(
    tmp_path: Path, seams: Seams
) -> None:
    workspace = make_workspace(
        tmp_path,
        axes={"temperature": [0.5], "bt_ratio": [0.1], "tag_placement": ["src"]},
    )

    lines = _dry_run_lines(workspace)

    variant = "t0.5_bt0.1_src_s1234"
    assert [line for line in lines if line.startswith(f"variant {variant} bt ")] == [
        f"variant {variant} bt {EN_MN}: available 300, requested 120, used 120",
        f"variant {variant} bt {MN_EN}: available 300, requested 120, used 120",
        f"variant {variant} bt {MN_RU}: available 300, requested 60, used 60",
        f"variant {variant} bt {RU_MN}: available 300, requested 60, used 60",
    ]


def test_dry_run_prints_the_weight_total_of_the_matrix_and_weights_that_add_up_to_it(
    tmp_path: Path, seams: Seams
) -> None:
    workspace = make_workspace(tmp_path, weight_total=50)

    lines = _dry_run_lines(workspace)

    assert "weight total: 50" in lines
    weights = [
        int(line.split("weight ")[1].split(",")[0])
        for line in lines
        if line.startswith(f"variant {ID_SRC} corpus ")
    ]
    assert len(weights) == 8 and sum(weights) == 50


def test_dry_run_creates_nothing_and_starts_no_training(
    workspace: Workspace, seams: Seams, tmp_path: Path
) -> None:
    before = _snapshot(tmp_path)

    _run_ok(workspace.args("--dry-run"))

    assert not workspace.out_dir.exists()
    assert _snapshot(tmp_path) == before
    assert _no_seam_was_called(seams)


def test_dry_run_hashes_nothing(
    workspace: Workspace, seams: Seams, monkeypatch: pytest.MonkeyPatch
) -> None:
    def refuse_to_hash(*args: object) -> None:
        raise AssertionError("a dry run hashed an input")

    monkeypatch.setattr(train_cli, "build_inputs", refuse_to_hash)

    _run_ok(workspace.args("--dry-run"))


def test_dry_run_prints_what_plan_ablation_returns(
    workspace: Workspace, seams: Seams, monkeypatch: pytest.MonkeyPatch
) -> None:
    planned: list[AblationPlan] = []
    real_plan = train_cli.plan_ablation

    def spying_plan(*args: Any, **kwargs: Any) -> AblationPlan:
        planned.append(real_plan(*args, **kwargs))
        return planned[-1]

    monkeypatch.setattr(train_cli, "plan_ablation", spying_plan)

    lines = _dry_run_lines(workspace)

    (plan,) = planned
    assert f"rows kept: {plan.training.stats.rows_kept}" in lines
    for variant in plan.variants:
        for corpus in variant.corpora:
            assert (
                f"variant {variant.variant.id} corpus {corpus.name}: rows "
                f"{corpus.rows}, probability {corpus.probability:.6f}, weight "
                f"{corpus.weight}, upsampling {corpus.upsampling:.3f}"
            ) in lines


def test_dry_run_reads_every_corpus_file(
    workspace: Workspace, seams: Seams, tmp_path: Path
) -> None:
    extra = write_corpus_parquet(
        [corpus_row(f"Extra {n}", f"Нэмэлт {n}") for n in range(5)],
        tmp_path / "more" / "extra.parquet",
    )

    lines = _run_ok(
        workspace.args("--dry-run", corpora=(workspace.corpus, extra))
    ).stdout.splitlines()

    assert "corpus files: 2" in lines
    assert "rows read: 5605" in lines
    assert "rows kept: 4805" in lines


def test_dry_run_leaves_a_finished_out_dir_as_it_was(
    workspace: Workspace, seams: Seams
) -> None:
    _run_ok(workspace.args())
    before = _snapshot(workspace.out_dir)

    _run_ok(workspace.args("--dry-run"))

    assert _snapshot(workspace.out_dir) == before


def test_dry_run_of_a_missing_sentencepiece_model_is_a_config_error(
    workspace: Workspace, seams: Seams, tmp_path: Path
) -> None:
    missing = tmp_path / "missing.model"

    result = runner.invoke(app, workspace.args("--dry-run", spm_model=missing))

    _assert_one_error(result, 2, f"SentencePiece model not found: {missing}")
    assert not workspace.out_dir.exists()


# --- a run -------------------------------------------------------------------------


def _timed_run(
    monkeypatch: pytest.MonkeyPatch, workspace: Workspace, *extra: str
) -> Result:
    """A run whose clock is scripted: variant 1 trains 55.34 s and evaluates 2.62 s,
    variant 2 trains 40.06 s and evaluates 1.51 s."""
    clock = FakeClock()
    install_seams(
        monkeypatch,
        train=FakeTrain(clock=clock, seconds={ID_SRC: 55.34, ID_TGT: 40.06}),
        evaluate=FakeEvaluate(clock=clock, seconds={ID_SRC: 2.62, ID_TGT: 1.51}),
    )
    real_run = train_cli.run_ablation

    def run_with_the_scripted_clock(
        plan: AblationPlan, **kwargs: Any
    ) -> AblationResult:
        return real_run(plan, clock=clock, **kwargs)

    monkeypatch.setattr(train_cli, "run_ablation", run_with_the_scripted_clock)
    return _run_ok(workspace.args(*extra))


def test_a_run_reports_one_line_per_variant_and_names_the_result_files(
    workspace: Workspace, monkeypatch: pytest.MonkeyPatch
) -> None:
    result = _timed_run(monkeypatch, workspace)

    assert result.stdout.splitlines() == [
        f"variant 1/2 {ID_SRC} trained train 55.3 s eval 2.6 s",
        f"variant 2/2 {ID_TGT} trained train 40.1 s eval 1.5 s",
        "variants: 2 (trained 2, evaluated 0, reused 0)",
        f"manifest: {workspace.manifest}",
        f"results.tsv: {workspace.out_dir / 'results.tsv'}",
        f"results.json: {workspace.out_dir / 'results.json'}",
        f"results.md: {workspace.out_dir / 'results.md'}",
    ]
    assert result.stderr == ""


def test_a_run_leaves_the_tables_the_variants_and_the_manifest(
    workspace: Workspace, seams: Seams
) -> None:
    _run_ok(workspace.args())

    assert sorted(path.name for path in workspace.out_dir.iterdir()) == [
        "data",
        "manifest.json",
        "results.json",
        "results.md",
        "results.tsv",
        "variants",
    ]
    assert sorted(path.name for path in (workspace.out_dir / "variants").iterdir()) == (
        sorted(VARIANT_IDS)
    )
    rows = workspace.out_dir.joinpath("results.tsv").read_text(encoding="utf-8")
    assert [line.split("\t")[0] for line in rows.splitlines()] == [
        "variant",
        *VARIANT_IDS,
    ]
    assert _part_files(workspace.out_dir) == []


def test_each_variant_trains_on_its_own_validated_config_and_is_evaluated_on_the_dev_sets(
    workspace: Workspace, seams: Seams
) -> None:
    _run_ok(workspace.args())

    assert seams.train.variant_ids == list(VARIANT_IDS)
    assert seams.evaluate.variant_ids == list(VARIANT_IDS)
    for call in seams.train.calls:
        assert call.validate is validate_train_config
        assert call.cfg_path == workspace.config_file(call.variant_id)
        assert call.run_dir == workspace.variant_dir(call.variant_id) / "run"
    for call in seams.evaluate.calls:
        assert sorted(call.dev) == list(DIRECTIONS)
        assert call.spm_model == workspace.spm_model
        assert call.out_dir == workspace.variant_dir(call.variant_id) / "eval"


def test_the_manifest_records_inputs_outputs_config_and_environment(
    workspace: Workspace, seams: Seams
) -> None:
    _run_ok(workspace.args())

    manifest = _manifest(workspace)
    assert manifest.stage == "train_ablate"
    assert manifest.seed == 1234
    assert manifest.config_sha256 == _sha256(workspace.matrix)
    assert manifest.code_version
    assert datetime.fromisoformat(manifest.created).utcoffset() == timedelta(0)
    assert [(i.path, i.sha256, i.bytes, i.rows) for i in manifest.inputs] == [
        (str(path), _sha256(path), path.stat().st_size, None)
        for path in workspace.inputs
    ]
    expected_outputs = [
        (workspace.out_dir / "results.tsv", 2),
        (workspace.out_dir / "results.json", None),
        (workspace.out_dir / "results.md", None),
    ]
    for variant_id in VARIANT_IDS:
        expected_outputs += [
            (workspace.config_file(variant_id), None),
            (workspace.checkpoint(variant_id), None),
        ]
    assert [(o.path, o.sha256, o.bytes, o.rows) for o in manifest.outputs] == [
        (str(path), _sha256(path), path.stat().st_size, rows)
        for path, rows in expected_outputs
    ]
    assert list(manifest.tool_versions) == TOOL_PACKAGES
    assert manifest.tool_versions["polars"] == pl.__version__
    for package in ("eole", "torch", "sentencepiece", "sacrebleu"):
        assert manifest.tool_versions[package] == importlib.metadata.version(package)
    assert manifest.status == {"ablation": "ok"}


def test_the_manifest_counts_the_rows_the_variants_and_the_run_parameters(
    workspace: Workspace, seams: Seams
) -> None:
    _run_ok(workspace.args("--threads", "3"))

    directions = {direction: 100.0 for direction in DIRECTIONS}
    assert _manifest(workspace).metrics == {
        "count.rows_read": 5600.0,
        "count.rows_kept": 4800.0,
        "count.excluded.not_train": 800.0,
        "count.excluded.filtered": 0.0,
        "count.excluded.excluded_origin": 0.0,
        "count.excluded.empty": 0.0,
        "count.excluded.line_break": 0.0,
        "count.excluded_by_origin.ft": 0.0,
        "count.excluded_by_origin.synth_llm": 0.0,
        "count.excluded_by_origin.synth_nmt": 0.0,
        "count.duplicate_ids": 0.0,
        **{
            f"count.rows_by_corpus.{name}": float(rows)
            for name, rows, *_ in PLANNED_CORPORA
        },
        "count.source_collision_rate": 0.0,
        "count.dev.rows_read": 5600.0,
        "count.dev.rows_usable": 400.0,
        "count.dev.excluded.not_dev": 5200.0,
        "count.dev.excluded.filtered": 0.0,
        "count.dev.excluded.empty": 0.0,
        "count.dev.excluded.line_break": 0.0,
        **{f"count.dev.available_by_direction.{d}": n for d, n in directions.items()},
        **{f"count.dev.lines_by_direction.{d}": n for d, n in directions.items()},
        "count.dev.dev_in_train": 0.0,
        "variants.total": 2.0,
        "variants.trained": 2.0,
        "variants.evaluated": 0.0,
        "variants.reused": 0.0,
        "threads": 3.0,
        "seed.0": 1234.0,
    }


def test_the_manifest_lists_every_seed_of_the_matrix_and_takes_the_first_as_its_seed(
    tmp_path: Path, seams: Seams
) -> None:
    workspace = make_workspace(
        tmp_path,
        axes={"temperature": [0.5], "bt_ratio": [0.5], "tag_placement": ["src"]},
        seeds=[7, 1234],
    )

    _run_ok(workspace.args())

    manifest = _manifest(workspace)
    assert manifest.seed == 7
    assert {k: v for k, v in manifest.metrics.items() if k.startswith("seed.")} == {
        "seed.0": 7.0,
        "seed.1": 1234.0,
    }
    assert manifest.metrics["variants.total"] == 2.0


def test_the_checkpoint_hashes_of_the_manifest_are_those_of_the_files_and_of_the_results(
    workspace: Workspace, seams: Seams
) -> None:
    _run_ok(workspace.args())

    outputs = {item.path: item.sha256 for item in _manifest(workspace).outputs}
    for variant_id in VARIANT_IDS:
        checkpoint = workspace.checkpoint(variant_id)
        recorded = json.loads(workspace.result_file(variant_id).read_text("utf-8"))
        assert outputs[str(checkpoint)] == _sha256(checkpoint)
        assert recorded["checkpoint_sha256"] == _sha256(checkpoint)
    assert _sha256(workspace.checkpoint(ID_SRC)) != _sha256(
        workspace.checkpoint(ID_TGT)
    )


def test_every_rendered_config_passes_the_validator_and_its_weights_sum_to_100(
    workspace: Workspace, seams: Seams
) -> None:
    _run_ok(workspace.args())

    for variant_id in VARIANT_IDS:
        cfg = read_config(workspace.config_file(variant_id), "config")
        validate_train_config(cfg)
        weights = [
            corpus["weight"] for name, corpus in cfg["data"].items() if name != "valid"
        ]
        assert sorted(weights) == [6, 6, 7, 7, 15, 15, 22, 22]
        assert sum(weights) == 100


# --- the commit protocol -----------------------------------------------------------


def test_the_manifest_is_written_once_and_last_when_everything_else_is_on_disk(
    workspace: Workspace, seams: Seams, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: list[dict[str, Any]] = []
    real_write = store.write

    def spying_write(stage_dir: Path, manifest: Manifest) -> Path:
        seen.append(
            {
                "manifest_present": workspace.manifest.exists(),
                "results": [path.is_file() for path in workspace.result_files()],
                "configs": [workspace.config_file(v).is_file() for v in VARIANT_IDS],
                "checkpoints": [workspace.checkpoint(v).is_file() for v in VARIANT_IDS],
                "result_json": [
                    workspace.result_file(v).is_file() for v in VARIANT_IDS
                ],
                "part_files": _part_files(workspace.out_dir),
            }
        )
        return real_write(stage_dir, manifest)

    monkeypatch.setattr(store, "write", spying_write)

    _run_ok(workspace.args())

    assert seen == [
        {
            "manifest_present": False,
            "results": [True, True, True],
            "configs": [True, True],
            "checkpoints": [True, True],
            "result_json": [True, True],
            "part_files": [],
        }
    ]


class _ManifestProbingTrain(FakeTrain):
    def __init__(self, workspace: Workspace, record: dict[str, bool]) -> None:
        super().__init__()
        self.workspace = workspace
        self.record = record

    def __call__(self, *args: Any, **kwargs: Any) -> Path:
        self.record["training"] = self.workspace.manifest.exists()
        return super().__call__(*args, **kwargs)


def test_an_earlier_manifest_stands_while_the_plan_is_made_and_goes_before_any_output(
    workspace: Workspace, seams: Seams, monkeypatch: pytest.MonkeyPatch
) -> None:
    _run_ok(workspace.args())
    old_manifest = workspace.manifest.read_bytes()
    present_while: dict[str, bool] = {}
    real_plan = train_cli.plan_ablation
    real_run = train_cli.run_ablation

    def probing_plan(*args: Any, **kwargs: Any) -> AblationPlan:
        present_while["planning"] = workspace.manifest.exists()
        return real_plan(*args, **kwargs)

    def probing_run(plan: AblationPlan, **kwargs: Any) -> AblationResult:
        present_while["starting the run"] = workspace.manifest.exists()
        return real_run(plan, **kwargs)

    monkeypatch.setattr(train_cli, "plan_ablation", probing_plan)
    monkeypatch.setattr(train_cli, "run_ablation", probing_run)
    probe_seams = install_seams(
        monkeypatch, train=_ManifestProbingTrain(workspace, present_while)
    )

    _run_ok(workspace.args("--force"))

    assert present_while == {
        "planning": True,
        "starting the run": False,
        "training": False,
    }
    assert workspace.manifest.read_bytes() != old_manifest
    assert len(probe_seams.train.calls) == 2


def test_the_out_dir_is_made_after_the_plan_and_before_the_run_starts(
    workspace: Workspace, seams: Seams, monkeypatch: pytest.MonkeyPatch
) -> None:
    existed: dict[str, bool] = {}
    real_plan = train_cli.plan_ablation
    real_run = train_cli.run_ablation

    def probing_plan(*args: Any, **kwargs: Any) -> AblationPlan:
        plan = real_plan(*args, **kwargs)
        existed["after planning"] = workspace.out_dir.exists()
        return plan

    def probing_run(plan: AblationPlan, **kwargs: Any) -> AblationResult:
        existed["starting the run"] = workspace.out_dir.is_dir()
        return real_run(plan, **kwargs)

    monkeypatch.setattr(train_cli, "plan_ablation", probing_plan)
    monkeypatch.setattr(train_cli, "run_ablation", probing_run)

    _run_ok(workspace.args())

    assert existed == {"after planning": False, "starting the run": True}


def test_an_out_dir_that_cannot_be_created_is_a_plain_failure_before_any_seam(
    workspace: Workspace, seams: Seams, tmp_path: Path
) -> None:
    blocker = tmp_path / "blocker"
    blocker.write_text("a file where a directory should go", encoding="utf-8")

    result = runner.invoke(app, workspace.args(out_dir=blocker / "out"))

    _assert_one_error(
        result, 1, f"cannot write the ablation outputs under {blocker / 'out'}: "
    )
    assert _no_seam_was_called(seams)
    assert not (blocker / "out").exists()


def test_a_failure_of_the_plan_creates_nothing_and_keeps_an_earlier_manifest(
    workspace: Workspace, seams: Seams
) -> None:
    _run_ok(workspace.args())
    committed = _snapshot(workspace.out_dir)
    fresh_out = workspace.root / "fresh"
    unknown_language = write_corpus_parquet(
        [corpus_row("Hello", "Hi", src_lang="xxx_Latn", tgt_lang="khk_Cyrl")],
        workspace.root / "odd" / "unknown.parquet",
    )

    again = runner.invoke(app, workspace.args(corpora=(unknown_language,)))
    elsewhere = runner.invoke(
        app, workspace.args(corpora=(unknown_language,), out_dir=fresh_out)
    )

    for result in (again, elsewhere):
        _assert_one_error(result, 2, "training rows use languages that are not in")
    assert _snapshot(workspace.out_dir) == committed
    assert not fresh_out.exists()


@pytest.mark.parametrize("extra", [[], ["--dry-run"]])
def test_a_typo_in_an_override_is_a_config_error_before_anything_is_written_or_deleted(
    workspace: Workspace, seams: Seams, tmp_path: Path, extra: list[str]
) -> None:
    # The dry run passes for a run that would fail, and the real run used to delete
    # the manifest of the earlier run and write the data before it noticed.
    _run_ok(workspace.args())
    committed = _snapshot(workspace.out_dir)
    for seam in (seams.train, seams.evaluate):
        seam.calls.clear()
    write_matrix_file(
        workspace.matrix,
        workspace.base_config,
        overrides={"training": {"train_stepz": 100}},
    )
    fresh_out = tmp_path / "fresh"

    again = runner.invoke(app, workspace.args(*extra))
    elsewhere = runner.invoke(app, workspace.args(*extra, out_dir=fresh_out))

    for result in (again, elsewhere):
        _assert_one_error(result, 2, "invalid Eole config: training.train_stepz")
    assert _snapshot(workspace.out_dir) == committed
    assert workspace.manifest.is_file()
    assert not fresh_out.exists()
    assert _no_seam_was_called(seams)


@pytest.mark.parametrize("extra", [[], ["--dry-run"]])
@pytest.mark.parametrize("what", ["garbage", "directory"])
def test_a_sentencepiece_model_that_does_not_load_is_a_config_error_before_any_write(
    workspace: Workspace, seams: Seams, tmp_path: Path, what: str, extra: list[str]
) -> None:
    _run_ok(workspace.args())
    committed = _snapshot(workspace.out_dir)
    for seam in (seams.train, seams.evaluate):
        seam.calls.clear()
    bad = tmp_path / "bad.model"
    if what == "garbage":
        bad.write_bytes(b"not a SentencePiece model")
    else:
        bad.mkdir()

    result = runner.invoke(app, workspace.args(*extra, spm_model=bad))

    _assert_one_error(result, 2, "SentencePiece model")
    assert _snapshot(workspace.out_dir) == committed
    assert _no_seam_was_called(seams)


# --- resuming ----------------------------------------------------------------------


def test_a_rerun_reuses_every_variant_says_so_and_trains_nothing(
    workspace: Workspace, seams: Seams
) -> None:
    first = _run_ok(workspace.args())
    first_files = {
        name: workspace.out_dir.joinpath(name).read_bytes()
        for name in RESULT_FILE_NAMES
    }
    first_manifest = _manifest(workspace)

    second = _run_ok(workspace.args())

    assert [line.split()[:4] for line in first.stdout.splitlines()[:2]] == [
        ["variant", "1/2", ID_SRC, "trained"],
        ["variant", "2/2", ID_TGT, "trained"],
    ]
    assert second.stdout.splitlines()[:3] == [
        f"variant 1/2 {ID_SRC} reused train 0.0 s eval 0.0 s",
        f"variant 2/2 {ID_TGT} reused train 0.0 s eval 0.0 s",
        "variants: 2 (trained 0, evaluated 0, reused 2)",
    ]
    assert len(seams.train.calls) == len(seams.evaluate.calls) == 2
    assert {
        name: workspace.out_dir.joinpath(name).read_bytes()
        for name in RESULT_FILE_NAMES
    } == first_files
    again = _manifest(workspace)
    assert again.model_copy(update={"created": "", "metrics": {}}) == (
        first_manifest.model_copy(update={"created": "", "metrics": {}})
    )
    assert again.metrics["variants.reused"] == 2.0
    assert again.metrics["variants.trained"] == 0.0


def test_force_trains_every_variant_again(workspace: Workspace, seams: Seams) -> None:
    _run_ok(workspace.args())
    before = {v: workspace.checkpoint(v).read_bytes() for v in VARIANT_IDS}

    forced = _run_ok(workspace.args("--force"))

    assert [line.split()[3] for line in forced.stdout.splitlines()[:2]] == [
        "trained"
    ] * 2
    assert len(seams.train.calls) == len(seams.evaluate.calls) == 4
    for variant_id in VARIANT_IDS:
        assert workspace.checkpoint(variant_id).read_bytes() != before[variant_id]
    assert _manifest(workspace).metrics["variants.trained"] == 2.0


# --- failures ----------------------------------------------------------------------


@pytest.mark.parametrize(
    ("error", "exit_code"),
    [
        (ConfigError("the config is odd"), 2),
        (DataError("the data is odd"), 3),
        (UpstreamError("eole exited with code 1"), 4),
        (RinchenError("some other trouble"), 1),
    ],
    ids=["ConfigError", "DataError", "UpstreamError", "RinchenError"],
)
def test_a_failing_variant_stops_the_run_with_the_code_of_its_error_and_keeps_the_first(
    workspace: Workspace,
    monkeypatch: pytest.MonkeyPatch,
    error: RinchenError,
    exit_code: int,
) -> None:
    seams = install_seams(monkeypatch, train=FakeTrain(fail={ID_TGT: error}))

    result = runner.invoke(app, workspace.args())

    _assert_one_error(result, exit_code, str(error))
    (line,) = result.stdout.splitlines()
    assert _is_progress_line(line, "1/2", ID_SRC, "trained")
    assert seams.train.variant_ids == [ID_SRC, ID_TGT]
    assert workspace.result_file(ID_SRC).is_file()
    assert workspace.checkpoint(ID_SRC).is_file()
    assert not workspace.result_file(ID_TGT).exists()
    assert not workspace.manifest.exists()
    assert [path.exists() for path in workspace.result_files()] == [False] * 3
    assert _part_files(workspace.out_dir) == []


def test_a_failing_evaluation_stops_the_run_with_exit_4_and_no_manifest(
    workspace: Workspace, monkeypatch: pytest.MonkeyPatch
) -> None:
    install_seams(
        monkeypatch, evaluate=FakeEvaluate(fail={ID_SRC: UpstreamError("predict died")})
    )

    result = runner.invoke(app, workspace.args())

    _assert_one_error(result, 4, "predict died")
    assert not workspace.manifest.exists()
    assert not workspace.result_file(ID_SRC).exists()


def test_a_write_failure_inside_the_run_is_a_plain_failure_naming_the_out_dir(
    workspace: Workspace, monkeypatch: pytest.MonkeyPatch
) -> None:
    install_seams(monkeypatch, train=FakeTrain(fail={ID_SRC: OSError("disk full")}))

    result = runner.invoke(app, workspace.args())

    _assert_one_error(
        result, 1, f"cannot write the ablation outputs under {workspace.out_dir}: "
    )
    assert "disk full" in result.stderr
    assert not workspace.manifest.exists()


def test_an_output_path_blocked_by_a_file_is_a_plain_failure_naming_the_out_dir(
    workspace: Workspace, seams: Seams
) -> None:
    workspace.out_dir.mkdir()
    (workspace.out_dir / "data").write_text("a file, not a directory", encoding="utf-8")

    result = runner.invoke(app, workspace.args())

    _assert_one_error(
        result, 1, f"cannot write the ablation outputs under {workspace.out_dir}: "
    )
    assert _no_seam_was_called(seams)
    assert not workspace.manifest.exists()


def test_the_rerun_after_a_failure_reuses_the_finished_variant_and_trains_the_rest(
    workspace: Workspace, monkeypatch: pytest.MonkeyPatch
) -> None:
    install_seams(
        monkeypatch, train=FakeTrain(fail={ID_TGT: UpstreamError("eole died")})
    )
    runner.invoke(app, workspace.args())
    seams = install_seams(monkeypatch)

    result = _run_ok(workspace.args())

    assert [line.split()[:4] for line in result.stdout.splitlines()[:2]] == [
        ["variant", "1/2", ID_SRC, "reused"],
        ["variant", "2/2", ID_TGT, "trained"],
    ]
    assert seams.train.variant_ids == [ID_TGT]
    assert _manifest(workspace).metrics["variants.reused"] == 1.0
    assert all(path.is_file() for path in workspace.result_files())


def test_the_rerun_after_a_failing_evaluation_scores_the_trained_checkpoint_only(
    workspace: Workspace, monkeypatch: pytest.MonkeyPatch
) -> None:
    first = install_seams(
        monkeypatch, evaluate=FakeEvaluate(fail={ID_SRC: UpstreamError("predict died")})
    )
    failed = runner.invoke(app, workspace.args())
    _assert_one_error(failed, 4, "predict died")
    assert first.train.variant_ids == [ID_SRC]
    assert workspace.train_record(ID_SRC).is_file()
    assert not workspace.manifest.exists()
    seams = install_seams(monkeypatch)

    result = _run_ok(workspace.args())

    lines = result.stdout.splitlines()
    assert [line.split()[:4] for line in lines[:2]] == [
        ["variant", "1/2", ID_SRC, "evaluated"],
        ["variant", "2/2", ID_TGT, "trained"],
    ]
    # Nothing trained in this run for the first variant, whatever its training took.
    assert lines[0].split()[4:7] == ["train", "0.0", "s"]
    assert lines[2] == "variants: 2 (trained 1, evaluated 1, reused 0)"
    assert seams.train.variant_ids == [ID_TGT]
    assert seams.evaluate.variant_ids == [ID_SRC, ID_TGT]
    manifest = _manifest(workspace)
    assert manifest.metrics["variants.trained"] == 1.0
    assert manifest.metrics["variants.evaluated"] == 1.0
    assert manifest.metrics["variants.reused"] == 0.0
    outputs = {item.path: item.sha256 for item in manifest.outputs}
    for variant_id in (ID_SRC, ID_TGT):
        checkpoint = workspace.checkpoint(variant_id)
        assert outputs[str(checkpoint)] == _sha256(checkpoint)
    assert all(path.is_file() for path in workspace.result_files())
    assert str(workspace.train_record(ID_SRC)) not in outputs


def test_an_interrupt_exits_130_and_leaves_the_finished_variant_and_no_manifest(
    workspace: Workspace, monkeypatch: pytest.MonkeyPatch
) -> None:
    install_seams(monkeypatch)
    _run_ok(workspace.args())
    install_seams(monkeypatch, train=FakeTrain(fail={ID_TGT: KeyboardInterrupt()}))

    result = runner.invoke(app, workspace.args("--force"))

    assert result.exit_code == 130, result.output
    (line,) = result.stdout.splitlines()
    assert _is_progress_line(line, "1/2", ID_SRC, "trained")
    assert not workspace.manifest.exists()
    assert not workspace.result_file(ID_TGT).exists()
    assert workspace.result_file(ID_SRC).is_file()
    assert workspace.checkpoint(ID_SRC).is_file()
    assert _part_files(workspace.out_dir) == []


# --- inputs ------------------------------------------------------------------------


def _rewrite_text(path: Path, workspace: Workspace) -> None:
    path.write_text(path.read_text(encoding="utf-8") + "# edited\n", encoding="utf-8")


def _rewrite_parquet(path: Path, workspace: Workspace) -> None:
    pl.read_parquet(path).write_parquet(path, compression="uncompressed")


def _swap_for_another_model(path: Path, workspace: Workspace) -> None:
    other = train_toy_spm(workspace.root / "other_spm", vocab_size=600)
    shutil.copyfile(other, path)


@pytest.mark.parametrize(
    ("seam", "which", "rewrite"),
    [
        ("load_ablation", "matrix", _rewrite_text),
        ("plan_ablation", "corpus", _rewrite_parquet),
        ("run_ablation", "base_config", _rewrite_text),
        ("run_ablation", "spm_model", _swap_for_another_model),
    ],
    ids=["matrix", "corpus", "base config", "SentencePiece model"],
)
def test_the_manifest_vouches_for_the_bytes_an_input_had_before_it_was_read(
    workspace: Workspace,
    seams: Seams,
    monkeypatch: pytest.MonkeyPatch,
    seam: str,
    which: str,
    rewrite: Callable[[Path, Workspace], None],
) -> None:
    path: Path = getattr(workspace, which)
    before = [_sha256(each) for each in workspace.inputs]
    real = getattr(train_cli, seam)

    def after_a_change(*args: Any, **kwargs: Any) -> Any:
        rewrite(path, workspace)  # the file changes just before the reader gets it
        return real(*args, **kwargs)

    monkeypatch.setattr(train_cli, seam, after_a_change)

    _run_ok(workspace.args())

    assert _sha256(path) != before[workspace.inputs.index(path)]
    manifest = _manifest(workspace)
    assert [item.sha256 for item in manifest.inputs] == before
    assert manifest.config_sha256 == before[0]


@pytest.mark.parametrize(
    ("which", "what"),
    [
        ("matrix", "ablation config file"),
        ("base_config", "base Eole config"),
        ("corpus", "corpus file"),
        ("spm_model", "SentencePiece model"),
    ],
)
def test_an_input_that_cannot_be_read_is_a_data_error_naming_it(
    workspace: Workspace,
    seams: Seams,
    monkeypatch: pytest.MonkeyPatch,
    which: str,
    what: str,
) -> None:
    unreadable: Path = getattr(workspace, which)
    real_open = Path.open

    def open_unless_unreadable(self: Path, mode: str = "r", *args: Any, **kwargs: Any):
        if self == unreadable and "b" in mode:
            raise PermissionError(13, "Permission denied", str(self))
        return real_open(self, mode, *args, **kwargs)

    monkeypatch.setattr(Path, "open", open_unless_unreadable)

    result = runner.invoke(app, workspace.args())

    _assert_one_error(result, 3, f"cannot read {what} {unreadable}: ")
    assert not workspace.out_dir.exists()
    assert _no_seam_was_called(seams)


def test_a_missing_matrix_is_a_config_error(
    workspace: Workspace, seams: Seams, tmp_path: Path
) -> None:
    missing = tmp_path / "missing.yaml"

    result = runner.invoke(app, workspace.args(matrix=missing))

    _assert_one_error(result, 2, f"ablation config file not found: {missing}")
    assert not workspace.out_dir.exists()


def test_an_invalid_matrix_is_a_config_error(
    workspace: Workspace, seams: Seams
) -> None:
    workspace.matrix.write_text("name: toy\nbogus: 1\n", encoding="utf-8")

    result = runner.invoke(app, workspace.args())

    assert result.exit_code == 2, result.output
    assert result.stderr.startswith("error: invalid ablation config")
    assert not workspace.out_dir.exists()


def test_a_matrix_that_overrides_what_the_run_sets_is_a_config_error_at_plan_time(
    tmp_path: Path, seams: Seams
) -> None:
    workspace = make_workspace(tmp_path, overrides={"training": {"model_path": "x"}})

    result = runner.invoke(app, workspace.args())

    _assert_one_error(result, 2, "invalid overrides: training.model_path")
    assert not workspace.out_dir.exists()
    assert _no_seam_was_called(seams)


@pytest.mark.parametrize("extra", [[], ["--dry-run"]])
def test_a_temperature_whose_powers_overflow_is_a_config_error_naming_the_variant(
    tmp_path: Path, seams: Seams, extra: list[str]
) -> None:
    workspace = make_workspace(
        tmp_path,
        axes={"temperature": [400.0], "bt_ratio": [0.5], "tag_placement": ["src"]},
    )

    result = runner.invoke(app, workspace.args(*extra))

    _assert_one_error(result, 2, "variant t400_bt0.5_src_s1234: temperature 400.0")
    assert not workspace.out_dir.exists()
    assert _no_seam_was_called(seams)


@pytest.mark.parametrize("extra", [[], ["--dry-run"]])
def test_an_out_dir_with_a_dollar_sign_is_a_config_error(
    workspace: Workspace, seams: Seams, tmp_path: Path, extra: list[str]
) -> None:
    out_dir = tmp_path / "out$HOME"

    result = runner.invoke(app, workspace.args(*extra, out_dir=out_dir))

    _assert_one_error(result, 2, "the output directory holds")
    assert not out_dir.exists()
    assert _no_seam_was_called(seams)


def test_a_missing_sentencepiece_model_is_a_config_error(
    workspace: Workspace, seams: Seams, tmp_path: Path
) -> None:
    missing = tmp_path / "missing.model"

    result = runner.invoke(app, workspace.args(spm_model=missing))

    _assert_one_error(result, 2, f"SentencePiece model not found: {missing}")
    assert not workspace.out_dir.exists()
    assert _no_seam_was_called(seams)


@pytest.mark.parametrize("extra", [[], ["--dry-run"]])
def test_a_missing_corpus_is_a_data_error(
    workspace: Workspace, seams: Seams, tmp_path: Path, extra: list[str]
) -> None:
    missing = tmp_path / "in" / "missing.parquet"

    result = runner.invoke(
        app, workspace.args(*extra, corpora=(workspace.corpus, missing))
    )

    _assert_one_error(result, 3, f"corpus file not found: {missing}")
    assert not workspace.out_dir.exists()
    assert _no_seam_was_called(seams)


@pytest.mark.parametrize("extra", [[], ["--dry-run"]])
def test_a_corpus_that_is_not_parquet_is_a_data_error(
    workspace: Workspace, seams: Seams, extra: list[str]
) -> None:
    workspace.corpus.write_bytes(b"not a parquet file")

    result = runner.invoke(app, workspace.args(*extra))

    _assert_one_error(result, 3, "cannot read corpus file")
    assert not workspace.out_dir.exists()


@pytest.mark.parametrize("extra", [[], ["--dry-run"]])
def test_a_corpus_without_training_rows_is_a_data_error(
    workspace: Workspace, seams: Seams, extra: list[str]
) -> None:
    write_corpus_parquet([corpus_row("Hello", "Сайн", split="dev")], workspace.corpus)

    result = runner.invoke(app, workspace.args(*extra))

    _assert_one_error(result, 3, "no training rows")
    assert not workspace.out_dir.exists()
    assert _no_seam_was_called(seams)


@pytest.mark.parametrize("extra", [[], ["--dry-run"]])
def test_a_corpus_in_a_language_of_no_registry_entry_is_a_config_error(
    workspace: Workspace, seams: Seams, tmp_path: Path, extra: list[str]
) -> None:
    unknown = write_corpus_parquet(
        [corpus_row("Hello", "Hi", src_lang="xxx_Latn", tgt_lang="khk_Cyrl")],
        tmp_path / "odd" / "unknown.parquet",
    )

    result = runner.invoke(
        app, workspace.args(*extra, corpora=(workspace.corpus, unknown))
    )

    _assert_one_error(result, 2, "training rows use languages that are not in")
    assert "xxx_Latn" in result.stderr
    assert not workspace.out_dir.exists()


@pytest.mark.parametrize("extra", [[], ["--dry-run"]])
@pytest.mark.parametrize("repeat", ["same", "alias", "dotdot"])
def test_a_corpus_given_twice_is_a_config_error_naming_it(
    workspace: Workspace, seams: Seams, tmp_path: Path, repeat: str, extra: list[str]
) -> None:
    (tmp_path / "x").mkdir()
    alias = tmp_path / "alias.parquet"
    alias.symlink_to(workspace.corpus)
    again = {
        "same": workspace.corpus,
        "alias": alias,
        "dotdot": tmp_path / "x" / ".." / "in" / "toy.parquet",
    }[repeat]

    result = runner.invoke(
        app, workspace.args(*extra, corpora=(workspace.corpus, again))
    )

    _assert_one_error(
        result,
        2,
        f"--corpus {again} names the same file as --corpus {workspace.corpus}",
    )
    assert "twice" in result.stderr
    assert not workspace.out_dir.exists()
    assert _no_seam_was_called(seams)


def test_two_different_corpus_files_are_both_read(
    workspace: Workspace, seams: Seams, tmp_path: Path
) -> None:
    extra = write_corpus_parquet(
        [corpus_row(f"Extra {n}", f"Нэмэлт {n}") for n in range(5)],
        tmp_path / "more" / "extra.parquet",
    )

    _run_ok(workspace.args(corpora=(workspace.corpus, extra)))

    manifest = _manifest(workspace)
    assert [item.path for item in manifest.inputs] == [
        str(path)
        for path in (
            workspace.matrix,
            workspace.base_config,
            workspace.corpus,
            extra,
            workspace.spm_model,
        )
    ]
    assert manifest.metrics["count.rows_kept"] == 4805.0


# --- refusals ----------------------------------------------------------------------


@pytest.mark.parametrize("extra", [[], ["--force"], ["--dry-run"]])
def test_an_out_dir_holding_the_manifest_of_another_stage_is_a_data_error(
    workspace: Workspace, seams: Seams, extra: list[str]
) -> None:
    store.write(
        workspace.out_dir,
        Manifest(
            stage="synth",
            created="2026-11-07T00:00:00Z",
            code_version="deadbeef",
            config_sha256="cfg-abc",
            seed=0,
        ),
    )
    before = _snapshot(workspace.out_dir)

    result = runner.invoke(app, workspace.args(*extra))

    _assert_one_error(result, 3, f"{workspace.out_dir} holds the manifest of stage ")
    assert "'synth'" in result.stderr
    assert "'train_ablate'" in result.stderr
    assert _snapshot(workspace.out_dir) == before
    assert _no_seam_was_called(seams)


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
def test_an_out_dir_that_is_the_directory_of_a_corpus_is_a_config_error(
    workspace: Workspace, seams: Seams, tmp_path: Path, spelling: str, extra: list[str]
) -> None:
    out_dir, corpus = spelling_of_one_place(workspace, tmp_path)[spelling]
    before = _snapshot(tmp_path)

    result = runner.invoke(
        app, workspace.args(*extra, corpora=(corpus,), out_dir=out_dir)
    )

    _assert_one_error(result, 2, f"--out-dir {out_dir} is the directory of the input")
    assert f"file {corpus}; pass a different --out-dir" in result.stderr
    assert _snapshot(tmp_path) == before
    assert _no_seam_was_called(seams)


@pytest.mark.parametrize("extra", [[], ["--dry-run"]])
@pytest.mark.parametrize("which", ["matrix", "base_config", "spm_model"])
def test_an_out_dir_that_is_the_directory_of_another_input_is_a_config_error(
    workspace: Workspace, seams: Seams, tmp_path: Path, which: str, extra: list[str]
) -> None:
    if which == "base_config":
        configs = tmp_path / "configs"
        configs.mkdir()
        moved = configs / "base.yaml"
        shutil.move(workspace.base_config, moved)
        write_matrix_file(workspace.matrix, moved)
        path = moved
    else:
        path = getattr(workspace, which)

    result = runner.invoke(app, workspace.args(*extra, out_dir=path.parent))

    _assert_one_error(
        result, 2, f"--out-dir {path.parent} is the directory of the input file {path}"
    )
    assert _no_seam_was_called(seams)


def test_an_out_dir_below_the_directory_of_a_corpus_is_allowed(
    workspace: Workspace, seams: Seams
) -> None:
    out_dir = workspace.corpus.parent / "out"

    _run_ok(workspace.args(out_dir=out_dir))

    assert (out_dir / "results.tsv").is_file()


# --- odd paths ---------------------------------------------------------------------


@pytest.mark.parametrize(
    "name", ["out dir", "ünï-хураангуй"], ids=["space", "non-ASCII"]
)
def test_an_out_dir_with_a_space_or_non_ascii_letters_gets_every_output(
    workspace: Workspace, seams: Seams, tmp_path: Path, name: str
) -> None:
    out_dir = tmp_path / name

    _run_ok(workspace.args(out_dir=out_dir))

    assert sorted(path.name for path in out_dir.iterdir()) == [
        "data",
        "manifest.json",
        "results.json",
        "results.md",
        "results.tsv",
        "variants",
    ]
    manifest = store.read(out_dir)
    assert manifest is not None
    assert all(Path(item.path).is_file() for item in manifest.outputs)
    assert all(str(item.path).startswith(str(out_dir)) for item in manifest.outputs)
    assert manifest.metrics["variants.total"] == 2.0
    assert _part_files(out_dir) == []


@pytest.mark.parametrize("extra", [[], ["--dry-run"]])
@pytest.mark.parametrize("char", ["[", "]", "*", "?"])
def test_an_out_dir_with_a_glob_character_is_a_config_error_before_any_work(
    workspace: Workspace, seams: Seams, tmp_path: Path, char: str, extra: list[str]
) -> None:
    # Eole globs the checkpoint directory (`model.*.safetensors`): `out[1]` trained a
    # variant for 18 s and then failed to predict, `out*` loaded the checkpoints of
    # its siblings. Both the dry run and a real run refuse it, naming the character.
    out_dir = tmp_path / f"out{char}1"

    result = runner.invoke(app, workspace.args(*extra, out_dir=out_dir))

    _assert_one_error(
        result, 2, f"the output directory {out_dir} holds {char!r}, which Eole reads"
    )
    assert not out_dir.exists()
    assert _no_seam_was_called(seams)


def test_paths_with_spaces_work_for_every_input_and_the_out_dir(
    tmp_path: Path, seams: Seams
) -> None:
    workspace = make_workspace(tmp_path / "a dir with spaces")

    result = _run_ok(workspace.args("--timeout", "30"))

    assert workspace.manifest.is_file()
    assert result.stdout.splitlines()[-1] == (
        f"results.md: {workspace.out_dir / 'results.md'}"
    )
    manifest = _manifest(workspace)
    assert [item.path for item in manifest.inputs] == [str(p) for p in workspace.inputs]


def test_a_relative_out_dir_is_resolved_for_the_outputs(
    workspace: Workspace, seams: Seams, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(workspace.root)

    _run_ok(workspace.args(out_dir=Path("relative out")))

    manifest = store.read(workspace.root / "relative out")
    assert manifest is not None
    assert all(
        item.path.startswith(str(workspace.root / "relative out"))
        for item in manifest.outputs
    )
