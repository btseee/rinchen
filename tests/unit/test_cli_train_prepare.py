"""`rinchen train ablate --prepare-only`: what it renders, that it is what `ablate` would
have rendered, what it prints, the manifest of stage `train_prepare` and how that stage
and `train_ablate` keep apart, and the options it refuses.

Eole never starts, and neither does the ablation loop: the CLI's `run_training` and
`evaluate_variant` come from the same fakes as in `test_cli_train.py`, and the tests
check that neither is called. The toy corpus and the toy SentencePiece model are real, so
the corpora, validation files and Eole configs are the ones a real run makes.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import shutil
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from click.testing import Result
from typer.testing import CliRunner

from rinchen.cli import train as train_cli
from rinchen.cli import train_common
from rinchen.cli.main import app
from rinchen.errors import RinchenError
from rinchen.manifest import store
from rinchen.manifest.model import Manifest
from tests.support.fake_ablation import ID_SRC, ID_TGT
from tests.support.launch_cli import assert_one_error, snapshot
from tests.support.train_cli import (
    VARIANT_IDS,
    Seams,
    Workspace,
    install_seams,
    make_workspace,
)

runner = CliRunner()

TOOL_PACKAGES = ["eole", "torch", "sentencepiece", "sacrebleu", "polars"]


@pytest.fixture
def workspace(tmp_path: Path) -> Workspace:
    return make_workspace(tmp_path)


@pytest.fixture
def seams(monkeypatch: pytest.MonkeyPatch) -> Seams:
    return install_seams(monkeypatch)


@pytest.fixture
def loop_calls(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """The calls of the ablation loop (`run_ablation`), which still runs when called:
    preparing is not running, so a prepare must leave this empty."""
    calls: list[str] = []
    real = train_cli.run_ablation

    def recording(*args: Any, **kwargs: Any) -> Any:
        calls.append("run_ablation")
        return real(*args, **kwargs)

    monkeypatch.setattr(train_cli, "run_ablation", recording)
    return calls


def run_ok(args: list[str]) -> Result:
    result = runner.invoke(app, args)
    assert result.exit_code == 0, result.output
    return result


def prepare_args(workspace: Workspace, *extra: str) -> list[str]:
    return workspace.args("--prepare-only", *extra)


def manifest_of(workspace: Workspace) -> Manifest:
    manifest = store.read(workspace.out_dir)
    assert manifest is not None
    return manifest


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepared_files(out_dir: Path) -> dict[str, bytes]:
    """What preparing writes: every variant's config and validation files, and the
    corpus files."""
    found: dict[str, bytes] = {}
    for path in sorted(out_dir.rglob("*")):
        relative = path.relative_to(out_dir)
        if not path.is_file() or relative.parts[0] not in ("data", "variants"):
            continue
        if relative.parts[0] == "variants" and relative.name not in (
            "config.yaml",
            "valid.src",
            "valid.tgt",
        ):
            continue
        found[str(relative)] = path.read_bytes()
    return found


# --- help and options --------------------------------------------------------------------


def test_ablate_help_names_the_prepare_only_option() -> None:
    result = runner.invoke(app, ["train", "ablate", "--help"])

    assert result.exit_code == 0
    assert "--prepare-only" in result.stdout


# --- what it writes and prints -------------------------------------------------------------


def test_it_prints_one_line_per_variant_and_the_manifest(
    workspace: Workspace, seams: Seams
) -> None:
    lines = run_ok(prepare_args(workspace)).stdout.splitlines()

    assert lines == [
        *(
            f"prepared {variant_id}: {workspace.config_file(variant_id)}"
            for variant_id in VARIANT_IDS
        ),
        f"manifest: {workspace.manifest}",
    ]


def test_it_writes_the_configs_the_validation_files_and_the_corpora_and_nothing_else(
    workspace: Workspace, seams: Seams
) -> None:
    run_ok(prepare_args(workspace))

    for variant_id in VARIANT_IDS:
        directory = workspace.variant_dir(variant_id)
        assert sorted(path.name for path in directory.iterdir()) == [
            "config.yaml",
            "valid.src",
            "valid.tgt",
        ]
    assert (workspace.out_dir / "data" / "auth").is_dir()
    assert (workspace.out_dir / "data" / "bt").is_dir()
    assert sorted(path.name for path in workspace.out_dir.iterdir()) == [
        "data",
        "manifest.json",
        "variants",
    ]
    assert not any(workspace.result_files()[i].exists() for i in range(3))


def test_nothing_is_trained_or_evaluated(
    workspace: Workspace, seams: Seams, loop_calls: list[str]
) -> None:
    run_ok(prepare_args(workspace))

    assert loop_calls == []
    assert seams.train.calls == []
    assert seams.evaluate.calls == []
    assert not list(workspace.out_dir.rglob("ckpt"))
    assert not list(workspace.out_dir.rglob("train.json"))
    assert not list(workspace.out_dir.rglob("result.json"))


def test_the_files_are_what_a_training_run_prepares(
    workspace: Workspace, seams: Seams
) -> None:
    run_ok(prepare_args(workspace))
    prepared = prepared_files(workspace.out_dir)
    shutil.rmtree(workspace.out_dir)

    run_ok(workspace.args())

    assert prepared
    assert prepared_files(workspace.out_dir) == prepared
    assert "variants/" + ID_SRC + "/config.yaml" in prepared
    assert "variants/" + ID_TGT + "/valid.tgt" in prepared


def test_it_prepares_through_the_function_the_run_uses_once_per_variant(
    workspace: Workspace, seams: Seams, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[tuple[str, int]] = []
    real = train_cli.prepare_variant

    def recording(plan: Any, planned: Any, *, threads: int) -> Any:
        calls.append((planned.variant.id, threads))
        return real(plan, planned, threads=threads)

    monkeypatch.setattr(train_cli, "prepare_variant", recording)

    run_ok(prepare_args(workspace))
    run_ok(prepare_args(workspace, "--threads", "3"))

    assert calls == [(variant_id, 2) for variant_id in VARIANT_IDS] + [
        (variant_id, 3) for variant_id in VARIANT_IDS
    ]


def test_preparing_again_gives_the_same_files_and_a_new_manifest(
    workspace: Workspace, seams: Seams
) -> None:
    run_ok(prepare_args(workspace))
    first = prepared_files(workspace.out_dir)
    before = manifest_of(workspace)

    run_ok(prepare_args(workspace))

    assert prepared_files(workspace.out_dir) == first
    after = manifest_of(workspace)
    assert [item.sha256 for item in after.outputs] == [
        item.sha256 for item in before.outputs
    ]


# --- the manifest ---------------------------------------------------------------------------


def test_the_manifest_has_its_own_stage_and_the_inputs_of_the_run(
    workspace: Workspace, seams: Seams
) -> None:
    run_ok(prepare_args(workspace))

    manifest = manifest_of(workspace)
    assert manifest.stage == "train_prepare"
    assert manifest.seed == 1234
    assert manifest.status == {"prepare": "ok"}
    assert [item.path for item in manifest.inputs] == [
        str(path) for path in workspace.inputs
    ]
    assert [item.sha256 for item in manifest.inputs] == [
        sha256(path) for path in workspace.inputs
    ]
    assert manifest.config_sha256 == sha256(workspace.matrix)
    assert manifest.tool_versions == {
        package: importlib.metadata.version(package) for package in TOOL_PACKAGES
    }


def test_the_manifest_lists_the_configs_the_validation_files_and_every_corpus_file_once(
    workspace: Workspace, seams: Seams
) -> None:
    run_ok(prepare_args(workspace))

    manifest = manifest_of(workspace)
    paths = [item.path for item in manifest.outputs]
    assert len(paths) == len(set(paths))
    expected = {
        str(path)
        for path in workspace.out_dir.rglob("*")
        if path.is_file()
        and path.name != "manifest.json"
        and (
            path.parts[len(workspace.out_dir.parts)] == "data"
            or path.name in ("config.yaml", "valid.src", "valid.tgt")
        )
    }
    assert set(paths) == expected
    for variant_id in VARIANT_IDS:
        assert str(workspace.config_file(variant_id)) in paths
    # Eight corpora (four directions, authentic and back-translated) of three files each,
    # and three files for each of the two variants.
    assert len(paths) == 8 * 3 + 2 * 3
    for item in manifest.outputs:
        assert item.sha256 == sha256(Path(item.path)), item.path
        assert item.bytes == Path(item.path).stat().st_size, item.path


def test_the_corpus_files_carry_their_row_counts_and_their_sidecars_are_listed(
    workspace: Workspace, seams: Seams
) -> None:
    run_ok(prepare_args(workspace))

    outputs = {item.path: item for item in manifest_of(workspace).outputs}
    sidecars = sorted(path for path in outputs if path.endswith(".json"))
    assert len(sidecars) == 8
    for sidecar in sidecars:
        record = json.loads(Path(sidecar).read_text(encoding="utf-8"))
        stem = sidecar.removesuffix(".json")
        assert outputs[stem + ".src"].rows == record["rows"]
        assert outputs[stem + ".tgt"].rows == record["rows"]
        assert outputs[stem + ".src"].sha256 == record["src_sha256"]
        assert outputs[stem + ".tgt"].sha256 == record["tgt_sha256"]


def test_the_metrics_count_the_variants_the_corpora_and_the_rows(
    workspace: Workspace, seams: Seams
) -> None:
    run_ok(prepare_args(workspace, "--threads", "3"))

    metrics = manifest_of(workspace).metrics
    assert metrics["variants.total"] == 2.0
    assert metrics["corpora.files"] == 8.0
    assert metrics["threads"] == 3.0
    assert metrics["seed.0"] == 1234.0
    assert metrics["count.rows_kept"] == 4800.0
    assert "count.dev.rows_usable" in metrics


def test_the_manifest_is_written_after_every_file_it_lists(
    workspace: Workspace, seams: Seams, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: list[tuple[list[bool], int]] = []
    real_write = train_common.store.write

    def looking(out_dir: Path, manifest: Manifest) -> Path:
        seen.append(
            (
                [Path(item.path).is_file() for item in manifest.outputs],
                sum(1 for path in out_dir.rglob("*") if path.is_file()),
            )
        )
        return real_write(out_dir, manifest)

    monkeypatch.setattr(train_common.store, "write", looking)

    run_ok(prepare_args(workspace))

    ((exist, files),) = seen
    assert all(exist)
    assert (
        sum(1 for path in workspace.out_dir.rglob("*") if path.is_file()) == files + 1
    )


@pytest.mark.parametrize(
    ("seam", "which", "rewrite"),
    [
        ("load_ablation", "matrix", "text"),
        ("plan_ablation", "corpus", "parquet"),
        ("prepare_variant", "base_config", "text"),
        ("prepare_variant", "spm_model", "other model"),
    ],
    ids=["matrix", "corpus", "base config", "SentencePiece model"],
)
def test_the_manifest_vouches_for_the_bytes_an_input_had_before_it_was_read(
    workspace: Workspace,
    seams: Seams,
    monkeypatch: pytest.MonkeyPatch,
    seam: str,
    which: str,
    rewrite: str,
) -> None:
    import polars as pl

    from tests.support.tiny_parallel import train_toy_spm

    path: Path = getattr(workspace, which)
    before = [sha256(each) for each in workspace.inputs]
    real = getattr(train_cli, seam)
    changed: list[Callable[[], None]] = []

    def after_a_change(*args: Any, **kwargs: Any) -> Any:
        if not changed:
            changed.append(lambda: None)
            if rewrite == "text":
                path.write_text(
                    path.read_text(encoding="utf-8") + "# edited\n", encoding="utf-8"
                )
            elif rewrite == "parquet":
                pl.read_parquet(path).write_parquet(path, compression="uncompressed")
            else:
                other = train_toy_spm(workspace.root / "other_spm", vocab_size=600)
                shutil.copyfile(other, path)
        return real(*args, **kwargs)

    monkeypatch.setattr(train_cli, seam, after_a_change)

    run_ok(prepare_args(workspace))

    assert sha256(path) != before[workspace.inputs.index(path)]
    manifest = manifest_of(workspace)
    assert [item.sha256 for item in manifest.inputs] == before
    assert manifest.config_sha256 == before[0]


# --- the two stages keep apart --------------------------------------------------------------


def test_a_prepare_into_a_directory_with_the_manifest_of_a_training_run_is_refused(
    workspace: Workspace, seams: Seams
) -> None:
    run_ok(workspace.args())
    before = snapshot(workspace.out_dir)

    result = runner.invoke(app, prepare_args(workspace))

    assert_one_error(
        result,
        3,
        f"{workspace.out_dir} holds the manifest of stage 'train_ablate', not of "
        "'train_prepare'",
    )
    assert snapshot(workspace.out_dir) == before


def test_a_training_run_into_a_directory_with_the_manifest_of_a_prepare_is_refused(
    workspace: Workspace, seams: Seams
) -> None:
    run_ok(prepare_args(workspace))
    before = snapshot(workspace.out_dir)

    result = runner.invoke(app, workspace.args())

    assert_one_error(
        result,
        3,
        f"{workspace.out_dir} holds the manifest of stage 'train_prepare', not of "
        "'train_ablate'",
    )
    assert snapshot(workspace.out_dir) == before
    assert seams.train.calls == []


@pytest.mark.parametrize("first", ["ablate", "prepare"])
def test_a_dry_run_is_refused_for_the_stage_it_would_replace(
    workspace: Workspace, seams: Seams, first: str
) -> None:
    run_ok(workspace.args() if first == "ablate" else prepare_args(workspace))
    other = prepare_args(workspace) if first == "ablate" else workspace.args()

    result = runner.invoke(app, [*other, "--dry-run"])

    assert_one_error(result, 3, f"{workspace.out_dir} holds the manifest of stage")


def test_the_same_stage_again_is_not_refused(
    workspace: Workspace, seams: Seams
) -> None:
    run_ok(prepare_args(workspace))

    run_ok(prepare_args(workspace))
    run_ok(prepare_args(workspace, "--dry-run"))

    assert manifest_of(workspace).stage == "train_prepare"


# --- --dry-run, --force and --timeout ---------------------------------------------------------


def test_a_dry_run_wins_over_prepare_only_and_creates_nothing(
    workspace: Workspace, seams: Seams, monkeypatch: pytest.MonkeyPatch
) -> None:
    def refuse(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("a dry run prepared a variant")

    monkeypatch.setattr(train_cli, "prepare_variant", refuse)
    expected = run_ok(workspace.args("--dry-run")).stdout

    result = run_ok(prepare_args(workspace, "--dry-run"))

    assert result.stdout == expected
    assert not workspace.out_dir.exists()
    assert seams.train.calls == []


def test_force_with_prepare_only_is_a_config_error(
    workspace: Workspace, seams: Seams
) -> None:
    result = runner.invoke(app, prepare_args(workspace, "--force"))

    assert_one_error(result, 2, "--force")
    assert not workspace.out_dir.exists()


def test_force_with_prepare_only_and_a_dry_run_is_a_config_error_too(
    workspace: Workspace, seams: Seams
) -> None:
    result = runner.invoke(app, prepare_args(workspace, "--force", "--dry-run"))

    assert_one_error(result, 2, "--force")
    assert not workspace.out_dir.exists()


@pytest.mark.parametrize("extra", [[], ["--dry-run"]])
def test_a_timeout_with_prepare_only_is_a_config_error(
    workspace: Workspace, seams: Seams, extra: list[str]
) -> None:
    result = runner.invoke(app, prepare_args(workspace, "--timeout", "30", *extra))

    assert_one_error(result, 2, "--timeout")
    assert not workspace.out_dir.exists()


def test_an_unusable_timeout_is_reported_as_such_before_the_combination(
    workspace: Workspace, seams: Seams
) -> None:
    result = runner.invoke(app, prepare_args(workspace, "--timeout", "0"))

    assert_one_error(result, 2, "--timeout must be a positive, finite number")


def test_threads_are_accepted_and_recorded(workspace: Workspace, seams: Seams) -> None:
    run_ok(prepare_args(workspace, "--threads", "4"))

    assert manifest_of(workspace).metrics["threads"] == 4.0


# --- failures --------------------------------------------------------------------------------


def test_a_failure_while_preparing_leaves_no_manifest_and_removes_the_old_one(
    workspace: Workspace, seams: Seams, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_ok(prepare_args(workspace))
    real = train_cli.prepare_variant
    calls: list[str] = []

    def failing_on_the_second(plan: Any, planned: Any, *, threads: int) -> Any:
        calls.append(planned.variant.id)
        if len(calls) == 2:
            raise OSError(28, "No space left on device")
        return real(plan, planned, threads=threads)

    monkeypatch.setattr(train_cli, "prepare_variant", failing_on_the_second)

    result = runner.invoke(app, prepare_args(workspace))

    assert_one_error(
        result, 1, f"cannot write the ablation outputs under {workspace.out_dir}: "
    )
    assert not workspace.manifest.exists()


def test_a_domain_error_while_preparing_exits_with_its_code(
    workspace: Workspace, seams: Seams, monkeypatch: pytest.MonkeyPatch
) -> None:
    def failing(*args: Any, **kwargs: Any) -> Any:
        raise RinchenError("something else")

    monkeypatch.setattr(train_cli, "prepare_variant", failing)

    result = runner.invoke(app, prepare_args(workspace))

    assert_one_error(result, 1, "something else")
    assert not workspace.manifest.exists()


def test_an_interrupt_while_preparing_exits_130_and_writes_no_manifest(
    workspace: Workspace, seams: Seams, monkeypatch: pytest.MonkeyPatch
) -> None:
    def interrupted(*args: Any, **kwargs: Any) -> Any:
        raise KeyboardInterrupt

    monkeypatch.setattr(train_cli, "prepare_variant", interrupted)

    result = runner.invoke(app, prepare_args(workspace))

    assert result.exit_code == 130, result.output
    assert not workspace.manifest.exists()


def test_a_missing_corpus_is_a_data_error_before_anything_is_written(
    workspace: Workspace, seams: Seams
) -> None:
    missing = workspace.root / "nowhere.parquet"

    result = runner.invoke(app, workspace.args("--prepare-only", corpora=(missing,)))

    assert_one_error(result, 3, f"corpus file not found: {missing}")
    assert not workspace.out_dir.exists()
