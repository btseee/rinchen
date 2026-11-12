"""`rinchen train docify`: what it plans and prints, what it writes (the doc corpora, the
validation windows, the config of the continuation and the manifest) and how a rerun
leaves the output directory. What it refuses is in `test_cli_train_docify_refusals.py`.

Nothing here starts Eole. The corpus, the SentencePiece model and the base config are
real, so the files are the ones a real run writes. The counts of `hand_built_rows()` are
worked out in `tests.support.doc_cli`: with windows of exactly two sentences 5 windows are
cut (3 English to Mongolian, 2 Mongolian to English) from the 10 of the 20 rows that can be
in one.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
from pathlib import Path
from typing import Any

import polars as pl
import pytest
import yaml

from rinchen.cli import train_common, train_docify
from rinchen.cli.main import app
from rinchen.manifest import store
from rinchen.manifest.model import Manifest
from rinchen.train import launch as launch_module
from rinchen.train.eole_cfg import read_config, validate_train_config
from tests.support.doc_cli import (
    EN_MN_CORPUS,
    MN_EN_CORPUS,
    PAIRS,
    TEST_ONLY_TEXT,
    WRITTEN_FILES,
    DocifyWorkspace,
    hand_built_rows,
    make_docify_workspace,
    run_ok,
    runner,
    toy_document_rows,
)
from tests.support.fake_launch import EOLE
from tests.support.launch_cli import snapshot

TOOL_PACKAGES = ["sentencepiece", "polars"]
EN_MN_WINDOWS = {
    "This is the book. <sep> The water is here.",
    "Where is the dog? <sep> The tree is there.",
    "The book and the water. <sep> This is the home.",
}
EN_MN_TARGETS = {
    "Энэ бол ном. <sep> Ус энд байна.",
    "Нохой хаана байна вэ? <sep> Тэнд мод байна.",
    "Ном ба ус. <sep> Энэ бол гэр.",
}
STATISTICS = [
    "corpus files: 1",
    "max pieces: 62",
    "rows read: 20",
    "rows in windows: 10",
    "windows: 5",
    "windows of size 1: 0",
    "windows of size 2: 5",
    "excluded not_train: 7",
    "excluded filtered: 0",
    "excluded excluded_origin: 1",
    "excluded no_doc_id: 1",
    "excluded no_seg_idx: 0",
    "excluded empty: 0",
    "excluded line_break: 0",
    "excluded contains_sep: 0",
    "excluded duplicate_seg_idx: 0",
    "excluded too_long: 0",
    "excluded gap_single: 0",
    "excluded tail: 1",
    "pieces src: min 12, mean 18.800, p95 22, max 22",
    "pieces tgt: min 10, mean 15.400, p95 20, max 20",
    "dev windows: 2",
    f"corpus {EN_MN_CORPUS}: windows 3, weight 60",
    f"corpus {MN_EN_CORPUS}: windows 2, weight 40",
]


@pytest.fixture
def workspace(tmp_path: Path) -> DocifyWorkspace:
    return make_docify_workspace(tmp_path)


def manifest_of(workspace: DocifyWorkspace) -> Manifest:
    manifest = store.read(workspace.out_dir)
    assert manifest is not None
    return manifest


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def lines_of(path: Path) -> list[str]:
    return path.read_text(encoding="utf-8").splitlines()


def config_of(workspace: DocifyWorkspace) -> dict[str, Any]:
    return yaml.safe_load(workspace.config.read_text(encoding="utf-8"))


# --- help and defaults ---------------------------------------------------------------------


def test_the_train_group_lists_docify() -> None:
    result = runner.invoke(app, ["train", "--help"])

    assert result.exit_code == 0
    assert "docify" in result.stdout


def test_help_names_every_option() -> None:
    result = runner.invoke(app, ["train", "docify", "--help"])

    assert result.exit_code == 0
    for option in (
        "--corpus",
        "--spm-model",
        "--config",
        "--out-dir",
        "--min-size",
        "--max-size",
        "--max-pieces",
        "--temperature",
        "--seed",
        "--dev-max-windows",
        "--weight-total",
        "--dry-run",
    ):
        assert option in result.stdout, option


def test_the_help_states_the_defaults_that_do_not_come_from_the_base_config() -> None:
    text = " ".join(runner.invoke(app, ["train", "docify", "--help"]).stdout.split())

    for option, default in (
        ("--min-size", "2"),
        ("--max-size", "4"),
        ("--temperature", "1.0"),
        ("--seed", "1234"),
        ("--dev-max-windows", "1000"),
        ("--weight-total", "100"),
    ):
        assert f"[default: {default}]" in text.split(option, 1)[1].split("--", 1)[0]


def test_the_defaults_are_windows_of_two_to_four_the_budget_of_the_base_seed_1234(
    workspace: DocifyWorkspace,
) -> None:
    run_ok(workspace.args())

    manifest = manifest_of(workspace)
    assert manifest.seed == 1234
    assert manifest.metrics["min_size"] == 2.0
    assert manifest.metrics["max_size"] == 4.0
    assert manifest.metrics["max_pieces"] == 62.0
    assert manifest.metrics["temperature"] == 1.0
    assert manifest.metrics["weight_total"] == 100.0
    assert manifest.metrics["dev_max_windows"] == 1000.0
    # Every size from one to the largest has a count, and none above it.
    assert [
        key for key in manifest.metrics if key.startswith("count.windows_by_size.")
    ] == [f"count.windows_by_size.{size}" for size in (1, 2, 3, 4)]
    assert manifest.metrics["count.windows_by_size.1"] == 0.0


# --- the dry run -----------------------------------------------------------------------------


def test_a_dry_run_prints_the_statistics_the_windows_and_the_weights(
    workspace: DocifyWorkspace,
) -> None:
    result = run_ok(workspace.args("--dry-run", *PAIRS))

    assert result.stdout.splitlines() == [*STATISTICS, "nothing was written"]


def test_a_dry_run_creates_nothing_not_even_the_output_directory(
    workspace: DocifyWorkspace,
) -> None:
    run_ok(workspace.args("--dry-run", *PAIRS))

    assert not workspace.out_dir.exists()


def test_a_dry_run_writes_nothing_beside_an_output_directory_that_exists(
    workspace: DocifyWorkspace,
) -> None:
    workspace.out_dir.mkdir()

    run_ok(workspace.args("--dry-run", *PAIRS))

    assert list(workspace.out_dir.iterdir()) == []


def test_a_dry_run_and_the_run_that_follows_it_agree_on_the_weights(
    workspace: DocifyWorkspace,
) -> None:
    dry = run_ok(workspace.args("--dry-run", *PAIRS)).stdout.splitlines()
    run_ok(workspace.args(*PAIRS))

    weights = {
        corpus: entry["weight"]
        for corpus, entry in config_of(workspace)["data"].items()
        if corpus != "valid"
    }
    assert weights == {EN_MN_CORPUS: 60, MN_EN_CORPUS: 40}
    assert f"corpus {EN_MN_CORPUS}: windows 3, weight 60" in dry


# --- what a run writes and prints -------------------------------------------------------------


def test_a_run_prints_the_statistics_then_what_it_wrote(
    workspace: DocifyWorkspace,
) -> None:
    result = run_ok(workspace.args(*PAIRS))

    assert result.stdout.splitlines() == [
        *STATISTICS,
        "corpora: 2 (reused 0)",
        "validation windows: 2",
        f"config: {workspace.config}",
        f"manifest: {workspace.manifest}",
    ]


def test_it_writes_the_corpora_the_validation_files_the_config_and_the_manifest_only(
    workspace: DocifyWorkspace,
) -> None:
    run_ok(workspace.args(*PAIRS))

    assert sorted(workspace.written()) == sorted(WRITTEN_FILES)


def test_the_corpus_files_hold_one_joined_window_per_line_and_no_tags(
    workspace: DocifyWorkspace,
) -> None:
    run_ok(workspace.args(*PAIRS))

    directory = workspace.out_dir / "data" / "doc"
    sources = lines_of(directory / f"{EN_MN_CORPUS}.src")
    targets = lines_of(directory / f"{EN_MN_CORPUS}.tgt")
    assert set(sources) == EN_MN_WINDOWS
    assert set(targets) == EN_MN_TARGETS
    assert len(sources) == len(targets) == 3
    assert len(lines_of(directory / f"{MN_EN_CORPUS}.src")) == 2


def test_the_validation_files_hold_the_dev_windows_as_tag_doc_and_pieces(
    workspace: DocifyWorkspace,
) -> None:
    run_ok(workspace.args(*PAIRS))

    sources = lines_of(workspace.out_dir / "valid.src")
    targets = lines_of(workspace.out_dir / "valid.tgt")
    assert len(sources) == len(targets) == 2
    assert sorted(line.split()[0] for line in sources) == ["<2eng_Latn>", "<2khk_Cyrl>"]
    assert all(line.split()[1] == "<doc>" for line in sources)
    assert all("<sep>" in line.split() for line in sources + targets)


def test_no_row_of_the_test_split_is_in_any_file(workspace: DocifyWorkspace) -> None:
    run_ok(workspace.args(*PAIRS))

    for relative, content in workspace.written().items():
        text = content.decode("utf-8")
        for sentence in TEST_ONLY_TEXT:
            assert sentence not in text, relative


def test_the_config_is_the_base_with_the_data_the_paths_and_nothing_to_continue_from(
    workspace: DocifyWorkspace,
) -> None:
    run_ok(workspace.args(*PAIRS))

    cfg = config_of(workspace)
    base = read_config(workspace.base_config, "base")
    out = workspace.out_dir
    spm = workspace.spm_model.as_posix()
    assert cfg["model"] == base["model"]
    assert cfg["src_vocab"] == (out / "run" / "vocab.src").as_posix()
    assert cfg["training"]["model_path"] == (out / "run" / "ckpt").as_posix()
    assert cfg["transforms_configs"]["sentencepiece"] == {
        "src_subword_model": spm,
        "tgt_subword_model": spm,
    }
    assert cfg["transforms_configs"]["filtertoolong"] == {
        "src_seq_length": 64,
        "tgt_seq_length": 64,
    }
    assert cfg["data"][EN_MN_CORPUS] == {
        "path_src": (out / "data" / "doc" / f"{EN_MN_CORPUS}.src").as_posix(),
        "path_tgt": (out / "data" / "doc" / f"{EN_MN_CORPUS}.tgt").as_posix(),
        "transforms": ["sentencepiece", "prefix", "filtertoolong"],
        "weight": 60,
        "src_prefix": "<2khk_Cyrl> <doc>",
        "tgt_prefix": "",
    }
    assert cfg["data"][MN_EN_CORPUS]["src_prefix"] == "<2eng_Latn> <doc>"
    assert cfg["data"]["valid"] == {
        "path_src": (out / "valid.src").as_posix(),
        "path_tgt": (out / "valid.tgt").as_posix(),
    }
    assert "train_from" not in cfg["training"]
    assert "reset_optim" not in cfg["training"]
    validate_train_config(cfg)


def test_the_launcher_takes_the_config_into_the_run_directory_it_names(
    workspace: DocifyWorkspace, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_ok(workspace.args(*PAIRS))
    monkeypatch.setattr(launch_module, "eole_command", lambda: EOLE)
    run_dir = workspace.out_dir / "run"

    result = run_ok(
        [
            "train",
            "launch",
            "--config",
            str(workspace.config),
            "--run-dir",
            str(run_dir),
            "--write-only",
        ]
    )

    assert result.stdout.strip() == str(run_dir / "command.sh")


def test_the_paths_in_the_config_are_absolute_whatever_the_working_directory(
    workspace: DocifyWorkspace, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(workspace.root)

    run_ok(
        workspace.args(*PAIRS, out_dir=Path("out"), spm_model=Path("spm") / "toy.model")
    )

    cfg = config_of(workspace)
    assert cfg["src_vocab"] == (workspace.root / "out" / "run" / "vocab.src").as_posix()
    assert (
        cfg["transforms_configs"]["sentencepiece"]["src_subword_model"]
        == (workspace.root / "spm" / "toy.model").as_posix()
    )
    assert cfg["data"]["valid"]["path_src"].startswith(workspace.root.as_posix())


def test_every_statistic_is_a_metric_and_the_rows_are_all_counted(
    workspace: DocifyWorkspace,
) -> None:
    run_ok(workspace.args(*PAIRS))

    metrics = manifest_of(workspace).metrics
    reasons = (
        "not_train",
        "filtered",
        "excluded_origin",
        "no_doc_id",
        "no_seg_idx",
        "empty",
        "line_break",
        "contains_sep",
        "duplicate_seg_idx",
        "too_long",
        "gap_single",
        "tail",
    )
    assert {reason: metrics[f"count.excluded.{reason}"] for reason in reasons} == {
        "not_train": 7.0,
        "filtered": 0.0,
        "excluded_origin": 1.0,
        "no_doc_id": 1.0,
        "no_seg_idx": 0.0,
        "empty": 0.0,
        "line_break": 0.0,
        "contains_sep": 0.0,
        "duplicate_seg_idx": 0.0,
        "too_long": 0.0,
        "gap_single": 0.0,
        "tail": 1.0,
    }
    assert metrics["count.rows_read"] == 20.0
    assert metrics["count.rows_in_windows"] == 10.0
    assert metrics["count.rows_read"] == metrics["count.rows_in_windows"] + sum(
        metrics[f"count.excluded.{reason}"] for reason in reasons
    )
    assert metrics["count.windows"] == 5.0
    assert metrics["count.windows_by_size.2"] == 5.0
    assert metrics["count.pieces.src.min"] == 12.0
    assert metrics["count.pieces.src.max"] == 22.0
    assert metrics["count.pieces.tgt.mean"] == pytest.approx(15.4)
    assert metrics["count.dev.windows"] == 2.0
    assert metrics["count.dev.excluded.not_dev"] == 16.0


def test_the_metrics_also_record_the_parameters_and_the_weights(
    workspace: DocifyWorkspace,
) -> None:
    run_ok(workspace.args(*PAIRS, "--seed", "9", "--temperature", "0.5"))

    metrics = manifest_of(workspace).metrics
    assert metrics["max_pieces"] == 62.0
    assert metrics["min_size"] == 2.0
    assert metrics["max_size"] == 2.0
    assert metrics["temperature"] == 0.5
    assert metrics["weight_total"] == 100.0
    assert metrics["corpora.files"] == 2.0
    assert metrics["valid.windows"] == 2.0
    assert metrics[f"weight.{EN_MN_CORPUS}"] == 55.0
    assert metrics[f"weight.{MN_EN_CORPUS}"] == 45.0


def test_the_manifest_has_the_stage_the_inputs_and_the_tool_versions(
    workspace: DocifyWorkspace,
) -> None:
    run_ok(workspace.args(*PAIRS))

    manifest = manifest_of(workspace)
    assert manifest.stage == "train_docify"
    assert manifest.status == {"docify": "ok"}
    assert manifest.seed == 1234
    assert [item.path for item in manifest.inputs] == [
        str(path) for path in workspace.inputs
    ]
    assert [item.sha256 for item in manifest.inputs] == [
        sha256(path) for path in workspace.inputs
    ]
    assert manifest.config_sha256 == sha256(workspace.base_config)
    assert manifest.tool_versions == {
        package: importlib.metadata.version(package) for package in TOOL_PACKAGES
    }


def test_the_manifest_lists_every_written_file_once_with_its_digest_and_size(
    workspace: DocifyWorkspace,
) -> None:
    run_ok(workspace.args(*PAIRS))

    manifest = manifest_of(workspace)
    paths = [item.path for item in manifest.outputs]
    assert len(paths) == len(set(paths))
    assert sorted(
        Path(path).relative_to(workspace.out_dir).as_posix() for path in paths
    ) == (sorted(name for name in WRITTEN_FILES if name != "manifest.json"))
    for item in manifest.outputs:
        assert item.sha256 == sha256(Path(item.path)), item.path
        assert item.bytes == Path(item.path).stat().st_size, item.path


def test_the_corpus_files_carry_their_row_counts_and_the_sidecars_are_listed(
    workspace: DocifyWorkspace,
) -> None:
    run_ok(workspace.args(*PAIRS))

    outputs = {Path(item.path).name: item for item in manifest_of(workspace).outputs}
    assert outputs[f"{EN_MN_CORPUS}.src"].rows == 3
    assert outputs[f"{EN_MN_CORPUS}.tgt"].rows == 3
    assert outputs[f"{MN_EN_CORPUS}.src"].rows == 2
    assert outputs["valid.src"].rows == 2
    assert outputs["valid.tgt"].rows == 2
    record = json.loads(
        Path(outputs[f"{EN_MN_CORPUS}.json"].path).read_text(encoding="utf-8")
    )
    assert record["rows"] == 3
    assert record["src_sha256"] == outputs[f"{EN_MN_CORPUS}.src"].sha256


def test_the_manifest_is_written_after_every_file_it_lists(
    workspace: DocifyWorkspace, monkeypatch: pytest.MonkeyPatch
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

    run_ok(workspace.args(*PAIRS))

    ((exist, files),) = seen
    assert all(exist)
    assert len(workspace.written()) == files + 1


@pytest.mark.parametrize(
    ("seam", "which", "rewrite"),
    [
        ("read_corpus_rows", "corpus", "parquet"),
        ("check_continuation_base", "base_config", "text"),
        ("docify", "spm_model", "other model"),
    ],
    ids=["corpus", "base config", "SentencePiece model"],
)
def test_the_manifest_vouches_for_the_bytes_an_input_had_before_it_was_read(
    workspace: DocifyWorkspace,
    monkeypatch: pytest.MonkeyPatch,
    seam: str,
    which: str,
    rewrite: str,
) -> None:
    from tests.support.tiny_parallel import train_toy_spm

    path: Path = getattr(workspace, which)
    before = [sha256(each) for each in workspace.inputs]
    real = getattr(train_docify, seam)
    changed: list[bool] = []

    def after_a_change(*args: Any, **kwargs: Any) -> Any:
        if not changed:
            changed.append(True)
            if rewrite == "text":
                path.write_text(
                    path.read_text(encoding="utf-8") + "# edited\n", encoding="utf-8"
                )
            elif rewrite == "parquet":
                pl.read_parquet(path).write_parquet(path, compression="uncompressed")
            else:
                other = train_toy_spm(workspace.root / "other_spm", vocab_size=600)
                path.write_bytes(other.read_bytes())
        return real(*args, **kwargs)

    monkeypatch.setattr(train_docify, seam, after_a_change)

    run_ok(workspace.args(*PAIRS))

    assert sha256(path) != before[workspace.inputs.index(path)]
    manifest = manifest_of(workspace)
    assert [item.sha256 for item in manifest.inputs] == before
    assert manifest.config_sha256 == before[0]


# --- the windows ------------------------------------------------------------------------------


def test_the_same_options_give_the_same_files_in_another_directory(
    tmp_path: Path,
) -> None:
    first = make_docify_workspace(tmp_path / "a", rows=toy_document_rows())
    second = make_docify_workspace(tmp_path / "b", rows=toy_document_rows())
    run_ok(first.args("--seed", "5"))
    run_ok(second.args("--seed", "5"))

    def bodies(workspace: DocifyWorkspace) -> dict[str, bytes]:
        return {
            name: content
            for name, content in workspace.written().items()
            if name.startswith(("data/", "valid."))
        }

    assert bodies(first) == bodies(second)


def test_the_seed_changes_the_training_windows_and_never_the_validation_windows(
    tmp_path: Path,
) -> None:
    first = make_docify_workspace(tmp_path / "a", rows=toy_document_rows())
    second = make_docify_workspace(tmp_path / "b", rows=toy_document_rows())
    run_ok(first.args("--seed", "1"))
    run_ok(second.args("--seed", "2"))

    corpus = "data/doc/eng_Latn-khk_Cyrl.doc.src"
    assert first.written()[corpus] != second.written()[corpus]
    assert first.written()["valid.src"] == second.written()["valid.src"]
    assert first.written()["valid.tgt"] == second.written()["valid.tgt"]


def test_toy_documents_become_windows_inside_the_budget_and_every_row_is_counted(
    tmp_path: Path,
) -> None:
    workspace = make_docify_workspace(tmp_path, rows=toy_document_rows())

    run_ok(workspace.args())

    metrics = manifest_of(workspace).metrics
    assert metrics["count.windows"] > 0
    assert metrics["count.pieces.src.max"] <= metrics["max_pieces"]
    assert metrics["count.pieces.tgt.max"] <= metrics["max_pieces"]
    assert metrics["count.windows_by_size.1"] == 0.0
    excluded = sum(
        value for key, value in metrics.items() if key.startswith("count.excluded.")
    )
    assert metrics["count.rows_read"] == metrics["count.rows_in_windows"] + excluded
    assert len(lines_of(workspace.out_dir / "valid.src")) == metrics["valid.windows"]


def test_dev_max_windows_keeps_that_many_validation_windows_of_each_direction(
    tmp_path: Path,
) -> None:
    workspace = make_docify_workspace(tmp_path, rows=toy_document_rows())

    run_ok(workspace.args("--dev-max-windows", "1"))

    sources = lines_of(workspace.out_dir / "valid.src")
    assert len(sources) == 4
    assert sorted(line.split()[0] for line in sources) == [
        "<2eng_Latn>",
        "<2khk_Cyrl>",
        "<2khk_Cyrl>",
        "<2rus_Cyrl>",
    ]
    assert manifest_of(workspace).metrics["valid.windows"] == 4.0


@pytest.mark.parametrize(
    ("filtertoolong", "expected"),
    [
        ({"src_seq_length": 64, "tgt_seq_length": 64}, 62),
        ({"src_seq_length": 128, "tgt_seq_length": 128}, 126),
        ({"src_seq_length": 100, "tgt_seq_length": 128}, 100),
        ({"src_seq_length": 128, "tgt_seq_length": 100}, 98),
        ({"src_seq_length": 128}, 128),
        ({"tgt_seq_length": 128}, 126),
        (None, 190),
    ],
    ids=["64", "128", "short source", "short target", "no target", "no source", "none"],
)
def test_the_budget_comes_from_what_filtertoolong_keeps_in_the_base_config(
    tmp_path: Path, filtertoolong: dict[str, int] | None, expected: int
) -> None:
    def edit(cfg: dict[str, Any]) -> None:
        if filtertoolong is None:
            cfg["transforms_configs"].pop("filtertoolong")
        else:
            cfg["transforms_configs"]["filtertoolong"] = filtertoolong

    workspace = make_docify_workspace(tmp_path, edit_base=edit)

    dry = run_ok(workspace.args("--dry-run"))
    assert f"max pieces: {expected}" in dry.stdout.splitlines()

    run_ok(workspace.args())
    assert manifest_of(workspace).metrics["max_pieces"] == float(expected)


@pytest.mark.parametrize("budget", [50, 62])
def test_an_explicit_budget_up_to_what_the_base_keeps_is_used(
    workspace: DocifyWorkspace, budget: int
) -> None:
    run_ok(workspace.args("--max-pieces", str(budget), *PAIRS))

    metrics = manifest_of(workspace).metrics
    assert metrics["max_pieces"] == float(budget)
    assert metrics["count.pieces.src.max"] <= budget


def test_a_small_budget_shortens_windows_and_every_sentence_is_still_counted(
    workspace: DocifyWorkspace,
) -> None:
    # At 18 pieces the pairs of 19, 21 and 19 source pieces (English to Mongolian) and of
    # 22 (the Mongolian pair of the sentences 2 and 3) are cut to one sentence.
    run_ok(workspace.args("--max-pieces", "18", *PAIRS))

    metrics = manifest_of(workspace).metrics
    assert metrics["count.windows_by_size.1"] > 0
    assert metrics["count.rows_read"] == metrics["count.rows_in_windows"] + sum(
        value for key, value in metrics.items() if key.startswith("count.excluded.")
    )
    # The validation windows are cut with the same budget: the one pair of each
    # direction is over 18 pieces (20 and 22 on the source side), so both are cut.
    assert metrics["count.dev.windows_by_size.1"] == 2.0
    assert metrics["count.dev.pieces.src.max"] <= 18.0


def test_a_temperature_below_one_lifts_the_smaller_direction(
    workspace: DocifyWorkspace,
) -> None:
    result = run_ok(workspace.args("--dry-run", "--temperature", "0.5", *PAIRS))

    assert f"corpus {EN_MN_CORPUS}: windows 3, weight 55" in result.stdout.splitlines()
    assert f"corpus {MN_EN_CORPUS}: windows 2, weight 45" in result.stdout.splitlines()


def test_several_corpus_files_are_read_together(tmp_path: Path) -> None:
    rows = hand_built_rows()
    first = make_docify_workspace(tmp_path, rows=rows[:10])
    second = make_docify_workspace(tmp_path / "other", rows=rows[10:])

    result = run_ok(
        first.args("--dry-run", *PAIRS, corpora=[first.corpus, second.corpus])
    )

    assert "corpus files: 2" in result.stdout.splitlines()
    assert "rows read: 20" in result.stdout.splitlines()


def test_a_rerun_reuses_the_unchanged_corpora_and_writes_the_same_files(
    workspace: DocifyWorkspace,
) -> None:
    run_ok(workspace.args(*PAIRS))
    first = workspace.written()
    before = manifest_of(workspace)

    result = run_ok(workspace.args(*PAIRS))

    assert "corpora: 2 (reused 2)" in result.stdout.splitlines()
    assert workspace.written().keys() == first.keys()
    for name, content in first.items():
        if name != "manifest.json":
            assert workspace.written()[name] == content, name
    after = manifest_of(workspace)
    assert [item.sha256 for item in after.outputs] == [
        item.sha256 for item in before.outputs
    ]


def test_a_rerun_with_other_windows_writes_the_corpora_again(
    workspace: DocifyWorkspace,
) -> None:
    run_ok(workspace.args(*PAIRS))
    first = workspace.written()

    result = run_ok(workspace.args("--min-size", "1", "--max-size", "1"))

    assert "corpora: 2 (reused 0)" in result.stdout.splitlines()
    assert (
        workspace.written()[f"data/doc/{EN_MN_CORPUS}.src"]
        != first[f"data/doc/{EN_MN_CORPUS}.src"]
    )


def test_a_dry_run_after_a_run_leaves_the_files_as_they_were(
    workspace: DocifyWorkspace,
) -> None:
    run_ok(workspace.args(*PAIRS))
    before = snapshot(workspace.out_dir)

    run_ok(workspace.args("--dry-run", *PAIRS))

    assert snapshot(workspace.out_dir) == before


def test_the_directory_is_made_then_the_old_manifest_goes_then_the_files_are_written(
    workspace: DocifyWorkspace, monkeypatch: pytest.MonkeyPatch
) -> None:
    order: list[str] = []

    def spied(name: str) -> None:
        real = getattr(train_docify, name)

        def spy(*args: Any, **kwargs: Any) -> Any:
            order.append(name)
            return real(*args, **kwargs)

        monkeypatch.setattr(train_docify, name, spy)

    for name in (
        "create_out_dir",
        "begin_commit",
        "build_doc_corpora",
        "write_doc_validation",
        "render_continuation_config",
        "write_manifest",
    ):
        spied(name)

    run_ok(workspace.args(*PAIRS))

    assert order == [
        "create_out_dir",
        "begin_commit",
        "build_doc_corpora",
        "write_doc_validation",
        "render_continuation_config",
        "write_manifest",
    ]
